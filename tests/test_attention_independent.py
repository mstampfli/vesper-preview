"""Expectations computed outside the module.

No test here calls ``project``, ``output`` or ``rotary`` to produce its own expected value; the
numbers come from scalar arithmetic. They follow the attention/FFN review of 2026-10-14, whose
probes reproduced five defects on d2f9579: the precision, storage, empty-chunk, shape and base
tests below fail on that tip.
"""

import math

import pytest
import torch
from torch import Tensor

from vesper.attention import WindowedAttention, kv_cache_empty, kv_cache_push, rotary
from vesper.contracts import ContractError
from vesper.ffn import FFN

BASE = 10000.0
CPU = torch.device("cpu")
F64 = torch.float64


def mat(rows: list[list[float]], dtype: torch.dtype = F64) -> Tensor:
    out = torch.zeros(len(rows), len(rows[0]), dtype=dtype)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[i, j] = value
    return out


def positions(*values: int) -> Tensor:
    out = torch.zeros(1, len(values), dtype=torch.long)
    for i, v in enumerate(values):
        out[0, i] = v
    return out


def set_weights(attn: WindowedAttention, q: Tensor, k: Tensor, v: Tensor, o: Tensor) -> None:
    with torch.no_grad():
        _ = attn.w_q.copy_(q)
        _ = attn.w_k.copy_(k)
        _ = attn.w_v.copy_(v)
        _ = attn.w_o.copy_(o)


# --- independent expectations that already held on d2f9579 ---------------------------------


def test_two_tokens_match_scalar_arithmetic() -> None:
    attn = WindowedAttention(d=2, heads=1, kv_groups=1, head_dim=2, window=2, rotary_base=BASE).to(
        F64
    )
    eye = mat([[1.0, 0.0], [0.0, 1.0]])
    set_weights(attn, eye, eye, mat([[2.0, 0.0], [0.0, 3.0]]), mat([[1.0, 1.0], [2.0, -1.0]]))
    x = mat([[1.0, 0.0], [0.0, 1.0]]).reshape(1, 2, 2)
    y, _ = attn.forward(x, positions(0, 1), kv_cache_empty(1, 1, 2, 2, F64, CPU))
    # token 1 has q = k = (-sin 1, cos 1); against k0 = (1, 0) the scaled logit is
    # -sin(1) / sqrt(2), against itself 1 / sqrt(2); values are (2, 0) and (0, 3)
    s0, s1 = -math.sin(1.0) / math.sqrt(2.0), 1.0 / math.sqrt(2.0)
    a = math.exp(s0) / (math.exp(s0) + math.exp(s1))
    expected = mat([[2.0, 4.0], [2 * a + 3 * (1 - a), 4 * a - 3 * (1 - a)]])
    assert torch.allclose(y[0], expected, rtol=0, atol=1e-12)


def test_rotary_pairs_match_scalar_trig() -> None:
    x = torch.zeros(1, 1, 1, 4, dtype=F64)
    x[0, 0, 0, 0] = 1.0
    x[0, 0, 0, 2] = 1.0
    a0, a1 = 2.0, 2.0 / math.sqrt(BASE)  # pair 1 turns at base^(-2/4)
    r = rotary(x, positions(2), BASE)[0, 0, 0]
    expected = mat([[math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)]])[0]
    assert torch.allclose(r, expected, rtol=0, atol=1e-12)
    y = torch.zeros(1, 1, 1, 4, dtype=F64)
    y[0, 0, 0, 1] = 1.0
    y[0, 0, 0, 3] = 1.0
    r2 = rotary(y, positions(2), BASE)[0, 0, 0]
    expected2 = mat([[-math.sin(a0), math.cos(a0), -math.sin(a1), math.cos(a1)]])[0]
    assert torch.allclose(r2, expected2, rtol=0, atol=1e-12)


def test_gqa_replication_and_output_ordering() -> None:
    attn = WindowedAttention(d=8, heads=4, kv_groups=2, head_dim=2, window=1, rotary_base=BASE).to(
        F64
    )
    one_hot = torch.eye(8, dtype=F64)[:4]  # kv head 0 reads coordinates (0, 1), head 1 reads (2, 3)
    set_weights(
        attn,
        torch.zeros(8, 8, dtype=F64),
        torch.zeros(4, 8, dtype=F64),
        one_hot,
        torch.diag(torch.arange(1, 9, dtype=F64)),
    )
    x = torch.arange(1, 9, dtype=F64).reshape(1, 1, 8)
    y, _ = attn.forward(x, positions(0), kv_cache_empty(1, 2, 2, 1, F64, CPU))
    # query heads (0, 1) read kv head 0 = (1, 2), heads (2, 3) read kv head 1 = (3, 4);
    # the diagonal output matrix multiplies coordinate i by i + 1
    expected = mat([[1.0, 4.0, 3.0, 8.0, 15.0, 24.0, 21.0, 32.0]])
    assert torch.allclose(y[0], expected, rtol=0, atol=1e-12)


def test_ffn_squared_relu_against_scalar_arithmetic() -> None:
    ffn = FFN(d=2, ffn_mult=2).to(F64)
    with torch.no_grad():
        _ = ffn.w1.copy_(mat([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))
        _ = ffn.w2.copy_(mat([[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 2.0, -2.0]]))
    x = mat([[2.0, -1.0]]).reshape(1, 1, 2)
    # hidden preactivations (2, -1, -2, 1), squared relu (4, 0, 0, 1), output (5, 2);
    # a plain relu would give (3, 0)
    assert torch.allclose(ffn.forward(x)[0], mat([[5.0, 2.0]]), rtol=0, atol=1e-12)


# --- the five findings --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "position", "atol"),
    [(torch.float16, 100000, 2e-3), (torch.bfloat16, 256, 1e-2), (torch.float32, 10**7, 1e-6)],
)
def test_rotary_phases_do_not_follow_activation_precision(
    dtype: torch.dtype, position: int, atol: float
) -> None:
    x = torch.zeros(1, 2, 1, 2, dtype=dtype)
    x[..., 0] = 1.0
    r = rotary(x, positions(position, position + 1), BASE)
    assert r.dtype == dtype
    assert bool(torch.isfinite(r).all().item())
    assert not torch.equal(r[0, 0], r[0, 1])
    expected = mat(
        [
            [math.cos(position), math.sin(position)],
            [math.cos(position + 1), math.sin(position + 1)],
        ]
    )
    assert torch.allclose(r[0, :, 0, :].to(F64), expected, rtol=0, atol=atol)


def test_cache_storage_equals_its_logical_size() -> None:
    cache = kv_cache_empty(1, 1, 2, 16, torch.float32, CPU)
    g = torch.Generator().manual_seed(6)
    k = torch.randn(1, 4096, 1, 2, generator=g)
    v = torch.randn(1, 4096, 1, 2, generator=g)
    pushed = kv_cache_push(cache, k, v, torch.arange(4096).reshape(1, 4096))
    assert pushed.k.shape == (1, 16, 1, 2)
    for tensor in (pushed.k, pushed.v, pushed.pos):
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert torch.equal(pushed.k, k[:, -16:])
    assert torch.equal(pushed.v, v[:, -16:])
    # repeated single-token pushes stay at the window and never touch the cache they came from
    before = (pushed.k.clone(), pushed.v.clone(), pushed.pos.clone())
    current = pushed
    for step in range(4096, 4096 + 40):
        current = kv_cache_push(current, k[:, :1], v[:, :1], positions(step))
        assert current.k.untyped_storage().nbytes() == 16 * 2 * 4
        assert current.pos.shape == (1, 16)
    assert int(current.pos[0, -1].item()) == 4096 + 39
    assert int(current.pos[0, 0].item()) == 4096 + 24
    assert torch.equal(pushed.k, before[0])
    assert torch.equal(pushed.v, before[1])
    assert torch.equal(pushed.pos, before[2])


def test_partial_window_pushes_allocate_only_what_they_hold() -> None:
    cache = kv_cache_empty(2, 1, 2, 16, torch.float32, CPU)
    k = torch.ones(2, 5, 1, 2)
    first = kv_cache_push(cache, k, k, torch.arange(5).expand(2, 5))
    assert first.k.shape == (2, 5, 1, 2)
    assert first.k.untyped_storage().nbytes() == first.k.numel() * 4
    second = kv_cache_push(first, k, k, torch.arange(5, 10).expand(2, 5))
    assert second.k.shape == (2, 10, 1, 2)
    assert second.k.untyped_storage().nbytes() == second.k.numel() * 4
    assert kv_cache_push(second, k[:, :0], k[:, :0], torch.zeros(2, 0, dtype=torch.long)) is second


def test_empty_chunk_returns_an_empty_output_and_the_same_cache() -> None:
    attn = WindowedAttention(d=4, heads=2, kv_groups=1, head_dim=2, window=3, rotary_base=BASE).to(
        F64
    )
    g = torch.Generator().manual_seed(7)
    x = torch.randn(1, 5, 4, generator=g, dtype=F64)
    empty = kv_cache_empty(1, 2, 2, 3, F64, CPU)
    _, filled = attn.forward(x, torch.arange(5).reshape(1, 5), empty)
    for cache in (empty, filled):
        y, after = attn.forward(
            torch.zeros(1, 0, 4, dtype=F64), torch.zeros(1, 0, dtype=torch.long), cache
        )
        assert y.shape == (1, 0, 4)
        assert torch.equal(after.k, cache.k)
        assert torch.equal(after.v, cache.v)
        assert torch.equal(after.pos, cache.pos)
    # validation still runs in front of the empty path
    with pytest.raises(ContractError):
        _ = attn.forward(
            torch.zeros(1, 0, 4, dtype=F64), torch.zeros(2, 0, dtype=torch.long), empty
        )


def test_rotary_rejects_wrong_shapes_dtypes_and_bases() -> None:
    x = torch.ones(1, 2, 1, 2)
    with pytest.raises(ContractError):
        _ = rotary(x, torch.zeros(2, 2, dtype=torch.long), BASE)  # batch 2 against batch 1
    with pytest.raises(ContractError):
        _ = rotary(x, torch.zeros(1, 3, dtype=torch.long), BASE)
    with pytest.raises(ContractError):
        _ = rotary(x, torch.zeros(1, 2), BASE)  # float positions
    with pytest.raises(ValueError, match="even"):
        _ = rotary(torch.ones(1, 2, 1, 3), positions(0, 1), BASE)
    for base in (1.0, 0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="base"):
            _ = rotary(x, positions(0, 1), base)
        with pytest.raises(ValueError, match="base"):
            _ = WindowedAttention(d=2, heads=1, kv_groups=1, head_dim=2, window=2, rotary_base=base)


def test_rotary_with_a_non_default_base() -> None:
    x = torch.zeros(1, 1, 1, 4, dtype=F64)
    x[0, 0, 0, 0] = 1.0
    x[0, 0, 0, 2] = 1.0
    r = rotary(x, positions(3), 2.0)[0, 0, 0]
    a0, a1 = 3.0, 3.0 / math.sqrt(2.0)
    expected = mat([[math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)]])[0]
    assert torch.allclose(r, expected, rtol=0, atol=1e-12)
    wrong_base = mat([[math.cos(3.0), math.sin(3.0), math.cos(0.03), math.sin(0.03)]])[0]
    assert not torch.allclose(r, wrong_base, rtol=0, atol=1e-6)
