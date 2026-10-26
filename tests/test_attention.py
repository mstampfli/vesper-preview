import math

import pytest
import torch
from torch import Tensor

from vesper.attention import (
    KVCache,
    RingKVCache,
    WindowedAttention,
    kv_cache_empty,
    kv_cache_push,
    kv_ring_empty,
    kv_ring_write,
    rotary,
)
from vesper.contracts import ContractError

D, H, KV_GROUPS, DK, W = 32, 4, 2, 8, 16
BASE = 10000.0
HKV = H // KV_GROUPS
CPU = torch.device("cpu")
F64 = torch.float64


def _attn(seed: int = 0) -> WindowedAttention:
    g = torch.Generator().manual_seed(seed)
    return WindowedAttention(
        d=D, heads=H, kv_groups=KV_GROUPS, head_dim=DK, window=W, rotary_base=BASE, generator=g
    ).to(F64)


def _inputs(b: int, n: int, start: int, seed: int = 1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, D, generator=g, dtype=F64)
    pos = torch.arange(start, start + n).expand(b, n)
    return x, pos


def _empty(b: int) -> KVCache:
    return kv_cache_empty(b, HKV, DK, W, F64, CPU)


def _oracle(attn: WindowedAttention, x: Tensor, pos: Tensor, cache: KVCache) -> Tensor:
    """Full attention over cache + chunk with the explicit window mask, no blocking."""
    q, k, v = attn.project(x, pos)
    keys = torch.cat([cache.k, k], dim=1).repeat_interleave(KV_GROUPS, dim=2)
    vals = torch.cat([cache.v, v], dim=1).repeat_interleave(KV_GROUPS, dim=2)
    kpos = torch.cat([cache.pos, pos], dim=1)
    # [b, h, i, j] scores by plain matmul (einsum is untyped in the stubs).
    scores = (q.permute(0, 2, 1, 3) @ keys.permute(0, 2, 3, 1)) / math.sqrt(DK)
    allowed = (kpos[:, None, :] <= pos[:, :, None]) & (kpos[:, None, :] > pos[:, :, None] - W)
    scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
    out = scores.softmax(dim=-1) @ vals.permute(0, 2, 1, 3)  # [b, h, i, dk]
    return attn.output(out.permute(0, 2, 1, 3))


def test_blocked_equals_oracle_with_prefilled_cache() -> None:
    attn = _attn()
    b, m, n = 2, 20, 130
    xc, pc = _inputs(b, m, 0, seed=2)
    _, cache = attn.forward(xc, pc, _empty(b))
    x, pos = _inputs(b, n, m, seed=3)
    y, _ = attn.forward(x, pos, cache)
    assert (y - _oracle(attn, x, pos, cache)).abs().max().item() < 1e-10


def test_two_halves_equal_whole() -> None:
    attn = _attn()
    x, pos = _inputs(1, 70, 5)
    whole, _ = attn.forward(x, pos, _empty(1))
    first, cache = attn.forward(x[:, :33], pos[:, :33], _empty(1))
    second, _ = attn.forward(x[:, 33:], pos[:, 33:], cache)
    assert (torch.cat([first, second], dim=1) - whole).abs().max().item() < 1e-10


def test_window_boundary_is_exact() -> None:
    attn = _attn()
    x, pos = _inputs(1, 3 * W, 0)
    y, _ = attn.forward(x, pos, _empty(1))
    i = 2 * W + 3
    # A token W positions back is outside the window: no effect on token i.
    far = x.clone()
    far[:, i - W] += 1.0
    y_far, _ = attn.forward(far, pos, _empty(1))
    assert (y_far[:, i] - y[:, i]).abs().max().item() < 1e-12
    # A token W - 1 positions back is the oldest inside the window: it must matter.
    near = x.clone()
    near[:, i - W + 1] += 1.0
    y_near, _ = attn.forward(near, pos, _empty(1))
    assert (y_near[:, i] - y[:, i]).abs().max().item() > 1e-6


def test_causal_no_future_leakage() -> None:
    attn = _attn()
    x, pos = _inputs(1, 40, 0)
    y, _ = attn.forward(x, pos, _empty(1))
    j = 17
    bumped = x.clone()
    bumped[:, j] += 1.0
    y_b, _ = attn.forward(bumped, pos, _empty(1))
    assert (y_b[:, :j] - y[:, :j]).abs().max().item() < 1e-12
    assert (y_b[:, j:] - y[:, j:]).abs().max().item() > 1e-6


def test_rotary_makes_attention_relative() -> None:
    attn = _attn()
    x, pos = _inputs(1, 50, 0)
    y0, _ = attn.forward(x, pos, _empty(1))
    y1, _ = attn.forward(x, pos + 1000, _empty(1))
    assert (y1 - y0).abs().max().item() < 1e-9


def test_rotary_preserves_norm_and_is_identity_at_zero() -> None:
    g = torch.Generator().manual_seed(4)
    x = torch.randn(1, 5, 2, DK, generator=g, dtype=F64)
    pos = torch.arange(5).expand(1, 5)
    r = rotary(x, pos, BASE)
    assert (r.square().sum(-1) - x.square().sum(-1)).abs().max().item() < 1e-12
    assert torch.equal(rotary(x, torch.zeros(1, 5, dtype=torch.long), BASE), x)


def test_cache_keeps_the_last_window() -> None:
    g = torch.Generator().manual_seed(5)
    k = torch.randn(1, 2 * W, HKV, DK, generator=g, dtype=F64)
    v = torch.randn(1, 2 * W, HKV, DK, generator=g, dtype=F64)
    pos = torch.arange(2 * W).expand(1, 2 * W)
    cache = kv_cache_push(_empty(1), k, v, pos)
    assert cache.k.shape[1] == W
    assert torch.equal(cache.pos[0], torch.arange(W, 2 * W))
    assert torch.equal(cache.k, k[:, W:])


def test_rejects_mismatched_positions() -> None:
    attn = _attn()
    x, pos = _inputs(1, 10, 0)
    with pytest.raises(ContractError):
        _ = attn.forward(x, pos[:, :9], _empty(1))


def test_rejects_non_consecutive_positions() -> None:
    attn = _attn()
    x, pos = _inputs(1, 10, 0)
    gappy = pos.clone()
    gappy[:, 5:] += 1
    with pytest.raises(ContractError, match="consecutive"):
        _ = attn.forward(x, gappy, _empty(1))


def test_a_ring_attends_by_position_not_by_slot_order() -> None:
    # Maintainer B's example (b8d0a5f): slots hold positions [3, 1, 2] with values [30, 10, 20]; the
    # new token is position 4, value 40; window 3 and equal logits. The window is positions 2..4, so
    # the mean is 30; taking the last three array entries ([10, 20, 40]) would give 23.33.
    attn = WindowedAttention(d=2, heads=1, kv_groups=1, head_dim=2, window=3, rotary_base=BASE)
    attn = attn.to(F64)
    ring = RingKVCache(
        k=torch.zeros(1, 3, 1, 2, dtype=F64),
        v=torch.tensor((30.0, 10.0, 20.0), dtype=F64).view(1, 3, 1, 1).expand(1, 3, 1, 2),
        pos=torch.tensor(((3, 1, 2),)),
        valid=torch.ones(1, 3, dtype=torch.bool),
        window=3,
    )
    q = torch.zeros(1, 1, 1, 2, dtype=F64)  # equal logits for every visible key
    k = torch.zeros(1, 1, 1, 2, dtype=F64)
    v = torch.full((1, 1, 1, 2), 40.0, dtype=F64)
    y, after = attn.attend_ring(q, k, v, torch.tensor(((4,),)), ring)
    expected = attn.output(torch.full((1, 1, 1, 2), 30.0, dtype=F64))
    torch.testing.assert_close(y, expected, rtol=0, atol=1e-12)
    # the new token took slot 4 % 3 = 1 (position 1 left the window), and is valid
    assert torch.equal(after.pos, torch.tensor(((3, 4, 2),)))
    assert float(after.v[0, 1, 0, 0]) == 40.0
    assert after.valid.all()


def test_ring_decoding_equals_the_growing_cache_and_the_whole_chunk_past_the_window() -> None:
    attn = _attn()
    steps = 3 * W + 5  # past the window, the ring wraps more than once
    x, pos = _inputs(2, steps, 7)
    whole, _ = attn.forward(x, pos, _empty(2))
    ring, cache = kv_ring_empty(2, HKV, DK, W, F64, CPU), _empty(2)
    for i in range(steps):
        q, k, v = attn.project(x[:, i : i + 1], pos[:, i : i + 1])
        y_ring, ring = attn.attend_ring(q, k, v, pos[:, i : i + 1], ring)
        y_grow, cache = attn.attend(q, k, v, pos[:, i : i + 1], cache)
        torch.testing.assert_close(y_ring, y_grow, rtol=0, atol=1e-12)
        torch.testing.assert_close(y_ring, whole[:, i : i + 1], rtol=0, atol=1e-12)
        # the ring holds exactly the growing cache's entries, wherever they sit
        held = ring.valid[0]
        assert torch.equal(ring.pos[0][held].sort().values, cache.pos[0])
    assert ring.k.shape == (2, W, HKV, DK)  # never reallocated to another shape


def test_ring_rows_write_their_own_slots() -> None:
    ring = kv_ring_empty(2, HKV, DK, 4, F64, CPU)
    k = torch.ones(2, 1, HKV, DK, dtype=F64)
    v = torch.stack([torch.full((1, HKV, DK), 1.0), torch.full((1, HKV, DK), 2.0)]).to(F64)
    after = kv_ring_write(ring, k, v, torch.tensor(((5,), (6,))))
    written = torch.tensor(((False, True, False, False), (False, False, True, False)))
    assert torch.equal(after.valid, written)
    assert after.pos[0, 1] == 5
    assert after.pos[1, 2] == 6
    assert float(after.v[1, 2, 0, 0]) == 2.0
    assert float(after.v[0, 2].abs().sum()) == 0.0  # unwritten slots stay finite zeros
    assert not ring.valid.any()  # the input ring is unchanged


def test_a_ring_refuses_chunks_other_windows_and_negative_positions() -> None:
    attn = _attn()
    x, pos = _inputs(1, 2, 0)
    q, k, v = attn.project(x, pos)
    ring = kv_ring_empty(1, HKV, DK, W, F64, CPU)
    with pytest.raises(ContractError):
        _ = attn.attend_ring(q, k, v, pos, ring)  # two tokens
    other = kv_ring_empty(1, HKV, DK, W + 1, F64, CPU)
    with pytest.raises(ContractError):
        _ = attn.attend_ring(q[:, :1], k[:, :1], v[:, :1], pos[:, :1], other)
    with pytest.raises(ContractError, match="nonnegative"):
        _ = kv_ring_write(ring, k[:, :1], v[:, :1], torch.tensor(((-1,),)))
