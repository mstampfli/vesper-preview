import math

import pytest
import torch
from torch import Tensor

from vesper.attention import KVCache, WindowedAttention, kv_cache_empty, kv_cache_push, rotary
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
