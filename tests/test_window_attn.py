"""vesper.kernels.window_attn: ``WindowedAttention.attend`` on CUDA as fla's windowed kernel.

The oracle is the blocked ``scaled_dot_product_attention`` path with its position mask, run on a
CPU float32 copy (tests/test_attention.py pins it against a literal per-query softmax): output,
new cache and the gradients of q, k and v must agree to bf16 rounding with an empty, a partly
filled and a full cache, chunks shorter and longer than the window; the CUDA call must be the
kernel, refuse positions with a gap, equal itself captured and repeat.
"""

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

import vesper.attention
from vesper.attention import KVCache, WindowedAttention, kv_cache_empty, kv_cache_push
from vesper.contracts import ContractError
from vesper.graphs import WarmedGraph, deterministic_algorithms
from vesper.kernels.window_attn import window_attention

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs CUDA")
CUDA = torch.device("cuda")
CPU = torch.device("cpu")
H, KV_GROUPS, DK, W = 4, 2, 64, 16
D = H * DK


def _layer(device: torch.device) -> WindowedAttention:
    return WindowedAttention(D, H, KV_GROUPS, DK, W, 10000.0, torch.Generator().manual_seed(0)).to(
        device
    )


def _inputs(b: int, n: int, m: int, start: int) -> tuple[list[Tensor], KVCache]:
    g = torch.Generator().manual_seed(n * 100 + m)
    hkv = H // KV_GROUPS
    q = torch.randn(b, n, H, DK, generator=g)
    k = torch.randn(b, n, hkv, DK, generator=g)
    v = torch.randn(b, n, hkv, DK, generator=g)
    pos = (torch.arange(n) + start + m).expand(b, n).contiguous()
    cache = kv_cache_empty(b, hkv, DK, W, torch.float32, CPU)
    if m > 0:
        ck = torch.randn(b, m, hkv, DK, generator=g)
        cv = torch.randn(b, m, hkv, DK, generator=g)
        cache = kv_cache_push(cache, ck, cv, (torch.arange(m) + start).expand(b, m).contiguous())
    up = torch.randn(b, n, D, generator=g)
    return [q, k, v, pos, up], cache


def _on(cache: KVCache, device: torch.device) -> KVCache:
    return KVCache(cache.k.to(device), cache.v.to(device), cache.pos.to(device), cache.window)


def _attend(layer: WindowedAttention, values: Sequence[Tensor], cache: KVCache) -> list[Tensor]:
    q, k, v = (t.clone().requires_grad_() for t in values[:3])
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=q.is_cuda):
        qq, kk, vv = (t.bfloat16() if t.is_cuda else t for t in (q, k, v))
        y, new = layer.attend(qq, kk, vv, values[3], cache)
    torch.autograd.backward([(y.float() * values[4]).sum()])
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    return [y.detach().float(), new.k.float(), new.pos, q.grad, k.grad, v.grad]


@needs_cuda
@pytest.mark.parametrize(("n", "m"), [(8, 0), (40, 0), (8, 5), (40, 5), (8, W), (40, W)])
def test_the_kernel_equals_the_blocked_attention(n: int, m: int) -> None:
    values, cache = _inputs(3, n, m, start=7)
    want = _attend(_layer(CPU), values, cache)
    got = _attend(_layer(CUDA), [t.to(CUDA) for t in values], _on(cache, CUDA))
    names = ("y", "cache k", "cache pos", "dq", "dk", "dv")
    for a, b, name in zip(got, want, names, strict=True):
        if name == "cache pos":
            assert torch.equal(a.cpu(), b), name
            continue
        torch.testing.assert_close(a.cpu().float(), b.float(), rtol=3e-2, atol=3e-2, msg=name)


@needs_cuda
def test_a_gap_in_the_positions_is_refused() -> None:
    values, cache = _inputs(2, 8, W, start=0)
    gapped = values[3] + 1  # the chunk starts one past the cache's next position
    layer = _layer(CUDA)
    q, k, v = (t.to(CUDA).bfloat16() for t in values[:3])
    with pytest.raises(ContractError, match="consecutive"):
        _ = layer.attend(q, k, v, gapped.to(CUDA), _on(cache, CUDA))


@needs_cuda
def test_the_cuda_call_is_the_kernel_and_repeats_captured(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    kernel = window_attention

    def spy(q: Tensor, keys: Tensor, values: Tensor, window: int) -> Tensor:
        calls.append(window)
        return kernel(q, keys, values, window)

    monkeypatch.setattr(vesper.attention, "window_attention", spy)
    values, cache = _inputs(2, 40, W, start=3)
    layer = _layer(CUDA)
    dev = [t.to(CUDA) for t in values]
    cached = _on(cache, CUDA)

    def step(vals: Sequence[Tensor]) -> list[Tensor]:
        q, k, v = (t.detach().bfloat16().requires_grad_() for t in vals[:3])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, _ = layer.attend(q, k, v, vals[3], cached)
        torch.autograd.backward([(y.float() * vals[4]).sum()])
        grads = [t.grad for t in (q, k, v)]
        assert all(g is not None for g in grads)
        return [y.detach(), *(g for g in grads if g is not None)]

    with deterministic_algorithms():
        eager = [[t.clone() for t in step(dev)] for _ in range(2)]
        graph = WarmedGraph(warm=2)
        captured = [[t.clone() for t in graph(step, dev)] for _ in range(3)]
    assert calls  # the kernel, not the blocked path
    assert graph.captures == 1
    assert all(torch.equal(a, b) for a, b in zip(eager[0], eager[1], strict=True))
    for run in captured:
        assert all(torch.equal(a, b) for a, b in zip(run, eager[0], strict=True))
