"""vesper.kernels.keys: ``RecurrentSublayer.keys`` on CUDA as fused kernels.

The oracle is the module's own reference arithmetic run on a CPU copy (pinned there against the
operator's literal definition by tests/test_key_conv.py), never the kernels: the key, the next
history and the gradients of the key input, the taps and (when it requires one) the history must
agree to float32 rounding, for chunks shorter than the filter too; the CUDA call must be the
fused path, equal itself captured and repeat.
"""

from collections.abc import Sequence
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.column import ColumnShape, RecurrentSublayer
from vesper.graphs import WarmedGraph, deterministic_algorithms

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need CUDA")
CUDA = torch.device("cuda")
K = 4
SHAPE = ColumnShape(
    d=128,
    heads=2,
    kv_groups=2,
    head_dim=64,
    window=16,
    rotary_base=10000.0,
    ffn_mult=2,
    block=16,
    key_conv=K,
)


def _sublayer() -> RecurrentSublayer:
    layer = RecurrentSublayer(SHAPE, torch.Generator().manual_seed(0))
    with torch.no_grad():  # asymmetric taps: a reversed or shifted filter shows
        r = torch.arange(K, dtype=torch.float32)[:, None]
        j = torch.arange(SHAPE.d, dtype=torch.float32)[None, :]
        assert layer.conv_k is not None
        _ = layer.conv_k.copy_((r + 1) * 0.5 - 0.003 * j + 0.01 * r * j)
    return layer


def _keys(layer: RecurrentSublayer, z: Tensor, history: Tensor, up: Tensor) -> list[Tensor]:
    zs, hs = z.clone().requires_grad_(), history.clone().requires_grad_()
    layer.zero_grad(set_to_none=True)
    k, nxt = layer.keys(zs, hs)
    torch.autograd.backward([(k.double() * up.double()).sum()])
    assert layer.conv_k is not None
    assert layer.conv_k.grad is not None
    assert zs.grad is not None
    assert hs.grad is not None
    return [k.detach(), nxt, zs.grad, layer.conv_k.grad.clone(), hs.grad]


@needs_cuda
@pytest.mark.parametrize("n", [1, 2, 3, 257])  # shorter than the filter, and a long chunk
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_fused_keys_equal_the_reference(n: int, dtype: torch.dtype) -> None:
    g = torch.Generator().manual_seed(n)
    z = (torch.randn(3, n, SHAPE.d, generator=g) * 2).to(dtype)
    z[0, 0] = 0.0  # a zero key input row: the floor
    history = torch.randn(3, K - 1, SHAPE.d, generator=g)
    up = torch.randn(3, n, SHAPE.heads, SHAPE.head_dim, generator=g)
    cpu = _sublayer()
    gpu = _sublayer().to(CUDA)
    want = _keys(cpu, z.float(), history, up)  # the reference, float32 on the CPU
    got = _keys(gpu, z.to(CUDA), history.to(CUDA), up.to(CUDA))
    bound = 2e-5 if dtype == torch.float32 else 8e-3  # bf16: the key's own final rounding
    names = ("key", "next history", "dz", "dtaps", "dhistory")
    for a, b, name in zip(got, want, names, strict=True):
        torch.testing.assert_close(a.float().cpu(), b.float(), rtol=bound, atol=bound, msg=name)


@needs_cuda
def test_the_cuda_call_is_the_fused_kernel_and_repeats_captured() -> None:
    layer = _sublayer().to(CUDA)
    z = torch.randn(2, 64, SHAPE.d, device=CUDA)
    history = torch.randn(2, K - 1, SHAPE.d, device=CUDA)
    k, _ = layer.keys(z.clone().requires_grad_(), history)
    names: list[str] = []
    node = k.grad_fn
    while node is not None and len(names) < 4:
        names.append(node.name())
        node = node.next_functions[0][0] if node.next_functions else None
    assert any("FusedKeys" in name for name in names), names  # not silently the reference
    del k, node  # an autograd graph kept alive across a capture breaks it

    def step(values: Sequence[Tensor]) -> list[Tensor]:
        zs = values[0].detach().requires_grad_()
        conv = layer.conv_k
        assert conv is not None
        conv.grad = None
        key, nxt = layer.keys(zs, values[1])
        torch.autograd.backward([(key * values[2]).sum()])
        grad_conv = cast("Tensor | None", conv.grad)  # set to None above
        assert zs.grad is not None
        assert grad_conv is not None
        return [key.detach(), nxt, zs.grad, grad_conv]

    up = torch.randn(2, 64, SHAPE.heads, SHAPE.head_dim, device=CUDA)
    with deterministic_algorithms():
        eager = [[v.clone() for v in step([z, history, up])] for _ in range(2)]
        graph = WarmedGraph(warm=2)
        captured = [[v.clone() for v in graph(step, [z, history, up])] for _ in range(3)]
    assert graph.captures == 1
    assert all(torch.equal(a, b) for a, b in zip(eager[0], eager[1], strict=True))
    for run in captured:
        assert all(torch.equal(a, b) for a, b in zip(run, eager[0], strict=True))
