"""vesper.kernels.norm: ``RMSNorm`` on CUDA as one fused kernel per direction.

The oracle is the module's own reference arithmetic run on a CPU copy (the path every CPU test
pins: scale-safe, maintainer B's boundary cases of 10-15), never the kernel: values and both
gradients must agree to float32 rounding, at the range boundaries too; the CUDA call must take the
fused path (not silently the reference), equal itself captured and repeat.
"""

from collections.abc import Sequence
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.column import RMSNorm
from vesper.graphs import WarmedGraph, deterministic_algorithms

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel needs CUDA")
CUDA = torch.device("cuda")


def _run(norm: RMSNorm, x: Tensor, up: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    xs = x.clone().requires_grad_()
    norm.zero_grad(set_to_none=True)
    y = norm.forward(xs)
    torch.autograd.backward([(y.double() * up.double()).sum()])
    assert xs.grad is not None
    assert norm.weight.grad is not None
    return y.detach(), xs.grad, norm.weight.grad.clone()


def _pair(x: Tensor, gain: Tensor) -> tuple[list[Tensor], list[Tensor]]:
    up = torch.randn(x.shape, generator=torch.Generator().manual_seed(1))
    cpu, gpu = RMSNorm(x.shape[-1]), RMSNorm(x.shape[-1]).to(CUDA)
    with torch.no_grad():
        _ = cpu.weight.copy_(gain)
        _ = gpu.weight.copy_(gain)
    want = [t.double() for t in _run(cpu, x, up)]
    got = [t.double().cpu() for t in _run(gpu, x.to(CUDA), up.to(CUDA))]
    return got, want


def _rows(*rows: Tensor) -> Tensor:
    return torch.stack(rows)


@needs_cuda
@pytest.mark.parametrize("width", [512, 384, 33])  # the model's, another, one not a power of two
def test_values_and_both_gradients_equal_the_reference(width: int) -> None:
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, width, generator=g) * 3
    gain = 1 + 0.1 * torch.randn(width, generator=g)
    got, want = _pair(x, gain)
    for a, b, name in zip(got, want, ("y", "dx", "dg"), strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5, msg=name)


@needs_cuda
def test_the_range_boundaries_hold_as_in_the_reference() -> None:
    # Maintainer B's review of 10-15: rows near the float32 range, rows of zeros, a tiny constant
    # row
    width = 64
    x = _rows(
        torch.full((width,), 1e20),
        torch.zeros(width),
        torch.full((width,), 1e-4),
        torch.linspace(-1.0, 1.0, width) * 3e38,  # near float32's maximum (linspace itself
        # overflows computing its step between -3e38 and 3e38)
        torch.randn(width, generator=torch.Generator().manual_seed(2)),
    )
    got, want = _pair(x, torch.ones(width))
    assert torch.isfinite(got[0]).all()
    assert torch.isfinite(got[1]).all()
    torch.testing.assert_close(got[0], want[0], rtol=2e-5, atol=1e-6)
    torch.testing.assert_close(got[1], want[1], rtol=2e-5, atol=1e-6)
    assert float(got[0][2, 0]) == pytest.approx(0.0995, abs=5e-5)  # eps regularizes, not floors
    assert float(got[0][1].abs().max()) == 0.0


@needs_cuda
def test_bf16_activations_come_back_in_bf16_with_float32_statistics() -> None:
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(64, 512, generator=g) * 50).bfloat16()
    norm = RMSNorm(512).to(CUDA)
    y = norm.forward(x.to(CUDA))
    assert y.dtype == torch.bfloat16
    want = RMSNorm(512).forward(x.float()).bfloat16()  # float32 statistics, one rounding at the end
    torch.testing.assert_close(y.cpu().float(), want.float(), rtol=8e-3, atol=1e-6)


@needs_cuda
def test_the_cuda_call_is_the_fused_kernel_and_repeats_captured() -> None:
    norm = RMSNorm(512).to(CUDA)
    x = torch.randn(128, 512, device=CUDA)
    y = norm.forward(x.clone().requires_grad_())
    names: list[str] = []
    node = y.grad_fn
    while node is not None and len(names) < 4:
        names.append(node.name())
        node = node.next_functions[0][0] if node.next_functions else None
    assert any("FusedRMSNorm" in name for name in names), names  # not silently the reference
    del y, node  # an autograd graph kept alive across a capture breaks it (torch warns)

    def step(values: Sequence[Tensor]) -> list[Tensor]:
        xs = values[0].detach().requires_grad_()
        norm.weight.grad = None
        out = norm.forward(xs)
        torch.autograd.backward([(out * values[1]).sum()])
        grad_w = cast("Tensor | None", norm.weight.grad)  # set to None above
        assert xs.grad is not None
        assert grad_w is not None
        return [out.detach(), xs.grad, grad_w]

    up = torch.randn(128, 512, device=CUDA)
    with deterministic_algorithms():
        eager = [[v.clone() for v in step([x.detach(), up])] for _ in range(2)]
        graph = WarmedGraph(warm=2)
        captured = [[v.clone() for v in graph(step, [x.detach(), up])] for _ in range(3)]
    assert graph.captures == 1
    assert all(torch.equal(a, b) for a, b in zip(eager[0], eager[1], strict=True))
    for run in captured:
        assert all(torch.equal(a, b) for a, b in zip(run, eager[0], strict=True))
