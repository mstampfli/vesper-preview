"""``column.RMSNorm`` fused on CUDA: one kernel forward, one backward.

PRIMITIVE ``fused_rms_norm(x, weight, floor, eps)``: what ``RMSNorm.forward`` computes (each row
divided by its largest magnitude, floored at ``floor``, before squaring; the statistics and the
gain in float32; the result in ``x``'s dtype), as one Triton kernel per direction instead of
about six elementwise operations forward and twelve backward. Measured 2026-10-21 on the RTX
5090 (op census of the captured trainer's step): the norm wrote about 33 GiB per step, the
largest elementwise cost after the head. The forward keeps one float32 number per row,
``1 / sqrt(mean(x^2) + eps)``; the backward is the analytic gradient of
``x / sqrt(mean(x^2) + eps) * g`` (the gradient the reference's autograd computes, its scale
being a constant), the gain's summed in fixed per-block order (deterministic).

For: ``RMSNorm.forward`` on a CUDA tensor of float32, bfloat16 or float16. Not for: the CPU or
float64 (``RMSNorm``'s own operations stay the reference there and in every test).
"""

from typing import cast, override

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


def fused_rms_norm(x: Tensor, weight: Tensor, floor: float, eps: float) -> Tensor:
    """``[..., d] -> [..., d]`` in ``x``'s dtype; ``x`` on CUDA, ``weight [d]``."""
    rows = x.reshape(-1, x.shape[-1])
    out: Tensor = _FusedRMSNorm.apply(rows, weight, floor, eps)  # type: ignore[misc]  # pyright: ignore[reportAny]  # torch types apply as (*Any) -> Any
    return out.view(x.shape)


class _FusedRMSNorm(torch.autograd.Function):
    @override
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, weight: Tensor, floor: float, eps: float) -> Tensor:
        from vesper.kernels.norm_triton import rms_forward  # noqa: PLC0415  # the gpu extra

        rows = x.contiguous()
        y, inv = rms_forward(rows, weight, floor, eps)
        ctx.save_for_backward(rows, weight, inv)
        return y

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, dy: Tensor) -> tuple[Tensor, Tensor, None, None]:  # pyright: ignore[reportIncompatibleMethodOverride]  # torch: (*Any) -> Any
        from vesper.kernels.norm_triton import rms_backward  # noqa: PLC0415

        rows, weight, inv = cast(
            "tuple[Tensor, Tensor, Tensor]",
            ctx.saved_tensors,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        )
        dx, dg = rms_backward(rows, weight, dy.contiguous(), inv)
        return dx, dg.to(weight.dtype), None, None
