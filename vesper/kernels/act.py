"""The FFN's squared ReLU on CUDA: one kernel forward, one backward.

PRIMITIVE ``relu_square(z)``: ``relu(z)^2`` with the gradient ``2 relu(z) g``, each element
computed in float32 and rounded once to ``z``'s dtype. The same bits as ``h * h`` with
``h = relu(z)``: the product of two bfloat16 values is exact in float32, and so is doubling, so
one rounding of ``2 h g`` equals the sum of the two rounded ``h g`` autograd forms. Measured
2026-10-21 on the RTX 5090 (op census): the relu, the product and their four backward operations
wrote about 6 GiB per step; this writes one tensor each way and keeps only ``z``.

For: ``FFN.hidden`` on a CUDA tensor of float32, bfloat16 or float16. Not for: the CPU or
float64 (``h * h`` stays the reference there and in every test).
"""

from typing import cast, override

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx


def relu_square(z: Tensor) -> Tensor:
    """``relu(z)^2`` in ``z``'s dtype, ``z`` on CUDA."""
    out: Tensor = _ReluSquare.apply(z.contiguous())  # type: ignore[misc]  # pyright: ignore[reportAny]  # torch types apply as (*Any) -> Any
    return out


class _ReluSquare(torch.autograd.Function):
    @override
    @staticmethod
    def forward(ctx: FunctionCtx, z: Tensor) -> Tensor:
        from vesper.kernels.act_triton import relu_square_forward  # noqa: PLC0415  # the gpu extra

        ctx.save_for_backward(z)
        return relu_square_forward(z)

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]  # torch: (*Any) -> Any
        from vesper.kernels.act_triton import relu_square_backward  # noqa: PLC0415

        (z,) = cast(
            "tuple[Tensor]",
            ctx.saved_tensors,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        )
        return relu_square_backward(z, grad.contiguous())
