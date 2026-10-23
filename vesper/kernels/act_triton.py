"""The squared-ReLU kernels of ``vesper.kernels.act`` on CUDA (Triton; imported only there).

Elementwise over a flat contiguous tensor: forward ``relu(z)^2``, backward ``2 relu(z) g``, each
computed in float32 and rounded once to the tensor's dtype.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 4096


@triton.jit
def _forward_kernel(z: tl.tensor, y: tl.tensor, size: int, block: tl.constexpr) -> None:
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    inside = offsets < size
    h = tl.maximum(tl.load(z + offsets, mask=inside, other=0.0).to(tl.float32), 0.0)
    tl.store(y + offsets, (h * h).to(y.dtype.element_ty), mask=inside)


@triton.jit
def _backward_kernel(
    z: tl.tensor, g: tl.tensor, dz: tl.tensor, size: int, block: tl.constexpr
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
    inside = offsets < size
    h = tl.maximum(tl.load(z + offsets, mask=inside, other=0.0).to(tl.float32), 0.0)
    grad = tl.load(g + offsets, mask=inside, other=0.0).to(tl.float32)
    tl.store(dz + offsets, (2.0 * h * grad).to(dz.dtype.element_ty), mask=inside)


def relu_square_forward(z: Tensor) -> Tensor:
    """``relu(z)^2`` of a contiguous tensor, in its dtype."""
    y = torch.empty_like(z)
    size = z.numel()
    if size > 0:
        _forward_kernel[(triton.cdiv(size, BLOCK),)](z, y, size, BLOCK)
    return y


def relu_square_backward(z: Tensor, g: Tensor) -> Tensor:
    """``2 relu(z) g`` of contiguous tensors, in ``z``'s dtype."""
    dz = torch.empty_like(z)
    size = z.numel()
    if size > 0:
        _backward_kernel[(triton.cdiv(size, BLOCK),)](z, g, dz, size, BLOCK)
    return dz
