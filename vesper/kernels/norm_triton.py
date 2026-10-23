"""The fused RMSNorm row kernels of ``vesper.kernels.norm`` on CUDA (Triton; imported only there).

``rms_forward``: per row, in float32, the scale-safe statistics of ``column.RMSNorm`` (the row
divided by its largest magnitude, floored, before squaring) and ``y = x / rms * g`` stored in
the output's dtype, plus ``inv = 1 / sqrt(mean(x^2) + eps)`` for the backward; one read of the
row, one write. ``rms_backward``: per row, ``dx = inv * (dy g - xhat mean(dy g xhat))`` with
``xhat = x inv``, and the gain's gradient as fixed per-program partial sums over consecutive rows
(summed in order by the caller): no atomics, deterministic. A row is one program's block, so the
width must fit ``next_power_of_2(d)`` lanes (the model's widths do).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

ROWS_PER_PROGRAM = 32  # rows one backward program sums the gain's gradient over


@triton.jit
def _forward_kernel(  # noqa: PLR0913, PLR0917  # a kernel's parameters are its launch ABI
    x: tl.tensor,
    weight: tl.tensor,
    y: tl.tensor,
    inv_out: tl.tensor,
    stride_x: int,
    stride_y: int,
    width: int,
    floor: float,
    eps: float,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block)
    inside = offsets < width
    v = tl.load(x + row * stride_x + offsets, mask=inside, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(v), 0), floor)
    z = v / scale
    rms = tl.sqrt(tl.sum(z * z, 0) / width + eps / (scale * scale))
    g = tl.load(weight + offsets, mask=inside, other=0.0).to(tl.float32)
    out = z / rms * g
    tl.store(y + row * stride_y + offsets, out.to(y.dtype.element_ty), mask=inside)
    tl.store(inv_out + row, 1.0 / (scale * rms))


@triton.jit
def _backward_kernel(  # noqa: PLR0913, PLR0917  # a kernel's parameters are its launch ABI
    x: tl.tensor,
    weight: tl.tensor,
    dy: tl.tensor,
    inv_in: tl.tensor,
    dx: tl.tensor,
    dg_partial: tl.tensor,
    stride_x: int,
    stride_dy: int,
    stride_dx: int,
    rows: int,
    width: int,
    block: tl.constexpr,
    per_program: tl.constexpr,
) -> None:
    program = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, block)
    inside = offsets < width
    g = tl.load(weight + offsets, mask=inside, other=0.0).to(tl.float32)
    acc = tl.full((block,), 0.0, tl.float32)
    for i in range(per_program):
        row = program * per_program + i
        live = inside & (row < rows)
        v = tl.load(x + row * stride_x + offsets, mask=live, other=0.0).to(tl.float32)
        d = tl.load(dy + row * stride_dy + offsets, mask=live, other=0.0).to(tl.float32)
        inv = tl.load(inv_in + row, mask=row < rows, other=0.0)
        xhat = v * inv
        dg = d * g
        mean = tl.sum(dg * xhat, 0) / width
        tl.store(
            dx + row * stride_dx + offsets,
            (inv * (dg - xhat * mean)).to(dx.dtype.element_ty),
            mask=live,
        )
        acc = acc + d * xhat
    tl.store(dg_partial + program * width + offsets, acc, mask=inside)


def rms_forward(x: Tensor, weight: Tensor, floor: float, eps: float) -> tuple[Tensor, Tensor]:
    """``(y, inv)`` of row-contiguous ``x [N, d]``: ``y`` in ``x``'s dtype, ``inv`` float32."""
    n, width = x.shape
    y = torch.empty_like(x)
    inv = torch.empty(n, dtype=torch.float32, device=x.device)
    if n > 0:
        block = triton.next_power_of_2(width)
        _forward_kernel[(n,)](x, weight, y, inv, x.stride(0), y.stride(0), width, floor, eps, block)
    return y, inv


def rms_backward(x: Tensor, weight: Tensor, dy: Tensor, inv: Tensor) -> tuple[Tensor, Tensor]:
    """``(dx, dg)``: the input's gradient in ``x``'s dtype and the gain's in float32."""
    n, width = x.shape
    dx = torch.empty_like(x)
    programs = -(-n // ROWS_PER_PROGRAM)
    partial = torch.empty(max(programs, 1), width, dtype=torch.float32, device=x.device)
    if n == 0:
        return dx, torch.zeros(width, dtype=torch.float32, device=x.device)
    block = triton.next_power_of_2(width)
    _backward_kernel[(programs,)](
        x, weight, dy, inv, dx, partial, x.stride(0), dy.stride(0), dx.stride(0), n, width, block,
        ROWS_PER_PROGRAM,
    )  # fmt: skip
    return dx, partial.sum(0)
