"""The recurrent key filter of ``vesper.kernels.keys`` on CUDA (Triton; imported only there).

``keys_forward``: per token and head, ``u = sum_r taps[r] * src[t - r]`` (the causal depthwise
filter of ``RecurrentSublayer.keys``; rows before the chunk come from the history) and the
scale-safe L2 normalization of ``recurrence.normalize_keys``, in float32, one store in the key's
dtype. ``keys_backward_u``: per token and head, the filter output's gradient from the key's,
recomputing ``u`` (the normalization's scale held constant, as the reference detaches it; the
clamp at a squared norm of one passes gradient at and above it, as ``clamp_min`` does).
``keys_backward_z``: per batch row and block of tokens, the filter input's gradient and the
taps' gradient as fixed per-program partial sums (no atomics: deterministic).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

ROWS_PER_PROGRAM = 64  # tokens one input-gradient program walks (and sums the taps over)


@triton.jit
def _forward_kernel(  # noqa: PLR0913, PLR0917  # a kernel's parameters are its launch ABI
    z: tl.tensor,
    hist: tl.tensor,
    taps: tl.tensor,
    out: tl.tensor,
    n: int,
    width: int,
    floor: float,
    heads: int,
    taps_count: tl.constexpr,
    head_dim: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    b = row // n
    t = row % n
    j = head * head_dim + tl.arange(0, head_dim)
    u = tl.full((head_dim,), 0.0, tl.float32)
    for r in tl.static_range(taps_count):
        src = t - r
        from_z = tl.load(z + (b * n + src) * width + j, mask=(j < width) & (src >= 0), other=0.0)
        hist_row = taps_count - 1 + src
        from_hist = tl.load(
            hist + (b * (taps_count - 1) + hist_row) * width + j,
            mask=(j < width) & (src < 0),
            other=0.0,
        )
        tap = tl.load(taps + r * width + j)
        u = u + tap * (from_z.to(tl.float32) + from_hist.to(tl.float32))
    scale = tl.maximum(tl.max(tl.abs(u), 0), floor)
    s = u / scale
    den = tl.sqrt(tl.maximum(tl.sum(s * s, 0), 1.0))
    tl.store(out + row * (heads * head_dim) + j, (s / den).to(out.dtype.element_ty))


@triton.jit
def _backward_u_kernel(  # noqa: PLR0913, PLR0917
    z: tl.tensor,
    hist: tl.tensor,
    taps: tl.tensor,
    grad_k: tl.tensor,
    grad_u: tl.tensor,
    n: int,
    width: int,
    floor: float,
    taps_count: tl.constexpr,
    head_dim: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    b = row // n
    t = row % n
    j = head * head_dim + tl.arange(0, head_dim)
    u = tl.full((head_dim,), 0.0, tl.float32)
    for r in tl.static_range(taps_count):
        src = t - r
        from_z = tl.load(z + (b * n + src) * width + j, mask=(j < width) & (src >= 0), other=0.0)
        hist_row = taps_count - 1 + src
        from_hist = tl.load(
            hist + (b * (taps_count - 1) + hist_row) * width + j,
            mask=(j < width) & (src < 0),
            other=0.0,
        )
        tap = tl.load(taps + r * width + j)
        u = u + tap * (from_z.to(tl.float32) + from_hist.to(tl.float32))
    scale = tl.maximum(tl.max(tl.abs(u), 0), floor)
    s = u / scale
    sq = tl.sum(s * s, 0)
    den = tl.sqrt(tl.maximum(sq, 1.0))
    g = tl.load(grad_k + row * width + j).to(tl.float32)
    dot = tl.sum(g * s, 0)
    through = tl.where(sq >= 1.0, s * (dot / (den * den * den)), 0.0)
    tl.store(grad_u + row * width + j, (g / den - through) / scale)


@triton.jit
def _backward_z_kernel(  # noqa: PLR0913, PLR0917
    z: tl.tensor,
    hist: tl.tensor,
    taps: tl.tensor,
    grad_u: tl.tensor,
    grad_z: tl.tensor,
    tap_partial: tl.tensor,
    n: int,
    width: int,
    taps_count: tl.constexpr,
    taps_pad: tl.constexpr,
    block: tl.constexpr,
    per_program: tl.constexpr,
) -> None:
    b = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1).to(tl.int64)
    j = tl.arange(0, block)
    inside = j < width
    which = tl.arange(0, taps_pad)
    acc = tl.full((taps_pad, block), 0.0, tl.float32)
    for i in range(per_program):
        t = part * per_program + i
        live = inside & (t < n)
        g = tl.load(grad_u + (b * n + t) * width + j, mask=live, other=0.0)
        dz = tl.full((block,), 0.0, tl.float32)
        for r in tl.static_range(taps_count):
            tap = tl.load(taps + r * width + j, mask=inside, other=0.0)
            later = tl.load(
                grad_u + (b * n + t + r) * width + j, mask=live & (t + r < n), other=0.0
            )
            dz = dz + tap * later
            src = t - r
            from_z = tl.load(z + (b * n + src) * width + j, mask=live & (src >= 0), other=0.0)
            hist_row = taps_count - 1 + src
            from_hist = tl.load(
                hist + (b * (taps_count - 1) + hist_row) * width + j,
                mask=live & (src < 0),
                other=0.0,
            )
            term = g * (from_z.to(tl.float32) + from_hist.to(tl.float32))
            acc = acc + tl.where(which[:, None] == r, term[None, :], 0.0)
        tl.store(grad_z + (b * n + t) * width + j, dz.to(grad_z.dtype.element_ty), mask=live)
    program = b * tl.num_programs(1) + part
    tl.store(
        tap_partial + (program * taps_pad + which[:, None]) * width + j[None, :],
        acc,
        mask=(which[:, None] < taps_count) & inside[None, :],
    )


def keys_forward(z: Tensor, hist: Tensor, taps: Tensor, heads: int, floor: float) -> Tensor:
    """Normalized keys ``[b, n, heads, dk]`` in ``z``'s dtype; all inputs contiguous."""
    b, n, width = z.shape
    head_dim = width // heads
    out = torch.empty(b, n, heads, head_dim, dtype=z.dtype, device=z.device)
    _forward_kernel[(b * n, heads)](
        z, hist, taps, out, n, width, floor, heads, int(taps.shape[0]), head_dim
    )
    return out


def keys_backward(
    z: Tensor, hist: Tensor, taps: Tensor, grad_k: Tensor, heads: int, floor: float
) -> tuple[Tensor, Tensor, Tensor]:
    """``(grad_z, grad_taps, grad_u)``: ``grad_z`` in ``z``'s dtype, the others float32."""
    b, n, width = z.shape
    count = int(taps.shape[0])
    grad_u = torch.empty(b, n, width, dtype=torch.float32, device=z.device)
    _backward_u_kernel[(b * n, heads)](
        z, hist, taps, grad_k, grad_u, n, width, floor, count, width // heads
    )
    grad_z = torch.empty_like(z)
    parts = -(-n // ROWS_PER_PROGRAM)
    pad = triton.next_power_of_2(count)
    partial = torch.empty(b * parts, pad, width, dtype=torch.float32, device=z.device)
    _backward_z_kernel[(b, parts)](
        z, hist, taps, grad_u, grad_z, partial, n, width, count, pad,
        triton.next_power_of_2(width), ROWS_PER_PROGRAM,
    )  # fmt: skip
    return grad_z, partial.sum(0)[:count], grad_u
