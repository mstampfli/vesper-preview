"""The two row kernels of ``vesper.kernels.nll`` on CUDA (Triton; imported only there).

``row_stats``: per row of a logits matrix, the float32 maximum ``top``, the shifted log-sum-exp
``log sum exp(x - top)`` and the negative log-likelihood of the row's target, in one pass over
the row (an online maximum and sum), without writing anything ``[N, V]``. ``row_grad_``: per
row, the loss gradient ``g * (softmax - onehot)`` computed in float32 from the logits and the
two saved statistics, written IN PLACE over the logits in their own dtype. The statistics stay
shifted in both directions: a row is only ever used as ``x - top`` (exact for the entries near
the top), so a common offset of the logits cancels before any rounding (maintainer B's review of
571b5f9: an absolute log-sum-exp stored in float32 lost it, rows offset by 1e8 gave a loss of 0
instead of log 4). One program per row, no atomics: deterministic. Both read rows with a
stride, so a logits matrix need only be row-contiguous; the targets must be contiguous.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK_V = 4096  # vocabulary entries per loop iteration of one row's program


@triton.jit
def _stats_kernel(  # noqa: PLR0913, PLR0917  # a kernel's parameters are its launch ABI
    logits: tl.tensor,
    targets: tl.tensor,
    top_out: tl.tensor,
    log_total_out: tl.tensor,
    nll_out: tl.tensor,
    stride: int,
    vocab: int,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * stride
    top = tl.full((), float("-inf"), tl.float32)
    total = tl.full((), 0.0, tl.float32)
    for start in range(0, vocab, block):
        offsets = start + tl.arange(0, block)
        x = tl.load(base + offsets, mask=offsets < vocab, other=float("-inf")).to(tl.float32)
        new_top = tl.maximum(top, tl.max(x, 0))
        total = total * tl.exp(top - new_top) + tl.sum(tl.exp(x - new_top), 0)
        top = new_top
    log_total = tl.log(total)
    target = tl.load(targets + row)
    picked = tl.load(base + target).to(tl.float32)
    tl.store(top_out + row, top)
    tl.store(log_total_out + row, log_total)
    tl.store(nll_out + row, (top - picked) + log_total)


@triton.jit
def _grad_kernel(  # noqa: PLR0913, PLR0917  # a kernel's parameters are its launch ABI
    logits: tl.tensor,
    targets: tl.tensor,
    top_in: tl.tensor,
    log_total_in: tl.tensor,
    grad_in: tl.tensor,
    stride: int,
    vocab: int,
    block: tl.constexpr,
) -> None:
    row = tl.program_id(0).to(tl.int64)
    base = logits + row * stride
    top = tl.load(top_in + row)
    log_total = tl.load(log_total_in + row)
    g = tl.load(grad_in + row)
    target = tl.load(targets + row)
    for start in range(0, vocab, block):
        offsets = start + tl.arange(0, block)
        inside = offsets < vocab
        x = tl.load(base + offsets, mask=inside, other=0.0).to(tl.float32)
        p = tl.exp((x - top) - log_total) * g
        p = tl.where(offsets == target, p - g, p)
        tl.store(base + offsets, p.to(base.dtype.element_ty), mask=inside)


def row_stats(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(top, log_total, nll)``, float32 ``[N]`` each, of ``logits [N, V]`` and targets ``[N]``.

    ``logits`` row-contiguous, ``targets`` contiguous (the kernel reads ``targets + row``).
    """
    n, vocab = logits.shape
    top, log_total, nll = (
        torch.empty(n, dtype=torch.float32, device=logits.device) for _ in range(3)
    )
    if n > 0:
        _stats_kernel[(n,)](
            logits, targets.contiguous(), top, log_total, nll, logits.stride(0), vocab, BLOCK_V
        )
    return top, log_total, nll


def row_grad_(
    logits: Tensor, targets: Tensor, top: Tensor, log_total: Tensor, grad: Tensor
) -> Tensor:
    """Overwrite ``logits [N, V]`` with ``grad[:, None] * (softmax - onehot)``; return it."""
    n, vocab = logits.shape
    if n > 0:
        _grad_kernel[(n,)](
            logits, targets.contiguous(), top, log_total, grad, logits.stride(0), vocab, BLOCK_V
        )
    return logits
