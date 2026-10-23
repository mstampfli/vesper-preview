"""The per-token next-token loss of a linear head, without float32 vocabulary copies.

PRIMITIVE ``linear_nll(x, weight, targets) -> [N]``: ``cross_entropy(x @ weight.T, targets)`` per
row, the value AND the gradients (to ``x`` and ``weight``). Measured 2026-10-21 on the RTX 5090
(the captured trainer, d512, 8 passes, vocabulary 32768): of a step's 0.49 s of kernel time, the
head's float32 logits copies, fills and its softmax forward and backward took about 170 ms beside
its 87 ms of matmuls (the project owner: "i want as MUCH of the gpu maxed out as possible"). Here
the logits exist only in the matmul's own dtype (bfloat16 under CUDA autocast, as ``F.linear`` gives
them), one ``[N, V]`` buffer at a time: the forward reduces each row to two float32 statistics, its
maximum ``top`` and the shifted log-sum-exp ``log sum exp(x - top)``, and its loss
``(top - x[target]) + log_total`` (``nll_triton.row_stats``), keeping nothing ``[N, V]``; the
backward recomputes the logits with the same matmul, overwrites them in place with
``g * (exp((x - top) - log_total) - onehot)`` computed in float32 (``nll_triton.row_grad_``)
and runs the two gradient matmuls, the weight's accumulated in float32. Kept shifted in both
directions, a common offset of a row's logits cancels exactly (maintainer B's review of 571b5f9: the
absolute log-sum-exp stored in float32 turned rows offset by 1e8 into a loss of 0 and an input
gradient of 3e8, and gave -4.24 already at 1e4). The rounding points are those of the autocast
path it replaces (bfloat16 logits, float32 softmax, the gradient rounded to bfloat16 before the
matmuls); the arithmetic between them is not the same instructions, so results agree to
rounding, not bitwise.

The compute dtype is chosen once, in the call, as autocast would choose it for ``F.linear``: the
autocast dtype of the inputs' device when autocast is on there and neither input is float64,
else ``promote_types(x, weight)``. Every matmul then takes operands already cast to it, so the
backward (which runs outside any autocast region) recomputes the forward's arithmetic exactly,
on the CPU as on CUDA (maintainer B: under CPU
autocast the recomputed logits were float32 against bfloat16 ones, and kept logits failed the
backward on a dtype mismatch). On a CPU tensor, or without Triton, the two row steps run as
torch operations (the reference the CUDA kernels are tested against).

Kept logits are consumed by the backward (overwritten in place with the gradient, then
released); a second backward through the same graph (``retain_graph``) recomputes them.

For: every next-token loss of the tied head over rows of positions (training objectives,
evaluations). Not for: a loss over a different weight per row, or a reduction other than per row
(the caller masks and sums).
"""

from typing import cast, override

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from vesper.contracts import Dims, check_ids, check_shape


def linear_nll(x: Tensor, weight: Tensor, targets: Tensor, *, keep_logits: bool = False) -> Tensor:
    """Per-row ``-log softmax(x @ weight.T)[target]``, ``[N]``, at least float32.

    Args:
        x: ``[N, d]`` rows (the head's normalized inputs), any float dtype.
        weight: ``[V, d]`` the output projection (the tied embedding).
        targets: ``int32|int64 [N]`` ids in ``[0, V)``.
        keep_logits: keep this call's logits for the backward pass instead of recomputing them
            there: one ``[N, V]`` buffer held per call until the backward, one matmul of four
            saved (the trainer's ``recompute=False``: memory for speed). The same values and
            gradients either way (the recomputed logits are the same matmul's).

    Raises:
        ContractError: on a shape mismatch or a target outside ``[0, V)`` (eager; under capture
            the id check is a device assert, ``contracts.check_ids``).
    """
    dims: Dims = {}
    check_shape("x", x, ("n", "d"), dims)
    check_shape("weight", weight, ("V", "d"), dims)
    ids = check_ids("targets", targets[None, :], ("b", "n"), dims, dims["V"])[0]
    compute = _compute_dtype(x, weight)
    operand = weight.detach().to(compute)
    # torch types Function.apply as (*Any) -> Any; the one untyped call of this module
    out: Tensor = _LinearNLL.apply(x.to(compute), weight, operand, ids, keep_logits)  # type: ignore[misc]  # pyright: ignore[reportAny]
    return out


def _compute_dtype(x: Tensor, weight: Tensor) -> torch.dtype:
    """What autocast would run ``F.linear(x, weight)`` in: it leaves float64 alone."""
    kind = x.device.type
    if torch.is_autocast_enabled(kind) and torch.float64 not in (x.dtype, weight.dtype):
        return torch.get_autocast_dtype(kind)
    return torch.promote_types(x.dtype, weight.dtype)


class _LinearNLL(torch.autograd.Function):
    """Keeps ``x``, the cast weight, the targets, each row's ``top`` and ``log_total``.

    Kept logits are held on the context (not saved tensors: the backward overwrites them in
    place and then drops them); without them, or on a second backward, the logits are recomputed
    with the same matmul.
    """

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        weight: Tensor,
        operand: Tensor,
        targets: Tensor,
        keep_logits: bool,  # Function.apply takes positional arguments only
    ) -> Tensor:
        logits = x @ operand.T
        top, log_total, nll = _row_stats(logits, targets)
        ctx.save_for_backward(x, operand, targets, top, log_total, weight)
        _hold(ctx, logits if keep_logits else None)
        return nll

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad: Tensor) -> tuple[Tensor, Tensor, None, None, None]:  # pyright: ignore[reportIncompatibleMethodOverride]  # torch: (*Any) -> Any
        x, operand, targets, top, log_total, weight = _saved(ctx)
        kept = _take(ctx)
        logits = kept if kept is not None else x @ operand.T
        _ = _row_grad_(logits, targets, top, log_total, grad.to(top.dtype).contiguous())
        dx = logits @ operand
        if logits.dtype in (torch.bfloat16, torch.float16):  # the weight's accumulated in float32
            dw = (
                torch.mm(logits.T, x, out_dtype=torch.float32)
                if logits.is_cuda
                else logits.T.float() @ x.float()
            )
        else:
            dw = logits.T @ x
        return dx, dw.to(weight.dtype), None, None, None


def _saved(ctx: FunctionCtx) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The six tensors ``_LinearNLL.forward`` saved, typed (torch leaves the attribute untyped)."""
    return cast(
        "tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]",
        ctx.saved_tensors,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    )


def _hold(ctx: FunctionCtx, logits: Tensor | None) -> None:
    """Keep ``logits`` on the context (not a saved tensor: the backward overwrites it in place)."""
    ctx.kept = logits  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


def _take(ctx: FunctionCtx) -> Tensor | None:
    """The kept logits, once: the first backward consumes them, a second one recomputes."""
    kept = cast("Tensor | None", ctx.kept)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    ctx.kept = None  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    return kept


def _row_stats(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``(top, log_total, nll)`` of each row, at least float32 (the module docstring)."""
    if logits.is_cuda and logits.dtype != torch.float64:
        from vesper.kernels.nll_triton import row_stats  # noqa: PLC0415  # the gpu extra

        return row_stats(logits, targets)
    wide = logits.to(torch.promote_types(logits.dtype, torch.float32))
    top = wide.amax(dim=-1)
    log_total = torch.logsumexp(wide - top[:, None], dim=-1)
    return top, log_total, (top - wide.gather(1, targets[:, None])[:, 0]) + log_total


def _row_grad_(
    logits: Tensor, targets: Tensor, top: Tensor, log_total: Tensor, grad: Tensor
) -> Tensor:
    """Overwrite ``logits`` with ``grad[:, None] * (softmax - onehot)``, computed wide."""
    if logits.is_cuda and logits.dtype != torch.float64:
        from vesper.kernels.nll_triton import row_grad_  # noqa: PLC0415

        return row_grad_(logits, targets, top, log_total, grad)
    wide = logits.to(top.dtype)
    p = torch.exp((wide - top[:, None]) - log_total[:, None]) * grad[:, None]
    _ = p.scatter_add_(1, targets[:, None], -grad[:, None].to(p.dtype))
    return logits.copy_(p)
