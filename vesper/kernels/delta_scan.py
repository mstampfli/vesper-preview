"""The delta-rule scan on CUDA, fused: flash-linear-attention's kernels behind our contract.

PRIMITIVE fused delta scan. ``delta_scan_fused`` and ``delta_scan_fused_fp32`` compute what
``vesper.recurrence.delta_scan_chunked`` computes (the reference, which stays the truth: every
test compares against it), with the forward AND the backward pass in Triton kernels from fla-core
0.5.2 (MIT, pinned in the ``gpu`` extra). Their interface is ours: ``q, k, v [b, n, h, dk]``,
``beta [b, n, h]``, the state ``[b, h, dk, dk]`` read as ``H^T q`` after the token's own write,
the gradient reaching the initial state too. We pass ``scale=1`` (the reference does not scale
the query) and keys already normalized by ``normalize_keys`` (the kernel's own normalization
stays off). One contract check (``_checked_length``) serves both.

``delta_scan_fused``: ``fla.ops.delta_rule.chunk_delta_rule``, inputs in bfloat16 (the kernel
refuses float32), the state in float32. Against a float64 reference fed the same bfloat16 inputs:
outputs, state and all five gradients within about 0.5 % relative (three bfloat16 roundings of
the output), the same at 1 and at 4096 tokens. The kernel rounds its intermediates to bfloat16
where the reference keeps float32; choosing it is choosing that (``RecurrentSublayer.scan``). On
pair 4/4 of stage 1 that rounding kept two-step lookups from being learned where the reference
policy learned them (docs/REPORTS.md, 2026-10-17).

``delta_scan_fused_fp32``: ``fla.ops.delta_rule.fused_recurrent_delta_rule``, one token at a
time, inputs AND state in float32. Against a float64 reference on the same inputs: outputs, state
and every gradient within about 2e-7 relative at stage-1 size and 4e-7 at 4096 tokens, the float32
reference's own error (measured 2026-10-17, 5060 Ti); about 1 ms forward and backward at stage-1
size (b 32, n 88, h 2, dk 16) where the chunked bfloat16 kernel takes 2 ms and the reference 60.

For: the training and prefill scan on a CUDA device. Not for: CPU or float64 arithmetic (use
the reference), or the one-token decode step (``delta_scan_step``).
"""

import warnings
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from vesper.contracts import ContractError, Dims, check_shape

# fla-core's chunked kernels tile the key dimension up to 256 (fla/ops/common/chunk_delta_h.py);
# the recurrent kernel holds the whole key dimension in one block and was measured to 512, so the
# chunked kernel's bound is the one bound for both.
MAX_HEAD_DIM = 256
INPUT_DTYPE = torch.bfloat16  # delta_scan_fused's q, k, v, beta
FP32_INPUT_DTYPE = torch.float32  # delta_scan_fused_fp32's q, k, v, beta
STATE_DTYPE = torch.float32


def fused_scan_available() -> bool:
    """Whether this environment can run the fused scans: a CUDA device and fla-core."""
    if not torch.cuda.is_available():
        return False
    try:
        _ = _kernels()
    except ImportError:
        return False
    return True


@runtime_checkable  # beartype checks the returns of the kernel getters against it
class _DeltaRule(Protocol):
    """The part of both kernels' signature this module calls (stubs/fla)."""

    def __call__(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        beta: Tensor,
        scale: float | None = None,
        initial_state: Tensor | None = None,
        output_final_state: bool = False,  # noqa: FBT001, FBT002  (the library's signature)
    ) -> tuple[Tensor, Tensor | None]: ...


def _kernels() -> tuple[_DeltaRule, _DeltaRule]:
    """``(chunk_delta_rule, fused_recurrent_delta_rule)``, imported on first use (gpu extra).

    Importing any part of fla imports all of ``fla.ops``, whose sparse-attention module warns
    that flash-attn is missing; we use no op that needs it, so that one notice is silenced here
    and nowhere else (the tests turn every warning into an error).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Flash Attention is not installed", category=ImportWarning
        )
        from fla.ops.delta_rule import (  # noqa: PLC0415
            chunk_delta_rule,
            fused_recurrent_delta_rule,
        )

    return chunk_delta_rule, fused_recurrent_delta_rule


def _checked_length(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor, input_dtype: torch.dtype
) -> int:
    """The one contract check of both fused scans; returns the token count ``n``.

    Raises:
        ContractError: on a shape or dtype mismatch (inputs ``input_dtype``, the state
            ``STATE_DTYPE``), a tensor off CUDA or on another device than ``q``, or a head
            dimension above ``MAX_HEAD_DIM``.
    """
    dims: Dims = {}
    check_shape("q", q, ("b", "n", "h", "dk"), dims, input_dtype)
    check_shape("k", k, ("b", "n", "h", "dk"), dims, input_dtype)
    check_shape("v", v, ("b", "n", "h", "dk"), dims, input_dtype)
    check_shape("beta", beta, ("b", "n", "h"), dims, input_dtype)
    check_shape("state", state, ("b", "h", "dk", "dk"), dims, STATE_DTYPE)
    if q.device.type != "cuda":
        msg = f"the fused scan runs on CUDA only, got q on {q.device}"
        raise ContractError(msg)
    for name, tensor in (("k", k), ("v", v), ("beta", beta), ("state", state)):
        if tensor.device != q.device:
            msg = f"{name} is on {tensor.device}, q on {q.device}"
            raise ContractError(msg)
    if dims["dk"] > MAX_HEAD_DIM:
        msg = f"head dimension {dims['dk']} exceeds the kernel's {MAX_HEAD_DIM}"
        raise ContractError(msg)
    return dims["n"]


def _run(
    kernel: _DeltaRule, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    o, final = kernel(q, k, v, beta, scale=1.0, initial_state=state, output_final_state=True)
    if final is None:  # output_final_state=True always returns it; a None is a library change
        msg = "fla's delta-rule kernel returned no final state"
        raise ContractError(msg)
    return o, final


def delta_scan_fused(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply ``n`` tokens to the state with the chunked bfloat16 kernel (the reference's form).

    Args:
        q: ``[b, n, h, dk]`` queries, bfloat16, on a CUDA device.
        k: ``[b, n, h, dk]`` keys, bfloat16, L2 norm at most one.
        v: ``[b, n, h, dk]`` values, bfloat16.
        beta: ``[b, n, h]`` write strengths in [0, 1], bfloat16.
        state: ``[b, h, dk, dk]`` the state before the first token, float32.

    Returns:
        ``(o, state_n)``: ``o [b, n, h, dk]`` in bfloat16 and the state after the last token in
        float32. With no tokens, the empty query tensor and the unchanged state, as the reference.

    Raises:
        ContractError: from ``_checked_length``, before the library is imported.
    """
    if _checked_length(q, k, v, beta, state, INPUT_DTYPE) == 0:
        return q, state
    return _run(_kernels()[0], q, k, v, beta, state)


def delta_scan_fused_fp32(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply ``n`` tokens to the state with the float32 recurrent kernel (the reference's result).

    Args:
        q: ``[b, n, h, dk]`` queries, float32, on a CUDA device.
        k: ``[b, n, h, dk]`` keys, float32, L2 norm at most one.
        v: ``[b, n, h, dk]`` values, float32.
        beta: ``[b, n, h]`` write strengths in [0, 1], float32.
        state: ``[b, h, dk, dk]`` the state before the first token, float32.

    Returns:
        ``(o, state_n)``: ``o [b, n, h, dk]`` and the state after the last token, both float32.
        With no tokens, the empty query tensor and the unchanged state, as the reference.

    Raises:
        ContractError: from ``_checked_length``, before the library is imported.
    """
    if _checked_length(q, k, v, beta, state, FP32_INPUT_DTYPE) == 0:
        return q, state
    if (
        torch.is_grad_enabled()
        and not state.requires_grad
        and any(t.requires_grad for t in (q, k, v, beta))
    ):
        # fla-core 0.5.2's recurrent backward writes the initial state's gradient whenever an
        # initial state is given, but allocates it only when that state requires grad: a cold
        # or detached state (every chunk's first pass) failed to compile its backward. A fresh
        # leaf takes the gradient, which nothing reads (tests: a state without grad).
        state = state.detach().requires_grad_(True)  # noqa: FBT003
    return _run(_kernels()[1], q, k, v, beta, state)
