"""The delta-rule scan on CUDA, fused: flash-linear-attention's chunked kernel behind our contract.

PRIMITIVE fused delta scan. ``delta_scan_fused`` computes what ``vesper.recurrence.
delta_scan_chunked`` computes (the reference, which stays the truth: every test compares
against it), with the forward AND the backward pass in Triton kernels from fla-core 0.5.2
(``fla.ops.delta_rule.chunk_delta_rule``, MIT, pinned in the ``gpu`` extra). Its interface is
ours: ``q, k, v [b, n, h, dk]``, ``beta [b, n, h]``, the state ``[b, h, dk, dk]`` read as
``H^T q`` after the token's own write, the gradient reaching the initial state too. We pass
``scale=1`` (the reference does not scale the query) and keys already normalized by
``normalize_keys`` (the kernel's own normalization stays off).

For: the training and prefill scan on a CUDA device, inputs in bfloat16, the state in float32.
Not for: CPU, float32 or float64 arithmetic (use the reference), or the one-token decode step
(``delta_scan_step``). Precision, measured on the 5060 Ti against a float64 reference fed the
same bfloat16 inputs: outputs, state and all five gradients within about 0.5 % relative (three
bfloat16 roundings of the output), the same at 1 and at 4096 tokens; the float32 reference is
within 1e-6. The kernel rounds its intermediates to bfloat16 where the reference keeps float32;
choosing it is choosing that (``RecurrentSublayer.scan``).
"""

import warnings
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from vesper.contracts import ContractError, Dims, check_shape

# fla-core's chunked kernels tile the key dimension up to 256 (fla/ops/common/chunk_delta_h.py).
MAX_HEAD_DIM = 256
INPUT_DTYPE = torch.bfloat16
STATE_DTYPE = torch.float32


def fused_scan_available() -> bool:
    """Whether this environment can run ``delta_scan_fused``: a CUDA device and fla-core."""
    if not torch.cuda.is_available():
        return False
    try:
        _ = _kernel()
    except ImportError:
        return False
    return True


@runtime_checkable  # beartype checks the return of _kernel against it
class _ChunkDeltaRule(Protocol):
    """The part of ``chunk_delta_rule``'s signature this module calls (stubs/fla)."""

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


def _kernel() -> _ChunkDeltaRule:
    """``fla.ops.delta_rule.chunk_delta_rule``, imported on first use (the gpu extra only).

    Importing any part of fla imports all of ``fla.ops``, whose sparse-attention module warns
    that flash-attn is missing; we use no op that needs it, so that one notice is silenced here
    and nowhere else (the tests turn every warning into an error).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Flash Attention is not installed", category=ImportWarning
        )
        from fla.ops.delta_rule import chunk_delta_rule  # noqa: PLC0415

    return chunk_delta_rule


def delta_scan_fused(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply ``n`` tokens to the state with the fused kernel (the reference's training form).

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
        ContractError: on a shape or dtype mismatch, a tensor off CUDA or on another device than
            ``q``, or a head dimension above ``MAX_HEAD_DIM``.
    """
    dims: Dims = {}
    check_shape("q", q, ("b", "n", "h", "dk"), dims, INPUT_DTYPE)
    check_shape("k", k, ("b", "n", "h", "dk"), dims, INPUT_DTYPE)
    check_shape("v", v, ("b", "n", "h", "dk"), dims, INPUT_DTYPE)
    check_shape("beta", beta, ("b", "n", "h"), dims, INPUT_DTYPE)
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
    if dims["n"] == 0:
        return q, state
    o, final = _kernel()(q, k, v, beta, scale=1.0, initial_state=state, output_final_state=True)
    if final is None:  # output_final_state=True always returns it; a None is a library change
        msg = "fla's chunk_delta_rule returned no final state"
        raise ContractError(msg)
    return o, final
