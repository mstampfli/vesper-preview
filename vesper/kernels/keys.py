"""``RecurrentSublayer.keys`` fused on CUDA: the causal key filter and the key normalization.

PRIMITIVE ``fused_keys(z, history, taps, heads)``: what ``RecurrentSublayer.keys`` computes for
its key (``u[t] = sum_r taps[r] * z[t - r]`` with the chunk's predecessors from ``history``, then
``normalize_keys`` per head; float32 arithmetic, the key in ``z``'s dtype), as one Triton kernel
forward and two backward instead of about twenty elementwise operations each way. Measured
2026-10-21 on the RTX 5090 (op census of the captured trainer's step): the filter and the
normalization wrote about 16 GiB per step, the largest elementwise cost left. The backward is the
reference's own gradient: the normalization's scale held constant (it is detached there), the
clamp at a squared norm of one passing gradient at and above it (``clamp_min``); the taps'
gradient is summed in fixed per-block order (deterministic). The history's gradient is formed
only when the history requires one (a carried state does not).

For: ``RecurrentSublayer.keys`` on a CUDA tensor of float32, bfloat16 or float16 with a filter.
Not for: the CPU, float64 or ``key_conv`` 0 (the module's own operations stay the reference
there and in every test).
"""

from typing import cast, override

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx

from vesper.recurrence import KEY_NORM_FLOOR


def fused_keys(z: Tensor, history: Tensor, taps: Tensor, heads: int) -> Tensor:
    """Normalized keys ``[b, n, heads, d / heads]`` in ``z``'s dtype (``z [b, n, d]`` on CUDA)."""
    out: Tensor = _FusedKeys.apply(z.contiguous(), history, taps, heads)  # type: ignore[misc]  # pyright: ignore[reportAny]  # torch types apply as (*Any) -> Any
    return out


class _FusedKeys(torch.autograd.Function):
    @override
    @staticmethod
    def forward(ctx: FunctionCtx, z: Tensor, history: Tensor, taps: Tensor, heads: int) -> Tensor:
        from vesper.kernels.keys_triton import keys_forward  # noqa: PLC0415  # the gpu extra

        hist = history.to(torch.float32).contiguous()
        wide = taps.to(torch.float32).contiguous()
        k = keys_forward(z, hist, wide, heads, KEY_NORM_FLOOR)
        ctx.save_for_backward(z, hist, wide, history, taps)
        return k

    @override
    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]  # torch: (*Any) -> Any
        ctx: FunctionCtx, grad_k: Tensor
    ) -> tuple[Tensor, Tensor | None, Tensor, None]:
        from vesper.kernels.keys_triton import keys_backward  # noqa: PLC0415

        z, hist, wide, history, taps = cast(
            "tuple[Tensor, Tensor, Tensor, Tensor, Tensor]",
            ctx.saved_tensors,  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        )
        heads = int(grad_k.shape[2])
        grad_z, grad_taps, grad_u = keys_backward(
            z, hist, wide, grad_k.contiguous(), heads, KEY_NORM_FLOOR
        )
        grad_history = (
            _history_gradient(grad_u, wide).to(history.dtype) if history.requires_grad else None
        )
        return grad_z, grad_history, grad_taps.to(taps.dtype), None


def _history_gradient(grad_u: Tensor, taps: Tensor) -> Tensor:
    """``[b, K-1, d]``: history row ``p`` feeds ``u[t]`` through tap ``r = K - 1 - p + t``."""
    count, n = int(taps.shape[0]), int(grad_u.shape[1])
    rows: list[Tensor] = []
    for p in range(count - 1):
        total = torch.zeros_like(grad_u[:, 0])
        for r in range(count - 1 - p, count):
            t = p - (count - 1) + r
            if t < n:
                total = total + taps[r] * grad_u[:, t]
        rows.append(total)
    return torch.stack(rows, dim=1)
