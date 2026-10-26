"""Wake objective (spec, "Wake objective") and the per-token loss the modulator reads.

``L = L_ntp + lambda_pc * sum_c L_pc,c + lambda_h * L_ponder + lambda_g * L_gate``

Mode ``e2e`` (the milestone mode): ``lambda_pc = 0`` by the spec, so the loss is the
cross-entropy of the final, halting-weighted output plus the ponder and gate terms the loop
and the retrieval gate hand in as scalars. Mode ``local`` is refused here on purpose: the
spec's predictive-coding target (which pass of column ``c+1``'s input column ``c`` predicts)
is ambiguous, and that is a review question, not a guess (docs/reviews/2026-10-14-local-mode-
question.md).

``ntp_loss`` returns one value per token, unreduced and unshifted, because the modulator defines
its surprise as the loss of predicting token ``t``; the alignment of that loss to the arriving
token is the caller's contract and is spelled out on ``ntp_loss``.
"""

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from vesper.config import VesperConfig
from vesper.contracts import ContractError, Dims, check_ids, check_shape

Mode = Literal["e2e", "local"]


@dataclass(frozen=True)
class LossWeights:
    """The objective's weights, which travel together from the configuration.

    ``lambda_h`` may be a zero-dimensional device tensor: a captured training step reads the
    ramp's current value from it, filled in place before each replay (a float would be frozen
    into the graph at capture, ``vesper.halting_experiment.GraphedTraining``).
    """

    lambda_h: float | Tensor
    lambda_g: float


def warmup_cosine(
    step: int, steps: int, *, peak: float, warmup: int, final_fraction: float
) -> float:
    """PRIMITIVE the learning rate at ``step`` (from 0) of a run of ``steps``.

    Linear warm-up to ``peak`` over ``warmup`` steps (step ``i`` gets ``peak * (i + 1) /
    warmup``), then one cosine from ``peak`` to ``peak * final_fraction`` that the last step
    (index ``steps - 1``) reaches exactly, and holds there beyond it. For: every schedule of a
    training run (``vesper.train.phase_schedule``, ``vesper.instruct``). Not for: a schedule with
    restarts or a step below 0.
    """
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_steps = max(steps - 1 - warmup, 1)
    progress = min((step - warmup) / decay_steps, 1.0)
    final = peak * final_fraction
    return final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def loss_weights(cfg: VesperConfig) -> LossWeights:
    """``learning.lambda_h`` and ``learning.lambda_g`` from a validated configuration."""
    return LossWeights(lambda_h=cfg.learning.lambda_h, lambda_g=cfg.learning.lambda_g)


def ntp_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Per-token cross-entropy: ``[b, n, V]``, ``int32|int64[b, n] -> [b, n]``.

    Alignment: ``loss[i]`` scores ``logits[i]`` against ``targets[i]`` as given; nothing is
    shifted here. With the data stream's chunks, ``targets[i]`` is the token at stream position
    ``i + 1``, so ``loss[i]`` is the surprise of that next token, not of the token at ``i``. The
    modulator's surprise for the token arriving at position ``j`` is therefore ``loss[j - 1]``,
    with the previous chunk's last loss carried across the boundary for ``j = 0``. Feeding
    ``loss[j]`` into token ``j``'s own fast-state write would use the next token's surprise and
    leak a future target. An empty chunk gives an empty result.

    Raises:
        ContractError: on a shape mismatch, a target dtype other than int32 or int64, or a
            target outside ``[0, V)``.
    """
    dims: Dims = {}
    check_shape("logits", logits, ("b", "n", "V"), dims)
    ids = check_ids("targets", targets, ("b", "n"), dims, dims["V"])
    flat = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), reduction="none")
    return flat.view(ids.shape)


def wake_loss(
    logits: Tensor,
    targets: Tensor,
    mode: Mode,
    *,
    weights: LossWeights,
    ponder: Tensor | None = None,
    gate: Tensor | None = None,
    mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The wake objective and its parts.

    Args:
        logits: ``[b, n, V]`` of the final (halting-weighted) output.
        targets: ``int32|int64[b, n]`` next tokens.
        mode: ``"e2e"``; ``"local"`` raises until its target is pinned.
        weights: ``lambda_h`` and ``lambda_g`` from the configuration (``loss_weights(cfg)``). No
            default: a caller that omits them cannot train a different objective by accident.
        ponder: scalar ``L_ponder`` from the halting controller, or ``None`` when no loop ran.
        gate: scalar ``L_gate`` from the retrieval gate, or ``None`` when retrieval is off.
        mask: optional ``bool[b, n]``; only true positions enter the mean (the data stream's
            ``loss_mask``: padding and document boundaries carry no loss).

    Returns:
        The total and a dict of the weighted parts that were present: ``ntp``, ``ponder``, ``gate``.

    Raises:
        ContractError: if there is no position to aggregate (``b`` or ``n`` is zero, or the mask
            selects nothing); the driver skips such a batch on purpose instead of receiving NaN
            from an empty mean. Also on a mask that is not ``bool[b, n]``.
        NotImplementedError: for ``mode="local"``.
    """
    check_mode(mode)
    if mask is None:
        per_token = ntp_loss(logits, targets)
    else:
        # the live positions are selected before the loss, not after: a non-finite logit at a
        # masked position then has no path into the value or the gradient
        dims: Dims = {"b": targets.shape[0], "n": targets.shape[1]}
        check_shape("logits", logits, ("b", "n", "V"), dims)
        check_shape("mask", mask, ("b", "n"), dims, dtype=torch.bool)
        per_token = ntp_loss(logits[mask][None, :, :], targets[mask][None, :])
    if per_token.numel() == 0:
        msg = f"wake_loss: nothing to aggregate, targets have shape {tuple(targets.shape)}"
        if mask is not None:
            msg += " and the mask selects no position"
        raise ContractError(msg)
    return wake_total(per_token.mean(), weights=weights, ponder=ponder, gate=gate)


def check_mode(mode: Mode) -> None:
    """The one place a learning mode is admitted: ``e2e`` trains, ``local`` is refused.

    Every loss path (``wake_loss``, ``Vesper.objective``) and the run driver call it, so an
    unsupported mode is refused before any stream is advanced or any parameter moves (maintainer B's
    review of 629559f: the sliced objective had trained in local mode).

    Raises:
        NotImplementedError: for ``mode="local"``.
    """
    if mode == "local":
        msg = (
            "mode 'local' is not implemented: the spec's predictive-coding target is ambiguous "
            "(docs/reviews/2026-10-14-local-mode-question.md)"
        )
        raise NotImplementedError(msg)


def wake_total(
    ntp: Tensor,
    *,
    weights: LossWeights,
    ponder: Tensor | None = None,
    gate: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The objective from its parts: the mean next-token loss plus the weighted ponder and gate.

    The one place the parts are weighted and summed, shared by ``wake_loss`` (logits of every
    position) and ``Vesper.objective`` (the head over position slices).

    Raises:
        ContractError: on a ``ponder`` or ``gate`` that is not a scalar.
    """
    parts: dict[str, Tensor] = {"ntp": ntp}
    if ponder is not None:
        check_shape("ponder", ponder, (), {})
        parts["ponder"] = weights.lambda_h * ponder
    if gate is not None:
        check_shape("gate", gate, (), {})
        parts["gate"] = weights.lambda_g * gate
    total = parts["ntp"]
    for name in ("ponder", "gate"):
        if name in parts:
            total = total + parts[name]
    return total, parts
