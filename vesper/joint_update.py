"""The joint halting update: head and backbone trained on sampled stopping (shared by drivers).

Moved unchanged from vesper/halting_calibration.py (maintainer B f27b8a3 designed it, 4871b54
accepted it, 1eae844/2a16c44 made it capturable) so the synthetic calibration and the real-text
pilot (docs/reviews/2026-10-20-real-data-halting-pilot.md) run the same update, never two copies.

The batch is ``E`` unique sequences ``ROLLOUTS`` times over, rollout-major, with sampled stops
from given uniforms (``SampledRule``, copy-forward). ``update_loss`` is the loss whose gradient
is, in expectation, the gradient of the expected trajectory cost ``label term + depth_cost *
active depth / T``: the paired score surrogate (``actor_loss``) through the head AND the
backbone features it reads, the pathwise weighted label term, and an optional backbone-only
dense auxiliary. ``update_step`` is the whole update on the device with no read-back, so it can
be captured as one CUDA graph. Every trajectory starts from fresh states: a driver's sequences
are independent contexts. The label weights are the driver's (a mean per episode, per class,
or pooled per token); the core never looks at what a label is.
"""

import math
from collections.abc import Sequence
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from vesper.contracts import ContractError, Dims, check_shape, require
from vesper.loop import DenseRule, LoopResult, SampledRule
from vesper.model import Vesper, token_losses

ROLLOUTS = 2  # independent trajectories per sequence, each the other's baseline
HEAD_BIAS = -math.log(9.0)  # with the hazard baseline: the q = 0.5 policy exits at the last pass
PEAK_LR = 1e-4
BACKBONE_LR = 1e-4  # a joint run's backbone (its stage-1 training ran at 1e-3); exploratory
WARMUP = 100
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
OUTPUT_RANK = 3  # label_nll's outputs are at least [b, n, d]; pass dimensions may lead
NON_FINITE = "non-finite cost, surrogate or gradient (the update after the last one logged)"


def learning_rate(update: int, peak: float = PEAK_LR) -> float:
    """``peak`` after a linear warmup over the first ``WARMUP`` updates, constant after."""
    return peak * min((update + 1) / WARMUP, 1.0)


def optimizer(
    head: list[nn.Parameter],
    backbone: list[nn.Parameter],
    *,
    capturable: bool = False,
    backbone_lr: float = BACKBONE_LR,
) -> torch.optim.AdamW:
    """AdamW with one group per trainable set, each with its ``"peak"`` rate.

    The head at ``PEAK_LR``, a trainable backbone (joint runs) at ``backbone_lr`` (the
    caller's validated rate; ``BACKBONE_LR`` unless a run scales it); no weight
    decay in either, so the update descends the trajectory cost itself. ``capturable`` (CUDA,
    a captured update) keeps the step counts on the device and gives every group its own rate
    as a device tensor, filled in place by ``set_learning_rates`` (a float would be frozen into
    the graph at capture), as ``vesper.train.optimizer`` does for one shared rate.
    """
    device = head[0].device

    def rate(peak: float) -> float | Tensor:
        return torch.tensor(peak, device=device) if capturable else peak

    groups: list[dict[str, object]] = [{"params": head, "peak": PEAK_LR, "lr": rate(PEAK_LR)}]
    if backbone:
        groups.append({"params": backbone, "peak": backbone_lr, "lr": rate(backbone_lr)})
    return torch.optim.AdamW(
        groups, betas=BETAS, eps=ADAM_EPS, weight_decay=0.0, capturable=capturable
    )


def set_learning_rates(opt: torch.optim.AdamW, update: int) -> list[float]:
    """Every group's rate at ``update`` from its ``"peak"`` (``learning_rate``); the rates.

    A capturable optimizer's device rate is filled in place, never replaced.
    """
    rates: list[float] = []
    for group in cast("list[dict[str, object]]", opt.param_groups):
        if "peak" not in group:
            msg = "a parameter group without its peak rate: build the optimizer with optimizer"
            raise ValueError(msg)
        rate = learning_rate(update, cast("float", group["peak"]))
        current = group["lr"]
        if isinstance(current, float):
            group["lr"] = rate
        else:  # the capturable optimizer's device tensor (``optimizer``)
            _ = cast("Tensor", current).fill_(rate)
        rates.append(rate)
    return rates


def training_precision(*, cuda: bool) -> dict[str, object]:
    """TF32 off and the highest float32 matmul precision (the design's arithmetic), as set.

    On ``cuda`` the update is captured (``WarmedGraph``) with the capturable optimizer, whose
    device-side step arithmetic is not bitwise the CPU optimizer's, and trains under torch's
    deterministic algorithms (``vesper.graphs.deterministic_algorithms``, whose effective
    settings the result records as ``determinism``); recorded here. For every driver of this
    update (the synthetic calibration, the text pilot), one setting in one place.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "matmul_allow_tf32": cast("bool", torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast": None,
        "cuda_graph": cuda,
        "optimizer_capturable": cuda,
    }


def _group_grad_norms(opt: torch.optim.AdamW) -> Tensor:
    """``float64 [groups]``: each group's gradient L2 norm before clipping (0 without any).

    On the device, never read back (a captured update returns it).
    """
    groups = cast("list[dict[str, object]]", opt.param_groups)
    device = cast("list[nn.Parameter]", groups[0]["params"])[0].device
    norms: list[Tensor] = []
    for group in groups:
        grads = [p.grad for p in cast("list[nn.Parameter]", group["params"]) if p.grad is not None]
        squares = [(g.double() * g.double()).sum() for g in grads]
        zero = torch.zeros((), dtype=torch.float64, device=device)
        norms.append(torch.stack(squares).sum().sqrt() if squares else zero)
    return torch.stack(norms)


def start_head(model: Vesper) -> list[nn.Parameter]:
    """Freeze every backbone parameter and set the head's calibration start; its parameters.

    The head: direction 0, gain 1, bias ``HEAD_BIAS``; with ``hazard_baseline`` its stop
    hazard is ``1 / (1 + 9 (T - t))`` at every pass t < T, whatever the backbone computes.
    """
    _ = model.requires_grad_(requires_grad=False)
    head = model.loop.halting_head()
    head.zero_residual()
    with torch.no_grad():
        _ = head.b_h.fill_(HEAD_BIAS)
    _ = head.requires_grad_(requires_grad=True)
    return list(head.parameters())


def release_backbone(model: Vesper) -> list[nn.Parameter]:
    """Make every parameter outside the halting head trainable (a joint run); those parameters.

    Called after the admission, which measures the source with the head at its start; the head
    keeps that start, so a joint cell and its head-only reference begin from the same weights.
    """
    backbone = [p for name, p in model.named_parameters() if not name.startswith("loop.halting")]
    for p in backbone:
        _ = p.requires_grad_()
    return backbone


def actor_loss(costs: Tensor, log_probs: Tensor) -> Tensor:
    """The score-function surrogate with paired baselines: ``costs, log_probs [ROLLOUTS, E]``.

    ``mean_(r, e) stopgrad(C_r - C_other) * log_prob_r``: each rollout's baseline is the OTHER
    rollout's detached cost, independent of its own actions, so the expected gradient is the
    exact gradient of the expected cost (a baseline that includes the rollout's own cost halves
    it: the design's oracle). ``log_probs`` are the summed log-probabilities of the decisions each
    trajectory made, never divided by their random number.

    Raises:
        ContractError: unless both are ``[ROLLOUTS, E]`` with one ``E >= 1``.
    """
    dims: Dims = {}
    check_shape("costs", costs, (ROLLOUTS, "e"), dims)
    check_shape("log_probs", log_probs, (ROLLOUTS, "e"), dims)
    if dims["e"] < 1:
        msg = "actor_loss: no episode"
        raise ContractError(msg)
    advantage = (costs - costs.flip(0)).detach()
    return (advantage * log_probs).mean()


class Costs(NamedTuple):
    """Per trajectory, float64: ``cost``, ``objective``, ``nll``, ``depth`` ``[b]`` detached.

    ``ce [b, n]`` is the per-label NLL (0 off the labels) WITH its graph to the backbone, for
    the pathwise term of a joint update (no graph when the backbone is frozen).
    """

    cost: Tensor
    objective: Tensor
    nll: Tensor
    depth: Tensor
    ce: Tensor


def _costs(  # noqa: PLR0913  # the update's inputs plus label_nll's memory option
    model: Vesper,
    result: LoopResult,
    targets: Tensor,
    mask: Tensor,
    weights: Tensor,
    valid: Tensor,
    depth_cost: float,
    *,
    slices: int = 1,
    pooled_depth: bool = False,
) -> Costs:
    """The trajectories' costs, ``[b]`` each, and the per-label NLL ``[b, n]``.

    ``objective`` is the label NLL weighted by ``weights`` (one row per trajectory), ``nll`` the
    ordinary mean next-token NLL over the trajectory's labels (logged whatever the weights),
    ``depth`` the trajectory's active depth, and ``cost = objective + depth_cost * depth / T``.

    ``depth`` by default is the mean over the trajectory's own real tokens (the synthetic
    calibration's episodes, equal length, unchanged). ``pooled_depth`` prices depth over real
    tokens POOLED, as ``pooled_weights`` pools labels (the text pilot, rows of unequal length;
    maintainer B's review of 72c9d31): each trajectory's summed depth times ``b / V``, ``V`` the
    real tokens of all ``b`` rows, so the mean over a rollout's rows is the depth pooled over its
    real tokens. Both read only ``valid``, never a sampled action, so paired rollouts share
    the normalization.
    """
    ce = label_nll(model, result.y, targets, mask, slices)
    with torch.no_grad():
        plain_ce = ce.detach()
        nll = plain_ce.sum(1) / mask.sum(1).double()
        objective = (plain_ce * weights).sum(1)
        summed = torch.where(valid, result.n_iters.double(), 0.0).sum(1)
        depth = (
            summed * (valid.shape[0] / valid.sum().double())
            if pooled_depth
            else summed / valid.sum(1).double()
        )
        t_max = int(result.per_pass.shape[0])
        return Costs(objective + depth_cost * depth / t_max, objective, nll, depth, ce)


def label_nll(model: Vesper, y: Tensor, targets: Tensor, mask: Tensor, slices: int = 1) -> Tensor:
    """``float64 [..., n]``: the next-token NLL of the outputs ``y [..., n, d]``, with its graph.

    At the labels of ``mask [b, n]`` (broadcast over leading pass dimensions), 0 elsewhere. The
    head takes ``[b, n, d]``: leading dimensions are folded into its batch and back.
    ``slices = 1`` is the measured synthetic update, unchanged. ``slices > 1`` (memory only, for
    a real vocabulary: the text pilot's dense term alone would hold ``T x E x n x V`` logits)
    scores the flattened positions in that many consecutive ranges through
    ``model.token_losses``, whose head runs again in the backward pass, so one range's
    vocabulary logits exist at a time; the same values and gradients up to the float rounding
    of the head's batching.

    ``targets`` and ``mask`` are exactly ``[b, n]`` of ``y``'s last two leading dimensions,
    integer (int32 or int64) and boolean: a public boundary, so a single row's labels are never
    broadcast across the batch and fractional targets are never truncated (maintainer B's review of
    778ff10); only ``y``'s pass dimensions in front broadcast, as intended.

    Raises:
        ValueError: on ``slices < 1``.
        ContractError: on ``y`` of rank below 3, labels of another shape, non-integer targets
            or a non-boolean mask; before any arithmetic.
    """
    if slices < 1:
        msg = f"slices must be at least 1, got {slices}"
        raise ValueError(msg)
    if y.dim() < OUTPUT_RANK:
        msg = f"y must be [..., b, n, d], got shape {tuple(y.shape)}"
        raise ContractError(msg)
    dims: Dims = {"b": int(y.shape[-3]), "n": int(y.shape[-2])}
    check_shape("targets", targets, ("b", "n"), dims)
    check_shape("mask", mask, ("b", "n"), dims, dtype=torch.bool)
    if targets.dtype not in {torch.int32, torch.int64}:
        msg = f"targets must be int32 or int64, got {targets.dtype}"
        raise ContractError(msg)
    lead = y.shape[:-1]
    safe = torch.where(mask, targets.long(), 0).expand(lead)  # every index valid; masked below
    if slices == 1:
        logits = model.head(y.reshape(-1, *y.shape[-2:])).float()
        ce = functional.cross_entropy(
            logits.flatten(0, -2), safe.reshape(-1), reduction="none"
        ).view(lead)
    else:
        rows, flat = y.reshape(-1, y.shape[-1]), safe.reshape(-1)
        width = -(-int(rows.shape[0]) // slices)
        ce = torch.cat(
            [
                token_losses(model, rows[s : s + width], flat[s : s + width])
                for s in range(0, int(rows.shape[0]), width)
            ]
        ).view(lead)
    return torch.where(mask, ce.double(), 0.0)


class Dense(NamedTuple):
    """The optional dense auxiliary of a joint update: the UNIQUE sequences once each."""

    tokens: Tensor
    pos: Tensor
    targets: Tensor
    mask: Tensor
    weights: Tensor
    weight: float


class Loss(NamedTuple):
    """One update's differentiable loss and its parts (``total = score + pathwise + dense``)."""

    total: Tensor
    score: Tensor
    pathwise: Tensor
    dense: Tensor
    costs: Costs
    rule: SampledRule
    result: LoopResult


def update_loss(  # noqa: PLR0913, PLR0917
    model: Vesper,
    tokens: Tensor,
    pos: Tensor,
    targets: Tensor,
    mask: Tensor,
    valid: Tensor,
    weights: Tensor,
    uniforms: Tensor,
    epsilon: float | Tensor,
    depth_cost: float,
    dense: Dense | None = None,
    *,
    recompute: bool = False,
    slices: int = 1,
    pooled_depth: bool = False,
) -> Loss:
    """The loss whose gradient, in expectation, is the gradient of the expected trajectory cost.

    The batch is the sequences ``ROLLOUTS`` times over, rollout-major, with sampled stops from
    ``uniforms`` (``SampledRule``). For ``J = E[label term + depth_cost * depth / T]`` a
    parameter moves both a trajectory's cost and its probability, so the loss has two parts
    (maintainer B, channel f27b8a3): ``score``, the paired score surrogate (``actor_loss``) whose
    costs are detached and whose log-probabilities keep their graph through the head AND the
    backbone features it reads; and ``pathwise``, the mean over trajectories of the weighted
    label term WITH its graph (the depth has no pathwise derivative). With the backbone frozen
    only the head has gradients, from ``score`` alone. ``dense`` (a joint run's auxiliary,
    ``weight`` rho) adds rho times the mean over passes of the same weighted label term on fresh
    all-pass states of the unique sequences: a backbone-only pathwise term with no decision
    credit. ``recompute`` (the loop's activation recomputation, both forwards) and ``slices``
    (``label_nll``'s) change memory only; both off is the measured synthetic update.
    ``pooled_depth`` (``_costs``) prices depth over pooled real tokens; off is the synthetic
    per-trajectory mean, unchanged.

    Raises:
        RuntimeError: if the rule recorded no decisions.
        ValueError: on ``slices < 1``.
    """
    rule = SampledRule(model.loop.halting_head(), epsilon, uniforms, valid)
    states = model.init_states(int(tokens.shape[0]), torch.float32, tokens.device)
    result = model.run(tokens, pos, states, recompute=recompute, exit_rule=rule)
    costs = _costs(
        model,
        result,
        targets,
        mask,
        weights,
        valid,
        depth_cost,
        slices=slices,
        pooled_depth=pooled_depth,
    )
    if rule.log_prob is None or rule.decisions is None:
        msg = "the sampled rule recorded no decisions"
        raise RuntimeError(msg)
    score = actor_loss(costs.cost.view(ROLLOUTS, -1), rule.log_prob.sum(1).view(ROLLOUTS, -1))
    pathwise = (costs.ce * weights).sum(1).mean()
    dense_term = torch.zeros((), dtype=torch.float64, device=tokens.device)
    if dense is not None and dense.weight > 0.0:
        fresh = model.init_states(int(dense.tokens.shape[0]), torch.float32, tokens.device)
        passes = model.run(
            dense.tokens, dense.pos, fresh, recompute=recompute, exit_rule=DenseRule()
        ).per_pass
        per_pass = (
            label_nll(model, passes, dense.targets, dense.mask, slices) * dense.weights
        ).sum(-1)
        dense_term = dense.weight * per_pass.mean(0).mean()
    return Loss(score + pathwise + dense_term, score, pathwise, dense_term, costs, rule, result)


class Step(NamedTuple):
    """One update's outputs, detached, on the device: what its record is read from."""

    score: Tensor
    pathwise: Tensor
    dense: Tensor
    cost: Tensor
    objective: Tensor
    nll: Tensor
    depth: Tensor
    ce: Tensor
    grad_norm: Tensor
    group_norms: Tensor
    decisions: Tensor
    n_iters: Tensor


def update_step(  # noqa: PLR0913  # the six update inputs plus the cost and memory options
    model: Vesper,
    opt: torch.optim.AdamW,
    params: Sequence[nn.Parameter],
    depth_cost: float,
    dense_weight: float,
    values: Sequence[Tensor],
    *,
    recompute: bool = False,
    slices: int = 1,
    pooled_depth: bool = False,
) -> Step:
    """The update on the device, with no read-back: capturable as one CUDA graph.

    ``values``, all on the model's device: tokens, positions, targets, loss mask, valid (each
    ``[ROLLOUTS * E, n]``, rollout-major), the label weights (float64, the same), the uniforms
    and the floor (a float64 scalar). The unique sequences, for the dense auxiliary, are the
    first ``E`` rows. Its checks are device contracts (``require``). ``recompute`` and
    ``slices`` are ``update_loss``'s memory options, ``pooled_depth`` its depth pricing.

    Raises:
        FloatingPointError: eagerly, on a non-finite cost, surrogate or gradient, before the step.
        RuntimeError: if the rule recorded no decisions.
    """
    tokens, pos, targets, mask, valid, weights, uniforms, epsilon = values
    e = int(tokens.shape[0]) // ROLLOUTS
    dense = (
        Dense(tokens[:e], pos[:e], targets[:e], mask[:e], weights[:e], dense_weight)
        if dense_weight > 0.0
        else None
    )
    loss = update_loss(
        model,
        tokens,
        pos,
        targets,
        mask,
        valid,
        weights,
        uniforms,
        epsilon,
        depth_cost,
        dense,
        recompute=recompute,
        slices=slices,
        pooled_depth=pooled_depth,
    )
    costs, rule = loss.costs, loss.rule
    if rule.decisions is None:
        msg = "the sampled rule recorded no decisions"
        raise RuntimeError(msg)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([loss.total])
    group_norms = _group_grad_norms(opt)
    grad_norm = nn.utils.clip_grad_norm_(list(params), GRAD_CLIP)
    finite = (
        torch.isfinite(costs.cost).all() & torch.isfinite(loss.total) & torch.isfinite(grad_norm)
    )
    try:
        require(finite, NON_FINITE)
    except ContractError as error:
        raise FloatingPointError(NON_FINITE) from error
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    return Step(
        loss.score.detach(),
        loss.pathwise.detach(),
        loss.dense.detach(),
        costs.cost,
        costs.objective,
        costs.nll,
        costs.depth,
        costs.ce.detach(),
        grad_norm.detach(),
        group_norms,
        rule.decisions,
        loss.result.n_iters,
    )
