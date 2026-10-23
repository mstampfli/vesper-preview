"""The halting exploration calibration: learn the stopping decisions on a frozen capable backbone.

Design: docs/reviews/2026-10-18-halting-exploration-calibration.md (maintainer B, 7255c6e;
implemented by maintainer A; maintainer B's review of cc69a25 in channel 3c47621, closure pending),
a bounded calibration under the adopted, NOT PROVEN halting-in-phases
plan (docs/reviews/2026-10-17-halting-in-phases.md). It asks whether a trained head can route an
already capable backbone under real stopping; it claims no deployed speedup.

    python -m vesper.halting_calibration --source RUN/model.pt --source-sha256 HEX \
        --seed 0 --floor none|taper --out RUN_DIR [--device cpu|cuda] [--updates 2000] \
        [--eval-batches 32] [--depth-cost 0.02] [--save-at 0 500 1000 1500 2000] \
        [--required-accuracy 0.5] [--label-weight episode|class] \
        [--train head|joint] [--dense-weight 0.0] [--preflight]

The defaults are the design's; any other ``--updates``, ``--eval-batches``, ``--depth-cost``,
``--save-at``, ``--required-accuracy`` or ``--label-weight`` makes the run exploratory, named
as a difference by ``trial_protocol`` and DIAGNOSTIC in its verdict. ``--required-accuracy``
lowers only the admission's class floor (the source and initial-policy checks stay; the
verdict keeps ``REQUIRED_ACCURACY``). ``--label-weight class`` counts every class equally in
the NLL term (``_label_weights``), a changed task preference.

One run:

- Source: a run's weights (``vesper.halting_common.read_warm_start``) whose sha256 must equal
  ``--source-sha256``; its config snapshot is the run's config. Before anything is written
  (``_admit``): the dense per-pass counts of the source's validation set are recomputed and must
  equal its result.json's ``counts.dense_pass``; the full-depth model must score above 50 % on
  every class of this calibration's evaluation set; and the initial head's deterministic policy
  must exit every token at the last pass with the dense output. A failed check is written to
  result.json and stops the run; no other checkpoint or set is tried instead.
- Model: every backbone parameter frozen (``start_head``); the halting head alone trains, from
  direction 0, gain 1 and bias ``HEAD_BIAS`` on top of the hazard baseline. A new run: fresh
  optimizer, data stream, action stream and progress.
- Each update: ``EPISODES`` fresh episodes (terminal mix, ``generator_for(TRAIN_SPLIT, seed)``),
  each run ``ROLLOUTS`` times with real sampled stops (``vesper.loop.SampledRule``) under the
  floor of ``--floor`` (``_floor_at``), with uniforms drawn from
  ``generator_for(ACTION_SPLIT, seed)`` as one fixed-shape array per update. The cost of a
  trajectory is its episode's mean label NLL (or the class-balanced term of ``--label-weight
  class``) plus ``DEPTH_COST`` times its mean active depth
  over T, over every real token (a routing proxy, not measured work). The head learns by the
  score-function estimator, each rollout's baseline the other rollout's detached cost
  (``actor_loss``); AdamW on the head only (``PEAK_LR`` after a ``WARMUP``-update linear warmup,
  ``BETAS``, ``ADAM_EPS``, no decay, clip ``GRAD_CLIP``). Runs with the same seed and different
  floors consume identical data and uniforms (``consumed_digest``, ``draw_digest``).
- Evaluation at every update of ``SAVE_AT`` (the head saved there), on a CPU reference copy over
  ``EVAL_BATCHES`` x ``EVAL_SIZE`` episodes of ``generator_for(EVAL_SPLIT, 0)``
  (``evaluate_policy``): the DETERMINISTIC live CDF policy at ``Q`` (no floor, no sampling,
  copy-forward, own-exit output) with accuracy and depth per class, per role and over all
  tokens, and depth entropies; a token-independent random-depth control drawn from the policy's
  own depth distribution (seeds ``RANDOM_SEEDS``) with its measured depth difference; per-episode
  differences from the unchanged full-depth model with bootstrap intervals. The full-depth model
  and every static depth are measured once (``fixed_depths``; the backbone never changes). The
  terminal evaluation carries the design's calibration criteria (``verdict``).
- ``--preflight``: the disposable timing run of ``PREFLIGHT_UPDATES`` updates: seconds and peak
  memory per update and the spread of the two cost terms between paired rollouts; no
  evaluation, and nothing from it becomes a trial update.

- ``--train joint`` (exploratory, maintainer B f27b8a3, accepted 4871b54): the backbone trains too,
from
  the same source and head start, in a second AdamW group at ``BACKBONE_LR``. Its update loss
  (``joint_update.update_loss``) adds to the score surrogate (whose features are NOT detached,
  so the backbone also learns from the decisions) the pathwise weighted label term of the sampled
  trajectories, and with ``--dense-weight`` rho the mean over passes of that term on fresh
  all-pass states (backbone only). A joint run saves the whole weights and the CURRENT
  backbone's full and static profiles at every save point; the source's stay the target.
- On CUDA every update is one captured CUDA graph after three eager ones
  (``vesper.graphs.WarmedGraph``): the update runs on the device without a read-back
  (``joint_update.update_step``), the floor and the batch are its inputs, the rates are device
  tensors of a capturable AdamW, and the record is read from its outputs; captured updates
  equal eager ones bitwise. The capturable optimizer's arithmetic is not bitwise the CPU one's
  (recorded).

The loop executes every pass for every token whatever the policy (stopped tokens are masked and
their writes copied forward), so a lower active depth here is not a lower executed cost; the
evaluation records the number of passes the loop executed.
"""

import argparse
import itertools
import json
import math
import time
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, cast

import torch
from torch import Tensor, nn
from torch.nn import functional

from vesper.column import check_capturable_scan, scan_of, set_scan
from vesper.graphs import WarmedGraph, deterministic_algorithms, memory_peak, reset_memory_peak
from vesper.halting_common import (
    CPU,
    SNAPSHOT,
    WarmStart,
    chained_digest,
    chunk_tensors,
    class_accuracy,
    conditional_entropy,
    evaluating,
    load_weights,
    plain,
    read_warm_start,
    version_or_none,
)
from vesper.halting_experiment import class_counts, held_out
from vesper.joint_update import (
    ADAM_EPS,
    BACKBONE_LR,
    BETAS,
    GRAD_CLIP,
    HEAD_BIAS,
    PEAK_LR,
    ROLLOUTS,
    WARMUP,
    Step,
    optimizer,
    release_backbone,
    set_learning_rates,
    start_head,
    training_precision,
    update_step,
)
from vesper.loop import CdfRule, DenseRule, ExitRule, TraceRule
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.synthetic import CLASSES, ROLES, TERMINAL_MIX, Episodes, episodes, generator_for

TRAIN_SPLIT = "halting-head-train"
ACTION_SPLIT = "halting-head-action"
EVAL_SPLIT = "halting-head-eval"
REQUIRED_ACCURACY = 0.5  # every class above this: admission (full depth) and the verdict (policy)
CLASS_TOLERANCE = 0.01  # the verdict: every class within one point of the full-depth model
DEPTH_REDUCTION = 0.2  # the verdict: all-token mean active depth at least 20 % below full depth
LABEL_WEIGHTS = ("episode", "class")  # how a trajectory's labels form its NLL term
LABEL_WEIGHT = "episode"  # the design's: each episode's mean label NLL
EPISODES = 32  # unique episodes per update
UPDATES = 2000
SAVE_AT = (0, 500, 1000, 1500, 2000)
DEPTH_COST = 0.02
TRAIN_MODES = ("head", "joint")  # what trains: the head on a frozen backbone, or both
TRAIN_MODE = "head"  # the design's
FLOORS = ("none", "taper")
FLOOR_START = 0.5
FLOOR_END = 1500  # the taper reaches zero at this update and stays there
Q = 0.5
RANDOM_SEEDS = tuple(range(8))
BOOTSTRAP = 1000
EVAL_BATCHES = 32
EVAL_SIZE = 32
PREFLIGHT_UPDATES = 100
TRIAL_RUNS = 4  # the design's trial: seeds 0 and 1 crossed with the two floors
DEVICES = ("cpu", "cuda")
TRAIN_SCAN = {"cpu": "reference", "cuda": "fused-fp32"}

RuleFor = Callable[[int], ExitRule]


def _floor_at(schedule: str, update: int) -> float:
    """The continuation floor at ``update``: 0 for ``none``, ``0.5 max(1 - u/1500, 0)`` (taper).

    Raises:
        ValueError: on an unknown schedule or a negative update.
    """
    if schedule not in FLOORS:
        msg = f"floor must be one of {FLOORS}, got {schedule!r}"
        raise ValueError(msg)
    if update < 0:
        msg = f"update must be non-negative, got {update}"
        raise ValueError(msg)
    if schedule == "none":
        return 0.0
    return FLOOR_START * max(1.0 - update / FLOOR_END, 0.0)


def _label_weights(task: Tensor, scheme: str) -> Tensor:
    """``float64 [E, n]``: the weight of every label in its episode's NLL term; 0 elsewhere.

    ``task`` is ``Episodes.task`` of the E UNIQUE episodes (class index at a label, -1 elsewhere).
    ``"episode"`` (the design's ``LABEL_WEIGHT``): each episode's mean, 1 / (its label count).
    ``"class"`` (exploratory, maintainer B 9f1eceb): every class present among the E episodes'
    labels counts equally; a label of class c weighs E / (C * N_c), with N_c its labels POOLED over
    the E episodes (not per-episode class means, so an episode's lone read label weighs what each
    read label of another episode weighs) and C the number of classes present; an absent class
    has no label and is not counted. Both schemes sum to E over all labels, so the term keeps
    the scale of a mean NLL per episode against the depth cost. The weights read only ``task``,
    never a sampled action, so both rollouts of an episode share them.

    Raises:
        ValueError: on an unknown scheme or an episode without a label.
    """
    if scheme not in LABEL_WEIGHTS:
        msg = f"unknown label weight scheme {scheme!r} (one of {LABEL_WEIGHTS})"
        raise ValueError(msg)
    mask = task >= 0
    labels = mask.sum(1)
    if bool((labels == 0).any()):
        msg = "an episode has no label"
        raise ValueError(msg)
    if scheme == "episode":
        return torch.where(mask, 1.0 / labels.double()[:, None], 0.0)
    safe = torch.where(mask, task, 0)
    counts = torch.bincount(safe[mask], minlength=len(CLASSES)).double()
    present = int((counts > 0).sum())
    per_class = torch.where(counts > 0, task.shape[0] / (present * counts), 0.0)
    return torch.where(mask, per_class[safe], 0.0)


def _class_nll(ce: Tensor, task: Tensor) -> list[float | None]:
    """The mean NLL of each class in ``CLASSES`` order, pooled over every label; None if absent."""
    mask = task >= 0
    safe = torch.where(mask, task, 0)
    sums = torch.zeros(len(CLASSES), dtype=torch.float64).index_add_(0, safe[mask], ce[mask])
    counts = torch.bincount(safe[mask], minlength=len(CLASSES))
    return [
        float(s) / int(c) if int(c) > 0 else None
        for s, c in zip(sums.unbind(), counts.unbind(), strict=True)
    ]


def train_update(  # noqa: PLR0913, PLR0917
    model: Vesper,
    opt: torch.optim.AdamW,
    params: Sequence[nn.Parameter],
    ep: Episodes,
    uniforms: Tensor,
    epsilon: float,
    update: int,
    depth_cost: float = DEPTH_COST,
    label_weight: str = LABEL_WEIGHT,
    dense_weight: float = 0.0,
    graph: WarmedGraph | None = None,
) -> dict[str, object]:
    """One update of the trainable parameters from ``ep``'s sampled trajectories; its record.

    ``joint_update.update_loss`` is the loss (the head's score term; with a trainable backbone
    also the pathwise label term and, at ``dense_weight`` above 0, the dense auxiliary); ``opt``
    holds one parameter group per trainable set, each with its own ``"peak"`` learning rate
    under the shared warmup (``joint_update.set_learning_rates``); ``params`` is every
    trainable parameter, clipped together.

    The batch is the episodes ``ROLLOUTS`` times over, rollout-major, and ``uniforms`` are
    float64 ``[T - 1, ROLLOUTS * E, n]`` in the same order. ``depth_cost`` weighs the mean active
    depth in the cost (the design's ``DEPTH_COST``; other values are exploratory);
    ``label_weight`` forms the NLL term from the unique episodes (``_label_weights``; the
    design's ``LABEL_WEIGHT``). Every decision of a trajectory is credited with its whole cost.
    The record keeps the ordinary mean NLL (``nll``) beside the trained term
    (``objective_nll``) and each class's NLL pooled over the batch (``class_nll``).

    The update itself (``update_step``) runs on the device without a read-back; the record is
    read from its outputs after it. With ``graph`` (CUDA) the update is captured once and
    replayed (``vesper.graphs.WarmedGraph``): the batch, the uniforms and the floor are its
    values, the rates are filled into a capturable ``opt`` (``optimizer``) before it.

    Raises:
        FloatingPointError: on a non-finite cost, surrogate or gradient, before the step (in a
            replay a device assert: the process stops with the same message).
    """
    device = next(model.parameters()).device
    c = ep.chunk
    rates = set_learning_rates(opt, update)
    unique = _label_weights(ep.task, label_weight)
    values = [
        torch.cat([t] * ROLLOUTS).to(device)
        for t in (c.tokens, c.pos, c.targets, c.loss_mask, c.valid, unique)
    ]
    values += [uniforms.to(device), torch.tensor(epsilon, dtype=torch.float64, device=device)]

    def step(inputs: Sequence[Tensor]) -> Step:
        return update_step(model, opt, params, depth_cost, dense_weight, inputs)

    out = step(values) if graph is None else Step(*graph(step, values))
    t_max = model.loop.t_max
    pair_nll = out.objective.view(ROLLOUTS, -1)  # the NLL term the depth cost competes with
    pair_depth = depth_cost * out.depth.view(ROLLOUTS, -1) / t_max
    deciding = values[4] & (out.decisions > 0)
    return {
        "update": update,
        "lr": rates[0],
        "lr_groups": rates,
        "epsilon": epsilon,
        "cost": float(out.cost.mean()),
        "nll": float(out.nll.mean()),
        "objective_nll": float(out.objective.mean()),
        "class_nll": _class_nll(out.ce.cpu(), torch.cat([ep.task] * ROLLOUTS)),
        "depth": float(out.depth.mean()),
        "surrogate": float(out.score),
        "pathwise": float(out.pathwise),
        "dense": float(out.dense),
        "grad_norm": float(out.grad_norm),
        "grad_norm_groups": [float(norm) for norm in out.group_norms.unbind()],
        "decisions": int(out.decisions.sum()),
        "stopped_early": float((out.n_iters[deciding] < t_max).double().mean()),
        "pair_nll_diff_std": float((pair_nll[0] - pair_nll[1]).std()),
        "pair_depth_cost_diff_std": float((pair_depth[0] - pair_depth[1]).std()),
    }


def _episode_counts(model: Vesper, y: Tensor, ep: Episodes) -> Tensor:
    """``int64 [b, C, 2]``: correct and total per episode and class from the outputs ``y``."""
    mask = ep.chunk.loss_mask
    right = (model.head(y).argmax(-1) == ep.chunk.targets.long()) & mask
    one_hot = functional.one_hot(torch.where(mask, ep.task, 0), len(CLASSES)) * mask[..., None]
    return torch.stack([(one_hot * right[..., None]).sum(1), one_hot.sum(1)], dim=-1)


def _run_policy(
    model: Vesper, batches: Sequence[Episodes], rule_for: RuleFor
) -> tuple[Tensor, Tensor, int]:
    """``(episode counts [N, C, 2], depths [N, n], passes executed)`` of a policy, prefill."""
    counts: list[Tensor] = []
    depths: list[Tensor] = []
    executed = 0
    for i, ep in enumerate(batches):
        c = ep.chunk
        states = model.init_states(int(c.tokens.shape[0]), torch.float32, CPU)
        result = model.run(c.tokens, c.pos, states, exit_rule=rule_for(i))
        counts.append(_episode_counts(model, result.y, ep))
        depths.append(result.n_iters.long())
        executed = max(executed, int(result.per_pass.shape[0]))
    return torch.cat(counts), torch.cat(depths), executed


def _traces(depths: Sequence[Tensor]) -> RuleFor:
    """Batch ``i`` replays the exit trace ``depths[i]`` (``TraceRule``)."""
    return lambda i: TraceRule(depths[i])


def _mean_depth(histogram: Tensor) -> float | None:
    total = float(histogram.sum())
    steps = torch.arange(1, int(histogram.shape[-1]) + 1, dtype=torch.float64)
    return float((histogram.double() * steps).sum()) / total if total else None


def _summary(
    counts: Tensor, depths: Tensor, batches: Sequence[Episodes], t_max: int
) -> dict[str, object]:
    """Accuracy per class; depth per class, role and over all real tokens; depth entropies."""
    task = torch.cat([ep.task for ep in batches])
    role = torch.cat([ep.role for ep in batches])
    valid = torch.cat([ep.chunk.valid for ep in batches])
    labels = task >= 0
    by_class = torch.zeros(len(CLASSES), t_max, dtype=torch.int64)
    _ = by_class.index_put_(
        (task[labels], depths[labels] - 1), torch.ones_like(task[labels]), accumulate=True
    )
    by_role = torch.zeros(len(ROLES), t_max, dtype=torch.int64)
    _ = by_role.index_put_(
        (role[valid], depths[valid] - 1), torch.ones_like(role[valid]), accumulate=True
    )
    labels_marginal, given_class = conditional_entropy(by_class)
    all_marginal, given_role = conditional_entropy(by_role)
    return {
        "accuracy": class_accuracy(counts.sum(0)),
        "counts": plain(counts.sum(0)),
        "depth_all": _mean_depth(by_role.sum(0)),
        "depth_histogram": plain(by_role.sum(0)),
        "depth_by_class": {name: _mean_depth(by_class[i]) for i, name in enumerate(CLASSES)},
        "depth_by_role": {name: _mean_depth(by_role[i]) for i, name in enumerate(ROLES)},
        "entropy_labels": labels_marginal,
        "entropy_labels_given_class": given_class,
        "entropy_all": all_marginal,
        "entropy_all_given_role": given_role,
    }


def _bootstrap(differences: Tensor) -> dict[str, float]:
    """Mean of per-episode differences and its 95 % percentile bootstrap interval (seed 0)."""
    g = torch.Generator().manual_seed(0)
    n = int(differences.shape[0])
    index = torch.randint(0, n, (BOOTSTRAP, n), generator=g)
    means = differences.double()[index].mean(1)
    return {
        "mean": float(differences.double().mean()),
        "low": float(torch.quantile(means, 0.025)),
        "high": float(torch.quantile(means, 0.975)),
    }


def _paired(policy: Tensor, full: Tensor) -> dict[str, object]:
    """Per-episode accuracy differences, policy minus full depth, overall and per class."""

    def rate(counts: Tensor) -> Tensor:
        return counts[..., 0].double() / counts[..., 1].double()

    per_class: dict[str, dict[str, float] | None] = {}
    for i, name in enumerate(CLASSES):
        seen = full[:, i, 1] > 0
        per_class[name] = (
            _bootstrap(rate(policy[seen, i]) - rate(full[seen, i])) if bool(seen.any()) else None
        )
    return {"overall": _bootstrap(rate(policy.sum(1)) - rate(full.sum(1))), "per_class": per_class}


def evaluate_policy(model: Vesper, batches: Sequence[Episodes], full: Tensor) -> dict[str, object]:
    """The deterministic policy at ``Q``, its random-depth controls and the paired differences.

    ``full`` holds the full-depth model's ``[N, C, 2]`` episode counts on the same batches.
    """
    with evaluating(model) as m:
        t_max = m.loop.t_max
        head = m.loop.halting_head()
        counts, depths, executed = _run_policy(m, batches, lambda _: CdfRule(head, Q))
        histogram = torch.bincount((depths - 1).flatten(), minlength=t_max).double()
        controls: list[dict[str, object]] = []
        for seed in RANDOM_SEEDS:
            g = torch.Generator().manual_seed(seed)
            drawn = [
                torch.multinomial(histogram, ep.chunk.tokens.numel(), replacement=True, generator=g)
                .view(ep.chunk.tokens.shape)
                .add(1)
                for ep in batches
            ]
            r_counts, r_depths, _ = _run_policy(m, batches, _traces(drawn))
            control = _summary(r_counts, r_depths, batches, t_max)
            control["seed"] = seed
            control["depth_difference"] = float(r_depths.double().mean() - depths.double().mean())
            controls.append(control)
    return {
        "policy": _summary(counts, depths, batches, t_max),
        "random": controls,
        "paired_vs_full_depth": _paired(counts, full),
        "executed_passes": executed,
    }


def fixed_depths(model: Vesper, batches: Sequence[Episodes]) -> tuple[dict[str, object], Tensor]:
    """The unchanged full-depth model and every static depth; and the former's episode counts."""
    with evaluating(model) as m:
        t_max = m.loop.t_max
        full_counts, full_depths, _ = _run_policy(m, batches, lambda _: DenseRule())
        static: list[dict[str, object]] = []
        for d in range(1, t_max + 1):
            trace = [torch.full_like(ep.chunk.tokens, d, dtype=torch.int64) for ep in batches]
            s_counts, s_depths, _ = _run_policy(m, batches, _traces(trace))
            static.append(_summary(s_counts, s_depths, batches, t_max) | {"depth": d})
    record: dict[str, object] = {
        "full": _summary(full_counts, full_depths, batches, t_max),
        "static": static,
    }
    return record, full_counts


def static_frontier(points: Sequence[tuple[float, float]], depth: float) -> float:
    """The best overall accuracy a fixed-depth policy reaches at mean active depth <= ``depth``.

    ``points`` are the measured static policies as ``(mean active depth, overall accuracy)``.
    Candidates: every point at or below ``depth``, and every whole-episode mixture of two points
    whose mean depth is exactly ``depth`` (one below it, one above; a mixture's accuracy is linear
    in its mean depth, so on a segment the best point within the budget is either its lower end
    or the budget itself). An expected LOGICAL-depth benchmark (maintainer B's review of cc69a25),
    not a measured wall-time curve; -inf when no point lies at or below ``depth``.
    """
    best = max((accuracy for d, accuracy in points if d <= depth), default=-math.inf)
    for (d1, a1), (d2, a2) in itertools.combinations(sorted(points), 2):
        if d1 < depth < d2:
            share = (d2 - depth) / (d2 - d1)  # of episodes at the lower depth
            best = max(best, share * a1 + (1.0 - share) * a2)
    return best


def trial_protocol(  # noqa: PLR0913
    updates: int,
    eval_batches: int,
    save_at: Sequence[int],
    required_accuracy: float,
    *,
    preflight: bool,
    depth_cost: float = DEPTH_COST,
    label_weight: str = LABEL_WEIGHT,
    train: str = TRAIN_MODE,
    dense_weight: float = 0.0,
) -> dict[str, object]:
    """The run's budget against the design's fixed trial protocol, with every difference named.

    Only a run that matches it is ASSESSED by ``verdict``; any other (a preflight, a test, a
    shortened debug run) reports diagnostic criterion values that claim nothing about the trial.
    """
    differences = [
        name
        for name, actual, prescribed in (
            ("updates", updates, UPDATES),
            ("eval_batches", eval_batches, EVAL_BATCHES),
            ("save_at", tuple(save_at), SAVE_AT),
            ("required_accuracy", required_accuracy, REQUIRED_ACCURACY),
            ("preflight", preflight, False),
            ("depth_cost", depth_cost, DEPTH_COST),
            ("label_weight", label_weight, LABEL_WEIGHT),
            ("train", train, TRAIN_MODE),
            ("dense_weight", dense_weight, 0.0),
        )
        if actual != prescribed
    ]
    return {
        "depth_cost": depth_cost,
        "label_weight": label_weight,
        "train": train,
        "dense_weight": dense_weight,
        "updates": updates,
        "eval_episodes": eval_batches * EVAL_SIZE,
        "save_at": list(save_at),
        "required_accuracy": required_accuracy,
        "trial": not differences,
        "differences_from_trial": differences,
    }


def verdict(
    terminal: dict[str, object], fixed: dict[str, object], protocol: dict[str, object]
) -> dict[str, object]:
    """The design's calibration criteria on the terminal evaluation (not a statistical test).

    Every class within one point of the full-depth model and above 50 %; all-token mean active
    depth at least 20 % below the full depth's; overall accuracy above the random controls' mean,
    whose measured depth difference from the policy is kept beside it (quality alone is not a cost
    match); and above ``static_frontier`` at the policy's mean depth. ``promising`` is given only
    under the trial protocol (``trial_protocol``); otherwise it is None and the values are
    diagnostic.
    """
    policy = cast("dict[str, object]", terminal["policy"])
    full = cast("dict[str, object]", fixed["full"])
    accuracy = cast("dict[str, float | None]", policy["accuracy"])
    full_accuracy = cast("dict[str, float | None]", full["accuracy"])
    per_class: dict[str, bool] = {}
    for name in CLASSES:
        a, f = accuracy[name], full_accuracy[name]
        per_class[name] = (
            a is not None and f is not None and a > REQUIRED_ACCURACY and a >= f - CLASS_TOLERANCE
        )
    depth = cast("float", policy["depth_all"])
    full_depth = cast("float", full["depth_all"])
    controls = cast("list[dict[str, object]]", terminal["random"])
    random_overall = sum(
        cast("float", cast("dict[str, float | None]", c["accuracy"])["overall"]) for c in controls
    ) / len(controls)
    random_depth_difference = sum(cast("float", c["depth_difference"]) for c in controls) / len(
        controls
    )
    points = [
        (
            cast("float", s["depth_all"]),
            cast("float", cast("dict[str, float | None]", s["accuracy"])["overall"]),
        )
        for s in cast("list[dict[str, object]]", fixed["static"])
    ]
    static_best = static_frontier(points, depth)
    overall = cast("float", accuracy["overall"])
    checks = {
        "every_class_within_one_point_and_above_half": all(per_class.values()),
        "depth_reduced_by_20_percent": depth <= (1.0 - DEPTH_REDUCTION) * full_depth,
        "above_random_control": overall > random_overall,
        "above_static_frontier": overall > static_best,
    }
    trial = protocol["trial"] is True
    return {
        "per_class": per_class,
        "mean_depth": depth,
        "full_depth": full_depth,
        "random_overall": random_overall,
        "random_depth_difference": random_depth_difference,
        "static_frontier_at_policy_depth": static_best,
        "checks": checks,
        "criteria_met": all(checks.values()),
        "assessment": "trial" if trial else "diagnostic",
        "promising": all(checks.values()) if trial else None,
        "protocol": protocol,
        "scope": " ".join(
            [
                f"the design's calibration criteria on {protocol['eval_episodes']} episodes",
                f"after {protocol['updates']} updates; logical depth, not wall time;",
                "not an equivalence test"
                + ("" if trial else "; DIAGNOSTIC, not the prescribed trial"),
            ]
        ),
    }


def _dense_counts(model: Vesper, batches: Sequence[Episodes]) -> Tensor:
    """``int64 [T, C, 2]``: each pass's own readout under the dense forward, per class."""
    total = torch.zeros(model.loop.t_max, len(CLASSES), 2, dtype=torch.int64)
    for ep in batches:
        c = ep.chunk
        states = model.init_states(int(c.tokens.shape[0]), torch.float32, CPU)
        result = model.run(c.tokens, c.pos, states, exit_rule=DenseRule())
        total += torch.stack([class_counts(model, h, ep) for h in result.per_pass.unbind()])
    return total


def _admit(
    model: Vesper,
    source_result: dict[str, object],
    batches: Sequence[Episodes],
    required_accuracy: float,
) -> dict[str, object]:
    """The design's checks before training, on the CPU reference model with the starting head.

    The source's dense per-pass counts are recomputed on the evaluation set its own result
    records (split, batches, size) and must equal its ``counts.dense_pass``.
    """
    counts = cast("dict[str, object]", source_result["counts"])
    source_set = held_out(
        cast("int", source_result["eval_batches"]),
        cast("int", source_result["eval_size"]),
        cast("str", source_result["eval_split"]),
    )
    with evaluating(model) as m:
        reproduced = plain(_dense_counts(m, source_set)) == counts["dense_pass"]
        baseline = class_accuracy(_dense_counts(m, batches)[-1])
        above = all(
            v is not None and v > required_accuracy for k, v in baseline.items() if k != "overall"
        )
        c = batches[0].chunk
        b = int(c.tokens.shape[0])
        dense = m.run(c.tokens, c.pos, m.init_states(b, torch.float32, CPU), exit_rule=DenseRule())
        rule = CdfRule(m.loop.halting_head(), Q)
        live = m.run(c.tokens, c.pos, m.init_states(b, torch.float32, CPU), exit_rule=rule)
        last = bool((live.n_iters == m.loop.t_max).all()) and torch.equal(live.y, dense.y)
    return {
        "source_counts_reproduced": reproduced,
        "full_depth_baseline": baseline,
        "required_accuracy": required_accuracy,
        "every_class_above_required": above,
        "initial_policy_exits_last_with_dense_output": last,
        "admitted": reproduced and above and last,
    }


def _write_line(log: TextIO, record: dict[str, object]) -> None:
    _ = log.write(json.dumps(record) + "\n")
    log.flush()


@dataclass(frozen=True, slots=True)
class _Plan:
    """What one run's training loop follows."""

    seed: int
    floor: str
    updates: int
    save_at: tuple[int, ...]
    preflight: bool
    target: torch.device
    depth_cost: float
    label_weight: str
    dense_weight: float


def _checked(
    floor: str, device: str, updates: int, save_at: Sequence[int], *, preflight: bool
) -> tuple[torch.device, str]:
    """The training device and scan after checking every setting (``run``'s refusals)."""
    _ = _floor_at(floor, 0)
    if device not in DEVICES:
        msg = f"device must be one of {DEVICES}, got {device!r}"
        raise ValueError(msg)
    target = torch.device(device)
    scan = TRAIN_SCAN[device]
    _ = check_capturable_scan(scan, target)
    if updates < 1 or (not preflight and any(not 0 <= s <= updates for s in save_at)):
        msg = f"updates must be at least 1 and every save point within 0..{updates}"
        raise ValueError(msg)
    return target, scan


def _source_model(source: Path, source_sha256: str) -> tuple[WarmStart, Vesper, list[nn.Parameter]]:
    """The source's weights in a model with the head at its calibration start (CPU, eval mode).

    Raises:
        ValueError: on a sha256 other than ``source_sha256`` or weights that do not fit.
    """
    warm = read_warm_start(source)
    if warm.record["sha256"] != source_sha256:
        msg = f"{source}: sha256 {warm.record['sha256']} is not the required {source_sha256}"
        raise ValueError(msg)
    model = Vesper(warm.source_config, torch.Generator().manual_seed(0))
    load_weights(model, warm.state)
    params = start_head(model)
    _ = model.eval()  # nothing here trains in train mode: every buffer stays as loaded
    return warm, model, params


def _check_options(
    label_weight: str, depth_cost: float, required_accuracy: float, train: str, dense_weight: float
) -> None:
    """Refuse an invalid exploratory option before anything is read or written (``run``).

    Raises:
        ValueError: on an unknown label weight or train mode, a depth cost or dense weight that
            is not finite and at least 0, a required accuracy that is not finite, or a dense
            weight above 0 outside a joint run.
    """
    if label_weight not in LABEL_WEIGHTS:
        msg = f"unknown label weight {label_weight!r} (one of {LABEL_WEIGHTS})"
        raise ValueError(msg)
    if not (math.isfinite(depth_cost) and depth_cost >= 0.0):
        msg = f"the depth cost must be finite and at least 0, not {depth_cost}"
        raise ValueError(msg)
    if not math.isfinite(required_accuracy):
        msg = f"the required accuracy must be finite, not {required_accuracy}"
        raise ValueError(msg)
    if train not in TRAIN_MODES:
        msg = f"unknown train mode {train!r} (one of {TRAIN_MODES})"
        raise ValueError(msg)
    if not (math.isfinite(dense_weight) and dense_weight >= 0.0):
        msg = f"the dense weight must be finite and at least 0, not {dense_weight}"
        raise ValueError(msg)
    if dense_weight > 0.0 and train != "joint":
        msg = "a dense weight above 0 trains the backbone: only in a joint run"
        raise ValueError(msg)


def _train(
    model: Vesper,
    head: list[nn.Parameter],
    backbone: list[nn.Parameter],
    plan: _Plan,
    out: Path,
    eval_set: Sequence[Episodes],
    full_counts: Tensor | None,
) -> dict[str, object]:
    """The training loop with its checkpoints; the identity, timing and evaluation records.

    A joint run (``backbone`` not empty) also saves the whole model's weights at every save
    point and measures the CURRENT backbone's full-depth and static profiles there, beside the
    evaluation against the source's (which stays the comparison target).
    """
    cuda = plan.target.type == "cuda"
    opt = optimizer(head, backbone, capturable=cuda)
    graph = WarmedGraph() if cuda else None  # every update on CUDA: captured after the warm ones
    params = head + backbone
    data = generator_for(TRAIN_SPLIT, plan.seed)
    action = generator_for(ACTION_SPLIT, plan.seed)
    t_max = model.loop.t_max
    consumed, draws, training = "", "", 0.0
    evaluations: dict[str, object] = {}
    reset_memory_peak(plan.target)
    started = time.monotonic()
    with (
        # CUDA runs repeat (10-19); the settings in effect are recorded from inside the scope
        deterministic_algorithms() if cuda else nullcontext(None) as determinism,
        (out / "train.jsonl").open("w", encoding="utf-8") as log,
        (out / "eval.jsonl").open("w", encoding="utf-8") as evals,
    ):

        def checkpoint(after: int) -> None:
            if plan.preflight or after not in plan.save_at:
                return
            (out / "heads").mkdir(exist_ok=True)
            halting = model.loop.halting_head()
            torch.save(
                {name: p.detach().to(CPU).clone() for name, p in halting.named_parameters()},
                out / "heads" / f"after-{after:05d}.pt",
            )
            if backbone:
                (out / "weights").mkdir(exist_ok=True)
                torch.save(
                    {
                        name: t.detach().to(CPU).clone()
                        for name, t in cast("dict[str, Tensor]", model.state_dict()).items()
                    },
                    out / "weights" / f"after-{after:05d}.pt",
                )
            if full_counts is not None:
                evaluation = evaluate_policy(model, eval_set, full_counts) | {"after": after}
                if backbone:
                    evaluation["current_fixed"] = fixed_depths(model, eval_set)[0]
                evaluations[str(after)] = evaluation
                _write_line(evals, evaluation)

        checkpoint(0)
        for update in range(plan.updates):
            ep = episodes(data, EPISODES, f"{TRAIN_SPLIT}-{plan.seed}-{update}", TERMINAL_MIX)
            n = int(ep.chunk.tokens.shape[1])
            uniforms = torch.rand(
                (t_max - 1, ROLLOUTS, EPISODES, n), generator=action, dtype=torch.float64
            ).reshape(t_max - 1, ROLLOUTS * EPISODES, n)
            consumed = chained_digest(consumed, chunk_tensors(ep.chunk))
            draws = chained_digest(draws, [uniforms])
            epsilon = _floor_at(plan.floor, update)
            tick = time.monotonic()
            record = train_update(
                model,
                opt,
                params,
                ep,
                uniforms,
                epsilon,
                update,
                plan.depth_cost,
                plan.label_weight,
                plan.dense_weight,
                graph,
            )
            if cuda:
                torch.cuda.synchronize(plan.target)
            training += time.monotonic() - tick
            _write_line(log, record | {"consumed_digest": consumed, "draw_digest": draws})
            checkpoint(update + 1)
    return {
        "consumed_digest": consumed,
        "draw_digest": draws,
        "train_seconds": training,
        "seconds_per_update": training / plan.updates,
        "wall_seconds": time.monotonic() - started,
        "peak_reserved_bytes": memory_peak(plan.target) if cuda else None,
        "evaluations": evaluations,
        "determinism": determinism,
    }


def _preflight_summary(out: Path, seconds_per_update: float) -> dict[str, object]:
    """The spread of the two cost terms between paired rollouts, and the projected trial time."""
    lines = [
        cast("dict[str, float]", json.loads(line))
        for line in (out / "train.jsonl").read_text("utf-8").splitlines()
    ]
    count = len(lines)
    return {
        "preflight_spread": {
            "pair_nll_diff_std_mean": sum(r["pair_nll_diff_std"] for r in lines) / count,
            "pair_depth_cost_diff_std_mean": sum(r["pair_depth_cost_diff_std"] for r in lines)
            / count,
        },
        "projected_trial_seconds": TRIAL_RUNS * UPDATES * seconds_per_update,
    }


def run(  # noqa: PLR0913
    source: Path,
    source_sha256: str,
    seed: int,
    floor: str,
    out: Path,
    *,
    device: str = "cpu",
    updates: int = UPDATES,
    eval_batches: int = EVAL_BATCHES,
    preflight: bool = False,
    save_at: Sequence[int] = SAVE_AT,
    required_accuracy: float = REQUIRED_ACCURACY,
    depth_cost: float = DEPTH_COST,
    label_weight: str = LABEL_WEIGHT,
    train: str = TRAIN_MODE,
    dense_weight: float = 0.0,
) -> dict[str, object]:
    """One calibration run (module docstring); writes ``out`` and returns its result.

    ``depth_cost`` other than the design's ``DEPTH_COST``, ``label_weight`` other than
    ``LABEL_WEIGHT`` (``_label_weights``), ``train="joint"`` (the backbone trains too, from the
    same source and head start, at ``BACKBONE_LR``; ``joint_update.update_loss``) or a
    ``dense_weight`` above 0 (joint only) is an exploratory run, diagnostic by ``trial_protocol``.

    ``preflight`` runs ``updates`` disposable updates with no evaluation and no saved heads.
    ``required_accuracy`` is the admission floor every full-depth class must exceed (the
    design's 50 %; recorded; lowered by tests, whose untrained sources cannot reach it, and
    diagnostically by ``--required-accuracy`` for sources weak in some class).

    Raises:
        ValueError: on an unknown floor, device, label weight or train mode, a depth cost
            that is not finite and at least 0, a required accuracy that is not finite, a dense
            weight that is not finite and at least 0 or above 0 outside a joint run, a scan the
            device cannot run, updates below 1, a save point outside ``0..updates``, a source
            whose sha256 differs or that does not fit, all before ``out`` exists.
        FileExistsError: if ``out`` exists.
        RuntimeError: when an admission check fails (result.json written first).
        FloatingPointError: on a non-finite update (the updates before it logged).
    """
    _check_options(label_weight, depth_cost, required_accuracy, train, dense_weight)
    target, scan = _checked(floor, device, updates, save_at, preflight=preflight)
    warm, model, params = _source_model(source, source_sha256)
    initial = chained_digest("", list(cast("dict[str, Tensor]", model.state_dict()).values()))
    source_result = cast(
        "dict[str, object]", json.loads((warm.directory / "result.json").read_text("utf-8"))
    )
    eval_generator = generator_for(EVAL_SPLIT, 0)
    eval_set = [
        episodes(eval_generator, EVAL_SIZE, f"{EVAL_SPLIT}-{i}", TERMINAL_MIX)
        for i in range(eval_batches)
    ]
    admission = _admit(model, source_result, eval_set, required_accuracy)
    out.mkdir(parents=True, exist_ok=False)
    _ = (out / SNAPSHOT).write_bytes(warm.recipe)
    result: dict[str, object] = {
        "design": "docs/reviews/2026-10-18-halting-exploration-calibration.md",
        "plan": "docs/reviews/2026-10-17-halting-in-phases.md (adopted, NOT PROVEN)",
        "preflight": preflight,
        "source": warm.record,
        "source_required_sha256": source_sha256,
        "config_snapshot": SNAPSHOT,
        "head_start": {"direction": 0.0, "gain": 1.0, "bias": HEAD_BIAS},
        "initial_state_digest": initial,
        "seed": seed,
        "floor": floor,
        "floor_start": FLOOR_START if floor == "taper" else 0.0,
        "floor_end": FLOOR_END if floor == "taper" else None,
        "splits": {"train": TRAIN_SPLIT, "action": ACTION_SPLIT, "eval": EVAL_SPLIT},
        "episodes": EPISODES,
        "rollouts": ROLLOUTS,
        "updates": updates,
        "depth_cost": depth_cost,
        "label_weight": label_weight,
        "train": train,
        "dense_weight": dense_weight,
        "optimizer": {
            "peak_lr": PEAK_LR,
            "backbone_peak_lr": BACKBONE_LR if train == "joint" else None,
            "warmup": WARMUP,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": 0.0,
            "grad_clip": GRAD_CLIP,
        },
        "q": Q,
        "eval_batches": eval_batches,
        "eval_size": EVAL_SIZE,
        "save_at": list(save_at),
        "protocol": trial_protocol(
            updates,
            eval_batches,
            save_at,
            required_accuracy,
            preflight=preflight,
            depth_cost=depth_cost,
            label_weight=label_weight,
            train=train,
            dense_weight=dense_weight,
        ),
        "admission": admission,
    }
    if not admission["admitted"]:
        write_json(out / "result.json", result)
        msg = f"admission failed: {admission}"
        raise RuntimeError(msg)
    fixed, full_counts = (None, None) if preflight else fixed_depths(model, eval_set)
    if fixed is not None:
        result["fixed"] = fixed
    backbone = release_backbone(model) if train == "joint" else []
    precision = training_precision(cuda=target.type == "cuda")
    model = model.to(target)
    _ = set_scan(model, scan)
    plan = _Plan(
        seed,
        floor,
        updates,
        tuple(save_at),
        preflight,
        target,
        depth_cost,
        label_weight,
        dense_weight,
    )
    trained = _train(model, params, backbone, plan, out, eval_set, full_counts)
    result |= trained | {
        "device": device,
        "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else None,
        "scan": scan_of(model),
        "precision": precision,
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fla-core": version_or_none("fla-core"),
        },
        "environment": environment(),
    }
    terminal = cast("dict[str, object]", trained["evaluations"]).get(str(updates))
    if fixed is not None and terminal is not None:
        protocol = cast("dict[str, object]", result["protocol"])
        result["verdict"] = verdict(cast("dict[str, object]", terminal), fixed, protocol)
    if preflight:
        result |= _preflight_summary(out, cast("float", trained["seconds_per_update"]))
    write_json(out / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the stopping head on a frozen backbone under sampled stops (the design), "
            "or with --train joint the backbone too (exploratory)."
        )
    )
    _ = parser.add_argument("--source", type=Path, required=True)
    _ = parser.add_argument("--source-sha256", required=True)
    _ = parser.add_argument("--seed", type=int, required=True)
    _ = parser.add_argument("--floor", choices=FLOORS, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--device", choices=DEVICES, default="cpu")
    _ = parser.add_argument("--updates", type=int, default=None)
    _ = parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    _ = parser.add_argument("--threads", type=int, default=2)
    _ = parser.add_argument(
        "--depth-cost",
        type=float,
        default=DEPTH_COST,
        help=f"the depth term's weight ({DEPTH_COST} is the design's; others are exploratory)",
    )
    _ = parser.add_argument(
        "--save-at",
        type=int,
        nargs="+",
        default=SAVE_AT,
        help="updates to save and evaluate the head at (others than the design's: exploratory)",
    )
    _ = parser.add_argument(
        "--required-accuracy",
        type=float,
        default=REQUIRED_ACCURACY,
        help=f"admission floor of every full-depth class ({REQUIRED_ACCURACY}: the design's)",
    )
    _ = parser.add_argument(
        "--label-weight",
        choices=LABEL_WEIGHTS,
        default=LABEL_WEIGHT,
        help=f"how labels form the NLL term ({LABEL_WEIGHT}: the design's; class: exploratory)",
    )
    _ = parser.add_argument(
        "--train",
        choices=TRAIN_MODES,
        default=TRAIN_MODE,
        help=f"what trains ({TRAIN_MODE}: the design's; joint: the backbone too, exploratory)",
    )
    _ = parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.0,
        help="rho of a joint run's dense all-pass auxiliary (0: none)",
    )
    _ = parser.add_argument(
        "--preflight",
        action="store_true",
        help=f"a disposable timing run ({PREFLIGHT_UPDATES} updates unless --updates)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the module docstring."""
    args = _parser().parse_args(argv)
    torch.set_num_threads(cast("int", args.threads))
    preflight = cast("bool", args.preflight)
    given = cast("int | None", args.updates)
    updates = given if given is not None else (PREFLIGHT_UPDATES if preflight else UPDATES)
    _ = run(
        cast("Path", args.source),
        cast("str", args.source_sha256),
        cast("int", args.seed),
        cast("str", args.floor),
        cast("Path", args.out),
        device=cast("str", args.device),
        updates=updates,
        eval_batches=cast("int", args.eval_batches),
        depth_cost=cast("float", args.depth_cost),
        save_at=tuple(cast("Sequence[int]", args.save_at)),
        required_accuracy=cast("float", args.required_accuracy),
        label_weight=cast("str", args.label_weight),
        train=cast("str", args.train),
        dense_weight=cast("float", args.dense_weight),
        preflight=preflight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
