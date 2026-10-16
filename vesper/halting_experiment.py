"""The halting experiment's stage-1 driver: train one condition on the synthetic task, evaluate it.

Plan: docs/reviews/2026-10-15-halting-experiment-plan.md (stage 1); task: ``vesper.synthetic``.

    python -m vesper.halting_experiment --condition fixed --seed 0 --out RUN_DIR \
        [--config configs/halting-stage1.yaml] [--steps N] [--beta 0.1] [--eval-batches 8]

Conditions (the plan's table): ``act`` (A: ACT with its mixture readout and ponder cost, the
trainer's own phase schedule and head freeze, its copy-forward state policy), ``fixed`` (B:
uniform exit weights, no head trained, deployed with ``CdfRule(None, q)``), ``local`` (C) and
``aggregate`` (D) (the head from ``zero_residual``, trained from the first step, deployed with
``CdfRule(head, q)``). B-D train DENSE (``DenseRule``). Every condition shares the model
initialization, the data order (``generator_for("train", seed)``), the optimizer and the
learning-rate schedule; each step is one batch of fresh sequences with cold states (a sequence
is a whole document), so the aggregate KL's population is exactly the optimizer batch.

Training draws each step's batch with the hop mix of that step (``synthetic.hop_mix``).
Evaluation, on one held-out set shared by every run (``held_out``: ``generator_for(split, 0)``,
the terminal mix; ``--eval-split validation`` for the probe's gates, ``final`` for the stage-1
comparison):

- ``dense_pass``: each pass's own readout under the dense forward, per class: the dense state
  policy's static depth curve, and from B each class's empirical depth need.
- for B-D, per q of ``Q_GRID``: (i) ``dense_exit``, the dense forward read at the CDF exit;
  (ii) ``trace``, the same exit trace replayed with copy-forward (``TraceRule``); (iii)
  ``live``, the gates recomputed under copy-forward (``CdfRule``), with their measured passes.
- ``deployed``: the deployed policy (B-D at ``Q_REFERENCE``, A its own rule) over ``prefill``,
  token-by-token ``decode`` and ``carried`` chunks of ``CARRIED_TOKENS``, with the largest
  output difference from prefill and the agreement of the depths.
- ``static``: every token forced to depth d under copy-forward, d = 1..T.
- ``random``: the deployed prefill trace permuted across the held-out positions (the same
  passes exactly), replayed with ``TraceRule``; for A that replays a trace without ACT's
  mixture readout, and is labelled so.
- ``depth``: the deployed depth histogram per role and per class (marginal and conditional
  entropies), and for B-D the soft information between token and exit from the dense head.
- ``register``: reader accuracy over forced reader depth r and register depth g, the rest of
  the trace as deployed; g index 0 is the register's own deployed depth.

Every accuracy is kept as counts (correct, total) per class, so seeds can be pooled.
"""

import argparse
import copy
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from vesper.column import Column
from vesper.config import VesperConfig, parse_bytes
from vesper.exits import exit_log_probs, hazard_cdf_exits, hazard_logits
from vesper.learning import LossWeights
from vesper.loop import CdfRule, DenseRule, ExitRule, TraceRule
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.synthetic import (
    CLASSES,
    GAP,
    READ_POSITION,
    REGISTER_POSITION,
    ROLES,
    SEQUENCE_TOKENS,
    TERMINAL_MIX,
    VOCAB,
    Episodes,
    episodes,
    generator_for,
    hop_mix,
)
from vesper.train import optimizer, phase_schedule, set_halting, set_learning_rate, total_steps

CONDITIONS = ("act", "fixed", "local", "aggregate")  # the plan's A, B, C, D
Q_GRID = tuple(k / 10 for k in range(1, 11))
Q_REFERENCE = 0.5
CARRIED_TOKENS = 8
EVAL_SPLITS = ("validation", "final")
WEIGHTS_EVERY = 1000  # updates between diagnostic weight snapshots
# a run's recipe snapshot, named relative to its result.json: a copied run directory reads
# without the creator's paths (maintainer B's re-review of 9044238)
SNAPSHOT = "config.yaml"
EVAL_BATCH = 32  # 32 batches of 32: the probe's 1024 fixed validation episodes
CPU = torch.device("cpu")

RuleFor = Callable[[int, int], ExitRule | None]


def _check_setup(cfg: VesperConfig, condition: str) -> None:
    """The configuration fits the task and the condition is known.

    Raises:
        ValueError: on an unknown condition, a vocabulary or chunk size other than the task's,
            or an attention window that reaches past the task's gap (a query could then attend
            to a definition directly; stacked attentions can relay across it anyway, so the
            task is hybrid and the recurrence ablation measures the reliance).
    """
    if condition not in CONDITIONS:
        msg = f"condition must be one of {CONDITIONS}, got {condition!r}"
        raise ValueError(msg)
    m = cfg.model
    if m.vocab != VOCAB or cfg.training.chunk_tokens != SEQUENCE_TOKENS or m.window > GAP:
        msg = (
            f"the task needs vocab {VOCAB}, chunk_tokens {SEQUENCE_TOKENS} and window <= {GAP}; "
            f"got {m.vocab}, {cfg.training.chunk_tokens} and {m.window}"
        )
        raise ValueError(msg)


def build(cfg: VesperConfig, condition: str, seed: int) -> Vesper:
    """The model for one run: the seed's initialization, the condition's halting head.

    ``local`` and ``aggregate`` start from ``zero_residual`` (exactly uniform with the
    baseline, never ACT's frozen bias); ``fixed`` freezes the head, which it never reads.

    Raises:
        ValueError: on an unknown condition or a config the task cannot use (``_check_setup``).
    """
    _check_setup(cfg, condition)
    model = Vesper(cfg, torch.Generator().manual_seed(seed))
    head = model.loop.halting_head()
    if condition in {"local", "aggregate"}:
        head.zero_residual()
    if condition == "fixed":
        for parameter in head.parameters():
            parameter.requires_grad = False
    return model


def train_step(
    model: Vesper,
    opt: torch.optim.AdamW,
    cfg: VesperConfig,
    condition: str,
    beta: float,
    ep: Episodes,
    step: int,
) -> dict[str, object]:
    """One optimizer step on one batch; returns the step's record."""
    c = ep.chunk
    b = int(c.tokens.shape[0])
    phase = phase_schedule(step * b * SEQUENCE_TOKENS, step, cfg)
    states = model.init_states(b, torch.float32, CPU)
    if condition == "act":
        set_halting(model, phase)
        result = model.run(c.tokens, c.pos, states, None, phase.t_max)
        weights = LossWeights(lambda_h=phase.lambda_h, lambda_g=cfg.learning.lambda_g)
        loss, parts = model.objective(c, result, cfg, weights)
    else:
        result = model.run(c.tokens, c.pos, states, exit_rule=DenseRule())
        loss, parts = model.all_pass_objective(c, result, condition=condition, beta=beta)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([loss])
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(nn.utils.clip_grad_norm_(parameters, cfg.optimizer.grad_clip).item())
    set_learning_rate(opt, phase.lr)
    # the optimizer's step is untyped in torch; its result (a closure's loss) is not used here
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    record: dict[str, object] = {
        "step": step,
        "loss": float(loss.item()),
        "lr": phase.lr,
        "grad_norm": grad_norm,
        "loss_depth": float(result.n_iters[c.loss_mask].float().mean().item()),
    }
    for name, value in parts.items():
        record[name] = [float(v) for v in value.unbind()] if value.dim() else float(value.item())
    return record


def class_counts(model: Vesper, y: Tensor, ep: Episodes) -> Tensor:
    """``int64[C, 2]``: correct and total per class of ``CLASSES`` from ``y [b, n, d]``.

    The prediction is the head's argmax at the loss positions.
    """
    mask = ep.chunk.loss_mask
    predicted = model.head(y[mask][None])[0].argmax(-1)
    right = predicted == ep.chunk.targets[mask].long()
    task = ep.task[mask]
    correct = torch.bincount(task[right], minlength=len(CLASSES))
    total = torch.bincount(task, minlength=len(CLASSES))
    return torch.stack([correct, total], dim=1)


def _run_chunks(
    model: Vesper, ep: Episodes, rule_for: RuleFor, size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """``(y [b, n, d], n_iters [b, n], per_pass [T, b, n, d])`` over chunks with carried states.

    ``size = n`` is prefill, 1 is token-by-token decode; ``rule_for(start, end)`` gives the
    exit rule of one chunk (a trace's slice, or the same live rule for every chunk).
    """
    c = ep.chunk
    b, n = int(c.tokens.shape[0]), int(c.tokens.shape[1])
    states = model.init_states(b, torch.float32, CPU)
    ys: list[Tensor] = []
    iters: list[Tensor] = []
    passes: list[Tensor] = []
    for start in range(0, n, size):
        end = min(start + size, n)
        result = model.run(
            c.tokens[:, start:end], c.pos[:, start:end], states, exit_rule=rule_for(start, end)
        )
        states = list(result.states)
        ys.append(result.y)
        iters.append(result.n_iters)
        passes.append(result.per_pass)
    return torch.cat(ys, dim=1), torch.cat(iters, dim=1), torch.cat(passes, dim=2)


def _dense_rule(start: int, end: int) -> ExitRule | None:
    """Every chunk dense (``DenseRule``)."""
    del start, end
    return DenseRule()


def _readout(per_pass: Tensor, exits: Tensor) -> Tensor:
    """Each token's own output at its exit pass: ``per_pass [T, b, n, d]``, ``exits [b, n]``."""
    b, n, d = per_pass.shape[1], per_pass.shape[2], per_pass.shape[3]
    index = (exits.long() - 1)[None, :, :, None].expand(1, b, n, d)
    return per_pass.gather(0, index)[0]


def recurrence_ablated(model: Vesper) -> Vesper:
    """A copy of ``model`` whose recurrent sublayers read nothing into the residual stream.

    Every column's recurrent output projection ``w_o`` is zeroed (the only path from the
    recurrent read to ``x``): the states are still written, never read. The task is hybrid
    (stacked window attentions relay across the gap, maintainer B's review of c7d25c5), so this copy
    measures how much each answer depends on the recurrent state. ``model`` is not touched.
    """
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        for column in ablated.loop.columns:
            if not isinstance(column, Column):
                msg = "the recurrence ablation needs vesper.column.Column columns"
                raise TypeError(msg)
            _ = column.rec.w_o.zero_()
    return ablated


def _trace_of(depths: Tensor) -> RuleFor:
    """A ``TraceRule`` per chunk, from one trace over the whole sequence."""

    def rule_for(start: int, end: int) -> ExitRule | None:
        return TraceRule(depths[:, start:end])

    return rule_for


def _deployed_rule(model: Vesper, condition: str, q: float) -> RuleFor:
    """The condition's deployed policy: ACT's own rule for A, the CDF rule at ``q`` for B-D."""
    if condition == "act":
        rule: ExitRule | None = None
    else:
        rule = CdfRule(None if condition == "fixed" else model.loop.halting_head(), q)

    def rule_for(start: int, end: int) -> ExitRule | None:
        del start, end  # one live rule for every chunk: its decision is per token
        return rule

    return rule_for


def _dense_hazards(model: Vesper, condition: str, per_pass: Tensor) -> Tensor:
    """Hazard logits ``[T, b, n]`` (float64) of B-D from the dense forward's pass outputs."""
    t_max, b, n = per_pass.shape[0], per_pass.shape[1], per_pass.shape[2]
    if condition == "fixed":
        return hazard_logits(torch.zeros(t_max, b, n))
    head = model.loop.halting_head()
    return hazard_logits(torch.stack([head.logit(per_pass[t], t) for t in range(t_max)]))


class Totals:
    """Tensors summed over the held-out batches, by name."""

    def __init__(self) -> None:
        """Nothing summed yet."""
        self.values: dict[str, Tensor] = {}

    def add(self, key: str, value: Tensor) -> None:
        """Add ``value`` to the running sum under ``key``."""
        previous = self.values.get(key)
        self.values[key] = value.clone() if previous is None else previous + value

    def peak(self, key: str, value: float) -> None:
        """Keep the largest ``value`` seen under ``key``."""
        previous = self.values.get(key)
        tensor = torch.tensor(value, dtype=torch.float64)
        self.values[key] = tensor if previous is None else torch.maximum(previous, tensor)


def _passes(n_iters: Tensor, ep: Episodes) -> Tensor:
    """``[4]``: summed passes and count over all positions, then over the loss positions."""
    mask = ep.chunk.loss_mask
    return torch.stack(
        [
            n_iters.sum().double(),
            torch.tensor(float(n_iters.numel()), dtype=torch.float64),
            n_iters[mask].sum().double(),
            mask.sum().double(),
        ]
    )


def evaluate_batch(  # noqa: PLR0915
    model: Vesper,
    condition: str,
    ep: Episodes,
    totals: Totals,
    ablated: Vesper | None = None,
) -> None:
    """Every measurement of the module docstring on one held-out batch, added to ``totals``.

    ``ablated`` is ``model`` with the recurrent read removed (``recurrence_ablated``), made
    here when not given; ``evaluate`` makes it once for all batches.
    """
    c = ep.chunk
    b, n = int(c.tokens.shape[0]), int(c.tokens.shape[1])
    _, _, per_pass = _run_chunks(model, ep, _dense_rule, n)
    t_max = int(per_pass.shape[0])
    totals.add(
        "dense_pass", torch.stack([class_counts(model, per_pass[t], ep) for t in range(t_max)])
    )
    if condition != "act":
        z = _dense_hazards(model, condition, per_pass)
        log_p = exit_log_probs(z)
        for scope, where in (("all", c.valid), ("loss", c.loss_mask)):
            p = log_p.exp()[:, where]  # [T, N]
            totals.add(f"soft/{scope}/p", p.sum(1))
            totals.add(f"soft/{scope}/plogp", (p * log_p[:, where]).sum().reshape(1))
            totals.add(f"soft/{scope}/count", _vector(float(p.shape[1])))
        for q in Q_GRID:
            exits = hazard_cdf_exits(z, q).long()
            totals.add(f"dense_exit/{q}", class_counts(model, _readout(per_pass, exits), ep))
            totals.add(f"trace_passes/{q}", _passes(exits, ep))
            traced, _, _ = _run_chunks(model, ep, _trace_of(exits), n)
            totals.add(f"trace/{q}", class_counts(model, traced, ep))
            live, live_iters, _ = _run_chunks(model, ep, _deployed_rule(model, condition, q), n)
            totals.add(f"live/{q}", class_counts(model, live, ep))
            totals.add(f"live_passes/{q}", _passes(live_iters, ep))
            agree = _vector(float((live_iters == exits).sum()), float(exits.numel()))
            totals.add(f"live_agrees_with_dense_exits/{q}", agree)
    d0 = _measure_modes(model, condition, ep, totals, per_pass)
    static: list[Tensor] = []
    for depth in range(1, t_max + 1):
        fixed = torch.full((b, n), depth, dtype=torch.int64)
        y, _, _ = _run_chunks(model, ep, _trace_of(fixed), n)
        static.append(class_counts(model, y, ep))
    totals.add("static", torch.stack(static))
    order = torch.randperm(b * n, generator=torch.Generator().manual_seed(b * n))
    shuffled = d0.reshape(-1)[order].reshape(b, n).long()
    y, _, _ = _run_chunks(model, ep, _trace_of(shuffled), n)
    totals.add("random", class_counts(model, y, ep))
    totals.add("random_passes", _passes(shuffled, ep))
    mask = c.loss_mask
    depth_index = d0.long() - 1
    totals.add(
        "depth/role",
        torch.bincount(
            ep.role.reshape(-1) * t_max + depth_index.reshape(-1), minlength=len(ROLES) * t_max
        ).reshape(len(ROLES), t_max),
    )
    totals.add(
        "depth/class",
        torch.bincount(
            ep.task[mask] * t_max + depth_index[mask], minlength=len(CLASSES) * t_max
        ).reshape(len(CLASSES), t_max),
    )
    grid: list[Tensor] = []
    for reader in range(1, t_max + 1):
        row: list[Tensor] = []
        for register in range(t_max + 1):  # 0: the register's own deployed depth
            trace = d0.clone().long()
            trace[:, READ_POSITION] = reader
            if register:
                trace[:, REGISTER_POSITION] = register
            y, _, _ = _run_chunks(model, ep, _trace_of(trace), n)
            row.append(class_counts(model, y, ep))
        grid.append(torch.stack(row))
    totals.add("register", torch.stack(grid))
    # the recurrence ablation, paired: the deployed exit trace replayed (TraceRule, one exit per
    # token) by the model and by its ablated copy, and the dense last pass of the copy against
    # the model's own; the only difference within a pair is the recurrent read. Not against
    # "deployed/prefill": A's deployed readout is its ACT mixture, so that comparison changed
    # predictions with no weight changed (maintainer B's re-review of 9044238)
    without = recurrence_ablated(model) if ablated is None else ablated
    replay = _trace_of(d0.long())
    y_reference, _, _ = _run_chunks(model, ep, replay, n)
    y, _, _ = _run_chunks(without, ep, replay, n)
    totals.add("ablated/reference", class_counts(model, y_reference, ep))
    totals.add("ablated/trace", class_counts(without, y, ep))
    totals.peak("ablated_max_output_difference", float((y - y_reference).abs().max()))
    _, _, ablated_passes = _run_chunks(without, ep, _dense_rule, n)
    totals.add("ablated/dense_last", class_counts(without, ablated_passes[-1], ep))


def _measure_modes(
    model: Vesper, condition: str, ep: Episodes, totals: Totals, dense_passes: Tensor
) -> Tensor:
    """Prefill, decode and carried chunks of the deployed policy and, for B-D, of (i) and (ii).

    At ``Q_REFERENCE``: the deployed policy's accuracy, passes, largest output difference from
    prefill and its depth and answer agreement; for B-D also (i) the dense forward read at its
    own CDF exits and (ii) those exits replayed with copy-forward, each per mode, with the dense
    exits' agreement with prefill's. Returns the deployed prefill trace.
    """
    c = ep.chunk
    n = int(c.tokens.shape[1])
    mask = c.loss_mask
    rule = _deployed_rule(model, condition, Q_REFERENCE)
    y0, d0, _ = _run_chunks(model, ep, rule, n)
    predicted0 = model.head(y0[mask][None])[0].argmax(-1)
    exits0 = (
        None
        if condition == "act"
        else hazard_cdf_exits(_dense_hazards(model, condition, dense_passes), Q_REFERENCE).long()
    )
    for mode, size in (("prefill", n), ("decode", 1), ("carried", CARRIED_TOKENS)):
        y, d, _ = (y0, d0, None) if mode == "prefill" else _run_chunks(model, ep, rule, size)
        totals.add(f"deployed/{mode}", class_counts(model, y, ep))
        totals.add(f"deployed_passes/{mode}", _passes(d, ep))
        totals.peak(f"deployed_max_output_difference/{mode}", float((y - y0).abs().max()))
        same_depth = float((d == d0).sum())
        same_answer = float((model.head(y[mask][None])[0].argmax(-1) == predicted0).sum())
        totals.add(
            f"deployed_agreement/{mode}",
            _vector(same_depth, float(d.numel()), same_answer, float(mask.sum())),
        )
        if exits0 is None:
            continue
        passes = dense_passes if mode == "prefill" else _run_chunks(model, ep, _dense_rule, size)[2]
        exits = hazard_cdf_exits(_dense_hazards(model, condition, passes), Q_REFERENCE).long()
        totals.add(f"dense_mode/{mode}", class_counts(model, _readout(passes, exits), ep))
        totals.add(
            f"dense_mode_exits/{mode}",
            _vector(float((exits == exits0).sum()), float(exits.numel())),
        )
        traced, _, _ = _run_chunks(model, ep, _trace_of(exits), size)
        totals.add(f"trace_mode/{mode}", class_counts(model, traced, ep))
    return d0


def _vector(*values: float) -> Tensor:
    """A float64 vector of ``values`` (a typed list, so the strict checker sees no ``Any``)."""
    data: list[float] = list(values)
    return torch.tensor(data, dtype=torch.float64)


def _plain(value: Tensor) -> object:
    """A tensor as nested lists of floats, for the JSON record (``tolist`` is untyped)."""
    if value.dim() == 0:
        return float(value.item())
    return [_plain(v) for v in value.unbind()]


def _entropy(counts: Tensor) -> float:
    """Entropy in nats of a histogram (zero for an empty one)."""
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    p = counts.double() / total
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def _conditional_entropy(histogram: Tensor) -> tuple[float, float]:
    """``(H(D), H(D | row))`` of a ``[rows, depths]`` histogram, in nats."""
    total = float(histogram.sum())
    marginal = _entropy(histogram.sum(0))
    conditional = sum(
        float(row.sum()) / total * _entropy(row) for row in histogram.unbind() if total
    )
    return marginal, conditional


def _accuracy(counts: Tensor) -> dict[str, float | None]:
    """Per-class and overall accuracy from ``[C, 2]`` counts; None for a class never seen.

    None and not NaN: the record is JSON, which has no NaN (Python writes a non-standard token
    and a NaN never compares equal to itself after reading it back).
    """
    out: dict[str, float | None] = {
        name: float(counts[i, 0]) / float(counts[i, 1]) if float(counts[i, 1]) else None
        for i, name in enumerate(CLASSES)
    }
    out["overall"] = float(counts[:, 0].sum()) / float(counts[:, 1].sum())
    return out


def _cost(passes: Tensor, t_max: int, columns: int) -> dict[str, float]:
    """Mean executed passes over all and over the loss positions, and copy-forward writes."""
    mean_all = float(passes[0] / passes[1])
    return {
        "passes": mean_all,
        "loss_passes": float(passes[2] / passes[3]),
        "replayed_writes": columns * (t_max - mean_all),
    }


def summarize(values: dict[str, Tensor], t_max: int, columns: int) -> dict[str, object]:
    """Accuracies, costs, entropies and the register summary from the summed counts."""
    out: dict[str, object] = {}
    out["dense_pass"] = [_accuracy(v) for v in values["dense_pass"].unbind()]
    out["static"] = [_accuracy(v) for v in values["static"].unbind()]
    out["random"] = _accuracy(values["random"]) | _cost(values["random_passes"], t_max, columns)
    for key, value in values.items():
        kind, _, suffix = key.partition("/")
        if kind in {
            "dense_exit",
            "trace",
            "live",
            "deployed",
            "dense_mode",
            "trace_mode",
            "ablated",
        }:
            costs = values.get(f"{kind}_passes/{suffix}")
            entry = _accuracy(value)
            if costs is not None and kind != "dense_exit":
                entry |= _cost(costs, t_max, columns)
            out[key] = entry
        elif "_max_" in kind:  # a peak (Totals.peak); the report pools these by max
            out[key] = float(value)
        elif kind == "deployed_agreement":
            out[key] = {"depth": float(value[0] / value[1]), "answer": float(value[2] / value[3])}
        elif kind in {"live_agrees_with_dense_exits", "dense_mode_exits"}:
            out[key] = float(value[0] / value[1])
    for name, histogram in (("role", values["depth/role"]), ("class", values["depth/class"])):
        marginal, conditional = _conditional_entropy(histogram)
        out[f"depth/{name}"] = {
            "histogram": [[int(v) for v in row.unbind()] for row in histogram.unbind()],
            "entropy": marginal,
            "conditional_entropy": conditional,
            "information": marginal - conditional,
        }
    for scope in ("all", "loss"):
        if f"soft/{scope}/count" in values:
            count = float(values[f"soft/{scope}/count"])
            mean = values[f"soft/{scope}/p"] / count
            mean_plogp = float(values[f"soft/{scope}/plogp"]) / count
            out[f"soft_information/{scope}"] = mean_plogp - float(
                (mean * mean.clamp_min(1e-300).log()).sum()
            )
    read = [CLASSES.index(f"read{k}") for k in range(1, 5)]
    grid = values["register"][:, :, read, :].sum(2)  # [reader, register, 2]
    out["register"] = [[float(cell[0] / cell[1]) for cell in row.unbind()] for row in grid.unbind()]
    return out


def evaluate(model: Vesper, condition: str, batches: Sequence[Episodes]) -> dict[str, object]:
    """Every measurement over ``batches``: the summed counts and their summary."""
    totals = Totals()
    was_training = model.training
    _ = model.eval()
    ablated = recurrence_ablated(model)
    with torch.no_grad():
        for ep in batches:
            evaluate_batch(model, condition, ep, totals, ablated)
    _ = model.train(was_training)
    counts = {key: _plain(value) for key, value in totals.values.items()}
    summary = summarize(totals.values, model.t_max, model.columns)
    return {"counts": counts, "summary": summary}


def held_out(batches: int, size: int = EVAL_BATCH, split: str = "validation") -> list[Episodes]:
    """A shared held-out set: ``batches`` of ``size`` from ``generator_for(split, 0)``.

    The terminal hop mix (``TERMINAL_MIX``). ``validation`` calibrates (the probe's gates);
    ``final`` is for the stage-1 comparison and stays unread until then
    (docs/reviews/2026-10-15-retrieval-binding.md, 5.2).

    Raises:
        ValueError: on a split other than these two.
    """
    if split not in EVAL_SPLITS:
        msg = f"split must be one of {EVAL_SPLITS}, got {split!r}"
        raise ValueError(msg)
    g = generator_for(split, 0)
    return [episodes(g, size, f"{split}-{i}", TERMINAL_MIX) for i in range(batches)]


def _probe(model: Vesper, condition: str, ep: Episodes) -> dict[str, object]:
    """A cheap held-out reading for the learning curve: deployed and last-pass accuracy."""
    c = ep.chunk
    n = int(c.tokens.shape[1])
    was_training = model.training
    _ = model.eval()
    with torch.no_grad():
        y, d, _ = _run_chunks(model, ep, _deployed_rule(model, condition, Q_REFERENCE), n)
        dense = model.run(
            c.tokens,
            c.pos,
            model.init_states(int(c.tokens.shape[0]), torch.float32, CPU),
            exit_rule=DenseRule(),
        )
        record: dict[str, object] = {
            "deployed": _accuracy(class_counts(model, y, ep)),
            "deployed_loss_passes": float(d[c.loss_mask].float().mean()),
            "dense_last": _accuracy(class_counts(model, dense.per_pass[-1], ep)),
        }
    _ = model.train(was_training)
    return record


def run(  # noqa: PLR0913
    config: Path,
    condition: str,
    seed: int,
    out: Path,
    *,
    steps: int | None = None,
    beta: float = 0.1,
    eval_batches: int = 32,
    eval_size: int = EVAL_BATCH,
    eval_split: str = "validation",
    log_every: int = 50,
    probe_every: int = 250,
    weights_every: int = WEIGHTS_EVERY,
) -> dict[str, object]:
    """Train one condition and seed, evaluate it, write the run directory ``out``.

    ``out`` receives ``config.yaml`` (``SNAPSHOT``, recorded relative to ``result.json``: the
    config's bytes as read once, before the first step; the run parses and hashes exactly these
    bytes, so a later edit of the source cannot be recorded as the recipe, maintainer B's review of
    c7d25c5), ``train.jsonl``, ``model.pt`` and ``result.json``, and ``weights/after-NNNNN.pt``
    after every ``weights_every`` updates: the model's weights only, DIAGNOSTIC snapshots for
    learning curves and interventions, not recovery checkpoints (no optimizer, generator or
    progress state; an interrupted run starts again). The probe's gates use only the final result.

    Raises:
        FileExistsError: if ``out`` exists (a run directory is written once).
    """
    evaluation_set = held_out(eval_batches, eval_size, eval_split)  # refuses a bad split first
    out.mkdir(parents=True, exist_ok=False)
    recipe = config.read_bytes()
    snapshot = out / SNAPSHOT
    _ = snapshot.write_bytes(recipe)
    cfg = parse_bytes(recipe, str(snapshot))
    model = build(cfg, condition, seed)
    planned = total_steps(cfg)
    steps = planned if steps is None else steps
    opt = optimizer(model, cfg)
    data = generator_for("train", seed)
    probe_set = episodes(generator_for("probe", 0), eval_size, "probe")
    started = time.monotonic()
    with (out / "train.jsonl").open("w", encoding="utf-8") as log:
        for step in range(steps):
            ep = episodes(data, cfg.training.batch_chunks, f"train-{seed}-{step}", hop_mix(step))
            record = train_step(model, opt, cfg, condition, beta, ep, step)
            if (step + 1) % weights_every == 0:
                weights = out / "weights"
                weights.mkdir(exist_ok=True)
                state = cast("dict[str, Tensor]", model.state_dict())
                torch.save(state, weights / f"after-{step + 1:05d}.pt")
            last = step == steps - 1
            if step % log_every == 0 or last:
                record["seconds"] = time.monotonic() - started
                # this batch's labels per class, binding the mix schedule to the run's log
                labels = torch.bincount(ep.task[ep.task >= 0], minlength=len(CLASSES))
                record["labels"] = [int(v) for v in labels.unbind()]
                if step % probe_every == 0 or last:
                    record["probe"] = _probe(model, condition, probe_set)
                _ = log.write(json.dumps(record) + "\n")
                log.flush()
    torch.save(cast("dict[str, Tensor]", model.state_dict()), out / "model.pt")
    trained = time.monotonic()
    evaluation = evaluate(model, condition, evaluation_set)
    result: dict[str, object] = {
        "plan": "docs/reviews/2026-10-15-halting-experiment-plan.md",
        "condition": condition,
        "seed": seed,
        "beta": beta if condition in {"local", "aggregate"} else None,
        "steps": steps,
        "planned_steps": planned,
        "config": str(config),
        "config_snapshot": SNAPSHOT,
        "config_sha256": hashlib.sha256(recipe).hexdigest(),
        "q_reference": Q_REFERENCE,
        "eval_sequences": eval_batches * eval_size,
        "eval_batches": eval_batches,
        "eval_size": eval_size,
        "eval_split": eval_split,
        "train_seconds": trained - started,
        "eval_seconds": time.monotonic() - trained,
        "environment": environment(),
        **evaluation,
    }
    write_json(out / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one halting-experiment condition on the stage-1 task, evaluate it."
    )
    _ = parser.add_argument("--condition", required=True, choices=CONDITIONS)
    _ = parser.add_argument("--seed", type=int, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--config", type=Path, default=Path("configs/halting-stage1.yaml"))
    _ = parser.add_argument("--steps", type=int, default=None)
    _ = parser.add_argument("--beta", type=float, default=0.1)
    _ = parser.add_argument("--eval-batches", type=int, default=32)
    _ = parser.add_argument("--eval-split", choices=EVAL_SPLITS, default="validation")
    _ = parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the module docstring."""
    args = _parser().parse_args(argv)
    torch.set_num_threads(cast("int", args.threads))
    _ = run(
        cast("Path", args.config),
        cast("str", args.condition),
        cast("int", args.seed),
        cast("Path", args.out),
        steps=cast("int | None", args.steps),
        beta=cast("float", args.beta),
        eval_batches=cast("int", args.eval_batches),
        eval_split=cast("str", args.eval_split),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
