"""Joint halting on real text: phase 2 of the pilot.

The design is docs/reviews/2026-10-20-real-data-halting-pilot.md.

The update is ``vesper.joint_update``'s, unchanged (score + pathwise + backbone-only dense, two
rollouts with sampled stops and copy-forward); this module adapts it to text. A plan trains the
backbone with the head (``"joint"``) or keeps it the source's (``"frozen"``: head only, from the
score term alone, no dense term). Its context
contract (maintainer B 0617971): every row of a chunk is an INDEPENDENT context, read from cold
states, in training and in every evaluation compared with it; a carried stream would need its
own state and credit contract. Labels are pooled by their real count (``pooled_weights``), the
active depth is priced over pooled real tokens (``joint_update``'s cost with ``pooled_depth``),
so rows of unequal length weigh by their tokens in both terms, and every evaluation number
is a pooled next-token NLL; perplexity only by exponentiating a pooled NLL, never by mixing
perplexities.

``load_text_source`` reads a ``vesper.train`` checkpoint (manifest hashes, the config file's
own hash, the recipe's objective), ``training_stream`` the training side of the finalized
corpus, ``depth_nll`` every token's NLL at every fixed depth from cold states, and
``evaluate_text`` the deterministic policy (``CdfRule`` at q = 0.5) against the model's own
fixed depths, their lower frontier and depth-matched random controls.
"""

import argparse
import io
import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import Tensor, nn

from vesper.column import (
    ColumnState,
    check_capturable_scan,
    scan_of,
    set_scan,
    state_tensors,
)
from vesper.config import VesperConfig, parse_bytes
from vesper.corpus import (
    Component,
    Split,
    check_inputs,
    iter_mixed_documents,
    load_finalization,
)
from vesper.data import Chunk, Stream
from vesper.evals import EvalGraphs, depth_losses, run_chunks
from vesper.graphs import (
    SignatureGraph,
    WarmedGraph,
    deterministic_algorithms,
    memory_peak,
    reset_memory_peak,
)
from vesper.halting_calibration import static_frontier
from vesper.halting_common import chained_digest, chunk_tensors, version_or_none
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
from vesper.keys import hashed_bytes, verified_bytes
from vesper.loop import CdfRule, ExitRule, TraceRule
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.synthetic import generator_for
from vesper.tokenizer import Tokenizer
from vesper.train import checked_record, held_out_chunks, resumed_components

log = logging.getLogger(__name__)
Q = 0.5  # the deployed policy's exit quantile, as in the synthetic calibration
RANDOM_SEEDS = (0, 1, 2, 3)


def pooled_weights(mask: Tensor) -> Tensor:
    """``float64 [E, n]``: ``E / N`` at every label of the ``E`` unique rows, 0 elsewhere.

    ``N`` is the rows' label count POOLED: the mean over the rows of each row's weighted NLL is
    then the NLL pooled over every real label, whatever the rows' label counts (padding and
    document ends carry no label). The weights read only the mask, never a sampled action, so
    both rollouts of a row share them.

    Raises:
        ValueError: if no row has a label.
    """
    labels = int(mask.sum())
    if labels == 0:
        msg = "the batch has no label"
        raise ValueError(msg)
    return torch.where(mask, mask.shape[0] / labels, 0.0).to(torch.float64)


def training_stream(
    parts: Sequence[Component], seed: int, tokenizer: Tokenizer, split: Split, rows: int, n: int
) -> Stream:
    """The training side of the finalized corpus as a stream of ``rows`` slots of ``n`` positions.

    Consecutive chunks of one slot continue its documents, but phase 2 reads every row from
    cold states, so a row is a fresh context whatever came before it.

    Raises:
        ValueError: for the held-out side (phase 2 never trains on evaluation documents).
    """
    if split.held_out:
        msg = "phase 2 trains on the training side, never the held-out split"
        raise ValueError(msg)
    return Stream(iter_mixed_documents(parts, seed, tokenizer, split), slots=rows, chunk=n)


@dataclass(frozen=True)
class Pinned:
    """Validation chunks bound to the accepted test pin: the chunks and the pin's sha256."""

    chunks: list[Chunk]
    pin_sha256: str


def validation_chunks(
    parts: Sequence[Component],
    tokenizer: Tokenizer,
    held: Split,
    cfg: VesperConfig,
    pin: Path,
    *,
    seed: int = 0,
    micro: int = 8,
) -> Pinned:
    """Phase 2's validation set, BOUND to the accepted test pin (maintainer B e97d5ca).

    The held-out stream's first ``eval_chunks`` chunks; every document they read must already
    be in the pin's ``read_documents``, so no validation read reaches the test part; the pin's
    hash travels with every result.

    Raises:
        ValueError: on a training-side split, or a chunk reading a document outside the pin.
    """
    if not held.held_out:
        msg = "validation reads the held-out side"
        raise ValueError(msg)
    data, pin_sha256 = hashed_bytes(pin)  # the hash reported is the hash of the pin that admits
    record = cast("dict[str, object]", json.loads(data))
    allowed = set(cast("list[str]", record["read_documents"]))
    chunks = held_out_chunks(parts, seed, tokenizer, held, micro, cfg)
    outside = sorted({d for c in chunks for slot in c.doc_ids for d in slot} - allowed)
    if outside:
        msg = (
            f"{len(outside)} validation documents are outside the pin's read set, e.g. {outside[0]}"
        )
        raise ValueError(msg)
    return Pinned(chunks, pin_sha256)


@dataclass(frozen=True)
class TextSource:
    """A phase-1 checkpoint as a starting point: the model and what identifies it."""

    cfg: VesperConfig
    model: Vesper
    model_sha256: str
    config_sha256: str
    objective: str
    step: int
    identity: dict[str, str]  # the run's tokenizer, config, finalization and recipe sha256


def load_text_source(checkpoint: Path, config_path: Path) -> TextSource:
    """The model of a ``vesper.train`` checkpoint, its files checked against the manifest.

    ``train.json`` through ``train.checked_record`` (one snapshot, hashed and parsed from the
    same bytes, its schema refused before any field is read); the weights and the config each
    read once, checked (``keys.verified_bytes``) and loaded from the bytes that were checked
    (maintainer B's review of 72c9d31: a hash-then-reread let a replacement in). ``config_path``
    must be the file the run trained from (its sha256 is the checkpoint's
    ``identity.config_sha256``). The weights load strictly, on the CPU.

    Raises:
        ValueError: on a missing file, a hash that differs from the manifest, another schema,
            or another config.
    """
    record, manifest = checked_record(checkpoint)
    identity = cast("dict[str, str]", record["identity"])
    config_sha256 = identity["config_sha256"]
    config_bytes = verified_bytes(
        config_path, config_sha256, "the config this checkpoint trained from"
    )
    cfg = parse_bytes(config_bytes, str(config_path))
    model_bytes = verified_bytes(
        checkpoint / "model.pt", manifest.get("model.pt"), "model.pt, against the manifest"
    )
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    weights = cast(
        "dict[str, Tensor]",
        torch.load(io.BytesIO(model_bytes), weights_only=True, map_location="cpu"),
    )
    _ = cast("object", model.load_state_dict(weights))
    recipe = cast("dict[str, object]", record["recipe"])
    progress = cast("dict[str, object]", record["progress"])
    return TextSource(
        cfg=cfg,
        model=model,
        model_sha256=cast("str", manifest["model.pt"]),
        config_sha256=config_sha256,
        objective=cast("str", recipe["objective"]),
        step=cast("int", progress["step"]),
        identity=dict(identity),
    )


def _run_cold(
    model: Vesper,
    chunks: Sequence[Chunk],
    graphs: EvalGraphs | None,
    traces: Sequence[Tensor] | None = None,
) -> list[tuple[Tensor, Tensor]]:
    """Per chunk ``(per-token NLL float64 [b, n], passes int64 [b, n])``, each from cold states.

    The deployed policy (``CdfRule`` at ``Q``), or with ``traces`` each chunk's given exit depths
    (``TraceRule``; the depths are values of the captured pass, never constants in it). Captured
    on CUDA (``evals.run_chunks``), read back once.
    """
    head = model.loop.halting_head()

    def cold(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        rule: ExitRule = CdfRule(head, Q) if traces is None else TraceRule(inputs[3])
        result = model.run(inputs[0], inputs[1], states, exit_rule=rule)
        nll = model.position_losses(result.y, inputs[2])
        return [nll, result.n_iters, *state_tensors(result.states)]

    graph = _rule_graph(graphs, trace=traces is not None)
    kind = ("cdf", Q) if traces is None else ("trace",)
    extras = None if traces is None else [[t] for t in traces]
    out = run_chunks(model, chunks, graph, 2, cold, carried=False, switches=kind, extras=extras)
    return [(nll.double().cpu(), passes.long().cpu()) for nll, passes in out]


def _rule_graph(graphs: EvalGraphs | None, *, trace: bool) -> SignatureGraph | None:
    """The graph a cold rule pass runs under: the trace replays' or the deployed rule's."""
    if graphs is None:
        return None
    return graphs.trace if trace else graphs.rule


def depth_nll(
    model: Vesper, chunks: Sequence[Chunk], *, graphs: EvalGraphs | None = None
) -> Tensor:
    """``float64 [T, N]``: every label's NLL at every fixed depth, each chunk from cold states.

    ``evals.depth_losses`` with fresh states per chunk (independent contexts): one dense
    traversal per chunk, labels in chunk order, then row-major within a chunk; no gradient,
    float32 arithmetic; captured on CUDA (``graphs`` required there).
    """
    return depth_losses(model, chunks, carried=False, graphs=graphs)


class PolicyRecord(TypedDict):
    """A policy on the evaluation chunks: pooled NLL over labels, mean depth over real tokens."""

    nll: float
    depth: float
    depth_histogram: list[int]  # real tokens exiting at pass 1..T
    labels: int
    tokens: int
    seed: int | None  # a random control's seed; None for the policy itself


class Paired(TypedDict):
    """Per-label ``fixed - policy`` (positive: the policy is better), pooled summaries."""

    mean: float
    improved: float
    worse: float
    depth: int  # the fixed depth compared against


class TextEvaluation(TypedDict):
    """``evaluate_text``'s record."""

    fixed_nll: list[float]
    policy: PolicyRecord
    frontier_nll_at_policy_depth: float
    random: list[PolicyRecord]
    paired_vs_full_depth: Paired
    paired_vs_best_fixed: Paired


def _policy_record(
    nll: Tensor, passes: Tensor, chunks: Sequence[Chunk], t_max: int, seed: int | None = None
) -> PolicyRecord:
    labels = torch.cat([c.loss_mask.flatten() for c in chunks])
    valid = torch.cat([c.valid.flatten() for c in chunks])
    depths = passes.flatten()[valid]
    histogram = torch.bincount(depths - 1, minlength=t_max)
    return {
        "nll": float(nll.flatten()[labels].mean()),
        "depth": float(depths.double().mean()),
        "depth_histogram": [int(histogram[t]) for t in range(t_max)],
        "labels": int(labels.sum()),
        "tokens": int(valid.sum()),
        "seed": seed,
    }


def _paired(policy: Tensor, fixed: Tensor, depth: int) -> Paired:
    d = fixed - policy
    return {
        "mean": float(d.mean()),
        "improved": float((d > 0).double().mean()),
        "worse": float((d < 0).double().mean()),
        "depth": depth,
    }


def evaluate_text(
    model: Vesper,
    chunks: Sequence[Chunk],
    random_seeds: Sequence[int] = RANDOM_SEEDS,
    *,
    graphs: EvalGraphs | None = None,
) -> TextEvaluation:
    """The deterministic policy on independent chunks against the model's own fixed depths.

    ``fixed_nll``: the pooled NLL at each fixed depth (``depth_nll``); ``policy``: ``CdfRule``
    at ``Q`` (pooled NLL over labels, mean depth over real tokens, the depth histogram);
    ``frontier_nll_at_policy_depth``: the best fixed-depth policy or two-depth mixture at the
    policy's mean depth (``static_frontier`` on negated NLL); ``random``: per seed, depths drawn
    from the policy's own histogram and replayed (``TraceRule``); ``paired_vs_full_depth`` and
    ``paired_vs_best_fixed``: per-label differences, positive where the policy is better.

    On the model's own device and scan, in eval mode (defensive: no Vesper module reads the
    mode today; ``recurrence.state_read_noise`` is declared but not applied), the caller's mode
    restored after: as ``vesper.train`` evaluates (decision 2026-10-16), not the synthetic
    drivers' CPU reference copy (``halting_common.evaluating``), which a text model's
    evaluation set would make hours long. Every pass is a captured CUDA graph on CUDA
    (``graphs`` required there; the project owner 2026-10-21), the random controls' depths its
    values.
    """
    was_training = model.training
    _ = model.eval()
    try:
        return _evaluate(model, chunks, random_seeds, graphs)
    finally:
        _ = model.train(was_training)


def _evaluate(
    m: Vesper, chunks: Sequence[Chunk], random_seeds: Sequence[int], graphs: EvalGraphs | None
) -> TextEvaluation:
    t_max = m.loop.t_max
    fixed = depth_nll(m, chunks, graphs=graphs)
    runs = _run_cold(m, chunks, graphs)
    nll = torch.cat([r[0].flatten() for r in runs])
    passes = torch.cat([r[1].flatten() for r in runs])
    policy = _policy_record(nll, passes, chunks, t_max)
    labels = torch.cat([c.loss_mask.flatten() for c in chunks])
    policy_labels = nll[labels]
    fixed_means = [float(fixed[d].mean()) for d in range(t_max)]
    histogram = torch.tensor(policy["depth_histogram"], dtype=torch.float64)
    controls: list[PolicyRecord] = []
    for seed in random_seeds:
        g = torch.Generator().manual_seed(seed)
        traces = [
            torch.multinomial(histogram, c.tokens.numel(), replacement=True, generator=g)
            .view(c.tokens.shape)
            .add(1)
            for c in chunks
        ]
        replay = _run_cold(m, chunks, graphs, traces)
        r_nll = torch.cat([r[0].flatten() for r in replay])
        r_passes = torch.cat([r[1].flatten() for r in replay])
        controls.append(_policy_record(r_nll, r_passes, chunks, t_max, seed))
    points = [(float(d + 1), -v) for d, v in enumerate(fixed_means)]
    best = fixed_means.index(min(fixed_means))  # the first best fixed depth, 0-based
    return {
        "fixed_nll": fixed_means,
        "policy": policy,
        "frontier_nll_at_policy_depth": -static_frontier(points, policy["depth"]),
        "random": controls,
        "paired_vs_full_depth": _paired(policy_labels, fixed[-1], t_max),
        "paired_vs_best_fixed": _paired(policy_labels, fixed[best], best + 1),
    }


ACTION_SPLIT = "text-joint-action"  # the uniforms' generator (synthetic.generator_for)
TRAIN_SCAN = {"cpu": "reference", "cuda": "fused-fp32"}


def train_text_update(  # noqa: PLR0913
    model: Vesper,
    opt: torch.optim.AdamW,
    params: Sequence[nn.Parameter],
    chunk: Chunk,
    uniforms: Tensor,
    update: int,
    plan: "Plan",
    *,
    graph: WarmedGraph | None = None,
) -> dict[str, object]:
    """One joint update from ``chunk``'s rows as independent contexts; its record.

    The rows are the ``E`` unique sequences, ``ROLLOUTS`` times over, rollout-major; the label
    weights are ``pooled_weights`` (both rollouts share them); ``uniforms`` float64
    ``[T - 1, ROLLOUTS * E, n]``; no floor (epsilon 0: joint mode's long runs ran without one).
    The update is ``joint_update.update_step`` with the plan's memory options, captured on CUDA
    (``graph``), its record read after it.
    """
    device = next(model.parameters()).device
    rates = set_learning_rates(opt, update)
    unique = pooled_weights(chunk.loss_mask)
    c = chunk
    values = [
        torch.cat([t] * ROLLOUTS).to(device)
        for t in (c.tokens, c.pos, c.targets, c.loss_mask, c.valid, unique)
    ]
    values += [uniforms.to(device), torch.zeros((), dtype=torch.float64, device=device)]

    def step(inputs: Sequence[Tensor]) -> Step:
        return update_step(
            model,
            opt,
            params,
            plan.depth_cost,
            plan.dense_weight,
            inputs,
            recompute=plan.recompute,
            slices=plan.slices,
            pooled_depth=True,  # depth priced over pooled real tokens, as labels are
        )

    out = step(values) if graph is None else Step(*graph(step, values))
    t_max = model.loop.t_max
    pair_nll = out.objective.view(ROLLOUTS, -1)
    pair_depth = plan.depth_cost * out.depth.view(ROLLOUTS, -1) / t_max
    deciding = values[4] & (out.decisions > 0)
    labels = values[3]
    return {
        "update": update,
        "lr_groups": rates,
        "cost": float(out.cost.mean()),
        # the pooled NLL of the sampled trajectories: per-trajectory objectives are E/N-weighted
        # sums, so their mean over each rollout's E rows is that rollout's pooled NLL
        "pooled_nll": float(out.objective.mean()),
        "nll_row_mean": float(out.nll.mean()),
        "depth": float(out.depth.mean()),
        "labels": int(labels.sum()) // ROLLOUTS,
        "surrogate": float(out.score),
        "pathwise": float(out.pathwise),
        "dense": float(out.dense),
        "grad_norm": float(out.grad_norm),
        "grad_norm_groups": [float(norm) for norm in out.group_norms.unbind()],
        "stopped_early": float((out.n_iters[deciding] < t_max).double().mean()),
        # population std over the E pairs (defined for any E, 0 at E = 1): the spread of the
        # two cost terms between paired rollouts, the preflight's scale check
        "pair_nll_diff_std": float((pair_nll[0] - pair_nll[1]).std(correction=0)),
        "pair_depth_cost_diff_std": float((pair_depth[0] - pair_depth[1]).std(correction=0)),
    }


BACKBONE_MODES = ("joint", "frozen")


@dataclass(frozen=True, slots=True)
class Plan:
    """What one phase-2 run follows: the price, the auxiliary, the budget, the memory options.

    ``backbone`` is ``"joint"`` (the backbone trains with the head) or ``"frozen"`` (head only,
    the backbone stays the source's: maintainer B 32e7587, the diagnostic that separates the head's
    allocation from the backbone's drift; its dense weight is 0, the dense term trains only
    the backbone).
    """

    seed: int
    depth_cost: float
    dense_weight: float
    updates: int
    rows: int
    save_at: tuple[int, ...]
    slices: int
    recompute: bool
    device: str
    preflight: bool
    backbone: str

    def record(self) -> dict[str, object]:
        """Every field, for the result."""
        return {
            "seed": self.seed,
            "depth_cost": self.depth_cost,
            "dense_weight": self.dense_weight,
            "updates": self.updates,
            "rows": self.rows,
            "save_at": list(self.save_at),
            "slices": self.slices,
            "recompute": self.recompute,
            "device": self.device,
            "preflight": self.preflight,
            "backbone": self.backbone,
        }


def check_plan(plan: Plan) -> torch.device:
    """The training device after every refusal, before anything is read or written.

    Raises:
        ValueError: on a depth cost or dense weight that is not finite and at least 0, updates
            or rows below 1, a save point outside ``0..updates``, slices below 1, an unknown
            device or a scan it cannot run, an unknown backbone mode, or a frozen backbone
            with a dense weight above 0 (a term that would train nothing).
    """
    if plan.backbone not in BACKBONE_MODES:
        msg = f"backbone must be one of {BACKBONE_MODES}, got {plan.backbone!r}"
        raise ValueError(msg)
    if not (math.isfinite(plan.depth_cost) and plan.depth_cost >= 0.0):
        msg = f"the depth cost must be finite and at least 0, not {plan.depth_cost}"
        raise ValueError(msg)
    if not (math.isfinite(plan.dense_weight) and plan.dense_weight >= 0.0):
        msg = f"the dense weight must be finite and at least 0, not {plan.dense_weight}"
        raise ValueError(msg)
    if plan.backbone == "frozen" and plan.dense_weight > 0.0:
        msg = f"a frozen backbone takes no dense term: dense weight 0, not {plan.dense_weight}"
        raise ValueError(msg)
    if plan.updates < 1 or plan.rows < 1 or plan.slices < 1:
        msg = (
            f"updates, rows and slices must be at least 1"
            f" ({plan.updates}, {plan.rows}, {plan.slices})"
        )
        raise ValueError(msg)
    if any(not 0 <= s <= plan.updates for s in plan.save_at):
        msg = f"every save point must lie within 0..{plan.updates}"
        raise ValueError(msg)
    if plan.device not in TRAIN_SCAN:
        msg = f"device must be one of {tuple(TRAIN_SCAN)}, got {plan.device!r}"
        raise ValueError(msg)
    target = torch.device(plan.device)
    _ = check_capturable_scan(TRAIN_SCAN[plan.device], target)
    return target


def run(  # noqa: PLR0913, PLR0915
    source: Path,
    config: Path,
    pin: Path,
    out: Path,
    plan: Plan,
    *,
    parts: Sequence[Component],
    finalization: Path,
    tokenizer_path: Path,
) -> dict[str, object]:
    """One phase-2 run: the phase-1 checkpoint ``source`` trained jointly on text; its result.

    Reads the source (``load_text_source``) and checks that ``finalization`` and
    ``tokenizer_path`` are the files its run trained with (identity hashes) and that ``parts``
    are the files the finalization was taken over (``corpus.check_inputs``; ``main`` resolves
    them from the source's own pins, ``train.resumed_components``); the validation chunks are
    bound to the test ``pin`` (``validation_chunks``). The head starts at ``start_head``; a
    ``"joint"`` plan releases the backbone (``joint_update``'s optimizer, rates and clipping),
    a ``"frozen"`` one keeps it the source's (no backbone group, nothing else trains); each
    update reads the next chunk of the training stream (``plan.rows`` independent rows) and runs
    ``train_text_update``; the run stops early, recorded in ``stopped``, when the stream ends
    or a chunk has a row without a label (a trajectory needs one: its NLL and depth are means
    over its own labels and real tokens). At every save point the whole weights are saved and
    ``evaluate_text`` runs on the validation chunks; ``plan.preflight`` runs the updates with
    neither. On CUDA the update is captured (``WarmedGraph``) inside
    ``deterministic_algorithms``.

    Raises:
        ValueError: as ``check_plan``, on another finalization or tokenizer than the source's,
            components the finalization was not taken over, or a validation document outside
            the pin; all before ``out`` exists.
        FileExistsError: if ``out`` exists.
        FloatingPointError: on a non-finite update (the updates before it logged).
    """
    target = check_plan(plan)
    src = load_text_source(source, config)
    # each identity file parsed from the bytes whose hash was checked, never read again
    fin = load_finalization(finalization, sha256=src.identity["finalization_sha256"])
    tokenizer = Tokenizer.parse(
        verified_bytes(
            tokenizer_path,
            src.identity["tokenizer_sha256"],
            "the tokenizer the source trained with",
        )
    )
    cfg, model = src.cfg, src.model
    check_inputs(fin, parts)
    fraction = cfg.training.held_out_fraction
    pinned = validation_chunks(parts, tokenizer, Split(fin, fraction, held_out=True), cfg, pin)
    stream = training_stream(
        parts, plan.seed, tokenizer, Split(fin, fraction), plan.rows, cfg.training.chunk_tokens
    )
    out.mkdir(parents=True, exist_ok=False)
    head = start_head(model)
    backbone = release_backbone(model) if plan.backbone == "joint" else []
    cuda = target.type == "cuda"
    precision = training_precision(cuda=cuda)
    model = model.to(target)
    _ = set_scan(model, TRAIN_SCAN[plan.device])
    opt = optimizer(head, backbone, capturable=cuda)
    graph = WarmedGraph() if cuda else None
    params = head + backbone
    action = generator_for(ACTION_SPLIT, plan.seed)
    t_max = model.loop.t_max
    consumed, draws, training = "", "", 0.0
    evaluations: dict[str, TextEvaluation] = {}
    stopped: str | None = None
    reset_memory_peak(target)
    started = time.monotonic()
    with (
        deterministic_algorithms() if cuda else nullcontext(None) as determinism,
        (out / "train.jsonl").open("w", encoding="utf-8") as log,
        (out / "eval.jsonl").open("w", encoding="utf-8") as evals,
    ):

        def checkpoint(after: int) -> None:
            if plan.preflight or after not in plan.save_at:
                return
            (out / "weights").mkdir(exist_ok=True)
            torch.save(
                {
                    name: t.detach().to("cpu").clone()
                    for name, t in cast("dict[str, Tensor]", model.state_dict()).items()
                },
                out / "weights" / f"after-{after:06d}.pt",
            )
            # captured on CUDA; built per save point so their memory is not held while training
            eval_graphs = EvalGraphs() if cuda else None
            evaluation = evaluate_text(model, pinned.chunks, graphs=eval_graphs)
            evaluations[str(after)] = evaluation
            line: dict[str, object] = {"after": after, **evaluation}
            _ = evals.write(json.dumps(line) + "\n")
            evals.flush()

        checkpoint(0)
        for update in range(plan.updates):
            chunk = stream.next()
            if chunk is None or not bool(chunk.loss_mask.any(1).all()):
                stopped = f"the training stream ran dry before update {update}"
                break
            n = int(chunk.tokens.shape[1])
            uniforms = torch.rand(
                (t_max - 1, ROLLOUTS, plan.rows, n), generator=action, dtype=torch.float64
            ).reshape(t_max - 1, ROLLOUTS * plan.rows, n)
            consumed = chained_digest(consumed, chunk_tensors(chunk))
            draws = chained_digest(draws, [uniforms])
            tick = time.monotonic()
            record = train_text_update(
                model, opt, params, chunk, uniforms, update, plan, graph=graph
            )
            if cuda:
                torch.cuda.synchronize(target)
            training += time.monotonic() - tick
            _ = log.write(
                json.dumps(record | {"consumed_digest": consumed, "draw_digest": draws}) + "\n"
            )
            log.flush()
            checkpoint(update + 1)
    done = len(evaluations) and max(int(k) for k in evaluations)
    result: dict[str, object] = {
        "design": "docs/reviews/2026-10-20-real-data-halting-pilot.md (phase 2, exploratory)",
        "source": str(source),
        "source_model_sha256": src.model_sha256,
        "source_config_sha256": src.config_sha256,
        "source_objective": src.objective,
        "source_step": src.step,
        "pin": str(pin),
        "pin_sha256": pinned.pin_sha256,
        "plan": plan.record(),
        "rollouts": ROLLOUTS,
        "head_start": {"direction": 0.0, "gain": 1.0, "bias": HEAD_BIAS},
        "optimizer": {
            "peak_lr": PEAK_LR,
            "backbone_peak_lr": BACKBONE_LR if backbone else None,
            "warmup": WARMUP,
            "betas": list(BETAS),
            "eps": ADAM_EPS,
            "weight_decay": 0.0,
            "grad_clip": GRAD_CLIP,
        },
        "q": Q,
        "consumed_digest": consumed,
        "draw_digest": draws,
        "stopped": stopped,
        "last_evaluation": done or None,
        "train_seconds": training,
        "wall_seconds": time.monotonic() - started,
        "peak_reserved_bytes": memory_peak(target) if cuda else None,
        # the training update's own graph (evaluation graphs are separate): eager warm-up
        # calls, then one capture per signature and replays (maintainer B's review of 8415310)
        "update_graph": None
        if graph is None
        else {"warm": graph.warm, "captures": graph.captures, "replays": graph.replays},
        "determinism": determinism,
        "device_name": torch.cuda.get_device_name(target) if cuda else None,
        "scan": scan_of(model),
        "precision": precision,
        "versions": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "fla-core": version_or_none("fla-core"),
        },
        "environment": environment(),
        "evaluations": evaluations,
    }
    write_json(out / "result.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    _ = parser.add_argument("--source", type=Path, required=True, help="a vesper.train checkpoint")
    _ = parser.add_argument("--config", type=Path, required=True, help="the source run's config")
    _ = parser.add_argument("--pin", type=Path, required=True, help="the accepted test pin")
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--data", type=Path, required=True)
    _ = parser.add_argument("--finalization", type=Path, required=True)
    _ = parser.add_argument("--tokenizer", type=Path, required=True)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument("--depth-cost", type=float, required=True)
    _ = parser.add_argument("--backbone", choices=BACKBONE_MODES, default="joint")
    _ = parser.add_argument(
        "--dense-weight", type=float, default=None, help="default 0.1 joint, 0 frozen"
    )
    _ = parser.add_argument("--updates", type=int, required=True)
    _ = parser.add_argument("--rows", type=int, default=4, help="unique rows per update (E)")
    _ = parser.add_argument("--save-at", type=int, nargs="*", default=None)
    _ = parser.add_argument("--slices", type=int, default=4)
    _ = parser.add_argument("--no-recompute", action="store_true")
    _ = parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _ = parser.add_argument("--preflight", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: 0 after a completed run, 1 on a refused option."""
    args = _parser().parse_args(argv)
    updates = cast("int", args.updates)
    save_at = cast("list[int] | None", args.save_at)
    backbone = cast("str", args.backbone)
    dense = cast("float | None", args.dense_weight)
    plan = Plan(
        seed=cast("int", args.seed),
        depth_cost=cast("float", args.depth_cost),
        dense_weight=dense if dense is not None else (0.1 if backbone == "joint" else 0.0),
        updates=updates,
        rows=cast("int", args.rows),
        save_at=tuple(save_at if save_at is not None else (0, updates // 2, updates)),
        slices=cast("int", args.slices),
        recompute=not cast("bool", args.no_recompute),
        device=cast("str", args.device),
        preflight=cast("bool", args.preflight),
        backbone=backbone,
    )
    source = cast("Path", args.source)
    try:
        # the plan is refused before the source is read: resolving the source's components
        # below reads its record (maintainer B's review of 72c9d31); run checks the plan again
        _ = check_plan(plan)
        _ = run(
            source,
            cast("Path", args.config),
            cast("Path", args.pin),
            cast("Path", args.out),
            plan,
            parts=resumed_components(cast("Path", args.data), source),  # the source's pins
            finalization=cast("Path", args.finalization),
            tokenizer_path=cast("Path", args.tokenizer),
        )
    except ValueError:
        log.exception("refused")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
