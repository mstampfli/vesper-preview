"""Instruction tuning: one bounded test of whether supervised tuning makes the model follow tasks.

    python -m vesper.instruct tune --source CKPT --config CFG --tokenizer TOK --dolly FILE \
        --instruct configs/instruct-dolly.yaml --panel configs/instruct-panel.json \
        --data ROOT --finalization FIN --pin PIN --out DIR [--device cuda] [--recompute] \
        [--loss-slices N] [--limit-examples N]
    python -m vesper.instruct final --source CKPT --config CFG --tokenizer TOK --dolly FILE \
        --instruct configs/instruct-dolly.yaml --panel configs/instruct-panel.json --out DIR \
        [--device cuda]
    python -m vesper.instruct view --source CKPT --config CFG --tokenizer TOK \
        --instruct configs/instruct-dolly.yaml --data ROOT --finalization FIN --pin PIN

The contract is docs/reviews/2026-10-25-instruction-tuning.md (maintainer B's reviews of 6080029 and
e178dab); every setting is the typed ``config.InstructConfig`` (``--instruct``), beside the
source run's own configuration. ``tune`` starts from a ``vesper.train`` checkpoint and trains
every parameter except the halting head (``tuning_optimizer``; the head is frozen and checked
unchanged) on the train split of ``vesper.instruct_data`` with the existing FIXED objective at
the model's ``T``: every pass of a dense forward scored with equal weight
(``Vesper.all_pass_objective``, condition ``fixed``), zero ponder, no halting gradient, over the
response body and terminator targets only, pooled over the batch's real labels. Every batch is a
set of independent examples from cold states, created inside the step (``train_step``), so
nothing is carried between examples, batches or replays. One setting (``Recipe``): a fresh
AdamW, warm-up then cosine (``learning.warmup_cosine``), a fixed number of epochs; the weights
after every epoch are kept. Each epoch, the unchanged checkpoint included as epoch 0, is scored on
the validation split at a fixed depth (``evaluate_examples``); the epoch with the lowest pooled
validation body NLL among 1.. is selected, ties to the earlier.

Replay (optional, the config's ``replay`` section; maintainer B's design, 758b4fd): every update
also scores rows of the source run's own TRAINING text (``instruct_replay.ReplaySource``: the
checkpoint's pinned components and finalization, the training side, the replay's own seed,
every row checked against the accepted pin's held-out documents and groups and recorded in
``train.jsonl``) and adds ``weight`` times their mean loss to the instruction loss; the
gradients of both accumulate at the same weights, are clipped once and make ONE optimizer step
(``_device_step``). Weight 0, or no section, is the update without replay and reads no row.

Forgetting: the days run's own held-out view (``days_view``: the checkpoint's PINNED components,
its finalization checked against its identity, the first chunks of the held-out stream, every
document inside the accepted test pin, the view's digest equal to the config's) is checked
before any training and scored for the source and the selected weights with the same
fixed-depth evaluator (``evals.static_profile``). ``view`` prints that digest without a model.

``final`` reads the reserved test split once per output directory. Before it claims the
directory or reads a test example it refuses (2) a preflight run, any input that differs from
the tuning run's (``Binding``: source weights and identity, the config's and the panel's bytes,
the scorer's version), a changed or missing ``data.json``, a membership its own preparation
does not reproduce, and selected weights that do not hash as recorded; a second attempt gets 3.
After the claim a failure is recorded in the claim's ``status.json``. It writes the paired test
comparison (``paired``) and the panel, generated greedily at the fixed depth from cold states
on the CPU (``generate``, ``instruct_panel``).

A positive fit result says the model fits these answers better; it is instruction following only
if the panel moves too. Validation intervals after the epoch choice are descriptive.
"""

import argparse
import hashlib
import io
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from vesper.column import ColumnState, set_scan, state_tensors
from vesper.config import InstructConfig, VesperConfig, parse_instruct_bytes
from vesper.corpus import Component, Finalization, Split, load_finalization
from vesper.data import Chunk
from vesper.decode import CapturedDecoder
from vesper.evals import EvalGraphs, run_chunks, static_profile
from vesper.graphs import (
    GraphFamily,
    MemoryPhases,
    deterministic_algorithms,
    signature,
)
from vesper.halting_text import TextSource, load_text_source, validation_chunks
from vesper.instruct_data import (
    CANONICAL,
    TERMINATOR,
    Batch,
    Example,
    Prepared,
    SplitRule,
    batches,
    int64_bytes,
    load_records,
    prepare,
    prompt_text,
)
from vesper.instruct_panel import SCORING, Panel, answer_of, load_panel, score_answer
from vesper.instruct_replay import ReplayRows, ReplaySource
from vesper.keys import SequenceDigest, hashed_bytes, verified_bytes
from vesper.learning import warmup_cosine
from vesper.loop import DenseRule
from vesper.model import Vesper
from vesper.provenance import environment
from vesper.tokenizer import Tokenizer
from vesper.train import checked_record, components_of_record, decays, set_learning_rate

ATTRIBUTION = (
    "databricks-dolly-15k, CC BY-SA 3.0: Databricks, Inc. and the Wikipedia contributors whose"
    " text the contexts quote; derived examples are shared under the same license"
)
HALTING_PREFIX = "loop.halting."
log = logging.getLogger("vesper.instruct")


@dataclass(frozen=True)
class Recipe:
    """The one tuning setting, every value explicit in the result."""

    peak_lr: float
    warmup_steps: int
    final_lr_fraction: float
    epochs: int
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    batch_tokens: int
    buckets: tuple[int, ...]
    max_inputs: int
    train_passes: int
    eval_depth: int
    seed: int
    resamples: int
    max_new_tokens: int
    recompute: bool
    slices: int
    replay_weight: float = 0.0  # the replay loss's coefficient; 0: no replay (config.ReplayConfig)


def recipe_for(
    cfg: VesperConfig, icfg: InstructConfig, *, recompute: bool = False, slices: int = 1
) -> Recipe:
    """The setting: the instruction config's values, the source run's betas, decay and clip.

    Raises:
        ValueError: when the evaluation depth is outside the model's passes, or ``slices`` < 1.
    """
    if not 1 <= icfg.eval_depth <= cfg.model.t_max:
        msg = f"eval depth {icfg.eval_depth} outside the model's passes 1..{cfg.model.t_max}"
        raise ValueError(msg)
    if slices < 1:
        msg = f"loss slices must be at least 1, got {slices}"
        raise ValueError(msg)
    o = cfg.optimizer
    return Recipe(
        peak_lr=icfg.peak_lr,
        warmup_steps=icfg.warmup_steps,
        final_lr_fraction=icfg.final_lr_fraction,
        epochs=icfg.epochs,
        weight_decay=o.weight_decay,
        beta1=o.beta1,
        beta2=o.beta2,
        grad_clip=o.grad_clip,
        batch_tokens=icfg.batch_tokens,
        buckets=icfg.buckets,
        max_inputs=icfg.max_inputs,
        train_passes=cfg.model.t_max,
        eval_depth=icfg.eval_depth,
        seed=icfg.seed,
        resamples=icfg.resamples,
        max_new_tokens=icfg.max_new_tokens,
        recompute=recompute,
        slices=slices,
        replay_weight=0.0 if icfg.replay is None else icfg.replay.weight,
    )


def tuning_optimizer(model: Vesper, recipe: Recipe, *, capturable: bool) -> torch.optim.AdamW:
    """AdamW over every parameter but the halting head's, decayed by role (``train.decays``).

    The head's parameters stop requiring a gradient here and are in no group, so no step can
    move them. ``capturable``: the CUDA-graph variant, its rate a device tensor filled in place
    (``train.set_learning_rate``).
    """
    named = list(model.named_parameters())
    for name, parameter in named:
        if name.startswith(HALTING_PREFIX):
            _ = parameter.requires_grad_(False)  # noqa: FBT003  # torch's positional flag
    trained = [(n, p) for n, p in named if not n.startswith(HALTING_PREFIX)]
    groups = [
        {"params": [p for n, p in trained if decays(n)], "weight_decay": recipe.weight_decay},
        {"params": [p for n, p in trained if not decays(n)], "weight_decay": 0.0},
    ]
    betas = (recipe.beta1, recipe.beta2)
    if not capturable:
        return torch.optim.AdamW(groups, lr=recipe.peak_lr, betas=betas)
    device = next(model.parameters()).device
    lr = torch.tensor(recipe.peak_lr, device=device)
    return torch.optim.AdamW(groups, lr=lr, betas=betas, capturable=True)


@dataclass(frozen=True)
class StepRecord:
    """One update: its losses, the gradient's norm before the clip and the labels.

    ``loss`` and ``pass_loss`` are the instruction batch's; with replay, ``replay_loss`` and
    ``replay_labels`` are the replay rows', and ``objective`` is loss + weight * replay_loss
    (the instruction loss itself without replay).
    """

    loss: float
    grad_norm: float
    pass_loss: list[float]
    labels: int
    seconds: float
    objective: float
    replay_loss: float | None
    replay_labels: int


def _batch_values(batch: Batch | ReplayRows, device: torch.device) -> list[Tensor]:
    return [t.to(device) for t in (batch.tokens, batch.targets, batch.valid, batch.pos)] + [
        batch.loss_mask.to(device)
    ]


def _chunk_of(tokens: Tensor, targets: Tensor, valid: Tensor, pos: Tensor, mask: Tensor) -> Chunk:
    """The objective's view of a batch: only tokens, targets, valid and the mask are read."""
    zeros = torch.zeros_like(valid)
    return Chunk(tokens, targets, valid, zeros, torch.zeros_like(pos), mask, pos, (), ())


def _objective(
    model: Vesper, recipe: Recipe, values: Sequence[Tensor]
) -> tuple[Tensor, dict[str, Tensor]]:
    """The fixed all-pass objective of one set of rows, each row from cold states made here."""
    tokens, targets, valid, pos, mask = values
    device = tokens.device
    states = model.init_states(int(tokens.shape[0]), torch.float32, device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        result = model.run(
            tokens,
            pos,
            states,
            passes=recipe.train_passes,
            recompute=recipe.recompute,
            exit_rule=DenseRule(),
        )
        return model.all_pass_objective(
            _chunk_of(tokens, targets, valid, pos, mask),
            result,
            condition="fixed",
            beta=0.0,
            slices=recipe.slices,
            keep_logits=not recipe.recompute,
        )


def _device_step(
    model: Vesper, opt: torch.optim.AdamW, recipe: Recipe, values: Sequence[Tensor]
) -> list[Tensor]:
    """The whole update with no read-back: capturable.

    ``values`` are the instruction batch's five tensors, followed by the replay rows' five when
    ``recipe.replay_weight`` > 0. The gradients are cleared once; the instruction loss's backward
    and then the weighted replay loss's backward accumulate at the same weights (one after the
    other, so the two activation sets are never held together); the combined gradient is
    clipped once and ONE optimizer step is taken (maintainer B 758b4fd). Without replay this is the
    original update, operation for operation.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    value, parts = _objective(model, recipe, values[:5])
    torch.autograd.backward([value])
    out = [value.detach().float()]
    if recipe.replay_weight > 0:
        replayed, _ = _objective(model, recipe, values[5:])
        torch.autograd.backward([replayed * recipe.replay_weight])
        out.append(replayed.detach().float())
    grad_norm = nn.utils.clip_grad_norm_(params, recipe.grad_clip).float()
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    return [*out, grad_norm, parts["pass_ntp"].float()]


def train_step(
    model: Vesper,
    opt: torch.optim.AdamW,
    batch: Batch,
    lr: float,
    recipe: Recipe,
    graphs: GraphFamily | None,
    replay: ReplayRows | None = None,
) -> StepRecord:
    """One update on ``batch``, every row a cold, independent example, plus ``replay``'s rows.

    The loss is pooled over the batch's real labels; with ``recipe.replay_weight`` > 0 the
    replay rows' loss, pooled over their own scored targets, is added with that weight in the
    same update (``_device_step``). On CUDA a captured graph per batch length (``graphs``'
    member ``("train", length)``, required: each length warms eagerly ``WARM_STEPS`` times,
    then captures and replays; the family's members share one memory pool; the replay rows'
    fixed shape and the weight are part of its signature); on the CPU the same device step
    runs directly.

    Raises:
        ValueError: on CUDA without ``graphs``; replay rows without a weight, or a weight
            without rows.
    """
    device = next(model.parameters()).device
    if device.type == "cuda" and graphs is None:
        msg = "a CUDA update runs as a captured CUDA graph: pass graphs (warm, capture, replay)"
        raise ValueError(msg)
    replaying = recipe.replay_weight > 0
    if replaying != (replay is not None):
        msg = (
            "a replay weight needs replay rows"
            if replaying
            else "replay rows with weight 0: weight 0 reads no replay rows"
        )
        raise ValueError(msg)
    started = time.monotonic()
    set_learning_rate(opt, lr)
    values = _batch_values(batch, device)
    if replay is not None:
        values += _batch_values(replay, device)
    if graphs is None:
        out: Sequence[Tensor] = _device_step(model, opt, recipe, values)
    else:
        length = int(batch.tokens.shape[1])
        graph = graphs(("train", length))
        key = signature(
            values,
            "instruct",
            recipe.train_passes,
            recipe.recompute,
            recipe.slices,
            recipe.replay_weight,
        )
        out = graph(key, lambda v: _device_step(model, opt, recipe, v), values)
    scalars = 3 if replaying else 2
    record = torch.cat([*(t.reshape(1) for t in out[:scalars]), out[scalars]]).double().cpu()
    loss = float(record[0])
    replay_loss = float(record[1]) if replaying else None
    return StepRecord(
        loss=loss,
        grad_norm=float(record[scalars - 1]),
        pass_loss=cast("list[float]", record[scalars:].tolist()),  # pyright: ignore[reportUnknownMemberType]
        labels=int(batch.loss_mask.sum()),
        seconds=time.monotonic() - started,
        objective=loss if replay_loss is None else loss + recipe.replay_weight * replay_loss,
        replay_loss=replay_loss,
        replay_labels=0 if replay is None else int(replay.loss_mask.sum()),
    )


@dataclass(frozen=True)
class ExampleLosses:
    """Per example, in record-index order: its group, body and terminator loss sums and counts.

    At one fixed depth; float64 sums, so a pooled NLL is a ratio of sums.
    """

    index: Tensor
    group: Tensor
    body_sum: Tensor
    body_count: Tensor
    term_sum: Tensor
    term_count: Tensor

    def pooled(self, part: str = "body") -> float | None:
        """Token-pooled NLL of ``part`` (``body`` or ``term``); None when it has no target."""
        total = self.body_sum if part == "body" else self.term_sum
        count = self.body_count if part == "body" else self.term_count
        targets = int(count.sum())
        return float(total.sum()) / targets if targets else None


def evaluate_examples(
    model: Vesper,
    evaluation: Sequence[Batch],
    *,
    depth: int,
    graphs: GraphFamily | None,
    slices: int = 1,
) -> ExampleLosses:
    """Every example's body and terminator losses with every token stopping at ``depth``.

    One no-gradient dense forward of ``depth`` passes per batch from cold states
    (``evals.run_chunks``), the head on the last pass's output, scored ``slices`` row groups at a
    time (at most ``slices`` groups, one row each at least): the head's float32 ``[rows, V]``
    buffer then holds the largest group, ``ceil(b / min(slices, b))`` of the batch's ``b`` rows
    (five rows in three slices: 2/5; one row: all of it) (8192 positions x 32768 ids is 1 GiB, the
    allocation the 2026-10-25 preflights failed beside). The losses are the same per row
    whatever the grouping, to rounding. On CUDA ``graphs`` holds one
    captured graph per batch length (member ``("eval", length)``), kept by the caller across
    evaluations of the same model: each length warms once per run and then replays (one graph
    shared by every length was dropped at each switch and re-warmed every epoch, so the f5fe880
    preflight ran eager). ``run_chunks`` copies each batch's results out at once, as a shared
    pool requires.

    Raises:
        ValueError: on ``slices < 1``.
    """
    if slices < 1:
        msg = f"loss slices must be at least 1, got {slices}"
        raise ValueError(msg)

    def fixed(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        tokens, pos, targets, body, term = inputs
        result = model.run(tokens, pos, states, passes=depth, exit_rule=DenseRule())
        groups = min(slices, int(tokens.shape[0]))
        losses = torch.cat(
            [
                model.position_losses(y, t)
                for y, t in zip(
                    result.y.tensor_split(groups), targets.tensor_split(groups), strict=True
                )
            ]
        ).double()
        zero = torch.zeros_like(losses)
        return [
            torch.where(body, losses, zero).sum(1),
            body.sum(1),
            torch.where(term, losses, zero).sum(1),
            term.sum(1),
            *state_tensors(result.states),
        ]

    def length(batch: Batch) -> int:
        return int(batch.tokens.shape[1])

    ordered = sorted(evaluation, key=length)  # each length's batches together
    out: list[list[Tensor]] = []
    for size in sorted({length(b) for b in ordered}):
        same = [b for b in ordered if length(b) == size]
        out += run_chunks(
            model,
            [b.chunk() for b in same],
            None if graphs is None else graphs(("eval", size)),
            4,
            fixed,
            carried=False,
            switches=("instruct-fixed", depth, slices),
            extras=[[b.body_mask, b.term_mask] for b in same],
        )
    rows = [b.indices >= 0 for b in ordered]
    pick = [[t.cpu()[keep] for t in o] for o, keep in zip(out, rows, strict=True)]
    index = torch.cat([b.indices[keep] for b, keep in zip(ordered, rows, strict=True)])
    group = torch.cat([b.groups[keep] for b, keep in zip(ordered, rows, strict=True)])
    order = torch.argsort(index)
    columns = [torch.cat([p[k] for p in pick])[order] for k in range(4)]
    return ExampleLosses(
        index=index[order],
        group=group[order],
        body_sum=columns[0].double(),
        body_count=columns[1].long(),
        term_sum=columns[2].double(),
        term_count=columns[3].long(),
    )


def paired(
    base: ExampleLosses, tuned: ExampleLosses, *, resamples: int, seed: int
) -> dict[str, object]:
    """Tuned minus base pooled NLL, body and terminator apart, on the same examples.

    For each part, the groups with at least one target of that part CONTRIBUTE; the 95 %
    percentile interval resamples the contributing groups whole with their loss sums and counts,
    so each resample recomputes a token-pooled NLL (a mean of example means is another
    estimand) and no resample has zero targets. A group without targets of a part (an empty
    response's group, for the body) adds nothing to that part and is listed by count. A part
    with no target at all is reported ``unassessed``, with no estimate and no interval: never a
    NaN (maintainer B's review of e178dab).

    Raises:
        ValueError: when the two do not cover the same examples, groups and target counts.
    """
    same = (
        torch.equal(base.index, tuned.index)
        and torch.equal(base.group, tuned.group)
        and torch.equal(base.body_count, tuned.body_count)
        and torch.equal(base.term_count, tuned.term_count)
    )
    if not same:
        msg = "paired: the two evaluations differ in examples, groups or target counts"
        raise ValueError(msg)
    ordered = torch.sort(base.group).values
    first = torch.ones_like(ordered, dtype=torch.bool)
    first[1:] = ordered[1:] != ordered[:-1]
    groups = ordered[first]  # the distinct groups, sorted
    inverse = torch.searchsorted(groups, base.group)
    n_groups = int(groups.shape[0])
    generator = torch.Generator().manual_seed(seed)
    report: dict[str, object] = {"groups": n_groups, "examples": int(base.index.shape[0])}
    for part in ("body", "term"):
        count = base.body_count if part == "body" else base.term_count
        sums = [e.body_sum if part == "body" else e.term_sum for e in (base, tuned)]
        per_group = [
            torch.zeros(n_groups, dtype=torch.float64).index_add_(0, inverse, x)
            for x in (*sums, count.double())
        ]
        contributing = per_group[2] > 0
        n_contributing = int(contributing.sum())
        if n_contributing == 0:
            report[part] = {"status": "unassessed", "targets": 0, "groups_contributing": 0}
            continue
        kept = [g[contributing] for g in per_group]
        b_nll = float(kept[0].sum() / kept[2].sum())
        t_nll = float(kept[1].sum() / kept[2].sum())
        draws = torch.randint(n_contributing, (resamples, n_contributing), generator=generator)
        resampled = [g[draws].sum(1) for g in kept]
        delta = (resampled[1] - resampled[0]) / resampled[2]
        bounds = (float(torch.quantile(delta, 0.025)), float(torch.quantile(delta, 0.975)))
        report[part] = {
            "status": "assessed",
            "targets": int(count.sum()),
            "groups_contributing": n_contributing,
            "groups_without_targets": n_groups - n_contributing,
            "base_nll": b_nll,
            "tuned_nll": t_nll,
            "delta": t_nll - b_nll,
            "base_ppl": math.exp(b_nll),
            "tuned_ppl": math.exp(t_nll),
            "interval_95": list(bounds),
            "resamples": resamples,
        }
    return report


@dataclass(frozen=True)
class Generation:
    """A greedy continuation: the answer before the marker, how it stopped, the new ids."""

    answer: str
    stop: str  # "terminator" or "length"
    ids: tuple[int, ...]
    raw: str


def generate(
    model: Vesper,
    tok: Tokenizer,
    prompt: str,
    *,
    depth: int,
    max_new: int,
    decoder: CapturedDecoder | None = None,
) -> Generation:
    """Greedy decoding from cold states, every token at ``depth`` passes, on the model's device.

    The prompt is encoded alone (the training prefix exactly); each new id is fed with the
    carried states and the next position. The generated ids are decoded together after every
    step and the run stops when the text holds the marker, wherever BPE split or merged it
    (``instruct_panel.answer_of``). Without a marker after ``max_new`` ids the answer is the
    whole text and the stop is "length". Only ids the tokenizer can decode compete.

    On the CPU the prompt is one chunk and every new id one step (the behavioural reference).
    On CUDA every prompt token and every new id is one replay of ``decoder``'s captured
    single-token step (``vesper.decode``; one is made when none is given, pass one to reuse its
    capture across prompts); the marker check and the stops are the same.

    Raises:
        ValueError: before any prompt work, when ``decoder`` was made for another model, depth,
            vocabulary or device (maintainer B's review of 0e342c3: it was used silently), or when
            the prompt encodes to no ids (nothing would be computed; the captured path emitted id
            0).
    """
    device = next(model.parameters()).device
    if decoder is not None and (
        decoder.model is not model
        or decoder.depth != depth
        or decoder.vocab != tok.vocab_size
        or decoder.device != device
    ):
        msg = (
            f"the decoder was made for depth {decoder.depth}, vocab {decoder.vocab} on "
            f"{decoder.device} and another model or settings than this call's (depth {depth}, "
            f"vocab {tok.vocab_size} on {device})"
        )
        raise ValueError(msg)
    ids = tok.encode(prompt).long().to(device)
    if int(ids.shape[0]) == 0:
        msg = "generate: the prompt encodes to no ids"
        raise ValueError(msg)
    if device.type == "cuda":
        made = decoder or CapturedDecoder(model, depth, tok.vocab_size)
        return _generate_captured(made, tok, ids, max_new=max_new)
    states: Sequence[ColumnState] = model.init_states(1, torch.float32, device)
    tokens, pos = ids[None], torch.arange(int(ids.shape[0]), device=device)[None]
    new: list[int] = []
    text = ""
    with torch.no_grad():
        for _ in range(max_new):
            result = model.run(tokens, pos, states, passes=depth, exit_rule=DenseRule())
            states = result.states
            logits = model.head(result.y[:, -1:])[..., : tok.vocab_size]
            nxt = int(logits.argmax(-1).item())
            new.append(nxt)
            text = tok.decode(torch.tensor(new, dtype=torch.int64))
            if (answer := answer_of(text)) is not None:
                return Generation(answer, "terminator", tuple(new), text)
            tokens = torch.full((1, 1), nxt, dtype=torch.int64, device=device)
            pos = pos[:, -1:] + 1
    return Generation(text, "length", tuple(new), text)


def _generate_captured(
    decoder: CapturedDecoder, tok: Tokenizer, ids: Tensor, *, max_new: int
) -> Generation:
    """``generate`` on CUDA: the prompt's ids one by one, then each new id, through ``decoder``."""
    device = decoder.device

    def at(position: int) -> Tensor:
        return torch.full((1, 1), position, dtype=torch.int64, device=device)

    flat = decoder.start()
    nxt_t = torch.zeros(1, 1, dtype=torch.int64, device=device)
    for i in range(int(ids.shape[0])):
        nxt_t, _, flat = decoder.step(ids[i].view(1, 1), at(i), flat)
    new: list[int] = []
    text = ""
    for k in range(max_new):
        nxt = int(nxt_t.item())  # read before the next replay overwrites it
        new.append(nxt)
        text = tok.decode(torch.tensor(new, dtype=torch.int64))
        if (answer := answer_of(text)) is not None:
            return Generation(answer, "terminator", tuple(new), text)
        if k + 1 < max_new:
            token = torch.full((1, 1), nxt, dtype=torch.int64, device=device)
            nxt_t, _, flat = decoder.step(token, at(int(ids.shape[0]) + k), flat)
    return Generation(text, "length", tuple(new), text)


@dataclass(frozen=True)
class Binding:
    """What a tuning run read, which ``final`` must read again before any reserved forward.

    ``tune`` writes it into ``result.json``; ``final`` builds the same record from ITS inputs
    and refuses on any difference before it claims the directory or reads a test example (maintainer
    B a62ff1e and e178dab: a different source or a different correctly hashed file reached
    evaluation). ``limited`` marks a preflight run, which has no reserved read at all: ``final``
    is always unlimited, so a preflight's binding differs there. The config's bytes cover the
    dataset pin, the split, the example limit, the batching and the evaluation settings.
    """

    limited: bool
    model_sha256: str
    identity: tuple[tuple[str, str], ...]
    instruct_sha256: str
    panel_sha256: str
    scoring: str

    def record(self) -> dict[str, object]:
        """As written into ``result.json``."""
        return {
            "limited": self.limited,
            "model_sha256": self.model_sha256,
            "identity": dict(self.identity),
            "instruct_sha256": self.instruct_sha256,
            "panel_sha256": self.panel_sha256,
            "scoring": self.scoring,
        }

    @staticmethod
    def parse(record: object) -> "Binding":
        """A record written by ``record``, read back.

        Raises:
            ValueError: when it is not such a record.
        """
        if not isinstance(record, dict):
            msg = "result.json: no binding record"
            raise ValueError(msg)  # noqa: TRY004  # a malformed file is a value error
        fields = cast("dict[str, object]", record)
        identity = cast("dict[str, str]", fields["identity"])
        return Binding(
            limited=cast("bool", fields["limited"]),
            model_sha256=cast("str", fields["model_sha256"]),
            identity=tuple(sorted(identity.items())),
            instruct_sha256=cast("str", fields["instruct_sha256"]),
            panel_sha256=cast("str", fields["panel_sha256"]),
            scoring=cast("str", fields["scoring"]),
        )

    def differences(self, other: "Binding") -> list[str]:
        """The names of the fields in which ``other`` differs."""
        pairs: list[tuple[str, object, object]] = [
            ("limited", self.limited, other.limited),
            ("model_sha256", self.model_sha256, other.model_sha256),
            ("identity", self.identity, other.identity),
            ("instruct_sha256", self.instruct_sha256, other.instruct_sha256),
            ("panel_sha256", self.panel_sha256, other.panel_sha256),
            ("scoring", self.scoring, other.scoring),
        ]
        return [name for name, a, b in pairs if a != b]


def _source_corpus(
    source: Path, loaded: TextSource, *, data: Path, finalization: Path
) -> tuple[list[Component], Finalization]:
    """The checkpoint's own pinned components and its finalization, checked against its identity.

    ``loaded`` is the checkpoint at ``source`` as the caller loaded and checked it. The record is
    read ONCE here; that snapshot must be the loaded checkpoint (identity and model.pt hash,
    maintainer B's review of eb4d9f9), and the components are resolved from the same snapshot
    (``train.components_of_record``, whose recipe must hash to that identity), so nothing read
    later can change them (maintainer B's review of 01e2b3f).

    Raises:
        ValueError: when the checkpoint is not the loaded one, its recipe does not hash to its
            identity, or the finalization does not hash as recorded.
    """
    identity = loaded.identity
    checked = checked_record(source)
    same = (
        checked.record.get("identity") == identity
        and checked.manifest.get("model.pt") == loaded.model_sha256
    )
    if not same:
        msg = f"{source} is no longer the checkpoint that was loaded"
        raise ValueError(msg)
    parts = components_of_record(data, checked, source)
    return parts, load_finalization(finalization, sha256=identity["finalization_sha256"])


def days_view(  # noqa: PLR0913
    source: Path,
    loaded: TextSource,
    tok: Tokenizer,
    icfg: InstructConfig,
    *,
    data: Path,
    finalization: Path,
    pin: Path,
    expected: bool = True,
) -> tuple[list[Chunk], dict[str, object]]:
    """The days run's held-out view and its record, checked before any model reads it.

    ``loaded`` is the checkpoint at ``source`` as the caller already loaded and checked it; the
    record read once in ``_source_corpus`` must be that one (its identity and model.pt hash) and
    the components are resolved from that snapshot, so the view belongs to the model that is
    scored on it (maintainer B's reviews of eb4d9f9 and 01e2b3f). Then the checkpoint's own
    pinned components (today's selection cannot change them), its finalization receipt checked
    against the identity, the first ``eval_chunks`` chunks of the seed-0 held-out stream at
    ``icfg.days_micro`` slots, every document inside the accepted test pin
    (``halting_text.validation_chunks``), the pin's hash equal to the config's and the view's
    digest (``SequenceDigest`` over every chunk's tokens, targets, loss mask, positions and
    document ids) equal to the config's. ``expected=False`` skips only that last comparison (the
    ``view`` command that prints the digest).

    Raises:
        ValueError: on any of those checks.
    """
    cfg = loaded.cfg
    parts, fin = _source_corpus(source, loaded, data=data, finalization=finalization)
    held = Split(fin, cfg.training.held_out_fraction, held_out=True)
    pinned = validation_chunks(parts, tok, held, cfg, pin, micro=icfg.days_micro)
    if pinned.pin_sha256 != icfg.pin_sha256:
        msg = f"the pin {pin} is not the config's ({pinned.pin_sha256} != {icfg.pin_sha256})"
        raise ValueError(msg)
    digest = SequenceDigest()
    documents: set[str] = set()
    for chunk in pinned.chunks:
        for t in (chunk.tokens, chunk.targets, chunk.loss_mask, chunk.pos):
            digest.add(int64_bytes(t))
        digest.add("\n".join("\t".join(slot) for slot in chunk.doc_ids).encode())
        documents.update(d for slot in chunk.doc_ids for d in slot)
    view_sha256 = digest.hexdigest()
    if expected and view_sha256 != icfg.days_view_sha256:
        msg = f"the days view's digest {view_sha256} is not the config's {icfg.days_view_sha256}"
        raise ValueError(msg)
    return pinned.chunks, {
        "rule": (
            "train.resumed_components(checkpoint pins), finalization checked against the"
            " checkpoint identity, halting_text.validation_chunks(seed 0, micro"
            f" {icfg.days_micro}) inside the accepted pin, digest over tokens/targets/mask/pos/"
            "document ids"
        ),
        "pin_sha256": pinned.pin_sha256,
        "view_sha256": view_sha256,
        "chunks": len(pinned.chunks),
        "documents": len(documents),
        "labels": sum(int(c.loss_mask.sum()) for c in pinned.chunks),
    }


# --- the commands -----------------------------------------------------------------------------


def _snapshot(model: Vesper) -> dict[str, Tensor]:
    """The model's parameters and buffers, cloned on the CPU: the checked source, kept."""
    state = cast("dict[str, Tensor]", model.state_dict())
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _fresh(cfg: VesperConfig, state: dict[str, Tensor]) -> Vesper:
    """A model built from the checked config and a kept snapshot: never a reload of files."""
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    _ = cast("object", model.load_state_dict(state))
    return model


def _halting_tensors(model: Vesper) -> dict[str, Tensor]:
    return {
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if n.startswith(HALTING_PREFIX)
    }


def _sha256_of_state(model: Vesper, path: Path) -> str:
    buffer = io.BytesIO()
    state = cast("dict[str, Tensor]", model.state_dict())
    torch.save({k: v.cpu() for k, v in state.items()}, buffer)
    data = buffer.getvalue()
    _ = path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _load_weights(model: Vesper, path: Path, sha256: str) -> None:
    data = verified_bytes(path, sha256, "tuned weights")
    weights = cast(
        "dict[str, Tensor]", torch.load(io.BytesIO(data), weights_only=True, map_location="cpu")
    )
    _ = cast("object", model.load_state_dict(weights))


def _write_json(path: Path, record: dict[str, object]) -> str:
    """``record`` as JSON at ``path``; the sha256 of exactly the bytes written."""
    data = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    _ = path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _losses_record(losses: ExampleLosses) -> dict[str, object]:
    return {
        "body_nll": losses.pooled("body"),
        "term_nll": losses.pooled("term"),
        "body_targets": int(losses.body_count.sum()),
        "term_targets": int(losses.term_count.sum()),
        "examples": int(losses.index.shape[0]),
    }


def _save_losses(losses: ExampleLosses, path: Path) -> str:
    buffer = io.BytesIO()
    torch.save(cast("dict[str, object]", asdict(losses)), buffer)
    data = buffer.getvalue()
    _ = path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _tokenizer(path: Path, sha256: str) -> Tokenizer:
    return Tokenizer.parse(verified_bytes(path, sha256, "the source run's tokenizer"))


def _days_depth_loss(
    model: Vesper, chunks: list[Chunk], cfg: VesperConfig, depth: int
) -> dict[str, object]:
    device = next(model.parameters()).device
    graphs = EvalGraphs() if device.type == "cuda" else None
    profile = static_profile(model, chunks, cfg, graphs=graphs)
    loss = profile.loss_sums[depth - 1] / profile.tokens_with_loss
    return {"loss": loss, "perplexity": math.exp(loss), "labels": profile.tokens_with_loss}


def _on(model: Vesper, device: torch.device) -> Vesper:
    model = model.to(device)
    _ = set_scan(model, "fused-fp32" if device.type == "cuda" else "reference")
    return model


def _limited(prepared: Prepared, limit: int, buckets: Sequence[int]) -> Prepared:
    """A preflight's workload: per split, the first ``limit`` examples of EACH length bucket.

    So every captured length, and the alternation between them, is exercised however short
    the first examples are (maintainer B's review of e178dab). 0: everything.
    """
    if limit == 0:
        return prepared

    def bucket(e: Example) -> int:
        return next(b for b in buckets if e.inputs <= b)

    examples = {
        s: tuple(e for b in buckets for e in [x for x in kept if bucket(x) == b][:limit])
        for s, kept in prepared.examples.items()
    }
    return Prepared(examples, prepared.groups, prepared.counts, prepared.membership)


def _data_record(icfg: InstructConfig, prepared: Prepared) -> dict[str, object]:
    """``data.json``: the dataset's identity, the rules, the counts and every record's fate.

    The config as a whole is bound by ``Binding`` (one check per fact); this record holds what
    the preparation produced, which ``final`` reproduces.
    """
    return {
        "dataset": icfg.dataset_name,
        "revision": icfg.dataset_revision,
        "file_sha256": icfg.dataset_sha256,
        "split": [icfg.split_key, icfg.train_below, icfg.validation_below],
        "max_inputs": icfg.max_inputs,
        "attribution": ATTRIBUTION,
        "canonical": CANONICAL,
        "terminator": TERMINATOR,
        "counts": prepared.counts,
        "membership": [list(m) for m in prepared.membership],
        "claim": "exact-text connected groups; no semantic decontamination",
    }


@dataclass(frozen=True)
class _Inputs:
    """The inputs both commands read and check before anything else happens."""

    source: TextSource
    icfg: InstructConfig
    instruct_sha256: str
    panel: Panel
    panel_sha256: str
    tok: Tokenizer
    prepared: Prepared


def _inputs(args: argparse.Namespace) -> _Inputs:
    """Source, configs, panel, tokenizer and the prepared dataset, each read once and checked.

    Raises:
        ValueError: on any failed check (a hash, a malformed panel, a refused config).
    """
    source = load_text_source(cast("Path", args.source), cast("Path", args.config))
    instruct_path = cast("Path", args.instruct)
    instruct_bytes, instruct_sha256 = hashed_bytes(instruct_path)
    icfg = parse_instruct_bytes(instruct_bytes, str(instruct_path))
    panel_bytes, panel_sha256 = hashed_bytes(cast("Path", args.panel))
    panel = load_panel(panel_bytes)
    tok = _tokenizer(cast("Path", args.tokenizer), source.identity["tokenizer_sha256"])
    records = load_records(cast("Path", args.dolly), icfg.dataset_sha256)
    rule = SplitRule(icfg.split_key, icfg.train_below, icfg.validation_below)
    prepared = prepare(records, tok, max_inputs=icfg.max_inputs, rule=rule)
    return _Inputs(source, icfg, instruct_sha256, panel, panel_sha256, tok, prepared)


def _binding(inputs: _Inputs, *, limited: bool) -> Binding:
    return Binding(
        limited=limited,
        model_sha256=inputs.source.model_sha256,
        identity=tuple(sorted(inputs.source.identity.items())),
        instruct_sha256=inputs.instruct_sha256,
        panel_sha256=inputs.panel_sha256,
        scoring=SCORING,
    )


@dataclass(frozen=True)
class _Training:
    """What the training scope reads: the run's directory, the checked source and the recipe."""

    out: Path
    cfg: VesperConfig
    base_state: dict[str, Tensor]
    source_sha256: str
    recipe: Recipe
    prepared: Prepared
    device: torch.device
    replay: ReplaySource | None  # one row set per update when the recipe's weight is above 0


@dataclass(frozen=True)
class _Trained:
    """What a tuning run keeps from its training (``_train``); nothing here holds device memory."""

    epochs: list[dict[str, object]]
    per_epoch: list[ExampleLosses]
    halting_unchanged: bool
    graphs: dict[str, list[int]]
    settings: dict[str, object]
    steps: int
    steps_per_epoch: int


def _train(run: _Training, phases: MemoryPhases) -> _Trained:  # noqa: PLR0915
    """Train and validate every epoch, writing ``train.jsonl``, each epoch's weights and losses.

    The scope that owns the training: its model (built from the checked source snapshot, so the
    loaded source model is never trained or moved), its optimizer and its graph family exist only
    inside this call, and ``phases`` holds the family only while it runs. What the run keeps
    (records, copied-out losses, the halting check, the graph counters, the determinism
    settings) is returned; the rest ends with the call, before the forgetting phase builds its
    models (maintainer B's design, af89c52; r7 measured the card full there).
    """
    recipe, out, device = run.recipe, run.out, run.device
    cuda = device.type == "cuda"
    graphs = GraphFamily() if cuda else None  # training and validation, one memory pool
    phases.graphs = graphs
    try:
        with phases.phase("setup"):
            model = _on(_fresh(run.cfg, run.base_state), device)
            halting_before = _halting_tensors(model)
            opt = tuning_optimizer(model, recipe, capturable=cuda)

        def plan(split: str, epoch: int) -> list[Batch]:
            return batches(
                run.prepared.examples[split],
                epoch=epoch,
                seed=recipe.seed,
                batch_tokens=recipe.batch_tokens,
                buckets=recipe.buckets,
            )

        validation = plan("validation", 0)
        weights_dir = out / "weights"
        weights_dir.mkdir()
        epochs: list[dict[str, object]] = []
        per_epoch: list[ExampleLosses] = []

        def score(epoch: int, weights_sha256: str) -> None:
            losses = evaluate_examples(
                model, validation, depth=recipe.eval_depth, graphs=graphs, slices=recipe.slices
            )
            per_epoch.append(losses)
            digest = _save_losses(losses, out / f"validation-epoch-{epoch}.pt")
            epochs.append(
                {
                    "epoch": epoch,
                    "weights_sha256": weights_sha256,
                    "losses_sha256": digest,
                    **_losses_record(losses),
                }
            )
            log.info("epoch %d validation body NLL %s", epoch, losses.pooled("body"))

        steps_per_epoch = len(plan("train", 1))
        total = steps_per_epoch * recipe.epochs
        log_path = out / "train.jsonl"
        cpu_settings: dict[str, object] = {"deterministic": "not set on the CPU"}
        with deterministic_algorithms() if cuda else nullcontext(cpu_settings) as effective:
            settings = dict(effective)
            with phases.phase("validation-0"):
                score(0, run.source_sha256)
            step = 0
            for epoch in range(1, recipe.epochs + 1):
                with phases.phase(f"train-{epoch}"):
                    for batch in plan("train", epoch):
                        lr = warmup_cosine(
                            step,
                            total,
                            peak=recipe.peak_lr,
                            warmup=recipe.warmup_steps,
                            final_fraction=recipe.final_lr_fraction,
                        )
                        rows = None if run.replay is None else run.replay.next()
                        rec = train_step(model, opt, batch, lr, recipe, graphs, replay=rows)
                        if not math.isfinite(rec.objective):
                            msg = f"non-finite loss at step {step}"
                            raise FloatingPointError(msg)
                        line: dict[str, object] = {
                            "step": step,
                            "epoch": epoch,
                            "lr": lr,
                            "length": int(batch.tokens.shape[1]),
                            **cast("dict[str, object]", asdict(rec)),
                        }
                        if rows is not None:
                            line["replay_rows"] = rows.record
                        with log_path.open("a") as handle:
                            _ = handle.write(json.dumps(line) + "\n")
                        step += 1
                with phases.phase(f"save-{epoch}"):
                    digest = _sha256_of_state(model, weights_dir / f"epoch-{epoch}.pt")
                with phases.phase(f"validation-{epoch}"):
                    score(epoch, digest)
        with phases.phase("halting-check"):
            halting_after = _halting_tensors(model)
            unchanged = all(
                torch.equal(halting_before[n], halting_after[n]) for n in halting_before
            )
        return _Trained(
            epochs=epochs,
            per_epoch=per_epoch,
            halting_unchanged=unchanged,
            graphs={} if graphs is None else graphs.counts(),
            settings=settings,
            steps=total,
            steps_per_epoch=steps_per_epoch,
        )
    finally:
        phases.graphs = None  # the records keep the counts; the family ends with this scope


def _replay_source(
    args: argparse.Namespace, inputs: _Inputs
) -> tuple[ReplaySource | None, dict[str, object]]:
    """The run's replay rows (``instruct_replay``) and their record; none without a weight.

    The source's own pinned components and finalization (``_source_corpus``), its tokenizer
    (``_inputs``), the configured held-out fraction, the training side, and the accepted pin's
    bytes checked against the config's hash; all of it identified in the record. Weight 0, or
    no ``replay`` section, reads nothing.

    Raises:
        ValueError: on any failed check of those inputs.
    """
    replay = inputs.icfg.replay
    config = None if replay is None else cast("dict[str, object]", asdict(replay))
    if replay is None or replay.weight == 0:
        return None, {"config": config, "rows_read": False}
    source = inputs.source
    parts, fin = _source_corpus(
        cast("Path", args.source),
        source,
        data=cast("Path", args.data),
        finalization=cast("Path", args.finalization),
    )
    fraction = source.cfg.training.held_out_fraction
    pin = verified_bytes(cast("Path", args.pin), inputs.icfg.pin_sha256, "the accepted test pin")
    rows = ReplaySource(parts, inputs.tok, Split(fin, fraction), replay, pin)
    return rows, {
        "config": config,
        "rows_read": True,
        "side": "training",
        "held_out_fraction": fraction,
        "finalization_sha256": source.identity["finalization_sha256"],
        "tokenizer_sha256": source.identity["tokenizer_sha256"],
        "pin_sha256": inputs.icfg.pin_sha256,
        "components": [
            {
                "name": c.name,
                "share": c.share,
                "generation": c.generation,
                "paths": [str(p) for p in c.paths],
            }
            for c in parts
        ],
        "objective": (
            "instruction mean + weight * replay mean, each over its own scored targets and the"
            " dense passes; one clipped AdamW step per update"
        ),
    }


def tune(args: argparse.Namespace) -> int:
    """The tuning command (module docstring); 0 when complete and the head unchanged.

    Every input is read and checked, the days view included, before the output directory is
    made. Writes ``data.json``, ``train.jsonl``, the weights of every epoch, each epoch's
    validation losses and ``result.json``.
    """
    out = cast("Path", args.out)
    started = time.monotonic()
    inputs = _inputs(args)
    source, icfg = inputs.source, inputs.icfg
    cfg = source.cfg
    recipe = recipe_for(
        cfg, icfg, recompute=cast("bool", args.recompute), slices=cast("int", args.loss_slices)
    )
    limit = cast("int", args.limit_examples)
    if limit < 0:
        msg = f"--limit-examples must be at least 0, got {limit}"
        raise ValueError(msg)
    view, view_record = days_view(
        cast("Path", args.source),
        source,
        inputs.tok,
        icfg,
        data=cast("Path", args.data),
        finalization=cast("Path", args.finalization),
        pin=cast("Path", args.pin),
    )
    replay, replay_record = _replay_source(args, inputs)
    base_state = _snapshot(source.model)  # the checked source, kept: no later reload
    prepared = _limited(inputs.prepared, limit, recipe.buckets)
    out.mkdir(parents=True, exist_ok=False)
    data_sha256 = _write_json(out / "data.json", _data_record(icfg, inputs.prepared))
    device = torch.device(cast("str", args.device))
    # all device work from here to the end runs inside a phase (between phases only CPU work:
    # the selection, the paired report on copied-out losses), so the phases' largest reserved
    # peak is the run's peak_reserved_bytes
    phases = MemoryPhases(device, out / "memory.jsonl")
    # the training model, its optimizer and graphs end with this call; only its results remain
    trained = _train(
        _Training(out, cfg, base_state, source.model_sha256, recipe, prepared, device, replay),
        phases,
    )
    if replay is not None:
        replay_record["consumed"] = replay.summary()
    epochs, per_epoch, unchanged = trained.epochs, trained.per_epoch, trained.halting_unchanged
    weights_dir = out / "weights"
    nlls = [cast("float", e["body_nll"]) for e in epochs[1:]]
    selected = 1 + nlls.index(min(nlls))  # ties to the earlier epoch
    selected_sha = cast("str", epochs[selected]["weights_sha256"])
    with phases.phase("forgetting-build-base"):
        base_days = _on(_fresh(cfg, base_state), device)
    with phases.phase("forgetting-build-selected"):
        # the selected epoch from its saved bytes, whichever epoch it is (one path, no live model)
        tuned_days = _fresh(cfg, base_state)
        _load_weights(tuned_days, weights_dir / f"epoch-{selected}.pt", selected_sha)
        tuned_days = _on(tuned_days, device)
    with phases.phase("forgetting-base"):
        base_loss = _days_depth_loss(base_days, view, cfg, recipe.eval_depth)
    with phases.phase("forgetting-selected"):
        tuned_loss = _days_depth_loss(tuned_days, view, cfg, recipe.eval_depth)
    forgetting = {
        "view": view_record,
        "depth": recipe.eval_depth,
        "base": base_loss,
        "selected": tuned_loss,
    }
    result: dict[str, object] = {
        "recipe": cast("dict[str, object]", asdict(recipe)),
        "binding": _binding(inputs, limited=limit != 0).record(),
        "data_json_sha256": data_sha256,
        "limit_examples": limit,
        "run_kind": "preflight (limited examples)" if limit else "trial",
        "source": {
            "checkpoint": str(cast("Path", args.source)),
            "model_sha256": source.model_sha256,
            "identity": source.identity,
            "step": source.step,
        },
        "steps": trained.steps,
        "steps_per_epoch": trained.steps_per_epoch,
        "epochs": epochs,
        "selected_epoch": selected,
        "selected_weights_sha256": selected_sha,
        "selection": "lowest pooled validation body NLL among epochs 1..; ties to the earlier",
        "validation_paired_descriptive": paired(
            per_epoch[0], per_epoch[selected], resamples=recipe.resamples, seed=recipe.seed
        ),
        "halting_unchanged": unchanged,
        "forgetting": forgetting,
        "replay": replay_record,
        # the training/validation family per (kind, batch length): [captures, replays]; the
        # forgetting forwards run their own graphs, not counted here
        "graphs": trained.graphs,
        # the largest reserved peak over the phases, which cover the run from the model's move
        # to the device to the last forgetting forward; each phase's record in memory_phases
        "peak_reserved_bytes": phases.reserved_peak(),
        "memory_phases": phases.records,
        "determinism": trained.settings,
        "seconds": time.monotonic() - started,
        "environment": environment(),
        "device": str(device),
    }
    _ = _write_json(out / "result.json", result)
    return 0 if unchanged else 1


def _final_checks(
    args: argparse.Namespace, out: Path
) -> tuple[_Inputs, dict[str, object], str] | None:
    """Every check ``final`` makes before its claim, or None (logged).

    Returns the inputs, the run's result and the hash of exactly the result bytes parsed (it is
    carried into final.json, never recomputed from the path: maintainer B's review of eb4d9f9).
    Reads no test example into a model: the dataset is parsed and prepared (deterministic filtering
    reads every record's bytes) to reproduce the run's membership.
    """
    result_bytes, result_sha = hashed_bytes(out / "result.json")
    result = cast("dict[str, object]", json.loads(result_bytes))
    inputs = _inputs(args)
    if differences := Binding.parse(result.get("binding")).differences(
        _binding(inputs, limited=False)
    ):
        log.error("final's inputs differ from the tuning run's in %s: refused", differences)
        return None
    data_path = out / "data.json"
    if not data_path.is_file():
        log.error("%s: missing; refused", data_path)
        return None
    data_bytes, data_sha = hashed_bytes(data_path)
    if data_sha != result.get("data_json_sha256"):
        log.error("%s changed since the tuning run; refused", data_path)
        return None
    recorded = cast("dict[str, object]", json.loads(data_bytes))
    again = cast(
        "dict[str, object]",
        json.loads(json.dumps(_data_record(inputs.icfg, inputs.prepared))),
    )
    if recorded != again:
        log.error("the dataset's preparation does not reproduce %s; refused", data_path)
        return None
    return inputs, result, result_sha


def final(args: argparse.Namespace) -> int:
    """The one reserved read (module docstring), into ``final.json`` in the tuning directory.

    0 when complete; 2 when refused before the claim; 3 when this directory's claim already
    exists (never a second read).
    """
    out = cast("Path", args.out)
    claim = out / "final.attempt"
    if claim.exists():  # the cheap refusal first; the mkdir below is the exclusive claim
        log.error("final already attempted in %s; not reading the reserved split again", out)
        return 3
    try:
        checked = _final_checks(args, out)
        if checked is None:
            return 2
        inputs, result, result_sha = checked
        slices = _recorded_slices(result)
        selected = cast("int", result["selected_epoch"])
        # the tuned model from the checked config and the loaded source, never a reload of files
        tuned = _fresh(inputs.source.cfg, _snapshot(inputs.source.model))
        weights = out / "weights" / f"epoch-{selected}.pt"
        _load_weights(tuned, weights, cast("str", result["selected_weights_sha256"]))
    except (ValueError, KeyError) as exc:
        log.error("final refused before its claim: %r", exc)  # noqa: TRY400
        return 2
    try:
        claim.mkdir()
    except FileExistsError:
        log.error("final already attempted in %s; not reading the reserved split again", out)  # noqa: TRY400
        return 3
    try:
        _final_read(args, out, inputs, (result, result_sha), tuned, slices)
    except BaseException as exc:
        _ = _write_json(claim / "status.json", {"status": "failed", "error": repr(exc)})
        raise
    _ = _write_json(claim / "status.json", {"status": "complete"})
    return 0


def _recorded_slices(result: dict[str, object]) -> int:
    """The tuning run's loss slices, from its checked ``result.json``.

    ``final`` scores the head in the same row groups as the run's validations (maintainer B's review
    of 71f3093: it used 1).

    Raises:
        ValueError: when the record has no positive integer ``recipe.slices``.
    """
    raw = result.get("recipe")
    recipe = cast("dict[str, object]", raw) if isinstance(raw, dict) else {}
    slices = recipe.get("slices")
    if isinstance(slices, bool) or not isinstance(slices, int) or slices < 1:
        msg = f"result.json: recipe.slices must be a positive integer, got {slices!r}"
        raise ValueError(msg)
    return slices


def _final_read(
    args: argparse.Namespace,
    out: Path,
    inputs: _Inputs,
    run: tuple[dict[str, object], str],
    tuned: Vesper,
    slices: int,
) -> None:
    """After the claim: the test split for both models, then the panel; ``final.json``."""
    result, result_sha = run
    recipe = recipe_for(inputs.source.cfg, inputs.icfg, slices=slices)
    device = torch.device(cast("str", args.device))
    test = batches(
        inputs.prepared.examples["test"],
        epoch=0,
        seed=recipe.seed,
        batch_tokens=recipe.batch_tokens,
        buckets=recipe.buckets,
    )
    models = {"base": _on(inputs.source.model, device), "tuned": _on(tuned, device)}
    losses: dict[str, ExampleLosses] = {}
    for name, model in models.items():
        losses[name] = _test_losses(model, test, recipe, device)
        _ = _save_losses(losses[name], out / f"test-{name}.pt")
    # on CUDA one captured decoder per model serves its whole panel (one capture per model)
    decoders = (
        {n: CapturedDecoder(m, recipe.eval_depth, inputs.tok.vocab_size) for n, m in models.items()}
        if device.type == "cuda"
        else {}
    )
    scored, illustrations = _panel_rows(models, decoders, inputs.tok, inputs.panel, recipe)

    def passed(name: str) -> list[bool]:
        return [cast("bool", cast("dict[str, object]", r[name])["pass"]) for r in scored]

    base_pass, tuned_pass = passed("base"), passed("tuned")
    pairs = list(zip(base_pass, tuned_pass, strict=True))
    _ = _write_json(
        out / "final.json",
        {
            "result_sha256": result_sha,
            "selected_epoch": result["selected_epoch"],
            "loss_slices": recipe.slices,
            "test_paired": paired(
                losses["base"], losses["tuned"], resamples=recipe.resamples, seed=recipe.seed
            ),
            "panel_sha256": inputs.panel_sha256,
            "scoring": SCORING,
            "panel": {
                "items": len(scored),
                "base_passed": sum(base_pass),
                "tuned_passed": sum(tuned_pass),
                "both": sum(a and b for a, b in pairs),
                "only_base": sum(a and not b for a, b in pairs),
                "only_tuned": sum(b and not a for a, b in pairs),
                "neither": sum(not a and not b for a, b in pairs),
                "rows": scored,
            },
            "illustrations": illustrations,
            "generation": (
                f"greedy, fixed depth {recipe.eval_depth}, cold states, {_backend(decoders)},"
                f" at most {recipe.max_new_tokens} new ids, stop at the marker in the decoded text"
            ),
            "environment": environment(),
        },
    )


def _test_losses(
    model: Vesper, test: Sequence[Batch], recipe: Recipe, device: torch.device
) -> ExampleLosses:
    """One model's test losses; its graph family (one per model: its weights) ends on return."""
    graphs = GraphFamily() if device.type == "cuda" else None
    return evaluate_examples(
        model, test, depth=recipe.eval_depth, graphs=graphs, slices=recipe.slices
    )


def _backend(decoders: dict[str, CapturedDecoder]) -> str:
    """How the panel was generated, for ``final.json``."""
    return "CUDA, one captured single-token decoder per model" if decoders else "CPU"


def _panel_rows(
    models: dict[str, Vesper],
    decoders: dict[str, CapturedDecoder],
    tok: Tokenizer,
    panel: Panel,
    recipe: Recipe,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Every model's greedy answer to every panel item, scored, and to every illustration.

    ``decoders`` holds the model's captured decoder on CUDA (reused for every prompt) and is
    empty on the CPU.
    """
    depth, most = recipe.eval_depth, recipe.max_new_tokens

    def answer(name: str, prompt: str) -> Generation:
        return generate(
            models[name], tok, prompt, depth=depth, max_new=most, decoder=decoders.get(name)
        )

    scored: list[dict[str, object]] = []
    for item in panel.scored:
        prompt = prompt_text(str(item["instruction"]), str(item["context"]))
        row: dict[str, object] = {"id": item["id"]}
        checks = cast("list[dict[str, object]]", item["checks"])
        for name in models:
            g = answer(name, prompt)
            row[name] = {"answer": g.answer, "stop": g.stop, "pass": score_answer(checks, g.answer)}
        scored.append(row)
    illustrations: list[dict[str, object]] = []
    for item in panel.illustrations:
        prompt = prompt_text(str(item["instruction"]), str(item["context"]))
        shown: dict[str, object] = {"instruction": item["instruction"]}
        for name in models:
            g = answer(name, prompt)
            shown[name] = {"answer": g.answer, "stop": g.stop}
        illustrations.append(shown)
    return scored, illustrations


def _view(args: argparse.Namespace) -> int:
    """Print the days view's record (its digest included); no model reads it."""
    source = load_text_source(cast("Path", args.source), cast("Path", args.config))
    instruct_path = cast("Path", args.instruct)
    instruct_bytes, _ = hashed_bytes(instruct_path)
    icfg = parse_instruct_bytes(instruct_bytes, str(instruct_path))
    tok = _tokenizer(cast("Path", args.tokenizer), source.identity["tokenizer_sha256"])
    _, record = days_view(
        cast("Path", args.source),
        source,
        tok,
        icfg,
        data=cast("Path", args.data),
        finalization=cast("Path", args.finalization),
        pin=cast("Path", args.pin),
        expected=False,
    )
    print(json.dumps(record, indent=2, sort_keys=True))  # noqa: T201  # the command's output
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesper.instruct", description=(__doc__ or "").split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("tune", "final", "view"):
        sub = commands.add_parser(name)
        _ = sub.add_argument("--source", type=Path, required=True)
        _ = sub.add_argument("--config", type=Path, required=True)
        _ = sub.add_argument("--tokenizer", type=Path, required=True)
        _ = sub.add_argument("--instruct", type=Path, required=True)
    for name in ("tune", "final"):
        sub = commands.choices[name]
        _ = sub.add_argument("--dolly", type=Path, required=True)
        _ = sub.add_argument("--panel", type=Path, required=True)
        _ = sub.add_argument("--out", type=Path, required=True)
        _ = sub.add_argument("--device", default="cpu")
    for name in ("tune", "view"):
        sub = commands.choices[name]
        _ = sub.add_argument("--data", type=Path, required=True)
        _ = sub.add_argument("--finalization", type=Path, required=True)
        _ = sub.add_argument("--pin", type=Path, required=True)
    tune_cmd = commands.choices["tune"]
    _ = tune_cmd.add_argument("--recompute", action="store_true")
    _ = tune_cmd.add_argument("--loss-slices", type=int, default=1)
    _ = tune_cmd.add_argument(
        "--limit-examples", type=int, default=0, help="a preflight: the first N of every split"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: ``tune``, ``final`` or ``view`` (module docstring)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    command = cast("str", args.command)
    if command == "tune":
        return tune(args)
    if command == "final":
        return final(args)
    return _view(args)


if __name__ == "__main__":
    sys.exit(main())
