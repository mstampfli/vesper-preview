"""The training loop (docs/reviews/2026-10-15-training-loop.md): phases, steps, checkpoints.

    python -m vesper.train --config configs/vesper-s.yaml --data ROOT \
        --finalization finalize-<stamp>.json --tokenizer vesper-s-32768-<stamp>.json \
        --out DIR --micro-chunks 2 [--steps N] [--resume DIR/step-N] [--seed 0] [--device cuda]

The first version: e2e wake training without retrieval, the ledger or dusk. Through the recipe's
phase 1 that is the recipe; through phase 2 it is the no-retrieval ablation, and every log line
and receipt carries that label. ``phase_schedule`` is a pure function of tokens seen and the step
index; ``optimizer`` decays weights by role, decided by parameter name; ``step`` takes one
optimizer step over ``batch_chunks`` slots as independent streams, with the token-weighted mean
loss over the whole batch; ``checkpoint`` and ``restore`` write and read a directory with a sha256
manifest, the streams' packing state, the carried states and the identity of what the run reads.
"""

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from vesper.attention import KVCache
from vesper.column import ColumnState, check_column_state, map_state
from vesper.config import VesperConfig, load
from vesper.corpus import (
    FINALIZE_RULE,
    SPLIT_RULE,
    Component,
    Mixer,
    MixerState,
    Split,
    check_inputs,
    iter_mixed_documents,
    load_finalization,
)
from vesper.data import Chunk, Document, SlotState, Stream, StreamState
from vesper.evals import evaluate, iteration_histogram
from vesper.generations import parse_pin, pinned
from vesper.keys import file_sha256
from vesper.learning import LossWeights, check_mode
from vesper.loop import HaltingHead
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.publish import staged
from vesper.sample import components_for
from vesper.tokenizer import Tokenizer

LABEL = (
    "e2e wake training; no retrieval, no ledger, no dusk: the recipe through phase 1, the"
    " no-retrieval ablation beyond"
)
DECAYED = frozenset(
    {
        "embedding.weight",
        "rec.w_q",
        "rec.w_k",
        "rec.w_v",
        "rec.w_o",
        "attn.w_q",
        "attn.w_k",
        "attn.w_v",
        "attn.w_o",
        "ffn.w1",
        "ffn.w2",
    }
)
UNDECAYED = frozenset(
    {
        "rec.w_beta",
        "rec.conv_k",  # the key filter's taps: a filter, like a gate (retrieval-binding 4.1)
        "halting.w_h",
        "halting.b_h",
        "halting.norm.weight",
        "loop.u",
        "norm_rec.weight",
        "norm_attn.weight",
        "norm_ffn.weight",
        "norm_out.weight",
    }
)
FILES = ("model.pt", "optimizer.pt", "states.pt", "train.json")
# the checkpoint schema changes whenever a stored shape does: 2 progress with the documents digest
# and the recipe in the identity; 3 the mixer state in states.pt; 4 the queued documents' source and
# progress.tokens_consumed (maintainer B's review of 629559f: a shape changed under an unchanged
# number); 5 the recipe's generation pins; 6 the halting head's norm gain in model.pt and its Adam
# state; 7 the recurrent key filter (rec.conv_k in model.pt) and its per-pass key history in
# states.pt
CHECKPOINT_SCHEMA = 7
log = logging.getLogger("vesper.train")


@dataclass(frozen=True)
class Phase:
    """What the recipe says for one step: the pass count, the frozen head, the weights, the lr."""

    name: str
    t_max: int
    halting_frozen: bool
    lambda_h: float
    lr: float
    retrieval: bool
    ledger_writes: bool
    dusk_eta_fraction: float
    qat: bool


@dataclass
class Group:
    """One micro-batch: its stream of ``micro_chunks`` slots and their carried states."""

    stream: Stream
    states: list[ColumnState]


@dataclass(frozen=True)
class StepResult:
    """What one optimizer step did; ``tokens`` is the step's own count, not the run's.

    ``consumed`` counts the step's input tokens per source as the streams emitted them (the
    chunks' ``spans``), which is what the model saw; the mixer's hand-out counts are not that.
    ``peak_memory_bytes`` is the accelerator's peak of allocated bytes during the step, with the
    carried states of every slot and the optimizer resident (0 on the CPU, which has no counter).
    """

    loss: float
    ntp: float
    ponder: float
    tokens_with_loss: int
    tokens: int
    consumed: dict[str, int]
    groups_active: int
    histogram: list[float]
    grad_norm: float
    seconds: float
    peak_memory_bytes: int


@dataclass(frozen=True)
class Recipe:
    """What decides the order and membership of the data a run consumes, beyond its three files.

    The seed, the ordered components with their shares, the held-out fraction, the finalize and
    split rules and the checkpoint schema. Its sha256 is part of the ``Identity`` and the plain
    record goes into the run header and ``train.json``: a run under another recipe is a labeled
    fork, never a continuation (maintainer B's review of b797837: another seed from the same
    checkpoint passed a count check over equal-length documents and diverged).
    """

    seed: int
    components: tuple[tuple[str, float], ...]
    generations: tuple[str | None, ...]
    held_out_fraction: float
    finalize_rule: str
    split_rule: str
    schema: int


@dataclass(frozen=True)
class Layout:
    """The micro-batch layout a run executes: ``groups`` streams of ``slots`` slots each."""

    groups: int
    slots: int


@dataclass(frozen=True)
class Identity:
    """The hashes of what a run reads and follows; a checkpoint with other ones is refused."""

    tokenizer_sha256: str
    config_sha256: str
    finalization_sha256: str
    recipe_sha256: str


@dataclass(frozen=True)
class Progress:
    """Where a run stands: the step, the tokens seen, the realized shares, the documents' digest.

    ``documents_sha256`` is the mixer's digest over the ids it handed out (``Mixer.digest``): a
    resumed run's state must reproduce it, not only the counts. ``tokens_emitted`` are the
    mixer's hand-out counts per component, ``tokens_consumed`` the inputs the model saw per
    source, summed over the steps.
    """

    step: int
    tokens_seen: int
    tokens_emitted: dict[str, int]
    tokens_consumed: dict[str, int]
    documents_sha256: str


def total_steps(cfg: VesperConfig) -> int:
    """Steps in the run: ``total_tokens`` over the batch's tokens, rounded up."""
    batch = cfg.training.chunk_tokens * cfg.training.batch_chunks
    return math.ceil(cfg.training.total_tokens / batch)


def phase_schedule(tokens_seen: int, step: int, cfg: VesperConfig) -> Phase:
    """The recipe at ``tokens_seen`` and ``step``, pure (spec: Phases; Optimizer and schedule).

    ``name`` by the fraction of ``total_tokens`` seen against the phase ends. ``t_max`` linear
    from ``t_max_anneal_from`` to ``model.t_max`` over the warm phase, rounded, never decreasing;
    the halting head is frozen in warm (``p = 1/t_max``) and free after; ``lambda_h`` is 0 in
    warm and ramps linearly to ``learning.lambda_h`` across the loop phase; the learning rate
    warms up linearly over ``warmup_steps`` and then follows one cosine from the peak to
    ``peak * final_lr_fraction`` at the last step. The retrieval, ledger, dusk and QAT switches
    carry the recipe's values and are not read by this version.

    Raises:
        ValueError: on a negative ``tokens_seen`` or ``step``.
    """
    if tokens_seen < 0 or step < 0:
        msg = f"tokens_seen and step must be at least 0, got {tokens_seen} and {step}"
        raise ValueError(msg)
    t, o, m = cfg.training, cfg.optimizer, cfg.model
    fraction = min(tokens_seen / t.total_tokens, 1.0)
    if fraction < t.phase_warm_end:
        name = "warm"
    elif fraction < t.phase_loop_end:
        name = "loop"
    elif fraction < t.phase_ledger_end:
        name = "ledger"
    else:
        name = "settle"
    if name == "warm":
        progress = fraction / t.phase_warm_end
        t_max = min(
            m.t_max, round(t.t_max_anneal_from + progress * (m.t_max - t.t_max_anneal_from))
        )
        lambda_h = 0.0
    else:
        t_max = m.t_max
        ramp = min((fraction - t.phase_warm_end) / (t.phase_loop_end - t.phase_warm_end), 1.0)
        lambda_h = cfg.learning.lambda_h * ramp
    steps = total_steps(cfg)
    if step < o.warmup_steps:
        lr = o.peak_lr * (step + 1) / o.warmup_steps
    else:
        # the last step of the run (index steps - 1) reaches the final fraction exactly
        decay_steps = max(steps - 1 - o.warmup_steps, 1)
        progress_lr = min((step - o.warmup_steps) / decay_steps, 1.0)
        final = o.peak_lr * o.final_lr_fraction
        lr = final + (o.peak_lr - final) * 0.5 * (1.0 + math.cos(math.pi * progress_lr))
    return Phase(
        name=name,
        t_max=max(t_max, 1),
        halting_frozen=name == "warm",
        lambda_h=lambda_h,
        lr=lr,
        retrieval=name != "warm",
        ledger_writes=name in {"ledger", "settle"},
        dusk_eta_fraction=0.0 if name in {"warm", "loop"} else cfg.dusk.eta_slow_fraction,
        qat=fraction >= t.qat_from,
    )


def decays(name: str) -> bool:
    """Whether the parameter ``name`` takes weight decay, by its role (spec: Optimizer).

    Raises:
        ValueError: for a name outside both lists; a new parameter gets a role, not a guess.
    """
    if any(name == role or name.endswith("." + role) for role in DECAYED):
        return True
    if any(name == role or name.endswith("." + role) for role in UNDECAYED):
        return False
    msg = f"parameter {name!r} has no weight-decay role; add it to DECAYED or UNDECAYED"
    raise ValueError(msg)


def optimizer(model: nn.Module, cfg: VesperConfig) -> torch.optim.AdamW:
    """AdamW with two groups by role: decayed linear maps, undecayed gates, norms and ``u``."""
    decayed = [p for name, p in model.named_parameters() if decays(name)]
    plain = [p for name, p in model.named_parameters() if not decays(name)]
    o = cfg.optimizer
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": o.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=o.peak_lr,
        betas=(o.beta1, o.beta2),
    )


def set_learning_rate(opt: torch.optim.AdamW, lr: float) -> None:
    """Write the phase's learning rate into every parameter group."""
    for group in cast("list[dict[str, object]]", opt.param_groups):
        group["lr"] = lr


def set_halting(model: Vesper, phase: Phase) -> None:
    """Freeze the halting head at ``p = 1/t_max`` in warm, or let it learn after.

    Frozen means the direction zero, the norm's gain one, the bias the logit of ``1/t_max`` and no
    updates; the frozen values are where the unfrozen head starts, so its first update moves a
    logit by at most ``lr * (d + 1)`` (docs/reviews/2026-10-15-halting-diagnosis.md). At one pass
    the loop halts by construction whatever the head says, so the head is held at the two-pass
    value (``p = 1/2``, logit 0): a saturated bias would survive the transition to a free head
    with a zero gradient and halt every token at the first pass forever (maintainer B's review of
    b797837).

    Raises:
        TypeError: when the model's halting head is not a ``HaltingHead``.
    """
    head = model.loop.halting
    if not isinstance(head, HaltingHead):
        msg = "the model's halting head is not a HaltingHead; the schedule cannot freeze it"
        raise TypeError(msg)
    if phase.halting_frozen:
        p = 1.0 / max(phase.t_max, 2)
        logit = math.log(p / (1.0 - p))
        with torch.no_grad():
            _ = head.w_h.zero_()
            _ = head.norm.weight.fill_(1.0)
            _ = head.b_h.fill_(logit)
    for parameter in head.parameters():
        _ = parameter.requires_grad_(not phase.halting_frozen)


def detach_states(states: Sequence[ColumnState]) -> list[ColumnState]:
    """The carried states without their graph: truncated backpropagation at the chunk."""
    return [map_state(state, _detached) for state in states]


def _detached(t: Tensor) -> Tensor:
    return t.detach()


def step(  # noqa: PLR0913
    model: Vesper,
    opt: torch.optim.AdamW,
    groups: Sequence[Group],
    phase: Phase,
    cfg: VesperConfig,
    device: torch.device,
    *,
    recompute: bool = False,
    slices: int = 1,
) -> StepResult | None:
    """One optimizer step over the next chunk of every group that still has one.

    Every group's chunk is drawn and its loss positions counted first; each group's objective
    (``wake_objective`` with the phase's weights) is scaled by its share of the batch's loss
    tokens before its backward pass, so the gradient is the token-weighted mean of the unsplit
    batch; the states are detached after the step and never reset; one clip, one step. The tail
    policy: a group whose stream has run dry drops out of the batch and the others train on
    (``groups_active`` says how many did); None only when every stream is dry, and then nothing
    was drawn, no state moved and the head was not touched (maintainer B's review of b797837: a draw
    from the live groups followed by a None return consumed their chunks without a step).

    Raises:
        NotImplementedError: for an unsupported learning mode, before anything is drawn or
            touched (maintainer B's review of 1725ca6: a direct step under local mode drew a
            document, replaced the group's states and zeroed the halting head before the objective
            refused).
        ValueError: when no drawn position carries a loss.
    """
    check_mode(cfg.learning.mode)  # the public boundary refuses first; the streams and states stay
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    drawn = [(group, chunk) for group in groups if (chunk := group.stream.next()) is not None]
    if not drawn:
        return None
    set_halting(model, phase)
    counts = [int(chunk.loss_mask.sum().item()) for _, chunk in drawn]
    total = sum(counts)
    if total == 0:
        msg = "no position in the batch carries a loss"
        raise ValueError(msg)
    weights = LossWeights(lambda_h=phase.lambda_h, lambda_g=cfg.learning.lambda_g)
    opt.zero_grad(set_to_none=True)
    loss_sum = 0.0
    ntp_sum = 0.0
    ponder_sum = 0.0
    seen = 0
    consumed: dict[str, int] = {}
    histogram: list[Tensor] = []
    for (group, chunk), count in zip(drawn, counts, strict=True):
        # one call per group, so its logits, outputs and graph are released before the next
        # group's forward (maintainer B's review of 444b966: the step held the previous group's
        # logits, y and per_pass at the next forward)
        loss, ntp, ponder, iters = _train_group(
            model,
            group,
            chunk,
            count / total,
            phase=phase,
            cfg=cfg,
            device=device,
            weights=weights,
            recompute=recompute,
            slices=slices,
        )
        loss_sum += loss
        ntp_sum += ntp
        ponder_sum += ponder
        if iters is not None:
            histogram.append(iters)
        seen += int(chunk.valid.sum().item())
        for slot_spans in chunk.spans:
            for source, inputs in slot_spans:
                consumed[source] = consumed.get(source, 0) + inputs
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(nn.utils.clip_grad_norm_(parameters, cfg.optimizer.grad_clip).item())
    set_learning_rate(opt, phase.lr)
    # the optimizer's step is untyped in torch; its result (a closure's loss) is not used here
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    return StepResult(
        loss=loss_sum,
        ntp=ntp_sum,
        ponder=ponder_sum,
        tokens_with_loss=total,
        tokens=seen,
        consumed=consumed,
        groups_active=len(drawn),
        histogram=iteration_histogram(histogram, cfg.model.t_max),
        grad_norm=grad_norm,
        seconds=time.monotonic() - started,
        peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    )


def _train_group(  # noqa: PLR0913
    model: Vesper,
    group: Group,
    chunk: Chunk,
    share: float,
    *,
    phase: Phase,
    cfg: VesperConfig,
    device: torch.device,
    weights: LossWeights,
    recompute: bool,
    slices: int,
) -> tuple[float, float, float, Tensor | None]:
    """One group's forward, objective scaled by ``share``, backward and state hand-off.

    Returns the share-weighted loss, next-token and ponder parts and the iteration counts of the
    loss positions (None for a chunk without a loss position, whose forward still advances the
    states). Every tensor of the forward dies with this call's frame.
    """
    tokens = chunk.tokens.to(device)
    pos = chunk.pos.to(device)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        result = model.run(tokens, pos, group.states, passes=phase.t_max, recompute=recompute)
        group.states = detach_states(result.states)
        if share == 0.0:
            return 0.0, 0.0, 0.0, None
        # the head over the loss positions in slices (float32 logits per slice), never [b, n, V]
        objective, parts = model.objective(chunk, result, cfg, weights, slices=slices)
    torch.autograd.backward([objective * share])
    return (
        float(objective.item()) * share,
        float(parts["ntp"].item()) * share,
        float(parts["ponder"].item()) * share,
        result.n_iters[chunk.loss_mask.to(device)].detach().cpu(),
    )


def _state_record(states: Sequence[ColumnState]) -> list[dict[str, object]]:
    return [
        {
            "h": list(state.h),
            "kv": [{"k": c.k, "v": c.v, "pos": c.pos, "window": c.window} for c in state.kv],
            "key_history": list(state.key_history),
        }
        for state in states
    ]


def _states_from_record(record: list[dict[str, object]], model: Vesper) -> list[ColumnState]:
    """The carried states of a checkpoint, each validated against the model's geometry.

    Raises:
        ValueError, ContractError: from ``check_column_state``, on a state the model cannot run.
    """
    out: list[ColumnState] = []
    for column in record:
        caches = cast("list[dict[str, object]]", column["kv"])
        state = ColumnState(
            h=tuple(cast("list[Tensor]", column["h"])),
            kv=tuple(
                KVCache(
                    k=cast("Tensor", c["k"]),
                    v=cast("Tensor", c["v"]),
                    pos=cast("Tensor", c["pos"]),
                    window=cast("int", c["window"]),
                )
                for c in caches
            ),
            key_history=tuple(cast("list[Tensor]", column["key_history"])),
        )
        check_column_state(state, model.shape, model.t_max)
        out.append(state)
    return out


def _stream_record(state: StreamState) -> dict[str, object]:
    return {
        "slots": [
            {
                "queued": [(d.doc_id, d.tokens, d.source) for d in slot.queued],
                "offset": slot.offset,
                "segment": slot.segment,
                "emitted": slot.emitted,
            }
            for slot in state.slots
        ],
        "chunk": state.chunk,
        "exhausted": state.exhausted,
        "documents_taken": state.documents_taken,
    }


def _stream_from_record(record: dict[str, object]) -> StreamState:
    slots: list[SlotState] = []
    for slot in cast("list[dict[str, object]]", record["slots"]):
        queued = tuple(
            Document(doc_id=doc_id, tokens=tokens, source=source)
            for doc_id, tokens, source in cast("list[tuple[str, Tensor, str]]", slot["queued"])
        )
        slots.append(
            SlotState(
                queued=queued,
                offset=cast("int", slot["offset"]),
                segment=cast("int", slot["segment"]),
                emitted=cast("int", slot["emitted"]),
            )
        )
    return StreamState(
        slots=tuple(slots),
        chunk=cast("int", record["chunk"]),
        exhausted=cast("bool", record["exhausted"]),
        documents_taken=cast("int", record["documents_taken"]),
    )


def checkpoint(  # noqa: PLR0913
    path: Path,
    model: Vesper,
    opt: torch.optim.AdamW,
    groups: Sequence[Group],
    progress: Progress,
    identity: Identity,
    recipe: Recipe,
    *,
    reason: str,
    mixer: MixerState | None = None,
) -> Path:
    """Write ``model.pt``, ``optimizer.pt``, ``states.pt``, ``train.json`` and ``manifest.json``.

    ``states.pt`` holds every group's carried states (detached), its stream's packing state and
    the mixer's state (the readers' cursors, balances and digest: what a resume continues from
    at bounded cost; None only in unit tests that step over plain iterators, and such a
    checkpoint cannot be resumed by ``run``); ``train.json`` the progress, the identity, the
    recipe, the layout, the reason (``interval``, ``budget``, ``steps`` or ``exhausted``) and the
    label; ``manifest.json`` the sha256 of each file. The directory is staged and published by
    one rename (``publish.staged``), so a failed save leaves nothing at ``path`` and a retry is
    not refused. Returns the manifest's path.

    Raises:
        ValueError: when ``path`` already holds a checkpoint (they are immutable).
    """
    if path.exists():
        msg = f"{path}: a checkpoint already exists there"
        raise ValueError(msg)
    streams = [g.stream.state() for g in groups]
    layout = Layout(groups=len(groups), slots=len(streams[0].slots) if streams else 0)
    with staged(path) as partial:
        partial.mkdir(parents=True)
        torch.save(cast("dict[str, Tensor]", model.state_dict()), partial / "model.pt")
        torch.save(cast("dict[str, object]", opt.state_dict()), partial / "optimizer.pt")
        torch.save(
            {
                "groups": [
                    {"states": _state_record(g.states), "stream": _stream_record(state)}
                    for g, state in zip(groups, streams, strict=True)
                ],
                "mixer": None if mixer is None else _plain(mixer),
            },
            partial / "states.pt",
        )
        record: dict[str, object] = {
            "progress": _plain(progress),
            "identity": _plain(identity),
            "recipe": _plain(recipe),
            "layout": _plain(layout),
            "reason": reason,
            "label": LABEL,
            "written_at_utc": datetime.now(UTC).isoformat(),
        }
        write_json(partial / "train.json", record)
        manifest = {name: file_sha256(partial / name) for name in FILES}
        write_json(partial / "manifest.json", dict(manifest))
    return path / "manifest.json"


@dataclass(frozen=True)
class Restored:
    """What a checkpoint gives back.

    The groups' states, their streams' states, the mixer's state (None when the checkpoint was
    written without one) and the progress.
    """

    states: list[list[ColumnState]]
    streams: list[StreamState]
    mixer: MixerState | None
    progress: Progress


def restore(
    path: Path, model: Vesper, opt: torch.optim.AdamW, identity: Identity, layout: Layout
) -> Restored:
    """Verify every hash, check the identity and the layout, load with ``weights_only=True``.

    Raises:
        ValueError: on a missing or changed file, a checkpoint from a run with another
            tokenizer, config, finalization or recipe, or one whose groups and slots differ from
            ``layout`` (regrouping logical slots is not supported; the run refuses before it
            trains, so the layout it logs is the one it executes).
    """
    record = _checked_record(path)  # one snapshot of train.json, its schema checked first
    manifest = cast("dict[str, str]", json.loads((path / "manifest.json").read_text()))
    for name in FILES:
        file = path / name
        if not file.is_file() or file_sha256(file) != manifest.get(name):
            msg = f"{file}: missing or its hash differs from the manifest"
            raise ValueError(msg)
    saved = cast("dict[str, object]", record["identity"])
    if saved != _plain(identity):
        msg = f"{path}: checkpoint of a run with another tokenizer, config, finalization or recipe"
        raise ValueError(msg)
    if cast("dict[str, object]", record["layout"]) != _plain(layout):
        msg = (
            f"{path}: checkpoint of {record['layout']} groups and slots; this run asks for"
            f" {_plain(layout)}; regrouping is not supported"
        )
        raise ValueError(msg)
    loaded = cast("dict[str, Tensor]", torch.load(path / "model.pt", weights_only=True))
    _ = cast("object", model.load_state_dict(loaded))
    opt.load_state_dict(
        cast("dict[str, object]", torch.load(path / "optimizer.pt", weights_only=True))
    )
    saved_states = cast("dict[str, object]", torch.load(path / "states.pt", weights_only=True))
    groups = cast("list[dict[str, object]]", saved_states["groups"])
    mixer_record = cast("dict[str, object] | None", saved_states.get("mixer"))
    progress_record = cast("dict[str, object]", record["progress"])
    progress = Progress(
        step=cast("int", progress_record["step"]),
        tokens_seen=cast("int", progress_record["tokens_seen"]),
        tokens_emitted=cast("dict[str, int]", progress_record["tokens_emitted"]),
        tokens_consumed=cast("dict[str, int]", progress_record["tokens_consumed"]),
        documents_sha256=cast("str", progress_record["documents_sha256"]),
    )
    return Restored(
        states=[
            _states_from_record(cast("list[dict[str, object]]", g["states"]), model) for g in groups
        ],
        streams=[_stream_from_record(cast("dict[str, object]", g["stream"])) for g in groups],
        mixer=None if mixer_record is None else _mixer_from_record(mixer_record),
        progress=progress,
    )


def _mixer_from_record(record: dict[str, object]) -> MixerState:
    return MixerState(
        cursors=tuple(
            (int(pair), int(row)) for pair, row in cast("list[tuple[int, int]]", record["cursors"])
        ),
        active=tuple(int(i) for i in cast("list[int]", record["active"])),
        emitted=dict(cast("dict[str, int]", record["emitted"])),
        pulled=cast("int", record["pulled"]),
        digest=cast("str", record["digest"]),
    )


def _checked_record(path: Path) -> dict[str, object]:
    """The checkpoint's ``train.json`` as one validated snapshot.

    Hashed and parsed from the same bytes, its schema checked before any field is read (maintainer
    B's review of f55bfb3: a hash-then-reread let a replacement land between the two, and a schema-3
    record reached a KeyError first). Returns the record with ``recipe`` known to be a mapping of
    the current schema.

    Raises:
        ValueError: when the manifest or the record is missing, the hash differs, the record is
            not JSON or not a mapping with a mapping ``recipe``, or the schema is another one.
    """
    manifest_path = path / "manifest.json"
    file = path / "train.json"
    if not manifest_path.is_file() or not file.is_file():
        msg = f"{path}: not a checkpoint (manifest.json and train.json expected)"
        raise ValueError(msg)
    manifest = _mapping(
        cast("object", json.loads(manifest_path.read_bytes())), f"{manifest_path}: malformed"
    )
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest.get("train.json"):
        msg = f"{file}: missing or its hash differs from the manifest"
        raise ValueError(msg)
    try:
        parsed = cast("object", json.loads(data))
    except ValueError as exc:
        msg = f"{file}: malformed record: {exc}"
        raise ValueError(msg) from exc
    record = _mapping(parsed, f"{file}: malformed record")
    recipe = _mapping(record.get("recipe"), f"{file}: malformed record: no recipe mapping")
    saved_schema = recipe.get("schema")
    if saved_schema != CHECKPOINT_SCHEMA:
        msg = (
            f"{path}: checkpoint schema {saved_schema!r}, this code reads {CHECKPOINT_SCHEMA};"
            " no conversion exists, the run is a fork from this checkpoint's own code"
        )
        raise ValueError(msg)
    return record


def _mapping(value: object, what: str) -> dict[str, object]:
    """``value`` as a JSON mapping, or a ``ValueError`` naming ``what`` is malformed."""
    if not isinstance(value, dict):
        msg = f"{what}: a mapping was expected"
        raise ValueError(msg)  # noqa: TRY004  # a malformed file is a value error, as everywhere here
    return cast("dict[str, object]", value)


def resumed_components(data: Path, resume: Path) -> list[Component]:
    """The components a resumed run reads: the checkpoint's own pins resolved under the data root.

    The selection chooses the inputs of a new run; a resume resolves the exact references its
    checkpoint stored, whatever the selection says now, and never reads or rewrites it
    (generation contract, amendment 1; maintainer B's review of 1725ca6: a switch of the active web
    generation made a valid checkpoint unresumable through the command line). Component order
    and shares come from the stored recipe, each pin is validated by ``generations.pinned`` and
    the component's files are the generation's inputs in manifest order.

    Raises:
        ValueError: when the checkpoint's record is missing, tampered or inconsistent (a pin
            whose kind is not its component's, counts that differ), a component carries no pin
            (a legacy-layout run resumes under ``--legacy-layout --web --mix``), a pin is
            malformed, or a pinned generation is missing, changed or re-hashed.
    """
    record = _checked_record(resume)
    recipe = cast("dict[str, object]", record["recipe"])  # a mapping: _checked_record said so
    components = recipe.get("components")
    pins = recipe.get("generations")
    if not isinstance(components, list) or not isinstance(pins, list):
        msg = f"{resume}: malformed recipe: components and generations must be lists"
        raise ValueError(msg)  # noqa: TRY004  # a malformed record is a value error, as everywhere here
    entries = cast("list[object]", components)
    raw_pins = cast("list[object]", pins)
    if len(entries) != len(raw_pins):
        msg = f"{resume}: inconsistent recipe: {len(entries)} components, {len(raw_pins)} pins"
        raise ValueError(msg)
    out: list[Component] = []
    for raw_entry, raw_pin in zip(entries, raw_pins, strict=True):
        if (
            not isinstance(raw_entry, list)
            or len(cast("list[object]", raw_entry)) != 2  # noqa: PLR2004
            or not isinstance(cast("list[object]", raw_entry)[0], str)
            or not isinstance(cast("list[object]", raw_entry)[1], (int, float))
            or not (raw_pin is None or isinstance(raw_pin, str))
        ):
            msg = f"{resume}: malformed recipe: a component is [name, share] with a pin or null"
            raise ValueError(msg)
        entry = cast("list[object]", raw_entry)
        name, share = cast("str", entry[0]), float(cast("int | float", entry[1]))
        pin = raw_pin
        if pin is None:
            msg = (
                f"{resume}: component {name!r} carries no generation pin; a legacy-layout run"
                " resumes with --legacy-layout --web --mix"
            )
            raise ValueError(msg)
        ref = parse_pin(pin)
        if ref.kind != name:
            msg = (
                f"{resume}: inconsistent recipe: component {name!r} pins a {ref.kind!r} generation"
            )
            raise ValueError(msg)
        generation = pinned(data, ref)
        out.append(Component(name, generation.inputs, share, generation=ref.pin))
    return out


def recipe_of(parts: Sequence[Component], seed: int, cfg: VesperConfig) -> Recipe:
    """The run's data recipe.

    The seed, the ordered components with their shares and generation pins, the split, the rules.
    """
    return Recipe(
        seed=seed,
        components=tuple((c.name, c.share) for c in parts),
        generations=tuple(c.generation for c in parts),
        held_out_fraction=cfg.training.held_out_fraction,
        finalize_rule=FINALIZE_RULE,
        split_rule=SPLIT_RULE,
        schema=CHECKPOINT_SCHEMA,
    )


def identity_of(
    tokenizer_path: Path, config_path: Path, finalization_receipt: Path, recipe: Recipe
) -> Identity:
    """The hashes of the three files a run reads and of its recipe."""
    canonical = json.dumps(_plain(recipe), sort_keys=True, separators=(",", ":"))
    return Identity(
        tokenizer_sha256=file_sha256(tokenizer_path),
        config_sha256=file_sha256(config_path),
        finalization_sha256=file_sha256(finalization_receipt),
        recipe_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def held_out_chunks(
    parts: Sequence[Component],
    seed: int,
    tokenizer: Tokenizer,
    split: Split,
    micro: int,
    cfg: VesperConfig,
) -> list[Chunk]:
    """The first ``training.eval_chunks`` chunks of a held-out stream of ``micro`` slots."""
    documents = iter_mixed_documents(parts, seed, tokenizer, split)
    stream = Stream(documents, slots=micro, chunk=cfg.training.chunk_tokens)
    chunks: list[Chunk] = []
    while len(chunks) < cfg.training.eval_chunks and (chunk := stream.next()) is not None:
        chunks.append(chunk)
    return chunks


def _plain(
    value: Progress | Identity | StepResult | Recipe | Layout | MixerState,
) -> dict[str, object]:
    # through JSON and back, so tuples become lists and the result compares equal to a record
    # read from train.json
    plain = cast("dict[str, object]", asdict(value))
    return cast("dict[str, object]", json.loads(json.dumps(plain)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesper.train", description=(__doc__ or "").split("\n")[0]
    )
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--data", type=Path, default=None, help="the data root (selection)")
    _ = parser.add_argument(
        "--legacy-layout", action="store_true", help="read today's flat --web and --mix dirs"
    )
    _ = parser.add_argument("--web", type=Path, default=None)
    _ = parser.add_argument("--mix", type=Path, default=None)
    _ = parser.add_argument("--finalization", type=Path, required=True)
    _ = parser.add_argument("--tokenizer", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--micro-chunks", type=int, required=True)
    _ = parser.add_argument("--steps", type=int, default=None)
    _ = parser.add_argument("--resume", type=Path, default=None)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    _ = parser.add_argument(
        "--recompute",
        action="store_true",
        help="recompute column activations in the backward pass: same gradients, far less memory",
    )
    _ = parser.add_argument(
        "--loss-slices",
        type=int,
        default=1,
        help="compute the head and the loss over this many position slices: same objective",
    )
    return parser


def run(  # noqa: C901, PLR0913, PLR0915
    cfg: VesperConfig,
    *,
    config_path: Path,
    parts: Sequence[Component],
    finalization_receipt: Path,
    tokenizer_path: Path,
    out: Path,
    micro: int,
    steps: int | None,
    resume: Path | None,
    seed: int,
    device: torch.device,
    recompute: bool = False,
    slices: int = 1,
) -> Path:
    """The run: streams, schedule, steps, log lines, checkpoints with evaluations.

    The run ends when the token budget is reached, the optional step cap is hit or every stream
    is dry (the ``end`` line's ``reason``: ``budget``, ``steps``, ``exhausted``); the last
    completed step is checkpointed exactly once, with that reason, unless an interval checkpoint
    already holds it. Every line carries the ``run_id`` of its header; a fresh run never appends
    to a log that holds one (a resume does, with ``resumed_from``). Returns the log's path.

    Raises:
        ValueError: when ``batch_chunks`` is not a multiple of ``micro``, the finalization does
            not cover the components, ``out`` already holds a run and this is not a resume, the
            checkpoint has another identity or layout, or its replay does not reproduce the
            checkpoint's documents (their digest and the realized shares).
    """
    t = cfg.training
    check_mode(cfg.learning.mode)  # an unsupported mode is refused before anything is read
    if micro < 1 or t.batch_chunks % micro != 0:
        msg = f"micro_chunks must divide batch_chunks ({t.batch_chunks}), got {micro}"
        raise ValueError(msg)
    finalization = load_finalization(finalization_receipt)
    check_inputs(finalization, parts)
    recipe = recipe_of(parts, seed, cfg)
    identity = identity_of(tokenizer_path, config_path, finalization_receipt, recipe)
    layout = Layout(groups=t.batch_chunks // micro, slots=micro)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    if resume is None and log_path.exists():
        msg = f"{log_path}: already holds a run; resume one of its checkpoints or use another out"
        raise ValueError(msg)
    tokenizer = Tokenizer.load(tokenizer_path)
    split = Split(finalization, t.held_out_fraction)
    held = Split(finalization, t.held_out_fraction, held_out=True)
    # every random choice goes through an explicit generator: the model's initialization here,
    # the corpus order through the seed; nothing reads torch's global generator
    model = Vesper(cfg, torch.Generator().manual_seed(seed)).to(device)
    opt = optimizer(model, cfg)
    counts: dict[str, int] = {}
    mixer = Mixer(parts, seed, tokenizer, split, tokens_emitted=counts)
    progress = Progress(
        step=0,
        tokens_seen=0,
        tokens_emitted={},
        tokens_consumed={},
        documents_sha256=mixer.digest(),
    )
    if resume is None:
        documents = mixer.documents()
        groups = [
            Group(
                Stream(documents, slots=micro, chunk=t.chunk_tokens),
                model.init_states(micro, torch.float32, device),
            )
            for _ in range(layout.groups)
        ]
    else:
        restored = restore(resume, model, opt, identity, layout)
        progress = restored.progress
        if restored.mixer is None:
            msg = f"{resume}: the checkpoint holds no mixer state; it cannot be resumed"
            raise ValueError(msg)
        # bounded cost: the readers seek to their cursors, the digest continues from its state;
        # the three records of the same facts (mixer state, progress, streams) must agree
        mixer = Mixer.restore(parts, seed, tokenizer, split, restored.mixer, tokens_emitted=counts)
        taken = sum(s.documents_taken for s in restored.streams)
        if (
            mixer.digest() != progress.documents_sha256
            or counts != progress.tokens_emitted
            or mixer.pulled != taken
        ):
            msg = f"{resume}: the mixer state, the progress and the streams disagree"
            raise ValueError(msg)
        documents = mixer.documents()
        groups = [
            Group(Stream.restore(stream_state, documents), _to_device(states, device))
            for stream_state, states in zip(restored.streams, restored.states, strict=True)
        ]
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{os.getpid()}"
    emit = _emitter(log_path, run_id)
    emit(
        {
            "kind": "run",
            "label": LABEL,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "identity": _plain(identity),
            "recipe": _plain(recipe),
            "layout": _plain(layout),
            "config": str(config_path),
            "total_steps": total_steps(cfg),
            "micro_chunks": micro,
            "recompute": recompute,
            "loss_slices": slices,
            "device": str(device),
            "resumed_from": str(resume) if resume else None,
            "environment": environment(),
        }
    )
    every = max(t.total_tokens // t.checkpoints_per_run, 1)
    next_checkpoint = (progress.tokens_seen // every + 1) * every
    step_index = progress.step
    tokens_seen = progress.tokens_seen
    step_cap = None if steps is None else step_index + steps
    checkpointed_at = step_index
    active = layout.groups
    consumed: dict[str, int] = dict(progress.tokens_consumed)
    histogram: list[float] = []
    training_positions = 0
    training_passes = 0
    phase = phase_schedule(tokens_seen, step_index, cfg)

    def save(reason: str) -> None:
        nonlocal checkpointed_at
        current = Progress(
            step=step_index,
            tokens_seen=tokens_seen,
            tokens_emitted=dict(counts),
            tokens_consumed=dict(consumed),
            documents_sha256=mixer.digest(),
        )
        where = out / f"step-{step_index}"
        _ = checkpoint(
            where, model, opt, groups, current, identity, recipe, reason=reason, mixer=mixer.state()
        )
        checkpointed_at = step_index
        chunks = held_out_chunks(parts, seed, tokenizer, held, micro, cfg)
        evaluation = evaluate(model, chunks, cfg, passes=phase.t_max)
        emit(
            {
                "kind": "checkpoint",
                "step": step_index,
                "tokens_seen": tokens_seen,
                "reason": reason,
                "path": str(where),
                "held_out_perplexity": evaluation.perplexity,
                "held_out_loss_sum": evaluation.loss_sum,
                "held_out_tokens_with_loss": evaluation.tokens_with_loss,
                "held_out_iteration_counts": evaluation.iteration_counts,
                "held_out_iteration_histogram": evaluation.iteration_histogram,
                "held_out_measurement": "checkpoint-post-update",
                "held_out_precision": "float32",
                "training_iteration_histogram": histogram,
                "training_tokens_with_loss": training_positions,
                "training_measurement": "last-step-pre-update",
                "held_out_t_max": phase.t_max,
                "training_t_max": training_passes,
            }
        )

    def stopped() -> str | None:
        if tokens_seen >= t.total_tokens:
            return "budget"
        if step_cap is not None and step_index >= step_cap:
            return "steps"
        return None

    while (reason := stopped()) is None:
        phase = phase_schedule(tokens_seen, step_index, cfg)
        result = step(model, opt, groups, phase, cfg, device, recompute=recompute, slices=slices)
        if result is None:
            reason = "exhausted"
            break
        step_index += 1
        tokens_seen += result.tokens
        histogram = result.histogram
        training_positions = result.tokens_with_loss
        training_passes = phase.t_max
        if result.groups_active < active:
            active = result.groups_active
            emit(
                {
                    "kind": "exhausted",
                    "step": step_index,
                    "tokens_seen": tokens_seen,
                    "groups_active": active,
                }
            )
        line: dict[str, object] = {"kind": "step", "step": step_index}
        line.update(_plain(result))
        # after the result, so that no field of the step's own record can shadow the run's count
        line.update(
            {
                "tokens_seen": tokens_seen,
                "phase": phase.name,
                "t_max": phase.t_max,
                "lr": phase.lr,
                "lambda_h": phase.lambda_h,
            }
        )
        for source, inputs in result.consumed.items():
            consumed[source] = consumed.get(source, 0) + inputs
        line["shares"] = dict(counts)  # tokens the mixer handed to the streams
        line["consumed"] = dict(consumed)  # inputs the model saw, per source, over the run
        emit(line)
        if tokens_seen >= next_checkpoint:
            next_checkpoint += every
            save("interval")
    if step_index != checkpointed_at:
        save(reason)
    emit({"kind": "end", "step": step_index, "tokens_seen": tokens_seen, "reason": reason})
    return log_path


def _emitter(log_path: Path, run_id: str) -> Callable[[dict[str, object]], None]:
    """Append log lines, each stamped with the run id that links it to its header."""

    def emit(line: dict[str, object]) -> None:
        line["run_id"] = run_id
        _append(log_path, line)

    return emit


def _to_device(states: Sequence[ColumnState], device: torch.device) -> list[ColumnState]:
    return [map_state(state, lambda t: t.to(device)) for state in states]


def _append(path: Path, line: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        _ = handle.write(json.dumps(line) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: 0 when the run reached its last step or its data, 1 on an error."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    config_path = cast("Path", args.config)
    cfg = load(config_path)
    resume = cast("Path | None", args.resume)
    data = cast("Path | None", args.data)
    legacy_layout = cast("bool", args.legacy_layout)
    web = cast("Path | None", args.web)
    mix = cast("Path | None", args.mix)
    try:
        if (
            resume is not None
            and data is not None
            and not legacy_layout
            and web is None
            and mix is None
        ):
            parts = resumed_components(data, resume)  # the checkpoint's pins, not today's selection
        else:
            parts = components_for(data=data, legacy_layout=legacy_layout, web=web, mix=mix)
        log_path = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=cast("Path", args.finalization),
            tokenizer_path=cast("Path", args.tokenizer),
            out=cast("Path", args.out),
            micro=cast("int", args.micro_chunks),
            recompute=cast("bool", args.recompute),
            slices=cast("int", args.loss_slices),
            steps=cast("int | None", args.steps),
            resume=resume,
            seed=cast("int", args.seed),
            device=torch.device(cast("str", args.device)),
        )
    except (RuntimeError, ValueError, TypeError):
        log.exception("stopped")
        return 1
    log.info("log at %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
