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
import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple, cast

import torch
from torch import Tensor, nn

from vesper.attention import KVCache, RingKVCache
from vesper.column import (
    SCAN_IMPLS,
    ColumnState,
    check_capturable_scan,
    check_column_state,
    map_state,
    reset_recurrent,
    scan_of,
    set_scan,
    state_tensors,
    with_state_tensors,
)
from vesper.config import VesperConfig, load, parse_bytes
from vesper.corpus import (
    CYCLE_ORDER_RULE,
    FINALIZE_RULE,
    SPLIT_RULE,
    Component,
    Finalization,
    Mixer,
    MixerState,
    Split,
    check_inputs,
    iter_mixed_documents,
    load_finalization,
)
from vesper.data import Chunk, Document, SlotState, Stream, StreamState
from vesper.evals import EvalGraphs, evaluate, iteration_histogram, static_profile
from vesper.generations import parse_pin, pinned
from vesper.graphs import (
    WARM_STEPS,
    SignatureGraph,
    memory_peak,
    reset_memory_peak,
    signature,
)
from vesper.keys import file_sha256, hashed_bytes, verified_bytes
from vesper.learning import LossWeights, check_mode, warmup_cosine
from vesper.loop import DenseRule, HaltingHead
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.publish import staged
from vesper.sample import components_for
from vesper.tokenizer import Tokenizer

LABEL = (
    "e2e wake training; no retrieval, no ledger, no dusk: the recipe through phase 1, the"
    " no-retrieval ablation beyond"
)
# the training objective: "act" the recipe's (wake_objective, the ACT mixture); "fixed" every
# pass supervised with equal weight on a dense forward (all_pass_objective, stage 1's condition
# B), the depth-first phase of the real-data halting pilot
# (docs/reviews/2026-10-20-real-data-halting-pilot.md). Part of a run's recipe: a resume keeps it.
OBJECTIVES = ("act", "fixed")
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
# states.pt; 8 the training objective in the recipe (maintainer B 0617971: part of the run's
# identity)
CHECKPOINT_SCHEMA = 8
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
    """One micro-batch: its stream of ``micro_chunks`` slots and their carried states.

    ``draw_ahead`` draws the group's next chunk early: ``step`` calls it (through its ``ahead``
    argument) while the device runs the current step, so the host's drawing overlaps the GPU's
    work instead of leaving the card idle (measured 2026-10-21 on the RTX 5090 with real data:
    drawing took 42 of a 274 ms step). ``next`` hands that chunk out before drawing anew, so the
    chunks and their order are the stream's own; ``stream_state`` is the stream as of the last
    chunk handed out, so a checkpoint never records a chunk that was drawn but not trained.
    """

    stream: Stream
    states: list[ColumnState]
    _ahead: list[Chunk | None] = dataclasses.field(default_factory=list, init=False)
    _before: StreamState | None = dataclasses.field(default=None, init=False)

    def next(self) -> Chunk | None:
        """The chunk drawn ahead if there is one, else the stream's next."""
        if self._ahead:
            self._before = None
            return self._ahead.pop()
        return self.stream.next()

    def draw_ahead(self) -> None:
        """Draw the next chunk now and keep the stream's state from before it.

        Raises:
            ValueError: when a chunk is already drawn ahead (one at most).
        """
        if self._ahead:
            msg = "a chunk is already drawn ahead"
            raise ValueError(msg)
        self._before = self.stream.state()
        self._ahead.append(self.stream.next())

    def stream_state(self) -> StreamState:
        """The stream's state as of the last chunk handed out (what a checkpoint records)."""
        return self.stream.state() if self._before is None else self._before


@dataclass(frozen=True)
class StepResult:
    """What one optimizer step did; ``tokens`` is the step's own count, not the run's.

    ``consumed`` counts the step's input tokens per source as the streams emitted them (the
    chunks' ``spans``), which is what the model saw; the mixer's hand-out counts are not that.
    ``peak_reserved_bytes`` is ``graphs.memory_peak`` over the step: what the allocator held
    from the card, the captured graph's pool, the carried states of every slot and the optimizer
    included (0 on the CPU, which has no counter).
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
    peak_reserved_bytes: int
    # the "fixed" objective's next-token loss of each pass, token-weighted over the batch (None
    # under "act"): what depth-first training watches, whether later passes improve
    pass_ntp: list[float] | None = None


@dataclass(frozen=True)
class Recipe:
    """What decides the order and membership of the data a run consumes, beyond its three files.

    The seed, the ordered components with their shares, the held-out fraction, the finalize and
    split rules, the checkpoint schema and the training objective (``OBJECTIVES``: a resume under
    another objective is another run). Its sha256 is part of the ``Identity`` and the plain
    record goes into the run header and ``train.json``: a run under another recipe is a labeled
    fork, never a continuation (maintainer B's review of b797837: another seed from the same
    checkpoint passed a count check over equal-length documents and diverged).

    ``component_passes`` is "finite" (a component that runs out leaves the mix) or the cycling
    rule (``corpus.CYCLE_ORDER_RULE``: it starts again in a new order). The finite value is left
    out of the plain record (``_plain``), so a finite recipe's record and hash are the ones
    written before the field existed and every finite checkpoint stays resumable as it was.
    """

    seed: int
    components: tuple[tuple[str, float], ...]
    generations: tuple[str | None, ...]
    held_out_fraction: float
    finalize_rule: str
    split_rule: str
    schema: int
    objective: str
    component_passes: str = "finite"


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
    lr = warmup_cosine(
        step,
        total_steps(cfg),
        peak=o.peak_lr,
        warmup=o.warmup_steps,
        final_fraction=o.final_lr_fraction,
    )
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


def optimizer(
    model: nn.Module, cfg: VesperConfig, *, capturable: bool = False
) -> torch.optim.AdamW:
    """AdamW with two groups by role: decayed linear maps, undecayed gates, norms and ``u``.

    ``capturable`` builds the variant a CUDA graph can capture: its step keeps every count on
    the device, and the learning rate is a device tensor shared by the groups, so a captured step
    reads the rate ``set_learning_rate`` writes into it before each replay (a Python float would
    be frozen into the graph at capture). Every parameter must then be on one CUDA device.

    Raises:
        ValueError: if ``capturable`` and a parameter is not on CUDA.
    """
    decayed = [p for name, p in model.named_parameters() if decays(name)]
    plain = [p for name, p in model.named_parameters() if not decays(name)]
    o = cfg.optimizer
    groups = [
        {"params": decayed, "weight_decay": o.weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]
    if not capturable:
        return torch.optim.AdamW(groups, lr=o.peak_lr, betas=(o.beta1, o.beta2))
    devices = {p.device for p in model.parameters()}
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        msg = f"a capturable optimizer needs every parameter on one CUDA device, got {devices}"
        raise ValueError(msg)
    lr = torch.tensor(o.peak_lr, device=next(iter(devices)))
    return torch.optim.AdamW(groups, lr=lr, betas=(o.beta1, o.beta2), capturable=True)


def set_learning_rate(opt: torch.optim.AdamW, lr: float) -> None:
    """Write the phase's learning rate into every parameter group.

    A capturable optimizer's rate is a device tensor that a captured step reads: it is filled in
    place, never replaced.
    """
    for group in cast("list[dict[str, object]]", opt.param_groups):
        current = group["lr"]
        if isinstance(current, float):
            group["lr"] = lr
        else:  # the capturable optimizer's device tensor (``optimizer``)
            _ = cast("Tensor", current).fill_(lr)


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
    objective: str = "act",
    graphs: "StepGraphs | None" = None,
    ahead: Callable[[], None] | None = None,
) -> StepResult | None:
    """One optimizer step over the next chunk of every group that still has one.

    ``ahead``, when given, is called once the step is launched on the device and before its
    record is read: the run draws the next step's chunks there (``Group.draw_ahead``), on the
    host while the device works, and ``Group.next`` hands them out to the next step.

    On CUDA the whole step (every group's forward, objective and backward, the clip and the
    optimizer's step) is ONE captured CUDA graph (``graphs``; the project owner 2026-10-21: CUDA
    graphs are a hard requirement for anything on the GPU, so a CUDA step without ``graphs`` is
    refused); the step reads nothing back until it has run, then its record in one transfer. On the
    CPU the same device step runs directly. The model's optimizer must be the capturable one on
    CUDA (``optimizer(..., capturable=True)``).

    ``objective`` (``OBJECTIVES``): "act" trains ``wake_objective``; "fixed" runs every token
    through every pass of the phase (``DenseRule``) and trains ``all_pass_objective`` with equal
    pass weights, so the halting head gets no gradient and the ponder part is zero.

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
        ValueError: for an unknown objective, before anything is drawn; when no drawn position
            carries a loss.
    """
    check_mode(cfg.learning.mode)  # the public boundary refuses first; the streams and states stay
    _check_objective(objective)
    cuda = device.type == "cuda"
    if cuda and graphs is None:
        msg = "a CUDA step runs as a captured CUDA graph: pass StepGraphs (no eager GPU step)"
        raise ValueError(msg)
    if cuda and scan_of(model) == "reference":
        msg = "the reference scan cannot run as a captured CUDA graph; set a fused scan"
        raise ValueError(msg)
    started = time.monotonic()
    reset_memory_peak(device)
    drawn = [
        (index, group, chunk)
        for index, group in enumerate(groups)
        if (chunk := group.next()) is not None
    ]
    if not drawn:
        return None
    set_halting(model, phase)
    # the chunks are still on the CPU: counting there costs no device synchronization
    counts = [int(chunk.loss_mask.sum()) for _, _, chunk in drawn]
    total = sum(counts)
    if total == 0:
        msg = "no position in the batch carries a loss"
        raise ValueError(msg)
    set_learning_rate(opt, phase.lr)  # a capturable optimizer's rate is filled in place
    sizes = [group.stream.slots for group in groups]  # every group's, so a dry one keeps its place
    plan = _StepPlan(
        has_loss=tuple(count > 0 for count in counts),
        t_max=phase.t_max,
        halting_frozen=phase.halting_frozen,
        fixed=objective == "fixed",
        recompute=recompute,
        slices=slices,
        templates=tuple(tuple(group.states) for _, group, _ in drawn),
        chunks=tuple(chunk for _, _, chunk in drawn),
        reset_every=cfg.training.state_reset_chunks,
        first_slots=tuple(sum(sizes[:index]) for index, _, _ in drawn),
        total_slots=sum(sizes),
    )
    values: list[Tensor] = []
    for _, group, chunk in drawn:
        values += [t.to(device) for t in _chunk_tensors(chunk)]
        values += state_tensors(group.states)
    shares: list[float] = [c / total for c in counts]
    values.append(torch.tensor(shares, dtype=torch.float32, device=device))
    if graphs is None:
        out: Sequence[Tensor] = _device_step(model, opt, cfg, plan, phase.lambda_h, values)
    else:
        lambda_h = graphs.lambda_h(device)
        _ = lambda_h.fill_(phase.lambda_h)
        out = graphs(
            plan.key(values),
            lambda v: _device_step(model, opt, cfg, plan, lambda_h, v),
            values,
        )
    if ahead is not None:
        ahead()  # on the host while the device runs the launched step
    read = _read_step(out, plan)
    histogram = [
        n_iters[chunk.loss_mask].to(torch.int32)
        for n_iters, chunk in zip(read.n_iters, plan.chunks, strict=True)
        if bool(chunk.loss_mask.any())
    ]
    for (_, group, _), states in zip(drawn, read.states, strict=True):
        group.states = with_state_tensors(group.states, states)
    loss, ntp, ponder, grad_norm = read.scalars
    seen = sum(int(chunk.valid.sum()) for _, _, chunk in drawn)
    consumed = _consumed(plan.chunks)
    return StepResult(
        loss=loss,
        ntp=ntp,
        ponder=ponder,
        tokens_with_loss=total,
        tokens=seen,
        consumed=consumed,
        groups_active=len(drawn),
        histogram=iteration_histogram(histogram, cfg.model.t_max),
        grad_norm=grad_norm,
        seconds=time.monotonic() - started,
        peak_reserved_bytes=memory_peak(device),
        pass_ntp=read.pass_ntp if plan.fixed else None,
    )


def _consumed(chunks: Sequence[Chunk]) -> dict[str, int]:
    """The input tokens per source the chunks fed the model (their ``spans``)."""
    consumed: dict[str, int] = {}
    for chunk in chunks:
        for slot_spans in chunk.spans:
            for source, inputs in slot_spans:
                consumed[source] = consumed.get(source, 0) + inputs
    return consumed


_CHUNK_FIELDS = 7  # tokens, targets, valid, doc_start, segment, loss_mask, pos


def _chunk_tensors(chunk: Chunk) -> list[Tensor]:
    """A chunk's ``_CHUNK_FIELDS`` tensors, in the order ``_chunk_with`` takes them back."""
    c = chunk
    return [c.tokens, c.targets, c.valid, c.doc_start, c.segment, c.loss_mask, c.pos]


def _chunk_with(chunk: Chunk, tensors: Sequence[Tensor]) -> Chunk:
    """``chunk`` with its tensors replaced (a step's device values), its metadata kept."""
    tokens, targets, valid, doc_start, segment, loss_mask, pos = tensors
    return dataclasses.replace(
        chunk,
        tokens=tokens,
        targets=targets,
        valid=valid,
        doc_start=doc_start,
        segment=segment,
        loss_mask=loss_mask,
        pos=pos,
    )


@dataclass(frozen=True)
class _StepRead:
    """A device step's outputs on the host: read in one transfer after the step ran."""

    scalars: tuple[float, float, float, float]  # loss, ntp, ponder, grad norm
    pass_ntp: list[float]
    n_iters: list[Tensor]  # per group, [b, n] on the CPU
    states: list[list[Tensor]]  # per group, its new state tensors, still on the device


def _read_step(out: Sequence[Tensor], plan: "_StepPlan") -> _StepRead:
    """``_device_step``'s outputs read back: the only point the host waits for the device."""
    groups = len(plan.chunks)
    iters = list(out[5 : 5 + groups])
    record = torch.cat(
        [torch.stack(list(out[:4])).double(), out[4].double()]
        + [n.flatten().double() for n in iters]
    ).cpu()
    values = [float(v) for v in record.unbind()]
    t_max = plan.t_max
    n_iters: list[Tensor] = []
    offset = 4 + t_max
    for n in iters:
        n_iters.append(record[offset : offset + n.numel()].view(n.shape).to(torch.int64))
        offset += n.numel()
    states: list[list[Tensor]] = []
    rest = list(out[5 + groups :])
    for template in plan.templates:
        size = len(state_tensors(template))
        states.append(rest[:size])
        rest = rest[size:]
    return _StepRead(
        scalars=(values[0], values[1], values[2], values[3]),
        pass_ntp=values[4 : 4 + t_max],
        n_iters=n_iters,
        states=states,
    )


class StepGraphs(SignatureGraph):
    """The trainer's captured steps on CUDA (``graphs.SignatureGraph``) and its halting weight.

    The step's signature (``_StepPlan.key``) is the phase's pass count and head freeze, the
    objective, which groups drew a chunk and which of them carry a loss, the memory options and
    every value's shape (the attention windows fill during the first chunk, a dry group leaves
    the batch); phases and the tail only move forward, so an old signature never returns. The
    halting weight ``lambda_h``, which ramps every step, is a device scalar filled before each
    step, never a captured float.
    """

    def __init__(self, warm: int = WARM_STEPS) -> None:
        """Nothing captured yet."""
        super().__init__(warm)
        self._lambda_h: Tensor | None = None

    def lambda_h(self, device: torch.device) -> Tensor:
        """The device scalar the captured objective reads the halting weight from."""
        if self._lambda_h is None:
            self._lambda_h = torch.zeros((), dtype=torch.float32, device=device)
        return self._lambda_h


@dataclass(frozen=True)
class _StepPlan:
    """What one step's device work is, beyond its values: the signature and the structure."""

    # which drawn groups carry a loss position: a group without one skips the head (a Python
    # decision frozen into the graph). Which groups drew needs no field: the device work never
    # reads a group's index, and a dry group changes the values' count
    has_loss: tuple[bool, ...]
    t_max: int
    halting_frozen: bool
    fixed: bool
    recompute: bool
    slices: int
    templates: tuple[tuple[ColumnState, ...], ...]  # each group's state structure
    chunks: tuple[Chunk, ...]  # each group's chunk, for its metadata (the tensors are values)
    reset_every: int = 0  # training.state_reset_chunks (reset_keep); 0 = never
    first_slots: tuple[int, ...] = ()  # each drawn group's first slot in the whole batch
    total_slots: int = 0  # the batch's slots over every group, drawn or not

    def key(self, values: Sequence[Tensor]) -> tuple[object, ...]:
        """The capture signature: the plan's switches and every value's shape and dtype."""
        return signature(
            values,
            self.has_loss,
            self.t_max,
            self.halting_frozen,
            self.fixed,
            self.recompute,
            self.slices,
            self.reset_every,
            self.first_slots,
            self.total_slots,
        )


RESET_LEVELS = 6  # memory lifetimes per group of slots: K, 4K, 16K, 64K, 256K chunks, never
RESET_RATIO = 4


def reset_keep(
    pos: Tensor, chunk_tokens: int, shortest: int, first_slot: int, total_slots: int
) -> Tensor:
    """PRIMITIVE ``float32 [b]``: 0 where a slot's recurrent memory restarts cold, else 1.

    MIXED LIFETIMES: the ``b`` rows of ``pos`` are the batch's slots ``first_slot ..
    first_slot + b - 1`` of ``total_slots``; slot ``s`` has level ``s % RESET_LEVELS`` and period
    ``shortest * RESET_RATIO ** level`` chunks; the last level never restarts. So at every step
    the batch holds memories of every age, from cold to never cleared (a single period would
    trade the never-reset run's failure for its mirror image: no memory older than the period
    is ever trained). Slots are numbered across the whole batch, not within a micro-group, so
    how the batch is split into groups does not change any slot's lifetime (maintainer B 9e3066c:
    per group, six groups of two slots never had the never-cleared level). A slot's chunk index is
    ``pos[:, 0] // chunk_tokens`` (positions count every emitted position, padding included: the
    slot's own chunk count, resume-safe, no host state); it restarts when ``(index + offset) %
    period == 0`` and never at its first chunk (already cold). Slots that share a level are
    staggered by ``offset = copy * period // copies`` (``copy = s // RESET_LEVELS``, ``copies``
    the batch's slots per level, rounded up). Computed on the device, inside the captured step.

    For: training a model that must work from a cold start AND from long-carried memory (the 16x
    d512 run of 2026-10-21, never reset, reached held-out perplexity 36.4 from its own carried
    states and 351 from cold ones). Not for: ``shortest`` < 1.

    Raises:
        ValueError: when the rows do not lie inside ``0 .. total_slots - 1``.
    """
    b = int(pos.shape[0])
    if first_slot < 0 or first_slot + b > total_slots:
        msg = f"slots {first_slot}..{first_slot + b - 1} are not inside 0..{total_slots - 1}"
        raise ValueError(msg)
    slot = first_slot + torch.arange(b, device=pos.device)
    level = torch.remainder(slot, RESET_LEVELS)
    period = shortest * torch.pow(torch.full_like(slot, RESET_RATIO), level)
    copies = -(-total_slots // RESET_LEVELS)
    offset = torch.div(slot, RESET_LEVELS, rounding_mode="floor") * period // copies
    chunk_index = torch.div(pos[:, 0], chunk_tokens, rounding_mode="floor")
    due = torch.remainder(chunk_index + offset, period).eq(0)
    restart = due & chunk_index.gt(0) & level.lt(RESET_LEVELS - 1)
    return restart.logical_not().to(torch.float32)


def _device_step(
    model: Vesper,
    opt: torch.optim.AdamW,
    cfg: VesperConfig,
    plan: _StepPlan,
    lambda_h: float | Tensor,
    values: Sequence[Tensor],
) -> list[Tensor]:
    """The whole optimizer step on the device with no read-back: capturable as one CUDA graph.

    ``values``: per group that drew, its chunk's ``_CHUNK_FIELDS`` then its state tensors
    (``state_tensors``), and last the groups' shares of the batch's loss tokens (float32).
    Each group's forward (autocast bf16 on CUDA), its objective scaled by its share and its
    backward run in turn, so a group's logits and graph are released before the next forward
    (maintainer B's review of 444b966); a group without a loss position only advances its states.
    Then one clip over the trainable parameters and the optimizer's step. Returns the share-weighted
    loss, next-token and ponder parts, the gradient norm (scalars), the per-pass next-token loss
    ``[T]`` (zeros under "act"), every group's pass counts ``[b, n]`` and its new detached
    state tensors.
    """
    device = values[0].device
    shares = values[-1]
    weights = LossWeights(lambda_h=lambda_h, lambda_g=cfg.learning.lambda_g)
    params = [p for p in model.parameters() if p.requires_grad]
    opt.zero_grad(set_to_none=True)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss, ntp, ponder = zero, zero, zero
    pass_ntp = torch.zeros(plan.t_max, dtype=torch.float32, device=device)
    iters: list[Tensor] = []
    states_out: list[Tensor] = []
    cursor = 0
    for g, (template, chunk) in enumerate(zip(plan.templates, plan.chunks, strict=True)):
        local = _chunk_with(chunk, values[cursor : cursor + _CHUNK_FIELDS])
        cursor += _CHUNK_FIELDS
        size = len(state_tensors(template))
        states = with_state_tensors(template, values[cursor : cursor + size])
        cursor += size
        if plan.reset_every > 0:
            carry = reset_keep(
                local.pos,
                cfg.training.chunk_tokens,
                plan.reset_every,
                plan.first_slots[g],
                plan.total_slots,
            )
            states = [reset_recurrent(state, carry) for state in states]
        share = shares[g]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            result = model.run(
                local.tokens,
                local.pos,
                states,
                passes=plan.t_max,
                recompute=plan.recompute,
                exit_rule=DenseRule() if plan.fixed else None,
            )
            states_out += state_tensors(detach_states(result.states))
            iters.append(result.n_iters.detach())
            if not plan.has_loss[g]:
                continue
            # the head over the loss positions in slices; without recomputation its logits are
            # kept for the backward pass too (one switch trades memory for speed everywhere)
            keep = not plan.recompute
            if plan.fixed:
                value, all_pass = model.all_pass_objective(
                    local, result, condition="fixed", beta=0.0, slices=plan.slices, keep_logits=keep
                )
                part_ntp, part_ponder = all_pass["expected_ntp"], zero
                pass_ntp = pass_ntp + all_pass["pass_ntp"].float() * share
            else:
                value, parts = model.objective(
                    local, result, cfg, weights, slices=plan.slices, keep_logits=keep
                )
                part_ntp, part_ponder = parts["ntp"], parts["ponder"]
        torch.autograd.backward([value * share])
        loss = loss + value.detach().float() * share
        ntp = ntp + part_ntp.detach().float() * share
        ponder = ponder + part_ponder.detach().float() * share
    grad_norm = nn.utils.clip_grad_norm_(params, cfg.optimizer.grad_clip).float()
    # the optimizer's step is untyped in torch; its result (a closure's loss) is not used here
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    return [loss, ntp, ponder, grad_norm, pass_ntp, *iters, *states_out]


def _check_objective(objective: str) -> None:
    if objective not in OBJECTIVES:
        msg = f"objective must be one of {OBJECTIVES}, got {objective!r}"
        raise ValueError(msg)


# a safety save's directory name: the run's id (UTC stamp to the microsecond, pid), then the step
_SAFETY_SAVE_NAME = re.compile(r"\d{8}T\d{12}Z-\d+-step-\d+")


def _remove_older_safety_saves(safety: Path, keep: Path) -> tuple[list[str], str | None]:
    """Remove the safety saves in ``safety`` other than ``keep``, never raising.

    Returns the removed names and the error that stopped the cleanup (None when it finished).
    Owned entries only: a real directory (not a symlink) whose name is a safety save's
    (``_SAFETY_SAVE_NAME``). Anything else there stays: staging directories, an operator's files
    and directories. A failure while listing or removing stops the cleanup and is returned, never
    raised, because ``keep`` is already published: the caller reports it and trains on, and the
    saves left over are removed by a later cleanup.

    Not for: evaluated checkpoints (immutable, never removed by the run).
    """
    removed: list[str] = []
    try:
        for entry in sorted(safety.iterdir()):
            owned = _SAFETY_SAVE_NAME.fullmatch(entry.name) is not None
            if entry == keep or not owned or entry.is_symlink() or not entry.is_dir():
                continue
            shutil.rmtree(entry)
            removed.append(entry.name)
    except OSError as error:
        return removed, f"{type(error).__name__}: {error}"
    return removed, None


def _check_safety_seconds(safety_seconds: float | None) -> None:
    if safety_seconds is not None and not 0 < safety_seconds < math.inf:
        msg = f"safety_seconds must be a positive finite number of seconds, got {safety_seconds}"
        raise ValueError(msg)


def _read_inputs(
    tokenizer_path: Path, config_path: Path, finalization_receipt: Path, cfg: VesperConfig
) -> tuple[bytes, Finalization, tuple[str, str, str]]:
    """The tokenizer's bytes, the finalization and the three files' hashes, one snapshot each.

    The tokenizer and config are each read once here. The caller parses those tokenizer bytes;
    the config bytes must parse to the supplied ``cfg``. The receipt is read for its hash, then
    reread with that hash verified before parsing (``load_finalization``). Thus parsed content
    and recorded identity stay bound even though the receipt has two reads (maintainer B's
    identity_of follow-up of 10-21: the identity hashed paths independently of the content the run
    used).

    Raises:
        ValueError: when the config file no longer parses to ``cfg``, or the receipt changes
            between its hash and its parse.
    """
    tokenizer_bytes, tokenizer_sha256 = hashed_bytes(tokenizer_path)
    config_bytes, config_sha256 = hashed_bytes(config_path)
    if parse_bytes(config_bytes, str(config_path)) != cfg:
        msg = f"{config_path}: changed since it was parsed for this run"
        raise ValueError(msg)
    _, finalization_sha256 = hashed_bytes(finalization_receipt)
    finalization = load_finalization(finalization_receipt, sha256=finalization_sha256)
    return tokenizer_bytes, finalization, (tokenizer_sha256, config_sha256, finalization_sha256)


def _check_transition(*, cycle: bool, from_finite: bool, resume: Path | None) -> None:
    if from_finite and not (cycle and resume is not None):
        msg = "from_finite is the transition of a resumed finite run to cycling: it needs both"
        raise ValueError(msg)


def _state_record(states: Sequence[ColumnState]) -> list[dict[str, object]]:
    """The carried states as plain containers for ``weights_only`` loads.

    Raises:
        ValueError: on a ``RingKVCache`` (a decoding window, never a trainer's state).
    """
    if any(isinstance(c, RingKVCache) for state in states for c in state.kv):
        msg = "a decoding ring window is not a trainer state and is never saved"
        raise ValueError(msg)
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
    recipe, the layout, the reason (``interval``, ``budget``, ``steps``, ``exhausted`` or
    ``safety``) and the
    label; ``manifest.json`` the sha256 of each file. The directory is staged and published by
    one rename (``publish.staged``), so a failed save leaves nothing at ``path`` and a retry is
    not refused. Returns the manifest's path.

    Raises:
        ValueError: when ``path`` already holds a checkpoint (they are immutable).
    """
    if path.exists():
        msg = f"{path}: a checkpoint already exists there"
        raise ValueError(msg)
    streams = [g.stream_state() for g in groups]
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
    # one snapshot of train.json and the manifest, its schema checked first; every other file
    # checked against that manifest and loaded from the bytes that were checked
    record, manifest = checked_record(path)
    blobs = {
        name: verified_bytes(path / name, manifest.get(name), f"{name}, against the manifest")
        for name in FILES
        if name != "train.json"
    }
    saved = cast("dict[str, object]", record["identity"])
    if saved != _plain(identity):
        differ = sorted(k for k, v in _plain(identity).items() if saved.get(k) != v)
        msg = (
            f"{path}: checkpoint of a run with another tokenizer, config, finalization or recipe"
            f" (differs: {', '.join(differ)}; the recipe includes the objective)"
        )
        raise ValueError(msg)
    if cast("dict[str, object]", record["layout"]) != _plain(layout):
        msg = (
            f"{path}: checkpoint of {record['layout']} groups and slots; this run asks for"
            f" {_plain(layout)}; regrouping is not supported"
        )
        raise ValueError(msg)
    loaded = cast("dict[str, Tensor]", torch.load(io.BytesIO(blobs["model.pt"]), weights_only=True))
    _ = cast("object", model.load_state_dict(loaded))
    opt.load_state_dict(
        cast("dict[str, object]", torch.load(io.BytesIO(blobs["optimizer.pt"]), weights_only=True))
    )
    # states load on the CPU whatever device saved them, are checked there (one device, real
    # floating dtypes) and moved to the run's device by _to_device: the explicit migration
    saved_states = cast(
        "dict[str, object]",
        torch.load(io.BytesIO(blobs["states.pt"]), weights_only=True, map_location="cpu"),
    )
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
        # absent in records written before cycling existed: pass 0 everywhere (MixerState)
        passes=tuple(int(p) for p in cast("list[int]", record.get("passes", []))),
        yielded=tuple(int(y) for y in cast("list[int]", record.get("yielded", []))),
    )


class CheckedRecord(NamedTuple):
    """A checkpoint's ``train.json`` and the ONE ``manifest.json`` snapshot it was checked against.

    Every other file of the checkpoint is checked against this same ``manifest`` (read once),
    with ``keys.verified_bytes``, and loaded from the bytes that were checked.
    """

    record: dict[str, object]
    manifest: dict[str, object]


def checked_record(path: Path) -> CheckedRecord:
    """The checkpoint's ``train.json`` as one validated snapshot, and the manifest it matched.

    Hashed and parsed from the same bytes, its schema checked before any field is read (maintainer
    B's review of f55bfb3: a hash-then-reread let a replacement land between the two, and a schema-3
    record reached a KeyError first). The record's ``recipe`` is known to be a mapping of the
    current schema. Shared by ``restore``, ``resumed_components`` and the text pilot's
    ``halting_text.load_text_source`` (maintainer B's review of 72c9d31).

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
    data = verified_bytes(file, manifest.get("train.json"), "train.json, against the manifest")
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
    return CheckedRecord(record, manifest)


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
        ValueError: when the checkpoint's record is missing, tampered or inconsistent (a recipe
            that does not hash to its identity's ``recipe_sha256``, the digest ``identity_of``
            computes; a pin whose kind is not its component's, counts that differ), a component
            carries no pin
            (a legacy-layout run resumes under ``--legacy-layout --web --mix``), a pin is
            malformed, or a pinned generation is missing, changed or re-hashed.
    """
    return components_of_record(data, checked_record(resume), resume)


def components_of_record(data: Path, checked: CheckedRecord, resume: Path) -> list[Component]:
    """``resumed_components`` for a record snapshot the caller already read and checked.

    For a caller that must bind the components to what it loaded: it reads ``checked_record``
    once, compares that snapshot's identity with its loaded one, and resolves from the same
    snapshot, so no later read of the files can change what is resolved (maintainer B's review of
    01e2b3f: a replacement inside the resolver, restored before the caller's next check,
    resolved other shares). ``resume`` names the checkpoint in messages; it is not read.

    Raises:
        ValueError: as ``resumed_components``.
    """
    record = checked.record
    recipe = cast("dict[str, object]", record["recipe"])  # a mapping: checked_record said so
    # the recipe consumed here is the one the record's identity names (maintainer B's review of
    # a771dee: other shares with an updated manifest passed every check beside it)
    raw_identity = record.get("identity")
    identity = cast("dict[str, object]", raw_identity) if isinstance(raw_identity, dict) else {}
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if identity.get("recipe_sha256") != actual:
        msg = f"{resume}: the recipe does not hash to its identity's recipe_sha256 ({actual})"
        raise ValueError(msg)
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


def recipe_of(
    parts: Sequence[Component],
    seed: int,
    cfg: VesperConfig,
    objective: str = "act",
    *,
    cycle: bool = False,
) -> Recipe:
    """The run's recipe.

    The seed, the ordered components with their shares and generation pins, the split, the rules,
    the training objective, and whether the training mix cycles its components.
    """
    return Recipe(
        seed=seed,
        components=tuple((c.name, c.share) for c in parts),
        generations=tuple(c.generation for c in parts),
        held_out_fraction=cfg.training.held_out_fraction,
        finalize_rule=FINALIZE_RULE,
        split_rule=SPLIT_RULE,
        schema=CHECKPOINT_SCHEMA,
        objective=objective,
        component_passes=CYCLE_ORDER_RULE if cycle else "finite",
    )


def identity_of(
    tokenizer_sha256: str, config_sha256: str, finalization_sha256: str, recipe: Recipe
) -> Identity:
    """The identity of a run: the hashes of the three files it read and of its recipe.

    The caller hashes the bytes it actually parses (``keys.hashed_bytes``), so the identity
    describes what the run used, never a later read of the same paths.
    """
    canonical = json.dumps(_plain(recipe), sort_keys=True, separators=(",", ":"))
    return Identity(
        tokenizer_sha256=tokenizer_sha256,
        config_sha256=config_sha256,
        finalization_sha256=finalization_sha256,
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
    if isinstance(value, Recipe) and value.component_passes == "finite":
        del plain["component_passes"]  # the record and hash a finite recipe always had
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
    _ = parser.add_argument(
        "--safety-minutes",
        type=float,
        default=None,
        help="also save the resumable state (no evaluation, newest only, out/safety) this often",
    )
    _ = parser.add_argument(
        "--cycle-components",
        action="store_true",
        help="a training component that runs out starts again in a new order (part of the recipe)",
    )
    _ = parser.add_argument(
        "--from-finite-checkpoint",
        action="store_true",
        help="with --cycle-components and --resume: the recorded transition from a finite run",
    )
    _ = parser.add_argument(
        "--scan",
        choices=SCAN_IMPLS,
        default=None,
        help=(
            "the recurrent scan: the reference (CPU only), the fused CUDA kernel (bf16"
            " intermediates), or the fused float32 CUDA kernel (the reference's precision);"
            " default fused-fp32 on CUDA, the reference on the CPU"
        ),
    )
    _ = parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        default="act",
        help=(
            "act: the recipe's ACT objective; fixed: every pass supervised with equal weight on a "
            "dense forward (depth first); a resume keeps the run's objective"
        ),
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
    scan: str = "reference",
    objective: str = "act",
    safety_seconds: float | None = None,
    cycle: bool = False,
    from_finite: bool = False,
) -> Path:
    """The run: streams, schedule, steps, log lines, checkpoints with evaluations.

    ``cycle`` makes the training mix start a component that runs out again in a new order
    (``corpus.Mixer``, ``CYCLE_ORDER_RULE``) instead of dropping it, so the shares hold for any
    budget; it is part of the recipe and so of the identity, and the held-out stream stays
    finite. ``from_finite`` (with ``cycle`` and ``resume`` only) is the one recorded transition
    from a finite run: the checkpoint is checked against the identity of the SAME recipe with
    finite passes, not against the cycling one, and the run line names it (``transition``);
    every checkpoint after it carries the cycling identity. Without it, a finite checkpoint
    under ``cycle`` is refused like any other identity mismatch.

    ``safety_seconds`` (None: off) adds safety saves between the evaluated checkpoints. The
    interval is checked after each step: the first step that ends at least that much wall time
    after the last save of either kind writes its full resumable state without an evaluation to
    ``out/safety/<run_id>-step-<n>`` (a target cadence: a save lands up to one step later, and a
    failed save moves the next attempt one interval on). After a successful save the older
    safety saves there are removed (``_remove_older_safety_saves``: owned entries only); a
    failed cleanup is reported on the save's line (``cleanup_error``) and the run trains on. A
    failed save is logged (``safety_failed``), the previous one kept, and the run trains on. So
    a crash loses the training since the last SUCCESSFUL save. A safety save resumes like any
    checkpoint. It is no part of the recipe (it changes no training).

    ``objective`` (``OBJECTIVES``, ``step``) is part of the run's recipe and so of its identity:
    the run line records it and ``restore`` refuses a checkpoint of another objective. Every
    checkpoint also records the held-out static profile (``static_profile``: the loss at each
    fixed depth).

    ``scan`` chooses the recurrent scan for the whole run, evaluations included (``set_scan``):
    "reference", "fused" or "fused-fp32" (CUDA and the gpu extra); the run line records what the
    model's sublayers report (``scan_of``), and a resume may change it, since it is no part of
    the checkpoint.

    The run ends when the token budget is reached, the optional step cap is hit or every stream
    is dry (the ``end`` line's ``reason``: ``budget``, ``steps``, ``exhausted``); the last
    completed step is checkpointed exactly once, with that reason, unless an interval checkpoint
    already holds it. Every line carries the ``run_id`` of its header; a fresh run never appends
    to a log that holds one (a resume does, with ``resumed_from``). Returns the log's path.

    Raises:
        ValueError: when ``batch_chunks`` is not a multiple of ``micro``, the objective is
            unknown or differs from the resumed run's, the finalization does not cover the
            components, ``out`` already holds a run and this is not a resume, the checkpoint has
            another identity or layout, or its replay does not reproduce the checkpoint's
            documents (their digest and the realized shares).
    """
    t = cfg.training
    check_mode(cfg.learning.mode)  # an unsupported mode is refused before anything is read
    _check_objective(objective)
    if micro < 1 or t.batch_chunks % micro != 0:
        msg = f"micro_chunks must divide batch_chunks ({t.batch_chunks}), got {micro}"
        raise ValueError(msg)
    _check_safety_seconds(safety_seconds)
    _check_transition(cycle=cycle, from_finite=from_finite, resume=resume)
    tokenizer_bytes, finalization, files = _read_inputs(
        tokenizer_path, config_path, finalization_receipt, cfg
    )
    check_inputs(finalization, parts)
    recipe = recipe_of(parts, seed, cfg, objective, cycle=cycle)
    identity = identity_of(*files, recipe)
    # the identity the checkpoint must carry: the finite recipe's for the recorded transition
    finite = recipe_of(parts, seed, cfg, objective)
    parent = identity_of(*files, finite) if from_finite else identity
    layout = Layout(groups=t.batch_chunks // micro, slots=micro)
    # every random choice goes through an explicit generator: the model's initialization here,
    # the corpus order through the seed; nothing reads torch's global generator. The scan is
    # chosen before anything is written, so a refused "fused" leaves no trace.
    model = Vesper(cfg, torch.Generator().manual_seed(seed)).to(device)
    _ = check_capturable_scan(scan, device)  # CUDA work is captured: no reference scan there
    _ = set_scan(model, scan)
    # on CUDA every step is one captured graph (the project owner 2026-10-21: a hard requirement),
    # so the optimizer is the capturable variant whose rate is a device tensor
    cuda = device.type == "cuda"
    opt = optimizer(model, cfg, capturable=cuda)
    graphs = StepGraphs() if cuda else None
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    if resume is None and log_path.exists():
        msg = f"{log_path}: already holds a run; resume one of its checkpoints or use another out"
        raise ValueError(msg)
    tokenizer = Tokenizer.parse(tokenizer_bytes)
    split = Split(finalization, t.held_out_fraction)
    held = Split(finalization, t.held_out_fraction, held_out=True)
    counts: dict[str, int] = {}
    mixer = Mixer(parts, seed, tokenizer, split, tokens_emitted=counts, cycle=cycle)
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
        restored = restore(resume, model, opt, parent, layout)
        progress = restored.progress
        if restored.mixer is None:
            msg = f"{resume}: the checkpoint holds no mixer state; it cannot be resumed"
            raise ValueError(msg)
        # bounded cost: the readers seek to their cursors, the digest continues from its state;
        # the three records of the same facts (mixer state, progress, streams) must agree
        mixer = Mixer.restore(
            parts, seed, tokenizer, split, restored.mixer, tokens_emitted=counts, cycle=cycle
        )
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
            "scan": scan_of(model),
            "objective": objective,
            "device": str(device),
            "resumed_from": str(resume) if resume else None,
            "transition": (
                {"from": "finite", "to": recipe.component_passes, "parent_identity": _plain(parent)}
                if from_finite
                else None
            ),
            "safety_seconds": safety_seconds,
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
    # the mixer as of the chunks handed out to a step: taken right before the groups draw ahead,
    # so a checkpoint and the log agree with the groups' stream_state (all three describe the
    # documents pulled for trained chunks only; a resume rebuilds the drawn-ahead chunks)
    handed: tuple[MixerState, dict[str, int], str] | None = None

    def handed_out() -> tuple[MixerState, dict[str, int], str]:
        return handed if handed is not None else (mixer.state(), dict(counts), mixer.digest())

    def ahead() -> None:
        nonlocal handed
        handed = (mixer.state(), dict(counts), mixer.digest())
        for group in groups:
            group.draw_ahead()

    def write(where: Path, reason: str) -> None:
        mixer_state, emitted, digest = handed_out()
        current = Progress(
            step=step_index,
            tokens_seen=tokens_seen,
            tokens_emitted=dict(emitted),
            tokens_consumed=dict(consumed),
            documents_sha256=digest,
        )
        _ = checkpoint(
            where, model, opt, groups, current, identity, recipe, reason=reason, mixer=mixer_state
        )

    safety = out / "safety"
    saved_at = time.monotonic()  # the last save of either kind, for the safety interval

    def safety_save() -> None:
        nonlocal saved_at
        due = safety_seconds is not None and time.monotonic() - saved_at >= safety_seconds
        if not due or step_index == checkpointed_at:  # an evaluated checkpoint was just saved
            return
        where = safety / f"{run_id}-step-{step_index}"
        started = time.monotonic()
        try:
            safety.mkdir(exist_ok=True)
            write(where, "safety")
        # the serialization boundary only: the file system raises OSError, torch.save's writer a
        # RuntimeError (maintainer B 8d6c1c7: a real file-size limit, inline_container.cc); the
        # staged directory is never published, the previous safety save stays and the run goes on
        except (OSError, RuntimeError) as error:
            emit(
                {
                    "kind": "safety_failed",
                    "step": step_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        else:
            # the new save is published: a failed cleanup is reported with it, never raised
            removed, cleanup_error = _remove_older_safety_saves(safety, where)
            emit(
                {
                    "kind": "safety",
                    "step": step_index,
                    "path": str(where),
                    "seconds": time.monotonic() - started,
                    "removed": removed,
                    "cleanup_error": cleanup_error,
                }
            )
        saved_at = time.monotonic()

    def save(reason: str) -> None:
        nonlocal checkpointed_at, saved_at
        where = out / f"step-{step_index}"
        write(where, reason)
        checkpointed_at = step_index
        chunks = held_out_chunks(parts, seed, tokenizer, held, micro, cfg)
        # captured on CUDA; built per checkpoint so their memory is not held while training
        eval_graphs = EvalGraphs() if cuda else None
        evaluation = evaluate(model, chunks, cfg, passes=phase.t_max, graphs=eval_graphs)
        profile = static_profile(model, chunks, cfg, graphs=eval_graphs)
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
                "held_out_static_loss": profile.mean_losses,
                "held_out_static_tokens_with_loss": profile.tokens_with_loss,
                "training_t_max": training_passes,
            }
        )
        saved_at = time.monotonic()  # after the evaluation: the safety interval counts training

    def stopped() -> str | None:
        if tokens_seen >= t.total_tokens:
            return "budget"
        if step_cap is not None and step_index >= step_cap:
            return "steps"
        return None

    while (reason := stopped()) is None:
        phase = phase_schedule(tokens_seen, step_index, cfg)
        result = step(
            model,
            opt,
            groups,
            phase,
            cfg,
            device,
            recompute=recompute,
            slices=slices,
            objective=objective,
            graphs=graphs,
            ahead=ahead,
        )
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
        line["shares"] = dict(handed_out()[1])  # tokens the mixer handed out for trained chunks
        line["consumed"] = dict(consumed)  # inputs the model saw, per source, over the run
        emit(line)
        if tokens_seen >= next_checkpoint:
            next_checkpoint += every
            save("interval")
        safety_save()
    if step_index != checkpointed_at:
        save(reason)
    emit(
        {
            "kind": "end",
            "step": step_index,
            "tokens_seen": tokens_seen,
            "reason": reason,
            # the proof that a CUDA run trained captured: its graphs' captures and replays
            "step_graph": None
            if graphs is None
            else {"captures": graphs.captures, "replays": graphs.replays},
        }
    )
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


def _scan_for(chosen: str | None, device: torch.device) -> str:
    """The scan asked for, else the device's default.

    fused-fp32 on CUDA (capturable, the reference's precision), the reference on the CPU.
    """
    if chosen is not None:
        return chosen
    return "fused-fp32" if device.type == "cuda" else "reference"


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: 0 when the run reached its last step or its data, 1 on an error."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    device = torch.device(cast("str", args.device))
    config_path = cast("Path", args.config)
    cfg = load(config_path)
    resume = cast("Path | None", args.resume)
    data = cast("Path | None", args.data)
    legacy_layout = cast("bool", args.legacy_layout)
    web = cast("Path | None", args.web)
    mix = cast("Path | None", args.mix)
    safety_minutes = cast("float | None", args.safety_minutes)
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
            scan=_scan_for(cast("str | None", args.scan), device),
            objective=cast("str", args.objective),
            steps=cast("int | None", args.steps),
            resume=resume,
            seed=cast("int", args.seed),
            device=device,
            safety_seconds=None if safety_minutes is None else safety_minutes * 60,
            cycle=cast("bool", args.cycle_components),
            from_finite=cast("bool", args.from_finite_checkpoint),
        )
    except (RuntimeError, ValueError, TypeError):
        log.exception("stopped")
        return 1
    log.info("log at %s", log_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
