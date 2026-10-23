"""The halting experiment's stage-1 driver: train one condition on the synthetic task, evaluate it.

Plan: docs/reviews/2026-10-15-halting-experiment-plan.md (stage 1); task: ``vesper.synthetic``.

    python -m vesper.halting_experiment --condition fixed --seed 0 --out RUN_DIR \
        [--config configs/halting-stage1.yaml] [--steps N] [--beta 0.1] [--eval-batches 8]
        [--mix v1|v2-slow] [--device cpu|cuda] [--scan reference|fused|fused-fp32] [--cuda-graph]
        [--data-seed N] [--static-every N] [--static-batches N] [--init-weights WEIGHTS]

Training runs on ``--device`` with the recurrent ``--scan``; ``--cuda-graph`` (on CUDA)
captures one step and replays it (``vesper.graphs``). Probes and the evaluation always run on
a CPU copy with the reference scan.

``--init-weights`` starts the run from another run's weights instead of the seed's
initialization (the warm start of docs/reviews/2026-10-17-halting-in-phases.md): a run's
``model.pt`` or ``weights/after-NNNNN.pt`` (``vesper.halting_common.read_warm_start``), whose
run's config snapshot must agree with this run's in its model and recurrence sections; every
name, shape and dtype must match the
model and every value be finite. The halting head starts as the condition's own (``build``),
never as the file's: the escaped weights come from condition B, whose head was never trained,
so they are not a trained stopping policy. It is a new run: the optimizer, the data order, the
schedule and the progress start fresh. result.json records the file (``init_weights``: path,
sha256 of the bytes loaded, digest of its tensors, the source run's config sha256, ``head``).

Conditions (the plan's table): ``act`` (A: ACT with its mixture readout and ponder cost, the
trainer's own phase schedule and head freeze, its copy-forward state policy), ``fixed`` (B:
uniform exit weights, no head trained, deployed with ``CdfRule(None, q)``), ``local`` (C) and
``aggregate`` (D) (the head from ``zero_residual``, trained from the first step, deployed with
``CdfRule(head, q)``). B-D train DENSE (``DenseRule``). Every condition shares the model
initialization, the data order (``generator_for("train", seed)``, or ``--data-seed``'s for a
crossed-seed run), the optimizer and the
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
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO, cast

import torch
from torch import Tensor, nn

from vesper.column import (
    FUSED_INPUT_DTYPES,
    SCAN_IMPLS,
    Column,
    check_capturable_scan,
    scan_of,
    set_scan,
)
from vesper.config import VesperConfig, parse_bytes
from vesper.data import Chunk
from vesper.exits import exit_log_probs, hazard_cdf_exits, hazard_logits
from vesper.graphs import WarmedGraph
from vesper.halting_common import (
    CPU,
    EVAL_SCAN,
    SNAPSHOT,
    WarmStart,
    chained_digest,
    chunk_tensors,
    class_accuracy,
    conditional_entropy,
    evaluation_model,
    load_weights,
    plain,
    read_warm_start,
    version_or_none,
)
from vesper.learning import LossWeights
from vesper.loop import CdfRule, DenseRule, ExitRule, TraceRule
from vesper.model import Vesper
from vesper.provenance import environment, write_json
from vesper.synthetic import (
    CLASSES,
    DEFAULT_MIX,
    GAP,
    MIX_NAMES,
    MIX_SCHEDULES,
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
from vesper.train import (
    Phase,
    optimizer,
    phase_schedule,
    set_halting,
    set_learning_rate,
    total_steps,
)

CONDITIONS = ("act", "fixed", "local", "aggregate")  # the plan's A, B, C, D
Q_GRID = tuple(k / 10 for k in range(1, 11))
Q_REFERENCE = 0.5
CARRIED_TOKENS = 8
EVAL_SPLITS = ("validation", "final")
WEIGHTS_EVERY = 1000  # updates between diagnostic weight snapshots
STATIC_CHECKS = (
    "static_checks.jsonl"  # a run's static-depth checks (``static_every``), one per line
)
STATIC_SPLIT = "validation"  # the static checks read the first episodes of this split
TRAIN_STATE_DTYPE = torch.float32  # the recurrent states every training step starts from
EVAL_BATCH = 32  # 32 batches of 32: the probe's 1024 fixed validation episodes
DEVICES = ("cpu", "cuda")

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


def build(
    cfg: VesperConfig, condition: str, seed: int, weights: Mapping[str, Tensor] | None = None
) -> Vesper:
    """The model for one run: the seed's initialization or ``weights``, the condition's head.

    ``local`` and ``aggregate`` start from ``zero_residual`` (exactly uniform with the
    baseline, never ACT's frozen bias); ``fixed`` freezes the head, which it never reads.
    ``weights`` (a warm start) replace the whole initialization BEFORE the head is set, so the
    head starts as its condition's whatever the file holds (``load_weights``; ACT's is set by
    the trainer's ``set_halting`` at the first step, as for a fresh start).

    Raises:
        ValueError: on an unknown condition, a config the task cannot use (``_check_setup``),
            or ``weights`` that do not fit the model (``load_weights``).
    """
    _check_setup(cfg, condition)
    model = Vesper(cfg, torch.Generator().manual_seed(seed))
    if weights is not None:
        load_weights(model, weights)
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
    """One optimizer step on one batch, eagerly, on the model's device; returns its record."""
    c = _on(ep.chunk, _device_of(model))
    phase = phase_schedule(step * int(c.tokens.shape[0]) * SEQUENCE_TOKENS, step, cfg)
    set_learning_rate(opt, phase.lr)
    return _record(step, phase.lr, _update(model, opt, cfg, condition, beta, c, phase), c)


@dataclass(frozen=True, slots=True)
class StepOutputs:
    """What one update leaves on the device; ``_record`` reads it after the step."""

    loss: Tensor
    grad_norm: Tensor
    n_iters: Tensor
    parts: dict[str, Tensor]


def _update(  # noqa: PLR0913
    model: Vesper,
    opt: torch.optim.AdamW,
    cfg: VesperConfig,
    condition: str,
    beta: float,
    chunk: Chunk,
    phase: Phase,
    *,
    lambda_h: Tensor | None = None,
) -> StepOutputs:
    """Forward, objective, backward, clipping and the optimizer's step, with no host read-back.

    The one update both training paths run: eagerly (``train_step``) and captured as a CUDA
    graph (``GraphedTraining``). The learning rate is set before it. For B-D nothing here
    depends on the step but the batch; ``act`` also reads ``phase``: its head freeze and
    ``T_max`` are fixed within a capture (``_signature``), and its ponder weight is read from
    ``lambda_h``, a device tensor filled before each replay, when one is given (else the
    phase's float, eagerly).
    """
    states = model.init_states(int(chunk.tokens.shape[0]), TRAIN_STATE_DTYPE, _device_of(model))
    if condition == "act":
        set_halting(model, phase)
        result = model.run(chunk.tokens, chunk.pos, states, None, phase.t_max)
        ponder_weight = phase.lambda_h if lambda_h is None else lambda_h
        weights = LossWeights(lambda_h=ponder_weight, lambda_g=cfg.learning.lambda_g)
        loss, parts = model.objective(chunk, result, cfg, weights)
    else:
        result = model.run(chunk.tokens, chunk.pos, states, exit_rule=DenseRule())
        loss, parts = model.all_pass_objective(chunk, result, condition=condition, beta=beta)
    opt.zero_grad(set_to_none=True)
    torch.autograd.backward([loss])
    parameters = [p for p in model.parameters() if p.requires_grad]
    grad_norm = nn.utils.clip_grad_norm_(parameters, cfg.optimizer.grad_clip)
    # the optimizer's step is untyped in torch; its result (a closure's loss) is not used here
    _ = cast("float | None", opt.step())  # pyright: ignore[reportUnknownMemberType]
    return StepOutputs(loss.detach(), grad_norm.detach(), result.n_iters, parts)


def _record(step: int, lr: float, out: StepOutputs, chunk: Chunk) -> dict[str, object]:
    """The step's log record, read from its outputs after the step (the only host read-back)."""
    mask = chunk.loss_mask.to(out.n_iters.device)
    record: dict[str, object] = {
        "step": step,
        "loss": float(out.loss.item()),
        "lr": lr,
        "grad_norm": float(out.grad_norm.item()),
        "loss_depth": float(out.n_iters[mask].float().mean().item()),
    }
    for name, value in out.parts.items():
        record[name] = [float(v) for v in value.unbind()] if value.dim() else float(value.item())
    return record


def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _check_backend(device: str, scan: str, condition: str, *, cuda_graph: bool) -> torch.device:
    """The training device, after checking the device, scan and capture choices together.

    Raises:
        ValueError: on an unknown condition, a device outside ``DEVICES``, a scan it cannot
            run or cannot capture (the reference on CUDA), ``cuda_graph`` off CUDA, or CUDA
            without ``cuda_graph`` (CUDA graphs for anything on the GPU, the project owner
            2026-10-21).
    """
    if condition not in CONDITIONS:
        msg = f"condition must be one of {CONDITIONS}, got {condition!r}"
        raise ValueError(msg)
    if device not in DEVICES:
        msg = f"device must be one of {DEVICES}, got {device!r}"
        raise ValueError(msg)
    target = torch.device(device)
    _ = check_capturable_scan(scan, target)
    if cuda_graph and target.type != "cuda":
        msg = f"cuda_graph needs device cuda, got {device}"
        raise ValueError(msg)
    if target.type == "cuda" and not cuda_graph:
        # CUDA graphs are a hard requirement for anything on the GPU (the project owner 2026-10-21)
        msg = "on CUDA the step runs captured: pass cuda_graph (no eager GPU training)"
        raise ValueError(msg)
    return target


def _labels(ep: Episodes) -> list[int]:
    """The batch's labels per class, binding the mix schedule to the run's log."""
    labels = torch.bincount(ep.task[ep.task >= 0], minlength=len(CLASSES))
    return [int(v) for v in labels.unbind()]


def _cpu_state(model: nn.Module) -> dict[str, Tensor]:
    """The model's weights as CPU tensors, whatever device it trains on (portable snapshots)."""
    state = cast("dict[str, Tensor]", model.state_dict())
    return {name: tensor.detach().cpu() for name, tensor in state.items()}


def _backend(model: Vesper, *, cuda_graph: bool) -> dict[str, object]:
    """What a run's training arithmetic ran on, and what evaluated it (maintainer B 5dba6af).

    The device and its name, the scan and the dtype it takes its inputs in, the parameter and
    state dtypes, autocast (training never enables it), the float32 matmul policy, the capture
    choice, the versions of the libraries that compute, and the evaluation backend.
    """
    device = _device_of(model)
    scan = scan_of(model)
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    widest = torch.promote_types(next(model.parameters()).dtype, torch.float32)  # the reference's
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "scan": scan,
        "scan_input_dtype": str(FUSED_INPUT_DTYPES.get(scan, widest)),
        "parameter_dtypes": parameter_dtypes,
        "state_dtype": str(TRAIN_STATE_DTYPE),
        "autocast": None,
        "matmul_allow_tf32": cast("bool", torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_graph": cuda_graph,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": version_or_none("triton"),
        "fla-core": version_or_none("fla-core"),
        "evaluation": {"device": str(CPU), "scan": EVAL_SCAN},
    }


def _mapped(chunk: Chunk, f: Callable[[Tensor], Tensor]) -> Chunk:
    """``chunk`` with ``f`` applied to each of its tensors; the document lists are kept."""
    return _with_tensors(chunk, [f(t) for t in chunk_tensors(chunk)])


def _with_tensors(chunk: Chunk, tensors: Sequence[Tensor]) -> Chunk:
    """``chunk`` with ``tensors`` in ``chunk_tensors`` order; the document lists are kept."""
    tokens, targets, valid, doc_start, segment, loss_mask, pos = tensors
    return replace(
        chunk,
        tokens=tokens,
        targets=targets,
        valid=valid,
        doc_start=doc_start,
        segment=segment,
        loss_mask=loss_mask,
        pos=pos,
    )


def _on(chunk: Chunk, device: torch.device) -> Chunk:
    """``chunk`` with its tensors on ``device`` (the chunk itself on its own device)."""
    if chunk.tokens.device == device:
        return chunk
    return _mapped(chunk, lambda t: t.to(device))


def _signature(condition: str, phase: Phase) -> tuple[bool, int] | None:
    """What a capture of ``condition``'s update fixes besides the batch, rate and ponder weight.

    B-D: nothing (their update reads the batch only). ``act``: the head freeze, which decides
    which parameters train, and the pass count; a new signature needs a new capture.
    """
    return (phase.halting_frozen, phase.t_max) if condition == "act" else None


class GraphedTraining:
    """Training steps on CUDA as captured graphs (``vesper.graphs``), every condition.

    One capture per signature (``_signature``), by ``vesper.graphs.WarmedGraph``: after a run
    starts and after every change of signature, ``WARM_STEPS`` steps run eagerly on a side
    stream (real steps, logged as any other; a newly trained parameter's optimizer state is
    created there, never during a capture), then the next step captures ``_update`` over a
    static batch and every step with the same signature copies its batch in and replays. The
    learning rate and the ponder weight are device tensors filled before each step, read by the
    eager steps too. The records are read from the same outputs as the eager ones, so the log
    has one format.
    """

    def __init__(
        self,
        model: Vesper,
        opt: torch.optim.AdamW,
        cfg: VesperConfig,
        condition: str,
        beta: float,
    ) -> None:
        """Hold the run's model and optimizer; nothing is captured yet.

        Raises:
            ValueError: for a model off CUDA.
        """
        device = _device_of(model)
        if device.type != "cuda":
            msg = f"a captured step needs the model on CUDA, got {device}"
            raise ValueError(msg)
        self.model: Vesper = model
        self.opt: torch.optim.AdamW = opt
        self.cfg: VesperConfig = cfg
        self.condition: str = condition
        self.beta: float = beta
        self.lambda_h: Tensor = torch.zeros((), device=device)  # read by every act update
        self.graph: WarmedGraph = WarmedGraph()
        self._names: tuple[str, ...] = ()
        self._signature: tuple[bool, int] | None = None

    @property
    def captures(self) -> int:
        """How many captures the run has made (one per signature)."""
        return self.graph.captures

    def step(self, ep: Episodes, step: int) -> dict[str, object]:
        """One training step: eager while warming, then a replay; returns the step's record."""
        device = _device_of(self.model)
        phase = phase_schedule(
            step * int(ep.chunk.tokens.shape[0]) * SEQUENCE_TOKENS, step, self.cfg
        )
        set_learning_rate(self.opt, phase.lr)
        _ = self.lambda_h.fill_(phase.lambda_h)
        signature = _signature(self.condition, phase)
        if signature != self._signature:
            self.graph.reset()
            self._signature = signature
        c = _on(ep.chunk, device)

        def update(values: Sequence[Tensor]) -> list[Tensor]:
            out = _update(
                self.model,
                self.opt,
                self.cfg,
                self.condition,
                self.beta,
                _with_tensors(c, values),
                phase,
                lambda_h=self.lambda_h,
            )
            self._names = tuple(out.parts)
            return [out.loss, out.grad_norm, out.n_iters, *out.parts.values()]

        values = self.graph(update, chunk_tensors(c))
        parts = dict(zip(self._names, values[3:], strict=True))
        out = StepOutputs(values[0], values[1], values[2], parts)
        return _record(step, phase.lr, out, c)


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
    totals.add("static", _static_profile(model, ep))
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


def _static_profile(model: Vesper, ep: Episodes) -> Tensor:
    """``int64[T, C, 2]``: class counts with every token forced to depth d, d = 1..T.

    Copy-forward (``TraceRule``) at a constant depth: B's ``A(seed, class, j)`` (the report's
    gates) and the run's static checks (``run``'s ``static_every``), one implementation.
    """
    b, n = int(ep.chunk.tokens.shape[0]), int(ep.chunk.tokens.shape[1])
    rows: list[Tensor] = []
    for depth in range(1, model.t_max + 1):
        fixed = torch.full((b, n), depth, dtype=torch.int64)
        y, _, _ = _run_chunks(model, ep, _trace_of(fixed), n)
        rows.append(class_counts(model, y, ep))
    return torch.stack(rows)


def _static_check(model: Vesper, batches: Sequence[Episodes]) -> Tensor:
    """``_static_profile`` summed over ``batches``, read from the evaluation model (no grad)."""
    seen = evaluation_model(model)
    was_training = seen.training
    _ = seen.eval()
    with torch.no_grad():
        total = torch.stack([_static_profile(seen, ep) for ep in batches]).sum(0)
    _ = seen.train(was_training)
    return total


def _check_static(static_every: int, static_batches: int) -> None:
    """Static checks off (0) or every ``static_every`` updates over ``static_batches`` >= 1.

    Raises:
        ValueError: on a negative interval, or no batches with checks on.
    """
    if static_every < 0 or (static_every and static_batches < 1):
        msg = (
            "static checks need static_every >= 0 and static_batches >= 1, "
            f"got {static_every} and {static_batches}"
        )
        raise ValueError(msg)


def _write_static_check(
    checks: TextIO, model: Vesper, batches: Sequence[Episodes], updates: int
) -> None:
    """One static check after ``updates`` updates, one JSON line, flushed."""
    line = {"after": updates, "static": plain(_static_check(model, batches))}
    _ = checks.write(json.dumps(line) + "\n")
    checks.flush()


def _prepared(
    config: Path, out: Path, condition: str, seed: int, init_weights: Path | None
) -> tuple[bytes, VesperConfig, WarmStart | None, Vesper]:
    """The run's recipe, config, warm start and CPU model, then ``out`` with its snapshot.

    The config's bytes are read once and parsed, hashed and snapshotted from the same bytes
    (``SNAPSHOT``); the config, the warm start and the model are checked before ``out`` is
    created, so a refused run writes nothing.

    Raises:
        FileExistsError: if ``out`` exists.
        ValueError: on a config the task cannot use or a warm start that does not fit
            (``build``, ``read_warm_start``).
    """
    recipe = config.read_bytes()
    cfg = parse_bytes(recipe, str(out / SNAPSHOT))
    start = None if init_weights is None else read_warm_start(init_weights, cfg)
    model = build(cfg, condition, seed, None if start is None else start.state)
    out.mkdir(parents=True, exist_ok=False)
    _ = (out / SNAPSHOT).write_bytes(recipe)
    return recipe, cfg, start, model


def _save_snapshot(out: Path, model: Vesper, updates: int) -> None:
    """The weights after ``updates`` updates, as CPU tensors, into ``weights/after-NNNNN.pt``."""
    weights = out / "weights"
    weights.mkdir(exist_ok=True)
    torch.save(_cpu_state(model), weights / f"after-{updates:05d}.pt")


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
    out["dense_pass"] = [class_accuracy(v) for v in values["dense_pass"].unbind()]
    out["static"] = [class_accuracy(v) for v in values["static"].unbind()]
    out["random"] = class_accuracy(values["random"]) | _cost(
        values["random_passes"], t_max, columns
    )
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
            entry = class_accuracy(value)
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
        marginal, conditional = conditional_entropy(histogram)
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
    counts = {key: plain(value) for key, value in totals.values.items()}
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
            "deployed": class_accuracy(class_counts(model, y, ep)),
            "deployed_loss_passes": float(d[c.loss_mask].float().mean()),
            "dense_last": class_accuracy(class_counts(model, dense.per_pass[-1], ep)),
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
    mix: str = DEFAULT_MIX,
    device: str = "cpu",
    scan: str = "reference",
    cuda_graph: bool = False,
    data_seed: int | None = None,
    static_every: int = 0,
    static_batches: int = 8,
    init_weights: Path | None = None,
) -> dict[str, object]:
    """Train one condition and seed, evaluate it, write the run directory ``out``.

    ``seed`` draws the initialization and, unless ``data_seed`` is given, the data order
    (``generator_for("train", data_seed)``); both are recorded (``seed``, ``data_seed``), so a
    crossed-seed run (maintainer B 5dba6af) is keyed by its pair. ``static_every`` > 0 adds a static
    check after every ``static_every`` updates: ``_static_profile`` of the evaluation model
    summed over the first ``static_batches`` batches of ``eval_size`` episodes of
    ``STATIC_SPLIT``, one line ``{"after": updates, "static": counts}`` in ``STATIC_CHECKS``.

    Training runs on ``device`` with the recurrent ``scan`` (``vesper.column.set_scan``); on
    CUDA the optimizer is the capturable one, and ``cuda_graph`` replays a captured step, one
    capture per signature (``GraphedTraining``). The probes and the final evaluation always run
    on a CPU copy with the reference scan, so every run's evaluation is computed the same way;
    a CPU run evaluates
    the model itself, as before. Weights are saved as CPU tensors. The result records
    ``device``, ``scan`` (read back from the model) and ``cuda_graph``.

    ``out`` receives ``config.yaml`` (``SNAPSHOT``, recorded relative to ``result.json``: the
    config's bytes as read once, before the first step; the run parses and hashes exactly these
    bytes, so a later edit of the source cannot be recorded as the recipe, maintainer B's review of
    c7d25c5), ``train.jsonl``, ``model.pt`` and ``result.json``, and ``weights/after-NNNNN.pt``
    after every ``weights_every`` updates: the model's weights only, DIAGNOSTIC snapshots for
    learning curves and interventions, not recovery checkpoints (no optimizer, generator or
    progress state; an interrupted run starts again). The probe's gates use only the final result.
    ``mix`` names the training mix recipe of ``MIX_SCHEDULES``; the result records the name and
    its stages, so a report can bind its gates to the recipe that actually ran.

    ``init_weights`` starts the run from another run's weights (``read_warm_start``; the
    module docstring): ``seed`` then only names the run and, unless ``data_seed`` is given,
    its data order; ``initial_state_digest`` is the state the first step starts from (the
    file's weights with the condition's head) and ``init_weights`` records the file.

    Raises:
        FileExistsError: if ``out`` exists (a run directory is written once).
        ValueError: on an unknown ``mix``, evaluation split or device, a scan the device cannot
            run, ``cuda_graph`` off CUDA, a negative ``static_every`` or ``static_batches``
            below 1 with checks on, a config the task cannot use, or ``init_weights`` that are
            not a run's, come from a run with another model, or do not fit the model, before
            anything is written.
    """
    if mix not in MIX_SCHEDULES:
        msg = f"unknown mix schedule {mix!r}; known: {sorted(MIX_SCHEDULES)}"
        raise ValueError(msg)
    _check_static(static_every, static_batches)
    target = _check_backend(device, scan, condition, cuda_graph=cuda_graph)
    evaluation_set = held_out(eval_batches, eval_size, eval_split)  # refuses a bad split first
    static_set = held_out(static_batches, eval_size, STATIC_SPLIT) if static_every else []
    order = seed if data_seed is None else data_seed
    recipe, cfg, start, model = _prepared(config, out, condition, seed, init_weights)
    model = model.to(target)
    _ = set_scan(model, scan)
    planned = total_steps(cfg)
    steps = planned if steps is None else steps
    opt = optimizer(model, cfg, capturable=target.type == "cuda")
    graphed = GraphedTraining(model, opt, cfg, condition, beta) if cuda_graph else None
    data = generator_for("train", order)
    probe_set = episodes(generator_for("probe", 0), eval_size, "probe")
    initial = chained_digest("", list(_cpu_state(model).values()))
    consumed = ""
    started = time.monotonic()
    with ExitStack() as files:
        log = files.enter_context((out / "train.jsonl").open("w", encoding="utf-8"))
        checks = (
            files.enter_context((out / STATIC_CHECKS).open("w", encoding="utf-8"))
            if static_every
            else None
        )
        for step in range(steps):
            ep = episodes(
                data, cfg.training.batch_chunks, f"train-{order}-{step}", hop_mix(step, mix)
            )
            consumed = chained_digest(consumed, chunk_tensors(ep.chunk))
            if graphed is None:
                record = train_step(model, opt, cfg, condition, beta, ep, step)
            else:
                record = graphed.step(ep, step)
            if (step + 1) % weights_every == 0:
                _save_snapshot(out, model, step + 1)
            if checks is not None and (step + 1) % static_every == 0:
                _write_static_check(checks, model, static_set, step + 1)
            last = step == steps - 1
            if step % log_every == 0 or last:
                record["seconds"] = time.monotonic() - started
                record["consumed_digest"] = consumed
                record["labels"] = _labels(ep)
                if step % probe_every == 0 or last:
                    record["probe"] = _probe(evaluation_model(model), condition, probe_set)
                _ = log.write(json.dumps(record) + "\n")
                log.flush()
    torch.save(_cpu_state(model), out / "model.pt")
    trained = time.monotonic()
    evaluation = evaluate(evaluation_model(model), condition, evaluation_set)
    result: dict[str, object] = {
        "plan": "docs/reviews/2026-10-15-halting-experiment-plan.md",
        "condition": condition,
        "seed": seed,
        "data_seed": order,
        "initial_state_digest": initial,
        # the condition's own head start (build), never the file's
        "init_weights": None if start is None else start.record | {"head": "condition"},
        "consumed_digest": consumed,
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
        "mix_schedule": mix,
        "mix_stages": [[start, list(stage)] for start, stage in MIX_SCHEDULES[mix]],
        "device": device,
        "scan": scan_of(model),
        "cuda_graph": cuda_graph,
        "backend": _backend(model, cuda_graph=cuda_graph),
        "static_every": static_every,
        "static_batches": static_batches if static_every else None,
        "static_size": eval_size if static_every else None,
        "static_split": STATIC_SPLIT if static_every else None,
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
    _ = parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="the data order's seed when it differs from --seed (the initialization's)",
    )
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--config", type=Path, default=Path("configs/halting-stage1.yaml"))
    _ = parser.add_argument("--steps", type=int, default=None)
    _ = parser.add_argument("--beta", type=float, default=0.1)
    _ = parser.add_argument("--eval-batches", type=int, default=32)
    _ = parser.add_argument("--eval-split", choices=EVAL_SPLITS, default="validation")
    _ = parser.add_argument("--mix", choices=MIX_NAMES, default=DEFAULT_MIX)
    _ = parser.add_argument("--threads", type=int, default=4)
    _ = parser.add_argument("--device", choices=DEVICES, default="cpu")
    _ = parser.add_argument("--scan", choices=SCAN_IMPLS, default="reference")
    _ = parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="B-D on CUDA: capture one training step as a CUDA graph and replay it",
    )
    _ = parser.add_argument(
        "--static-every",
        type=int,
        default=0,
        help="a static-depth check after every N updates (0: none), into " + STATIC_CHECKS,
    )
    _ = parser.add_argument("--static-batches", type=int, default=8)
    _ = parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="start from a run's model.pt or weights/after-N.pt (the head: the condition's own)",
    )
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
        mix=cast("str", args.mix),
        device=cast("str", args.device),
        scan=cast("str", args.scan),
        cuda_graph=cast("bool", args.cuda_graph),
        data_seed=cast("int | None", args.data_seed),
        static_every=cast("int", args.static_every),
        static_batches=cast("int", args.static_batches),
        init_weights=cast("Path | None", args.init_weights),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
