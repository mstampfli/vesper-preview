"""Evaluations that run at every checkpoint (spec: Evaluation suite), each a plain function.

Two of the suite's metrics exist with the first training loop: the held-out perplexity and the
iteration histogram. The others (needle at distance, session learning, dusk gain, joules) need
the ledger, dusk or the energy meter and keep their registry rows as specified.
"""

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional

from vesper.column import ColumnState, scan_of, state_tensors, with_state_tensors
from vesper.config import VesperConfig
from vesper.data import Chunk
from vesper.graphs import SignatureGraph, signature
from vesper.loop import DenseRule
from vesper.model import Vesper


@dataclass(frozen=True)
class Evaluation:
    """Loss and depth measured on the same positions in one evaluation traversal.

    Counts index passes ``1..cfg.model.t_max``, including zeroes for unused passes.
    Raw integer counts and the loss sum retain their unrounded values for aggregation.
    """

    loss_sum: float
    tokens_with_loss: int
    iteration_counts: tuple[int, ...]

    @property
    def mean_loss(self) -> float:
        """Mean next-token loss; ``inf`` when no position carries a loss."""
        return self.loss_sum / self.tokens_with_loss if self.tokens_with_loss else math.inf

    @property
    def perplexity(self) -> float:
        """Exponential of the mean next-token loss, or ``inf`` without observations."""
        return math.exp(self.mean_loss)

    @property
    def iteration_histogram(self) -> list[float]:
        """Unrounded fractions of the measured positions at each pass."""
        return _fractions(self.iteration_counts)


@dataclass
class EvalGraphs:
    """The evaluations' captured chunk passes on CUDA, held across a run's checkpoints.

    One ``SignatureGraph`` per kind of pass (the policy forward of ``evaluate``, the dense
    traversal of ``depth_losses``), so alternating the two never re-warms; the weights are
    updated in place between checkpoints, so a captured pass stays valid for the run. A carried
    pass's FIRST chunk starts from cold states whose attention windows are empty, a signature
    of its own, so each carried kind has a second graph for it (``policy_first``,
    ``dense_first``): short streams, one or two chunks per call (a document at a time), would
    otherwise switch the one graph's key on every call and never capture (2026-10-28: the
    per-document policy evaluation ran eager). The project owner 2026-10-21: CUDA graphs for
    ANYTHING on the GPU, so a CUDA evaluation without them is refused.
    """

    policy: SignatureGraph = field(default_factory=SignatureGraph)
    policy_first: SignatureGraph = field(default_factory=SignatureGraph)
    dense: SignatureGraph = field(default_factory=SignatureGraph)  # depth_losses
    dense_first: SignatureGraph = field(default_factory=SignatureGraph)
    static: SignatureGraph = field(default_factory=SignatureGraph)  # static_profile's sums
    static_first: SignatureGraph = field(default_factory=SignatureGraph)
    rule: SignatureGraph = field(default_factory=SignatureGraph)  # a deployed exit rule, cold
    trace: SignatureGraph = field(default_factory=SignatureGraph)  # replayed depths, cold

    def counts(self) -> dict[str, list[int]]:
        """Each graph's ``[eager, captures, replays]`` by field name (for a record).

        Whether a CUDA evaluation ran captured is measured here, never assumed (2026-10-29: a
        run's deferred evaluations recorded no counts, so their capture was unknown). The eager
        count tells a pass that ran without a graph (warm-up only) from one never run.
        """
        graphs = {  # every field, by its name (a test checks the two lists agree)
            "policy": self.policy,
            "policy_first": self.policy_first,
            "dense": self.dense,
            "dense_first": self.dense_first,
            "static": self.static,
            "static_first": self.static_first,
            "rule": self.rule,
            "trace": self.trace,
        }
        return {name: [g.eager, g.captures, g.replays] for name, g in graphs.items()}


PassFn = Callable[[Sequence[Tensor], list[ColumnState]], list[Tensor]]


def run_chunks(  # noqa: PLR0913
    model: Vesper,
    chunks: Sequence[Chunk],
    graph: SignatureGraph | None,
    outputs: int,
    pass_fn: PassFn,
    *,
    carried: bool,
    switches: tuple[object, ...] = (),
    extras: Sequence[Sequence[Tensor]] | None = None,
    running: Sequence[Tensor] | None = None,
    first: SignatureGraph | None = None,
) -> list[list[Tensor]]:
    """PRIMITIVE a no-gradient pass over every chunk, captured on CUDA; each chunk's results.

    ``pass_fn(inputs, states)`` gets ``[tokens, pos, targets, *extra]`` on the device (``extras``
    gives each chunk's extra tensors, e.g. a trace of depths: values, never constants frozen
    into the graph) and returns its ``outputs`` result tensors, then the new state tensors
    (``column.state_tensors``). ``carried``: the states run from chunk to chunk (cold at the
    first); else every chunk starts cold (independent contexts). A carried pass's first chunk
    runs under ``first`` (required with ``graph`` on CUDA): its cold states have empty attention
    windows, another signature than the carried chunks', and one graph for both would re-warm
    at every switch and, for short streams, never capture. No gradient, float32 (autocast off).
    ``switches`` name the pass's kind in the capture signature. The results are copied out of
    the graph's buffers and stay on the device until the caller reads them once; the carried
    states are the graph's outputs, read as the next chunk's inputs (copied in) before that
    graph replays again.

    ``running``: an aggregate streamed through the chunks (e.g. zero sums). The pass then also
    gets the running values after its other inputs and returns their updated values as its
    ``outputs`` results (same count, shapes and dtypes); only the latest are kept, and the
    result is the one row of final values (``running`` itself when there is no chunk). So the
    storage is the aggregate's whatever the number of chunks (maintainer B 9e3066c: a row per chunk
    is O(chunks * T)). Without it, every chunk's results are returned, for callers that need them.

    For: every evaluation on the GPU (the project owner 2026-10-21: CUDA graphs for ANYTHING on the
    GPU). Not for: training (``vesper.train.step``, ``joint_update``), which owns its graphs.

    Raises:
        ValueError: on CUDA without ``graph``, or carried without ``first`` (no GPU evaluation
            outside a graph, which still warms up eagerly before it captures); when
            ``running`` has another count than ``outputs`` or a pass changes
            a running value's shape or dtype.
    """
    results, _ = _chunk_loop(
        model,
        chunks,
        graph,
        outputs,
        pass_fn,
        carried=carried,
        switches=switches,
        extras=extras,
        running=running,
        first=first,
        initial=None,
    )
    return results


def run_carried(  # noqa: PLR0913
    model: Vesper,
    chunks: Sequence[Chunk],
    graph: SignatureGraph | None,
    outputs: int,
    pass_fn: PassFn,
    *,
    first: SignatureGraph | None,
    initial: Sequence[ColumnState] | None = None,
    switches: tuple[object, ...] = (),
) -> tuple[list[list[Tensor]], list[ColumnState]]:
    """PRIMITIVE ``run_chunks`` carried, from given states, returning its final states too.

    The same chunk loop as ``run_chunks(..., carried=True)`` (one implementation), with two
    differences: ``initial`` (states after an earlier call) replaces the cold start, its first
    chunk then running under ``graph`` like every carried chunk (``first`` is only for a cold
    start, required on CUDA then); and the states after the last chunk come back as CLONES,
    owned by the caller, so several continuations can start from the same states without one
    inheriting another's (each passes them as values, which a graph copies into its inputs).

    For: branching a carried stream, e.g. scoring several continuations of one saved prefix
    state (``vesper.recall``). Not for: running aggregates (``run_chunks``' ``running``).

    Raises:
        ValueError: as ``run_chunks``; on no chunks.
    """
    if not chunks:
        msg = "run_carried needs at least one chunk"
        raise ValueError(msg)
    results, carry = _chunk_loop(
        model,
        chunks,
        graph,
        outputs,
        pass_fn,
        carried=True,
        switches=switches,
        extras=None,
        running=None,
        first=first,
        initial=initial,
    )
    if carry is None:  # carried with at least one chunk always sets it
        msg = "run_carried: the chunk loop returned no states"
        raise RuntimeError(msg)
    owned = [t.clone() for t in state_tensors(carry)]  # the graph overwrites its outputs
    return results, with_state_tensors(carry, owned)


def _chunk_loop(  # noqa: PLR0913
    model: Vesper,
    chunks: Sequence[Chunk],
    graph: SignatureGraph | None,
    outputs: int,
    pass_fn: PassFn,
    *,
    carried: bool,
    switches: tuple[object, ...],
    extras: Sequence[Sequence[Tensor]] | None,
    running: Sequence[Tensor] | None,
    first: SignatureGraph | None,
    initial: Sequence[ColumnState] | None,
) -> tuple[list[list[Tensor]], list[ColumnState] | None]:
    """The chunk loop of ``run_chunks`` and ``run_carried``: results, and the last carry.

    The carry refers to the last graph's output buffers (``run_carried`` clones it).
    """
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    if cuda and (graph is None or (carried and first is None and initial is None)):
        msg = (
            "a CUDA evaluation runs as captured CUDA graphs: pass EvalGraphs, and a carried pass"
            " a graph for its cold first chunk (eager warm-up, then capture and replays)"
        )
        raise ValueError(msg)
    if cuda and scan_of(model) == "reference":
        msg = "the reference scan cannot run as a captured CUDA graph; set a fused scan"
        raise ValueError(msg)
    current = None if running is None else _running_layout(running, outputs, device)
    results: list[list[Tensor]] = []
    carry: list[ColumnState] | None = None if initial is None else list(initial)
    for index, chunk in enumerate(chunks):
        b = int(chunk.tokens.shape[0])
        cold = model.init_states(b, torch.float32, device)
        templates = carry if carried and carry is not None else cold
        inputs = [chunk.tokens.to(device), chunk.pos.to(device), chunk.targets.to(device)]
        if extras is not None:
            inputs += [t.to(device) for t in extras[index]]
        if current is not None:
            inputs += current
        values = inputs + state_tensors(templates)
        n_inputs = len(inputs)

        def run(
            v: Sequence[Tensor], templates: list[ColumnState] = templates, n: int = n_inputs
        ) -> list[Tensor]:
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                return pass_fn(v[:n], with_state_tensors(templates, v[n:]))

        chunk_graph = first if carried and index == 0 and initial is None else graph
        out: Sequence[Tensor] = (
            run(values)
            if chunk_graph is None
            else chunk_graph(signature(values, *switches), run, values)
        )
        kept = [t.clone() for t in out[:outputs]]  # graph buffers are overwritten
        if current is None:
            results.append(kept)
        else:
            current = _running_layout(kept, outputs, device, like=current)
        if carried:
            carry = with_state_tensors(templates, list(out[outputs:]))
    return (results if current is None else [current]), carry


def _running_layout(
    values: Sequence[Tensor],
    outputs: int,
    device: torch.device,
    like: Sequence[Tensor] | None = None,
) -> list[Tensor]:
    """``values`` on ``device``, checked as ``run_chunks``' running aggregate.

    There must be ``outputs`` of them and, given ``like`` (the previous chunk's), with the same
    shapes and dtypes.
    """
    if len(values) != outputs:
        msg = f"running has {len(values)} values, the pass returns {outputs}"
        raise ValueError(msg)
    for old, new in zip(like or values, values, strict=True):
        if new.shape != old.shape or new.dtype != old.dtype:
            msg = f"a running value changed from {old.shape} {old.dtype} to {new.shape} {new.dtype}"
            raise ValueError(msg)
    return [t.to(device) for t in values]


def evaluate(
    model: Vesper,
    chunks: Sequence[Chunk],
    cfg: VesperConfig,
    passes: int,
    *,
    graphs: EvalGraphs | None = None,
) -> Evaluation:
    """Measure next-token loss and pass counts together over positions with a loss.

    ``chunks`` are consecutive chunks of one held-out stream (the same slots throughout), run
    with cold states carried from chunk to chunk and ``passes`` loop passes, no gradient, in
    float32 (even under an outer autocast). Every chunk runs once, as a captured CUDA graph on
    CUDA (``graphs`` required there); its loss mask selects both losses and pass counts, added
    on the device to running totals (a float64 loss sum, a label count, a pass histogram) that
    are read back once at the end. Empty or fully masked inputs give zero sums/counts and
    infinite mean loss/perplexity.
    """
    t_max = cfg.model.t_max

    def policy(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        # each chunk added on the device to the running float64 loss sum, label count and pass
        # histogram: O(T_max) metric storage whatever the number of chunks (maintainer B's reviews
        # of 571b5f9 and af63bf4: per-token outputs, then a row per chunk, were retained)
        tokens, pos, targets, mask, loss_sum, labels, histogram = inputs
        result = model.run(tokens, pos, states, passes=passes)
        per_token = model.position_losses(result.y, targets).double()
        chunk_sum = torch.where(mask, per_token, torch.zeros_like(per_token)).sum()
        depth = functional.one_hot(result.n_iters.long().clamp(0, t_max), t_max + 1)
        chunk_histogram = (depth * mask.unsqueeze(-1).long()).sum(dim=(0, 1))
        return [
            loss_sum + chunk_sum,
            labels + mask.sum(),
            histogram + chunk_histogram,
            *state_tensors(result.states),
        ]

    masks = [[chunk.loss_mask] for chunk in chunks]
    zeros = [
        torch.zeros((), dtype=torch.float64),
        torch.zeros((), dtype=torch.int64),
        torch.zeros(t_max + 1, dtype=torch.int64),
    ]
    ((total, count, depths),) = run_chunks(
        model,
        chunks,
        None if graphs is None else graphs.policy,
        3,
        policy,
        carried=True,
        switches=("policy", passes),
        extras=masks,
        running=zeros,
        first=None if graphs is None else graphs.policy_first,
    )
    return Evaluation(float(total), int(count), _pass_totals(depths.cpu()))


def depth_losses(
    model: Vesper,
    chunks: Sequence[Chunk],
    *,
    carried: bool,
    graphs: EvalGraphs | None = None,
) -> Tensor:
    """``float64 [T, N]`` on the CPU: every label's next-token loss at every fixed depth.

    One dense traversal per chunk (``DenseRule``; with copy-forward, pass ``j`` of a dense run
    is exactly the run in which every token stops at ``j``), the head applied one depth at a
    time, labels in chunk order then row-major. ``carried``: the states run from chunk to chunk
    (a run's held-out stream, ``static_profile``); else each chunk starts cold (independent
    contexts, the text pilot's phase 2). Captured on CUDA (``graphs`` required there), read
    back once. The one per-depth measurement the static profile, phase 2's fixed depths and the
    paired per-token profiles all use.
    """
    t_max = model.loop.t_max

    def dense(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        tokens, pos, targets = inputs
        result = model.run(tokens, pos, states, passes=t_max, exit_rule=DenseRule())
        losses = torch.stack(
            [model.position_losses(result.per_pass[d], targets) for d in range(t_max)]
        )
        return [losses, *state_tensors(result.states)]

    graph = None if graphs is None else graphs.dense
    first = None if graphs is None or not carried else graphs.dense_first
    switches = ("dense", carried, t_max)
    out = run_chunks(
        model, chunks, graph, 1, dense, carried=carried, switches=switches, first=first
    )
    columns = [
        losses.cpu()[:, chunk.loss_mask].double()
        for (losses,), chunk in zip(out, chunks, strict=True)
    ]
    return torch.cat(columns, dim=1) if columns else torch.zeros(t_max, 0, dtype=torch.float64)


@dataclass(frozen=True)
class StaticProfile:
    """The held-out next-token loss at every fixed depth ``1..T`` on the same positions."""

    loss_sums: tuple[float, ...]
    tokens_with_loss: int

    @property
    def mean_losses(self) -> list[float]:
        """Mean loss per depth; ``inf`` everywhere when no position carries a loss."""
        n = self.tokens_with_loss
        return [s / n if n else math.inf for s in self.loss_sums]


def static_profile(
    model: Vesper,
    chunks: Sequence[Chunk],
    cfg: VesperConfig,
    *,
    graphs: EvalGraphs | None = None,
) -> StaticProfile:
    """The loss if every token stopped at depth ``j``, for every ``j``, from one dense traversal.

    The same dense traversal as ``depth_losses`` with the states carried (``chunks`` are
    consecutive chunks of one held-out stream, as in ``evaluate``), but each chunk's masked
    losses are added on the device to running float64 sums per depth, so the storage is
    ``[T]`` whatever the number of chunks; the per-label matrix exists only where a caller
    asks for it (``depth_losses``; maintainer B 9e3066c). For the halting pilot's phase-1 check
    (docs/reviews/2026-10-20-real-data-halting-pilot.md): does the loss fall with depth?
    """
    t_max = model.loop.t_max
    if t_max != cfg.model.t_max:
        msg = f"the model runs {t_max} passes, the config says {cfg.model.t_max}"
        raise ValueError(msg)

    def dense_sums(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        tokens, pos, targets, mask, sums = inputs
        result = model.run(tokens, pos, states, passes=t_max, exit_rule=DenseRule())
        losses = torch.stack(
            [model.position_losses(result.per_pass[d], targets) for d in range(t_max)]
        ).double()
        chunk_sums = torch.where(mask, losses, torch.zeros_like(losses)).sum(dim=(1, 2))
        return [sums + chunk_sums, *state_tensors(result.states)]

    masks = [[chunk.loss_mask] for chunk in chunks]
    ((sums,),) = run_chunks(
        model,
        chunks,
        None if graphs is None else graphs.static,
        1,
        dense_sums,
        carried=True,
        switches=("static", t_max),
        extras=masks,
        running=[torch.zeros(t_max, dtype=torch.float64)],
        first=None if graphs is None else graphs.static_first,
    )
    count = sum(int(chunk.loss_mask.sum()) for chunk in chunks)
    return StaticProfile(tuple(float(s) for s in sums.cpu().unbind()), count)


def perplexity(model: Vesper, chunks: Sequence[Chunk], cfg: VesperConfig, passes: int) -> float:
    """Compatibility float API for ``evaluate(...).perplexity``; one traversal."""
    return evaluate(model, chunks, cfg, passes).perplexity


def iteration_histogram(n_iters: Sequence[Tensor], t_max: int) -> list[float]:
    """The fraction of tokens that halted at each pass ``1..t_max``, over the given counts.

    ``n_iters`` are the loop's ``int32`` pass counts of the tokens to include (the caller selects
    the positions with a loss). An empty input gives all zeros.

    Raises:
        ValueError: if ``t_max`` is below 1.
    """
    counts = _iteration_counts(n_iters, t_max)
    return _fractions(_pass_totals(counts))


def _iteration_counts(n_iters: Iterable[Tensor], t_max: int) -> Tensor:
    """Shared integer accumulation; transfer only per-pass counts from an accelerator."""
    if t_max < 1:
        msg = f"t_max must be at least 1, got {t_max}"
        raise ValueError(msg)
    counts = torch.zeros(t_max + 1, dtype=torch.int64)
    for part in n_iters:
        flat = part.reshape(-1).to(dtype=torch.int64)
        counts += torch.bincount(flat, minlength=t_max + 1)[: t_max + 1].to(device="cpu")
    return counts


def _fractions(counts: Sequence[int]) -> list[float]:
    """Normalize exact pass counts without rounding the returned fractions."""
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [n / total for n in counts]


def _pass_totals(counts: Tensor) -> tuple[int, ...]:
    """Convert the CPU int64 bins for passes 1..T_max to exact Python integers."""
    return tuple(int(counts[t].item()) for t in range(1, counts.numel()))
