"""One training step captured as a CUDA graph and replayed (the small model's dispatch cost).

PRIMITIVE captured step. A stage-1 step is about 6,700 small GPU launches whose work takes about
13 ms of a 110 ms step; the rest is Python and the dispatcher issuing them one by one. A CUDA
graph records the launches once and replays them with one call. ``CapturedStep`` owns that: the
step function, run once under capture, reads only the static input tensors given to it and
returns its outputs; ``replay`` copies a new batch into those inputs (same shapes and dtypes,
checked), replays, and returns the same output tensors, overwritten.

What the step may do: any CUDA work without a read-back to the host (``vesper.contracts.require``
records its checks as device asserts while capturing), no data-dependent shapes, no new Python
decisions per batch (a decision taken at capture is replayed forever), an optimizer built
``capturable`` whose learning rate is filled in place (``vesper.train``). The caller runs the
first steps eagerly inside ``side_stream`` before capturing, as CUDA graphs require (lazy
initialization and allocator warm-up must not happen during capture); those are real steps.

For: the stage-1 training step on CUDA (``vesper.halting_experiment --cuda-graph``), every
condition; ACT's loop has no host sync (all passes run, halted tokens are masked), its head freeze
and pass count are fixed per capture; the halting calibration's update (head-only and joint).
``WarmedGraph`` is the warm-capture-replay life both drivers share. Not for: evaluation, or
anything on the CPU. ``deterministic_algorithms`` makes a CUDA training loop repeat itself,
captured or eager (measured 2026-10-19: without it two runs with equal inputs diverged).
"""

import json
import os
import time
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import Tensor
from torch.cuda import _POOL_HANDLE  # pyright: ignore[reportPrivateUsage]  # graph_pool_handle's type

from vesper.contracts import ContractError


@contextmanager
def side_stream() -> Generator[None]:
    """Run the enclosed eager steps on a fresh CUDA stream, then join it to the current one."""
    stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch leaves it unannotated
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        yield
    torch.cuda.current_stream().wait_stream(stream)


class CapturedStep:
    """A step function captured once over static inputs; ``replay`` runs it on new values."""

    def __init__(
        self,
        step: Callable[[], Sequence[Tensor]],
        inputs: Sequence[Tensor],
        pool: _POOL_HANDLE | None = None,
    ) -> None:
        """Capture ``step``, which must read its batch only from ``inputs`` (CUDA tensors).

        Capturing records the work without running it: the first ``replay`` is the first time
        it runs. ``pool``: capture into that shared memory pool (``GraphFamily``), else the
        graph's own.

        Raises:
            ContractError: if an input is not a CUDA tensor.
        """
        for index, tensor in enumerate(inputs):
            if not tensor.is_cuda:
                msg = f"captured input {index} is on {tensor.device}, not CUDA"
                raise ContractError(msg)
        self.inputs: tuple[Tensor, ...] = tuple(inputs)
        self._graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=pool):
            outputs = tuple(step())
        self.outputs: tuple[Tensor, ...] = outputs

    def replay(self, values: Sequence[Tensor]) -> tuple[Tensor, ...]:
        """Copy ``values`` into the static inputs, replay, and return the (overwritten) outputs.

        Raises:
            ContractError: on a different number of values or a value whose shape or dtype is
                not its input's (a captured graph has fixed shapes; nothing is broadcast).
        """
        if len(values) != len(self.inputs):
            msg = f"{len(values)} values for {len(self.inputs)} captured inputs"
            raise ContractError(msg)
        for index, (static, value) in enumerate(zip(self.inputs, values, strict=True)):
            if value.shape != static.shape or value.dtype != static.dtype:
                msg = (
                    f"captured input {index} is {static.dtype} {tuple(static.shape)}, "
                    f"got {value.dtype} {tuple(value.shape)}"
                )
                raise ContractError(msg)
            _ = static.copy_(value)
        self._graph.replay()
        return self.outputs


WARM_STEPS = 3  # eager steps on a side stream before a capture (lazy init, autotuning)
# the cuBLAS workspace settings its documentation names as reproducible; a preset outside them
# is refused, an unset variable is the measured default (below)
CUBLAS_WORKSPACES = (":4096:8", ":16:8")


@contextmanager
def deterministic_algorithms() -> Generator[dict[str, object]]:
    """PRIMITIVE repeatable GPU training: torch's strict deterministic algorithms for the block.

    Measured 2026-10-19 on the calibration (torch 2.14.1+cu130, RTX 5060 Ti): two CUDA runs with
    bitwise-equal inputs diverged within the first updates, eager and captured, with the fused
    or the reference scan. With ``CUBLAS_WORKSPACE_CONFIG`` unset (cuBLAS's default 32 MiB
    workspace here), the repository's test repeats two captured 120-update trajectories in
    fresh processes, including warmup and an evaluation copy. Exported 30-update probes on the
    real source compare eager and captured execution with ``:4096:8``; their recorded metrics
    and final weight digests agree. The measured cost was a few percent per captured update
    under load. torch
    substitutes deterministic implementations without raising, so this does not name the op that
    differed and does not certify third-party kernels in general: the guarantee is the measured
    workload, software and card.

    Inside: deterministic algorithms on with ``warn_only`` off (an op without a deterministic
    implementation raises). On exit, exceptions included, BOTH previous settings (enabled and
    warn_only) are restored, so nesting and callers see nothing change. Yields the effective
    settings for the run's record: both flags, the cuBLAS variable as found, and the cuBLAS
    workspace size in effect (which an explicit ``torch.backends.cuda.cublas_workspace_size``
    call sets over the variable).

    For: a training loop on CUDA whose runs must repeat. Not for: CPU work (unchanged) or code
    that must stay fast at any cost.

    Raises:
        RuntimeError: if ``CUBLAS_WORKSPACE_CONFIG`` is set to a value outside
            ``CUBLAS_WORKSPACES`` (a preset is never silently kept or overridden).
    """
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is not None and workspace not in CUBLAS_WORKSPACES:
        allowed = ", ".join(CUBLAS_WORKSPACES)
        msg = f"CUBLAS_WORKSPACE_CONFIG must be unset or one of {allowed}, got {workspace!r}"
        raise RuntimeError(msg)
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = fill_uninitialized_memory()
    torch.use_deterministic_algorithms(True, warn_only=False)  # noqa: FBT003  # torch's flag
    # torch's deterministic mode also fills every new tensor with NaN so that a read of
    # uninitialized memory repeats; measured 2026-10-21 on the 5090: ~2700 fill kernels of a
    # ~6000-kernel captured step. It is off here on this evidence only: the separate-process
    # repeat test and the captured == eager tests pass bitwise with it off, so the paths THEY run
    # read no memory they did not write (an uninitialized read there would differ between runs).
    # It is not a proof for other paths; an op reading unwritten memory elsewhere would not be
    # caught by them (maintainer B's review of 571b5f9).
    _set_fill_uninitialized_memory(value=False)
    try:
        yield {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "fill_uninitialized_memory": fill_uninitialized_memory(),
            "cublas_workspace_config": workspace,
            "cublas_workspace_size": (
                torch.backends.cuda.cublas_workspace_size() if torch.cuda.is_available() else None
            ),
        }
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        _set_fill_uninitialized_memory(value=fill)


def fill_uninitialized_memory() -> bool:
    """Torch's ``torch.utils.deterministic.fill_uninitialized_memory``, typed.

    Torch defines it on a module subclass the type checkers cannot see.
    """
    value: object = torch.utils.deterministic.fill_uninitialized_memory  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    return bool(value)  # pyright: ignore[reportUnknownArgumentType]


def _set_fill_uninitialized_memory(*, value: bool) -> None:
    torch.utils.deterministic.fill_uninitialized_memory = value  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


class WarmedGraph:
    """PRIMITIVE warm, capture, replay: one training update's whole life as a CUDA graph.

    Each call runs ``step(values)``: the first ``warm`` calls after construction or ``reset``
    eagerly inside ``side_stream`` (real updates, as CUDA graphs require before a capture), the
    next one captures ``step`` over owned copies of ``values`` (``CapturedStep``) and replays it
    at once on those values, and every later call copies its values in and replays. The outputs
    are the step's tensors; after a replay they are the graph's, overwritten by the next call, so
    the caller reads them first. What ``step`` may do is ``CapturedStep``'s contract; a value
    that changes between updates (a learning rate, a floor) is a device tensor among ``values``
    or filled in place before the call, never a Python number read inside ``step``. The step
    function of the capturing call is the one replayed: later calls' ``step`` is not called.

    For: a driver's update on CUDA (``vesper.halting_experiment.GraphedTraining``,
    ``vesper.halting_calibration.train_update``). Not for: work whose shapes or Python
    decisions change between calls without a ``reset``.
    """

    def __init__(self, warm: int = WARM_STEPS, pool: _POOL_HANDLE | None = None) -> None:
        """Nothing captured yet; the first ``warm`` calls run eagerly; captures go to ``pool``."""
        self.warm: int = warm
        self.captures: int = 0
        self.replays: int = 0
        self._eager: int = 0
        self._captured: CapturedStep | None = None
        self._pool: _POOL_HANDLE | None = pool

    def reset(self) -> None:
        """Drop the capture: the next ``warm`` calls are eager again, then a new capture."""
        self._eager, self._captured = 0, None

    def __call__(
        self, step: Callable[[Sequence[Tensor]], Sequence[Tensor]], values: Sequence[Tensor]
    ) -> tuple[Tensor, ...]:
        """One update: eager while warming, else a replay over ``values`` (CUDA tensors).

        Raises:
            ContractError: if a value is not on CUDA, or (on a replay) not shaped and typed like
                the captured one.
        """
        for index, value in enumerate(values):
            if not value.is_cuda:
                msg = f"value {index} is on {value.device}, not CUDA"
                raise ContractError(msg)
        if self._eager < self.warm:
            self._eager += 1
            with side_stream():
                return tuple(step(values))
        if self._captured is None:
            static = [value.clone() for value in values]  # owned buffers, never the caller's
            self._captured = CapturedStep(lambda: step(static), static, self._pool)
            self.captures += 1
        self.replays += 1
        return self._captured.replay(values)


class SignatureGraph:
    """PRIMITIVE one ``WarmedGraph`` per work signature: GPU work whose shape changes rarely.

    Each call names its signature ``key`` (everything the captured work depends on beyond the
    values copied in: shapes, dtypes, a phase, which inputs exist). The same key replays the
    current graph; a new key drops it and warms and captures a new one, so the memory of only
    one graph is held. ``captures`` counts captures over the object's life, ``replays`` replays.

    For: work that moves forward through few signatures and repeats each many times (the
    trainer's steps across phases and the tail, the evaluations' chunks after the first one's
    empty windows; the project owner 2026-10-21: CUDA graphs for ANYTHING on the GPU). Not for: work
    that alternates between signatures (each switch re-warms; hold one ``SignatureGraph`` per
    alternating kind instead). ``warm`` larger than the work never captures: the eager oracle an
    equality test compares against, not a run mode.
    """

    def __init__(self, warm: int = WARM_STEPS, pool: _POOL_HANDLE | None = None) -> None:
        """Nothing captured yet; every capture goes to ``pool`` (``GraphFamily``) if given."""
        self.warm: int = warm
        self.captures: int = 0
        self.replays: int = 0
        self._key: tuple[object, ...] | None = None
        self._graph: WarmedGraph | None = None
        self._pool: _POOL_HANDLE | None = pool

    def __call__(
        self,
        key: tuple[object, ...],
        step: Callable[[Sequence[Tensor]], Sequence[Tensor]],
        values: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        """``step(values)`` under ``key``'s graph: eager while warming, then captured, replayed.

        Raises:
            ContractError: as ``WarmedGraph``.
        """
        if key != self._key or self._graph is None:
            self._key, self._graph = key, WarmedGraph(self.warm, self._pool)
        graph = self._graph
        captures, replays = graph.captures, graph.replays
        out = graph(step, values)
        self.captures += graph.captures - captures
        self.replays += graph.replays - replays
        return out


class GraphFamily:
    """PRIMITIVE one ``SignatureGraph`` per key, every member captured into ONE memory pool.

    Members never run at the same time (one stream, one call at a time), so their working
    memory can be the same memory: freed working memory of one member's capture is reused by the
    others, so the family can hold much less than the sum of separate pools; what each member
    keeps (its outputs) and the allocator's block overhead stay, so the size is not bounded by
    the largest member (the test measures one saving, not a bound; maintainer B, ad39b41). Measured
    need: the instruction-tuning preflight of 709df83 ran out of
    memory on the 16 GB card with one separate pool per batch length for training and for
    evaluation. ``captures`` and ``replays`` per key are the members' own counters.

    For: alternating kinds of GPU work with distinct shapes, e.g. one graph per batch length,
    whose outputs the caller reads or copies before the next replay of ANY member (a later
    member's replay may reuse that memory). Not for: outputs that must survive another member's
    replay, members run on concurrent streams, or one shape repeated (a plain ``SignatureGraph``).
    """

    def __init__(self, warm: int = WARM_STEPS) -> None:
        """No member and no pool yet: the pool is made at the first member, on CUDA only."""
        self.warm: int = warm
        self.members: dict[object, SignatureGraph] = {}
        self._pool: _POOL_HANDLE | None = None

    def __call__(self, key: object) -> SignatureGraph:
        """The member for ``key``, made (in the family's pool) on first use."""
        if key not in self.members:
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            self.members[key] = SignatureGraph(self.warm, self._pool)
        return self.members[key]

    def counts(self) -> dict[str, list[int]]:
        """Each member's ``[captures, replays]``, by ``str(key)`` in key order (for a record)."""
        named = {str(k): [g.captures, g.replays] for k, g in self.members.items()}
        return {name: named[name] for name in sorted(named)}


def reset_memory_peak(device: torch.device) -> None:
    """Start a new ``memory_peak`` window on ``device`` (nothing on the CPU)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_peak(device: torch.device) -> int:
    """PRIMITIVE the bytes a run holds on ``device``: the allocator's peak RESERVATION.

    Since the last ``reset_memory_peak``, the most the caching allocator held from the card,
    which includes every captured graph's private pool. Not the peak of allocated bytes: a
    replay allocates nothing, so a window of replays reports only the resting tensors (the
    5090 sweep of 2026-10-21 read 0.99 GiB for steps that held 15 GB). Reserved bytes can
    exceed what live tensors need by the allocator's cached blocks; that is memory the process
    holds and another job cannot have. 0 on the CPU, which has no counter.

    For: every reported memory figure of a GPU run (training steps, drivers, the throughput
    sweep). Not for: the size of one tensor or of one graph alone.
    """
    if device.type != "cuda":
        return 0
    return torch.cuda.max_memory_reserved(device)


class MemoryPhases:
    """PRIMITIVE where a GPU run's memory goes: one record per named phase of the run.

    Each ``phase(name)`` synchronizes the device, reads the allocator's current allocated and
    reserved bytes, starts a new peak window (``reset_peak_memory_stats``, which resets BOTH
    peaks), and at its end synchronizes again and reads both current values, both peaks and the
    given graph family's ``[captures, replays]`` (that family only: graphs the phase makes
    elsewhere are not counted). The record is appended to ``log`` (one JSON line) as the phase
    ends, also when it ends in an exception, which it records and re-raises; a counter read that
    fails is recorded as unreadable, and every reading taken before it is kept (maintainer B,
    420fffb; the phase's own exception takes precedence over the read's). Phases do
    not nest (a nested reset would shorten the outer window) and are never entered or left while
    the current stream captures. ``reserved_peak()`` is the maximum over the phases' reserved
    peaks: for phases that cover a run without gaps it is the run's ``memory_peak``.

    What the numbers are: allocator statistics. The allocated peak counts tensors only, and a
    replay allocates nothing (``memory_peak``); a captured graph's private pool must keep its
    addresses across replays, so neither peak is the smallest card the captured run fits on.
    Nothing here empties caches or changes a graph's lifetime. On the CPU the memory fields are
    None (no counters); names, seconds and errors are still recorded.

    For: a run's per-phase memory record (the instruction tuning's ``memory.jsonl``). Not for:
    timing kernels (it synchronizes), or code inside a captured step.
    """

    def __init__(
        self, device: torch.device, log: Path | None = None, graphs: GraphFamily | None = None
    ) -> None:
        """No phase yet; ``log`` (if given) receives one JSON line per ended phase."""
        self.device: torch.device = device
        self.log: Path | None = log
        self.graphs: GraphFamily | None = graphs
        self.records: list[dict[str, object]] = []
        self._open: str | None = None

    def _readings(self) -> dict[str, int | None]:
        if self.device.type != "cuda":
            return {"allocated": None, "reserved": None}
        torch.cuda.synchronize(self.device)
        return {
            "allocated": torch.cuda.memory_allocated(self.device),
            "reserved": torch.cuda.memory_reserved(self.device),
        }

    def _peaks(self) -> dict[str, int | None]:
        if self.device.type != "cuda":
            return {"allocated_peak": None, "reserved_peak": None}
        return {
            "allocated_peak": torch.cuda.max_memory_allocated(self.device),
            "reserved_peak": torch.cuda.max_memory_reserved(self.device),
        }

    @staticmethod
    def _not_capturing(name: str) -> None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            msg = f"memory phase {name!r} at a boundary inside a CUDA graph capture"
            raise RuntimeError(msg)

    @contextmanager
    def phase(self, name: str) -> Generator[None]:
        """Measure the enclosed work as phase ``name`` (see the class docstring)."""
        if self._open is not None:
            msg = f"memory phase {name!r} inside phase {self._open!r}: phases do not nest"
            raise RuntimeError(msg)
        self._not_capturing(name)
        entry = self._readings()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self._open = name
        started = time.monotonic()
        error: str | None = None
        try:
            yield
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            self._open = None
            # every reading is kept as soon as it is taken: a later failed read loses nothing
            record: dict[str, object] = {
                "phase": name,
                "error": error,
                "allocated_entry": entry["allocated"],
                "reserved_entry": entry["reserved"],
            }
            failed_read: Exception | None = None
            try:
                self._not_capturing(name)
                exit_ = self._readings()
                record |= {"allocated_exit": exit_["allocated"], "reserved_exit": exit_["reserved"]}
                record |= self._peaks()
            except Exception as read_exc:  # noqa: BLE001  # recorded; raised below if nothing else is
                record["unreadable"] = repr(read_exc)
                failed_read = read_exc
            record["graphs"] = None if self.graphs is None else self.graphs.counts()
            record["seconds"] = time.monotonic() - started
            self.records.append(record)
            if self.log is not None:
                with self.log.open("a") as handle:
                    _ = handle.write(json.dumps(record) + "\n")
            if failed_read is not None and error is None:
                raise failed_read  # the phase's own exception, if any, propagates instead

    def reserved_peak(self) -> int:
        """The largest reserved peak over the recorded phases (0 with none, or on the CPU)."""
        peaks = [r.get("reserved_peak") for r in self.records]
        return max((p for p in peaks if isinstance(p, int)), default=0)


def signature(values: Sequence[Tensor], *switches: object) -> tuple[object, ...]:
    """A ``SignatureGraph`` key: the ``switches`` and every value's shape, dtype and device."""
    return (*switches, tuple((tuple(v.shape), v.dtype, v.device) for v in values))
