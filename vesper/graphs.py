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

import os
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager

import torch
from torch import Tensor

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

    def __init__(self, step: Callable[[], Sequence[Tensor]], inputs: Sequence[Tensor]) -> None:
        """Capture ``step``, which must read its batch only from ``inputs`` (CUDA tensors).

        Capturing records the work without running it: the first ``replay`` is the first time
        it runs.

        Raises:
            ContractError: if an input is not a CUDA tensor.
        """
        for index, tensor in enumerate(inputs):
            if not tensor.is_cuda:
                msg = f"captured input {index} is on {tensor.device}, not CUDA"
                raise ContractError(msg)
        self.inputs: tuple[Tensor, ...] = tuple(inputs)
        self._graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
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

    def __init__(self, warm: int = WARM_STEPS) -> None:
        """Nothing is captured yet; the first ``warm`` calls run eagerly."""
        self.warm: int = warm
        self.captures: int = 0
        self.replays: int = 0
        self._eager: int = 0
        self._captured: CapturedStep | None = None

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
            self._captured = CapturedStep(lambda: step(static), static)
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

    def __init__(self, warm: int = WARM_STEPS) -> None:
        """Nothing captured yet."""
        self.warm: int = warm
        self.captures: int = 0
        self.replays: int = 0
        self._key: tuple[object, ...] | None = None
        self._graph: WarmedGraph | None = None

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
            self._key, self._graph = key, WarmedGraph(self.warm)
        graph = self._graph
        captures, replays = graph.captures, graph.replays
        out = graph(step, values)
        self.captures += graph.captures - captures
        self.replays += graph.replays - replays
        return out


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


def signature(values: Sequence[Tensor], *switches: object) -> tuple[object, ...]:
    """A ``SignatureGraph`` key: the ``switches`` and every value's shape, dtype and device."""
    return (*switches, tuple((tuple(v.shape), v.dtype, v.device) for v in values))
