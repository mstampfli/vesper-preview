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
anything on the CPU.
"""

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
