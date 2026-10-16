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
and pass count are fixed per capture. Not for: evaluation, or anything on the CPU.
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
