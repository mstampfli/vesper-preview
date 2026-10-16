"""Tensor contracts: the primitives every module uses to check its boundary.

PRIMITIVE ``check_shape``: asserts a tensor's rank, its named dimensions and, when asked, its
dtype at a function boundary. Dimension names are bound on first sight and enforced afterwards,
so every tensor of one call must agree on ``b``, ``n``, ``h`` and ``dk``.

Use it for: the inputs and outputs listed in docs/registry.md, once per public call.
Not for: value checks (ranges, finiteness, normalization) and not inside inner loops. It checks
shape and dtype only. A failed contract raises ``ContractError`` naming the tensor, the axis and
both values, which is the message a kernel author needs.

PRIMITIVE ``check_ids``: the model's integer boundary. The data module stores token ids as
``int32``; lookups and losses need ``int64`` and ids inside ``[0, vocab)``. This is the one place
where the shape is bound, the dtype is checked, the range is checked and the widening happens.

Use it for: every token or target tensor entering ``vesper.embedding`` or ``vesper.learning``.
Not for: positions (``int64`` by contract, ``check_shape`` with a dtype), float tensors, or any
integer tensor without a vocabulary.

PRIMITIVE ``require``: a value contract on device data that also holds inside a captured CUDA
graph (a device-side assert there, ``ContractError`` everywhere else). Every data-dependent
check on the training step's path goes through it; a bare ``.item()`` there ends a capture.
"""

import torch

Dims = dict[str, int]
DimSpec = tuple[str | int, ...]


class ContractError(ValueError):
    """A tensor did not meet its contract (rank, dimension binding or dtype)."""


def check_shape(
    name: str,
    tensor: torch.Tensor,
    spec: DimSpec,
    dims: Dims,
    dtype: torch.dtype | None = None,
) -> None:
    """Check ``tensor`` against ``spec`` and bind the unbound dimension names in ``dims``.

    Args:
        name: the tensor's name, used in the error message.
        tensor: the tensor to check.
        spec: one entry per axis, either a dimension name or a literal size.
        dims: the bindings shared by all tensors of one call; extended in place.
        dtype: when given, the exact dtype the tensor must have.

    Raises:
        ContractError: on a rank, dimension or dtype mismatch.
    """
    shape = tuple(tensor.shape)
    if len(shape) != len(spec):
        msg = f"{name}: expected rank {len(spec)} {spec}, got shape {shape}"
        raise ContractError(msg)
    for axis, (want, got) in enumerate(zip(spec, shape, strict=True)):
        if isinstance(want, int):
            if got != want:
                msg = f"{name}: axis {axis} must be {want}, got {got} (shape {shape})"
                raise ContractError(msg)
            continue
        bound = dims.get(want)
        if bound is None:
            dims[want] = got
        elif bound != got:
            msg = f"{name}: axis {axis} '{want}' is {got}, but '{want}' = {bound} elsewhere"
            raise ContractError(msg)
    if dtype is not None and tensor.dtype != dtype:
        msg = f"{name}: dtype must be {dtype}, got {tensor.dtype}"
        raise ContractError(msg)


def check_ids(
    name: str, tensor: torch.Tensor, spec: DimSpec, dims: Dims, vocab: int
) -> torch.Tensor:
    """Check an id tensor against ``spec`` and ``[0, vocab)`` and return it as ``int64``.

    Args:
        name: the tensor's name, used in the error message.
        tensor: ``int32`` or ``int64`` ids.
        spec: one entry per axis, as for ``check_shape``.
        dims: the bindings shared by all tensors of one call; extended in place.
        vocab: the exclusive upper bound of a valid id.

    Returns:
        The same ids as ``int64`` (the tensor itself when it already is).

    Raises:
        ContractError: on a shape mismatch, a dtype other than int32 or int64, or an id outside
            ``[0, vocab)``.
    """
    check_shape(name, tensor, spec, dims)
    if tensor.dtype not in (torch.int32, torch.int64):
        msg = f"{name}: dtype must be int32 or int64, got {tensor.dtype}"
        raise ContractError(msg)
    if tensor.numel() > 0:
        inside = ((tensor >= 0) & (tensor < vocab)).all()
        require(inside, f"{name}: every id must lie in [0, vocab={vocab})")
    return tensor.long()


def require(condition: torch.Tensor, message: str) -> None:
    """Hold a value contract computed on the device: raise ``ContractError`` unless ``condition``.

    PRIMITIVE value check. ``condition`` is a one-element bool tensor (``(x >= 0).all()``,
    ``mask.any()``). Eagerly the check reads it back and raises ``ContractError(message)``. While
    a CUDA graph is being captured on the current stream, a read-back is not allowed (it would
    end the capture), so the check is recorded as a device-side assert instead: it runs on every
    replay and a violation stops the process with ``message`` (a CUDA error, not an exception
    the step can catch). The contract holds in both modes; only the way it fails differs.

    Use it for: every data-dependent check on a tensor in the training step (id ranges,
    consecutive positions, a non-empty loss mask, a dense forward). Not for: shapes and dtypes
    (``check_shape``, which never touches the data) or checks on Python values.

    Raises:
        ContractError: eagerly, if ``condition`` is false; or if it is not a one-element bool.
    """
    if condition.dtype != torch.bool or condition.numel() != 1:
        shape = tuple(condition.shape)
        msg = f"require: the condition must be one bool, got {condition.dtype} {shape}"
        raise ContractError(msg)
    if condition.is_cuda and torch.cuda.is_current_stream_capturing():
        # torch's only capturable assert is private; pinned by tests/test_contracts.py, which
        # replays a captured check on good and on bad data
        torch._assert_async(condition, message)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage, reportPrivateImportUsage]
        return
    if not bool(condition.item()):
        raise ContractError(message)
