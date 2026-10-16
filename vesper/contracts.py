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
    if tensor.numel() > 0 and bool(((tensor < 0) | (tensor >= vocab)).any().item()):
        msg = f"{name}: every id must lie in [0, vocab={vocab})"
        raise ContractError(msg)
    return tensor.long()
