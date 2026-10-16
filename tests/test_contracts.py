import pytest
import torch

from vesper.contracts import ContractError, Dims, check_ids, check_shape


def test_binds_then_enforces_named_dims() -> None:
    dims: Dims = {}
    check_shape("q", torch.zeros(2, 5, 3, 4), ("b", "n", "h", "dk"), dims)
    assert dims == {"b": 2, "n": 5, "h": 3, "dk": 4}
    check_shape("beta", torch.zeros(2, 5, 3), ("b", "n", "h"), dims)
    with pytest.raises(ContractError, match="'h' is 7"):
        check_shape("v", torch.zeros(2, 5, 7, 4), ("b", "n", "h", "dk"), dims)


def test_rank_mismatch() -> None:
    with pytest.raises(ContractError, match="expected rank 3"):
        check_shape("x", torch.zeros(2, 3), ("a", "b", "c"), {})


def test_literal_dim() -> None:
    check_shape("x", torch.zeros(2, 4), ("b", 4), {})
    with pytest.raises(ContractError, match="axis 1 must be 4"):
        check_shape("x", torch.zeros(2, 5), ("b", 4), {})


def test_dtype() -> None:
    check_shape("x", torch.zeros(2, dtype=torch.float16), ("b",), {}, dtype=torch.float16)
    with pytest.raises(ContractError, match=r"dtype must be torch\.float16"):
        check_shape("x", torch.zeros(2), ("b",), {}, dtype=torch.float16)


def test_ids_are_widened_bound_and_range_checked() -> None:
    dims: Dims = {}
    ids32 = torch.arange(6, dtype=torch.int32).view(2, 3)
    out = check_ids("tokens", ids32, ("b", "n"), dims, vocab=6)
    assert out.dtype == torch.int64
    assert torch.equal(out, ids32.long())
    assert dims == {"b": 2, "n": 3}
    same = check_ids("targets", ids32.long(), ("b", "n"), dims, vocab=6)
    assert same.dtype == torch.int64
    empty = check_ids("tokens", torch.zeros(1, 0, dtype=torch.int32), ("b", "n"), {}, vocab=6)
    assert empty.shape == (1, 0)


def test_ids_reject_wrong_dtype_range_and_shape() -> None:
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = check_ids("tokens", torch.zeros(2, 3), ("b", "n"), {}, vocab=6)
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = check_ids("tokens", torch.zeros(2, 3, dtype=torch.bool), ("b", "n"), {}, vocab=6)
    with pytest.raises(ContractError, match=r"\[0, vocab=6\)"):
        _ = check_ids("tokens", torch.full((2, 3), 6, dtype=torch.int32), ("b", "n"), {}, vocab=6)
    with pytest.raises(ContractError, match=r"\[0, vocab=6\)"):
        _ = check_ids("tokens", torch.full((2, 3), -1), ("b", "n"), {}, vocab=6)
    dims: Dims = {"n": 4}
    with pytest.raises(ContractError, match="'n' is 3"):
        _ = check_ids("tokens", torch.zeros(2, 3, dtype=torch.int32), ("b", "n"), dims, vocab=6)
