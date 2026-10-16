import subprocess
import sys

import pytest
import torch

from vesper.contracts import ContractError, Dims, check_ids, check_shape, require


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


def test_require_raises_eagerly_and_refuses_a_condition_that_is_not_one_bool() -> None:
    require(torch.tensor(True), "never")
    with pytest.raises(ContractError, match="the ids are bad"):
        require(torch.tensor(False), "the ids are bad")
    with pytest.raises(ContractError, match="one bool"):
        require(torch.ones(2, dtype=torch.bool), "two")
    with pytest.raises(ContractError, match="one bool"):
        require(torch.tensor(1), "an int")


_CAPTURED_REQUIRE = """
import sys, torch
from vesper.contracts import require
x = torch.ones(4, device="cuda")
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    require((x > 0).all(), "warm")
torch.cuda.current_stream().wait_stream(side)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    require((x > 0).all(), "x must stay positive")
    y = x * 2
graph.replay()
torch.cuda.synchronize()
print("good replay", float(y.sum()))
if sys.argv[1] == "bad":
    x.fill_(-1.0)
    graph.replay()
    torch.cuda.synchronize()
    print("bad replay passed")
"""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("case", ["good", "bad"])
def test_require_inside_a_captured_graph_checks_every_replay(case: str) -> None:
    # a device-side assert ends its CUDA context, so each case runs in its own interpreter;
    # eagerly a capture would have died at the read-back instead of recording the check
    done = subprocess.run(  # noqa: S603  (our own interpreter and a fixed script)
        [sys.executable, "-c", _CAPTURED_REQUIRE, case],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert "good replay 8.0" in done.stdout, done.stderr
    if case == "good":
        assert done.returncode == 0, done.stderr
    else:
        assert done.returncode != 0
        assert "bad replay passed" not in done.stdout
        assert "x must stay positive" in done.stderr
