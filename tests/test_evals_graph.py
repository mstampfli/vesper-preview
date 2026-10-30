"""Every evaluation on CUDA runs as captured graphs, equal to the same passes run eagerly.

The project owner 2026-10-21: CUDA graphs for ANYTHING on the GPU. Each test runs the real
evaluation twice on CUDA, captured (``EvalGraphs()``) and as the eager oracle (every
``SignatureGraph`` with a warm count larger than the work, so it never captures), and requires
bitwise-equal results AND captures and replays on the captured side (an evaluation that silently
stayed eager would pass the equality alone).
"""

from collections.abc import Sequence

import pytest
import torch

from conftest import small_raw
from vesper.column import ColumnState, set_scan
from vesper.config import VesperConfig, parse
from vesper.data import Chunk, Document, Stream
from vesper.evals import EvalGraphs, depth_losses, evaluate, run_chunks, static_profile
from vesper.graphs import SignatureGraph
from vesper.model import Vesper

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
CUDA = torch.device("cuda")
NEVER = 10**9


def _config() -> VesperConfig:
    raw = small_raw()
    raw["model"]["t_max"] = 3
    return parse(raw, "test")


def _chunks(count: int) -> list[Chunk]:
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 60, (90,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(6)
    ]
    stream = Stream(iter(docs), slots=2, chunk=16)
    chunks: list[Chunk] = []
    while len(chunks) < count and (chunk := stream.next()) is not None:
        chunks.append(chunk)
    assert len(chunks) == count
    return chunks


def _eager() -> EvalGraphs:
    return EvalGraphs(
        policy=SignatureGraph(NEVER),
        policy_first=SignatureGraph(NEVER),
        dense=SignatureGraph(NEVER),
        dense_first=SignatureGraph(NEVER),
        static=SignatureGraph(NEVER),
        static_first=SignatureGraph(NEVER),
        rule=SignatureGraph(NEVER),
        trace=SignatureGraph(NEVER),
    )


def _document_chunks(lengths: list[int]) -> list[list[Chunk]]:
    """One single-slot stream per document, chunks of 16: a document of n tokens is
    ceil(n / 16) chunks, as the per-document Pythia comparison streams them."""
    out: list[list[Chunk]] = []
    for i, n in enumerate(lengths):
        tokens = torch.randint(1, 60, (n,), generator=torch.Generator().manual_seed(100 + i))
        stream = Stream(iter([Document(f"doc{i}", tokens.to(torch.int32))]), slots=1, chunk=16)
        chunks: list[Chunk] = []
        while (chunk := stream.next()) is not None:
            chunks.append(chunk)
        out.append(chunks)
    return out


def test_short_carried_documents_capture_both_chunk_kinds_and_equal_the_eager_ones() -> None:
    # documents of one, two and three chunks in turn, three single-chunk ones in a row first
    # (so a first-chunk graph is captured and then a second chunk follows it): every carried
    # evaluation, called once per document with one EvalGraphs, equals the eager oracle's, and
    # after the warm-ups both chunk kinds REPLAY (2026-10-28: with one graph per kind the key
    # switched on every document and nothing was ever captured)
    cfg = _config()
    model = _model(cfg)
    docs = _document_chunks([10, 12, 15, 20, 40, 9, 30, 14, 44, 11, 25])
    captured, eager = EvalGraphs(), _eager()
    for chunks in docs:
        assert static_profile(model, chunks, cfg, graphs=captured) == static_profile(
            model, chunks, cfg, graphs=eager
        )
        assert evaluate(model, chunks, cfg, passes=3, graphs=captured) == evaluate(
            model, chunks, cfg, passes=3, graphs=eager
        )
        got = depth_losses(model, chunks, carried=True, graphs=captured)
        assert torch.equal(got, depth_losses(model, chunks, carried=True, graphs=eager))
    firsts = len(docs)
    rests = sum(len(c) - 1 for c in docs)
    assert rests >= 4
    for graph, calls in (
        (captured.policy_first, firsts),
        (captured.policy, rests),
        (captured.static_first, firsts),
        (captured.static, rests),
        (captured.dense_first, firsts),
        (captured.dense, rests),
    ):
        assert graph.captures == 1
        assert graph.replays == calls - 3  # three eager warm-ups, then capture-and-replay
    assert eager.policy.captures == eager.policy_first.captures == 0


def _model(cfg: VesperConfig) -> Vesper:
    model = Vesper(cfg, torch.Generator().manual_seed(0)).to(CUDA)
    _ = set_scan(model, "fused-fp32")  # the reference scan cannot be captured
    with torch.no_grad():  # a head that halts some tokens early, so the policy's depths vary
        for p in model.loop.halting_head().parameters():
            _ = p.normal_(0.0, 0.5, generator=torch.Generator(device=CUDA).manual_seed(3))
    return model


@pytest.mark.parametrize("carried", [True, False])
def test_captured_depth_losses_equal_the_eager_ones(carried: bool) -> None:  # noqa: FBT001
    cfg = _config()
    model, chunks = _model(cfg), _chunks(9)
    captured = EvalGraphs()
    got = depth_losses(model, chunks, carried=carried, graphs=captured)
    want = depth_losses(model, chunks, carried=carried, graphs=_eager())
    assert captured.dense.captures >= 1
    assert captured.dense.replays >= 4
    assert got.shape == (cfg.model.t_max, sum(int(c.loss_mask.sum()) for c in chunks))
    assert torch.equal(got, want)


def test_a_captured_policy_evaluation_and_static_profile_equal_the_eager_ones() -> None:
    cfg = _config()
    model, chunks = _model(cfg), _chunks(9)
    captured = EvalGraphs()
    got = evaluate(model, chunks, cfg, passes=3, graphs=captured)
    want = evaluate(model, chunks, cfg, passes=3, graphs=_eager())
    assert captured.policy.captures >= 1
    assert captured.policy.replays >= 4
    assert got == want
    assert len([c for c in got.iteration_counts if c]) > 1  # the policy's depths really vary
    got_profile = static_profile(model, chunks, cfg, graphs=captured)
    want_profile = static_profile(model, chunks, cfg, graphs=_eager())
    assert got_profile == want_profile


def test_a_cuda_evaluation_without_graphs_is_refused() -> None:
    cfg = _config()
    model, chunks = _model(cfg), _chunks(2)
    with pytest.raises(ValueError, match="captured CUDA graphs"):
        _ = evaluate(model, chunks, cfg, passes=3)
    with pytest.raises(ValueError, match="captured CUDA graphs"):
        _ = depth_losses(model, chunks, carried=False)


def test_a_carried_cuda_pass_without_a_first_chunk_graph_is_refused() -> None:
    cfg = _config()
    model, chunks = _model(cfg), _chunks(2)

    def nothing(_inputs: Sequence[torch.Tensor], _states: list[ColumnState]) -> list[torch.Tensor]:
        msg = "refused before any pass"
        raise AssertionError(msg)

    with pytest.raises(ValueError, match="cold first chunk"):
        _ = run_chunks(model, chunks, SignatureGraph(), 1, nothing, carried=True)
