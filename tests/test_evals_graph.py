"""Every evaluation on CUDA runs as captured graphs, equal to the same passes run eagerly.

The project owner 2026-10-21: CUDA graphs for ANYTHING on the GPU. Each test runs the real
evaluation twice on CUDA, captured (``EvalGraphs()``) and as the eager oracle (every
``SignatureGraph`` with a warm count larger than the work, so it never captures), and requires
bitwise-equal results AND captures and replays on the captured side (an evaluation that silently
stayed eager would pass the equality alone).
"""

import pytest
import torch

from conftest import small_raw
from vesper.column import set_scan
from vesper.config import VesperConfig, parse
from vesper.data import Chunk, Document, Stream
from vesper.evals import EvalGraphs, depth_losses, evaluate, static_profile
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
        dense=SignatureGraph(NEVER),
        rule=SignatureGraph(NEVER),
        trace=SignatureGraph(NEVER),
    )


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
