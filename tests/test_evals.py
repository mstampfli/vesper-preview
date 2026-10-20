import math
from collections.abc import Sequence
from dataclasses import replace
from typing import override

import pytest
import torch

from conftest import small_config, small_raw
from vesper.column import ColumnState, Modulation
from vesper.config import VesperConfig, parse
from vesper.data import Chunk, Document, Stream
from vesper.evals import evaluate, iteration_histogram, perplexity, static_profile
from vesper.learning import ntp_loss
from vesper.loop import DenseRule, LoopResult
from vesper.model import Vesper

CPU = torch.device("cpu")


class ScriptedEvaluation(Vesper):
    """Known probabilities and depths, with a state sentinel across chunks."""

    def __init__(self) -> None:
        super().__init__(small_config(), torch.Generator().manual_seed(0))
        self.calls: int = 0

    @override
    def forward(
        self,
        tokens: torch.Tensor,
        pos: torch.Tensor,
        states: Sequence[ColumnState],
        modulation: Modulation | None = None,
        passes: int | None = None,
        *,
        recompute: bool = False,
    ) -> tuple[torch.Tensor, LoopResult]:
        del pos, modulation, recompute
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        assert passes == 2
        assert tuple(tokens.shape) == (1, 4)
        assert all(bool((h == self.calls).all().item()) for s in states for h in s.h)
        self.calls += 1
        carried = tuple(replace(s, h=tuple(h + 1 for h in s.h)) for s in states)
        probabilities = [0.5, 0.0, 0.0, 0.0] if self.calls == 1 else [0.25, 0.125, 0.0, 0.0]
        logits = torch.full((1, 4, 64), math.nan)
        for i, p in enumerate(probabilities):
            if p:
                logits[0, i, :] = math.log((1 - p) / 63)
                logits[0, i, 0] = math.log(p)
        depths = [1, 2, 2, 2] if self.calls == 1 else [2, 1, 2, 2]
        return logits, LoopResult(
            y=torch.zeros(1, 4, 32),
            states=carried,
            n_iters=ints(*depths).reshape(1, 4),
            ponder=torch.zeros(1, 4),
            per_pass=torch.zeros(2, 1, 4, 32),
        )


def test_evaluate_counts_the_same_masked_positions_once_with_carried_states() -> None:
    model = ScriptedEvaluation()
    stream = Stream(iter([Document("a", torch.arange(1, 18, dtype=torch.int32))]), slots=1, chunk=4)
    chunks: list[Chunk] = []
    for mask in ([True, False, False, False], [True, True, False, False], [False] * 4):
        chunk = stream.next()
        assert chunk is not None
        selected = torch.zeros_like(chunk.loss_mask)
        for i, include in enumerate(mask):
            selected[0, i] = include
        chunks.append(replace(chunk, targets=torch.zeros_like(chunk.targets), loss_mask=selected))
    # The evaluator's documented float32 contract also survives an outer autocast.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = evaluate(model, chunks, small_config(), passes=2)
    assert model.calls == 3  # no second forward for depths; fully masked chunks still carry state
    assert result.tokens_with_loss == 3
    assert result.iteration_counts == (2, 1)
    assert result.iteration_histogram == [2 / 3, 1 / 3]
    assert result.loss_sum == pytest.approx(math.log(2) + math.log(4) + math.log(8))
    assert result.mean_loss == pytest.approx(math.log(4))
    assert result.perplexity == pytest.approx(4.0)


def test_evaluate_empty_and_all_masked_positions_have_no_measurement() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    chunk = Stream(iter([Document("a", ints(1, 2, 3, 4, 5))]), slots=1, chunk=4).next()
    assert chunk is not None
    masked = replace(chunk, loss_mask=torch.zeros_like(chunk.loss_mask))
    for chunks in ([], [masked]):
        result = evaluate(model, chunks, cfg, passes=2)
        assert result.loss_sum == 0.0
        assert result.tokens_with_loss == 0
        assert result.iteration_counts == (0, 0)
        assert result.iteration_histogram == [0.0, 0.0]
        assert result.mean_loss == math.inf
        assert result.perplexity == math.inf


def ints(*values: int) -> torch.Tensor:
    out = torch.zeros(len(values), dtype=torch.int32)
    for i, v in enumerate(values):
        out[i] = v
    return out


def test_iteration_histogram_counts_fractions_per_pass() -> None:
    counts = iteration_histogram([ints(1, 1, 2)], t_max=3)
    first: list[float] = [2 / 3, 1 / 3, 0.0]
    assert counts == pytest.approx(first)
    two = iteration_histogram([ints(1, 3).reshape(1, 2), ints(3)], 3)
    second: list[float] = [1 / 3, 0.0, 2 / 3]
    assert two == pytest.approx(second)
    assert iteration_histogram([], t_max=2) == [0.0, 0.0]
    with pytest.raises(ValueError, match="t_max"):
        _ = iteration_histogram([], t_max=0)


def test_perplexity_is_the_exponential_of_the_mean_loss_with_carried_states() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(4)
    docs = [
        Document(f"d{i}", torch.randint(1, 64, (40,), generator=g).to(torch.int32))
        for i in range(3)
    ]
    stream = Stream(iter(docs), slots=2, chunk=16)
    chunks: list[Chunk] = []
    while (chunk := stream.next()) is not None:
        chunks.append(chunk)
    assert len(chunks) >= 3
    value = perplexity(model, chunks, cfg, passes=2)
    assert math.isfinite(value)
    # an untrained model over 64 symbols sits near the uniform perplexity of 64
    assert 20.0 < value < 200.0
    # the same chunks again give the same number: no state leaks between calls
    assert perplexity(model, chunks, cfg, passes=2) == value
    # fewer passes is a different model: a different number
    assert perplexity(model, chunks, cfg, passes=1) != value
    assert perplexity(model, [], cfg, passes=1) == math.inf


def test_perplexity_matches_a_hand_computation_over_the_masked_positions() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [
        Document("a", torch.arange(1, 10, dtype=torch.int32)),
        Document("b", torch.arange(20, 26, dtype=torch.int32)),
    ]
    stream = Stream(iter(docs), slots=1, chunk=8)
    chunks: list[Chunk] = []
    while (chunk := stream.next()) is not None:
        chunks.append(chunk)
    assert not all(bool(c.loss_mask.all().item()) for c in chunks)  # a boundary is masked
    value = perplexity(model, chunks, cfg, passes=2)
    total = 0.0
    count = 0
    states = model.init_states(1, torch.float32, CPU)
    with torch.no_grad():
        for chunk in chunks:
            logits, result = model.forward(chunk.tokens, chunk.pos, states, passes=2)
            states = list(result.states)
            losses = ntp_loss(logits, chunk.targets)[chunk.loss_mask]
            total += float(losses.sum().item())
            count += int(losses.numel())
    assert value == pytest.approx(math.exp(total / count))


def test_perplexity_leaves_the_model_and_the_callers_states_alone() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    before = [p.detach().clone() for p in model.parameters()]
    docs = [Document("a", torch.arange(1, 20, dtype=torch.int32))]
    stream = Stream(iter(docs), slots=1, chunk=8)
    chunks: list[Chunk] = []
    while (chunk := stream.next()) is not None:
        chunks.append(chunk)
    _ = perplexity(model, chunks, cfg, passes=2)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    assert all(p.grad is None for p in model.parameters())


def three_pass_config() -> VesperConfig:
    raw = small_raw()
    raw["model"]["t_max"] = 3
    return parse(raw, "test")


def test_the_static_profile_is_every_fixed_depth_from_one_dense_traversal() -> None:
    # the real-data pilot's phase-1 check (docs/reviews/2026-10-20-real-data-halting-pilot.md):
    # the held-out loss if every token stopped at depth j, for every j at once. With copy-forward
    # a token's pass-j output reads only the states of passes 1..j, so one dense traversal gives
    # each depth; the check here runs every depth on its own, states carried across chunks.
    cfg = three_pass_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [Document("a", torch.arange(1, 40, dtype=torch.int32) % 60 + 1)]
    stream = Stream(iter(docs), slots=1, chunk=8)
    chunks: list[Chunk] = []
    while (chunk := stream.next()) is not None:
        chunks.append(chunk)
    assert len(chunks) > 2
    profile = static_profile(model, chunks, cfg)
    count = sum(int(c.loss_mask.sum().item()) for c in chunks)
    assert profile.tokens_with_loss == count
    assert len(profile.loss_sums) == cfg.model.t_max
    with torch.no_grad():
        for depth in range(1, cfg.model.t_max + 1):
            states = model.init_states(1, torch.float32, CPU)
            total = 0.0
            for chunk in chunks:
                result = model.run(
                    chunk.tokens, chunk.pos, states, passes=depth, exit_rule=DenseRule()
                )
                states = list(result.states)
                losses = ntp_loss(model.head(result.y), chunk.targets)[chunk.loss_mask]
                total += float(losses.sum().item())
            assert profile.loss_sums[depth - 1] == pytest.approx(total, rel=1e-5), depth
    assert len(set(profile.mean_losses)) == cfg.model.t_max  # three different depths, not one
    assert all(p.grad is None for p in model.parameters())


def test_an_empty_static_profile_has_no_measurement() -> None:
    profile = static_profile(
        Vesper(three_pass_config(), torch.Generator().manual_seed(0)), [], three_pass_config()
    )
    assert (profile.tokens_with_loss, profile.mean_losses) == (0, [math.inf] * 3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the histogram's inputs come from the GPU"
)
def test_iteration_histogram_accepts_pass_counts_on_the_gpu() -> None:
    cuda = torch.device("cuda")
    expected: list[float] = [1 / 3, 2 / 3]
    assert iteration_histogram([ints(1, 2, 2).to(cuda)], t_max=2) == pytest.approx(expected)
