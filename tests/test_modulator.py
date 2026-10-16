import pytest
import torch

from vesper.modulator import ModulatorState, init_state, normalize, update


def test_cold_state_returns_zeros_then_warms() -> None:
    state = init_state()
    loss = torch.full((2, 50), 3.0)
    m, state = update(state, loss, half_life_tokens=1000)
    assert torch.equal(m, torch.zeros(2, 50))
    assert state.tokens == 50
    assert state.mean.item() == pytest.approx(3.0)


def test_half_life_is_in_tokens_per_stream() -> None:
    state = init_state()
    _, state = update(state, torch.ones(4, 100), half_life_tokens=10_000)
    for _ in range(10):  # 10 chunks of 1,000 tokens per stream = exactly one half-life
        _, state = update(state, torch.full((4, 1000), 2.0), half_life_tokens=10_000)
    assert state.mean.item() == pytest.approx(1.5, abs=1e-5)


def test_normalizes_to_zero_mean_unit_scale() -> None:
    g = torch.Generator().manual_seed(0)
    state = init_state()
    for _ in range(40):
        loss = 3.0 + 0.5 * torch.randn(8, 500, generator=g)
        _, state = update(state, loss, half_life_tokens=2000)
    m = normalize(state, 3.0 + 0.5 * torch.randn(8, 5000, generator=g))
    assert m.mean().item() == pytest.approx(0.0, abs=0.05)
    assert m.std().item() == pytest.approx(1.0, abs=0.05)


def test_normalize_does_not_advance() -> None:
    state = init_state()
    _, state = update(state, torch.ones(1, 10), half_life_tokens=100)
    before = ModulatorState(state.mean.clone(), state.mean_sq.clone(), state.tokens)
    _ = normalize(state, torch.full((1, 10), 9.0))
    assert state.tokens == before.tokens
    assert torch.equal(state.mean, before.mean)


def test_rejects_bad_half_life() -> None:
    with pytest.raises(ValueError, match="half_life_tokens"):
        _ = update(init_state(), torch.ones(1, 4), half_life_tokens=0)
