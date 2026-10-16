import math

import pytest
import torch

from vesper.contracts import ContractError
from vesper.recurrence import write_strength

ALPHA = 0.5


def grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Logits ``[2, 5, 3]`` across the sigmoid's range and ``m [2, 5]`` across the clip range."""
    gen = torch.Generator().manual_seed(11)
    logits = torch.randn(2, 5, 3, dtype=torch.float64, generator=gen) * 6
    m = torch.rand(2, 5, dtype=torch.float64, generator=gen) * 4 - 1  # uniform in [-1, 3]
    return logits, m


def test_without_modulator_is_the_sigmoid() -> None:
    logits, _ = grid()
    assert torch.equal(write_strength(logits), torch.sigmoid(logits))


def test_zero_modulation_changes_nothing() -> None:
    logits, m = grid()
    assert torch.allclose(write_strength(logits, torch.zeros_like(m), ALPHA), torch.sigmoid(logits))


def test_modulation_scales_the_odds() -> None:
    logits, m = grid()
    beta = torch.sigmoid(logits)
    factor = (1 + ALPHA * m)[..., None]
    expected = beta * factor / (1 + beta * (factor - 1))
    assert torch.allclose(write_strength(logits, m, ALPHA), expected, rtol=1e-12, atol=0)


def test_small_beta_recovers_the_plain_product() -> None:
    logits = torch.full((1, 1, 1), -8.0, dtype=torch.float64)
    m = torch.full((1, 1), 3.0, dtype=torch.float64)
    beta = torch.sigmoid(logits)
    plain = beta * (1 + ALPHA * 3.0)
    ratio = (write_strength(logits, m, ALPHA) / plain).item()
    assert abs(ratio - 1) < 1e-3


def test_stays_inside_the_unit_interval_where_the_plain_product_leaves_it() -> None:
    # beta 0.9 at m = 3: the spec's product gives 2.25 and an erase eigenvalue of -1.25
    logits = torch.full((1, 1, 1), math.log(0.9 / 0.1), dtype=torch.float64)
    m = torch.full((1, 1), 3.0, dtype=torch.float64)
    beta_eff = write_strength(logits, m, ALPHA)
    assert torch.allclose(beta_eff, torch.full_like(beta_eff, 2.25 / 2.35), rtol=1e-12, atol=0)
    assert float(beta_eff.item()) < 1.0
    assert 1.0 - float(beta_eff.item()) >= 0.0  # erase eigenvalue on a unit key


@pytest.mark.parametrize("logit", [-50.0, -3.0, 0.0, 3.0, 50.0])
@pytest.mark.parametrize("m_value", [-1.0, 0.0, 1.5, 3.0])
def test_bounds_hold_across_the_clip_range(logit: float, m_value: float) -> None:
    logits = torch.full((1, 1, 1), logit, dtype=torch.float32)
    m = torch.full((1, 1), m_value, dtype=torch.float32)
    beta_eff = float(write_strength(logits, m, ALPHA).item())
    assert 0.0 <= beta_eff <= 1.0


def test_monotone_in_the_modulator() -> None:
    logits = torch.linspace(-6, 6, 7, dtype=torch.float64).reshape(1, 7, 1)
    low = write_strength(logits, torch.full((1, 7), -1.0, dtype=torch.float64), ALPHA)
    mid = write_strength(logits, torch.zeros(1, 7, dtype=torch.float64), ALPHA)
    high = write_strength(logits, torch.full((1, 7), 3.0, dtype=torch.float64), ALPHA)
    assert bool((low <= mid).all().item())
    assert bool((mid <= high).all().item())
    assert bool((low < high).all().item())


def test_alpha_zero_is_the_identity() -> None:
    logits, m = grid()
    assert torch.equal(write_strength(logits, m, 0.0), torch.sigmoid(logits))


def test_interior_mapping_differs_from_the_plain_product() -> None:
    # beta 1/2 (logit 0), alpha 1/2, m 1: the product gives 3/4, the odds rule 3/5
    logits = torch.zeros(1, 1, 1, dtype=torch.float64)
    m = torch.ones(1, 1, dtype=torch.float64)
    beta_eff = write_strength(logits, m, ALPHA)
    assert torch.allclose(beta_eff, torch.full_like(beta_eff, 0.6), rtol=1e-12, atol=0)
    assert float(beta_eff.item()) != 0.75


@pytest.mark.parametrize("logit", [0.0, 50.0, 1e4])
def test_zero_multiplier_is_an_exact_zero_write_with_zero_gradient(logit: float) -> None:
    # alpha 1 is in the spec's ablation; at the lower clip m = -1 the odds multiplier is zero
    logits = torch.full((1, 1, 2), logit, dtype=torch.float32, requires_grad=True)
    m = torch.full((1, 1), -1.0, dtype=torch.float32, requires_grad=True)
    beta_eff = write_strength(logits, m, 1.0)
    assert torch.equal(beta_eff, torch.zeros_like(beta_eff))
    g_logits, g_m = torch.autograd.grad(beta_eff.sum(), (logits, m))
    assert torch.equal(g_logits, torch.zeros_like(logits))
    assert torch.equal(g_m, torch.zeros_like(m))


def test_zero_multiplier_in_one_token_leaves_the_others_alone() -> None:
    logits = torch.zeros(1, 2, 1, dtype=torch.float64)
    m = torch.zeros(1, 2, dtype=torch.float64)
    m[0, 1] = -2.0  # 1 + 0.5 * (-2) = 0 for the second token only
    beta_eff = write_strength(logits, m, ALPHA)
    assert float(beta_eff[0, 0, 0].item()) == 0.5
    assert float(beta_eff[0, 1, 0].item()) == 0.0


def test_modulator_requires_an_explicit_finite_non_negative_alpha() -> None:
    logits, m = grid()
    with pytest.raises(ValueError, match="alpha must be given"):
        _ = write_strength(logits, m)
    with pytest.raises(ValueError, match="non-negative"):
        _ = write_strength(logits, m, -0.5)
    with pytest.raises(ValueError, match="non-negative"):
        _ = write_strength(logits, m, math.inf)
    assert torch.equal(write_strength(logits), torch.sigmoid(logits))  # the plain call stays


def test_rejects_a_negative_multiplier() -> None:
    logits = torch.zeros(1, 2, 3)
    m = torch.zeros(1, 2)
    m[0, 1] = -3.0  # 1 + 0.5 * (-3) < 0
    with pytest.raises(ValueError, match="negative"):
        _ = write_strength(logits, m, ALPHA)
    with pytest.raises(ValueError, match="negative"):
        _ = write_strength(logits, torch.full((1, 2), -1.5), 1.0)


def test_rejects_a_modulator_of_the_wrong_shape() -> None:
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ContractError):
        _ = write_strength(logits, torch.zeros(1, 3), ALPHA)
    with pytest.raises(ContractError):
        _ = write_strength(torch.zeros(1, 2), torch.zeros(1, 2), ALPHA)
