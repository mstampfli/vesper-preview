"""The exit distribution for all-pass supervision (the halting experiment plan, 2026-10-15).

The fixtures are maintainer B's adversarial review's (docs/reviews/2026-10-15-halting-objective-
adversarial.md), recomputed here as an independent oracle: uniform exits from the fixed hazard
baseline, local against aggregate KL, the CDF policy's jumps, the two-pass full-loss gradient.
"""

import math

import pytest
import torch

from vesper.contracts import ContractError
from vesper.exits import (
    exit_log_probs,
    expected_pass_loss,
    hazard_baseline,
    hazard_cdf_exits,
    hazard_logits,
    kl_uniform_aggregate,
    kl_uniform_local,
)

F64 = torch.float64


def _baseline(t_max: int) -> torch.Tensor:
    """The uniform hazard offsets written out independently: -log(T - t), then 0 at T."""
    return _vec([-math.log(t_max - t) for t in range(1, t_max)] + [0.0])


def _z(t_max: int, b: int = 1, n: int = 1) -> torch.Tensor:
    """Zero learned logits plus the uniform baseline, the last pass's hazard left at zero."""
    return torch.zeros(t_max, b, n, dtype=F64) + _baseline(t_max)[:, None, None]


def _vec(values: list[float], dtype: torch.dtype = F64) -> torch.Tensor:
    """A 1-D tensor from a list, element by element (strict typing has no typed constructor)."""
    out = torch.zeros(len(values), dtype=dtype)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _log_p_of(rows: list[list[float]]) -> torch.Tensor:
    """Exit distributions given per token as rows -> log p [T, 1, tokens]."""
    p = torch.stack([_vec(row) for row in rows], dim=1)[:, None, :]
    return torch.log(p)


def _exits(*values: int) -> torch.Tensor:
    """The expected ``hazard_cdf_exits`` result for one row of tokens."""
    return _vec([float(v) for v in values], torch.int32)[None, :]


@pytest.mark.parametrize("t_max", [1, 2, 8])
def test_zero_learned_logits_give_exactly_uniform_exits(t_max: int) -> None:
    p = exit_log_probs(_z(t_max)).exp()[:, 0, 0]
    assert torch.allclose(p, torch.full((t_max,), 1.0 / t_max, dtype=F64), atol=1e-15)
    assert torch.equal(hazard_baseline(t_max), _baseline(t_max))
    assert torch.equal(hazard_baseline(1), torch.zeros(1, dtype=F64))
    with pytest.raises(ValueError, match="at least 1"):
        _ = hazard_baseline(0)


def test_the_distribution_sums_to_one_and_the_last_hazard_is_unused() -> None:
    z = torch.randn(8, 3, 5, generator=torch.Generator().manual_seed(0), dtype=F64) * 4
    log_p = exit_log_probs(z)
    assert torch.allclose(log_p.exp().sum(0), torch.ones(3, 5, dtype=F64), atol=1e-12)
    changed = z.clone()
    changed[-1] = 1e6  # the forced final exit has no hazard of its own
    assert torch.equal(exit_log_probs(changed), log_p)


def test_extreme_logits_stay_finite_forward_and_backward() -> None:
    z = _vec([1000.0, -1000.0, 40.0, -40.0, 0.0, 0.0, 0.0, 0.0])
    z = z[:, None, None].repeat(1, 1, 2).requires_grad_()
    log_p = exit_log_probs(z)
    assert bool(torch.isfinite(log_p).all())
    loss = log_p.exp().sum() + kl_uniform_local(log_p, torch.ones(1, 2, dtype=torch.bool))
    (grad,) = torch.autograd.grad(loss, [z])
    assert bool(torch.isfinite(grad).all())


def test_local_kl_taxes_token_specific_routing_and_aggregate_kl_does_not() -> None:
    # Maintainer B's fixture: eight tokens, T = 8; all uniform against one token at each depth
    mask = torch.ones(1, 8, dtype=torch.bool)
    uniform = _log_p_of([[1 / 8] * 8 for _ in range(8)])
    eps = 1e-300  # log(0) stays finite in float64 and contributes 0 * log 0 = 0 to the KL
    distinct = _log_p_of([[1.0 - 7 * eps if t == i else eps for t in range(8)] for i in range(8)])
    assert kl_uniform_local(uniform, mask).item() == pytest.approx(0.0, abs=1e-12)
    assert kl_uniform_aggregate(uniform, mask).item() == pytest.approx(0.0, abs=1e-12)
    assert kl_uniform_local(distinct, mask).item() == pytest.approx(math.log(8), abs=1e-9)
    assert kl_uniform_aggregate(distinct, mask).item() == pytest.approx(0.0, abs=1e-9)
    # and the CDF policy at q = 0.5 on hazard logits: everyone at pass 4 against one token at
    # each depth (token i's hazard saturates at pass i, the last token takes the forced exit)
    assert torch.equal(hazard_cdf_exits(hazard_logits(torch.zeros(8, 1, 8)), 0.5), _exits(*[4] * 8))
    saturated = torch.full((8, 1, 8), -60.0, dtype=F64)
    for i in range(7):
        saturated[i, 0, i] = 60.0
    assert torch.equal(hazard_cdf_exits(saturated, 0.5), _exits(*range(1, 9)))


def test_the_cdf_policy_jumps_where_the_expected_depth_barely_moves() -> None:
    # Maintainer B's fixture: expected depth 4.57 exits at 8, expected depth 4.43 exits at 1 (p_1 =
    # 0.49 or 0.51, passes 2..7 at a hazard of sigmoid(-60), the rest at the forced exit)
    def first_pass(p: float) -> torch.Tensor:
        z = torch.full((8, 1, 1), -60.0, dtype=F64)
        z[0] = math.log(p / (1 - p))
        return z

    assert torch.equal(hazard_cdf_exits(first_pass(0.49), 0.5), _exits(8))
    assert torch.equal(hazard_cdf_exits(first_pass(0.51), 0.5), _exits(1))


def test_masked_positions_are_excluded_and_an_empty_mask_is_refused() -> None:
    log_p = _log_p_of([[0.5, 0.5], [1.0 - 1e-300, 1e-300]])
    losses = torch.stack([_vec([2.0, 7.0]), _vec([1.0, 9.0])])[:, None, :]
    only_first = _vec([1.0, 0.0]).bool()[None, :]
    assert expected_pass_loss(log_p, losses, only_first).item() == pytest.approx(1.5)
    assert kl_uniform_local(log_p, only_first).item() == pytest.approx(0.0, abs=1e-12)
    nothing = torch.zeros(1, 2, dtype=torch.bool)
    for f in (kl_uniform_local, kl_uniform_aggregate):
        with pytest.raises(ContractError, match="no position"):
            _ = f(log_p, nothing)
    with pytest.raises(ContractError, match="no position"):
        _ = expected_pass_loss(log_p, losses, nothing)


def test_the_full_loss_keeps_a_gradient_where_act_has_none() -> None:
    # Maintainer B's two-pass fixture: losses (2, 1), beta 0.1, first hazard logit 6: p_1 = 0.997527
    # and d(expected loss + beta KL)/dz_1 = 0.0039464; ACT halting at pass 1 has exactly zero there
    z = _vec([6.0, 0.0])[:, None, None].requires_grad_()
    log_p = exit_log_probs(z)
    mask = torch.ones(1, 1, dtype=torch.bool)
    losses = _vec([2.0, 1.0])[:, None, None]
    objective = expected_pass_loss(log_p, losses, mask) + 0.1 * kl_uniform_local(log_p, mask)
    (grad,) = torch.autograd.grad(objective, [z])
    assert log_p.exp()[0, 0, 0].item() == pytest.approx(0.9975273768, abs=1e-9)
    assert grad[0, 0, 0].item() == pytest.approx(0.0039464, abs=1e-6)
    assert grad[1, 0, 0].item() == 0.0  # the forced final exit has no hazard parameter


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("t_max", range(2, 9))
def test_an_exact_quantile_of_the_uniform_cdf_exits_at_its_own_pass(
    t_max: int, dtype: torch.dtype
) -> None:
    # condition B's CDF is exactly k / T at pass k: q = k / T must exit at k, whatever rounding
    # leaves in the sum (measured before the fix: float64, T = 4, q = 1/4 exited at pass 2);
    # the learned logits come in the head's dtype, the hazard logits are built by the primitive
    z = hazard_logits(torch.zeros(t_max, 1, 1, dtype=dtype))
    for k in range(1, t_max + 1):
        assert torch.equal(hazard_cdf_exits(z, k / t_max), _exits(k)), (t_max, k)


def test_hazard_logits_are_float64_learned_logits_plus_the_baseline() -> None:
    learned = torch.randn(5, 2, 3, generator=torch.Generator().manual_seed(1))
    z = hazard_logits(learned)
    assert z.dtype == F64
    expected = learned.double() + _baseline(5)[:, None, None]
    assert torch.equal(z, expected)
    with pytest.raises(ContractError, match="no pass"):
        _ = hazard_logits(torch.zeros(0, 2))
    with pytest.raises(ContractError, match="floating"):
        _ = hazard_cdf_exits(torch.zeros(4, 2, dtype=torch.int64), 0.5)


def test_a_non_finite_loss_at_a_masked_position_reaches_no_gradient() -> None:
    # Maintainer B's review of 49c92cf, finding 1: the product over every position before the mask
    # gave a finite 1.5 and a NaN hazard-logit gradient at the masked token
    z = torch.zeros(2, 1, 2, dtype=F64, requires_grad=True)
    losses = torch.stack([_vec([2.0, math.nan]), _vec([1.0, math.nan])])[:, None, :]
    _ = losses.requires_grad_(True)
    mask = _vec([1.0, 0.0]).bool()[None, :]
    value = expected_pass_loss(exit_log_probs(z), losses, mask)
    grad_z, grad_losses = torch.autograd.grad(value, [z, losses])
    assert value.item() == pytest.approx(1.5)
    assert bool(torch.isfinite(grad_z).all())
    assert bool(torch.isfinite(grad_losses).all())
    assert grad_losses[:, 0, 1].abs().sum().item() == 0.0


def test_the_reductions_refuse_mismatched_shapes_and_dtypes() -> None:
    log_p = exit_log_probs(_z(4, b=2, n=3))
    mask = torch.ones(2, 3, dtype=torch.bool)
    losses = torch.ones(4, 2, 3, dtype=F64)
    bad = {
        "broadcast losses": (log_p, torch.ones(4, 1, 3, dtype=F64), mask),
        "losses dtype": (log_p, losses.float(), mask),
        "mask dtype": (log_p, losses, mask.double()),
        "mask shape": (log_p, losses, torch.ones(3, dtype=torch.bool)),
        "rank": (log_p[:, 0], losses[:, 0], mask[0]),
        "no pass": (log_p[:0], losses[:0], mask),
    }
    for name, (lp, ls, m) in bad.items():
        with pytest.raises(ContractError):
            _ = expected_pass_loss(lp, ls, m)
        if name not in {"broadcast losses", "losses dtype"}:
            for f in (kl_uniform_local, kl_uniform_aggregate):
                with pytest.raises(ContractError):
                    _ = f(lp, m)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.float32, 3e38), (torch.bfloat16, 3e38), (torch.float64, 1e308)],
)
def test_every_finite_logit_saturates_to_finite_values(
    dtype: torch.dtype, magnitude: float
) -> None:
    # Maintainer B's review of 230a5e5, gap a: eight float32 logits of 3e38 overflowed the survival
    # sum
    for sign in (1.0, -1.0):
        z = torch.full((8, 1, 1), sign * magnitude, dtype=dtype, requires_grad=True)
        log_p = exit_log_probs(z)
        mask = torch.ones(1, 1, dtype=torch.bool)
        for kl in (kl_uniform_local(log_p, mask), kl_uniform_aggregate(log_p, mask)):
            (grad,) = torch.autograd.grad(kl, [z], retain_graph=True)
            assert math.isfinite(kl.item())
            assert kl.item() == pytest.approx(math.log(8), abs=1e-6)  # all mass on one exit
            assert bool(torch.isfinite(grad).all())
        assert bool(torch.isfinite(log_p).all())
        # the dense CDF exits agree: everything at pass 1, or nothing before the forced last
        assert torch.equal(hazard_cdf_exits(z.detach(), 0.5), _exits(1 if sign > 0 else 8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_precision_logits_are_promoted_and_stay_finite(dtype: torch.dtype) -> None:
    # Maintainer B's review of 49c92cf, finding 6: eight float16 logits of 60000 overflowed the
    # log-survival sum and gave a non-finite KL and gradient
    for value in (60000.0, -60000.0):
        z = torch.full((8, 1, 1), value, dtype=dtype, requires_grad=True)
        log_p = exit_log_probs(z)
        assert log_p.dtype == torch.float32
        kl = kl_uniform_local(log_p, torch.ones(1, 1, dtype=torch.bool))
        (grad,) = torch.autograd.grad(kl, [z])
        assert bool(torch.isfinite(log_p).all())
        assert math.isfinite(kl.item())
        assert bool(torch.isfinite(grad).all())


def test_an_exit_mass_that_underflows_to_zero_contributes_exactly_zero() -> None:
    # float32: the first hazard saturates, every later exit probability is exactly 0.0
    z = torch.full((8, 1, 1), 0.0, requires_grad=True)
    with torch.no_grad():
        _ = z[0].fill_(1000.0)
    log_p = exit_log_probs(z)
    assert bool((log_p.exp()[1:] == 0.0).all())
    mask = torch.ones(1, 1, dtype=torch.bool)
    kl = kl_uniform_local(log_p, mask)
    (grad,) = torch.autograd.grad(kl, [z])
    assert kl.item() == pytest.approx(math.log(8), abs=1e-6)
    assert bool(torch.isfinite(grad).all())
