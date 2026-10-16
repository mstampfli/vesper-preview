import math

import pytest
import torch

from vesper.contracts import ContractError
from vesper.learning import LossWeights, ntp_loss, wake_loss

V = 7
W = LossWeights(lambda_h=0.01, lambda_g=0.01)


def _logits_and_targets() -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 5, V, generator=g, dtype=torch.float64)
    targets = torch.randint(0, V, (2, 5), generator=g)
    return logits, targets


def test_per_token_loss_matches_log_softmax() -> None:
    logits, targets = _logits_and_targets()
    loss = ntp_loss(logits, targets)
    assert loss.shape == (2, 5)
    manual = -(logits - logits.logsumexp(dim=-1, keepdim=True)).gather(-1, targets[..., None])[
        ..., 0
    ]
    assert (loss - manual).abs().max().item() < 1e-12
    # Uniform logits give exactly log V, a value one can check by hand.
    assert (
        ntp_loss(torch.zeros(1, 1, V, dtype=torch.float64), torch.full((1, 1), 3, dtype=torch.long))
        - math.log(V)
    ).abs().item() < 1e-12


def test_e2e_wake_loss_is_the_spec_formula() -> None:
    logits, targets = _logits_and_targets()
    ponder = torch.tensor(2.5, dtype=torch.float64)
    gate = torch.tensor(0.4, dtype=torch.float64)
    total, parts = wake_loss(logits, targets, mode="e2e", ponder=ponder, gate=gate, weights=W)
    ntp = ntp_loss(logits, targets).mean()
    assert (parts["ntp"] - ntp).abs().item() < 1e-12
    assert (parts["ponder"] - 0.01 * ponder).abs().item() < 1e-12
    assert (parts["gate"] - 0.01 * gate).abs().item() < 1e-12
    assert (total - (ntp + 0.01 * ponder + 0.01 * gate)).abs().item() < 1e-12


def test_e2e_without_loop_terms_is_plain_cross_entropy() -> None:
    logits, targets = _logits_and_targets()
    total, parts = wake_loss(logits, targets, mode="e2e", weights=W)
    assert set(parts) == {"ntp"}
    assert (total - ntp_loss(logits, targets).mean()).abs().item() < 1e-12


def test_gradient_flows_to_logits() -> None:
    logits, targets = _logits_and_targets()
    leaf = logits.clone().requires_grad_()
    total, _ = wake_loss(leaf, targets, mode="e2e", weights=W)
    (g,) = torch.autograd.grad(total, [leaf])
    assert g.abs().sum().item() > 0
    # Softmax gradient rows sum to zero: a sign-sensitive check, not a magnitude one.
    assert g.sum(dim=-1).abs().max().item() < 1e-12


def test_local_mode_is_refused_until_its_target_is_pinned() -> None:
    logits, targets = _logits_and_targets()
    with pytest.raises(NotImplementedError, match="local"):
        _ = wake_loss(logits, targets, mode="local", weights=W)


def test_rejects_mismatched_targets() -> None:
    logits, targets = _logits_and_targets()
    with pytest.raises(ContractError):
        _ = ntp_loss(logits, targets[:, :4])
    with pytest.raises(ContractError, match="vocab"):
        _ = ntp_loss(logits, torch.full_like(targets, V))


def test_empty_aggregate_is_refused_not_nan() -> None:
    cases = (
        (torch.zeros(1, 0, V), torch.zeros(1, 0, dtype=torch.long)),
        (torch.zeros(0, 3, V), torch.zeros(0, 3, dtype=torch.long)),
    )
    for logits, targets in cases:
        assert ntp_loss(logits, targets).shape == targets.shape
        with pytest.raises(ContractError, match="nothing to aggregate"):
            _ = wake_loss(logits, targets, mode="e2e", weights=W)


def test_int32_targets_are_widened_at_the_boundary() -> None:
    logits, targets = _logits_and_targets()
    assert torch.equal(ntp_loss(logits, targets.to(torch.int32)), ntp_loss(logits, targets))
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = ntp_loss(logits, targets.to(torch.float64))


def test_loss_is_aligned_to_the_given_target_and_not_shifted() -> None:
    logits, targets = _logits_and_targets()
    base = ntp_loss(logits, targets)
    moved = targets.clone()
    moved[0, 2] = (moved[0, 2] + 1) % V
    changed = (ntp_loss(logits, moved) - base).abs() > 0
    assert bool(changed[0, 2].item())
    assert int(changed.sum().item()) == 1


def test_scalar_case_by_hand() -> None:
    # logits (log 2, 0) against target 0: cross-entropy log(3/2), logit gradient (-1/3, 1/3)
    logits = torch.zeros(1, 1, 2, dtype=torch.float64)
    logits[0, 0, 0] = math.log(2.0)
    leaf = logits.clone().requires_grad_()
    loss = ntp_loss(leaf, torch.zeros(1, 1, dtype=torch.long))
    assert abs(loss.item() - math.log(1.5)) < 1e-12
    (g,) = torch.autograd.grad(loss.sum(), [leaf])
    assert abs(g[0, 0, 0].item() + 1 / 3) < 1e-12
    assert abs(g[0, 0, 1].item() - 1 / 3) < 1e-12


def test_masked_positions_have_no_path_into_the_value_or_the_gradient() -> None:
    # the review's case: NaN logits at the masked position, finite loss log 2 at the live one,
    # and a gradient of exactly zero (not NaN) at the masked logits
    logits = torch.zeros(1, 2, 2, dtype=torch.float64)
    logits[0, 1] = float("nan")
    leaf = logits.clone().requires_grad_()
    targets = torch.zeros(1, 2, dtype=torch.long)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    mask[0, 0] = True
    total, _ = wake_loss(leaf, targets, mode="e2e", weights=W, mask=mask)
    assert abs(total.item() - math.log(2.0)) < 1e-12
    (g,) = torch.autograd.grad(total, [leaf])
    assert torch.equal(g[0, 1], torch.zeros(2, dtype=torch.float64))
    assert abs(g[0, 0, 0].item() + 0.5) < 1e-12
    assert abs(g[0, 0, 1].item() - 0.5) < 1e-12


def test_mask_selects_the_positions_that_enter_the_mean() -> None:
    logits, targets = _logits_and_targets()
    per_token = ntp_loss(logits, targets)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 4] = True
    total, parts = wake_loss(logits, targets, mode="e2e", weights=W, mask=mask)
    expected = (per_token[0, 1] + per_token[1, 4]) / 2
    assert (parts["ntp"] - expected).abs().item() < 1e-12
    assert (total - expected).abs().item() < 1e-12
    with pytest.raises(ContractError, match="selects no position"):
        _ = wake_loss(logits, targets, mode="e2e", weights=W, mask=torch.zeros_like(mask))
    with pytest.raises(ContractError):
        _ = wake_loss(logits, targets, mode="e2e", weights=W, mask=mask[:, :4])
    with pytest.raises(ContractError):
        _ = wake_loss(logits, targets, mode="e2e", weights=W, mask=mask.to(torch.int64))
