"""Large joint updates by accumulation over micro-batches (plans A9, 2026-11-08).

A 64-row update with both rollouts does not fit one forward on a 16 GB card. Every term of the
joint loss is a row mean under pooled weights (score: actor_loss's mean over [ROLLOUTS, e];
pathwise; dense; the anchor), so ``update_step(..., parts=k)`` runs k micro-batches of the rows,
each backward scaled by its share of the rows, with the pooled depth priced over the WHOLE
update's real tokens, and then one clip and step. These tests hold it to the one-forward update.
"""

import math

import pytest
import torch
from torch import Tensor, nn

from conftest import small_raw
from vesper.config import parse
from vesper.joint_update import ROLLOUTS, Step, optimizer, release_backbone, start_head, update_step
from vesper.model import Vesper

T = 3
VOCAB = 96
E = 4
N = 8


def _model() -> tuple[Vesper, list[nn.Parameter], list[nn.Parameter]]:
    """A joint-mode model (head and backbone trainable) whose head stops some tokens early."""
    raw = small_raw(vocab=VOCAB)
    raw["model"]["t_max"] = T
    raw["recurrence"]["state_read_noise"] = 0.0
    model = Vesper(parse(raw, "test"), torch.Generator().manual_seed(0))
    head = start_head(model)
    backbone = release_backbone(model)
    with torch.no_grad():  # mixed routing: some tokens stop before the last pass
        for p in head:
            _ = p.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(3))
    return model, head, backbone


def _values(e: int = E, n: int = N) -> list[Tensor]:
    """``update_step``'s values for ``e`` unique rows of unequal real length (so a per-part depth
    normalization differs from the pooled one), rollout-major, weights pooled over all rows."""
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32)
    targets = torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32)
    valid = torch.ones(e, n, dtype=torch.bool)
    valid[-1, -3:] = False  # only the last row is short: its part has fewer real tokens
    mask = (torch.rand(e, n, generator=g) < 0.7) & valid
    mask[:, 0] = True
    weights = torch.where(mask, e / float(mask.sum()), 0.0).double()
    pos = torch.arange(n).expand(e, n).clone()
    uniforms = torch.rand((T - 1, ROLLOUTS * e, n), generator=g, dtype=torch.float64)
    rows = [torch.cat([t] * ROLLOUTS) for t in (tokens, pos, targets, mask, valid, weights)]
    return [*rows, uniforms, torch.zeros((), dtype=torch.float64)]


def _norm(t: Tensor) -> float:
    d = t.detach().double()
    return math.sqrt(float((d * d).sum()))


def _update(parts: int, *, pooled: bool = True) -> tuple[Step, list[Tensor], Vesper]:
    """One update from the same start; its outputs, the (clipped) gradients it applied, the
    model after it."""
    model, head, backbone = _model()
    params = head + backbone
    out = update_step(
        model,
        optimizer(head, backbone),
        params,
        2.5,
        0.1,
        _values(),
        pooled_depth=pooled,
        parts=parts,
    )
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
    return out, grads, model


@pytest.mark.parametrize("parts", [2, 4])
@pytest.mark.parametrize("depth", ["pooled", "per-row"])
def test_an_accumulated_update_applies_the_one_forward_gradient(parts: int, depth: str) -> None:
    # the same rows through float32 matmuls batched 2 or 4 at a time, then gradients summed over
    # parts: rounding only, normwise per tensor. Measured 2026-11-08 (the accumulation through
    # private helpers before update_step took `parts`): worst 1.97e-7 (2 parts), 1.75e-7 (4); a
    # per-part depth normalization differed by 2.9e-2 overall. Bound 1e-5, the one
    # test_joint_update_memory measured for a batching change
    whole, g_whole, _ = _update(1, pooled=depth == "pooled")
    got, g_got, _ = _update(parts, pooled=depth == "pooled")
    assert any(_norm(g) > 0 for g in g_whole)
    for g, h in zip(g_whole, g_got, strict=True):
        assert _norm(h - g) <= 1e-5 * _norm(g) + 1e-12
    assert float(got.grad_norm) == pytest.approx(float(whole.grad_norm), rel=1e-5)


@pytest.mark.parametrize("parts", [2, 4])
def test_an_accumulated_update_reports_the_whole_update(parts: int) -> None:
    # the per-trajectory outputs in the whole update's rollout-major order and the loss parts
    # summed with their row shares: what the driver's record reads
    whole, _, _ = _update(1)
    got, _, _ = _update(parts)
    per_row = [
        ("cost", whole.cost, got.cost),
        ("objective", whole.objective, got.objective),
        ("nll", whole.nll, got.nll),
        ("depth", whole.depth, got.depth),
        ("ce", whole.ce, got.ce),
    ]
    for name, a, b in per_row:
        assert a.shape == b.shape, name
        assert _norm(b - a) <= 1e-5 * _norm(a) + 1e-12, name
    assert torch.equal(whole.decisions, got.decisions)
    assert torch.equal(whole.n_iters, got.n_iters)
    parts_of_loss = [
        ("score", whole.score, got.score),
        ("pathwise", whole.pathwise, got.pathwise),
        ("dense", whole.dense, got.dense),
    ]
    for name, a, b in parts_of_loss:
        assert float(b) == pytest.approx(float(a), rel=1e-5), name


@pytest.mark.parametrize("parts", [0, 3, 5])
def test_a_part_count_that_does_not_divide_the_rows_is_refused_before_any_work(parts: int) -> None:
    model, head, backbone = _model()
    before = [p.detach().clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="parts must divide"):
        _ = update_step(
            model, optimizer(head, backbone), head + backbone, 2.5, 0.1, _values(), parts=parts
        )
    for p, q in zip(before, model.parameters(), strict=True):
        assert torch.equal(p, q)


def test_one_part_is_the_one_forward_update_bitwise() -> None:
    _, g_default, model_a = _update_default()
    _, g_one, model_b = _update(1)
    for g, h in zip(g_default, g_one, strict=True):
        assert torch.equal(g, h)
    for p, q in zip(model_a.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p, q)


def _update_default() -> tuple[Step, list[Tensor], Vesper]:
    """The update with NO parts argument: the function's own default."""
    model, head, backbone = _model()
    params = head + backbone
    out = update_step(
        model, optimizer(head, backbone), params, 2.5, 0.1, _values(), pooled_depth=True
    )
    grads = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
    return out, grads, model
