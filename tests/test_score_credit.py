"""The joint update without score credit to the backbone (backbone mode "pathwise").

With ``score_to_backbone=False`` the halting head reads DETACHED features in the sampled
forward, so the paired score surrogate trains the head alone and the backbone learns only from
the label terms (pathwise and dense). Nothing else in the forward reads the head (its output is
a discrete stop), so every value, every decision and every gradient of the label terms stays
bitwise the default's; only the score term's path into the backbone is gone. The default keeps
it, as every earlier joint run had it (maintainer B, channel f27b8a3).
"""

import math
from typing import TypedDict, cast

import torch
from torch import Tensor, nn

from conftest import small_raw
from vesper.config import parse
from vesper.joint_update import (
    ROLLOUTS,
    Dense,
    Loss,
    optimizer,
    release_backbone,
    start_head,
    update_loss,
    update_step,
)
from vesper.model import Vesper


class ScoreOptions(TypedDict, total=False):
    score_to_backbone: bool


T = 3


def _model() -> tuple[Vesper, list[nn.Parameter], list[nn.Parameter]]:
    """A joint-mode model whose head reads its features (a nonzero direction): mixed stops."""
    raw = small_raw(vocab=96)
    raw["model"]["t_max"] = T
    raw["recurrence"]["state_read_noise"] = 0.0
    model = Vesper(parse(raw, "test"), torch.Generator().manual_seed(0))
    head = start_head(model)
    backbone = release_backbone(model)
    with torch.no_grad():
        for p in head:
            _ = p.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(3))
    return model, head, backbone


def _batch(e: int = 2, n: int = 8) -> dict[str, Tensor]:
    """``e`` unique rows twice over, rollout-major, with independent uniforms per rollout."""
    g = torch.Generator().manual_seed(1)
    unique = {
        "tokens": torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32),
        "targets": torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32),
        "mask": torch.ones(e, n, dtype=torch.bool),
        "valid": torch.ones(e, n, dtype=torch.bool),
        "weights": torch.full((e, n), 1.0 / (e * n), dtype=torch.float64),
    }
    batch = {k: torch.cat([v] * ROLLOUTS) for k, v in unique.items()}
    batch["pos"] = torch.arange(n).expand(ROLLOUTS * e, n).clone()
    batch["uniforms"] = torch.rand((T - 1, ROLLOUTS * e, n), generator=g, dtype=torch.float64)
    return batch


def _loss(model: Vesper, b: dict[str, Tensor], *, dense: float, credit: bool | None) -> Loss:
    """``update_loss`` on ``b``; ``credit`` None calls it WITHOUT the argument (its default)."""
    e = int(b["tokens"].shape[0]) // ROLLOUTS
    aux = Dense(
        b["tokens"][:e], b["pos"][:e], b["targets"][:e], b["mask"][:e], b["weights"][:e], dense
    )
    batch = (b["tokens"], b["pos"], b["targets"], b["mask"], b["valid"], b["weights"])
    if credit is None:
        return update_loss(model, *batch, b["uniforms"], 0.0, 2.5, aux)
    return update_loss(model, *batch, b["uniforms"], 0.0, 2.5, aux, score_to_backbone=credit)


def _grads(loss: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    """Every gradient of ``loss``, an unreached parameter's as exact zeros; the graph kept."""
    grads = cast(  # allow_unused: an unreached parameter comes back None
        "tuple[Tensor | None, ...]",
        torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True),
    )
    return [
        torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params, strict=True)
    ]


def _norm(ts: list[Tensor]) -> float:
    return math.sqrt(sum(float((t.double() * t.double()).sum()) for t in ts))


def test_without_score_credit_the_score_term_trains_the_head_only() -> None:
    model, head, backbone = _model()
    b = _batch()
    credited = _loss(model, b, dense=0.1, credit=None)  # the default: every earlier joint run
    plain = _loss(model, b, dense=0.1, credit=False)
    assert float(credited.score.detach()) != 0.0  # the rollouts differ: a real score term
    on_backbone = _grads(credited.score, backbone)
    off_backbone = _grads(plain.score, backbone)
    assert _norm(on_backbone) > 0.0  # control: by default the score reaches the backbone
    assert _norm(off_backbone) == 0.0
    on_head, off_head = _grads(credited.score, head), _grads(plain.score, head)
    assert _norm(on_head) > 0.0
    assert all(torch.equal(a, c) for a, c in zip(on_head, off_head, strict=True))  # head kept


def test_without_score_credit_values_decisions_and_label_gradients_are_unchanged() -> None:
    model, head, backbone = _model()
    b = _batch()
    credited = _loss(model, b, dense=0.1, credit=True)
    plain = _loss(model, b, dense=0.1, credit=False)
    assert torch.equal(credited.total.detach(), plain.total.detach())
    assert torch.equal(credited.score.detach(), plain.score.detach())
    assert credited.rule.decisions is not None
    assert plain.rule.decisions is not None
    assert torch.equal(credited.rule.decisions, plain.rule.decisions)
    assert torch.equal(credited.result.n_iters, plain.result.n_iters)
    params = head + backbone
    on = _grads(credited.pathwise + credited.dense, params)
    off = _grads(plain.pathwise + plain.dense, params)
    assert _norm(on) > 0.0
    assert all(torch.equal(a, c) for a, c in zip(on, off, strict=True))


def test_the_update_step_passes_score_credit_through() -> None:
    # behaviour, not the argument: the backbone group's gradient norm before clipping is the
    # label term's alone without credit, and differs from it by default
    b = _batch()
    settings: list[tuple[str, ScoreOptions]] = [
        ("default", {}),
        ("plain", {"score_to_backbone": False}),
    ]
    values = [
        b["tokens"],
        b["pos"],
        b["targets"],
        b["mask"],
        b["valid"],
        b["weights"],
        b["uniforms"],
        torch.tensor(0.0, dtype=torch.float64),
    ]
    norms: dict[str, list[float]] = {}
    for name, kwargs in settings:
        model, head, backbone = _model()
        opt = optimizer(head, backbone)
        step = update_step(model, opt, head + backbone, 2.5, 0.0, values, **kwargs)
        norms[name] = [float(x) for x in step.group_norms.unbind()]
    model, _, backbone = _model()
    expected = _norm(_grads(_loss(model, b, dense=0.0, credit=False).pathwise, backbone))
    assert math.isclose(norms["plain"][1], expected, rel_tol=1e-12, abs_tol=0.0)
    assert not math.isclose(norms["default"][1], expected, rel_tol=1e-6, abs_tol=0.0)
    assert norms["default"][0] == norms["plain"][0]  # the head's gradient is the same
