"""The dense-only control of the joint text runs (backbone mode "dense"; plans A0, 11-03).

J2's backbone regressed at every fixed depth (depth 2, where its policy mostly stops, by 0.13
nats, growing with updates), so the drift is backbone-wide. Is it the stopping objective, or
the fine-tuning setup itself (fresh optimizer, four rows per update, cold-start chunks)? With
``stopping=False`` the joint update's loss is its dense all-pass term ALONE: the sampled
trajectories still run and are recorded, but neither the score surrogate nor the pathwise term
trains anything, and the halting head gets no gradient. Same setup, no stopping objective.
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


class StoppingOptions(TypedDict, total=False):
    stopping: bool


T = 3


def _model() -> tuple[Vesper, list[nn.Parameter], list[nn.Parameter]]:
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


def _loss(model: Vesper, b: dict[str, Tensor], *, stopping: bool | None) -> Loss:
    """``update_loss`` on ``b`` with dense weight 1; ``stopping`` None: the default call."""
    e = int(b["tokens"].shape[0]) // ROLLOUTS
    aux = Dense(
        b["tokens"][:e], b["pos"][:e], b["targets"][:e], b["mask"][:e], b["weights"][:e], 1.0
    )
    batch = (b["tokens"], b["pos"], b["targets"], b["mask"], b["valid"], b["weights"])
    if stopping is None:
        return update_loss(model, *batch, b["uniforms"], 0.0, 2.5, aux)
    return update_loss(model, *batch, b["uniforms"], 0.0, 2.5, aux, stopping=stopping)


def _grads(loss: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    grads = cast(
        "tuple[Tensor | None, ...]",
        torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True),
    )
    return [
        torch.zeros_like(p) if g is None else g.detach() for g, p in zip(grads, params, strict=True)
    ]


def _norm(ts: list[Tensor]) -> float:
    return math.sqrt(sum(float((t.double() * t.double()).sum()) for t in ts))


def test_without_stopping_the_loss_is_the_dense_term_alone() -> None:
    model, head, backbone = _model()
    b = _batch()
    plain = _loss(model, b, stopping=False)
    assert torch.equal(plain.total, plain.dense)
    assert float(plain.score.detach()) != 0.0  # still computed and recorded
    assert float(plain.pathwise.detach()) != 0.0
    assert _norm(_grads(plain.total, head)) == 0.0  # the head gets nothing
    on = _grads(plain.total, backbone)
    want = _grads(plain.dense, backbone)
    assert _norm(on) > 0.0
    assert all(torch.equal(a, c) for a, c in zip(on, want, strict=True))


def test_by_default_the_loss_keeps_the_stopping_terms() -> None:
    model, _, _ = _model()
    b = _batch()
    joint = _loss(model, b, stopping=None)
    assert torch.equal(joint.total, joint.score + joint.pathwise + joint.dense)
    plain = _loss(model, b, stopping=False)
    assert torch.equal(joint.dense.detach(), plain.dense.detach())  # same forward, same terms
    assert torch.equal(joint.score.detach(), plain.score.detach())


def test_the_update_step_passes_stopping_through() -> None:
    b = _batch()
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
    settings: list[tuple[str, StoppingOptions]] = [("default", {}), ("dense", {"stopping": False})]
    for name, kwargs in settings:
        model, head, backbone = _model()
        step = update_step(model, optimizer(head, backbone), head + backbone, 2.5, 1.0, values,
                           **kwargs)  # fmt: skip
        norms[name] = [float(x) for x in step.group_norms.unbind()]
    model, _, backbone = _model()
    expected = _norm(_grads(_loss(model, b, stopping=False).dense, backbone))
    assert norms["dense"][0] == 0.0  # the head group has no gradient
    assert norms["default"][0] > 0.0
    assert math.isclose(norms["dense"][1], expected, rel_tol=1e-12, abs_tol=0.0)
    assert not math.isclose(norms["default"][1], expected, rel_tol=1e-6, abs_tol=0.0)
