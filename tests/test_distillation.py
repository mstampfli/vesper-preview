"""Independent dense-distribution values, gradients, ownership and update integration."""

import copy
import dataclasses
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from test_halting_text import PLAN, PhaseOne, _phase_one, text_chunks, three_pass_config  # pyright: ignore[reportPrivateUsage]
from test_train import Builder
from vesper.contracts import ContractError
from vesper.distillation import Anchor, depth_kl
from vesper.halting_text import check_plan, train_text_update
from vesper.joint_update import (
    ROLLOUTS,
    Dense,
    optimizer,
    release_backbone,
    start_head,
    update_loss,
)
from vesper.loop import DenseRule
from vesper.model import Vesper


def _unchanged(t: Tensor) -> Tensor:
    return t


def _model() -> Vesper:
    return Vesper(three_pass_config(), torch.Generator().manual_seed(3))


def _dense(model: Vesper, tokens: Tensor, pos: Tensor) -> Tensor:
    parameter = next(model.parameters())
    states = model.init_states(tokens.shape[0], parameter.dtype, parameter.device)
    return model.run(tokens, pos, states, exit_rule=DenseRule()).per_pass


def _oracle(model: Vesper, teacher: Vesper, y: Tensor, old: Tensor, mask: Tensor) -> Tensor:
    # Full explicit distributions, independently indexed by depth and position; no slices,
    # checkpoint or production KL helper. Selection occurs before the nonlinear head.
    parts: list[Tensor] = []
    for depth in range(y.shape[0]):
        student = model.head(y[depth][mask][None]).double().log_softmax(-1)[0]
        with torch.no_grad():
            target = teacher.head(old[depth][mask][None]).double().log_softmax(-1)[0]
        parts.append((target.exp() * (target - student)).sum(-1))
    return torch.stack(parts)


@pytest.mark.parametrize("slices", [1, 3, 100])
def test_values_and_all_gradients_match_full_distribution_oracle(slices: int) -> None:
    model = _model().double()
    teacher = copy.deepcopy(model)
    g = torch.Generator().manual_seed(5)
    y = torch.randn(3, 2, 5, 32, generator=g, dtype=torch.float64, requires_grad=True)
    old = torch.randn(3, 2, 5, 32, generator=g, dtype=torch.float64, requires_grad=True)
    mask_rows: list[list[bool]] = [
        [True, False, True, True, False],
        [False, True, False, False, False],
    ]
    mask = torch.tensor(mask_rows)
    got = depth_kl(model, teacher, y, old, mask, slices=slices)
    expected = _oracle(model, teacher, y, old, mask)
    torch.testing.assert_close(got[:, mask], expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(got[:, ~mask], torch.zeros_like(got[:, ~mask]))
    params = [y, model.embedding.weight, model.norm_out.weight]
    actual = torch.autograd.grad(got.sum(), params, retain_graph=True)
    wanted = torch.autograd.grad(expected.sum(), params)
    for a, b in zip(actual, wanted, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)
    assert old.grad is None
    assert all(p.grad is None for p in teacher.parameters())


def test_masked_nan_rows_cannot_poison_value_or_gradient_and_empty_is_zero() -> None:
    model = _model()
    teacher = copy.deepcopy(model)
    y = torch.randn(3, 1, 2, 32, requires_grad=True)
    old = y.detach().clone()
    with torch.no_grad():
        y[:, :, 1] = torch.nan
        old[:, :, 1] = torch.nan
    mask_rows: list[list[bool]] = [[True, False]]
    value = depth_kl(model, teacher, y, old, torch.tensor(mask_rows), slices=3)
    torch.autograd.backward([value.sum()])
    assert bool(torch.isfinite(value).all())
    assert y.grad is not None
    assert bool(torch.isfinite(y.grad).all())
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    empty = depth_kl(
        model, teacher, y[:, :, :0], old[:, :, :0], torch.zeros(1, 0, dtype=torch.bool)
    )
    assert empty.shape == (3, 1, 0)


def test_shapes_slices_and_model_vocabulary_are_checked() -> None:
    model = _model()
    y = torch.zeros(3, 2, 4, 32)
    mask = torch.ones(2, 4, dtype=torch.bool)
    for bad in [mask[:1], mask.T, mask.long()]:
        with pytest.raises(ContractError):
            _ = depth_kl(model, model, y, y, bad)
    with pytest.raises(ValueError, match="slices"):
        _ = depth_kl(model, model, y, y, mask, slices=0)
    with pytest.raises(ContractError):
        _ = depth_kl(model, model, y, y[:1], mask)


def test_no_vocabulary_activation_is_saved_for_backward() -> None:
    model = _model()
    teacher = copy.deepcopy(model)
    y = torch.randn(3, 2, 5, 32, requires_grad=True)
    saved: list[Tensor] = []

    def pack(t: Tensor) -> Tensor:
        saved.append(t)
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, _unchanged):
        value = depth_kl(
            model, teacher, y, y.detach() + 0.2, torch.ones(2, 5, dtype=torch.bool), slices=3
        )
        torch.autograd.backward([value.sum()])
    assert not any(t.dim() and t.shape[-1] == model.embedding.vocab for t in saved)
    assert y.grad is not None
    assert float(y.grad.abs().sum()) > 0


def test_teacher_is_original_owned_frozen_and_cold_and_loss_is_pooled() -> None:
    model = _model()
    anchor = Anchor(model, 0.7)
    before = {
        k: v.clone() for k, v in cast("dict[str, Tensor]", anchor.teacher.state_dict()).items()
    }
    assert not any(p.requires_grad for p in anchor.teacher.parameters())
    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(model.parameters(), anchor.teacher.parameters(), strict=True)
    )
    chunks = text_chunks(2, 8, 2)
    with torch.no_grad():
        _ = model.embedding.weight.add_(0.1)
    c = chunks[0]
    mask = c.loss_mask.clone()
    mask[1, 1:] = False
    weights = torch.where(mask, 2.0 / mask.sum(), 0.0).double()
    y = _dense(model, c.tokens, c.pos)
    old = _dense(anchor.teacher, c.tokens, c.pos)
    got = anchor.loss(model, y, c.tokens, c.pos, mask, weights, slices=3)
    expected = 0.7 * _oracle(model, anchor.teacher, y, old, mask).mean()
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=1e-7)
    torch.autograd.backward([got])
    assert all(p.grad is None for p in anchor.teacher.parameters())
    assert all(
        torch.equal(v, cast("dict[str, Tensor]", anchor.teacher.state_dict())[k])
        for k, v in before.items()
    )
    profile = anchor.profile(model, chunks, slices=3)
    labels = sum(int(c.loss_mask.sum()) for c in chunks)
    with torch.no_grad():
        wanted = (
            sum(
                (
                    _oracle(
                        model,
                        anchor.teacher,
                        _dense(model, c.tokens, c.pos),
                        _dense(anchor.teacher, c.tokens, c.pos),
                        c.loss_mask,
                    ).sum(1)
                    for c in chunks
                ),
                torch.zeros(3, dtype=torch.float64),
            )
            / labels
        )
    torch.testing.assert_close(
        torch.tensor(profile, dtype=torch.float64), wanted, rtol=2e-5, atol=1e-7
    )
    assert anchor.profile(model, [chunks[1]], slices=3) == anchor.profile(
        model, [chunks[1]], slices=3
    )


@pytest.mark.parametrize("stopping", [True, False])
def test_anchor_reuses_dense_forward_and_adds_the_independent_loss(*, stopping: bool) -> None:
    model = _model()
    anchor = Anchor(model, 0.4)
    _ = start_head(model)
    _ = release_backbone(model)
    with torch.no_grad():
        _ = model.norm_out.weight.mul_(1.1)
    c = text_chunks(2, 8, 1)[0]
    weights = torch.where(c.loss_mask, 2.0 / c.loss_mask.sum(), 0.0).double()
    values = [
        torch.cat([v] * ROLLOUTS)
        for v in (c.tokens, c.pos, c.targets, c.loss_mask, c.valid, weights)
    ]
    uniforms = torch.rand(2, 4, 8, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    dense = Dense(c.tokens, c.pos, c.targets, c.loss_mask, weights, 0.2)
    with patch.object(model, "run", wraps=model.run) as calls:
        got = update_loss(
            model,
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            uniforms,
            0.0,
            0.3,
            dense,
            stopping=stopping,
            anchor=anchor,
            slices=3,
        )
    assert calls.call_count == 2  # sampled + ONE dense, shared by CE and KL
    want = anchor.loss(
        model, _dense(model, c.tokens, c.pos), c.tokens, c.pos, c.loss_mask, weights, slices=3
    )
    torch.testing.assert_close(got.anchor, want)
    base = got.score + got.pathwise + got.dense if stopping else got.dense
    torch.testing.assert_close(got.total, base + want)
    dense_outputs = _dense(model, c.tokens, c.pos)
    original_outputs = _dense(anchor.teacher, c.tokens, c.pos)
    independent = (
        0.4 * _oracle(model, anchor.teacher, dense_outputs, original_outputs, c.loss_mask).mean()
    )
    params = list(model.parameters())
    actual_grads = cast(
        "tuple[Tensor | None, ...]",
        torch.autograd.grad(got.total, params, retain_graph=True, allow_unused=True),
    )
    expected_grads = cast(
        "tuple[Tensor | None, ...]",
        torch.autograd.grad(base + independent, params, allow_unused=True),
    )
    for actual, expected in zip(actual_grads, expected_grads, strict=True):
        if actual is None:
            assert expected is None
        else:
            assert expected is not None
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-7)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_bad_weight_refuses_before_source_work(weight: float) -> None:
    with pytest.raises(ValueError, match="anchor"):
        _ = check_plan(dataclasses.replace(PLAN, anchor_weight=weight))


def test_driver_records_teacher_and_distance_at_every_save(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1: PhaseOne = _phase_one(tmp_path, book_builder)
    plan = dataclasses.replace(
        PLAN, backbone="dense", anchor_weight=0.5, evaluation="after_training"
    )
    result = p1.run(tmp_path / "a3", plan)
    distances = cast("dict[str, list[float]]", result["anchor_kl"])
    assert set(distances) == {"0", "2"}
    assert max(abs(x) for x in distances["0"]) < 1e-7
    assert max(distances["2"]) > 0
    assert (
        cast("dict[str, object]", result["anchor"])["source_model_sha256"]
        == result["source_model_sha256"]
    )


def test_validation_only_distance_preserves_training_and_saved_weights(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    plain = p1.run(tmp_path / "plain", PLAN)
    measured = p1.run(tmp_path / "measured", dataclasses.replace(PLAN, anchor_eval=True))
    assert (tmp_path / "plain/train.jsonl").read_bytes() == (
        tmp_path / "measured/train.jsonl"
    ).read_bytes()
    assert plain["consumed_digest"] == measured["consumed_digest"]
    assert plain["draw_digest"] == measured["draw_digest"]
    assert plain["evaluations"] == measured["evaluations"]
    assert plain["anchor"] is None
    assert set(cast("dict[str, object]", measured["anchor_kl"])) == {"0", "2"}
    for step in (0, 2):
        left = cast(
            "dict[str, Tensor]",
            torch.load(tmp_path / f"plain/weights/after-{step:06d}.pt", weights_only=True),
        )
        right = cast(
            "dict[str, Tensor]",
            torch.load(tmp_path / f"measured/weights/after-{step:06d}.pt", weights_only=True),
        )
        assert all(torch.equal(v, right[k]) for k, v in left.items())


def test_zero_anchor_skips_teacher_and_missing_enabled_anchor_refuses() -> None:
    model = _model()
    anchor = Anchor(model, 0.0)
    head = start_head(model)
    backbone = release_backbone(model)
    c = text_chunks(1, 8, 1)[0]
    u = torch.rand(2, 2, 8, dtype=torch.float64)
    opt = optimizer(head, backbone)
    before = [p.clone() for p in model.parameters()]
    with pytest.raises(ValueError, match="configured anchor"):
        _ = train_text_update(
            model, opt, head + backbone, c, u, 0, dataclasses.replace(PLAN, anchor_weight=0.5)
        )
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
    with patch.object(anchor.teacher, "run", side_effect=AssertionError("disabled teacher ran")):
        record = train_text_update(model, opt, head + backbone, c, u, 0, PLAN, anchor=anchor)
    assert record["anchor"] == 0.0


def test_anchor_without_label_auxiliary_reuses_one_dense_pass() -> None:
    model = _model()
    anchor = Anchor(model, 0.6)
    head = start_head(model)
    backbone = release_backbone(model)
    with torch.no_grad():
        _ = model.norm_out.weight.add_(0.1)
    c = text_chunks(1, 8, 1)[0]
    u = torch.rand(2, 2, 8, dtype=torch.float64)
    plan = dataclasses.replace(PLAN, dense_weight=0.0, anchor_weight=0.6)
    with patch.object(model, "run", wraps=model.run) as calls:
        record = train_text_update(
            model, optimizer(head, backbone), head + backbone, c, u, 0, plan, anchor=anchor
        )
    assert calls.call_count == 2
    assert record["dense"] == 0.0
    assert cast("float", record["anchor"]) > 0.0


def test_teacher_receives_no_gradient_even_when_caller_outputs_require_it() -> None:
    model = _model()
    teacher = copy.deepcopy(model)
    y = torch.randn(3, 1, 2, 32, requires_grad=True)
    old = torch.randn_like(y, requires_grad=True)
    result = depth_kl(model, teacher, y, old, torch.ones(1, 2, dtype=torch.bool))
    grads = torch.autograd.grad(result.sum(), [old, *teacher.parameters()], allow_unused=True)
    assert all(g is None for g in grads)


def test_profile_refuses_no_labels_and_restores_training_mode() -> None:
    model = _model()
    anchor = Anchor(model, 0.0)
    with pytest.raises(ValueError, match="no labels"):
        _ = anchor.profile(model, [])
    assert model.training
    other = _model()
    other.loop.t_max = 2
    other.t_max = 2
    y = torch.zeros(3, 1, 2, 32)
    with pytest.raises(ContractError, match="teacher depth"):
        _ = depth_kl(model, other, y, y, torch.ones(1, 2, dtype=torch.bool))


def test_nonfinite_anchor_refuses_before_optimizer_changes_the_student() -> None:
    model = _model()
    anchor = Anchor(model, 0.5)
    with torch.no_grad():
        _ = anchor.teacher.norm_out.weight.fill_(float("nan"))
    head = start_head(model)
    backbone = release_backbone(model)
    opt = optimizer(head, backbone)
    before = [p.detach().clone() for p in model.parameters()]
    c = text_chunks(1, 8, 1)[0]
    u = torch.rand(2, 2, 8, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="non-finite"):
        _ = train_text_update(
            model,
            opt,
            head + backbone,
            c,
            u,
            0,
            dataclasses.replace(PLAN, anchor_weight=0.5),
            anchor=anchor,
        )
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before, strict=True))
    assert not cast("dict[Tensor, object]", opt.state)
