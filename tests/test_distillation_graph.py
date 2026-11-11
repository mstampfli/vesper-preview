"""Anchor updates across actual CUDA capture, including changing batches and rates."""

import dataclasses
from typing import cast

import pytest
import torch
from torch import Tensor

from test_halting_text import PLAN, text_chunks, three_pass_config
from vesper.column import set_scan
from vesper.distillation import Anchor
from vesper.graphs import SignatureGraph, WarmedGraph, deterministic_algorithms
from vesper.halting_text import train_text_update
from vesper.joint_update import optimizer, release_backbone, start_head, training_precision
from vesper.kernels.delta_scan import fused_scan_available
from vesper.model import Vesper

needs_fused = pytest.mark.skipif(not fused_scan_available(), reason="requires fused CUDA scan")
CUDA = torch.device("cuda")
UPDATES = 7


def _snapshot(model: Vesper, opt: torch.optim.AdamW) -> list[Tensor]:
    state = cast("dict[Tensor, dict[str, Tensor]]", opt.state)
    return [p.detach().clone() for p in model.parameters()] + [
        value.detach().clone() for p in model.parameters() for value in state.get(p, {}).values()
    ]


def _trajectory(
    mode: str, *, graph: WarmedGraph | None, recompute: bool
) -> tuple[list[dict[str, object]], list[list[Tensor]]]:
    model = Vesper(three_pass_config(), torch.Generator().manual_seed(3)).to(CUDA)
    _ = set_scan(model, "fused-fp32")
    anchor = Anchor(model, 0.7)
    original = [p.detach().clone() for p in anchor.teacher.parameters()]
    head = start_head(model)
    backbone = release_backbone(model)
    with torch.no_grad():
        _ = model.norm_out.weight.add_(0.1)
        for p in head:
            _ = p.normal_(0.0, 0.3, generator=torch.Generator(device=CUDA).manual_seed(5))
    opt = optimizer(head, backbone, capturable=True)
    plan = dataclasses.replace(
        PLAN,
        backbone=mode,
        device="cuda",
        updates=UPDATES,
        rows=2,
        slices=3,
        recompute=recompute,
        anchor_weight=0.7,
    )
    records: list[dict[str, object]] = []
    states: list[list[Tensor]] = []
    for update, chunk in enumerate(text_chunks(2, 8, UPDATES)):
        uniforms = torch.rand(
            2, 4, 8, generator=torch.Generator().manual_seed(100 + update), dtype=torch.float64
        )
        records.append(
            train_text_update(
                model,
                opt,
                head + backbone,
                chunk,
                uniforms,
                update,
                plan,
                graph=graph,
                anchor=anchor,
            )
        )
        states.append(_snapshot(model, opt))
        assert all(p.grad is None for p in anchor.teacher.parameters())
        assert all(
            torch.equal(p, q) for p, q in zip(anchor.teacher.parameters(), original, strict=True)
        )
    assert all(cast("float", r["anchor"]) > 0 for r in records)
    assert len({tuple(cast("list[float]", r["lr_groups"])) for r in records}) == UPDATES
    return records, states


@needs_fused
@pytest.mark.parametrize("mode", ["joint", "pathwise", "dense"])
@pytest.mark.parametrize("recompute", [False, True])
def test_captured_anchor_updates_match_every_eager_parameter_and_adam_state(
    mode: str, *, recompute: bool
) -> None:
    _ = training_precision(cuda=True)
    with deterministic_algorithms():
        expected, states = _trajectory(mode, graph=None, recompute=recompute)
        graph = WarmedGraph()
        actual, captured = _trajectory(mode, graph=graph, recompute=recompute)
    assert graph.captures == 1
    assert graph.replays == UPDATES - graph.warm
    assert actual == expected
    for left, right in zip(states, captured, strict=True):
        assert len(left) == len(right)
        assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


@needs_fused
def test_captured_validation_distance_uses_changing_values_and_model_identity() -> None:
    _ = training_precision(cuda=True)
    model = Vesper(three_pass_config(), torch.Generator().manual_seed(3)).to(CUDA)
    _ = set_scan(model, "fused-fp32")
    anchor = Anchor(model, 0.0)
    chunks = text_chunks(2, 8, 7)
    with torch.no_grad():
        _ = model.norm_out.weight.add_(0.1)
    graph = SignatureGraph()
    with deterministic_algorithms():
        expected = anchor.profile(model, chunks, slices=3, graph=SignatureGraph(warm=100))
        actual = anchor.profile(model, chunks, slices=3, graph=graph)
        assert actual == expected
        assert graph.captures == 1
        assert graph.replays == 4
        other = Vesper(three_pass_config(), torch.Generator().manual_seed(7)).to(CUDA)
        _ = set_scan(other, "fused-fp32")
        changed = anchor.profile(other, chunks, slices=3, graph=graph)
        reference = anchor.profile(other, chunks, slices=3, graph=SignatureGraph(warm=100))
    assert changed == reference
    assert changed != actual
    assert graph.captures == 2
