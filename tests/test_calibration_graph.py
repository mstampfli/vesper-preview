"""The captured calibration update (halting_calibration.train_update with a ``WarmedGraph``).

The eager update on the same device with the same capturable optimizer is the truth: a captured
run must equal it BITWISE after every update: every record field, every parameter, every Adam
moment and step counter. Head-only and joint (with the dense auxiliary), under the taper floor and
the warmup, so the floor and both learning rates change every update.
"""

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.column import set_scan
from vesper.config import load
from vesper.contracts import ContractError
from vesper.exits import floored_stop_log_probs
from vesper.graphs import WARM_STEPS, WarmedGraph
from vesper.halting_calibration import (
    _floor_at,  # pyright: ignore[reportPrivateUsage]
    _label_weights,  # pyright: ignore[reportPrivateUsage]
    train_update,
)
from vesper.joint_update import (
    ROLLOUTS,
    Dense,
    optimizer,
    release_backbone,
    start_head,
    update_loss,
)
from vesper.kernels.delta_scan import fused_scan_available
from vesper.model import Vesper
from vesper.synthetic import TERMINAL_MIX, episodes, generator_for

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"
CUDA = torch.device("cuda")
EPISODES = 4
UPDATES = WARM_STEPS + 4  # three eager, one capture (its first replay), three more replays
needs_fused = pytest.mark.skipif(
    not fused_scan_available(), reason="captured steps need CUDA and the fused scan"
)


class _Trained:
    """What a run of updates leaves: each update's record and its parameters and Adam state."""

    def __init__(self, records: list[dict[str, object]], after: list[list[Tensor]]) -> None:
        self.records: list[dict[str, object]] = records
        self.after: list[list[Tensor]] = after  # per update: parameters, then m, v, step each


def _snapshot(params: list[torch.nn.Parameter], opt: torch.optim.AdamW) -> list[Tensor]:
    adam = cast("dict[Tensor, dict[str, Tensor]]", opt.state)
    state = [
        adam[p][name].detach().clone() for p in params for name in ("exp_avg", "exp_avg_sq", "step")
    ]
    return [p.detach().clone() for p in params] + state


def _trained(*, graph: WarmedGraph | None, joint: bool, dense_weight: float) -> _Trained:
    model = Vesper(load(CONFIG), torch.Generator().manual_seed(0)).to(CUDA)
    _ = set_scan(model, "fused-fp32")
    head = start_head(model)
    w_h = model.loop.halting_head().w_h
    with torch.no_grad():  # a head whose stops vary across tokens, so decisions differ
        _ = w_h.copy_(torch.randn(w_h.shape, generator=torch.Generator().manual_seed(7)))
    backbone = release_backbone(model) if joint else []
    opt = optimizer(head, backbone, capturable=True)
    data, action = generator_for("graph-train", 0), generator_for("graph-action", 0)
    t_max = model.loop.t_max
    records: list[dict[str, object]] = []
    after: list[list[Tensor]] = []
    for update in range(UPDATES):
        ep = episodes(data, EPISODES, f"graph-{update}", TERMINAL_MIX)
        n = int(ep.chunk.tokens.shape[1])
        uniforms = torch.rand(
            (t_max - 1, ROLLOUTS * EPISODES, n), generator=action, dtype=torch.float64
        )
        epsilon = _floor_at("taper", update)
        records.append(
            train_update(
                model,
                opt,
                head + backbone,
                ep,
                uniforms,
                epsilon,
                update,
                depth_cost=10.0,
                label_weight="class",
                dense_weight=dense_weight,
                graph=graph,
            )
        )
        after.append(_snapshot(head + backbone, opt))
    return _Trained(records, after)


@needs_fused
@pytest.mark.parametrize(
    ("joint", "dense_weight"),
    [(False, 0.0), (True, 0.0), (True, 0.1)],
    ids=["head", "joint", "dense"],
)
def test_the_captured_updates_equal_the_eager_updates_bitwise(
    *, joint: bool, dense_weight: float
) -> None:
    eager = _trained(graph=None, joint=joint, dense_weight=dense_weight)
    graph = WarmedGraph()
    captured = _trained(graph=graph, joint=joint, dense_weight=dense_weight)
    assert graph.captures == 1
    assert graph.replays == UPDATES - WARM_STEPS
    for update, (want, got) in enumerate(zip(eager.records, captured.records, strict=True)):
        assert got == want, update
    epsilons = {cast("float", r["epsilon"]) for r in captured.records}
    rates = {tuple(cast("list[float]", r["lr_groups"])) for r in captured.records}
    costs = {cast("float", r["cost"]) for r in captured.records}
    assert len(epsilons) == UPDATES  # the floor changed at every replay
    assert len(rates) == UPDATES  # and the rates
    assert len(costs) == UPDATES  # a new batch each
    for update, (want_all, got_all) in enumerate(zip(eager.after, captured.after, strict=True)):
        for index, (want_t, got_t) in enumerate(zip(want_all, got_all, strict=True)):
            assert torch.equal(want_t, got_t), (update, index)


def test_the_update_gives_the_dense_auxiliary_each_unique_episode_once() -> None:
    # the update slices the unique episodes from its rollout-major batch; the dense term it
    # trains on equals the term of the unique episodes given explicitly (before the step)
    model = Vesper(load(CONFIG), torch.Generator().manual_seed(0))
    head = start_head(model)
    backbone = release_backbone(model)
    ep = episodes(generator_for("dense-wiring", 0), 3, "d", TERMINAL_MIX)
    c = ep.chunk
    weights = _label_weights(ep.task, "class")
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    uniforms = torch.zeros((t_max - 1, ROLLOUTS * 3, n), dtype=torch.float64)

    def twice(t: Tensor) -> Tensor:
        return torch.cat([t] * ROLLOUTS)

    want = update_loss(
        model,
        twice(c.tokens),
        twice(c.pos),
        twice(c.targets),
        twice(c.loss_mask),
        twice(c.valid),
        twice(weights),
        uniforms,
        0.0,
        10.0,
        Dense(c.tokens, c.pos, c.targets, c.loss_mask, weights, 0.1),
    ).dense
    record = train_update(
        model,
        optimizer(head, backbone),
        head + backbone,
        ep,
        uniforms,
        0.0,
        0,
        depth_cost=10.0,
        label_weight="class",
        dense_weight=0.1,
    )
    assert record["dense"] == float(want.detach())
    assert float(want.detach()) > 0.0


def test_a_floor_tensor_outside_a_probability_is_refused_on_the_device() -> None:
    # a captured update reads the floor from a device tensor (it changes every update); the
    # check is a device contract then, a ContractError eagerly; a float keeps its ValueError
    z = torch.zeros(2, dtype=torch.float64)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ContractError, match="epsilon must lie in"):
            _ = floored_stop_log_probs(z, torch.tensor(bad, dtype=torch.float64))
        with pytest.raises(ValueError, match="epsilon must lie in"):
            _ = floored_stop_log_probs(z, bad)
    for other in (torch.tensor(0.25), torch.tensor(0.25, dtype=torch.float64, device="meta")):
        with pytest.raises(ContractError, match="epsilon: a float64 tensor on cpu"):
            _ = floored_stop_log_probs(z, other)  # another dtype or device: never cast silently
    good = floored_stop_log_probs(z, torch.tensor(0.25, dtype=torch.float64))
    same = floored_stop_log_probs(z, 0.25)
    assert all(torch.equal(a, b) for a, b in zip(good, same, strict=True))


def test_a_non_finite_update_raises_before_the_step_and_changes_nothing() -> None:
    model = Vesper(load(CONFIG), torch.Generator().manual_seed(0))
    head = start_head(model)
    with torch.no_grad():
        _ = model.loop.halting_head().b_h.fill_(float("nan"))  # every stop logit is NaN
    opt = optimizer(head, [])
    before = [p.detach().clone() for p in head]
    ep = episodes(generator_for("nan", 0), 2, "nan", TERMINAL_MIX)
    n = int(ep.chunk.tokens.shape[1])
    uniforms = torch.full((model.loop.t_max - 1, ROLLOUTS * 2, n), 0.5, dtype=torch.float64)
    with pytest.raises(FloatingPointError, match="non-finite cost, surrogate or gradient"):
        _ = train_update(model, opt, head, ep, uniforms, 0.0, 0)
    for want, got in zip(before, head, strict=True):
        torch.testing.assert_close(got, want, rtol=0.0, atol=0.0, equal_nan=True)
    assert len(cast("dict[Tensor, object]", opt.state)) == 0  # no step taken
