"""The halting exploration calibration (vesper/halting_calibration.py) and its two primitives.

Design: docs/reviews/2026-10-18-halting-exploration-calibration.md. The sampled stop's
probabilities (``vesper.exits.floored_stop_log_probs``), the sampled exit rule
(``vesper.loop.SampledRule``), the paired-baseline estimator against an enumerated oracle, the
frozen backbone, and the run's refusals, admission, record and identity.
"""

import hashlib
import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import NamedTuple, cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from vesper.column import ColumnState
from vesper.config import VesperConfig, load
from vesper.contracts import ContractError
from vesper.exits import HAZARD_LIMIT, floored_stop_log_probs, hazard_baseline
from vesper.halting_calibration import (
    DEPTH_COST,
    LABEL_WEIGHT,
    PREFLIGHT_UPDATES,
    REQUIRED_ACCURACY,
    SAVE_AT,
    TRAIN_MODE,
    UPDATES,
    _admit,  # pyright: ignore[reportPrivateUsage]
    _floor_at,  # pyright: ignore[reportPrivateUsage]
    _label_weights,  # pyright: ignore[reportPrivateUsage]
    evaluate_policy,
    fixed_depths,
    main,
    run,
    static_frontier,
    train_update,
    trial_protocol,
    verdict,
)
from vesper.halting_common import load_weights
from vesper.halting_experiment import run as experiment_run
from vesper.joint_update import (
    BACKBONE_LR,
    HEAD_BIAS,
    PEAK_LR,
    ROLLOUTS,
    Dense,
    actor_loss,
    learning_rate,
    optimizer,
    start_head,
    update_loss,
)
from vesper.loop import CdfRule, DenseRule, SampledRule
from vesper.model import Vesper
from vesper.synthetic import CLASSES, TERMINAL_MIX, Episodes, episodes, generator_for

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"
# the admission floor for an untrained test source, some of whose classes score exactly 0:
# below every accuracy, so the training path runs; recorded in the result like the real 0.5
NO_FLOOR = -1.0


def _vec(values: list[float]) -> Tensor:
    """A float64 1-D tensor, element by element (strict typing has no typed constructor)."""
    out = torch.zeros(len(values), dtype=torch.float64)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _cfg() -> VesperConfig:
    return load(CONFIG)


def _model(seed: int = 0) -> Vesper:
    return Vesper(_cfg(), torch.Generator().manual_seed(seed))


def _moderate_head(model: Vesper) -> None:
    """A head whose stop probabilities sit well inside (0, 1) for this model's outputs."""
    head = model.loop.halting_head()
    with torch.no_grad():
        _ = head.w_h.copy_(torch.randn(head.w_h.shape, generator=torch.Generator().manual_seed(7)))
        _ = head.b_h.fill_(1.0)


# --- the stop probabilities -------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.0, 0.2, 0.5, 1.0])
def test_the_floored_stop_is_a_mixture_with_finite_gradients(epsilon: float) -> None:
    z = _vec([-1e5, -30.0, -2.0, 0.0, 2.0, 30.0, 1e5]).requires_grad_()
    log_stop, log_continue = floored_stop_log_probs(z, epsilon)
    saturated = z.detach().clamp(-HAZARD_LIMIT, HAZARD_LIMIT)
    stop = (1.0 - epsilon) * torch.sigmoid(saturated)
    torch.testing.assert_close(log_stop.exp(), stop, rtol=1e-12, atol=1e-300)
    torch.testing.assert_close(log_continue.exp(), 1.0 - stop, rtol=1e-12, atol=1e-15)
    if epsilon == 1.0:
        assert bool(torch.isneginf(log_stop).all())  # a stop that cannot be sampled
        assert bool((log_continue == 0.0).all())
    else:
        assert bool(torch.isfinite(log_stop).all())
    assert bool(torch.isfinite(log_continue).all())
    for chosen in (log_stop, log_continue):
        (grad,) = torch.autograd.grad(chosen.sum(), z, retain_graph=True)
        assert bool(torch.isfinite(grad).all()), (epsilon, chosen)


def test_the_mixture_keeps_the_head_derivative_where_a_clamp_would_not() -> None:
    # Maintainer B's design: max(epsilon, continue) is flat where the floor binds; the mixture is
    # not
    z = _vec([2.0]).requires_grad_()  # sigmoid(2) = 0.88 stop
    _, log_continue = floored_stop_log_probs(z, 0.2)
    (mixed,) = torch.autograd.grad(log_continue.exp().sum(), z)
    clamped = torch.clamp_min(1.0 - torch.sigmoid(z), 0.2)  # binds: 0.12 < 0.2
    (flat,) = torch.autograd.grad(clamped.sum(), z)
    assert float(flat) == 0.0
    assert float(mixed) < 0.0  # more stop logit, less continuation, everywhere


def test_the_floor_must_be_a_probability() -> None:
    z = torch.zeros(2, dtype=torch.float64)
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="epsilon must lie in"):
            _ = floored_stop_log_probs(z, bad)


# --- the sampled rule ------------------------------------------------------------------------


def _manual_trace(
    model: Vesper, per_pass: Tensor, uniforms: Tensor, epsilon: float
) -> tuple[Tensor, Tensor]:
    """Exits and summed log-probabilities recomputed from the pass outputs, token by token."""
    head = model.loop.halting_head()
    t_max = per_pass.shape[0]
    baseline = hazard_baseline(t_max)
    b, n = per_pass.shape[1], per_pass.shape[2]
    exits = torch.full((b, n), t_max, dtype=torch.int32)
    log_prob = torch.zeros(b, n, dtype=torch.float64)
    with torch.no_grad():
        for t in range(t_max - 1):
            z = head.logit(per_pass[t], t).double() + baseline[t]
            stop_p = (1.0 - epsilon) * torch.sigmoid(z)
            open_ = exits == t_max
            stops = open_ & (uniforms[t] < stop_p)
            taken = torch.where(stops, stop_p.log(), torch.log1p(-stop_p))
            log_prob = log_prob + torch.where(open_, taken, 0.0)
            exits = torch.where(stops, torch.full_like(exits, t + 1), exits)
    return exits, log_prob


def test_the_sampled_rule_stops_where_the_uniform_falls_below_the_stop_probability() -> None:
    model = _model()
    _moderate_head(model)
    ep = episodes(generator_for("rule", 0), 3, "r")
    c = ep.chunk
    t_max = model.loop.t_max
    uniforms = torch.rand(
        (t_max - 1, 3, c.tokens.shape[1]),
        generator=torch.Generator().manual_seed(1),
        dtype=torch.float64,
    )
    deciding = torch.ones_like(c.valid)
    deciding[1, 5:9] = False  # not deciding: no log-probability, no count
    rule = SampledRule(model.loop.halting_head(), 0.2, uniforms, deciding)
    with torch.no_grad():
        result = model.run(
            c.tokens,
            c.pos,
            model.init_states(3, torch.float32, torch.device("cpu")),
            exit_rule=rule,
        )
    exits, expected = _manual_trace(model, result.per_pass, uniforms, 0.2)
    assert torch.equal(result.n_iters, exits)
    assert bool((exits < t_max).any())  # both kinds decided: early stops
    assert bool((exits == t_max).any())  # and tokens that ran to the forced exit
    assert rule.log_prob is not None
    assert rule.decisions is not None
    expected = torch.where(deciding, expected, 0.0)
    torch.testing.assert_close(rule.log_prob.detach(), expected, rtol=1e-6, atol=1e-9)
    decisions = torch.where(deciding, torch.clamp(exits, max=t_max - 1), 0)
    assert torch.equal(rule.decisions, decisions.to(torch.int32))
    # each token's output is its exit pass's own
    index = (exits.long() - 1)[None, :, :, None].expand(1, *result.per_pass.shape[1:])
    torch.testing.assert_close(result.y, result.per_pass.gather(0, index)[0])


def test_the_sampled_rule_refuses_uniforms_of_another_shape_type_or_range() -> None:
    model = _model()
    ep = episodes(generator_for("rule", 0), 2, "r")
    c = ep.chunk
    t_max = model.loop.t_max
    n = int(c.tokens.shape[1])
    good = torch.zeros((t_max - 1, 2, n), dtype=torch.float64)
    for uniforms, deciding, match in (
        (good[:-1], c.valid, f"uniforms: axis 0 must be {t_max - 1}"),
        (good[:, :1], c.valid, "uniforms: axis 1 'b' is 1"),
        (good.float(), c.valid, "uniforms: dtype must be torch.float64"),
        (good + 1.0, c.valid, r"uniforms must lie in \[0, 1\)"),
        (good - 1e-3, c.valid, r"uniforms must lie in \[0, 1\)"),
        (good, c.valid[:, :-1], "deciding: axis 1 'n' is"),
        (good, c.valid.long(), "deciding: dtype must be torch.bool"),
    ):
        rule = SampledRule(model.loop.halting_head(), 0.0, uniforms, deciding)
        states = model.init_states(2, torch.float32, torch.device("cpu"))
        with pytest.raises(ContractError, match=match):
            _ = model.run(c.tokens, c.pos, states, exit_rule=rule)


def test_a_replay_with_the_same_uniforms_takes_the_same_decisions_in_chunks_or_tokens() -> None:
    # float64 like tests/test_key_conv.py's chunked equality: one forward, a token-by-token
    # decode and uneven chunks, given the matching slices of one uniform array, make the same
    # decisions and leave the same outputs, log-probabilities and carried states
    f64, cpu = torch.float64, torch.device("cpu")
    model = _model().to(f64).eval()
    _moderate_head(model)
    ep = episodes(generator_for("replay", 0), 2, "r")
    c = ep.chunk
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    uniforms = torch.rand((t_max - 1, 2, n), generator=torch.Generator().manual_seed(3), dtype=f64)

    def chunked(sizes: list[int]) -> tuple[Tensor, Tensor, Tensor, list[ColumnState]]:
        states = model.init_states(2, f64, cpu)
        ys: list[Tensor] = []
        iters: list[Tensor] = []
        logs: list[Tensor] = []
        start = 0
        for size in sizes:
            end = start + size
            rule = SampledRule(
                model.loop.halting_head(),
                0.2,
                uniforms[:, :, start:end].contiguous(),
                c.valid[:, start:end],
            )
            with torch.no_grad():
                result = model.run(
                    c.tokens[:, start:end], c.pos[:, start:end], states, exit_rule=rule
                )
            states = list(result.states)
            assert rule.log_prob is not None
            ys.append(result.y)
            iters.append(result.n_iters)
            logs.append(rule.log_prob)
            start = end
        assert start == n
        return torch.cat(ys, 1), torch.cat(iters, 1), torch.cat(logs, 1), states

    whole = chunked([n])
    exits = whole[1]
    assert bool((exits < t_max).any())  # both kinds decided: early stops
    assert bool((exits == t_max).any())  # and tokens that ran to the forced exit
    for sizes in ([1] * n, [5, 0, 30, 1, 52], [n // 2, n - n // 2]):
        y, d, lp, states = chunked(sizes)
        assert torch.equal(d, exits), sizes
        assert torch.allclose(y, whole[0], rtol=0, atol=1e-10), sizes
        assert torch.allclose(lp, whole[2], rtol=0, atol=1e-10), sizes
        for a, b in zip(states, whole[3], strict=True):
            for ha, hb in zip(a.h, b.h, strict=True):
                assert torch.allclose(ha, hb, rtol=0, atol=1e-10), sizes


# --- the estimator ----------------------------------------------------------------------------


def _oracle_terms(z: Tensor, epsilon: float) -> list[tuple[Tensor, float]]:
    """The design's two-decision oracle through this module's stop probabilities.

    Two decisions, each CONTINUE with probability ``1 - s``, ``s = (1 - eps) sigmoid(-z)``
    (z is the continue logit). Early work helps only a later reader: the cost is 1 unless both
    continue, plus 0.05 per continuation. Returns ``(log-probability, cost)`` per trajectory.
    """
    log_stop, log_continue = floored_stop_log_probs(-z, epsilon)
    terms: list[tuple[Tensor, float]] = []
    for a, b in itertools.product((0, 1), repeat=2):
        log_p = (log_continue[0] if a else log_stop[0]) + (log_continue[1] if b else log_stop[1])
        terms.append((log_p, 1.0 - a * b + 0.05 * (a + b)))
    return terms


@pytest.mark.parametrize("epsilon", [0.0, 0.2, 0.5])
def test_the_oracle_s_exact_gradient_through_the_floored_stop(epsilon: float) -> None:
    z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    exact = sum(log_p.exp() * cost for log_p, cost in _oracle_terms(z, epsilon))
    (grad,) = torch.autograd.grad(cast("Tensor", exact), z)
    expected = (0.05 - (epsilon + (1 - epsilon) * 0.5)) * (1 - epsilon) * 0.25
    torch.testing.assert_close(grad, torch.full_like(grad, expected), rtol=0, atol=1e-14)


def test_the_paired_baseline_estimator_is_exact_and_an_own_baseline_halves_it() -> None:
    # the design's oracle at epsilon .2: exact gradient (-.11, -.11); the paired surrogate,
    # in expectation over two independent trajectories, reproduces it through actor_loss
    z = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    terms = _oracle_terms(z, 0.2)
    paired = torch.zeros((), dtype=torch.float64)
    own = torch.zeros((), dtype=torch.float64)
    for (lp1, c1), (lp2, c2) in itertools.product(terms, repeat=2):
        weight = (lp1 + lp2).exp().detach()
        costs = _vec([c1, c2]).view(ROLLOUTS, 1)
        paired = paired + weight * actor_loss(costs, torch.stack([lp1, lp2]).view(ROLLOUTS, 1))
        mean = (c1 + c2) / 2  # a baseline that includes the rollout's own cost
        own = own + weight * ((c1 - mean) * lp1 + (c2 - mean) * lp2) / 2
    (g,) = torch.autograd.grad(paired, z, retain_graph=True)
    (bad,) = torch.autograd.grad(own, z)
    torch.testing.assert_close(g, torch.full_like(g, -0.11), rtol=0, atol=1e-14)
    torch.testing.assert_close(bad, torch.full_like(bad, -0.055), rtol=0, atol=1e-14)


def test_the_surrogate_sends_no_gradient_into_the_costs_and_refuses_other_shapes() -> None:
    costs = _vec([1.0, 2.0, 3.0, 0.5]).view(2, 2).requires_grad_()
    log_probs = _vec([-1.0, -2.0, -0.5, -1.5]).view(2, 2).requires_grad_()
    loss = actor_loss(costs, log_probs)
    grads = torch.autograd.grad(loss, [costs, log_probs], allow_unused=True)
    grad_costs, grad_logs = cast("tuple[Tensor | None, Tensor]", tuple(grads))
    assert grad_costs is None
    advantage = costs.detach() - costs.detach().flip(0)
    torch.testing.assert_close(grad_logs, advantage / 4)
    for c, lp, match in (
        (costs[:1], log_probs[:1], "costs: axis 0 must be 2"),
        (costs, log_probs[:, :1], "log_probs: axis 1 'e' is 1"),
        (costs.flatten(), log_probs.flatten(), "costs: expected rank 2"),
        (torch.zeros(2, 0), torch.zeros(2, 0), "no episode"),
    ):
        with pytest.raises(ContractError, match=match):
            _ = actor_loss(c, lp)


# --- schedule and start -------------------------------------------------------------------------


def test_the_floor_and_rate_schedules() -> None:
    assert [_floor_at("taper", u) for u in (0, 750, 1500, 1999)] == [0.5, 0.25, 0.0, 0.0]
    assert {_floor_at("none", u) for u in (0, 750, 1999)} == {0.0}
    with pytest.raises(ValueError, match="floor must be one of"):
        _ = _floor_at("linear", 0)
    with pytest.raises(ValueError, match="non-negative"):
        _ = _floor_at("taper", -1)
    assert learning_rate(0) == pytest.approx(PEAK_LR / 100)
    assert learning_rate(99) == PEAK_LR
    assert learning_rate(1999) == PEAK_LR


def test_the_start_freezes_the_backbone_and_its_policy_exits_at_the_last_pass() -> None:
    model = _model(5)
    params = start_head(model)
    head = model.loop.halting_head()
    trained = {id(p) for p in model.parameters() if p.requires_grad}
    assert trained == {id(p) for p in head.parameters()} == {id(p) for p in params}
    assert float(head.b_h.detach()) == pytest.approx(HEAD_BIAS)
    # the hazard at pass t < T is 1 / (1 + 9 (T - t)) whatever the backbone computes, and the
    # cumulative exit probability before the forced exit stays below 1/2
    t_max = model.loop.t_max
    z = HEAD_BIAS + hazard_baseline(t_max)[:-1]
    hazards = torch.sigmoid(z)
    expected = 1.0 / (1.0 + 9.0 * (t_max - torch.arange(1, t_max, dtype=torch.float64)))
    torch.testing.assert_close(hazards, expected)
    assert float(1.0 - torch.prod(1.0 - hazards)) < 0.5
    ep = episodes(generator_for("start", 0), 2, "s")
    c = ep.chunk
    cpu = torch.device("cpu")
    with torch.no_grad():
        dense = model.run(
            c.tokens, c.pos, model.init_states(2, torch.float32, cpu), exit_rule=DenseRule()
        )
        live = model.run(
            c.tokens, c.pos, model.init_states(2, torch.float32, cpu), exit_rule=CdfRule(head, 0.5)
        )
    assert bool((live.n_iters == t_max).all())
    assert torch.equal(live.y, dense.y)


def test_training_moves_only_the_head_and_its_records_are_finite() -> None:
    model = _model(2)
    params = start_head(model)
    _ = model.eval()
    _moderate_head(model)
    before = {k: v.clone() for k, v in cast("dict[str, Tensor]", model.state_dict()).items()}
    opt = optimizer(params, [])
    data = generator_for("frozen", 0)
    action = generator_for("frozen-action", 0)
    t_max = model.loop.t_max
    for update in range(3):
        ep = episodes(data, 4, f"f-{update}")
        n = int(ep.chunk.tokens.shape[1])
        uniforms = torch.rand((t_max - 1, ROLLOUTS * 4, n), generator=action, dtype=torch.float64)
        record = train_update(model, opt, params, ep, uniforms, 0.2, update)
        for key in ("cost", "nll", "depth", "surrogate", "grad_norm"):
            assert math.isfinite(cast("float", record[key])), key
        assert cast("int", record["decisions"]) > 0
    after = cast("dict[str, Tensor]", model.state_dict())
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved
    assert all(k.startswith("loop.halting") for k in moved), moved


@pytest.mark.parametrize(
    ("depth_cost", "scheme"), [(0.02, "episode"), (1.0, "episode"), (1.0, "class")]
)
def test_an_update_minimizes_the_paired_surrogate_of_the_trajectory_cost(
    depth_cost: float, scheme: str
) -> None:
    # recomputed here from the same forward: cost = the label NLL term + depth_cost * mean depth
    # / T, the decisions' log-probabilities SUMMED per trajectory, each rollout against the
    # other; the NLL term is each episode's mean ("episode") or class-balanced over the unique
    # episodes ("class": a label of class c weighs E / (C * N_c), counted here by hand)
    model = _model(4)
    params = start_head(model)
    _ = model.eval()
    _moderate_head(model)
    episodes_ = 3
    ep = episodes(generator_for("surrogate", 0), episodes_, "s")
    c = ep.chunk
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    uniforms = torch.rand(
        (t_max - 1, ROLLOUTS * episodes_, n),
        generator=torch.Generator().manual_seed(5),
        dtype=torch.float64,
    )
    tokens, pos, targets, mask, valid = (
        torch.cat([t] * ROLLOUTS) for t in (c.tokens, c.pos, c.targets, c.loss_mask, c.valid)
    )
    rule = SampledRule(model.loop.halting_head(), 0.3, uniforms, valid)
    with torch.no_grad():
        result = model.run(
            tokens,
            pos,
            model.init_states(ROLLOUTS * episodes_, torch.float32, torch.device("cpu")),
            exit_rule=rule,
        )
        logits = model.head(result.y).double()
    assert rule.log_prob is not None
    nll = torch.stack(
        [
            functional.cross_entropy(logits[i][mask[i]], targets[i][mask[i]].long())
            for i in range(ROLLOUTS * episodes_)
        ]
    )
    unique_labels = [int(c) for c in ep.task[ep.task >= 0].unbind()]
    pooled = {c: unique_labels.count(c) for c in set(unique_labels)}
    tasks = torch.cat([ep.task] * ROLLOUTS)
    per_label: list[list[tuple[int, float]]] = []  # (class, ce) of every label, per trajectory
    for i in range(ROLLOUTS * episodes_):
        ce = functional.cross_entropy(
            logits[i][mask[i]], targets[i][mask[i]].long(), reduction="none"
        )
        per_label.append(
            [
                (int(c), float(v))
                for c, v in zip(tasks[i][mask[i]].unbind(), ce.unbind(), strict=True)
            ]
        )
    if scheme == "episode":
        objective = _vec([sum(v for _, v in row) / len(row) for row in per_label])
    else:
        weight = {c: episodes_ / (len(pooled) * n_c) for c, n_c in pooled.items()}
        objective = _vec([sum(weight[c] * v for c, v in row) for row in per_label])
    depth = result.n_iters.double().mean(1)  # every token of the task is real
    cost = (objective + depth_cost * depth / t_max).view(ROLLOUTS, episodes_)
    log_probs = rule.log_prob.detach().sum(1).view(ROLLOUTS, episodes_)
    expected = float(((cost - cost.flip(0)) * log_probs).mean())
    opt = optimizer(params, [])
    record = train_update(model, opt, params, ep, uniforms, 0.3, 0, depth_cost, scheme)
    assert record["surrogate"] == pytest.approx(expected, rel=1e-5)
    assert record["nll"] == pytest.approx(float(nll.mean()), rel=1e-5)  # ordinary, any scheme
    assert record["objective_nll"] == pytest.approx(float(objective.mean()), rel=1e-5)
    assert record["depth"] == pytest.approx(float(depth.mean()))
    every = [label for row in per_label for label in row]
    class_nll = cast("list[float | None]", record["class_nll"])
    assert len(class_nll) == len(CLASSES)
    for k, value in enumerate(class_nll):  # pooled over both rollouts; None for an absent class
        values = [v for c, v in every if c == k]
        if values:
            assert value == pytest.approx(sum(values) / len(values), rel=1e-5), CLASSES[k]
        else:
            assert value is None, CLASSES[k]


# --- evaluation and verdict ----------------------------------------------------------------------


def test_the_starting_policy_scores_as_the_full_depth_model() -> None:
    model = _model(3)
    _ = start_head(model)
    batches = [episodes(generator_for("eval", 0), 2, f"e{i}") for i in range(2)]
    fixed, full = fixed_depths(model, batches)
    evaluation = evaluate_policy(model, batches, full)
    policy = cast("dict[str, object]", evaluation["policy"])
    t_max = model.loop.t_max
    assert policy["accuracy"] == cast("dict[str, object]", fixed["full"])["accuracy"]
    assert policy["depth_all"] == float(t_max)
    static = cast("list[dict[str, object]]", fixed["static"])
    assert static[-1]["accuracy"] == cast("dict[str, object]", fixed["full"])["accuracy"]
    paired = cast("dict[str, dict[str, float]]", evaluation["paired_vs_full_depth"])
    assert paired["overall"] == {"mean": 0.0, "low": 0.0, "high": 0.0}
    assert evaluation["executed_passes"] == t_max
    # a policy at full depth: its random control draws only the last pass
    for control in cast("list[dict[str, object]]", evaluation["random"]):
        assert control["depth_difference"] == 0.0


def _verdict_inputs(
    accuracy: float, depth: float, random: float, static_curve: list[float]
) -> tuple[dict[str, object], dict[str, object]]:
    full = dict.fromkeys(CLASSES, 0.9) | {"overall": 0.9}
    policy = dict.fromkeys(CLASSES, accuracy) | {"overall": accuracy}
    terminal: dict[str, object] = {
        "policy": {"accuracy": policy, "depth_all": depth},
        "random": [{"accuracy": {"overall": random}, "depth_difference": 0.25}],
    }
    fixed: dict[str, object] = {
        "full": {"accuracy": full, "depth_all": 8.0},
        "static": [
            {"accuracy": {"overall": v}, "depth_all": float(d)}
            for d, v in enumerate(static_curve, start=1)
        ],
    }
    return terminal, fixed


TRIAL = trial_protocol(2000, 32, (0, 500, 1000, 1500, 2000), 0.5, preflight=False)


def test_the_verdict_needs_every_criterion() -> None:
    curve = [0.2, 0.5, 0.7, 0.8, 0.85, 0.88, 0.89, 0.9]
    good = verdict(*_verdict_inputs(0.895, 6.0, 0.8, curve), TRIAL)
    assert good["promising"] is True
    assert good["random_depth_difference"] == 0.25  # the control's measured cost, kept
    checks = {
        "every_class_within_one_point_and_above_half": _verdict_inputs(0.885, 6.0, 0.8, curve),
        "depth_reduced_by_20_percent": _verdict_inputs(0.895, 6.5, 0.8, curve),
        "above_random_control": _verdict_inputs(0.895, 6.0, 0.9, curve),
        "above_static_frontier": _verdict_inputs(0.895, 6.0, 0.8, [*curve[:5], 0.9, 0.9, 0.9]),
    }
    for name, inputs in checks.items():
        result = verdict(*inputs, TRIAL)
        failing = cast("dict[str, bool]", result["checks"])
        assert result["promising"] is False, name
        assert [k for k, v in failing.items() if not v] == [name]


def test_the_static_frontier_takes_every_cheaper_point_and_two_depth_mixture() -> None:
    curve = [0.2, 0.5, 0.7, 0.8, 0.85, 0.88, 0.89, 0.9]
    points = [(float(d), a) for d, a in enumerate(curve, start=1)]
    # between neighbours the frontier is their mixture: 0.88 + 0.5 * (0.89 - 0.88)
    assert static_frontier(points, 6.5) == pytest.approx(0.885)
    # Maintainer B's fixture (review of cc69a25): a cheaper fixed depth beats the candidate
    cheaper = [0.95, *curve[1:]]
    result = verdict(*_verdict_inputs(0.895, 6.0, 0.8, cheaper), TRIAL)
    assert result["static_frontier_at_policy_depth"] == pytest.approx(0.95)
    assert result["promising"] is False
    # a mixture of non-adjacent depths: 1/3 of episodes at depth 2, 2/3 at depth 8 has mean
    # depth 6 and accuracy .5/3 + .99 * 2/3, above every point at or below 6
    valley = [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.99]
    mixed = [(float(d), a) for d, a in enumerate(valley, start=1)]
    assert static_frontier(mixed, 6.0) == pytest.approx(0.5 / 3 + 0.99 * 2 / 3)
    assert verdict(*_verdict_inputs(0.8, 6.0, 0.1, valley), TRIAL)["checks"] == {
        "every_class_within_one_point_and_above_half": False,
        "depth_reduced_by_20_percent": True,
        "above_random_control": True,
        "above_static_frontier": False,
    }
    assert static_frontier(points, 0.5) == -math.inf  # nothing that cheap was measured


def test_only_the_trial_protocol_is_assessed() -> None:
    # Maintainer B's review of cc69a25: a shortened or preflight run reports diagnostic values only
    assert TRIAL["trial"] is True
    assert TRIAL["eval_episodes"] == 1024
    curve = [0.2, 0.5, 0.7, 0.8, 0.85, 0.88, 0.89, 0.9]
    for protocol, differences in (
        (trial_protocol(3, 32, (0, 500, 1000, 1500, 2000), 0.5, preflight=False), ["updates"]),
        (
            trial_protocol(2000, 1, (0, 500, 1000, 1500, 2000), 0.5, preflight=False),
            ["eval_batches"],
        ),
        (trial_protocol(2000, 32, (0, 2000), 0.5, preflight=False), ["save_at"]),
        (
            trial_protocol(2000, 32, (0, 500, 1000, 1500, 2000), -1.0, preflight=False),
            ["required_accuracy"],
        ),
        (trial_protocol(2000, 32, (0, 500, 1000, 1500, 2000), 0.5, preflight=True), ["preflight"]),
        (
            trial_protocol(
                2000, 32, (0, 500, 1000, 1500, 2000), 0.5, preflight=False, depth_cost=1.0
            ),
            ["depth_cost"],
        ),
    ):
        assert protocol["trial"] is False
        assert protocol["differences_from_trial"] == differences
        result = verdict(*_verdict_inputs(0.895, 6.0, 0.8, curve), protocol)
        assert result["criteria_met"] is True
        assert result["promising"] is None
        assert result["assessment"] == "diagnostic"
        assert "DIAGNOSTIC" in cast("str", result["scope"])
        counted = f"on {protocol['eval_episodes']} episodes after {protocol['updates']} updates"
        assert counted in cast("str", result["scope"])
    assert "on 1024 episodes after 2000 updates" in cast(
        "str", verdict(*_verdict_inputs(0.895, 6.0, 0.8, curve), TRIAL)["scope"]
    )


# --- the run ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("source") / "run"
    _ = experiment_run(CONFIG, "fixed", 0, out, steps=2, eval_batches=1, eval_size=2)
    return out / "model.pt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small(source: Path, out: Path, *, seed: int = 0, floor: str = "taper") -> dict[str, object]:
    return run(
        source,
        _sha(source),
        seed,
        floor,
        out,
        updates=3,
        eval_batches=1,
        save_at=(0, 3),
        required_accuracy=NO_FLOOR,
    )


def test_a_run_trains_the_head_and_records_its_identity(tmp_path: Path, source: Path) -> None:
    out = tmp_path / "taper"
    result = _small(source, out)
    assert {p.name for p in out.iterdir()} == {
        "config.yaml",
        "train.jsonl",
        "eval.jsonl",
        "heads",
        "result.json",
    }
    assert sorted(p.name for p in (out / "heads").iterdir()) == ["after-00000.pt", "after-00003.pt"]
    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (out / "train.jsonl").read_text("utf-8").splitlines()
    ]
    assert [line["update"] for line in lines] == [0, 1, 2]
    assert [line["epsilon"] for line in lines] == [_floor_at("taper", u) for u in range(3)]
    record = cast("dict[str, object]", json.loads((out / "result.json").read_text("utf-8")))
    assert record["source"] == result["source"]
    assert cast("dict[str, object]", record["source"])["sha256"] == _sha(source)
    assert record["head_start"] == {"direction": 0.0, "gain": 1.0, "bias": HEAD_BIAS}
    assert cast("dict[str, object]", record["admission"])["admitted"] is True
    assert cast("dict[str, object]", record["admission"])["required_accuracy"] == NO_FLOOR
    assert set(cast("dict[str, object]", record["evaluations"])) == {"0", "3"}
    judged = cast("dict[str, object]", record["verdict"])
    assert judged["assessment"] == "diagnostic"  # 3 updates on 32 episodes: not the trial
    assert judged["promising"] is None
    assert cast("dict[str, object]", record["protocol"])["eval_episodes"] == 32
    assert (out / "config.yaml").read_bytes() == (source.parent / "config.yaml").read_bytes()
    heads = [
        cast("dict[str, Tensor]", torch.load(out / "heads" / f, weights_only=True))
        for f in ("after-00000.pt", "after-00003.pt")
    ]
    assert any(not torch.equal(heads[0][k], heads[1][k]) for k in heads[0])
    # the same seed under the other floor consumes the same data and uniforms; another seed not
    other_floor = _small(source, tmp_path / "none", floor="none")
    other_seed = _small(source, tmp_path / "seed1", seed=1)
    keys = ("consumed_digest", "draw_digest", "initial_state_digest")
    assert [result[k] for k in keys] == [other_floor[k] for k in keys]
    assert result["consumed_digest"] != other_seed["consumed_digest"]
    assert result["draw_digest"] != other_seed["draw_digest"]
    # both chains recomputed from independent draws, step by step as logged
    data = generator_for("halting-head-train", 0)
    action = generator_for("halting-head-action", 0)
    consumed, draws = "", ""
    for update, line in enumerate(lines):
        c = episodes(data, 32, "x", TERMINAL_MIX).chunk
        consumed = _chain(
            consumed, [c.tokens, c.targets, c.valid, c.doc_start, c.segment, c.loss_mask, c.pos]
        )
        uniforms = torch.rand((7, ROLLOUTS, 32, 88), generator=action, dtype=torch.float64)
        draws = _chain(draws, [uniforms.reshape(7, ROLLOUTS * 32, 88)])
        assert (line["consumed_digest"], line["draw_digest"]) == (consumed, draws), update
    assert (result["consumed_digest"], result["draw_digest"]) == (consumed, draws)


def _chain(previous: str, tensors: list[Tensor]) -> str:
    """The digest chain written out independently: sha256 over dtype, shape and bytes."""
    h = hashlib.sha256(previous.encode())
    for t in tensors:
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.contiguous().numpy().tobytes())  # type: ignore[misc]  # ndarray: generic over Any
    return h.hexdigest()


def test_the_admission_refuses_a_head_that_is_not_at_its_start(source: Path) -> None:
    # the design's check that the initial deterministic policy exits every token at the last
    # pass with the dense output: a head that already stops some tokens fails it
    record = cast(
        "dict[str, object]", json.loads((source.parent / "result.json").read_text("utf-8"))
    )
    model = _model()
    load_weights(model, cast("dict[str, Tensor]", torch.load(source, weights_only=True)))
    _ = start_head(model)
    # 32 episodes: every class labelled (with 2, read3 and read4 are unseen, and an unseen
    # class fails admission, which a test of another check must not depend on)
    batches = [episodes(generator_for("halting-head-eval", 0), 32, "e")]
    assert _admit(model, record, batches, NO_FLOOR)["admitted"] is True
    _moderate_head(model)
    admission = _admit(model, record, batches, NO_FLOOR)
    assert admission["initial_policy_exits_last_with_dense_output"] is False
    assert admission["source_counts_reproduced"] is True  # the dense counts do not read the head
    assert admission["admitted"] is False


def test_a_run_is_refused_before_anything_is_written(tmp_path: Path, source: Path) -> None:
    out = tmp_path / "refused"
    good = _sha(source)
    cases: list[tuple[dict[str, object], type[Exception], str]] = [
        ({"source_sha256": "0" * 64}, ValueError, "is not the required"),
        ({"floor": "linear"}, ValueError, "floor must be one of"),
        ({"device": "tpu"}, ValueError, "device must be one of"),
        ({"save_at": (0, 4)}, ValueError, "every save point"),
        ({"updates": 0}, ValueError, "at least 1"),
        ({"label_weight": "token"}, ValueError, "unknown label weight"),
        ({"depth_cost": math.nan}, ValueError, "depth cost"),
        ({"depth_cost": math.inf}, ValueError, "depth cost"),
        ({"depth_cost": -1.0}, ValueError, "depth cost"),
        ({"required_accuracy": math.nan}, ValueError, "required accuracy"),
        ({"required_accuracy": math.inf}, ValueError, "required accuracy"),
        ({"required_accuracy": -math.inf}, ValueError, "required accuracy"),
    ]
    for change, error, match in cases:
        arguments: dict[str, object] = {
            "source_sha256": good,
            "floor": "taper",
            "device": "cpu",
            "save_at": (0, 3),
            "updates": 3,
            "label_weight": "episode",
            "depth_cost": 1.0,
            "required_accuracy": NO_FLOOR,
        } | change
        with pytest.raises(error, match=match):
            _ = run(
                source,
                cast("str", arguments["source_sha256"]),
                0,
                cast("str", arguments["floor"]),
                out,
                device=cast("str", arguments["device"]),
                updates=cast("int", arguments["updates"]),
                eval_batches=1,
                save_at=cast("tuple[int, ...]", arguments["save_at"]),
                required_accuracy=cast("float", arguments["required_accuracy"]),
                label_weight=cast("str", arguments["label_weight"]),
                depth_cost=cast("float", arguments["depth_cost"]),
            )
        assert not out.exists(), change


def test_an_admission_failure_is_recorded_and_stops_the_run(tmp_path: Path, source: Path) -> None:
    # the untrained source cannot score above 50 % on every class: reported, never trained
    out = tmp_path / "admission"
    with pytest.raises(RuntimeError, match="admission failed"):
        _ = run(source, _sha(source), 0, "none", out, updates=3, eval_batches=1, save_at=(0, 3))
    record = cast("dict[str, object]", json.loads((out / "result.json").read_text("utf-8")))
    admission = cast("dict[str, object]", record["admission"])
    assert admission["admitted"] is False
    assert admission["every_class_above_required"] is False
    assert admission["source_counts_reproduced"] is True
    assert admission["initial_policy_exits_last_with_dense_output"] is True
    assert not (out / "train.jsonl").exists()


def test_a_source_whose_counts_do_not_reproduce_is_not_admitted(
    tmp_path: Path, source: Path
) -> None:
    copied = tmp_path / "copied"
    copied.mkdir()
    for name in ("model.pt", "config.yaml"):
        _ = (copied / name).write_bytes((source.parent / name).read_bytes())
    record = cast(
        "dict[str, object]", json.loads((source.parent / "result.json").read_text("utf-8"))
    )
    counts = cast("dict[str, list[list[list[int]]]]", record["counts"])
    counts["dense_pass"][0][0][0] += 1  # one count that the weights do not give
    _ = (copied / "result.json").write_text(json.dumps(record), encoding="utf-8")
    out = tmp_path / "not-reproduced"
    with pytest.raises(RuntimeError, match="admission failed"):
        _ = run(
            copied / "model.pt",
            _sha(source),
            0,
            "none",
            out,
            updates=3,
            eval_batches=1,
            save_at=(0, 3),
            required_accuracy=NO_FLOOR,
        )
    written = cast("dict[str, object]", json.loads((out / "result.json").read_text("utf-8")))
    admission = cast("dict[str, object]", written["admission"])
    assert admission["source_counts_reproduced"] is False


def _cli(source: Path, out: Path, flags: list[str]) -> None:
    """``main`` on ``source`` into ``out``; the process-wide thread count is restored after it,
    whatever it raises (``main`` sets it from ``--threads``)."""
    threads = torch.get_num_threads()
    try:
        _ = main(
            [
                "--source",
                str(source),
                "--source-sha256",
                _sha(source),
                "--seed",
                "0",
                "--floor",
                "none",
                "--out",
                str(out),
                "--eval-batches",
                "1",
                *flags,
            ]
        )
    finally:
        torch.set_num_threads(threads)


def test_a_command_line_test_leaves_the_thread_count_as_it_found_it(
    tmp_path: Path, source: Path
) -> None:
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with pytest.raises(RuntimeError, match="admission failed"):
            _cli(source, tmp_path / "threads", ["--threads", "2"])
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize(
    ("flags", "match"),
    [
        (["--depth-cost", "nan"], "depth cost"),
        (["--depth-cost", "inf"], "depth cost"),
        (["--required-accuracy", "nan"], "required accuracy"),
    ],
)
def test_the_command_line_refuses_a_nonfinite_option_before_anything_is_written(
    tmp_path: Path, source: Path, flags: list[str], match: str
) -> None:
    out = tmp_path / "refused"
    with pytest.raises(ValueError, match=match):
        _cli(source, out, flags)
    assert not out.exists()


@pytest.mark.parametrize(
    ("flags", "expected", "differences"),
    [
        (
            [],
            (UPDATES, DEPTH_COST, list(SAVE_AT), REQUIRED_ACCURACY, LABEL_WEIGHT, TRAIN_MODE, 0.0),
            ["eval_batches"],
        ),
        (
            ["--preflight"],
            (
                PREFLIGHT_UPDATES,
                DEPTH_COST,
                list(SAVE_AT),
                REQUIRED_ACCURACY,
                LABEL_WEIGHT,
                TRAIN_MODE,
                0.0,
            ),
            ["updates", "eval_batches", "preflight"],
        ),
        (
            [
                "--updates",
                "3",
                "--depth-cost",
                "20",
                "--save-at",
                "0",
                "2",
                "3",
                "--required-accuracy",
                "0.25",
                "--label-weight",
                "class",
                "--train",
                "joint",
                "--dense-weight",
                "0.1",
            ],
            (3, 20.0, [0, 2, 3], 0.25, "class", "joint", 0.1),
            [
                "updates",
                "eval_batches",
                "save_at",
                "required_accuracy",
                "depth_cost",
                "label_weight",
                "train",
                "dense_weight",
            ],
        ),
    ],
)
def test_the_command_line_hands_every_option_to_the_run(
    tmp_path: Path,
    source: Path,
    flags: list[str],
    expected: tuple[object, ...],
    differences: list[str],
) -> None:
    # the untrained source is refused at admission, after the record of the run's budget is
    # written; that record shows what the command line handed over
    out = tmp_path / "cli"
    with pytest.raises(RuntimeError, match="admission failed"):
        _cli(source, out, flags)
    record = cast("dict[str, object]", json.loads((out / "result.json").read_text("utf-8")))
    admission = cast("dict[str, object]", record["admission"])
    protocol = cast("dict[str, object]", record["protocol"])
    got = (
        record["updates"],
        record["depth_cost"],
        record["save_at"],
        admission["required_accuracy"],
        record["label_weight"],
        record["train"],
        record["dense_weight"],
    )
    assert got == expected
    assert protocol["required_accuracy"] == expected[3]
    assert protocol["label_weight"] == expected[4]
    assert protocol["differences_from_trial"] == differences
    assert protocol["trial"] is False
    assert record["eval_batches"] == 1


def test_a_run_trains_and_records_under_its_own_depth_cost(tmp_path: Path, source: Path) -> None:
    # the same seed and floor consume the same episodes and uniforms, so the first update's
    # trajectories are identical across depth costs and only the weight differs
    t_max = _cfg().model.t_max
    firsts: list[dict[str, float]] = []
    for depth_cost in (1.0, 2.0):
        out = tmp_path / f"dc{depth_cost}"
        _ = run(
            source,
            _sha(source),
            0,
            "none",
            out,
            updates=3,
            eval_batches=1,
            save_at=(0, 3),
            required_accuracy=NO_FLOOR,
            depth_cost=depth_cost,
        )
        lines = [
            cast("dict[str, float]", json.loads(line))
            for line in (out / "train.jsonl").read_text("utf-8").splitlines()
        ]
        assert len(lines) == 3
        for line in lines:  # the cost every update trained on carries the run's weight
            assert line["cost"] == pytest.approx(
                line["nll"] + depth_cost * line["depth"] / t_max, rel=1e-12
            )
        firsts.append(lines[0])
    assert firsts[0]["depth"] == firsts[1]["depth"]
    assert firsts[0]["pair_depth_cost_diff_std"] > 0.0
    assert firsts[1]["pair_depth_cost_diff_std"] == pytest.approx(
        2.0 * firsts[0]["pair_depth_cost_diff_std"], rel=1e-12
    )


# --- class-balanced label weights (maintainer B 9f1eceb)
# -------------------------------------------------


def _rows(values: list[list[int]]) -> Tensor:
    """An int64 2-D tensor (strict typing has no typed constructor)."""
    return torch.stack([_vec([float(v) for v in row]).long() for row in values])


def test_episode_label_weights_are_each_episodes_mean() -> None:
    weights = _label_weights(_rows([[-1, 0, 0, 7], [3, -1, -1, -1]]), "episode")
    third = 1 / 3
    assert torch.equal(
        weights, torch.stack([_vec([0.0, third, third, third]), _vec([1.0, 0.0, 0.0, 0.0])])
    )


def test_class_label_weights_pool_each_class_over_the_unique_episodes() -> None:
    # class 0: 3 labels over both episodes; class 7: 1 label; class 3: absent from the batch;
    # E = 2 episodes, 2 classes present: a label of class c weighs E / (2 * N_c)
    weights = _label_weights(_rows([[-1, 0, 0, 7], [0, -1, -1, -1]]), "class")
    w0, w7 = 2 / (2 * 3), 2 / (2 * 1)
    assert torch.equal(weights, torch.stack([_vec([0.0, w0, w0, w7]), _vec([w0, 0.0, 0.0, 0.0])]))
    assert float(weights.sum()) == pytest.approx(2.0)  # mean weight per episode 1, like "episode"
    # pooled, not per-episode class means: episode 1's lone class-0 label weighs the same as
    # each of episode 0's two
    assert weights[1, 0] == weights[0, 1]


def test_label_weights_refuse_an_unknown_scheme_and_an_episode_without_labels() -> None:
    with pytest.raises(ValueError, match="scheme"):
        _ = _label_weights(_rows([[0]]), "token")
    with pytest.raises(ValueError, match="no label"):
        _ = _label_weights(_rows([[0], [-1]]), "episode")


def test_a_class_weighted_run_trains_on_the_balanced_term_and_logs_both(
    tmp_path: Path, source: Path
) -> None:
    t_max = _cfg().model.t_max
    out = tmp_path / "class"
    result = run(
        source,
        _sha(source),
        0,
        "none",
        out,
        updates=3,
        eval_batches=1,
        save_at=(0, 3),
        required_accuracy=NO_FLOOR,
        depth_cost=1.0,
        label_weight="class",
    )
    assert result["label_weight"] == "class"
    protocol = cast("dict[str, object]", result["protocol"])
    assert protocol["differences_from_trial"] == [
        "updates",
        "eval_batches",
        "save_at",
        "required_accuracy",
        "depth_cost",
        "label_weight",
    ]
    lines = [
        cast("dict[str, float]", json.loads(line))
        for line in (out / "train.jsonl").read_text("utf-8").splitlines()
    ]
    for line in lines:  # trained on the balanced term; the ordinary NLL is still logged
        assert line["cost"] == pytest.approx(
            line["objective_nll"] + 1.0 * line["depth"] / t_max, rel=1e-12
        )
    assert any(abs(line["objective_nll"] - line["nll"]) > 1e-6 for line in lines)


# --- joint adaptation (maintainer B f27b8a3)
# -------------------------------------------------------------

_HIGH = 1.0 - 1e-9  # a uniform above every stop probability here: the token continues


def _forced(exits: tuple[int, ...], positions: tuple[int, ...], t_max: int, n: int) -> Tensor:
    """``float64 [T - 1, n]`` uniforms that make token ``positions[i]`` exit at pass ``exits[i]``
    (1-based) under any stop probability strictly inside (0, 1): 0 stops, ``_HIGH`` continues;
    every other token continues to the last pass (``SampledRule`` stops a non-deciding token too
    when its uniform is below the stop probability, it only leaves out its log-probability)."""
    u = torch.full((t_max - 1, n), _HIGH, dtype=torch.float64)
    for position, exit_ in zip(positions, exits, strict=True):
        for t in range(t_max - 1):
            u[t, position] = 0.0 if t == exit_ - 1 else _HIGH
    return u


class _TinyJoint(NamedTuple):
    """A three-pass model with a trainable backbone and one episode in which two tokens decide."""

    model: Vesper
    episode: Episodes
    positions: tuple[int, ...]
    deciding: Tensor
    weights: Tensor


def _tiny_joint() -> _TinyJoint:
    cfg = _cfg()
    tiny = replace(
        cfg,
        model=replace(cfg.model, t_max=3),
        training=replace(cfg.training, t_max_anneal_from=3),
    )
    model = Vesper(tiny, torch.Generator().manual_seed(3))
    _ = model.eval()
    _moderate_head(model)
    head = model.loop.halting_head()
    with torch.no_grad():  # a nonzero direction, scaled so that every pattern has real weight
        _ = head.w_h.mul_(0.05)
        _ = head.b_h.fill_(-0.5)
    ep = episodes(generator_for("joint-exact", 0), 1, "j")
    labels = ep.chunk.loss_mask[0].nonzero().flatten()
    positions = (int(labels[0]) - 1, int(labels[-1]))  # one before the first label, the last
    deciding = torch.zeros_like(ep.chunk.valid)
    for position in positions:
        deciding[0, position] = True
    return _TinyJoint(model, ep, positions, deciding, _label_weights(ep.task, "episode"))


def _pattern(j: _TinyJoint, exits: tuple[int, ...], depth_cost: float) -> tuple[Tensor, Tensor]:
    """``(log p, cost)`` of one stop pattern, both with their graphs; log p rebuilt HERE.

    The log-probability comes from the head and the pass features it reads, not from the rule's
    own record, which would share any defect of the rule (such as detached features): a
    continue at every pass before the exit, then the stop.
    """
    model, c = j.model, j.episode.chunk
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    head = model.loop.halting_head()
    uniforms = _forced(exits, j.positions, t_max, n)[:, None]
    rule = SampledRule(head, 0.0, uniforms, j.deciding)
    states = model.init_states(1, torch.float32, torch.device("cpu"))
    result = model.run(c.tokens, c.pos, states, exit_rule=rule)
    labels = c.loss_mask[0]
    ce = functional.cross_entropy(
        model.head(result.y)[0][labels].double(), c.targets[0][labels].long(), reduction="none"
    )
    label_term = (ce * j.weights[0][labels]).sum()
    # the update's depth is over its deciding tokens (in a run: every real token decides)
    depth = result.n_iters[0][j.deciding[0]].double().mean()
    baseline = hazard_baseline(t_max)
    log_p = _vec([0.0]).sum()
    for position, exit_ in zip(j.positions, exits, strict=True):
        for t in range(min(exit_, t_max - 1)):
            z = head.logit(result.per_pass[t], t)[0, position].double() + baseline[t]
            log_p = log_p + functional.logsigmoid(z if t == exit_ - 1 else -z)
    assert rule.log_prob is not None
    assert float(rule.log_prob.detach().sum()) == pytest.approx(float(log_p.detach()))
    return log_p, label_term + depth_cost * depth / t_max


def _twice(t: Tensor) -> Tensor:
    """The episode once per rollout, rollout-major."""
    return torch.cat([t, t])


def test_the_update_loss_is_unbiased_for_the_exact_joint_gradient() -> None:
    # J = sum over every stop pattern tau of p_tau * (L_tau + dc * D_tau / T), with the backbone
    # trainable: its gradient needs the pathwise label term AND the score term through the
    # features the head reads (maintainer B's one-parameter case: pathwise only is wrong by 6x). The
    # update loss's expectation over independent pairs of patterns must equal grad J exactly,
    # for the head and for every backbone parameter.
    j, depth_cost = _tiny_joint(), 1.0
    model, c = j.model, j.episode.chunk
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    params = list(model.parameters())
    patterns = list(itertools.product(range(1, t_max + 1), repeat=len(j.positions)))
    terms = [_pattern(j, exits, depth_cost) for exits in patterns]
    probability = torch.stack([log_p.exp() for log_p, _ in terms])
    assert float(probability.detach().sum()) == pytest.approx(1.0, abs=1e-9)  # all enumerated
    assert float(probability.detach().min()) > 0.05  # none negligible: the check sees them all
    exact_j = (probability * torch.stack([cost for _, cost in terms])).sum()
    # without allow_unused: a parameter the exact objective does not reach is an error
    exact = torch.autograd.grad(exact_j, params)
    p = probability.detach()
    expected = [torch.zeros_like(q, dtype=torch.float64) for q in params]
    for (a, first), (b, second) in itertools.product(enumerate(patterns), repeat=2):
        uniforms = torch.stack(
            [_forced(first, j.positions, t_max, n), _forced(second, j.positions, t_max, n)], dim=1
        )
        loss = update_loss(
            model,
            _twice(c.tokens),
            _twice(c.pos),
            _twice(c.targets),
            _twice(c.loss_mask),
            _twice(j.deciding),
            _twice(j.weights),
            uniforms,
            0.0,
            depth_cost,
        )
        grads = torch.autograd.grad(loss.total, params)
        for k, g in enumerate(grads):
            expected[k] += float(p[a] * p[b]) * g.double()
    # per parameter, relative to the exact gradient's norm (float32 arithmetic agrees to ~5e-7;
    # leaving out either part of the loss is off by order one)
    for (name, _), want, got in zip(model.named_parameters(), exact, expected, strict=True):
        size = float((want.double() * want.double()).sum().sqrt())
        error = float(((got - want.double()) * (got - want.double())).sum().sqrt())
        assert size > 0.0, name  # every parameter, backbone and head, has a gradient
        assert error / size < 1e-5, name


def test_the_dense_auxiliary_is_rho_times_the_mean_over_passes_and_reaches_only_the_backbone() -> (
    None
):
    j = _tiny_joint()
    model, c = j.model, j.episode.chunk
    t_max, n = model.loop.t_max, int(c.tokens.shape[1])
    rho = 0.1
    dense = Dense(c.tokens, c.pos, c.targets, c.loss_mask, j.weights, rho)
    uniforms = torch.stack(
        [_forced((1, 2), j.positions, t_max, n), _forced((3, 1), j.positions, t_max, n)], dim=1
    )
    loss = update_loss(
        model,
        _twice(c.tokens),
        _twice(c.pos),
        _twice(c.targets),
        _twice(c.loss_mask),
        _twice(j.deciding),
        _twice(j.weights),
        uniforms,
        0.0,
        1.0,
        dense,
    )
    # by hand: every pass of a fresh all-pass forward, its weighted label NLL, the mean over passes
    states = model.init_states(1, torch.float32, torch.device("cpu"))
    passes = model.run(c.tokens, c.pos, states, exit_rule=DenseRule()).per_pass
    labels = c.loss_mask[0]
    per_pass = [
        (
            functional.cross_entropy(
                model.head(passes[t])[0][labels].double(),
                c.targets[0][labels].long(),
                reduction="none",
            )
            * j.weights[0][labels]
        ).sum()
        for t in range(t_max)
    ]
    want = rho * torch.stack(per_pass).mean()
    assert float(loss.dense.detach()) == pytest.approx(float(want.detach()), rel=1e-6)
    parts = float((loss.score + loss.pathwise + loss.dense).detach())
    assert float(loss.total.detach()) == pytest.approx(parts, abs=1e-12)
    head = list(model.loop.halting_head().parameters())
    on_head = torch.autograd.grad(loss.dense, head, allow_unused=True, retain_graph=True)
    assert all(g is None for g in on_head)  # no decision credit: the head never sees it
    on_embedding = torch.autograd.grad(loss.dense, [model.embedding.weight])[0]
    assert float((on_embedding * on_embedding).sum()) > 0.0
    without = update_loss(
        model,
        _twice(c.tokens),
        _twice(c.pos),
        _twice(c.targets),
        _twice(c.loss_mask),
        _twice(j.deciding),
        _twice(j.weights),
        uniforms,
        0.0,
        1.0,
    )
    assert float(without.dense) == 0.0


def test_a_joint_run_trains_the_backbone_and_measures_it_beside_the_source(
    tmp_path: Path, source: Path
) -> None:
    out = tmp_path / "joint"
    result = run(
        source,
        _sha(source),
        0,
        "none",
        out,
        updates=3,
        eval_batches=1,
        save_at=(0, 3),
        required_accuracy=NO_FLOOR,
        train="joint",
        dense_weight=0.1,
    )
    protocol = cast("dict[str, object]", result["protocol"])
    assert protocol["differences_from_trial"] == [
        "updates",
        "eval_batches",
        "save_at",
        "required_accuracy",
        "train",
        "dense_weight",
    ]
    assert (result["train"], result["dense_weight"]) == ("joint", 0.1)
    optimizer = cast("dict[str, object]", result["optimizer"])
    assert optimizer["backbone_peak_lr"] == BACKBONE_LR
    before = cast(
        "dict[str, Tensor]", torch.load(out / "weights" / "after-00000.pt", weights_only=True)
    )
    after = cast(
        "dict[str, Tensor]", torch.load(out / "weights" / "after-00003.pt", weights_only=True)
    )
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert any(not k.startswith("loop.halting") for k in moved)  # the backbone trained
    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (out / "train.jsonl").read_text("utf-8").splitlines()
    ]
    for line in lines:
        assert len(cast("list[float]", line["lr_groups"])) == 2
        norms = cast("list[float]", line["grad_norm_groups"])
        assert len(norms) == 2
        assert all(v > 0.0 for v in norms)
        # the groups partition the clipped set: their norms make up the total, before clipping
        assert math.hypot(*norms) == pytest.approx(cast("float", line["grad_norm"]), rel=1e-5)
        assert cast("float", line["dense"]) > 0.0
    evaluations = cast("dict[str, dict[str, object]]", result["evaluations"])
    for after_ in ("0", "3"):  # the current backbone measured at every save, the source kept
        assert set(cast("dict[str, object]", evaluations[after_]["current_fixed"])) == {
            "full",
            "static",
        }
    # before any update the current backbone IS the source: the two profiles agree exactly
    assert json.dumps(evaluations["0"]["current_fixed"], sort_keys=True) == json.dumps(
        result["fixed"], sort_keys=True
    )


def test_a_head_run_refuses_a_dense_weight_and_an_unknown_train_mode(
    tmp_path: Path, source: Path
) -> None:
    out = tmp_path / "refused"
    for train, dense_weight, match in (
        ("head", 0.1, "only in a joint run"),
        ("all", 0.0, "unknown train mode"),
        ("joint", math.nan, "dense weight"),
        ("joint", -0.1, "dense weight"),
    ):
        with pytest.raises(ValueError, match=match):
            _ = run(
                source,
                _sha(source),
                0,
                "none",
                out,
                updates=3,
                eval_batches=1,
                save_at=(0, 3),
                required_accuracy=NO_FLOOR,
                train=train,
                dense_weight=dense_weight,
            )
        assert not out.exists()
