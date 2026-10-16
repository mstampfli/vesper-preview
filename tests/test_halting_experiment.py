"""The stage-1 driver (vesper/halting_experiment.py): setup, training step and the measurements.

The evaluation's measurements are checked against relations that must hold whatever the model
learned: an untrained model is enough, and a wrong trace, a mislabelled count or a cost that is
not the policy's own breaks one of them.
"""

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from vesper.column import Column
from vesper.config import VesperConfig, load
from vesper.halting_experiment import (
    CONDITIONS,
    Q_GRID,
    Totals,
    build,
    class_counts,
    evaluate_batch,
    held_out,
    recurrence_ablated,
    run,
    summarize,
    train_step,
)
from vesper.loop import TraceRule
from vesper.synthetic import (
    BACKS,
    CLASSES,
    HOP_QUERIES,
    READ_POSITION,
    SEQUENCE_TOKENS,
    TERMINAL_MIX,
    episodes,
    generator_for,
    hop_mix,
)
from vesper.train import optimizer

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"
T = 8
LABELS = HOP_QUERIES + BACKS + 1  # scored positions per sequence


def _cfg() -> VesperConfig:
    return load(CONFIG)


def test_the_setup_refuses_a_config_the_task_cannot_use() -> None:
    cfg = _cfg()
    _ = build(cfg, "fixed", 0)
    with pytest.raises(ValueError, match="condition"):
        _ = build(cfg, "ponder", 0)
    wide = replace(cfg, model=replace(cfg.model, window=9))
    with pytest.raises(ValueError, match="window"):
        _ = build(wide, "fixed", 0)
    other = replace(cfg, model=replace(cfg.model, vocab=65))
    with pytest.raises(ValueError, match="vocab"):
        _ = build(other, "fixed", 0)


def test_each_condition_starts_from_its_own_head() -> None:
    cfg = _cfg()
    act = build(cfg, "act", 0)
    fixed = build(cfg, "fixed", 0)
    local = build(cfg, "local", 0)
    # one initialization for every condition: only the halting head differs
    act_state = cast("dict[str, torch.Tensor]", act.state_dict())
    local_state = cast("dict[str, torch.Tensor]", local.state_dict())
    for name, value in act_state.items():
        if not name.startswith("loop.halting"):  # the head, registered under two names
            assert torch.equal(value, local_state[name]), name
    assert not any(p.requires_grad for p in fixed.loop.halting_head().parameters())
    head = local.loop.halting_head()
    assert float(head.w_h.detach().abs().sum()) == 0.0
    assert float(head.b_h.detach().abs().sum()) == 0.0
    assert bool((head.norm.weight.detach() == 1.0).all())
    # ACT keeps the trainer's path: its head is left to set_halting
    assert float(act.loop.halting_head().w_h.detach().abs().sum()) > 0.0


@pytest.mark.parametrize("condition", CONDITIONS)
def test_a_training_step_moves_the_model_and_only_the_trained_head(condition: str) -> None:
    cfg = _cfg()
    model = build(cfg, condition, 0)
    opt = optimizer(model, cfg)
    head_before = [p.detach().clone() for p in model.loop.halting_head().parameters()]
    u_before = model.loop.u.detach().clone()
    data = generator_for("train", 0)
    for step in range(2):
        record = train_step(model, opt, cfg, condition, 0.1, episodes(data, 4, "t"), step)
        assert isinstance(record["loss"], float)
        assert torch.isfinite(torch.tensor(record["loss"]))
    head = model.loop.halting_head()
    head_after = list(head.parameters())
    head_moved = any(not torch.equal(a, b) for a, b in zip(head_before, head_after, strict=True))
    if condition == "act":
        # the trainer's warm phase holds ACT's head frozen at p = 1/T (set_halting)
        assert float(head.w_h.abs().sum()) == 0.0
        assert float(head.b_h) == pytest.approx(math.log(1 / 7))
        assert not any(p.requires_grad for p in head.parameters())
    else:
        # fixed never trains a head; the learned conditions train theirs from the first step
        assert head_moved == (condition in {"local", "aggregate"})
    assert not torch.equal(u_before, model.loop.u.detach())


def test_class_counts_score_the_argmax_at_each_loss_position() -> None:
    model = build(_cfg(), "fixed", 0)
    ep = episodes(generator_for("test", 0), 3, "t")
    y = torch.randn(3, SEQUENCE_TOKENS, 32, generator=torch.Generator().manual_seed(0))
    counts = class_counts(model, y, ep)
    assert counts.shape == (len(CLASSES), 2)
    expected = torch.zeros(len(CLASSES), 2, dtype=torch.int64)
    with torch.no_grad():
        logits = model.head(y)
    for s in range(3):
        for p in range(SEQUENCE_TOKENS):
            task = int(ep.task[s, p])
            if task >= 0:
                expected[task, 1] += 1
                if int(logits[s, p].argmax()) == int(ep.chunk.targets[s, p]):
                    expected[task, 0] += 1
    assert torch.equal(counts, expected)


@pytest.fixture(scope="module")
def fixed_eval() -> dict[str, torch.Tensor]:
    model = build(_cfg(), "fixed", 0).eval()
    totals = Totals()
    with torch.no_grad():
        evaluate_batch(model, "fixed", episodes(generator_for("test", 1), 3, "t"), totals)
    return totals.values


def test_the_ablation_removes_the_recurrent_read_from_a_copy_only() -> None:
    model = build(_cfg(), "fixed", 0).eval()
    ep = episodes(generator_for("test", 3), 2, "t")
    with torch.no_grad():
        # given the model itself as its "ablation", the ablated replay IS its paired reference,
        # output for output, and for B the reference is the deployed run
        same = Totals()
        evaluate_batch(model, "fixed", ep, same, ablated=model)
        assert torch.equal(same.values["ablated/trace"], same.values["ablated/reference"])
        assert float(same.values["ablated_max_output_difference"]) == 0.0
        assert torch.equal(same.values["ablated/reference"], same.values["deployed/prefill"])
        # the default ablation is a copy with every recurrent output projection zeroed: the
        # original keeps its weights, and the copy's outputs differ
        before = [p.detach().clone() for p in model.parameters()]
        evaluate_batch(model, "fixed", ep, Totals())
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
        without = recurrence_ablated(model)
        columns = [c for c in without.loop.columns if isinstance(c, Column)]
        assert len(columns) == len(without.loop.columns)
        assert all(float(c.rec.w_o.detach().abs().sum()) == 0.0 for c in columns)
        c = ep.chunk
        states = model.init_states(2, torch.float32, torch.device("cpu"))
        y = model.run(c.tokens, c.pos, states).y
        y_without = without.run(c.tokens, c.pos, states).y
        assert not torch.allclose(y, y_without)


def test_an_identity_ablation_of_act_changes_no_output() -> None:
    # Maintainer B's re-review of 9044238: A's deployed readout is its ACT mixture, so comparing the
    # one-exit replay of an unchanged model against it changed predictions with no weight
    # changed. The pair is the replay with and without the recurrent read; with the model as
    # its own "ablation" every output must be identical, not only the scores, which an
    # untrained model at chance cannot tell apart.
    model = build(_cfg(), "act", 0).eval()
    head = model.loop.halting_head()
    with torch.no_grad():
        _ = head.w_h.zero_()
        _ = head.b_h.fill_(math.log(0.3 / 0.7))  # p = 0.3 at every pass: a nontrivial mixture
    ep = episodes(generator_for("test", 4), 2, "t")
    c = ep.chunk
    with torch.no_grad():
        deployed = model.run(
            c.tokens, c.pos, model.init_states(2, torch.float32, torch.device("cpu"))
        )
        replayed = model.run(
            c.tokens,
            c.pos,
            model.init_states(2, torch.float32, torch.device("cpu")),
            exit_rule=TraceRule(deployed.n_iters.long()),
        )
        # the fixture's point: the mixture readout is not the one-exit readout at the same exits
        assert float((deployed.y - replayed.y).abs().max()) > 1e-3
        totals = Totals()
        evaluate_batch(model, "act", ep, totals, ablated=model)
    assert float(totals.values["ablated_max_output_difference"]) == 0.0
    assert torch.equal(totals.values["ablated/trace"], totals.values["ablated/reference"])


def test_dense_and_trace_readouts_are_measured_over_every_mode(
    fixed_eval: dict[str, torch.Tensor],
) -> None:
    for mode in ("prefill", "decode", "carried"):
        # B's exits are its quantile whatever the outputs, so every mode takes prefill's
        agreement = fixed_eval[f"dense_mode_exits/{mode}"]
        assert float(agreement[0]) == float(agreement[1]), mode
        for kind in ("dense_mode", "trace_mode"):
            assert int(fixed_eval[f"{kind}/{mode}"][:, 1].sum()) == 3 * LABELS, (kind, mode)
    assert torch.equal(fixed_eval["dense_mode/prefill"], fixed_eval["dense_exit/0.5"])
    assert torch.equal(fixed_eval["trace_mode/prefill"], fixed_eval["trace/0.5"])


def test_the_fixed_policy_exits_every_token_at_its_quantile(
    fixed_eval: dict[str, torch.Tensor],
) -> None:
    for q in Q_GRID:
        depth = min(T, -(-round(q * T * 1000) // 1000))  # ceil(q T), exact for q = k / 10
        passes = fixed_eval[f"live_passes/{q}"]
        assert float(passes[0] / passes[1]) == depth, q
        assert float(passes[2] / passes[3]) == depth, q
        agree = fixed_eval[f"live_agrees_with_dense_exits/{q}"]
        assert float(agree[0]) == float(agree[1]), q
        # the same trace under the same states: replay and live gates score identically
        assert torch.equal(fixed_eval[f"trace/{q}"], fixed_eval[f"live/{q}"]), q


def test_measurements_that_must_agree_whatever_the_model(
    fixed_eval: dict[str, torch.Tensor],
) -> None:
    v = fixed_eval
    # every token at depth d: its first d passes are fresh, so pass d equals the dense pass d
    assert torch.equal(v["static"], v["dense_pass"])
    # the permuted trace costs exactly what the deployed one does
    assert torch.equal(v["random_passes"], v["deployed_passes/prefill"])
    # per sequence: HOP_QUERIES hops with every depth at least once, each copy class once, one
    # read; sequence s of a batch reads at depth 1 + s mod 4, so three sequences read 1, 2, 3
    totals = [int(x) for x in v["dense_pass"][0, :, 1].unbind()]
    hops, backs, reads = totals[:4], totals[4:7], totals[7:]
    assert sum(hops) == 3 * HOP_QUERIES
    assert min(hops) >= 3
    assert backs == [3] * 3
    assert reads == [1, 1, 1, 0]
    # decode and carried chunks take the same exits as prefill
    for mode in ("decode", "carried"):
        agreement = v[f"deployed_agreement/{mode}"]
        assert float(agreement[0]) == float(agreement[1]), mode
    # the register grid's "own depth" column equals forcing the register to that depth (4)
    assert torch.equal(v["register"][:, 0], v["register"][:, 4])
    assert v["register"].shape == (T, T + 1, len(CLASSES), 2)
    # depth histograms count every position once, every loss position once
    assert int(v["depth/role"].sum()) == 3 * SEQUENCE_TOKENS
    assert int(v["depth/class"].sum()) == 3 * LABELS


def test_the_summary_reads_the_counts() -> None:
    model = build(_cfg(), "local", 0).eval()
    totals = Totals()
    with torch.no_grad():
        evaluate_batch(model, "local", episodes(generator_for("test", 2), 2, "t"), totals)
    s = summarize(totals.values, T, 2)
    assert isinstance(s["live/0.5"], dict)
    live = cast("dict[str, float]", s["live/0.5"])
    counts = totals.values["live/0.5"]
    assert live["overall"] == float(counts[:, 0].sum()) / float(counts[:, 1].sum())
    passes = totals.values["live_passes/0.5"]
    assert live["passes"] == pytest.approx(float(passes[0] / passes[1]))
    assert live["replayed_writes"] == pytest.approx(2 * (T - live["passes"]))
    assert isinstance(s["depth/role"], dict)
    depth = cast("dict[str, float]", s["depth/role"])
    assert depth["information"] == pytest.approx(depth["entropy"] - depth["conditional_entropy"])
    assert isinstance(s["register"], list)
    register = cast("list[list[float]]", s["register"])
    assert len(register) == T
    assert all(len(row) == T + 1 for row in register)


def test_a_run_writes_its_record_once(tmp_path: Path) -> None:
    out = tmp_path / "run"
    result = run(CONFIG, "fixed", 0, out, steps=2, eval_batches=1, eval_size=2, log_every=1)
    assert {p.name for p in out.iterdir()} == {
        "config.yaml",
        "train.jsonl",
        "model.pt",
        "result.json",
    }
    # the recipe is the snapshot taken before the first step, hashed from the same bytes
    snapshot = (out / "config.yaml").read_bytes()
    assert snapshot == CONFIG.read_bytes()
    assert result["config_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert result["config_snapshot"] == "config.yaml"  # relative to result.json
    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["step"] for line in lines] == [0, 1]
    # each batch is drawn with its step's mix: step 0's labels are an independent draw's
    cfg = load(CONFIG)
    data = generator_for("train", 0)
    for step, line in enumerate(lines):
        ep = episodes(data, cfg.training.batch_chunks, "x", hop_mix(step))
        expected = torch.bincount(ep.task[ep.task >= 0], minlength=len(CLASSES))
        assert line["labels"] == [int(v) for v in expected.unbind()], step
    record = cast("dict[str, object]", json.loads((out / "result.json").read_text("utf-8")))
    assert record["condition"] == "fixed"
    assert record["steps"] == 2
    assert record["planned_steps"] == 12000
    assert record["eval_sequences"] == 2
    assert (record["eval_batches"], record["eval_size"]) == (1, 2)
    assert record["eval_split"] == "validation"
    assert result["summary"] == record["summary"]
    with pytest.raises(FileExistsError):
        _ = run(CONFIG, "fixed", 0, out, steps=1, eval_batches=1, eval_size=2)
    # an unknown split is refused before a run directory exists
    with pytest.raises(ValueError, match="split must be one of"):
        _ = run(CONFIG, "fixed", 0, tmp_path / "other", steps=1, eval_split="test")
    assert not (tmp_path / "other").exists()


def test_weight_snapshots_are_the_weights_after_their_update(tmp_path: Path) -> None:
    # Maintainer B's review of 1678173: the promised snapshots every 1000 updates did not exist
    longer = tmp_path / "three"
    _ = run(CONFIG, "fixed", 0, longer, steps=3, eval_batches=1, eval_size=2, weights_every=2)
    assert sorted(p.name for p in (longer / "weights").iterdir()) == ["after-00002.pt"]
    shorter = tmp_path / "two"
    _ = run(CONFIG, "fixed", 0, shorter, steps=2, eval_batches=1, eval_size=2, weights_every=2)
    snapshot = cast(
        "dict[str, torch.Tensor]",
        torch.load(longer / "weights" / "after-00002.pt", weights_only=True),
    )
    after_two = cast("dict[str, torch.Tensor]", torch.load(shorter / "model.pt", weights_only=True))
    assert snapshot.keys() == after_two.keys()
    assert all(torch.equal(snapshot[k], after_two[k]) for k in snapshot)
    final = cast("dict[str, torch.Tensor]", torch.load(longer / "model.pt", weights_only=True))
    assert not all(torch.equal(snapshot[k], final[k]) for k in snapshot)  # a third update moved


def test_the_held_out_set_is_shared_and_disjoint_from_training() -> None:
    a = held_out(2, 3)
    b = held_out(2, 3)
    assert all(torch.equal(x.chunk.tokens, y.chunk.tokens) for x, y in zip(a, b, strict=True))
    train = episodes(generator_for("train", 0), 3, "t")
    assert not torch.equal(a[0].chunk.tokens, train.chunk.tokens)
    assert int(a[0].chunk.tokens[0, READ_POSITION]) == 2
    # the validation set is the terminal mix from its own generator; final is another set
    direct = episodes(generator_for("validation", 0), 3, "v", TERMINAL_MIX)
    assert torch.equal(a[0].chunk.tokens, direct.chunk.tokens)
    final = held_out(2, 3, "final")
    assert not torch.equal(a[0].chunk.tokens, final[0].chunk.tokens)
    with pytest.raises(ValueError, match="split must be one of"):
        _ = held_out(1, 3, "eval")
