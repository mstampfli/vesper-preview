"""The stage-1 driver (vesper/halting_experiment.py): setup, training step and the measurements.

The evaluation's measurements are checked against relations that must hold whatever the model
learned: an untrained model is enough, and a wrong trace, a mislabelled count or a cost that is
not the policy's own breaks one of them.
"""

import hashlib
import json
import math
from collections.abc import Mapping
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
from vesper.model import Vesper
from vesper.synthetic import (
    BACKS,
    CLASSES,
    HOP_QUERIES,
    MIX_SCHEDULES,
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
    # the default mix recipe, named and spelled out, so a report binds gates to what ran
    assert record["mix_schedule"] == "v1"
    assert record["mix_stages"] == [[s, list(m)] for s, m in MIX_SCHEDULES["v1"]]
    assert result["summary"] == record["summary"]
    with pytest.raises(FileExistsError):
        _ = run(CONFIG, "fixed", 0, out, steps=1, eval_batches=1, eval_size=2)
    # an unknown split or mix is refused before a run directory exists
    with pytest.raises(ValueError, match="split must be one of"):
        _ = run(CONFIG, "fixed", 0, tmp_path / "other", steps=1, eval_split="test")
    with pytest.raises(ValueError, match="unknown mix schedule"):
        _ = run(CONFIG, "fixed", 0, tmp_path / "other", steps=1, mix="v3")
    assert not (tmp_path / "other").exists()


def test_a_run_trains_and_records_the_named_mix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the stages differ only from update 1000 on, so the driver's calls are what is checked:
    # every step draws its batch under the run's named recipe, and the record says which
    calls: list[tuple[int, str]] = []

    def spy(step: int, schedule: str = "v1") -> tuple[float, ...]:
        calls.append((step, schedule))
        return hop_mix(step, schedule)

    monkeypatch.setattr("vesper.halting_experiment.hop_mix", spy)
    out = tmp_path / "slow"
    result = run(CONFIG, "fixed", 0, out, steps=3, eval_batches=1, eval_size=2, mix="v2-slow")
    assert calls == [(0, "v2-slow"), (1, "v2-slow"), (2, "v2-slow")]
    assert result["mix_schedule"] == "v2-slow"
    assert result["mix_stages"] == [
        [0, [0.7, 0.1, 0.1, 0.1]],
        [8000, [0.4, 0.3, 0.15, 0.15]],
        [16000, [0.25, 0.25, 0.25, 0.25]],
    ]


def test_the_data_order_has_its_own_seed_and_both_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B 5dba6af: a seed chose both the initialization and the data order; a crossed run
    # keeps --seed's initialization and draws --data-seed's batches, and records both
    built: list[int] = []
    drawn: list[tuple[str, int]] = []
    real_build, real_generator = build, generator_for

    def spy_build(
        cfg: VesperConfig,
        condition: str,
        seed: int,
        weights: Mapping[str, torch.Tensor] | None = None,
    ) -> Vesper:
        assert weights is None  # a fresh run: the seed draws the initialization
        built.append(seed)
        return real_build(cfg, condition, seed)

    def spy_generator(split: str, seed: int) -> torch.Generator:
        drawn.append((split, seed))
        return real_generator(split, seed)

    monkeypatch.setattr("vesper.halting_experiment.build", spy_build)
    monkeypatch.setattr("vesper.halting_experiment.generator_for", spy_generator)

    def small(seed: int, name: str, data_seed: int | None = None) -> dict[str, object]:
        out = tmp_path / name
        return run(
            CONFIG, "fixed", seed, out, steps=1, eval_batches=1, eval_size=2, data_seed=data_seed
        )

    crossed = small(1, "c", data_seed=4)
    assert built == [1]
    assert ("train", 4) in drawn
    assert ("train", 1) not in drawn
    assert (crossed["seed"], crossed["data_seed"]) == (1, 4)
    built.clear()
    drawn.clear()
    plain = small(4, "p")
    assert built == [4]
    assert ("train", 4) in drawn
    assert (plain["seed"], plain["data_seed"]) == (4, 4)  # by default the data order is the seed's

    def labels(out: Path) -> object:
        first = (out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
        return cast("dict[str, object]", json.loads(first))["labels"]

    assert labels(tmp_path / "c") == labels(tmp_path / "p")  # the same batches
    other = small(1, "o")
    assert other["data_seed"] == 1
    assert labels(tmp_path / "o") != labels(tmp_path / "c")


def _independent_chain(previous: str, tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256(previous.encode())
    for t in tensors:
        h.update(f"{t.dtype}{tuple(t.shape)}".encode())
        h.update(t.contiguous().numpy().tobytes())  # type: ignore[misc]  # ndarray: generic over Any
    return h.hexdigest()


def test_a_run_digests_its_initial_state_and_every_consumed_batch(tmp_path: Path) -> None:
    # Maintainer B 7ad7211: hash what is consumed, not class counts; identical starts provable by
    # digest
    def go(seed: int, data_seed: int, name: str) -> dict[str, object]:
        out = tmp_path / name
        return run(
            CONFIG,
            "fixed",
            seed,
            out,
            steps=2,
            eval_batches=1,
            eval_size=2,
            data_seed=data_seed,
            log_every=1,
        )

    a, again, other_init, other_data = go(1, 4, "a"), go(1, 4, "b"), go(4, 4, "c"), go(1, 1, "d")
    keys = ("initial_state_digest", "consumed_digest")
    assert [a[k] for k in keys] == [again[k] for k in keys]
    assert other_init["initial_state_digest"] != a["initial_state_digest"]
    assert other_init["consumed_digest"] == a["consumed_digest"]  # the same data order
    assert other_data["initial_state_digest"] == a["initial_state_digest"]
    assert other_data["consumed_digest"] != a["consumed_digest"]
    # the chain, recomputed from an independent draw, step by step as logged
    cfg = load(CONFIG)
    g = generator_for("train", 4)
    chain = ""
    lines = (tmp_path / "a" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    for step, line in enumerate(lines):
        c = episodes(g, cfg.training.batch_chunks, f"train-4-{step}", hop_mix(step)).chunk
        chain = _independent_chain(
            chain, [c.tokens, c.targets, c.valid, c.doc_start, c.segment, c.loss_mask, c.pos]
        )
        assert cast("dict[str, object]", json.loads(line))["consumed_digest"] == chain, step
    assert a["consumed_digest"] == chain
    state = cast("dict[str, torch.Tensor]", build(cfg, "fixed", 1).state_dict())
    assert a["initial_state_digest"] == _independent_chain("", list(state.values()))


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


def _variant(directory: Path, old: str, new: str) -> Path:
    """The stage-1 config with one line's ``old`` replaced by ``new``, as a file of its own."""
    text = CONFIG.read_text(encoding="utf-8")
    assert text.count(old) == 1, old
    path = directory / f"variant-{len(list(directory.glob('variant-*')))}.yaml"
    _ = path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def _two_updates(config: Path, out: Path) -> Path:
    """A finished run of B, two updates, a snapshot after the second: a warm start's source."""
    _ = run(config, "fixed", 3, out, steps=2, eval_batches=1, eval_size=2, weights_every=2)
    return out


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _two_updates(CONFIG, tmp_path_factory.mktemp("source") / "run")


def _warm(out: Path, weights: Path, condition: str = "local") -> dict[str, object]:
    return run(
        CONFIG, condition, 0, out, steps=1, eval_batches=1, eval_size=2, init_weights=weights
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_a_run_starts_from_another_runs_weights_with_its_own_head(
    tmp_path: Path, source: Path, condition: str
) -> None:
    # the warm start of docs/reviews/2026-10-17-halting-in-phases.md
    weights = source / "model.pt"
    raw = weights.read_bytes()
    state = cast("dict[str, torch.Tensor]", torch.load(weights, weights_only=True))
    result = _warm(tmp_path / "warm", weights, condition)
    assert result["init_weights"] == {
        "path": str(weights),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "state_digest": _independent_chain("", list(state.values())),
        "source_config_sha256": hashlib.sha256((source / "config.yaml").read_bytes()).hexdigest(),
        "head": "condition",
    }
    # the first step starts from the file's weights, the head as the condition's own: C and D
    # from zero_residual (B's head in the file is its untrained random direction), A and B as
    # loaded (A's set_halting acts at the first step, B never reads its head)
    expected = dict(state)
    if condition in {"local", "aggregate"}:
        for name in (k for k in expected if k.startswith("loop.halting")):
            fill = 1.0 if name.endswith("norm.weight") else 0.0
            expected[name] = torch.full_like(expected[name], fill)
        assert any(
            not torch.equal(state[k], expected[k]) for k in state
        )  # the file's head is not already the condition's
    assert result["initial_state_digest"] == _independent_chain("", list(expected.values()))
    fresh = cast("dict[str, torch.Tensor]", build(_cfg(), condition, 0).state_dict())
    assert result["initial_state_digest"] != _independent_chain("", list(fresh.values()))
    # a fresh run records that it is one
    plain = cast("dict[str, object]", json.loads((source / "result.json").read_text("utf-8")))
    assert plain["init_weights"] is None


def test_a_weight_snapshot_is_a_start_too(tmp_path: Path, source: Path) -> None:
    snapshot = source / "weights" / "after-00002.pt"
    result = _warm(tmp_path / "warm", snapshot)
    record = cast("dict[str, object]", result["init_weights"])
    assert record["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    # past 99999 updates the name has six digits (the 144000-update runs)
    six = source / "weights" / "after-144000.pt"
    _ = six.write_bytes(snapshot.read_bytes())
    try:
        record = cast("dict[str, object]", _warm(tmp_path / "six", six)["init_weights"])
    finally:
        six.unlink()
    assert record["path"] == str(six)


def test_a_start_of_unknown_origin_another_model_or_another_shape_is_refused(
    tmp_path: Path, source: Path
) -> None:
    good = cast("dict[str, torch.Tensor]", torch.load(source / "model.pt", weights_only=True))
    u = good["loop.u"]

    def fake(label: str, payload: object) -> Path:
        """A run directory with the stage-1 snapshot and ``payload`` saved as its model.pt."""
        directory = tmp_path / label
        directory.mkdir()
        _ = (directory / "config.yaml").write_bytes(CONFIG.read_bytes())
        torch.save(payload, directory / "model.pt")
        return directory / "model.pt"

    def refused(weights: Path, match: str) -> None:
        out = tmp_path / "refused"
        with pytest.raises(ValueError, match=match):
            _ = _warm(out, weights)
        assert not out.exists()  # refused before anything is written

    # weights of no run: no snapshot of the recipe they were trained under
    loose = tmp_path / "loose.pt"
    _ = loose.write_bytes((source / "model.pt").read_bytes())
    refused(loose, "not a run's model.pt")
    refused(fake("named", good).parent / "weights" / "after-2.pt", "not a run's model.pt")
    # a source run of another model, whether or not its shapes differ
    epsilon = _variant(tmp_path, "halt_epsilon: 0.01", "halt_epsilon: 0.02")
    refused(_two_updates(epsilon, tmp_path / "eps") / "model.pt", "run's model config differs")
    conv = _variant(tmp_path, "key_conv: 4 ", "key_conv: 3 ")
    refused(_two_updates(conv, tmp_path / "conv") / "model.pt", "run's recurrence config differs")
    # weights that do not fit the model, name by name, before load_state_dict could cast them
    missing = {k: v for k, v in good.items() if k != "loop.u"}
    refused(fake("missing", missing), r"missing \['loop.u'\], unexpected \[\]")
    refused(fake("extra", good | {"loop.v": u}), r"missing \[\], unexpected \['loop.v'\]")
    refused(fake("shape", good | {"loop.u": u[:-1]}), r"loop.u is torch.float32\(7, ")
    refused(fake("dtype", good | {"loop.u": u.double()}), r"loop.u is torch.float64\(8, ")
    for bad in (float("nan"), float("inf")):
        one = u.clone()  # a single non-finite element among finite ones
        one[3, 0] = bad
        refused(fake(f"value-{bad}", good | {"loop.u": one}), "loop.u has a non-finite value")
    refused(fake("list", list(good.values())), "not a mapping of names to tensors")
    number: dict[str, object] = {**good, "loop.u": 1.0}
    refused(fake("number", number), "not a mapping of names to tensors")
    # the training sections may differ: another schedule from the same weights is the point
    rate = _variant(tmp_path, "peak_lr: 1.0e-3", "peak_lr: 5.0e-4")
    other = _two_updates(rate, tmp_path / "rate") / "model.pt"
    assert _warm(tmp_path / "accepted", other)["init_weights"] is not None
