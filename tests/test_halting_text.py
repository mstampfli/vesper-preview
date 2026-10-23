"""Tests for vesper.halting_text: the joint halting update on real text (pilot phase 2)."""

import dataclasses
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
import yaml
from torch import Tensor

from conftest import replaced_after_first_read, small_raw
from test_train import Builder, run_fixture
from vesper import halting_text
from vesper.config import VesperConfig, load, parse
from vesper.corpus import Component, Split, load_finalization
from vesper.data import Chunk, Document, Stream
from vesper.evals import static_profile
from vesper.exits import hazard_baseline
from vesper.halting_calibration import static_frontier
from vesper.halting_text import (
    Plan,
    depth_nll,
    evaluate_text,
    load_text_source,
    pooled_weights,
    train_text_update,
    training_stream,
    validation_chunks,
)
from vesper.halting_text import run as run_text
from vesper.joint_update import (
    GRAD_CLIP,
    ROLLOUTS,
    optimizer,
    release_backbone,
    start_head,
    update_step,
)
from vesper.loop import CdfRule, LoopResult, TraceRule
from vesper.model import Vesper
from vesper.tokenizer import Tokenizer
from vesper.train import held_out_chunks, run

CPU = torch.device("cpu")
PLAN = Plan(
    seed=0,
    depth_cost=0.3,
    dense_weight=0.1,
    updates=2,
    rows=1,
    save_at=(0, 2),
    slices=2,
    recompute=True,
    device="cpu",
    preflight=False,
    backbone="joint",
)
FROZEN = dataclasses.replace(PLAN, backbone="frozen", dense_weight=0.0)


def three_pass_config() -> VesperConfig:
    raw = small_raw()
    raw["model"]["t_max"] = 3
    return parse(raw, "test")


def text_chunks(rows: int, n: int, count: int) -> list[Chunk]:
    docs = [
        Document(f"d{i}", (torch.arange(1, 30 + 7 * i, dtype=torch.int32) * (i + 3)) % 60 + 1)
        for i in range(12)
    ]
    stream = Stream(iter(docs), slots=rows, chunk=n)
    chunks: list[Chunk] = []
    while len(chunks) < count and (chunk := stream.next()) is not None:
        chunks.append(chunk)
    assert len(chunks) == count
    return chunks


def early_stopping_model() -> Vesper:
    """A three-pass model whose head stops some tokens before the last pass."""
    model = Vesper(three_pass_config(), torch.Generator().manual_seed(0))
    with torch.no_grad():
        for p in model.loop.halting_head().parameters():
            _ = p.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(3))
    return model


def weights(path: Path) -> dict[str, Tensor]:
    return cast("dict[str, Tensor]", torch.load(path, weights_only=True))


def test_pooled_weights_make_the_mean_trajectory_objective_the_pooled_nll() -> None:
    # unequal label counts per row and padding: the mean over the E rows of each row's weighted
    # NLL is the NLL pooled over every real label (maintainer B 0617971: pool by actual loss-token
    # counts), not the mean of per-row means
    rows: list[list[bool]] = [[True, True, True, False], [True, False, False, False]]
    values: list[list[float]] = [[1.0, 2.0, 3.0, 50.0], [10.0, 60.0, 70.0, 80.0]]
    mask = torch.tensor(rows)
    ce = torch.tensor(values, dtype=torch.float64)
    w = pooled_weights(mask)
    assert w.dtype == torch.float64
    assert bool((w[~mask] == 0).all())
    assert float((ce * w).sum(1).mean()) == pytest.approx((1 + 2 + 3 + 10) / 4)
    assert float((ce * w).sum(1).mean()) != pytest.approx((2.0 + 10.0) / 2)  # per-row means


def test_pooled_weights_refuse_a_batch_without_a_label() -> None:
    with pytest.raises(ValueError, match="no label"):
        _ = pooled_weights(torch.zeros(2, 4, dtype=torch.bool))


def test_depth_nll_reads_every_chunk_from_fresh_states() -> None:
    # the phase-2 context (maintainer B 0617971): independent chunks, one reset policy. A chunk's
    # per-token NLL must not depend on the chunks before it, and each depth's sum is that
    # chunk's static profile alone (cold states)
    cfg = three_pass_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    chunks = text_chunks(rows=2, n=8, count=4)
    nll = depth_nll(model, chunks)
    assert nll.dtype == torch.float64
    labels = [int(c.loss_mask.sum()) for c in chunks]
    assert tuple(nll.shape) == (cfg.model.t_max, sum(labels))
    alone = depth_nll(model, chunks[2:])
    assert torch.equal(nll[:, sum(labels[:2]) :], alone)
    start = 0
    for chunk, count in zip(chunks, labels, strict=True):
        sums = static_profile(model, [chunk], cfg).loss_sums
        for depth in range(cfg.model.t_max):
            got = float(nll[depth, start : start + count].sum())
            assert got == pytest.approx(sums[depth], rel=1e-6), depth
        start += count
    carried = static_profile(model, chunks, cfg).loss_sums  # the carried context differs
    assert any(
        float(nll[d].sum()) != pytest.approx(carried[d], rel=1e-6) for d in range(cfg.model.t_max)
    )


def test_the_starting_head_policy_is_full_depth_and_its_paired_difference_is_zero() -> None:
    # start_head's bias with the hazard baseline: the q = 0.5 policy exits at the last pass, so
    # the policy IS full depth and every paired per-token difference is exactly 0
    cfg = three_pass_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    _ = start_head(model)
    record = evaluate_text(model, text_chunks(rows=2, n=8, count=3), random_seeds=(0, 1))
    t_max = cfg.model.t_max
    assert record["policy"]["depth"] == pytest.approx(float(t_max))
    assert record["policy"]["nll"] == pytest.approx(record["fixed_nll"][-1], rel=1e-9)
    paired = record["paired_vs_full_depth"]
    assert (paired["mean"], paired["improved"], paired["worse"]) == (0.0, 0.0, 0.0)
    assert paired["depth"] == t_max


def test_the_policy_is_the_cdf_rule_and_the_frontier_is_built_on_negated_nll() -> None:
    # a head pushed to exit early: the policy's depth and pooled NLL are CdfRule(head, 0.5)'s
    # on fresh states, and the frontier is static_frontier over (depth, -NLL) of every fixed
    # depth, negated back (maintainer B 0617971: the frontier in pooled NLL, never in perplexity)
    model = early_stopping_model()
    head = model.loop.halting_head()
    chunks = text_chunks(rows=2, n=8, count=3)
    record = evaluate_text(model, chunks, random_seeds=(0,))
    nll_sum, labels, depth_sum, tokens = 0.0, 0, 0.0, 0
    with torch.no_grad():
        for c in chunks:
            states = model.init_states(int(c.tokens.shape[0]), torch.float32, CPU)
            result = model.run(c.tokens, c.pos, states, exit_rule=CdfRule(head, 0.5))
            ce = torch.nn.functional.cross_entropy(
                model.head(result.y).flatten(0, 1).float(),
                c.targets.long().flatten(),
                reduction="none",
            ).view(c.targets.shape)
            nll_sum += float(ce[c.loss_mask].double().sum())
            labels += int(c.loss_mask.sum())
            depth_sum += float(result.n_iters[c.valid].double().sum())
            tokens += int(c.valid.sum())
    policy = record["policy"]
    assert policy["nll"] == pytest.approx(nll_sum / labels, rel=1e-6)
    assert policy["depth"] == pytest.approx(depth_sum / tokens)
    assert 1.0 <= policy["depth"] < 3  # the head does stop early
    points = [(float(d + 1), -v) for d, v in enumerate(record["fixed_nll"])]
    want = -static_frontier(points, policy["depth"])
    assert record["frontier_nll_at_policy_depth"] == pytest.approx(want)
    best = record["paired_vs_best_fixed"]["depth"]
    assert record["fixed_nll"][best - 1] == min(record["fixed_nll"])


def test_evaluation_restores_the_callers_mode_and_repeats() -> None:
    # evaluate_text switches to eval mode and restores the caller's mode; nothing in Vesper
    # reads the mode today (recurrence.state_read_noise is declared in the config but no module
    # applies it: measured 2026-10-20, a mutant without the switch is equivalent), so the switch
    # is defensive and what this checks is the restored mode and repeatable results
    model = Vesper(three_pass_config(), torch.Generator().manual_seed(0))
    chunks = text_chunks(rows=2, n=8, count=2)
    _ = model.train()
    first = evaluate_text(model, chunks, random_seeds=(0,))
    after_train = _mode(model)
    second = evaluate_text(model, chunks, random_seeds=(0,))
    _ = model.eval()
    third = evaluate_text(model, chunks, random_seeds=(0,))
    after_eval = _mode(model)
    assert (after_train, after_eval) == (True, False)
    assert first == second == third


def _mode(model: Vesper) -> bool:
    return model.training


def test_random_controls_replay_draws_from_the_policys_own_histogram() -> None:
    # exact oracle: per seed, every chunk's depths are torch.multinomial over the policy's
    # depth histogram with that seed's generator, in chunk order; the control's histogram over
    # real tokens is exactly the replayed draws'
    chunks = text_chunks(rows=2, n=8, count=3)
    record = evaluate_text(early_stopping_model(), chunks, random_seeds=(0, 1, 2))
    histogram = record["policy"]["depth_histogram"]
    weights = torch.tensor(histogram, dtype=torch.float64)
    assert [c["seed"] for c in record["random"]] == [0, 1, 2]
    for seed, control in zip((0, 1, 2), record["random"], strict=True):
        g = torch.Generator().manual_seed(seed)
        want = torch.zeros(len(histogram), dtype=torch.int64)
        for c in chunks:
            drawn = torch.multinomial(weights, c.tokens.numel(), replacement=True, generator=g)
            want += torch.bincount(drawn.view(c.tokens.shape)[c.valid], minlength=len(histogram))
        assert control["depth_histogram"] == [int(want[t]) for t in range(len(histogram))]


def test_the_training_stream_reads_only_the_training_side(
    tmp_path: Path, book_builder: Builder
) -> None:
    cfg, _, parts, receipt, tokenizer_path = run_fixture(tmp_path, book_builder, 1000, 1)
    fin = load_finalization(receipt)
    tokenizer = Tokenizer.load(tokenizer_path)
    fraction = cfg.training.held_out_fraction
    held = Split(fin, fraction, held_out=True)
    evaluation = set(_read_documents(held_out_chunks(parts, 0, tokenizer, held, 8, cfg)))
    stream = training_stream(parts, 0, tokenizer, Split(fin, fraction), rows=2, n=8)
    read: list[Chunk] = []
    while (chunk := stream.next()) is not None:
        read.append(chunk)
    trained = set(_read_documents(read))
    assert trained
    assert evaluation
    assert not trained & evaluation  # the two sides never share a document
    with pytest.raises(ValueError, match="held-out"):
        _ = training_stream(parts, 0, tokenizer, Split(fin, fraction, held_out=True), 2, 8)


def _read_documents(chunks: list[Chunk]) -> list[str]:
    return sorted({d for c in chunks for slot in c.doc_ids for d in slot})


def _write_pin(path: Path, documents: list[str]) -> None:
    record: dict[str, list[str]] = {"read_documents": documents}
    _ = path.write_text(json.dumps(record))


def _step_of(path: Path) -> int:
    return int(path.name.split("-")[1])


def test_validation_is_bound_to_the_test_pin(tmp_path: Path, book_builder: Builder) -> None:
    # Maintainer B e97d5ca: every validation read stays inside the accepted read set; a pin that
    # lacks one of the chunks' documents refuses, and the pin's hash travels with the chunks
    cfg, _, parts, receipt, tokenizer_path = run_fixture(tmp_path, book_builder, 1000, 1)
    fin = load_finalization(receipt)
    tokenizer = Tokenizer.load(tokenizer_path)
    held = Split(fin, cfg.training.held_out_fraction, held_out=True)
    docs = _read_documents(held_out_chunks(parts, 0, tokenizer, held, 1, cfg))
    assert docs
    pin = tmp_path / "pin.json"
    _write_pin(pin, docs)
    pinned = validation_chunks(parts, tokenizer, held, cfg, pin, micro=1)
    assert pinned.pin_sha256 == hashlib.sha256(pin.read_bytes()).hexdigest()
    assert len(pinned.chunks) == cfg.training.eval_chunks
    _write_pin(pin, docs[1:])
    with pytest.raises(ValueError, match="outside the pin"):
        _ = validation_chunks(parts, tokenizer, held, cfg, pin, micro=1)
    with pytest.raises(ValueError, match="held-out"):
        _ = validation_chunks(
            parts, tokenizer, Split(fin, cfg.training.held_out_fraction), cfg, pin, micro=1
        )


@dataclasses.dataclass(frozen=True)
class PhaseOne:
    """A tiny fixed-objective vesper.train run: its files, its components, a pin of its reads."""

    files: dict[str, Path]
    parts: list[Component]

    def run(
        self, out: Path, plan: Plan, *, pin: Path | None = None, tokenizer: Path | None = None
    ) -> dict[str, object]:
        return run_text(
            self.files["checkpoint"],
            self.files["config"],
            pin or self.files["pin"],
            out,
            plan,
            parts=self.parts,
            finalization=self.files["receipt"],
            tokenizer_path=tokenizer or self.files["tokenizer"],
        )


def _phase_one(tmp_path: Path, book_builder: Builder) -> PhaseOne:
    """A tiny fixed-objective vesper.train run, its last checkpoint, and a pin of its reads."""
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 1
    )
    out = tmp_path / "phase1"
    _ = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=out,
        micro=1,
        steps=None,
        resume=None,
        seed=0,
        device=CPU,
        objective="fixed",
    )
    checkpoint = max(out.glob("step-*"), key=_step_of)
    held = Split(load_finalization(receipt), cfg.training.held_out_fraction, held_out=True)
    tokenizer = Tokenizer.load(tokenizer_path)
    pin = tmp_path / "pin.json"
    _write_pin(pin, _read_documents(held_out_chunks(parts, 0, tokenizer, held, 8, cfg)))
    files = {
        "checkpoint": checkpoint,
        "config": config_path,
        "pin": pin,
        "receipt": receipt,
        "tokenizer": tokenizer_path,
    }
    return PhaseOne(files, list(parts))


def _lines(path: Path) -> list[dict[str, float]]:
    return [cast("dict[str, float]", json.loads(x)) for x in path.read_text().splitlines()]


def test_a_short_run_trains_jointly_evaluates_and_records_its_bindings(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    result = p1.run(tmp_path / "p2", PLAN)
    lines = _lines(tmp_path / "p2" / "train.jsonl")
    assert len(lines) == 2
    assert all(math.isfinite(x["cost"]) and x["labels"] > 0 for x in lines)
    assert result["stopped"] is None
    assert sorted(cast("dict[str, object]", result["evaluations"])) == ["0", "2"]
    assert result["pin_sha256"] == hashlib.sha256(p1.files["pin"].read_bytes()).hexdigest()
    assert result["source_objective"] == "fixed"
    saved = weights(p1.files["checkpoint"] / "model.pt")
    after = weights(tmp_path / "p2" / "weights" / "after-000002.pt")
    changed = {k for k in saved if not torch.equal(saved[k], after[k])}
    assert any("halting" in k for k in changed)  # the head trained
    assert any("halting" not in k for k in changed)  # and the backbone (joint)


def test_two_runs_of_one_plan_repeat_exactly(tmp_path: Path, book_builder: Builder) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    first = p1.run(tmp_path / "a", PLAN)
    second = p1.run(tmp_path / "b", PLAN)
    assert first["consumed_digest"] == second["consumed_digest"]
    assert first["draw_digest"] == second["draw_digest"]
    assert (tmp_path / "a" / "train.jsonl").read_text() == (
        tmp_path / "b" / "train.jsonl"
    ).read_text()
    wa = weights(tmp_path / "a" / "weights" / "after-000002.pt")
    wb = weights(tmp_path / "b" / "weights" / "after-000002.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_the_plan_record_names_every_field() -> None:
    assert set(PLAN.record()) == set(inspect.signature(Plan).parameters)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (dataclasses.replace(PLAN, depth_cost=math.nan), "depth cost"),
        (dataclasses.replace(PLAN, depth_cost=-1.0), "depth cost"),
        (dataclasses.replace(PLAN, dense_weight=math.inf), "dense weight"),
        (dataclasses.replace(PLAN, updates=0), "at least 1"),
        (dataclasses.replace(PLAN, rows=0), "at least 1"),
        (dataclasses.replace(PLAN, slices=0), "at least 1"),
        (dataclasses.replace(PLAN, save_at=(0, 3)), "save point"),
        (dataclasses.replace(PLAN, device="tpu"), "device"),
        (dataclasses.replace(PLAN, backbone="partial"), "backbone"),
        (dataclasses.replace(FROZEN, dense_weight=0.1), "dense"),
    ],
)
def test_a_refused_plan_writes_nothing(
    tmp_path: Path, book_builder: Builder, plan: Plan, message: str
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    with pytest.raises(ValueError, match=message):
        _ = p1.run(tmp_path / "p2", plan)
    assert not (tmp_path / "p2").exists()


def test_a_run_stops_cleanly_when_the_stream_runs_dry(
    tmp_path: Path, book_builder: Builder
) -> None:
    # far more updates than the tiny corpus feeds: a row without a label would make that
    # trajectory's mean NLL and depth 0 / 0, so the run stops before such a chunk, records why,
    # and every logged update is finite
    p1 = _phase_one(tmp_path, book_builder)
    # rows 3: this fixture's stream ends with two chunks in which some rows are already empty
    # (measured; rows 4 ends cleanly and would not exercise the stop)
    plan = dataclasses.replace(PLAN, updates=10_000, save_at=(0,), rows=3)
    result = p1.run(tmp_path / "p2", plan)
    lines = _lines(tmp_path / "p2" / "train.jsonl")
    assert 0 < len(lines) < 10_000
    assert str(result["stopped"]).startswith("the training stream ran dry")
    assert all(math.isfinite(x["cost"]) and math.isfinite(x["depth"]) for x in lines)


def test_components_the_finalization_was_not_taken_over_are_refused(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    shuffled = PhaseOne(p1.files, list(reversed(p1.parts)))
    with pytest.raises(ValueError, match="finalization"):
        _ = shuffled.run(tmp_path / "p2", PLAN)
    assert not (tmp_path / "p2").exists()


def test_another_tokenizer_or_a_pin_without_the_reads_is_refused_before_writing(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    other = tmp_path / "other-tok.json"
    _ = other.write_bytes(p1.files["tokenizer"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="tokenizer"):
        _ = p1.run(tmp_path / "p2", PLAN, tokenizer=other)
    empty = tmp_path / "empty-pin.json"
    _write_pin(empty, [])
    with pytest.raises(ValueError, match="outside the pin"):
        _ = p1.run(tmp_path / "p2", PLAN, pin=empty)
    assert not (tmp_path / "p2").exists()


def test_a_phase_one_checkpoint_loads_with_its_identity_and_a_changed_file_is_refused(
    tmp_path: Path, book_builder: Builder
) -> None:
    p1 = _phase_one(tmp_path, book_builder)
    checkpoint, config_path = p1.files["checkpoint"], p1.files["config"]
    source = load_text_source(checkpoint, config_path)
    saved = weights(checkpoint / "model.pt")
    loaded = cast("dict[str, Tensor]", source.model.state_dict())
    assert sorted(loaded) == sorted(saved)
    assert all(torch.equal(loaded[k], saved[k]) for k in saved)
    assert source.objective == "fixed"
    assert source.model_sha256 == hashlib.sha256((checkpoint / "model.pt").read_bytes()).hexdigest()
    assert source.cfg == load(config_path)
    other = tmp_path / "other.yaml"
    _ = other.write_text(config_path.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="config"):
        _ = load_text_source(checkpoint, other)
    data = bytearray((checkpoint / "model.pt").read_bytes())
    data[-1] ^= 1
    _ = (checkpoint / "model.pt").write_bytes(bytes(data))
    with pytest.raises(ValueError, match=r"model\.pt"):
        _ = load_text_source(checkpoint, config_path)


def _floats(values: Tensor) -> list[float]:
    return [float(v) for v in values.unbind()]


def _unequal_rows() -> tuple[Vesper, Chunk, Tensor]:
    """Maintainer B's 04b07fa case: rows with 3 and 1 real tokens, the first exiting at pass 1 and
    the second at pass 3 in both rollouts (the same draws); a three-pass model."""
    model = Vesper(three_pass_config(), torch.Generator().manual_seed(91))
    long_ids: list[int] = [2, 3, 4, 5]
    short_ids: list[int] = [6, 7]
    chunk = Stream(
        iter(
            [
                Document("long", torch.tensor(long_ids, dtype=torch.int32)),
                Document("short", torch.tensor(short_ids, dtype=torch.int32)),
            ]
        ),
        slots=2,
        chunk=3,
    ).next()
    assert chunk is not None
    assert _floats(chunk.valid.sum(1)) == [3.0, 1.0]
    draws: list[list[float]] = [[0.0] * 3, [0.999] * 3, [0.0] * 3, [0.999] * 3]
    passes: list[list[list[float]]] = [draws, draws]  # [T - 1, ROLLOUTS * E, n]
    uniforms = torch.tensor(passes, dtype=torch.float64)
    return model, chunk, uniforms


def test_the_text_update_prices_depth_over_pooled_real_tokens() -> None:
    # Maintainer B 04b07fa: the text contract pools real tokens for the depth as for the labels; the
    # per-row mean reported depth 2.0 and cost 5.39 where the pooled depth is 1.5 (4 tokens at
    # pass 1 of 3 at pass 1 and 1 at pass 3, over 4 real tokens per rollout)
    model, chunk, uniforms = _unequal_rows()
    head, backbone = start_head(model), release_backbone(model)
    plan = dataclasses.replace(
        PLAN, rows=2, recompute=False, slices=1, dense_weight=0.0, depth_cost=1.0
    )
    record = train_text_update(
        model, optimizer(head, backbone), head + backbone, chunk, uniforms, 0, plan
    )
    assert record["depth"] == pytest.approx(1.5, abs=1e-12)
    pooled_nll = cast("float", record["pooled_nll"])
    assert record["cost"] == pytest.approx(pooled_nll + 1.0 * 1.5 / 3, abs=1e-12)


def test_the_joint_core_keeps_the_per_trajectory_depth_mean_by_default() -> None:
    # the synthetic calibration's cost is unchanged: without pooled_depth each trajectory's
    # depth is the mean over its own real tokens; with it, its summed depth times b / V
    model, chunk, uniforms = _unequal_rows()
    head, backbone = start_head(model), release_backbone(model)
    params = head + backbone
    rows = [torch.cat([t] * ROLLOUTS) for t in (chunk.tokens, chunk.pos, chunk.targets)]
    mask, valid = torch.cat([chunk.loss_mask] * ROLLOUTS), torch.cat([chunk.valid] * ROLLOUTS)
    values = [
        *rows,
        mask,
        valid,
        torch.cat([pooled_weights(chunk.loss_mask)] * ROLLOUTS),
        uniforms,
        torch.zeros((), dtype=torch.float64),
    ]
    snapshot = {k: v.clone() for k, v in cast("dict[str, Tensor]", model.state_dict()).items()}
    per_row = update_step(model, optimizer(head, backbone), params, 1.0, 0.0, values)
    assert _floats(per_row.depth) == [1.0, 3.0, 1.0, 3.0]
    _ = cast("object", model.load_state_dict(snapshot))
    pooled = update_step(
        model, optimizer(head, backbone), params, 1.0, 0.0, values, pooled_depth=True
    )
    assert _floats(pooled.depth) == [1.5, 1.5, 1.5, 1.5]  # 3 * 4 / 8 and 3 * 4 / 8
    assert torch.equal(pooled.objective, per_row.objective)  # the label term is untouched
    shift = (pooled.depth - per_row.depth) / 3  # depth cost 1, T 3; float rounding only
    assert torch.allclose(pooled.cost - per_row.cost, shift, rtol=0.0, atol=1e-12)


def _changed_config(path: Path) -> bytes:
    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(path.read_text()))
    raw["training"]["eval_chunks"] = cast("int", raw["training"]["eval_chunks"]) + 1
    return yaml.safe_dump(raw).encode()


@pytest.mark.parametrize("name", ["train.json", "model.pt", "config"])
def test_a_source_file_replaced_after_its_first_read_never_reaches_the_load(
    tmp_path: Path, book_builder: Builder, name: str
) -> None:
    # Maintainer B 04b07fa: the loader hashed train.json and then read it again, so a replacement in
    # between was loaded under the manifest's hash; the weights and the config had the same
    # shape. Every file is now parsed from the bytes that were hashed.
    p1 = _phase_one(tmp_path, book_builder)
    checkpoint, config_path = p1.files["checkpoint"], p1.files["config"]
    good = load_text_source(checkpoint, config_path)
    record = cast(
        "dict[str, dict[str, object]]", json.loads((checkpoint / "train.json").read_text())
    )
    record["progress"]["step"] = 123456
    zeros = {k: torch.zeros_like(v) for k, v in weights(checkpoint / "model.pt").items()}
    blob = io.BytesIO()
    torch.save(zeros, blob)
    path, replacement = {
        "train.json": (checkpoint / "train.json", json.dumps(record).encode()),
        "model.pt": (checkpoint / "model.pt", blob.getvalue()),
        "config": (config_path, _changed_config(config_path)),
    }[name]
    with replaced_after_first_read(path, replacement) as opens:
        source = load_text_source(checkpoint, config_path)
    assert len(opens) == 1
    assert source.step == good.step
    assert source.cfg == good.cfg
    loaded = cast("dict[str, Tensor]", source.model.state_dict())
    assert all(torch.equal(loaded[k], v) for k, v in weights(checkpoint / "model.pt").items())


def test_a_source_of_another_checkpoint_schema_is_refused(
    tmp_path: Path, book_builder: Builder
) -> None:
    # the schema refusal of train.checked_record, shared (maintainer B 04b07fa: a schema-3 record
    # with a matching manifest loaded)
    p1 = _phase_one(tmp_path, book_builder)
    checkpoint = p1.files["checkpoint"]
    record = cast(
        "dict[str, dict[str, object]]", json.loads((checkpoint / "train.json").read_text())
    )
    record["recipe"]["schema"] = 3
    data = json.dumps(record).encode()
    _ = (checkpoint / "train.json").write_bytes(data)
    manifest = cast("dict[str, str]", json.loads((checkpoint / "manifest.json").read_text()))
    manifest["train.json"] = hashlib.sha256(data).hexdigest()
    _ = (checkpoint / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="schema 3"):
        _ = load_text_source(checkpoint, p1.files["config"])


def test_the_reported_pin_hash_is_the_hash_of_the_pin_that_admitted_the_reads(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B 04b07fa: the pin was parsed, then hashed by path after the chunks were read, so a
    # pin replaced in between (one admitting nothing) was the hash reported
    p1 = _phase_one(tmp_path, book_builder)
    source = load_text_source(p1.files["checkpoint"], p1.files["config"])
    held = Split(
        load_finalization(p1.files["receipt"]),
        source.cfg.training.held_out_fraction,
        held_out=True,
    )
    tokenizer = Tokenizer.load(p1.files["tokenizer"])
    pin = p1.files["pin"]
    admitting = hashlib.sha256(pin.read_bytes()).hexdigest()
    empty: dict[str, list[str]] = {"read_documents": []}
    with replaced_after_first_read(pin, json.dumps(empty).encode()) as opens:
        pinned = validation_chunks(p1.parts, tokenizer, held, source.cfg, pin)
    assert len(opens) == 1
    assert pinned.chunks
    assert pinned.pin_sha256 == admitting


@pytest.mark.parametrize("name", ["receipt", "dropped", "tokenizer"])
def test_a_run_parses_the_finalization_and_tokenizer_it_checked(
    tmp_path: Path, book_builder: Builder, name: str
) -> None:
    # the run's identity checks hashed the finalization receipt and the tokenizer and then
    # loaded them by path again, and the finalization's dropped-ids file was hashed and then
    # read by path (the shape of maintainer B 04b07fa); a replacement after the first read (here
    # unreadable bytes) must never be what the run parses
    p1 = _phase_one(tmp_path, book_builder)
    plan = dataclasses.replace(PLAN, preflight=True)
    receipt = p1.files["receipt"]
    record = cast("dict[str, str]", json.loads(receipt.read_text()))
    dropped = receipt.parent / record["dropped_file"]
    path = dropped if name == "dropped" else p1.files[name]
    with replaced_after_first_read(path, b"not the file that was checked") as opens:
        result = p1.run(tmp_path / "p2", plan)
    assert len(opens) == 1
    assert result["stopped"] is None


def test_the_command_line_refuses_a_plan_before_reading_the_source(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B 04b07fa: main resolved the source's components (reading its record) before the
    # plan's refusal of a NaN depth cost; the plan is refused first, nothing read or written
    p1 = _phase_one(tmp_path, book_builder)
    out = tmp_path / "bad"
    argv = [
        "--source", str(p1.files["checkpoint"]),
        "--config", str(p1.files["config"]),
        "--pin", str(p1.files["pin"]),
        "--out", str(out),
        "--data", str(tmp_path),
        "--finalization", str(p1.files["receipt"]),
        "--tokenizer", str(p1.files["tokenizer"]),
        "--depth-cost", "nan",
        "--updates", "2",
        "--device", "cpu",
    ]  # fmt: skip

    def source_read(*_: object) -> list[Component]:
        msg = "the source was read before the plan was checked"
        raise AssertionError(msg)

    with patch.object(halting_text, "resumed_components", source_read):
        assert halting_text.main(argv) == 1
    assert not out.exists()


# --- the frozen backbone (head-only), maintainer B 32e7587
# -------------------------------------------


def _assert_only_the_head_moved(
    source: dict[str, Tensor], start: dict[str, Tensor], after: dict[str, Tensor]
) -> None:
    """Every backbone entry of the state dict, parameters AND buffers, bitwise the source's at
    the start and at the end; the head changed between the run's own start (after-0, after
    ``start_head`` reset it) and its end. Comparing the head with the phase-1 source instead
    passed with the optimizer step disabled (maintainer B's review of 8415310)."""
    assert sorted(source) == sorted(start) == sorted(after)
    backbone = [k for k in source if not k.startswith("loop.halting")]
    assert backbone
    for snapshot in (start, after):
        unchanged = [k for k in backbone if torch.equal(source[k], snapshot[k])]
        assert unchanged == backbone
    head = [k for k in source if k.startswith("loop.halting")]
    assert any(not torch.equal(start[k], after[k]) for k in head)


def _frozen_run_holds_the_backbone(
    tmp_path: Path, book_builder: Builder, plan: Plan
) -> dict[str, object]:
    p1 = _phase_one(tmp_path, book_builder)
    result = p1.run(tmp_path / "p2", plan)
    last = plan.updates
    assert plan.save_at == (0, last)
    source = weights(p1.files["checkpoint"] / "model.pt")
    start = weights(tmp_path / "p2" / "weights" / "after-000000.pt")
    after = weights(tmp_path / "p2" / "weights" / f"after-{last:06d}.pt")
    _assert_only_the_head_moved(source, start, after)
    evaluations = cast("dict[str, dict[str, object]]", result["evaluations"])
    # the fixed-depth predictions of the frozen model are the source's, exactly
    assert evaluations["0"]["fixed_nll"] == evaluations[str(last)]["fixed_nll"]
    assert cast("dict[str, object]", result["plan"])["backbone"] == "frozen"
    assert cast("dict[str, object]", result["optimizer"])["backbone_peak_lr"] is None
    raw = (tmp_path / "p2" / "train.jsonl").read_text().splitlines()
    lines = [cast("dict[str, object]", json.loads(x)) for x in raw]
    assert len(lines) == last
    assert all(len(cast("list[float]", x["lr_groups"])) == 1 for x in lines)  # the head's only
    assert all(x["dense"] == 0.0 for x in lines)
    return result


def test_a_frozen_backbone_run_changes_only_the_head(tmp_path: Path, book_builder: Builder) -> None:
    result = _frozen_run_holds_the_backbone(tmp_path, book_builder, FROZEN)
    assert result["update_graph"] is None  # the CPU update is not captured


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_frozen_backbone_run_changes_only_the_head_across_captured_updates(
    tmp_path: Path, book_builder: Builder
) -> None:
    # WARM_STEPS eager updates, then the training graph's capture and replays: 6 updates are
    # 3 eager, 1 capture (replayed at once) and 2 more replays (maintainer B's review of 8415310: at
    # 2 updates no captured training update ran)
    plan = dataclasses.replace(FROZEN, device="cuda", updates=6, save_at=(0, 6))
    result = _frozen_run_holds_the_backbone(tmp_path, book_builder, plan)
    assert result["update_graph"] == {"warm": 3, "captures": 1, "replays": 3}


def test_the_frozen_update_s_head_gradient_is_the_paired_score_gradient() -> None:
    # independent of the update's own cost and rule code: the trajectories are re-derived from
    # the uniforms and the head's logits (with the hazard baseline), the costs from each token's
    # exit-pass loss (pooled over labels) plus the pooled depth price, and the gradient is that
    # of mean_r stopgrad(C_r - C_other) * log p_r through the head alone
    model = early_stopping_model()
    chunk = text_chunks(1, 8, 1)[0]
    t_max = model.loop.t_max
    n = int(chunk.tokens.shape[1])
    uniforms = torch.rand(
        (t_max - 1, ROLLOUTS, n), generator=torch.Generator().manual_seed(5), dtype=torch.float64
    )
    head = start_head(model)
    with torch.no_grad():  # a head that decides: the start's stop probabilities are tiny
        for p in head:
            _ = p.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(7))
    reference = {k: v.clone() for k, v in cast("dict[str, Tensor]", model.state_dict()).items()}
    plan = dataclasses.replace(FROZEN, rows=1, recompute=False, slices=1, depth_cost=0.6)
    record = train_text_update(model, optimizer(head, []), head, chunk, uniforms, 0, plan)
    assert cast("float", record["grad_norm"]) < GRAD_CLIP  # unclipped: compare raw gradients
    got = [cast("Tensor", p.grad).double().clone() for p in head]

    expected = _paired_score_gradient(reference, chunk, uniforms, 0.6)
    assert any(float(e.abs().max()) > 1e-6 for e in expected)  # the check is not vacuous
    for g, e in zip(got, expected, strict=True):
        torch.testing.assert_close(g, e, rtol=1e-4, atol=1e-7)


def _paired_score_gradient(
    reference: dict[str, Tensor], chunk: Chunk, uniforms: Tensor, depth_cost: float
) -> list[Tensor]:
    """The head's gradient of mean_r stopgrad(C_r - C_other) * log p_r for a one-row chunk,
    from a copy of the model built here (the frozen update's independent oracle)."""
    oracle = early_stopping_model()
    _ = cast("object", oracle.load_state_dict(reference))  # its result is untyped in torch
    _ = oracle.requires_grad_(requires_grad=False)
    oracle_head = oracle.loop.halting_head()
    params = list(oracle_head.parameters())
    for p in params:
        p.requires_grad = True
    t_max = oracle.loop.t_max
    n = int(chunk.tokens.shape[1])
    baseline = hazard_baseline(t_max)
    valid, labels = chunk.valid[0], chunk.loss_mask[0]

    def trace(per_pass: Tensor, r: int) -> tuple[Tensor, Tensor]:
        """Exits and the summed log-probability of the taken actions, from the pass outputs."""
        exits = torch.full((n,), t_max, dtype=torch.int64)
        log_p = torch.zeros((), dtype=torch.float64)
        for t in range(t_max - 1):
            stop = torch.sigmoid(oracle_head.logit(per_pass[t], t)[0].double() + baseline[t])
            open_ = (exits == t_max) & valid
            stops = open_ & (uniforms[t, r] < stop.detach())
            taken = torch.where(stops, stop.log(), torch.log1p(-stop))
            log_p = log_p + torch.where(open_, taken, 0.0).sum()
            exits = torch.where(stops, torch.full_like(exits, t + 1), exits)
        return exits, log_p

    def traced_run(exits: Tensor) -> LoopResult:
        with torch.no_grad():
            return oracle.run(
                chunk.tokens,
                chunk.pos,
                oracle.init_states(1, torch.float32, CPU),
                exit_rule=TraceRule(exits[None]),
            )

    costs: list[float] = []
    log_probs: list[Tensor] = []
    for r in range(ROLLOUTS):
        # copy-forward makes a later pass depend on earlier exits: replay the trace until it
        # reproduces itself (pass-major, so at most t_max rounds), then score that run
        exits = torch.full((n,), t_max, dtype=torch.int64)
        result = traced_run(exits)
        for _ in range(t_max + 1):
            new, _ = trace(result.per_pass, r)
            if torch.equal(new, exits):
                break
            exits = new
            result = traced_run(exits)
        assert torch.equal(trace(result.per_pass, r)[0], exits)  # a fixed point was reached
        _, log_p = trace(result.per_pass, r)
        nll = oracle.position_losses(result.y, chunk.targets)[0].double()[labels].mean()
        depth = exits[valid].double().mean()
        costs.append(float(nll) + depth_cost * float(depth) / t_max)
        log_probs.append(log_p)
    assert costs[0] != costs[1]  # the two rollouts took different trajectories
    surrogate = ((costs[0] - costs[1]) * log_probs[0] + (costs[1] - costs[0]) * log_probs[1]) / 2
    return [g.double() for g in torch.autograd.grad(surrogate, params)]


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], ("joint", 0.1)),
        (["--backbone", "frozen"], ("frozen", 0.0)),  # the dense term trains only the backbone
        (["--backbone", "joint", "--dense-weight", "0.3"], ("joint", 0.3)),
    ],
)
def test_the_command_line_s_dense_weight_follows_the_backbone_mode(
    extra: list[str], expected: tuple[str, float]
) -> None:
    argv = ["--source", "s", "--config", "c", "--pin", "p", "--out", "o", "--data", "d",
            "--finalization", "f", "--tokenizer", "t", "--depth-cost", "0.5", "--updates", "3",
            "--device", "cpu", *extra]  # fmt: skip
    plans: list[Plan] = []

    def capture(*args: object, **_: object) -> dict[str, object]:
        plans.append(cast("Plan", args[4]))
        return {}

    def no_components(*_: object) -> list[Component]:
        return []

    with (
        patch.object(halting_text, "resumed_components", no_components),
        patch.object(halting_text, "run", capture),
    ):
        assert halting_text.main(argv) == 0
    assert (plans[0].backbone, plans[0].dense_weight) == expected
