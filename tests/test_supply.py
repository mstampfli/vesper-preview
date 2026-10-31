import json
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from conftest import small_raw
from test_train import Builder, read_log, run_fixture
from vesper.config import parse
from vesper.corpus import Component, Mixer, RawReader, Split, finalize, load_finalization
from vesper.data import Stream
from vesper.keys import file_sha256
from vesper.publish import copied_snapshot
from vesper.supply import SupplyInputs, assess, count_supply, prepare_supply, read_supply
from vesper.supply import main as count_main
from vesper.tokenizer import Tokenizer, train_from_texts
from vesper.train import main as train_main
from vesper.train import run


def fixture(
    root: Path, texts: tuple[str, ...] = ("abcd", "efgh")
) -> tuple[list[Component], Tokenizer, Split, SupplyInputs]:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "input.parquet"
    pq.write_table(
        pa.Table.from_pydict({"id": [str(i) for i in range(len(texts))], "text": texts}),
        str(path),
        row_group_size=1,
    )
    parts = [Component("text", (path,), 1.0)]
    fin = finalize(parts, root / "fin")
    # No merges are learned: every ASCII character is exactly one byte-alphabet token.
    tok = train_from_texts(["a", "b"], vocab=257)
    tok.save(root / "tok.json")
    split = Split(load_finalization(fin.receipt), 0.0)
    inputs = prepare_supply(
        parts, (root / "tok.json").read_bytes(), split, file_sha256(fin.receipt)
    )
    return parts, tok, split, inputs


def test_literal_counts_roundtrip_and_mixer_exhaustion(tmp_path: Path) -> None:
    parts, tok, split, inputs = fixture(tmp_path / "data")
    supply = count_supply(inputs, tmp_path / "cache")
    assert supply.counts[0].tokens == 8
    assert supply.counts[0].documents == 2
    assert supply.counts[0].potential_labels == 6
    assert supply.counts[0].max_document_tokens == 4
    assert read_supply(tmp_path / "cache", inputs) == supply
    mixer = Mixer(parts, 19, tok, split)
    assert sum(d.tokens.numel() for d in mixer.documents()) == 8
    result = assess(supply, parts, 10)
    assert result["completion"] == "unassessed"
    assert result["warnings"]
    assert result["valid"] is True  # a nominal shortfall does not refuse a finite run


def test_corrupt_or_wrong_cache_refuses(tmp_path: Path) -> None:
    _, _, _, inputs = fixture(tmp_path / "data")
    _ = count_supply(inputs, tmp_path / "cache")
    path = tmp_path / "cache" / "supply.json"
    original = path.read_bytes()
    _ = path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="hash"):
        _ = read_supply(tmp_path / "cache", inputs)
    _ = path.write_bytes(original)
    other = replace(inputs, identity={**inputs.identity, "held_out_fraction": 0.5})
    with pytest.raises(ValueError, match="identity"):
        _ = read_supply(tmp_path / "cache", other)


def test_changed_worker_input_leaves_no_cache(tmp_path: Path) -> None:
    parts, _, _, inputs = fixture(tmp_path / "data")
    _ = parts[0].paths[0].write_bytes(b"replaced after preparation")
    with pytest.raises(ValueError, match="hash"):
        _ = count_supply(inputs, tmp_path / "cache")
    assert not (tmp_path / "cache").exists()


def test_empty_and_one_token_supply_are_not_trainable(tmp_path: Path) -> None:
    for name, texts in (("empty", ("",)), ("one", ("x",))):
        parts, _, _, inputs = fixture(tmp_path / name, texts)
        supply = count_supply(inputs, tmp_path / (name + "-cache"))
        result = assess(supply, parts, 100, cycle=True)
        assert result["valid"] is False
        assert result["errors"]
        assert result["completion"] == "unassessed"
        assert "NaN" not in json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("texts", [("abcd", "efgh"), ("",), ("x",)])
def test_plan_warns_before_model_and_saves_bound_assessment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    texts: tuple[str, ...],
) -> None:
    parts, tok, split, inputs = fixture(tmp_path / "data", texts)
    raw = small_raw(vocab=tok.vocab_size)
    raw["training"].update({"held_out_fraction": 1e-9, "total_tokens": 100})
    cfg = parse(raw, "supply test")
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text(yaml.safe_dump(raw))
    split = replace(split, held_out_fraction=1e-9)
    inputs = prepare_supply(parts, inputs.tokenizer, split, file_sha256(split.finalization.receipt))
    _ = count_supply(inputs, tmp_path / "cache")

    def no_model(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a supply plan must not construct a model")

    monkeypatch.setattr("vesper.train.Vesper", no_model)

    def plan(cache: Path | None) -> Path:
        return run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=split.finalization.receipt,
            tokenizer_path=tmp_path / "data" / "tok.json",
            out=tmp_path / "plan",
            micro=1,
            steps=None,
            resume=None,
            seed=0,
            device=torch.device("cpu"),
            supply_cache=cache,
            plan_only=True,
        )

    with pytest.raises(ValueError, match="needs a checked"):
        _ = plan(None)
    if texts != ("abcd", "efgh"):
        with pytest.raises(ValueError, match=r"supply|loss positions"):
            _ = plan(tmp_path / "cache")
        assert not (tmp_path / "plan").exists()
        return
    path = plan(tmp_path / "cache")
    result = cast("dict[str, object]", json.loads(path.read_text()))
    report = cast("dict[str, object]", result["supply"])
    assert report["completion"] == "unassessed"
    assert cast("list[dict[str, object]]", report["components"])[0]["nominal_demand"] == 100
    assert cast("dict[str, object]", result["identity"])["tokenizer_sha256"] == file_sha256(
        tmp_path / "data" / "tok.json"
    )
    assert "nominal demand 100 exceeds finite supply 8" in caplog.text
    assert not (tmp_path / "plan" / "log.jsonl").exists()


def two_files(root: Path) -> tuple[list[Component], SupplyInputs]:
    parts, _, split, inputs = fixture(root)
    second = root / "second.parquet"
    pq.write_table(pa.Table.from_pydict({"id": ["other"], "text": ["xyz"]}), str(second))
    parts = [replace(parts[0], share=0.5), Component("other", (second,), 0.5)]
    fin = finalize(parts, root / "fin-two")
    split = replace(split, finalization=load_finalization(fin.receipt))
    return parts, prepare_supply(parts, inputs.tokenizer, split, file_sha256(fin.receipt))


def test_parallel_count_and_changed_shares_reuse_the_same_inventory(tmp_path: Path) -> None:
    parts, inputs = two_files(tmp_path / "data")
    serial = count_supply(inputs, tmp_path / "serial")
    parallel = count_supply(inputs, tmp_path / "parallel", workers=2)
    assert serial.counts == parallel.counts
    assert [c.tokens for c in parallel.counts] == [8, 3]
    changed = [replace(parts[0], share=0.1), replace(parts[1], share=0.9)]
    new = prepare_supply(
        changed, inputs.tokenizer, inputs.split, file_sha256(inputs.split.finalization.receipt)
    )
    assert new.identity == inputs.identity
    cached = read_supply(tmp_path / "serial", new)
    report = assess(cached, changed, 20, cycle=True)
    rows = cast("list[dict[str, object]]", report["components"])
    assert [r["nominal_demand"] for r in rows] == [2, 18]
    assert [r["nominal_exposure"] for r in rows] == [0.25, 6.0]
    assert report["valid"] is True


@pytest.mark.parametrize("workers", [1, 2])
def test_partial_count_does_not_publish(tmp_path: Path, workers: int) -> None:
    parts, inputs = two_files(tmp_path / "data")
    _ = parts[1].paths[0].write_bytes(b"bad second file")
    with pytest.raises(ValueError, match="hash"):
        _ = count_supply(inputs, tmp_path / "cache", workers=workers)
    assert not (tmp_path / "cache").exists()
    assert not list(tmp_path.glob("*.staging"))


def test_source_changes_after_snapshot_do_not_change_the_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, _, _, inputs = fixture(tmp_path / "data")

    def reader(paths: Sequence[Path], seed: int) -> RawReader:
        _ = parts[0].paths[0].write_bytes(b"changed after snapshot verification")
        return RawReader(paths, seed)

    monkeypatch.setattr("vesper.supply.RawReader", reader)
    result = count_supply(inputs, tmp_path / "cache")
    assert result.counts[0].tokens == 8
    with pytest.raises(ValueError, match="as it is now"):
        _ = prepare_supply(
            parts, inputs.tokenizer, inputs.split, file_sha256(inputs.split.finalization.receipt)
        )


def test_snapshot_refuses_fifo_and_cleans_on_body_failure(tmp_path: Path) -> None:
    source = tmp_path / "pipe"
    os.mkfifo(source)
    with pytest.raises(ValueError, match="regular file"), copied_snapshot(source, "x", tmp_path):
        pytest.fail("FIFO must not yield")
    source.unlink()
    _ = source.write_bytes(b"abc")

    def fail_inside_snapshot() -> None:
        with copied_snapshot(source, file_sha256(source), tmp_path) as snapshot:
            assert snapshot.read_bytes() == b"abc"
            _ = source.write_bytes(b"replacement")
            assert snapshot.read_bytes() == b"abc"
            msg = "body"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="body"):
        fail_inside_snapshot()
    assert not (tmp_path / "staging").exists()


def test_count_cannot_overwrite_a_completed_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, inputs = fixture(tmp_path / "data")
    rendezvous = Barrier(2)

    def reader(paths: Sequence[Path], seed: int) -> RawReader:
        _ = rendezvous.wait(timeout=10)
        return RawReader(paths, seed)

    monkeypatch.setattr("vesper.supply.RawReader", reader)

    def attempt() -> str | None:
        try:
            return count_supply(inputs, tmp_path / "cache").sha256
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt) for _ in range(2)]
        results = [f.result() for f in futures]
    assert results.count(None) == 1
    assert read_supply(tmp_path / "cache", inputs).sha256 in results
    with pytest.raises(ValueError, match="new output"):
        _ = count_supply(inputs, tmp_path / "cache")


@pytest.mark.parametrize(
    "field",
    [
        "tokenizer_sha256",
        "finalization_sha256",
        "dropped_ids_sha256",
        "split_rule",
        "counter_sources",
        "dependencies",
        "files",
    ],
)
def test_every_identity_binding_is_checked(tmp_path: Path, field: str) -> None:
    _, _, _, inputs = fixture(tmp_path / "data")
    _ = count_supply(inputs, tmp_path / "cache")
    changed = replace(inputs, identity={**inputs.identity, field: "different"})
    with pytest.raises(ValueError, match="identity"):
        _ = read_supply(tmp_path / "cache", changed)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("tokens", -1),
        ("tokens", True),
        ("tokens", 1.5),
        ("tokens", "8"),
        ("tokens", None),
        ("potential_labels", 7),
        ("potential_labels", 8),
    ],
)
def test_invalid_count_refuses_even_with_matching_digest(
    tmp_path: Path, field: str, bad: object
) -> None:
    _, _, _, inputs = fixture(tmp_path / "data")
    _ = count_supply(inputs, tmp_path / "cache")
    path = tmp_path / "cache" / "supply.json"
    raw: dict[str, object] = cast("dict[str, object]", json.loads(path.read_bytes()))
    counts = cast("list[dict[str, object]]", raw["counts"])
    counts[0][field] = bad
    _ = path.write_text(json.dumps(raw))
    manifest: dict[str, str] = {"supply.json": file_sha256(path)}
    _ = (path.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="counts"):
        _ = read_supply(path.parent, inputs)


def test_equal_nominal_supply_is_not_completion_and_resume_keeps_buffers(tmp_path: Path) -> None:
    parts, tok, split, inputs = fixture(tmp_path / "short")
    supply = count_supply(inputs, tmp_path / "short-cache")
    assert (
        cast("list[dict[str, object]]", assess(supply, parts, 8)["components"])[0][
            "supply_vs_nominal"
        ]
        == "at"
    )
    stream = Stream(Mixer(parts, 0, tok, split).documents(), slots=1, chunk=4)
    valid = 0
    while (chunk := stream.next()) is not None:
        valid += int(chunk.valid.sum().item())
    assert valid == 7
    parts, tok, split, inputs = fixture(tmp_path / "long", ("x" * 100,))
    mixer = Mixer(parts, 0, tok, split)
    stream = Stream(mixer.documents(), slots=1, chunk=4)
    assert stream.next() is not None
    state = stream.state()
    restored = Stream.restore(state, Mixer.restore(parts, 0, tok, split, mixer.state()).documents())
    expected, actual = stream.next(), restored.next()
    assert expected is not None
    assert actual is not None
    assert torch.equal(expected.tokens, actual.tokens)
    assert sum(mixer.emitted.values()) == 100
    supply = count_supply(inputs, tmp_path / "long-cache")
    assert assess(supply, parts, 100)["completion"] == "unassessed"


def test_actual_cli_count_plan_train_and_buffered_resume(
    tmp_path: Path, book_builder: Builder, caplog: pytest.LogCaptureFixture
) -> None:
    cfg, config, parts, fin, tok = run_fixture(tmp_path, book_builder, 128, 2)
    cache = tmp_path / "cache"
    common = [
        "--config",
        str(config),
        "--legacy-layout",
        "--web",
        str(parts[0].paths[0].parent),
        "--mix",
        str(parts[1].paths[0].parent.parent),
        "--finalization",
        str(fin),
        "--tokenizer",
        str(tok),
    ]
    assert count_main([*common, "--out", str(cache), "--workers", "2"]) == 0
    plan = tmp_path / "plan"
    command = [*common, "--micro-chunks", "1", "--device", "cpu", "--supply", str(cache)]
    assert train_main([*command, "--plan", "--out", str(plan)]) == 0
    assert train_main([*command, "--plan", "--out", str(plan)]) == 1
    assert not (plan / "log.jsonl").exists()
    out = tmp_path / "run"
    assert train_main([*command, "--out", str(out), "--steps", "2"]) == 0
    header = read_log(out / "log.jsonl")[0]
    assert cast("dict[str, object]", header["supply"])["status"] == "assessed"
    checkpoint = next(out.glob("step-*"))
    before = (checkpoint / "train.json").read_bytes()
    resumed_plan = tmp_path / "resume-plan"
    assert (
        train_main([*command, "--plan", "--resume", str(checkpoint), "--out", str(resumed_plan)])
        == 0
    )
    planned = cast("dict[str, object]", json.loads((resumed_plan / "plan.json").read_bytes()))
    assert cast("dict[str, object]", planned["supply"])["total_tokens"] == 128
    assert (checkpoint / "train.json").read_bytes() == before
    continuation = tmp_path / "continued"
    assert train_main([*command, "--resume", str(checkpoint), "--out", str(continuation)]) == 0
    full = tmp_path / "full"
    assert train_main([*command, "--out", str(full)]) == 0
    continued_rows = [r for r in read_log(continuation / "log.jsonl") if r["kind"] == "step"]
    full_rows = [r for r in read_log(full / "log.jsonl") if r["kind"] == "step"]
    for row in continued_rows:
        equivalent = next(r for r in full_rows if r["step"] == row["step"])
        assert {k: v for k, v in row.items() if k not in {"seconds", "run_id"}} == {
            k: v for k, v in equivalent.items() if k not in {"seconds", "run_id"}
        }
    # Optional evidence changes the header, not any update or prefetch/restore state.
    no_cache = tmp_path / "no-cache"
    _ = run(
        cfg,
        config_path=config,
        parts=parts,
        finalization_receipt=fin,
        tokenizer_path=tok,
        out=no_cache,
        micro=1,
        steps=None,
        resume=None,
        seed=0,
        device=torch.device("cpu"),
    )
    no_rows = [r for r in read_log(no_cache / "log.jsonl") if r["kind"] == "step"]
    for actual, expected in zip(no_rows, full_rows, strict=True):
        assert {k: v for k, v in actual.items() if k not in {"seconds", "run_id"}} == {
            k: v for k, v in expected.items() if k not in {"seconds", "run_id"}
        }
    assert "Training supply is unknown" in caplog.text
    assert (
        train_main(
            [
                *command,
                "--resume",
                str(checkpoint),
                "--seed",
                "12",
                "--plan",
                "--out",
                str(tmp_path / "bad"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "bad").exists()


def test_dropped_held_and_empty_documents_match_training_membership(tmp_path: Path) -> None:
    parts, tok, split, inputs = fixture(tmp_path / "data", ("abcd", "abcd", "", "xyz"))
    supply = count_supply(inputs, tmp_path / "cache")
    assert supply.counts[0].tokens == 7  # duplicate and empty removed; 4 + 3 bytes
    assert supply.counts[0].potential_labels == 5
    for seed in (0, 1, 99):
        assert sum(d.tokens.numel() for d in Mixer(parts, seed, tok, split).documents()) == 7
    held = replace(split, held_out_fraction=1.0)
    excluded = prepare_supply(
        parts, inputs.tokenizer, held, file_sha256(split.finalization.receipt)
    )
    empty = count_supply(excluded, tmp_path / "held")
    assert empty.counts[0].tokens == 0
    assert Mixer(parts, 0, tok, held, cycle=True).pull() is None
    assert assess(empty, parts, 10, cycle=True)["valid"] is False


def test_changed_worker_membership_and_tokenizer_refuse(tmp_path: Path) -> None:
    _, _, split, inputs = fixture(tmp_path / "data")
    changed = replace(inputs, split=replace(split, held_out_fraction=0.5))
    with pytest.raises(ValueError, match="snapshots"):
        _ = count_supply(changed, tmp_path / "wrong-split")
    changed = replace(inputs, tokenizer=inputs.tokenizer + b" ")
    with pytest.raises(ValueError, match="snapshots"):
        _ = count_supply(changed, tmp_path / "wrong-tokenizer")
    assert not (tmp_path / "wrong-split").exists()
    assert not (tmp_path / "wrong-tokenizer").exists()


def test_invalid_budget_share_and_worker_options_refuse(tmp_path: Path) -> None:
    parts, _, _, inputs = fixture(tmp_path / "data")
    supply = count_supply(inputs, tmp_path / "cache")
    for budget in (0, -1, True, 10**400):
        with pytest.raises(ValueError, match=r"budget|range"):
            _ = assess(supply, parts, budget)
    for share in (float("nan"), float("inf"), -1.0, 1e308):
        with pytest.raises(ValueError, match="shares"):
            _ = assess(supply, [replace(parts[0], share=share)], 10)
    for workers in (0, -1, True):
        with pytest.raises(ValueError, match="workers"):
            _ = count_supply(inputs, tmp_path / "bad-workers", workers=workers)
    assert not (tmp_path / "bad-workers").exists()
