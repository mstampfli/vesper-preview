"""The instruction-tuning commands end to end on a tiny ``vesper.train`` checkpoint (CPU)."""

import hashlib
import json
import shutil
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from torch import Tensor

from test_halting_text import PhaseOne, _phase_one  # pyright: ignore[reportPrivateUsage]
from test_train import Builder
from vesper.config import VesperConfig, parse_instruct_bytes
from vesper.corpus import Component
from vesper.decode import CapturedDecoder
from vesper.graphs import WARM_STEPS, GraphFamily
from vesper.halting_text import load_text_source
from vesper.instruct import (
    HALTING_PREFIX,
    ExampleLosses,
    Recipe,
    _fresh,  # pyright: ignore[reportPrivateUsage]
    days_view,
    evaluate_examples,
    main,
    tuning_optimizer,
)
from vesper.instruct_data import Batch
from vesper.model import Vesper
from vesper.tokenizer import Tokenizer

CATEGORIES = ("open_qa", "closed_qa", "classification", "brainstorming")


def write_dolly(path: Path, salt: str = "") -> str:
    rows: list[dict[str, str]] = []
    for i in range(60):
        context = f"Shared passage {i % 7} about colours{salt}." if i % 3 == 0 else ""
        rows.append(
            {
                "instruction": f"Question {i}: which colour is item {i}?{salt}",
                "context": context,
                # the last responses long enough for the second length bucket: the first examples
                # of every split are all short, so only a per-bucket preflight reaches it
                "response": f"Item {i} is {'blue' if i % 2 else 'red'}."
                + (" " + " ".join(f"shade{j}" for j in range(12)) if i >= 45 else ""),
                "category": CATEGORIES[i % 4],
            }
        )
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    _ = path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def write_panel(path: Path, salt: str = "") -> None:
    item = {
        "id": "a",
        "instruction": f"Say red.{salt}",
        "context": "",
        "checks": [{"kind": "exact", "value": "red"}],
        "pass": ["red"],
        "fail": ["blue"],
    }
    panel = {
        "version": "test",
        "scored": [item],
        "illustrations": [{"instruction": "Name a colour.", "context": ""}],
    }
    _ = path.write_text(json.dumps(panel))


def weights(path: Path) -> dict[str, Tensor]:
    return cast("dict[str, Tensor]", torch.load(path, weights_only=True, map_location="cpu"))


def other_checkpoint(source: Path, where: Path) -> Path:
    """A valid copy of ``source`` with one weight changed (its manifest updated to match)."""
    _ = shutil.copytree(source, where)
    changed = weights(where / "model.pt")
    first = next(iter(changed))
    changed[first] = changed[first] + 1.0
    torch.save(changed, where / "model.pt")
    manifest = cast("dict[str, object]", json.loads((where / "manifest.json").read_text()))
    manifest["model.pt"] = hashlib.sha256((where / "model.pt").read_bytes()).hexdigest()
    _ = (where / "manifest.json").write_text(json.dumps(manifest))
    return where


def write_instruct(path: Path, p1: PhaseOne, dolly_sha: str, view_sha: str) -> None:
    raw: dict[str, dict[str, object]] = {
        "dataset": {"name": "test", "revision": "test", "sha256": dolly_sha},
        "split": {"key": "test-split", "train_below": 0.7, "validation_below": 0.85},
        "examples": {"max_inputs": 256, "buckets": [128, 256], "batch_tokens": 512},
        "schedule": {
            "peak_lr": 1e-3,
            "warmup_steps": 1,
            "final_lr_fraction": 0.1,
            "epochs": 3,
            "seed": 0,
        },
        "evaluation": {"depth": 2, "resamples": 50, "max_new_tokens": 8},
        "forgetting": {
            "micro": 8,
            "pin_sha256": hashlib.sha256(p1.files["pin"].read_bytes()).hexdigest(),
            "view_sha256": view_sha,
        },
    }
    _ = path.write_text(yaml.safe_dump(raw))


def test_tune_then_final_reads_the_reserved_split_once_and_only_with_the_bound_inputs(  # noqa: PLR0915
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    p1 = _phase_one(tmp_path, book_builder)

    def pinned(data: Path, resume: Path) -> Sequence[Component]:
        del data, resume
        return p1.parts  # the owned fixture is a legacy-layout run: no generation pins

    monkeypatch.setattr("vesper.instruct.resumed_components", pinned)
    source = p1.files["checkpoint"]
    days = [
        "--data", str(tmp_path), "--finalization", str(p1.files["receipt"]),
        "--pin", str(p1.files["pin"]),
    ]  # fmt: skip
    dolly, panel, instruct = tmp_path / "dolly.jsonl", tmp_path / "panel.json", tmp_path / "i.yaml"
    sha = write_dolly(dolly)
    write_panel(panel)
    write_instruct(instruct, p1, sha, "0" * 64)
    loaded = load_text_source(source, p1.files["config"])
    _, view = days_view(
        source,
        loaded,
        Tokenizer.load(p1.files["tokenizer"]),
        parse_instruct_bytes(instruct.read_bytes(), "test"),
        data=tmp_path,
        finalization=p1.files["receipt"],
        pin=p1.files["pin"],
        expected=False,
    )
    view_sha = cast("str", view["view_sha256"])

    def common(src: Path, data: Path, panel_path: Path, config: Path) -> list[str]:
        return [
            "--source", str(src), "--config", str(p1.files["config"]),
            "--tokenizer", str(p1.files["tokenizer"]), "--dolly", str(data),
            "--panel", str(panel_path), "--instruct", str(config),
        ]  # fmt: skip

    # a days view that is not the config's is refused before anything is written
    wrong = tmp_path / "wrong-view.yaml"
    write_instruct(wrong, p1, sha, "1" * 64)
    with pytest.raises(ValueError, match="days view"):
        _ = main(
            ["tune", *common(source, dolly, panel, wrong), *days, "--out", str(tmp_path / "w")]
        )
    assert not (tmp_path / "w").exists()

    write_instruct(instruct, p1, sha, view_sha)
    bound = common(source, dolly, panel, instruct)
    out = tmp_path / "tuned"
    assert main(["tune", *bound, *days, "--out", str(out)]) == 0
    result = cast("dict[str, object]", json.loads((out / "result.json").read_text()))
    assert result["halting_unchanged"] is True
    epochs = cast("list[dict[str, object]]", result["epochs"])
    assert [e["epoch"] for e in epochs] == [0, 1, 2, 3]
    assert result["selected_epoch"] in {1, 2, 3}
    saved, before = weights(out / "weights" / "epoch-3.pt"), weights(source / "model.pt")
    assert all(torch.equal(saved[k], before[k]) for k in saved if k.startswith(HALTING_PREFIX))
    assert any(not torch.equal(saved[k], before[k]) for k in saved)
    forgetting = cast("dict[str, dict[str, object]]", result["forgetting"])
    assert forgetting["view"]["view_sha256"] == view_sha
    assert cast("int", forgetting["base"]["labels"]) > 0
    # every phase of the run, in order, each line written as it ended (maintainer B, e919d2d)
    phases = cast("list[dict[str, object]]", result["memory_phases"])
    per_epoch = [f"{kind}-{e}" for e in (1, 2, 3) for kind in ("train", "save", "validation")]
    assert [p["phase"] for p in phases] == [
        "setup",
        "validation-0",
        *per_epoch,
        "halting-check",
        "forgetting-build-base",
        "forgetting-build-selected",
        "forgetting-base",
        "forgetting-selected",
    ]
    lines = (out / "memory.jsonl").read_text().splitlines()
    assert [cast("dict[str, object]", json.loads(line)) for line in lines] == phases
    assert all(p["error"] is None for p in phases)
    assert result["peak_reserved_bytes"] == 0  # no counters on the CPU

    # a phase that fails keeps its line, with the error (the run's record stays readable)
    def no_forgetting(*args: object, **kwargs: object) -> object:
        del args, kwargs
        msg = "injected in the forgetting forward"
        raise RuntimeError(msg)

    broken_run = tmp_path / "broken-run"
    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct._days_depth_loss", no_forgetting)
        with pytest.raises(RuntimeError, match="injected"):
            _ = main(["tune", *bound, *days, "--limit-examples", "3", "--out", str(broken_run)])
    ended = [
        cast("dict[str, object]", json.loads(line))
        for line in (broken_run / "memory.jsonl").read_text().splitlines()
    ]
    assert ended[-1]["phase"] == "forgetting-base"
    assert "injected" in cast("str", ended[-1]["error"])
    assert not (broken_run / "result.json").exists()

    # a preflight run has no reserved read
    pre = tmp_path / "pre"
    assert main(["tune", *bound, *days, "--limit-examples", "3", "--out", str(pre)]) == 0
    assert main(["final", *bound, "--out", str(pre)]) == 2
    assert not (pre / "final.attempt").exists()
    # the preflight trains every length bucket, alternating, however short the first examples
    lines = (pre / "train.jsonl").read_text().splitlines()
    steps = [cast("dict[str, object]", json.loads(line)) for line in lines]
    lengths = [cast("int", s["length"]) for s in steps]
    assert set(lengths) == {128, 256}
    assert lengths != sorted(lengths)

    # another setting the dataset does not show (more resamples): refused by the binding alone
    resampled = tmp_path / "resampled.yaml"
    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(instruct.read_text()))
    raw["evaluation"]["resamples"] = 51
    _ = resampled.write_text(yaml.safe_dump(raw))
    assert main(["final", *common(source, dolly, panel, resampled), "--out", str(out)]) == 2
    assert not (out / "final.attempt").exists()

    # another correctly hashed dataset (with its own config), another panel, a malformed panel,
    # another source: all refused before the claim
    other = tmp_path / "other.jsonl"
    other_config = tmp_path / "other.yaml"
    write_instruct(other_config, p1, write_dolly(other, salt=" (other)"), view_sha)
    other_panel, broken_panel = tmp_path / "other-panel.json", tmp_path / "broken-panel.json"
    write_panel(other_panel, salt=" again")
    _ = broken_panel.write_text("{")
    refused = [
        common(source, other, panel, other_config),
        common(source, dolly, other_panel, instruct),
        common(source, dolly, broken_panel, instruct),
        common(other_checkpoint(source, tmp_path / "other-source"), dolly, panel, instruct),
    ]
    for args in refused:
        assert main(["final", *args, "--out", str(out)]) == 2
        assert not (out / "final.attempt").exists()

    # a missing or changed data.json: refused before the claim
    data_json = out / "data.json"
    kept = data_json.read_bytes()
    data_json.unlink()
    assert main(["final", *bound, "--out", str(out)]) == 2
    _ = data_json.write_bytes(kept.replace(b'"kept"', b'"too_long"', 1))
    assert main(["final", *bound, "--out", str(out)]) == 2
    # the hash alone: the same record, other bytes
    _ = data_json.write_bytes(kept + b"\n")
    assert main(["final", *bound, "--out", str(out)]) == 2
    # the reproduction alone: a changed record whose hash the result was made to carry
    result_json = out / "result.json"
    kept_result = result_json.read_bytes()
    edited = kept.replace(b'"kept"', b'"too_long"', 1)
    _ = data_json.write_bytes(edited)
    forged = cast("dict[str, object]", json.loads(kept_result))
    forged["data_json_sha256"] = hashlib.sha256(edited).hexdigest()
    _ = result_json.write_text(json.dumps(forged))
    assert main(["final", *bound, "--out", str(out)]) == 2
    assert not (out / "final.attempt").exists()
    _ = data_json.write_bytes(kept)
    _ = result_json.write_bytes(kept_result)

    # a failure after the claim is recorded in the claim, and the claim stays
    failing = tmp_path / "failing"
    assert main(["tune", *bound, *days, "--out", str(failing)]) == 0

    def broken(*args: object, **kwargs: object) -> object:
        del args, kwargs
        msg = "injected after the claim"
        raise RuntimeError(msg)

    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct.evaluate_examples", broken)
        with pytest.raises(RuntimeError, match="injected"):
            _ = main(["final", *bound, "--out", str(failing)])
    failed = cast("dict[str, str]", json.loads((failing / "final.attempt/status.json").read_text()))
    assert failed["status"] == "failed"
    assert "injected" in failed["error"]
    assert main(["final", *bound, "--out", str(failing)]) == 3

    # the bound inputs: the reserved split once, never again
    assert main(["final", *bound, "--out", str(out)]) == 0
    status = cast("object", json.loads((out / "final.attempt" / "status.json").read_text()))
    assert status == {"status": "complete"}
    final = cast("dict[str, object]", json.loads((out / "final.json").read_text()))
    test = cast("dict[str, object]", final["test_paired"])
    assert cast("int", test["examples"]) > 0
    assert cast("dict[str, object]", final["panel"])["items"] == 1
    assert final["loss_slices"] == 1
    assert main(["final", *bound, "--out", str(out)]) == 3

    # the run's loss slices reach final's evaluations (maintainer B's review of 71f3093: final used
    # 1)
    sliced = tmp_path / "sliced"
    assert main(["tune", *bound, *days, "--loss-slices", "3", "--out", str(sliced)]) == 0
    sliced_result = sliced / "result.json"
    good = sliced_result.read_bytes()
    sliced_run = cast("dict[str, dict[str, object]]", json.loads(good))
    assert sliced_run["recipe"]["slices"] == 3
    # a missing or malformed recorded value: refused before the claim
    for bad in (None, 0, -1, "3", True, 2.0):
        forged_run = cast("dict[str, dict[str, object]]", json.loads(good))
        if bad is None:
            del forged_run["recipe"]["slices"]
        else:
            forged_run["recipe"]["slices"] = bad
        _ = sliced_result.write_text(json.dumps(forged_run))
        assert main(["final", *bound, "--out", str(sliced)]) == 2, bad
        assert not (sliced / "final.attempt").exists(), bad
    _ = sliced_result.write_bytes(good)
    seen: list[int] = []

    def spy(
        model: Vesper,
        evaluation: Sequence[Batch],
        *,
        depth: int,
        graphs: GraphFamily | None,
        slices: int = 1,
    ) -> ExampleLosses:
        seen.append(slices)
        return evaluate_examples(model, evaluation, depth=depth, graphs=graphs, slices=slices)

    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct.evaluate_examples", spy)
        assert main(["final", *bound, "--out", str(sliced)]) == 0
    assert seen == [3, 3]  # base and tuned
    sliced_final = cast("dict[str, object]", json.loads((sliced / "final.json").read_text()))
    assert sliced_final["loss_slices"] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_final_on_cuda_answers_the_panel_with_one_captured_decoder_per_model(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # final generated its panel on the CPU even on a CUDA run (maintainer B, 203c7b8): on CUDA each
    # model gets one captured decoder for all its prompts, and the answers equal the CPU's
    args = _tune_args(tmp_path, book_builder, monkeypatch)
    tuned = tmp_path / "tuned"
    assert main(["tune", *args, "--out", str(tuned)]) == 0
    on_cpu, on_cuda = tmp_path / "final-cpu", tmp_path / "final-cuda"
    _ = shutil.copytree(tuned, on_cpu)
    _ = shutil.copytree(tuned, on_cuda)
    final_args = args[: args.index("--data")]  # final takes no days-view inputs
    made: list[CapturedDecoder] = []

    def counted(model: Vesper, depth: int, vocab: int) -> CapturedDecoder:
        made.append(CapturedDecoder(model, depth, vocab))
        return made[-1]

    assert main(["final", *final_args, "--device", "cpu", "--out", str(on_cpu)]) == 0
    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct.CapturedDecoder", counted)
        assert main(["final", *final_args, "--device", "cuda", "--out", str(on_cuda)]) == 0
    cpu, cuda = (
        cast("dict[str, object]", json.loads((d / "final.json").read_text()))
        for d in (on_cpu, on_cuda)
    )
    assert ", CPU," in cast("str", cpu["generation"])
    assert "one captured single-token decoder per model" in cast("str", cuda["generation"])
    assert len(made) == 2  # base and tuned, each reused for every prompt
    assert all(d.graph.captures == 1 for d in made)
    panel_cpu, panel_cuda = (cast("dict[str, object]", r["panel"]) for r in (cpu, cuda))
    assert panel_cuda["rows"] == panel_cpu["rows"]
    assert cuda["illustrations"] == cpu["illustrations"]


def _tune_args(tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``tune``'s arguments for the owned fixture, every input bound (all but ``--out``)."""
    p1 = _phase_one(tmp_path, book_builder)

    def pinned(data: Path, resume: Path) -> Sequence[Component]:
        del data, resume
        return p1.parts

    monkeypatch.setattr("vesper.instruct.resumed_components", pinned)
    source = p1.files["checkpoint"]
    dolly, panel, instruct = tmp_path / "dolly.jsonl", tmp_path / "panel.json", tmp_path / "i.yaml"
    sha = write_dolly(dolly)
    write_panel(panel)
    write_instruct(instruct, p1, sha, "0" * 64)
    _, view = days_view(
        source,
        load_text_source(source, p1.files["config"]),
        Tokenizer.load(p1.files["tokenizer"]),
        parse_instruct_bytes(instruct.read_bytes(), "test"),
        data=tmp_path,
        finalization=p1.files["receipt"],
        pin=p1.files["pin"],
        expected=False,
    )
    write_instruct(instruct, p1, sha, cast("str", view["view_sha256"]))
    return [
        "--source", str(source), "--config", str(p1.files["config"]),
        "--tokenizer", str(p1.files["tokenizer"]), "--dolly", str(dolly),
        "--panel", str(panel), "--instruct", str(instruct),
        "--data", str(tmp_path), "--finalization", str(p1.files["receipt"]),
        "--pin", str(p1.files["pin"]),
    ]  # fmt: skip


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
        ),
    ],
)
def test_the_training_scope_ends_before_the_forgetting_models_are_built(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch, device: str
) -> None:
    # r7 measured the card full in the forgetting phase beside the finished training; the
    # training's model, optimizer and graph family must be gone by then, released by their
    # owners and not by a garbage collection (maintainer B, af89c52)
    args = _tune_args(tmp_path, book_builder, monkeypatch)
    alive: dict[str, weakref.ReferenceType[object]] = {}
    still: list[list[str]] = []
    real_optimizer, real_family, real_fresh = tuning_optimizer, GraphFamily, _fresh

    def optimizer(model: Vesper, recipe: Recipe, *, capturable: bool) -> torch.optim.AdamW:
        opt = real_optimizer(model, recipe, capturable=capturable)
        alive["model"], alive["optimizer"] = weakref.ref(model), weakref.ref(opt)
        return opt

    def family(warm: int = WARM_STEPS) -> GraphFamily:
        made = real_family(warm)
        alive["graphs"] = weakref.ref(made)
        return made

    def fresh(cfg: VesperConfig, state: dict[str, Tensor]) -> Vesper:
        if "model" in alive and not still:  # the first build after training: forgetting's base
            still.append([name for name, ref in alive.items() if ref() is not None])
        return real_fresh(cfg, state)

    monkeypatch.setattr("vesper.instruct.tuning_optimizer", optimizer)
    monkeypatch.setattr("vesper.instruct.GraphFamily", family)
    monkeypatch.setattr("vesper.instruct._fresh", fresh)
    out = tmp_path / f"scope-{device}"
    assert (
        main(["tune", *args, "--limit-examples", "3", "--device", device, "--out", str(out)]) == 0
    )
    assert sorted(alive) == (
        ["graphs", "model", "optimizer"] if device == "cuda" else ["model", "optimizer"]
    )
    assert still == [[]]  # checked once, and nothing of the training was alive
    result = cast("dict[str, object]", json.loads((out / "result.json").read_text()))
    assert result["halting_unchanged"] is True
    if device == "cuda":
        assert cast("dict[str, list[int]]", result["graphs"])  # the counters outlive the family
