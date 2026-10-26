"""Files replaced after their first checked load change nothing (maintainer B's review of eb4d9f9).

``tune`` keeps the checked source as a snapshot for the forgetting baseline; ``final`` builds
the tuned model from the checked config and carries the hash of the result bytes it parsed.
Each swap below happens AFTER the first load, the moment a reload would have seen it.
"""

import argparse
import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml

import vesper.instruct as instruct_module
from test_halting_text import PhaseOne, _phase_one  # pyright: ignore[reportPrivateUsage]
from test_instruct_lifecycle import weights, write_dolly, write_instruct, write_panel
from test_train import Builder
from vesper.config import InstructConfig, parse_instruct_bytes
from vesper.corpus import Component
from vesper.data import Chunk
from vesper.halting_text import TextSource, load_text_source
from vesper.instruct import days_view, main
from vesper.tokenizer import Tokenizer


def copy_flat(src: Path, dst: Path) -> None:
    """A checkpoint directory's files (it has no subdirectories), copied byte for byte."""
    dst.mkdir()
    for f in src.iterdir():
        _ = (dst / f.name).write_bytes(f.read_bytes())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_manifest(source: Path, name: str) -> None:
    manifest = cast("dict[str, object]", json.loads((source / "manifest.json").read_text()))
    manifest[name] = sha(source / name)
    _ = (source / "manifest.json").write_text(json.dumps(manifest))


def replace_weights(source: Path) -> None:
    """A consistent checkpoint with other weights at the same path."""
    changed = weights(source / "model.pt")
    first = next(iter(changed))
    changed[first] = changed[first] + 1.0
    torch.save(changed, source / "model.pt")
    rewrite_manifest(source, "model.pt")


def replace_config(source: Path, config: Path) -> None:
    """A consistent checkpoint whose config (attention window) is another one."""
    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(config.read_text()))
    raw["model"]["window"] = 1
    _ = config.write_text(yaml.safe_dump(raw))
    record = cast("dict[str, dict[str, object]]", json.loads((source / "train.json").read_text()))
    record["identity"]["config_sha256"] = sha(config)
    _ = (source / "train.json").write_text(json.dumps(record))
    rewrite_manifest(source, "train.json")


def setup(tmp_path: Path, p1: PhaseOne) -> tuple[list[str], list[str]]:
    source = p1.files["checkpoint"]
    dolly, panel, instruct = tmp_path / "dolly.jsonl", tmp_path / "panel.json", tmp_path / "i.yaml"
    dolly_sha = write_dolly(dolly)
    write_panel(panel)
    write_instruct(instruct, p1, dolly_sha, "0" * 64)
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
    write_instruct(instruct, p1, dolly_sha, cast("str", view["view_sha256"]))
    bound = [
        "--source", str(source), "--config", str(p1.files["config"]),
        "--tokenizer", str(p1.files["tokenizer"]), "--dolly", str(dolly),
        "--panel", str(panel), "--instruct", str(instruct),
    ]  # fmt: skip
    days = [
        "--data", str(tmp_path), "--finalization", str(p1.files["receipt"]),
        "--pin", str(p1.files["pin"]),
    ]  # fmt: skip
    return bound, days


def read(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text()))


def test_files_replaced_after_their_first_load_change_nothing(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    p1 = _phase_one(tmp_path, book_builder)

    def pinned(data: Path, resume: Path) -> Sequence[Component]:
        del data, resume
        return p1.parts  # the owned fixture is a legacy-layout run: no generation pins

    monkeypatch.setattr("vesper.instruct.resumed_components", pinned)
    bound, days = setup(tmp_path, p1)
    source, config = p1.files["checkpoint"], p1.files["config"]
    kept_source, kept_config = tmp_path / "kept-source", tmp_path / "kept-config.yaml"
    copy_flat(source, kept_source)
    _ = kept_config.write_bytes(config.read_bytes())

    def restore() -> None:
        shutil.rmtree(source)
        copy_flat(kept_source, source)
        _ = config.write_bytes(kept_config.read_bytes())

    clean, swapped = tmp_path / "clean", tmp_path / "swapped"
    assert main(["tune", *bound, *days, "--out", str(clean)]) == 0

    # tune: the source's weights AND config replaced right after the view (after the first load)
    real_view = days_view

    def view_then_swap(
        source_dir: Path, loaded: TextSource, tok: Tokenizer, icfg: InstructConfig, **kw: Path
    ) -> tuple[list[Chunk], dict[str, object]]:
        made = real_view(
            source_dir, loaded, tok, icfg, data=kw["data"], finalization=kw["finalization"],
            pin=kw["pin"],
        )  # fmt: skip
        replace_weights(source)
        replace_config(source, config)
        return made

    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct.days_view", view_then_swap)
        assert main(["tune", *bound, *days, "--out", str(swapped)]) == 0
    restore()
    clean_result, swapped_result = read(clean / "result.json"), read(swapped / "result.json")
    assert swapped_result["forgetting"] == clean_result["forgetting"]
    assert swapped_result["epochs"] == clean_result["epochs"]

    # final: the source's config replaced and result.json rewritten right after the checks
    real_checks = instruct_module._final_checks  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    original_result = (swapped / "result.json").read_bytes()

    def checks_then_swap(args: argparse.Namespace, out: Path) -> object:
        checked: object = real_checks(args, out)
        replace_config(source, config)
        forged = read(swapped / "result.json")
        forged["selected_epoch"] = 999
        _ = (swapped / "result.json").write_text(json.dumps(forged))
        return checked

    assert main(["final", *bound, "--out", str(clean)]) == 0
    with monkeypatch.context() as patch:
        patch.setattr("vesper.instruct._final_checks", checks_then_swap)
        assert main(["final", *bound, "--out", str(swapped)]) == 0
    restore()
    clean_final, swapped_final = read(clean / "final.json"), read(swapped / "final.json")
    assert swapped_final["test_paired"] == clean_final["test_paired"]
    assert swapped_final["result_sha256"] == hashlib.sha256(original_result).hexdigest()
    assert swapped_final["selected_epoch"] == clean_final["selected_epoch"]
