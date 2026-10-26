"""The forgetting view follows the checkpoint's pins, not today's selection (maintainer B's
e178dab).

Adapted from maintainer B's days_probe.py: a real pinned-generation data root, a checkpoint over its
selection, then a switch of the active web generation. The view must not move.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from test_train import Builder, selection_fixture
from vesper import train as t
from vesper.config import InstructConfig, VesperConfig
from vesper.corpus import Component, Split, load_finalization
from vesper.data import Document, Stream, is_held_out
from vesper.generations import Unit, publish_generation, reference, select, selection
from vesper.halting_text import TextSource, load_text_source
from vesper.instruct import days_view
from vesper.model import Vesper
from vesper.sample import selected_components
from vesper.tokenizer import Tokenizer

NO_HASH = "0" * 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Owned:
    """An owned checkpoint over a pinned selection, its pin and an unbound view config."""

    cfg: VesperConfig
    parts: list[Component]
    source: Path
    loaded: TextSource
    data: Path
    fin: Path
    pin: Path
    tok: Tokenizer
    icfg: InstructConfig


def owned(tmp_path: Path, book_builder: Builder) -> Owned:
    cfg, config, data, fin, tokfile = selection_fixture(tmp_path, book_builder)
    parts = list(selected_components(data))
    recipe = t.recipe_of(parts, 0, cfg, "fixed")
    identity = t.identity_of(sha(tokfile), sha(config), sha(fin), recipe)
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    stream = Stream(iter([Document("owned", torch.arange(1, 41, dtype=torch.int32))]), 1, 16)
    group = t.Group(stream, model.init_states(1, torch.float32, torch.device("cpu")))
    source = tmp_path / "source"
    _ = t.checkpoint(
        source,
        model,
        t.optimizer(model, cfg),
        [group],
        t.Progress(0, 0, {}, {}, "owned"),
        identity,
        recipe,
        reason="owned",
    )
    tok = Tokenizer.load(tokfile)
    held = Split(load_finalization(fin), cfg.training.held_out_fraction, held_out=True)
    reads = t.held_out_chunks(parts, 0, tok, held, 1, cfg)
    pin = tmp_path / "pin.json"
    documents = sorted({d for c in reads for slot in c.doc_ids for d in slot})
    pinned: dict[str, list[str]] = {"read_documents": documents}
    _ = pin.write_text(json.dumps(pinned))
    icfg = InstructConfig(
        dataset_name="test",
        dataset_revision="test",
        dataset_sha256=NO_HASH,
        split_key="k",
        train_below=0.9,
        validation_below=0.95,
        max_inputs=64,
        buckets=(64,),
        batch_tokens=64,
        peak_lr=1e-3,
        warmup_steps=1,
        final_lr_fraction=0.1,
        epochs=1,
        seed=0,
        eval_depth=1,
        resamples=10,
        max_new_tokens=4,
        days_micro=1,
        pin_sha256=sha(pin),
        days_view_sha256=NO_HASH,
    )
    loaded = load_text_source(source, config)
    return Owned(cfg, parts, source, loaded, data, fin, pin, tok, icfg)


def bound_view(o: Owned) -> tuple[InstructConfig, dict[str, object]]:
    """The view's record and the config bound to its own digest."""
    _, record = days_view(
        o.source, o.loaded, o.tok, o.icfg, data=o.data, finalization=o.fin, pin=o.pin,
        expected=False,
    )  # fmt: skip
    return replace(o.icfg, days_view_sha256=str(record["view_sha256"])), record


def test_the_days_view_refuses_a_checkpoint_other_than_the_loaded_one(
    tmp_path: Path, book_builder: Builder
) -> None:
    # with the view's own digest bound nothing else refuses, so only the loaded-checkpoint check
    # can (round-7 mutant m53 died only on a zero digest's message)
    o = owned(tmp_path, book_builder)
    bound, before = bound_view(o)
    data, fin, pin = o.data, o.fin, o.pin
    other = replace(o.loaded, model_sha256=NO_HASH)
    with pytest.raises(ValueError, match="no longer the checkpoint"):
        _ = days_view(o.source, other, o.tok, bound, data=data, finalization=fin, pin=pin)
    other = replace(o.loaded, identity={**o.loaded.identity, "config_sha256": NO_HASH})
    with pytest.raises(ValueError, match="no longer the checkpoint"):
        _ = days_view(o.source, other, o.tok, bound, data=data, finalization=fin, pin=pin)
    _, again = days_view(o.source, o.loaded, o.tok, bound, data=data, finalization=fin, pin=pin)
    assert again == before


def _reshared(o: Owned) -> tuple[bytes, bytes]:
    """A self-consistent replacement of the checkpoint's header and manifest: other shares, the
    recipe's own identity digest, the manifest's train.json hash (maintainer B's 01e2b3f
    fixture)."""
    record = cast(
        "dict[str, dict[str, object]]", json.loads((o.source / "train.json").read_bytes())
    )
    recipe, identity = record["recipe"], record["identity"]
    names = [cast("str", c[0]) for c in cast("list[list[object]]", recipe["components"])]
    recipe["components"] = [
        [name, 0.99 if i == 0 else 0.01 / (len(names) - 1)] for i, name in enumerate(names)
    ]
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    identity["recipe_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    header = json.dumps(record).encode()
    manifest = cast("dict[str, str]", json.loads((o.source / "manifest.json").read_bytes()))
    manifest["train.json"] = hashlib.sha256(header).hexdigest()
    return header, json.dumps(manifest).encode()


def test_the_source_corpus_resolves_the_one_snapshot_it_bound_to_the_loaded_checkpoint(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's review of 01e2b3f: files replaced while the components were resolved, and
    # restored before the next check, resolved other shares; the snapshot read once is what resolves
    from vesper.instruct import _source_corpus  # pyright: ignore[reportPrivateUsage]  # noqa: PLC0415

    o = owned(tmp_path, book_builder)
    header, manifest = o.source / "train.json", o.source / "manifest.json"
    original = (header.read_bytes(), manifest.read_bytes())
    replacement = _reshared(o)
    shares = [[c.name, c.share] for c in o.parts]
    real_read = t.checked_record

    def swapped_while_read(path: Path) -> t.CheckedRecord:
        # every later read of the record (a resolver's own) sees the replacement, restored after
        _ = header.write_bytes(replacement[0])
        _ = manifest.write_bytes(replacement[1])
        try:
            return real_read(path)
        finally:
            _ = header.write_bytes(original[0])
            _ = manifest.write_bytes(original[1])

    monkeypatch.setattr("vesper.train.checked_record", swapped_while_read)
    parts, _ = _source_corpus(o.source, o.loaded, data=o.data, finalization=o.fin)
    assert [[c.name, c.share] for c in parts] == shares
    monkeypatch.undo()
    # the replacement in place from the start: its own identity is not the loaded one
    _ = header.write_bytes(replacement[0])
    _ = manifest.write_bytes(replacement[1])
    with pytest.raises(ValueError, match="no longer the checkpoint"):
        _ = _source_corpus(o.source, o.loaded, data=o.data, finalization=o.fin)
    _ = header.write_bytes(original[0])
    _ = manifest.write_bytes(original[1])


def test_the_days_view_is_the_checkpoints_pins_and_checks_every_input(
    tmp_path: Path, book_builder: Builder
) -> None:
    o = owned(tmp_path, book_builder)
    bound, before = bound_view(o)
    source, loaded, tok, data, fin, pin = o.source, o.loaded, o.tok, o.data, o.fin, o.pin

    # switch the selection's web generation to one with another held-out document
    old = selection(data)
    stage = tmp_path / "new-web"
    stage.mkdir()
    doc = next(
        f"<urn:changed:{i}>"
        for i in range(100)
        if is_held_out(f"<urn:changed:{i}>", o.cfg.training.held_out_fraction)
    )
    table = pa.Table.from_pydict({"id": [doc], "text": ["changed oranges purple planets " * 40]})
    pq.write_table(table, str(stage / "000.parquet"))
    _ = (stage / "000.receipt.json").write_text("{}")
    units = [Unit("000", ("000.parquet",), ("000.receipt.json",))]
    ref = reference(publish_generation("web", stage, data, units=units))
    _ = select(data, [ref if r.kind == "web" else r for r in old.references], replacing=old.sha256)
    assert list(selected_components(data)) != o.parts  # today's selection moved

    _, after = days_view(source, loaded, tok, bound, data=data, finalization=fin, pin=pin)
    assert after == before  # the checkpoint's pins did not

    # another finalization receipt (same content, other bytes) and another pin are refused
    other_fin = tmp_path / "other-finalize.json"
    _ = other_fin.write_bytes(fin.read_bytes() + b" ")
    with pytest.raises(ValueError, match="finalization receipt"):
        _ = days_view(source, loaded, tok, bound, data=data, finalization=other_fin, pin=pin)
    other_pin = tmp_path / "other-pin.json"
    _ = other_pin.write_bytes(pin.read_bytes() + b" ")
    with pytest.raises(ValueError, match="not the config's"):
        _ = days_view(source, loaded, tok, bound, data=data, finalization=fin, pin=other_pin)
    # a wider view reads documents outside the pin: refused there
    wider = replace(bound, days_micro=2)
    with pytest.raises(ValueError, match="outside the pin"):
        _ = days_view(source, loaded, tok, wider, data=data, finalization=fin, pin=pin)
    # a view whose digest is not the config's
    with pytest.raises(ValueError, match="days view's digest"):
        _ = days_view(
            source,
            loaded,
            tok,
            replace(bound, days_view_sha256="1" * 64),
            data=data,
            finalization=fin,
            pin=pin,
        )
