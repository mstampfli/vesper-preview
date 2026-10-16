import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vesper.books import Edition
from vesper.collect import collect_books, collect_code, collect_wikipedia
from vesper.corpus import file_manifest
from vesper.sample import SHARES, components, main
from vesper.tokenizer import Tokenizer, train_from_texts

Builder = Callable[[Path, str, dict[str, str]], Path]
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def write_mix(tmp_path: Path, book_builder: Builder) -> tuple[Path, Path]:
    web = tmp_path / "web"
    web.mkdir()
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": [f"<urn:{i}>" for i in range(30)],
                "text": [
                    f"web document {i} says the delta rule erases and writes" for i in range(30)
                ],
            }
        ),
        str(web / "000.parquet"),
        row_group_size=10,
    )
    mix = tmp_path / "mix"
    code_shard = tmp_path / "code-shard.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "repo_name": ["o/r"] * 6,
                "path": [f"f{i}.py" for i in range(6)],
                "content": [f"def f{i}():\n    return {i}\n" for i in range(6)],
                "license": ["mit"] * 6,
            }
        ),
        str(code_shard),
    )
    _ = collect_code(code_shard, mix / "code" / "00000.parquet")
    wiki_shard = tmp_path / "wiki-shard.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": ["1", "2", "3"],
                "url": ["u1", "u2", "u3"],
                "title": ["A", "B", "C"],
                "text": ["An article about alpha.", "An article about beta.", "Gamma."],
            }
        ),
        str(wiki_shard),
    )
    _ = collect_wikipedia(wiki_shard, mix / "wiki" / "00000.parquet")
    repo = book_builder(tmp_path / "book", "en", {"chapter-1.xhtml": "<p>Once upon a time.</p>"})
    _ = collect_books([Edition("x_book", repo, "c1")], mix / "books" / "titles.parquet", "listing")
    return web, mix


def test_components_and_manifest_cover_every_file(tmp_path: Path, book_builder: Builder) -> None:
    web, mix = write_mix(tmp_path, book_builder)
    parts = components(web, mix)
    assert [(c.name, c.share) for c in parts] == list(SHARES)
    assert sum(c.share for c in parts) == pytest.approx(1.0)
    rows = file_manifest(parts)
    assert [cast("str", r["component"]) for r in rows] == ["web", "code", "wiki", "books"]
    assert all(len(cast("str", r["sha256"])) == 64 and cast("int", r["bytes"]) > 0 for r in rows)
    (mix / "books" / "titles.parquet").unlink()
    with pytest.raises(ValueError, match="books"):
        _ = components(web, mix)


def test_main_trains_saves_and_receipts_the_frozen_tokenizer(
    tmp_path: Path, book_builder: Builder
) -> None:
    web, mix = write_mix(tmp_path, book_builder)
    out = tmp_path / "tokenizer"
    # the shrunk config keeps every relation; the vocab is the small one the tiny sample can fill
    small = tmp_path / "small.yaml"
    _ = small.write_text(
        (CONFIGS / "vesper-s.yaml").read_text().replace("vocab: 32768", "vocab: 300"),
        encoding="utf-8",
    )
    assert "vocab: 300" in small.read_text()
    code = main(
        [
            "--legacy-layout",
            "--web",
            str(web),
            "--mix",
            str(mix),
            "--out",
            str(out),
            "--config",
            str(small),
            "--budget-bytes",
            "800",
            "--seed",
            "3",
            "--label",
            "test",
        ]
    )
    assert code == 0
    receipts = sorted(out.glob("*.receipt.json"))
    assert len(receipts) == 1
    finalizations = sorted(out.glob("finalize-*.json"))
    assert len(finalizations) == 1  # the driver finalized first
    record = cast("dict[str, object]", json.loads(receipts[0].read_text()))
    artifact = out / cast("str", record["artifact"])
    assert artifact.is_file()
    assert (out / f"{artifact.name}.sha256").read_text().split()[0] == record["artifact_sha256"]
    assert record["classification"] == "test"
    inputs = cast("dict[str, object]", record["input"])
    assert inputs["seed"] == 3
    assert inputs["byte_budget"] == 800
    finalization = cast("dict[str, object]", inputs["finalization"])
    assert finalization["receipt"] == str(finalizations[0])
    assert finalization["dropped"] == 0
    per = cast("dict[str, dict[str, object]]", inputs["components"])
    assert set(per) == {"web", "code", "wiki", "books"}
    assert per["web"]["budget_bytes"] == 640
    assert sum(cast("int", p["budget_bytes"]) for p in per.values()) == 800
    assert cast("int", per["web"]["utf8_bytes_used"]) >= 640
    assert cast("int", per["books"]["utf8_bytes_used"]) == len(b"Once upon a time.")
    assert cast("int", inputs["documents_used"]) == sum(
        cast("int", p["documents_used"]) for p in per.values()
    )
    tok = cast("dict[str, object]", record["tokenizer"])
    assert tok["vocab_size_actual"] == 300
    assert cast("dict[str, object]", tok["round_trip_check"])["identical"] is True
    loaded = Tokenizer.load(artifact)
    assert loaded.decode(loaded.encode("alpha beta")) == "alpha beta"
    with pytest.raises(SystemExit):
        _ = main([])


def test_an_input_that_changes_during_the_run_is_refused(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, fifth round): a source replaced between the plan and
    # the training pass; eligibility and the artifact must not describe files that are gone
    web, mix = write_mix(tmp_path, book_builder)

    def replacing(texts: Iterable[str], vocab: int, *, rule: str = "words-v1") -> Tokenizer:
        tokenizer = train_from_texts(texts, vocab, rule=rule)
        pq.write_table(
            pa.Table.from_pydict({"id": ["<urn:new>"], "text": ["replaced after the plan"]}),
            str(web / "000.parquet"),
        )
        return tokenizer

    monkeypatch.setattr("vesper.sample.train_from_texts", replacing)
    small = tmp_path / "small.yaml"
    _ = small.write_text(
        (CONFIGS / "vesper-s.yaml").read_text().replace("vocab: 32768", "vocab: 300"),
        encoding="utf-8",
    )
    out = tmp_path / "tokenizer"
    code = main(
        [
            "--legacy-layout",
            "--web",
            str(web),
            "--mix",
            str(mix),
            "--out",
            str(out),
            "--config",
            str(small),
            "--budget-bytes",
            "800",
            "--label",
            "test",
        ]
    )
    assert code == 1
    assert list(out.glob("vesper-s-*.json")) == []


def test_a_source_replaced_and_restored_during_training_is_not_eligible(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, seventh round): the source is replaced while the
    # trainer reads and restored to its exact bytes before the final check; counts and endpoint
    # hashes agree, only the consumed bytes differ
    web, mix = write_mix(tmp_path, book_builder)
    source = web / "000.parquet"
    original_bytes = source.read_bytes()
    table = pq.ParquetFile(str(source)).read_row_group(0)
    texts = cast("list[str]", table.column("text").to_pylist())
    ids = cast("list[str]", table.column("id").to_pylist())
    replaced = [t[:-1] + ("Y" if t[-1] != "Y" else "Z") for t in texts]  # same length, other bytes

    def replacing(stream: Iterable[str], vocab: int, *, rule: str = "words-v1") -> Tokenizer:
        pq.write_table(
            pa.Table.from_pydict({"id": ids, "text": replaced}), str(source), row_group_size=10
        )
        tokenizer = train_from_texts(list(stream), vocab, rule=rule)
        _ = source.write_bytes(original_bytes)
        return tokenizer

    monkeypatch.setattr("vesper.sample.train_from_texts", replacing)
    small = tmp_path / "small.yaml"
    _ = small.write_text(
        (CONFIGS / "vesper-s.yaml").read_text().replace("vocab: 32768", "vocab: 300"),
        encoding="utf-8",
    )
    out = tmp_path / "tokenizer"
    code = main(
        [
            "--legacy-layout",
            "--web",
            str(web),
            "--mix",
            str(mix),
            "--out",
            str(out),
            "--config",
            str(small),
            "--budget-bytes",
            "800",
            "--label",
            "frozen",
        ]
    )
    assert code == 0
    record = cast("dict[str, object]", json.loads(min(out.glob("*.receipt.json")).read_text()))
    assert record["freeze_eligible"] is False
    assert "differs from the plan" in cast("str", record["classification"])
    assert record["plan_consumed_sha256"] != record["trained_consumed_sha256"]


def test_components_resolve_through_the_selection_and_pin_their_generations(
    tmp_path: Path, book_builder: Builder
) -> None:
    from vesper.generations import (  # noqa: PLC0415
        Reference,
        Unit,
        publish_generation,
        reference,
        select,
    )
    from vesper.sample import components_for, selected_components  # noqa: PLC0415

    web, mix = write_mix(tmp_path, book_builder)
    data = tmp_path / "data"
    refs: list[Reference] = []
    for name, _ in SHARES:
        directory = web if name == "web" else mix / name
        staging = tmp_path / f"staging-{name}"
        staging.mkdir()
        units: list[Unit] = []
        for path in sorted(directory.glob("*.parquet")):
            _ = shutil.copyfile(path, staging / path.name)
            _ = (staging / f"{path.stem}.receipt.json").write_text("{}", encoding="utf-8")
            units.append(Unit(path.stem, (path.name,), (f"{path.stem}.receipt.json",)))
        refs.append(reference(publish_generation(name, staging, data, units=units)))
    with pytest.raises(ValueError, match="no selection"):
        _ = selected_components(data)
    chosen = select(data, refs, replacing=None)
    parts = selected_components(data)
    assert [(c.name, c.share) for c in parts] == list(SHARES)
    for part, ref in zip(parts, refs, strict=True):
        assert part.generation == ref.pin
        assert all(p.parent == data / ref.kind / ref.identity for p in part.paths)
        assert len(part.paths) >= 1
    rows = file_manifest(parts)
    assert all(cast("str", r["generation"]).startswith(f"{r['component']}/gen-") for r in rows)
    # a selection without one kind is refused for the components
    partial = select(data, refs[:3], replacing=chosen.sha256)
    with pytest.raises(ValueError, match="no generation for 'books'"):
        _ = selected_components(data)
    _ = select(data, refs, replacing=partial.sha256)
    # the command line names the inputs one way only
    assert components_for(data=data, legacy_layout=False, web=None, mix=None) == parts
    legacy = components_for(data=None, legacy_layout=True, web=web, mix=mix)
    assert all(c.generation is None for c in legacy)
    with pytest.raises(ValueError, match="one way"):
        _ = components_for(data=data, legacy_layout=True, web=web, mix=mix)
    with pytest.raises(ValueError, match="one way"):
        _ = components_for(data=None, legacy_layout=False, web=web, mix=mix)
    with pytest.raises(ValueError, match="one way"):
        _ = components_for(data=None, legacy_layout=True, web=web, mix=None)
