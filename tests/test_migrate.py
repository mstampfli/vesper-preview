import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from test_sample import write_mix
from vesper.corpus import finalize, load_finalization
from vesper.generations import SELECTION, pinned, select, selection, validate_generation
from vesper.keys import file_sha256
from vesper.migrate import (
    EXIT_COMMITTED_INCOMPLETE,
    ActivationUnrecordedError,
    apply,
    main,
    plan,
    recover,
)
from vesper.sample import components, selected_components

Builder = Callable[[Path, str, dict[str, str]], Path]


def legacy_fixture(tmp_path: Path, book_builder: Builder) -> tuple[Path, Path, Path, Path]:
    """Today's layout: flat web and mix directories with one receipt beside a collected file and
    one duplicated web document, finalized once the legacy way (it drops that duplicate)."""
    web, mix = write_mix(tmp_path, book_builder)
    first = pq.read_table(str(web / "000.parquet")).column("text").to_pylist()[0]
    pq.write_table(
        pa.Table.from_pydict({"id": ["<urn:dup>"], "text": [first]}), str(web / "001.parquet")
    )
    _ = (mix / "code" / "00000.receipt.json").write_text("{}", encoding="utf-8")
    fin = finalize(components(web, mix), tmp_path / "legacy-finalization")
    assert fin.dropped == frozenset({"<urn:dup>"})
    return mix, web, mix, fin.receipt


def test_the_plan_names_every_legacy_file_and_writes_nothing(
    tmp_path: Path, book_builder: Builder
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    planned = plan(data, web, mix, receipt)
    assert [k.kind for k in planned.kinds] == ["web", "code", "wiki", "books"]
    legacy = components(web, mix)
    for kind, component in zip(planned.kinds, legacy, strict=True):
        assert kind.share == component.share
        assert [Path(f.path) for f in kind.files] == list(component.paths)
        assert all(f.bytes == Path(f.path).stat().st_size for f in kind.files)
        assert kind.generation_parent == str(data / kind.kind)
    artifacts = [f.artifact for k in planned.kinds for f in k.files]
    assert artifacts == [None, None, "00000.receipt.json", None, None]  # the one receipt beside
    assert planned.selection == str(data / SELECTION)
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    with pytest.raises(ValueError, match="does not exist"):
        _ = plan(data, web, mix, tmp_path / "absent.json")


def test_apply_links_publishes_compares_and_selects_only_when_identical(
    tmp_path: Path, book_builder: Builder
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    legacy_files = {
        p: (p.stat().st_ino, p.read_bytes()) for c in components(web, mix) for p in c.paths
    }
    planned = plan(data, web, mix, receipt)
    migration = apply(planned, out=tmp_path / "migrated-finalization")
    assert migration.comparison.identical
    assert migration.comparison.inputs_match
    assert migration.comparison.legacy_dropped == 1
    assert migration.selection is not None
    assert [g.kind for g in migration.generations] == ["web", "code", "wiki", "books"]
    by_kind_and_name = {(p.parent.name, p.name): p for p in legacy_files}
    for generation in migration.generations:
        assert validate_generation(generation.path) == generation
        for linked in generation.inputs:
            source = by_kind_and_name[(generation.kind, linked.name)]
            assert linked.stat().st_ino == source.stat().st_ino  # one inode, no copy
    code = next(g for g in migration.generations if g.kind == "code")
    assert "00000.receipt.json" in [f.name for f in code.files]  # the receipt travelled, bound
    # the legacy files are exactly as they were, and nothing of the old layout is gone
    for path, (inode, content) in legacy_files.items():
        assert path.stat().st_ino == inode
        assert path.read_bytes() == content
    assert load_finalization(receipt).dropped == migration.finalization.dropped
    assert not (data / "staging").exists()
    # new runs now read the generations through the selection, with pins
    parts = selected_components(data)
    assert [c.name for c in parts] == ["web", "code", "wiki", "books"]
    assert all(c.generation is not None for c in parts)
    assert selection(data).sha256 == migration.selection.sha256
    for ref in selection(data).references:
        assert pinned(data, ref).path.parent == data / ref.kind
    record = cast("dict[str, object]", json.loads(migration.receipt.read_text()))
    assert record["activation"] == "selected"
    assert record["selection_sha256"] == migration.selection.sha256
    assert cast("dict[str, object]", record["comparison"])["inputs_match"] is True
    # a second migration over the same root is refused before anything is written
    with pytest.raises(ValueError, match="already holds a selection"):
        _ = plan(data, web, mix, receipt)


def test_a_legacy_finalization_over_other_inputs_is_refused_before_any_link(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's review of e1273b2: a finalization over three of the four inputs was accepted
    # when its empty drop set happened to agree; the drop set alone is not evidence of the same
    # inputs
    data, web, mix, _ = legacy_fixture(tmp_path, book_builder)
    partial = finalize(components(web, mix)[1:], tmp_path / "partial-finalization")
    planned = plan(data, web, mix, partial.receipt)
    with pytest.raises(ValueError, match="not taken over the planned web file"):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    assert not list(data.glob("*/gen-*"))  # refused before the first link
    assert not (data / SELECTION).exists()
    assert not (data / "migrations").exists()


def test_a_source_changed_after_planning_is_refused_and_nothing_is_selected(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's fixture: one unique web document changed between the plan and the links; the
    # published hash differed from the plan and the selection still succeeded
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    changed = web / "000.parquet"
    pq.write_table(
        pa.Table.from_pydict({"id": ["<urn:changed>"], "text": ["another document entirely"]}),
        str(changed),
    )
    with pytest.raises(ValueError, match="differs from the plan"):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    assert len(list((data / "web").glob("gen-*"))) == 1  # published, for inspection
    assert not (data / SELECTION).exists()
    assert not (data / "migrations").exists()
    assert not (data / "staging").exists()


def test_a_different_drop_set_over_the_same_inputs_publishes_but_selects_nothing(
    tmp_path: Path, book_builder: Builder
) -> None:
    # the legacy receipt says it dropped nothing (its dropped file replaced consistently); the
    # inputs bind, the decisions differ: nothing is selected and the receipt says refused
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    record = cast("dict[str, object]", json.loads(receipt.read_text()))
    empty = receipt.parent / "finalize-empty.parquet"
    pq.write_table(pa.Table.from_pydict({"doc_id": []}), str(empty))
    record["dropped_file"] = empty.name
    record["dropped_sha256"] = file_sha256(empty)
    record["dropped"] = 0
    _ = receipt.write_text(json.dumps(record), encoding="utf-8")
    assert load_finalization(receipt).dropped == frozenset()
    planned = plan(data, web, mix, receipt)
    migration = apply(planned, out=tmp_path / "migrated-finalization")
    assert migration.comparison.inputs_match
    assert not migration.comparison.identical
    assert (migration.comparison.only_migrated, migration.comparison.only_legacy) == (1, 0)
    assert migration.selection is None
    assert not (data / SELECTION).exists()
    assert len(migration.generations) == 4  # published, for inspection; nothing selected
    written = cast("dict[str, object]", json.loads(migration.receipt.read_text()))
    assert written["activation"] == "refused"
    assert written["selection_sha256"] is None
    with pytest.raises(ValueError, match="no selection"):
        _ = selected_components(data)


def _fail_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    """Maintainer B's fixture: the receipt write that records the activation fails; the others
    work."""
    from vesper.provenance import write_json as real_write  # noqa: PLC0415

    def flaky(path: Path, record: dict[str, object]) -> None:
        if record.get("activation") == "selected":
            msg = "injected completion receipt failure"
            raise OSError(msg)
        real_write(path, record)

    monkeypatch.setattr("vesper.migrate.write_json", flaky)


def _receipt(data: Path) -> tuple[Path, dict[str, object]]:
    (path,) = list((data / "migrations").glob("migrate-*.json"))
    return path, cast("dict[str, object]", json.loads(path.read_text()))


def test_the_receipt_is_durable_before_the_selection_is_written(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture: an OSError on the last receipt write left a live selection and no
    # record; then (review of bf92389) the state had no name and nothing could finish the record
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    _fail_completion(monkeypatch)
    with pytest.raises(
        ActivationUnrecordedError, match="committed, its receipt incomplete"
    ) as caught:
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    assert isinstance(caught.value.__cause__, OSError)
    assert (data / SELECTION).exists()  # the activation happened
    written_path, written = _receipt(data)
    assert written["activation"] == "pending"  # the intent and the evidence, recorded first
    assert written["selection_sha256"] is None
    assert len(cast("list[object]", written["generations"])) == 4
    assert cast("dict[str, object]", written["comparison"])["inputs_match"] is True
    assert caught.value.receipt == written_path
    assert caught.value.selection_sha256 == selection(data).sha256
    assert list(caught.value.generations) == written["generations"]
    assert selected_components(data)  # the selection itself is whole
    with pytest.raises(ValueError, match=re.escape(f"--recover {written_path}")):
        _ = plan(data, web, mix, receipt)  # a retry is refused and names the way out
    monkeypatch.undo()
    before = selection(data)
    before_bytes = (data / SELECTION).read_bytes()
    recovered = recover(written_path)
    assert recovered.selection_sha256 == before.sha256  # completed, not reselected
    assert (recovered.completed_now, recovered.current) == (True, True)
    assert (data / SELECTION).read_bytes() == before_bytes
    _, done = _receipt(data)
    assert done["activation"] == "selected"
    assert done["selection_sha256"] == before.sha256
    assert done["recovered_at_utc"]
    again = recover(written_path)  # idempotent: a complete receipt is left as it is
    assert again.selection_sha256 == before.sha256
    assert (again.completed_now, again.current) == (False, True)
    assert _receipt(data)[1] == done


def test_a_completed_receipt_reports_its_own_selection_not_a_later_one(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's review of e51d5f5: recover(old receipt) returned whatever selection was
    # current, as if it were that receipt's result, while the receipt recorded another hash
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    migration = apply(plan(data, web, mix, receipt), out=tmp_path / "migrated-finalization")
    assert migration.selection is not None
    ours = migration.selection
    later = select(data, ours.references[:3], replacing=ours.sha256)
    result = recover(migration.receipt)
    assert result.selection_sha256 == ours.sha256  # the receipt's own completion identity
    assert result.selection_sha256 != later.sha256
    assert (result.completed_now, result.current) == (False, False)
    assert selection(data).sha256 == later.sha256  # nothing touched


def test_recovery_refuses_a_finalization_whose_inputs_are_not_the_pinned_files(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's review of e51d5f5: one input hash corrupted in the finalization receipt, with
    # the generation labels and the drop count kept, and pending recovery marked it selected
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    _fail_completion(monkeypatch)
    with pytest.raises(ActivationUnrecordedError):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    monkeypatch.undo()
    written_path, pending = _receipt(data)
    fin_path = Path(cast("str", pending["finalization"]))
    fin = cast("dict[str, object]", json.loads(fin_path.read_text()))
    rows = cast("list[dict[str, object]]", fin["inputs"])
    original = fin_path.read_text()
    sha = rows[0]["sha256"]
    rows[0]["sha256"] = "0" * 64  # same component, path and generation label
    _ = fin_path.write_text(json.dumps(fin), encoding="utf-8")
    with pytest.raises(ValueError, match="not taken over the pinned generations' files"):
        _ = recover(written_path)
    assert _receipt(data)[1] == pending
    # a path outside the pinned generation, same hash and label, is refused the same way
    rows[0]["sha256"] = sha
    rows[0]["path"] = str(tmp_path / "elsewhere" / Path(cast("str", rows[0]["path"])).name)
    _ = fin_path.write_text(json.dumps(fin), encoding="utf-8")
    with pytest.raises(ValueError, match="not taken over the pinned generations' files"):
        _ = recover(written_path)
    _ = fin_path.write_text(original, encoding="utf-8")
    assert recover(written_path).completed_now


def test_a_receipt_artifact_changed_after_planning_is_refused_and_nothing_is_selected(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's review of bf92389: the plan named the artifact but bound no size or hash, so a
    # code receipt changed after planning was selected with the changed provenance bytes
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    artifact = mix / "code" / "00000.receipt.json"
    _ = artifact.write_text('{"policy":"original"}', encoding="utf-8")
    planned = plan(data, web, mix, receipt)
    (bound,) = [f for k in planned.kinds for f in k.files if f.artifact is not None]
    assert bound.artifact_bytes == artifact.stat().st_size
    assert bound.artifact_sha256 == file_sha256(artifact)
    _ = artifact.write_text('{"policy":"changed after plan"}', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("artifact 00000.receipt.json differs from")):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    assert not (data / SELECTION).exists()
    assert not (data / "migrations").exists()
    assert not (data / "staging").exists()


def test_recovery_refuses_an_intervening_different_selection(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    _fail_completion(monkeypatch)
    with pytest.raises(ActivationUnrecordedError):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    monkeypatch.undo()
    written_path, pending = _receipt(data)
    ours = selection(data)
    # an operator selected three kinds since: other pins, so not the migration's selection
    other = select(data, ours.references[:3], replacing=ours.sha256)
    with pytest.raises(ValueError, match="not the selection this migration wrote"):
        _ = recover(written_path)
    assert _receipt(data)[1] == pending
    assert selection(data).sha256 == other.sha256
    # the same pins chosen again on top of another selection: still not the migration's own write
    again = select(data, ours.references, replacing=other.sha256)
    with pytest.raises(ValueError, match="not the selection this migration wrote"):
        _ = recover(written_path)
    assert _receipt(data)[1] == pending
    assert selection(data).sha256 == again.sha256


def test_recovery_validates_the_receipt_before_completing_it(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    _fail_completion(monkeypatch)
    with pytest.raises(ActivationUnrecordedError):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    monkeypatch.undo()
    written_path, pending = _receipt(data)
    original = written_path.read_text()
    # a finalization over other inputs (the legacy one: no generation pins) is not this migration's
    tampered: dict[str, object] = {**pending, "finalization": str(receipt)}
    _ = written_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="not taken over the pinned generations"):
        _ = recover(written_path)
    # a pin whose manifest hash is not the published one fails validation
    pins = cast("list[str]", pending["generations"])
    bad: dict[str, object] = {**pending, "generations": [*pins[:-1], pins[-1][:-4] + "0000"]}
    _ = written_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest sha256 differs"):
        _ = recover(written_path)
    # a comparison that was not identical never activated anything
    recorded = cast("dict[str, object]", pending["comparison"])
    differing: dict[str, object] = {**pending, "comparison": {**recorded, "only_migrated": 1}}
    _ = written_path.write_text(json.dumps(differing), encoding="utf-8")
    with pytest.raises(ValueError, match="not identical"):
        _ = recover(written_path)
    _ = written_path.write_text(original, encoding="utf-8")
    assert recover(written_path).selection_sha256 == selection(data).sha256


def test_recovery_refuses_a_receipt_that_never_activated(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)

    def no_select(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "injected failure before the selection"
        raise OSError(msg)

    monkeypatch.setattr("vesper.migrate.select", no_select)
    with pytest.raises(OSError, match="before the selection"):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    monkeypatch.undo()
    written_path, pending = _receipt(data)
    assert pending["activation"] == "pending"
    with pytest.raises(ValueError, match="no selection"):
        _ = recover(written_path)
    assert _receipt(data)[1] == pending
    assert not (data / SELECTION).exists()
    refused: dict[str, object] = {**pending, "activation": "refused"}
    _ = written_path.write_text(json.dumps(refused), encoding="utf-8")
    with pytest.raises(ValueError, match="refused activation"):
        _ = recover(written_path)


def test_a_link_that_fails_publishes_nothing_for_that_kind_and_leaves_no_staging(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    planned = plan(data, web, mix, receipt)
    real_link = os.link

    def failing_link(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        if Path(src).parent.name == "wiki":
            msg = "Invalid cross-device link"
            raise OSError(18, msg)
        real_link(src, dst)

    monkeypatch.setattr(os, "link", failing_link)
    with pytest.raises(OSError, match="cross-device"):
        _ = apply(planned, out=tmp_path / "migrated-finalization")
    assert not (data / "staging").exists()
    assert not list((data / "wiki").glob("gen-*"))
    assert not (data / SELECTION).exists()
    # the kinds before it were published (additive, inspectable), the legacy files untouched
    assert len(list((data / "web").glob("gen-*"))) == 1
    assert len(list((data / "code").glob("gen-*"))) == 1
    assert (mix / "wiki" / "00000.parquet").is_file()


def test_the_command_line_prints_the_plan_then_applies(
    tmp_path: Path, book_builder: Builder, capsys: pytest.CaptureFixture[str]
) -> None:
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    arguments = [
        "--data", str(data), "--web", str(web), "--mix", str(mix),
        "--legacy-finalization", str(receipt), "--out", str(tmp_path / "out"),
    ]  # fmt: skip
    assert main(arguments) == 0
    printed = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    kinds = [cast("dict[str, object]", k)["kind"] for k in cast("list[object]", printed["kinds"])]
    assert kinds == ["web", "code", "wiki", "books"]
    assert not (data / SELECTION).exists()  # the plan alone writes nothing
    assert main([*arguments, "--apply"]) == 0
    out = capsys.readouterr().out
    plan_text, result_text = out.split("}\n{", 1)
    result = cast("dict[str, object]", json.loads("{" + result_text))
    assert isinstance(cast("object", json.loads(plan_text + "}")), dict)
    assert result["activation"] == "selected"
    assert result["selection_sha256"] == selection(data).sha256
    assert cast("dict[str, object]", result["comparison"])["only_migrated"] == 0
    assert main(arguments) == 1  # refused: the root holds a selection now


def test_the_command_line_names_the_committed_incomplete_state_and_recovers(
    tmp_path: Path,
    book_builder: Builder,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Maintainer B's review of bf92389: a generic "stopped" with exit 1 and no outcome, a pending
    # receipt with a null selection hash, and a retry refused with no supported way to finish
    data, web, mix, receipt = legacy_fixture(tmp_path, book_builder)
    arguments = [
        "--data", str(data), "--web", str(web), "--mix", str(mix),
        "--legacy-finalization", str(receipt), "--out", str(tmp_path / "out"), "--apply",
    ]  # fmt: skip
    _fail_completion(monkeypatch)
    assert main(arguments) == EXIT_COMMITTED_INCOMPLETE
    out = capsys.readouterr().out
    outcome = cast("dict[str, object]", json.loads("{" + out.split("}\n{", 1)[1]))
    written_path, _ = _receipt(data)
    assert outcome["activation"] == "committed-incomplete"
    assert outcome["selection_sha256"] == selection(data).sha256
    assert outcome["receipt"] == str(written_path)
    assert outcome["recovery"] == f"python -m vesper.migrate --recover {written_path}"
    assert "injected completion receipt failure" in cast("str", outcome["error"])
    monkeypatch.undo()
    assert main(["--recover", str(written_path)]) == 0
    recovered = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert recovered["activation"] == "selected"
    assert recovered["selection_sha256"] == selection(data).sha256
    assert _receipt(data)[1]["activation"] == "selected"
    with pytest.raises(SystemExit):
        _ = main(["--recover", str(written_path), "--apply"])  # recovery takes nothing else
    with pytest.raises(SystemExit):
        _ = main(["--data", str(data)])  # a plan needs all of its inputs
