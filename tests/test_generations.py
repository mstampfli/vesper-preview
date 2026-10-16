import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vesper.generations import (
    MANIFEST,
    MANIFEST_SCHEMA,
    SELECTION,
    Generation,
    Reference,
    Unit,
    pinned,
    publish_generation,
    reference,
    select,
    selection,
    validate_generation,
)


def stage(root: Path, name: str, units: dict[str, int]) -> tuple[Path, list[Unit]]:
    """A staging directory with one parquet and one receipt per unit, ``units`` rows each."""
    staging = root / name
    staging.mkdir()
    listed: list[Unit] = []
    for unit, rows in units.items():
        pq.write_table(
            pa.Table.from_pydict(
                {"id": [f"<urn:{unit}:{i}>" for i in range(rows)], "text": ["t"] * rows}
            ),
            str(staging / f"{unit}.parquet"),
        )
        receipt: dict[str, object] = {"unit": unit, "rows": rows}
        _ = (staging / f"{unit}.receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        listed.append(Unit(unit, (f"{unit}.parquet",), (f"{unit}.receipt.json",)))
    return staging, listed


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_a_generation_is_published_complete_and_validates_as_exactly_its_manifest(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    staging, units = stage(tmp_path, "staging", {"shard-0": 3, "shard-1": 2})
    gen = publish_generation("code", staging, data, units=units)
    assert gen.kind == "code"
    assert gen.identity.startswith("gen-")
    assert gen.path == data / "code" / gen.identity
    assert not staging.exists()  # renamed, not copied
    assert names(gen.path) == [
        MANIFEST,
        "shard-0.parquet",
        "shard-0.receipt.json",
        "shard-1.parquet",
        "shard-1.receipt.json",
    ]
    assert gen.inputs == (gen.path / "shard-0.parquet", gen.path / "shard-1.parquet")
    assert [u.name for u in gen.units] == ["shard-0", "shard-1"]
    assert gen.parent is None
    assert gen.provenance["vesper_dirty"] in {"true", "false", "unknown"}
    record = cast(
        "dict[str, object]", json.loads((gen.path / MANIFEST).read_text(encoding="utf-8"))
    )
    assert record["schema"] == MANIFEST_SCHEMA
    assert len(cast("list[object]", record["files"])) == 4
    again = validate_generation(gen.path, expected_kind="code")
    assert again == gen
    with pytest.raises(ValueError, match="expected 'wiki'"):
        _ = validate_generation(gen.path, expected_kind="wiki")


def test_a_generation_is_refused_when_its_content_differs_from_its_manifest(tmp_path: Path) -> None:
    data = tmp_path / "data"
    staging, units = stage(tmp_path, "staging", {"shard-0": 3})
    gen = publish_generation("code", staging, data, units=units)
    # an unlisted file
    extra = gen.path / "stray.txt"
    _ = extra.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not in the manifest"):
        _ = validate_generation(gen.path)
    extra.unlink()
    # a changed byte
    receipt = gen.path / "shard-0.receipt.json"
    original = receipt.read_bytes()
    _ = receipt.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="size or sha256"):
        _ = validate_generation(gen.path)
    _ = receipt.write_bytes(original)
    assert validate_generation(gen.path) == gen
    # a listed file replaced by a symlink to content elsewhere
    outside = tmp_path / "outside.parquet"
    _ = outside.write_bytes((gen.path / "shard-0.parquet").read_bytes())
    (gen.path / "shard-0.parquet").unlink()
    (gen.path / "shard-0.parquet").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        _ = validate_generation(gen.path)
    # not a generation at all
    with pytest.raises(ValueError, match="not a generation"):
        _ = validate_generation(tmp_path)


def test_publication_refuses_bad_units_and_leaves_nothing_on_failure(tmp_path: Path) -> None:
    data = tmp_path / "data"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="at least one unit"):
        _ = publish_generation("code", empty, data, units=[])
    with pytest.raises(ValueError, match="real directory"):
        _ = publish_generation("code", tmp_path / "absent", data, units=[])
    staging, units = stage(tmp_path, "staging", {"shard-0": 3})
    bad = [Unit("shard-0", ("../shard-0.parquet",), ("shard-0.receipt.json",))]
    with pytest.raises(ValueError, match="plain file name"):
        _ = publish_generation("code", staging, data, units=bad)
    missing = [Unit("shard-0", ("shard-0.parquet", "absent.parquet"), ("shard-0.receipt.json",))]
    with pytest.raises(ValueError, match="missing"):
        _ = publish_generation("code", staging, data, units=missing)
    not_parquet = [Unit("shard-0", ("shard-0.receipt.json",), ("shard-0.parquet",))]
    with pytest.raises(ValueError, match="not a parquet file"):
        _ = publish_generation("code", staging, data, units=not_parquet)
    assert not (data / "code").exists() or names(data / "code") == []  # nothing published
    assert staging.is_dir()  # the staging survives a refused publication
    with pytest.raises(ValueError, match="needs a parent"):
        _ = publish_generation("code", staging, data, units=units, replaces=["x"])


def test_an_extension_inherits_the_parent_by_hard_link_and_refuses_collisions(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    staging, units = stage(tmp_path, "first", {"shard-0": 3, "shard-1": 2})
    parent = publish_generation("code", staging, data, units=units)
    more, new_units = stage(tmp_path, "second", {"shard-2": 4})
    child = publish_generation("code", more, data, units=new_units, parent=parent)
    assert [u.name for u in child.units] == ["shard-0", "shard-1", "shard-2"]
    assert child.parent == (parent.identity, parent.manifest_sha256)
    for name in ("shard-0.parquet", "shard-1.receipt.json"):
        assert (child.path / name).stat().st_ino == (parent.path / name).stat().st_ino  # one inode
    assert validate_generation(parent.path) == parent  # the parent is untouched
    assert len(child.inputs) == 3
    # the same unit again is a collision unless replaced
    again, again_units = stage(tmp_path, "third", {"shard-1": 5})
    with pytest.raises(ValueError, match="not replaced"):
        _ = publish_generation("code", again, data, units=again_units, parent=child)
    replaced = publish_generation(
        "code", again, data, units=again_units, parent=child, replaces=["shard-1"]
    )
    assert [u.name for u in replaced.units] == ["shard-0", "shard-2", "shard-1"]
    assert pq.read_table(str(replaced.path / "shard-1.parquet")).num_rows == 5
    with pytest.raises(ValueError, match="does not have"):
        _ = publish_generation(
            "code",
            stage(tmp_path, "fourth", {"shard-9": 1})[0],
            data,
            units=[Unit("shard-9", ("shard-9.parquet",), ("shard-9.receipt.json",))],
            parent=replaced,
            replaces=["nope"],
        )
    assert isinstance(replaced, Generation)


def test_a_selection_chooses_once_pins_each_generation_and_refuses_stale_or_tampered(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    code = publish_generation(
        "code", stage(tmp_path, "c", {"shard-0": 2})[0], data, units=stage_units("shard-0")
    )
    wiki = publish_generation(
        "wiki", stage(tmp_path, "w", {"w-0": 2})[0], data, units=stage_units("w-0")
    )
    with pytest.raises(ValueError, match="no selection"):
        _ = selection(data)  # a missing selection is never a fallback to the files present
    with pytest.raises(ValueError, match="each kind once"):
        _ = select(data, [reference(code), reference(code)], replacing=None)
    first = select(data, [reference(code), reference(wiki)], replacing=None)
    assert [r.kind for r in first.references] == ["code", "wiki"]
    assert first.previous_sha256 is None
    assert selection(data) == first
    assert names(data) == [".selection.lock", SELECTION, "code", "wiki"]
    # a newer code generation; the switch records what it replaces, and a stale switch is refused
    newer = publish_generation(
        "code",
        stage(tmp_path, "c2", {"shard-1": 3})[0],
        data,
        units=stage_units("shard-1"),
        parent=code,
    )
    with pytest.raises(ValueError, match="read it again"):
        _ = select(data, [reference(newer), reference(wiki)], replacing=None)
    second = select(data, [reference(newer), reference(wiki)], replacing=first.sha256)
    assert second.previous_sha256 == first.sha256
    assert selection(data).references[0].identity == newer.identity
    with pytest.raises(ValueError, match="read it again"):
        _ = select(data, [reference(code), reference(wiki)], replacing=first.sha256)  # stale
    # a run pinned to the old generation resolves it after the selection moved on
    old = pinned(data, reference(code))
    assert old == code
    # a tampered pinned generation is refused, through the selection as well
    _ = (newer.path / "shard-1.receipt.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="size or sha256"):
        _ = pinned(data, reference(newer))
    with pytest.raises(ValueError, match="size or sha256"):
        _ = selection(data)
    # a reference to a generation that is gone, or with another recorded hash, is refused
    with pytest.raises(ValueError, match="missing"):
        _ = pinned(data, Reference("code", "gen-nowhere", "0" * 64))
    with pytest.raises(ValueError, match="differs from the pinned"):
        _ = pinned(data, Reference("wiki", wiki.identity, "0" * 64))
    with pytest.raises(ValueError, match="plain file name"):
        _ = pinned(data, Reference("wiki", "../wiki", wiki.manifest_sha256))


def stage_units(name: str) -> list[Unit]:
    return [Unit(name, (f"{name}.parquet",), (f"{name}.receipt.json",))]


def test_control_names_and_paths_are_checked_before_anything_is_written(tmp_path: Path) -> None:
    # Maintainer B's fixtures (review of 5726c87): a traversing kind, a staged manifest symlink
    # aimed at an outside sentinel, a symlinked generation root, an identity that is not the
    # directory's
    data = tmp_path / "data"
    sentinel = tmp_path / "sentinel.txt"
    _ = sentinel.write_text("untouched", encoding="utf-8")
    staging, units = stage(tmp_path, "s1", {"shard-0": 2})
    with pytest.raises(ValueError, match="plain file name"):
        _ = publish_generation("../outside-kind", staging, data, units=units)
    assert not (tmp_path / "outside-kind").exists()
    assert not data.exists()
    (staging / MANIFEST).symlink_to(sentinel)
    with pytest.raises(ValueError, match="control file is already there"):
        _ = publish_generation("code", staging, data, units=units)
    assert sentinel.read_text(encoding="utf-8") == "untouched"
    (staging / MANIFEST).unlink()
    gen = publish_generation("code", staging, data, units=units)
    # a generation root that is a symlink to a directory outside the data root
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    staging2, units2 = stage(tmp_path, "s2", {"shard-9": 1})
    other = publish_generation("code", staging2, tmp_path / "other-data", units=units2)
    link = data / "code" / "gen-linked"
    link.symlink_to(other.path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _ = pinned(data, Reference("code", "gen-linked", other.manifest_sha256))
    with pytest.raises(ValueError, match="not a real directory"):
        _ = validate_generation(link)
    link.unlink()
    # the manifest's identity must be the directory's name
    renamed = data / "code" / "gen-renamed"
    _ = gen.path.rename(renamed)
    with pytest.raises(ValueError, match="is not this directory's"):
        _ = validate_generation(renamed)
    _ = renamed.rename(gen.path)
    assert validate_generation(gen.path) == gen
    # and its kind the parent directory's name
    moved_kind = data / "wiki"
    moved_kind.mkdir()
    _ = gen.path.rename(moved_kind / gen.identity)
    with pytest.raises(ValueError, match="expected 'wiki'"):
        _ = validate_generation(moved_kind / gen.identity)


def test_a_stale_or_foreign_parent_is_refused_before_linking(tmp_path: Path) -> None:
    data = tmp_path / "data"
    staging, units = stage(tmp_path, "p", {"shard-0": 2})
    parent = publish_generation("code", staging, data, units=units)
    # the object outlives a change to its directory: the bytes on disk are not what it says
    receipt = parent.path / "shard-0.receipt.json"
    original = receipt.read_bytes()
    _ = receipt.write_bytes(original + b" ")
    child_staging, child_units = stage(tmp_path, "c", {"shard-1": 1})
    with pytest.raises(ValueError, match="size or sha256"):
        _ = publish_generation("code", child_staging, data, units=child_units, parent=parent)
    assert names(child_staging) == ["shard-1.parquet", "shard-1.receipt.json"]  # nothing linked
    _ = receipt.write_bytes(original)
    # a parent of another kind cannot be extended by this kind
    with pytest.raises(ValueError, match="cannot extend"):
        _ = publish_generation("wiki", child_staging, data, units=child_units, parent=parent)
    child = publish_generation("code", child_staging, data, units=child_units, parent=parent)
    assert [u.name for u in child.units] == ["shard-0", "shard-1"]


def test_selection_compare_and_publish_is_one_operation_under_the_writer_lock(
    tmp_path: Path,
) -> None:
    # Maintainer B's fixture (review of 5726c87): two contenders based on the same predecessor; the
    # second must fail explicitly instead of overwriting the first
    import threading  # noqa: PLC0415

    from vesper.generations import writer_lock  # noqa: PLC0415

    data = tmp_path / "data"
    code = publish_generation(
        "code", stage(tmp_path, "c", {"shard-0": 2})[0], data, units=stage_units("shard-0")
    )
    first = select(data, [reference(code)], replacing=None)
    newer = publish_generation(
        "code", stage(tmp_path, "c2", {"shard-1": 1})[0], data, units=stage_units("shard-1")
    )
    outcomes: list[str] = []
    released = threading.Event()
    holding = threading.Event()

    def holder() -> None:
        with writer_lock(data):
            holding.set()
            _ = released.wait(timeout=10)

    def contender() -> None:
        try:
            _ = select(data, [reference(newer)], replacing=first.sha256)
            outcomes.append("selected")
        except ValueError as exc:
            outcomes.append(f"refused: {exc}")

    hold = threading.Thread(target=holder)
    hold.start()
    assert holding.wait(timeout=10)
    contend = threading.Thread(target=contender)
    contend.start()
    contend.join(timeout=0.5)
    assert contend.is_alive()  # blocked on the writer lock while another writer holds it
    assert selection(data) == first  # nothing published meanwhile
    released.set()
    hold.join(timeout=10)
    contend.join(timeout=10)
    assert outcomes == ["selected"]
    second = selection(data)
    assert second.previous_sha256 == first.sha256
    # the other contender, based on the same predecessor, is refused
    with pytest.raises(ValueError, match="read it again"):
        _ = select(data, [reference(code)], replacing=first.sha256)
    assert selection(data) == second
    # the hash is of the bytes that were parsed
    import hashlib  # noqa: PLC0415

    assert second.sha256 == hashlib.sha256((data / SELECTION).read_bytes()).hexdigest()


def test_the_command_line_selects_and_shows_with_pins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from vesper.generations import main, parse_pin  # noqa: PLC0415

    data = tmp_path / "data"
    code = publish_generation(
        "code", stage(tmp_path, "c", {"shard-0": 2})[0], data, units=stage_units("shard-0")
    )
    pin = reference(code).pin
    assert parse_pin(pin) == reference(code)
    with pytest.raises(ValueError, match="kind/identity@sha256"):
        _ = parse_pin("code/gen-x")
    assert main(["show", "--data", str(data)]) == 1  # no selection yet
    assert "refused" in capsys.readouterr().err
    assert main(["select", "--data", str(data), "--ref", pin]) == 0
    shown = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert shown["references"] == [asdict_reference(reference(code))]
    assert shown["previous_sha256"] is None
    assert main(["select", "--data", str(data), "--ref", pin]) == 1  # stale: no --replacing
    assert "read it again" in capsys.readouterr().err
    assert main(["show", "--data", str(data)]) == 0
    current = cast("dict[str, object]", json.loads(capsys.readouterr().out))["sha256"]
    assert isinstance(current, str)
    assert main(["select", "--data", str(data), "--ref", pin, "--replacing", current]) == 0
    replaced = cast("dict[str, object]", json.loads(capsys.readouterr().out))
    assert replaced["previous_sha256"] == current


def asdict_reference(ref: Reference) -> dict[str, str]:
    return {"kind": ref.kind, "identity": ref.identity, "manifest_sha256": ref.manifest_sha256}
