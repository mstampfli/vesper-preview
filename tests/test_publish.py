from pathlib import Path

import pytest

from vesper.publish import staged


class BoomError(Exception):
    pass


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def write_then_fail(final: Path) -> None:
    with staged(final) as partial:
        _ = partial.write_text("half", encoding="utf-8")
        raise BoomError


def write_nothing(final: Path) -> None:
    with staged(final):
        pass


def make_directory_then_fail(final: Path) -> None:
    with staged(final) as partial:
        partial.mkdir()
        _ = (partial / "model.pt").write_bytes(b"weights")
        raise BoomError


def make_directory(final: Path, files: dict[str, bytes]) -> None:
    with staged(final) as partial:
        partial.mkdir()
        for name, content in files.items():
            _ = (partial / name).write_bytes(content)


def nested_inner_fails(final: Path) -> None:
    # the outer call has written its file when a nested call on the same destination fails
    with staged(final) as outer:
        _ = outer.write_text("outer", encoding="utf-8")
        with pytest.raises(BoomError):
            write_then_fail(final)
        assert outer.read_text(encoding="utf-8") == "outer"  # untouched by the inner cleanup


def test_a_file_appears_complete_or_not_at_all(tmp_path: Path) -> None:
    final = tmp_path / "receipt.json"
    with staged(final) as partial:
        assert partial.parent.parent == tmp_path  # staged beside the destination
        assert partial.name == final.name
        assert not partial.exists()  # nothing is there until the body writes
        _ = partial.write_text("{}", encoding="utf-8")
        assert not final.exists()  # nothing at the final name before the exit
    assert final.read_text(encoding="utf-8") == "{}"
    assert names(tmp_path) == ["receipt.json"]  # no staging left
    with pytest.raises(BoomError):
        write_then_fail(tmp_path / "other.json")
    assert names(tmp_path) == ["receipt.json"]  # neither the final nor a staging directory


def test_a_reservation_is_published_onto_and_an_empty_body_is_refused(tmp_path: Path) -> None:
    reserved = tmp_path / "finalize.json"
    reserved.touch(exist_ok=False)
    with staged(reserved) as partial:
        _ = partial.write_text("published", encoding="utf-8")
    assert reserved.read_text(encoding="utf-8") == "published"
    with pytest.raises(ValueError, match="nothing was written"):
        write_nothing(tmp_path / "never.json")
    assert names(tmp_path) == ["finalize.json"]


def test_a_directory_appears_complete_or_not_at_all(tmp_path: Path) -> None:
    final = tmp_path / "step-1"
    make_directory(final, {"model.pt": b"weights", "manifest.json": b"{}"})
    assert names(final) == ["manifest.json", "model.pt"]
    assert names(tmp_path) == ["step-1"]
    with pytest.raises(BoomError):
        make_directory_then_fail(tmp_path / "step-2")
    assert names(tmp_path) == ["step-1"]  # the staging directory and its content are gone
    # a non-empty directory at the final name is not replaced: the caller checks existence first
    with pytest.raises(OSError, match="step-1"):
        make_directory(final, {"x": b"x"})
    assert names(tmp_path) == ["step-1"]
    assert names(final) == ["manifest.json", "model.pt"]


def test_calls_on_one_destination_own_their_staging_and_leftovers_are_never_published(
    tmp_path: Path,
) -> None:
    # Maintainer B's cases (review of 444b966): a nested call's failure must not delete the outer
    # call's file, and a crashed run's leftover must not be published by a body that writes nothing
    final = tmp_path / "receipt.json"
    nested_inner_fails(final)
    assert final.read_text(encoding="utf-8") == "outer"
    assert names(tmp_path) == ["receipt.json"]
    leftover = tmp_path / ".receipt.json.crashed.staging"
    leftover.mkdir()
    _ = (leftover / "receipt.json").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="nothing was written"):
        write_nothing(final)
    assert final.read_text(encoding="utf-8") == "outer"  # the stale content did not land
    assert (leftover / "receipt.json").read_text(encoding="utf-8") == "stale"  # not ours, untouched
    # two concurrent calls on one destination: each publishes its own content, the later wins
    with staged(final) as first, staged(final) as second:
        _ = first.write_text("first", encoding="utf-8")
        _ = second.write_text("second", encoding="utf-8")
        assert first != second
    assert final.read_text(encoding="utf-8") == "first"  # exits run inner first: second, then first
    assert names(tmp_path) == [".receipt.json.crashed.staging", "receipt.json"]


def test_owned_staging_is_exclusive_and_releases_without_masking(tmp_path: Path) -> None:
    # Maintainer B's review of e39edd3: a label-plus-clock name gave two nested operations one
    # directory (the inner exit deleted the outer one's files), and a parent vanishing between two
    # checks raised over the body's own exception
    import shutil  # noqa: PLC0415

    from vesper.publish import owned_staging  # noqa: PLC0415

    with owned_staging(tmp_path, "x") as outer:
        _ = (outer / "sentinel").write_text("keep", encoding="utf-8")
        with owned_staging(tmp_path, "x") as inner:
            assert inner != outer
            assert inner.parent == outer.parent == tmp_path / "staging"
            assert inner.name.startswith("x-")
        assert (outer / "sentinel").read_text(encoding="utf-8") == "keep"
        assert outer.stat().st_mode & 0o044  # readable by the readers a generation will have
    assert not (tmp_path / "staging").exists()

    # the body's exception is the one that propagates, even when the parent vanished underneath
    def vanish_then_fail() -> None:
        with owned_staging(tmp_path, "y"):
            shutil.rmtree(tmp_path / "staging")
            raise BoomError

    with pytest.raises(BoomError):
        vanish_then_fail()
    # a neighbour's staging is never touched; a staging renamed away leaves only the parent tidy
    neighbour = tmp_path / "staging" / "another-operation"
    neighbour.mkdir(parents=True)
    _ = (neighbour / "keep").write_text("x", encoding="utf-8")
    with owned_staging(tmp_path, "z") as own:
        _ = (own / "file").write_text("x", encoding="utf-8")
        _ = own.rename(tmp_path / "published")  # the caller published it
    assert (tmp_path / "published" / "file").is_file()
    assert (neighbour / "keep").is_file()
