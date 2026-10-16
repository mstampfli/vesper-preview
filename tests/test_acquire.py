import json
import platform
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vesper.acquire import acquire_books, acquire_shards, ensure_free, main
from vesper.books import Edition, TitleListing
from vesper.collect import CODE_REPO, CODE_REVISION, WIKI_REPO, WIKI_REVISION
from vesper.generations import validate_generation
from vesper.provenance import environment

Builder = Callable[[Path, str, dict[str, str]], Path]


def load(path: Path) -> dict[str, object]:
    return cast("dict[str, object]", json.loads(path.read_text()))


def ids_of(path: Path) -> list[object]:
    return pq.ParquetFile(str(path)).read_row_group(0).column("id").to_pylist()


def test_environment_names_the_interpreter_the_libraries_and_the_commit() -> None:
    env = environment()
    assert env["python"] == platform.python_version()
    assert set(env) >= {"vesper", "pyarrow", "huggingface-hub", "defusedxml", "requests"}
    assert len(env["vesper_commit"]) in {40, len("unknown")}
    assert env["vesper_dirty"] in {"true", "false", "unknown"}
    assert (env["vesper_dirty"] == "unknown") == (env["vesper_commit"] == "unknown")


def test_ensure_free_reports_and_refuses(tmp_path: Path) -> None:
    assert ensure_free(tmp_path, 0.0) > 0.0
    with pytest.raises(RuntimeError, match="below the floor"):
        _ = ensure_free(tmp_path, 1e9)


def test_acquire_shards_collects_each_shard_with_a_receipt_and_releases_staging(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str, str]] = []
    stagings: list[Path] = []

    def fetch(repo: str, revision: str, path: str, staging: Path) -> Path:
        calls.append((repo, revision, path))
        stagings.append(staging)
        staging.mkdir(parents=True, exist_ok=True)
        shard = staging / Path(path).name
        if repo == CODE_REPO:
            rows: dict[str, list[str]] = {
                "repo_name": ["o/a", "o/b"],
                "path": ["x.py", "y.c"],
                "content": ["print(1)\n", "int x;\n"],
                "license": ["mit", "gpl-3.0"],
            }
        else:
            rows = {
                "id": ["7"],
                "url": ["https://en.wikipedia.org/wiki/Q"],
                "title": ["Q"],
                "text": ["Q is."],
            }
        pq.write_table(pa.Table.from_pydict(rows), str(shard))
        return shard

    data = tmp_path / "mix"
    receipts = list(
        acquire_shards("code", [0, 5], data, floor_gb=0.0, fetch=fetch, layout="legacy").receipts
    )
    assert calls == [
        (CODE_REPO, CODE_REVISION, "data/train-00000-of-01126.parquet"),
        (CODE_REPO, CODE_REVISION, "data/train-00005-of-01126.parquet"),
    ]
    # one staging directory per run, under data/staging, so concurrent runs never share one
    assert len(set(stagings)) == 1
    assert stagings[0].parent == data / "staging"
    assert stagings[0].name.startswith("code-")
    assert receipts == [
        data / "code" / "train-00000-of-01126.receipt.json",
        data / "code" / "train-00005-of-01126.receipt.json",
    ]
    assert not (data / "staging").exists()
    record = load(receipts[1])
    assert (record["rows_read"], record["rows_kept"]) == (2, 1)
    assert record["rejected"] == {"gpl-3.0": 1}
    assert cast("int", record["input_bytes"]) > 0
    assert record["source_path"] == "data/train-00005-of-01126.parquet"
    assert cast("dict[str, str]", record["environment"])["python"] == platform.python_version()
    assert set(record) >= {"collected_at_utc", "seconds", "input_sha256", "output_sha256", "rule"}
    assert ids_of(data / "code" / "train-00005-of-01126.parquet") == [
        f"code:o/a/x.py@{CODE_REVISION}"
    ]
    wiki = acquire_shards("wiki", [40], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    assert calls[-1] == (WIKI_REPO, WIKI_REVISION, "20231101.en/train-00040-of-00041.parquet")
    assert wiki.receipts == (data / "wiki" / "train-00040-of-00041.receipt.json",)
    assert wiki.generation is None
    with pytest.raises(ValueError, match="outside"):
        _ = acquire_shards("wiki", [41], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    with pytest.raises(ValueError, match="kind"):
        _ = acquire_shards("books", [0], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    with pytest.raises(RuntimeError, match="floor"):
        _ = acquire_shards("code", [1], data, floor_gb=1e9, fetch=fetch, layout="legacy")


def test_acquire_shards_publishes_a_generation_by_default_and_extends_a_parent(
    tmp_path: Path,
) -> None:
    def fetch(repo: str, revision: str, path: str, staging: Path) -> Path:
        del repo, revision
        staging.mkdir(parents=True, exist_ok=True)
        shard = staging / Path(path).name
        pq.write_table(
            pa.Table.from_pydict(
                {
                    "repo_name": ["o/a"],
                    "path": [f"{Path(path).stem}.py"],
                    "content": ["print(1)\n"],
                    "license": ["mit"],
                }
            ),
            str(shard),
        )
        return shard

    data = tmp_path / "data"
    first = acquire_shards("code", [0, 5], data, floor_gb=0.0, fetch=fetch)
    generation = first.generation
    assert generation is not None
    assert generation.kind == "code"
    assert generation.path.parent == data / "code"
    assert [u.name for u in generation.units] == ["train-00000-of-01126", "train-00005-of-01126"]
    assert [p.name for p in generation.inputs] == [
        "train-00000-of-01126.parquet",
        "train-00005-of-01126.parquet",
    ]
    assert first.receipts == (
        generation.path / "train-00000-of-01126.receipt.json",
        generation.path / "train-00005-of-01126.receipt.json",
    )
    assert all(r.is_file() for r in first.receipts)
    assert not (data / "staging").exists()  # both stagings released or renamed away
    assert sorted(p.name for p in (data / "code").iterdir()) == [generation.identity]
    assert validate_generation(generation.path) == generation
    # a second run extends the first: the parent's units come first, by hard link
    second = acquire_shards("code", [7], data, floor_gb=0.0, fetch=fetch, parent=generation)
    child = second.generation
    assert child is not None
    assert [u.name for u in child.units] == [
        "train-00000-of-01126",
        "train-00005-of-01126",
        "train-00007-of-01126",
    ]
    assert (child.path / "train-00000-of-01126.parquet").stat().st_ino == (
        generation.path / "train-00000-of-01126.parquet"
    ).stat().st_ino
    assert child.parent == (generation.identity, generation.manifest_sha256)
    # the same shard again is a collision, and nothing half-collected is left behind
    with pytest.raises(ValueError, match="not replaced"):
        _ = acquire_shards("code", [5], data, floor_gb=0.0, fetch=fetch, parent=child)
    assert sorted(p.name for p in (data / "code").iterdir()) == sorted(
        [generation.identity, child.identity]
    )
    assert not (data / "staging").exists()
    with pytest.raises(ValueError, match="legacy layout"):
        _ = acquire_shards(
            "code", [1], data, floor_gb=0.0, fetch=fetch, layout="legacy", parent=child
        )


def test_acquire_books_lists_selects_fetches_and_writes_the_manifest(
    tmp_path: Path, book_builder: Builder
) -> None:
    fetched: list[str] = []

    def listing() -> TitleListing:
        return TitleListing(("b_two", "a_one", "c_three"), ("tools",))

    def fetch(slug: str, staging: Path) -> Edition:
        fetched.append(slug)
        language = "fr" if slug == "c_three" else "en"
        repo = book_builder(staging / slug, language, {"chapter-1.xhtml": f"<p>{slug}</p>"})
        return Edition(slug, repo, f"commit-{slug}")

    data = tmp_path / "mix"
    books = acquire_books(2, data, floor_gb=0.0, listing=listing, fetch=fetch, layout="legacy")
    (receipt_path,) = books.receipts
    assert books.generation is None
    assert len(fetched) == 2
    assert not (data / "staging").exists()
    record = load(receipt_path)
    listings = sorted((data / "books").glob("listing-*.json"))
    assert len(listings) == 1
    assert load(listings[0])["skipped"] == ["tools"]
    assert record["listing"] == {"file": listings[0].name, "titles": 3, "skipped": ["tools"]}
    assert cast("dict[str, object]", record["selection"])["count"] == 2
    assert record["rows_read"] == 2
    titles = cast("list[dict[str, object]]", record["titles"])
    assert [t["slug"] for t in titles] == fetched
    kept = [t for t in titles if t["status"] == "kept"]
    rejected = cast("dict[str, int]", record["rejected"])
    assert len(kept) + rejected.get("language", 0) == 2
    out = Path(cast("str", record["output"]))
    assert out.parent == data / "books"
    assert ids_of(out) == [f"book:{t['slug']}@commit-{t['slug']}" for t in kept]
    # the generation layout: one unit, the parquet its input, receipt and listing its artifacts;
    # a larger selection replaces the parent's titles instead of repeating them
    gen_data = tmp_path / "gen"
    first = acquire_books(2, gen_data, floor_gb=0.0, listing=listing, fetch=fetch)
    generation = first.generation
    assert generation is not None
    assert generation.kind == "books"
    assert len(generation.units) == 1
    unit = generation.units[0]
    assert unit.name.startswith("titles-")
    assert unit.inputs == (f"{unit.name}.parquet",)
    assert sorted(unit.artifacts) == sorted(
        [f"{unit.name}.receipt.json", unit.name.replace("titles-", "listing-") + ".json"]
    )
    assert first.receipts == (generation.path / f"{unit.name}.receipt.json",)
    assert not (gen_data / "staging").exists()
    bigger = acquire_books(
        3, gen_data, floor_gb=0.0, listing=listing, fetch=fetch, parent=generation
    )
    assert bigger.generation is not None
    assert len(bigger.generation.units) == 1  # replaced, not inherited beside the old titles
    assert bigger.generation.parent == (generation.identity, generation.manifest_sha256)


def test_main_parses_and_reports_a_stop(tmp_path: Path) -> None:
    # a floor no disk meets stops before any network call; the exit code says so
    assert main(["--data", str(tmp_path), "--floor-gb", "1e9", "code", "--shards", "0"]) == 1
    with pytest.raises(SystemExit):
        _ = main(["nothing"])


def test_two_runs_have_their_own_staging_and_one_release_leaves_the_other(tmp_path: Path) -> None:
    # 2026-10-15: a shard run released data/staging while a books run was cloning into it
    seen: list[Path] = []

    def fetch(repo: str, revision: str, path: str, staging: Path) -> Path:
        del repo, revision
        seen.append(staging)
        staging.mkdir(parents=True, exist_ok=True)
        other = tmp_path / "mix" / "staging" / "books-elsewhere"
        other.mkdir(parents=True, exist_ok=True)
        _ = (other / "clone-in-progress").write_text("x")
        shard = staging / Path(path).name
        pq.write_table(
            pa.Table.from_pydict(
                {"repo_name": ["o/a"], "path": ["x.py"], "content": ["c"], "license": ["mit"]}
            ),
            str(shard),
        )
        return shard

    data = tmp_path / "mix"
    _ = acquire_shards("code", [0], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    assert not seen[0].exists()  # this run's staging released
    assert (
        data / "staging" / "books-elsewhere" / "clone-in-progress"
    ).is_file()  # the other's kept


def test_a_failed_fetch_releases_the_runs_staging(tmp_path: Path) -> None:
    # Maintainer B's fixture (collector review, third round): acquire_shards left staging behind
    # when a fetch failed
    seen: list[Path] = []

    def fetch(repo: str, revision: str, path: str, staging: Path) -> Path:
        del repo, revision, path
        seen.append(staging)
        staging.mkdir(parents=True, exist_ok=True)
        _ = (staging / "half-downloaded").write_text("x")
        msg = "connection reset"
        raise RuntimeError(msg)

    data = tmp_path / "mix"
    with pytest.raises(RuntimeError, match="connection reset"):
        _ = acquire_shards("code", [0], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    assert not seen[0].exists()
    assert not (data / "staging").exists()


def test_each_title_is_released_before_the_next_is_fetched(
    tmp_path: Path, book_builder: Builder
) -> None:
    repos: list[Path] = []

    def listing() -> TitleListing:
        return TitleListing(("a_one", "b_two", "c_three"), ())

    def fetch(slug: str, staging: Path) -> Edition:
        # the previous title's clone is gone by the time the next one is fetched
        assert all(not r.exists() for r in repos)
        repo = book_builder(staging / slug, "en", {"chapter-1.xhtml": f"<p>{slug}</p>"})
        repos.append(repo)
        return Edition(slug, repo, f"commit-{slug}")

    data = tmp_path / "mix"
    _ = acquire_books(3, data, floor_gb=0.0, listing=listing, fetch=fetch, layout="legacy")
    assert len(repos) == 3
    assert not (data / "staging").exists()


def test_a_release_tolerates_a_parent_refilled_between_check_and_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # two acquisitions run at once: the other run creates its staging after this run has seen
    # the parent empty and before it removes it; the removal fails and that is not an error
    import errno  # noqa: PLC0415

    def fetch(_repo: str, _revision: str, path: str, staging: Path) -> Path:
        staging.mkdir(parents=True, exist_ok=True)
        shard = staging / Path(path).name
        pq.write_table(
            pa.Table.from_pydict(
                {
                    "id": ["1"],
                    "url": ["https://en.wikipedia.org/wiki/A"],
                    "title": ["A"],
                    "text": ["A."],
                }
            ),
            str(shard),
        )
        return shard

    def refilled(self: Path) -> None:
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(self))

    monkeypatch.setattr(Path, "rmdir", refilled)
    data = tmp_path / "mix"
    result = acquire_shards("wiki", [0], data, floor_gb=0.0, fetch=fetch, layout="legacy")
    assert [r.name for r in result.receipts] == ["train-00000-of-00041.receipt.json"]
    assert all(r.is_file() for r in result.receipts)
    assert (data / "staging").is_dir()  # left to the run that refilled it


def test_a_run_owns_its_staging_through_every_failure_window(  # noqa: C901, PLR0915
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixtures (review of f55bfb3): a listing failure left an empty books-publish
    # staging, a receipt-write failure left the listing and a 3,617-byte parquet, no generation
    # published
    import vesper.acquire as acquire_module  # noqa: PLC0415
    from vesper.generations import publish_generation  # noqa: PLC0415

    def listing() -> TitleListing:
        return TitleListing(("a_one", "b_two"), ())

    def fetch_book(slug: str, staging: Path) -> Edition:
        repo = book_builder(staging / slug, "en", {"chapter-1.xhtml": f"<p>{slug}</p>"})
        return Edition(slug, repo, f"commit-{slug}")

    def fetch_shard(_repo: str, _revision: str, path: str, staging: Path) -> Path:
        staging.mkdir(parents=True, exist_ok=True)
        shard = staging / Path(path).name
        pq.write_table(
            pa.Table.from_pydict(
                {
                    "id": ["1"],
                    "url": ["https://en.wikipedia.org/wiki/A"],
                    "title": ["A"],
                    "text": ["A."],
                }
            ),
            str(shard),
        )
        return shard

    data = tmp_path / "mix"
    other = (
        data / "staging" / "another-operation"
    )  # a concurrent run's staging, never ours to touch
    other.mkdir(parents=True)
    _ = (other / "keep").write_text("x", encoding="utf-8")

    def nothing_of_ours_left() -> None:
        assert sorted(p.name for p in (data / "staging").iterdir()) == ["another-operation"]
        assert (other / "keep").is_file()
        assert not list(data.glob("*/gen-*"))

    def books() -> None:
        _ = acquire_books(1, data, floor_gb=0.0, listing=listing, fetch=fetch_book)

    # the listing itself fails
    def failing_listing() -> TitleListing:
        msg = "injected listing failure"
        raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="listing failure"):
        _ = acquire_books(1, data, floor_gb=0.0, listing=failing_listing, fetch=fetch_book)
    nothing_of_ours_left()
    # the listing write, then the receipt write, fail
    from vesper.provenance import write_json as real_write  # noqa: PLC0415

    def failing_write(suffix: str) -> Callable[[Path, dict[str, object]], None]:
        def write(path: Path, record: dict[str, object]) -> None:
            if path.name.endswith(suffix):
                msg = f"injected write failure for {suffix}"
                raise OSError(msg)
            real_write(path, record)

        return write

    monkeypatch.setattr(acquire_module, "write_json", failing_write("listing-"[:0] + ".json"))
    with pytest.raises(OSError, match="write failure"):
        books()
    nothing_of_ours_left()
    monkeypatch.setattr(acquire_module, "write_json", failing_write(".receipt.json"))
    with pytest.raises(OSError, match="receipt"):
        books()
    nothing_of_ours_left()
    with pytest.raises(OSError, match="receipt"):
        _ = acquire_shards("wiki", [0], data, floor_gb=0.0, fetch=fetch_shard)
    nothing_of_ours_left()
    monkeypatch.setattr(acquire_module, "write_json", real_write)

    # the provenance fails after the collection
    def failing_environment() -> dict[str, str]:
        msg = "injected provenance failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(acquire_module, "environment", failing_environment)
    with pytest.raises(RuntimeError, match="provenance"):
        books()
    nothing_of_ours_left()
    monkeypatch.setattr(acquire_module, "environment", environment)

    # the publication fails
    def failing_publish(*_args: object, **_kwargs: object) -> object:
        msg = "injected publication failure"
        raise ValueError(msg)

    monkeypatch.setattr(acquire_module, "publish_generation", failing_publish)
    with pytest.raises(ValueError, match="publication"):
        books()
    nothing_of_ours_left()
    monkeypatch.setattr(acquire_module, "publish_generation", publish_generation)
    # a success leaves nothing of ours behind either, and the other operation's staging stands
    result = acquire_books(1, data, floor_gb=0.0, listing=listing, fetch=fetch_book)
    assert result.generation is not None
    assert sorted(p.name for p in (data / "staging").iterdir()) == ["another-operation"]
    assert (other / "keep").is_file()
    # the allocation fails: the data root is a file; the exception is the allocation's own
    blocked = tmp_path / "a-file"
    _ = blocked.write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _ = acquire_books(1, blocked, floor_gb=0.0, listing=listing, fetch=fetch_book)
    assert blocked.read_text(encoding="utf-8") == ""
