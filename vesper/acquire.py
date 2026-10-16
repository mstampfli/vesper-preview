"""Acquire the non-web components of the mix, one unit at a time, each with a receipt.

The mixture document's acquisition and receipt contracts
(docs/reviews/2026-10-15-mixture-sources.md): one compressed shard, or one title, at a time;
selected rows written to compact parquet with provenance columns; staging released before the
next unit; free disk checked before each; nothing downloaded whole. Every output has a receipt
beside it: the collector's totals, when, how long, the input's bytes, the interpreter and library
versions and the vesper commit; for books also the title universe, the selection rule and the
per-title manifest.

    python -m vesper.acquire code --shards 1 2 3
    python -m vesper.acquire wiki --shards 0
    python -m vesper.acquire books --count 300
    (--data DIR, default ~/data/mix; --floor-gb 40)

The network functions are parameters, so the unit tests run the driver on local files.
"""

import argparse
import contextlib
import logging
import shutil
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from vesper.books import Edition, TitleListing, fetch_title, list_titles, select_titles
from vesper.collect import (
    CODE_REPO,
    CODE_REVISION,
    WIKI_REPO,
    WIKI_REVISION,
    BooksReceipt,
    CollectReceipt,
    TitleOutcome,
    collect_books,
    collect_code,
    collect_wikipedia,
    fetch_shard,
)
from vesper.generations import Generation, Unit, publish_generation, reference, validate_generation
from vesper.provenance import environment, write_json
from vesper.publish import owned_staging

FREE_FLOOR_GB = 40.0
DEFAULT_COUNT = 300
_SHARDS = {
    "code": (CODE_REPO, CODE_REVISION, "data/train-{index:05d}-of-01126.parquet", 1126),
    "wiki": (WIKI_REPO, WIKI_REVISION, "20231101.en/train-{index:05d}-of-00041.parquet", 41),
}
log = logging.getLogger("vesper.acquire")


Layout = Literal["generation", "legacy"]


@dataclass(frozen=True)
class AcquireResult:
    """What a driver run produced: the receipt paths, and the generation when one was published."""

    receipts: tuple[Path, ...]
    generation: Generation | None


def ensure_free(path: Path, floor_gb: float) -> float:
    """The free space under ``path`` in GB; a floor below it stops the run before a download.

    Raises:
        RuntimeError: when free space is below ``floor_gb``.
    """
    free = shutil.disk_usage(path).free / 1e9
    if free < floor_gb:
        msg = f"{free:.1f} GB free under {path}, below the floor of {floor_gb:.1f} GB"
        raise RuntimeError(msg)
    return free


def _plain(value: CollectReceipt | TitleOutcome) -> dict[str, object]:
    return cast("dict[str, object]", asdict(value))


def _empty(staging: Path) -> None:
    """Remove this run's downloads from its staging so the next shard's have the space.

    The directory stays this run's own (``publish.owned_staging`` releases it at the end).
    """
    for child in staging.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def acquire_shards(
    kind: str,
    indices: Sequence[int],
    data: Path,
    *,
    floor_gb: float = FREE_FLOOR_GB,
    fetch: Callable[[str, str, str, Path], Path] = fetch_shard,
    layout: Layout = "generation",
    parent: Generation | None = None,
) -> AcquireResult:
    """Fetch and collect the numbered shards of ``code`` or ``wiki`` one at a time.

    Under the ``generation`` layout (the default) every shard's parquet and receipt are collected
    into this run's own output staging and published together as one generation,
    ``data/<kind>/gen-<stamp>`` (``generations.publish_generation``; with ``parent`` its units
    are inherited first); a failure publishes nothing. Under the explicit ``legacy`` layout each
    shard lands as ``data/<kind>/<shard stem>.parquet`` with its receipt beside it. The download
    staging is released after every shard. Returns the receipt paths and the generation.

    Raises:
        ValueError: on an unknown kind, an index outside the dataset, or a parent the generation
            cannot extend.
        RuntimeError: when free space is below the floor (nothing more is fetched).
    """
    if kind not in _SHARDS:
        msg = f"kind must be one of {sorted(_SHARDS)}, got {kind!r}"
        raise ValueError(msg)
    repo, revision, pattern, total = _SHARDS[kind]
    for index in indices:
        if not 0 <= index < total:
            msg = f"{kind} shard index {index} outside [0, {total})"
            raise ValueError(msg)
    if layout == "legacy" and parent is not None:
        msg = "a parent generation has no meaning under the legacy layout"
        raise ValueError(msg)
    collect = collect_code if kind == "code" else collect_wikipedia
    data.mkdir(parents=True, exist_ok=True)
    receipts: list[Path] = []
    units: list[Unit] = []
    with owned_staging(data, kind) as staging, _outputs(data, kind, layout) as outputs:
        for index in indices:
            free = ensure_free(data, floor_gb)
            path = pattern.format(index=index)
            started = time.monotonic()
            shard = fetch(repo, revision, path, staging)
            size = shard.stat().st_size
            log.info("%s: fetched %s (%.0f MB) with %.1f GB free", kind, path, size / 1e6, free)
            receipt_path, unit = _collect_shard(kind, path, shard, outputs, collect, started)
            receipts.append(receipt_path)
            units.append(unit)
            _empty(staging)  # the download goes before the next shard's
        if layout == "legacy":
            return AcquireResult(receipts=tuple(receipts), generation=None)
        generation = publish_generation(kind, outputs, data, units=units, parent=parent)
    log.info("%s: published %s (%s)", kind, generation.identity, generation.manifest_sha256[:12])
    return AcquireResult(
        receipts=tuple(generation.path / r.name for r in receipts), generation=generation
    )


def _outputs(data: Path, kind: str, layout: Layout) -> contextlib.AbstractContextManager[Path]:
    """Where a run's outputs go.

    Its own output staging under the generation layout (``publish.owned_staging``: allocated
    exclusively, released with the operation however it ends, renamed away by a publication;
    maintainer B's reviews of f55bfb3 and e39edd3); the flat ``data/<kind>`` directory under the
    explicit legacy layout (the real destination, never released).
    """
    if layout == "generation":
        return owned_staging(data, f"{kind}-publish")
    directory = data / kind
    directory.mkdir(parents=True, exist_ok=True)
    return contextlib.nullcontext(directory)


def _collect_shard(
    kind: str,
    path: str,
    shard: Path,
    outputs: Path,
    collect: Callable[[Path, Path], CollectReceipt],
    started: float,
) -> tuple[Path, Unit]:
    """Collect one fetched shard into ``outputs`` and write its receipt beside it.

    Returns the receipt path and the shard's unit (the parquet its input, the receipt its
    artifact).
    """
    stem = Path(path).stem
    out = outputs / f"{stem}.parquet"
    receipt = collect(shard, out)
    record: dict[str, object] = _plain(receipt)
    record["collected_at_utc"] = datetime.now(UTC).isoformat()
    record["seconds"] = round(time.monotonic() - started, 1)
    record["source_path"] = path
    record["input_bytes"] = shard.stat().st_size
    record["environment"] = environment()
    receipt_path = out.with_suffix(".receipt.json")
    write_json(receipt_path, record)
    log.info(
        "%s: read %d kept %d (%.1f MB text) rejected %s -> %s",
        kind,
        receipt.rows_read,
        receipt.rows_kept,
        receipt.bytes_kept / 1e6,
        receipt.rejected,
        out,
    )
    return receipt_path, Unit(stem, (f"{stem}.parquet",), (f"{stem}.receipt.json",))


def acquire_books(
    count: int,
    data: Path,
    *,
    floor_gb: float = FREE_FLOOR_GB,
    listing: Callable[[], TitleListing] = list_titles,
    fetch: Callable[[str, Path], Edition] = fetch_title,
    layout: Layout = "generation",
    parent: Generation | None = None,
) -> AcquireResult:
    """List the title universe, select ``count`` titles, fetch and collect them one at a time.

    Writes ``listing-<stamp>.json`` (the universe and the skipped names), ``titles-<stamp>.parquet``
    and its receipt with the selection rule and the per-title manifest: under the ``generation``
    layout (the default) into this run's output staging, published together as one generation
    with the unit ``titles-<stamp>`` (the parquet its input, receipt and listing its artifacts; a
    ``parent`` books generation is replaced, not inherited, since a larger selection repeats its
    titles); under the explicit ``legacy`` layout into ``data/books/``. The clone staging is
    released at the end. Returns the receipt path and the generation.

    Raises:
        RuntimeError: when free space is below the floor or a fetch fails; a title that fails to
            read is counted in the receipt, not raised.
        ValueError: on a parent under the legacy layout.
    """
    if layout == "legacy" and parent is not None:
        msg = "a parent generation has no meaning under the legacy layout"
        raise ValueError(msg)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    data.mkdir(parents=True, exist_ok=True)
    with owned_staging(data, "books") as staging, _outputs(data, "books", layout) as outputs:
        universe = listing()
        write_json(
            outputs / f"listing-{stamp}.json",
            {
                "fetched_at_utc": stamp,
                "titles": list(universe.titles),
                "skipped": list(universe.skipped),
            },
        )
        selected = select_titles(universe.titles, count)
        log.info(
            "books: %d titles in the universe, %d selected", len(universe.titles), len(selected)
        )
        started = time.monotonic()

        def editions() -> Iterator[Edition]:
            for slug in selected:
                _ = ensure_free(data, floor_gb)
                edition = fetch(slug, staging)
                log.info("books: fetched %s at %s", slug, edition.commit[:12])
                yield edition
                # the consumer has read this title before asking for the next: release its clone
                shutil.rmtree(edition.repo, ignore_errors=True)

        out = outputs / f"titles-{stamp}.parquet"
        result = collect_books(editions(), out, f"listing-{stamp}")
        receipt_path = out.with_suffix(".receipt.json")
        write_json(
            receipt_path, _books_record(result, universe, count, len(selected), stamp, started)
        )
        log.info(
            "books: read %d kept %d (%.1f MB text) rejected %s -> %s",
            result.receipt.rows_read,
            result.receipt.rows_kept,
            result.receipt.bytes_kept / 1e6,
            result.receipt.rejected,
            out,
        )
        if layout == "legacy":
            return AcquireResult(receipts=(receipt_path,), generation=None)
        unit = Unit(
            f"titles-{stamp}",
            (f"titles-{stamp}.parquet",),
            (f"titles-{stamp}.receipt.json", f"listing-{stamp}.json"),
        )
        replaces = [u.name for u in parent.units] if parent is not None else []
        generation = publish_generation(
            "books", outputs, data, units=[unit], parent=parent, replaces=replaces
        )
    log.info("books: published %s (%s)", generation.identity, generation.manifest_sha256[:12])
    return AcquireResult(receipts=(generation.path / receipt_path.name,), generation=generation)


def _books_record(
    result: BooksReceipt,
    universe: TitleListing,
    count: int,
    selected: int,
    stamp: str,
    started: float,
) -> dict[str, object]:
    """The books receipt.

    The collector's receipt plus when, how long, the environment, the listing file, the selection
    rule and the per-title manifest.
    """
    record: dict[str, object] = _plain(result.receipt)
    record["collected_at_utc"] = datetime.now(UTC).isoformat()
    record["seconds"] = round(time.monotonic() - started, 1)
    record["environment"] = environment()
    record["listing"] = {
        "file": f"listing-{stamp}.json",
        "titles": len(universe.titles),
        "skipped": list(universe.skipped),
    }
    record["selection"] = {
        "rule": "the first count titles in blake2b order of the slug (vesper.books.select_titles)",
        "count": count,
        "selected": selected,
    }
    record["titles"] = [_plain(outcome) for outcome in result.titles]
    return record


def _parser() -> argparse.ArgumentParser:
    summary = (__doc__ or "").split("\n\n")[0]
    parser = argparse.ArgumentParser(prog="vesper.acquire", description=summary)
    _ = parser.add_argument("--data", type=Path, default=Path.home() / "data" / "mix")
    _ = parser.add_argument("--floor-gb", type=float, default=FREE_FLOOR_GB)
    _ = parser.add_argument(
        "--legacy-layout",
        action="store_true",
        help="write today's flat data/<kind>/ files instead of publishing a generation",
    )
    _ = parser.add_argument(
        "--parent",
        default=None,
        help="the generation (gen-<stamp>) under data/<kind>/ this one extends or replaces",
    )
    sub = parser.add_subparsers(dest="component", required=True)
    for kind in _SHARDS:
        shards = sub.add_parser(kind, help=f"{kind} shards by index")
        _ = shards.add_argument("--shards", type=int, nargs="+", required=True)
    books = sub.add_parser("books", help="Standard Ebooks titles")
    _ = books.add_argument("--count", type=int, default=DEFAULT_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: returns 0 when every requested unit was collected, 1 otherwise."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    component = cast("str", args.component)
    data = cast("Path", args.data)
    floor_gb = cast("float", args.floor_gb)
    layout: Layout = "legacy" if cast("bool", args.legacy_layout) else "generation"
    try:
        parent_name = cast("str | None", args.parent)
        parent = (
            None
            if parent_name is None
            else validate_generation(data / component / parent_name, expected_kind=component)
        )
        if component == "books":
            result = acquire_books(
                cast("int", args.count), data, floor_gb=floor_gb, layout=layout, parent=parent
            )
        else:
            result = acquire_shards(
                component,
                cast("list[int]", args.shards),
                data,
                floor_gb=floor_gb,
                layout=layout,
                parent=parent,
            )
    except (RuntimeError, ValueError):
        log.exception("stopped")
        return 1
    if result.generation is not None:
        log.info(
            "published %s; it is not selected until `python -m vesper.generations select` names it",
            reference(result.generation).pin,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
