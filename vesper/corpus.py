"""Corpus reader (spec, Data): parquet files to whole documents, finalized, shuffled, split.

The pretraining corpus is FineWeb-Edu's ``sample/10BT`` as parquet files with ``id`` and ``text``
columns, 1,000 documents per row group, plus the collected components (``vesper.collect``), whose
files carry ``group_key``, ``dedup_key`` and ``dedup_version`` as well. Documents are shuffled
relative to each other by a block shuffle keyed by the seed: a permutation of the ``(file, row
group)`` pairs and a permutation of the rows inside each group. Its limits: the rows of one group
stay contiguous in the output, so two documents of one group are never further apart than the
group size and documents of different groups interleave only at group granularity; inside a
document nothing moves. Files are opened for one read and closed again, so the number of open
descriptors does not grow with the file count.

Finalization (docs/reviews/2026-10-15-mixture-sources.md, Grouping and Deduplication) is decided
once, against the frozen set of input files and independently of any shuffle: ``finalize`` reads
every document of every component, finds the dedup keys that occur more than once, keeps for each
the document with the smallest id (a property of the ids, not of an order) and writes the ids of
the others as the dropped set, with a receipt naming the inputs by hash. A reader then goes
through ``select_documents`` with a ``Split`` (that ``Finalization``, the fraction and the side):
dropped documents are skipped and the held-out split is ``is_held_out(group_key, fraction)`` on
the group key, so a repository's files or a title's chapters land on one side together, and the
same document is on the same side under every seed. Two orders of the same inputs give the same
membership; the first-seen rule this replaced did not (collector review, second round). A
duplicate whose copies sit in groups on different sides is resolved conservatively: the surviving
copy's group decides, the other copies are in no split at all. Keys are versioned: ``text-v2``
documents (web, encyclopedia, books) deduplicate against each other across sources, code
deduplicates among code (``code-v1``); byte-identical content in a code file and a text document
is two documents, by design, and that is the whole cross-source claim. A web file without the key
columns groups by its id and takes ``text-v2`` of its text as the key, computed here.

Cost of finalization: two reads of every input (the first finds the keys that repeat, the second
the ids under them) and a set of sixteen-byte digests per distinct document (about one gigabyte
for the ten-billion-token web sample); an external sort would bound that if the corpus grows past
the box, and is the known alternative. Reading afterwards holds one row group's text plus the
dropped set.

Tokenization happens here as well, with the frozen tokenizer, into the ``int32`` ``Document``
the ``Stream`` takes. This module is off the hot path: pure Python over pyarrow, no torch work.
"""

import hashlib
import io
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from vesper.data import Document, is_held_out
from vesper.keys import SequenceDigest, dedup_key, file_sha256, verified_bytes
from vesper.publish import staged
from vesper.tokenizer import Tokenizer

COLUMNS = ("id", "text")
KEY_COLUMNS = ("group_key", "dedup_key", "dedup_version")
FINALIZE_RULE = (
    "for every dedup key that occurs more than once across all inputs, the document with the"
    " smallest id survives and the others are dropped; independent of any order"
)
SPLIT_RULE = (
    "a dropped document is skipped; then a document is held out when the first 64 bits of"
    " blake2b of its group key lie below held_out_fraction * 2^64 (vesper.data.is_held_out)"
)
_SHARE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RawDocument:
    """One document before tokenization: its stable id, its text, and its two keys.

    ``group_key`` decides the split; ``dedup_key`` (``<version>:<sha256>``) decides duplicates.
    """

    doc_id: str
    text: str
    group_key: str
    dedup_key: str


@dataclass(frozen=True)
class Component:
    """One component of the mix: its name, its parquet files, its share, and its generation pin.

    ``generation`` is ``"<kind>/<identity>@<manifest sha256>"`` when the files come from a
    published generation chosen by the selection (vesper.generations), None under the explicit
    legacy layout; it travels into the finalization receipt and the training recipe, so a run
    names the generation it read, not only the files.
    """

    name: str
    paths: tuple[Path, ...]
    share: float
    generation: str | None = None


@dataclass(frozen=True)
class Finalization:
    """The membership decision for a frozen set of inputs: the ids dropped as duplicates.

    ``inputs`` are ``(component, path, sha256, generation)`` of every file the decision was
    taken over, in order, so a reader can check it is reading the same files under the same
    assignment to components from the same generations (``generation`` is the component's pin or
    None under the legacy layout); ``receipt`` is the JSON written beside the dropped set.
    """

    dropped: frozenset[str]
    inputs: tuple[tuple[str, str, str, str | None], ...]
    receipt: Path


@dataclass(frozen=True)
class Split:
    """What a reader yields: one side of the finalized corpus.

    ``held_out=False`` is the training side, ``True`` the evaluation side; ``held_out_fraction``
    is the configured ``training.held_out_fraction``.
    """

    finalization: Finalization
    held_out_fraction: float
    held_out: bool = False


def _permutation(count: int, key: str) -> list[int]:
    """A fixed permutation of ``range(count)`` for ``key``: indices sorted by a keyed blake2b.

    A keyed hash rather than a seeded generator, chosen for reproducibility: the order is a
    property of the key alone, the same in every process, on every machine and across library
    versions, with no generator state to carry. A seeded generator would have been correct
    too; this is the deliberate choice, not a security requirement.
    """

    def rank(index: int) -> bytes:
        return hashlib.blake2b(f"{key}:{index}".encode(), digest_size=8).digest()

    return sorted(range(count), key=rank)


def string_value(value: object, name: str, path: Path, group: int) -> str:
    """PRIMITIVE: the data boundary's string check for one value of one parquet column.

    Raises:
        TypeError: naming the file, the row group and the column, on a non-string value.
    """
    if not isinstance(value, str):
        msg = f"{path} row group {group}: column {name!r} holds {type(value).__name__}, not str"
        raise TypeError(msg)
    return value


def string_column(table_column: list[object], name: str, path: Path, group: int) -> list[str]:
    """PRIMITIVE: ``string_value`` over a whole column of one row group.

    Raises:
        TypeError: as ``string_value``, on the first non-string value.
    """
    return [string_value(value, name, path, group) for value in table_column]


def _columns(path: Path) -> tuple[int, bool]:
    """Row-group count and whether the file carries the key columns.

    Raises:
        ValueError: on a missing ``id`` or ``text`` column, or on only some of the key columns.
    """
    with pq.ParquetFile(str(path)) as parquet:
        names = parquet.schema_arrow.names
        missing = [c for c in COLUMNS if c not in names]
        if missing:
            msg = f"{path}: missing column(s) {missing}; found {names}"
            raise ValueError(msg)
        keyed = [c for c in KEY_COLUMNS if c in names]
        if keyed and len(keyed) != len(KEY_COLUMNS):
            msg = f"{path}: key columns must come together, found only {keyed}"
            raise ValueError(msg)
        return parquet.metadata.num_row_groups, bool(keyed)


def _read_group(path: Path, group: int, *, keyed: bool) -> list[RawDocument]:
    """One row group of one file as documents, in the file's row order."""
    columns: list[str] = [*COLUMNS, *KEY_COLUMNS] if keyed else [*COLUMNS]
    with pq.ParquetFile(str(path)) as parquet:
        table = parquet.read_row_group(group, columns=columns)
    ids = string_column(table.column("id").to_pylist(), "id", path, group)
    texts = string_column(table.column("text").to_pylist(), "text", path, group)
    if keyed:
        groups = string_column(table.column("group_key").to_pylist(), "group_key", path, group)
        hashes = string_column(table.column("dedup_key").to_pylist(), "dedup_key", path, group)
        versions = string_column(
            table.column("dedup_version").to_pylist(), "dedup_version", path, group
        )
        keys = [f"{v}:{h}" for v, h in zip(versions, hashes, strict=True)]
    else:
        groups = ids
        keys = [":".join(dedup_key(text, "text")) for text in texts]
    return [
        RawDocument(doc_id=i, text=t, group_key=g, dedup_key=k)
        for i, t, g, k in zip(ids, texts, groups, keys, strict=True)
    ]


@dataclass(frozen=True)
class ReaderCursor:
    """Where a ``RawReader`` stands.

    The next document is row ``row`` of the permuted rows of the ``pair``-th ``(file, row group)``
    in the permuted pair order; ``pair`` equal to the pair count is the end.
    """

    pair: int
    row: int


class RawReader:
    """Every document of ``paths`` exactly once, in a seeded block-shuffled order, whole, keyed.

    The order is a permutation of the ``(file, row group)`` pairs keyed by ``seed`` and, inside
    each group, a permutation of its rows keyed by ``seed`` and the pair (``_permutation``), so
    the same seed gives the same order in every process and a different seed a different one.
    The rows of one group stay contiguous (see the module docstring for the limits). Each file
    is opened once for its metadata and once per row group read, and closed each time, so any
    number of files needs a handful of descriptors. A collected file's keys are read; a web file
    groups by its id and is keyed by ``text-v2`` of its text. ``cursor()`` is the position of the
    next document and ``seek(cursor)`` continues from a stored one at the cost of one row-group
    read: the bounded-cost resume (a replay of the consumed prefix is the test oracle for it).

    Raises:
        ValueError: if ``paths`` is empty, a file lacks the ``id`` or ``text`` column, or carries
            only some of the key columns.
        TypeError: if a value in the read columns is not a string.
    """

    def __init__(self, paths: Sequence[Path], seed: int) -> None:
        """Open every file once for its layout; nothing is read yet."""
        if len(paths) == 0:
            msg = "paths must name at least one parquet file"
            raise ValueError(msg)
        self.paths: Sequence[Path] = paths
        self.seed: int = seed
        self._layout: list[tuple[int, bool]] = [_columns(path) for path in paths]
        self._pairs: list[tuple[int, int]] = [
            (fi, g) for fi, (count, _) in enumerate(self._layout) for g in range(count)
        ]
        self._order: list[int] = _permutation(len(self._pairs), f"groups:{seed}")
        self._pair: int = 0  # position in the permuted pair order
        self._row: int = 0  # position in the permuted rows of the current group
        self._rows: list[RawDocument] | None = None  # the current group, in permuted order

    def cursor(self) -> ReaderCursor:
        """The position of the next document."""
        return ReaderCursor(pair=self._pair, row=self._row)

    def seek(self, cursor: ReaderCursor) -> None:
        """Continue from ``cursor``; the group it names is read once.

        Raises:
            ValueError: when the cursor lies outside this reader's pairs or the group's rows.
        """
        if not 0 <= cursor.pair <= len(self._pairs) or cursor.row < 0:
            msg = f"cursor {cursor} outside {len(self._pairs)} pairs"
            raise ValueError(msg)
        self._pair, self._row, self._rows = cursor.pair, cursor.row, None
        if cursor.pair < len(self._pairs):
            rows = self._load(cursor.pair)
            if cursor.row > len(rows):
                msg = f"cursor {cursor} outside the {len(rows)} rows of its group"
                raise ValueError(msg)
            self._rows = rows
        elif cursor.row != 0:
            msg = f"cursor {cursor} at the end must have row 0"
            raise ValueError(msg)

    def next(self) -> RawDocument | None:
        """The next document, or None at the end."""
        while self._pair < len(self._pairs):
            if self._rows is None:
                self._rows = self._load(self._pair)
            if self._row < len(self._rows):
                document = self._rows[self._row]
                self._row += 1
                return document
            self._pair, self._row, self._rows = self._pair + 1, 0, None
        return None

    def documents(self) -> Iterator[RawDocument]:
        """Every remaining document."""
        while (document := self.next()) is not None:
            yield document

    def _load(self, pair: int) -> list[RawDocument]:
        fi, group = self._pairs[self._order[pair]]
        rows = _read_group(self.paths[fi], group, keyed=self._layout[fi][1])
        return [rows[r] for r in _permutation(len(rows), f"rows:{self.seed}:{fi}:{group}")]


def iter_raw_documents(paths: Sequence[Path], seed: int) -> Iterator[RawDocument]:
    """``RawReader(paths, seed).documents()``: every document once, seeded block-shuffled order.

    Raises:
        ValueError, TypeError: as ``RawReader``.
    """
    return RawReader(paths, seed).documents()


def _iter_in_file_order(paths: Sequence[Path]) -> Iterator[RawDocument]:
    """Every document of ``paths`` in file and row order (finalization needs no shuffle)."""
    for path in paths:
        count, keyed = _columns(path)
        for group in range(count):
            yield from _read_group(path, group, keyed=keyed)


def file_manifest(components: Sequence[Component]) -> list[dict[str, object]]:
    """Every input file with its component, generation pin, path, bytes and sha256, in order."""
    rows: list[dict[str, object]] = []
    for component in components:
        rows.extend(
            {
                "component": component.name,
                "generation": component.generation,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in component.paths
        )
    return rows


def _digest(key: str) -> bytes:
    return hashlib.blake2b(key.encode(), digest_size=16).digest()


def _inputs(rows: list[dict[str, object]]) -> tuple[tuple[str, str, str, str | None], ...]:
    """The receipt's rows as ``(component, path, sha256, generation)``: assignment and pin too.

    Maintainer B's fixture (review of 444b966): the same files under another assignment to
    components had the same flattened path and hash list and was accepted as a continuation. The
    generation pin (None in a receipt written under the legacy layout) binds which published
    generation the files came from.
    """
    return tuple(
        (
            cast("str", r["component"]),
            cast("str", r["path"]),
            cast("str", r["sha256"]),
            cast("str | None", r.get("generation")),
        )
        for r in rows
    )


def _repeated_keys(components: Sequence[Component]) -> tuple[set[bytes], int]:
    """First read: the dedup keys that occur more than once, and the document count.

    Raises:
        ValueError: when a document id occurs more than once across the inputs.
    """
    once: set[bytes] = set()
    repeated: set[bytes] = set()
    identities: set[bytes] = set()
    documents = 0
    for component in components:
        for raw in _iter_in_file_order(component.paths):
            documents += 1
            identity = _digest(raw.doc_id)
            if identity in identities:
                # one id is one document; a second occurrence (a repeated acquisition run in the
                # same directory, or two sources claiming one id) is an input error, not a
                # duplicate to resolve, because the two occurrences may not even agree
                msg = f"document id {raw.doc_id!r} occurs more than once across the inputs"
                raise ValueError(msg)
            identities.add(identity)
            digest = _digest(raw.dedup_key)
            if digest in once:
                repeated.add(digest)
            else:
                once.add(digest)
    return repeated, documents


def _dropped_ids(
    components: Sequence[Component], repeated: set[bytes]
) -> tuple[set[str], dict[str, int]]:
    """Second read: under every repeated key the smallest id survives; the rest, by component."""
    members: dict[bytes, list[str]] = {}
    owner: dict[str, str] = {}
    for component in components:
        for raw in _iter_in_file_order(component.paths):
            digest = _digest(raw.dedup_key)
            if digest in repeated:
                members.setdefault(digest, []).append(raw.doc_id)
                owner[raw.doc_id] = component.name
    dropped: set[str] = set()
    for ids in members.values():
        survivor = min(ids)
        dropped.update(i for i in ids if i != survivor)
    per_component = {c.name: 0 for c in components}
    for doc_id in dropped:
        per_component[owner[doc_id]] += 1
    return dropped, per_component


def finalize(components: Sequence[Component], out: Path) -> Finalization:
    """Decide membership once for these inputs: the dropped duplicates, written with a receipt.

    Two reads of every input in file order (``FINALIZE_RULE``): the first finds the dedup keys
    that occur more than once and checks that every document id occurs once, the second the ids
    under each repeated key; every id but the smallest is dropped. Writes
    ``out/finalize-<stamp>.parquet`` (column ``doc_id``, sorted) and ``out/finalize-<stamp>.json``
    (the inputs with their hashes, the rule, the counts per component). Returns the
    ``Finalization`` to pass to the readers.

    Raises:
        ValueError: on no components, a component without files, a document id that occurs
            more than once across the inputs (an occurrence is identified by its id; a repeated
            id is an input error, since the occurrences need not agree), or a receipt with this
            second's stamp already in ``out`` (receipts are immutable).
    """
    if len(components) == 0 or any(len(c.paths) == 0 for c in components):
        msg = "finalize needs at least one component, each with at least one file"
        raise ValueError(msg)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parquet_path = out / f"finalize-{stamp}.parquet"
    receipt = out / f"finalize-{stamp}.json"
    try:
        # the receipt path is reserved exclusively before any work, so two runs in one second
        # cannot both pass a check and publish onto one path; the reserved file stays empty until
        # the complete receipt is renamed onto it, and ``load_finalization`` refuses an empty one
        receipt.touch(exist_ok=False)
    except FileExistsError as exc:
        msg = f"{receipt}: a finalization with this stamp already exists"
        raise ValueError(msg) from exc
    try:
        inputs = file_manifest(components)
        repeated, documents = _repeated_keys(components)
        dropped, per_component = _dropped_ids(components, repeated)
        dropped_sha256 = _publish_dropped(sorted(dropped), parquet_path)
        record: dict[str, object] = {
            "finalized_at_utc": stamp,
            "rule": FINALIZE_RULE,
            "inputs": inputs,
            "documents": documents,
            "repeated_keys": len(repeated),
            "dropped": len(dropped),
            "dropped_per_component": per_component,
            "dropped_file": parquet_path.name,
            "dropped_sha256": dropped_sha256,
        }
        # published last and atomically: a reader finds the receipt empty (reserved) or complete
        with staged(receipt) as partial:
            _ = partial.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except BaseException:
        # a failed run releases its reservation; the published dropped file, if any, goes too
        receipt.unlink(missing_ok=True)
        parquet_path.unlink(missing_ok=True)
        raise
    return Finalization(dropped=frozenset(dropped), inputs=_inputs(inputs), receipt=receipt)


def _publish_dropped(ids: list[str], parquet_path: Path) -> str:
    """Write the dropped ids to a staged file this run owns, hash it, publish it by rename.

    Returns the file's sha256; a failure removes the staged file and leaves no destination.
    """
    with staged(parquet_path) as partial:
        pq.write_table(pa.Table.from_pydict({"doc_id": ids}), str(partial))
        return file_sha256(partial)  # the rename happens on the way out, after the hash


def load_finalization(receipt: Path, *, sha256: str | None = None) -> Finalization:
    """Read a finalization back from its receipt; the dropped file's hash is verified.

    Each file is read once and parsed from the bytes that were hashed (``keys.verified_bytes``),
    never re-read after its check. ``sha256``, when given, is the receipt's own expected hash
    (a run's recorded identity): the receipt that is parsed is then the one that was checked.

    Raises:
        ValueError: when the receipt's hash is not ``sha256``, the receipt is empty (reserved by
            a run that is still working or was abandoned; the dropped file beside it is not
            reader-ready), or the dropped file is missing or its hash differs from the receipt.
    """
    data = (
        receipt.read_bytes()
        if sha256 is None
        else verified_bytes(receipt, sha256, "the finalization receipt")
    )
    text = data.decode("utf-8")
    if not text.strip():
        msg = f"{receipt}: reserved, not published (a finalization in progress or abandoned)"
        raise ValueError(msg)
    record = cast("dict[str, object]", json.loads(text))
    dropped_path = receipt.parent / cast("str", record["dropped_file"])
    dropped = verified_bytes(dropped_path, record["dropped_sha256"], f"{receipt}: its dropped file")
    column = pq.read_table(io.BytesIO(dropped), columns=["doc_id"]).column("doc_id").to_pylist()
    ids = string_column(column, "doc_id", dropped_path, 0)
    inputs = cast("list[dict[str, object]]", record["inputs"])
    return Finalization(dropped=frozenset(ids), inputs=_inputs(inputs), receipt=receipt)


def check_inputs(finalization: Finalization, components: Sequence[Component]) -> None:
    """The finalization was taken over exactly these files (path and hash, in order).

    Raises:
        ValueError: naming the first difference.
    """
    current = _inputs(file_manifest(components))
    if current != finalization.inputs:
        expected = {
            path: (component, digest, generation)
            for component, path, digest, generation in finalization.inputs
        }
        for component, path, digest, generation in current:
            known = expected.get(path)
            if known is None or known[1] != digest:
                msg = f"finalization {finalization.receipt} was not taken over {path} as it is now"
                raise ValueError(msg)
            if known[0] != component:
                msg = (
                    f"finalization {finalization.receipt} took {path} as component {known[0]!r},"
                    f" not {component!r}: the files' assignment to components is pinned"
                )
                raise ValueError(msg)
            if known[2] != generation:
                msg = (
                    f"finalization {finalization.receipt} took {path} from generation"
                    f" {known[2]!r}, not {generation!r}: the generation is pinned"
                )
                raise ValueError(msg)
        msg = f"finalization {finalization.receipt} covers a different set of files"
        raise ValueError(msg)


def keeps(raw: RawDocument, split: Split) -> bool:
    """The reader's finalizer step for one document: not dropped, and on the split's side.

    A document whose id is in the finalization's dropped set is out; then
    ``is_held_out(group_key, held_out_fraction) == held_out`` keeps the requested side. Both
    decisions are properties of the document, so the membership is the same under every order.
    """
    if raw.doc_id in split.finalization.dropped:
        return False
    return is_held_out(raw.group_key, split.held_out_fraction) == split.held_out


def select_documents(documents: Iterable[RawDocument], split: Split) -> Iterator[RawDocument]:
    """``keeps`` over an iterator: the documents of the split's side, dropped duplicates skipped."""
    for raw in documents:
        if keeps(raw, split):
            yield raw


def iter_documents(
    paths: Sequence[Path], seed: int, tokenizer: Tokenizer, split: Split
) -> Iterator[Document]:
    """Tokenized documents of one side of the split, in the seeded order of ``iter_raw_documents``.

    The two sides are disjoint and together cover the finalized corpus (``select_documents``).
    """
    for raw in select_documents(iter_raw_documents(paths, seed), split):
        yield Document(doc_id=raw.doc_id, tokens=tokenizer.encode(raw.text))


def _check_shares(components: Sequence[Component]) -> None:
    if len(components) == 0:
        msg = "the mix needs at least one component"
        raise ValueError(msg)
    shares = [c.share for c in components]
    if min(shares) <= 0.0 or abs(sum(shares) - 1.0) > _SHARE_TOLERANCE:
        msg = f"shares must be positive and sum to one, got {shares}"
        raise ValueError(msg)


def budgets(components: Sequence[Component], byte_budget: int) -> list[int]:
    """Each component's byte budget: ``share * byte_budget`` floored, the remainder to the last.

    Raises:
        ValueError: on no components, a non-positive share, shares not summing to one, a budget
            below 1, or a budget too small to give every component a byte.
    """
    _check_shares(components)
    if byte_budget < 1:
        msg = f"byte_budget must be at least 1, got {byte_budget}"
        raise ValueError(msg)
    out = [int(c.share * byte_budget) for c in components]
    out[-1] = byte_budget - sum(out[:-1])
    if min(out) < 1:
        msg = f"byte_budget {byte_budget} leaves a component without a byte: {out}"
        raise ValueError(msg)
    return out


def iter_mixed_texts(
    components: Sequence[Component], seed: int, split: Split, byte_budget: int
) -> Iterator[tuple[str, str]]:
    """``(component name, text)`` pairs of the tokenizer's sample: the mix by byte share.

    Every component contributes its texts of the split's side in the seeded order, finalized
    (``select_documents``). Each component's budget is ``share * byte_budget`` floored, the
    remainder going to the last one, so the budgets sum to the budget exactly. At each step the
    component whose emitted bytes are the smallest fraction of its budget yields its next text
    (ties go to the earlier component), so the running proportions track the shares at document
    granularity. A component stops when its budget is reached (the text that crosses it is
    included whole) or when it runs out of documents, in which case the sample is short of it and
    the caller's receipt shows the bytes it did give. Nothing is cut inside a document.

    Raises:
        ValueError: as ``budgets``.
    """
    limits = budgets(components, byte_budget)
    streams = [select_documents(iter_raw_documents(c.paths, seed), split) for c in components]
    emitted = [0] * len(components)
    active = list(range(len(components)))

    def ratio(index: int) -> float:
        return emitted[index] / limits[index]

    while active:
        index = min(active, key=ratio)
        raw = next(streams[index], None)
        if raw is None:
            active.remove(index)
            continue
        yield components[index].name, raw.text
        emitted[index] += len(raw.text.encode("utf-8"))
        if emitted[index] >= limits[index]:
            active.remove(index)


def iter_training_texts(
    paths: Sequence[Path], seed: int, split: Split, byte_budget: int
) -> Iterator[str]:
    """Texts of the split's side in the seeded order until ``byte_budget`` UTF-8 bytes are reached.

    The one-component case of ``iter_mixed_texts``: the other side and the dropped duplicates
    never enter it, the last text may carry the total past the budget, nothing is cut inside a
    document.

    Raises:
        ValueError: if ``byte_budget`` is below 1.
    """
    component = Component("all", tuple(paths), 1.0)
    for _, text in iter_mixed_texts((component,), seed, split, byte_budget):
        yield text


def iter_mixed_documents(
    components: Sequence[Component],
    seed: int,
    tokenizer: Tokenizer,
    split: Split,
    *,
    skip: int = 0,
    tokens_emitted: dict[str, int] | None = None,
) -> Iterator[Document]:
    """The training mix by tokens: whole, tokenized documents interleaved by the components' shares.

    Every component contributes its documents of the split's side in the seeded order, finalized
    (``select_documents``). The shares are relative and there is no total budget: at each
    document the component with the smallest ``tokens_emitted / share`` yields its next document
    (ties to the earlier component), so the realized token proportions track the shares at
    document granularity; a component that runs out leaves the others running, and the caller's
    ``tokens_emitted`` (updated in place, by component name) shows the realized shares. The first
    ``skip`` documents with text are replayed eagerly, before this function returns (tokenized
    and counted, not yielded), so a resumed run makes the same component choices, continues with
    the same document as an uninterrupted one, and can compare ``tokens_emitted`` with its
    checkpoint before consuming anything (``Stream.documents_taken`` summed over the streams is
    the count to pass; it skips empty documents as the stream does, and a replay costs the
    tokenization of the skipped documents). Documents are whole; nothing moves inside one.

    Raises:
        ValueError: on no components, a share that is not positive, shares that do not sum to
            one, or a negative ``skip``.
    """
    mixer = Mixer(components, seed, tokenizer, split, tokens_emitted=tokens_emitted)
    mixer.replay(skip)
    return mixer.documents()


@dataclass(frozen=True)
class MixerState:
    """Where a ``Mixer`` stands.

    Every reader's cursor as ``(pair, row)``, the components still active, the tokens handed out
    per component, the documents handed out and the digest over their ids.
    """

    cursors: tuple[tuple[int, int], ...]
    active: tuple[int, ...]
    emitted: dict[str, int]
    pulled: int
    digest: str


class Mixer:
    """The token-share mixer: one seeded reader per component, the balances, the pulled ids.

    ``emitted`` holds the tokens handed out per component (the caller's dict when given, updated
    in place); ``pulled`` counts the documents handed out and ``digest()`` is the chained sha256
    over their ids in order: the identity of what was consumed, which equal counts cannot give
    (maintainer B's review of b797837: equal-length documents under another seed pass a count
    check). ``state()`` and ``restore(state)`` are the bounded-cost resume: the readers seek to
    their cursors at the cost of one row-group read each and the digest continues from its stored
    state; ``replay`` (pulling without yielding) stays as the test oracle that a restored mixer
    hands out the same documents as the uninterrupted one.

    Raises:
        ValueError: on no components, a share that is not positive or shares that do not sum to
            one.
    """

    def __init__(
        self,
        components: Sequence[Component],
        seed: int,
        tokenizer: Tokenizer,
        split: Split,
        *,
        tokens_emitted: dict[str, int] | None = None,
    ) -> None:
        """Open one seeded reader per component; nothing is pulled yet."""
        _check_shares(components)
        emitted = {c.name: 0 for c in components} if tokens_emitted is None else tokens_emitted
        for c in components:
            _ = emitted.setdefault(c.name, 0)
        self.components: Sequence[Component] = components
        self.tokenizer: Tokenizer = tokenizer
        self.split: Split = split
        self.emitted: dict[str, int] = emitted
        self.pulled: int = 0
        self.readers: list[RawReader] = [RawReader(c.paths, seed) for c in components]
        self.active: list[int] = list(range(len(components)))
        self._digest: SequenceDigest = SequenceDigest()

    @classmethod
    def restore(
        cls,
        components: Sequence[Component],
        seed: int,
        tokenizer: Tokenizer,
        split: Split,
        state: MixerState,
        *,
        tokens_emitted: dict[str, int] | None = None,
    ) -> "Mixer":
        """A mixer standing where ``state`` says, over the same components, seed and split.

        Raises:
            ValueError: when the state has another number of readers, names an unknown
                component, or a cursor lies outside its reader (``RawReader.seek``).
        """
        mixer = cls(components, seed, tokenizer, split, tokens_emitted=tokens_emitted)
        if len(state.cursors) != len(mixer.readers):
            msg = (
                f"mixer state has {len(state.cursors)} readers, the components {len(mixer.readers)}"
            )
            raise ValueError(msg)
        names = {c.name for c in components}
        if set(state.emitted) - names or any(not 0 <= i < len(components) for i in state.active):
            msg = f"mixer state names components outside {sorted(names)}"
            raise ValueError(msg)
        for reader, (pair, row) in zip(mixer.readers, state.cursors, strict=True):
            reader.seek(ReaderCursor(pair=pair, row=row))
        mixer.active = list(state.active)
        for name, count in state.emitted.items():
            mixer.emitted[name] = count
        mixer.pulled = state.pulled
        mixer._digest = SequenceDigest(state.digest, state.pulled)
        return mixer

    def state(self) -> MixerState:
        """Where the mixer stands, for a checkpoint."""
        cursors = tuple((c.pair, c.row) for c in (reader.cursor() for reader in self.readers))
        return MixerState(
            cursors=cursors,
            active=tuple(self.active),
            emitted=dict(self.emitted),
            pulled=self.pulled,
            digest=self.digest(),
        )

    def ratio(self, index: int) -> float:
        """Tokens emitted over the share: the component with the smallest ratio goes next."""
        return self.emitted[self.components[index].name] / self.components[index].share

    def pull(self) -> Document | None:
        """The next document of the mix, tokenized and counted, or None when all are out."""
        while self.active:
            index = min(self.active, key=self.ratio)
            raw = self._next_kept(index)
            if raw is None:
                self.active.remove(index)
                continue
            if not raw.text:
                continue
            tokens = self.tokenizer.encode(raw.text)
            self.emitted[self.components[index].name] += int(tokens.numel())
            self.pulled += 1
            self._digest.add(raw.doc_id.encode("utf-8"))
            return Document(doc_id=raw.doc_id, tokens=tokens, source=self.components[index].name)
        return None

    def _next_kept(self, index: int) -> RawDocument | None:
        """The component's next document on the split's side, or None when it is out."""
        reader = self.readers[index]
        while (raw := reader.next()) is not None:
            if keeps(raw, self.split):
                return raw
        return None

    def replay(self, count: int) -> None:
        """Pull ``count`` documents without yielding them (tokenized, counted, digested).

        Raises:
            ValueError: on a negative ``count``.
        """
        if count < 0:
            msg = f"replay count must be at least 0, got {count}"
            raise ValueError(msg)
        for _ in range(count):
            if self.pull() is None:
                break

    def digest(self) -> str:
        """The sha256 (hex) over the ids pulled so far, in order."""
        return self._digest.hexdigest()

    def documents(self) -> Iterator[Document]:
        """Every remaining document, in the mix's order."""
        while (document := self.pull()) is not None:
            yield document
