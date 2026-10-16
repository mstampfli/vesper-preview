"""Collector for the non-web components of the mix (docs/reviews/2026-10-15-mixture-sources.md).

One shard in, one compact parquet out, provenance kept, one row group at a time. Every row written
carries a namespaced ``id``, the ``text``, the ``source`` and its pinned ``revision``, the
``group_key`` the split is taken on (the repository for code, the document id for the
encyclopedia, the title for books), the versioned ``dedup_key`` that decides duplicates before the
split, and the provenance columns ``license`` (the label the filter saw), ``header`` (the SPDX
expression found in the file, code only), ``origin`` (the file path or the URL) and ``title``. The
output has the ``id`` and ``text`` columns the corpus reader expects, and the reader applies the
group split and the deduplication when the key columns are present (``vesper.corpus``).

The code filter has two parts, both recorded: the repository's license label must be in the
allowlist, and a file that carries an explicit ``SPDX-License-Identifier`` naming a license
outside the allowlist is rejected as conflicting evidence. The label is still not a file audit; a
file without a header is kept on its repository's label, and the receipt's ``rule`` says exactly
which policy ran.

Outputs are published whole or not at all: a collector writes to a temporary sibling, closes it on
every path, and moves it onto the destination only after success, so a failed run leaves a prior
output untouched; the input is never the output. Hashes are computed in bounded chunks
(``vesper.keys.file_sha256``). Code is never normalized: its dedup key is the hash of the exact
text, line endings included. Text sources use ``normalize_text_v2`` for the key only.

The network step, ``fetch_shard``, is a thin call into ``huggingface_hub`` for one file of one
pinned revision; the unit tests do not exercise it, they run the filters and writers on small
local parquet files. Nothing here downloads a repository whole.
"""

import hashlib
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from vesper.books import BOOK_LICENSE, BOOK_SOURCE, GITHUB_REMOTE, Edition, book_text
from vesper.corpus import string_column, string_value
from vesper.keys import dedup_key, file_sha256

CODE_REPO = "codeparrot/github-code"
# The pinned dataset revisions below are public git commit ids, not secrets.
CODE_REVISION = "b5661e6b17396364b2bcf8e68977b0d28e1ebd19"  # pragma: allowlist secret
CODE_ALLOWLIST = frozenset(
    {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "unlicense"}
)
WIKI_REPO = "wikimedia/wikipedia"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # pragma: allowlist secret
WIKI_RULE = "every article whose text is not empty or whitespace; attribution (URL, title) kept"
BOOKS_RULE = "titles whose dc:language is en or en-*, with non-empty text in spine order"

OUTPUT_COLUMNS = (
    "id",
    "text",
    "source",
    "revision",
    "group_key",
    "dedup_key",
    "dedup_version",
    "license",
    "header",
    "origin",
    "title",
)
_SCHEMA = pa.schema([pa.field(name, pa.string()) for name in OUTPUT_COLUMNS])
_ROWS_PER_GROUP = 1000
_SPDX_TAG = "SPDX-License-Identifier:"
_SPDX_WINDOW = 4096
_SPDX_OPERATORS = frozenset({"and", "or"})
_SPDX_CLOSERS = ("*/", "-->", "*)", "#", "//")
# an SPDX identifier is letters, digits, '.', '-' and a trailing '+'; what follows in the same
# token (a sentence's period, markup in a file that quotes the header) is not part of it
_SPDX_ID = re.compile(r"[A-Za-z0-9.+-]+")


@dataclass(frozen=True)
class CollectReceipt:
    """What one input contributed, under which policy, from which bytes, to which output.

    ``rule`` states the keep policy that ran, in words a reader can compare between runs;
    ``input_sha256`` is the shard's content hash (for books, the hash of the sorted
    ``slug@commit`` manifest); ``rejected`` counts rows by the reason they were dropped.
    """

    source: str
    revision: str
    shard: str
    rule: str
    rows_read: int
    rows_kept: int
    bytes_kept: int
    rejected: dict[str, int]
    input_sha256: str
    output: str
    output_sha256: str


@dataclass(frozen=True)
class TitleOutcome:
    """One fetched title's fate: ``kept``, or rejected as ``language``, ``malformed``, ``empty``."""

    slug: str
    commit: str
    status: str
    title: str
    text_bytes: int
    note: str


@dataclass(frozen=True)
class BooksReceipt:
    """The books component's receipt: the shard-style totals and the per-title manifest."""

    receipt: CollectReceipt
    titles: tuple[TitleOutcome, ...]


def fetch_shard(repo: str, revision: str, path: str, staging: Path) -> Path:
    """Download one file of one pinned dataset revision into ``staging`` and return its path."""
    staging.mkdir(parents=True, exist_ok=True)
    local = hf_hub_download(
        repo_id=repo, filename=path, repo_type="dataset", revision=revision, cache_dir=str(staging)
    )
    return Path(local)


@dataclass(frozen=True)
class SpdxHeader:
    """What a file's first ``SPDX-License-Identifier`` says: absent, parsed or unsupported.

    ``parsed`` carries the lower-cased license identifiers the expression names; ``unsupported``
    is a header that is there but not evidence this check can read (a malformed expression, an
    unknown token, an exception clause), and is never approval; ``absent`` is no header at all.
    """

    status: str
    expression: str
    identifiers: tuple[str, ...]


def spdx_header(text: str) -> SpdxHeader:
    """PRIMITIVE: the first ``SPDX-License-Identifier`` in the first 4 KiB of a file, read strictly.

    The expression after the tag (comment closers and whitespace trimmed) is split on whitespace,
    parentheses and ``/`` (the old dual-license form ``MIT/GPL-3.0-only`` names two licenses). A
    token may be the operator ``AND`` or ``OR`` (dropped) or an identifier: letters, digits,
    ``.``, ``-`` and ``+``, with one trailing period allowed for a header that ends a sentence
    (``Apache-2.0.``); ``GPL-2.0+`` keeps its plus. Anything else (``***``, markup such as
    ``BSD-3-Clause</span>``, a ``WITH`` exception clause this check does not interpret) makes the
    header ``unsupported``: evidence that is present and unreadable is not approval. Not a parser
    of the full SPDX grammar: it names the licenses an expression mentions, which is what a
    conflict check needs, and refuses what it cannot name.
    """
    head = text[:_SPDX_WINDOW]
    at = head.find(_SPDX_TAG)
    if at < 0:
        return SpdxHeader("absent", "", ())
    line = head[at + len(_SPDX_TAG) :].split("\n", 1)[0].strip()
    for closer in _SPDX_CLOSERS:
        if line.endswith(closer):
            line = line[: -len(closer)].strip()
    tokens = line.replace("(", " ").replace(")", " ").replace("/", " ").split()
    identifiers: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in _SPDX_OPERATORS:
            continue
        bare = lowered.removesuffix(".")
        if lowered == "with" or _SPDX_ID.fullmatch(bare) is None:
            return SpdxHeader("unsupported", line, ())
        identifiers.append(bare)
    if not identifiers:
        return SpdxHeader("unsupported", line, ())
    return SpdxHeader("parsed", line, tuple(identifiers))


def _distinct(shard: Path, out: Path) -> None:
    """The input is never the output (the temporary sibling has a name of its own).

    Raises:
        ValueError: when they name the same file.
    """
    if shard.resolve() == out.resolve():
        msg = f"input and output are the same file: {shard}"
        raise ValueError(msg)


class ParquetOutput:
    """One compact parquet file, written to a temporary sibling and published whole or not at all.

    Fixed row groups; ``close()`` moves the file onto its destination, ``abort()`` or leaving the
    ``with`` block through an exception removes it. Never relies on destructor timing to close
    the writer.
    """

    @staticmethod
    def temporary(out: Path) -> Path:
        """A temporary sibling this writer alone owns: unique per process and per writer."""
        ParquetOutput.count += 1
        return out.with_name(f".{out.name}.{os.getpid()}-{ParquetOutput.count}.partial")

    count: int = 0

    def __init__(self, out: Path) -> None:
        """Open the writer on a fresh temporary beside ``out``; nothing touches ``out`` yet."""
        out.parent.mkdir(parents=True, exist_ok=True)
        self.out: Path = out
        self.temp: Path = self.temporary(out)
        self.writer: pq.ParquetWriter = pq.ParquetWriter(str(self.temp), _SCHEMA)
        self.open: bool = True
        self.pending: dict[str, list[str]] = {name: [] for name in OUTPUT_COLUMNS}
        self.rows: int = 0
        self.text_bytes: int = 0

    def __enter__(self) -> Self:
        """The writer itself; ``close()`` inside the block publishes, an exception aborts."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Abort when the block ends with the writer still open (an exception before close)."""
        del exc_type, exc, tb
        if self.open:
            self.abort()

    def add(self, row: dict[str, str]) -> None:
        """Queue one row (every output column present); a full row group is written at once."""
        for name in OUTPUT_COLUMNS:
            self.pending[name].append(row[name])
        self.rows += 1
        self.text_bytes += len(row["text"].encode("utf-8"))
        if len(self.pending["id"]) >= _ROWS_PER_GROUP:
            self.flush()

    def flush(self) -> None:
        """Write the queued rows as one row group, if there are any."""
        if self.pending["id"]:
            self.writer.write_table(pa.Table.from_pydict(self.pending, schema=_SCHEMA))
            self.pending = {name: [] for name in OUTPUT_COLUMNS}

    def close(self) -> str:
        """Finish the file, publish it onto the destination and return its hash."""
        self.flush()
        if self.rows == 0:
            self.writer.write_table(pa.Table.from_pydict(self.pending, schema=_SCHEMA))
        self.writer.close()
        self.open = False
        try:
            # the hash is this writer's content, taken on the temporary it alone owns: after the
            # rename the destination may already be another writer's publication
            digest = file_sha256(self.temp)
            _ = self.temp.replace(self.out)
        except BaseException:
            # the publication did not complete: the unpublished temporary is this writer's to
            # remove, whether or not its handle is still open
            self.temp.unlink(missing_ok=True)
            raise
        return digest

    def abort(self) -> None:
        """Close the writer and remove the unpublished temporary; the destination is untouched."""
        if self.open:
            self.writer.close()
            self.open = False
        self.temp.unlink(missing_ok=True)


def _groups(shard: Path, columns: list[str]) -> Iterator[tuple[int, pa.Table]]:
    """The shard's row groups one at a time, with the needed columns only."""
    with pq.ParquetFile(str(shard)) as parquet:
        names = parquet.schema_arrow.names
        missing = [c for c in columns if c not in names]
        if missing:
            msg = f"{shard}: missing column(s) {missing}; found {names}"
            raise ValueError(msg)
        for group in range(parquet.metadata.num_row_groups):
            yield group, parquet.read_row_group(group, columns=columns)


def _count(rejected: dict[str, int], reason: str) -> None:
    rejected[reason] = rejected.get(reason, 0) + 1


def collect_code(
    shard: Path,
    out: Path,
    allowlist: frozenset[str] = CODE_ALLOWLIST,
    revision: str = CODE_REVISION,
) -> CollectReceipt:
    """Keep the rows of one github-code shard whose label and header are both in the allowlist.

    The repository's license label must be in the allowlist, and the file's SPDX header, if it
    names licenses, must name only allowed ones. The physical columns are ``repo_name, path,
    content, license`` (the dataset card's ``code`` is ``content`` on disk). Labels are compared
    lower-cased; a missing or unlisted label is rejected and counted under its value, before the
    row's other fields are looked at, so a rejected row needs no string payload. A kept label's
    ``content`` must be a string; an SPDX header naming a license outside the allowlist rejects
    the file as ``header:<id>``, a header this check cannot read as ``header:unsupported``
    (present evidence is never approval), and the header expression is kept in the ``header``
    column. The id is
    ``code:<repo_name>/<path>@<revision>``, the group key ``code:<repo_name>``, the dedup key the
    hash of the exact content.

    Raises:
        ValueError: on a missing column, or when ``shard`` and ``out`` are the same file.
        TypeError: on a non-string value in a kept row.
    """
    _distinct(shard, out)
    allowed = {label.lower() for label in allowlist}
    rule = f"repository license label in {sorted(allowed)}; no SPDX header naming another license"
    rejected: dict[str, int] = {}
    read = 0
    with ParquetOutput(out) as output:
        for group, table in _groups(shard, ["repo_name", "path", "content", "license"]):
            labels = table.column("license").to_pylist()
            repos = table.column("repo_name").to_pylist()
            paths = table.column("path").to_pylist()
            contents = table.column("content").to_pylist()
            read += len(labels)
            for i, label in enumerate(labels):
                tag = label.lower() if isinstance(label, str) else "missing"
                if tag not in allowed:
                    _count(rejected, tag)
                    continue
                repo = string_value(repos[i], "repo_name", shard, group)
                path = string_value(paths[i], "path", shard, group)
                content = string_value(contents[i], "content", shard, group)
                header = spdx_header(content)
                if header.status == "unsupported":
                    _count(rejected, "header:unsupported")
                    continue
                outside = [name for name in header.identifiers if name not in allowed]
                if outside:
                    _count(rejected, f"header:{outside[0]}")
                    continue
                version, key = dedup_key(content, "code")
                output.add(
                    {
                        "id": f"code:{repo}/{path}@{revision}",
                        "text": content,
                        "source": CODE_REPO,
                        "revision": revision,
                        "group_key": f"code:{repo}",
                        "dedup_key": key,
                        "dedup_version": version,
                        "license": tag,
                        "header": header.expression,
                        "origin": path,
                        "title": "",
                    }
                )
        digest = output.close()
    return CollectReceipt(
        source=CODE_REPO,
        revision=revision,
        shard=shard.name,
        rule=rule,
        rows_read=read,
        rows_kept=output.rows,
        bytes_kept=output.text_bytes,
        rejected=rejected,
        input_sha256=file_sha256(shard),
        output=str(out),
        output_sha256=digest,
    )


def collect_wikipedia(shard: Path, out: Path, revision: str = WIKI_REVISION) -> CollectReceipt:
    """Keep every non-empty article of one Wikipedia shard with its attribution.

    The columns are ``id, url, title, text``; the id becomes ``wiki:<id>``, the group key the
    same, the dedup key the hash of the normalized text; the URL and title are kept because the
    license (CC BY-SA 3.0 / GFDL) asks for attribution. Empty texts are rejected and counted.

    Raises:
        ValueError: on a missing column, or when ``shard`` and ``out`` are the same file.
        TypeError: on a non-string value.
    """
    _distinct(shard, out)
    rejected: dict[str, int] = {}
    read = 0
    with ParquetOutput(out) as output:
        for group, table in _groups(shard, ["id", "url", "title", "text"]):
            ids = string_column(table.column("id").to_pylist(), "id", shard, group)
            urls = string_column(table.column("url").to_pylist(), "url", shard, group)
            titles = string_column(table.column("title").to_pylist(), "title", shard, group)
            texts = string_column(table.column("text").to_pylist(), "text", shard, group)
            read += len(ids)
            for doc_id, url, title, text in zip(ids, urls, titles, texts, strict=True):
                if not text.strip():
                    _count(rejected, "empty")
                    continue
                version, key = dedup_key(text, "text")
                output.add(
                    {
                        "id": f"wiki:{doc_id}",
                        "text": text,
                        "source": WIKI_REPO,
                        "revision": revision,
                        "group_key": f"wiki:{doc_id}",
                        "dedup_key": key,
                        "dedup_version": version,
                        "license": "cc-by-sa-3.0",
                        "header": "",
                        "origin": url,
                        "title": title,
                    }
                )
        digest = output.close()
    return CollectReceipt(
        source=WIKI_REPO,
        revision=revision,
        shard=shard.name,
        rule=WIKI_RULE,
        rows_read=read,
        rows_kept=output.rows,
        bytes_kept=output.text_bytes,
        rejected=rejected,
        input_sha256=file_sha256(shard),
        output=str(out),
        output_sha256=digest,
    )


def collect_books(editions: Iterable[Edition], out: Path, listing: str) -> BooksReceipt:
    """One row per fetched title whose language is English and whose text is non-empty.

    ``editions`` are the clones ``vesper.books.fetch_title`` returned, consumed lazily so a caller
    may fetch as it goes; ``listing`` names the title universe the selection came from (the
    listing's UTC stamp) and is the receipt's ``revision``. The id is ``book:<slug>@<commit>``,
    the group key ``book:<slug>`` (a title's chapters stay on one side of the split), the dedup key
    ``text-v2``, the license the publisher's statement (``public-domain-us``), the origin the
    repository URL, the title ``dc:title``. A title whose language is not ``en`` or ``en-*`` is
    rejected as ``language``; a title ``book_text`` cannot read as ``malformed`` (the error kept in
    the outcome's note); a title without text as ``empty``. Nothing is normalized in the text.
    The receipt's input hash is the SHA-256 of the sorted ``slug@commit`` lines.
    """
    rejected: dict[str, int] = {}
    outcomes: list[TitleOutcome] = []
    manifest: list[str] = []
    with ParquetOutput(out) as output:
        for edition in editions:
            manifest.append(f"{edition.slug}@{edition.commit}")
            try:
                book = book_text(edition.repo)
            except ValueError as exc:
                _count(rejected, "malformed")
                outcomes.append(
                    TitleOutcome(edition.slug, edition.commit, "malformed", "", 0, str(exc))
                )
                continue
            size = len(book.text.encode("utf-8"))
            language = book.language.lower()
            if not (language == "en" or language.startswith("en-")):
                _count(rejected, "language")
                outcomes.append(
                    TitleOutcome(
                        edition.slug, edition.commit, "language", book.title, size, book.language
                    )
                )
                continue
            if not book.text.strip():
                _count(rejected, "empty")
                outcomes.append(
                    TitleOutcome(edition.slug, edition.commit, "empty", book.title, 0, "")
                )
                continue
            version, key = dedup_key(book.text, "text")
            output.add(
                {
                    "id": f"book:{edition.slug}@{edition.commit}",
                    "text": book.text,
                    "source": BOOK_SOURCE,
                    "revision": edition.commit,
                    "group_key": f"book:{edition.slug}",
                    "dedup_key": key,
                    "dedup_version": version,
                    "license": BOOK_LICENSE,
                    "header": "",
                    "origin": f"{GITHUB_REMOTE}/{edition.slug}",
                    "title": book.title,
                }
            )
            outcomes.append(
                TitleOutcome(edition.slug, edition.commit, "kept", book.title, size, book.author)
            )
        digest = output.close()
    receipt = CollectReceipt(
        source=BOOK_SOURCE,
        revision=listing,
        shard=f"titles:{len(manifest)}",
        rule=BOOKS_RULE,
        rows_read=len(manifest),
        rows_kept=output.rows,
        bytes_kept=output.text_bytes,
        rejected=rejected,
        input_sha256=hashlib.sha256("\n".join(sorted(manifest)).encode()).hexdigest(),
        output=str(out),
        output_sha256=digest,
    )
    return BooksReceipt(receipt=receipt, titles=tuple(outcomes))
