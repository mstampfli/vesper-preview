"""Finite training-side supply, counted explicitly once and assessed before a run.

Counts describe encoded inventory, not usable training inputs or guaranteed completion. A
nominal shortage warns without changing finite training or selecting cycling. The count CLI
does no model work; ordinary training never starts it implicitly.
"""

import argparse
import hashlib
import json
import logging
import math
import multiprocessing
import os
import platform
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import cast

from vesper.config import load
from vesper.corpus import (
    SPLIT_RULE,
    Component,
    RawReader,
    Split,
    check_inputs,
    keeps,
    load_finalization,
)
from vesper.keys import SequenceDigest, file_sha256, hashed_bytes, verified_bytes
from vesper.provenance import environment, write_json
from vesper.publish import copied_snapshot, owned_staging, staged
from vesper.sample import components_for
from vesper.tokenizer import Tokenizer

_SCHEMA = 1
_RULE = "finite training side; keeps; skip empty text; encode without special tokens; labels L-1"
_SOURCES = ("supply.py", "corpus.py", "data.py", "keys.py", "tokenizer.py", "publish.py")
log = logging.getLogger("vesper.supply")


@dataclass(frozen=True)
class SupplyInputs:
    """Checked counting identity and immutable tokenizer/membership snapshots for workers."""

    identity: dict[str, object]
    tokenizer: bytes
    split: Split


@dataclass(frozen=True)
class Counts:
    """Finite encoded tokens and nonempty eligible documents, with potential label positions."""

    tokens: int
    documents: int
    potential_labels: int
    max_document_tokens: int


@dataclass(frozen=True)
class Supply:
    """Complete per-file counts in identity order, bound to the checked receipt's digest."""

    identity: dict[str, object]
    counts: tuple[Counts, ...]
    sha256: str


def _plain(value: object) -> object:
    """Normalize tuples to JSON lists for exact saved-record comparison."""
    return cast("object", json.loads(json.dumps(value, allow_nan=False)))


def _parts(parts: Sequence[Component]) -> None:
    names = [c.name for c in parts]
    if not names or len(set(names)) != len(names) or any(not c.name or not c.paths for c in parts):
        msg = "supply needs uniquely named components, each with files"
        raise ValueError(msg)
    if any(not math.isfinite(c.share) or not 0 < c.share <= 1 for c in parts) or not math.isclose(
        math.fsum(c.share for c in parts), 1.0, rel_tol=0, abs_tol=1e-9
    ):
        msg = "component shares must be finite, positive and sum to one"
        raise ValueError(msg)


def prepare_supply(
    parts: Sequence[Component], tokenizer_bytes: bytes, split: Split, finalization_sha256: str
) -> SupplyInputs:
    """Bind exactly the finite inventory and encoding, independently of run seed/share/budget.

    Recheck the finalization against its expected digest and the supplied membership snapshot;
    rehash its input files. Generation pins include manifest digests. Legacy files bind actual
    paths/hashes too. This reads bytes but never tokenizes the corpus. A changed identity needs
    another explicit count; unrelated report/commit changes do not invalidate source hashes.

    Raises:
        ValueError: on invalid components, a non-training split or any changed binding.
    """
    _parts(parts)
    if (
        split.held_out
        or not math.isfinite(split.held_out_fraction)
        or not (0 <= split.held_out_fraction <= 1)
    ):
        msg = "supply counts require a training split with finite fraction in [0, 1]"
        raise ValueError(msg)
    checked = load_finalization(split.finalization.receipt, sha256=finalization_sha256)
    if checked != split.finalization:
        msg = "supply finalization differs from its checked snapshot"
        raise ValueError(msg)
    check_inputs(checked, parts)
    return SupplyInputs(
        _inventory_identity(tokenizer_bytes, split, finalization_sha256), tokenizer_bytes, split
    )


def _inventory_identity(
    tokenizer_bytes: bytes, split: Split, finalization_sha256: str
) -> dict[str, object]:
    """Identity of the actual immutable membership/encoding values passed to workers."""
    checked = split.finalization
    dropped = SequenceDigest()
    for doc_id in sorted(checked.dropped):
        dropped.add(doc_id.encode("utf-8"))
    tokenizer = Tokenizer.parse(tokenizer_bytes)
    identity: dict[str, object] = {
        "files": _plain(checked.inputs),
        "finalization_sha256": finalization_sha256,
        "dropped_ids_sha256": dropped.hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "tokenizer_rule": tokenizer.rule,
        "held_out_fraction": split.held_out_fraction,
        "held_out": False,
        "split_rule": SPLIT_RULE,
        "counter_rule": _RULE,
        "counter_sources": {name: file_sha256(Path(__file__).with_name(name)) for name in _SOURCES},
        "dependencies": {name: version(name) for name in ("tokenizers", "pyarrow")},
        "python": platform.python_version(),
    }
    return identity


def _counts(value: object) -> Counts:
    """The one persisted-count boundary: exact fields, nonnegative integers, coherent totals."""
    fields = ("tokens", "documents", "potential_labels", "max_document_tokens")
    if not isinstance(value, dict):
        msg = "supply counts must be a mapping"
        raise ValueError(msg)  # noqa: TRY004
    record = cast("dict[str, object]", value)
    if set(record) != set(fields) or any(type(record[k]) is not int for k in fields):
        msg = "supply counts need exactly the four integer count fields"
        raise ValueError(msg)
    tokens, documents, labels, maximum = (cast("int", record[k]) for k in fields)
    if (
        min(tokens, documents, labels, maximum) < 0
        or labels > tokens
        or tokens - labels > documents
        or (tokens - labels) * maximum < tokens
        or maximum > tokens
        or (documents == 0 and tokens != 0)
        or (tokens == 0) != (maximum == 0)
        or (maximum > 0 and labels < maximum - 1)
        or tokens > documents * maximum
    ):
        msg = "inconsistent supply counts"
        raise ValueError(msg)
    return Counts(tokens, documents, labels, maximum)


def read_supply(path: Path, inputs: SupplyInputs) -> Supply:
    """Read one complete cache, verify its manifest and exact current input identity.

    Missing, partial, malformed, changed-input or corrupt evidence refuses; never zero supply.
    The caller obtains inputs through prepare_supply, including actual current-file hashes.

    Raises:
        ValueError: on a mismatched identity, malformed receipt or invalid counts.
        OSError: on missing or unreadable files.
    """
    manifest = cast("object", json.loads((path / "manifest.json").read_bytes()))
    if not isinstance(manifest, dict):
        msg = "supply manifest must be a mapping"
        raise ValueError(msg)  # noqa: TRY004
    digest = cast("dict[str, object]", manifest).get("supply.json")
    data = verified_bytes(path / "supply.json", digest, "supply receipt")
    record = cast("object", json.loads(data))
    if not isinstance(record, dict):
        msg = "supply receipt must be a mapping"
        raise ValueError(msg)  # noqa: TRY004
    item = cast("dict[str, object]", record)
    rows = item.get("counts")
    if (
        type(item.get("schema")) is not int
        or item["schema"] != _SCHEMA
        or item.get("complete") is not True
        or item.get("identity") != _plain(inputs.identity)
        or not isinstance(rows, list)
        or len(cast("list[object]", rows)) != len(inputs.split.finalization.inputs)
    ):
        msg = "supply identity/schema/completeness does not match these inputs"
        raise ValueError(msg)
    counts = tuple(_counts(row) for row in cast("list[object]", rows))
    return Supply(inputs.identity, counts, hashlib.sha256(data).hexdigest())


@dataclass
class _Counter:
    tokenizer: Tokenizer
    split: Split
    scratch: Path

    def file(self, entry: tuple[str, str, str, str | None]) -> Counts:
        _, source, digest, _ = entry
        tokens = documents = labels = maximum = 0
        with copied_snapshot(Path(source), digest, self.scratch) as snapshot:
            for raw in RawReader((snapshot,), seed=0).documents():
                if not raw.text or not keeps(raw, self.split):
                    continue
                length = self.tokenizer.encode(raw.text).numel()
                tokens += length
                documents += 1
                labels += max(length - 1, 0)
                maximum = max(maximum, length)
        return Counts(tokens, documents, labels, maximum)


_worker: _Counter | None = None


def _initialize(tokenizer: bytes, split: Split, scratch: Path) -> None:
    """One private tokenizer/drop set per spawned worker, not one copy per submitted file."""
    global _worker  # noqa: PLW0603  # isolated process initializer, never shared training state
    _worker = _Counter(Tokenizer.parse(tokenizer), split, scratch / str(os.getpid()))


def _count_file(entry: tuple[str, str, str, str | None]) -> Counts:
    if _worker is None:
        msg = "supply worker was not initialized"
        raise RuntimeError(msg)
    return _worker.file(entry)


def count_supply(inputs: SupplyInputs, out: Path, *, workers: int = 1) -> Supply:
    """Explicit finite count, complete immutable directory publication, bounded parallelism.

    Each worker holds a private verified file copy on disk and at most one row group plus its
    encoded document in memory. Peak temporary disk is the sum of concurrently counted file
    sizes. Workers each hold the tokenizer and drop set. No model, GPU or training traversal.
    Failure publishes no cache; an existing completed cache is never overwritten.

    Raises:
        ValueError: on workers below one, an existing destination or changed worker input.
        OSError: on insufficient disk or unreadable input. Partial owned staging is removed.
    """
    if type(workers) is not int or workers < 1 or out.exists() or out.is_symlink():
        msg = "supply needs positive workers and a new output directory"
        raise ValueError(msg)
    digest = inputs.identity.get("finalization_sha256")
    if (
        not isinstance(digest, str)
        or _inventory_identity(inputs.tokenizer, inputs.split, digest) != inputs.identity
        or load_finalization(inputs.split.finalization.receipt, sha256=digest)
        != inputs.split.finalization
        or inputs.split.held_out
    ):
        msg = "supply worker snapshots differ from their prepared identity"
        raise ValueError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = inputs.split.finalization.inputs
    with staged(out) as partial:
        partial.mkdir()
        with owned_staging(partial, "workers") as scratch:
            if workers == 1:
                counter = _Counter(Tokenizer.parse(inputs.tokenizer), inputs.split, scratch)
                counts = tuple(counter.file(entry) for entry in entries)
            else:
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(entries)),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_initialize,
                    initargs=(inputs.tokenizer, inputs.split, scratch),
                ) as pool:
                    counts = tuple(pool.map(_count_file, entries))
        record: dict[str, object] = {
            "schema": _SCHEMA,
            "complete": True,
            "identity": inputs.identity,
            "counts": [cast("dict[str, object]", asdict(c)) for c in counts],
            "environment": environment(),
        }
        write_json(partial / "supply.json", record)
        write_json(partial / "manifest.json", {"supply.json": file_sha256(partial / "supply.json")})
        # Read our own complete bytes before publishing, through the same consumer boundary.
        return read_supply(partial, inputs)


def _nominal_demand(share: float, total_tokens: int) -> float:
    """Refuse a demand outside finite floating-point reporting rather than emit Infinity."""
    msg = "nominal demand exceeds the finite reporting range"
    try:
        demand = share * total_tokens
    except OverflowError as exc:
        raise ValueError(msg) from exc
    if not math.isfinite(demand):
        raise ValueError(msg)
    return demand


def assess(
    supply: Supply | None, parts: Sequence[Component], total_tokens: int, *, cycle: bool = False
) -> dict[str, object]:
    """Whole-run nominal demand, never remaining inventory or a completion certificate.

    Seed, document order, buffers, successors, layout and step rounding are deliberately not
    simulated. A resumed run reports this SAME whole-run nominal comparison, not supply minus
    handed tokens. The caller binds this report to its checked run/checkpoint identity.
    Zero positive-share supply or an entirely zero-label mix is invalid; ordinary shortfalls
    only warn, including under cycling where the ratio describes corpus-equivalent exposure.

    Raises:
        ValueError: on invalid components/budget or supply for a different ordered inventory.
    """
    _parts(parts)
    if type(total_tokens) is not int or total_tokens < 1:
        msg = "supply demand needs a positive whole-run token budget"
        raise ValueError(msg)
    warnings: list[str] = []
    errors: list[str] = []
    rows: list[dict[str, object]] = []
    if supply is None:
        warnings.append("Training supply is unknown: no checked cache; no count was started.")
    else:
        files = cast("list[list[object]]", supply.identity["files"])
        expected = [(c.name, str(p), c.generation) for c in parts for p in c.paths]
        if [(row[0], row[1], row[3]) for row in files] != expected or len(files) != len(
            supply.counts
        ):
            msg = "supply identity does not match the assessment's ordered components"
            raise ValueError(msg)
        cursor = 0
        for component in parts:
            subset = supply.counts[cursor : cursor + len(component.paths)]
            cursor += len(subset)
            tokens = sum(c.tokens for c in subset)
            labels = sum(c.potential_labels for c in subset)
            demand = _nominal_demand(component.share, total_tokens)
            exposure = demand / tokens if tokens else None
            relation = "below" if tokens < demand else "above" if tokens > demand else "at"
            rows.append(
                {
                    "component": component.name,
                    "supply_tokens": tokens,
                    "documents": sum(c.documents for c in subset),
                    "potential_labels": labels,
                    "max_document_tokens": max(c.max_document_tokens for c in subset),
                    "share": component.share,
                    "nominal_demand": demand,
                    "nominal_exposure": exposure,
                    "supply_vs_nominal": relation,
                }
            )
            if tokens == 0:
                errors.append(f"{component.name}: zero training supply for a positive share")
            elif tokens < demand:
                warning = (
                    f"{component.name}: nominal demand {demand:g} exceeds finite supply {tokens}; "
                )
                warning += f"nominal exposure {exposure:g}; completion remains unassessed."
                warnings.append(warning)
            if labels == 0:
                warnings.append(f"{component.name}: no potential within-document loss positions")
        if sum(c.potential_labels for c in supply.counts) == 0:
            errors.append("The entire mix has no potential within-document loss positions")
    return {
        "status": "unknown" if supply is None else "assessed",
        "valid": None if supply is None else not errors,
        "completion": "unassessed",
        "demand_scope": "configured-whole-run",
        "total_tokens": total_tokens,
        "component_mode": "cycling" if cycle else "finite",
        "supply_sha256": None if supply is None else supply.sha256,
        "components": rows,
        "warnings": warnings,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Count explicitly from selected or legacy components; no training or implicit launch."""
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "finalization", "tokenizer", "out"):
        _ = parser.add_argument(f"--{option}", type=Path, required=True)
    for option in ("data", "web", "mix"):
        _ = parser.add_argument(f"--{option}", type=Path, default=None)
    _ = parser.add_argument("--legacy-layout", action="store_true")
    _ = parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    try:
        cfg = load(cast("Path", args.config))
        parts = components_for(
            data=cast("Path | None", args.data),
            legacy_layout=cast("bool", args.legacy_layout),
            web=cast("Path | None", args.web),
            mix=cast("Path | None", args.mix),
        )
        tokenizer, _ = hashed_bytes(cast("Path", args.tokenizer))
        receipt = cast("Path", args.finalization)
        _, digest = hashed_bytes(receipt)
        split = Split(load_finalization(receipt, sha256=digest), cfg.training.held_out_fraction)
        inputs = prepare_supply(parts, tokenizer, split, digest)
        result = count_supply(inputs, cast("Path", args.out), workers=cast("int", args.workers))
    except (OSError, ValueError, TypeError, RuntimeError):
        log.exception("supply count failed; no complete cache published")
        return 1
    log.info("complete supply %s at %s", result.sha256, cast("Path", args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
