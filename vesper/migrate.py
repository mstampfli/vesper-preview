"""Bring today's flat data layout into generations without moving or deleting anything.

``python -m vesper.migrate --data ROOT --web DIR --mix DIR --legacy-finalization RECEIPT --out DIR``
prints its plan and stops; with ``--apply`` it carries the plan out. The plan is the exact set of
files a legacy run reads (``sample.components(web, mix)``, so nothing is guessed from what else
lies in a directory), each with its size and hash, the unit it becomes, the generation directory
that will hold it, and the selection that will be written. Applying it hard-links every file into
a staging of its own and publishes one generation per kind (``generations.publish_generation``:
one inode, no copy, the old path untouched), finalizes the corpus once more over the generations
(``corpus.finalize``), and compares the dropped set with the legacy finalization's: identical,
the selection is written and new runs read the generations; different, nothing is selected and
the difference is reported, since the same bytes under new names must decide the same membership.
The legacy files, receipts and finalization stay where they are; nothing is cleaned up. The
contract and its amendments: docs/reviews/2026-10-15-generations.md, Migration.
"""

import argparse
import json
import logging
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

from vesper.corpus import Component, Finalization, finalize, load_finalization
from vesper.generations import (
    SELECTION,
    Generation,
    Reference,
    Selection,
    Unit,
    parse_pin,
    pinned,
    publish_generation,
    reference,
    select,
    selection,
    writer_lock,
)
from vesper.keys import file_sha256
from vesper.provenance import environment, write_json
from vesper.publish import owned_staging, staged
from vesper.sample import components

log = logging.getLogger("vesper.migrate")


EXIT_COMMITTED_INCOMPLETE = 3  # the selection is written, its receipt is not complete: --recover
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class PlannedFile:
    """One legacy file and what becomes of it: the unit it forms and the artifact beside it.

    The artifact (an acquisition receipt) is bound by size and hash like the file itself: it is
    provenance that travels into the generation, so changed bytes after the plan are refused
    (maintainer B's review of bf92389).
    """

    path: str
    bytes: int
    sha256: str
    unit: str
    artifact: str | None
    artifact_bytes: int | None = None
    artifact_sha256: str | None = None


class ActivationUnrecordedError(RuntimeError):
    """The selection was written and the receipt's completion was not: committed, incomplete.

    The receipt says ``pending`` with the pins and the evidence; ``recover(receipt)`` validates
    them against the current selection and finishes the record without selecting again.
    """

    def __init__(self, receipt: Path, selection_sha256: str, generations: Sequence[str]) -> None:
        """The pending receipt, the committed selection's hash and the pins it names."""
        self.receipt: Path = receipt
        self.selection_sha256: str = selection_sha256
        self.generations: tuple[str, ...] = tuple(generations)
        msg = (
            f"{receipt}: the selection {selection_sha256[:12]} is committed, its receipt incomplete"
            f" (activation pending); finish it with: python -m vesper.migrate --recover {receipt}"
        )
        super().__init__(msg)


@dataclass(frozen=True)
class PlannedKind:
    """One component: its share, its files in reading order, and the generation it will become."""

    kind: str
    share: float
    files: tuple[PlannedFile, ...]
    generation_parent: str


@dataclass(frozen=True)
class Plan:
    """What the migration will do, written before anything is done and recorded afterwards."""

    data: str
    web: str
    mix: str
    legacy_finalization: str
    kinds: tuple[PlannedKind, ...]
    selection: str


@dataclass(frozen=True)
class Comparison:
    """The new finalization against the legacy one: the inputs it was taken over, and the drop set.

    ``inputs_match`` says both finalizations were taken over the same component-to-hash
    assignment (the paths differ by design); the dropped counts say what each decided. Equal drop
    sets over different inputs are a coincidence, not equivalence (maintainer B's review of
    e1273b2).
    """

    legacy_dropped: int
    migrated_dropped: int
    only_legacy: int
    only_migrated: int
    inputs_match: bool

    @property
    def identical(self) -> bool:
        """The same documents dropped over the same input bytes under the same assignment."""
        return self.inputs_match and self.only_legacy == 0 and self.only_migrated == 0


@dataclass(frozen=True)
class Migration:
    """What a migration did: the generations, the finalization, the comparison, the selection."""

    generations: tuple[Generation, ...]
    finalization: Finalization
    comparison: Comparison
    selection: Selection | None
    receipt: Path


def plan(data: Path, web: Path, mix: Path, legacy_finalization: Path) -> Plan:
    """The migration plan: the legacy layout's files as the legacy reader sees them.

    Raises:
        ValueError: when the data root already holds a selection (a migrated root is not migrated
            again), the legacy finalization receipt does not exist, or a component has no file.
    """
    if (data / SELECTION).exists():
        pending = "".join(f"; recover it with --recover {p}" for p in _pending_receipts(data))
        msg = f"{data / SELECTION}: this root already holds a selection; not migrating over it" + (
            f" (a migration receipt is still pending{pending})" if pending else ""
        )
        raise ValueError(msg)
    if not legacy_finalization.is_file():
        msg = f"{legacy_finalization}: the legacy finalization receipt does not exist"
        raise ValueError(msg)
    kinds: list[PlannedKind] = []
    for component in components(web, mix):
        files: list[PlannedFile] = []
        for path in component.paths:
            receipt = path.with_suffix(".receipt.json")
            has_artifact = receipt.is_file()
            files.append(
                PlannedFile(
                    path=str(path),
                    bytes=path.stat().st_size,
                    sha256=file_sha256(path),
                    unit=path.stem,
                    artifact=receipt.name if has_artifact else None,
                    artifact_bytes=receipt.stat().st_size if has_artifact else None,
                    artifact_sha256=file_sha256(receipt) if has_artifact else None,
                )
            )
        kinds.append(
            PlannedKind(
                kind=component.name,
                share=component.share,
                files=tuple(files),
                generation_parent=str(data / component.name),
            )
        )
    return Plan(
        data=str(data),
        web=str(web),
        mix=str(mix),
        legacy_finalization=str(legacy_finalization),
        kinds=tuple(kinds),
        selection=str(data / SELECTION),
    )


def apply(planned: Plan, *, out: Path) -> Migration:
    """Carry the plan out: link, publish, finalize, compare, and select only when identical.

    Every generation is published before the finalization starts (the slow step, about half an
    hour on the live corpus), so a difference leaves the generations in place for inspection and
    nothing selected. The migration receipt (``data/migrations/migrate-<stamp>.json``) records the
    plan, the pins, the finalization receipt, the comparison and the environment.

    Raises:
        ValueError: when the plan's root holds a selection by now, or publication refuses.
        OSError: when a file cannot be hard-linked into the data root (another filesystem; the
            migration never copies), with nothing published for that kind.
        ActivationUnrecordedError: when the selection was written and the receipt's completion
            failed (the receipt stays ``pending`` with the pins; ``recover`` finishes it).
    """
    data = Path(planned.data)
    if (data / SELECTION).exists():
        msg = f"{data / SELECTION}: a selection appeared since the plan; not migrating over it"
        raise ValueError(msg)
    legacy = load_finalization(Path(planned.legacy_finalization))
    _check_legacy_binding(planned, legacy)  # the same files, hashes and assignment, before any link
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    generations: list[Generation] = []
    parts: list[Component] = []
    for kind in planned.kinds:
        generation = _publish_kind(kind, data)
        _check_published(kind, generation)  # the bytes linked are the bytes planned
        generations.append(generation)
        parts.append(
            Component(
                kind.kind, generation.inputs, kind.share, generation=reference(generation).pin
            )
        )
        log.info("%s: published %s from %d files", kind.kind, generation.identity, len(kind.files))
    log.info("finalizing over the generations (the slow step)")
    migrated = finalize(parts, out)
    comparison = Comparison(
        legacy_dropped=len(legacy.dropped),
        migrated_dropped=len(migrated.dropped),
        only_legacy=len(legacy.dropped - migrated.dropped),
        only_migrated=len(migrated.dropped - legacy.dropped),
        inputs_match=_assignment(legacy) == _assignment(migrated),
    )
    refs: list[Reference] = [reference(g) for g in generations]
    receipt = data / "migrations" / f"migrate-{stamp}.json"
    record: dict[str, object] = {
        "migrated_at_utc": datetime.now(UTC).isoformat(),
        "plan": cast("dict[str, object]", asdict(planned)),
        "generations": [r.pin for r in refs],
        "finalization": str(migrated.receipt),
        "comparison": cast("dict[str, object]", asdict(comparison)),
        "activation": "pending" if comparison.identical else "refused",
        "selection_sha256": None,
        "environment": environment(),
    }
    # the intent and the evidence are durable before the selection is written, so a failure
    # after activation never leaves an unrecorded selection (maintainer B's review of e1273b2)
    _write_receipt(receipt, record)
    chosen: Selection | None = None
    if comparison.identical:
        chosen = select(data, refs, replacing=None)
        log.info("dropped sets identical (%d): selection written", comparison.legacy_dropped)
        record["activation"] = "selected"
        record["selection_sha256"] = chosen.sha256
        try:
            _write_receipt(receipt, record)
        except OSError as exc:
            raise ActivationUnrecordedError(receipt, chosen.sha256, [r.pin for r in refs]) from exc
    else:
        difference = cast("dict[str, object]", asdict(comparison))
        log.error("finalizations differ: %s; nothing selected", difference)
    return Migration(
        generations=tuple(generations),
        finalization=migrated,
        comparison=comparison,
        selection=chosen,
        receipt=receipt,
    )


def _check_legacy_binding(planned: Plan, legacy: Finalization) -> None:
    """The legacy finalization was taken over exactly the planned files: component, path, hash.

    Raises:
        ValueError: naming the first planned file the legacy finalization did not cover, or the
            first legacy input the plan does not name (a finalization over other inputs whose
            drop set happens to agree is not evidence; maintainer B's review of e1273b2).
    """
    planned_set = {(k.kind, f.path, f.sha256) for k in planned.kinds for f in k.files}
    legacy_set = {(component, path, sha256) for component, path, sha256, _ in legacy.inputs}
    for component, path, sha256 in sorted(planned_set - legacy_set):
        msg = (
            f"{legacy.receipt}: not taken over the planned {component} file {path} ({sha256[:12]});"
            " the plan and the legacy finalization must name the same files and hashes"
        )
        raise ValueError(msg)
    for component, path, sha256 in sorted(legacy_set - planned_set):
        msg = (
            f"{legacy.receipt}: taken over {component} file {path} ({sha256[:12]}) that the plan"
            " does not name; the plan and the legacy finalization must name the same files"
        )
        raise ValueError(msg)


def _check_published(kind: PlannedKind, generation: Generation) -> None:
    """The published generation holds exactly the planned bytes under the planned names.

    Raises:
        ValueError: naming the first file whose size or hash differs from the plan (a source
            changed between the plan and the links), or a planned artifact the generation lacks.
    """
    records = {f.name: f for f in generation.files}
    for file in kind.files:
        name = Path(file.path).name
        record = records.get(name)
        if record is None or record.bytes != file.bytes or record.sha256 != file.sha256:
            msg = (
                f"{generation.path / name}: differs from the plan ({file.bytes} bytes,"
                f" {file.sha256[:12]}); the source changed after planning, nothing selected"
            )
            raise ValueError(msg)
        if file.artifact is None:
            continue
        artifact = records.get(file.artifact)
        if artifact is None:
            msg = (
                f"{generation.path}: the planned artifact {file.artifact} is not in the generation"
            )
            raise ValueError(msg)
        if artifact.bytes != file.artifact_bytes or artifact.sha256 != file.artifact_sha256:
            planned_hash = (file.artifact_sha256 or "unbound")[:12]
            msg = (
                f"{generation.path}: the artifact {file.artifact} differs from the plan"
                f" ({file.artifact_bytes} bytes, {planned_hash}); its provenance changed after"
                " planning, nothing selected"
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class Recovery:
    """What ``recover`` established about a receipt.

    ``selection_sha256`` is the selection that receipt records as its own activation, never
    simply whatever selection is current; ``completed_now`` says this call wrote the completion;
    ``current`` says that selection is still the current one (a later operator selection makes it
    False without making the receipt wrong).
    """

    receipt: Path
    selection_sha256: str
    completed_now: bool
    current: bool


def recover(receipt: Path) -> Recovery:
    """Finish a migration whose selection is committed and whose receipt is not (``pending``).

    Validates, under the data root's writer lock: the comparison the receipt records is identical
    (only then was anything selected); every pinned generation resolves with its recorded
    manifest hash; the recorded finalization verifies and was taken over exactly the pinned
    generations' files (component, resolved path, sha256 and pin of every input, against each
    generation's own manifest records: labels alone are not evidence, maintainer B's review of
    e51d5f5); and the current selection names exactly those pins and replaced nothing
    (``previous_sha256`` is None, as the migration's own ``select(..., replacing=None)`` wrote it).
    Then the receipt becomes ``selected`` with the current selection's hash and the recovery time.
    Nothing is selected again. A receipt already ``selected`` is reported with its own recorded
    selection, whether or not that is still the current one, and left as it is.

    Raises:
        ValueError: on a receipt that refused activation or has another phase, a selected receipt
            without a well-formed selection hash, a comparison that is not identical, a pin or
            finalization that does not verify, no current selection (the migration never
            activated; the receipt stays pending), or a current selection that is not the one
            this migration wrote (an intervening selection is never overwritten by a record).
    """
    record = cast("dict[str, object]", json.loads(receipt.read_text(encoding="utf-8")))
    data = Path(cast("str", cast("dict[str, object]", record["plan"])["data"]))
    phase = record.get("activation")
    if phase == "refused":
        msg = f"{receipt}: this migration refused activation; there is nothing to recover"
        raise ValueError(msg)
    if phase == "selected":
        recorded = record.get("selection_sha256")
        if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
            msg = f"{receipt}: a selected receipt without a well-formed selection hash"
            raise ValueError(msg)
        now = selection(data).sha256 if (data / SELECTION).is_file() else None
        return Recovery(receipt, recorded, completed_now=False, current=now == recorded)
    if phase != "pending":
        msg = f"{receipt}: activation {phase!r} is not one this command knows"
        raise ValueError(msg)
    pins = _check_pending_evidence(receipt, record, data)
    with writer_lock(data):
        if not (data / SELECTION).is_file():
            msg = (
                f"{receipt}: no selection at {data / SELECTION}; this migration never activated,"
                " the receipt stays pending"
            )
            raise ValueError(msg)
        current = selection(data)
        ours = sorted(r.pin for r in current.references) == sorted(pins)
        if not ours or current.previous_sha256 is not None:
            msg = (
                f"{receipt}: the current selection {current.sha256[:12]} is not the selection this"
                " migration wrote; the receipt stays pending and the selection is not touched"
            )
            raise ValueError(msg)
        record["activation"] = "selected"
        record["selection_sha256"] = current.sha256
        record["recovered_at_utc"] = datetime.now(UTC).isoformat()
        _write_receipt(receipt, record)
    log.info("receipt %s completed for selection %s", receipt, current.sha256)
    return Recovery(receipt, current.sha256, completed_now=True, current=True)


def _check_pending_evidence(receipt: Path, record: dict[str, object], data: Path) -> list[str]:
    """What a pending receipt must prove before ``recover`` completes it; returns its pins.

    The recorded comparison is identical; every pin resolves with its manifest hash; the recorded
    finalization verifies and its inputs are exactly the pinned generations' files, component,
    resolved path, sha256 and pin each, against the generations' own manifest records; and its
    drop count is the one the comparison recorded.

    Raises:
        ValueError: naming the first piece of evidence that does not hold.
    """
    comparison = cast("dict[str, object]", record["comparison"])
    identical = (
        comparison.get("inputs_match") is True
        and comparison.get("only_legacy") == 0
        and comparison.get("only_migrated") == 0
    )
    if not identical:
        msg = f"{receipt}: the recorded comparison is not identical; nothing was to be selected"
        raise ValueError(msg)
    pins = [cast("str", p) for p in cast("list[object]", record["generations"])]
    expected: set[tuple[str, str, str, str | None]] = set()
    for pin in pins:
        ref = parse_pin(pin)
        generation = pinned(data, ref)
        hashes = {f.name: f.sha256 for f in generation.files}
        expected |= {
            (ref.kind, str(path.resolve()), hashes[path.name], pin) for path in generation.inputs
        }
    finalization = load_finalization(Path(cast("str", record["finalization"])))
    taken_over = {
        (component, str(Path(path).resolve()), sha256, generation)
        for component, path, sha256, generation in finalization.inputs
    }
    if taken_over != expected or len(finalization.inputs) != len(expected):
        msg = (
            f"{receipt}: the finalization {finalization.receipt} is not taken over the pinned"
            " generations' files (component, path, sha256 and pin of every input)"
        )
        raise ValueError(msg)
    if len(finalization.dropped) != comparison.get("migrated_dropped"):
        msg = f"{receipt}: the finalization's drop set is not the one the comparison recorded"
        raise ValueError(msg)
    return pins


def _pending_receipts(data: Path) -> list[Path]:
    """The migration receipts under ``data/migrations`` still recording a pending activation."""
    found: list[Path] = []
    for path in sorted((data / "migrations").glob("migrate-*.json")):
        try:
            record = cast("dict[str, object]", json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if record.get("activation") == "pending":
            found.append(path)
    return found


def _assignment(finalization: Finalization) -> frozenset[tuple[str, str]]:
    """What a finalization was taken over, paths aside: each component's input hashes."""
    return frozenset((component, sha256) for component, _, sha256, _ in finalization.inputs)


def _write_receipt(path: Path, record: dict[str, object]) -> None:
    """The migration receipt, complete or absent (``publish.staged``), rewritten as it advances."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with staged(path) as partial:
        write_json(partial, record)


def _publish_kind(kind: PlannedKind, data: Path) -> Generation:
    """Hard-link one kind's files into a staging of its own and publish it as a generation.

    The staging is ``publish.owned_staging``: exclusive, released however this ends (a failed
    link leaves nothing for this kind; releasing removes links only, the sources are untouched).
    """
    with owned_staging(data, f"migrate-{kind.kind}") as staging:
        units: list[Unit] = []
        for file in kind.files:
            source = Path(file.path)
            os.link(source, staging / source.name)
            artifacts: tuple[str, ...] = ()
            if file.artifact is not None:
                os.link(source.with_name(file.artifact), staging / file.artifact)
                artifacts = (file.artifact,)
            units.append(Unit(file.unit, (source.name,), artifacts))
        return publish_generation(kind.kind, staging, data, units=units)


_PLAN_ARGUMENTS = ("data", "web", "mix", "legacy_finalization", "out")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesper.migrate", description=(__doc__ or "").split("\n\n")[0]
    )
    _ = parser.add_argument("--data", type=Path, help="the data root (generations)")
    _ = parser.add_argument("--web", type=Path, help="the flat web directory")
    _ = parser.add_argument("--mix", type=Path, help="the flat mix directory")
    _ = parser.add_argument("--legacy-finalization", type=Path)
    _ = parser.add_argument("--out", type=Path, help="where the new finalization goes")
    _ = parser.add_argument("--apply", action="store_true", help="carry the printed plan out")
    _ = parser.add_argument(
        "--recover",
        type=Path,
        metavar="RECEIPT",
        help="finish a pending receipt whose selection is committed (takes no other option)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: the plan to stdout, and with ``--apply`` the result.

    Exit 1 on a refusal or on a difference in the dropped sets (nothing selected then), and
    ``EXIT_COMMITTED_INCOMPLETE`` when the selection is written and its receipt is not (the
    outcome names the receipt, the selection and the recovery command). ``--recover RECEIPT``
    finishes such a receipt (``recover``).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = _parser()
    args = parser.parse_args(argv)
    out = cast("TextIO", sys.stdout)
    given = [name for name in _PLAN_ARGUMENTS if cast("object", getattr(args, name)) is not None]
    receipt = cast("Path | None", args.recover)
    if receipt is not None:
        if given or cast("bool", args.apply):
            parser.error("--recover takes no other option")
        try:
            recovered = recover(receipt)
        except (OSError, ValueError):
            log.exception("recovery refused")
            return 1
        done: dict[str, object] = {
            "activation": "selected",
            "selection_sha256": recovered.selection_sha256,
            "completed_now": recovered.completed_now,
            "current": recovered.current,
            "receipt": str(receipt),
        }
        _ = out.write(json.dumps(done, indent=2) + "\n")
        return 0
    missing = [f"--{name.replace('_', '-')}" for name in _PLAN_ARGUMENTS if name not in given]
    if missing:
        parser.error(f"a plan needs {', '.join(missing)}")
    try:
        planned = plan(
            cast("Path", args.data),
            cast("Path", args.web),
            cast("Path", args.mix),
            cast("Path", args.legacy_finalization),
        )
        _ = out.write(json.dumps(cast("dict[str, object]", asdict(planned)), indent=2) + "\n")
        if not cast("bool", args.apply):
            return 0
        migration = apply(planned, out=cast("Path", args.out))
    except ActivationUnrecordedError as exc:
        log.exception("selection committed, receipt incomplete")
        outcome: dict[str, object] = {
            "activation": "committed-incomplete",
            "generations": list(exc.generations),
            "selection_sha256": exc.selection_sha256,
            "receipt": str(exc.receipt),
            "error": str(exc.__cause__),
            "recovery": f"python -m vesper.migrate --recover {exc.receipt}",
        }
        _ = out.write(json.dumps(outcome, indent=2) + "\n")
        return EXIT_COMMITTED_INCOMPLETE
    except (OSError, RuntimeError, ValueError):
        log.exception("stopped")
        return 1
    pins: list[str] = [reference(g).pin for g in migration.generations]
    result: dict[str, object] = {
        "generations": pins,
        "finalization": str(migration.finalization.receipt),
        "comparison": cast("dict[str, object]", asdict(migration.comparison)),
        "activation": "selected" if migration.selection is not None else "refused",
        "selection_sha256": None if migration.selection is None else migration.selection.sha256,
        "receipt": str(migration.receipt),
    }
    _ = out.write(json.dumps(result, indent=2) + "\n")
    return 0 if migration.comparison.identical else 1


if __name__ == "__main__":
    sys.exit(main())
