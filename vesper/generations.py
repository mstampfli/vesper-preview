"""Acquisition outputs as immutable generations (docs/reviews/2026-10-15-generations.md).

A generation is a directory under ``data/<kind>/`` that is complete or absent: one
``MANIFEST.json`` naming, in order, the logical acquisition units (a shard, a title selection)
with each unit's parquet inputs (what a reader opens) and its artifacts (receipts, listings:
hashed and bound, never read as data), the parent generation it extends, and the provenance of
the run that published it. A generation is never modified after publication; more units make a
new generation that hard-links the parent's files. ``validate_generation`` is the one boundary at
which names, containment, symlinks, duplicates and hashes are checked, before publication and
whenever a generation is read. The selection manifest that chooses the generations a new run
reads, and the pinning of generations in finalization receipts and checkpoints, are the next
rows of this module.
"""

import argparse
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO, cast

from vesper.keys import file_sha256
from vesper.provenance import environment, write_json
from vesper.publish import contained, publish, staged

MANIFEST = "MANIFEST.json"
MANIFEST_SCHEMA = 1
_INPUT_SUFFIX = ".parquet"


@dataclass(frozen=True)
class Unit:
    """One logical acquisition unit: its name and its files, by relative name in the generation.

    ``inputs`` are the parquet files a reader opens, in reading order; ``artifacts`` the receipts
    and listings bound to them.
    """

    name: str
    inputs: tuple[str, ...]
    artifacts: tuple[str, ...]


@dataclass(frozen=True)
class FileRecord:
    """A file of the generation with its size and sha256."""

    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Generation:
    """A validated generation: where it is, what it holds, where it came from."""

    kind: str
    identity: str
    path: Path
    units: tuple[Unit, ...]
    files: tuple[FileRecord, ...]
    parent: tuple[str, str] | None
    provenance: dict[str, str]
    manifest_sha256: str

    @property
    def inputs(self) -> tuple[Path, ...]:
        """The parquet inputs in reading order: unit order, then each unit's own order."""
        return tuple(self.path / name for unit in self.units for name in unit.inputs)


def publish_generation(
    kind: str,
    staging: Path,
    data: Path,
    *,
    units: Sequence[Unit],
    parent: Generation | None = None,
    replaces: Sequence[str] = (),
) -> Generation:
    """Publish the files staged under ``staging`` as ``data/<kind>/gen-<stamp>``, or nothing.

    The units' files must already lie flat in ``staging``. With ``parent``, its units are inherited
    first in its order (their files hard-linked into ``staging``: same filesystem, no copy is ever
    started) except the names in ``replaces``, which the new units replace; a unit named in both
    without being replaced is a collision and refused, as is a unit in ``replaces`` the parent does
    not have. The manifest is written into ``staging``, the content validated as a generation,
    and the directory renamed onto its final name in one step; a failure leaves ``staging`` as
    it was (plus the links and the manifest) and publishes nothing. Returns the published
    generation, validated at its final path.

    Raises:
        ValueError: on an empty or malformed unit list, a collision, a file that fails
            validation, or a final name that already exists.
        OSError: when ``staging`` and ``data`` are not on one filesystem (the link or the rename
            cannot be atomic).
    """
    # every path-controlling value is checked before any path is formed or anything is written
    # (maintainer B's review of 5726c87: a kind of '../x' published outside the data root, a staged
    # MANIFEST.json symlink overwrote an outside file before validation)
    _check_name(kind, data)
    root = data.resolve()
    if staging.is_symlink() or not staging.is_dir():
        msg = f"{staging}: the staging must be a real directory"
        raise ValueError(msg)
    control = staging / MANIFEST
    if control.exists() or control.is_symlink():
        msg = f"{control}: a control file is already there; the staging is not this run's"
        raise ValueError(msg)
    if not units:
        msg = "a generation needs at least one unit"
        raise ValueError(msg)
    if parent is None and replaces:
        msg = "replaces needs a parent"
        raise ValueError(msg)
    if parent is not None:
        _check_parent(parent, kind, root)
    inherited = [] if parent is None else _inherit(staging, parent, units, replaces)
    all_units = (*inherited, *units)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    identity = f"gen-{stamp}"
    record = _manifest(kind, identity, all_units, staging, parent)
    write_json(control, record)
    try:
        _ = validate_generation(staging, expected_kind=kind, expected_identity=identity)
    except ValueError:
        control.unlink(missing_ok=True)  # ours, written just above; a retry must not trip on it
        raise
    kind_dir = root / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    if kind_dir.is_symlink():
        msg = f"{kind_dir}: the kind directory is a symlink; refusing to publish through it"
        raise ValueError(msg)
    final = kind_dir / identity
    if final.exists() or final.is_symlink():
        msg = f"{final}: a generation already exists there"
        raise ValueError(msg)
    publish(staging, final)
    return validate_generation(final, expected_kind=kind)


def _check_parent(parent: Generation, kind: str, root: Path) -> None:
    """The parent, as it is on disk now, is the one the caller holds and is of this kind.

    A ``Generation`` object can outlive a change to its directory; the inheritance links the
    bytes on disk, so they are validated again and compared with the object's manifest hash.

    Raises:
        ValueError: on another kind, a parent outside the data root, or a parent whose content
            no longer matches the object.
    """
    if parent.kind != kind:
        msg = f"the parent is a {parent.kind!r} generation; a {kind!r} generation cannot extend it"
        raise ValueError(msg)
    contained(parent.path, root)
    now = validate_generation(parent.path, expected_kind=kind)
    if now.manifest_sha256 != parent.manifest_sha256 or now.files != parent.files:
        msg = f"{parent.path}: the parent changed since it was read; read it again"
        raise ValueError(msg)


def _inherit(
    staging: Path, parent: Generation, units: Sequence[Unit], replaces: Sequence[str]
) -> list[Unit]:
    """Hard-link the parent's units that are not replaced into ``staging``; return them in order.

    Raises:
        ValueError: on a replaced unit the parent lacks, a unit in both without replacement, or
            a parent file that collides with a staged file.
        OSError: when the link crosses filesystems.
    """
    replaced = set(replaces)
    known = {unit.name for unit in parent.units}
    if replaced - known:
        msg = f"replaces names units the parent does not have: {sorted(replaced - known)}"
        raise ValueError(msg)
    collisions = (known & {unit.name for unit in units}) - replaced
    if collisions:
        msg = f"units {sorted(collisions)} exist in the parent and are not replaced"
        raise ValueError(msg)
    inherited: list[Unit] = []
    for unit in parent.units:
        if unit.name in replaced:
            continue
        for name in (*unit.inputs, *unit.artifacts):
            target = staging / name
            if target.exists() or target.is_symlink():
                msg = f"{target}: the parent's file collides with a staged file"
                raise ValueError(msg)
            os.link(parent.path / name, target)
        inherited.append(unit)
    return inherited


def validate_generation(
    path: Path, *, expected_kind: str | None = None, expected_identity: str | None = None
) -> Generation:
    """The one boundary: read the manifest and check the directory is exactly what it says.

    The directory is a real directory (not a symlink) and the manifest a real file, read once:
    the bytes hashed are the bytes parsed. The manifest's identity is the directory's name and
    its kind the parent directory's name, unless the caller names what to expect (a staging
    directory before publication). Every listed name is a plain file name (no separator, no
    ``..``, not absolute), exists as a regular file that is not a symlink and resolves inside
    ``path``, has the recorded size and sha256, is listed once, and inputs end in ``.parquet``;
    every file in the directory other than the manifest is listed. Returns the generation with
    the manifest's sha256.

    Raises:
        ValueError: naming the first violation, or a manifest of another schema, kind or
            identity.
    """
    if path.is_symlink() or not path.is_dir():
        msg = f"{path}: not a real directory; not a generation"
        raise ValueError(msg)
    manifest_path = path / MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        msg = f"{path}: no regular {MANIFEST}; not a generation"
        raise ValueError(msg)
    raw = manifest_path.read_bytes()
    record = cast("dict[str, object]", json.loads(raw.decode("utf-8")))
    if record.get("schema") != MANIFEST_SCHEMA:
        msg = f"{manifest_path}: schema {record.get('schema')!r}, this code reads {MANIFEST_SCHEMA}"
        raise ValueError(msg)
    kind = cast("str", record["kind"])
    _check_name(kind, manifest_path)
    kind_expected = path.parent.name if expected_kind is None else expected_kind
    if kind != kind_expected:
        msg = f"{manifest_path}: a {kind!r} generation, expected {kind_expected!r}"
        raise ValueError(msg)
    identity = cast("str", record["identity"])
    _check_name(identity, manifest_path)
    identity_expected = path.name if expected_identity is None else expected_identity
    if identity != identity_expected:
        msg = (
            f"{manifest_path}: identity {identity!r} is not this directory's {identity_expected!r}"
        )
        raise ValueError(msg)
    units = tuple(
        Unit(
            name=cast("str", u["name"]),
            inputs=tuple(cast("list[str]", u["inputs"])),
            artifacts=tuple(cast("list[str]", u["artifacts"])),
        )
        for u in cast("list[dict[str, object]]", record["units"])
    )
    files = tuple(
        FileRecord(
            name=cast("str", f["name"]),
            bytes=cast("int", f["bytes"]),
            sha256=cast("str", f["sha256"]),
        )
        for f in cast("list[dict[str, object]]", record["files"])
    )
    _check_units(units, files, manifest_path)
    _check_files(files, path, manifest_path)
    parent_record = cast("dict[str, str] | None", record.get("parent"))
    return Generation(
        kind=kind,
        identity=identity,
        path=path,
        units=units,
        files=files,
        parent=None
        if parent_record is None
        else (parent_record["identity"], parent_record["manifest_sha256"]),
        provenance=dict(cast("dict[str, str]", record["provenance"])),
        manifest_sha256=hashlib.sha256(raw).hexdigest(),  # the bytes parsed above
    )


def _check_units(units: Sequence[Unit], files: Sequence[FileRecord], manifest_path: Path) -> None:
    """Units named once, non-empty, their files exactly the file list, inputs parquet."""
    names = [f.name for f in files]
    if len(set(names)) != len(names):
        msg = f"{manifest_path}: a file is listed twice"
        raise ValueError(msg)
    unit_names = [u.name for u in units]
    if len(set(unit_names)) != len(unit_names) or not units:
        msg = f"{manifest_path}: units must be non-empty and named once each"
        raise ValueError(msg)
    referenced = [name for u in units for name in (*u.inputs, *u.artifacts)]
    if sorted(referenced) != sorted(names):
        msg = f"{manifest_path}: the units' files and the file list differ"
        raise ValueError(msg)
    for u in units:
        for name in u.inputs:
            if not name.endswith(_INPUT_SUFFIX):
                msg = f"{manifest_path}: input {name!r} of unit {u.name!r} is not a parquet file"
                raise ValueError(msg)


def _check_files(files: Sequence[FileRecord], path: Path, manifest_path: Path) -> None:
    """Every listed file is a plain name and a regular file inside ``path`` with its hash.

    Nothing else lies in the directory.
    """
    inside = path.resolve()
    for f in files:
        _check_name(f.name, manifest_path)
        file = path / f.name
        if file.is_symlink() or not file.is_file():
            msg = f"{file}: missing, not a regular file, or a symlink"
            raise ValueError(msg)
        contained(file, inside)
        if file.stat().st_size != f.bytes or file_sha256(file) != f.sha256:
            msg = f"{file}: size or sha256 differ from the manifest"
            raise ValueError(msg)
    names = {f.name for f in files}
    present = {p.name for p in path.iterdir() if p.name != MANIFEST}
    if present != names:
        msg = f"{path}: files not in the manifest: {sorted(present - names)}"
        raise ValueError(msg)


def _manifest(
    kind: str, identity: str, units: Sequence[Unit], staging: Path, parent: Generation | None
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for unit in units:
        for name in (*unit.inputs, *unit.artifacts):
            _check_name(name, staging)
            file = staging / name
            if file.is_symlink() or not file.is_file():
                msg = f"{file}: unit {unit.name!r} names a file that is missing or a symlink"
                raise ValueError(msg)
            files.append({"name": name, "bytes": file.stat().st_size, "sha256": file_sha256(file)})
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": kind,
        "identity": identity,
        "published_at_utc": datetime.now(UTC).isoformat(),
        "units": [cast("dict[str, object]", asdict(unit)) for unit in units],
        "files": files,
        "parent": None
        if parent is None
        else {"identity": parent.identity, "manifest_sha256": parent.manifest_sha256},
        "provenance": environment(),
    }


def _check_name(name: str, where: Path) -> None:
    """A plain file name inside the generation: no separator, no ``..``, not absolute, not empty."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name or name == MANIFEST:
        msg = f"{where}: {name!r} is not a plain file name inside the generation"
        raise ValueError(msg)


SELECTION = "SELECTION.json"
SELECTION_SCHEMA = 1


@dataclass(frozen=True)
class Reference:
    """What a selection, a receipt or a recipe stores about a generation: enough to pin it."""

    kind: str
    identity: str
    manifest_sha256: str

    @property
    def pin(self) -> str:
        """The one-string form a component, a receipt and a recipe carry: kind/identity@sha256."""
        return f"{self.kind}/{self.identity}@{self.manifest_sha256}"


@dataclass(frozen=True)
class Selection:
    """The corpus selection a new run reads: one reference per kind, chosen in one step."""

    references: tuple[Reference, ...]
    previous_sha256: str | None
    selected_at_utc: str
    provenance: dict[str, str]
    sha256: str


def reference(generation: Generation) -> Reference:
    """The reference that pins ``generation``."""
    return Reference(generation.kind, generation.identity, generation.manifest_sha256)


def pinned(data: Path, ref: Reference) -> Generation:
    """The generation a stored reference names, validated and with the recorded manifest hash.

    A resume resolves its pinned references through this, whatever the current selection says.

    Raises:
        ValueError: when the generation is missing, fails validation, or its manifest hash is not
            the recorded one (the pinned content is not what was pinned).
    """
    _check_name(ref.kind, data)
    _check_name(ref.identity, data / ref.kind)
    root = data.resolve()
    path = root / ref.kind / ref.identity
    if path.is_symlink() or not path.is_dir():
        msg = f"{path}: the pinned generation is missing or is a symlink"
        raise ValueError(msg)
    contained(path, root)  # the generation lies under the data root itself, not redirected
    generation = validate_generation(path, expected_kind=ref.kind)
    if generation.manifest_sha256 != ref.manifest_sha256:
        msg = f"{path}: manifest sha256 differs from the pinned {ref.manifest_sha256[:12]}"
        raise ValueError(msg)
    return generation


def select(data: Path, references: Sequence[Reference], *, replacing: str | None) -> Selection:
    """Choose the generations new runs read: one reference per kind, written in one rename.

    Every referenced generation is validated first (``pinned``); the kinds must be distinct. The
    selection records the sha256 of the selection it replaces: ``replacing`` is the current
    selection's hash as the operator saw it, or None when there is none, so a stale concurrent
    activation is refused instead of silently overwriting another operator's choice.

    Raises:
        ValueError: on a repeated kind, a reference that fails ``pinned``, or a ``replacing`` that
            is not the current selection's hash.
    """
    kinds = [r.kind for r in references]
    if len(set(kinds)) != len(kinds) or not references:
        msg = f"a selection names each kind once, got {kinds}"
        raise ValueError(msg)
    for ref in references:
        _ = pinned(data, ref)
    # compare and publish are one operation under the data root's writer lock: of two contenders
    # based on the same predecessor, the second finds another current hash and is refused
    # (maintainer B's review of 5726c87: both passed the check and one overwrote the other)
    with writer_lock(data):
        current_sha = _snapshot(data / SELECTION)[1] if (data / SELECTION).is_file() else None
        if replacing != current_sha:
            msg = (
                f"{data / SELECTION}: the current selection is {current_sha}, not {replacing}; read"
                " it again before selecting"
            )
            raise ValueError(msg)
        record: dict[str, object] = {
            "schema": SELECTION_SCHEMA,
            "references": [cast("dict[str, object]", asdict(r)) for r in references],
            "previous_sha256": current_sha,
            "selected_at_utc": datetime.now(UTC).isoformat(),
            "provenance": environment(),
        }
        with staged(data / SELECTION) as partial:
            write_json(partial, record)
        return selection(data)  # read back under the lock: the selection this call committed


@contextmanager
def writer_lock(data: Path) -> Generator[None, None, None]:
    """The data root's writer lock (``.selection.lock``), exclusive for the body.

    ``select`` holds it across its compare and publish; an operator tool that must keep the
    selection still while it works holds it the same way. Advisory (``flock``), per data root.
    """
    data.mkdir(parents=True, exist_ok=True)
    fd = os.open(data / ".selection.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _snapshot(path: Path) -> tuple[dict[str, object], str]:
    """One read of a control file: the record parsed from, and the sha256 of, the same bytes.

    Raises:
        ValueError: when the control file is a symlink.
    """
    if path.is_symlink():
        msg = f"{path}: a control file must not be a symlink"
        raise ValueError(msg)
    raw = path.read_bytes()
    return cast("dict[str, object]", json.loads(raw.decode("utf-8"))), hashlib.sha256(
        raw
    ).hexdigest()


def selection(data: Path) -> Selection:
    """The current selection, every referenced generation validated against its recorded hash.

    Raises:
        ValueError: when there is no selection (a missing selection is never a fallback to the
            files that happen to be there), the schema is another one, or a referenced generation
            fails ``pinned``.
    """
    path = data / SELECTION
    if not path.is_file():
        msg = f"{path}: no selection chooses the inputs (a legacy layout is an explicit mode)"
        raise ValueError(msg)
    record, sha = _snapshot(path)  # the hash is of the bytes parsed, whatever lands afterwards
    if record.get("schema") != SELECTION_SCHEMA:
        msg = f"{path}: schema {record.get('schema')!r}, this code reads {SELECTION_SCHEMA}"
        raise ValueError(msg)
    references = tuple(
        Reference(
            kind=cast("str", r["kind"]),
            identity=cast("str", r["identity"]),
            manifest_sha256=cast("str", r["manifest_sha256"]),
        )
        for r in cast("list[dict[str, object]]", record["references"])
    )
    for ref in references:
        _ = pinned(data, ref)
    return Selection(
        references=references,
        previous_sha256=cast("str | None", record.get("previous_sha256")),
        selected_at_utc=cast("str", record["selected_at_utc"]),
        provenance=dict(cast("dict[str, str]", record["provenance"])),
        sha256=sha,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesper.generations", description=(__doc__ or "").split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select", help="choose one generation per kind for new runs")
    _ = choose.add_argument("--data", type=Path, required=True)
    _ = choose.add_argument(
        "--ref",
        action="append",
        required=True,
        help="a pin kind/identity@sha256 as the driver printed it; one per kind",
    )
    _ = choose.add_argument(
        "--replacing",
        default=None,
        help="the current selection's sha256 as `show` printed it; omit for the first selection",
    )
    show = sub.add_parser(
        "show", help="print the current selection with every generation validated"
    )
    _ = show.add_argument("--data", type=Path, required=True)
    return parser


def parse_pin(pin: str) -> Reference:
    """``kind/identity@sha256`` back into a reference.

    Raises:
        ValueError: on another shape.
    """
    head, sep, digest = pin.partition("@")
    kind, slash, identity = head.partition("/")
    if not (sep and slash and kind and identity and len(digest) == 64):  # noqa: PLR2004
        msg = f"a pin is kind/identity@sha256, got {pin!r}"
        raise ValueError(msg)
    return Reference(kind=kind, identity=identity, manifest_sha256=digest)


def main(argv: Sequence[str] | None = None) -> int:
    """The operator's command line: ``select`` writes the selection, ``show`` prints it."""
    args = _parser().parse_args(argv)
    data = cast("Path", args.data)
    try:
        if cast("str", args.action) == "select":
            refs = [parse_pin(p) for p in cast("list[str]", args.ref)]
            chosen = select(data, refs, replacing=cast("str | None", args.replacing))
        else:
            chosen = selection(data)
    except ValueError as exc:
        _ = cast("TextIO", sys.stderr).write(f"refused: {exc}\n")
        return 1
    record = cast("dict[str, object]", asdict(chosen))
    _ = cast("TextIO", sys.stdout).write(json.dumps(record, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
