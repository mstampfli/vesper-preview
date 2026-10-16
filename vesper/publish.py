"""Atomic publication of a file or a directory by rename.

A reader of a receipt, a dropped-id file or a checkpoint must find it complete or not at all.
``staged(final)`` yields a path this operation owns; the caller writes there; a normal exit
renames it onto ``final`` in one step and an exception removes it, so nothing partial ever
carries the final name (maintainer B's reviews of 2026-10-15: a checkpoint written into its final
directory left a partial one a retry refused; a finalization receipt was readable while empty;
a partial named by destination and pid let a nested call delete the outer call's file and
published a crashed run's leftover).
"""

import shutil
import tempfile
from collections.abc import Generator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path


@contextmanager
def staged(final: Path) -> Generator[Path, None, None]:
    """PRIMITIVE: write to the yielded path; it is published onto ``final`` by one rename.

    For a file or a directory that readers must see complete. Each call owns a fresh staging
    directory beside ``final`` (``.<name>.<random>.staging``, created exclusively) and yields
    the initially absent child ``<name>`` inside it; the caller writes the file or makes the
    directory there. On a normal exit the child is renamed onto ``final`` in one step
    (replacing a reserved empty file, never a non-empty directory) and the staging directory is
    removed. On an exception only this call's staging directory is removed and ``final`` is
    untouched; the exception propagates. Two calls on one destination, nested or concurrent,
    never touch each other's files, and a leftover of a crashed run is never published.

    Not for: a destination updated in place while readers hold it (that is a generation with an
    active manifest, ARCHITECTURE.md "Not yet written"), nor for making the reservation itself.

    Raises:
        ValueError: when the body wrote nothing at the yielded path (nothing to publish).
    """
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", suffix=".staging", dir=final.parent))
    partial = staging / final.name
    try:
        yield partial
        publish(partial, final)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@contextmanager
def owned_staging(root: Path, label: str) -> Generator[Path, None, None]:
    """PRIMITIVE: one operation's own staging directory under ``root/staging``, for its lifetime.

    Allocated exclusively (``mkdtemp`` with the label and the clock as the prefix: two nested or
    concurrent operations with one label under one clock tick get two directories, never one),
    made readable like a directory other readers will open (it may become a generation by
    rename), yielded, and released on exit however the body ends: the directory is removed if it
    is still there (a publication that renamed it away leaves nothing to remove) and the shared
    parent only when it is empty; a parent another owner refilled or removed in between is left
    alone. Nothing is raised from the exit, so the body's exception is the one that propagates
    (maintainer B's review of e39edd3: a label-plus-clock name let a nested operation delete the
    outer one's files, and a parent vanishing between two checks raised over the body's error).

    Not for: a destination readers open while it is written (``staged`` publishes onto a final
    name). The caller publishes the staging itself or lets it be released.
    """
    parent = root / "staging"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(tempfile.mkdtemp(prefix=f"{label}-{stamp}-", dir=parent))
    path.chmod(0o755)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
        with suppress(OSError):
            if not any(parent.iterdir()):
                parent.rmdir()


def publish(partial: Path, final: Path) -> None:
    """PRIMITIVE: the rename step alone, for a caller that already owns its staging.

    ``staged`` uses it; ``generations.publish_generation`` renames the acquisition's own staging
    directory with it. One rename on one filesystem: a file replaces an existing file, a
    directory only an absent or empty one.

    Raises:
        ValueError: when nothing exists at ``partial``.
        OSError: when the rename cannot be done in one step (another filesystem, a non-empty
            directory at ``final``).
    """
    if not partial.exists() and not partial.is_symlink():
        msg = f"{partial}: nothing was written there, nothing to publish at {final}"
        raise ValueError(msg)
    _ = partial.replace(final)


def contained(path: Path, inside: Path) -> None:
    """PRIMITIVE: ``path`` resolves, symlinks followed, to a place under ``inside``.

    The lexical checks on a name do not constrain where the filesystem takes it; this is the
    check every path that came from outside (an href, a manifest entry) passes before it is read
    or written. ``inside`` must already be resolved.

    Raises:
        ValueError: when the resolved path leaves ``inside``.
    """
    if not path.resolve().is_relative_to(inside):
        msg = f"{path}: resolves outside {inside}"
        raise ValueError(msg)
