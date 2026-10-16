"""The corpus's identity primitives: the dedup normalization and key, and a bounded file hash.

Every place that decides whether two documents are the same, or attests to a file's content,
calls one of these. The dedup key is versioned so a change of rule rebuilds the keys instead of
silently mixing two rules (docs/reviews/2026-10-15-mixture-sources.md, Deduplication). The module
imports nothing from the package, so the reader and the collector can both depend on it.
"""

import hashlib
from pathlib import Path
from typing import final

_CHUNK = 1 << 20


def normalize_text_v2(text: str) -> str:
    """PRIMITIVE: version 2 of the text dedup normalization, for the key only.

    Exactly: the text is split on the line feed; in every line, runs of spaces and tabs collapse
    to one space and leading and trailing Unicode whitespace is removed (``str.strip``: so a
    carriage return before the line feed goes, CRLF endings become LF, and a leading or trailing
    no-break space goes); empty lines stay; the lines are joined with the line feed. Version 1
    kept a leading no-break space (its leading trim was the space split alone, its trailing trim
    ``rstrip``); the collector review's second round found the two implementations of "v1"
    disagreed on that character, so the clean rule is version 2 and version 1 is retired
    (2026-10-15). The training text is never rewritten by it. Not for code, whose key is the
    exact text.
    """
    return "\n".join(
        " ".join(part for part in line.replace("\t", " ").split(" ") if part).strip()
        for line in text.split("\n")
    )


TEXT_KEY_VERSION = "text-v2"
CODE_KEY_VERSION = "code-v1"


def dedup_key(text: str, kind: str) -> tuple[str, str]:
    """PRIMITIVE: the versioned dedup key of ``text``, ``(version, sha256 hex)``.

    ``"code"`` hashes the exact UTF-8 bytes (``code-v1``); ``"text"`` hashes
    ``normalize_text_v2`` (``text-v2``). The one place a dedup hash is computed. Keys of
    different versions never match each other: text sources deduplicate against each other,
    code among code; byte-identical content in a code file and a text document is two documents.

    Raises:
        ValueError: on a kind other than ``"code"`` or ``"text"``.
    """
    if kind == "code":
        return CODE_KEY_VERSION, hashlib.sha256(text.encode("utf-8")).hexdigest()
    if kind == "text":
        digest = hashlib.sha256(normalize_text_v2(text).encode("utf-8")).hexdigest()
        return TEXT_KEY_VERSION, digest
    msg = f"kind must be 'code' or 'text', got {kind!r}"
    raise ValueError(msg)


def file_sha256(path: Path) -> str:
    """PRIMITIVE: the SHA-256 of a file read in one-mebibyte chunks; never the whole file at once.

    For every receipt's input and output hash. Not for in-memory strings (``hashlib`` directly).
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@final
class SequenceDigest:
    """PRIMITIVE: a SHA-256 chain over a sequence of byte strings, each length-prefixed.

    For the identity of what a run or a sample consumed, in order: the ids a mixer handed out,
    the texts a tokenizer trainer read. Each item folds into the state as
    ``sha256(state || len(item) || item)``, so the hex state is the digest and a digest can be
    continued from its hex (a checkpoint stores it, a resumed run goes on from it). The length
    prefix makes the framing unambiguous, so ``add(b"ab"); add(b"c")`` and
    ``add(b"a"); add(b"bc")`` differ. Not for files (``file_sha256``) and not for equality of
    multisets: the order is part of the identity.
    """

    def __init__(self, state: str | None = None, count: int = 0) -> None:
        """An empty sequence, or the sequence a stored ``(hexdigest, count)`` pair stands for.

        Raises:
            ValueError: on a state that is not 64 hex characters or a negative count.
        """
        msg = f"a sequence digest state is 64 hex characters and a count >= 0, got {state!r}"
        try:
            self._state: bytes = bytes(32) if state is None else bytes.fromhex(state)
        except ValueError as exc:
            raise ValueError(msg) from exc
        if len(self._state) != 32 or count < 0:  # noqa: PLR2004
            raise ValueError(msg)
        self.count: int = count

    def add(self, item: bytes) -> None:
        """Append one item."""
        self._state = hashlib.sha256(self._state + len(item).to_bytes(8, "big") + item).digest()
        self.count += 1

    def hexdigest(self) -> str:
        """The digest so far, as hex; the digest can still grow, and can be resumed from it."""
        return self._state.hex()
