"""Replay rows for instruction tuning: the source run's training-side text beside every update.

PRIMITIVE ``ReplaySource`` (maintainer B's design, 758b4fd). Instruction tuning forgets general text
(the 3e-5 and 1e-5 trials of 2026-10-26); replay adds, to every tuning update, the loss of a
few rows of the text the source model was trained on. The rows come from ONE stream
(``data.Stream`` over ``corpus.iter_mixed_documents``) of the finalized corpus's TRAINING side,
with the replay's own seed, independent of the example order; the stream's packing is kept as
it is: documents are concatenated within a row, a target that opens a new document is not
scored, padding only where the text ends, positions continue as the stream gives them. The
caller runs every row from cold states (all recurrent states, key histories and attention
caches empty per row and update), so nothing is carried between rows or updates.

Every consumed row is checked before it is returned: each document's split group is known (a
document without one is refused, not passed), no document is one the accepted test pin names
as held out, no group is one of the pin's held-out groups, no group falls on the held-out side
of the split, and the row scores at least one target. A stream that runs out is refused; it is
never started again (no silent cycling). Every row is recorded: the sha256 of its tokens,
targets, loss mask and positions, its documents and their groups, its scored targets.

Train-side exclusion is what this establishes; it says nothing about overlap between this text
and the instruction dataset's contexts.

For: ``instruct.tune`` with a ``replay`` section. Not for: evaluation, or any held-out read.
"""

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import cast

from torch import Tensor

from vesper.config import ReplayConfig
from vesper.corpus import Component, Split, iter_mixed_documents
from vesper.data import Document, Stream, is_held_out
from vesper.instruct_data import int64_bytes
from vesper.tokenizer import Tokenizer

PACKING = (
    "documents concatenated within a row as the training stream packs them; a target opening a"
    " new document is not scored; padding only where the text ends; stream positions kept; every"
    " row from cold states (recurrent states, key histories and attention caches empty)"
)
CLAIM = (
    "training side of the source's split only; no decontamination from the instruction dataset"
    " is claimed"
)


@dataclass(frozen=True)
class ReplayRows:
    """One update's replay rows, every tensor ``[rows, length]``, and each row's record."""

    tokens: Tensor
    targets: Tensor
    valid: Tensor
    pos: Tensor
    loss_mask: Tensor
    record: list[dict[str, object]]


def _row_sha256(*tensors: Tensor) -> str:
    digest = hashlib.sha256()
    for t in tensors:
        digest.update(int64_bytes(t))
    return digest.hexdigest()


class ReplaySource:
    """The replay rows of one tuning run, checked and recorded; see the module docstring.

    ``pin`` is the accepted test pin's bytes, already read and checked by the caller; its
    ``held_out_group_of`` (every held-out document's split group) is what consumed rows are
    checked against.

    Raises:
        ValueError: for the held-out side, or a pin without ``held_out_group_of``.
    """

    def __init__(
        self,
        parts: Sequence[Component],
        tokenizer: Tokenizer,
        train: Split,
        replay: ReplayConfig,
        pin: bytes,
    ) -> None:
        """Open the stream; nothing is read yet."""
        if train.held_out:
            msg = "replay reads the training side, never the held-out split"
            raise ValueError(msg)
        record = cast("dict[str, object]", json.loads(pin))
        if not isinstance(record.get("held_out_group_of"), dict):
            msg = "the pin has no held_out_group_of"
            raise ValueError(msg)  # noqa: TRY004  # a malformed file is a value error
        self._held_documents: dict[str, str] = cast("dict[str, str]", record["held_out_group_of"])
        self._held_groups: set[str] = set(self._held_documents.values())
        self._fraction: float = train.held_out_fraction
        self._groups: dict[str, str] = {}
        self._stream: Stream = Stream(
            self._grouped(iter_mixed_documents(parts, replay.seed, tokenizer, train)),
            slots=replay.rows,
            chunk=replay.length,
        )
        self._rows: int = 0
        self._batches: int = 0
        self._labels: int = 0
        self._documents: set[str] = set()
        self._consumed_groups: set[str] = set()

    def _grouped(self, documents: Iterator[Document]) -> Iterator[Document]:
        for document in documents:
            self._groups[document.doc_id] = document.group_key
            yield document

    def _check(self, documents: Sequence[str]) -> list[str]:
        groups: list[str] = []
        for doc in documents:
            group = self._groups.get(doc, "")
            if not group:
                msg = f"replay document {doc} has no group: its side cannot be checked"
                raise ValueError(msg)
            if doc in self._held_documents:
                msg = f"replay document {doc} is a held-out document of the pin"
                raise ValueError(msg)
            if group in self._held_groups:
                msg = f"replay document {doc} belongs to the held-out group {group}"
                raise ValueError(msg)
            if is_held_out(group, self._fraction):
                msg = f"replay document {doc}'s group {group} is on the held-out side"
                raise ValueError(msg)
            groups.append(group)
        return groups

    def next(self) -> ReplayRows:
        """The next update's rows, checked and recorded.

        Raises:
            ValueError: when the stream ran out, or on a failed check (module docstring).
        """
        chunk = self._stream.next()
        if chunk is None:
            msg = f"the replay text ran out after {self._batches} updates; it is not restarted"
            raise ValueError(msg)
        record: list[dict[str, object]] = []
        for r, documents in enumerate(chunk.doc_ids):
            labels = int(chunk.loss_mask[r].sum())
            if labels == 0:
                msg = f"replay row {r} of update {self._batches} has no scored target"
                raise ValueError(msg)
            record.append(
                {
                    "sha256": _row_sha256(
                        chunk.tokens[r], chunk.targets[r], chunk.loss_mask[r], chunk.pos[r]
                    ),
                    "documents": list(documents),
                    "groups": self._check(documents),
                    "labels": labels,
                }
            )
        for row in record:
            self._documents.update(cast("list[str]", row["documents"]))
            self._consumed_groups.update(cast("list[str]", row["groups"]))
            self._labels += cast("int", row["labels"])
        self._rows += len(record)
        self._batches += 1
        return ReplayRows(
            tokens=chunk.tokens.long(),
            targets=chunk.targets.long(),
            valid=chunk.valid,
            pos=chunk.pos,
            loss_mask=chunk.loss_mask,
            record=record,
        )

    def summary(self) -> dict[str, object]:
        """What the run consumed, for ``result.json``: counts and the held-out overlap (0, 0)."""
        return {
            "batches": self._batches,
            "rows": self._rows,
            "labels": self._labels,
            "documents": len(self._documents),
            "groups": len(self._consumed_groups),
            "held_out_overlap": {
                "documents": len(self._documents & set(self._held_documents)),
                "groups": len(self._consumed_groups & self._held_groups),
            },
            "packing": PACKING,
            "claim": CLAIM,
        }
