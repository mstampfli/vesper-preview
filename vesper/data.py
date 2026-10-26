"""Document stream (spec, Data): whole documents, in order, packed per slot, cut into chunks.

A ``Stream`` owns ``b`` slots. Each slot is one text stream built from whole documents drawn
in order from a shared iterator, concatenated, never split across slots and never shuffled
inside; the order of documents is the iterator's business (the corpus reader shuffles
documents relative to each other). Chunks of ``n`` positions are cut from each slot; the
column's recurrent state and window carry across chunks, which is what trains persistence.
Document boundaries inside a chunk are marked per position, so a column can carry state
across documents or reset at them (the spec's open question) without a second data path,
and the one position whose target is the next document's first token carries no loss.

An input position is a stream position that has a successor; the last token of a slot's
stream appears only as a target. Padding (token 0, ``valid`` false) occurs only once the
iterator is exhausted and some slot runs dry before the others.

Cost: a slot keeps its pending documents in a queue with a cursor into the first one and
writes each token once into the chunk's buffers. Nothing is concatenated, so packing short
documents is linear in the chunk length (a design that re-concatenated the pending tokens per
document was quadratic; tests pin the absence of ``torch.cat`` and ``torch.stack`` here).
"""

import hashlib
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor

from vesper.contracts import check_shape


@dataclass(frozen=True)
class Document:
    """One document: a stable id, its ``int32[L]`` token ids and the source that handed it out.

    ``source`` is the mixer's component name (empty for documents built without one, as in
    tests); the stream carries it into every chunk's ``spans`` so consumed tokens are counted
    per source as the streams emit them, not as the mixer hands documents out. ``group_key`` is
    the split group that decided the document's side (``is_held_out``), as the corpus readers
    give it; empty where no reader built the document (tests, a restored stream's queue), so a
    check of consumed groups must refuse an empty key rather than pass it.
    """

    doc_id: str
    tokens: Tensor
    source: str = ""
    group_key: str = ""


@dataclass(frozen=True)
class Chunk:
    """``n`` positions of every slot's stream; every tensor is ``[b, n]``.

    ``tokens`` (int32) are the inputs, ``targets`` (int32) the next token of the same stream,
    ``valid`` (bool) is true where the input is a real stream position and false on padding,
    ``doc_start`` (bool) marks a document's first token, ``segment`` (int64) counts documents
    per slot from zero, ``loss_mask`` (bool) is false on padding and where the target opens a
    new document, so ``loss_mask`` implies ``valid``. ``pos`` (int64) is the slot's stream
    position of every input, consecutive within the chunk and continuing across chunks and
    documents (the attention's rotary positions; its scores depend on differences, and the
    window carrying across a document boundary is the persistence the stream is for).
    ``doc_ids`` lists, per slot, the documents that gave inputs to this chunk, in order: the
    pointer a replay log needs. ``spans`` lists, per slot and in the same order, each document's
    ``(source, inputs)``: how many of this chunk's input positions came from it, so the tokens
    the model consumed can be counted per source (the mixer's hand-out counts cannot say this:
    a long document handed out once feeds its slot for many chunks).
    """

    tokens: Tensor
    targets: Tensor
    valid: Tensor
    doc_start: Tensor
    segment: Tensor
    loss_mask: Tensor
    pos: Tensor
    doc_ids: tuple[tuple[str, ...], ...]
    spans: tuple[tuple[tuple[str, int], ...], ...]


def is_held_out(doc_id: str, fraction: float) -> bool:
    """True for the ``fraction`` of documents reserved for evaluation.

    Decided by the first 64 bits of blake2b of the id against ``fraction * 2^64``, so the split
    is the same in every process and on every machine (Python's own ``hash`` is salted per
    process and must not be used here), and the held-out set of a smaller fraction is a subset
    of a larger one's. The caller passes the configured ``training.held_out_fraction``; 0 and 1
    are accepted here and mean none and all. Held-out documents never train, never replay,
    never enter the ledger.

    Raises:
        ValueError: if ``fraction`` is outside ``[0, 1]``.
    """
    if not 0.0 <= fraction <= 1.0:
        msg = f"fraction must lie in [0, 1], got {fraction}"
        raise ValueError(msg)
    digest = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") < fraction * 2.0**64


@dataclass(frozen=True)
class SlotState:
    """One slot's packing state: the queued documents, the cursor into the first, the counters."""

    queued: tuple[Document, ...]
    offset: int
    segment: int
    emitted: int


@dataclass(frozen=True)
class StreamState:
    """Everything a ``Stream`` needs to continue exactly where it stopped.

    A checkpoint saves it; ``Stream.restore`` rebuilds the stream from it and from a document
    iterator positioned after the ``documents_taken`` documents this stream already pulled, so the
    next chunk equals the next chunk of the uninterrupted stream, positions included (a fresh
    stream over a skipped iterator would emit the next document at position zero and lose the
    queued tokens: the training-loop review's counterexample).
    """

    slots: tuple[SlotState, ...]
    chunk: int
    exhausted: bool
    documents_taken: int


class _Slot:
    """One stream: a queue of whole documents and a cursor into the first."""

    def __init__(self) -> None:
        """Start empty, with no document seen."""
        self.queue: deque[Document] = deque()
        self.offset: int = 0  # tokens of queue[0] already emitted as inputs
        self.pending: int = 0  # tokens not yet emitted as inputs, over the whole queue
        self.segment: int = 0  # document number of queue[0] within this slot
        self.emitted: int = 0  # positions emitted so far, padding included: the next one's

    def state(self) -> SlotState:
        """The packing state as values (the queued documents are shared, not copied)."""
        return SlotState(tuple(self.queue), self.offset, self.segment, self.emitted)

    @classmethod
    def from_state(cls, state: SlotState) -> "_Slot":
        """A slot continuing from ``state``."""
        slot = cls()
        for document in state.queued:
            slot.append(document)
        slot.offset = state.offset
        slot.pending -= state.offset
        slot.segment = state.segment
        slot.emitted = state.emitted
        return slot

    def append(self, document: Document) -> None:
        """Queue one non-empty document behind the pending ones."""
        self.queue.append(document)
        self.pending += document.tokens.numel()

    def pairs(self) -> int:
        """Pending positions that have a successor."""
        return max(self.pending - 1, 0)

    def fill(
        self,
        tokens: Tensor,
        targets: Tensor,
        valid: Tensor,
        doc_start: Tensor,
        segment: Tensor,
        loss_mask: Tensor,
        pos: Tensor,
    ) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
        """Write up to ``n`` positions into zero-initialized row views of length ``n``.

        Returns the ids of the documents that gave inputs, in order, and their ``(source,
        inputs)`` spans in the same order.
        """
        n = tokens.numel()
        m = min(n, self.pairs())
        pos[:] = torch.arange(self.emitted, self.emitted + n, dtype=torch.int64)
        ids: list[str] = []
        spans: list[tuple[str, int]] = []
        filled = 0
        while filled < m:
            doc = self.queue[0].tokens
            ids.append(self.queue[0].doc_id)
            length = doc.numel()
            take = min(length - self.offset, m - filled)
            spans.append((self.queue[0].source, take))
            end = filled + take
            tokens[filled:end] = doc[self.offset : self.offset + take]
            if self.offset == 0:
                doc_start[filled] = True
            segment[filled:end] = self.segment
            # successors inside this document carry a loss; the one input whose successor is the
            # next document's first token does not
            inside = min(take, length - self.offset - 1)
            targets[filled : filled + inside] = doc[self.offset + 1 : self.offset + 1 + inside]
            loss_mask[filled : filled + inside] = True
            if inside < take:
                targets[end - 1] = self.queue[1].tokens[0]
            self.offset += take
            self.pending -= take
            filled = end
            if self.offset == length:
                _ = self.queue.popleft()
                self.offset = 0
                self.segment += 1
        valid[:m] = True
        segment[m:] = self.segment
        self.emitted += n  # a padded tail continues the count (Chunk.pos)
        return tuple(ids), tuple(spans)


class Stream:
    """``b`` document streams cut into ``n``-position chunks; see the module docstring."""

    def __init__(self, documents: Iterator[Document], slots: int, chunk: int) -> None:
        """Take ownership of ``documents``; ``slots`` is ``b``, ``chunk`` is ``n``.

        Raises:
            ValueError: if ``slots`` or ``chunk`` is below 1.
        """
        if slots < 1:
            msg = f"slots must be at least 1, got {slots}"
            raise ValueError(msg)
        if chunk < 1:
            msg = f"chunk must be at least 1, got {chunk}"
            raise ValueError(msg)
        self._documents: Iterator[Document] = documents
        self._n: int = chunk
        self._slots: list[_Slot] = [_Slot() for _ in range(slots)]
        self._exhausted: bool = False
        self.documents_taken: int = 0  # non-empty documents pulled from the iterator so far

    @property
    def slots(self) -> int:
        """``b``: the number of document streams (rows of every chunk)."""
        return len(self._slots)

    def state(self) -> StreamState:
        """The stream's packing state for a checkpoint."""
        return StreamState(
            slots=tuple(slot.state() for slot in self._slots),
            chunk=self._n,
            exhausted=self._exhausted,
            documents_taken=self.documents_taken,
        )

    @classmethod
    def restore(cls, state: StreamState, documents: Iterator[Document]) -> "Stream":
        """A stream continuing from ``state`` over ``documents``.

        ``documents`` must be the same iterator positioned after the ``state.documents_taken``
        non-empty documents the saved stream had pulled (the corpus readers replay to that count).
        """
        stream = cls(documents, slots=len(state.slots), chunk=state.chunk)
        stream._slots = [_Slot.from_state(slot) for slot in state.slots]
        stream._exhausted = state.exhausted
        stream.documents_taken = state.documents_taken
        return stream

    def _pull(self) -> Document | None:
        """Next non-empty document, or None once the iterator is exhausted.

        Raises:
            ContractError: if a document's tokens are not ``int32[L]``.
        """
        while not self._exhausted:
            document = next(self._documents, None)
            if document is None:
                self._exhausted = True
                break
            check_shape("tokens", document.tokens, ("L",), {}, dtype=torch.int32)
            if document.tokens.numel() > 0:
                self.documents_taken += 1
                return document
        return None

    def _fill(self, slot: _Slot) -> None:
        """Queue documents until the slot can cut a full chunk, or the iterator ends."""
        while slot.pending < self._n + 1:
            document = self._pull()
            if document is None:
                return
            slot.append(document)

    def next(self) -> Chunk | None:
        """The next chunk, or None when no slot has an input position left.

        Slots are filled in index order, each until it holds ``n + 1`` pending tokens, so the
        assignment of documents to slots is deterministic for a given iterator.
        """
        for slot in self._slots:
            self._fill(slot)
        if all(slot.pairs() == 0 for slot in self._slots):
            return None
        b, n = len(self._slots), self._n
        tokens = torch.zeros(b, n, dtype=torch.int32)
        targets = torch.zeros(b, n, dtype=torch.int32)
        valid = torch.zeros(b, n, dtype=torch.bool)
        doc_start = torch.zeros(b, n, dtype=torch.bool)
        segment = torch.zeros(b, n, dtype=torch.int64)
        loss_mask = torch.zeros(b, n, dtype=torch.bool)
        pos = torch.zeros(b, n, dtype=torch.int64)
        filled = [
            slot.fill(
                tokens[row],
                targets[row],
                valid[row],
                doc_start[row],
                segment[row],
                loss_mask[row],
                pos[row],
            )
            for row, slot in enumerate(self._slots)
        ]
        return Chunk(
            tokens=tokens,
            targets=targets,
            valid=valid,
            doc_start=doc_start,
            segment=segment,
            loss_mask=loss_mask,
            pos=pos,
            doc_ids=tuple(ids for ids, _ in filled),
            spans=tuple(spans for _, spans in filled),
        )
