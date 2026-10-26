"""Instruction tuning's data: records, connected groups, the split, serialization and batches.

The first instruction-tuning test (docs/reviews/2026-10-25-instruction-tuning.md, contract amended
by maintainer B's review of 6080029) reads databricks-dolly-15k: 15,011 human-written records of an
instruction, an optional context, a response and a category, CC BY-SA 3.0, attribution to
Databricks and the Wikipedia contributors whose text the contexts quote. The source file is read
once, hashed and parsed from the checked bytes (``load_records``); a record's id is its line
index in that file.

Records that share an instruction or a context are not independent (the dataset card describes
rephrased questions and reused Wikipedia passages), so the split is over CONNECTED GROUPS: two
records are linked when their canonical instructions are equal or their canonical contexts are
equal and not empty, and a group is a connected component of those links (``group_records``).
Whole groups go to train, validation or test by a keyed hash (``SplitRule``), so equal text never
straddles two splits. This is exact-text grouping after ``canonical``, not semantic
decontamination.

An example is three separately encoded segments (``serialize``): the prompt prefix, the response
body and the terminator ``TERMINATOR``, their ids concatenated. Encoding the whole string at once
would let BPE merge across a segment boundary (a response that starts with two newlines changes
the prefix's own ids: maintainer B's boundary probe), so the prefix a model is trained on is exactly
the prefix it is given at inference. The terminator is ordinary ids (the tokenizer has no special
tokens): with the frozen tokenizer 65f54642 ``"<end>"`` alone is three ids and the line-feed-
prefixed ``TERMINATOR`` four, [198, 27, 450, 29]. A field that contains the literal ``MARKER``
is refused, not escaped (``prepare`` counts them). An example whose inputs exceed the limit is
dropped and counted, never truncated.

Training and evaluation batches (``batches``, ``collate``) hold one example per row from position
0, padded to a length bucket; the loss positions are the SHIFTED targets: the last prefix position
predicts the first body token, and the masks mark the body's targets and the terminator's apart.
Each batch is an independent set of cold contexts; nothing is carried between rows or batches.
"""

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from vesper.data import Chunk
from vesper.keys import normalize_text_v2, verified_bytes
from vesper.tokenizer import Tokenizer

SOURCE = "databricks/databricks-dolly-15k"
FIELDS = ("instruction", "context", "response", "category")
MARKER = "<end>"
TERMINATOR = "\n" + MARKER
SPLITS = ("train", "validation", "test")
CANONICAL = (
    "keys.normalize_text_v2 (per line: spaces and tabs collapsed, Unicode whitespace stripped),"
    " then leading and trailing line feeds removed, then str.casefold"
)
_HASH_SPACE = 1 << 64  # a shift, not 2**64: the type checkers widen a large literal power to Any


@dataclass(frozen=True)
class Record:
    """One source line: its index in the file and the four fields, exactly as read."""

    index: int
    instruction: str
    context: str
    response: str
    category: str


def load_records(path: Path, sha256: str) -> tuple[Record, ...]:
    """The records of the JSON Lines file at ``path``, read once and checked against ``sha256``.

    Every non-empty line is one JSON object with exactly the string fields ``FIELDS``; a final
    line feed ends the file (an empty last line is no record). The index is the line number from
    0.

    Raises:
        ValueError: when the file is missing or hashes to another digest, a line is not such an
            object, or an empty line stands anywhere but at the end.
    """
    data = verified_bytes(path, sha256, f"the {SOURCE} file")
    lines = data.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    records: list[Record] = []
    for index, line in enumerate(lines):
        instruction, context, response, category = _fields(
            cast("object", json.loads(line)) if line else None, index
        )
        records.append(Record(index, instruction, context, response, category))
    return tuple(records)


def _fields(value: object, index: int) -> list[str]:
    """A parsed line's ``FIELDS`` in order, or a ``ValueError`` naming the line."""
    if not isinstance(value, dict):
        msg = f"line {index}: not an object with exactly the fields {FIELDS}"
        raise ValueError(msg)  # noqa: TRY004  # a malformed file is a value error, as everywhere
    mapping = cast("dict[str, object]", value)
    if set(mapping) != set(FIELDS):
        msg = f"line {index}: not an object with exactly the fields {FIELDS}"
        raise ValueError(msg)
    fields = [mapping[name] for name in FIELDS]
    strings = [f for f in fields if isinstance(f, str)]
    if len(strings) != len(fields):
        msg = f"line {index}: every field must be a string"
        raise ValueError(msg)
    return strings


def canonical(text: str) -> str:
    """The grouping key of a field: ``CANONICAL``, exactly. Not a training text."""
    return normalize_text_v2(text).strip("\n").casefold()


def group_records(records: Sequence[Record]) -> tuple[int, ...]:
    """Each record's group: the smallest record index of its connected component.

    Two records are linked when their canonical instructions are equal, or their canonical
    contexts are equal and not empty; a group is the closure of those links. A function of the
    records' fields and indices only, not of their order.
    """
    parent = {r.index: r.index for r in records}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[tuple[str, str], int] = {}
    for r in records:
        keys = [("instruction", canonical(r.instruction))]
        if (context := canonical(r.context)) != "":
            keys.append(("context", context))
        for key in keys:
            seen = first.setdefault(key, r.index)
            a, b = root(seen), root(r.index)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return tuple(root(r.index) for r in records)


@dataclass(frozen=True)
class SplitRule:
    """The keyed group split (config ``split``): train below a bound, validation below the next."""

    key: str
    train_below: float
    validation_below: float

    def of(self, group: int) -> str:
        """The split of a group, so every record of a group lands where its group does.

        blake2b-64 of the group's id keyed by ``key``, as a fraction of 2**64, cut at the bounds.
        """
        digest = hashlib.blake2b(str(group).encode(), digest_size=8, key=self.key.encode())
        fraction = int.from_bytes(digest.digest(), "big") / _HASH_SPACE
        if fraction < self.train_below:
            return SPLITS[0]
        if fraction < self.validation_below:
            return SPLITS[1]
        return SPLITS[2]


def int64_bytes(t: Tensor) -> bytes:
    """PRIMITIVE the bytes a digest of a data tensor hashes: its values as int64, little-endian.

    The values' native bytes as numpy's ``tobytes`` would give them, whatever the tensor's own
    integer or bool dtype, so one digest rule covers tokens, targets, masks and positions (the
    days view's digest, the replay rows' digests). Not for floating tensors: they would be
    truncated to integers.
    """
    flat = t.to(torch.int64).contiguous().view(torch.uint8).flatten()
    return bytes(cast("list[int]", flat.tolist()))  # pyright: ignore[reportUnknownMemberType]


def prompt_text(instruction: str, context: str) -> str:
    """The prompt prefix: the instruction, the context when there is one, the response header."""
    shown = f"Context:\n{context}\n\n" if context else ""
    return f"Instruction:\n{instruction}\n\n{shown}Response:\n"


@dataclass(frozen=True)
class Example:
    """A serialized record: its segments' ids (``int64``) and where it came from."""

    index: int
    group: int
    split: str
    category: str
    prefix: Tensor
    body: Tensor
    terminator: Tensor

    @property
    def ids(self) -> Tensor:
        """The segments' ids, concatenated."""
        return torch.cat((self.prefix, self.body, self.terminator))

    @property
    def inputs(self) -> int:
        """Input positions the example occupies: every id but the last is an input."""
        return int(self.ids.shape[0]) - 1


def serialize(tok: Tokenizer, record: Record, group: int, split: str) -> Example:
    """The record as three separately encoded segments.

    ``prompt_text``, the response and ``TERMINATOR``; the prefix's ids are exactly those
    ``tok.encode(prompt_text(...))`` gives alone, as at inference.

    Raises:
        ValueError: when a field contains ``MARKER`` (refused, never escaped).
    """
    if any(MARKER in text for text in (record.instruction, record.context, record.response)):
        msg = f"record {record.index}: a field contains the terminator marker {MARKER!r}"
        raise ValueError(msg)
    return Example(
        index=record.index,
        group=group,
        split=split,
        category=record.category,
        prefix=tok.encode(prompt_text(record.instruction, record.context)).long(),
        body=tok.encode(record.response).long(),
        terminator=tok.encode(TERMINATOR).long(),
    )


@dataclass(frozen=True)
class Prepared:
    """Every split's retained examples and the counts of what was kept and dropped, and why."""

    examples: dict[str, tuple[Example, ...]]
    groups: tuple[int, ...]
    counts: dict[str, object] = field(default_factory=dict)
    # per record, in file order: [index, group, split, fate] with fate "kept", "marker" (a
    # collision) or "too_long": the membership a run writes and ``final`` must reproduce
    membership: tuple[tuple[int, int, str, str], ...] = ()


def prepare(
    records: Sequence[Record], tok: Tokenizer, *, max_inputs: int, rule: SplitRule
) -> Prepared:
    """Group, split, serialize and filter the records.

    Examples with more than ``max_inputs`` inputs and records with a ``MARKER`` collision are
    dropped and counted per split and category, never truncated or escaped.
    """
    groups = group_records(records)
    kept: dict[str, list[Example]] = {s: [] for s in SPLITS}
    dropped_long: Counter[str] = Counter()
    collisions: Counter[str] = Counter()
    before: Counter[str] = Counter()
    membership: list[tuple[int, int, str, str]] = []
    for record, group in zip(records, groups, strict=True):
        split = rule.of(group)
        before[f"{split}/{record.category}"] += 1
        try:
            example = serialize(tok, record, group, split)
        except ValueError:
            collisions[f"{split}/{record.category}"] += 1
            membership.append((record.index, group, split, "marker"))
            continue
        if example.inputs > max_inputs:
            dropped_long[f"{split}/{record.category}"] += 1
            membership.append((record.index, group, split, "too_long"))
            continue
        kept[split].append(example)
        membership.append((record.index, group, split, "kept"))
    examples = {s: tuple(kept[s]) for s in SPLITS}
    counts: dict[str, object] = {
        "records": len(records),
        "groups": len(set(groups)),
        "groups_per_split": {s: len({e.group for e in examples[s]}) for s in SPLITS},
        "before_filter": dict(sorted(before.items())),
        "dropped_marker_collision": dict(sorted(collisions.items())),
        "dropped_over_max_inputs": dict(sorted(dropped_long.items())),
        "max_inputs": max_inputs,
        "retained": {s: len(examples[s]) for s in SPLITS},
        "retained_per_category": {
            s: dict(sorted(Counter(e.category for e in examples[s]).items())) for s in SPLITS
        },
        "retained_inputs": {s: sum(e.inputs for e in examples[s]) for s in SPLITS},
        "retained_body_tokens": {s: sum(int(e.body.shape[0]) for e in examples[s]) for s in SPLITS},
    }
    return Prepared(examples, groups, counts, tuple(membership))


@dataclass(frozen=True)
class Batch:
    """``rows`` examples from position 0, padded to ``length``; every tensor is ``[rows, length]``.

    ``targets`` are the shifted next ids; ``body_mask`` and ``term_mask`` mark the targets that
    are the response body's and the terminator's ids; ``indices [rows]`` the record ids, -1 for a
    filler row (no input, no loss).
    """

    tokens: Tensor
    targets: Tensor
    valid: Tensor
    pos: Tensor
    body_mask: Tensor
    term_mask: Tensor
    indices: Tensor
    groups: Tensor

    @property
    def loss_mask(self) -> Tensor:
        """The trained targets: body and terminator."""
        return self.body_mask | self.term_mask

    def chunk(self) -> Chunk:
        """The batch as the model's ``Chunk``: independent rows, one document each."""
        rows = int(self.tokens.shape[0])
        indices = cast("list[int]", self.indices.tolist())  # pyright: ignore[reportUnknownMemberType]
        doc_start = torch.zeros_like(self.valid)
        doc_start[:, 0] = self.valid[:, 0]
        return Chunk(
            tokens=self.tokens,
            targets=self.targets,
            valid=self.valid,
            doc_start=doc_start,
            segment=torch.zeros_like(self.pos),
            loss_mask=self.loss_mask,
            pos=self.pos,
            doc_ids=tuple((str(i),) if i >= 0 else () for i in indices),
            spans=tuple(() for _ in range(rows)),
        )


def collate(examples: Sequence[Example], length: int, rows: int) -> Batch:
    """``examples`` as one batch of ``rows`` rows of ``length`` inputs, filler rows after them.

    Raises:
        ValueError: on more examples than rows, or an example with more inputs than ``length``.
    """
    if len(examples) > rows:
        msg = f"{len(examples)} examples for {rows} rows"
        raise ValueError(msg)
    tokens = torch.zeros(rows, length, dtype=torch.int64)
    targets = torch.zeros(rows, length, dtype=torch.int64)
    valid = torch.zeros(rows, length, dtype=torch.bool)
    body_mask = torch.zeros(rows, length, dtype=torch.bool)
    term_mask = torch.zeros(rows, length, dtype=torch.bool)
    indices = torch.full((rows,), -1, dtype=torch.int64)
    groups = torch.full((rows,), -1, dtype=torch.int64)
    for row, example in enumerate(examples):
        n = example.inputs
        if n > length:
            msg = f"record {example.index} has {n} inputs, the batch holds {length}"
            raise ValueError(msg)
        ids = example.ids
        tokens[row, :n] = ids[:-1]
        targets[row, :n] = ids[1:]
        valid[row, :n] = True
        # target j is ids[j + 1]: the body's ids sit at targets p - 1 .. p + b - 2, the
        # terminator's right after them (the last prefix position predicts the first body id)
        p, b = int(example.prefix.shape[0]), int(example.body.shape[0])
        body_mask[row, p - 1 : p - 1 + b] = True
        term_mask[row, p - 1 + b : n] = True
        indices[row] = example.index
        groups[row] = example.group
    pos = torch.arange(length, dtype=torch.int64).expand(rows, length).clone()
    return Batch(tokens, targets, valid, pos, body_mask, term_mask, indices, groups)


def _rank(*parts: object) -> bytes:
    return hashlib.blake2b(":".join(str(p) for p in parts).encode(), digest_size=8).digest()


def batches(
    examples: Sequence[Example],
    *,
    epoch: int,
    seed: int,
    batch_tokens: int,
    buckets: Sequence[int],
) -> list[Batch]:
    """One epoch of ``examples`` in batches: every example exactly once.

    Each example goes to the smallest bucket that holds its inputs; a bucket's batches have
    ``batch_tokens // bucket`` rows (the last one filled with filler rows), its examples taken in
    the order of a blake2b rank of ``(seed, epoch, index)``, and the batches of all buckets are
    ordered by a rank of ``(seed, epoch, bucket, k)``. A pure function of its arguments.

    Raises:
        ValueError: when an example fits no bucket, or a bucket is larger than ``batch_tokens``.
    """
    if any(b > batch_tokens for b in buckets):
        msg = f"a bucket larger than batch_tokens {batch_tokens}: {tuple(buckets)}"
        raise ValueError(msg)

    def example_rank(e: Example) -> bytes:
        return _rank("example", seed, epoch, e.index)

    def batch_rank(pair: tuple[bytes, Batch]) -> bytes:
        return pair[0]

    ordered = sorted(examples, key=example_rank)
    by_bucket: dict[int, list[Example]] = {b: [] for b in sorted(buckets)}
    for example in ordered:
        bucket = next((b for b in sorted(buckets) if example.inputs <= b), None)
        if bucket is None:
            msg = f"record {example.index} has {example.inputs} inputs, beyond every bucket"
            raise ValueError(msg)
        by_bucket[bucket].append(example)
    ranked: list[tuple[bytes, Batch]] = []
    for bucket, members in by_bucket.items():
        rows = batch_tokens // bucket
        for k, start in enumerate(range(0, len(members), rows)):
            batch = collate(members[start : start + rows], bucket, rows)
            ranked.append((_rank("batch", seed, epoch, bucket, k), batch))
    return [batch for _, batch in sorted(ranked, key=batch_rank)]
