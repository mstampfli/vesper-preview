"""The replay rows beside instruction tuning: training-side text, recorded and checked (CPU)."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from vesper.config import ReplayConfig
from vesper.corpus import Component, Split, finalize, iter_mixed_documents, iter_raw_documents
from vesper.data import Document, Stream
from vesper.instruct_replay import ReplaySource
from vesper.tokenizer import Tokenizer, train_from_texts

CFG = ReplayConfig(weight=1.0, rows=2, length=16, seed=3)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_from_texts([f"word{i} other{i % 7} text" for i in range(200)], vocab=300)


def corpus(tmp_path: Path) -> tuple[list[Component], Split, Split]:
    path = tmp_path / "c.parquet"
    ids = [f"d{i:03d}" for i in range(60)]
    texts = [" ".join(f"word{i} other{j % 7}" for j in range(6)) for i in range(60)]
    pq.write_table(pa.Table.from_pydict({"id": ids, "text": texts}), str(path))
    parts = [Component("c", (path,), 1.0)]
    fin = finalize(parts, tmp_path / "fin")
    return parts, Split(fin, 0.3), Split(fin, 0.3, held_out=True)


def pin_of(held: dict[str, str]) -> bytes:
    """The one field of the accepted pin the replay reads: every held-out document's group."""
    record: dict[str, dict[str, str]] = {"held_out_group_of": held}
    return json.dumps(record).encode()


def documents_of(row: dict[str, object]) -> list[str]:
    return cast("list[str]", row["documents"])


def held_map(parts: list[Component], tok: Tokenizer, held: Split) -> dict[str, str]:
    return {d.doc_id: d.group_key for d in iter_mixed_documents(parts, 0, tok, held)}


def test_rows_are_the_training_stream_with_documents_groups_and_digests(
    tmp_path: Path, tok: Tokenizer
) -> None:
    parts, train, held = corpus(tmp_path)
    held_docs = held_map(parts, tok, held)
    assert held_docs  # the fixture has both sides
    source = ReplaySource(parts, tok, train, CFG, pin_of(held_docs))
    rows = source.next()
    for t in (rows.tokens, rows.targets, rows.pos):
        assert t.shape == (2, 16)
        assert t.dtype == torch.int64
    assert rows.valid.dtype == torch.bool
    assert rows.loss_mask.dtype == torch.bool
    # the stream's own packing, padding and cross-document mask, unchanged (the oracle: the
    # plain stream over the same seeded training mix)
    plain = Stream(iter_mixed_documents(parts, CFG.seed, tok, train), slots=2, chunk=16).next()
    assert plain is not None
    assert torch.equal(rows.tokens, plain.tokens.long())
    assert torch.equal(rows.targets, plain.targets.long())
    assert torch.equal(rows.loss_mask, plain.loss_mask)
    assert torch.equal(rows.valid, plain.valid)
    assert torch.equal(rows.pos, plain.pos)
    raw_groups = {r.doc_id: r.group_key for r in iter_raw_documents([parts[0].paths[0]], 0)}
    recorded = rows.record
    assert len(recorded) == 2
    for r, row in enumerate(recorded):
        assert row["documents"] == list(plain.doc_ids[r])
        assert row["groups"] == [raw_groups[d] for d in plain.doc_ids[r]]
        assert row["labels"] == int(plain.loss_mask[r].sum()) > 0
        assert len(str(row["sha256"])) == 64
        assert set(documents_of(row)).isdisjoint(held_docs)
    assert recorded[0]["sha256"] != recorded[1]["sha256"]
    # its own seed decides the rows; the same seed repeats them
    again = ReplaySource(parts, tok, train, CFG, pin_of(held_docs)).next()
    assert torch.equal(again.tokens, rows.tokens)
    assert again.record == rows.record
    other = ReplaySource(parts, tok, train, ReplayConfig(1.0, 2, 16, 4), pin_of(held_docs)).next()
    assert not torch.equal(other.tokens, rows.tokens)
    summary = source.summary()
    assert summary["batches"] == 1
    assert summary["rows"] == 2
    assert summary["labels"] == sum(cast("int", row["labels"]) for row in recorded)
    assert summary["held_out_overlap"] == {"documents": 0, "groups": 0}


def test_a_consumed_document_or_group_named_held_out_by_the_pin_is_refused(
    tmp_path: Path, tok: Tokenizer
) -> None:
    parts, train, held = corpus(tmp_path)
    held_docs = held_map(parts, tok, held)
    first = ReplaySource(parts, tok, train, CFG, pin_of(held_docs)).next().record[0]
    doc, group = documents_of(first)[0], cast("list[str]", first["groups"])[0]
    as_document = {**held_docs, doc: "some-held-group"}
    with pytest.raises(ValueError, match="held-out document"):
        _ = ReplaySource(parts, tok, train, CFG, pin_of(as_document)).next()
    as_group = {**held_docs, "another-held-doc": group}
    with pytest.raises(ValueError, match="held-out group"):
        _ = ReplaySource(parts, tok, train, CFG, pin_of(as_group)).next()


def test_a_held_out_side_document_is_refused_even_when_the_pin_misses_it(
    tmp_path: Path, tok: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a reader that leaked a held-out document into the training mix: the split's own rule
    # catches it although the pin given does not list it
    parts, train, held = corpus(tmp_path)

    def leaking(
        components: list[Component], seed: int, tokenizer: Tokenizer, split: Split
    ) -> Iterator[Document]:
        del split
        yield from iter_mixed_documents(components, seed, tokenizer, held)

    monkeypatch.setattr("vesper.instruct_replay.iter_mixed_documents", leaking)
    with pytest.raises(ValueError, match="on the held-out side"):
        _ = ReplaySource(parts, tok, train, CFG, pin_of({})).next()


def test_the_held_out_side_a_missing_group_and_an_exhausted_stream_are_refused(
    tmp_path: Path, tok: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, train, held = corpus(tmp_path)
    held_docs = held_map(parts, tok, held)
    with pytest.raises(ValueError, match="training side"):
        _ = ReplaySource(parts, tok, held, CFG, pin_of(held_docs))
    # the stream runs out: refused, never started again (no silent cycling)
    source = ReplaySource(parts, tok, train, ReplayConfig(1.0, 2, 64, 3), pin_of(held_docs))
    seen: list[int] = []

    def drain() -> None:
        while len(seen) < 1000:  # the refusal ends it long before
            _ = source.next()
            seen.append(1)

    with pytest.raises(ValueError, match="ran out"):
        drain()
    assert 0 < len(seen) < 1000
    # a document whose group is unknown cannot be checked: refused, not passed

    def ungrouped(
        components: list[Component], seed: int, tokenizer: Tokenizer, split: Split
    ) -> Iterator[Document]:
        for d in iter_mixed_documents(components, seed, tokenizer, split):
            yield Document(d.doc_id, d.tokens, d.source)

    monkeypatch.setattr("vesper.instruct_replay.iter_mixed_documents", ungrouped)
    with pytest.raises(ValueError, match="no group"):
        _ = ReplaySource(parts, tok, train, CFG, pin_of(held_docs)).next()


def test_a_batch_with_a_row_that_scores_nothing_is_refused(
    tmp_path: Path, tok: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    parts, train, held = corpus(tmp_path)
    held_docs = held_map(parts, tok, held)
    real_next = Stream.next

    def first_row_unscored(self: Stream) -> object:
        chunk = real_next(self)
        if chunk is not None:
            chunk.loss_mask[0] = False
        return chunk

    monkeypatch.setattr(Stream, "next", first_row_unscored)
    with pytest.raises(ValueError, match="no scored target"):
        _ = ReplaySource(parts, tok, train, CFG, pin_of(held_docs)).next()
