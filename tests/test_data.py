import os
import subprocess
import sys
from itertools import pairwise
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from vesper.contracts import ContractError
from vesper.data import Chunk, Document, Stream, is_held_out

FRACTION = 0.005  # training.held_out_fraction of configs/vesper-s.yaml


def ints(*values: int) -> Tensor:
    t = torch.zeros(len(values), dtype=torch.int32)
    for i, v in enumerate(values):
        t[i] = v
    return t


def doc(doc_id: str, *tokens: int) -> Document:
    return Document(doc_id=doc_id, tokens=ints(*tokens))


def drain(docs: list[Document], slots: int, chunk: int) -> list[Chunk]:
    stream = Stream(iter(docs), slots=slots, chunk=chunk)
    out: list[Chunk] = []
    while (c := stream.next()) is not None:
        out.append(c)
    return out


def bools(*values: bool) -> Tensor:
    t = torch.zeros(len(values), dtype=torch.bool)
    for i, v in enumerate(values):
        t[i] = v
    return t


# --- held-out split -----------------------------------------------------------------------


def test_held_out_is_deterministic_and_matches_the_fraction() -> None:
    ids = [f"<urn:uuid:{i:032x}>" for i in range(200_000)]
    held = [is_held_out(i, FRACTION) for i in ids]
    assert held == [is_held_out(i, FRACTION) for i in ids]
    fraction = sum(held) / len(held)
    assert 0.004 < fraction < 0.006
    half = sum(is_held_out(i, 0.5) for i in ids[:20_000]) / 20_000
    assert 0.47 < half < 0.53


def test_fraction_endpoints_and_nesting() -> None:
    ids = [f"d{i}" for i in range(2_000)]
    assert not any(is_held_out(i, 0.0) for i in ids)
    assert all(is_held_out(i, 1.0) for i in ids)
    small = {i for i in ids if is_held_out(i, 0.01)}
    large = {i for i in ids if is_held_out(i, 0.05)}
    assert small <= large
    assert small
    assert len(large) > len(small)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _ = is_held_out("x", -0.1)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _ = is_held_out("x", 1.1)


def test_golden_ids_pin_the_hash_rule() -> None:
    # computed once from the rule (first 64 bits of blake2b below fraction * 2^64) and pinned
    assert [i for i in range(64) if is_held_out(f"d{i}", 0.05)] == [35, 45]
    assert [i for i in range(2000) if is_held_out(f"d{i}", 0.005)] == [188, 343, 1415, 1669, 1773]


def test_held_out_does_not_depend_on_the_process_hash_seed() -> None:
    code = (
        "from vesper.data import is_held_out; "
        "print([is_held_out(f'd{i}', 0.05) for i in range(64)])"
    )
    outputs: set[str] = set()
    for seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        run = subprocess.run(  # noqa: S603  (own interpreter, fixed code string)
            [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
        )
        outputs.add(run.stdout)
    assert len(outputs) == 1
    assert "True" in next(iter(outputs))


# --- one slot, exact -------------------------------------------------------------------------


def test_one_slot_packs_documents_in_order_and_masks_the_boundary() -> None:
    chunks = drain([doc("a", 1, 2, 3, 4), doc("b", 5, 6), doc("c", 7)], slots=1, chunk=3)
    # stream: 1 2 3 4 | 5 6 | 7 ; six (input, target) pairs, so exactly two chunks
    assert len(chunks) == 2
    first, second = chunks
    assert torch.equal(first.tokens[0], ints(1, 2, 3))
    assert torch.equal(first.targets[0], ints(2, 3, 4))
    assert torch.equal(first.doc_start[0], bools(True, False, False))
    assert torch.equal(first.segment[0], torch.zeros(3, dtype=torch.int64))
    assert torch.equal(first.loss_mask[0], bools(True, True, True))
    assert torch.equal(second.tokens[0], ints(4, 5, 6))
    assert torch.equal(second.targets[0], ints(5, 6, 7))
    assert torch.equal(second.doc_start[0], bools(False, True, False))
    assert torch.equal(second.segment[0], torch.arange(3, dtype=torch.int64).clamp(max=1))
    # the targets 5 and 7 open documents b and c: those positions carry no loss
    assert torch.equal(second.loss_mask[0], bools(False, True, False))


def test_chunk_shapes_and_dtypes() -> None:
    (chunk,) = drain([doc("a", 1, 2, 3, 4, 5)], slots=2, chunk=4)
    assert chunk.tokens.shape == (2, 4)
    assert chunk.tokens.dtype == torch.int32
    assert chunk.targets.dtype == torch.int32
    assert chunk.valid.dtype == torch.bool
    assert chunk.doc_start.dtype == torch.bool
    assert chunk.segment.dtype == torch.int64
    assert chunk.loss_mask.dtype == torch.bool


# --- several slots ---------------------------------------------------------------------------


def test_slots_fill_in_index_order_and_pad_only_at_the_end() -> None:
    chunks = drain([doc("a", 1, 2, 3), doc("b", 4, 5), doc("c", 6, 7, 8)], slots=2, chunk=2)
    # slot 0 takes a (3 tokens, enough for chunk + 1); slot 1 takes b, then c to reach 3
    assert len(chunks) == 2
    first, second = chunks
    assert torch.equal(first.tokens, torch.stack([ints(1, 2), ints(4, 5)]))
    assert torch.equal(first.targets, torch.stack([ints(2, 3), ints(5, 6)]))
    assert torch.equal(first.loss_mask, torch.stack([bools(True, True), bools(True, False)]))
    # slot 0 is out of pairs (3 was its last token): padding, no loss; slot 1 continues with c
    assert torch.equal(second.tokens, torch.stack([ints(0, 0), ints(6, 7)]))
    assert torch.equal(second.targets, torch.stack([ints(0, 0), ints(7, 8)]))
    assert torch.equal(second.valid, torch.stack([bools(False, False), bools(True, True)]))
    assert bool(first.valid.all().item())
    assert torch.equal(second.doc_start, torch.stack([bools(False, False), bools(True, False)]))
    assert torch.equal(second.loss_mask, torch.stack([bools(False, False), bools(True, True)]))
    assert torch.equal(second.segment[0], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(second.segment[1], torch.ones(2, dtype=torch.int64))


def test_every_document_is_used_whole_exactly_once() -> None:
    gen = torch.Generator().manual_seed(7)
    docs: list[Document] = []
    for i in range(60):
        length = int(torch.randint(1, 23, (1,), generator=gen).item())
        tokens = torch.randint(1, 1000, (length,), generator=gen).to(torch.int32)
        docs.append(Document(doc_id=f"d{i}", tokens=tokens))
    slots, n = 4, 8
    chunks = drain(docs, slots=slots, chunk=n)
    assert chunks
    # rebuild each slot's stream from the valid inputs
    streams: list[list[int]] = [[] for _ in range(slots)]
    starts: list[list[bool]] = [[] for _ in range(slots)]
    for c in chunks:
        # a loss position is a valid position; padding is never a document start
        assert bool((c.loss_mask <= c.valid).all().item())
        assert bool((c.doc_start <= c.valid).all().item())
        assert bool((c.tokens[~c.valid] == 0).all().item())
        # within a chunk, the target of a valid position is the next valid input
        both = c.valid[:, :-1] & c.valid[:, 1:]
        assert bool((c.targets[:, :-1] == c.tokens[:, 1:])[both].all().item())
        # between two valid positions, the loss is masked exactly where the next opens a document
        opens_next = c.doc_start[:, 1:]
        assert bool(((c.loss_mask[:, :-1] == ~opens_next)[both]).all().item())
        for s in range(slots):
            for j in range(n):
                if bool(c.valid[s, j].item()):
                    streams[s].append(int(c.tokens[s, j].item()))
                    starts[s].append(bool(c.doc_start[s, j].item()))
    # the document boundaries recovered from doc_start cut each stream into whole documents,
    # the last one missing only its final token (which appears only as a target)
    seen: dict[str, int] = {}
    expected = {d.doc_id: [int(t.item()) for t in d.tokens.unbind()] for d in docs}
    for s in range(slots):
        cuts = [j for j, st in enumerate(starts[s]) if st] + [len(streams[s])]
        pieces = [streams[s][a:b] for a, b in pairwise(cuts)]
        for i, piece in enumerate(pieces):
            is_last = i == len(pieces) - 1
            matches = [
                k
                for k, v in expected.items()
                if k not in seen and (v[:-1] == piece if is_last else v == piece)
            ]
            assert matches, piece
            seen[matches[0]] = s
    assert len(seen) == len(docs)


def test_targets_across_chunks_continue_each_slot_stream() -> None:
    # the last target of one chunk is the first valid input of the next chunk in the same slot
    docs = [doc(f"d{i}", *range(10 * i + 1, 10 * i + 1 + (i % 5) + 1)) for i in range(30)]
    chunks = drain(docs, slots=3, chunk=4)
    for earlier, later in pairwise(chunks):
        for s in range(3):
            if bool(earlier.valid[s, -1].item()) and bool(later.valid[s, 0].item()):
                assert int(earlier.targets[s, -1].item()) == int(later.tokens[s, 0].item())


def test_segment_counts_documents_per_slot() -> None:
    chunks = drain([doc("a", 1, 2), doc("b", 3, 4), doc("c", 5, 6), doc("d", 7, 8)], 1, 4)
    seg = torch.cat([c.segment[0] for c in chunks])
    starts = torch.cat([c.doc_start[0] for c in chunks])
    # the segment rises by one exactly at each document start and is constant otherwise
    assert bool((seg[1:] - seg[:-1] == starts[1:].to(torch.int64)).all().item())
    assert int(seg[0].item()) == 0


def test_last_token_of_each_document_is_only_ever_a_target() -> None:
    chunks = drain([doc("a", 1, 2, 3), doc("b", 4, 5)], slots=1, chunk=10)
    (chunk,) = chunks
    # pairs: (1,2) (2,3) (3,4) (4,5); 5 never appears as an input
    assert torch.equal(chunk.tokens[0, :4], ints(1, 2, 3, 4))
    assert torch.equal(chunk.targets[0, :4], ints(2, 3, 4, 5))
    assert not bool(chunk.loss_mask[0, 4:].any().item())
    assert not bool(chunk.valid[0, 4:].any().item())
    assert torch.equal(chunk.loss_mask[0, :4], bools(True, True, False, True))


# --- cost ------------------------------------------------------------------------------------


def test_packing_short_documents_copies_without_concatenation() -> None:
    # 256 one-token documents into one slot: the pending queue is never concatenated or stacked
    docs = [doc(f"d{i}", i % 7 + 1) for i in range(256)]
    with (
        patch.object(torch, "cat", wraps=torch.cat) as cat_mock,
        patch.object(torch, "stack", wraps=torch.stack) as stack_mock,
    ):
        chunks = drain(docs, slots=1, chunk=256)
    assert cat_mock.call_count == 0
    assert stack_mock.call_count == 0
    (chunk,) = chunks
    assert int(chunk.valid.sum().item()) == 255
    assert bool(chunk.doc_start[0, :255].all().item())
    assert not bool(chunk.loss_mask.any().item())


# --- edge cases ------------------------------------------------------------------------------


def test_empty_documents_are_skipped() -> None:
    chunks = drain([doc("e"), doc("a", 1, 2), doc("e2"), doc("b", 3, 4)], slots=1, chunk=4)
    (chunk,) = chunks
    assert torch.equal(chunk.doc_start[0], bools(True, False, True, False))
    assert torch.equal(chunk.segment[0], ints(0, 0, 1, 1).to(torch.int64))


def test_empty_source_yields_nothing() -> None:
    assert drain([], slots=3, chunk=4) == []
    assert drain([doc("one", 9)], slots=1, chunk=4) == []


def test_more_slots_than_documents_pads_the_idle_slots() -> None:
    (chunk,) = drain([doc("a", 1, 2, 3)], slots=3, chunk=2)
    assert torch.equal(chunk.tokens[0], ints(1, 2))
    assert not bool(chunk.loss_mask[1:].any().item())
    assert not bool(chunk.doc_start[1:].any().item())


def test_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="slots"):
        _ = Stream(iter([]), slots=0, chunk=4)
    with pytest.raises(ValueError, match="chunk"):
        _ = Stream(iter([]), slots=1, chunk=0)
    bad_dtype = Document(doc_id="x", tokens=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ContractError):
        _ = Stream(iter([bad_dtype]), slots=1, chunk=2).next()
    bad_rank = Document(doc_id="y", tokens=torch.zeros(1, 3, dtype=torch.int32))
    with pytest.raises(ContractError):
        _ = Stream(iter([bad_rank]), slots=1, chunk=2).next()


def test_positions_are_stream_positions_consecutive_within_and_across_chunks() -> None:
    # stream of slot 0: 1 2 3 4 | 5 6 | 7 ; the positions count inputs of the slot's stream and
    # keep counting across the document boundary and the chunk boundary (the attention needs
    # consecutive positions inside a chunk; its scores depend on differences only)
    chunks = drain([doc("a", 1, 2, 3, 4), doc("b", 5, 6), doc("c", 7)], slots=1, chunk=3)
    first, second = chunks
    assert first.pos.dtype == torch.int64
    assert torch.equal(first.pos[0], torch.arange(0, 3))
    assert torch.equal(second.pos[0], torch.arange(3, 6))
    # two slots count independently; a padded tail still continues the count
    two = drain([doc("a", 1, 2, 3), doc("b", 4, 5, 6, 7, 8)], slots=2, chunk=2)
    for s in range(2):
        seen = torch.cat([c.pos[s] for c in two])
        assert torch.equal(seen, torch.arange(seen.numel()))


def test_doc_ids_name_the_documents_that_gave_inputs_in_order() -> None:
    chunks = drain([doc("a", 1, 2, 3, 4), doc("b", 5, 6), doc("c", 7)], slots=1, chunk=3)
    first, second = chunks
    assert first.doc_ids == (("a",),)
    assert second.doc_ids == (("a", "b"),)  # c is only ever a target: no input, not listed
    two = drain([doc("a", 1, 2, 3), doc("b", 4, 5, 6, 7, 8)], slots=2, chunk=2)
    assert two[0].doc_ids == (("a",), ("b",))
    assert all(len(c.doc_ids) == 2 for c in two)


def test_spans_count_the_inputs_each_source_gave_to_a_chunk() -> None:
    # consumed tokens per source as the stream emits them: a long document handed out once feeds
    # its slot for several chunks, which the mixer's hand-out counts cannot say
    web = Document("a", torch.arange(1, 8, dtype=torch.int32), "web")
    code = Document("b", torch.arange(8, 10, dtype=torch.int32), "code")
    chunks = drain([web, code], slots=1, chunk=3)
    assert [c.spans for c in chunks] == [
        ((("web", 3),),),
        ((("web", 3),),),
        ((("web", 1), ("code", 1)),),  # a's last token is an input whose target opens b
    ]
    for c in chunks:
        assert sum(n for slot in c.spans for _, n in slot) == int(c.valid.sum().item())
        assert tuple(s for s, _ in c.spans[0]) == tuple(
            {"a": "web", "b": "code"}[d] for d in c.doc_ids[0]
        )
    assert drain([doc("a", 1, 2, 3)], slots=1, chunk=4)[0].spans == ((("", 2),),)  # no source


def test_documents_taken_counts_what_the_stream_pulled() -> None:
    stream = Stream(iter([doc("a", 1, 2), doc("e"), doc("b", 3, 4, 5)]), slots=1, chunk=2)
    assert stream.documents_taken == 0
    _ = stream.next()
    assert stream.documents_taken == 2  # the empty document is skipped, not counted


def same_chunk(a: Chunk, e: Chunk) -> bool:
    return (
        torch.equal(a.tokens, e.tokens)
        and torch.equal(a.targets, e.targets)
        and torch.equal(a.valid, e.valid)
        and torch.equal(a.doc_start, e.doc_start)
        and torch.equal(a.segment, e.segment)
        and torch.equal(a.loss_mask, e.loss_mask)
        and torch.equal(a.pos, e.pos)
        and a.doc_ids == e.doc_ids
    )


def test_a_restored_stream_continues_exactly_where_the_saved_one_stopped() -> None:
    # Maintainer B's counterexample (training-loop review): a=[10..19], b=[20..29], chunk 3; after
    # one chunk the stream has taken one document and its slot still holds 7 queued tokens; a fresh
    # stream over the iterator skipped by one would emit [20, 21, 22] at positions [0, 1, 2]
    docs = [doc("a", *range(10, 20)), doc("b", *range(20, 30))]
    original = Stream(iter(docs), slots=1, chunk=3)
    first = original.next()
    assert first is not None
    assert torch.equal(first.tokens[0], ints(10, 11, 12))
    assert original.documents_taken == 1
    saved = original.state()
    assert saved.documents_taken == 1
    assert saved.slots[0].offset == 3
    assert saved.slots[0].emitted == 3
    restored = Stream.restore(saved, iter(docs[saved.documents_taken :]))
    expected = original.next()
    actual = restored.next()
    assert expected is not None
    assert actual is not None
    assert torch.equal(actual.tokens, expected.tokens)
    assert torch.equal(actual.pos, expected.pos)
    assert torch.equal(actual.pos[0], ints(3, 4, 5).to(torch.int64))
    assert actual.doc_ids == expected.doc_ids
    # and so on until both run dry, with the same document count taken
    while (a := restored.next()) is not None:
        e = original.next()
        assert e is not None
        assert same_chunk(a, e)
    assert original.next() is None
    assert restored.documents_taken == original.documents_taken
    wrong = Stream(iter(docs[1:]), slots=1, chunk=3).next()
    assert wrong is not None
    assert torch.equal(wrong.tokens[0], ints(20, 21, 22))  # what a skip alone would have given


def test_a_restored_stream_keeps_several_slots_and_a_partly_used_queue() -> None:
    docs = [doc(f"d{i}", *range(100 * i, 100 * i + 2 + (i % 4))) for i in range(12)]
    original = Stream(iter(docs), slots=3, chunk=4)
    seen = [original.next(), original.next()]
    assert all(c is not None for c in seen)
    saved = original.state()
    restored = Stream.restore(saved, iter(docs[saved.documents_taken :]))
    while (e := original.next()) is not None:
        a = restored.next()
        assert a is not None
        assert torch.equal(a.tokens, e.tokens)
        assert torch.equal(a.pos, e.pos)
        assert torch.equal(a.segment, e.segment)
        assert a.doc_ids == e.doc_ids
    assert restored.next() is None
