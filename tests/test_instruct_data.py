import hashlib
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.instruct_data import (
    MARKER,
    SPLITS,
    TERMINATOR,
    Example,
    Prepared,
    Record,
    SplitRule,
    batches,
    canonical,
    collate,
    group_records,
    load_records,
    prepare,
    prompt_text,
    serialize,
)
from vesper.tokenizer import Tokenizer, train_from_texts

RULE = SplitRule("dolly-split-v1", 0.90, 0.95)


def split_of(group: int) -> str:
    return RULE.of(group)


def ser(tok: Tokenizer, r: Record, group: int) -> Example:
    return serialize(tok, r, group, "train")


def prep(records: list[Record], tok: Tokenizer, max_inputs: int) -> Prepared:
    return prepare(records, tok, max_inputs=max_inputs, rule=RULE)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    # newline runs in the training texts make BPE merge "\n\n" and "\n\n\n": the boundary trap
    texts = [f"Response:\n\n\nanswer {i}\n\nmore text {i * 3}\n<end>" for i in range(60)]
    return train_from_texts(texts + ["Instruction:\nContext:\né😀 café"] * 10, vocab=320)


def record(
    index: int, instruction: str = "Say hi.", context: str = "", response: str = "hi"
) -> Record:
    return Record(index, instruction, context, response, "open_qa")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    _ = path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


ROW: dict[str, object] = {"instruction": "a", "context": "", "response": "b", "category": "open_qa"}


def ints(t: Tensor) -> list[int]:
    return cast("list[int]", t.tolist())  # pyright: ignore[reportUnknownMemberType]


def test_load_records_reads_the_checked_bytes(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    digest = write_jsonl(path, [ROW, {**ROW, "instruction": "c"}])
    records = load_records(path, digest)
    assert [(r.index, r.instruction) for r in records] == [(0, "a"), (1, "c")]
    with pytest.raises(ValueError, match="hash differs"):
        _ = load_records(path, "0" * 64)


@pytest.mark.parametrize(
    "bad",
    [
        {k: v for k, v in ROW.items() if k != "context"},
        {**ROW, "extra": "x"},
        {**ROW, "response": 3},
    ],
)
def test_load_records_refuses_a_malformed_line(tmp_path: Path, bad: dict[str, object]) -> None:
    path = tmp_path / "d.jsonl"
    digest = write_jsonl(path, [ROW, bad])
    with pytest.raises(ValueError, match="line 1"):
        _ = load_records(path, digest)


def test_load_records_refuses_an_inner_empty_line(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    data = (json.dumps(ROW) + "\n\n" + json.dumps(ROW) + "\n").encode()
    _ = path.write_bytes(data)
    with pytest.raises(ValueError, match="line 1"):
        _ = load_records(path, hashlib.sha256(data).hexdigest())


def test_canonical_ignores_case_and_spacing_only() -> None:
    assert canonical("  What IS\tthe  Sky?\n") == canonical("what is the sky?")
    assert canonical("what is the sky") != canonical("what is the sky?")
    assert canonical("\n \n") == ""


def test_groups_link_equal_instructions_and_nonempty_contexts_transitively() -> None:
    records = [
        record(0, "Q one", "passage A"),
        record(1, "q  ONE", ""),  # same instruction as 0
        record(2, "Q two", "passage a"),  # same context as 0
        record(3, "Q three", "passage B"),
        record(4, "Q four", ""),  # empty context links nothing
        record(5, "Q five", " "),
        record(6, "q three", "passage C"),  # same instruction as 3
    ]
    assert group_records(records) == (0, 0, 0, 3, 4, 5, 3)
    assert group_records(list(reversed(records))) == tuple(reversed((0, 0, 0, 3, 4, 5, 3)))


def test_split_is_a_keyed_function_of_the_group() -> None:
    counts: dict[str, int] = dict.fromkeys(SPLITS, 0)
    for group in range(20000):
        counts[split_of(group)] += 1
    assert counts["train"] / 20000 == pytest.approx(0.90, abs=0.01)
    assert counts["validation"] / 20000 == pytest.approx(0.05, abs=0.005)
    assert [split_of(g) for g in range(200)] == [split_of(g) for g in range(200)]
    other = SplitRule("other", 0.90, 0.95)
    assert [split_of(g) for g in range(200)] != [other.of(g) for g in range(200)]
    # the bounds are the rule's: everything below 1.0 is train or validation
    assert {SplitRule("k", 0.5, 1.0).of(g) for g in range(500)} == {"train", "validation"}


def test_serialize_keeps_the_prefix_ids_whatever_the_response(tok: Tokenizer) -> None:
    for response in ("hello", " hello", "\n\nhello", "\n\n\nanswer 3", "é😀", "", "x\n"):
        r = record(7, "Return it.", "", response)
        e = ser(tok, r, 7)
        prompt = prompt_text(r.instruction, r.context)
        assert torch.equal(e.prefix, tok.encode(prompt).long())
        assert torch.equal(e.body, tok.encode(response).long())
        assert torch.equal(e.terminator, tok.encode(TERMINATOR).long())
        assert tok.decode(e.ids) == prompt + response + TERMINATOR
    # the trap the segments avoid: encoded whole, this response changes the prefix's ids
    trap = prompt_text("Return it.", "") + "\n\n\nanswer 3" + TERMINATOR
    prefix = tok.encode(prompt_text("Return it.", ""))
    assert not torch.equal(tok.encode(trap)[: prefix.shape[0]], prefix)


def test_serialize_shows_the_context_only_when_there_is_one() -> None:
    assert "Context:" not in prompt_text("a", "")
    assert prompt_text("a", "c") == "Instruction:\na\n\nContext:\nc\n\nResponse:\n"


@pytest.mark.parametrize("where", ["instruction", "context", "response"])
def test_serialize_refuses_the_marker_in_any_field(tok: Tokenizer, where: str) -> None:
    fields = {"instruction": "a", "context": "", "response": "b", where: f"x {MARKER} y"}
    r = Record(0, fields["instruction"], fields["context"], fields["response"], "open_qa")
    with pytest.raises(ValueError, match="terminator marker"):
        _ = ser(tok, r, 0)


def test_collate_masks_the_shifted_body_and_terminator_targets(tok: Tokenizer) -> None:
    examples = [
        ser(tok, record(0, "Say it.", "", "\n\nanswer 1"), 0),
        ser(tok, record(1, "Say nothing.", "", ""), 1),
    ]
    batch = collate(examples, 64, 3)
    for row, e in enumerate(examples):
        p, b, t = (int(x.shape[0]) for x in (e.prefix, e.body, e.terminator))
        assert torch.equal(batch.targets[row][batch.body_mask[row]], e.body)
        assert torch.equal(batch.targets[row][batch.term_mask[row]], e.terminator)
        # the last prefix position predicts the first response id (body, or the terminator)
        assert bool(batch.loss_mask[row, p - 1])
        assert not bool(batch.loss_mask[row, p - 2])
        assert int(batch.loss_mask[row].sum()) == b + t
        assert int(batch.valid[row].sum()) == p + b + t - 1
        assert torch.equal(batch.tokens[row, : e.inputs], e.ids[:-1])
    assert not bool(batch.valid[2].any())
    assert not bool(batch.loss_mask[2].any())
    assert ints(batch.indices) == [0, 1, -1]
    assert torch.equal(batch.pos[1], torch.arange(64))
    chunk = batch.chunk()
    assert torch.equal(chunk.loss_mask, batch.loss_mask)
    assert chunk.doc_ids == (("0",), ("1",), ())


def test_collate_refuses_what_does_not_fit(tok: Tokenizer) -> None:
    e = ser(tok, record(0), 0)
    with pytest.raises(ValueError, match="inputs"):
        _ = collate([e], e.inputs - 1, 1)
    _ = collate([e], e.inputs, 1)  # exactly full is fine
    with pytest.raises(ValueError, match="rows"):
        _ = collate([e, e], e.inputs, 1)


def test_batches_hold_every_example_once_in_a_seeded_order(tok: Tokenizer) -> None:
    examples = [ser(tok, record(i, f"Q {i}", "", "answer " * (1 + i % 40)), i) for i in range(50)]
    one = batches(examples, epoch=0, seed=0, batch_tokens=512, buckets=(32, 64, 128))
    seen = sorted(i for b in one for i in ints(b.indices) if i >= 0)
    assert seen == list(range(50))
    for b in one:
        rows, length = b.tokens.shape
        assert rows == 512 // length
    again = batches(examples, epoch=0, seed=0, batch_tokens=512, buckets=(32, 64, 128))
    assert [ints(b.indices) for b in one] == [ints(b.indices) for b in again]
    other = batches(examples, epoch=1, seed=0, batch_tokens=512, buckets=(32, 64, 128))
    assert [ints(b.indices) for b in one] != [ints(b.indices) for b in other]
    with pytest.raises(ValueError, match="beyond every bucket"):
        _ = batches(examples, epoch=0, seed=0, batch_tokens=512, buckets=(8,))


def test_prepare_drops_and_counts_long_and_colliding_records(tok: Tokenizer) -> None:
    records = [record(i, f"Q {i}", "", "answer") for i in range(30)]
    records.append(record(30, "Q long", "", "answer " * 200))
    records.append(record(31, "Q marker", "", f"a {MARKER}"))
    fits = ser(tok, records[0], 0).inputs
    prepared = prep(records, tok, max(fits, 64))

    def total(key: str) -> int:
        return sum(cast("dict[str, int]", prepared.counts[key]).values())

    assert total("retained") == 30
    assert total("dropped_over_max_inputs") == 1
    assert total("dropped_marker_collision") == 1
    # every record's fate, in file order, with its group and split
    fates = [m[3] for m in prepared.membership]
    assert fates == ["kept"] * 30 + ["too_long", "marker"]
    assert [m[0] for m in prepared.membership] == list(range(32))
    assert all(m[2] == split_of(m[1]) for m in prepared.membership)
    # a boundary-length example stays
    exact = prep(records[:1], tok, fits)
    assert sum(len(v) for v in exact.examples.values()) == 1
    assert sum(len(v) for v in prep(records[:1], tok, fits - 1).examples.values()) == 0
    for split, kept in prepared.examples.items():
        assert all(e.split == split for e in kept)
        assert all(e.split == split_of(e.group) for e in kept)
