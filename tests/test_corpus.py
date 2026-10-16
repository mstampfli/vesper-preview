import json
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from vesper.corpus import (
    Component,
    Mixer,
    MixerState,
    RawReader,
    ReaderCursor,
    Split,
    check_inputs,
    finalize,
    iter_documents,
    iter_mixed_documents,
    iter_mixed_texts,
    iter_raw_documents,
    iter_training_texts,
    load_finalization,
)
from vesper.data import is_held_out
from vesper.keys import dedup_key
from vesper.tokenizer import Tokenizer, train_from_texts

FRACTION = 0.25  # large enough for a tiny corpus to have both splits


def write_parquet(path: Path, first: int, count: int, group_size: int) -> list[tuple[str, str]]:
    rows = [
        (f"<urn:uuid:{i:032x}>", f"document {i} text with\nnewline and {i * 7}")
        for i in range(first, first + count)
    ]
    table = pa.Table.from_pydict({"id": [r[0] for r in rows], "text": [r[1] for r in rows]})
    pq.write_table(table, str(path), row_group_size=group_size)
    return rows


def make_corpus(tmp_path: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """Two files, three and two row groups of four documents: 20 documents in all."""
    a = write_parquet(tmp_path / "a.parquet", 0, 12, group_size=4)
    b = write_parquet(tmp_path / "b.parquet", 100, 8, group_size=4)
    return [tmp_path / "a.parquet", tmp_path / "b.parquet"], a + b


def splits(paths: Sequence[Path], where: Path, fraction: float) -> tuple[Split, Split]:
    """The training and held-out splits of ``paths`` as one component, finalized under ``where``."""
    fin = finalize([Component("all", tuple(paths), 1.0)], where)
    return Split(fin, fraction), Split(fin, fraction, held_out=True)


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_from_texts(
        [f"document {i} text with newline and {i * 7}" for i in range(50)], vocab=300
    )


def test_every_document_once_whole_and_shuffled(tmp_path: Path) -> None:
    paths, rows = make_corpus(tmp_path)
    seen = [(d.doc_id, d.text) for d in iter_raw_documents(paths, seed=3)]
    assert sorted(seen) == sorted(rows)
    assert seen != rows  # shuffled relative to each other
    assert [(d.doc_id, d.text) for d in iter_raw_documents(paths, seed=3)] == seen
    assert [(d.doc_id, d.text) for d in iter_raw_documents(paths, seed=4)] != seen


def test_the_shuffle_is_a_block_shuffle_over_row_groups(tmp_path: Path) -> None:
    # groups of four rows (file a: rows 0-3, 4-7, 8-11; file b: 100-103, 104-107) stay
    # contiguous in the output; the groups are permuted and so are the rows inside each group
    paths, rows = make_corpus(tmp_path)
    seen = [d.doc_id for d in iter_raw_documents(paths, seed=3)]
    positions = {doc_id: k for k, doc_id in enumerate(seen)}
    groups = [rows[i : i + 4] for i in range(0, 20, 4)]
    block_starts: list[int] = []
    inner_orders: list[list[int]] = []
    for group in groups:
        where = sorted(positions[doc_id] for doc_id, _ in group)
        assert where == list(range(where[0], where[0] + 4))  # contiguous block of four
        block_starts.append(where[0])
        inner_orders.append([positions[doc_id] - where[0] for doc_id, _ in group])
    assert block_starts != sorted(block_starts)  # the groups moved relative to each other
    assert any(order != [0, 1, 2, 3] for order in inner_orders)  # rows moved inside a group


def test_seeded_order_is_the_same_in_another_process(tmp_path: Path) -> None:
    # the permutations are keyed blake2b ranks: no generator state, no salted hash
    paths, _ = make_corpus(tmp_path)
    here = [d.doc_id for d in iter_raw_documents(paths, seed=3)]
    code = (
        "import sys; from pathlib import Path; from vesper.corpus import iter_raw_documents; "
        "print('\\n'.join(d.doc_id for d in iter_raw_documents("
        "[Path(p) for p in sys.argv[1:]], seed=3)))"
    )
    for seed in ("0", "1", "random"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        run = subprocess.run(  # noqa: S603  (own interpreter, fixed code string)
            [sys.executable, "-c", code, *map(str, paths)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert run.stdout.split("\n")[: len(here)] == here


def test_many_files_need_only_a_handful_of_descriptors(tmp_path: Path) -> None:
    # 80 files under a hard limit of 64 open descriptors: readers are opened per read and
    # closed again, so the count of files is not a count of open handles
    for i in range(80):
        _ = write_parquet(tmp_path / f"f{i:03d}.parquet", 1000 * i, 2, group_size=2)
    code = (
        "import resource, sys; from pathlib import Path; "
        "from vesper.corpus import iter_raw_documents; "
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64)); "
        "paths = sorted(Path(sys.argv[1]).glob('*.parquet')); "
        "print(sum(1 for _ in iter_raw_documents(paths, seed=0)))"
    )
    run = subprocess.run(  # noqa: S603  (own interpreter, fixed code string)
        [sys.executable, "-c", code, str(tmp_path)], check=True, capture_output=True, text=True
    )
    assert run.stdout.strip() == "160"


def test_splits_are_disjoint_and_cover_the_corpus(tmp_path: Path, tok: Tokenizer) -> None:
    paths, rows = make_corpus(tmp_path)
    training, evaluation = splits(paths, tmp_path / "fin", FRACTION)
    train = list(iter_documents(paths, 3, tok, training))
    held = list(iter_documents(paths, 3, tok, evaluation))
    assert {d.doc_id for d in train}.isdisjoint({d.doc_id for d in held})
    assert sorted(d.doc_id for d in train + held) == sorted(r[0] for r in rows)
    assert held  # the fraction is chosen so both sides are populated
    assert all(is_held_out(d.doc_id, FRACTION) for d in held)
    assert not any(is_held_out(d.doc_id, FRACTION) for d in train)


def test_documents_are_tokenized_with_the_given_tokenizer(tmp_path: Path, tok: Tokenizer) -> None:
    paths, rows = make_corpus(tmp_path)
    training, _ = splits(paths, tmp_path / "fin", FRACTION)
    text_of = dict(rows)
    for d in iter_documents(paths, 5, tok, training):
        assert d.tokens.dtype == torch.int32
        assert torch.equal(d.tokens, tok.encode(text_of[d.doc_id]))
        assert tok.decode(d.tokens) == text_of[d.doc_id]


def test_training_texts_stop_at_the_byte_budget_and_skip_held_out(tmp_path: Path) -> None:
    paths, rows = make_corpus(tmp_path)
    training, _ = splits(paths, tmp_path / "fin", FRACTION)
    text_of = dict(rows)
    texts = list(iter_training_texts(paths, 3, training, byte_budget=120))
    total = sum(len(t.encode("utf-8")) for t in texts)
    assert total >= 120
    assert total - len(texts[-1].encode("utf-8")) < 120  # the last text crossed the budget
    held_texts = {text_of[r[0]] for r in rows if is_held_out(r[0], FRACTION)}
    assert not any(t in held_texts for t in texts)
    everything = list(iter_training_texts(paths, 3, training, byte_budget=10**9))
    assert len(everything) == sum(not is_held_out(r[0], FRACTION) for r in rows)
    with pytest.raises(ValueError, match="byte_budget"):
        _ = next(iter_training_texts(paths, 3, training, byte_budget=0))


def test_rejects_missing_columns_wrong_types_and_no_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _ = next(iter_raw_documents([], seed=0))
    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pydict({"id": ["x"], "body": ["y"]}), str(bad))
    with pytest.raises(ValueError, match="missing column"):
        _ = next(iter_raw_documents([bad], seed=0))
    typed = tmp_path / "typed.parquet"
    pq.write_table(pa.Table.from_pydict({"id": [1, 2], "text": ["a", "b"]}), str(typed))
    with pytest.raises(TypeError, match="not str"):
        _ = next(iter_raw_documents([typed], seed=0))
    partial = tmp_path / "partial.parquet"
    pq.write_table(
        pa.Table.from_pydict({"id": ["x"], "text": ["y"], "group_key": ["g"]}), str(partial)
    )
    with pytest.raises(ValueError, match="key columns"):
        _ = next(iter_raw_documents([partial], seed=0))


def write_collected(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    """A collected file: ``(id, text, group_key, dedup_key)`` rows with the collector's columns."""
    table = pa.Table.from_pydict(
        {
            "id": [r[0] for r in rows],
            "text": [r[1] for r in rows],
            "group_key": [r[2] for r in rows],
            "dedup_key": [r[3] for r in rows],
            "dedup_version": ["code-v1"] * len(rows),
        }
    )
    pq.write_table(table, str(path))


def opposite_sides(fraction: float) -> dict[bool, str]:
    """Two ids of one repository that fall on different sides of the split at ``fraction``."""
    sides: dict[bool, str] = {}
    for i in range(1000):
        doc_id = f"code:owner/repo/f{i}.py@rev"
        _ = sides.setdefault(is_held_out(doc_id, fraction), doc_id)
        if len(sides) == 2:
            break
    return sides


def test_duplicates_are_read_once_and_a_group_stays_on_one_side(
    tmp_path: Path, tok: Tokenizer
) -> None:
    # Maintainer B's fixture (collector review, finding 1): two identical files of one repository
    # whose ids fall on different sides at fraction 0.5; the reader yields one of them, on the side
    # of the group, not one on each side
    sides = opposite_sides(0.5)
    same = dedup_key("print(1)\n", "code")[1]
    rows = [
        (sides[False], "print(1)\n", "code:owner/repo", same),
        (sides[True], "print(1)\n", "code:owner/repo", same),
        ("code:o/two/a.py@rev", "a\n", "code:o/two", dedup_key("a\n", "code")[1]),
        ("code:o/two/b.py@rev", "b\n", "code:o/two", dedup_key("b\n", "code")[1]),
    ]
    path = tmp_path / "collected.parquet"
    write_collected(path, rows)
    training, evaluation = splits([path], tmp_path / "fin", 0.5)
    assert training.finalization.dropped == {max(sides.values())}  # the smaller id survives
    train = [d.doc_id for d in iter_documents([path], 0, tok, training)]
    held = [d.doc_id for d in iter_documents([path], 0, tok, evaluation)]
    everything = train + held
    assert len(everything) == 3  # the duplicate is read once
    assert min(sides.values()) in everything
    two = held if is_held_out("code:o/two", 0.5) else train
    assert {"code:o/two/a.py@rev", "code:o/two/b.py@rev"} <= set(two)
    texts = list(iter_training_texts([path], 0, training, byte_budget=10**9))
    assert texts.count("print(1)\n") <= 1


def test_membership_does_not_depend_on_the_seed(tmp_path: Path, tok: Tokenizer) -> None:
    # Maintainer B's fixture (collector review, second round): identical content in two groups whose
    # hashes choose opposite sides; under first-seen deduplication the content's partition
    # followed the shuffle order. Finalization decides by the ids, so every seed agrees.
    groups: dict[bool, str] = {}
    for i in range(1000):
        group = f"code:owner/repo{i}"
        _ = groups.setdefault(is_held_out(group, 0.5), group)
        if len(groups) == 2:
            break
    same = dedup_key("shared\n", "code")[1]
    rows = [
        (f"{groups[False]}/x.py@rev", "shared\n", groups[False], same),
        (f"{groups[True]}/x.py@rev", "shared\n", groups[True], same),
    ]
    # two files so the two seeds can order the two documents differently
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    write_collected(first, rows[:1])
    write_collected(second, rows[1:])
    training, evaluation = splits([first, second], tmp_path / "fin", 0.5)
    orders = {
        seed: [d.doc_id for d in iter_raw_documents([first, second], seed)] for seed in range(8)
    }
    assert len({tuple(o) for o in orders.values()}) == 2  # both orders occur
    memberships: set[tuple[frozenset[str], frozenset[str]]] = set()
    for seed in range(8):
        train = frozenset(d.doc_id for d in iter_documents([first, second], seed, tok, training))
        held = frozenset(d.doc_id for d in iter_documents([first, second], seed, tok, evaluation))
        memberships.add((train, held))
    assert len(memberships) == 1
    survivor = min(r[0] for r in rows)
    train_ids, held_ids = next(iter(memberships))
    assert train_ids | held_ids == {survivor}


def test_finalize_writes_a_receipt_that_loads_back_and_checks_its_inputs(tmp_path: Path) -> None:
    paths, _ = make_corpus(tmp_path)
    extra = tmp_path / "extra.parquet"
    # two more web documents with document 3's text under other ids
    text = f"document 3 text with\nnewline and {3 * 7}"
    pq.write_table(
        pa.Table.from_pydict({"id": ["<urn:zzz>", "<urn:aaa>"], "text": [text, text]}), str(extra)
    )
    parts = [Component("web", (paths[0],), 0.5), Component("more", (paths[1], extra), 0.5)]
    fin = finalize(parts, tmp_path / "fin")
    # three ids share the key; the smallest, "<urn:aaa>", survives
    assert fin.dropped == {"<urn:uuid:" + f"{3:032x}" + ">", "<urn:zzz>"}
    record = cast("dict[str, object]", json.loads(fin.receipt.read_text()))
    assert record["documents"] == 22
    assert record["repeated_keys"] == 1
    assert record["dropped"] == 2
    assert record["dropped_per_component"] == {"web": 1, "more": 1}
    inputs = cast("list[dict[str, object]]", record["inputs"])
    assert [cast("str", r["component"]) for r in inputs] == ["web", "more", "more"]
    again = load_finalization(fin.receipt)
    assert again.dropped == fin.dropped
    assert again.inputs == fin.inputs
    check_inputs(again, parts)
    with pytest.raises(ValueError, match="different set"):
        check_inputs(again, parts[:1])
    _ = write_parquet(paths[0], 0, 12, group_size=3)  # same rows, other layout: other hash
    with pytest.raises(ValueError, match="not taken over"):
        check_inputs(again, parts)
    dropped_file = fin.receipt.parent / cast("str", record["dropped_file"])
    _ = dropped_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        _ = load_finalization(fin.receipt)
    with pytest.raises(ValueError, match="at least one"):
        _ = finalize([], tmp_path / "fin2")


def test_web_files_without_keys_group_by_id_and_dedup_by_normalized_text(
    tmp_path: Path, tok: Tokenizer
) -> None:
    path = tmp_path / "web.parquet"
    table = pa.Table.from_pydict(
        {
            "id": ["<urn:a>", "<urn:b>", "<urn:c>"],
            "text": ["Alpha  beta\nline", "Alpha beta\nline", "Gamma"],
        }
    )
    pq.write_table(table, str(path))
    training, _ = splits([path], tmp_path / "fin", 0.0)
    assert training.finalization.dropped == {"<urn:b>"}  # whitespace runs: one document
    ids = [d.doc_id for d in iter_documents([path], 0, tok, training)]
    assert sorted(ids) == ["<urn:a>", "<urn:c>"]
    assert len(list(iter_training_texts([path], 0, training, byte_budget=10**9))) == 2


def mixed_components(tmp_path: Path) -> tuple[list[Component], Split]:
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "id": [f"a{i:02d}" for i in range(40)],
                "text": [f"A{i:02d}" + "x" * 96 for i in range(40)],
            }
        ),
        str(a),
    )
    # b's second text duplicates one of a's: across components it is one document, a's (the
    # smaller id)
    pq.write_table(
        pa.Table.from_pydict(
            {"id": ["b0", "b1", "b2"], "text": ["B0" + "y" * 98, "A00" + "x" * 96, "B2"]}
        ),
        str(b),
    )
    parts = [Component("a", (a,), 0.75), Component("b", (b,), 0.25)]
    return parts, Split(finalize(parts, tmp_path / "fin"), 0.0)


def test_mixed_texts_track_the_shares_dedup_across_components_and_stop_when_short(
    tmp_path: Path,
) -> None:
    parts, training = mixed_components(tmp_path)
    assert training.finalization.dropped == {"b1"}
    # budgets 4500 and 1500 exhaust both components: 40 + 3 documents minus the one duplicate
    whole = list(iter_mixed_texts(parts, 0, training, byte_budget=6000))
    assert len(whole) == 42
    assert sum(1 for name, _ in whole if name == "a") == 40
    assert [t for _, t in whole].count("A00" + "x" * 96) == 1
    # budgets 900 and 300: a crosses 900 by one whole document at most; b runs out of documents
    # before its budget and the sample is short of it, nothing invented
    pairs = list(iter_mixed_texts(parts, 0, training, byte_budget=1200))
    a_bytes = sum(len(t) for name, t in pairs if name == "a")
    b_bytes = sum(len(t) for name, t in pairs if name == "b")
    assert 900 <= a_bytes < 1000
    assert b_bytes == 100 + 2
    # the first document comes from the component with the smallest emitted/budget ratio: a tie
    # at zero goes to the earlier component; b appears before a has used three times b's bytes
    assert pairs[0][0] == "a"
    first_b = next(k for k, (name, _) in enumerate(pairs) if name == "b")
    assert first_b <= 4
    a_only = [Component("a", parts[0].paths, 0.5)]
    with pytest.raises(ValueError, match="shares"):
        _ = next(iter_mixed_texts(a_only, 0, training, 100))
    with pytest.raises(ValueError, match="at least one"):
        _ = next(iter_mixed_texts([], 0, training, 100))
    with pytest.raises(ValueError, match="without a byte"):
        _ = next(iter_mixed_texts(parts, 0, training, 1))  # budgets [0, 1]: a gets no byte
    # the single-component sample is the same function
    alone = list(iter_training_texts(parts[0].paths, 0, training, byte_budget=500))
    whole_a = [Component("a", parts[0].paths, 1.0)]
    assert alone == [t for _, t in iter_mixed_texts(whole_a, 0, training, 500)]


def test_mixed_documents_track_token_shares_and_resume_by_skipping(
    tmp_path: Path, tok: Tokenizer
) -> None:
    parts, training = mixed_components(tmp_path)
    counts: dict[str, int] = {}
    docs = list(iter_mixed_documents(parts, 0, tok, training, tokens_emitted=counts))
    assert len(docs) == 42
    assert set(counts) == {"a", "b"}
    assert counts["a"] == sum(int(d.tokens.numel()) for d in docs if d.doc_id.startswith("a"))
    # b runs out first; until then the token ratio follows 3:1 at document granularity
    first_b = next(k for k, d in enumerate(docs) if d.doc_id.startswith("b"))
    assert first_b <= 4
    # a resumed run that skips the first five documents continues with the sixth, having made
    # the same component choices, and ends with the same realized counts
    replayed: dict[str, int] = {}
    resumed = list(iter_mixed_documents(parts, 0, tok, training, skip=5, tokens_emitted=replayed))
    assert [d.doc_id for d in resumed] == [d.doc_id for d in docs[5:]]
    assert replayed == counts
    with pytest.raises(ValueError, match="count"):
        _ = next(iter_mixed_documents(parts, 0, tok, training, skip=-1))
    # the mixer's digest is the identity of the order: a replay reproduces it, another seed
    # does not even when every count agrees (maintainer B's review of b797837: equal-length
    # documents under another seed passed the count check and the run diverged)
    whole = Mixer(parts, 0, tok, training)
    whole.replay(5)
    partial = Mixer(parts, 0, tok, training)
    for _ in range(5):
        assert partial.pull() is not None
    assert whole.digest() == partial.digest()
    assert whole.pulled == partial.pulled == 5
    other = Mixer(parts, 1, tok, training)
    other.replay(5)
    assert other.digest() != whole.digest()
    assert Mixer(parts, 0, tok, training).digest() == Mixer(parts, 1, tok, training).digest()
    # the bounded-cost resume: a mixer restored from the state after five documents hands out
    # exactly what the uninterrupted one hands out next (the replay is the oracle), with the same
    # balances and digest at the end
    state = whole.state()
    assert state.pulled == 5
    restored = Mixer.restore(parts, 0, tok, training, state)
    assert restored.digest() == whole.digest()
    rest_restored = [d.doc_id for d in restored.documents()]
    rest_whole = [d.doc_id for d in whole.documents()]
    assert rest_restored == rest_whole
    assert len(rest_restored) == 42 - 5
    assert restored.emitted == whole.emitted
    assert restored.digest() == whole.digest()
    with pytest.raises(ValueError, match="readers"):
        _ = Mixer.restore(parts, 0, tok, training, MixerState(((0, 0),), (0,), {}, 0, "0" * 64))
    with pytest.raises(ValueError, match="outside"):
        _ = Mixer.restore(
            parts, 0, tok, training, MixerState(((0, 0), (0, 0)), (0, 7), {}, 0, "0" * 64)
        )
    halves = Split(training.finalization, 0.5)
    held = list(iter_mixed_documents(parts, 0, tok, Split(halves.finalization, 0.5, held_out=True)))
    train = list(iter_mixed_documents(parts, 0, tok, halves))
    assert len(held) + len(train) == 42
    assert {d.doc_id for d in held}.isdisjoint({d.doc_id for d in train})


def test_a_reader_continues_from_its_cursor(tmp_path: Path) -> None:
    paths, _ = make_corpus(tmp_path)
    reader = RawReader(paths, 0)
    first = [reader.next() for _ in range(5)]
    assert all(d is not None for d in first)
    cursor = reader.cursor()
    rest = [d.doc_id for d in reader.documents()]
    again = RawReader(paths, 0)
    again.seek(cursor)
    assert [d.doc_id for d in again.documents()] == rest
    end = again.cursor()
    assert again.next() is None
    fresh = RawReader(paths, 0)
    fresh.seek(end)
    assert fresh.next() is None
    for bad in (ReaderCursor(end.pair + 1, 0), ReaderCursor(0, 10**6), ReaderCursor(-1, 0)):
        with pytest.raises(ValueError, match=r"outside|end"):
            RawReader(paths, 0).seek(bad)
    with pytest.raises(ValueError, match="row 0"):
        RawReader(paths, 0).seek(ReaderCursor(end.pair, 1))


def test_check_inputs_pins_the_files_assignment_to_components(tmp_path: Path) -> None:
    # Maintainer B's fixture (review of 444b966): the same files, the same flattened path and hash
    # list, another assignment to components: another mixture, refused
    paths, _ = make_corpus(tmp_path)
    extra = tmp_path / "extra.parquet"
    pq.write_table(
        pa.Table.from_pydict({"id": ["<urn:e1>", "<urn:e2>"], "text": ["eins", "zwei"]}),
        str(extra),
    )
    original = [Component("a", (paths[0],), 0.5), Component("b", (paths[1], extra), 0.5)]
    changed = [Component("a", (paths[0], paths[1]), 0.5), Component("b", (extra,), 0.5)]
    fin = finalize(original, tmp_path / "fin")
    check_inputs(fin, original)
    with pytest.raises(ValueError, match="assignment to components is pinned"):
        check_inputs(fin, changed)
    assert [c for c, _, _, _ in load_finalization(fin.receipt).inputs] == ["a", "b", "b"]
    # the generation pin is part of the receipt: the same files under another pin are refused
    assert [g for _, _, _, g in fin.inputs] == [None, None, None]
    pinned_parts = [
        Component("a", (paths[0],), 0.5, generation="a/gen-x@" + "0" * 64),
        Component("b", (paths[1], extra), 0.5),
    ]
    with pytest.raises(ValueError, match="the generation is pinned"):
        check_inputs(fin, pinned_parts)
    fin2 = finalize(pinned_parts, tmp_path / "fin2")
    assert [g for _, _, _, g in load_finalization(fin2.receipt).inputs] == [
        "a/gen-x@" + "0" * 64,
        None,
        None,
    ]
    check_inputs(fin2, pinned_parts)


def test_finalize_refuses_a_repeated_document_id(tmp_path: Path) -> None:
    # Maintainer B's fixture (collector review, fifth round): two acquisition runs of one unchanged
    # book in one directory give one id twice; a conflicting second occurrence is the same input
    # error
    a = tmp_path / "a.parquet"
    b = tmp_path / "b.parquet"
    same = dedup_key("same book", "code")[1]
    write_collected(a, [("book:title@commit", "same book", "book:title", same)])
    write_collected(b, [("book:title@commit", "same book", "book:title", same)])
    parts = [Component("x", (a,), 0.5), Component("y", (b,), 0.5)]
    with pytest.raises(ValueError, match="occurs more than once"):
        _ = finalize(parts, tmp_path / "fin")
    other = dedup_key("other text", "code")[1]
    write_collected(b, [("book:title@commit", "other text", "book:other", other)])
    with pytest.raises(ValueError, match="occurs more than once"):
        _ = finalize(parts, tmp_path / "fin")


def test_finalize_never_overwrites_a_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, fifth round): two runs in one second got one stamp
    paths, _ = make_corpus(tmp_path)
    parts = [Component("all", tuple(paths), 1.0)]
    fixed = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)

    class FixedClock:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            del tz
            return fixed

    monkeypatch.setattr("vesper.corpus.datetime", FixedClock)
    first = finalize(parts, tmp_path / "fin")
    before = first.receipt.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        _ = finalize(parts, tmp_path / "fin")
    assert first.receipt.read_bytes() == before


def test_finalize_reserves_its_receipt_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixture (collector review, seventh round): a second run that passed an
    # existence check before the first published; the reservation is the receipt file itself,
    # created exclusively before any work
    paths, _ = make_corpus(tmp_path)
    parts = [Component("all", tuple(paths), 1.0)]
    fixed = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)

    class FixedClock:
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            del tz
            return fixed

    monkeypatch.setattr("vesper.corpus.datetime", FixedClock)
    out = tmp_path / "fin"
    out.mkdir()
    reserved = out / "finalize-20261015T030000Z.json"
    reserved.touch()  # another run holds this stamp
    with pytest.raises(ValueError, match="already exists"):
        _ = finalize(parts, out)
    assert reserved.read_bytes() == b""  # the other run's reservation is untouched
    assert sorted(p.name for p in out.iterdir()) == [reserved.name]  # no partial left
    # the reservation is not a finalization: a reader refuses it instead of failing on its JSON
    with pytest.raises(ValueError, match="reserved, not published"):
        _ = load_finalization(reserved)
