"""The stage-1 task, checked by an oracle that reads only the tokens (vesper/synthetic.py).

The oracle rebuilds each sequence's graph from its definition pairs and recomputes every target
from the markers it sees, so a layout or target defect in the generator cannot agree with it by
sharing the generator's own bookkeeping.
"""

import hashlib
from collections import Counter
from pathlib import Path
from typing import cast

import pytest
import torch

from vesper.config import load
from vesper.synthetic import (
    BACKS,
    CLASSES,
    CYCLE,
    DEFAULT_MIX,
    FIRST_NODE,
    GAP,
    HOP_QUERIES,
    HOPS,
    MIX_SCHEDULES,
    READ,
    READ_POSITION,
    REGISTER_POSITION,
    ROLES,
    SEP,
    SEQUENCE_TOKENS,
    TERMINAL_MIX,
    VOCAB,
    _ints,  # pyright: ignore[reportPrivateUsage]
    back_token,
    episodes,
    generator_for,
    hop_mix,
    hop_token,
    register_token,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"


def _row(values: torch.Tensor) -> list[int]:
    return [int(v.item()) for v in values.unbind()]


def _graph(tokens: list[int]) -> dict[int, int]:
    """The definition pairs ``(a, b, SEP)`` at the start, as a successor map."""
    succ: dict[int, int] = {}
    at = 0
    while tokens[at + 2] == SEP and tokens[at] >= FIRST_NODE:
        assert tokens[at] not in succ, "a node bound twice"
        succ[tokens[at]] = tokens[at + 1]
        at += 3
    return succ


def _hop(succ: dict[int, int], node: int, k: int) -> int:
    for _ in range(k):
        node = succ[node]
    return node


def _oracle(tokens: list[int]) -> dict[int, tuple[str, int]]:
    """Every loss position the markers imply, with its class and target."""
    succ = _graph(tokens)
    expected: dict[int, tuple[str, int]] = {}
    for p, token in enumerate(tokens):
        for k in range(1, HOPS + 1):
            if token == hop_token(k):
                expected[p + 1] = (f"hop{k}", _hop(succ, tokens[p + 1], k))
            if token == register_token(k):
                expected[READ_POSITION] = (f"read{k}", _hop(succ, tokens[p + 1], k))
        for j in range(1, BACKS + 1):
            if token == back_token(j):
                expected[p] = (f"back{j}", tokens[p - j])
    return expected


def test_every_target_and_loss_position_matches_the_oracle() -> None:
    ep = episodes(generator_for("test", 0), 64, "t")
    c = ep.chunk
    for s in range(64):
        tokens = _row(c.tokens[s])
        succ = _graph(tokens)
        # one cycle through all eight nodes: from any node, CYCLE hops and no fewer return
        assert len(succ) == CYCLE
        start = tokens[0]
        assert [_hop(succ, start, k) == start for k in range(1, CYCLE + 1)] == [False] * 7 + [True]
        expected = _oracle(tokens)
        assert set(expected) == {p for p in range(SEQUENCE_TOKENS) if bool(c.loss_mask[s, p])}
        for p, (name, target) in expected.items():
            assert int(c.targets[s, p]) == target, (s, p, name)
            assert CLASSES[int(ep.task[s, p])] == name
            if name.startswith(("hop", "read")):
                assert target != tokens[p if name.startswith("hop") else REGISTER_POSITION]
        assert tokens[READ_POSITION] == READ
        # HOP_QUERIES hops with every depth at least once, each copy class once, one read; copy
        # content never names a cycle node
        kinds = Counter(name[:-1] for name, _ in expected.values())
        assert kinds == {"hop": HOP_QUERIES, "back": BACKS, "read": 1}
        hops = Counter(name for name, _ in expected.values() if name.startswith("hop"))
        assert set(hops) == {f"hop{k}" for k in range(1, HOPS + 1)}
        for p, (name, _) in expected.items():
            if name.startswith("back"):
                assert all(tokens[q] not in succ for q in range(p - 3, p))


def test_the_mix_sets_the_free_queries_and_every_depth_stays_mandatory() -> None:
    # all free queries at one depth: exactly that depth fills them, the others appear once each
    for k in range(1, HOPS + 1):
        mix = tuple(1.0 if i == k else 0.0 for i in range(1, HOPS + 1))
        ep = episodes(generator_for("test", 5), 8, "t", mix)
        for s in range(8):
            hops = Counter(CLASSES[t] for t in _row(ep.task[s]) if t >= 0)
            expected = {f"hop{i}": 1 for i in range(1, HOPS + 1)}
            expected[f"hop{k}"] += HOP_QUERIES - HOPS
            assert {n: v for n, v in hops.items() if n.startswith("hop")} == expected, (k, s)
    # the terminal mix spreads the free queries evenly: 16 x 400 / 4 = 1600 per depth expected,
    # 1400 is more than five sigma below
    ep = episodes(generator_for("test", 6), 400, "t")
    hops = Counter(CLASSES[t] for t in _row(ep.task.reshape(-1)) if t >= 0)
    assert all(hops[f"hop{k}"] > 400 + 1400 for k in range(1, HOPS + 1))


def test_the_training_mix_follows_its_schedule() -> None:
    # v1 is the default and unchanged: the 12000-update probe trained under it
    assert hop_mix(0) == hop_mix(0, "v1") == (0.7, 0.1, 0.1, 0.1)
    assert hop_mix(999) == (0.7, 0.1, 0.1, 0.1)
    assert hop_mix(1000) == (0.4, 0.3, 0.15, 0.15)
    assert hop_mix(2999) == (0.4, 0.3, 0.15, 0.15)
    assert hop_mix(3000) == TERMINAL_MIX
    assert hop_mix(10**6) == TERMINAL_MIX
    # v2-slow, the one-column calibration's recipe (maintainer B, 2026-10-16), at every stage
    # boundary
    assert hop_mix(7999, "v2-slow") == (0.7, 0.1, 0.1, 0.1)
    assert hop_mix(8000, "v2-slow") == (0.4, 0.3, 0.15, 0.15)
    assert hop_mix(15999, "v2-slow") == (0.4, 0.3, 0.15, 0.15)
    assert hop_mix(16000, "v2-slow") == TERMINAL_MIX
    assert hop_mix(35999, "v2-slow") == TERMINAL_MIX
    for stages in MIX_SCHEDULES.values():
        assert stages[0][0] == 0
        assert [start for start, _ in stages] == sorted({start for start, _ in stages})
        for _, mix in stages:
            assert sum(mix) == pytest.approx(1.0)
    assert DEFAULT_MIX == "v1"
    with pytest.raises(ValueError, match="at least 0"):
        _ = hop_mix(-1)
    with pytest.raises(ValueError, match="unknown mix schedule"):
        _ = hop_mix(0, "v3")


def test_a_mix_that_is_not_a_distribution_is_refused() -> None:
    g = generator_for("test", 0)
    for bad in ((0.5, 0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (1.5, -0.5, 0.0, 0.0), (float("nan"),) * 4):
        with pytest.raises(ValueError, match="probabilities"):
            _ = episodes(g, 1, "t", bad)


def test_every_register_depth_occurs_in_every_batch() -> None:
    ep = episodes(generator_for("test", 7), 12, "t")
    reads = [CLASSES[int(t)] for t in _row(ep.task[:, READ_POSITION])]
    assert reads == [f"read{1 + s % HOPS}" for s in range(12)]
    marker = _row(ep.chunk.tokens[:, REGISTER_POSITION - 1])
    assert marker == [register_token(1 + s % HOPS) for s in range(12)]


def test_no_query_attends_to_a_definition_directly() -> None:
    # a direct edge only: stacked window attentions can still relay a definition across the
    # gap (maintainer B's review of c7d25c5), which the driver's recurrence ablation measures
    cfg = load(CONFIG)
    assert cfg.model.vocab == VOCAB
    assert cfg.model.window <= GAP
    assert cfg.training.chunk_tokens == SEQUENCE_TOKENS
    ep = episodes(generator_for("test", 1), 32, "t")
    last_definition = int((ep.role[0] == ROLES.index("definition")).nonzero().max())
    for s in range(32):
        for p in range(SEQUENCE_TOKENS):
            name = CLASSES[int(ep.task[s, p])] if int(ep.task[s, p]) >= 0 else ""
            if name.startswith(("hop", "read")):
                # a query sits beyond the window of every definition (attention sees p-w+1..p)
                assert p - last_definition >= cfg.model.window, (s, p)
        assert cfg.model.window <= READ_POSITION - REGISTER_POSITION


def test_roles_cover_every_position_and_agree_with_the_tokens() -> None:
    ep = episodes(generator_for("test", 2), 16, "t")
    c = ep.chunk
    assert torch.equal(ep.task >= 0, c.loss_mask)
    for s in range(16):
        tokens = _row(c.tokens[s])
        roles = [ROLES[r] for r in _row(ep.role[s])]
        assert roles[REGISTER_POSITION - 1] == "register_marker"
        assert roles[REGISTER_POSITION] == "register"
        assert roles[READ_POSITION - 1] == "fill"
        assert roles[READ_POSITION] == "read"
        assert Counter(roles) == {
            "definition": 2 * CYCLE,
            "separator": CYCLE,
            "gap": GAP,
            "register_marker": 1,
            "register": 1,
            "hop_marker": HOP_QUERIES,
            "hop": HOP_QUERIES,
            "copy_content": 3 * BACKS,
            "back": BACKS,
            "fill": 1,
            "read": 1,
        }
        assert sum(Counter(roles).values()) == SEQUENCE_TOKENS
        for p, role in enumerate(roles):
            if role in {"hop", "back", "read"}:
                assert bool(c.loss_mask[s, p])
            else:
                assert not bool(c.loss_mask[s, p])
            if role in {"gap", "fill"}:
                assert tokens[p] == 0


def test_the_chunk_is_one_valid_document_per_slot() -> None:
    c = episodes(generator_for("test", 3), 4, "tag").chunk
    assert c.tokens.dtype == torch.int32
    assert c.targets.dtype == torch.int32
    assert bool(c.valid.all())
    assert c.doc_start[:, 0].all()
    assert not c.doc_start[:, 1:].any()
    assert torch.equal(c.pos[1], torch.arange(SEQUENCE_TOKENS))
    assert c.doc_ids == (("tag-0",), ("tag-1",), ("tag-2",), ("tag-3",))
    assert int(c.tokens.max()) < VOCAB
    assert int(c.targets.max()) < VOCAB
    assert int(c.loss_mask.sum()) == 4 * (HOP_QUERIES + BACKS + 1)


def test_splits_and_seeds_are_reproducible_and_independent() -> None:
    a = episodes(generator_for("train", 0), 8, "x").chunk.tokens
    assert torch.equal(a, episodes(generator_for("train", 0), 8, "x").chunk.tokens)
    assert not torch.equal(a, episodes(generator_for("train", 1), 8, "x").chunk.tokens)
    assert not torch.equal(a, episodes(generator_for("validation", 0), 8, "x").chunk.tokens)
    v = episodes(generator_for("validation", 0), 8, "x").chunk.tokens
    assert not torch.equal(v, episodes(generator_for("final", 0), 8, "x").chunk.tokens)


def test_items_take_every_order() -> None:
    # the first item is a copy item in about 3 of 23 sequences (240 of 1840 expected)
    ep = episodes(generator_for("test", 4), 1840, "t")
    first = _row(ep.chunk.tokens[:, REGISTER_POSITION + 1])
    hop_markers = {hop_token(k) for k in range(1, HOPS + 1)}
    copies = sum(1 for t in first if t not in hop_markers)
    assert 160 < copies < 320


def test_a_batch_needs_a_sequence() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        _ = episodes(generator_for("test", 0), 0, "t")


# The generator's output as of 07b39f2 (the recipe every stage-1 run so far trained on): batches
# from three seeds, mixes and sizes, and the first validation batch. Any change to a draw, its
# order or the layout changes this digest; a faster generator must reproduce it exactly.
# (a sha256 of generated batches, not a secret)
GENERATOR_DIGEST = (
    "65c58dd3b172c1d6e2d477a8224455cc158b3bf917c70b283e326e33c99740a4"  # pragma: allowlist secret
)


def test_the_generator_reproduces_its_recorded_batches_exactly() -> None:
    h = hashlib.sha256()
    cases = [
        ("train", 0, 32, "v2-slow", 0),
        ("train", 0, 32, "v2-slow", 20000),
        ("train", 3, 7, "v1", 500),
    ]
    for split, seed, b, mix, step in cases:
        g = generator_for(split, seed)
        for i in range(3):
            ep = episodes(g, b, f"{split}-{i}", hop_mix(step + i, mix))
            c = ep.chunk
            tensors = (c.tokens, c.targets, c.valid, c.doc_start, c.segment, c.loss_mask, c.pos)
            for t in (*tensors, ep.role, ep.task):
                h.update(_record(t))
            h.update(repr((c.doc_ids, c.spans)).encode())
    ep = episodes(generator_for("validation", 0), 32, "validation-0", TERMINAL_MIX)
    for t in (ep.chunk.tokens, ep.chunk.targets, ep.task):
        h.update(_record(t))
    assert h.hexdigest() == GENERATOR_DIGEST


def _record(t: torch.Tensor) -> bytes:
    """A tensor's dtype, shape and values as bytes, for the digest."""
    values = cast("list[int]", t.flatten().long().tolist())  # pyright: ignore[reportUnknownMemberType]
    return f"{t.dtype}{tuple(t.shape)}{values}".encode()


def test_ints_takes_only_a_one_dimensional_integer_tensor() -> None:
    assert _ints(torch.arange(3, 0, -1)) == [3, 2, 1]
    with pytest.raises(ValueError, match="1-D integer"):
        _ = _ints(torch.full((1,), 1.5))
    with pytest.raises(ValueError, match="1-D integer"):
        _ = _ints(torch.zeros(2, 2, dtype=torch.int64))
