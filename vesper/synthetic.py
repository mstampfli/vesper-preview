"""The halting experiment's stage-1 task: synthetic sequences whose useful depth differs by token.

Plan: docs/reviews/2026-10-15-halting-experiment-plan.md, stage 1; the task's second version and
its frozen recipe: docs/reviews/2026-10-15-retrieval-binding.md, section 5 (the first version, 4
hop queries per 56-token sequence, was not learned by the model as built in 16000 steps). Every
sequence is one document of ``SEQUENCE_TOKENS`` positions with its own random graph, so nothing
about a graph can be memorized across sequences; a split's sequences come from a generator seeded
by the split name and the seed (``generator_for``), and a held-out graph equals a training graph
only by a coincidence of about one in ``48! / 40! / 8``.

Layout (positions, ``n = 88``):

- ``0..23`` definitions: an 8-node cycle ``c_0 -> c_1 -> ... -> c_7 -> c_0`` over nodes drawn
  without replacement, its eight bindings ``(c_i, c_{i+1}, SEP)`` in random order. A cycle and
  not a chain: a chain's end would answer every long query without a single hop. A hop of 1..4
  never returns the query node, so chance among the other seven nodes is 1/7 (maintainer B).
- ``24..31`` gap: ``GAP`` fills. With the model's attention window at most ``GAP`` (the stage-1
  config has 8), no query attends to a definition DIRECTLY. The task is still HYBRID: stacked
  window attentions (columns x passes layers, each relaying up to ``window - 1`` positions)
  can carry a definition across the gap, which maintainer B measured in the real model with every
  recurrent state zeroed (review of c7d25c5), and the recurrence also relays (maintainer B's review
  of d9eada0). The driver's recurrence ablation measures how much each answer depends on the
  recurrent state.
- ``32, 33`` register: ``REG_k`` and a cycle node ``a``. The register carries no loss; its value
  is needed at ``READ``. Within a batch, sequence ``s`` holds ``k = 1 + (s mod HOPS)``, so every
  register depth occurs in every batch of at least ``HOPS`` sequences.
- ``34..85`` ``HOP_QUERIES + BACKS`` items in random order: hop items ``(HOP_k, a)``, loss at
  ``a``, target ``succ^k(a)``: one of each ``k = 1..HOPS`` in every sequence and the other
  ``HOP_QUERIES - HOPS`` with ``k`` drawn from the batch's ``mix`` (``hop_mix`` gives the training
  schedule, ``TERMINAL_MIX`` the held-out sets); copy items ``(x_1, x_2, x_3, BACK_j)`` for
  ``j = 1..BACKS`` over non-cycle nodes, loss at ``BACK_j``, target the token ``j`` positions back.
- ``86`` fill, ``87`` ``READ``: loss, target ``succ^k(a)`` of the register's ``k`` and ``a``,
  54 positions after the register, so beyond the window too.

Hop depth ``k`` is an algorithmic count, not a depth need: with two columns a pass answered hop2
in the diagnostics (maintainer B), and each class's empirical depth profile is measured from
fixed-depth readouts on held-out sequences. Loss positions are exactly the eleven classes of
``CLASSES``; every other position carries none (``loss_mask`` false), so a halting head is trained
only where an answer is scored.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor

from vesper.data import Chunk

VOCAB = 64
FILL = 0
SEP = 1
READ = 2
HOPS = 4  # hop depths k = 1..HOPS
BACKS = 3  # copy queries j = 1..BACKS
HOP_QUERIES = 20  # hop items per sequence: one of each depth, the rest from the mix
FIRST_NODE = 16
NODES = VOCAB - FIRST_NODE  # 48 node symbols
CYCLE = 8
GAP = 8
SEQUENCE_TOKENS = 88
REGISTER_POSITION = 33
READ_POSITION = SEQUENCE_TOKENS - 1
TERMINAL_MIX = (0.25, 0.25, 0.25, 0.25)
MIX_TOLERANCE = 1e-9  # a mix sums to 1 within this
# the training mixes of the free hop queries over k = 1..HOPS, a versioned recipe per name:
# (first step of the stage, mix). A run records the name and its stages; a recipe never changes
# under its name, a new one gets a new name.
MIX_SCHEDULES: dict[str, tuple[tuple[int, tuple[float, ...]], ...]] = {
    # the 12000-update learnability probe (1678173; docs/reviews/2026-10-15-retrieval-binding.md)
    "v1": ((0, (0.7, 0.1, 0.1, 0.1)), (1000, (0.4, 0.3, 0.15, 0.15)), (3000, TERMINAL_MIX)),
    # the one-column calibration (docs/reviews/2026-10-16-stage1-c1-recipe.md): the stages moved
    # to the measured onsets of hop1 and hop2, 3500-6500 and 8000-10000 updates
    "v2-slow": ((0, (0.7, 0.1, 0.1, 0.1)), (8000, (0.4, 0.3, 0.15, 0.15)), (16000, TERMINAL_MIX)),
}
MIX_NAMES: tuple[str, ...] = tuple(MIX_SCHEDULES.keys())
DEFAULT_MIX = "v1"

ROLES = (
    "definition",
    "separator",
    "gap",
    "register_marker",
    "register",
    "hop_marker",
    "hop",
    "copy_content",
    "back",
    "fill",
    "read",
)
CLASSES = (
    *(f"hop{k}" for k in range(1, HOPS + 1)),
    *(f"back{j}" for j in range(1, BACKS + 1)),
    *(f"read{k}" for k in range(1, HOPS + 1)),
)

_ITEMS_START = REGISTER_POSITION + 1
_DEFINITIONS = 3 * CYCLE


def hop_token(k: int) -> int:
    """The marker of a ``k``-hop query (``1 <= k <= HOPS``)."""
    return READ + k


def back_token(j: int) -> int:
    """The marker of a copy query ``j`` positions back (``1 <= j <= BACKS``)."""
    return READ + HOPS + j


def register_token(k: int) -> int:
    """The marker of a register that holds a ``k``-hop resolution (``1 <= k <= HOPS``)."""
    return READ + HOPS + BACKS + k


def hop_mix(step: int, schedule: str = DEFAULT_MIX) -> tuple[float, ...]:
    """The training mix of the free hop queries at ``step`` under a recipe of ``MIX_SCHEDULES``.

    Raises:
        ValueError: if ``step`` is negative or ``schedule`` is not a recipe name.
    """
    if schedule not in MIX_SCHEDULES:
        msg = f"unknown mix schedule {schedule!r}; known: {sorted(MIX_SCHEDULES)}"
        raise ValueError(msg)
    if step < 0:
        msg = f"step must be at least 0, got {step}"
        raise ValueError(msg)
    return next(mix for start, mix in reversed(MIX_SCHEDULES[schedule]) if step >= start)


@dataclass(frozen=True)
class Episodes:
    """``b`` synthetic sequences: the chunk the model sees and what every position is.

    ``role int64[b, n]`` indexes ``ROLES`` at every position; ``task int64[b, n]`` indexes
    ``CLASSES`` at the loss positions and is -1 elsewhere, so ``task >= 0`` equals
    ``chunk.loss_mask``.
    """

    chunk: Chunk
    role: Tensor
    task: Tensor


def generator_for(split: str, seed: int) -> torch.Generator:
    """A generator for one split and seed; distinct names give independent streams."""
    digest = hashlib.sha256(f"vesper-synthetic:{split}:{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)


def episodes(
    generator: torch.Generator, b: int, tag: str, mix: Sequence[float] = TERMINAL_MIX
) -> Episodes:
    """Draw ``b`` sequences from ``generator``; ``tag`` names them in ``chunk.doc_ids``.

    ``mix`` is the probability of each depth ``k = 1..HOPS`` for the free hop queries.

    Raises:
        ValueError: if ``b`` is below 1, or ``mix`` is not ``HOPS`` finite non-negative
            probabilities summing to 1.
    """
    if b < 1:
        msg = f"b must be at least 1, got {b}"
        raise ValueError(msg)
    probabilities: list[float] = [float(p) for p in mix]
    weights = torch.tensor(probabilities, dtype=torch.float64)
    if (
        weights.shape != (HOPS,)
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or abs(float(weights.sum()) - 1.0) > MIX_TOLERANCE
    ):
        msg = f"mix must be {HOPS} non-negative probabilities summing to 1, got {tuple(mix)}"
        raise ValueError(msg)
    n = SEQUENCE_TOKENS
    rows = [_draw(generator, weights, 1 + s % HOPS) for s in range(b)]
    token_rows: list[list[int]] = [r[0] for r in rows]
    target_rows: list[list[int]] = [r[1] for r in rows]
    role_rows: list[list[int]] = [r[2] for r in rows]
    task_rows: list[list[int]] = [r[3] for r in rows]
    tokens = torch.tensor(token_rows, dtype=torch.int64)
    targets = torch.tensor(target_rows, dtype=torch.int64)
    role = torch.tensor(role_rows, dtype=torch.int64)
    task = torch.tensor(task_rows, dtype=torch.int64)
    loss_mask = task >= 0
    doc_start = torch.zeros(b, n, dtype=torch.bool)
    doc_start[:, 0] = True
    chunk = Chunk(
        tokens=tokens.to(torch.int32),
        targets=targets.to(torch.int32),
        valid=torch.ones(b, n, dtype=torch.bool),
        doc_start=doc_start,
        segment=torch.zeros(b, n, dtype=torch.int64),
        loss_mask=loss_mask,
        pos=torch.arange(n, dtype=torch.int64).expand(b, n).contiguous(),
        doc_ids=tuple((f"{tag}-{s}",) for s in range(b)),
        spans=tuple((("synthetic", n),) for _ in range(b)),
    )
    return Episodes(chunk=chunk, role=role, task=task)


Row = tuple[list[int], list[int], list[int], list[int]]
_ROLE = {name: index for index, name in enumerate(ROLES)}
_CLASS = {name: index for index, name in enumerate(CLASSES)}


def _draw(g: torch.Generator, weights: Tensor, k_reg: int) -> Row:
    """One sequence's rows (the layout of the module docstring): tokens, targets, role, task.

    Built as Python lists and turned into tensors once per batch by ``episodes``: an element
    write into a tensor costs a dispatch each. The draws from ``g`` are the same calls in the same
    order as when the rows were tensors, so a seed gives the same batches (tests/test_synthetic.py
    pins a digest of them).
    """
    n = SEQUENCE_TOKENS
    tokens, targets = [FILL] * n, [0] * n
    role, task = [_ROLE["gap"]] * n, [-1] * n
    nodes = _ints(torch.randperm(NODES, generator=g) + FIRST_NODE)
    cycle, pool = nodes[:CYCLE], nodes[CYCLE:]
    for slot, i in enumerate(_ints(torch.randperm(CYCLE, generator=g))):
        at = 3 * slot
        tokens[at : at + 3] = [cycle[i], cycle[(i + 1) % CYCLE], SEP]
        role[at : at + 3] = [_ROLE["definition"], _ROLE["definition"], _ROLE["separator"]]
    i_reg = _below(g, CYCLE)
    tokens[REGISTER_POSITION - 1], tokens[REGISTER_POSITION] = register_token(k_reg), cycle[i_reg]
    role[REGISTER_POSITION - 1] = _ROLE["register_marker"]
    role[REGISTER_POSITION] = _ROLE["register"]
    free = HOP_QUERIES - HOPS
    drawn = _ints(torch.multinomial(weights, free, replacement=True, generator=g) + 1)
    depths: list[int] = list(range(1, HOPS + 1)) + drawn
    items: list[tuple[str, int]] = [("hop", k) for k in depths]
    items += [("back", j) for j in range(1, BACKS + 1)]
    at = _ITEMS_START
    for index in _ints(torch.randperm(len(items), generator=g)):
        kind, level = items[index]
        if kind == "hop":
            i = _below(g, CYCLE)
            tokens[at], tokens[at + 1] = hop_token(level), cycle[i]
            role[at], role[at + 1] = _ROLE["hop_marker"], _ROLE["hop"]
            targets[at + 1] = cycle[(i + level) % CYCLE]
            task[at + 1] = _CLASS[f"hop{level}"]
            at += 2
        else:
            content = [pool[_below(g, len(pool))] for _ in range(3)]
            tokens[at : at + 3] = content
            tokens[at + 3] = back_token(level)
            role[at : at + 3] = [_ROLE["copy_content"]] * 3
            role[at + 3] = _ROLE["back"]
            targets[at + 3] = content[3 - level]
            task[at + 3] = _CLASS[f"back{level}"]
            at += 4
    role[at] = _ROLE["fill"]  # tokens[at] is already FILL
    tokens[READ_POSITION] = READ
    role[READ_POSITION] = _ROLE["read"]
    targets[READ_POSITION] = cycle[(i_reg + k_reg) % CYCLE]
    task[READ_POSITION] = _CLASS[f"read{k_reg}"]
    return tokens, targets, role, task


def _ints(values: Tensor) -> list[int]:
    """A 1-D integer tensor as Python ints, in one call (``tolist`` is untyped, hence the cast)."""
    if values.dim() != 1 or values.dtype.is_floating_point:
        msg = f"_ints needs a 1-D integer tensor, got {values.dtype} {tuple(values.shape)}"
        raise ValueError(msg)
    return cast("list[int]", values.tolist())  # pyright: ignore[reportUnknownMemberType]


def _below(g: torch.Generator, bound: int) -> int:
    """One integer in ``[0, bound)`` from ``g``."""
    return int(torch.randint(bound, (1,), generator=g).item())
