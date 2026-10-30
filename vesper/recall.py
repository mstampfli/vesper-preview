"""Fact-sensitive recall at a stated token gap: a behaviour probe of a fixed model.

An episode is one token stream from a cold start: a FACT sentence that plants a value for a
synthetic key ("The access code for the Varden vault is 4817."), ``gap`` tokens of held-out
FILLER, then a QUERY that asks for the key's value ("... the Varden vault is"). Every candidate
value is scored as a complete suffix with teacher forcing, each from the SAME state saved before
the query (``evals.run_carried``; the query's tokens are replayed with every candidate), so no
candidate inherits another's state. Episodes come
in pairs that differ only in the planted value (v or w, equal token lengths): the paired contrast
is the change in the log-odds of v against w when the fact switches from v to w, so a prior
preference for one value cancels.

What a result can say (maintainer B 85543e6): fact-sensitive recall at the stated gap for this model
and this cue/value format. A gap beyond one attention window does NOT mean the recurrent state
carried the fact: stacked columns and passes relay through attention further than one window.
Attributing recall to a component needs its own ablation or a receptive-field bound. A failure
does not show that the model's memory does nothing on text either.

Design and review: maintainer B 85543e6 (channel), maintainer A 2bb7b3a.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from vesper.column import ColumnState, state_tensors
from vesper.data import Chunk
from vesper.evals import run_carried
from vesper.graphs import SignatureGraph
from vesper.loop import DenseRule
from vesper.model import Vesper

MIN_CANDIDATES = 2  # a rank among fewer than two values says nothing
KEY_OCCURRENCES = 2  # once in the fact, once in the query


@dataclass(frozen=True)
class Episode:
    """One planted fact, the filler and the query, with the candidate values to score.

    ``tokens`` (int32 ``[P]``) is the whole prefix: fact, filler, query. ``fact`` and ``value``
    are ``[start, end)`` spans in it (the value inside the fact), ``query`` the index of the
    query's first token, ``candidates`` (int32 ``[K, c]``) every candidate value's ids, distinct
    rows of one length, and ``planted`` the row of the value the fact carries.
    """

    tokens: Tensor
    fact: tuple[int, int]
    value: tuple[int, int]
    query: int
    candidates: Tensor
    planted: int

    @property
    def gap(self) -> int:
        """Tokens strictly between the fact's end and the query's start."""
        return self.query - self.fact[1]


def _distinct_rows(rows: Tensor) -> bool:
    """True when no two rows of ``rows [K, c]`` are equal."""
    same = (rows[:, None, :] == rows[None, :, :]).all(dim=2)
    return int(same.sum()) == rows.shape[0]


def build_episode(
    fact_head: Tensor,
    fact_tail: Tensor,
    filler: Tensor,
    query: Tensor,
    candidates: Tensor,
    planted: int,
) -> Episode:
    """The episode ``fact_head + candidates[planted] + fact_tail + filler + query``.

    Raises:
        ValueError: on a piece that is not int32 ``[L]`` (empty allowed only for ``fact_tail``
            and ``filler``), candidates that are not int32 ``[K, c]`` with ``K >= 2``, ``c >= 1``
            and distinct rows, or ``planted`` outside ``[0, K)``.
    """
    for name, piece, empty_ok in (
        ("fact_head", fact_head, False),
        ("fact_tail", fact_tail, True),
        ("filler", filler, True),
        ("query", query, False),
    ):
        if piece.dtype != torch.int32 or piece.dim() != 1 or (piece.numel() == 0 and not empty_ok):
            msg = f"{name} must be int32 [L] and non-empty, got {piece.dtype} {tuple(piece.shape)}"
            raise ValueError(msg)
    if candidates.dtype != torch.int32 or candidates.dim() != len(("K", "c")):
        msg = f"candidates must be int32 [K, c], got {candidates.dtype} {tuple(candidates.shape)}"
        raise ValueError(msg)
    k, c = candidates.shape
    if k < MIN_CANDIDATES or c < 1:
        msg = f"candidates must be at least 2 values of at least 1 token, got [{k}, {c}]"
        raise ValueError(msg)
    if not _distinct_rows(candidates):
        msg = "candidates must be distinct"
        raise ValueError(msg)
    if not 0 <= planted < k:
        msg = f"planted must lie in [0, {k}), got {planted}"
        raise ValueError(msg)
    head, mid = fact_head.numel(), fact_head.numel() + c
    fact_end = mid + fact_tail.numel()
    tokens = torch.cat([fact_head, candidates[planted], fact_tail, filler, query])
    return Episode(
        tokens=tokens,
        fact=(0, fact_end),
        value=(head, mid),
        query=fact_end + filler.numel(),
        candidates=candidates,
        planted=planted,
    )


def occurrences(tokens: Tensor, sub: Tensor) -> list[int]:
    """Every start index where ``sub`` occurs in ``tokens`` (both 1-d), overlaps included."""
    n, m = tokens.numel(), sub.numel()
    if m == 0 or m > n:
        return []
    hits = (tokens.unfold(0, m, 1) == sub).all(dim=1).nonzero().flatten()
    return [int(hits[i]) for i in range(hits.shape[0])]


def check_isolation(episode: Episode, key: Tensor) -> None:
    """Refuse an episode in which the answer could come from anywhere but the planted fact.

    The planted value occurs exactly once, at its span; no other candidate occurs anywhere; the
    key occurs exactly twice, once in the fact before the value and once in the query.

    Raises:
        ValueError: naming the first isolation failure.
    """
    tokens = episode.tokens
    for row in range(episode.candidates.shape[0]):
        hits = occurrences(tokens, episode.candidates[row])
        expected = [episode.value[0]] if row == episode.planted else []
        if hits != expected:
            msg = f"isolation: candidate {row} occurs at {hits}, expected {expected}"
            raise ValueError(msg)
    hits = occurrences(tokens, key)
    in_fact = [h for h in hits if h + key.numel() <= episode.value[0]]
    in_query = [h for h in hits if h >= episode.query]
    if len(hits) != KEY_OCCURRENCES or len(in_fact) != 1 or len(in_query) != 1:
        msg = f"isolation: the key occurs at {hits}, expected once in the fact, once in the query"
        raise ValueError(msg)


def layout(length: int, chunk: int) -> list[int]:
    """Segment lengths of a ``length``-token prefix: an odd first segment, then whole chunks.

    The first segment is cold and carries the remainder (``1..chunk`` tokens), so every later
    segment has exactly ``chunk`` tokens: one captured signature for all of them.

    Raises:
        ValueError: on ``length`` or ``chunk`` below 1.
    """
    if length < 1 or chunk < 1:
        msg = f"length and chunk must be at least 1, got {length} and {chunk}"
        raise ValueError(msg)
    first = (length - 1) % chunk + 1
    return [first] + [chunk] * ((length - first) // chunk)


def _chunk(tokens: Tensor, pos: Tensor, targets: Tensor, valid: Tensor, loss: Tensor) -> Chunk:
    """A ``Chunk`` of ``[b, n]`` tensors for a probe stream (no documents, no spans)."""
    b = tokens.shape[0]
    return Chunk(
        tokens=tokens,
        targets=targets,
        valid=valid,
        doc_start=torch.zeros_like(valid),
        segment=torch.zeros(tokens.shape, dtype=torch.int64),
        loss_mask=loss,
        pos=pos,
        doc_ids=((),) * b,
        spans=((),) * b,
    )


@dataclass
class RecallGraphs:
    """The probe's captured passes on CUDA: cold first segment, carried segments, candidates.

    A ``SignatureGraph`` keys only tensor shapes, dtypes and devices, so a graph captured for one
    model or depth would replay that work for another of the same shapes (maintainer B's review of
    f4f15b2: a depth-1 request returned the captured depth-2 scores). ``bind`` ties the set to the
    first model object and depth it serves and refuses any other.
    """

    first: SignatureGraph = field(default_factory=SignatureGraph)
    carried: SignatureGraph = field(default_factory=SignatureGraph)
    final: SignatureGraph = field(default_factory=SignatureGraph)
    bound: tuple[Vesper, int] | None = None

    def bind(self, model: Vesper, passes: int) -> None:
        """Tie the graphs to ``model`` and ``passes`` on first use; refuse any other later.

        Raises:
            ValueError: when the graphs already serve another model object or depth.
        """
        if self.bound is None:
            self.bound = (model, passes)
        elif self.bound[0] is not model or self.bound[1] != passes:
            msg = (
                f"these RecallGraphs serve another model or depth (bound to passes"
                f" {self.bound[1]}); use a fresh RecallGraphs"
            )
            raise ValueError(msg)

    def counts(self) -> dict[str, list[int]]:
        """Each graph's ``[eager, captures, replays]``, measured, never assumed."""
        graphs = {"first": self.first, "carried": self.carried, "final": self.final}
        return {name: [g.eager, g.captures, g.replays] for name, g in graphs.items()}


def score_batch(
    model: Vesper,
    episodes: Sequence[Episode],
    *,
    chunk: int,
    final: int,
    passes: int,
    graphs: RecallGraphs | None,
) -> Tensor:
    """``float64 [b, K]``: every candidate's summed log-probability, one row per episode.

    The episodes are the batch's rows and must share their layout (prefix length, query index,
    candidate shape). The prefix up to the query runs once, carried (``layout``), at a fixed
    depth (``DenseRule``, ``passes``); then each candidate k runs one final segment of ``final``
    positions, the query plus candidate k (padded), from the SAME saved states, and its score is
    minus the summed loss of the positions that predict its tokens (the query's last position
    predicts the first). Positions are the stream's own, consecutive from 0.

    Raises:
        ValueError: on no episodes, episodes of different layouts, a query plus candidate
            longer than ``final``, or ``graphs`` bound to another model or depth.
    """
    if not episodes:
        msg = "score_batch needs at least one episode"
        raise ValueError(msg)
    if graphs is not None:
        graphs.bind(model, passes)
    head = episodes[0]
    shape = (head.tokens.numel(), head.query, tuple(head.candidates.shape))
    if any((e.tokens.numel(), e.query, tuple(e.candidates.shape)) != shape for e in episodes):
        msg = "every episode of a batch must share prefix length, query index and candidate shape"
        raise ValueError(msg)
    length, query, (k, c) = shape
    q = length - query
    if q + c > final:
        msg = f"query ({q}) plus candidate ({c}) exceed the final segment ({final})"
        raise ValueError(msg)
    b = len(episodes)
    rows = torch.stack([e.tokens for e in episodes])  # [b, P]
    zeros = torch.zeros(b, 1, dtype=torch.int32)

    def advance(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        result = model.run(inputs[0], inputs[1], states, passes=passes, exit_rule=DenseRule())
        return state_tensors(result.states)

    prefix: list[Chunk] = []
    start = 0
    for n in layout(query, chunk):
        tokens = rows[:, start : start + n].contiguous()
        pos = torch.arange(start, start + n, dtype=torch.int64).expand(b, n).contiguous()
        none = torch.zeros(b, n, dtype=torch.bool)
        prefix.append(_chunk(tokens, pos, torch.zeros_like(tokens), ~none, none))
        start += n
    _, saved = run_carried(
        model,
        prefix,
        None if graphs is None else graphs.carried,
        0,
        advance,
        first=None if graphs is None else graphs.first,
    )

    def score(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        result = model.run(inputs[0], inputs[1], states, passes=passes, exit_rule=DenseRule())
        return [model.position_losses(result.y, inputs[2]), *state_tensors(result.states)]

    out = torch.zeros(b, k, dtype=torch.float64)
    for j in range(k):
        cand = torch.stack([e.candidates[j] for e in episodes])  # [b, c]
        body = torch.cat([rows[:, query:], cand], dim=1)  # [b, q + c]
        pad = final - body.shape[1]
        tokens = torch.cat([body, zeros.expand(b, pad)], dim=1).contiguous()
        targets = torch.cat([tokens[:, 1:], zeros], dim=1).contiguous()
        valid = torch.zeros(b, final, dtype=torch.bool)
        valid[:, : q + c] = True
        loss = torch.zeros(b, final, dtype=torch.bool)
        loss[:, q - 1 : q + c - 1] = True  # the positions that predict candidate j's tokens
        pos = torch.arange(query, query + final, dtype=torch.int64).expand(b, final).contiguous()
        results, _ = run_carried(
            model,
            [_chunk(tokens, pos, targets, valid, loss)],
            None if graphs is None else graphs.final,
            1,
            score,
            first=None,
            initial=saved,
        )
        nll = results[0][0].double().cpu()
        out[:, j] = -(nll * loss).sum(dim=1)
    return out


def rank(scores: Tensor, planted: int) -> tuple[int, int]:
    """``(rank, ties)`` of ``scores[planted]``, ties reported and never broken.

    The rank is 1 plus how many candidates score strictly higher; ``ties`` is how many OTHER
    candidates score exactly the same.
    """
    s = scores[planted]
    higher = int((scores > s).sum())
    ties = int((scores == s).sum()) - 1
    return 1 + higher, ties


def paired_contrast(planted_v: Tensor, planted_w: Tensor, v: int, w: int) -> float:
    """The change in the log-odds of v against w when the fact switches from v to w.

    ``planted_v`` and ``planted_w`` are the candidate scores of the episode pair (same filler and
    query, the fact carrying v resp. w). Positive: the model follows the planted value; zero: the
    fact makes no difference to v against w; a prior preference for v cancels.
    """
    return float((planted_v[v] - planted_v[w]) - (planted_w[v] - planted_w[w]))


def plan_pairs(
    count: int, values: int, candidates: int, seed: int
) -> list[tuple[list[int], int, int]]:
    """``count`` pairs over a pool of ``values``: each ``(candidate pool indices, v, w)``.

    ``v`` and ``w`` index into the candidate list (both planted, one per episode of the pair);
    the pool's values are planted (as v or w) in shuffled whole cycles, so equally often up to
    one, and each candidate list holds ``candidates`` distinct pool values in a random order.
    Deterministic in ``seed`` (a ``torch.Generator``).

    Raises:
        ValueError: when ``candidates`` is below 2 or above ``values``, or ``count`` below 1.
    """
    if not MIN_CANDIDATES <= candidates <= values or count < 1:
        msg = f"need 2 <= candidates <= values and count >= 1, got {candidates}, {values}, {count}"
        raise ValueError(msg)
    g = torch.Generator().manual_seed(seed)

    def perm(n: int) -> list[int]:
        order = torch.randperm(n, generator=g)
        return [int(order[i]) for i in range(n)]

    planted: list[int] = []
    while len(planted) < 2 * count:
        planted += perm(values)
    pairs: list[tuple[list[int], int, int]] = []
    for i in range(count):
        a, b = planted[2 * i], planted[2 * i + 1]
        if a == b:
            # a cycle boundary can repeat a value: swap in the next different one from later in
            # the order, so every value keeps its count (maintainer B's review of f4f15b2: replacing
            # it broke the balance for odd pools)
            j = next((k for k in range(2 * i + 2, len(planted)) if planted[k] != a), None)
            if j is None:  # unreachable: whole cycles of >= 2 values always hold another later
                msg = "plan_pairs: no later value to swap in"
                raise RuntimeError(msg)
            planted[2 * i + 1], planted[j] = planted[j], planted[2 * i + 1]
            b = planted[2 * i + 1]
        others = [x for x in range(values) if x not in (a, b)]
        pick = perm(len(others))[: candidates - 2]
        chosen = [a, b, *[others[p] for p in pick]]
        chosen = [chosen[p] for p in perm(len(chosen))]
        pairs.append((chosen, chosen.index(a), chosen.index(b)))
    return pairs
