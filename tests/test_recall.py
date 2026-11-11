"""The recall probe (``vesper.recall``) and ``evals.run_carried``.

The scorer is checked against a literal oracle: one forward of the whole stream (prefix plus
candidate) from cold states, no saved state, no chunking. State isolation: every candidate's
score is the same whether it is scored alone, with the others, or in another order.
"""

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from conftest import small_config
from vesper.column import ColumnState, set_scan, state_tensors
from vesper.data import Chunk, Document, Stream
from vesper.evals import PassFn, run_carried, run_chunks
from vesper.graphs import SignatureGraph
from vesper.loop import DenseRule
from vesper.model import Vesper
from vesper.recall import (
    Episode,
    RecallGraphs,
    build_episode,
    check_isolation,
    layout,
    occurrences,
    paired_contrast,
    plan_pairs,
    rank,
    score_batch,
)

CPU = torch.device("cpu")
NEVER = 10**9  # a warm count no test reaches: that graph never captures (the eager oracle)


def _model() -> Vesper:
    return Vesper(small_config(), torch.Generator().manual_seed(0)).eval()


def ids(*values: int) -> Tensor:
    """An int32 token tensor."""
    out = torch.zeros(len(values), dtype=torch.int32)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _stream_chunks(count: int) -> list[Chunk]:
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 60, (70,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(4)
    ]
    stream = Stream(iter(docs), slots=2, chunk=16)
    chunks: list[Chunk] = []
    while len(chunks) < count and (chunk := stream.next()) is not None:
        chunks.append(chunk)
    return chunks


def _nll_pass(model: Vesper) -> PassFn:
    def step(inputs: Sequence[Tensor], states: list[ColumnState]) -> list[Tensor]:
        result = model.run(inputs[0], inputs[1], states, passes=2, exit_rule=DenseRule())
        return [model.position_losses(result.y, inputs[2]), *state_tensors(result.states)]

    return step


# --- evals.run_carried ----------------------------------------------------------------------


def test_a_carried_pass_resumed_from_saved_states_equals_one_pass() -> None:
    model = _model()
    chunks = _stream_chunks(5)
    step = _nll_pass(model)
    with torch.no_grad():
        whole = run_chunks(model, chunks, None, 1, step, carried=True)
        _, saved = run_carried(model, chunks[:2], None, 1, step, first=None)
        rest, final = run_carried(model, chunks[2:], None, 1, step, first=None, initial=saved)
        _, final_whole = run_carried(model, chunks, None, 1, step, first=None)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(whole[2:], rest, strict=True))
    pairs = zip(state_tensors(final), state_tensors(final_whole), strict=True)
    assert all(torch.equal(a, b) for a, b in pairs)


def test_continuations_from_one_saved_state_do_not_inherit_each_other() -> None:
    model = _model()
    chunks = _stream_chunks(4)
    step = _nll_pass(model)
    with torch.no_grad():
        _, saved = run_carried(model, chunks[:2], None, 1, step, first=None)
        before = [t.clone() for t in state_tensors(saved)]
        y_first, _ = run_carried(model, chunks[3:], None, 1, step, first=None, initial=saved)
        _ = run_carried(model, chunks[2:3], None, 1, step, first=None, initial=saved)
        y_again, _ = run_carried(model, chunks[3:], None, 1, step, first=None, initial=saved)
    assert torch.equal(y_first[0][0], y_again[0][0])
    assert all(torch.equal(a, b) for a, b in zip(before, state_tensors(saved), strict=True))


def test_run_carried_needs_a_chunk() -> None:
    model = _model()
    with pytest.raises(ValueError, match="at least one chunk"):
        _ = run_carried(model, [], None, 1, _nll_pass(model), first=None)


# --- the episode ----------------------------------------------------------------------------

CANDIDATES = torch.stack([ids(40, 41), ids(42, 43), ids(44, 45), ids(46, 47)])
KEY = ids(30, 31)


def _episode(filler: Tensor, planted: int = 1, query: Tensor | None = None) -> Episode:
    return build_episode(
        fact_head=ids(5, 30, 31, 6),
        fact_tail=ids(7),
        filler=filler,
        query=ids(8, 30, 31, 6) if query is None else query,
        candidates=CANDIDATES,
        planted=planted,
    )


def test_an_episode_lays_out_fact_value_filler_and_query() -> None:
    ep = _episode(ids(9, 10, 11), planted=2)
    assert torch.equal(ep.tokens, ids(5, 30, 31, 6, 44, 45, 7, 9, 10, 11, 8, 30, 31, 6))
    assert ep.value == (4, 6)
    assert ep.fact == (0, 7)
    assert ep.query == 10
    assert ep.gap == 3
    check_isolation(ep, KEY)


def _build(**changes: object) -> Episode:
    args: dict[str, object] = {
        "fact_head": ids(5, 30, 31, 6),
        "fact_tail": ids(7),
        "filler": ids(9),
        "query": ids(8, 30, 31, 6),
        "candidates": CANDIDATES,
        "planted": 1,
    }
    args.update(changes)
    return build_episode(**args)  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]


def test_a_malformed_episode_is_refused() -> None:
    cases: list[tuple[dict[str, object], str]] = [
        ({"candidates": torch.stack([ids(40, 41), ids(40, 41)])}, "distinct"),
        ({"candidates": ids(40, 41)[None]}, "at least 2"),
        ({"planted": 4}, "planted"),
        ({"query": ids()}, "query"),
        ({"filler": ids(9).to(torch.int64)}, "filler"),
    ]
    for changes, match in cases:
        with pytest.raises(ValueError, match=match):
            _ = _build(**changes)


@pytest.mark.parametrize(
    "case",
    [
        "value-in-query",
        "value-in-filler",
        "other-candidate-in-filler",
        "key-in-filler",
        "no-key-in-query",
    ],
)
def test_isolation_refuses_any_other_source_of_the_answer(case: str) -> None:
    filler, query = ids(9, 10, 11), None
    if case == "value-in-query":  # the leaked-query mutant: the answer handed over in the query
        query = ids(8, 30, 31, 6, 42, 43)
    elif case == "value-in-filler":
        filler = ids(9, 42, 43)
    elif case == "other-candidate-in-filler":
        filler = ids(9, 46, 47)
    elif case == "key-in-filler":
        filler = ids(9, 30, 31)
    else:
        query = ids(8, 6)
    with pytest.raises(ValueError, match="isolation"):
        check_isolation(_episode(filler, planted=1, query=query), KEY)


def test_occurrences_finds_overlapping_hits() -> None:
    assert occurrences(ids(1, 1, 1, 2), ids(1, 1)) == [0, 1]
    assert occurrences(ids(1, 2), ids(3)) == []
    assert occurrences(ids(1), ids(1, 1)) == []


@pytest.mark.parametrize(
    ("length", "chunk", "expected"),
    [
        (1, 16, [1]),
        (16, 16, [16]),
        (17, 16, [1, 16]),
        (40, 16, [8, 16, 16]),
        (48, 16, [16, 16, 16]),
    ],
)
def test_layout_puts_the_remainder_first(length: int, chunk: int, expected: list[int]) -> None:
    assert layout(length, chunk) == expected
    assert sum(expected) == length


# --- ranks, contrast and the plan -----------------------------------------------------------


def _f64(*values: float) -> Tensor:
    out = torch.zeros(len(values), dtype=torch.float64)
    for i, value in enumerate(values):
        out[i] = value
    return out


def test_rank_counts_strictly_higher_and_reports_ties() -> None:
    scores = _f64(1.0, 3.0, 3.0, 2.0)
    assert rank(scores, 1) == (1, 1)
    assert rank(scores, 2) == (1, 1)
    assert rank(scores, 3) == (3, 0)
    assert rank(scores, 0) == (4, 0)


def test_the_paired_contrast_cancels_a_prior_and_signs_recall() -> None:
    prior = _f64(-1.0, -3.0, -5.0)  # the model prefers value 0
    follows_v = prior + _f64(0.0, 2.0, 0.0)  # the fact carries value 1
    follows_w = prior + _f64(0.0, 0.0, 2.0)  # the fact carries value 2
    assert paired_contrast(follows_v, follows_w, 1, 2) == pytest.approx(4.0)
    assert paired_contrast(prior, prior, 1, 2) == 0.0
    assert paired_contrast(follows_w, follows_v, 1, 2) == pytest.approx(-4.0)


def test_the_plan_balances_planted_values_and_keeps_candidates_distinct() -> None:
    pairs = plan_pairs(count=50, values=10, candidates=10, seed=3)
    planted = [chosen[v] for chosen, v, _ in pairs] + [chosen[w] for chosen, _, w in pairs]
    counts = [planted.count(x) for x in range(10)]
    assert max(counts) - min(counts) <= 1
    for chosen, v, w in pairs:
        assert len(set(chosen)) == len(chosen) == 10
        assert v != w
    assert plan_pairs(count=50, values=10, candidates=10, seed=3) == pairs
    assert plan_pairs(count=50, values=10, candidates=10, seed=4) != pairs


def test_a_collision_at_a_cycle_boundary_keeps_every_count_for_odd_pools() -> None:
    # Maintainer B's review of f4f15b2: count 9 over 3 values gave [5, 7, 6] instead of [6, 6, 6]
    pairs = plan_pairs(count=9, values=3, candidates=3, seed=1)
    planted = [chosen[v] for chosen, v, _ in pairs] + [chosen[w] for chosen, _, w in pairs]
    assert [planted.count(x) for x in range(3)] == [6, 6, 6]
    assert all(v != w for _, v, w in pairs)


def test_graphs_bound_to_one_model_and_depth_refuse_another() -> None:
    # a SignatureGraph keys shapes only: reuse for another depth or model would replay old work
    # (that score_batch calls bind is checked on CUDA, where a captured graph would replay)
    model, other = _model(), _model()
    graphs = RecallGraphs()
    graphs.bind(model, 2)
    graphs.bind(model, 2)  # the same model and depth: accepted
    with pytest.raises(ValueError, match="another model or depth"):
        graphs.bind(model, 1)
    with pytest.raises(ValueError, match="another model or depth"):
        graphs.bind(other, 2)


# --- the scorer against a literal oracle ----------------------------------------------------


def _oracle(model: Vesper, ep: Episode, j: int) -> float:
    """Candidate j's log-probability from ONE forward of the whole stream from cold states."""
    full = torch.cat([ep.tokens, ep.candidates[j]])[None]
    n = full.shape[1]
    states = model.init_states(1, torch.float32, CPU)
    with torch.no_grad():
        result = model.run(full, torch.arange(n)[None], states, passes=2, exit_rule=DenseRule())
        targets = torch.cat([full[:, 1:], torch.zeros(1, 1, dtype=torch.int32)], dim=1)
        nll = model.position_losses(result.y, targets)[0].double()
    c = ep.candidates.shape[1]
    q_end = ep.tokens.numel()
    return float(-nll[q_end - 1 : q_end - 1 + c].sum())


def _filler(n: int, seed: int) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 30, (n,), generator=g).to(torch.int32)


@pytest.mark.parametrize("gap", [3, 25, 40])
def test_the_scorer_equals_a_whole_stream_oracle_for_every_candidate(gap: int) -> None:
    model = _model()
    eps = [_episode(_filler(gap, s), planted=s % 4) for s in range(2)]
    with torch.no_grad():
        got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=None)
    for row, ep in enumerate(eps):
        for j in range(CANDIDATES.shape[0]):
            want = _oracle(model, ep, j)
            assert got[row, j].item() == pytest.approx(want, rel=1e-5, abs=1e-5), (gap, row, j)


def test_every_candidate_scores_the_same_alone_with_others_and_in_another_order() -> None:
    model = _model()
    ep = _episode(_filler(25, 7), planted=1)
    order, pick = ids(3, 1, 0, 2).long(), ids(2, 3).long()
    reordered = Episode(ep.tokens, ep.fact, ep.value, ep.query, ep.candidates[order], 1)
    two = Episode(ep.tokens, ep.fact, ep.value, ep.query, ep.candidates[pick], 0)
    with torch.no_grad():
        together = score_batch(model, [ep], chunk=16, final=8, passes=2, graphs=None)[0]
        permuted = score_batch(model, [reordered], chunk=16, final=8, passes=2, graphs=None)[0]
        alone = score_batch(model, [two], chunk=16, final=8, passes=2, graphs=None)[0]
    assert torch.equal(permuted, together[order])
    assert torch.equal(alone, together[pick])


def test_the_scorer_refuses_mixed_layouts_and_a_short_final_segment() -> None:
    model = _model()
    mixed = [_episode(_filler(3, 0)), _episode(_filler(4, 0))]
    with pytest.raises(ValueError, match="share"):
        _ = score_batch(model, mixed, chunk=16, final=8, passes=2, graphs=None)
    with pytest.raises(ValueError, match="final segment"):
        _ = score_batch(model, mixed[:1], chunk=16, final=5, passes=2, graphs=None)


# --- a state cut at the fact's end (which path carries the fact) ----------------------------


def _cut_by_hand(model: Vesper, states: Sequence[ColumnState], cut: str) -> list[ColumnState]:
    """The cut built from the model's own cold state, not through vesper.recall."""
    cold = model.init_states(1, torch.float32, CPU)
    recurrent, window = cut in {"recurrent", "both"}, cut in {"window", "both"}
    return [
        ColumnState(
            h=c.h if recurrent else s.h,
            kv=c.kv if window else s.kv,
            key_history=c.key_history if recurrent else s.key_history,
        )
        for s, c in zip(states, cold, strict=True)
    ]


def _cut_oracle(model: Vesper, ep: Episode, j: int, cut: str) -> float:
    """Candidate j's log-probability from TWO forwards: the fact, the hand cut, then the rest."""
    f = ep.fact[1]
    head = ep.tokens[:f][None]
    rest = torch.cat([ep.tokens[f:], ep.candidates[j]])[None]
    n = rest.shape[1]
    with torch.no_grad():
        states = model.init_states(1, torch.float32, CPU)
        first = model.run(head, torch.arange(f)[None], states, passes=2, exit_rule=DenseRule())
        states = _cut_by_hand(model, first.states, cut)
        pos = torch.arange(f, f + n)[None]
        result = model.run(rest, pos, states, passes=2, exit_rule=DenseRule())
        targets = torch.cat([rest[:, 1:], torch.zeros(1, 1, dtype=torch.int32)], dim=1)
        nll = model.position_losses(result.y, targets)[0].double()
    c = ep.candidates.shape[1]
    q_end = ep.tokens.numel() - f
    return float(-nll[q_end - 1 : q_end - 1 + c].sum())


@pytest.mark.parametrize("cut", ["split", "recurrent", "window", "both"])
@pytest.mark.parametrize("gap", [3, 25, 40])
def test_a_cut_scorer_equals_a_two_forward_oracle_with_the_state_cut_by_hand(
    cut: str, gap: int
) -> None:
    model = _model()
    eps = [_episode(_filler(gap, s), planted=s % 4) for s in range(2)]
    with torch.no_grad():
        got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=None, cut=cut)
    for row, ep in enumerate(eps):
        for j in range(CANDIDATES.shape[0]):
            want = _cut_oracle(model, ep, j, cut)
            assert got[row, j].item() == pytest.approx(want, rel=1e-5, abs=1e-5), (cut, row, j)


def test_cutting_both_paths_at_the_fact_end_leaves_no_trace_of_the_fact() -> None:
    # the pair differs only in the planted value: with every state cold after the fact, nothing
    # about it can reach the candidates; uncut, it must (the control). Each episode in its own
    # call: two rows of one batch may round differently (measured 4.8e-7), which is no trace
    model = _model()
    filler = _filler(25, 3)
    pair = [_episode(filler, planted=1), _episode(filler, planted=2)]

    def scored(ep: Episode, cut: str) -> Tensor:
        return score_batch(model, [ep], chunk=16, final=8, passes=2, graphs=None, cut=cut)[0]

    with torch.no_grad():
        both = [scored(ep, "both") for ep in pair]
        uncut = [scored(ep, "none") for ep in pair]
    assert torch.equal(both[0], both[1])
    assert not torch.equal(uncut[0], uncut[1])


def test_a_cut_is_refused_for_an_unknown_name_or_differing_fact_spans() -> None:
    model = _model()
    ep = _episode(_filler(25, 0))
    with pytest.raises(ValueError, match="cut"):
        _ = score_batch(model, [ep], chunk=16, final=8, passes=2, graphs=None, cut="attention")
    # same prefix length, query and candidates, but the fact ends one token later
    longer = _build(fact_tail=ids(7, 7), filler=_filler(24, 0))
    assert (longer.tokens.numel(), longer.query) == (ep.tokens.numel(), ep.query)
    with pytest.raises(ValueError, match="fact"):
        _ = score_batch(model, [ep, longer], chunk=16, final=8, passes=2, graphs=None, cut="both")


# --- on CUDA: captured, and the saved states owned by the caller ---------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_saved_states_survive_later_replays_of_the_graph_that_made_them() -> None:
    # on CUDA the states after the last chunk live in the graph's output buffers; run_carried
    # must hand back its own copies, or the next replay of that graph rewrites them
    model = _model().to(torch.device("cuda"))
    _ = set_scan(model, "fused-fp32")
    chunks = _stream_chunks(5)
    other = _stream_chunks(5)[1:]
    step = _nll_pass(model)
    first, carried, cont = SignatureGraph(), SignatureGraph(), SignatureGraph()
    with torch.no_grad():
        _, saved = run_carried(model, chunks[:3], carried, 1, step, first=first)
        for _ in range(3):  # warm, then capture, then replay: the carried graph really replays
            _, saved = run_carried(model, chunks[:3], carried, 1, step, first=first)
        want, _ = run_carried(model, chunks[3:4], cont, 1, step, first=None, initial=saved)
        _ = run_carried(model, other[:3], carried, 1, step, first=first)  # other inputs
        got, _ = run_carried(model, chunks[3:4], cont, 1, step, first=None, initial=saved)
    assert carried.captures >= 1
    assert carried.replays >= 1
    assert torch.equal(want[0][0], got[0][0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_the_captured_scorer_equals_the_eager_one() -> None:
    model = _model().to(torch.device("cuda"))
    _ = set_scan(model, "fused-fp32")
    eps = [_episode(_filler(40, s), planted=s % 4) for s in range(2)]
    captured = RecallGraphs()
    never = RecallGraphs(SignatureGraph(NEVER), SignatureGraph(NEVER), SignatureGraph(NEVER))
    with torch.no_grad():
        got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=captured)
        for _ in range(4):  # the cold first segment runs once per call: 3 warm, capture, replay
            got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=captured)
        want = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=never)
    counts = captured.counts()
    used = ("first", "carried", "final")  # no cut: the resumed graph is never called
    assert all(counts[g][1] >= 1 and counts[g][2] >= 1 for g in used), counts
    assert counts["resumed"] == [0, 0, 0], counts
    assert all(c[1] == 0 for c in never.counts().values())
    assert torch.equal(got, want)
    # the captured graphs are bound: a depth-1 request is refused, not answered with depth-2 work
    with torch.no_grad(), pytest.raises(ValueError, match="another model or depth"):
        _ = score_batch(model, eps, chunk=16, final=8, passes=1, graphs=captured)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("cut", ["recurrent", "window"])
def test_the_captured_cut_scorer_equals_the_eager_one(cut: str) -> None:
    # filler 40 at chunk 16: [8, 16, 16], so the resumed graph and the carried graph both run;
    # the window (small config) fills within the first filler chunk plus what the cut keeps
    model = _model().to(torch.device("cuda"))
    _ = set_scan(model, "fused-fp32")
    eps = [_episode(_filler(40, s), planted=s % 4) for s in range(2)]
    captured = RecallGraphs()
    never = RecallGraphs(
        first=SignatureGraph(NEVER),
        carried=SignatureGraph(NEVER),
        final=SignatureGraph(NEVER),
        resumed=SignatureGraph(NEVER),
    )
    with torch.no_grad():
        got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=captured, cut=cut)
        for _ in range(4):  # 3 warm, capture, replay for the graphs called once per call
            got = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=captured, cut=cut)
        want = score_batch(model, eps, chunk=16, final=8, passes=2, graphs=never, cut=cut)
    counts = captured.counts()
    assert all(c[1] >= 1 and c[2] >= 1 for c in counts.values()), counts
    assert all(c[1] == 0 for c in never.counts().values())
    assert torch.equal(got, want)
