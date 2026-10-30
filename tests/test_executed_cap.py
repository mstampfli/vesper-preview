"""The executed-pass cap (``Loop.forward(executed=...)``) against the ordinary eight-pass forward.

The frozen Bounded23 policy exits only at pass 2 or 3 (maintainer B 7f10c8b): with ``executed=3``
the loop stops after pass 3, the rule keeps its eight-pass hazard horizon, and the states hold only
the executed passes, so an ordinary forward refuses them. Every comparison here is bitwise:
the capped path runs the same operations as the first three passes of the ordinary one.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor

from vesper.column import (
    Column,
    ColumnShape,
    ColumnState,
    first_passes,
    init_column_state,
    reset_recurrent,
)
from vesper.contracts import ContractError
from vesper.exits import cdf_advance, cdf_begin, cdf_reaches, hazard_baseline
from vesper.loop import CdfRule, ExitStep, HaltingHead, Loop, LoopResult, TraceRule

CPU = torch.device("cpu")
F64 = torch.float64
T_MAX = 8
CAP = 3
SHAPE = ColumnShape(
    d=8,
    heads=2,
    kv_groups=2,
    head_dim=4,
    window=4,
    rotary_base=10000.0,
    ffn_mult=2,
    block=2,
    key_conv=4,
)


@dataclass(frozen=True)
class Bounded23:
    """The frozen rule's decisions (phase2-final/live_bounded.py, sha256 95fd1d19), copied.

    Pass 1 accumulates hazard mass and never exits, pass 2 exits where the CDF reaches ``q``,
    pass 3 exits every remaining token; the baseline is ``hazard_baseline(runs)``, so the
    horizon the rule is given decides its exits (maintainer B's rule-horizon.json counterexample).
    """

    head: HaltingHead
    q: float

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        baseline = hazard_baseline(runs, active.device)
        survival, cdf = cdf_begin(active)

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            nonlocal survival, cdf
            if t <= 1:
                z = self.head.logit(h, t).to(torch.float64) + baseline[t]
                survival, cdf = cdf_advance(z, survival, cdf)
            if t == 0:
                halts = active & last
            elif t == 1:
                halts = active & (cdf_reaches(cdf, self.q) | last)
            else:
                halts = active.clone()
            return halts, halts.to(dtype)

        return step


def _loop(seed: int = 0) -> tuple[Loop, HaltingHead]:
    g = torch.Generator().manual_seed(seed)
    columns = [Column(SHAPE, g).to(F64) for _ in range(2)]
    head = HaltingHead(SHAPE.d, g).to(F64)
    loop = Loop(columns, head, d=SHAPE.d, t_max=T_MAX, halt_epsilon=0.01)
    with torch.no_grad():
        _ = loop.u.normal_(generator=g)
        _ = head.w_h.mul_(4.0)  # spread the logits: some tokens reach q at pass 2, some do not
        _ = head.b_h.fill_(0.8)
    return loop, head


def _cold(b: int) -> list[ColumnState]:
    return [init_column_state(SHAPE, b, T_MAX, F64, CPU) for _ in range(2)]


def _chunks(b: int, n: int, count: int) -> list[tuple[Tensor, Tensor]]:
    g = torch.Generator().manual_seed(11)
    return [
        (
            torch.randn(b, n, SHAPE.d, generator=g, dtype=F64),
            torch.arange(i * n, (i + 1) * n).expand(b, n).contiguous(),
        )
        for i in range(count)
    ]


def _same_passes(full: ColumnState, capped: ColumnState, passes: int) -> None:
    """``capped`` holds exactly ``passes`` passes, each equal to ``full``'s, field by field."""
    assert {len(capped.h), len(capped.kv), len(capped.key_history)} == {passes}
    for t in range(passes):
        assert torch.equal(full.h[t], capped.h[t]), t
        assert torch.equal(full.key_history[t], capped.key_history[t]), t
        a, b = full.kv[t], capped.kv[t]
        assert torch.equal(a.k, b.k), t
        assert torch.equal(a.v, b.v), t
        assert torch.equal(a.pos, b.pos), t


def _same_outputs(full: LoopResult, capped: LoopResult) -> None:
    assert torch.equal(full.y, capped.y)
    assert torch.equal(full.n_iters, capped.n_iters)
    assert torch.equal(full.ponder, capped.ponder)
    assert capped.per_pass.shape[0] == CAP
    assert torch.equal(full.per_pass[:CAP], capped.per_pass)
    assert not full.per_pass[CAP:].any()  # every token had left: the ordinary later passes are 0


def _stream(
    loop: Loop,
    rule: Callable[[], Bounded23],
    resets: dict[int, Tensor],
    *,
    executed: int | None,
) -> list[tuple[LoopResult, list[ColumnState]]]:
    """Three carried chunks of two rows, ``resets[i]`` restarting rows before chunk ``i``."""
    states = _cold(2)
    out: list[tuple[LoopResult, list[ColumnState]]] = []
    with torch.no_grad():
        for i, (x0, pos) in enumerate(_chunks(2, 6, 3)):
            if i in resets:
                states = [reset_recurrent(s, resets[i]) for s in states]
            result = loop.forward(x0, pos, states, exit_rule=rule(), executed=executed)
            states = list(result.states)
            out.append((result, states))
    return out


def test_the_capped_forward_equals_the_ordinary_one_over_carried_chunks_and_resets() -> None:
    loop, head = _loop()
    resets = {2: torch.eye(2, dtype=F64)[0]}  # keep [1, 0]: row 1 restarts before chunk three
    full = _stream(loop, lambda: Bounded23(head, 0.5), resets, executed=None)
    capped = _stream(loop, lambda: Bounded23(head, 0.5), resets, executed=CAP)
    exits: set[int] = set()
    for (f, f_states), (c, c_states) in zip(full, capped, strict=True):
        _same_outputs(f, c)
        for fs, cs in zip(f_states, c_states, strict=True):
            _same_passes(fs, cs, CAP)
        exits |= {d for d in range(1, T_MAX + 1) if bool((c.n_iters == d).any())}
    assert exits == {2, 3}  # the fixture exercises both exits, so the comparison discriminates
    # the ordinary forward wrote the later passes by copy-forward: the capped state drops real
    # state, which is why it must never be continued deeper
    later = full[-1][1][0].h[CAP:]
    assert all(h.abs().max().item() > 0.0 for h in later)


def test_an_ordinary_state_continues_under_the_cap_through_its_first_passes() -> None:
    # the ordinary first chunk leaves all eight passes written (copy-forward fills 4..8), so a
    # capped second chunk reading any other passes than the first three would differ
    loop, head = _loop()
    (x0, pos), (x1, pos1) = _chunks(2, 6, 2)
    with torch.no_grad():
        warm = list(loop.forward(x0, pos, _cold(2), exit_rule=Bounded23(head, 0.5)).states)
        assert all(not torch.equal(warm[0].h[0], h) for h in warm[0].h[CAP:])
        full = loop.forward(x1, pos1, warm, exit_rule=Bounded23(head, 0.5))
        capped = loop.forward(x1, pos1, warm, exit_rule=Bounded23(head, 0.5), executed=CAP)
    _same_outputs(full, capped)
    for fs, cs in zip(full.states, capped.states, strict=True):
        _same_passes(fs, cs, CAP)


def _capped_states(loop: Loop, head: HaltingHead) -> list[ColumnState]:
    x0, pos = _chunks(2, 6, 1)[0]
    with torch.no_grad():
        first = loop.forward(x0, pos, _cold(2), exit_rule=Bounded23(head, 0.5), executed=CAP)
    return list(first.states)


# each refusal is its own test, checked before any column runs: a later failure inside the
# column (pass_index out of range) would also stop the forward, but only after it has run
_REFUSED = "state of column 0 has .* passes, the loop runs 8"


def test_a_capped_state_is_refused_by_an_ordinary_forward() -> None:
    loop, head = _loop()
    states = _capped_states(loop, head)
    x1, pos1 = _chunks(2, 6, 2)[1]
    with torch.no_grad(), pytest.raises(ValueError, match=_REFUSED):
        _ = loop.forward(x1, pos1, states, exit_rule=Bounded23(head, 0.5))


def test_a_capped_state_is_refused_by_a_deeper_cap() -> None:
    loop, head = _loop()
    states = _capped_states(loop, head)
    x1, pos1 = _chunks(2, 6, 2)[1]
    with torch.no_grad(), pytest.raises(ValueError, match=_REFUSED + " and executes 4"):
        _ = loop.forward(x1, pos1, states, exit_rule=Bounded23(head, 0.5), executed=CAP + 1)


def test_a_state_of_another_pass_count_is_refused_by_the_cap() -> None:
    # four passes are neither the loop's eight nor the cap's three; cutting them to three would
    # run silently on a state that is neither ordinary nor this cap's
    loop, head = _loop()
    four = [first_passes(s, 4) for s in _cold(2)]
    x1, pos1 = _chunks(2, 6, 2)[1]
    with torch.no_grad(), pytest.raises(ValueError, match=_REFUSED + " and executes 3"):
        _ = loop.forward(x1, pos1, four, exit_rule=Bounded23(head, 0.5), executed=CAP)


def test_the_same_cap_continues_a_capped_state() -> None:
    loop, head = _loop()
    states = _capped_states(loop, head)
    x1, pos1 = _chunks(2, 6, 2)[1]
    with torch.no_grad():
        result = loop.forward(x1, pos1, states, exit_rule=Bounded23(head, 0.5), executed=CAP)
    assert all(len(s.h) == CAP for s in result.states)


def test_the_rule_keeps_the_loops_horizon_under_the_cap() -> None:
    # zero logits: with the eight-pass baseline the CDF after two passes is 2/8 < .5, so every
    # token exits at 3; given a three-pass horizon it would be 2/3 and every token would exit
    # at 2 (maintainer B's rule-horizon.json); the cap must not change the rule's horizon
    loop, head = _loop()
    with torch.no_grad():
        _ = head.w_h.zero_()
        _ = head.b_h.zero_()
    x0, pos = _chunks(2, 6, 1)[0]
    with torch.no_grad():
        capped = loop.forward(x0, pos, _cold(2), exit_rule=Bounded23(head, 0.5), executed=CAP)
        shorter = loop.forward(x0, pos, _cold(2), exit_rule=Bounded23(head, 0.5), passes=CAP)
    assert torch.equal(capped.n_iters, torch.full((2, 6), 3, dtype=torch.int32))
    assert torch.equal(shorter.n_iters, torch.full((2, 6), 2, dtype=torch.int32))


def test_a_token_still_active_at_the_cap_is_refused_not_forced_out() -> None:
    loop, head = _loop()
    x0, pos = _chunks(2, 6, 1)[0]
    depths = torch.full((2, 6), 2, dtype=torch.int64)
    depths[1, 4] = CAP + 1  # one token needs a fourth pass
    with torch.no_grad():
        ok = torch.full((2, 6), CAP, dtype=torch.int64)
        _ = loop.forward(x0, pos, _cold(2), exit_rule=TraceRule(ok), executed=CAP)
        with pytest.raises(ContractError, match="executed"):
            _ = loop.forward(x0, pos, _cold(2), exit_rule=TraceRule(depths), executed=CAP)
        with pytest.raises(ContractError, match="executed"):
            _ = loop.forward(x0, pos, _cold(2), exit_rule=CdfRule(head, 0.99), executed=CAP)


def test_the_cap_must_lie_inside_the_forward() -> None:
    loop, _ = _loop()
    x0, pos = _chunks(2, 6, 1)[0]
    for executed, passes in ((0, None), (T_MAX + 1, None), (4, 3)):
        with pytest.raises(ValueError, match="executed"):
            _ = loop.forward(x0, pos, _cold(2), None, passes, executed=executed)
