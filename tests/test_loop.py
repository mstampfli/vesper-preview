import math
from dataclasses import replace
from fractions import Fraction
from typing import override

import pytest
import torch
from torch import Tensor

from vesper.attention import KVCache, kv_cache_empty
from vesper.column import (
    Carry,
    Column,
    ColumnShape,
    ColumnState,
    Modulation,
    Projections,
    init_column_state,
)
from vesper.contracts import ContractError
from vesper.exits import cdf_advance, cdf_begin, hazard_cdf_exits, hazard_logits
from vesper.loop import (
    ActRule,
    CdfRule,
    DenseRule,
    HaltingHead,
    Loop,
    LoopResult,
    TraceRule,
)
from vesper.recurrence import delta_scan_chunked

CPU = torch.device("cpu")
F64 = torch.float64


# --- the scalar column of the pass-state review (section B) -----------------------------------


class ScalarColumn:
    """One head of width one: unit key and query, value = input, beta 1/2; output = state read.

    The column of docs/reviews/2026-10-14-pass-state-review.md, section B: it writes its input
    value into the pass's state and returns the state read, replacing its input. No attention
    (its KV window stays empty) and no residual, so the hand-computed fractions apply exactly.
    """

    beta: float = 0.5

    def step(
        self,
        x: Tensor,
        pos: Tensor,
        state: ColumnState,
        pass_index: int,
        *,
        modulation: Modulation | None = None,
        carry: Carry | None = None,
    ) -> tuple[Tensor, ColumnState, Projections]:
        del pos, modulation
        b, n = x.shape[0], x.shape[1]
        ones = torch.ones(b, n, 1, 1, dtype=x.dtype)
        k, v = ones, x.reshape(b, n, 1, 1)
        beta = torch.full((b, n, 1), self.beta, dtype=x.dtype)
        if carry is not None:
            mask = carry.active[..., None, None]
            k = torch.where(mask, k, carry.projections.rec_k)
            v = torch.where(mask, v, carry.projections.rec_v)
            beta = torch.where(carry.active[..., None], beta, carry.projections.rec_beta)
        o, h_new = delta_scan_chunked(ones, k, v, beta, state.h[pass_index], 1)
        h = tuple(h_new if i == pass_index else s for i, s in enumerate(state.h))
        empty = torch.zeros(b, n, 1, 1, dtype=x.dtype)
        no_z = torch.zeros(b, n, 1, dtype=x.dtype)  # this scalar column has no key filter
        written = Projections(
            rec_z=no_z, rec_k=k, rec_v=v, rec_beta=beta, attn_k=empty, attn_v=empty
        )
        new_state = ColumnState(h=h, kv=state.kv, key_history=state.key_history)
        return o.reshape(b, n, 1), new_state, written


class FixedHalting:
    """Halting probabilities given per pass and token: ``table[t]`` is ``[b, n]``."""

    def __init__(self, table: list[Tensor]) -> None:
        self.table: list[Tensor] = table

    def probability(self, x: Tensor, pass_index: int) -> Tensor:
        del x
        return self.table[pass_index]


def iters(rows: list[list[int]]) -> Tensor:
    """An ``int32`` tensor from nested lists, for comparing ``n_iters`` exactly."""
    out = torch.zeros(len(rows), len(rows[0]), dtype=torch.int32)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[i, j] = value
    return out


def scalar_state(initial: list[float]) -> ColumnState:
    """Per-pass scalar states for one stream: ``initial[t]`` fills ``h[t]``."""
    return ColumnState(
        h=tuple(torch.full((1, 1, 1, 1), value, dtype=F64) for value in initial),
        kv=tuple(kv_cache_empty(1, 1, 1, 1, F64, CPU) for _ in initial),
        key_history=tuple(torch.zeros(1, 0, 1, dtype=F64) for _ in initial),
    )


def run_section_b(a: float, b: float, *, requires_grad: bool = False) -> tuple[LoopResult, Tensor]:
    """Two tokens, two passes, token 1 halts after pass 1, token 2 uses both; H^(1)=0, H^(2)=10."""
    x0 = torch.zeros(1, 2, 1, dtype=F64)
    x0[0, 0, 0], x0[0, 1, 0] = a, b
    x0 = x0.requires_grad_(requires_grad)
    halting = FixedHalting(
        [
            torch.zeros(1, 2, dtype=F64).index_fill_(1, torch.zeros(1, dtype=torch.long), 1.0),
            torch.ones(1, 2, dtype=F64),
        ]
    )
    loop = Loop([ScalarColumn()], halting, d=1, t_max=2, halt_epsilon=0.01)
    result = loop.forward(x0, torch.arange(2).reshape(1, 2), [scalar_state([0.0, 10.0])])
    return result, x0


def test_section_b_copy_forward_table() -> None:
    # the review's table: pass-1 state 5/2 after both tokens; token 1 halted, so at pass 2 its
    # saved (k, v, beta) = (1, 2, 1/2) is written first: 10/2 + 2/2 = 6, then token 2 with its
    # pass-1 output 5/2 as value: 6/2 + (5/2)/2 = 17/4, which is also its pass-2 output
    result, _ = run_section_b(2.0, 4.0)
    assert result.states[0].h[0].item() == Fraction(5, 2)
    assert result.states[0].h[1].item() == Fraction(17, 4)
    assert result.y[0, 0, 0].item() == 1.0  # token 1 halted after pass 1 with output H1 = 1
    assert result.y[0, 1, 0].item() == Fraction(17, 4)
    assert torch.equal(result.n_iters, iters([[1, 2]]))
    assert result.per_pass[1, 0, 0, 0].item() == 0.0  # token 1 produced nothing at pass 2


def test_section_b_pass_two_state_after_token_one_is_six() -> None:
    # the same run with token 2 removed from pass 2's view: a one-token chunk shows the state
    # after the copied update alone
    x0 = torch.zeros(1, 1, 1, dtype=F64)
    x0[0, 0, 0] = 2.0
    halting = FixedHalting([torch.ones(1, 1, dtype=F64), torch.ones(1, 1, dtype=F64)])
    loop = Loop([ScalarColumn()], halting, d=1, t_max=2, halt_epsilon=0.01)
    result = loop.forward(x0, torch.arange(1).reshape(1, 1), [scalar_state([0.0, 10.0])])
    assert result.states[0].h[0].item() == 1.0
    assert result.states[0].h[1].item() == 6.0  # 10/2 + 2/2: the inactive token still writes


def test_section_c_copied_projections_keep_their_gradient_in_e2e_mode() -> None:
    # d(token 2's pass-2 output)/d(a) is 3/8 through the copied projection; a detached copy
    # would give 1/8 (the review's probe): H2 = 5/2 + 3a/8 + b/4 at (a, b) = (2, 4)
    result, x0 = run_section_b(2.0, 4.0, requires_grad=True)
    (grad,) = torch.autograd.grad(result.y[0, 1, 0], [x0])
    assert grad[0, 0, 0].item() == Fraction(3, 8)
    assert grad[0, 1, 0].item() == Fraction(1, 4)


def test_all_tokens_halted_passes_still_update_every_later_state() -> None:
    # both tokens halt after pass 1 of three: passes 2 and 3 see only copied updates, and both
    # states move from their initial values by exactly those writes
    x0 = torch.zeros(1, 2, 1, dtype=F64)
    x0[0, 0, 0], x0[0, 1, 0] = 2.0, 4.0
    halting = FixedHalting([torch.ones(1, 2, dtype=F64)] * 3)
    loop = Loop([ScalarColumn()], halting, d=1, t_max=3, halt_epsilon=0.01)
    result = loop.forward(x0, torch.arange(2).reshape(1, 2), [scalar_state([0.0, 10.0, 20.0])])
    assert torch.equal(result.n_iters, iters([[1, 1]]))
    # pass 2: 10 -> 10/2 + 2/2 = 6 -> 6/2 + 4/2 = 5; pass 3: 20 -> 11 -> 7.5
    assert result.states[0].h[1].item() == 5.0
    assert result.states[0].h[2].item() == 7.5
    assert torch.equal(result.per_pass[1:], torch.zeros(2, 1, 2, 1, dtype=F64))


# --- ACT weights, remainder, ponder, the T_max cap ----------------------------------------------


def test_halting_weights_remainder_and_ponder() -> None:
    # the ponder is the expected depth under the allocation weights, sum_t a_t t
    # token A: p = (0.3, 0.9) halts at pass 2 with weights (0.3, 0.7), ponder 0.3 + 1.4 = 1.7
    # token B: p = 0.995 halts at pass 1 with weight 1, ponder 1
    # token C: p = (0.1, 0.1, 0.1) reaches T_max = 3 with weights (0.1, 0.1, 0.8), ponder 2.7
    p1 = torch.zeros(1, 3, dtype=F64)
    p1[0, 0], p1[0, 1], p1[0, 2] = 0.3, 0.995, 0.1
    p2 = torch.zeros(1, 3, dtype=F64)
    p2[0, 0], p2[0, 1], p2[0, 2] = 0.9, 0.5, 0.1
    p3 = torch.full((1, 3), 0.1, dtype=F64)
    loop = Loop([ScalarColumn()], FixedHalting([p1, p2, p3]), d=1, t_max=3, halt_epsilon=0.01)
    x0 = torch.zeros(1, 3, 1, dtype=F64)
    x0[0, 0, 0], x0[0, 1, 0], x0[0, 2, 0] = 2.0, 4.0, 8.0
    result = loop.forward(x0, torch.arange(3).reshape(1, 3), [scalar_state([0.0, 0.0, 0.0])])
    assert torch.equal(result.n_iters, iters([[2, 1, 3]]))
    assert abs(result.ponder[0, 0].item() - 1.7) < 1e-12
    assert abs(result.ponder[0, 1].item() - 1.0) < 1e-12
    assert abs(result.ponder[0, 2].item() - 2.7) < 1e-12
    outs = result.per_pass[:, 0, :, 0]  # [T, n]
    expected_a = 0.3 * outs[0, 0] + 0.7 * outs[1, 0]
    expected_c = 0.1 * outs[0, 2] + 0.1 * outs[1, 2] + 0.8 * outs[2, 2]
    assert abs(result.y[0, 0, 0].item() - expected_a.item()) < 1e-12
    assert result.y[0, 1, 0].item() == outs[0, 1].item()
    assert abs(result.y[0, 2, 0].item() - expected_c.item()) < 1e-12


def test_ponder_is_the_expected_depth_and_falls_as_p_rises() -> None:
    # identical outputs at every pass, T_max = 3: never halting costs 3, halting at once costs 1,
    # and at a constant interior p the depth is 3 - 3p, so its gradient in p is -3 (the spec's
    # first formula gave 0, 0.995 and +6 here, rewarding the deepest token)
    x0 = torch.zeros(1, 1, 1, dtype=F64)
    x0[0, 0, 0] = 2.0
    pos = torch.arange(1).reshape(1, 1)
    for p_value, expected in ((0.0, 3.0), (0.995, 1.0), (0.1, 2.7)):
        p = torch.full((1, 1), p_value, dtype=F64, requires_grad=True)
        loop = Loop([ScalarColumn()], FixedHalting([p, p, p]), d=1, t_max=3, halt_epsilon=0.01)
        result = loop.forward(x0, pos, [scalar_state([0.0, 0.0, 0.0])])
        assert abs(result.ponder[0, 0].item() - expected) < 1e-12, p_value
        if 0.0 < p_value < 0.3:
            (grad,) = torch.autograd.grad(result.ponder.sum(), [p])
            assert abs(grad.item() + 3.0) < 1e-12


class PoisonColumn(ScalarColumn):
    """The scalar column, returning NaN at every inactive position: the review's stand-in."""

    @override
    def step(
        self,
        x: Tensor,
        pos: Tensor,
        state: ColumnState,
        pass_index: int,
        *,
        modulation: Modulation | None = None,
        carry: Carry | None = None,
    ) -> tuple[Tensor, ColumnState, Projections]:
        y, new_state, written = super().step(
            x, pos, state, pass_index, modulation=modulation, carry=carry
        )
        if carry is not None:
            y = torch.where(carry.active[..., None], y, torch.full_like(y, float("nan")))
        return y, new_state, written


class ScaleColumn:
    """A second stand-in: ``w * x`` per position, with ``w`` a leaf, so gradients are visible."""

    def __init__(self, w: Tensor) -> None:
        self.w: Tensor = w

    def step(
        self,
        x: Tensor,
        pos: Tensor,
        state: ColumnState,
        pass_index: int,
        *,
        modulation: Modulation | None = None,
        carry: Carry | None = None,
    ) -> tuple[Tensor, ColumnState, Projections]:
        del pos, pass_index, modulation, carry
        b, n = x.shape[0], x.shape[1]
        empty = torch.zeros(b, n, 1, 1, dtype=x.dtype)
        written = Projections(
            rec_z=torch.zeros(b, n, x.shape[2], dtype=x.dtype),
            rec_k=empty,
            rec_v=empty,
            rec_beta=torch.zeros(b, n, 1, dtype=x.dtype),
            attn_k=empty,
            attn_v=empty,
        )
        return self.w * x, state, written


def test_a_poisoned_column_cannot_reach_the_next_column_or_its_gradient() -> None:
    # the review's two-column case: the first column returns NaN at inactive positions, the
    # second multiplies by w; a token halted after pass 1 must keep a finite output and a finite
    # d(output)/dw, which means the mask applies between the columns, not after them
    w = torch.ones((), dtype=F64, requires_grad=True)
    x0 = torch.zeros(1, 2, 1, dtype=F64)
    x0[0, 0, 0], x0[0, 1, 0] = 2.0, 4.0
    halting = FixedHalting(
        [
            torch.zeros(1, 2, dtype=F64).index_fill_(1, torch.zeros(1, dtype=torch.long), 1.0),
            torch.ones(1, 2, dtype=F64),
        ]
    )
    loop = Loop([PoisonColumn(), ScaleColumn(w)], halting, d=1, t_max=2, halt_epsilon=0.01)
    states = [scalar_state([0.0, 10.0]), scalar_state([0.0, 0.0])]
    result = loop.forward(x0, torch.arange(2).reshape(1, 2), states)
    assert bool(torch.isfinite(result.y).all().item())
    assert result.y[0, 0, 0].item() == 1.0  # token 1: pass-1 state read 1, times w = 1
    (grad,) = torch.autograd.grad(result.y[0, 0, 0], [w])
    assert grad.item() == 1.0


def test_inactive_positions_cannot_poison_outputs_or_gradients() -> None:
    x0 = torch.zeros(1, 2, 1, dtype=F64)
    x0[0, 0, 0], x0[0, 1, 0] = 2.0, 4.0
    x0 = x0.requires_grad_()
    halting = FixedHalting(
        [
            torch.zeros(1, 2, dtype=F64).index_fill_(1, torch.zeros(1, dtype=torch.long), 1.0),
            torch.ones(1, 2, dtype=F64),
        ]
    )
    loop = Loop([PoisonColumn()], halting, d=1, t_max=2, halt_epsilon=0.01)
    result = loop.forward(x0, torch.arange(2).reshape(1, 2), [scalar_state([0.0, 10.0])])
    assert bool(torch.isfinite(result.y).all().item())
    assert result.y[0, 0, 0].item() == 1.0
    assert result.y[0, 1, 0].item() == Fraction(17, 4)
    assert bool(torch.isfinite(result.per_pass).all().item())
    assert bool(torch.isfinite(result.ponder).all().item())
    (grad,) = torch.autograd.grad(result.y.sum(), [x0])
    assert bool(torch.isfinite(grad).all().item())
    assert grad[0, 0, 0].item() == Fraction(3, 8) + 1.0 / 2  # token 2's 3/8 plus token 1's own 1/2


# --- the real column --------------------------------------------------------------------------


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


def _real_loop(t_max: int, seed: int = 0) -> tuple[Loop, list[Column], HaltingHead]:
    g = torch.Generator().manual_seed(seed)
    columns = [Column(SHAPE, g).to(F64) for _ in range(2)]
    head = HaltingHead(SHAPE.d, g).to(F64)
    return Loop(columns, head, d=SHAPE.d, t_max=t_max, halt_epsilon=0.01), columns, head


def _real_states(b: int, t_max: int) -> list[ColumnState]:
    return [init_column_state(SHAPE, b, t_max, F64, CPU) for _ in range(2)]


def test_without_halting_pressure_the_loop_is_the_plain_pass_major_stack() -> None:
    loop, columns, head = _real_loop(t_max=2)
    with torch.no_grad():
        _ = head.b_h.fill_(-50.0)  # p ~ 0: every token runs to T_max
        _ = loop.u.normal_(generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(1)
    x0 = torch.randn(2, 5, SHAPE.d, generator=g, dtype=F64)
    pos = torch.arange(5).expand(2, 5)
    result = loop.forward(x0, pos, _real_states(2, 2))
    # by hand: pass t adds u[t], then column 0 then column 1 through their pass-t states
    x = x0
    states = _real_states(2, 2)
    for t in range(2):
        h = x + loop.u[t]
        for c, column in enumerate(columns):
            h, states[c] = column.forward(h, pos, states[c], t)
        x = h
    assert (result.y - x).abs().max().item() < 1e-10
    assert torch.equal(result.n_iters, iters([[2] * 5] * 2))
    for c in range(2):
        for t in range(2):
            assert (result.states[c].h[t] - states[c].h[t]).abs().max().item() < 1e-10


def test_gradient_reaches_conditioning_halting_and_every_column() -> None:
    loop, _, _ = _real_loop(t_max=3)
    g = torch.Generator().manual_seed(2)
    x0 = torch.randn(1, 4, SHAPE.d, generator=g, dtype=F64)
    result = loop.forward(x0, torch.arange(4).reshape(1, 4), _real_states(1, 3))
    loss = result.y.square().sum() + result.ponder.sum()
    params = dict(loop.named_parameters())
    grads = torch.autograd.grad(loss, list(params.values()))
    for (name, _), grad in zip(params.items(), grads, strict=True):
        assert grad.abs().sum().item() > 0, name
    assert "u" in params
    assert any(name.endswith("w_h") for name in params)


def test_the_halting_head_is_blind_to_the_scale_of_its_input() -> None:
    # docs/reviews/2026-10-15-halting-diagnosis.md: the head read the raw residual state (L2 about
    # 1,400 at pass 1, growing with every pass), so its logit carried the state's scale
    head = HaltingHead(64, torch.Generator().manual_seed(0)).to(F64)
    x = torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(1), dtype=F64)
    # up to RMSNorm's 1e-6 inside the root, which is about 5e-7 of a unit-RMS row's mean square
    # and moves p by less than 1e-6 here (it is a regularizer, not a floor: column.RMSNorm)
    for scale in (30.0, 1e4):
        assert torch.allclose(head.probability(x * scale, 0), head.probability(x, 0), atol=1e-6)


@pytest.mark.parametrize(
    "precision",
    [
        "float32",
        "bf16-cpu",
        pytest.param(
            "bf16-cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
        ),
    ],
)
@pytest.mark.parametrize("kind", ["constant", "random"])
@pytest.mark.parametrize("scale", [1.0, 30.0, 1000.0])
def test_one_fresh_adam_step_moves_a_halting_logit_by_at_most_lr_times_d_plus_one(
    scale: float, kind: str, precision: str
) -> None:
    # the short run's trigger (halting diagnosis; maintainer B's fixture shape): the head is freed
    # from its warm values (w = 0, p = 1/8) and Adam's first step is lr * g / (|g| + eps), lr *
    # sign(-g) up to eps, on every coordinate; on a raw input of 30 per coordinate that moved every
    # logit by about 6.9 at once, depth 8 -> 1. Reading a normalized state with unit gain,
    # |RMSNorm(h)|_1 <= d, so the first update moves no logit by more than lr * (d + 1) (maintainer
    # B's bound, 0.2293 at the run's lr), whatever the input's scale. The bound is for exact
    # arithmetic with fixed inputs, a zero initial direction, a unit gain and one fresh update; a
    # learned gain or later updates change it.
    d, lr = 768, 2.98139e-4
    device = torch.device("cuda" if precision == "bf16-cuda" else "cpu")
    head = HaltingHead(d, torch.Generator().manual_seed(0)).to(device)
    with torch.no_grad():
        _ = head.w_h.zero_()
        _ = head.b_h.fill_(math.log((1 / 8) / (7 / 8)))
    g = torch.Generator().manual_seed(2)
    raw = torch.ones(4, 16, d) if kind == "constant" else torch.randn(4, 16, d, generator=g)
    x = (scale * raw).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    bf16 = precision != "float32"

    def probability() -> Tensor:
        # pre and post are observed inside the same autocast as the training forward (maintainer B)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
            return head.probability(x, 0)

    with torch.no_grad():
        p_before = probability().double()
    loss = -torch.log(probability().float()).mean()  # a task gradient favouring halting
    torch.autograd.backward([loss])
    opt.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
    with torch.no_grad():
        p_after = probability().double()
    moved = (torch.logit(p_after) - torch.logit(p_before)).abs().max().item()
    bound = lr * (d + 1)
    if bf16:
        # under autocast the logit is a bf16 matmul plus a 0-dim bias (no promotion) and the
        # probability returns in bf16: each of the two observations is off by at most half a bf16
        # spacing of the logit (|logit| < 2, spacing <= 2^-7) plus half a spacing of p (p < 1/4,
        # spacing <= 2^-10) seen through dlogit/dp = 1 / (p (1 - p)); maintainer B measured 0.22957
        # against the exact 0.22927
        assert bool((p_before < 0.25).all() and (p_after < 0.25).all())
        p_low = float(min(p_before.min().item(), p_after.min().item()))
        tolerance = 2 * (2**-8 + 2**-11 / (p_low * (1 - p_low)))
    else:
        tolerance = bound * 1e-4
    assert 0 < moved <= bound + tolerance
    assert bool((p_after < 1 - 0.01).all())  # no token halts at pass 1 yet


def _same_run(a: LoopResult, b: LoopResult) -> None:
    assert torch.equal(a.n_iters, b.n_iters)
    assert (a.y - b.y).abs().max().item() < 1e-12
    assert (a.per_pass - b.per_pass).abs().max().item() < 1e-12
    for sa, sb in zip(a.states, b.states, strict=True):
        for ha, hb in zip(sa.h, sb.h, strict=True):
            assert (ha - hb).abs().max().item() < 1e-12


def test_the_dense_rule_runs_every_pass_with_fresh_writes() -> None:
    # dense = ACT that never halts before T: every pass active, the last pass's own output
    loop, _, _ = _real_loop(t_max=3)
    x0 = torch.randn(2, 5, SHAPE.d, generator=torch.Generator().manual_seed(4), dtype=F64)
    pos = torch.arange(5).expand(2, 5)
    dense = loop.forward(x0, pos, _real_states(2, 3), exit_rule=DenseRule())
    never = FixedHalting([torch.zeros(2, 5, dtype=F64)] * 3)
    act = loop.forward(x0, pos, _real_states(2, 3), exit_rule=ActRule(never, 0.01))
    _same_run(dense, act)
    assert torch.equal(dense.n_iters, iters([[3] * 5] * 2))
    assert (dense.y - dense.per_pass[2]).abs().max().item() == 0.0


def test_the_dense_fast_path_equals_the_merging_path_bitwise_with_gradients() -> None:
    # DenseRule skips every copy-forward selection (2026-10-21: ~1500 kernels of a training
    # step); a trace exiting every token at the last pass means the same but runs the merges, so
    # outputs, states and every parameter's gradient must be equal to the bit
    loop, _, _ = _real_loop(t_max=3)
    x0 = torch.randn(2, 5, SHAPE.d, generator=torch.Generator().manual_seed(9), dtype=F64)
    pos = torch.arange(5).expand(2, 5)
    up = torch.randn(3, 2, 5, SHAPE.d, generator=torch.Generator().manual_seed(10), dtype=F64)

    def run(rule: DenseRule | TraceRule) -> tuple[LoopResult, list[Tensor]]:
        loop.zero_grad(set_to_none=True)
        result = loop.forward(x0, pos, _real_states(2, 3), exit_rule=rule)
        torch.autograd.backward([(result.per_pass * up).sum() + result.y.sum()])
        grads = [p.grad.clone() for p in loop.parameters() if p.grad is not None]
        return result, grads

    fast, fast_grads = run(DenseRule())
    slow, slow_grads = run(TraceRule(iters([[3] * 5] * 2)))
    assert torch.equal(fast.y, slow.y)
    assert torch.equal(fast.per_pass, slow.per_pass)
    assert torch.equal(fast.n_iters, slow.n_iters)
    for sa, sb in zip(fast.states, slow.states, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(sa.h, sb.h, strict=True))
        assert all(torch.equal(a.k, b.k) for a, b in zip(sa.kv, sb.kv, strict=True))
    assert len(fast_grads) == len(slow_grads) > 0
    assert all(torch.equal(a, b) for a, b in zip(fast_grads, slow_grads, strict=True))


def test_a_trace_replays_a_given_exit_pattern_with_copy_forward() -> None:
    # a trace is ACT with a one-hot probability at the traced pass: the same halts, the same
    # allocation (1 - 0 = 1), so the same copy-forward states and outputs
    loop, _, _ = _real_loop(t_max=3)
    x0 = torch.randn(1, 6, SHAPE.d, generator=torch.Generator().manual_seed(5), dtype=F64)
    pos = torch.arange(6).reshape(1, 6)
    depths = iters([[1, 3, 2, 1, 2, 3]])
    table = [(depths == t + 1).to(F64) for t in range(3)]
    traced = loop.forward(x0, pos, _real_states(1, 3), exit_rule=TraceRule(depths))
    act = loop.forward(x0, pos, _real_states(1, 3), exit_rule=ActRule(FixedHalting(table), 0.01))
    _same_run(traced, act)
    assert torch.equal(traced.n_iters, depths)
    with pytest.raises(ContractError, match="trace"):
        _ = loop.forward(x0, pos, _real_states(1, 3), exit_rule=TraceRule(depths[:, :3]))


def test_the_cdf_rule_exits_where_the_dense_distribution_crosses_q() -> None:
    # one token: its own passes up to the exit do not depend on copy-forward, so the live exit
    # equals hazard_cdf_exits over the dense forward's per-pass hazard logits
    loop, _, head = _real_loop(t_max=4)
    with torch.no_grad():
        _ = head.w_h.mul_(1.5)  # logits of order one: the baseline still decides some exits
    seen: set[int] = set()
    for seed in range(8):
        x0 = torch.randn(1, 1, SHAPE.d, generator=torch.Generator().manual_seed(seed), dtype=F64)
        pos = torch.zeros(1, 1, dtype=torch.long)
        dense = loop.forward(x0, pos, _real_states(1, 4), exit_rule=DenseRule())
        z = hazard_logits(torch.stack([head.logit(dense.per_pass[t], t) for t in range(4)]))
        # the token's own CDF values are the boundaries: q exactly there and float32-rounded
        survival, cdf = cdf_begin(z[0])
        boundaries: list[float] = []
        for t in range(3):
            survival, cdf = cdf_advance(z[t], survival, cdf)
            boundaries += [float(cdf.item()), float(cdf.float().item())]
        for q in (0.2, 0.35, 0.5, 0.65, 0.8, *[b for b in boundaries if 0.0 < b <= 1.0]):
            live = loop.forward(x0, pos, _real_states(1, 4), exit_rule=CdfRule(head, q))
            assert torch.equal(live.n_iters, hazard_cdf_exits(z, q)), (seed, q)
            depth = int(live.n_iters.item())
            seen.add(depth)
            assert (live.y - live.per_pass[depth - 1]).abs().max().item() == 0.0
    assert seen == {1, 2, 3, 4}  # the fixture exercises every exit, so the check discriminates


def test_the_fixed_uniform_rule_exits_at_the_quantile_for_every_token() -> None:
    # condition B deploys with the fixed uniform distribution, not an untrained head (efeb8f3)
    loop, _, _ = _real_loop(t_max=4)
    x0 = torch.randn(2, 5, SHAPE.d, generator=torch.Generator().manual_seed(6), dtype=F64)
    pos = torch.arange(5).expand(2, 5)
    # q just past each quantile too: a baseline shifted by one pass exits early there
    cases = ((0.25, 1), (0.26, 2), (0.5, 2), (0.51, 3), (0.75, 3), (0.76, 4), (1.0, 4))
    for q, depth in cases:
        live = loop.forward(x0, pos, _real_states(2, 4), exit_rule=CdfRule(None, q))
        assert torch.equal(live.n_iters, iters([[depth] * 5] * 2)), q


def _live_exit(rule: CdfRule, t_max: int) -> int:
    """The pass at which ``rule`` exits one token (its decision alone, no columns)."""
    active = torch.ones(1, 1, dtype=torch.bool)
    decide = rule.begin(active, torch.float32, t_max)
    for t in range(t_max):
        halts, _ = decide(t, torch.zeros(1, 1, SHAPE.d), active, last=t == t_max - 1)
        if bool(halts.item()):
            return t + 1
    raise AssertionError


@pytest.mark.parametrize("t_max", range(2, 33))
def test_the_fixed_uniform_rule_agrees_with_the_dense_exits_at_every_boundary(t_max: int) -> None:
    # the uniform CDF is exactly k / T at pass k, so q = k / T sits on a boundary (the reference
    # q = 0.5 is one): the live rule exits at k and agrees with hazard_cdf_exits there and at the
    # float32-rounded q (maintainer B's review of 49c92cf, finding 4: T = 7, q = float32(5/7) gave 5
    # dense and 6 live; 127 disagreements over T = 2..32)
    z = hazard_logits(torch.zeros(t_max, 1, 1))
    for k in range(1, t_max + 1):
        exact = k / t_max
        assert _live_exit(CdfRule(None, exact), t_max) == k, (t_max, k)
        rounded = float(torch.tensor(exact, dtype=torch.float32).item())
        if rounded <= 1.0:
            dense = int(hazard_cdf_exits(z, rounded).item())
            assert _live_exit(CdfRule(None, rounded), t_max) == dense, (t_max, k)


def test_a_trace_outside_the_forward_or_not_integer_is_refused() -> None:
    # Maintainer B's review of 49c92cf, finding 5: depths 0, 9 and 1.5 were forced to the last pass
    loop, _, _ = _real_loop(t_max=3)
    x0 = torch.randn(1, 4, SHAPE.d, generator=torch.Generator().manual_seed(2), dtype=F64)
    pos = torch.arange(4).reshape(1, 4)
    for depths in (
        iters([[1, 0, 2, 3]]),
        iters([[1, 4, 2, 3]]),
        torch.full((1, 4), 1.5, dtype=F64),
        torch.ones(1, 4, dtype=torch.bool),
        # unsigned and complex reached NotImplementedError past the former denylist (maintainer B's
        # review of 230a5e5, gap b)
        torch.ones(1, 4, dtype=torch.uint8),
        torch.ones(1, 4, dtype=torch.uint16),
        torch.ones(1, 4, dtype=torch.uint32),
        torch.ones(1, 4, dtype=torch.uint64),
        torch.ones(1, 4, dtype=torch.complex64),
    ):
        with pytest.raises(ContractError, match="trace depths"):
            _ = loop.forward(x0, pos, _real_states(1, 3), exit_rule=TraceRule(depths))
    # a shorter forward narrows the range: depth 3 is outside two passes
    with pytest.raises(ContractError, match=r"\[1, 2\]"):
        _ = loop.forward(
            x0, pos, _real_states(1, 3), passes=2, exit_rule=TraceRule(iters([[1, 2, 3, 1]]))
        )


def test_loop_rejects_mismatched_states() -> None:
    loop, _, _ = _real_loop(t_max=2)
    x0 = torch.zeros(1, 3, SHAPE.d, dtype=F64)
    pos = torch.arange(3).reshape(1, 3)
    with pytest.raises(ValueError, match="states for"):
        _ = loop.forward(x0, pos, _real_states(1, 2)[:1])
    with pytest.raises(ValueError, match="passes"):
        _ = loop.forward(x0, pos, _real_states(1, 3))
    with pytest.raises(ContractError, match="x0"):
        _ = loop.forward(torch.zeros(1, 3, 1, dtype=F64), pos, _real_states(1, 2))
    with pytest.raises(ValueError, match="halt_epsilon"):
        _ = Loop([ScalarColumn()], FixedHalting([]), d=1, t_max=1, halt_epsilon=1.0)
    with pytest.raises(ValueError, match="at least one column"):
        _ = Loop([], FixedHalting([]), d=1, t_max=1, halt_epsilon=0.01)


def test_column_step_checks_the_carry_mask_and_returns_what_it_wrote() -> None:
    column = Column(SHAPE, torch.Generator().manual_seed(0)).to(F64)
    x = torch.zeros(1, 2, SHAPE.d, dtype=F64)
    pos = torch.arange(2).reshape(1, 2)
    state = init_column_state(SHAPE, 1, 1, F64, CPU)
    y, new_state, written = column.step(x, pos, state, 0)
    with pytest.raises(ContractError, match=r"carry\.active"):
        _ = column.step(x, pos, state, 0, carry=Carry(torch.ones(1, 3, dtype=torch.bool), written))
    with pytest.raises(ContractError, match=r"carry\.active"):
        _ = column.step(x, pos, state, 0, carry=Carry(torch.ones(1, 2), written))
    # carried projections of another chunk length or another column are rejected, not broadcast
    short = Projections(
        rec_z=written.rec_z,
        rec_k=written.rec_k[:, :1],
        rec_v=written.rec_v[:, :1],
        rec_beta=written.rec_beta[:, :1],
        attn_k=written.attn_k[:, :1],
        attn_v=written.attn_v[:, :1],
    )
    with pytest.raises(ContractError, match=r"carry\.rec_k"):
        _ = column.step(x, pos, state, 0, carry=Carry(torch.ones(1, 2, dtype=torch.bool), short))
    # the unfiltered key input a successor's filter reads is checked as well
    short_z = replace(written, rec_z=written.rec_z[:, :1])
    with pytest.raises(ContractError, match=r"carry\.rec_z"):
        _ = column.step(x, pos, state, 0, carry=Carry(torch.ones(1, 2, dtype=torch.bool), short_z))
    # with every token active and its own projections carried, the carry changes nothing
    again, _, _ = column.step(
        x, pos, state, 0, carry=Carry(torch.ones(1, 2, dtype=torch.bool), written)
    )
    assert torch.equal(again, y)
    assert y.shape == (1, 2, SHAPE.d)
    assert isinstance(new_state, ColumnState)
    assert written.rec_k.shape == (1, 2, SHAPE.heads, SHAPE.head_dim)
    assert written.attn_k.shape == (1, 2, SHAPE.kv_heads, SHAPE.head_dim)
    assert isinstance(new_state.kv[0], KVCache)
