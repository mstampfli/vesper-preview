"""The loop and halting controller (spec 4): columns ``1..C`` in order, ``T_max`` passes, ACT.

Pass ``t`` conditions the input with ``u_t`` (one learned vector per pass, shared across columns)
and runs every column through its pass-``t`` state. After the pass a halting head gives
``p_t = sigmoid(w_h . RMSNorm(x) + b_h)`` per token; ``P`` accumulates it and a token halts at the
first pass where ``P >= 1 - epsilon`` or at ``T_max``. The token's output is the halting-weighted
mean of its per-pass outputs, ``p_t`` for the passes before the halting one and the remainder
``1 - P`` at it, so the weights sum to one and the gradient reaches every pass used. The ponder
cost is the expected depth under those same allocation weights, ``sum_t a_t t`` per token: it is
``1`` when a token halts at once and ``T_max`` when it never halts, and it falls as ``p`` rises.
The spec's first formula, ``sum_t p_t t`` over the raw probabilities, rewarded never halting
(``p = 0`` used every pass at zero cost and its gradient pushed toward maximum depth); the
pass-state review's follow-up found it and the spec was corrected
(docs/reviews/2026-10-15-loop-review.md).

The halting decision is an exit rule (``ExitRule``): ACT above is the default; the halting
experiment adds ``DenseRule`` (every pass, fresh writes), ``TraceRule`` (a given exit trace) and
``CdfRule`` (the first pass whose exit CDF reaches ``q``); every rule shares the same copy-forward
(docs/reviews/2026-10-15-halting-experiment-plan.md).

Copy-forward (spec 4; docs/reviews/2026-10-14-pass-state-review.md, B and D): a halted token
leaves the active set, and its projections from its last active pass are written into every later
pass's recurrent state and KV window through the same equations an active token uses, so later
tokens see it at every pass and both execution orders agree. All ``T_max`` passes run even when
no token is active any more: the state-only work of the inactive tokens does not disappear. This
reference computes the inactive positions densely and ignores their outputs; a kernel skips them.

The loop takes any ``ColumnLike`` and ``HaltingLike``, so its arithmetic is tested against the
review's hand-computed table with a scalar column before it is trusted with the real one.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, override, runtime_checkable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

from vesper.column import Carry, ColumnState, Modulation, Projections, RMSNorm
from vesper.contracts import ContractError, Dims, check_shape, require
from vesper.exits import (
    cdf_advance,
    cdf_begin,
    cdf_reaches,
    floored_stop_log_probs,
    hazard_baseline,
)


@runtime_checkable
class ColumnLike(Protocol):
    """What the loop needs from a column: ``Column.step`` or a stand-in with its contract."""

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
        """See ``Column.step``."""
        ...


@runtime_checkable
class HaltingLike(Protocol):
    """``p [b, n]`` in ``[0, 1]`` from the pass output; the loop never sees the logits."""

    def probability(self, x: Tensor, pass_index: int) -> Tensor:
        """Halting probability per token after pass ``pass_index``."""
        ...


class HaltingHead(nn.Module):
    """The spec's halting head: ``sigmoid(w_h . RMSNorm(x) + b_h)``, ``2d + 1`` parameters.

    The head reads the pass's output through its own ``RMSNorm``, so its logit does not carry the
    residual state's scale, which grows with every pass (about 1,400 at pass 1 and 8,600 at pass
    8 in the short run, 354 at pass 1 already at initialization). Reading the raw state, the first
    free update (Adam's ``lr * g / (|g| + eps)``, ``lr * sign(-g)`` up to ``eps``, on every
    coordinate) shifted the held-out logits by about 8.7 in common and halted 98.91 % of the
    held-out positions at pass 1, every position from the next update on
    (docs/reviews/2026-10-15-halting-diagnosis.md); with a unit gain and a zero direction, the
    first update moves a logit by at most ``lr * (d + 1)`` in exact arithmetic.
    """

    def __init__(self, d: int, generator: torch.Generator | None = None) -> None:
        """Zero bias, a small random direction, a unit gain; ``generator`` makes it reproducible."""
        super().__init__()
        self.norm: RMSNorm = RMSNorm(d)
        self.w_h: nn.Parameter = nn.Parameter(torch.randn(d, generator=generator) / math.sqrt(d))
        self.b_h: nn.Parameter = nn.Parameter(torch.zeros(()))

    def logit(self, x: Tensor, pass_index: int) -> Tensor:
        """``[b, n, d] -> [b, n]``: ``w_h . RMSNorm(x) + b_h``; the pass index is unused here."""
        del pass_index
        return self.norm.forward(x) @ self.w_h + self.b_h

    def probability(self, x: Tensor, pass_index: int) -> Tensor:
        """``[b, n, d] -> [b, n]``: the sigmoid of ``logit``."""
        return torch.sigmoid(self.logit(x, pass_index))

    def zero_residual(self) -> None:
        """Direction zero, bias zero, gain one: zero logits, the all-pass conditions' start.

        With ``vesper.exits.hazard_baseline`` added, the exit distribution is exactly uniform.
        Not ACT's frozen start (``train.set_halting``), whose bias is the logit of ``1/T``: that
        bias on top of the baseline would apply the prior twice (maintainer B, channel efeb8f3).
        """
        with torch.no_grad():
            _ = self.w_h.zero_()
            _ = self.b_h.zero_()
            _ = self.norm.weight.fill_(1.0)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Same as ``probability`` for the generic module call."""
        return self.probability(x, 0)


@runtime_checkable
class ExitStep(Protocol):
    """One forward's halting decision after pass ``t``: ``(halts bool[b, n], allocation [b, n])``.

    ``halts`` must be a subset of ``active`` and must hold for every active token when ``last``;
    ``allocation`` is the output weight of pass ``t`` (zero where inactive).
    """

    def __call__(self, t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
        """Decide after pass ``t`` from its output ``h [b, n, d]``."""
        ...


@runtime_checkable
class ExitRule(Protocol):
    """A halting policy; ``begin`` makes the per-forward decision state."""

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """A fresh decision for one forward of ``runs`` passes over ``active bool[b, n]``."""
        ...


@dataclass(frozen=True)
class ActRule:
    """ACT (spec 4): accumulate ``p``, halt at ``P + p >= 1 - epsilon``, output the weighted mean.

    The loop's default rule, unchanged from before exit rules existed (the pass-state review's
    hand-computed table pins it): ``a_t = p_t`` before the halting pass, ``1 - P`` at it.
    """

    halting: HaltingLike
    epsilon: float

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """The accumulated probability starts at zero for every token."""
        del runs
        cumulative = torch.zeros(active.shape, dtype=dtype, device=active.device)

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            nonlocal cumulative
            p = torch.where(active, self.halting.probability(h, t), torch.zeros_like(cumulative))
            halts = active & ((cumulative + p >= 1.0 - self.epsilon) | last)
            allocation = torch.where(halts, 1.0 - cumulative, p)  # zero where inactive
            cumulative = cumulative + p
            return halts, allocation

        return step


@dataclass(frozen=True)
class DenseRule:
    """No exit before the last pass: every token runs every pass with fresh writes.

    The training state policy of the halting experiment's all-pass conditions (B-D); the output
    is the last pass's own, and the per-pass outputs carry the supervision.
    """

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """No state: the decision depends only on the pass."""
        del active, runs

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            del t, h
            halts = active & last
            return halts, halts.to(dtype)

        return step


@dataclass(frozen=True)
class TraceRule:
    """Exit every token at a given pass ``depths int[b, n]`` (1-based): the matched-trace policy.

    Replays an exit trace measured elsewhere (for example the dense forward's CDF exits) under
    copy-forward, isolating the state difference from the decision difference. The emitted
    output is the exit pass's own.
    """

    depths: Tensor

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """The trace must be integer, cover exactly the forward's tokens and lie in ``1..runs``.

        Checked before any state is touched: a depth outside the forward would otherwise be
        forced to the last pass silently and the matched-trace comparison would replay a
        different trace than the one measured (maintainer B's review of 49c92cf, finding 5).
        """
        if self.depths.shape != active.shape:
            msg = f"trace of shape {tuple(self.depths.shape)} for tokens {tuple(active.shape)}"
            raise ContractError(msg)
        # an allowlist, not a denylist: unsigned and complex traces reached NotImplementedError
        # past a denylist (maintainer B's review of 230a5e5, gap b)
        if self.depths.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            msg = f"trace depths must be a signed integer tensor, got {self.depths.dtype}"
            raise ContractError(msg)
        if self.depths.numel():
            # require: eager ContractError, a device assert inside a captured evaluation
            in_range = ((self.depths >= 1) & (self.depths <= runs)).all()
            require(in_range, f"trace depths must lie in [1, {runs}]")
        depths = self.depths.to(active.device)

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            del h
            halts = active & ((depths == t + 1) | last)
            return halts, halts.to(dtype)

        return step


@dataclass(frozen=True)
class CdfRule:
    """Exit at the first pass whose cumulative exit probability reaches ``q``: the deployed policy.

    Hazard logits ``head.logit(h, t) + hazard_baseline(runs)[t]`` (float64) from the live pass
    output, accumulated by ``cdf_advance`` and compared by ``cdf_reaches``, the primitives
    ``vesper.exits.hazard_cdf_exits`` uses, so the dense and live policies agree to the bit on
    the same logits; the forced final exit at ``runs``. The emitted output is the exit pass's own.
    ``head = None`` is the FIXED uniform distribution (zero learned logits): condition B deploys
    with it, never with an untrained head's random direction (maintainer B, channel efeb8f3).
    """

    head: HaltingHead | None
    q: float

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """Log-survival and the cumulative exit probability start at zero."""
        if not 0.0 < self.q <= 1.0:
            msg = f"q must lie in (0, 1], got {self.q}"
            raise ValueError(msg)
        baseline = hazard_baseline(runs, active.device)  # built where hazard_logits builds it
        survival, cdf = cdf_begin(active)

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            nonlocal survival, cdf
            if not last:
                learned = (
                    torch.zeros(active.shape, device=active.device)
                    if self.head is None
                    else self.head.logit(h, t)
                )
                # hazard_logits' operation on the same operands, one pass at a time
                z = learned.to(torch.float64) + baseline[t]
                survival, cdf = cdf_advance(z, survival, cdf)
            halts = active & (cdf_reaches(cdf, self.q) | last)
            return halts, halts.to(dtype)

        return step


@dataclass(frozen=True)
class LoopResult:
    """What one chunk through the loop produces.

    ``y [b, n, d]`` is the halting-weighted output; ``states`` the columns' states after the
    chunk; ``n_iters int32[b, n]`` the passes used per token; ``ponder [b, n]`` the expected depth
    ``sum_t a_t t`` under the allocation weights; ``per_pass [T_max, b, n, d]`` the pass outputs
    of active tokens (zero once halted), kept for the local-mode targets and for inspection.
    """

    y: Tensor
    states: tuple[ColumnState, ...]
    n_iters: Tensor
    ponder: Tensor
    per_pass: Tensor


class Loop(nn.Module):
    """``T_max`` passes of ``C`` columns with ACT halting and copy-forward (module docstring)."""

    def __init__(
        self,
        columns: Sequence[ColumnLike],
        halting: HaltingLike,
        d: int,
        t_max: int,
        halt_epsilon: float,
    ) -> None:
        """Conditioning vectors start at zero (no conditioning until learned).

        Raises:
            ValueError: on no columns, ``t_max < 1`` or ``halt_epsilon`` outside ``(0, 1)``.
        """
        super().__init__()
        if len(columns) == 0:
            msg = "the loop needs at least one column"
            raise ValueError(msg)
        if t_max < 1:
            msg = f"t_max must be at least 1, got {t_max}"
            raise ValueError(msg)
        if not 0.0 < halt_epsilon < 1.0:
            msg = f"halt_epsilon must lie in (0, 1), got {halt_epsilon}"
            raise ValueError(msg)
        self.columns: tuple[ColumnLike, ...] = tuple(columns)
        self.modules_: nn.ModuleList = nn.ModuleList(
            [module for c in columns if (module := _as_module(c)) is not None]
        )
        self.halting: HaltingLike = halting
        self.halting_module: nn.Module | None = _as_module(halting)
        self.t_max: int = t_max
        self.halt_epsilon: float = halt_epsilon
        self.u: nn.Parameter = nn.Parameter(torch.zeros(t_max, d))

    def halting_head(self) -> HaltingHead:
        """The loop's head as the spec's ``HaltingHead``.

        Raises:
            TypeError: when the loop was built with another ``HaltingLike`` (a test stand-in).
        """
        if not isinstance(self.halting, HaltingHead):
            msg = "the loop's halting head is not a HaltingHead"
            raise TypeError(msg)
        return self.halting

    @override
    def forward(
        self,
        x0: Tensor,
        pos: Tensor,
        states: Sequence[ColumnState],
        modulation: Modulation | None = None,
        passes: int | None = None,
        *,
        recompute: bool = False,
        exit_rule: ExitRule | None = None,
    ) -> LoopResult:
        """Run one chunk through all passes, or through the first ``passes`` of them.

        Args:
            x0: ``[b, n, d]`` the embedded chunk.
            pos: ``int64[b, n]`` positions, consecutive within the chunk.
            states: one ``ColumnState`` per column, each with ``t_max`` passes.
            modulation: the modulator's signal and strength (inference), or None.
            passes: run this many passes instead of ``t_max`` (the recipe's ``T_max`` anneal);
                the last one run is the forced halt, the states of later passes stay untouched.
            recompute: keep no activations inside a column application for the backward pass;
                recompute them from the column's inputs when the gradient needs them
                (``torch.utils.checkpoint``, non-reentrant). The same objective, gradients and
                states as without, about half again the step time; the internal activations of
                the ``C x T`` applications are rebuilt on demand, while the boundary tensors
                between applications and the carried states still scale with ``C`` and ``T``
                (measured: the recipe's 4096-token chunk at eight passes, one slot, 16 GiB
                without and 4.2 GiB with, on Vesper-S).
            exit_rule: the halting policy; None is ACT with the loop's own head and epsilon
                (``ActRule``). Whatever the rule, a halted token's projections are copied forward
                into every later pass (spec 4).

        Raises:
            ValueError: if ``states`` does not match the columns, a state has the wrong number of
                passes, or ``passes`` is outside ``[1, t_max]``.
            ContractError: on a shape mismatch of ``x0`` or ``pos``.
        """
        runs = self.t_max if passes is None else passes
        if not 1 <= runs <= self.t_max:
            msg = f"passes must lie in [1, {self.t_max}], got {runs}"
            raise ValueError(msg)
        if len(states) != len(self.columns):
            msg = f"{len(states)} states for {len(self.columns)} columns"
            raise ValueError(msg)
        for c, state in enumerate(states):
            if {len(state.h), len(state.kv), len(state.key_history)} != {self.t_max}:
                msg = f"state of column {c} has {len(state.h)} passes, the loop runs {self.t_max}"
                raise ValueError(msg)
        dims: Dims = {"d": self.u.shape[1]}
        check_shape("x0", x0, ("b", "n", "d"), dims)
        check_shape("pos", pos, ("b", "n"), dims, dtype=torch.long)
        b, n = dims["b"], dims["n"]
        device = x0.device
        active = torch.ones(b, n, dtype=torch.bool, device=device)
        rule = ActRule(self.halting, self.halt_epsilon) if exit_rule is None else exit_rule
        decide = rule.begin(active, x0.dtype, runs)
        # DenseRule: no token halts before the last pass, so every copy-forward selection below
        # would pick the fresh value; skipping them is exact (measured 2026-10-21 on the 5090:
        # their where/fill/add kernels were ~1500 of a ~5000-kernel captured training step)
        dense = isinstance(rule, DenseRule)
        y = torch.zeros_like(x0)
        depth = torch.zeros(b, n, dtype=x0.dtype, device=device)
        n_iters = torch.zeros(b, n, dtype=torch.int32, device=device)
        carried: list[Projections | None] = [None] * len(self.columns)
        current = list(states)
        x = x0
        per_pass: list[Tensor] = []
        for t in range(runs):
            h = x + self.u[t]
            fresh: list[Projections] = []
            for c, column in enumerate(self.columns):
                previous = carried[c]
                h, current[c], written = _apply(
                    column,
                    h,
                    pos,
                    current[c],
                    t,
                    modulation,
                    None if previous is None or dense else Carry(active, previous),
                    recompute=recompute,
                )
                fresh.append(written)
                # inactive positions hold whatever the dense pass produced there, meaningful or
                # not: they are zeroed before the next column, the head, the output or a
                # gradient can read them, so a non-finite value there stops at its column
                if not dense:
                    h = torch.where(active[..., None], h, torch.zeros_like(h))
            halts, allocation = decide(t, h, active, last=t == runs - 1)
            y = y + allocation[..., None] * h
            depth = depth + allocation * (t + 1)
            n_iters = n_iters + active.to(torch.int32)
            per_pass.append(h)
            if not dense:  # no token halted, so nothing is ever carried
                for c, written in enumerate(fresh):
                    previous = carried[c]
                    carried[c] = written if previous is None else _merge(halts, written, previous)
            x = h if dense else torch.where(active[..., None], h, x)
            active = active & ~halts
        return LoopResult(
            y=y,
            states=tuple(current),
            n_iters=n_iters,
            ponder=depth,
            per_pass=torch.stack(per_pass),
        )


def _apply(  # noqa: PLR0913
    column: ColumnLike,
    h: Tensor,
    pos: Tensor,
    state: ColumnState,
    t: int,
    modulation: Modulation | None,
    carry: Carry | None,
    *,
    recompute: bool,
) -> tuple[Tensor, ColumnState, Projections]:
    """One column application, with its activations kept or recomputed for the backward pass."""
    if not recompute:
        return column.step(h, pos, state, t, modulation=modulation, carry=carry)
    # non-reentrant: arbitrary (non-tensor) arguments and results, autocast state restored for
    # the recomputation; the column's forward is deterministic, so the recomputed activations
    # are the ones the saved ones would have been
    return checkpoint(
        column.step, h, pos, state, t, modulation=modulation, carry=carry, use_reentrant=False
    )


def _as_module(obj: object) -> nn.Module | None:
    """The object as a torch module when it is one, so its parameters register; else None.

    Spelled with an attribute check and a cast because the strict checker types the class
    object ``nn.Module`` itself as an untyped constructor.
    """
    return cast("nn.Module", obj) if hasattr(obj, "register_module") else None


def _merge(halts: Tensor, fresh: Projections, previous: Projections) -> Projections:
    """Tokens halting now freeze their fresh projections; everyone else keeps the previous ones."""

    def pick(a: Tensor, b: Tensor) -> Tensor:
        mask = halts.reshape(*halts.shape, *([1] * (a.dim() - halts.dim())))
        return torch.where(mask, a, b)

    return Projections(
        rec_z=pick(fresh.rec_z, previous.rec_z),
        rec_k=pick(fresh.rec_k, previous.rec_k),
        rec_v=pick(fresh.rec_v, previous.rec_v),
        rec_beta=pick(fresh.rec_beta, previous.rec_beta),
        attn_k=pick(fresh.attn_k, previous.attn_k),
        attn_v=pick(fresh.attn_v, previous.attn_v),
    )


class SampledRule:
    """Real, sampled stops with a continuation floor: the exploration policy of training.

    After every non-final pass each deciding token stops with probability
    ``s = (1 - epsilon) * sigmoid(head.logit(h, t) + hazard_baseline(runs)[t])``
    (``vesper.exits.floored_stop_log_probs``), decided by ``uniforms[t] < s`` from uniforms
    the caller draws: ``float64 [runs - 1, b, n]``, one per non-final pass and token, so a
    forward's decisions are a function of the uniforms alone and a chunked or token-by-token
    replay given the matching slices takes the same decisions. A stopped token emits that
    pass's own output and its write is copied forward (``Loop``); the final pass is a forced
    exit and decides nothing. Design: docs/reviews/2026-10-18-halting-exploration-calibration.md.

    After the forward, ``log_prob [b, n]`` (float64) holds each token's summed
    log-probability of the decisions it actually made, with the head's graph: the
    score-function estimator's term. Tokens outside ``deciding`` (padding) and decisions of
    inactive tokens contribute exactly zero; ``decisions [b, n]`` counts the decisions made.
    Not the deployed policy: a floor of zero still samples (``CdfRule`` is the deterministic
    one). One rule object serves one forward (``begin`` resets its record).
    """

    def __init__(
        self,
        head: HaltingHead,
        epsilon: float | Tensor,
        uniforms: Tensor,
        deciding: Tensor,
    ) -> None:
        """Keep the policy's parts; nothing is checked until ``begin`` knows the forward."""
        self.head: HaltingHead = head
        self.epsilon: float | Tensor = epsilon
        self.uniforms: Tensor = uniforms
        self.deciding: Tensor = deciding
        self.log_prob: Tensor | None = None
        self.decisions: Tensor | None = None

    def begin(self, active: Tensor, dtype: torch.dtype, runs: int) -> ExitStep:
        """Check the uniforms and the deciding mask against the forward; reset the record.

        Raises:
            ContractError: on a deciding mask that is not ``bool[b, n]`` like the forward's
                tokens, or uniforms that are not float64 ``[runs - 1, b, n]`` in ``[0, 1)``.
        """
        dims: Dims = {}
        check_shape("active", active, ("b", "n"), dims, dtype=torch.bool)
        check_shape("deciding", self.deciding, ("b", "n"), dims, dtype=torch.bool)
        check_shape("uniforms", self.uniforms, (runs - 1, "b", "n"), dims, dtype=torch.float64)
        if self.uniforms.numel():  # a device contract: no read-back while a graph is captured
            require(
                ((self.uniforms >= 0.0) & (self.uniforms < 1.0)).all(),
                "uniforms must lie in [0, 1)",
            )
        baseline = hazard_baseline(runs, active.device)
        uniforms = self.uniforms.to(active.device)
        deciding = self.deciding.to(active.device)
        zero = torch.zeros(active.shape, dtype=torch.float64, device=active.device)
        log_prob = zero
        decisions = torch.zeros(active.shape, dtype=torch.int32, device=active.device)
        self.log_prob, self.decisions = log_prob, decisions

        def step(t: int, h: Tensor, active: Tensor, *, last: bool) -> tuple[Tensor, Tensor]:
            nonlocal log_prob, decisions
            if last:  # the forced exit: no decision, no log-probability
                halts = active
                return halts, halts.to(dtype)
            z = self.head.logit(h, t).to(torch.float64) + baseline[t]
            log_stop, log_continue = floored_stop_log_probs(z, self.epsilon)
            deciders = active & deciding
            stop = active & (uniforms[t] < log_stop.detach().exp())
            # where, never a product with a mask: an undecided token's -inf (epsilon = 1)
            # reaches neither the sum nor its gradient
            taken = torch.where(stop, log_stop, log_continue)
            log_prob = log_prob + torch.where(deciders, taken, zero)
            decisions = decisions + deciders.to(torch.int32)
            self.log_prob, self.decisions = log_prob, decisions
            return stop, stop.to(dtype)

        return step
