"""The exit distribution of all-pass supervision (the halting experiment plan, 2026-10-15).

Plan: docs/reviews/2026-10-15-halting-experiment-plan.md. For the halting experiment's
conditions B-D every pass is supervised and the halting head gives each token a distribution
over exit passes instead of ACT's accumulated probabilities: hazards
``lambda_t = sigmoid(z_t)`` for ``t < T`` and a forced exit at ``T``, so
``p_t = lambda_t prod_{j<t} (1 - lambda_j)`` and ``p_T = prod_{j<T} (1 - lambda_j)``. Everything
is computed in log space through log-sigmoid and log-survival sums, never by subtracting rounded
probabilities from one, so extreme logits stay finite forward and backward. Half-precision
logits are promoted to float32 first: a float16 log-survival sum overflows at logits a float16
can hold (maintainer B's review of 49c92cf, finding 6).

``hazard_baseline`` is the fixed hazard offset that makes zero learned logits exactly uniform
(maintainer B's review, section 4): no learned per-pass biases. The two KL functions are the
review's section 1 distinction: the per-token (local) KL to the uniform prior also taxes the
dependence between token and exit depth, the batch-mean (aggregate) KL only the marginal. The caller
owns the population: both reduce over exactly the positions of the mask they are given, so an
aggregate KL over one micro-group is that group's KL, not the full batch's (review, section 2).
Every reduction selects the masked positions BEFORE any arithmetic, so a non-finite value at a
masked-out position reaches neither the value nor a gradient (finding 1).

The CDF exit policy has two callers that must agree to the bit: the dense ``hazard_cdf_exits``
over a whole forward's hazard logits and the live ``vesper.loop.CdfRule`` deciding pass by pass.
Both accumulate through ``cdf_advance`` (float64, the same operations in the same order) and
compare through ``cdf_reaches``; a vectorized cumulative sum in one and an incremental sum in
the other disagreed at boundaries (finding 4: 127 cases over T = 2..32).

PRIMITIVE ``exit_log_probs``: the one place hazard logits become exit log-probabilities.
Use it for: every condition's exit distribution in training. Not for: ACT's accumulated-
probability rule, which ``vesper.loop`` keeps as it was, or the exit decision (``cdf_advance``).
"""

import math

import torch
from torch import Tensor
from torch.nn import functional

from vesper.contracts import ContractError, Dims, check_shape, require


def hazard_baseline(t_max: int, device: torch.device | None = None) -> Tensor:
    """``[t_max]`` hazard offsets (float64): ``b0_t = -log(T - t)`` for ``t = 1..T-1``, then 0.

    Built on ``device`` (CPU when None), never copied there: a host-to-device copy inside a
    captured CUDA graph would end the capture.

    With zero learned logits the hazard at pass ``t`` is ``1 / (T - t + 1)``, which makes every
    exit probability exactly ``1 / T``; the last entry is a zero for the forced final exit,
    whose hazard is never read, so the baseline lines up with a ``[T, ...]`` logit stack.

    Raises:
        ValueError: if ``t_max`` is below 1.
    """
    if t_max < 1:
        msg = f"t_max must be at least 1, got {t_max}"
        raise ValueError(msg)
    passes = torch.arange(1, t_max, dtype=torch.float64, device=device)
    zero = torch.zeros(1, dtype=torch.float64, device=device)
    return torch.cat([-torch.log(t_max - passes), zero])


def hazard_logits(learned: Tensor) -> Tensor:
    """Learned logits ``[T, ...]`` plus ``hazard_baseline(T)``, in float64.

    PRIMITIVE: the one construction of the hazard logits from the head's output, for the
    training objective and the dense exit trace alike; ``vesper.loop.CdfRule`` adds the same
    float64 baseline pass by pass, the same operation on the same operands. float64 because a
    float32 baseline alone moves the uniform CDF by about 1e-8, off its exact quantiles.

    Raises:
        ContractError: on a non-floating ``learned`` or no pass.
    """
    _floating("learned", learned)
    t_max = learned.shape[0] if learned.dim() > 0 else 0
    if t_max < 1:
        msg = "hazard_logits: no pass"
        raise ContractError(msg)
    shape = (t_max,) + (1,) * (learned.dim() - 1)
    baseline = hazard_baseline(t_max, learned.device).reshape(shape)
    return learned.to(torch.float64) + baseline


def exit_log_probs(z: Tensor) -> Tensor:
    """Hazard logits ``[T, ...]`` to exit log-probabilities ``[T, ...]``; ``z[T-1]`` is unused.

    ``log p_t = logsigmoid(z_t) + sum_{j<t} logsigmoid(-z_j)`` for ``t < T`` and
    ``log p_T = sum_{j<T} logsigmoid(-z_j)``: the forced final exit takes all remaining survival.
    float16 and bfloat16 are computed and returned in float32. Hazard logits saturate at
    ``+-HAZARD_LIMIT``, so every finite logit of any dtype gives finite log-probabilities; the
    probabilities are exact, the log-probabilities past the limit are the saturated logit's
    (``HAZARD_LIMIT``'s note), and an exit whose probability underflows to zero contributes
    exactly zero to the KL and its gradient.

    Raises:
        ContractError: on a non-floating ``z`` or no pass.
    """
    _floating("z", z)
    t_max = z.shape[0] if z.dim() > 0 else 0
    if t_max < 1:
        msg = "exit_log_probs: no pass"
        raise ContractError(msg)
    z = z.to(_promoted(z.dtype))
    if t_max == 1:
        return torch.zeros_like(z)
    hazards = _saturated(z[: t_max - 1])
    survival = torch.cumsum(functional.logsigmoid(-hazards), dim=0)  # log S after pass t
    before = torch.cat([torch.zeros_like(survival[:1]), survival[:-1]], dim=0)
    return torch.cat([functional.logsigmoid(hazards) + before, survival[-1:]], dim=0)


def expected_pass_loss(log_p: Tensor, losses: Tensor, mask: Tensor) -> Tensor:
    """Mean over the masked positions of ``sum_t p_t loss_t``: ``[T, b, n]`` twice, ``bool[b, n]``.

    Raises:
        ContractError: on mismatched shapes or dtypes, or if no position is masked in.
    """
    _check(log_p, mask, "expected_pass_loss", losses)
    # static shapes (a captured step cannot select by a mask): every position is computed, with
    # the masked-out inputs replaced by zero BEFORE any arithmetic and the outputs after it
    # (where, never a multiplication by zero), so a non-finite value there reaches neither the
    # value nor a gradient, as with the selection this replaces
    lp, ls = torch.where(mask, log_p, 0.0), torch.where(mask, losses, 0.0)
    per_token = torch.where(mask, (lp.exp() * ls).sum(0), 0.0)
    return per_token.sum() / mask.sum()


def kl_uniform_local(log_p: Tensor, mask: Tensor) -> Tensor:
    """Mean over the masked positions of ``KL(p_i || uniform)``, the per-token regularizer.

    Raises:
        ContractError: on a mismatched shape or dtype, or if no position is masked in.
    """
    _check(log_p, mask, "kl_uniform_local")
    t_max = log_p.shape[0]
    lp = torch.where(mask, log_p, 0.0)  # inputs, then outputs (expected_pass_loss)
    per_token = torch.where(mask, (lp.exp() * (lp + math.log(t_max))).sum(0), 0.0)
    return per_token.sum() / mask.sum()


def kl_uniform_aggregate(log_p: Tensor, mask: Tensor) -> Tensor:
    """``KL(p_bar || uniform)`` of the mean exit distribution over exactly the masked positions.

    Raises:
        ContractError: on a mismatched shape or dtype, or if no position is masked in.
    """
    _check(log_p, mask, "kl_uniform_aggregate")
    t_max = log_p.shape[0]
    # masked-out positions enter the log-sum-exp as -inf: exp(-inf) adds nothing and their
    # gradient is exactly zero; at least one position is masked in (_check)
    masked = torch.where(mask, log_p, float("-inf")).flatten(1)  # [T, b * n]
    count = mask.sum().to(log_p.dtype)
    log_mean = torch.logsumexp(masked, dim=1) - torch.log(count)
    return (log_mean.exp() * (log_mean + math.log(t_max))).sum()


# Below this, a cumulative exit probability counts as having reached its quantile. The fixed
# uniform distribution's CDF is exactly k / T after pass k, so condition B's every quantile
# k / T (the reference q = 0.5 among them) lies on a boundary, and any rounded sum lands a few
# ulps either side of it (measured before the slack: float32 at T = 3, 5, 6, 7 and float64 at
# T = 4 exited one pass late). Logits and sum are float64 (``hazard_logits``, ``cdf_advance``),
# whose rounding over any plausible T is below 1e-13; 1e-9 holds the boundary with margin and
# is far below any q a caller distinguishes (a float32 q differs from its double by ~1e-8).
CDF_SLACK = 1e-9


def cdf_reaches(cdf: Tensor, q: float) -> Tensor:
    """``bool``: the cumulative exit probability has reached ``q`` (up to ``CDF_SLACK``).

    PRIMITIVE: the one comparison of every CDF exit, the dense ``hazard_cdf_exits`` and the live
    ``vesper.loop.CdfRule``. Not for: probabilities that are not a CDF of exits.
    """
    return cdf >= q - CDF_SLACK


def cdf_begin(like: Tensor) -> tuple[Tensor, Tensor]:
    """The accumulation's start for tokens shaped like ``like``: log-survival 0 and CDF 0 (float64).

    PRIMITIVE, with ``cdf_advance``: the one accumulation of the exit CDF, shared by the dense
    and the live policy so they agree to the bit.
    """
    zero = torch.zeros(like.shape, dtype=torch.float64, device=like.device)
    return zero, zero.clone()


def cdf_advance(z_t: Tensor, survival: Tensor, cdf: Tensor) -> tuple[Tensor, Tensor]:
    """One pass of the accumulation: the exit mass ``p_t`` added, the survival reduced (float64).

    Returns ``(survival, cdf)`` after the pass with hazard logit ``z_t``; called for the passes
    ``t < T``, the forced final exit takes the rest.
    """
    z = _saturated(z_t.to(torch.float64))
    cdf = cdf + (functional.logsigmoid(z) + survival).exp()
    return survival + functional.logsigmoid(-z), cdf


def hazard_cdf_exits(z: Tensor, q: float) -> Tensor:
    """``int32[...]``: the first pass (1-based) whose cumulative exit probability reaches ``q``.

    ``z [T, ...]`` are the hazard logits (``hazard_logits``), accumulated by ``cdf_advance`` and
    compared by ``cdf_reaches`` exactly as ``vesper.loop.CdfRule`` does pass by pass; the
    forced final exit is ``T``. Hazard LOGITS, not exit log-probabilities: both are floating
    ``[T, ...]``, so the name carries the difference (this was ``cdf_exits(log_p, q)``; a
    caller still passing log-probabilities now fails at the import instead of exiting wrongly).

    Raises:
        ValueError: if ``q`` lies outside ``(0, 1]``.
        ContractError: on a non-floating ``z`` or no pass.
    """
    if not 0.0 < q <= 1.0:
        msg = f"q must lie in (0, 1], got {q}"
        raise ValueError(msg)
    _floating("z", z)
    t_max = z.shape[0] if z.dim() > 0 else 0
    if t_max < 1:
        msg = "hazard_cdf_exits: no pass"
        raise ContractError(msg)
    survival, cdf = cdf_begin(z[0])
    exits = torch.full(z.shape[1:], t_max, dtype=torch.int32, device=z.device)
    for t in range(t_max - 1):
        survival, cdf = cdf_advance(z[t], survival, cdf)
        first = cdf_reaches(cdf, q) & (exits == t_max)
        exits = torch.where(first, torch.full_like(exits, t + 1), exits)
    return exits


# The magnitude at which a hazard logit saturates: a SATURATING contract. Beyond it the hazard
# is 0 or 1 exactly in float64 (sigmoid(-1e4) underflows), so every exit probability, and the
# value and gradient of anything computed from probabilities (the expected per-pass loss, both
# KLs, the CDF exits), is unchanged. The returned LOG-probabilities are not: they are those of
# the saturated logit, and the log-survival's own derivative, about -1 at a large positive
# hazard, is zero past the limit (maintainer B, channel 89a2af9). Unclamped, eight float32 logits of
# 3e38 overflowed the log-survival sum to -inf and the KL went NaN (review of 230a5e5, gap a).
HAZARD_LIMIT = 1e4


def _saturated(z: Tensor) -> Tensor:
    """Hazard logits clamped to ``+-HAZARD_LIMIT`` (exits' one saturation, dense and live)."""
    return z.clamp(-HAZARD_LIMIT, HAZARD_LIMIT)


def _promoted(dtype: torch.dtype) -> torch.dtype:
    """float16 and bfloat16 to float32; wider floats as they are."""
    return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype


def _floating(name: str, tensor: Tensor) -> None:
    if not tensor.dtype.is_floating_point:
        msg = f"{name}: a floating tensor is required, got {tensor.dtype}"
        raise ContractError(msg)


def _check(log_p: Tensor, mask: Tensor, what: str, losses: Tensor | None = None) -> None:
    """``log_p [T, b, n]`` floating, ``losses`` the same shape and dtype, ``mask bool[b, n]``."""
    dims: Dims = {}
    check_shape("log_p", log_p, ("t", "b", "n"), dims)
    _floating("log_p", log_p)
    if dims["t"] < 1:
        msg = f"{what}: no pass"
        raise ContractError(msg)
    check_shape("mask", mask, ("b", "n"), dims, dtype=torch.bool)
    if losses is not None:
        check_shape("losses", losses, ("t", "b", "n"), dims, dtype=log_p.dtype)
    require(mask.any(), f"{what}: no position in the mask")


def floored_stop_log_probs(z: Tensor, epsilon: float | Tensor) -> tuple[Tensor, Tensor]:
    """``(log s, log (1 - s))`` for a stop probability ``s = (1 - epsilon) * sigmoid(z)``.

    PRIMITIVE: the one place a sampled stop's probabilities are formed, for the sampled exit
    rule (``vesper.loop.SampledRule``) and every estimator built on its log-probabilities
    (docs/reviews/2026-10-18-halting-exploration-calibration.md). The floor ``epsilon`` is a
    MIXTURE that guarantees continuation with probability at least ``epsilon``; unlike
    ``max(epsilon, 1 - sigmoid(z))`` it has no region where the head's derivative is zero.
    ``z`` are hazard logits (``hazard_logits``); both outputs are float64, from the saturated
    logit (``HAZARD_LIMIT``), and finite for every finite ``z`` and every ``epsilon`` in
    ``[0, 1]`` except ``log s`` at ``epsilon = 1``, which is ``-inf`` (a stop that cannot be
    sampled); its gradient with respect to ``z`` stays finite. Not for: the deterministic CDF
    exit (``cdf_advance``) or the dense exit distribution (``exit_log_probs``).

    A float ``epsilon`` is checked on the host; a tensor (float64 on ``z``'s device: a captured
    update reads a floor that changes between replays from it) through ``require``, so it never
    reads the device back while a graph is captured.

    Raises:
        ValueError: if a float ``epsilon`` lies outside ``[0, 1]``.
        ContractError: if a tensor ``epsilon`` lies outside ``[0, 1]`` (a device assert under
            capture), or on a non-floating ``z``.
    """
    _floating("z", z)
    if isinstance(epsilon, float | int):
        if not 0.0 <= epsilon <= 1.0:
            msg = f"epsilon must lie in [0, 1], got {epsilon}"
            raise ValueError(msg)
        floor = torch.tensor(float(epsilon), dtype=torch.float64, device=z.device)
    else:
        floor = epsilon
        if floor.dtype != torch.float64 or floor.device != z.device:
            msg = f"epsilon: a float64 tensor on {z.device}, got {floor.dtype} on {floor.device}"
            raise ContractError(msg)
        inside = ((floor >= 0.0) & (floor <= 1.0)).all()
        require(inside, "epsilon must lie in [0, 1] (a device floor)")
    logit = _saturated(z.to(torch.float64))
    log_stop_hazard = functional.logsigmoid(logit)
    log_stop = torch.log1p(-floor) + log_stop_hazard  # log(1 - eps) + log sigmoid(z)
    # 1 - s = sigmoid(-z) + eps * sigmoid(z), summed in log space
    log_continue = torch.logaddexp(
        functional.logsigmoid(-logit), torch.log(floor) + log_stop_hazard
    )
    return log_stop, log_continue
