"""Recurrent core: the gated delta-rule fast-weight state (spec 3a), reference implementation.

Pure PyTorch and deliberately simple: the Triton kernels are tested against this file, never
the other way around. Per head the state is a matrix ``H`` of shape ``[dk, dk]`` read as
``H^T q``. Per token::

    H_t = (I - beta_t k_t k_t^T) H_{t-1} + beta_t k_t v_t^T        o_t = H_t^T q_t

which is the delta rule: read what is stored under ``k`` (``H^T k``), replace it by ``v`` with
strength ``beta``. ``k`` must have L2 norm at most one (``normalize_keys``) and ``beta`` in
``[0, 1]`` (``write_strength``); then the erase factor has eigenvalues in ``[0, 1]`` and the
state is stable. Floating-point sigmoid can reach either endpoint.

Layouts: ``q, k, v: [b, n, h, dk]``, ``beta: [b, n, h]``, ``state: [b, h, dk, dk]``. The blocked
form (``delta_scan_chunked``) uses the WY representation of the product of erase factors within a
block of ``block`` tokens. With ``K, V`` the block's keys and values as rows, let
``A = tril(diag(beta) K K^T, -1)`` and ``T = (I + A)^-1 diag(beta)``; then ``W = T K`` and
``U = T V`` satisfy ``w_i = beta_i (k_i - sum_{j<i} (k_i . k_j) w_j)`` (and ``u_i`` likewise), and
``H_end = H_0 + K^T (U - W H_0)``, ``O = Q H_0 + tril(Q K^T) (U - W H_0)``. ``I + A`` is unit
lower triangular, so its explicit inverse is cheap and well conditioned at block size 64.
"""

import torch
from torch import Tensor

from vesper.contracts import Dims, check_shape

KEY_NORM_FLOOR = 1e-6  # normalize_keys' denominator floor; kernels.keys uses the same one


def write_strength(logits: Tensor, m: Tensor | None = None, alpha: float | None = None) -> Tensor:
    """Turn the write-strength logits into ``beta`` in [0, 1], modulated when ``m`` is given.

    Without ``m`` this is ``sigmoid(logits)``. With the modulator ``m`` (spec, Learning: one
    scalar per token, clipped to ``[m_clip_low, m_clip_high]``) the write odds are scaled by
    ``1 + alpha m``::

        beta_eff = sigmoid(logits + ln(1 + alpha m)) = beta (1 + alpha m) / (1 + alpha beta m)

    This equals the plain product ``beta (1 + alpha m)`` to first order in ``beta`` and never
    leaves ``[0, 1]``, so the scan's contraction argument survives modulation; the plain product
    reaches 2.25 at ``beta = 0.9, m = 3`` (docs/reviews/2026-10-14-pass-state-review.md, E). It
    is a different interior mapping, not a clamp: at ``beta = 1/2, alpha = 1/2, m = 1`` the
    product gives 3/4 and the odds rule 3/5. A zero multiplier (``alpha = 1, m = -1``) is an
    exact zero write with a zero gradient; a negative multiplier is rejected here and in
    ``config.validate`` (``1 + alpha m_clip_low >= 0``). In exact arithmetic sigmoid lies in
    (0, 1); floating-point results can saturate.

    Args:
        logits: ``[b, n, h]`` from ``W_beta x`` (any shape when ``m`` is None).
        m: optional ``[b, n]`` modulator, broadcast over heads; ``1 + alpha m`` must not be
            negative.
        alpha: modulation strength, required with ``m``, finite and non-negative
            (``configs/*.yaml: learning.alpha``, which ``config.validate`` holds to the same
            contract); a negative ``alpha`` would write surprising tokens more weakly, the
            opposite of the spec's direction, and is refused rather than interpreted.

    Returns:
        ``beta`` of the same shape as ``logits``.

    Raises:
        ContractError: if ``m`` does not match the first two axes of ``logits``.
        ValueError: if ``m`` comes without ``alpha``, if ``alpha`` is negative or not finite, or
            if ``1 + alpha m`` is negative somewhere, i.e. ``m`` below its clip.
    """
    if m is None:
        return torch.sigmoid(logits)
    if alpha is None:
        msg = "alpha must be given with a modulator (configs/*.yaml: learning.alpha)"
        raise ValueError(msg)
    if not 0.0 <= alpha < float("inf"):
        msg = f"alpha must be finite and non-negative, got {alpha}"
        raise ValueError(msg)
    dims: Dims = {}
    check_shape("logits", logits, ("b", "n", "h"), dims)
    check_shape("m", m, ("b", "n"), dims)
    factor = 1.0 + alpha * m
    if bool((factor < 0).any().item()):
        msg = f"1 + alpha m must not be negative; alpha {alpha}, smallest m {m.min().item()}"
        raise ValueError(msg)
    # The log of a floored factor keeps infinities out of the backward pass; the where makes a
    # zero factor an exact zero write (and a zero gradient) instead of sigmoid(logits - large).
    shift = torch.log(factor.clamp_min(torch.finfo(factor.dtype).tiny))[..., None]
    beta = torch.sigmoid(logits + shift)
    return torch.where((factor > 0)[..., None], beta, torch.zeros_like(beta))


def normalize_keys(k: Tensor) -> Tensor:
    """L2-normalize finite keys with a denominator floor; a zero key stays zero.

    Args:
        k: ``[..., dk]``.

    Returns:
        ``k / max(||k||_2, 1e-6)`` with the same shape. Rows whose norm is below
        the floor remain subunit. Rescaling avoids overflow and underflow while
        computing the norm; derivatives still obey the input dtype's range.
    """
    if k.numel() == 0:
        return k
    # This positive scale cancels algebraically. Treating it as constant keeps
    # an unnecessary, numerically cancelling derivative out of autograd.
    scale = k.detach().abs().amax(dim=-1, keepdim=True).clamp_min(KEY_NORM_FLOOR)
    scaled = k / scale
    # If scale exceeds the floor, one scaled component has magnitude one;
    # otherwise scale is the floor. Thus max(||scaled||, 1) is the right
    # denominator in both cases. Clamp before sqrt so its derivative is finite
    # at zero. Accumulate wide, low-precision rows without overflowing the sum.
    accumulator = torch.float64 if scaled.dtype == torch.float64 else torch.float32
    squared_norm = scaled.square().sum(dim=-1, keepdim=True, dtype=accumulator)
    denominator = squared_norm.clamp_min(1.0).sqrt().to(dtype=scaled.dtype)
    return scaled / denominator


def delta_update(k: Tensor, v: Tensor, beta: Tensor, state: Tensor) -> Tensor:
    """Write one token into the state without reading it: the copy-forward path.

    ``state' = (I - beta k k^T) state + beta k v^T``. ``delta_scan_step`` is this plus the read,
    so a halted token whose saved ``(k, v, beta)`` is carried into a later pass updates that
    pass's state through the same equation as an active token (pass-state review, D).

    Args:
        k: ``[b, h, dk]`` key, L2 norm at most one.
        v: ``[b, h, dk]`` value.
        beta: ``[b, h]`` write strength in [0, 1].
        state: ``[b, h, dk, dk]`` the state before this token.

    Returns:
        The updated state, a new tensor; ``state`` is not modified.
    """
    dims: Dims = {}
    check_shape("k", k, ("b", "h", "dk"), dims)
    check_shape("v", v, ("b", "h", "dk"), dims)
    check_shape("beta", beta, ("b", "h"), dims)
    check_shape("state", state, ("b", "h", "dk", "dk"), dims)
    old = torch.einsum("bhij,bhi->bhj", state, k)
    return state + beta[..., None, None] * k[..., :, None] * (v - old)[..., None, :]


def delta_scan_step(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply one token to the state and read it (the inference form).

    Args:
        q: ``[b, h, dk]`` query.
        k: ``[b, h, dk]`` key, L2 norm at most one.
        v: ``[b, h, dk]`` value.
        beta: ``[b, h]`` write strength in [0, 1].
        state: ``[b, h, dk, dk]`` the state before this token.

    Returns:
        ``(o, state')`` with ``o: [b, h, dk]`` the read ``state'^T q`` and the updated state.
    """
    dims: Dims = {}
    check_shape("q", q, ("b", "h", "dk"), dims)
    check_shape("state", state, ("b", "h", "dk", "dk"), dims)
    new_state = delta_update(k, v, beta, state)
    o = torch.einsum("bhij,bhi->bhj", new_state, q)
    return o, new_state


def delta_scan_chunked(
    q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor, block: int
) -> tuple[Tensor, Tensor]:
    """Apply ``n`` tokens to the state in blocks (the training form).

    Equals ``delta_scan_step`` applied ``n`` times, to 1e-4 in float32 (tests/test_recurrence.py).

    Args:
        q: ``[b, n, h, dk]`` queries.
        k: ``[b, n, h, dk]`` keys, L2 norm at most one.
        v: ``[b, n, h, dk]`` values.
        beta: ``[b, n, h]`` write strengths in [0, 1].
        state: ``[b, h, dk, dk]`` the state before the first token.
        block: tokens per parallel block, at least 1 (``configs/*.yaml: recurrence.block``).

    Returns:
        ``(o, state_n)`` with ``o: [b, n, h, dk]`` and the state after the last token.
        With no tokens, return the empty query tensor and unchanged state.

    Raises:
        ValueError: if ``block`` is below 1.
    """
    dims: Dims = {}
    check_shape("q", q, ("b", "n", "h", "dk"), dims)
    check_shape("k", k, ("b", "n", "h", "dk"), dims)
    check_shape("v", v, ("b", "n", "h", "dk"), dims)
    check_shape("beta", beta, ("b", "n", "h"), dims)
    check_shape("state", state, ("b", "h", "dk", "dk"), dims)
    if block < 1:
        msg = f"block must be at least 1, got {block}"
        raise ValueError(msg)
    n = dims["n"]
    if n == 0:
        return q, state
    h_state = state
    outs: list[Tensor] = []
    for start in range(0, n, block):
        end = min(start + block, n)
        qb = q[:, start:end].transpose(1, 2)
        kb = k[:, start:end].transpose(1, 2)
        vb = v[:, start:end].transpose(1, 2)
        bb = beta[:, start:end].transpose(1, 2)
        eye = torch.eye(end - start, dtype=q.dtype, device=q.device)
        lower = torch.tril(bb[..., :, None] * (kb @ kb.transpose(-1, -2)), diagonal=-1)
        t_mat = torch.inverse(eye + lower) @ torch.diag_embed(bb)
        w_rows = t_mat @ kb
        u_rows = t_mat @ vb
        delta = u_rows - w_rows @ h_state
        causal_qk = torch.tril(qb @ kb.transpose(-1, -2))
        outs.append((qb @ h_state + causal_qk @ delta).transpose(1, 2))
        h_state = h_state + kb.transpose(-1, -2) @ delta
    return torch.cat(outs, dim=1), h_state
