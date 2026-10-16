import pytest
import torch
from torch import Tensor

from vesper.contracts import ContractError
from vesper.recurrence import delta_scan_chunked, delta_scan_step, normalize_keys, write_strength

Inputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def _inputs(b: int, n: int, h: int, dk: int, dtype: torch.dtype, seed: int = 0) -> Inputs:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, n, h, dk, generator=g, dtype=dtype)
    k = normalize_keys(torch.randn(b, n, h, dk, generator=g, dtype=dtype))
    v = torch.randn(b, n, h, dk, generator=g, dtype=dtype)
    beta = write_strength(torch.randn(b, n, h, generator=g, dtype=dtype))
    h0 = 0.1 * torch.randn(b, h, dk, dk, generator=g, dtype=dtype)
    return q, k, v, beta, h0


def _stepwise(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, h0: Tensor) -> tuple[Tensor, Tensor]:
    outs: list[Tensor] = []
    state = h0
    for t in range(q.shape[1]):
        o, state = delta_scan_step(q[:, t], k[:, t], v[:, t], beta[:, t], state)
        outs.append(o)
    return torch.stack(outs, dim=1), state


@pytest.mark.parametrize(("dtype", "tol"), [(torch.float64, 1e-10), (torch.float32, 1e-4)])
def test_chunked_equals_step_applied_n_times(dtype: torch.dtype, tol: float) -> None:
    # 130 tokens with block 64: two full blocks and a partial one.
    q, k, v, beta, h0 = _inputs(2, 130, 3, 16, dtype)
    o_ref, h_ref = _stepwise(q, k, v, beta, h0)
    o, h_n = delta_scan_chunked(q, k, v, beta, h0, block=64)
    assert (o - o_ref).abs().max().item() < tol
    assert (h_n - h_ref).abs().max().item() < tol


def test_block_one_is_the_step_form() -> None:
    q, k, v, beta, h0 = _inputs(1, 9, 2, 8, torch.float64)
    o_ref, h_ref = _stepwise(q, k, v, beta, h0)
    o, h_n = delta_scan_chunked(q, k, v, beta, h0, block=1)
    assert (o - o_ref).abs().max().item() < 1e-12
    assert (h_n - h_ref).abs().max().item() < 1e-12


def test_delta_rule_overwrites_exactly() -> None:
    # With beta = 1, reading k right after writing (k, v) returns v: the defining property.
    g = torch.Generator().manual_seed(1)
    state = torch.randn(1, 2, 8, 8, generator=g)
    k = normalize_keys(torch.randn(1, 2, 8, generator=g))
    v = torch.randn(1, 2, 8, generator=g)
    o, _ = delta_scan_step(k, k, v, torch.ones(1, 2), state)
    assert (o - v).abs().max().item() < 1e-5


def _reversed_erase_order(k: Tensor, v: Tensor, state: Tensor) -> Tensor:
    # The formula as the spec had it before the 2026-10-14 review: H (I - k k^T) + k v^T.
    eye = torch.eye(k.shape[-1], dtype=state.dtype)
    return state @ (eye - torch.outer(k, k)) + torch.outer(k, v)


def test_reversed_erase_order_is_not_a_delta_rule() -> None:
    g = torch.Generator().manual_seed(2)
    state = torch.randn(8, 8, generator=g)
    k = normalize_keys(torch.randn(8, generator=g))
    v = torch.randn(8, generator=g)
    read = _reversed_erase_order(k, v, state).T @ k
    assert (read - v).abs().max().item() > 0.1


def test_gradients_match_between_forms() -> None:
    q, k, v, beta, h0 = _inputs(1, 20, 2, 8, torch.float64)
    lq, lk, lv, lb, lh = (t.clone().requires_grad_() for t in (q, k, v, beta, h0))
    leaves = [lq, lk, lv, lb, lh]
    o_a, h_a = _stepwise(lq, lk, lv, lb, lh)
    grads_a = torch.autograd.grad(o_a.square().sum() + h_a.square().sum(), leaves)
    o_b, h_b = delta_scan_chunked(lq, lk, lv, lb, lh, block=8)
    grads_b = torch.autograd.grad(o_b.square().sum() + h_b.square().sum(), leaves)
    for ga, gb in zip(grads_a, grads_b, strict=True):
        assert (ga - gb).abs().max().item() < 1e-8


def test_state_norm_is_bounded_by_the_writes() -> None:
    # ||(I - b k k^T) H|| <= ||H|| for unit k and b in [0, 1], so ||H_n|| <= ||H_0|| + sum b ||v||.
    q, k, v, beta, h0 = _inputs(1, 2000, 2, 16, torch.float32)
    _, h_n = delta_scan_chunked(q, k, v, beta, h0, block=64)
    v_norms = v.square().sum(dim=-1).sqrt()
    bound = h0.flatten(1).square().sum(dim=1).sqrt() + (beta * v_norms).sum(dim=(1, 2))
    assert torch.isfinite(h_n).all()
    assert (h_n.flatten(1).square().sum(dim=1).sqrt() <= bound + 1e-3).all()


def test_rejects_block_zero() -> None:
    q, k, v, beta, h0 = _inputs(1, 4, 1, 4, torch.float32)
    with pytest.raises(ValueError, match="block must be at least 1"):
        _ = delta_scan_chunked(q, k, v, beta, h0, block=0)


def test_rejects_mismatched_heads() -> None:
    q, k, v, beta, h0 = _inputs(1, 4, 2, 4, torch.float32)
    with pytest.raises(ContractError, match="'h' is 1"):
        _ = delta_scan_chunked(q, k, v, beta, h0[:, :1], block=2)
