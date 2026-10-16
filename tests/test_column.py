import math
from pathlib import Path

import pytest
import torch
from torch import Tensor

from vesper.attention import kv_cache_empty
from vesper.column import (
    Column,
    ColumnShape,
    ColumnState,
    Modulation,
    RMSNorm,
    column_shape,
    init_column_state,
)
from vesper.config import load
from vesper.contracts import ContractError
from vesper.recurrence import delta_scan_step, delta_update, normalize_keys

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
T = 2
CPU = torch.device("cpu")
F64 = torch.float64


def _norm(t: Tensor) -> Tensor:
    return t.square().sum().sqrt()


def _column(seed: int = 0) -> Column:
    return Column(SHAPE, torch.Generator().manual_seed(seed)).to(F64)


def _state(b: int, t_max: int = T) -> ColumnState:
    return init_column_state(SHAPE, b, t_max, F64, CPU)


def _inputs(b: int, n: int, start: int = 0, seed: int = 1) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, n, SHAPE.d, generator=g, dtype=F64)
    return x, torch.arange(start, start + n).expand(b, n)


def pass_major(
    col: Column, x0: Tensor, pos: Tensor, state: ColumnState, *, shared: bool
) -> tuple[Tensor, ColumnState]:
    """The training order: every pass over the whole chunk, then the next pass."""
    x = x0
    for t in range(T):
        x, state = col.forward(x, pos, state, 0 if shared else t)
    return x, state


def token_major(
    col: Column, x0: Tensor, pos: Tensor, state: ColumnState, *, shared: bool
) -> tuple[Tensor, ColumnState]:
    """The inference order: every pass of one token, then the next token."""
    outs: list[Tensor] = []
    for i in range(x0.shape[1]):
        x = x0[:, i : i + 1]
        for t in range(T):
            x, state = col.forward(x, pos[:, i : i + 1], state, 0 if shared else t)
        outs.append(x)
    return torch.cat(outs, dim=1), state


# --- shapes, purity, per-pass isolation -------------------------------------------------------


def test_shapes_purity_and_pass_isolation() -> None:
    col = _column()
    x, pos = _inputs(2, 6)
    state = _state(2)
    before = [h.clone() for h in state.h]
    y, new = col.forward(x, pos, state, 0)
    assert y.shape == (2, 6, SHAPE.d)
    assert y.dtype == F64
    for h_before, h_now in zip(before, state.h, strict=True):
        assert torch.equal(h_before, h_now)  # the input state is untouched
    assert not torch.equal(new.h[0], state.h[0])  # this pass's state moved
    assert new.kv[0].k.shape[1] == min(6, SHAPE.window)
    assert new.h[1] is state.h[1]  # the other pass's state is the same object
    assert new.kv[1] is state.kv[1]


def test_zero_output_projections_make_the_column_the_identity() -> None:
    col = _column()
    with torch.no_grad():
        _ = col.rec.w_o.zero_()
        _ = col.attn.w_o.zero_()
        _ = col.ffn.w2.zero_()
    x, pos = _inputs(1, 5)
    state = _state(1)
    y, new = col.forward(x, pos, state, 1)
    assert torch.equal(y, x)
    assert not torch.equal(new.h[1], state.h[1])  # the recurrence still wrote its state
    assert new.h[0] is state.h[0]


def test_rms_norm_normalizes_to_unit_power_up_to_its_floor() -> None:
    norm = RMSNorm(SHAPE.d).to(F64)
    x, _ = _inputs(1, 3)
    y = norm.forward(x)
    # mean(y^2) = m / (m + 1e-6) for m = mean(x^2): within 1e-5 of one at these magnitudes
    assert (y.square().mean(-1) - 1.0).abs().max().item() < 1e-5
    assert torch.allclose(norm.forward(7.0 * x), y, rtol=0, atol=1e-5)
    with torch.no_grad():
        _ = norm.weight.mul_(2.0)
    assert torch.allclose(norm.forward(x), 2.0 * y, rtol=0, atol=1e-12)


# --- the state-ordering claim -----------------------------------------------------------------


def test_pass_major_equals_token_major_with_per_pass_states() -> None:
    col = _column()
    x0, pos = _inputs(2, 7, start=3)
    y_pm, s_pm = pass_major(col, x0, pos, _state(2), shared=False)
    y_tm, s_tm = token_major(col, x0, pos, _state(2), shared=False)
    assert (y_pm - y_tm).abs().max().item() < 1e-10
    for a, b in zip(s_pm.h, s_tm.h, strict=True):
        assert (a - b).abs().max().item() < 1e-10
    for ca, cb in zip(s_pm.kv, s_tm.kv, strict=True):
        assert torch.equal(ca.pos, cb.pos)
        assert (ca.k - cb.k).abs().max().item() < 1e-10
        assert (ca.v - cb.v).abs().max().item() < 1e-10


def test_one_shared_state_across_passes_breaks_the_equality() -> None:
    # the counterexample of the pass-state review: with one H for both passes, the training
    # order lets pass 2 of token 1 read tokens 2..n from pass 1, so the two orders disagree
    col = _column()
    x0, pos = _inputs(2, 7, start=3)
    y_pm, _ = pass_major(col, x0, pos, _state(2), shared=True)
    y_tm, _ = token_major(col, x0, pos, _state(2), shared=True)
    assert (y_pm - y_tm).abs().max().item() > 1e-6


def test_chunk_boundary_carries_the_state() -> None:
    # two chunks in a row equal one chunk of their concatenation, per pass
    col = _column()
    x0, pos = _inputs(1, 9, start=0)
    whole, s_whole = pass_major(col, x0, pos, _state(1), shared=False)
    first, s_first = pass_major(col, x0[:, :4], pos[:, :4], _state(1), shared=False)
    second, s_second = pass_major(col, x0[:, 4:], pos[:, 4:], s_first, shared=False)
    assert (torch.cat([first, second], dim=1) - whole).abs().max().item() < 1e-10
    for a, b in zip(s_whole.h, s_second.h, strict=True):
        assert (a - b).abs().max().item() < 1e-10


# --- modulation, gradients, empty chunk, contracts -------------------------------------------


def test_modulated_writes_change_the_output_only_when_m_is_nonzero() -> None:
    col = _column()
    x, pos = _inputs(1, 5)
    state = _state(1)
    plain, _ = col.forward(x, pos, state, 0)
    zero_m, _ = col.forward(x, pos, state, 0, Modulation(torch.zeros(1, 5, dtype=F64), 0.5))
    surprised, _ = col.forward(
        x, pos, state, 0, Modulation(torch.full((1, 5), 3.0, dtype=F64), 0.5)
    )
    assert torch.allclose(zero_m, plain, rtol=0, atol=1e-12)
    assert (surprised - plain).abs().max().item() > 1e-6
    with pytest.raises(ValueError, match="alpha"):
        _ = col.forward(x, pos, state, 0, Modulation(torch.zeros(1, 5, dtype=F64), -0.5))


def test_gradient_reaches_every_parameter() -> None:
    col = _column()
    x, pos = _inputs(2, 6)
    y, _ = col.forward(x, pos, _state(2), 1)
    names = [name for name, _ in col.named_parameters()]
    grads = torch.autograd.grad(y.square().sum(), list(col.parameters()))
    # three gains; recurrent q k v o beta and the key filter's taps; attention; ffn
    assert len(names) == 3 + 6 + 4 + 2
    assert "rec.conv_k" in names
    for name, g in zip(names, grads, strict=True):
        assert g.abs().sum().item() > 0, name


def test_empty_chunk_returns_empty_output_and_the_same_state() -> None:
    col = _column()
    state = _state(2)
    y, new = col.forward(
        torch.zeros(2, 0, SHAPE.d, dtype=F64), torch.zeros(2, 0, dtype=torch.long), state, 0
    )
    assert y.shape == (2, 0, SHAPE.d)
    assert new.h[0] is state.h[0]
    assert new.kv[0] is state.kv[0]


def test_rejects_a_bad_pass_index_or_state() -> None:
    col = _column()
    x, pos = _inputs(1, 3)
    for bad in (-1, T):
        with pytest.raises(ValueError, match="pass_index"):
            _ = col.forward(x, pos, _state(1), bad)
    state = _state(1)
    lopsided = ColumnState(h=state.h, kv=state.kv[:1], key_history=state.key_history)
    with pytest.raises(ValueError, match="windows"):
        _ = col.forward(x, pos, lopsided, 0)
    with pytest.raises(ValueError, match="t_max"):
        _ = init_column_state(SHAPE, 1, 0, F64, CPU)


def test_delta_update_is_the_step_without_the_read() -> None:
    g = torch.Generator().manual_seed(9)
    q = torch.randn(2, 3, 4, dtype=F64, generator=g)
    k = normalize_keys(torch.randn(2, 3, 4, dtype=F64, generator=g))
    v = torch.randn(2, 3, 4, dtype=F64, generator=g)
    beta = torch.rand(2, 3, dtype=F64, generator=g)
    state = torch.randn(2, 3, 4, 4, dtype=F64, generator=g)
    kept = state.clone()
    updated = delta_update(k, v, beta, state)
    _, from_step = delta_scan_step(q, k, v, beta, state)
    assert torch.equal(updated, from_step)
    assert torch.equal(state, kept)
    # with beta = 1 and a unit key, reading k from the new state returns exactly v
    one = delta_update(k, v, torch.ones(2, 3, dtype=F64), state)
    read = (one * k[..., :, None]).sum(dim=-2)  # state'^T k
    assert torch.allclose(read, v, rtol=0, atol=1e-12)


def test_parameter_count_at_vesper_s_dimensions() -> None:
    # the baselines document's column is 8,762,112 with retrieval (196,608 + the gate 768 + its
    # norm 768); without the retrieval sublayer the column holds 8,563,968, and the key filter
    # (the project owner's decision of 2026-10-15) adds K x d = 4 x 768 taps
    shape = column_shape(load(Path(__file__).resolve().parents[1] / "configs" / "vesper-s.yaml"))
    assert shape == ColumnShape(
        d=768,
        heads=12,
        kv_groups=4,
        head_dim=64,
        window=512,
        rotary_base=10000.0,
        ffn_mult=4,
        block=64,
        key_conv=4,
    )
    total = sum(p.numel() for p in Column(shape).parameters())
    assert total == 8_567_040
    assert total == 8_762_112 - 196_608 - 768 - 768 + 4 * 768


# --- precision ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "magnitude"), [(torch.float32, 1e20), (torch.bfloat16, 1e20), (torch.float64, 1e200)]
)
def test_rms_norm_does_not_square_out_of_range(dtype: torch.dtype, magnitude: float) -> None:
    # a constant row at the top of the dtype's range normalizes to the gain, not to zero
    norm = RMSNorm(SHAPE.d).to(dtype)
    row = torch.full((1, 1, SHAPE.d), magnitude, dtype=dtype)
    assert torch.allclose(norm.forward(row), torch.ones_like(row), rtol=1e-2, atol=0)


def test_rms_norm_native_fp16_small_and_zero_rows() -> None:
    # 1e-3 is the scale floor: a constant row there gives 1e-3 / sqrt(1e-6 + 1e-6) = 0.7071, and
    # the zero row's gradient is 1 / sqrt(eps) = 1000; computed in half precision the reciprocal
    # of the floored scale squared overflowed and both came out as zero
    norm = RMSNorm(SHAPE.d).to(torch.float16)
    small = torch.full((1, 1, SHAPE.d), 1e-3, dtype=torch.float16)
    assert torch.allclose(
        norm.forward(small), torch.full_like(small, 0.70710678), rtol=2e-3, atol=0
    )
    zero = torch.zeros(1, 1, SHAPE.d, dtype=torch.float16, requires_grad=True)
    y = norm.forward(zero)
    assert torch.equal(y, torch.zeros_like(y))
    (g,) = torch.autograd.grad(y.sum(), [zero])
    assert torch.allclose(g, torch.full_like(g, 1000.0), rtol=1e-2, atol=0)


def test_rms_norm_returns_the_input_dtype_under_a_float32_gain() -> None:
    norm = RMSNorm(SHAPE.d)  # float32 gain, as under autocast with float32 parameters
    for dtype in (torch.float16, torch.bfloat16):
        x, _ = _inputs(1, 3)
        y = norm.forward(x.to(dtype))
        assert y.dtype == dtype
        reference = norm.forward(x.to(torch.float32))
        assert torch.allclose(y.float(), reference, rtol=2e-2, atol=1e-3)


def test_rms_norm_rejects_a_wrong_width() -> None:
    norm = RMSNorm(SHAPE.d).to(F64)
    with pytest.raises(ContractError, match="last axis"):
        _ = norm.forward(torch.ones(1, 2, 1, dtype=F64))
    with pytest.raises(ContractError, match="last axis"):
        _ = norm.forward(torch.ones(1, 2, SHAPE.d + 1, dtype=F64))


def test_rms_norm_eps_is_a_regularizer_not_a_floor() -> None:
    norm = RMSNorm(SHAPE.d).to(F64)
    small = torch.full((1, 1, SHAPE.d), 1e-4, dtype=F64)
    expected = 1e-4 / math.sqrt(1e-8 + 1e-6)  # the formula, 0.0995...
    assert abs(norm.forward(small)[0, 0, 0].item() - expected) < 1e-12
    zero = torch.zeros(1, 1, SHAPE.d, dtype=F64, requires_grad=True)
    y = norm.forward(zero)
    assert torch.equal(y, torch.zeros_like(y))
    (g,) = torch.autograd.grad(y.sum(), [zero])
    assert torch.allclose(g, torch.full_like(g, 1e3), rtol=1e-9, atol=0)  # 1 / sqrt(eps)


def test_rms_norm_matches_the_formula_and_its_gradient_in_the_interior() -> None:
    norm = RMSNorm(SHAPE.d).to(F64)
    x, _ = _inputs(2, 3)
    leaf = x.clone().requires_grad_()
    y = norm.forward(leaf)
    reference = leaf / torch.sqrt(leaf.square().mean(-1, keepdim=True) + 1e-6)
    assert torch.allclose(y, reference, rtol=1e-12, atol=0)
    # a loss with an O(1) gradient: the squared norm of a normalized row is nearly constant and
    # its gradient (1e-6) would only compare rounding noise
    direction = torch.randn(y.shape, generator=torch.Generator().manual_seed(5), dtype=F64)
    (g_ours,) = torch.autograd.grad((y * direction).sum(), [leaf], retain_graph=True)
    (g_ref,) = torch.autograd.grad((reference * direction).sum(), [leaf])
    assert torch.allclose(g_ours, g_ref, rtol=1e-10, atol=1e-14)


def test_native_bfloat16_column_scans_in_float32() -> None:
    col = Column(SHAPE, torch.Generator().manual_seed(0)).to(torch.bfloat16)
    x, pos = _inputs(2, 6)
    state = init_column_state(SHAPE, 2, T, torch.bfloat16, CPU)
    assert state.h[0].dtype == torch.float32  # the state never lives below float32
    assert state.kv[0].k.dtype == torch.bfloat16
    y, new = col.forward(x.to(torch.bfloat16), pos, state, 0)
    assert y.dtype == torch.bfloat16
    assert bool(torch.isfinite(y).all().item())
    assert new.h[0].dtype == torch.float32
    assert new.kv[0].k.dtype == torch.bfloat16


def test_cpu_autocast_bfloat16_matches_float32_within_half_precision() -> None:
    col = _column().to(torch.float32)
    x, pos = _inputs(2, 6)
    x32 = x.to(torch.float32)
    reference, ref_state = col.forward(
        x32, pos, init_column_state(SHAPE, 2, T, torch.float32, CPU), 0
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        y, new = col.forward(x32, pos, init_column_state(SHAPE, 2, T, torch.float32, CPU), 0)
    assert bool(torch.isfinite(y).all().item())
    assert new.h[0].dtype == torch.float32
    rel = (_norm(y.float() - reference) / _norm(reference)).item()
    assert rel < 5e-2, rel
    assert (_norm(new.h[0] - ref_state.h[0]) / _norm(ref_state.h[0])).item() < 5e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA training path needs a GPU")
def test_cuda_autocast_bfloat16_matches_float32_within_half_precision() -> None:
    cuda = torch.device("cuda")
    col = _column().to(torch.float32).to(cuda)
    x, pos = _inputs(2, 6)
    x32, pos = x.to(torch.float32).to(cuda), pos.to(cuda)
    reference, ref_state = col.forward(
        x32, pos, init_column_state(SHAPE, 2, T, torch.float32, cuda), 0
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        y, new = col.forward(x32, pos, init_column_state(SHAPE, 2, T, torch.float32, cuda), 0)
    assert bool(torch.isfinite(y).all().item())
    assert new.h[0].dtype == torch.float32
    rel = (_norm(y.float() - reference) / _norm(reference)).item()
    assert rel < 5e-2, rel
    assert (_norm(new.h[0] - ref_state.h[0]) / _norm(ref_state.h[0])).item() < 5e-2
    (g,) = torch.autograd.grad(y.float().square().sum(), [col.rec.w_q])
    assert bool(torch.isfinite(g).all().item())


# --- an expectation computed outside the module ----------------------------------------------


def _mat(rows: list[list[float]]) -> Tensor:
    out = torch.zeros(len(rows), len(rows[0]), dtype=F64)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[i, j] = value
    return out


def test_one_token_against_scalar_arithmetic_in_sublayer_order() -> None:
    # d = 2, one head of width two, one token at position 0, so attention is a softmax over one
    # key (weight one) and rotary is the identity; every sublayer's arithmetic is written out
    # below without calling the column. Swapping attention and the FFN changes the result.
    # key_conv 0: the key is normalize(W_k x) exactly, the arithmetic as before the key filter
    shape = ColumnShape(
        d=2,
        heads=1,
        kv_groups=1,
        head_dim=2,
        window=4,
        rotary_base=10000.0,
        ffn_mult=1,
        block=1,
        key_conv=0,
    )
    col = Column(shape).to(F64)
    eye = _mat([[1.0, 0.0], [0.0, 1.0]])
    w_v_rec, w_o_rec = _mat([[2.0, 0.0], [0.0, 3.0]]), _mat([[1.0, 1.0], [0.0, 1.0]])
    w_v_attn, w_o_attn = _mat([[1.0, 0.0], [1.0, 1.0]]), _mat([[2.0, 0.0], [0.0, 1.0]])
    w1, w2 = _mat([[1.0, -1.0], [1.0, 1.0]]), _mat([[1.0, 0.0], [0.0, -1.0]])
    with torch.no_grad():
        for p, value in (
            (col.rec.w_q, eye),
            (col.rec.w_k, eye),
            (col.rec.w_v, w_v_rec),
            (col.rec.w_o, w_o_rec),
            (col.rec.w_beta, torch.zeros(1, 2, dtype=F64)),  # beta = sigmoid(0) = 1/2
            (col.attn.w_q, eye),
            (col.attn.w_k, eye),
            (col.attn.w_v, w_v_attn),
            (col.attn.w_o, w_o_attn),
            (col.ffn.w1, w1),
            (col.ffn.w2, w2),
        ):
            _ = p.copy_(value)
    h0 = _mat([[1.0, 2.0], [3.0, 4.0]])
    x = _mat([[0.5, -1.0]])[0]

    def rms(v: Tensor) -> Tensor:
        return v / torch.sqrt(v.square().mean() + 1e-6)

    xn = rms(x)
    k = xn / torch.clamp(_norm(xn), min=1e-6)
    v = w_v_rec @ xn
    h1 = h0 + 0.5 * torch.outer(k, v - h0.T @ k)  # (I - beta k k^T) H + beta k v^T
    x1 = x + w_o_rec @ (h1.T @ xn)  # the read H'^T q with q = xn
    x2 = x1 + w_o_attn @ (w_v_attn @ rms(x1))  # one key: softmax weight one, value = W_v xn
    hidden = torch.relu(w1 @ rms(x2)).square()
    x3 = x2 + w2 @ hidden

    state = ColumnState(
        h=(h0.reshape(1, 1, 2, 2),),
        kv=(kv_cache_empty(1, 1, 2, 4, F64, CPU),),
        key_history=(torch.zeros(1, 0, 2, dtype=F64),),
    )
    y, new = col.forward(x.reshape(1, 1, 2), torch.zeros(1, 1, dtype=torch.long), state, 0)
    assert torch.allclose(y[0, 0], x3, rtol=0, atol=1e-12)
    assert torch.allclose(new.h[0][0, 0], h1, rtol=0, atol=1e-12)
