"""The fused delta scan (vesper/kernels/delta_scan.py) against the reference scan.

The reference in float64, fed the SAME bfloat16-valued inputs, is the truth: the kernel's only
licence is its bfloat16 intermediates, measured at about 0.5 % relative, so 1.5e-2 is the bound
for outputs, state and gradients. A kernel that drops the initial state, scales the query,
skips the erase or loses a block misses it by tens of percent.
"""

import pytest
import torch
from torch import Tensor, nn

from conftest import small_config
from vesper.column import (
    ColumnShape,
    RecurrentSublayer,
    check_capturable_scan,
    check_scan,
    scan_of,
    set_scan,
)
from vesper.contracts import ContractError
from vesper.data import Document, Stream
from vesper.kernels import delta_scan as fused_module
from vesper.kernels.delta_scan import (
    delta_scan_fused,
    delta_scan_fused_fp32,
    fused_scan_available,
)
from vesper.model import Vesper
from vesper.recurrence import delta_scan_chunked, normalize_keys, write_strength

BF16 = torch.bfloat16
TOL = 1.5e-2
needs_fused = pytest.mark.skipif(
    not fused_scan_available(), reason="the fused scan needs a CUDA device and fla-core"
)
Inputs = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def _rel(a: Tensor, b: Tensor) -> float:
    a64, b64 = a.detach().double(), b.detach().double()
    err = (a64 - b64).square().sum().sqrt() / b64.square().sum().sqrt()
    return float(err.item())


def _inputs(b: int, n: int, h: int, dk: int, seed: int = 0) -> Inputs:
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, n, h, dk, generator=g)
    k = normalize_keys(torch.randn(b, n, h, dk, generator=g))
    v = torch.randn(b, n, h, dk, generator=g)
    beta = write_strength(torch.randn(b, n, h, generator=g))
    # a state as large as a chunk's writes, so a kernel that ignored it could not pass
    h0 = 0.5 * torch.randn(b, h, dk, dk, generator=g)
    cuda = torch.device("cuda")
    return (
        q.to(cuda, BF16),
        k.to(cuda, BF16),
        v.to(cuda, BF16),
        beta.to(cuda, BF16),
        h0.to(cuda),
    )


def _reference(q: Tensor, k: Tensor, v: Tensor, beta: Tensor, h0: Tensor) -> tuple[Tensor, Tensor]:
    return delta_scan_chunked(q.double(), k.double(), v.double(), beta.double(), h0.double(), 64)


# n = 1, one short of a 64-token chunk, one past it, two and a partial, and 4096 (the chunk of
# Vesper-S) to show the error does not grow along the sequence; dk 16 (stage 1) and 64 (Vesper-S)
SIZES = [(2, 1, 2, 16), (2, 63, 3, 16), (1, 65, 2, 64), (2, 130, 3, 16), (1, 4096, 2, 64)]


@needs_fused
@pytest.mark.parametrize(("b", "n", "h", "dk"), SIZES)
def test_forward_and_state_match_the_reference(b: int, n: int, h: int, dk: int) -> None:
    q, k, v, beta, h0 = _inputs(b, n, h, dk)
    o_ref, s_ref = _reference(q, k, v, beta, h0)
    o, s = delta_scan_fused(q, k, v, beta, h0)
    assert o.dtype == BF16
    assert s.dtype == torch.float32
    assert _rel(o, o_ref) < TOL
    assert _rel(s, s_ref) < TOL


@needs_fused
@pytest.mark.parametrize(("b", "n", "h", "dk"), SIZES[:4])
def test_every_gradient_matches_the_reference(b: int, n: int, h: int, dk: int) -> None:
    q, k, v, beta, h0 = _inputs(b, n, h, dk)
    g = torch.Generator(device="cuda").manual_seed(7)
    d_o = torch.randn(b, n, h, dk, generator=g, device="cuda")
    d_s = torch.randn(b, h, dk, dk, generator=g, device="cuda")
    ref_in = [t.double().requires_grad_() for t in (q, k, v, beta, h0)]
    o_ref, s_ref = delta_scan_chunked(ref_in[0], ref_in[1], ref_in[2], ref_in[3], ref_in[4], 64)
    torch.autograd.backward([(o_ref * d_o.double()).sum() + (s_ref * d_s.double()).sum()])
    fused_in = [t.clone().requires_grad_() for t in (q, k, v, beta, h0)]
    o, s = delta_scan_fused(fused_in[0], fused_in[1], fused_in[2], fused_in[3], fused_in[4])
    torch.autograd.backward([(o.float() * d_o).sum() + (s * d_s).sum()])
    for name, got, want in zip(["q", "k", "v", "beta", "state"], fused_in, ref_in, strict=True):
        assert got.grad is not None, name
        assert want.grad is not None, name
        assert _rel(got.grad, want.grad) < TOL, name


@needs_fused
def test_a_carried_state_continues_the_sequence() -> None:
    # 130 tokens in one call against 50 then 80 with the state carried: the split lies inside a
    # kernel chunk, so the second call starts from a state the first left mid-chunk
    q, k, v, beta, h0 = _inputs(2, 130, 3, 16)
    o_ref, s_ref = _reference(q, k, v, beta, h0)
    o_a, s_a = delta_scan_fused(q[:, :50], k[:, :50], v[:, :50], beta[:, :50], h0)
    o_b, s_b = delta_scan_fused(q[:, 50:], k[:, 50:], v[:, 50:], beta[:, 50:], s_a)
    assert _rel(torch.cat([o_a, o_b], dim=1), o_ref) < TOL
    assert _rel(s_b, s_ref) < TOL


@needs_fused
def test_no_tokens_return_the_state_unchanged() -> None:
    q, k, v, beta, h0 = _inputs(2, 0, 3, 16)
    o, s = delta_scan_fused(q, k, v, beta, h0)
    assert o.shape == (2, 0, 3, 16)
    assert s is h0


@needs_fused
def test_refuses_what_the_kernel_was_not_measured_for() -> None:
    q, k, v, beta, h0 = _inputs(1, 8, 2, 16)
    with pytest.raises(ContractError, match="dtype"):
        _ = delta_scan_fused(q.float(), k, v, beta, h0)
    with pytest.raises(ContractError, match="dtype"):
        _ = delta_scan_fused(q, k, v, beta, h0.to(BF16))
    with pytest.raises(ContractError, match="axis"):
        _ = delta_scan_fused(q, k[:, :4], v, beta, h0)
    with pytest.raises(ContractError, match="is on cpu"):
        _ = delta_scan_fused(q, k, v, beta, h0.cpu())
    wide = torch.zeros(1, 8, 1, 512, dtype=BF16, device="cuda")
    with pytest.raises(ContractError, match="exceeds"):
        _ = delta_scan_fused(
            wide,
            wide,
            wide,
            torch.zeros(1, 8, 1, dtype=BF16, device="cuda"),
            torch.zeros(1, 1, 512, 512, device="cuda"),
        )


def test_refuses_a_cpu_tensor_before_touching_the_kernel() -> None:
    # runs everywhere, fla-core or not: the contract is checked before the library is imported
    z = torch.zeros(1, 4, 1, 8, dtype=BF16)
    with pytest.raises(ContractError, match="CUDA only"):
        _ = delta_scan_fused(z, z, z, torch.zeros(1, 4, 1, dtype=BF16), torch.zeros(1, 1, 8, 8))


# --- the switch in the model -----------------------------------------------------------------

SHAPE = ColumnShape(
    d=32,
    heads=2,
    kv_groups=2,
    head_dim=16,
    window=4,
    rotary_base=10000.0,
    ffn_mult=2,
    block=4,
    key_conv=4,
)


def test_a_fused_sublayer_runs_the_fused_scan_and_never_the_reference() -> None:
    # on a CPU sublayer the fused branch must reach delta_scan_fused, which refuses the device;
    # a dispatch that fell back to the reference would return a result instead
    sub = RecurrentSublayer(SHAPE, torch.Generator().manual_seed(0))
    sub.scan_impl = "fused"
    x = torch.randn(2, 6, 32)
    state = torch.zeros(2, 2, 16, 16)
    history = torch.zeros(2, 3, 32)
    with pytest.raises(ContractError, match="CUDA only"):
        _ = sub.forward(x, state, history)


def test_set_scan_refuses_unknown_names_and_models_without_a_recurrence() -> None:
    model = Vesper(small_config(), torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="one of"):
        _ = set_scan(model, "triton")
    with pytest.raises(ValueError, match="no recurrent sublayer"):
        _ = set_scan(nn.Linear(2, 2), "reference")
    for impl in ("fused", "fused-fp32"):
        with pytest.raises(ValueError, match="fused scan needs"):
            _ = set_scan(model, impl)  # on CPU: no device here, or no CUDA parameters
    assert scan_of(model) == "reference"  # a refused switch changed nothing
    # the reference scan's block inverse cannot be captured, and CUDA work is always captured
    # (the project owner 2026-10-21): a driver's choice on CUDA is a fused scan; the check needs no
    # GPU to refuse. The plain validity check keeps the reference valid on CUDA: the eager oracle
    # the equivalence tests below compare the fused kernels against
    with pytest.raises(ValueError, match="cannot run as a captured CUDA graph"):
        _ = check_capturable_scan("reference", torch.device("cuda"))
    assert check_capturable_scan("reference", torch.device("cpu")) == "reference"
    assert check_scan("reference", torch.device("cuda")) == "reference"


def test_set_scan_reaches_every_sublayer_and_scan_of_reads_them_back() -> None:
    model = Vesper(small_config(), torch.Generator().manual_seed(0))
    sublayers = [m for m in model.modules() if isinstance(m, RecurrentSublayer)]
    for sublayer in sublayers:
        sublayer.scan_impl = "fused"
    assert set_scan(model, "reference") == len(sublayers) == small_config().model.columns
    assert scan_of(model) == "reference"
    sublayers[-1].scan_impl = "fused"
    with pytest.raises(ValueError, match="expected exactly one"):
        _ = scan_of(model)


@needs_fused
def test_the_model_trains_the_same_under_either_scan() -> None:
    # one bf16 forward and backward of the small model, reference scan against fused: loss and
    # every parameter's gradient within the kernel's bound (both runs round the same matmuls)
    cfg = small_config()
    cuda = torch.device("cuda")
    docs = [
        Document("a", torch.arange(1, 30, dtype=torch.int32)),
        Document("b", torch.arange(20, 51, dtype=torch.int32)),
    ]
    chunk = Stream(iter(docs), slots=2, chunk=16).next()
    assert chunk is not None
    outcomes: list[tuple[Tensor, list[Tensor]]] = []
    for impl in ("reference", "fused"):
        model = Vesper(cfg, torch.Generator().manual_seed(0)).to(cuda)
        assert set_scan(model, impl) == cfg.model.columns
        assert scan_of(model) == impl
        with torch.autocast(device_type="cuda", dtype=BF16):
            result = model.run(
                chunk.tokens.to(cuda), chunk.pos.to(cuda), model.init_states(2, torch.float32, cuda)
            )
            objective, _ = model.objective(chunk, result, cfg)
        torch.autograd.backward([objective])
        grads = [p.grad.detach().float().clone() for p in model.parameters() if p.grad is not None]
        outcomes.append((objective.detach().float(), grads))
    (loss_ref, grads_ref), (loss_fused, grads_fused) = outcomes
    assert len(grads_ref) == len(grads_fused)
    assert _rel(loss_fused, loss_ref) < TOL
    for got, want in zip(grads_fused, grads_ref, strict=True):
        assert _rel(got, want) < 5 * TOL


def test_the_module_names_one_head_dim_bound() -> None:
    assert fused_module.MAX_HEAD_DIM == 256


# --- the float32 recurrent kernel ------------------------------------------------------------
# Its licence is float32 rounding only: measured about 2e-7 (stage 1) to 4e-7 (4096 tokens)
# relative against the float64 reference on the same inputs, the float32 reference's own error.
# TOL_FP32 = 1e-5 keeps a margin and still refuses the bfloat16 kernel (about 5e-3) or any
# bfloat16 cast on the way.

TOL_FP32 = 1e-5


def _inputs32(b: int, n: int, h: int, dk: int, seed: int = 0) -> Inputs:
    q, k, v, beta, h0 = _inputs(b, n, h, dk, seed)
    g = torch.Generator().manual_seed(seed + 100)  # float32 values the bf16 grid cannot hold
    noise = [1e-3 * torch.randn(t.shape, generator=g).to(t.device) for t in (q, v)]
    k32 = normalize_keys(k.float() + 1e-3 * torch.randn(k.shape, generator=g).to(k.device))
    return (
        q.float() + noise[0],
        k32,
        v.float() + noise[1],
        beta.float(),
        h0,
    )


@needs_fused
@pytest.mark.parametrize(("b", "n", "h", "dk"), SIZES)
def test_fp32_forward_and_state_match_the_reference_to_float32(
    b: int, n: int, h: int, dk: int
) -> None:
    q, k, v, beta, h0 = _inputs32(b, n, h, dk)
    o_ref, s_ref = _reference(q, k, v, beta, h0)
    o, s = delta_scan_fused_fp32(q, k, v, beta, h0)
    assert (o.dtype, s.dtype) == (torch.float32, torch.float32)
    assert _rel(o, o_ref) < TOL_FP32
    assert _rel(s, s_ref) < TOL_FP32


@needs_fused
@pytest.mark.parametrize(("b", "n", "h", "dk"), SIZES[:4])
def test_fp32_every_gradient_matches_the_reference_to_float32(
    b: int, n: int, h: int, dk: int
) -> None:
    q, k, v, beta, h0 = _inputs32(b, n, h, dk)
    g = torch.Generator(device="cuda").manual_seed(7)
    d_o = torch.randn(b, n, h, dk, generator=g, device="cuda")
    d_s = torch.randn(b, h, dk, dk, generator=g, device="cuda")
    ref_in = [t.double().requires_grad_() for t in (q, k, v, beta, h0)]
    o_ref, s_ref = delta_scan_chunked(ref_in[0], ref_in[1], ref_in[2], ref_in[3], ref_in[4], 64)
    torch.autograd.backward([(o_ref * d_o.double()).sum() + (s_ref * d_s.double()).sum()])
    fused_in = [t.clone().requires_grad_() for t in (q, k, v, beta, h0)]
    o, s = delta_scan_fused_fp32(fused_in[0], fused_in[1], fused_in[2], fused_in[3], fused_in[4])
    torch.autograd.backward([(o * d_o).sum() + (s * d_s).sum()])
    for name, got, want in zip(["q", "k", "v", "beta", "state"], fused_in, ref_in, strict=True):
        assert got.grad is not None, name
        assert want.grad is not None, name
        assert _rel(got.grad, want.grad) < TOL_FP32, name


@needs_fused
def test_fp32_gradients_flow_from_a_state_that_needs_none() -> None:
    # a cold or detached state (the first pass of every chunk) does not require grad; fla's
    # recurrent backward failed to compile there, so the wrapper gives the kernel a fresh leaf
    q, k, v, beta, h0 = _inputs32(2, 63, 3, 16)
    ref_in = [t.double().requires_grad_() for t in (q, k, v, beta)]
    o_ref, _ = delta_scan_chunked(ref_in[0], ref_in[1], ref_in[2], ref_in[3], h0.double(), 64)
    torch.autograd.backward([o_ref.square().sum()])
    fused_in = [t.clone().requires_grad_() for t in (q, k, v, beta)]
    state = h0.clone()
    o, _ = delta_scan_fused_fp32(fused_in[0], fused_in[1], fused_in[2], fused_in[3], state)
    torch.autograd.backward([o.square().sum()])
    assert not state.requires_grad  # the caller's state is untouched
    assert state.grad is None
    for name, got, want in zip(["q", "k", "v", "beta"], fused_in, ref_in, strict=True):
        assert got.grad is not None, name
        assert want.grad is not None, name
        assert _rel(got.grad, want.grad) < TOL_FP32, name


@needs_fused
def test_fp32_a_carried_state_continues_the_sequence_and_no_tokens_change_nothing() -> None:
    q, k, v, beta, h0 = _inputs32(2, 130, 3, 16)
    o_ref, s_ref = _reference(q, k, v, beta, h0)
    o_a, s_a = delta_scan_fused_fp32(q[:, :50], k[:, :50], v[:, :50], beta[:, :50], h0)
    o_b, s_b = delta_scan_fused_fp32(q[:, 50:], k[:, 50:], v[:, 50:], beta[:, 50:], s_a)
    assert _rel(torch.cat([o_a, o_b], dim=1), o_ref) < TOL_FP32
    assert _rel(s_b, s_ref) < TOL_FP32
    o0, s0 = delta_scan_fused_fp32(q[:, :0], k[:, :0], v[:, :0], beta[:, :0], h0)
    assert o0.shape == (2, 0, 3, 16)
    assert s0 is h0


@needs_fused
def test_fp32_refuses_bfloat16_inputs_and_a_cpu_tensor() -> None:
    q, k, v, beta, h0 = _inputs32(1, 8, 2, 16)
    with pytest.raises(ContractError, match="dtype"):
        _ = delta_scan_fused_fp32(q.to(BF16), k, v, beta, h0)
    with pytest.raises(ContractError, match="dtype"):
        _ = delta_scan_fused_fp32(q, k, v, beta.to(BF16), h0)
    with pytest.raises(ContractError, match="is on cpu"):
        _ = delta_scan_fused_fp32(q, k, v, beta, h0.cpu())
    z = torch.zeros(1, 4, 1, 8)
    with pytest.raises(ContractError, match="CUDA only"):
        _ = delta_scan_fused_fp32(z, z, z, torch.zeros(1, 4, 1), torch.zeros(1, 1, 8, 8))


def test_a_fused_fp32_sublayer_runs_the_fp32_kernel_and_never_the_reference() -> None:
    # as for "fused": on a CPU sublayer the branch must reach delta_scan_fused_fp32, which
    # refuses the device; a dispatch that fell back to the reference would return a result
    sub = RecurrentSublayer(SHAPE, torch.Generator().manual_seed(0))
    sub.scan_impl = "fused-fp32"
    with pytest.raises(ContractError, match="CUDA only"):
        _ = sub.forward(torch.randn(2, 6, 32), torch.zeros(2, 2, 16, 16), torch.zeros(2, 3, 32))


@needs_fused
def test_the_model_trains_the_same_under_the_fp32_kernel_to_float32() -> None:
    # float32 throughout, no autocast: the reference scan and the fp32 kernel differ only in
    # float32 rounding, so the loss and every parameter gradient agree to 1e-4 relative (the
    # bfloat16 kernel misses this by an order of magnitude or more)
    cfg = small_config()
    cuda = torch.device("cuda")
    docs = [
        Document("a", torch.arange(1, 30, dtype=torch.int32)),
        Document("b", torch.arange(20, 51, dtype=torch.int32)),
    ]
    chunk = Stream(iter(docs), slots=2, chunk=16).next()
    assert chunk is not None
    outcomes: list[tuple[Tensor, list[Tensor]]] = []
    for impl in ("reference", "fused-fp32"):
        model = Vesper(cfg, torch.Generator().manual_seed(0)).to(cuda)
        assert set_scan(model, impl) == cfg.model.columns
        assert scan_of(model) == impl
        result = model.run(
            chunk.tokens.to(cuda), chunk.pos.to(cuda), model.init_states(2, torch.float32, cuda)
        )
        objective, _ = model.objective(chunk, result, cfg)
        torch.autograd.backward([objective])
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        outcomes.append((objective.detach(), grads))
    (loss_ref, grads_ref), (loss_fp32, grads_fp32) = outcomes
    assert len(grads_ref) == len(grads_fp32)
    assert _rel(loss_fp32, loss_ref) < 1e-6
    for got, want in zip(grads_fp32, grads_ref, strict=True):
        assert _rel(got, want) < 1e-4
