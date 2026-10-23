"""vesper.kernels.nll: the head's per-token loss without float32 vocabulary copies.

The oracle is torch's own ``cross_entropy`` over ``x @ weight.T`` with autograd, never this
module's code: in float64 on the CPU the values and both gradients must agree to rounding; on
CUDA under autocast the result must be at least as close to a float64 truth as the path it
replaces, the Triton row kernels must agree with the torch row operations, and a captured call
must equal an eager one bitwise and repeat itself. Boundaries from maintainer B's review of 571b5f9:
a common offset of a row's logits cancels; under CPU autocast kept and recomputed logits give the
same arithmetic as autocast's own linear; a second backward through kept logits; strided ids.
"""

from collections.abc import Callable, Sequence
from typing import cast

import pytest
import torch
from torch import Tensor
from torch.nn import functional

from vesper.contracts import ContractError
from vesper.graphs import WarmedGraph, deterministic_algorithms
from vesper.kernels.nll import linear_nll

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels need CUDA")
CUDA = torch.device("cuda")
Loss = Callable[[Tensor, Tensor, Tensor], Tensor]


def _inputs(
    n: int, d: int, vocab: int, dtype: torch.dtype, device: torch.device | str, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(n, d, generator=g, dtype=torch.float64) * 0.5).to(device, dtype)
    w = (torch.randn(vocab, d, generator=g, dtype=torch.float64) * 0.05).to(device, dtype)
    t = torch.randint(0, vocab, (n,), generator=g).to(device)
    up = torch.rand(n, generator=g, dtype=torch.float64).to(device)
    return x, w, t, up


def _grads(
    fn: Loss, x: Tensor, w: Tensor, t: Tensor, up: Tensor, *, autocast: bool = False
) -> list[Tensor]:
    xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        loss = fn(xs, ws, t)
    torch.autograd.backward([(loss.to(up.dtype) * up).sum()])
    assert xs.grad is not None
    assert ws.grad is not None
    return [loss.detach().double(), xs.grad.double(), ws.grad.double()]


def _oracle(x: Tensor, w: Tensor, t: Tensor) -> Tensor:
    logits = functional.linear(x, w)
    return functional.cross_entropy(
        logits.to(torch.promote_types(logits.dtype, torch.float32)), t, reduction="none"
    )


def _rel(a: Tensor, b: Tensor) -> float:
    return float(((a - b) ** 2).sum().sqrt() / (b**2).sum().sqrt())


@pytest.mark.parametrize(("dtype", "bound"), [(torch.float64, 1e-12), (torch.float32, 1e-5)])
def test_the_loss_and_both_gradients_equal_torchs_cross_entropy_on_the_cpu(
    dtype: torch.dtype, bound: float
) -> None:
    x, w, t, up = _inputs(64, 24, 300, dtype, "cpu")
    got = _grads(linear_nll, x, w, t, up.to(dtype))
    want = _grads(_oracle, x, w, t, up.to(dtype))
    for a, b, name in zip(got, want, ("loss", "dx", "dw"), strict=True):
        assert _rel(a, b) < bound, name
    assert linear_nll(x, w, t).dtype == torch.promote_types(dtype, torch.float32)


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=needs_cuda),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift", [0.0, 1e4, 1e8])
def test_a_common_offset_of_a_rows_logits_cancels(device: str, shift: float) -> None:
    # Maintainer B's fixture (review of 571b5f9): four equal logits at any offset are a uniform
    # softmax, so the loss is log 4, the input gradient exactly zero and the weight gradient
    # (p - onehot) x summed over both rows; an absolute log-sum-exp kept in float32 gave a loss
    # of 0 and an input gradient of 3e8 at 1e8, and -4.24 at 1e4
    x = torch.ones(2, 1, dtype=torch.float64).to(device, torch.float32)
    w = torch.full((4, 1), shift).to(device)
    t = torch.arange(0, 4, 3).to(device)  # targets 0 and 3
    loss, dx, dw = _grads(linear_nll, x, w, t, torch.ones(2, dtype=torch.float64).to(device))
    torch.testing.assert_close(loss, torch.full_like(loss, 4.0).log(), rtol=1e-6, atol=0.0)
    assert torch.equal(dx, torch.zeros_like(dx))
    targeted = (torch.arange(4) % 3 == 0).to(device, torch.float64)  # rows 0 and 3, once each
    want = (0.5 - targeted)[:, None]  # two rows of p = 1/4, minus each row's own target
    torch.testing.assert_close(dw, want, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("keep", [False, True])
def test_under_cpu_autocast_both_paths_take_autocasts_linear_arithmetic(*, keep: bool) -> None:
    # Maintainer B: under CPU autocast the forward's logits were bf16 (autocast cast the matmul)
    # while the backward recomputed them in float32 outside it, and kept bf16 logits failed the
    # backward against the float32 weight. Now kept and recomputed are the same bits, and both
    # follow autocast's own F.linear (bf16 logits, float32 softmax): the loss and the input
    # gradient to float32 rounding (measured: equal; the float32 recompute was 2.8e-3 off in
    # dx), the weight gradient, accumulated in float32 here and in bf16 there, to bf16 rounding
    x, w, t, up = _inputs(64, 24, 300, torch.float32, "cpu")

    def under(fn: Loss) -> list[Tensor]:
        xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = fn(xs, ws, t)
        torch.autograd.backward([(loss.double() * up).sum()])
        assert xs.grad is not None
        assert ws.grad is not None
        return [loss.detach().double(), xs.grad.double(), ws.grad.double()]

    got = under(lambda a, b, c: linear_nll(a, b, c, keep_logits=keep))
    other = under(lambda a, b, c: linear_nll(a, b, c, keep_logits=not keep))
    want = under(_oracle)
    assert all(torch.equal(a, b) for a, b in zip(got, other, strict=True))
    for a, b, name, bound in zip(got, want, ("loss", "dx", "dw"), (1e-6, 1e-6, 1e-2), strict=True):
        assert _rel(a, b) < bound, name


def test_autocast_leaves_float64_alone_as_it_does_for_a_linear() -> None:
    x, w, t, up = _inputs(8, 6, 20, torch.float64, "cpu")
    plain = _grads(linear_nll, x, w, t, up)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cast = linear_nll(x, w, t)
    assert cast.dtype == torch.float64
    assert torch.equal(cast, plain[0])


@pytest.mark.parametrize("device", DEVICES)
def test_a_second_backward_through_kept_logits_recomputes_them(device: str) -> None:
    # the backward overwrites kept logits with their gradient; a retained graph's second
    # backward must not read that (maintainer B: it failed on the version counter)
    x, w, t, up = _inputs(16, 8, 50, torch.float32, device)
    xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
    loss = (linear_nll(xs, ws, t, keep_logits=True).double() * up).sum()
    first = torch.autograd.grad(loss, (xs, ws), retain_graph=True)
    second = torch.autograd.grad(loss, (xs, ws))
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


@needs_cuda
def test_strided_targets_are_read_by_their_stride() -> None:
    # a view of every other id (maintainer B's probe): the kernels must read the view, not its
    # storage
    x, w, t, up = _inputs(64, 24, 300, torch.float32, CUDA)
    storage = torch.stack([t, torch.zeros_like(t)], dim=1).reshape(-1)
    strided = storage[::2]
    assert strided.stride() == (2,)
    got = _grads(linear_nll, x, w, strided, up)
    want = _grads(linear_nll, x, w, t, up)
    assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


def test_targets_outside_the_vocabulary_and_wrong_shapes_are_refused() -> None:
    x, w, t, _ = _inputs(4, 8, 10, torch.float32, "cpu")
    with pytest.raises(ContractError):
        _ = linear_nll(x, w, torch.arange(4) + 7)  # ids 7..10 of a vocabulary of 10
    with pytest.raises(ContractError):
        _ = linear_nll(x, w[:, :7], t)
    with pytest.raises(ContractError):
        _ = linear_nll(x, w, t[:3])
    assert linear_nll(x[:0], w, t[:0]).shape == (0,)


@needs_cuda
@pytest.mark.parametrize(
    ("vocab", "spread"),
    # spread rows as a trained head's are; and a vocabulary that is not a multiple of the
    # kernel's block with flat rows, where a padded tail read as anything but -inf would count
    [(32768, 40.0), (5003, 1.0)],
)
def test_the_triton_rows_equal_the_torch_rows(vocab: int, spread: float) -> None:
    from vesper.kernels import nll  # noqa: PLC0415
    from vesper.kernels.nll_triton import row_grad_, row_stats  # noqa: PLC0415

    x, w, t, up = _inputs(257, 64, vocab, torch.float32, CUDA)
    logits = (x @ w.T * spread).bfloat16()
    top, log_total, loss = row_stats(logits, t)
    wide = logits.float()
    want_lse = torch.logsumexp(wide, dim=-1)
    assert torch.equal(top, wide.amax(dim=-1))
    torch.testing.assert_close(top + log_total, want_lse, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        loss, want_lse - wide.gather(1, t[:, None])[:, 0], rtol=2e-6, atol=2e-5
    )
    g = up.float()
    got = row_grad_(logits.clone(), t, top, log_total, g)
    want = (torch.softmax(wide, dim=-1) - functional.one_hot(t, vocab)) * g[:, None]
    torch.testing.assert_close(got.float(), want.bfloat16().float(), rtol=8e-3, atol=1e-6)
    assert nll.linear_nll(x[:0], w, t[:0]).shape == (0,)


@needs_cuda
def test_under_autocast_it_is_at_least_as_close_to_a_float64_truth_as_the_old_path() -> None:
    x, w, t, up = _inputs(1024, 512, 32768, torch.float32, CUDA)
    new = _grads(linear_nll, x, w, t, up, autocast=True)
    old = _grads(_oracle, x, w, t, up, autocast=True)
    truth = _grads(_oracle, x.bfloat16().double(), w.bfloat16().double(), t, up)
    for a, b, c, name in zip(new, old, truth, ("loss", "dx", "dw"), strict=True):
        assert _rel(a, c) <= _rel(b, c) * 1.01, name  # the old path's error, or less
        assert _rel(a, c) < 5e-3, name


@needs_cuda
def test_a_captured_call_equals_the_eager_one_bitwise_and_repeats() -> None:
    x, w0, t, _ = _inputs(512, 128, 4099, torch.float32, CUDA)
    w = w0.clone().requires_grad_()

    def step(values: Sequence[Tensor]) -> list[Tensor]:
        w.grad = None
        xs = values[0].detach().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = linear_nll(xs, w, values[1])
        torch.autograd.backward([(loss * torch.arange(1, loss.shape[0] + 1, device=CUDA)).sum()])
        grad_x, grad_w = xs.grad, cast("Tensor | None", w.grad)  # w.grad was set to None above
        assert grad_x is not None
        assert grad_w is not None
        return [loss.detach(), grad_x, grad_w]

    with deterministic_algorithms():
        eager = [[v.clone() for v in step([x, t])] for _ in range(2)]
        graph = WarmedGraph(warm=2)
        captured = [[v.clone() for v in graph(step, [x, t])] for _ in range(4)]
    assert graph.captures == 1
    assert all(torch.equal(a, b) for a, b in zip(eager[0], eager[1], strict=True))
    for run in captured:
        assert all(torch.equal(a, b) for a, b in zip(run, eager[0], strict=True))


@needs_cuda
def test_under_autocast_no_float32_vocabulary_matrix_is_ever_held() -> None:
    # the point of the module (the project owner 2026-10-21: the GPU as full as possible): the
    # logits exist only as one bf16 [N, V] buffer at a time, forward and backward; a float32 copy of
    # them alone would be twice that. Rows outnumber the width here, as in training (4096 rows per
    # call against d = 512), so the weight-sized terms (its float32 gradient, its bf16 cast)
    # stay small beside the logits (measured: no copy of the logits in either gradient matmul)
    n, vocab = 8192, 32768
    x, w0, t, up = _inputs(n, 64, vocab, torch.float32, CUDA)
    w = w0.clone().requires_grad_()
    xs = x.clone().requires_grad_()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(CUDA)
    torch.cuda.reset_peak_memory_stats(CUDA)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = linear_nll(xs, w, t)
    torch.autograd.backward([(loss * up.float()).sum()])
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(CUDA) - base
    logits_bf16 = n * vocab * 2
    assert peak < 1.5 * logits_bf16  # one bf16 matrix plus small terms; float32 would be 2x


@needs_cuda
def test_keeping_the_logits_changes_memory_not_results() -> None:
    # recompute=False keeps each call's logits for the backward pass (one matmul of four saved);
    # the recomputed logits are the same deterministic matmul, so nothing else may differ
    n, vocab = 4096, 32768
    x, w, t, up = _inputs(n, 64, vocab, torch.float32, CUDA)

    def run(*, keep: bool) -> tuple[list[Tensor], int]:
        xs, ws = x.clone().requires_grad_(), w.clone().requires_grad_()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated(CUDA)
        with deterministic_algorithms(), torch.autocast("cuda", dtype=torch.bfloat16):
            loss = linear_nll(xs, ws, t, keep_logits=keep)
        held = torch.cuda.memory_allocated(CUDA) - base  # what waits for the backward pass
        torch.autograd.backward([(loss * up.float()).sum()])
        assert xs.grad is not None
        assert ws.grad is not None
        return [loss.detach(), xs.grad, ws.grad], held

    kept, held_kept = run(keep=True)
    recomputed, held_recomputed = run(keep=False)
    assert all(torch.equal(a, b) for a, b in zip(kept, recomputed, strict=True))
    assert held_kept - held_recomputed >= n * vocab * 2  # the bf16 logits, held
