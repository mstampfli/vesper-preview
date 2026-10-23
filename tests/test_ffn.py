import pytest
import torch

from vesper.ffn import FFN
from vesper.kernels.act import relu_square

D, MULT = 32, 4


def _ffn(seed: int = 0, dtype: torch.dtype = torch.float32) -> FFN:
    g = torch.Generator().manual_seed(seed)
    return FFN(d=D, ffn_mult=MULT, generator=g).to(dtype)


def test_shape_and_formula() -> None:
    ffn = _ffn(dtype=torch.float64)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 7, D, generator=g, dtype=torch.float64)
    y = ffn.forward(x)
    assert y.shape == (2, 7, D)
    hidden = torch.relu(x @ ffn.w1.T).square()
    assert hidden.shape[-1] == MULT * D
    assert (y - hidden @ ffn.w2.T).abs().max().item() < 1e-12


def test_relu_squared_is_sparse() -> None:
    # Half the pre-activations of a zero-mean input are negative, so at least about half the
    # hidden units are exactly zero. The spec's 85-95% is a trained-model figure; at init the
    # floor is the symmetric one.
    ffn = _ffn()
    g = torch.Generator().manual_seed(2)
    x = torch.randn(4, 64, D, generator=g)
    zeros = (ffn.hidden(x) == 0).float().mean().item()
    assert zeros > 0.4


def test_gradients_reach_both_weights() -> None:
    ffn = _ffn()
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 5, D, generator=g)
    g1, g2 = torch.autograd.grad(ffn.forward(x).square().sum(), [ffn.w1, ffn.w2])
    assert g1.abs().sum().item() > 0
    assert g2.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast's float32 pow is CUDA's")
def test_under_bf16_autocast_the_hidden_stays_bf16_and_equals_the_float32_square_bitwise() -> None:
    # the 5090 census of 2026-10-21: square() ran in float32 under autocast (a [b, n, 4d] float32
    # tensor and a cast back); the bf16 product gives the same bits, forward and backward
    cuda = torch.device("cuda")
    ffn = _ffn().to(cuda)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(2, 64, D, generator=g).to(cuda)
    up = torch.randn(2, 64, D, generator=g).to(cuda)

    def run(*, square: bool) -> list[torch.Tensor]:
        xs = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if square:  # the old formula, as an independent oracle
                hidden = torch.relu(torch.nn.functional.linear(xs, ffn.w1)).square()
            else:
                hidden = ffn.hidden(xs)
            y = torch.nn.functional.linear(hidden, ffn.w2)
        grads = torch.autograd.grad((y.float() * up).sum(), [xs, ffn.w1, ffn.w2])
        return [hidden.to(torch.bfloat16).detach(), y.detach(), *grads]

    new, old = run(square=False), run(square=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert ffn.hidden(x).dtype == torch.bfloat16
    for a, b, name in zip(new, old, ("hidden", "y", "dx", "dw1", "dw2"), strict=True):
        assert torch.equal(a, b), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused activation is CUDA's")
def test_the_cuda_activation_is_the_fused_kernel_with_the_same_bits_in_float32() -> None:
    # kernels.act: relu(z)^2 and 2 relu(z) g in one kernel each way, the bits of h * h
    cuda = torch.device("cuda")
    g = torch.Generator().manual_seed(5)
    z = (torch.randn(3, 33, 4 * D, generator=g) * 3).to(cuda)  # negatives, zeros' neighbours
    up = torch.randn(3, 33, 4 * D, generator=g).to(cuda)
    ffn = _ffn().to(cuda)
    zs = z.clone().requires_grad_()
    fused = relu_square(zs)
    node = fused.grad_fn
    assert node is not None
    assert "ReluSquare" in node.name()  # not silently h * h
    (dz,) = torch.autograd.grad((fused * up).sum(), [zs])
    zr = z.clone().requires_grad_()
    h = torch.relu(zr)
    plain = h * h
    (dr,) = torch.autograd.grad((plain * up).sum(), [zr])
    assert torch.equal(fused, plain)
    assert torch.equal(dz, dr)
    hidden = ffn.hidden(z[None, 0, :, :D].clone().requires_grad_())
    assert hidden.shape == (1, 33, 4 * D)
    assert hidden.grad_fn is not None
    assert "ReluSquare" in hidden.grad_fn.name()  # the FFN itself takes the fused path
