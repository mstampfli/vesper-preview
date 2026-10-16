import torch

from vesper.ffn import FFN

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
