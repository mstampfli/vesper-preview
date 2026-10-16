import math

import pytest
import torch

from vesper.recurrence import delta_scan_chunked, delta_scan_step, normalize_keys, write_strength


@pytest.mark.parametrize("block", [1, 2, 3, 64])
def test_two_token_rational_oracle(block: int) -> None:
    # Direct scalar arithmetic, independent of either scan implementation.
    # Nonorthogonal keys, unequal partial writes, q != k, and nonsymmetric H0.
    q = torch.tensor(((2.0, -1.0), (-1.0, 3.0)), dtype=torch.float64).reshape(1, 2, 1, 2)
    k = torch.tensor(((3 / 5, 4 / 5), (0.0, 1.0)), dtype=torch.float64).reshape(1, 2, 1, 2)
    v = torch.tensor(((5.0, -2.0), (-3.0, 1.0)), dtype=torch.float64).reshape(1, 2, 1, 2)
    beta = torch.tensor((1 / 2, 3 / 4), dtype=torch.float64).reshape(1, 2, 1)
    h0 = torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=torch.float64).reshape(1, 1, 2, 2)
    h1 = torch.tensor(((8 / 5, 2 / 25), (19 / 5, 36 / 25)), dtype=torch.float64).reshape(1, 1, 2, 2)
    h2 = torch.tensor(((8 / 5, 2 / 25), (-13 / 10, 111 / 100)), dtype=torch.float64).reshape(
        1, 1, 2, 2
    )
    expected = torch.tensor(((-3 / 5, -32 / 25), (-11 / 2, 13 / 4)), dtype=torch.float64).reshape(
        1, 2, 1, 2
    )

    first, state = delta_scan_step(q[:, 0], k[:, 0], v[:, 0], beta[:, 0], h0)
    assert torch.allclose(first, expected[:, 0], rtol=0, atol=1e-12)
    assert torch.allclose(state, h1, rtol=0, atol=1e-12)
    second, state = delta_scan_step(q[:, 1], k[:, 1], v[:, 1], beta[:, 1], state)
    assert torch.allclose(second, expected[:, 1], rtol=0, atol=1e-12)
    assert torch.allclose(state, h2, rtol=0, atol=1e-12)
    outputs, state = delta_scan_chunked(q, k, v, beta, h0, block)
    assert torch.allclose(outputs, expected, rtol=0, atol=1e-12)
    assert torch.allclose(state, h2, rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float32, 1e-6), (torch.float64, 1e-6), (torch.bfloat16, 1e-2)],
)
def test_zero_key_has_the_floored_linear_gradient(dtype: torch.dtype, tolerance: float) -> None:
    key = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    normalized = normalize_keys(key)
    (gradient,) = torch.autograd.grad(normalized.sum(), (key,))
    assert torch.equal(normalized, torch.zeros_like(key))
    assert torch.isfinite(gradient).all()
    assert torch.allclose(gradient, torch.full_like(key, 1e6), rtol=tolerance, atol=0)


@pytest.mark.parametrize(
    ("dtype", "magnitude"),
    [(torch.float16, 1e-4), (torch.bfloat16, 1e20), (torch.float32, 1e20), (torch.float64, 1e200)],
)
def test_finite_key_normalization_does_not_square_out_of_range(
    dtype: torch.dtype, magnitude: float
) -> None:
    key = torch.tensor(((magnitude, 0.0),), dtype=dtype)
    assert torch.equal(normalize_keys(key), torch.tensor(((1.0, 0.0),), dtype=dtype))


def test_wide_float16_key_accumulates_its_norm_without_overflow() -> None:
    key = torch.ones(1, 65536, dtype=torch.float16)
    assert torch.equal(normalize_keys(key), torch.full_like(key, 1 / 256))


@pytest.mark.parametrize("magnitude", [1.0, 1e200])
def test_normalization_gradient_matches_independent_derivative(magnitude: float) -> None:
    key = torch.tensor(((magnitude, 2 * magnitude),), dtype=torch.float64, requires_grad=True)
    (gradient,) = torch.autograd.grad(normalize_keys(key).sum(), (key,))
    # d((x+y)/sqrt(x*x+y*y)) at (s, 2s), written without squaring s.
    expected = torch.tensor(((2.0, -1.0),), dtype=torch.float64) / (5 * math.sqrt(5)) / magnitude
    assert torch.allclose(gradient, expected, rtol=1e-12, atol=0)


def test_small_keys_keep_the_existing_norm_floor() -> None:
    key = torch.tensor(((3e-8, 4e-8),), dtype=torch.float64)
    expected = torch.tensor(((0.03, 0.04),), dtype=torch.float64)
    assert torch.allclose(normalize_keys(key), expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_empty_key_arrays_keep_their_shape(shape: tuple[int, int]) -> None:
    key = torch.empty(shape, dtype=torch.float64, requires_grad=True)
    normalized = normalize_keys(key)
    assert normalized.shape == key.shape
    (gradient,) = torch.autograd.grad(normalized.sum(), (key,))
    assert gradient.shape == key.shape


@pytest.mark.parametrize("block", [1, 64])
def test_empty_scan_is_the_identity_including_gradients(block: int) -> None:
    q = torch.empty(2, 0, 3, 4, dtype=torch.float64, requires_grad=True)
    beta = torch.empty(2, 0, 3, dtype=torch.float64)
    state = torch.ones(2, 3, 4, 4, dtype=torch.float64, requires_grad=True)
    output, carried = delta_scan_chunked(q, q, q, beta, state, block)
    assert output.shape == q.shape
    assert torch.equal(carried, state)
    q_grad, state_grad = torch.autograd.grad(output.sum() + carried.sum(), (q, state))
    assert q_grad.shape == q.shape
    assert torch.equal(state_grad, torch.ones_like(state))


def test_sigmoid_endpoints_are_valid_floating_point_write_strengths() -> None:
    logits = torch.tensor((-1000.0, 0.0, 1000.0))
    assert torch.equal(write_strength(logits), torch.tensor((0.0, 0.5, 1.0)))


def test_empty_scan_still_rejects_an_invalid_block() -> None:
    q = torch.empty(1, 0, 1, 2)
    beta = torch.empty(1, 0, 1)
    state = torch.zeros(1, 1, 2, 2)
    with pytest.raises(ValueError, match="block must be at least 1"):
        _ = delta_scan_chunked(q, q, q, beta, state, block=0)
