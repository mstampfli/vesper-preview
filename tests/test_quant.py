import pytest
import torch

from vesper.quant import dequant_state, quant_state_int4
from vesper.recurrence import delta_scan_chunked, normalize_keys, write_strength

# Measured 0.107-0.108 on Gaussian and delta-rule states (docs/registry.md); a packing bug reads
# as about 1.0, so the bound catches both a broken packer and a regression of the scale scheme.
INT4_ROUND_TRIP_BOUND = 0.12


def _rel_error(state: torch.Tensor) -> float:
    codes, scale = quant_state_int4(state)
    back = dequant_state(codes, scale)
    err = (back - state).square().sum().sqrt() / state.square().sum().sqrt()
    return float(err.item())


def test_shapes_and_dtypes() -> None:
    state = torch.randn(2, 3, 64, 64)
    codes, scale = quant_state_int4(state)
    assert codes.shape == (2, 3, 64, 32)
    assert codes.dtype == torch.uint8
    assert scale.shape == (2, 3, 64)
    assert scale.dtype == torch.float16


def test_round_trip_error_on_a_gaussian_state() -> None:
    g = torch.Generator().manual_seed(0)
    assert _rel_error(torch.randn(2, 4, 64, 64, generator=g)) < INT4_ROUND_TRIP_BOUND


def test_round_trip_error_on_a_delta_rule_state() -> None:
    g = torch.Generator().manual_seed(1)
    b, n, h, dk = 1, 512, 4, 64
    q = torch.randn(b, n, h, dk, generator=g)
    k = normalize_keys(torch.randn(b, n, h, dk, generator=g))
    v = torch.randn(b, n, h, dk, generator=g)
    beta = write_strength(torch.randn(b, n, h, generator=g))
    _, state = delta_scan_chunked(q, k, v, beta, torch.zeros(b, h, dk, dk), block=64)
    assert _rel_error(state) < INT4_ROUND_TRIP_BOUND


def test_grid_points_round_trip_exactly() -> None:
    # Every row holds a 7 so its scale is exactly 0.5; all entries are then on the int4 grid.
    g = torch.Generator().manual_seed(2)
    codes = torch.randint(-7, 8, (1, 2, 8, 8), generator=g)
    codes[..., 0] = 7
    state = codes.float() * 0.5
    packed, scale = quant_state_int4(state)
    assert torch.equal(dequant_state(packed, scale), state)


def test_negative_values_keep_their_sign() -> None:
    state = torch.full((1, 1, 4, 4), -1.0)
    codes, scale = quant_state_int4(state)
    assert (dequant_state(codes, scale) + 1.0).abs().max().item() < 1e-3


def test_zero_state_round_trips_without_nan() -> None:
    codes, scale = quant_state_int4(torch.zeros(1, 1, 4, 4))
    assert torch.equal(dequant_state(codes, scale), torch.zeros(1, 1, 4, 4))


def test_odd_head_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="dk must be even"):
        _ = quant_state_int4(torch.zeros(1, 1, 5, 5))
