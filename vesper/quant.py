"""Quantization of the fast state ``H`` to int4 (spec: Hardware co-design, interface ``quant``).

Symmetric int4 with one fp16 scale per state row: codes in ``[-8, 7]``, ``scale = max|row| / 7``,
two codes packed per byte (even column in the low nibble). The row-wise scale costs ``dk`` fp16
values per head on top of ``dk*dk/2`` bytes of codes (about 6% at dk = 64) and halves the error
of a per-head scale. The measured round-trip error is in tests/test_quant.py and docs/registry.md.
"""

import torch
from torch import Tensor

from vesper.contracts import Dims, check_shape

_INT4_MAX = 7
_INT4_MIN = -8
_NIBBLE = 0xF
_NIBBLE_BITS = 4
_SIGN_BOUNDARY = 8
_UNSIGNED_RANGE = 16
_SCALE_FLOOR = 1e-8


def quant_state_int4(state: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize a floating-point state to packed int4 codes with per-row fp16 scales.

    Args:
        state: ``[b, h, dk, dk]`` floating point, ``dk`` even.

    Returns:
        ``(codes, scale)``: ``codes: [b, h, dk, dk/2]`` uint8 (two codes per byte) and
        ``scale: [b, h, dk]`` float16, one per state row.

    Raises:
        ValueError: if ``dk`` is odd.
    """
    dims: Dims = {}
    check_shape("state", state, ("b", "h", "dk", "dk"), dims)
    dk = dims["dk"]
    if dk % 2 != 0:
        msg = f"dk must be even to pack two int4 codes per byte, got {dk}"
        raise ValueError(msg)
    work = state.detach().float()
    scale16 = (work.abs().amax(dim=-1) / _INT4_MAX).clamp_min(_SCALE_FLOOR).to(torch.float16)
    codes = (
        torch.round(work / scale16.float()[..., None]).clamp(_INT4_MIN, _INT4_MAX).to(torch.int8)
    )
    low = (codes[..., 0::2] & _NIBBLE).to(torch.uint8)
    high = (codes[..., 1::2] & _NIBBLE).to(torch.uint8)
    return (high << _NIBBLE_BITS) | low, scale16


def dequant_state(codes: Tensor, scale: Tensor) -> Tensor:
    """Invert ``quant_state_int4``.

    Args:
        codes: ``[b, h, dk, dk/2]`` uint8 packed codes.
        scale: ``[b, h, dk]`` float16 per-row scales.

    Returns:
        ``[b, h, dk, dk]`` float32.
    """
    dims: Dims = {}
    check_shape("codes", codes, ("b", "h", "dk", "half"), dims, dtype=torch.uint8)
    check_shape("scale", scale, ("b", "h", "dk"), dims, dtype=torch.float16)
    low = (codes & _NIBBLE).to(torch.int8)
    high = (codes >> _NIBBLE_BITS).to(torch.int8)
    signed = torch.stack((low, high), dim=-1).flatten(-2)
    signed = torch.where(signed >= _SIGN_BOUNDARY, signed - _UNSIGNED_RANGE, signed)
    return signed.float() * scale.float()[..., None]
