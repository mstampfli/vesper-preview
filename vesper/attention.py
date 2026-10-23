"""Windowed attention (spec 3b): exact softmax over the last ``w`` tokens, rotary, GQA.

Reference implementation; the Triton kernel is tested against this file. A chunk of ``n``
tokens is processed in query blocks of ``w`` tokens. Block ``[s, e)`` attends to the cache (only
while ``s < w``, since older blocks can never reach it) and to chunk keys ``[s - w + 1, e)``, so
at most ``3w`` keys per block, under the exact mask ``p_i - w < p_j <= p_i``. Exactness against
the unblocked full-mask oracle is pinned by tests/test_attention.py.

Layouts: ``x: [b, n, d]``; ``pos: [b, n]`` int64 absolute positions, consecutive within the
chunk; ``q: [b, n, h, dk]``; ``k, v: [b, n, hkv, dk]`` with ``hkv = h / kv_groups``. The cache
holds the last ``m <= w`` rotated keys, values and positions, oldest first; it is one window,
and the column keeps one per pass (spec: state ordering). Rotary phases are computed in float64
whatever the activation dtype; the base comes from ``model.rotary_base``.
"""

import math
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vesper.contracts import ContractError, Dims, check_shape, require
from vesper.kernels.window_attn import window_attention

_WINDOW_KERNEL_DTYPES = (torch.bfloat16, torch.float16)  # float32 keeps the blocked path


@dataclass(frozen=True, slots=True)
class KVCache:
    """The last ``m <= window`` entries of one attention window, oldest first.

    ``k`` and ``v`` are ``[b, m, hkv, dk]`` (keys already rotated), ``pos`` is ``[b, m]`` int64.
    """

    k: Tensor
    v: Tensor
    pos: Tensor
    window: int


def kv_cache_empty(
    b: int, hkv: int, dk: int, window: int, dtype: torch.dtype, device: torch.device
) -> KVCache:
    """An empty cache (``m = 0``) for ``window`` entries.

    Raises:
        ValueError: if ``window`` is below 1.
    """
    if window < 1:
        msg = f"window must be at least 1, got {window}"
        raise ValueError(msg)
    return KVCache(
        k=torch.zeros(b, 0, hkv, dk, dtype=dtype, device=device),
        v=torch.zeros(b, 0, hkv, dk, dtype=dtype, device=device),
        pos=torch.zeros(b, 0, dtype=torch.long, device=device),
        window=window,
    )


def kv_cache_push(cache: KVCache, k: Tensor, v: Tensor, pos: Tensor) -> KVCache:
    """Append a chunk's rotated keys, values and positions and keep the last ``window`` entries.

    Args:
        cache: the window so far.
        k: ``[b, n, hkv, dk]`` rotated keys.
        v: ``[b, n, hkv, dk]`` values.
        pos: ``[b, n]`` int64 positions.

    Returns:
        A new cache holding the newest ``min(m + n, window)`` entries, allocated at exactly that
        size (a slice of the full concatenation would keep all ``m + n`` entries alive). The
        inputs are not modified; with no new entries the cache itself is returned. This is an
        ordered tail at O(window) per push, value-equivalent to the spec's ring; the ring's O(n)
        append belongs to the kernel, which is tested against this reference.
    """
    dims: Dims = {}
    check_shape("cache.k", cache.k, ("b", "m", "hkv", "dk"), dims)
    check_shape("cache.v", cache.v, ("b", "m", "hkv", "dk"), dims)
    check_shape("cache.pos", cache.pos, ("b", "m"), dims, dtype=torch.long)
    check_shape("k", k, ("b", "n", "hkv", "dk"), dims)
    check_shape("v", v, ("b", "n", "hkv", "dk"), dims)
    check_shape("pos", pos, ("b", "n"), dims, dtype=torch.long)
    w, m, n = cache.window, dims["m"], dims["n"]
    if n == 0:
        return cache
    keep_new = min(n, w)
    keep_old = min(m, w - keep_new)
    return KVCache(
        k=torch.cat([cache.k[:, m - keep_old :], k[:, n - keep_new :]], dim=1),
        v=torch.cat([cache.v[:, m - keep_old :], v[:, n - keep_new :]], dim=1),
        pos=torch.cat([cache.pos[:, m - keep_old :], pos[:, n - keep_new :]], dim=1),
        window=w,
    )


def rotary(x: Tensor, pos: Tensor, base: float) -> Tensor:
    """Apply rotary position embedding to ``x`` at absolute positions ``pos``.

    Pairs ``(x[2j], x[2j+1])`` are rotated by ``pos * base^(-2j/dk)``. The phases are computed
    in float64 whatever the activation dtype and only ``cos`` and ``sin`` are cast to ``x.dtype``:
    converting the position first overflows float16 past 65,504 and rounds neighbouring
    positions together in bfloat16 from 256 on. Attention scores then depend only on position
    differences, which tests/test_attention.py pins.

    Args:
        x: ``[b, n, heads, dk]`` with ``dk`` even.
        pos: ``[b, n]`` int64 positions.
        base: the rotary base, finite and above 1 (``configs/*.yaml: model.rotary_base``).

    Returns:
        The same shape and dtype, rotated; ``pos = 0`` returns ``x`` exactly.

    Raises:
        ContractError: if ``pos`` is not ``int64[b, n]`` for the ``x`` given.
        ValueError: if ``dk`` is odd or ``base`` is not a finite number above 1 (NaN included).
    """
    dims: Dims = {}
    check_shape("x", x, ("b", "n", "heads", "dk"), dims)
    check_shape("pos", pos, ("b", "n"), dims, dtype=torch.long)
    dk = dims["dk"]
    if dk % 2 != 0:
        msg = f"dk must be even for rotary pairs, got {dk}"
        raise ValueError(msg)
    if not 1.0 < base < float("inf"):  # written so that NaN fails too
        msg = f"rotary base must be finite and above 1, got {base}"
        raise ValueError(msg)
    half = dk // 2
    exponent = torch.arange(0, half, dtype=torch.float64, device=x.device) * (2.0 / dk)
    inv_freq = base ** (-exponent)
    angle = pos.to(torch.float64)[:, :, None] * inv_freq  # [b, n, half]
    cos = angle.cos().to(x.dtype)[:, :, None, :]
    sin = angle.sin().to(x.dtype)[:, :, None, :]
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class WindowedAttention(nn.Module):
    """One attention sublayer: projections, rotary, blocked windowed softmax, output projection.

    Parameters are ``2.5 d^2`` for ``kv_groups = 4`` (q and o at ``d^2``, k and v at ``d^2/4``),
    no biases. Call ``forward`` directly (the generic module call is untyped).
    """

    def __init__(
        self,
        d: int,
        heads: int,
        kv_groups: int,
        head_dim: int,
        window: int,
        rotary_base: float,
        generator: torch.Generator | None = None,
    ) -> None:
        """Build the sublayer; ``generator`` makes the initialization reproducible.

        Raises:
            ValueError: if ``heads`` is not a multiple of ``kv_groups``, ``window < 1`` or
                ``rotary_base <= 1``.
        """
        super().__init__()
        if heads % kv_groups != 0:
            msg = f"heads ({heads}) must be divisible by kv_groups ({kv_groups})"
            raise ValueError(msg)
        if window < 1:
            msg = f"window must be at least 1, got {window}"
            raise ValueError(msg)
        if not 1.0 < rotary_base < float("inf"):  # NaN fails this as well
            msg = f"rotary base must be finite and above 1, got {rotary_base}"
            raise ValueError(msg)
        self.heads: int = heads
        self.kv_heads: int = heads // kv_groups
        self.head_dim: int = head_dim
        self.window: int = window
        self.rotary_base: float = rotary_base
        std = 1.0 / math.sqrt(d)
        kv = self.kv_heads * head_dim
        self.w_q: nn.Parameter = nn.Parameter(
            torch.randn(heads * head_dim, d, generator=generator) * std
        )
        self.w_k: nn.Parameter = nn.Parameter(torch.randn(kv, d, generator=generator) * std)
        self.w_v: nn.Parameter = nn.Parameter(torch.randn(kv, d, generator=generator) * std)
        self.w_o: nn.Parameter = nn.Parameter(
            torch.randn(d, heads * head_dim, generator=generator) * std
        )

    def project(self, x: Tensor, pos: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project ``x`` to rotated ``q``, rotated ``k`` and ``v``.

        Args:
            x: ``[b, n, d]``.
            pos: ``[b, n]`` int64, consecutive within the chunk.

        Returns:
            ``q [b, n, h, dk]``, ``k [b, n, hkv, dk]``, ``v [b, n, hkv, dk]``.

        Raises:
            ContractError: on a shape mismatch or non-consecutive positions.
        """
        dims: Dims = {}
        check_shape("x", x, ("b", "n", "d"), dims)
        check_shape("pos", pos, ("b", "n"), dims, dtype=torch.long)
        if pos.shape[1] > 1:
            consecutive = (pos[:, 1:] - pos[:, :-1] == 1).all()
            require(consecutive, "pos: positions must be consecutive within a chunk")
        b, n = x.shape[0], x.shape[1]
        q = F.linear(x, self.w_q).view(b, n, self.heads, self.head_dim)
        k = F.linear(x, self.w_k).view(b, n, self.kv_heads, self.head_dim)
        v = F.linear(x, self.w_v).view(b, n, self.kv_heads, self.head_dim)
        return rotary(q, pos, self.rotary_base), rotary(k, pos, self.rotary_base), v

    def output(self, o: Tensor) -> Tensor:
        """Merge heads and apply the output projection: ``[b, n, h, dk] -> [b, n, d]``."""
        return F.linear(o.flatten(2), self.w_o)

    def attend(
        self, q: Tensor, k: Tensor, v: Tensor, pos: Tensor, cache: KVCache
    ) -> tuple[Tensor, KVCache]:
        """Windowed softmax over the cache and the chunk from projected ``q, k, v``; new cache.

        ``forward`` is ``project`` then this; the column calls the two apart so that it can write
        a halted token's carried ``k, v`` in place of fresh ones (spec 4, copy-forward).

        Args:
            q: ``[b, n, h, dk]`` rotated queries.
            k: ``[b, n, hkv, dk]`` rotated keys.
            v: ``[b, n, hkv, dk]`` values.
            pos: ``[b, n]`` int64, consecutive within the chunk.
            cache: the window carried from the previous chunk of this pass.

        Returns:
            ``y [b, n, d]`` and the cache holding the newest ``window`` entries. A chunk of zero
            tokens returns ``[b, 0, d]`` and the cache unchanged, after the same validation.
        """
        dims: Dims = {"hkv": self.kv_heads, "dk": self.head_dim, "h": self.heads}
        check_shape("q", q, ("b", "n", "h", "dk"), dims)
        check_shape("k", k, ("b", "n", "hkv", "dk"), dims)
        check_shape("v", v, ("b", "n", "hkv", "dk"), dims)
        check_shape("pos", pos, ("b", "n"), dims, dtype=torch.long)
        check_shape("cache.k", cache.k, ("b", "m", "hkv", "dk"), dims)
        check_shape("cache.v", cache.v, ("b", "m", "hkv", "dk"), dims)
        check_shape("cache.pos", cache.pos, ("b", "m"), dims, dtype=torch.long)
        if cache.window != self.window:
            msg = f"cache window {cache.window} does not match the layer's {self.window}"
            raise ContractError(msg)
        n, w = dims["n"], self.window
        if n == 0:
            return q.new_zeros(dims["b"], 0, self.w_o.shape[0]), cache
        if q.is_cuda and q.dtype in _WINDOW_KERNEL_DTYPES:
            # the same window over consecutive positions as one tiled kernel (kernels.window_attn);
            # the cache may be wider than q (an empty float32 cache promotes what is pushed onto
            # it), and the blocked path's attention runs in q's dtype under autocast too
            kpos = torch.cat([cache.pos, pos], dim=1)
            consecutive = (kpos[:, 1:] - kpos[:, :-1] == 1).all()
            require(consecutive, "attend: cache and chunk positions must be consecutive")
            keys = torch.cat([cache.k.to(q.dtype), k.to(q.dtype)], dim=1)
            vals = torch.cat([cache.v.to(q.dtype), v.to(q.dtype)], dim=1)
            y = self.output(window_attention(q, keys, vals, w))
            return y, kv_cache_push(cache, k, v, pos)
        outs: list[Tensor] = []
        for s in range(0, n, w):
            e = min(s + w, n)
            first = max(0, s - w + 1)
            if s < w:
                keys = torch.cat([cache.k, k[:, first:e]], dim=1)
                vals = torch.cat([cache.v, v[:, first:e]], dim=1)
                kpos = torch.cat([cache.pos, pos[:, first:e]], dim=1)
            else:
                keys, vals, kpos = k[:, first:e], v[:, first:e], pos[:, first:e]
            qpos = pos[:, s:e]
            allowed = (kpos[:, None, :] <= qpos[:, :, None]) & (
                kpos[:, None, :] > qpos[:, :, None] - w
            )
            block = F.scaled_dot_product_attention(
                q[:, s:e].transpose(1, 2),
                keys.transpose(1, 2),
                vals.transpose(1, 2),
                attn_mask=allowed[:, None, :, :],
                enable_gqa=True,
            )
            outs.append(block.transpose(1, 2))
        y = self.output(torch.cat(outs, dim=1))
        return y, kv_cache_push(cache, k, v, pos)

    @override
    def forward(self, x: Tensor, pos: Tensor, cache: KVCache) -> tuple[Tensor, KVCache]:
        """Attend over the cache and the chunk within the window; return output and new cache.

        Args:
            x: ``[b, n, d]``.
            pos: ``[b, n]`` int64, consecutive within the chunk.
            cache: the window carried from the previous chunk of this pass.

        Returns:
            ``y [b, n, d]`` and the cache holding the newest ``window`` entries (see ``attend``).
        """
        q, k, v = self.project(x, pos)
        return self.attend(q, k, v, pos, cache)
