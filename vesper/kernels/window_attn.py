"""Sliding-window attention on CUDA: flash-linear-attention's windowed kernel behind our contract.

PRIMITIVE ``window_attention(q, keys, values, window)``: the outputs of the ``n`` newest positions
of ``m + n`` consecutive key positions (the cache's ``m``, then the chunk's ``n``), each query at
position ``i`` attending the keys at ``[i - window + 1, i]``, scale ``1 / sqrt(dk)``, grouped KV
heads; what ``WindowedAttention.attend``'s blocked ``scaled_dot_product_attention`` with its
position mask computes, as fla-core 0.5.2's ``parallel_attn`` (MIT, the ``gpu`` extra; tiled
online softmax that skips the blocks outside the window, no atomics). The kernel takes equal
query and key lengths, so the cache's ``m`` positions get zero queries whose outputs are dropped
(they read nothing, their gradient is zero). Measured 2026-10-21 on the RTX 5090 at the recipe's
shape (b 16, chunk 1024, cache 512, 8 heads, 2 KV heads, dk 64): 1.46 ms forward and backward
against 2.27 ms; outputs and gradients within 2-4e-3 relative of the blocked path (two bf16
attentions with their own roundings).

For: ``WindowedAttention.attend`` on CUDA in bfloat16 or float16 with consecutive positions (the
caller checks them). Not for: float32 or the CPU (the blocked path stays the reference there and
in every test), or positions with gaps (the window here counts indices, not positions).
"""

import math
import warnings

import torch
from torch import Tensor


def window_attention(q: Tensor, keys: Tensor, values: Tensor, window: int) -> Tensor:
    """Outputs ``[b, n, h, dk]`` of ``q`` over ``keys, values [b, m + n, hkv, dk]``."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Flash Attention is not installed", category=ImportWarning
        )
        from fla.ops.attn.parallel import parallel_attn  # noqa: PLC0415  # the gpu extra

    b, n, h, dk = q.shape
    m = int(keys.shape[1]) - n
    full = torch.cat([q.new_zeros(b, m, h, dk), q], dim=1) if m > 0 else q
    o = parallel_attn(full, keys, values, scale=1.0 / math.sqrt(dk), window_size=window)
    return o[:, m:]
