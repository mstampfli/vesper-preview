# Precise local stubs for `fla.ops.attn.parallel` (vesper/kernels/window_attn.py calls it).
from torch import Tensor

def parallel_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor | None = None,
    scale: float | None = None,
    window_size: int | None = None,
    cu_seqlens: Tensor | None = None,
    chunk_indices: Tensor | None = None,
    *,
    sink_bias: Tensor | None = None,
) -> Tensor: ...
