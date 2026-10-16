from torch import Tensor

def chunk_delta_rule(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    scale: float | None = None,
    initial_state: Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_cpu: Tensor | None = None,
    *,
    chunk_size: int = ...,
) -> tuple[Tensor, Tensor | None]: ...
