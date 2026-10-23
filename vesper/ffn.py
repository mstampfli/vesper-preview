"""Feed-forward sublayer (spec 3d): ``W_2 relu(W_1 x)^2`` with hidden width ``ffn_mult d``.

Reference implementation. Squared ReLU is chosen for activation sparsity; the sparse kernel
skips the rows of ``W_2`` whose hidden unit is zero, which this reference does not attempt.
``hidden`` is exposed so the sparsity can be measured (tests/test_ffn.py, eval).
"""

import math
from typing import override

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vesper.contracts import Dims, check_shape
from vesper.kernels.act import relu_square


class FFN(nn.Module):
    """``W_2 relu(W_1 x)^2``; ``8 d^2`` parameters at ``ffn_mult = 4``, no biases.

    Call ``forward`` directly (the generic module call is untyped).
    """

    def __init__(self, d: int, ffn_mult: int, generator: torch.Generator | None = None) -> None:
        """Build the sublayer; ``generator`` makes the initialization reproducible.

        Raises:
            ValueError: if ``ffn_mult`` is below 1.
        """
        super().__init__()
        if ffn_mult < 1:
            msg = f"ffn_mult must be at least 1, got {ffn_mult}"
            raise ValueError(msg)
        hidden = ffn_mult * d
        self.w1: nn.Parameter = nn.Parameter(
            torch.randn(hidden, d, generator=generator) / math.sqrt(d)
        )
        self.w2: nn.Parameter = nn.Parameter(
            torch.randn(d, hidden, generator=generator) / math.sqrt(hidden)
        )

    def hidden(self, x: Tensor) -> Tensor:
        """The hidden activations ``relu(W_1 x)^2``: ``[b, n, d] -> [b, n, ffn_mult d]``."""
        dims: Dims = {}
        check_shape("x", x, ("b", "n", "d"), dims)
        z = F.linear(x, self.w1)
        if z.is_cuda and z.dtype in (torch.float32, torch.bfloat16, torch.float16):
            return relu_square(z)  # one kernel each way, the same bits (kernels.act)
        h = F.relu(z)
        # h * h, not h.square(): autocast runs pow in float32 (a [b, n, 4d] float32 tensor and a
        # cast back for W_2, measured 2026-10-21 as ~17 GiB per step on the 5090), while a
        # product stays in the activation dtype; a product of two bfloat16 values is exact in
        # float32, so one rounding to bfloat16 gives the same bits, forward and backward
        return h * h

    @override
    def forward(self, x: Tensor) -> Tensor:
        """``[b, n, d] -> [b, n, d]``."""
        return F.linear(self.hidden(x), self.w2)
