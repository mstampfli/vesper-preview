"""Embedding and head (spec 2): one tied matrix ``E`` of shape ``[V, d]``.

Input ``x0 = E[tokens]``; positions come from rotary inside attention, never from an added
table. Output ``logits = x E^T``. One matrix, two contracts, checked at the boundary. The
local next-token heads of local mode (``P_c``) are this same matrix by the spec, so nothing
else in the model may own a vocabulary-sized parameter. Token ids arrive as ``int32`` from the
data stream (its storage dtype) or as ``int64``; ``check_ids`` widens them here, the model's
integer boundary, so nothing downstream sees a narrow id.
"""

import math
from typing import override

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vesper.contracts import Dims, check_ids, check_shape


class TiedEmbedding(nn.Module):
    """``embed(int32|int64[b, n]) -> [b, n, d]`` and ``logits([b, n, d]) -> [b, n, V]``, one matrix.

    Call the two methods directly; ``forward`` is ``embed``.
    """

    def __init__(self, vocab: int, d: int, generator: torch.Generator | None = None) -> None:
        """Build the matrix with rows of norm about one (``std = 1/sqrt(d)``).

        Raises:
            ValueError: if ``vocab`` or ``d`` is below 1.
        """
        super().__init__()
        if vocab < 1 or d < 1:
            msg = f"vocab ({vocab}) and d ({d}) must be at least 1"
            raise ValueError(msg)
        self.vocab: int = vocab
        self.d: int = d
        self.weight: nn.Parameter = nn.Parameter(
            torch.randn(vocab, d, generator=generator) / math.sqrt(d)
        )

    def embed(self, tokens: Tensor) -> Tensor:
        """Look up token rows: ``int32|int64[b, n] -> [b, n, d]``.

        The gradient of a lookup reaches exactly the rows looked up, once per use; the output
        projection is the other path to the same matrix (tests pin both separately).

        Raises:
            ContractError: on a wrong shape, a dtype other than int32 or int64, or an id outside
                ``[0, V)``.
        """
        return F.embedding(check_ids("tokens", tokens, ("b", "n"), {}, self.vocab), self.weight)

    def logits(self, x: Tensor) -> Tensor:
        """Project onto the vocabulary with the same matrix: ``[b, n, d] -> [b, n, V]``."""
        dims: Dims = {"d": self.d}
        check_shape("x", x, ("b", "n", "d"), dims)
        return F.linear(x, self.weight)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Same as ``embed``."""
        return self.embed(tokens)
