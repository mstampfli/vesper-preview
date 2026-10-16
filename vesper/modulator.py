"""The modulator: one scalar per token, the normalized surprise (spec component 7).

``m_t = (l_t - mu) / sigma`` where ``l_t`` is the loss of predicting token ``t`` and ``mu``,
``sigma`` are exponential moving averages with a half-life of ``half_life_tokens`` per stream.
One ``(mu, E[l^2])`` pair is shared by all streams: a chunk's tokens are normalized with the
pre-chunk statistics, then the pair advances once by the chunk's statistics. Before the first
chunk there is no baseline, so ``m`` is zero for that chunk. Clipping to the configured range
belongs to the learning rules (learning.py), not here: the ledger stores the raw salience.
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from vesper.contracts import Dims, check_shape

_SIGMA_FLOOR = 1e-3
_HALF = 0.5


@dataclass(frozen=True, slots=True)
class ModulatorState:
    """The single EMA pair and the token count that says whether it is warm."""

    mean: Tensor
    mean_sq: Tensor
    tokens: int


def init_state(device: torch.device | None = None) -> ModulatorState:
    """A cold modulator: no tokens seen, zero statistics.

    Args:
        device: where the two scalars live.

    Returns:
        The cold state.
    """
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return ModulatorState(mean=zero, mean_sq=zero.clone(), tokens=0)


def normalize(state: ModulatorState, loss: Tensor) -> Tensor:
    """Read the modulator for a chunk without advancing it.

    Args:
        state: the current statistics.
        loss: ``[b, n]`` per-token losses, one row per stream.

    Returns:
        ``m: [b, n]``; all zeros while the state is cold.
    """
    dims: Dims = {}
    check_shape("loss", loss, ("b", "n"), dims)
    if state.tokens == 0:
        return torch.zeros_like(loss)
    var = (state.mean_sq - state.mean * state.mean).clamp_min(0.0)
    sigma = var.sqrt().clamp_min(_SIGMA_FLOOR)
    return (loss - state.mean) / sigma


def update(
    state: ModulatorState, loss: Tensor, half_life_tokens: int
) -> tuple[Tensor, ModulatorState]:
    """Read the modulator for a chunk, then advance the statistics by that chunk.

    Args:
        state: the current statistics.
        loss: ``[b, n]`` per-token losses, one row per stream; each stream advances by ``n`` tokens.
        half_life_tokens: the EMA half-life per stream (``modulator.half_life_tokens`` in configs/).

    Returns:
        ``(m, state')`` with ``m: [b, n]``.

    Raises:
        ValueError: if ``half_life_tokens`` is below 1.
    """
    if half_life_tokens < 1:
        msg = f"half_life_tokens must be at least 1, got {half_life_tokens}"
        raise ValueError(msg)
    m = normalize(state, loss)
    n = loss.shape[1]
    chunk_mean = loss.detach().float().mean()
    chunk_mean_sq = (loss.detach().float() ** 2).mean()
    if state.tokens == 0:
        return m, ModulatorState(mean=chunk_mean, mean_sq=chunk_mean_sq, tokens=n)
    keep = math.pow(_HALF, n / half_life_tokens)
    new_mean = keep * state.mean + (1.0 - keep) * chunk_mean
    new_mean_sq = keep * state.mean_sq + (1.0 - keep) * chunk_mean_sq
    return m, ModulatorState(mean=new_mean, mean_sq=new_mean_sq, tokens=state.tokens + n)
