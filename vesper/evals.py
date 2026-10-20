"""Evaluations that run at every checkpoint (spec: Evaluation suite), each a plain function.

Two of the suite's metrics exist with the first training loop: the held-out perplexity and the
iteration histogram. The others (needle at distance, session learning, dusk gain, joules) need
the ledger, dusk or the energy meter and keep their registry rows as specified.
"""

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from vesper.config import VesperConfig
from vesper.data import Chunk
from vesper.learning import ntp_loss
from vesper.loop import DenseRule
from vesper.model import Vesper


@dataclass(frozen=True)
class Evaluation:
    """Loss and depth measured on the same positions in one evaluation traversal.

    Counts index passes ``1..cfg.model.t_max``, including zeroes for unused passes.
    Raw integer counts and the loss sum retain their unrounded values for aggregation.
    """

    loss_sum: float
    tokens_with_loss: int
    iteration_counts: tuple[int, ...]

    @property
    def mean_loss(self) -> float:
        """Mean next-token loss; ``inf`` when no position carries a loss."""
        return self.loss_sum / self.tokens_with_loss if self.tokens_with_loss else math.inf

    @property
    def perplexity(self) -> float:
        """Exponential of the mean next-token loss, or ``inf`` without observations."""
        return math.exp(self.mean_loss)

    @property
    def iteration_histogram(self) -> list[float]:
        """Unrounded fractions of the measured positions at each pass."""
        return _fractions(self.iteration_counts)


def evaluate(model: Vesper, chunks: Sequence[Chunk], cfg: VesperConfig, passes: int) -> Evaluation:
    """Measure next-token loss and pass counts together over positions with a loss.

    ``chunks`` are consecutive chunks of one held-out stream (the same slots throughout), run
    with cold states carried from chunk to chunk and ``passes`` loop passes, no gradient, in
    float32 (even under an outer autocast). Every chunk runs once; its loss mask selects both
    losses and pass counts. Metric storage is O(T_max), independent of the stream length.
    Empty or fully masked inputs give zero sums/counts and infinite mean loss/perplexity.
    """
    device = next(model.parameters()).device
    total = 0.0
    count = 0
    depths = torch.zeros(cfg.model.t_max + 1, dtype=torch.int64)
    states = None
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        for chunk in chunks:
            b = chunk.tokens.shape[0]
            if states is None:
                states = model.init_states(b, torch.float32, device)
            logits, result = model.forward(
                chunk.tokens.to(device), chunk.pos.to(device), states, passes=passes
            )
            states = list(result.states)
            mask = chunk.loss_mask.to(device)
            depths += _iteration_counts([result.n_iters[mask]], cfg.model.t_max)
            if bool(mask.any().item()):
                per_token = ntp_loss(logits.float(), chunk.targets.to(device))
                total += float(per_token[mask].sum().item())
                count += int(mask.sum().item())
            # Do not retain one chunk's logits/pass outputs during the next forward.
            del logits, result
    return Evaluation(total, count, _pass_totals(depths))


@dataclass(frozen=True)
class StaticProfile:
    """The held-out next-token loss at every fixed depth ``1..T`` on the same positions."""

    loss_sums: tuple[float, ...]
    tokens_with_loss: int

    @property
    def mean_losses(self) -> list[float]:
        """Mean loss per depth; ``inf`` everywhere when no position carries a loss."""
        n = self.tokens_with_loss
        return [s / n if n else math.inf for s in self.loss_sums]


def static_profile(model: Vesper, chunks: Sequence[Chunk], cfg: VesperConfig) -> StaticProfile:
    """The loss if every token stopped at depth ``j``, for every ``j``, from one dense traversal.

    With copy-forward a token's pass-``j`` output reads only the states of passes ``1..j``, which
    earlier tokens wrote at those same passes, so pass ``j`` of a dense run (``DenseRule``, every
    token every pass) is exactly a run in which every token stops at ``j``. ``chunks`` are
    consecutive chunks of one held-out stream with cold states carried, as in ``evaluate``; no
    gradient, float32; one depth's logits exist at a time. For the halting pilot's phase-1 check
    (docs/reviews/2026-10-20-real-data-halting-pilot.md): does the loss fall with depth?
    """
    device = next(model.parameters()).device
    t_max = cfg.model.t_max
    totals = [0.0] * t_max
    count = 0
    states = None
    with torch.no_grad(), torch.autocast(device.type, enabled=False):
        for chunk in chunks:
            if states is None:
                states = model.init_states(chunk.tokens.shape[0], torch.float32, device)
            result = model.run(
                chunk.tokens.to(device),
                chunk.pos.to(device),
                states,
                passes=t_max,
                exit_rule=DenseRule(),
            )
            states = list(result.states)
            mask = chunk.loss_mask.to(device)
            if bool(mask.any().item()):
                targets = chunk.targets.to(device)
                for depth in range(t_max):
                    per_token = ntp_loss(model.head(result.per_pass[depth]).float(), targets)
                    totals[depth] += float(per_token[mask].sum().item())
                count += int(mask.sum().item())
            del result
    return StaticProfile(tuple(totals), count)


def perplexity(model: Vesper, chunks: Sequence[Chunk], cfg: VesperConfig, passes: int) -> float:
    """Compatibility float API for ``evaluate(...).perplexity``; one traversal."""
    return evaluate(model, chunks, cfg, passes).perplexity


def iteration_histogram(n_iters: Sequence[Tensor], t_max: int) -> list[float]:
    """The fraction of tokens that halted at each pass ``1..t_max``, over the given counts.

    ``n_iters`` are the loop's ``int32`` pass counts of the tokens to include (the caller selects
    the positions with a loss). An empty input gives all zeros.

    Raises:
        ValueError: if ``t_max`` is below 1.
    """
    counts = _iteration_counts(n_iters, t_max)
    return _fractions(_pass_totals(counts))


def _iteration_counts(n_iters: Iterable[Tensor], t_max: int) -> Tensor:
    """Shared integer accumulation; transfer only per-pass counts from an accelerator."""
    if t_max < 1:
        msg = f"t_max must be at least 1, got {t_max}"
        raise ValueError(msg)
    counts = torch.zeros(t_max + 1, dtype=torch.int64)
    for part in n_iters:
        flat = part.reshape(-1).to(dtype=torch.int64)
        counts += torch.bincount(flat, minlength=t_max + 1)[: t_max + 1].to(device="cpu")
    return counts


def _fractions(counts: Sequence[int]) -> list[float]:
    """Normalize exact pass counts without rounding the returned fractions."""
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [n / total for n in counts]


def _pass_totals(counts: Tensor) -> tuple[int, ...]:
    """Convert the CPU int64 bins for passes 1..T_max to exact Python integers."""
    return tuple(int(counts[t].item()) for t in range(1, counts.numel()))
