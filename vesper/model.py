"""The model assembled (spec 2 to 4): tied embedding, the loop of columns, final norm, tied head.

``x0 = E[tokens]``; the loop runs ``T_max`` passes of ``C`` columns with halting and copy-forward;
the halting-weighted output is RMS-normalized and projected onto the vocabulary with the same
matrix ``E``. This is the wake forward pass in e2e mode. Not built yet, and said so: the sparse
retrieval sublayer (3c), the ledger (5), dusk, and local mode; their registry rows stay
specified. ``wake_objective`` joins the model's outputs to a data ``Chunk`` so the loss skips the
positions the data marks (padding and document boundaries).
"""

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn

from vesper.column import (
    Column,
    ColumnShape,
    ColumnState,
    Modulation,
    RMSNorm,
    column_shape,
    init_column_state,
)
from vesper.config import VesperConfig
from vesper.contracts import Dims, check_shape, require
from vesper.data import Chunk
from vesper.embedding import TiedEmbedding
from vesper.exits import (
    exit_log_probs,
    expected_pass_loss,
    hazard_logits,
    kl_uniform_aggregate,
    kl_uniform_local,
)
from vesper.kernels.nll import linear_nll
from vesper.learning import LossWeights, check_mode, loss_weights, wake_loss, wake_total
from vesper.loop import ExitRule, HaltingHead, Loop, LoopResult

# the halting experiment's all-pass conditions (docs/reviews/2026-10-15-halting-experiment-plan.md):
# B fixed uniform exit weights, C a per-token KL to uniform, D the batch-mean KL to uniform
ALL_PASS_CONDITIONS = ("fixed", "local", "aggregate")


class Vesper(nn.Module):
    """Embedding, loop, final norm and head from one validated configuration."""

    def __init__(self, cfg: VesperConfig, generator: torch.Generator | None = None) -> None:
        """Build every part from ``cfg``; ``generator`` makes the initialization reproducible."""
        super().__init__()
        shape = column_shape(cfg)
        m = cfg.model
        self.embedding: TiedEmbedding = TiedEmbedding(m.vocab, m.d, generator)
        columns = [Column(shape, generator) for _ in range(m.columns)]
        self.loop: Loop = Loop(columns, HaltingHead(m.d, generator), m.d, m.t_max, m.halt_epsilon)
        self.norm_out: RMSNorm = RMSNorm(m.d)
        self.shape: ColumnShape = shape
        self.columns: int = m.columns
        self.t_max: int = m.t_max

    def init_states(
        self, b: int, dtype: torch.dtype, device: torch.device, *, ring: bool = False
    ) -> list[ColumnState]:
        """Cold per-pass states for every column, for ``b`` streams (``ring``: decoding windows)."""
        return [
            init_column_state(self.shape, b, self.t_max, dtype, device, ring=ring)
            for _ in range(self.columns)
        ]

    @override
    def forward(
        self,
        tokens: Tensor,
        pos: Tensor,
        states: Sequence[ColumnState],
        modulation: Modulation | None = None,
        passes: int | None = None,
        *,
        recompute: bool = False,
    ) -> tuple[Tensor, LoopResult]:
        """One chunk: ``int32|int64[b, n]`` tokens to ``logits [b, n, V]`` and the loop's result.

        Args:
            tokens: the chunk's input ids.
            pos: ``int64[b, n]`` absolute stream positions, consecutive within the chunk.
            recompute: the loop's activation recomputation for the backward pass (``Loop.forward``).
            states: one ``ColumnState`` per column, carried from the previous chunk.
            modulation: the modulator's signal and strength at inference, or None.
            passes: the loop's pass count for this chunk (``T_max`` annealed), or None for all.

        Returns:
            ``(logits, result)``; ``result.states`` are the states to carry into the next chunk.
        """
        result = self.run(tokens, pos, states, modulation, passes, recompute=recompute)
        return self.head(result.y), result

    def run(
        self,
        tokens: Tensor,
        pos: Tensor,
        states: Sequence[ColumnState],
        modulation: Modulation | None = None,
        passes: int | None = None,
        *,
        recompute: bool = False,
        exit_rule: ExitRule | None = None,
    ) -> LoopResult:
        """The forward without the head: ``forward`` is ``run`` followed by ``head``.

        The trainer's path: the objective then applies the head to the loss positions in slices
        (``objective``) instead of materializing ``[b, n, V]`` logits for every position.
        ``exit_rule`` is the loop's halting policy (None: ACT).
        """
        return self.loop.forward(
            self.embedding.embed(tokens),
            pos,
            states,
            modulation,
            passes,
            recompute=recompute,
            exit_rule=exit_rule,
        )

    def position_losses(self, y: Tensor, targets: Tensor) -> Tensor:
        """Next-token losses ``[...]`` of outputs ``y [..., d]`` against ``targets [...]``.

        ``token_losses`` through this model's head, without ``[..., V]`` float32 logits: what
        every evaluation scores (``evals``, ``halting_text``), one call per chunk or depth.
        """
        return token_losses(self, y, targets)

    def head(self, y: Tensor) -> Tensor:
        """Halting-weighted outputs ``[b, n, d]`` to logits ``[b, n, V]``: output norm, then ``E``.

        The last step of ``forward``; ``objective`` applies it to the loss positions in slices.
        """
        return self.embedding.logits(self.norm_out.forward(y))

    def objective(
        self,
        chunk: Chunk,
        result: LoopResult,
        cfg: VesperConfig,
        weights: LossWeights | None = None,
        *,
        slices: int = 1,
        keep_logits: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The wake objective from the loop's result, the head applied to the loss positions only.

        The same value and gradients as ``wake_objective(forward logits, ...)``: the positions
        ``chunk.loss_mask`` marks are selected from ``result.y`` first (a masked position's value
        has no path into the loss or its gradient), split into ``slices`` consecutive ranges, and
        each range is scored against its targets by ``token_losses``; the per-token losses are
        summed over the ranges and divided by the count, so the mean and the softmax over the
        full vocabulary per position are those of the unsliced objective. No range's logits are
        kept for the backward pass unless ``keep_logits`` (``token_losses``); ``slices`` bounds
        the one transient ``[N, V]`` buffer of a call (``b n / slices`` rows).

        Raises:
            ContractError: if no position carries a loss.
            ValueError: on ``slices < 1``.
        """
        check_mode(cfg.learning.mode)
        if slices < 1:
            msg = f"slices must be at least 1, got {slices}"
            raise ValueError(msg)
        device = result.y.device
        mask = chunk.loss_mask.to(device)
        require(mask.any(), "objective: no position in the chunk carries a loss")
        # static shapes for a captured step, as in all_pass_objective: every position is scored,
        # the rows and targets outside the mask are replaced before the head and the losses after
        # it, so nothing outside the loss positions reaches the value or a gradient
        rows, targets, keep = _loss_rows(result.y, chunk.targets.to(device), mask)
        width = -(-rows.shape[0] // slices)
        summed = torch.zeros((), dtype=torch.float32, device=device)
        for start in range(0, rows.shape[0], width):
            end = min(start + width, rows.shape[0])
            ranged = (rows[start:end], targets[start:end], keep[start:end])
            summed = summed + _slice_loss(self, *ranged, keep_logits=keep_logits)
        count = mask.sum()
        return wake_total(
            summed / count,
            weights=loss_weights(cfg) if weights is None else weights,
            ponder=torch.where(mask, result.ponder, 0.0).sum() / count,
        )

    def all_pass_objective(
        self,
        chunk: Chunk,
        result: LoopResult,
        *,
        condition: str,
        beta: float,
        slices: int = 1,
        keep_logits: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """The halting experiment's all-pass objective from a DENSE forward (``DenseRule``).

        Every pass's output is scored by the head at the loss positions (pass by pass, in
        ``slices`` ranges through ``token_losses``, whose logits the backward recomputes unless
        ``keep_logits``, so one range's vocabulary logits exist at a time by default); the
        exit distribution comes from the learned logits
        through ``hazard_logits`` and ``exit_log_probs``, in float64.
        ``condition``: ``fixed`` uses zero logits (exactly uniform weights, no head gradient);
        ``local`` and ``aggregate`` use the head's logits on the same per-pass outputs and add
        ``beta`` times the per-token or the batch-mean KL to uniform, over this chunk's loss
        positions (the caller owns the population: a split batch is not the batch).

        Returns the objective and its parts: ``expected_ntp``, ``kl`` (zero for ``fixed``),
        ``pass_ntp [T]`` (each pass's mean loss) and ``expected_depth``.

        Raises:
            ValueError: on an unknown condition or ``slices < 1``.
            ContractError: if no position carries a loss or a real input did not run every
                pass (the objective needs the dense forward).
        """
        if condition not in ALL_PASS_CONDITIONS:
            msg = f"condition must be one of {ALL_PASS_CONDITIONS}, got {condition!r}"
            raise ValueError(msg)
        if slices < 1:
            msg = f"slices must be at least 1, got {slices}"
            raise ValueError(msg)
        device = result.per_pass.device
        mask = chunk.loss_mask.to(device)
        require(mask.any(), "all_pass_objective: no position in the chunk carries a loss")
        t_max = result.per_pass.shape[0]
        # every real input, not only the scored ones: a context token that halted early would
        # have written copied-forward states the later supervised passes read (maintainer B's review
        # of 49c92cf, finding 2)
        dense = ((result.n_iters == t_max) | ~chunk.valid.to(device)).all()
        require(dense, "all_pass_objective needs a dense forward: a real input halted early")
        # Static shapes, so the step can be captured as a CUDA graph: every position is scored
        # (a position without a loss gets target 0, a valid id) and the reductions below keep
        # exactly the loss positions with where (vesper.exits). The values and gradients are the
        # selected positions' alone, as when they were selected by the mask.
        b, n = mask.shape[0], mask.shape[1]
        passes, targets, _ = _loss_rows(result.per_pass, chunk.targets.to(device), mask)
        width = -(-(b * n) // slices)
        losses = torch.stack(
            [
                torch.cat(
                    [
                        token_losses(
                            self,
                            rows[s : s + width],
                            targets[s : s + width],
                            keep_logits=keep_logits,
                        )
                        for s in range(0, b * n, width)
                    ]
                )
                for rows in passes.unbind()
            ]
        ).view(t_max, b, n)  # at least float32
        if condition == "fixed":
            learned = torch.zeros(t_max, b, n, dtype=torch.float32, device=device)
        else:
            head = self.loop.halting_head()
            per_pass = passes.view(t_max, b, n, -1)  # the replaced rows: no gradient outside
            learned = torch.stack([head.logit(per_pass[t], t) for t in range(t_max)])
        log_p = exit_log_probs(hazard_logits(learned))  # [T, b, n], float64
        expected = expected_pass_loss(log_p, losses.to(log_p.dtype), mask)
        if condition == "local":
            kl = kl_uniform_local(log_p, mask)
        elif condition == "aggregate":
            kl = kl_uniform_aggregate(log_p, mask)
        else:
            kl = torch.zeros((), dtype=log_p.dtype, device=device)
        depth = torch.arange(1, t_max + 1, dtype=log_p.dtype, device=device)
        count = mask.sum()
        pass_ntp = torch.where(mask, losses.detach(), 0.0).sum((1, 2)) / count
        per_token_depth = torch.where(mask, (log_p.exp() * depth[:, None, None]).sum(0), 0.0)
        parts = {
            "expected_ntp": expected.detach(),
            "kl": kl.detach(),
            "pass_ntp": pass_ntp,
            "expected_depth": (per_token_depth.sum() / count).detach(),
        }
        return expected + beta * kl, parts


def count_parameters(model: nn.Module) -> int:
    """Parameters of ``model``, a shared tensor counted once (the tied embedding is one matrix)."""
    return sum(p.numel() for p in model.parameters())


def _loss_rows(y: Tensor, targets: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """``y [..., b, n, d]`` as rows ``[..., b*n, d]``, targets ``[b*n]`` and the mask ``[b*n]``.

    The rows and targets outside ``mask`` are replaced by zero (a valid id) BEFORE any arithmetic,
    so the head never sees them: a non-finite hidden state outside the loss positions reaches
    neither the value nor a gradient, as when those positions were selected away. The objectives'
    one construction of their static-shape inputs.
    """
    b, n, d = y.shape[-3], y.shape[-2], y.shape[-1]
    keep = mask.reshape(b * n)
    rows = torch.where(keep[:, None], y.reshape(*y.shape[:-3], b * n, d), 0.0)
    return rows, torch.where(mask, targets, 0).reshape(b * n), keep


def _slice_loss(
    model: "Vesper", y: Tensor, targets: Tensor, keep: Tensor, *, keep_logits: bool
) -> Tensor:
    """The summed per-token loss of the kept rows of one range."""
    losses = token_losses(model, y, targets, keep_logits=keep_logits)
    return torch.where(keep, losses, 0.0).sum()


def token_losses(
    model: "Vesper", y: Tensor, targets: Tensor, *, keep_logits: bool = False
) -> Tensor:
    """PRIMITIVE per-token losses ``[...]`` (at least float32) of positions through the head.

    ``y [..., d]`` are positions (one range ``[N, d]``, or a chunk's ``[b, n, d]``) and
    ``targets [...]`` their targets, the same leading shape: the head's norm,
    then ``kernels.nll.linear_nll`` over the tied embedding, which keeps no vocabulary logits for
    the backward pass (it recomputes one call's logits there), so with ``T`` passes no ``T``
    vocabulary intermediates are held (maintainer B's review of 49c92cf, finding 3) and a caller's
    slices bound the one transient ``[N, V]`` buffer of a call. The logits are never copied to
    float32: bf16 under CUDA autocast as the matmul gives them, float32 or float64 otherwise (a
    plain ``.float()`` once demoted a float64 model's logits and failed a slice test on CI).

    For: every objective's loss of the head (``objective``, ``all_pass_objective``,
    ``joint_update.label_nll``) and the evaluations, one range at a time. Not for: a mask; every
    row is scored, the caller selects or zeroes afterwards.
    """
    # the declared contract before any flattening: targets exactly y's leading shape (maintainer B's
    # review of 571b5f9: y [2, 3, d] with targets [3, 2] was flattened and accepted)
    lead = tuple(f"lead{i}" for i in range(y.dim() - 1))
    dims: Dims = {}
    check_shape("y", y, (*lead, "d"), dims)
    check_shape("targets", targets, lead, dims)
    rows = y.reshape(1, -1, y.shape[-1])
    x = model.norm_out.forward(rows)[0]
    flat = targets.reshape(-1)
    losses = linear_nll(x, model.embedding.weight, flat, keep_logits=keep_logits)
    return losses.view(targets.shape)


def wake_objective(
    logits: Tensor,
    chunk: Chunk,
    result: LoopResult,
    cfg: VesperConfig,
    weights: LossWeights | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """The e2e wake loss of one chunk over the positions the data marks as carrying a loss.

    ``chunk.loss_mask`` excludes padding and the position whose target opens the next document;
    the ponder term is averaged over the same positions, with the configured weights, or with
    ``weights`` when the caller's schedule sets them (the ``lambda_h`` ramp).

    Raises:
        ContractError: if no position carries a loss.
    """
    mask = chunk.loss_mask
    require(mask.any(), "wake_objective: no position in the chunk carries a loss")
    return wake_loss(
        logits,
        chunk.targets.to(logits.device),
        cfg.learning.mode,
        weights=loss_weights(cfg) if weights is None else weights,
        ponder=result.ponder[mask.to(logits.device)].mean(),
        mask=mask.to(logits.device),
    )
