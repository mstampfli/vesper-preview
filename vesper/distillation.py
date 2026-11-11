"""Original-model distillation on cold, dense, matching-depth predictions.

The teacher owns a frozen copy, never optimizer parameters. Vocabulary intermediates
exist for one position slice of one pass, recomputed during backward. This is a
retention objective, not evidence that a trained model retains its source quality.
"""

import copy
import math
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint  # pyright: ignore[reportUnknownVariableType]

from vesper.column import scan_of, set_scan
from vesper.contracts import ContractError, Dims, check_shape, require
from vesper.data import Chunk
from vesper.graphs import SignatureGraph
from vesper.loop import DenseRule
from vesper.model import Vesper


def _row_kl(model: Vesper, teacher: Vesper, y: Tensor, old: Tensor, mask: Tensor) -> Tensor:
    # Static shapes, but masked NaNs are removed BEFORE norms, matmuls and softmax.
    rows = torch.where(mask[:, None], y, 0.0)
    original = torch.where(mask[:, None], old, 0.0)
    with torch.no_grad():
        logits = teacher.head(original[None])[0]
        dtype = torch.promote_types(logits.dtype, torch.float32)
        target = logits.to(dtype).log_softmax(-1)
    student = model.head(rows[None])[0].to(dtype).log_softmax(-1)
    divergence = (target.exp() * (target - student)).sum(-1)
    return torch.where(mask, divergence.double(), 0.0)


def depth_kl(
    model: Vesper,
    teacher: Vesper,
    passes: Tensor,
    original: Tensor,
    mask: Tensor,
    *,
    slices: int = 1,
) -> Tensor:
    """PRIMITIVE forward KL(teacher || student), float64 ``[T,b,n]``, zero off ``mask``.

    Matching dense outputs ``passes, original [T,b,n,d]`` use their respective heads;
    temperature is one. Only student outputs/head receive gradients. Used for the
    anchor and its validation distance, not a KL between stopping distributions or
    a different teacher depth. ``slices`` splits EACH pass's ``b*n`` positions, so
    no vocabulary intermediate scales with T, nor is one saved for backward.
    The checkpoint rebuilds both projections of that slice. Empty positions work.

    Raises:
        ContractError: on incompatible shapes, models, floating dtypes or devices.
        ValueError: on nonpositive slices.
    """
    dims: Dims = {"T": model.t_max, "d": model.shape.d}
    check_shape("passes", passes, ("T", "b", "n", "d"), dims)
    check_shape("original", original, ("T", "b", "n", "d"), dims)
    check_shape("mask", mask, ("b", "n"), dims, dtype=torch.bool)
    check_shape("student weight", model.embedding.weight, ("V", "d"), dims)
    check_shape("teacher weight", teacher.embedding.weight, ("V", "d"), dims)
    if slices < 1:
        msg = "slices must be positive"
        raise ValueError(msg)
    tensors = (passes, original, model.embedding.weight, teacher.embedding.weight)
    if (
        not passes.is_floating_point()
        or any(t.dtype != passes.dtype or t.device != passes.device for t in tensors)
        or mask.device != passes.device
        or teacher.t_max != dims["T"]
    ):
        msg = "matching floating dtype, device and teacher depth are required"
        raise ContractError(msg)
    if dims["b"] * dims["n"] == 0:
        return passes[..., 0].double() * 0.0
    width = -(-(dims["b"] * dims["n"]) // slices)
    flat_mask = mask.flatten()
    results: list[Tensor] = []
    for current_pass, original_pass in zip(
        passes.unbind(), original.detach().unbind(), strict=True
    ):
        current = current_pass.reshape(-1, dims["d"])
        previous = original_pass.reshape(-1, dims["d"])
        parts: list[Tensor] = []
        for start in range(0, current.shape[0], width):
            inputs = (
                current[start : start + width],
                previous[start : start + width],
                flat_mask[start : start + width],
            )
            value = (
                checkpoint(_row_kl, model, teacher, *inputs, use_reentrant=False)
                if torch.is_grad_enabled()
                else _row_kl(model, teacher, *inputs)
            )
            parts.append(value)
        results.append(torch.cat(parts).view(dims["b"], dims["n"]))
    return torch.stack(results)


class Anchor:
    """An owned frozen teacher copied from the checked source BEFORE student training.

    ``weight`` is explicit, finite and nonnegative. Zero can serve validation only;
    the training caller must skip it entirely. No source-path reload, shared storage,
    gradients, optimizer membership or persistent recurrent state in the teacher.
    """

    def __init__(self, model: Vesper, weight: float) -> None:
        """Copy the supplied source; refuse a nonfinite or negative weight first."""
        if not math.isfinite(weight) or weight < 0.0:
            msg = "anchor weight must be finite and nonnegative"
            raise ValueError(msg)
        self.weight: float = weight
        self.teacher: Vesper = copy.deepcopy(model)
        _ = self.teacher.requires_grad_(requires_grad=False)
        self.teacher.zero_grad(set_to_none=True)
        _ = self.teacher.eval()

    def to(self, device: torch.device, scan: str) -> None:
        """Move only the owned teacher and select the student's numerical scan."""
        _ = self.teacher.to(device)
        _ = set_scan(self.teacher, scan)

    def _outputs(self, tokens: Tensor, pos: Tensor) -> Tensor:
        with torch.no_grad():
            parameter = next(self.teacher.parameters())
            states = self.teacher.init_states(tokens.shape[0], parameter.dtype, parameter.device)
            return self.teacher.run(tokens, pos, states, exit_rule=DenseRule()).per_pass

    def loss(
        self,
        model: Vesper,
        passes: Tensor,
        tokens: Tensor,
        pos: Tensor,
        mask: Tensor,
        weights: Tensor,
        *,
        slices: int = 1,
    ) -> Tensor:
        """Weighted anchor scalar on ``tokens/pos/mask/weights [b,n]`` and ``passes [T,b,n,d]``.

        Mean over passes and rows of the weighted position sums, times ``weight``;
        text uses b/N weights, hence this is the pooled KL over labels and depths.
        Student dense outputs are supplied by the update, not computed twice.
        """
        dims: Dims = {}
        check_shape("tokens", tokens, ("b", "n"), dims)
        check_shape("pos", pos, ("b", "n"), dims, dtype=torch.int64)
        check_shape("mask", mask, ("b", "n"), dims, dtype=torch.bool)
        check_shape("weights", weights, ("b", "n"), dims)
        safe_weights = torch.where(mask, weights, 0.0)
        require(
            torch.isfinite(safe_weights).all() & (safe_weights >= 0).all(),
            "finite nonnegative anchor weights required",
        )
        original = self._outputs(tokens, pos)
        kl = depth_kl(model, self.teacher, passes, original, mask, slices=slices)
        return self.weight * (kl * safe_weights).sum(-1).mean()

    def profile(
        self,
        model: Vesper,
        chunks: Sequence[Chunk],
        *,
        slices: int = 1,
        graph: SignatureGraph | None = None,
    ) -> list[float]:
        """Pooled per-depth KL on these cold chunks, independent of the training weight.

        CUDA requires a graph; its key binds both model objects, scans, slicing and
        input signatures. Read each result before the next replay. Empty labels
        refuse; the caller's model mode is restored. No training/validation state
        is carried across chunks. The caller supplies only its admitted split.
        """
        parameter = next(model.parameters())
        if parameter.is_cuda and graph is None:
            msg = "CUDA anchor evaluation requires a graph"
            raise ValueError(msg)
        totals = torch.zeros(model.t_max, dtype=torch.float64)
        count = 0
        was_training = model.training
        _ = model.eval()
        try:
            with torch.no_grad():
                for chunk in chunks:
                    values = [
                        t.to(parameter.device) for t in (chunk.tokens, chunk.pos, chunk.loss_mask)
                    ]

                    def step(inputs: Sequence[Tensor]) -> list[Tensor]:
                        tokens, pos, mask = inputs
                        states = model.init_states(
                            tokens.shape[0], parameter.dtype, parameter.device
                        )
                        current = model.run(tokens, pos, states, exit_rule=DenseRule()).per_pass
                        original = self._outputs(tokens, pos)
                        kl = depth_kl(model, self.teacher, current, original, mask, slices=slices)
                        return [kl.sum((1, 2))]

                    key = (
                        id(model),
                        id(self.teacher),
                        scan_of(model),
                        scan_of(self.teacher),
                        slices,
                        *((tuple(t.shape), t.dtype, t.device) for t in values),
                    )
                    result = step(values) if graph is None else graph(key, step, values)
                    totals += result[0].cpu()
                    count += int(chunk.loss_mask.sum())
        finally:
            _ = model.train(was_training)
        if count == 0:
            msg = "anchor evaluation has no labels"
            raise ValueError(msg)
        return [float(v) / count for v in totals.unbind()]
