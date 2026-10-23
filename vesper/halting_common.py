"""What the halting drivers share: identity digests, warm starts, evaluation, depth statistics.

``vesper.halting_experiment`` (the A-D comparison) and ``vesper.halting_calibration`` (the
stopping-head calibration, docs/reviews/2026-10-18-halting-exploration-calibration.md) start
from the same kind of source, digest what they consume the same way, evaluate on the same CPU
reference copy and summarize depth with the same statistics: one implementation of each.
"""

import copy
import hashlib
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import cast

import torch
from torch import Tensor, nn

from vesper.column import scan_of, set_scan
from vesper.config import VesperConfig, parse_bytes
from vesper.data import Chunk
from vesper.model import Vesper
from vesper.synthetic import CLASSES

CPU = torch.device("cpu")
EVAL_SCAN = "reference"  # every probe and evaluation: a CPU model with this scan
# a run's recipe snapshot, named relative to its result.json: a copied run directory reads
# without the creator's paths (maintainer B's re-review of 9044238)
SNAPSHOT = "config.yaml"
SNAPSHOT_WEIGHTS = re.compile(r"after-\d{5,}\.pt")  # weight snapshots (144000 has six digits)


def chunk_tensors(chunk: Chunk) -> list[Tensor]:
    """The chunk's tensors in one fixed order.

    What a run digests as consumed, and what a captured step takes as its static inputs.
    """
    c = chunk
    return [c.tokens, c.targets, c.valid, c.doc_start, c.segment, c.loss_mask, c.pos]


def chained_digest(previous: str, tensors: Sequence[Tensor]) -> str:
    """sha256 over ``previous`` and each tensor's dtype, shape and bytes, in order (hex).

    Chained step by step over the batch tensors a run consumes (``consumed_digest``, maintainer B
    7ad7211: hash what is consumed, not class counts), over random draws, and over weights in
    ``state_dict`` order (``initial_state_digest``). Two runs agree on a digest only if they
    agree on every byte before it.
    """
    h = hashlib.sha256(previous.encode())
    for tensor in tensors:
        c = tensor.detach().to(CPU).contiguous()
        h.update(f"{c.dtype}{tuple(c.shape)}".encode())
        raw = c.reshape(-1).view(torch.uint8).numpy()  # type: ignore[misc]  # ndarray is generic over Any
        h.update(raw.tobytes())  # type: ignore[misc]
    return h.hexdigest()


def evaluation_model(model: Vesper) -> Vesper:
    """The model as every evaluation sees it: on the CPU with the reference scan.

    A CPU model with the reference scan is returned itself (CPU runs evaluate exactly as
    before); any other is copied, so training continues untouched on its device.
    """
    if next(model.parameters()).device == CPU and scan_of(model) == EVAL_SCAN:
        return model
    copy_ = copy.deepcopy(model).to(CPU)
    _ = set_scan(copy_, EVAL_SCAN)
    return copy_


class evaluating:  # noqa: N801  # a context manager, named as the statement reads
    """``with evaluating(model) as m``: ``evaluation_model(model)`` in eval mode, no gradients.

    The mode is restored on exit. Every measurement goes through this, so none depends on the
    caller's mode. Corrected 2026-10-20: this said train mode adds noise to the recurrent read
    (``recurrence.state_read_noise``); that field is declared in the config and applied by no
    module, and no module reads the mode today, so the switch is defensive.
    """

    def __init__(self, model: Vesper) -> None:
        """Choose the evaluation model; nothing changes until ``__enter__``."""
        self.model: Vesper = evaluation_model(model)
        self.was_training: bool = self.model.training
        self.no_grad: torch.no_grad = torch.no_grad()

    def __enter__(self) -> Vesper:
        """Eval mode and no gradients."""
        _ = self.model.eval()
        self.no_grad.__enter__()
        return self.model

    def __exit__(self, *exc: object) -> None:
        """Gradients and the caller's mode back."""
        self.no_grad.__exit__(None, None, None)
        _ = self.model.train(self.was_training)


def load_weights(model: nn.Module, weights: Mapping[str, Tensor]) -> None:
    """Load ``weights`` into ``model`` after checking them against it, all before any change.

    Raises:
        ValueError: on a missing or unexpected name, a shape or dtype other than the model's
            (``load_state_dict`` would cast a dtype silently), or a non-finite value.
    """
    own = cast("dict[str, Tensor]", model.state_dict())
    missing, unexpected = sorted(own.keys() - weights.keys()), sorted(weights.keys() - own.keys())
    if missing or unexpected:
        msg = f"init weights do not fit the model: missing {missing}, unexpected {unexpected}"
        raise ValueError(msg)
    for name, tensor in own.items():
        given = weights[name]
        if given.shape != tensor.shape or given.dtype != tensor.dtype:
            msg = (
                f"init weights: {name} is {given.dtype}{tuple(given.shape)}, "
                f"the model's {tensor.dtype}{tuple(tensor.shape)}"
            )
            raise ValueError(msg)
        if given.is_floating_point() and not bool(torch.isfinite(given).all()):
            msg = f"init weights: {name} has a non-finite value"
            raise ValueError(msg)
    _ = cast("object", model.load_state_dict(weights, strict=True))


def _init_sections(cfg: VesperConfig) -> dict[str, object]:
    """The config sections that say what a run's weights compute, by name.

    A warm start must agree with its source run in these; the training sections (schedule,
    mix, objective) may differ on purpose.
    """
    return {"model": cfg.model, "recurrence": cfg.recurrence}


@dataclass(frozen=True, slots=True)
class WarmStart:
    """A warm start's weights as loaded, its source run, and what a result records."""

    state: dict[str, Tensor]
    directory: Path  # the source run's directory
    recipe: bytes  # the source run's config snapshot, as read
    source_config: VesperConfig  # parsed from ``recipe``
    record: dict[str, object]


def read_warm_start(path: Path, cfg: VesperConfig | None = None) -> WarmStart:
    """The weights at ``path``, a run's ``model.pt`` or ``weights/after-NNNNN.pt``.

    The source run's config snapshot (``SNAPSHOT`` in its run directory) is read and parsed;
    given a ``cfg``, it must agree with it in ``_init_sections``. The file is read once: the
    tensors are loaded (CPU, weights only) from the same bytes the recorded sha256 digests.

    Raises:
        ValueError: for a path that is not a run's weights, a source run whose config differs
            in ``_init_sections``, or a file that is not a mapping of names to tensors.
        OSError: when the file or the source run's snapshot cannot be read.
    """
    if path.name == "model.pt":
        source = path.parent
    elif path.parent.name == "weights" and SNAPSHOT_WEIGHTS.fullmatch(path.name):
        source = path.parent.parent
    else:
        msg = f"{path}: not a run's model.pt or weights/after-NNNNN.pt"
        raise ValueError(msg)
    recipe = (source / SNAPSHOT).read_bytes()
    source_cfg = parse_bytes(recipe, str(source / SNAPSHOT))
    if cfg is not None:
        ours = _init_sections(cfg)
        for section, theirs in _init_sections(source_cfg).items():
            if theirs != ours[section]:
                msg = f"{path}: its run's {section} config differs from this run's"
                raise ValueError(msg)
    raw = path.read_bytes()
    loaded = cast("object", torch.load(io.BytesIO(raw), map_location=CPU, weights_only=True))
    if not isinstance(loaded, dict) or not all(
        isinstance(k, str) and isinstance(v, Tensor)  # type: ignore[misc]  # torch's Tensor class carries Any
        for k, v in cast("dict[object, object]", loaded).items()
    ):
        msg = f"{path}: not a mapping of names to tensors"
        raise ValueError(msg)
    state = cast("dict[str, Tensor]", loaded)
    record: dict[str, object] = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "state_digest": chained_digest("", list(state.values())),
        "source_config_sha256": hashlib.sha256(recipe).hexdigest(),
    }
    return WarmStart(state, source, recipe, source_cfg, record)


def plain(value: Tensor) -> object:
    """A tensor as nested lists of floats, for a JSON record (``tolist`` is untyped)."""
    if value.dim() == 0:
        return float(value.item())
    return [plain(v) for v in value.unbind()]


def version_or_none(distribution: str) -> str | None:
    """The installed version of ``distribution``, or None when it is not installed."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def entropy(counts: Tensor) -> float:
    """Entropy in nats of a histogram (zero for an empty one)."""
    total = float(counts.sum())
    if total == 0.0:
        return 0.0
    p = counts.double() / total
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def conditional_entropy(histogram: Tensor) -> tuple[float, float]:
    """``(H(D), H(D | row))`` of a ``[rows, depths]`` histogram, in nats."""
    total = float(histogram.sum())
    marginal = entropy(histogram.sum(0))
    conditional = sum(
        float(row.sum()) / total * entropy(row) for row in histogram.unbind() if total
    )
    return marginal, conditional


def class_accuracy(counts: Tensor) -> dict[str, float | None]:
    """Per-class and overall accuracy from ``[C, 2]`` counts; None for a class never seen.

    None and not NaN: the record is JSON, which has no NaN (Python writes a non-standard token
    and a NaN never compares equal to itself after reading it back).
    """
    out: dict[str, float | None] = {
        name: float(counts[i, 0]) / float(counts[i, 1]) if float(counts[i, 1]) else None
        for i, name in enumerate(CLASSES)
    }
    out["overall"] = float(counts[:, 0].sum()) / float(counts[:, 1].sum())
    return out
