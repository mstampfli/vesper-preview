"""The joint update's memory options: sliced label NLL and loop recomputation (pilot phase 2).

Both default off, and off is the update every synthetic result was measured with (its tests
run unchanged); on, they change memory only: the same value and gradients up to the float
rounding of the head's batching.
"""

import math
from collections import Counter
from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from conftest import small_raw
from vesper.config import parse
from vesper.contracts import ContractError
from vesper.joint_update import (
    ROLLOUTS,
    Dense,
    label_nll,
    optimizer,
    release_backbone,
    start_head,
    update_loss,
    update_step,
)
from vesper.model import Vesper

T = 3
VOCAB = 96
FFN = 64  # small_raw: d 32, ffn_mult 2

Grads = list[Tensor | None]


def _model() -> tuple[Vesper, list[nn.Parameter], list[nn.Parameter]]:
    """A joint-mode model (head and backbone trainable) whose head stops some tokens early."""
    raw = small_raw(vocab=VOCAB)  # not the FFN width (d 32 x ffn_mult 2 = 64): counted apart
    raw["model"]["t_max"] = T
    raw["recurrence"]["state_read_noise"] = 0.0
    model = Vesper(parse(raw, "test"), torch.Generator().manual_seed(0))
    head = start_head(model)
    backbone = release_backbone(model)
    with torch.no_grad():  # mixed routing: some tokens stop before the last pass
        for p in head:
            _ = p.normal_(0.0, 0.5, generator=torch.Generator().manual_seed(3))
    return model, head, backbone


def _batch(e: int = 2, n: int = 8) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32)
    targets = torch.randint(1, 60, (e, n), generator=g, dtype=torch.int32)
    mask = torch.rand(e, n, generator=g) < 0.7
    mask[:, 0] = True
    valid = torch.ones(e, n, dtype=torch.bool)
    valid[-1, -2:] = False
    mask &= valid
    weights = torch.where(mask, e / float(mask.sum()), 0.0).double()
    unique = {
        "tokens": tokens,
        "targets": targets,
        "mask": mask,
        "valid": valid,
        "weights": weights,
    }
    batch = {k: torch.cat([v] * ROLLOUTS) for k, v in unique.items()}
    batch["pos"] = torch.arange(n).expand(ROLLOUTS * e, n).clone()
    batch["uniforms"] = torch.rand((T - 1, ROLLOUTS * e, n), generator=g, dtype=torch.float64)
    return batch


def _loss(
    model: Vesper, b: dict[str, Tensor], memory: tuple[bool, int] | None = None
) -> tuple[float, Grads]:
    """The update loss and its gradients; ``memory`` = (recompute, slices) passed explicitly,
    or None to call with NO memory arguments (the function's own defaults)."""
    e = int(b["tokens"].shape[0]) // ROLLOUTS
    dense = Dense(
        b["tokens"][:e], b["pos"][:e], b["targets"][:e], b["mask"][:e], b["weights"][:e], 0.1
    )
    batch = (b["tokens"], b["pos"], b["targets"], b["mask"], b["valid"], b["weights"])
    if memory is None:
        loss = update_loss(model, *batch, b["uniforms"], 0.0, 2.5, dense)
    else:
        recompute, slices = memory
        loss = update_loss(
            model, *batch, b["uniforms"], 0.0, 2.5, dense, recompute=recompute, slices=slices
        )
    params = [p for p in model.parameters() if p.requires_grad]
    # allow_unused: a parameter the loss does not reach comes back None
    grads = cast(  # torch's stub says Tensor; allow_unused makes it Tensor | None
        "tuple[Tensor | None, ...]", torch.autograd.grad(loss.total, params, allow_unused=True)
    )
    return float(loss.total.detach()), [None if g is None else g.detach().clone() for g in grads]


def _norm(t: Tensor) -> float:
    """The Euclidean norm, in float64 (torch's norm functions are untyped here)."""
    d = t.detach().double()
    return math.sqrt(float((d * d).sum()))


@pytest.mark.parametrize("memory", [(False, 2), (False, 5), (True, 1), (True, 3)])
def test_the_memory_options_keep_the_loss_and_every_gradient(memory: tuple[bool, int]) -> None:
    # (recompute, slices); 5 does not divide the rows, so the last range is shorter
    model, _, _ = _model()
    b = _batch()
    value, grads = _loss(model, b)
    got, got_grads = _loss(model, b, memory)
    assert got == pytest.approx(value, rel=1e-6)
    assert sum(g is not None for g in grads) == sum(g is not None for g in got_grads)
    # normwise per tensor: the sliced head batches positions differently, so float32 sums round
    # differently, and how depends on the BLAS thread count. Measured 2026-10-20 (threads
    # 1/2/4/8): worst ||g - h|| / ||g|| 9.1e-7 (about 8 float32 ulps; the unchanged update is
    # thread-invariant); an element-wise relative bound failed on near-zero elements at 2 threads
    # in the full gate. 1e-5 keeps an 11x margin; a real defect (a detached path, a dropped
    # range) differs at order 1
    for g, h in zip(grads, got_grads, strict=True):
        if g is None:
            assert h is None
        else:
            assert h is not None
            assert _norm(h - g) <= 1e-5 * _norm(g)


def test_the_default_update_is_unchanged_bitwise() -> None:
    # off is the measured update: the explicit defaults give the identical loss and gradients
    model, _, _ = _model()
    b = _batch()
    value, grads = _loss(model, b)
    again, again_grads = _loss(model, b, (False, 1))
    assert again == value
    for g, h in zip(grads, again_grads, strict=True):
        if g is None:
            assert h is None
        else:
            assert h is not None
            assert torch.equal(g, h)


def _saved_widths(run: Callable[[], None]) -> Counter[int]:
    """The last-dimension widths of every distinct storage saved for the backward pass."""
    saved: list[Tensor] = []

    def pack(t: Tensor) -> Tensor:
        saved.append(t)
        return t

    def unpack(t: Tensor) -> Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        run()
    storages = {t.untyped_storage().data_ptr(): t for t in saved if t.dim()}
    return Counter(int(t.shape[-1]) for t in storages.values())


def test_sliced_label_nll_keeps_no_vocabulary_logits_for_the_backward_pass() -> None:
    # the point of the option: unsliced, the head's vocabulary intermediates of every pass are
    # saved (T x E x n x V floats: ~4 GB for the text pilot's dense term); sliced, the head
    # runs again in the backward pass and none is kept (model.token_losses)
    model, _, _ = _model()
    d = int(model.embedding.weight.shape[1])
    targets = torch.randint(1, 60, (2, 8), dtype=torch.int32)
    mask = torch.ones(2, 8, dtype=torch.bool)

    def run(slices: int) -> Callable[[], None]:
        def go() -> None:
            y = torch.randn(T, 2, 8, d, requires_grad=True)
            ce = label_nll(model, y, targets, mask, slices=slices)
            (grad,) = torch.autograd.grad(ce.sum(), [y])
            assert float(grad.abs().sum()) > 0

        return go

    assert _saved_widths(run(1))[VOCAB] > 0
    assert _saved_widths(run(2))[VOCAB] == 0


def test_update_loss_applies_both_options_to_both_forwards() -> None:
    # recompute must reach the trajectory forward AND the dense auxiliary's forward, and slices
    # both label terms: with both on, no FFN-width activation and no vocabulary-width logits
    # are kept for the backward pass; off, both are (equality tests cannot see a dropped
    # memory option, since it changes no value)
    model, _, _ = _model()
    b = _batch()

    def off() -> None:
        _ = _loss(model, b)

    def on() -> None:
        _ = _loss(model, b, (True, 2))

    widths_off, widths_on = _saved_widths(off), _saved_widths(on)
    assert widths_off[FFN] > 0
    assert widths_off[VOCAB] > 0
    assert widths_on[FFN] == 0
    assert widths_on[VOCAB] == 0


def test_update_step_passes_the_memory_options_through() -> None:
    model, head, backbone = _model()
    b = _batch()
    values = [
        b["tokens"],
        b["pos"],
        b["targets"],
        b["mask"],
        b["valid"],
        b["weights"],
        b["uniforms"],
        torch.tensor(0.0, dtype=torch.float64),
    ]

    def defaults() -> None:  # NO memory arguments: update_step's own defaults
        _ = update_step(model, optimizer(head, backbone), head + backbone, 2.5, 0.1, values)

    def on() -> None:
        opt = optimizer(head, backbone)
        _ = update_step(model, opt, head + backbone, 2.5, 0.1, values, recompute=True, slices=2)

    widths_off = _saved_widths(defaults)
    widths_on = _saved_widths(on)
    assert widths_off[FFN] > 0
    assert widths_off[VOCAB] > 0
    assert widths_on[FFN] == 0
    assert widths_on[VOCAB] == 0


def _labels() -> tuple[Tensor, Tensor, Tensor]:
    """y [2, 5, d] with integer targets and a boolean mask of exactly [2, 5]."""
    y = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(15))
    targets = torch.arange(1, 11).view(2, 5)
    mask = (torch.arange(10) % 3 != 1).view(2, 5)  # mixed labelled and unlabelled positions
    return y, targets, mask


@pytest.mark.parametrize("slices", [1, 3])
@pytest.mark.parametrize(
    "case",
    [
        "rank-one labels",
        "one-row labels",
        "rank-one targets, exact mask",
        "one-row targets, exact mask",
        "float targets",
        "integer mask",
        "rank-two y",
    ],
)
def test_label_nll_refuses_labels_that_do_not_match_its_outputs(case: str, slices: int) -> None:
    # Maintainer B's review of 778ff10: the public boundary silently repeated one row's labels
    # across the batch ([n] or [1, n] labels) and truncated float targets (1.9 -> 1)
    model, _, _ = _model()
    y, targets, mask = _labels()
    bad = {
        "rank-one labels": (y, targets[0], mask[0]),
        "one-row labels": (y, targets[:1], mask[:1]),
        # the targets alone wrong: each check must hold by itself, not through the mask's
        "rank-one targets, exact mask": (y, targets[0], mask),
        "one-row targets, exact mask": (y, targets[:1], mask),
        "float targets": (y, targets.float() + 0.9, mask),
        "integer mask": (y, targets, mask.long()),
        "rank-two y": (y[0], targets[0], mask[0]),
    }[case]
    with pytest.raises(ContractError):
        _ = label_nll(model, *bad, slices=slices)


@pytest.mark.parametrize("slices", [1, 3])
def test_label_nll_takes_exact_labels_and_broadcasts_only_pass_dimensions(slices: int) -> None:
    model, _, _ = _model()
    y, targets, mask = _labels()
    ordinary = label_nll(model, y, targets, mask, slices)
    assert tuple(ordinary.shape) == (2, 5)
    assert bool(torch.isfinite(ordinary).all())
    assert bool((ordinary[~mask] == 0).all())
    assert torch.equal(label_nll(model, y, targets.int(), mask, slices), ordinary)  # int32 too
    passes = label_nll(model, y.unsqueeze(0).expand(3, -1, -1, -1), targets, mask, slices)
    assert tuple(passes.shape) == (3, 2, 5)
    for t in range(3):
        assert _norm(passes[t] - ordinary) <= 1e-6 * _norm(ordinary)


def test_label_nll_refuses_a_slice_count_below_one() -> None:
    model, _, _ = _model()
    with pytest.raises(ValueError, match="slices"):
        _ = label_nll(
            model,
            torch.zeros(1, 4, int(model.embedding.weight.shape[1])),
            torch.zeros(1, 4, dtype=torch.int32),
            torch.ones(1, 4, dtype=torch.bool),
            slices=0,
        )
