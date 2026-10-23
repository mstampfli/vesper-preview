"""All-pass supervision (the halting experiment's conditions B-D, docs/reviews/2026-10-15-
halting-experiment-plan.md): the objective composed from per-pass losses and the exit
distribution, against an oracle built from the same public pieces by hand."""

import math
from dataclasses import replace

import pytest
import torch

from conftest import small_raw
from vesper.config import parse
from vesper.contracts import ContractError
from vesper.data import Chunk, Document, Stream
from vesper.exits import (
    exit_log_probs,
    expected_pass_loss,
    hazard_logits,
    kl_uniform_aggregate,
    kl_uniform_local,
)
from vesper.learning import ntp_loss
from vesper.loop import DenseRule, LoopResult, TraceRule
from vesper.model import Vesper

CPU = torch.device("cpu")
F64 = torch.float64
T = 4
# the objective scores per-token losses and logits in float32 by design (as ``objective`` does);
# against the float64 oracle that leaves about 1e-7 on values near five, while a wrong weighting
# (for example non-uniform weights in the fixed condition) moves the value by about 0.04
FLOAT32_TOLERANCE = 5e-6


def _model() -> Vesper:
    raw = small_raw()
    raw["model"]["t_max"] = T
    return Vesper(parse(raw, "test"), torch.Generator().manual_seed(0)).to(F64)


def _chunk() -> Chunk:
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (11,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(4)
    ]
    chunk = Stream(iter(docs), slots=2, chunk=16).next()
    assert chunk is not None
    assert bool((~chunk.loss_mask).any())  # the fixture has masked positions to exclude
    return chunk


def _dense(model: Vesper, chunk: Chunk) -> LoopResult:
    states = model.init_states(2, F64, CPU)
    return model.run(chunk.tokens, chunk.pos, states, exit_rule=DenseRule())


def _oracle(
    model: Vesper, chunk: Chunk, result: LoopResult, condition: str, beta: float
) -> torch.Tensor:
    mask = chunk.loss_mask
    losses = torch.stack(
        [ntp_loss(model.head(result.per_pass[t]).double(), chunk.targets) for t in range(T)]
    )
    if condition == "fixed":
        z = torch.zeros(T, *mask.shape, dtype=F64)
    else:
        z = torch.stack([model.loop.halting_head().logit(result.per_pass[t], t) for t in range(T)])
    # the uniform baseline written out independently: -log(T - t) for t < T, 0 at T
    baseline = torch.zeros(T, dtype=F64)
    for t in range(1, T):
        baseline[t - 1] = -math.log(T - t)
    z = z + baseline[:, None, None]
    log_p = exit_log_probs(z)
    value = expected_pass_loss(log_p, losses, mask)
    if condition == "local":
        value = value + beta * kl_uniform_local(log_p, mask)
    if condition == "aggregate":
        value = value + beta * kl_uniform_aggregate(log_p, mask)
    return value


@pytest.mark.parametrize("condition", ["fixed", "local", "aggregate"])
def test_the_objective_composes_per_pass_losses_and_the_exit_distribution(condition: str) -> None:
    model, chunk = _model(), _chunk()
    result = _dense(model, chunk)
    value, parts = model.all_pass_objective(chunk, result, condition=condition, beta=0.3)
    expected = _oracle(model, chunk, result, condition, 0.3)
    assert abs(value.item() - expected.item()) < FLOAT32_TOLERANCE
    assert set(parts) >= {"expected_ntp", "kl", "pass_ntp", "expected_depth"}
    assert parts["pass_ntp"].shape == (T,)


def test_the_fixed_condition_is_the_mean_per_pass_loss_and_trains_no_head() -> None:
    model, chunk = _model(), _chunk()
    result = _dense(model, chunk)
    value, parts = model.all_pass_objective(chunk, result, condition="fixed", beta=0.3)
    assert abs(value.item() - parts["pass_ntp"].mean().item()) < FLOAT32_TOLERANCE
    head = model.loop.halting_head()
    grads = torch.autograd.grad(value, [head.w_h, head.b_h, head.norm.weight], allow_unused=True)
    assert all(g is None or float(g.abs().sum()) == 0.0 for g in grads)


@pytest.mark.parametrize("condition", ["local", "aggregate"])
def test_the_learned_conditions_train_the_head_and_every_pass(condition: str) -> None:
    model, chunk = _model(), _chunk()
    model.loop.halting_head().zero_residual()
    with torch.no_grad():
        _ = model.loop.halting_head().w_h.normal_(generator=torch.Generator().manual_seed(3))
    value, _ = model.all_pass_objective(chunk, _dense(model, chunk), condition=condition, beta=0.3)
    params = dict(model.named_parameters())
    grads = dict(zip(params, torch.autograd.grad(value, list(params.values())), strict=True))
    assert float(grads["loop.halting.w_h"].abs().sum()) > 0
    assert float(grads["loop.u"][T - 1].abs().sum()) > 0  # the last pass is supervised


def test_slices_change_neither_the_value_nor_the_gradients() -> None:
    model, chunk = _model(), _chunk()
    result = _dense(model, chunk)
    params = list(model.parameters())
    one, _ = model.all_pass_objective(chunk, result, condition="local", beta=0.3, slices=1)
    three, _ = model.all_pass_objective(chunk, result, condition="local", beta=0.3, slices=3)
    assert abs(one.item() - three.item()) < 1e-12
    g1 = torch.autograd.grad(one, params, retain_graph=True)
    g3 = torch.autograd.grad(three, params)
    assert all((a - b).abs().max().item() < 1e-12 for a, b in zip(g1, g3, strict=True))


def test_a_zero_residual_head_starts_exactly_uniform_without_a_second_prior() -> None:
    # the learned conditions start at w = 0, b = 0, gain 1 plus the fixed baseline; ACT's frozen
    # bias (logit of 1/T) on top of the baseline would apply the prior twice (maintainer B, efeb8f3)
    model, chunk = _model(), _chunk()
    head = model.loop.halting_head()
    with torch.no_grad():
        _ = head.b_h.fill_(-1.0986)
        _ = head.norm.weight.fill_(3.0)
    head.zero_residual()
    _, parts = model.all_pass_objective(chunk, _dense(model, chunk), condition="local", beta=0.3)
    assert abs(parts["kl"].item()) < 1e-12
    assert abs(parts["expected_depth"].item() - (T + 1) / 2) < 1e-12


def test_the_objective_needs_a_dense_forward_and_a_loss_position() -> None:
    model, chunk = _model(), _chunk()
    with torch.no_grad():
        _ = model.loop.halting_head().b_h.fill_(50.0)  # ACT halts every token at pass 1
    act = model.run(chunk.tokens, chunk.pos, model.init_states(2, F64, CPU))
    assert bool((act.n_iters == 1).all())
    with pytest.raises(ContractError, match="dense"):
        _ = model.all_pass_objective(chunk, act, condition="local", beta=0.3)
    with pytest.raises(ValueError, match="condition"):
        _ = model.all_pass_objective(chunk, _dense(model, chunk), condition="act", beta=0.3)
    silenced = Chunk(
        tokens=chunk.tokens,
        targets=chunk.targets,
        valid=chunk.valid,
        doc_start=chunk.doc_start,
        segment=chunk.segment,
        loss_mask=torch.zeros_like(chunk.loss_mask),
        pos=chunk.pos,
        doc_ids=chunk.doc_ids,
        spans=chunk.spans,
    )
    with pytest.raises(ContractError, match="no position"):
        _ = model.all_pass_objective(silenced, _dense(model, chunk), condition="local", beta=0.3)


def test_a_context_input_that_halted_early_is_not_a_dense_forward() -> None:
    # Maintainer B's review of 49c92cf, finding 2: real inputs without a loss halted at pass 1 while
    # every loss position ran all passes; the later supervised outputs then read copied-forward
    # states, not the dense ones, and the objective accepted them
    model, chunk = _model(), _chunk()
    context = chunk.valid & ~chunk.loss_mask
    assert bool(context.any())
    trace = torch.where(context, 1, T).to(torch.int64)
    mixed = model.run(
        chunk.tokens, chunk.pos, model.init_states(2, F64, CPU), exit_rule=TraceRule(trace)
    )
    assert bool((mixed.n_iters[chunk.loss_mask] == T).all())
    with pytest.raises(ContractError, match="dense"):
        _ = model.all_pass_objective(chunk, mixed, condition="fixed", beta=0.1)


def _saved_for_backward(condition: str, slices: int) -> tuple[int, int]:
    """Storages the objective saves for its backward pass: ``(vocabulary-wide, [N, d] hidden)``.

    Counted by data pointer over every tensor autograd packs (parameters excluded), and the
    backward pass run once so the count is of a graph that works.
    """
    model, chunk = _model(), _chunk()
    result = _dense(model, chunk)
    vocab, d = model.embedding.weight.shape
    parameters = {p.untyped_storage().data_ptr() for p in model.parameters()}
    saved: list[torch.Tensor] = []

    def pack(t: torch.Tensor) -> torch.Tensor:
        saved.append(t)
        return t

    def unpack(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        value, _ = model.all_pass_objective(
            chunk, result, condition=condition, beta=0.3, slices=slices
        )
    storages = {t.untyped_storage().data_ptr(): t for t in saved}
    wide = [p for p, t in storages.items() if t.dim() and t.shape[-1] == vocab]
    # hidden rows by their exact shape, all rows or one slice of them: this fixture's b * n
    # equals d, so a [T, b * n] tensor would match a width test alone
    rows = int(chunk.tokens.numel())
    shapes = {(rows, d), (-(-rows // slices), d)}
    hidden = [p for p, t in storages.items() if tuple(t.shape) in shapes and p not in parameters]
    grads = torch.autograd.grad(value, list(model.parameters()), allow_unused=True)
    assert any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    return len(wide), len(hidden)


@pytest.mark.parametrize("slices", [1, 2, 4])
def test_no_vocabulary_logits_are_kept_for_the_backward_pass(slices: int) -> None:
    # Maintainer B's review of 49c92cf, finding 3: at slices = 1 every pass's vocabulary
    # intermediates were saved for the backward pass. They still are not: kernels.nll.linear_nll
    # keeps only the log-sum-exp per row. What it keeps instead of recomputing (2026-10-21, the 5090
    # profile: the recomputed head was ~21 ms of a step) is each call's normalized rows, one
    # [b * n / slices, d] per pass and slice: T * slices hidden storages, 1/V of the logits'
    # width each (until 2026-10-21 the head and its norm ran again and one storage was kept)
    assert _saved_for_backward("fixed", slices) == (0, T * slices)
    # the learned head keeps its own normalized input per pass, one more [b * n, d] each
    assert _saved_for_backward("local", slices) == (0, T * slices + T)
    assert _saved_for_backward("aggregate", slices) == (0, T * slices + T)


def test_the_parts_average_over_the_loss_positions_only() -> None:
    # the static-shape objective scores every position; its reported depth must still be the
    # mean over the loss positions, here with a head whose exits differ by position
    model, chunk = _model(), _chunk()
    head = model.loop.halting_head()
    with torch.no_grad():
        _ = head.w_h.copy_(torch.randn(head.w_h.shape, generator=torch.Generator().manual_seed(9)))
    result = _dense(model, chunk)
    _, parts = model.all_pass_objective(chunk, result, condition="local", beta=0.3)
    learned = torch.stack([head.logit(result.per_pass[t], t) for t in range(T)])
    p = exit_log_probs(hazard_logits(learned)).exp()
    depth = torch.arange(1, T + 1, dtype=p.dtype)[:, None, None]
    per_token = (p * depth).sum(0).detach()
    want = float(per_token[chunk.loss_mask].mean())
    assert abs(float(per_token.mean()) - want) > 1e-6  # the fixture tells the two means apart
    assert float(parts["expected_depth"]) == pytest.approx(want, rel=1e-12)


def test_targets_outside_the_loss_mask_are_never_read() -> None:
    # selection never touched them; the static-shape objective must not either (an
    # out-of-range id there would fail the loss's id contract if it were scored)
    model, chunk = _model(), _chunk()
    result = _dense(model, chunk)
    poisoned = replace(chunk, targets=torch.where(chunk.loss_mask, chunk.targets, 10_000))
    for condition in ("fixed", "local", "aggregate"):
        clean, _ = model.all_pass_objective(chunk, result, condition=condition, beta=0.3)
        dirty, _ = model.all_pass_objective(poisoned, result, condition=condition, beta=0.3)
        assert float(dirty.detach()) == float(clean.detach()), condition


def test_only_real_inputs_must_have_run_every_pass() -> None:
    # Maintainer B's review of 49c92cf, finding 2: every REAL input must be dense; a padding
    # position's depth writes nothing a scored pass reads, so it is not part of the contract
    model, chunk = _model(), _chunk()
    last, scored = torch.arange(15, 16), torch.arange(14, 16)
    padded = replace(
        chunk,
        valid=chunk.valid.clone().index_fill_(1, last, value=False),
        loss_mask=chunk.loss_mask.clone().index_fill_(1, scored, value=False),
    )
    result = _dense(model, padded)
    short = result.n_iters.clone().index_fill_(1, last, 1)
    _ = model.all_pass_objective(
        padded, replace(result, n_iters=short), condition="fixed", beta=0.0
    )
    real = result.n_iters.clone().index_fill_(1, torch.arange(3, 4), 1)
    with pytest.raises(ContractError, match="dense forward"):
        _ = model.all_pass_objective(
            padded, replace(result, n_iters=real), condition="fixed", beta=0.0
        )


def test_a_non_finite_hidden_state_outside_the_loss_positions_reaches_nothing() -> None:
    # selection never sent a non-loss position through the head; the static-shape objectives
    # replace those rows before it, so a NaN there changes no value and no gradient (both the
    # all-pass objective and ACT's)
    model, chunk = _model(), _chunk()
    outside = (~chunk.loss_mask).nonzero()[0]
    b, n = int(outside[0]), int(outside[1])
    dense = _dense(model, chunk)
    per_pass = dense.per_pass.detach().clone().requires_grad_(True)
    for condition in ("fixed", "local", "aggregate"):
        poisoned = per_pass.clone()  # a new graph per condition: each is differentiated once
        poisoned[:, b, n] = math.nan
        clean, _ = model.all_pass_objective(
            chunk, replace(dense, per_pass=per_pass), condition=condition, beta=0.3
        )
        dirty, _ = model.all_pass_objective(
            chunk, replace(dense, per_pass=poisoned), condition=condition, beta=0.3
        )
        assert float(dirty.detach()) == pytest.approx(float(clean.detach()), rel=1e-12)
        # the head's own weights too: their gradient is the logits' gradient times the row, so
        # a NaN row poisons them even where the row's own gradient stays finite (measured: the
        # embedding and the output norm, when the rows were not replaced)
        grads = torch.autograd.grad(dirty, [per_pass, *_trained_parameters(model)])
        assert all(bool(torch.isfinite(g).all()) for g in grads), condition
        assert float(grads[0][:, b, n].abs().sum()) == 0.0
    cfg = parse(small_raw() | {"model": small_raw()["model"] | {"t_max": T}}, "test")
    act = model.run(chunk.tokens, chunk.pos, model.init_states(2, F64, CPU))
    y = act.y.detach().clone().requires_grad_(True)
    y_poisoned = y.clone()
    y_poisoned[b, n] = math.nan
    clean_act, _ = model.objective(chunk, replace(act, y=y), cfg)
    dirty_act, _ = model.objective(chunk, replace(act, y=y_poisoned), cfg)
    assert float(dirty_act.detach()) == pytest.approx(float(clean_act.detach()), rel=1e-12)
    grads_act = torch.autograd.grad(dirty_act, [y, *_trained_parameters(model)])
    assert all(bool(torch.isfinite(g).all()) for g in grads_act)


def _trained_parameters(model: Vesper) -> list[torch.Tensor]:
    """The parameters an objective's head reaches: the tied embedding and the output norm."""
    return [model.embedding.weight, model.norm_out.weight]


def test_position_losses_refuse_targets_of_another_leading_shape() -> None:
    # Maintainer B's review of 571b5f9 (410cdc1): y [2, 3, d] with targets [3, 2] was accepted
    # (flattened first) and returned [3, 2]; the declared contract is targets with y's exact leading
    # shape
    model = _model()
    d = model.embedding.weight.shape[1]
    y = torch.randn(2, 3, d, dtype=F64)
    good = torch.zeros(2, 3, dtype=torch.int64)
    assert model.position_losses(y, good).shape == (2, 3)
    for bad in (torch.zeros(3, 2, dtype=torch.int64), torch.zeros(6, dtype=torch.int64)):
        with pytest.raises(ContractError):
            _ = model.position_losses(y, bad)
