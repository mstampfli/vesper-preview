import copy
import hashlib
import json
import math
import shutil
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from itertools import accumulate, pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml
from torch.nn import functional

import vesper.model
import vesper.train
from conftest import replaced_after_first_read, small_config, small_raw
from vesper.column import set_scan
from vesper.config import VesperConfig, parse
from vesper.corpus import CYCLE_ORDER_RULE, Component, MixerState, finalize
from vesper.data import Chunk, Document, Stream, is_held_out
from vesper.kernels.nll import linear_nll
from vesper.keys import file_sha256
from vesper.learning import LossWeights
from vesper.loop import DenseRule, HaltingHead
from vesper.model import Vesper, wake_objective
from vesper.sample import components
from vesper.tokenizer import train_from_texts
from vesper.train import (
    CHECKPOINT_SCHEMA,
    Group,
    Identity,
    Layout,
    Phase,
    Progress,
    Recipe,
    StepGraphs,
    StepResult,
    _device_step,  # pyright: ignore[reportPrivateUsage]
    _plain,  # pyright: ignore[reportPrivateUsage]
    _read_step,  # pyright: ignore[reportPrivateUsage]
    _scan_for,  # pyright: ignore[reportPrivateUsage]
    _to_device,  # pyright: ignore[reportPrivateUsage]
    checkpoint,
    decays,
    detach_states,
    identity_of,
    optimizer,
    phase_schedule,
    restore,
    run,
    set_halting,
    step,
    total_steps,
)

Builder = Callable[[Path, str, dict[str, str]], Path]
CPU = torch.device("cpu")


def training_config() -> VesperConfig:
    """The small model with a training budget small enough to cross every phase in 40 steps."""
    raw = small_raw()
    raw["training"].update(
        {"total_tokens": 40 * 2 * 16, "checkpoints_per_run": 4, "eval_chunks": 2}
    )
    raw["optimizer"].update({"warmup_steps": 4})
    return parse(raw, "test")


def test_the_schedule_names_the_phases_and_anneals_the_passes() -> None:
    cfg = training_config()
    total = cfg.training.total_tokens
    assert total_steps(cfg) == 40
    warm = phase_schedule(0, 0, cfg)
    assert (warm.name, warm.t_max, warm.halting_frozen, warm.lambda_h) == ("warm", 1, True, 0.0)
    assert not warm.retrieval
    assert not warm.ledger_writes
    assert warm.dusk_eta_fraction == 0.0
    late_warm = phase_schedule(int(total * 0.09), 3, cfg)
    assert late_warm.name == "warm"
    assert late_warm.t_max == 2  # annealed from 1 toward 2 before the phase ends
    loop = phase_schedule(int(total * 0.10), 4, cfg)
    assert (loop.name, loop.t_max, loop.halting_frozen) == ("loop", 2, False)
    assert loop.lambda_h == pytest.approx(0.0)
    assert loop.retrieval
    assert not loop.ledger_writes
    mid_loop = phase_schedule(int(total * 0.25), 10, cfg)
    assert mid_loop.lambda_h == pytest.approx(cfg.learning.lambda_h * 0.5)
    ledger = phase_schedule(int(total * 0.40), 16, cfg)
    assert (ledger.name, ledger.lambda_h, ledger.ledger_writes) == (
        "ledger",
        cfg.learning.lambda_h,
        True,
    )
    assert ledger.dusk_eta_fraction == cfg.dusk.eta_slow_fraction
    assert not phase_schedule(int(total * 0.49), 19, cfg).qat
    assert phase_schedule(int(total * 0.5), 20, cfg).qat
    settle = phase_schedule(int(total * 0.8), 32, cfg)
    assert settle.name == "settle"
    assert phase_schedule(total * 3, 200, cfg).name == "settle"  # past the end: still settle
    with pytest.raises(ValueError, match="at least 0"):
        _ = phase_schedule(-1, 0, cfg)


def test_the_learning_rate_warms_up_and_follows_one_cosine_to_the_final_fraction() -> None:
    cfg = training_config()
    o = cfg.optimizer
    rates = [phase_schedule(0, s, cfg).lr for s in range(total_steps(cfg))]
    assert rates[0] == pytest.approx(o.peak_lr / o.warmup_steps)
    assert rates[o.warmup_steps - 1] == pytest.approx(o.peak_lr)
    assert rates[-1] == pytest.approx(o.peak_lr * o.final_lr_fraction)
    after = rates[o.warmup_steps - 1 :]
    assert all(a >= b for a, b in pairwise(after))  # monotone decrease
    assert all(a <= b for a, b in pairwise(rates[: o.warmup_steps]))  # monotone warmup


def test_weight_decay_roles_cover_every_parameter_and_exempt_the_gates() -> None:
    model = Vesper(small_config(), torch.Generator().manual_seed(0))
    roles = {name: decays(name) for name, _ in model.named_parameters()}
    assert roles["embedding.weight"]
    assert roles["loop.modules_.0.rec.w_q"]
    assert roles["loop.modules_.0.ffn.w1"]
    assert not roles["loop.modules_.0.rec.w_beta"]  # a two-dimensional gate
    assert not roles["loop.halting.w_h"]
    assert not roles["loop.u"]
    assert not roles["norm_out.weight"]
    assert not roles["loop.modules_.1.norm_rec.weight"]
    with pytest.raises(ValueError, match="no weight-decay role"):
        _ = decays("loop.modules_.0.mystery")
    opt = optimizer(model, small_config())
    groups = cast("list[dict[str, object]]", opt.param_groups)
    counted = sum(len(cast("list[torch.Tensor]", g["params"])) for g in groups)
    assert counted == sum(1 for _ in model.parameters())
    assert groups[0]["weight_decay"] == small_config().optimizer.weight_decay
    assert groups[1]["weight_decay"] == 0.0


def test_the_frozen_head_gives_one_over_t_max_and_unfreezes_with_its_values() -> None:
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    frozen = phase_schedule(0, 0, cfg)
    assert frozen.t_max == 1
    head = model.loop.halting
    assert isinstance(head, HaltingHead)
    with torch.no_grad():
        _ = head.norm.weight.fill_(2.0)  # a learned gain the frozen phase must reset
    set_halting(model, frozen)
    assert torch.equal(head.norm.weight, torch.ones_like(head.norm.weight))
    x = torch.randn(3, 5, cfg.model.d)
    # one pass halts by construction; the head is held at the two-pass value, never saturated
    assert torch.allclose(model.loop.halting.probability(x, 0), torch.full((3, 5), 0.5))
    for p in (model.loop.halting_module or model.loop).parameters():
        assert not p.requires_grad
    half = phase_schedule(int(cfg.training.total_tokens * 0.09), 3, cfg)
    assert half.t_max == 2
    set_halting(model, half)
    assert torch.allclose(model.loop.halting.probability(x, 0), torch.full((3, 5), 0.5))
    free = phase_schedule(int(cfg.training.total_tokens * 0.2), 8, cfg)
    set_halting(model, free)
    assert torch.allclose(model.loop.halting.probability(x, 0), torch.full((3, 5), 0.5))
    assert all(p.requires_grad for p in (model.loop.halting_module or model.loop).parameters())


def test_a_warm_phase_that_only_ran_one_pass_leaves_a_head_that_can_learn(tmp_path: Path) -> None:
    # Maintainer B's fixture (review of b797837): warm executes only t_max = 1, then the head is
    # freed at t_max = 2; with a saturated bias both gradients were zero and every token halted at
    # pass one
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    chunk = cfg.training.chunk_tokens
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (40,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(8)
    ]
    groups = [
        Group(Stream(iter(docs), slots=2, chunk=chunk), model.init_states(2, torch.float32, CPU))
    ]
    warm = phase_schedule(0, 0, cfg)
    assert warm.t_max == 1
    assert warm.halting_frozen
    first = step(model, opt, groups, warm, cfg, CPU)
    assert first is not None
    head = model.loop.halting
    assert isinstance(head, HaltingHead)
    assert math.isfinite(float(head.b_h.item()))
    assert abs(float(head.b_h.item())) < 10
    # a checkpoint written in that warm step and restored keeps the finite head
    _ = checkpoint(
        tmp_path / "s", model, opt, groups, progress_at(1, 32), identity(), recipe(), reason="x"
    )
    fresh = Vesper(cfg, torch.Generator().manual_seed(5))
    _ = restore(tmp_path / "s", fresh, optimizer(fresh, cfg), identity(), Layout(1, 2))
    restored_head = fresh.loop.halting
    assert isinstance(restored_head, HaltingHead)
    assert torch.equal(restored_head.b_h, head.b_h)
    # freed at two passes: the head moves
    free = phase_schedule(int(cfg.training.total_tokens * 0.2), 8, cfg)
    assert free.t_max == 2
    assert not free.halting_frozen
    before = head.b_h.detach().clone()
    result = step(model, opt, groups, free, cfg, CPU)
    assert result is not None
    assert head.b_h.grad is not None
    assert head.w_h.grad is not None
    assert float(head.b_h.grad.abs().sum()) > 0
    assert float(head.w_h.grad.abs().sum()) > 0
    assert not torch.equal(head.b_h.detach(), before)
    assert result.histogram[1] > 0  # tokens reach the second pass


def docs_with_loss_counts(chunk: int) -> list[Document]:
    """Two documents: the first gives one loss position in a chunk, the second three."""
    g = torch.Generator().manual_seed(7)
    short = torch.randint(1, 64, (2,), generator=g).to(torch.int32)  # one pair
    longer = torch.randint(1, 64, (chunk,), generator=g).to(torch.int32)
    del chunk
    return [Document("short", short), Document("longer", longer[:4])]


def test_a_step_is_the_token_weighted_mean_of_the_unsplit_batch() -> None:
    # Maintainer B's fixture (training-loop review): groups with one and three loss tokens; equal
    # group means would weight the one token three times as much as the unsplit batch does
    cfg = training_config()
    docs = docs_with_loss_counts(cfg.training.chunk_tokens)
    seed = torch.Generator().manual_seed(0)
    model = Vesper(cfg, seed)
    reference = copy.deepcopy(model)
    phase = phase_schedule(int(cfg.training.total_tokens * 0.2), 8, cfg)  # loop: head free
    # two groups of one slot each
    groups = [
        Group(
            Stream(iter(docs[:1]), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        ),
        Group(
            Stream(iter(docs[1:]), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        ),
    ]
    opt = optimizer(model, cfg)
    result = step(model, opt, groups, phase, cfg, CPU)
    assert result is not None
    assert result.tokens_with_loss == 4
    assert result.tokens == 4  # the valid inputs: one pair and three pairs
    assert result.pass_ntp is None  # only the fixed objective reports each pass
    grads = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    # the unsplit batch: the same two chunks side by side as two slots, one objective over all
    # four loss positions (a stream over both documents would pack them into one slot instead)
    set_halting(reference, phase)
    first = Stream(iter(docs[:1]), slots=1, chunk=cfg.training.chunk_tokens).next()
    second = Stream(iter(docs[1:]), slots=1, chunk=cfg.training.chunk_tokens).next()
    assert first is not None
    assert second is not None
    chunk = Chunk(
        tokens=torch.cat([first.tokens, second.tokens]),
        targets=torch.cat([first.targets, second.targets]),
        valid=torch.cat([first.valid, second.valid]),
        doc_start=torch.cat([first.doc_start, second.doc_start]),
        segment=torch.cat([first.segment, second.segment]),
        loss_mask=torch.cat([first.loss_mask, second.loss_mask]),
        pos=torch.cat([first.pos, second.pos]),
        doc_ids=first.doc_ids + second.doc_ids,
        spans=first.spans + second.spans,
    )
    assert chunk.loss_mask.sum().item() == 4
    logits, loop_result = reference.forward(
        chunk.tokens, chunk.pos, reference.init_states(2, torch.float32, CPU), passes=phase.t_max
    )
    objective, _ = wake_objective(
        logits, chunk, loop_result, cfg, weights=LossWeights(phase.lambda_h, cfg.learning.lambda_g)
    )
    torch.autograd.backward([objective])
    # the step clips the whole gradient to grad_clip; the reference is clipped the same way
    _ = torch.nn.utils.clip_grad_norm_(
        [p for p in reference.parameters() if p.requires_grad], cfg.optimizer.grad_clip
    )
    for name, p in reference.named_parameters():
        if p.grad is None:
            assert name not in grads
            continue
        assert torch.allclose(grads[name], p.grad, atol=1e-6, rtol=1e-5), name
    # the carried states are detached: nothing reaches the previous chunk
    for group in groups:
        for state in group.states:
            assert all(not h.requires_grad and h.grad_fn is None for h in state.h)
            assert all(not c.k.requires_grad for c in state.kv)


def test_the_fixed_objective_supervises_every_pass_equally_on_a_dense_forward() -> None:
    # the real-data pilot's phase 1 (docs/reviews/2026-10-20-real-data-halting-pilot.md): stage
    # 1's condition B on text; the reference is written out here, not taken from the model
    raw = small_raw()
    raw["model"]["t_max"] = 3
    raw["training"]["t_max_anneal_from"] = 3  # depth first: every pass from the start
    cfg = parse(raw, "test")
    docs = docs_with_loss_counts(cfg.training.chunk_tokens)
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    reference = copy.deepcopy(model)
    phase = phase_schedule(0, 0, cfg)
    assert phase.t_max == 3
    chunk_tokens = cfg.training.chunk_tokens
    groups = [
        Group(
            Stream(iter(docs), slots=1, chunk=chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        )
    ]
    result = step(model, optimizer(model, cfg), groups, phase, cfg, CPU, objective="fixed")
    assert result is not None
    assert result.histogram == [0.0, 0.0, 1.0]  # every scored token ran every pass
    assert result.ponder == 0.0
    chunk = Stream(iter(docs), slots=1, chunk=chunk_tokens).next()
    assert chunk is not None
    out = reference.run(
        chunk.tokens,
        chunk.pos,
        reference.init_states(1, torch.float32, CPU),
        passes=3,
        exit_rule=DenseRule(),
    )
    assert torch.equal(out.n_iters[chunk.valid], torch.full_like(out.n_iters[chunk.valid], 3))
    mask = chunk.loss_mask
    per_pass = torch.stack(
        [
            functional.cross_entropy(
                reference.head(out.per_pass[t])[mask].float(), chunk.targets[mask].long()
            )
            for t in range(3)
        ]
    )
    each = [float(v) for v in per_pass.detach().unbind()]
    assert len(set(each)) == 3  # the passes differ, so the weights are visible
    expected = per_pass.mean()
    assert result.ntp == pytest.approx(float(expected.detach()), rel=1e-5)
    assert result.loss == pytest.approx(float(expected.detach()), rel=1e-5)
    assert result.pass_ntp == pytest.approx(each, rel=1e-5)
    torch.autograd.backward([expected])
    _ = torch.nn.utils.clip_grad_norm_(
        [p for p in reference.parameters() if p.requires_grad], cfg.optimizer.grad_clip
    )
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if q.grad is None or not bool(q.grad.any()):
            assert p.grad is None or not bool(p.grad.any()), name
            continue
        assert p.grad is not None, name
        assert torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5), name
    head = model.loop.halting_head()
    assert all(p.grad is None or not bool(p.grad.any()) for p in head.parameters())


def test_the_fixed_objective_ignores_a_head_that_would_stop_every_token() -> None:
    # depth first must not depend on the halting head's state: a free head biased to stop at
    # pass 1 halts every token under ACT, and the fixed step still runs and trains every pass
    raw = small_raw()
    raw["model"]["t_max"] = 3
    raw["training"]["t_max_anneal_from"] = 3
    cfg = parse(raw, "test")
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    phase = phase_schedule(int(cfg.training.total_tokens * 0.2), 8, cfg)  # loop: head free
    assert (phase.halting_frozen, phase.t_max) == (False, 3)
    set_halting(model, phase)
    with torch.no_grad():
        _ = model.loop.halting_head().b_h.fill_(30.0)  # p(halt) = 1 at the first pass
    docs = docs_with_loss_counts(cfg.training.chunk_tokens)
    chunk = Stream(iter(docs), slots=1, chunk=cfg.training.chunk_tokens).next()
    assert chunk is not None
    with torch.no_grad():
        act = model.run(chunk.tokens, chunk.pos, model.init_states(1, torch.float32, CPU), passes=3)
    assert bool((act.n_iters[chunk.valid] == 1).all())  # ACT would stop every token at pass 1
    groups = [
        Group(
            Stream(iter(docs), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        )
    ]
    result = step(model, optimizer(model, cfg), groups, phase, cfg, CPU, objective="fixed")
    assert result is not None
    assert result.histogram == [0.0, 0.0, 1.0]
    assert result.pass_ntp is not None
    assert len(result.pass_ntp) == 3


def test_a_checkpoint_from_before_the_objective_is_refused_by_its_schema(tmp_path: Path) -> None:
    # schema 7 stored a recipe without the objective; such a checkpoint is another run's, refused
    # by its schema number before the identity (not later as a mismatched recipe)
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    docs = [Document("d", torch.arange(1, 40, dtype=torch.int32))]
    groups = [
        Group(
            Stream(iter(docs), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        )
    ]
    where = tmp_path / "step-1"
    _ = checkpoint(where, model, opt, groups, progress_at(1, 16), identity(), recipe(), reason="x")
    record = cast("dict[str, object]", json.loads((where / "train.json").read_text()))
    old = cast("dict[str, object]", record["recipe"])
    del old["objective"]
    old["schema"] = 7
    _ = (where / "train.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = cast("dict[str, str]", json.loads((where / "manifest.json").read_text()))
    manifest["train.json"] = file_sha256(where / "train.json")
    _ = (where / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema 7, this code reads 8"):
        _ = restore(where, model, opt, identity(), Layout(1, 1))


def test_an_unknown_objective_is_refused_before_anything_is_drawn() -> None:
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    stream = Stream(
        iter(docs_with_loss_counts(cfg.training.chunk_tokens)),
        slots=1,
        chunk=cfg.training.chunk_tokens,
    )
    groups = [Group(stream, model.init_states(1, torch.float32, CPU))]
    with pytest.raises(ValueError, match="objective"):
        _ = step(
            model,
            optimizer(model, cfg),
            groups,
            phase_schedule(0, 0, cfg),
            cfg,
            CPU,
            objective="pondernet",
        )
    assert stream.next() is not None  # nothing was drawn


def test_a_step_trains_the_live_groups_when_a_stream_is_dry_and_none_moves_nothing() -> None:
    # Maintainer B's fixture (review of b797837): one stream dry, the other advanced from 16 to 32
    # and the step returned None, so that chunk was consumed without a model update
    cfg = training_config()
    chunk = cfg.training.chunk_tokens
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    long = Document("long", torch.arange(1, 4 * chunk + 2, dtype=torch.int32))
    live = Group(
        Stream(iter([long]), slots=1, chunk=chunk), model.init_states(1, torch.float32, CPU)
    )
    dry = Group(Stream(iter([]), slots=1, chunk=chunk), model.init_states(1, torch.float32, CPU))
    phase = phase_schedule(0, 0, cfg)
    before = [h.clone() for h in live.states[0].h]
    result = step(model, opt, [dry, live], phase, cfg, CPU)
    assert result is not None
    assert result.groups_active == 1
    assert result.tokens == chunk
    assert live.stream.state().slots[0].emitted == chunk
    assert any(not torch.equal(a, b) for a, b in zip(before, live.states[0].h, strict=True))
    assert dry.stream.state().slots[0].emitted == 0
    # every stream dry: None, and nothing was drawn, trained or re-frozen
    head = model.loop.halting
    assert isinstance(head, HaltingHead)
    with torch.no_grad():
        _ = head.b_h.fill_(0.25)  # a value the frozen phase would overwrite
    params = [p.detach().clone() for p in model.parameters()]
    both_dry = [dry, Group(Stream(iter([]), slots=1, chunk=chunk), live.states)]
    assert step(model, opt, both_dry, phase, cfg, CPU) is None
    assert all(torch.equal(a, b) for a, b in zip(params, model.parameters(), strict=True))
    assert float(head.b_h.item()) == 0.25


def test_detach_states_keeps_values_and_drops_the_graph() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    tokens = torch.randint(0, 64, (1, 16)).to(torch.int32)
    pos = torch.arange(16).reshape(1, 16)
    _, result = model.forward(tokens, pos, model.init_states(1, torch.float32, CPU))
    detached = detach_states(result.states)
    for before, after in zip(result.states, detached, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(before.h, after.h, strict=True))
        assert all(b.grad_fn is None for b in after.h)
        assert all(a.window == b.window for a, b in zip(before.kv, after.kv, strict=True))


def identity() -> Identity:
    return Identity("tok" * 20, "cfg" * 20, "fin" * 20, "rec" * 20)


def recipe() -> Recipe:
    return Recipe(0, (("a", 1.0),), (None,), 0.0, "finalize", "split", CHECKPOINT_SCHEMA, "act")


def progress_at(step: int, tokens_seen: int) -> Progress:
    return Progress(step, tokens_seen, {"a": 5}, {"a": 4}, "d" * 64)


def test_checkpoint_round_trip_restores_everything_and_refuses_tampering(tmp_path: Path) -> None:
    cfg = training_config()
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (20,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(6)
    ]
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    stream = Stream(iter(docs), slots=2, chunk=cfg.training.chunk_tokens)
    groups = [Group(stream, model.init_states(2, torch.float32, CPU))]
    phase = phase_schedule(0, 0, cfg)
    _ = step(model, opt, groups, phase, cfg, CPU)
    progress = progress_at(1, 32)
    where = tmp_path / "step-1"
    manifest = checkpoint(
        where, model, opt, groups, progress, identity(), recipe(), reason="interval"
    )
    assert manifest.is_file()
    assert sorted(p.name for p in where.iterdir()) == [
        "manifest.json",
        "model.pt",
        "optimizer.pt",
        "states.pt",
        "train.json",
    ]
    record = cast("dict[str, object]", json.loads((where / "train.json").read_text()))
    assert record["reason"] == "interval"
    assert record["layout"] == {"groups": 1, "slots": 2}
    assert record["recipe"] == {
        "seed": 0,
        "components": [["a", 1.0]],
        "generations": [None],
        "held_out_fraction": 0.0,
        "finalize_rule": "finalize",
        "split_rule": "split",
        "schema": CHECKPOINT_SCHEMA,
        "objective": "act",
    }
    with pytest.raises(ValueError, match="already exists"):
        _ = checkpoint(where, model, opt, groups, progress, identity(), recipe(), reason="interval")
    fresh = Vesper(cfg, torch.Generator().manual_seed(99))
    fresh_opt = optimizer(fresh, cfg)
    with pytest.raises(ValueError, match="regrouping"):
        _ = restore(where, fresh, fresh_opt, identity(), Layout(2, 1))
    restored = restore(where, fresh, fresh_opt, identity(), Layout(1, 2))
    for (name, a), (_, b) in zip(model.named_parameters(), fresh.named_parameters(), strict=True):
        assert torch.equal(a, b), name
    assert restored.progress == progress
    assert len(restored.states) == 1
    for before, after in zip(groups[0].states, restored.states[0], strict=True):
        assert all(torch.equal(x, y) for x, y in zip(before.h, after.h, strict=True))
        assert all(
            torch.equal(x.k, y.k) and torch.equal(x.pos, y.pos)
            for x, y in zip(before.kv, after.kv, strict=True)
        )
    saved_stream = groups[0].stream.state()
    assert restored.streams[0].documents_taken == saved_stream.documents_taken
    assert restored.streams[0].slots[0].offset == saved_stream.slots[0].offset
    assert [d.doc_id for d in restored.streams[0].slots[1].queued] == [
        d.doc_id for d in saved_stream.slots[1].queued
    ]
    with pytest.raises(ValueError, match="another tokenizer"):
        _ = restore(where, fresh, fresh_opt, Identity("x", "y", "z", "w"), Layout(1, 2))
    _ = (where / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash differs"):
        _ = restore(where, fresh, fresh_opt, identity(), Layout(1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the CUDA history transition needs a GPU")
def test_cuda_states_from_a_bf16_autocast_step_restore_on_the_cpu_and_move_back(
    tmp_path: Path,
) -> None:
    # Maintainer B's review of f4c9279: the cold -> saved key history under CUDA bf16 autocast,
    # through the real checkpoint and restore; states load on the CPU (one device, checked there)
    # and _to_device moves them back exactly
    cuda = torch.device("cuda")
    cfg = training_config()
    assert cfg.recurrence.key_conv > 0
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (20,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(4)
    ]
    model = Vesper(cfg, torch.Generator().manual_seed(0)).to(cuda)
    _ = set_scan(model, "fused-fp32")  # on CUDA the step is captured; the reference cannot be
    opt = optimizer(model, cfg, capturable=True)
    groups = [
        Group(
            Stream(iter(docs), slots=2, chunk=cfg.training.chunk_tokens),
            model.init_states(2, torch.float32, cuda),
        )
    ]
    _ = step(model, opt, groups, phase_schedule(0, 0, cfg), cfg, cuda, graphs=StepGraphs())
    saved = groups[0].states
    assert any(float(z.abs().max()) > 0 for s in saved for z in s.key_history)  # it transitioned
    where = tmp_path / "step-1"
    _ = checkpoint(
        where, model, opt, groups, progress_at(1, 32), identity(), recipe(), reason="interval"
    )
    fresh = Vesper(cfg, torch.Generator().manual_seed(99))
    restored = restore(where, fresh, optimizer(fresh, cfg), identity(), Layout(1, 2)).states[0]
    for before, after in zip(saved, restored, strict=True):
        tensors = (*after.h, *after.key_history, *(t for c in after.kv for t in (c.k, c.v, c.pos)))
        assert {t.device for t in tensors} == {CPU}
        assert {t.dtype for t in (*after.h, *after.key_history)} == {torch.float32}
        back = _to_device([after], cuda)[0]
        assert all(
            torch.equal(x, y) for x, y in zip(before.key_history, back.key_history, strict=True)
        )
        assert all(torch.equal(x, y) for x, y in zip(before.h, back.h, strict=True))
        assert all(
            torch.equal(x.k, y.k) and torch.equal(x.v, y.v) and torch.equal(x.pos, y.pos)
            for x, y in zip(before.kv, back.kv, strict=True)
        )


def test_a_checkpoint_of_another_schema_is_refused_before_any_weight_is_loaded(
    tmp_path: Path,
) -> None:
    # Maintainer B's fixture (review of 629559f): stored shapes changed under an unchanged schema
    # number and an old checkpoint failed deep inside restore with a KeyError; the schema is checked
    # right after the hashes, before the identity and before torch.load
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    docs = [Document("d", torch.arange(1, 40, dtype=torch.int32))]
    groups = [
        Group(
            Stream(iter(docs), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        )
    ]
    where = tmp_path / "step-1"
    _ = checkpoint(where, model, opt, groups, progress_at(1, 16), identity(), recipe(), reason="x")
    record = cast("dict[str, object]", json.loads((where / "train.json").read_text()))
    cast("dict[str, object]", record["recipe"])["schema"] = CHECKPOINT_SCHEMA - 1
    _ = (where / "train.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest = cast("dict[str, str]", json.loads((where / "manifest.json").read_text()))
    manifest["train.json"] = file_sha256(where / "train.json")
    _ = (where / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    fresh = Vesper(cfg, torch.Generator().manual_seed(9))
    before = [p.detach().clone() for p in fresh.parameters()]
    with pytest.raises(ValueError, match=f"schema {CHECKPOINT_SCHEMA - 1}, this code reads"):
        _ = restore(where, fresh, optimizer(fresh, cfg), identity(), Layout(1, 1))
    assert all(torch.equal(a, b) for a, b in zip(before, fresh.parameters(), strict=True))


def test_a_failed_save_leaves_no_checkpoint_and_a_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's finding (review of b797837): a checkpoint written into its final directory left
    # a partial one that the retry refused as "already exists"
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    docs = [Document("d", torch.arange(1, 40, dtype=torch.int32))]
    groups = [
        Group(
            Stream(iter(docs), slots=1, chunk=cfg.training.chunk_tokens),
            model.init_states(1, torch.float32, CPU),
        )
    ]
    where = tmp_path / "step-1"

    class BoomError(Exception):
        pass

    def failing_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise BoomError

    monkeypatch.setattr("vesper.train.torch.save", failing_save)
    with pytest.raises(BoomError):
        _ = checkpoint(
            where, model, opt, groups, progress_at(1, 16), identity(), recipe(), reason="x"
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # neither step-1 nor a partial
    monkeypatch.undo()
    manifest = checkpoint(
        where, model, opt, groups, progress_at(1, 16), identity(), recipe(), reason="x"
    )
    assert manifest.is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["step-1"]


def test_a_restored_run_follows_the_uninterrupted_trajectory(tmp_path: Path) -> None:
    cfg = training_config()
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (24,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(12)
    ]
    chunk = cfg.training.chunk_tokens

    def fresh() -> tuple[Vesper, torch.optim.AdamW]:
        model = Vesper(cfg, torch.Generator().manual_seed(0))
        return model, optimizer(model, cfg)

    # uninterrupted: three steps
    model_a, opt_a = fresh()
    groups_a = [
        Group(Stream(iter(docs), slots=2, chunk=chunk), model_a.init_states(2, torch.float32, CPU))
    ]
    losses_a: list[float] = []
    tokens = 0
    for index in range(3):
        result = step(model_a, opt_a, groups_a, phase_schedule(tokens, index, cfg), cfg, CPU)
        assert result is not None
        losses_a.append(result.loss)
        tokens += result.tokens
    # interrupted after one step, checkpointed, restored into fresh objects, two more steps
    model_b, opt_b = fresh()
    groups_b = [
        Group(Stream(iter(docs), slots=2, chunk=chunk), model_b.init_states(2, torch.float32, CPU))
    ]
    first = step(model_b, opt_b, groups_b, phase_schedule(0, 0, cfg), cfg, CPU)
    assert first is not None
    progress = Progress(1, first.tokens, {}, {}, "")
    _ = checkpoint(
        tmp_path / "step-1", model_b, opt_b, groups_b, progress, identity(), recipe(), reason="x"
    )
    model_c, opt_c = fresh()
    restored = restore(tmp_path / "step-1", model_c, opt_c, identity(), Layout(1, 2))
    taken = restored.streams[0].documents_taken
    groups_c = [Group(Stream.restore(restored.streams[0], iter(docs[taken:])), restored.states[0])]
    losses_c: list[float] = [first.loss]
    tokens = restored.progress.tokens_seen
    # the resumed steps recompute their activations: the same trajectory through the trainer
    for index in range(1, 3):
        phase = phase_schedule(tokens, index, cfg)
        result = step(model_c, opt_c, groups_c, phase, cfg, CPU, recompute=True)
        assert result is not None
        losses_c.append(result.loss)
        tokens += result.tokens
    assert losses_c == pytest.approx(losses_a, rel=1e-6, abs=1e-7)
    for (name, a), (_, c) in zip(
        model_a.named_parameters(), model_c.named_parameters(), strict=True
    ):
        assert torch.allclose(a, c, atol=1e-7, rtol=1e-6), name


def test_chunks_drawn_ahead_train_the_same_steps_and_are_not_checkpointed(tmp_path: Path) -> None:
    # the run draws the next step's chunks while the device runs the current one
    # (Group.draw_ahead through step's ahead): every step must train exactly the chunks it would
    # have drawn itself, and a checkpoint taken with a chunk drawn ahead records the stream from
    # before that chunk, so a resume continues the same trajectory
    cfg = training_config()
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (24,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(12)
    ]
    chunk = cfg.training.chunk_tokens

    def fresh() -> tuple[Vesper, torch.optim.AdamW, list[Group]]:
        model = Vesper(cfg, torch.Generator().manual_seed(0))
        stream = Stream(iter(docs), slots=2, chunk=chunk)
        return (
            model,
            optimizer(model, cfg),
            [Group(stream, model.init_states(2, torch.float32, CPU))],
        )

    model_a, opt_a, groups_a = fresh()
    losses_a: list[float] = []
    tokens = 0
    for index in range(3):
        result = step(model_a, opt_a, groups_a, phase_schedule(tokens, index, cfg), cfg, CPU)
        assert result is not None
        losses_a.append(result.loss)
        tokens += result.tokens

    model_b, opt_b, groups_b = fresh()
    drawn: list[int] = []

    def ahead() -> None:
        drawn.append(len(drawn))
        for group in groups_b:
            group.draw_ahead()

    first = step(model_b, opt_b, groups_b, phase_schedule(0, 0, cfg), cfg, CPU, ahead=ahead)
    assert first is not None
    assert drawn == [0]
    recorded, live = groups_b[0].stream_state(), groups_b[0].stream.state()
    assert recorded.slots[0].emitted < live.slots[0].emitted  # the drawn chunk is not recorded
    progress = Progress(1, first.tokens, {}, {}, "")
    _ = checkpoint(
        tmp_path / "step-1", model_b, opt_b, groups_b, progress, identity(), recipe(), reason="x"
    )
    losses_b, tokens = [first.loss], first.tokens
    for index in range(1, 3):
        phase = phase_schedule(tokens, index, cfg)
        result = step(model_b, opt_b, groups_b, phase, cfg, CPU, ahead=ahead)
        assert result is not None
        losses_b.append(result.loss)
        tokens += result.tokens
    assert losses_b == losses_a  # the same chunks, the same arithmetic

    model_c, opt_c, _ = fresh()
    restored = restore(tmp_path / "step-1", model_c, opt_c, identity(), Layout(1, 2))
    taken = restored.streams[0].documents_taken
    groups_c = [Group(Stream.restore(restored.streams[0], iter(docs[taken:])), restored.states[0])]
    losses_c, tokens = [first.loss], restored.progress.tokens_seen
    for index in range(1, 3):
        result = step(model_c, opt_c, groups_c, phase_schedule(tokens, index, cfg), cfg, CPU)
        assert result is not None
        losses_c.append(result.loss)
        tokens += result.tokens
    assert losses_c == losses_a
    with pytest.raises(ValueError, match="already drawn ahead"):
        groups_b[0].draw_ahead()  # the last step drew one; a second would skip it
    # a step without drawing ahead takes the drawn chunk; the record is the live stream again
    assert step(model_b, opt_b, groups_b, phase_schedule(tokens, 3, cfg), cfg, CPU) is not None
    assert (
        groups_b[0].stream_state().slots[0].emitted == groups_b[0].stream.state().slots[0].emitted
    )


def test_the_next_chunks_are_drawn_after_the_launch_and_before_the_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the point of drawing ahead is the overlap: launched, then drawn, then waited for
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [Document(f"d{i}", torch.arange(1, 30, dtype=torch.int32)) for i in range(4)]
    groups = [
        Group(
            Stream(iter(docs), slots=2, chunk=cfg.training.chunk_tokens),
            model.init_states(2, torch.float32, CPU),
        )
    ]
    events: list[str] = []

    def recorded[**P, R](name: str, fn: Callable[P, R]) -> Callable[P, R]:
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            events.append(name)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(vesper.train, "_device_step", recorded("launch", _device_step))
    monkeypatch.setattr(vesper.train, "_read_step", recorded("read", _read_step))
    result = step(
        model,
        optimizer(model, cfg),
        groups,
        phase_schedule(0, 0, cfg),
        cfg,
        CPU,
        ahead=lambda: events.append("ahead"),
    )
    assert result is not None
    assert events == ["launch", "ahead", "read"]


def run_fixture(
    tmp_path: Path,
    book_builder: Builder,
    total_tokens: int,
    checkpoints: int,
    *,
    long_document: bool = False,
) -> tuple[VesperConfig, Path, Sequence[Component], Path, Path]:
    """The small corpus, config file, finalization and tokenizer a run needs.

    With ``long_document`` one web document of the training side is many chunks long, so the
    slot that receives it runs on after every other stream is dry.
    """
    from test_sample import write_mix  # noqa: PLC0415

    web, mix = write_mix(tmp_path, book_builder)
    if long_document:
        doc_id = next(
            f"<urn:long:{i}>" for i in range(100) if not is_held_out(f"<urn:long:{i}>", 0.2)
        )
        words = " ".join(f"word{i % 37} erases and writes" for i in range(400))
        pq.write_table(
            pa.Table.from_pydict({"id": [doc_id], "text": [words]}), str(web / "001.parquet")
        )
    raw = small_raw(vocab=300)
    raw["training"].update(
        {
            "total_tokens": total_tokens,
            "checkpoints_per_run": checkpoints,
            "eval_chunks": 1,
            "held_out_fraction": 0.2,
        }
    )
    raw["optimizer"].update({"warmup_steps": 2})
    config_path = tmp_path / "small.yaml"
    _ = config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = parse(raw, "test")
    parts = components(web, mix)
    fin = finalize(parts, tmp_path / "fin")
    tok = train_from_texts(["web document says the delta rule erases and writes"] * 20, vocab=300)
    tokenizer_path = tmp_path / "tok.json"
    tok.save(tokenizer_path)
    return cfg, config_path, parts, fin.receipt, tokenizer_path


def read_log(log_path: Path) -> list[dict[str, object]]:
    return [
        cast("dict[str, object]", json.loads(line)) for line in log_path.read_text().splitlines()
    ]


def test_exhaustion_trains_the_live_groups_and_saves_the_terminal_boundary_once(
    tmp_path: Path, book_builder: Builder
) -> None:
    # a budget far beyond the corpus and one checkpoint interval beyond it: the run ends by
    # exhaustion, the live group trains on after the other ran dry, the last completed step is
    # saved exactly once with that reason, and the terminal save resumes to an immediate end
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 100_000, 1, long_document=True
    )
    out = tmp_path / "run"
    log_path = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=out,
        micro=1,
        steps=None,
        resume=None,
        seed=0,
        device=CPU,
    )
    lines = read_log(log_path)
    steps = [line for line in lines if line["kind"] == "step"]
    end = lines[-1]
    assert end["kind"] == "end"
    assert end["reason"] == "exhausted"
    assert steps
    assert end["step"] == steps[-1]["step"]
    exhausted = [line for line in lines if line["kind"] == "exhausted"]
    assert len(exhausted) == 1
    assert exhausted[0]["groups_active"] == 1
    tail = cast("int", exhausted[0]["step"])
    assert [line["groups_active"] for line in steps if cast("int", line["step"]) < tail] == [2] * (
        tail - 1
    )
    assert all(line["groups_active"] == 1 for line in steps if cast("int", line["step"]) >= tail)
    assert tail < cast("int", end["step"])  # the live group trained on past the first dry stream
    checkpoints = [line for line in lines if line["kind"] == "checkpoint"]
    assert len(checkpoints) == 1
    assert checkpoints[0]["reason"] == "exhausted"
    assert checkpoints[0]["step"] == end["step"]
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [f"step-{end['step']}"]
    log_path = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=out,
        micro=1,
        steps=None,
        resume=out / f"step-{end['step']}",
        seed=0,
        device=CPU,
    )
    again = read_log(log_path)
    assert again[-1]["kind"] == "end"
    assert again[-1]["reason"] == "exhausted"
    assert again[-1]["step"] == end["step"]
    assert len([line for line in again if line["kind"] == "step"]) == len(steps)
    assert len([line for line in again if line["kind"] == "checkpoint"]) == 1


def test_the_run_writes_a_log_checkpoints_and_resumes(  # noqa: PLR0915
    tmp_path: Path, book_builder: Builder
) -> None:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    out = tmp_path / "run"
    log_path = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=out,
        micro=1,
        steps=4,
        resume=None,
        seed=0,
        device=CPU,
    )
    lines = [
        cast("dict[str, object]", json.loads(line)) for line in log_path.read_text().splitlines()
    ]
    kinds = [line["kind"] for line in lines]
    assert kinds[0] == "run"
    assert kinds.count("step") == 4
    assert "checkpoint" in kinds
    steps = [line for line in lines if line["kind"] == "step"]
    assert [line["step"] for line in steps] == [1, 2, 3, 4]
    assert all(math.isfinite(cast("float", line["loss"])) for line in steps)
    assert steps[0]["phase"] == "warm"
    # the run's count is cumulative over the steps' own counts, and the checkpoint carries it
    seen = list(accumulate(cast("int", line["tokens"]) for line in steps))
    assert [line["tokens_seen"] for line in steps] == seen
    assert seen[0] > 0
    # consumed tokens per source, as the streams emitted them, sum to the run's count; the mixer's
    # hand-out counts (shares) are at least that and are not the same thing
    for line in steps:
        consumed = cast("dict[str, int]", line["consumed"])
        shares = cast("dict[str, int]", line["shares"])
        assert sum(consumed.values()) == line["tokens_seen"]
        assert set(consumed) <= set(shares)
        assert all(consumed[name] <= shares[name] for name in consumed)
    assert sum(cast("dict[str, int]", steps[-1]["consumed"]).values()) < sum(
        cast("dict[str, int]", steps[-1]["shares"]).values()
    )
    assert all(line["peak_reserved_bytes"] == 0 for line in steps)  # the CPU has no counter
    checkpoints = [line for line in lines if line["kind"] == "checkpoint"]
    assert checkpoints[-1]["tokens_seen"] == seen[-1]
    last = Path(cast("str", checkpoints[-1]["path"]))
    assert last.name == "step-4"
    assert math.isfinite(cast("float", checkpoints[-1]["held_out_perplexity"]))
    assert [line["reason"] for line in checkpoints] == ["interval", "steps"]
    header = lines[0]
    assert all(line["run_id"] == header["run_id"] for line in lines)
    assert header["layout"] == {"groups": cfg.training.batch_chunks, "slots": 1}
    assert cast("dict[str, object]", header["recipe"])["seed"] == 0
    assert header["scan"] == "reference"  # read back from the model's sublayers
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["reason"] == "steps"
    assert lines[-1]["step"] == 4
    # an unsupported learning mode is refused before anything is read or written
    local_raw = cast(
        "dict[str, dict[str, object]]", yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    local_raw["learning"]["mode"] = "local"
    local_path = tmp_path / "local.yaml"
    _ = local_path.write_text(yaml.safe_dump(local_raw), encoding="utf-8")
    with pytest.raises(NotImplementedError, match="local"):
        _ = run(
            parse(local_raw, "test"),
            config_path=local_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=tmp_path / "local-run",
            micro=1,
            steps=1,
            resume=None,
            seed=0,
            device=CPU,
        )
    assert not (tmp_path / "local-run").exists()
    # the fused scan on a CPU model is refused before anything is written
    with pytest.raises(ValueError, match="fused scan needs"):
        _ = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=tmp_path / "fused-run",
            micro=1,
            steps=1,
            resume=None,
            seed=0,
            device=CPU,
            scan="fused",
        )
    assert not (tmp_path / "fused-run").exists()
    # a fresh run never appends to a log that holds one
    with pytest.raises(ValueError, match="already holds a run"):
        _ = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=1,
            resume=None,
            seed=0,
            device=CPU,
        )
    # another seed is another recipe, not a continuation (maintainer B's fixture: equal-length
    # documents under seed 1 passed the count check and diverged)
    with pytest.raises(ValueError, match="recipe"):
        _ = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=1,
            resume=last,
            seed=1,
            device=CPU,
        )
    # resume for one more step from the last checkpoint
    log_path = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=out,
        micro=1,
        steps=1,
        resume=last,
        seed=0,
        device=CPU,
    )
    lines = [
        cast("dict[str, object]", json.loads(line)) for line in log_path.read_text().splitlines()
    ]
    resumed = [line for line in lines if line["kind"] == "run"][-1]
    assert resumed["resumed_from"] == str(last)
    assert [line["step"] for line in lines if line["kind"] == "step"][-1] == 5
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 5
    assert (out / "step-5" / "manifest.json").is_file()
    # the resume went through the mixer's cursors, not a replay: its fifth step equals the fifth
    # step of an uninterrupted run of the same recipe
    straight = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=tmp_path / "straight",
        micro=1,
        steps=5,
        resume=None,
        seed=0,
        device=CPU,
    )
    fifth = [line for line in read_log(straight) if line["kind"] == "step"][-1]
    resumed_fifth = [line for line in lines if line["kind"] == "step"][-1]
    assert resumed_fifth["loss"] == pytest.approx(cast("float", fifth["loss"]), rel=1e-6)
    assert resumed_fifth["shares"] == fifth["shares"]
    assert resumed_fifth["tokens_seen"] == fifth["tokens_seen"]
    with pytest.raises(ValueError, match="micro_chunks"):
        _ = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=3,
            steps=1,
            resume=None,
            seed=0,
            device=CPU,
        )


def test_the_run_draws_each_steps_chunks_ahead(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every step of a run draws the next one's chunks for every group while the device works
    # (the resume test above shows the checkpoints stay those of the trained chunks)
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    calls: list[int] = []
    draw = Group.draw_ahead

    def counted(group: Group) -> None:
        calls.append(1)
        draw(group)

    monkeypatch.setattr(Group, "draw_ahead", counted)
    _ = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=tmp_path / "run",
        micro=1,
        steps=3,
        resume=None,
        seed=0,
        device=CPU,
    )
    assert len(calls) == 3 * cfg.training.batch_chunks  # micro 1: one group per chunk of a batch


def test_a_fixed_run_records_its_objective_and_the_static_profile_and_keeps_it(
    tmp_path: Path, book_builder: Builder
) -> None:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    out = tmp_path / "run"

    def go(steps: int, resume: Path | None, objective: str) -> Path:
        return run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=steps,
            resume=resume,
            seed=0,
            device=CPU,
            objective=objective,
        )

    lines = read_log(go(2, None, "fixed"))
    assert lines[0]["objective"] == "fixed"
    assert cast("dict[str, object]", lines[0]["recipe"])["objective"] == "fixed"  # in the identity
    steps = [line for line in lines if line["kind"] == "step"]
    assert all(line["ponder"] == 0.0 for line in steps)
    for line in steps:  # the steps trained the fixed objective: each pass's loss, their mean
        losses = cast("list[float]", line["pass_ntp"])
        assert len(losses) == line["t_max"]
        assert sum(losses) / len(losses) == pytest.approx(cast("float", line["ntp"]), rel=1e-6)
    checkpoints = [line for line in lines if line["kind"] == "checkpoint"]
    assert checkpoints
    for line in checkpoints:
        losses = cast("list[float]", line["held_out_static_loss"])
        assert len(losses) == cfg.model.t_max
        assert all(math.isfinite(v) for v in losses)
        assert line["held_out_static_tokens_with_loss"] == line["held_out_tokens_with_loss"]
    last = Path(cast("str", checkpoints[-1]["path"]))
    before = (out / "log.jsonl").read_text()
    # the objective is in the recipe, the recipe's hash in the checkpoint's identity
    with pytest.raises(ValueError, match=r"differs: recipe_sha256; the recipe includes the obj"):
        _ = go(1, last, "act")
    assert (out / "log.jsonl").read_text() == before  # refused before anything was written
    resumed = read_log(go(1, last, "fixed"))
    assert [line["objective"] for line in resumed if line["kind"] == "run"] == ["fixed", "fixed"]


def test_a_direct_step_refuses_an_unsupported_mode_before_touching_anything() -> None:
    # Maintainer B's probe (review of 1725ca6): the exported step drew a document, replaced the
    # group's states and zeroed halting.w_h through the warm setup before model.objective refused
    # local mode; the shared check comes first at this boundary, and a refusal leaves everything as
    # it was
    from dataclasses import replace  # noqa: PLC0415

    cfg = training_config()
    local = replace(cfg, learning=replace(cfg.learning, mode="local"))
    chunk = cfg.training.chunk_tokens
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    opt = optimizer(model, cfg)
    head = model.loop.halting
    assert isinstance(head, HaltingHead)
    with torch.no_grad():
        _ = head.w_h.fill_(0.5)  # the one-pass warm setup would zero it
        _ = head.b_h.fill_(0.25)
    long = Document("long", torch.arange(1, 4 * chunk + 2, dtype=torch.int32))
    live_states = model.init_states(1, torch.float32, CPU)
    live = Group(Stream(iter([long]), slots=1, chunk=chunk), live_states)
    dry_states = model.init_states(1, torch.float32, CPU)
    dry = Group(Stream(iter([]), slots=1, chunk=chunk), dry_states)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    carried = [[h.clone() for h in state.h] for state in live_states]
    phase = phase_schedule(0, 0, local)  # warm: one pass, the head frozen
    with pytest.raises(NotImplementedError, match="local"):
        _ = step(model, opt, [dry, live], phase, local, CPU)
    assert live.stream.state().documents_taken == 0
    assert live.stream.state().slots[0].emitted == 0
    assert dry.stream.state().documents_taken == 0
    assert live.states is live_states
    assert dry.states is dry_states
    for state, snapshot in zip(live_states, carried, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(state.h, snapshot, strict=True))
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())
    assert {n: p.requires_grad for n, p in model.named_parameters()} == flags
    assert float(head.w_h.abs().sum().item()) > 0
    assert float(head.b_h.item()) == 0.25
    assert all(p.grad is None for p in model.parameters())


def selection_fixture(
    tmp_path: Path, book_builder: Builder
) -> tuple[VesperConfig, Path, Path, Path, Path]:
    """A data root with four published generations and a selection, the finalization over the
    selected components, and the config and tokenizer of ``run_fixture``."""
    import shutil  # noqa: PLC0415

    from vesper.generations import Reference, Unit, publish_generation, reference, select  # noqa: PLC0415
    from vesper.sample import selected_components  # noqa: PLC0415

    cfg, config_path, legacy_parts, _, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    data = tmp_path / "data"
    refs: list[Reference] = []
    for part in legacy_parts:
        staging = tmp_path / f"staging-{part.name}"
        staging.mkdir()
        units: list[Unit] = []
        for path in part.paths:
            _ = shutil.copyfile(path, staging / path.name)
            _ = (staging / f"{path.stem}.receipt.json").write_text("{}", encoding="utf-8")
            units.append(Unit(path.stem, (path.name,), (f"{path.stem}.receipt.json",)))
        refs.append(reference(publish_generation(part.name, staging, data, units=units)))
    _ = select(data, refs, replacing=None)
    fin = finalize(selected_components(data), tmp_path / "fin-selected")
    return cfg, config_path, data, fin.receipt, tokenizer_path


def test_a_resume_resolves_its_checkpoints_pins_not_todays_selection(  # noqa: PLR0915
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's probe (review of 1725ca6): a one-step checkpoint resumed through the command
    # line; the selection's web generation was switched with every original pin still valid, and the
    # same resume command exited 1. Amendment 1 of the generation contract: the selection chooses
    # a new run, a resume resolves the references its checkpoint stored and never touches it
    import shutil  # noqa: PLC0415

    from vesper.generations import (  # noqa: PLC0415
        SELECTION,
        Unit,
        publish_generation,
        reference,
        select,
        selection,
    )
    from vesper.sample import selected_components  # noqa: PLC0415
    from vesper.train import main, resumed_components  # noqa: PLC0415

    cfg, config_path, data, receipt, tokenizer_path = selection_fixture(tmp_path, book_builder)
    first = selection(data)
    parts = selected_components(data)
    _ = run(
        cfg,
        config_path=config_path,
        parts=parts,
        finalization_receipt=receipt,
        tokenizer_path=tokenizer_path,
        out=tmp_path / "first",
        micro=1,
        steps=1,
        resume=None,
        seed=0,
        device=CPU,
    )
    saved = tmp_path / "first" / "step-1"
    assert saved.is_dir()
    common = [
        "--config", str(config_path), "--data", str(data), "--finalization", str(receipt),
        "--tokenizer", str(tokenizer_path), "--micro-chunks", "1", "--steps", "1",
        "--seed", "0", "--device", "cpu",
    ]  # fmt: skip
    resume_args = [*common, "--resume", str(saved)]
    assert main([*resume_args, "--out", str(tmp_path / "resume-same")]) == 0
    # the selection moves to another web generation; every original pin stays valid
    web = next(c for c in parts if c.name == "web")
    staging = tmp_path / "staging-new-web"
    staging.mkdir()
    units: list[Unit] = []
    for path in web.paths:
        _ = shutil.copyfile(path, staging / path.name)
        _ = (staging / f"{path.stem}.receipt.json").write_text("{}", encoding="utf-8")
        units.append(Unit(path.stem, (path.name,), (f"{path.stem}.receipt.json",)))
    new_web = publish_generation("web", staging, data, units=units)
    others = [ref for ref in first.references if ref.kind != "web"]
    second = select(data, [reference(new_web), *others], replacing=first.sha256)
    assert main([*resume_args, "--out", str(tmp_path / "resume-switched")]) == 0
    assert selection(data).sha256 == second.sha256  # the pointer was never rewritten
    resumed = resumed_components(data, saved)
    assert [(c.name, c.share, c.generation, c.paths) for c in resumed] == [
        (c.name, c.share, c.generation, c.paths) for c in parts
    ]
    # a new run follows the selection: the old finalization was not taken over the new web file
    assert main([*common, "--out", str(tmp_path / "new-run")]) == 1
    # the selection absent: the resume still resolves its own pins
    _ = (data / SELECTION).rename(tmp_path / "SELECTION.aside")
    with pytest.raises(ValueError, match="no selection"):
        _ = selected_components(data)
    assert main([*resume_args, "--out", str(tmp_path / "resume-no-selection")]) == 0
    # a pinned generation missing: refused, never a fallback to the selection
    code = next(c for c in parts if c.name == "code")
    assert code.generation is not None
    identity = code.generation.split("/")[1].split("@")[0]
    _ = (data / "code" / identity).rename(tmp_path / "code-aside")
    with pytest.raises(ValueError, match="missing"):
        _ = resumed_components(data, saved)
    assert main([*resume_args, "--out", str(tmp_path / "resume-missing")]) == 1
    _ = (tmp_path / "code-aside").rename(data / "code" / identity)
    # a pinned generation changed: refused; restored, the resume runs again
    target = code.paths[0]
    original = target.read_bytes()
    _ = target.write_bytes(b"ALTERED")
    with pytest.raises(ValueError, match="differ"):
        _ = resumed_components(data, saved)
    _ = target.write_bytes(original)
    assert main([*resume_args, "--out", str(tmp_path / "resume-restored")]) == 0
    # inconsistent or tampered records: refused by name, before any generation is read
    record = cast("dict[str, object]", json.loads((saved / "train.json").read_text()))
    recipe = cast("dict[str, object]", record["recipe"])
    pins = cast("list[str | None]", recipe["generations"])

    record_identity = cast("dict[str, object]", record["identity"])
    declared = record_identity["recipe_sha256"]

    def digest_of(r: dict[str, object]) -> str:
        canonical = json.dumps(r, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    assert digest_of(recipe) == declared  # the saved recipe is the one its identity names

    def crafted(
        name: str,
        generations: list[str | None],
        *,
        tamper: bool = False,
        resign: bool = True,
        shares: list[list[object]] | None = None,
    ) -> Path:
        # resign: a record made consistent with its edit (the recipe's digest in its identity),
        # so the checks after the binding are reached; without it, the identity is the original
        directory = tmp_path / name
        directory.mkdir()
        recipe["generations"] = generations
        if shares is not None:
            recipe["components"] = shares
        record_identity["recipe_sha256"] = digest_of(recipe) if resign else declared
        text = json.dumps(record)
        _ = (directory / "train.json").write_text(text, encoding="utf-8")
        digest = file_sha256(directory / "train.json")
        if tamper:
            _ = (directory / "train.json").write_text(text + " ", encoding="utf-8")
        manifest: dict[str, str] = {"train.json": digest}
        _ = (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory

    with pytest.raises(ValueError, match="inconsistent"):
        _ = resumed_components(data, crafted("kind-swapped", [pins[1], *pins[1:]]))
    with pytest.raises(ValueError, match="inconsistent"):
        _ = resumed_components(data, crafted("one-short", pins[:-1]))
    with pytest.raises(ValueError, match="no generation pin"):
        _ = resumed_components(data, crafted("unpinned", [None] * len(pins)))
    with pytest.raises(ValueError, match="hash differs"):
        _ = resumed_components(data, crafted("tampered", pins, tamper=True))
    with pytest.raises(ValueError, match="not a checkpoint"):
        _ = resumed_components(data, tmp_path / "absent")
    # the recipe consumed is bound to the identity beside it (maintainer B's review of a771dee):
    # other shares under the original identity, the manifest updated, are refused; the same shares
    # with a matching identity resolve (they are another run's record, not a tampered one)
    saved_shares = cast("list[list[object]]", recipe["components"])
    other = [
        [saved_shares[0][0], 0.97],
        *([s[0], 0.03 / (len(saved_shares) - 1)] for s in saved_shares[1:]),
    ]
    with pytest.raises(ValueError, match="recipe"):
        _ = resumed_components(data, crafted("reshared", pins, resign=False, shares=other))
    resolved = resumed_components(data, crafted("reshared-signed", pins, shares=other))
    assert [c.share for c in resolved] == [cast("float", s[1]) for s in other]


def _signed(record: dict[str, object]) -> dict[str, object]:
    """``record`` with an identity naming its recipe's digest, as ``identity_of`` writes it.

    For hand-made records, so each case reaches the check it is about; the binding itself is
    tested with the pinned resume. A record that has an identity, or no recipe mapping, is kept.
    """
    recipe = record.get("recipe")
    if not isinstance(recipe, dict) or "identity" in record:
        return record
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return {**record, "identity": {"recipe_sha256": hashlib.sha256(canonical.encode()).hexdigest()}}


def test_the_resolver_parses_exactly_the_bytes_it_hashed_and_checks_the_schema_first(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's fixtures (review of f55bfb3): train.json replaced right after its hash was
    # taken gave share 0.5 from bytes the manifest never validated; a genuine schema-3 checkpoint
    # reached KeyError 'generations' before any schema check
    import hashlib  # noqa: PLC0415

    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    import vesper.train as train_module  # noqa: PLC0415
    from vesper.generations import Unit, publish_generation, reference  # noqa: PLC0415
    from vesper.train import resumed_components  # noqa: PLC0415

    data = tmp_path / "data"
    staging = tmp_path / "staging"
    staging.mkdir()
    pq.write_table(pa.Table.from_pydict({"id": ["1"], "text": ["one"]}), str(staging / "a.parquet"))
    _ = (staging / "a.receipt.json").write_text("{}", encoding="utf-8")
    pin = reference(
        publish_generation(
            "code", staging, data, units=[Unit("a", ("a.parquet",), ("a.receipt.json",))]
        )
    ).pin

    def checkpoint(name: str, record: dict[str, object]) -> tuple[Path, Path, bytes]:
        directory = tmp_path / name
        directory.mkdir()
        original = json.dumps(_signed(record)).encode("utf-8")
        meta = directory / "train.json"
        _ = meta.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        manifest: dict[str, str] = {"train.json": digest}
        _ = (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory, meta, original

    recipe_good: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "components": [["code", 1.0]],
        "generations": [pin],
    }
    directory, meta, original = checkpoint("snapshot", {"recipe": recipe_good})
    recipe_altered: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "components": [["code", 0.5]],
        "generations": [pin],
    }
    altered_record: dict[str, object] = {"recipe": recipe_altered}
    altered = json.dumps(altered_record).encode("utf-8")
    reads = {"train.json": 0}
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def swap_after_hash(path: Path) -> str:
        digest = file_sha256(path)
        if path == meta:
            _ = meta.write_bytes(altered)  # the replacement lands right after the hash
        return digest

    def read_bytes(self: Path) -> bytes:
        if self == meta:
            reads["train.json"] += 1
            if reads["train.json"] > 1:
                return altered  # a second read sees the replacement
        return real_read_bytes(self)

    def read_text(self: Path, encoding: str | None = None, errors: str | None = None) -> str:
        if self == meta:
            reads["train.json"] += 1
            if reads["train.json"] > 1:
                return altered.decode("utf-8")
        return real_read_text(self, encoding, errors)

    monkeypatch.setattr(train_module, "file_sha256", swap_after_hash)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(Path, "read_text", read_text)
    parts = resumed_components(data, directory)
    assert [(c.name, c.share, c.generation) for c in parts] == [("code", 1.0, pin)]
    assert reads["train.json"] == 1  # one snapshot: the bytes hashed are the bytes parsed
    _ = meta.write_bytes(original)
    monkeypatch.undo()
    # an older schema is refused by its number before any field is read
    old, _, _ = checkpoint("old-schema", {"recipe": {"schema": 3, "components": [["code", 1.0]]}})
    with pytest.raises(ValueError, match="schema 3"):
        _ = resumed_components(data, old)
    # malformed records are refused by name, never a KeyError or an IndexError
    no_recipe, _, _ = checkpoint("no-recipe", {"identity": {}})
    with pytest.raises(ValueError, match="malformed"):
        _ = resumed_components(data, no_recipe)
    bad_components, _, _ = checkpoint(
        "bad-components",
        {"recipe": {"schema": CHECKPOINT_SCHEMA, "components": "code", "generations": [pin]}},
    )
    with pytest.raises(ValueError, match="malformed"):
        _ = resumed_components(data, bad_components)
    short_entry, _, _ = checkpoint(
        "short-entry",
        {"recipe": {"schema": CHECKPOINT_SCHEMA, "components": [["code"]], "generations": [pin]}},
    )
    with pytest.raises(ValueError, match="malformed"):
        _ = resumed_components(data, short_entry)
    not_json = tmp_path / "not-json"
    not_json.mkdir()
    _ = (not_json / "train.json").write_text("{not json", encoding="utf-8")
    not_json_manifest: dict[str, str] = {"train.json": hashlib.sha256(b"{not json").hexdigest()}
    _ = (not_json / "manifest.json").write_text(json.dumps(not_json_manifest))
    with pytest.raises(ValueError, match="malformed"):
        _ = resumed_components(data, not_json)


@pytest.mark.parametrize("name", ["model.pt", "optimizer.pt", "states.pt"])
def test_a_resume_loads_each_checkpoint_file_from_the_bytes_it_checked(
    tmp_path: Path, book_builder: Builder, name: str
) -> None:
    # restore hashed every file against the manifest and then loaded it by path again (the
    # shape of maintainer B's reviews of f55bfb3 and 72c9d31): a file replaced after its check (here
    # by bytes torch cannot read) must never be what the resume loads
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    out = tmp_path / "run"

    def run_for(steps: int, resume: Path | None) -> Path:
        return run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=steps,
            resume=resume,
            seed=0,
            device=CPU,
        )

    _ = run_for(4, None)
    last = out / "step-4"
    with replaced_after_first_read(last / name, b"not the checked file") as opens:
        log_path = run_for(1, last)
    assert len(opens) == 1
    lines = [
        cast("dict[str, object]", json.loads(line)) for line in log_path.read_text().splitlines()
    ]
    assert [line["step"] for line in lines if line["kind"] == "step"][-1] == 5


def test_the_command_line_scan_defaults_to_a_capturable_one_on_cuda() -> None:
    # CUDA work is always captured (the project owner 2026-10-21) and the reference scan cannot be:
    # without --scan a CUDA run takes the fused float32 scan, a CPU run the reference; a choice
    # stays
    assert _scan_for(None, torch.device("cuda")) == "fused-fp32"
    assert _scan_for(None, CPU) == "reference"
    assert _scan_for("fused", torch.device("cuda")) == "fused"


@pytest.mark.parametrize("objective", ["act", "fixed"])
@pytest.mark.parametrize("recompute", [False, True])
def test_the_head_keeps_its_logits_exactly_when_activations_are_kept(
    objective: str,
    recompute: bool,  # noqa: FBT001
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # one switch trades memory for speed everywhere (2026-10-21): without recomputation the
    # head keeps its logits for the backward pass too, with it they are recomputed
    seen: list[bool] = []

    def spy(
        x: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor, *, keep_logits: bool = False
    ) -> torch.Tensor:
        seen.append(keep_logits)
        return linear_nll(x, weight, targets, keep_logits=keep_logits)

    monkeypatch.setattr(vesper.model, "linear_nll", spy)
    cfg = training_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (40,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(8)
    ]
    groups = [
        Group(
            Stream(iter(docs), slots=2, chunk=cfg.training.chunk_tokens),
            model.init_states(2, torch.float32, CPU),
        )
    ]
    phase = phase_schedule(int(cfg.training.total_tokens * 0.5), 20, cfg)
    result = step(
        model,
        optimizer(model, cfg),
        groups,
        phase,
        cfg,
        CPU,
        recompute=recompute,
        objective=objective,
    )
    assert result is not None
    assert seen  # the head ran through the kernel
    assert set(seen) == {not recompute}


def test_safety_saves_change_no_training_keep_only_the_newest_and_resume(
    tmp_path: Path, book_builder: Builder
) -> None:
    # any positive interval has passed by the end of a step here, so every step that is not an
    # evaluated checkpoint gets a safety save: the steps equal those of a run without them, only
    # the newest safety save is kept, and a resume from it equals the uninterrupted next step
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )

    def train(out: Path, steps: int, resume: Path | None, safety: float | None) -> Path:
        return run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=steps,
            resume=resume,
            seed=0,
            device=CPU,
            safety_seconds=safety,
        )

    lines = read_log(train(tmp_path / "safe", 4, None, 1e-9))
    assert lines[0]["safety_seconds"] == 1e-9
    evaluated = {line["step"] for line in lines if line["kind"] == "checkpoint"}
    saved = [line for line in lines if line["kind"] == "safety"]
    trained = [line["step"] for line in lines if line["kind"] == "step"]
    # every step got exactly one save of one kind (the last one both: its final checkpoint
    # follows the step's safety save)
    assert [line["step"] for line in saved] == [s for s in trained if s not in evaluated or s == 4]
    assert len(saved) >= 2
    assert all(cast("float", line["seconds"]) >= 0 for line in saved)
    newest = Path(cast("str", saved[-1]["path"]))
    assert newest.parent == tmp_path / "safe" / "safety"
    assert newest.name == f"{lines[0]['run_id']}-step-4"
    assert sorted(p.name for p in newest.parent.iterdir()) == [newest.name]
    assert (newest / "manifest.json").is_file()
    record = cast(
        "dict[str, object]", json.loads((newest / "train.json").read_text(encoding="utf-8"))
    )
    assert record["reason"] == "safety"
    assert cast("dict[str, object]", record["progress"])["step"] == 4
    plain = read_log(train(tmp_path / "plain", 4, None, None))
    assert plain[0]["safety_seconds"] is None
    assert not (tmp_path / "plain" / "safety").exists()
    assert not [line for line in plain if line["kind"] == "safety"]
    for ours, theirs in zip(
        [line for line in lines if line["kind"] == "step"],
        [line for line in plain if line["kind"] == "step"],
        strict=True,
    ):
        assert ours["loss"] == theirs["loss"]
        assert ours["shares"] == theirs["shares"]
        assert ours["tokens_seen"] == theirs["tokens_seen"]
    # resume from the second-to-last safety save's step: a fresh copy of the run up to there
    middle = cast("int", saved[-2]["step"])
    _ = train(tmp_path / "upto", middle, None, 1e-9)
    kept = sorted((tmp_path / "upto" / "safety").iterdir())
    assert len(kept) == 1
    assert kept[0].name.endswith(f"-step-{middle}")
    resumed = read_log(train(tmp_path / "upto", 1, kept[0], None))
    header = [line for line in resumed if line["kind"] == "run"][-1]
    assert header["resumed_from"] == str(kept[0])
    after = [line for line in resumed if line["kind"] == "step"][-1]
    straight = [line for line in plain if line["kind"] == "step"][middle]
    assert after["step"] == middle + 1
    assert after["loss"] == pytest.approx(cast("float", straight["loss"]), rel=1e-6)
    assert after["shares"] == straight["shares"]
    assert after["tokens_seen"] == straight["tokens_seen"]


def test_a_failed_safety_save_is_logged_keeps_the_previous_and_training_goes_on(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    calls: list[Path] = []

    def failing_second(  # noqa: PLR0913
        path: Path,
        model: Vesper,
        opt: torch.optim.AdamW,
        groups: Sequence[Group],
        progress: Progress,
        identity: Identity,
        recipe: Recipe,
        *,
        reason: str,
        mixer: MixerState | None = None,
    ) -> Path:
        if reason == "safety":
            calls.append(path)
            if len(calls) == 2:
                raise OSError(28, "No space left on device")
        return checkpoint(
            path, model, opt, groups, progress, identity, recipe, reason=reason, mixer=mixer
        )

    monkeypatch.setattr(vesper.train, "checkpoint", failing_second)
    out = tmp_path / "run"
    lines = read_log(
        run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=4,
            resume=None,
            seed=0,
            device=CPU,
            safety_seconds=1e-9,
        )
    )
    kinds = [line["kind"] for line in lines if line["kind"] in {"safety", "safety_failed"}]
    assert kinds[:2] == ["safety", "safety_failed"]
    failed = next(line for line in lines if line["kind"] == "safety_failed")
    assert "No space left" in cast("str", failed["error"])
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 4
    saved = [Path(cast("str", line["path"])) for line in lines if line["kind"] == "safety"]
    remaining = sorted(p.name for p in (out / "safety").iterdir())
    if len(saved) == 1:  # nothing succeeded after the failure: the first save is still there
        assert remaining == [saved[0].name]
    else:
        assert remaining == [saved[-1].name]
    assert not calls[1].exists()


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
def test_a_safety_interval_that_is_not_a_positive_finite_time_is_refused_first(
    tmp_path: Path, book_builder: Builder, bad: float
) -> None:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )
    with pytest.raises(ValueError, match="safety_seconds"):
        _ = run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=tmp_path / "run",
            micro=1,
            steps=1,
            resume=None,
            seed=0,
            device=CPU,
            safety_seconds=bad,
        )
    assert not (tmp_path / "run").exists()


def test_the_command_line_takes_safety_minutes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def fake_run(*_args: object, **kwargs: object) -> Path:
        seen.append(kwargs["safety_seconds"])
        return tmp_path / "log.jsonl"

    def no_components(**_: object) -> list[Component]:
        return []

    monkeypatch.setattr(vesper.train, "components_for", no_components)
    monkeypatch.setattr(vesper.train, "run", fake_run)
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text(yaml.safe_dump(small_raw()), encoding="utf-8")
    common = [
        "--config", str(config_path), "--finalization", str(tmp_path / "f.json"),
        "--tokenizer", str(tmp_path / "t.json"), "--out", str(tmp_path / "out"),
        "--micro-chunks", "1", "--device", "cpu",
    ]  # fmt: skip
    assert vesper.train.main([*common, "--safety-minutes", "10"]) == 0
    assert vesper.train.main(common) == 0
    assert seen == [600.0, None]


def test_the_command_line_takes_cycling_and_the_finite_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[object, object]] = []

    def fake_run(*_args: object, **kwargs: object) -> Path:
        seen.append((kwargs["cycle"], kwargs["from_finite"]))
        return tmp_path / "log.jsonl"

    def no_components(**_: object) -> list[Component]:
        return []

    monkeypatch.setattr(vesper.train, "components_for", no_components)
    monkeypatch.setattr(vesper.train, "run", fake_run)
    config_path = tmp_path / "config.yaml"
    _ = config_path.write_text(yaml.safe_dump(small_raw()), encoding="utf-8")
    common = [
        "--config", str(config_path), "--finalization", str(tmp_path / "f.json"),
        "--tokenizer", str(tmp_path / "t.json"), "--out", str(tmp_path / "out"),
        "--micro-chunks", "1", "--device", "cpu",
    ]  # fmt: skip
    assert vesper.train.main(common) == 0
    assert vesper.train.main([*common, "--cycle-components"]) == 0
    assert vesper.train.main([*common, "--cycle-components", "--from-finite-checkpoint"]) == 0
    assert seen == [(False, False), (True, False), (True, True)]


class Train(Protocol):
    def __call__(
        self,
        out: Path,
        steps: int | None,
        *,
        resume: Path | None = None,
        cycle: bool = False,
        from_finite: bool = False,
    ) -> list[dict[str, object]]: ...


def cycling_runs(tmp_path: Path, book_builder: Builder) -> tuple[Train, int]:
    """A runner over one small corpus, and the step at which its finite run is exhausted."""
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 100_000, 1
    )

    def train(
        out: Path,
        steps: int | None,
        *,
        resume: Path | None = None,
        cycle: bool = False,
        from_finite: bool = False,
    ) -> list[dict[str, object]]:
        return read_log(
            run(
                cfg,
                config_path=config_path,
                parts=parts,
                finalization_receipt=receipt,
                tokenizer_path=tokenizer_path,
                out=out,
                micro=1,
                steps=steps,
                resume=resume,
                seed=0,
                device=CPU,
                cycle=cycle,
                from_finite=from_finite,
            )
        )

    finite = train(tmp_path / "finite", None)
    assert finite[-1]["reason"] == "exhausted"
    return train, cast("int", finite[-1]["step"])


def steps_of(lines: list[dict[str, object]]) -> list[dict[str, object]]:
    return [line for line in lines if line["kind"] == "step"]


def assert_same_steps(ours: list[dict[str, object]], theirs: list[dict[str, object]]) -> None:
    assert [s["step"] for s in ours] == [s["step"] for s in theirs]
    for a, b in zip(ours, theirs, strict=True):
        assert a["loss"] == pytest.approx(cast("float", b["loss"]), rel=1e-6)
        assert a["shares"] == b["shares"]
        assert a["consumed"] == b["consumed"]
        assert a["tokens_seen"] == b["tokens_seen"]


def final_record(out: Path, step: int) -> dict[str, object]:
    return cast("dict[str, object]", json.loads((out / f"step-{step}" / "train.json").read_text()))


def test_a_cycling_run_trains_past_every_run_out_and_resumes_on_both_sides_of_one(
    tmp_path: Path, book_builder: Builder
) -> None:
    train, exhausted = cycling_runs(tmp_path, book_builder)
    total = exhausted + 8
    straight = train(tmp_path / "straight", total, cycle=True)
    header = straight[0]
    assert cast("dict[str, object]", header["recipe"])["component_passes"] == CYCLE_ORDER_RULE
    assert header["transition"] is None
    assert straight[-1]["reason"] == "steps"
    assert [s["groups_active"] for s in steps_of(straight)] == [2] * total
    saved = cast(
        "dict[str, object]",
        torch.load(tmp_path / "straight" / f"step-{total}" / "states.pt", weights_only=True),
    )
    mixer = cast("dict[str, object]", saved["mixer"])
    assert max(cast("list[int]", mixer["passes"])) >= 1  # some component wrapped
    # before any run-out the cycling run is the finite one
    finite = read_log(tmp_path / "finite" / "log.jsonl")
    assert steps_of(straight)[0]["loss"] == steps_of(finite)[0]["loss"]
    for cut in (2, exhausted + 3):  # before the first run-out, after a wrap
        out = tmp_path / f"upto-{cut}"
        _ = train(out, cut, cycle=True)
        resumed = train(out, total - cut, resume=out / f"step-{cut}", cycle=True)
        assert_same_steps(steps_of(resumed)[cut:], steps_of(straight)[cut:])
        ours, theirs = final_record(out, total), final_record(tmp_path / "straight", total)
        assert ours["progress"] == theirs["progress"]  # documents digest, counts, consumed
        a = cast(
            "dict[str, torch.Tensor]",
            torch.load(out / f"step-{total}" / "model.pt", weights_only=True),
        )
        b = cast(
            "dict[str, torch.Tensor]",
            torch.load(tmp_path / "straight" / f"step-{total}" / "model.pt", weights_only=True),
        )
        for name, tensor in a.items():
            torch.testing.assert_close(tensor, b[name], rtol=1e-5, atol=1e-6)


def test_a_finite_checkpoint_enters_cycling_only_through_the_recorded_transition(
    tmp_path: Path, book_builder: Builder
) -> None:
    train, exhausted = cycling_runs(tmp_path, book_builder)
    total = exhausted + 4
    out = tmp_path / "parent"
    parent = train(out, 2)
    checkpoint = out / "step-2"
    with pytest.raises(ValueError, match="needs both"):
        _ = train(tmp_path / "no-cycle", 1, resume=checkpoint, from_finite=True)
    with pytest.raises(ValueError, match="needs both"):
        _ = train(tmp_path / "no-resume", 1, cycle=True, from_finite=True)
    assert not (tmp_path / "no-cycle").exists()
    assert not (tmp_path / "no-resume").exists()
    # without the transition a finite checkpoint is another recipe's, refused as always
    with pytest.raises(ValueError, match="recipe"):
        _ = train(out, 1, resume=checkpoint, cycle=True)
    # a finite checkpoint taken after a run-out (here the exhausted terminal one) is refused
    # before any cycling run is recorded (maintainer B bba62d1: it accepted and then trained
    # nothing)
    terminal = tmp_path / "finite" / f"step-{exhausted}"
    with pytest.raises(ValueError, match="cycling cannot continue"):
        _ = train(tmp_path / "late", 2, resume=terminal, cycle=True, from_finite=True)
    assert not (tmp_path / "late" / "log.jsonl").exists()
    child = train(out, total - 2, resume=checkpoint, cycle=True, from_finite=True)
    header = [line for line in child if line["kind"] == "run"][-1]
    assert header["transition"] == {
        "from": "finite",
        "to": CYCLE_ORDER_RULE,
        "parent_identity": parent[0]["identity"],
    }
    assert header["identity"] != parent[0]["identity"]
    # it continues exactly as a run that cycled from the start (pass 0 is the finite order)
    straight = train(tmp_path / "straight", total, cycle=True)
    assert_same_steps(steps_of(child)[2:], steps_of(straight)[2:])
    assert final_record(out, total)["identity"] == straight[0]["identity"]
    # its checkpoints are the cycling recipe's: they resume under cycling, not under finite
    more = train(out, 1, resume=out / f"step-{total}", cycle=True)
    assert steps_of(more)[-1]["step"] == total + 1
    with pytest.raises(ValueError, match="recipe"):
        _ = train(out, 1, resume=out / f"step-{total}")


def test_the_run_hashes_exactly_the_files_it_uses(tmp_path: Path, book_builder: Builder) -> None:
    # the identity's three file hashes and what the run parses are one snapshot of each file
    # (maintainer B's identity_of follow-up of 10-21: it hashed the paths, then loaded them again)
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )

    def train(out: Path) -> list[dict[str, object]]:
        return read_log(
            run(
                cfg,
                config_path=config_path,
                parts=parts,
                finalization_receipt=receipt,
                tokenizer_path=tokenizer_path,
                out=out,
                micro=1,
                steps=1,
                resume=None,
                seed=0,
                device=CPU,
            )
        )

    # the tokenizer is read once: swapped after that read, the run still uses and names the
    # file it read
    original = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    with replaced_after_first_read(tokenizer_path, b"not the checked tokenizer") as opens:
        lines = train(tmp_path / "tok")
    assert len(opens) == 1
    identity = cast("dict[str, object]", lines[0]["identity"])
    assert identity["tokenizer_sha256"] == original
    # the finalization receipt swapped between its hash and its parse is refused, not recorded
    with (
        replaced_after_first_read(receipt, b"{}") as opens,
        pytest.raises(ValueError, match="finalization receipt"),
    ):
        _ = train(tmp_path / "fin")
    assert not (tmp_path / "fin" / "log.jsonl").exists()
    # a config file that no longer says what the run was given is refused before anything runs
    edited = cast("dict[str, dict[str, object]]", yaml.safe_load(config_path.read_text("utf-8")))
    edited["training"]["eval_chunks"] = 2
    _ = config_path.write_text(yaml.safe_dump(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="changed since"):
        _ = train(tmp_path / "cfg")
    assert not (tmp_path / "cfg").exists()


def test_a_finite_recipe_keeps_the_record_and_hash_it_had_before_cycling() -> None:
    finite = recipe()
    cycling = replace(finite, component_passes=CYCLE_ORDER_RULE)
    plain = _plain(finite)  # the record and hash under test
    assert "component_passes" not in plain
    assert _plain(cycling)["component_passes"] == CYCLE_ORDER_RULE
    t, c, f = "1" * 64, "2" * 64, "3" * 64
    canonical = json.dumps(plain, sort_keys=True, separators=(",", ":")).encode()
    assert identity_of(t, c, f, finite).recipe_sha256 == hashlib.sha256(canonical).hexdigest()
    assert identity_of(t, c, f, cycling).recipe_sha256 != identity_of(t, c, f, finite).recipe_sha256


def test_a_safety_save_waits_for_its_interval_since_the_last_save_of_either_kind(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a clock that moves only while a step trains (10 s per step), so evaluations take no time:
    # with a 25 s interval a safety save falls on every third step after the last save of
    # either kind, never sooner (a budget of about 12 steps with an evaluated checkpoint every 6)
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 12 * 2 * 16, 2
    )
    clock = [0.0]
    real_step = vesper.train.step

    def ticking(  # noqa: PLR0913
        model: Vesper,
        opt: torch.optim.AdamW,
        groups: Sequence[Group],
        phase: Phase,
        cfg: VesperConfig,
        device: torch.device,
        *,
        recompute: bool = False,
        slices: int = 1,
        objective: str = "act",
        graphs: StepGraphs | None = None,
        ahead: Callable[[], None] | None = None,
    ) -> StepResult | None:
        result = real_step(
            model,
            opt,
            groups,
            phase,
            cfg,
            device,
            recompute=recompute,
            slices=slices,
            objective=objective,
            graphs=graphs,
            ahead=ahead,
        )
        clock[0] += 10.0
        return result

    monkeypatch.setattr(vesper.train, "step", ticking)
    monkeypatch.setattr(vesper.train, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    lines = read_log(
        run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=tmp_path / "run",
            micro=1,
            steps=8,
            resume=None,
            seed=0,
            device=CPU,
            safety_seconds=25.0,
        )
    )
    # the final checkpoint is written after the loop, so only the interval ones reset the clock
    interval = {
        line["step"]
        for line in lines
        if line["kind"] == "checkpoint" and line["reason"] == "interval"
    }
    assert interval
    expected: list[int] = []
    last = 0.0
    for s in range(1, 9):
        now = 10.0 * s
        if s in interval:
            last = now
        elif now - last >= 25.0:
            expected.append(s)
            last = now
    assert [line["step"] for line in lines if line["kind"] == "safety"] == expected
    assert expected  # the interval was reached at least once


def _safety_run(tmp_path: Path, book_builder: Builder) -> Callable[[Path], Path]:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 6 * 2 * 16, 2
    )

    def train(out: Path) -> Path:
        return run(
            cfg,
            config_path=config_path,
            parts=parts,
            finalization_receipt=receipt,
            tokenizer_path=tokenizer_path,
            out=out,
            micro=1,
            steps=4,
            resume=None,
            seed=0,
            device=CPU,
            safety_seconds=1e-9,
        )

    return train


def test_a_failed_cleanup_is_reported_with_its_published_save_and_training_goes_on(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B's ccb4e69 fixture: removing an older save fails after the new one is published;
    # the run must record the save, report the cleanup failure and train on (it aborted before)
    train = _safety_run(tmp_path, book_builder)
    out = tmp_path / "run"
    safety = out / "safety"
    original = shutil.rmtree

    def refusing(path: str | Path, *_args: object, **_kwargs: object) -> None:
        if Path(path).parent == safety and not Path(path).name.startswith("."):
            raise PermissionError(13, "injected cleanup refusal", str(path))
        original(path)

    monkeypatch.setattr(shutil, "rmtree", refusing)
    lines = read_log(train(out))
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 4
    saved = [line for line in lines if line["kind"] == "safety"]
    assert len(saved) >= 2
    assert saved[0]["cleanup_error"] is None  # nothing older to remove yet
    assert saved[0]["removed"] == []
    for line in saved[1:]:
        assert "PermissionError" in cast("str", line["cleanup_error"])
        assert line["removed"] == []
    assert not [line for line in lines if line["kind"] == "safety_failed"]
    # every published save is still there and complete
    names = sorted(Path(cast("str", line["path"])).name for line in saved)
    assert sorted(p.name for p in safety.iterdir()) == names
    assert all((safety / name / "manifest.json").is_file() for name in names)


def test_the_cleanup_removes_only_safety_saves_it_owns(
    tmp_path: Path, book_builder: Builder
) -> None:
    # Maintainer B's ccb4e69 fixtures: an operator's directory and a regular file in out/safety were
    # deleted or aborted the run; only real directories named like a safety save are removed
    train = _safety_run(tmp_path, book_builder)
    out = tmp_path / "run"
    safety = out / "safety"
    safety.mkdir(parents=True)
    note = safety / "operator-note"
    note.mkdir()
    _ = (note / "keep.txt").write_text("not a checkpoint", encoding="utf-8")
    _ = (safety / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
    stale = safety / "20200101T000000000000Z-1-step-7"  # an older run's save: owned
    stale.mkdir()
    _ = (stale / "model.pt").write_bytes(b"old")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _ = (elsewhere / "keep.txt").write_text("outside", encoding="utf-8")
    linked = safety / "20200101T000000000000Z-2-step-9"  # a save's name, but a symlink
    linked.symlink_to(elsewhere, target_is_directory=True)
    named_file = safety / "20200101T000000000000Z-3-step-1"  # a save's name, but a file
    _ = named_file.write_text("not a directory", encoding="utf-8")
    lines = read_log(train(out))
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 4
    saved = [line for line in lines if line["kind"] == "safety"]
    assert all(line["cleanup_error"] is None for line in saved)
    assert saved[0]["removed"] == [stale.name]
    newest = Path(cast("str", saved[-1]["path"])).name
    assert sorted(p.name for p in safety.iterdir()) == sorted(
        [newest, note.name, "notes.txt", linked.name, named_file.name]
    )
    assert (note / "keep.txt").read_text(encoding="utf-8") == "not a checkpoint"
    assert (elsewhere / "keep.txt").read_text(encoding="utf-8") == "outside"
    assert linked.is_symlink()


def test_a_failed_listing_of_the_safety_saves_is_reported_and_training_goes_on(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = _safety_run(tmp_path, book_builder)
    out = tmp_path / "run"
    safety = out / "safety"
    original = Path.iterdir

    def refusing(self: Path) -> Iterator[Path]:
        if self == safety:
            raise OSError(5, "injected listing failure", str(self))
        return original(self)

    monkeypatch.setattr(Path, "iterdir", refusing)
    lines = read_log(train(out))
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 4
    saved = [line for line in lines if line["kind"] == "safety"]
    assert saved
    assert all("injected listing failure" in cast("str", line["cleanup_error"]) for line in saved)
    monkeypatch.undo()
    names = sorted(Path(cast("str", line["path"])).name for line in saved)
    assert sorted(p.name for p in safety.iterdir()) == names


def test_a_real_serializer_failure_in_a_safety_save_is_logged_and_training_goes_on(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Maintainer B 8d6c1c7: a real write limit makes torch.save raise RuntimeError
    # (inline_container.cc), not OSError, and the trainer stopped. Here the second safety save's
    # files are really written to /dev/full (every write fails with ENOSPC), through torch's own
    # writer
    train = _safety_run(tmp_path, book_builder)
    out = tmp_path / "run"
    original: Callable[[object, str | Path], None] = torch.save
    saves: list[str] = []
    safety = out / "safety"

    def full_disk(obj: object, f: Path) -> None:  # checkpoint's call: torch.save(obj, path)
        target: str | Path = f
        # a safety save's file: out/safety/.<name>.<random>.staging/<name>/<file>
        if f.parent.parent.parent == safety:
            if f.parent.name not in saves:
                saves.append(f.parent.name)
            if len(saves) == 2:
                target = "/dev/full"
        original(obj, target)

    monkeypatch.setattr(torch, "save", full_disk)
    lines = read_log(train(out))
    assert lines[-1]["kind"] == "end"
    assert lines[-1]["step"] == 4
    failed = [line for line in lines if line["kind"] == "safety_failed"]
    assert len(failed) == 1
    assert cast("str", failed[0]["error"]).startswith("RuntimeError: ")
    saved = [line for line in lines if line["kind"] == "safety"]
    assert len(saved) >= 2
    # the failed save left nothing published; the newest successful one is the only save kept
    assert sorted(p.name for p in (out / "safety").iterdir() if not p.name.startswith(".")) == [
        Path(cast("str", saved[-1]["path"])).name
    ]


def test_an_error_outside_the_save_boundary_is_not_absorbed_by_the_safety_policy(
    tmp_path: Path, book_builder: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    # only file-system and serializer failures are a failed safety save; anything else (here a
    # ValueError, as checkpoint raises for an existing directory) stops the run as before
    train = _safety_run(tmp_path, book_builder)

    def refusing(path: Path, *_args: object, reason: str, **_kwargs: object) -> Path:
        if reason == "safety":
            msg = f"{path}: a checkpoint already exists there"
            raise ValueError(msg)
        return path / "manifest.json"

    monkeypatch.setattr(vesper.train, "checkpoint", refusing)
    with pytest.raises(ValueError, match="already exists"):
        _ = train(tmp_path / "run")
