import copy
import json
import math
from collections.abc import Callable, Sequence
from itertools import accumulate, pairwise
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import yaml

from conftest import small_config, small_raw
from vesper.config import VesperConfig, parse
from vesper.corpus import Component, finalize
from vesper.data import Chunk, Document, Stream, is_held_out
from vesper.keys import file_sha256
from vesper.learning import LossWeights
from vesper.loop import HaltingHead
from vesper.model import Vesper, wake_objective
from vesper.sample import components
from vesper.tokenizer import train_from_texts
from vesper.train import (
    CHECKPOINT_SCHEMA,
    Group,
    Identity,
    Layout,
    Progress,
    Recipe,
    checkpoint,
    decays,
    detach_states,
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
    return Recipe(0, (("a", 1.0),), (None,), 0.0, "finalize", "split", CHECKPOINT_SCHEMA)


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
    assert all(line["peak_memory_bytes"] == 0 for line in steps)  # the CPU has no counter
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

    def crafted(name: str, generations: list[str | None], *, tamper: bool = False) -> Path:
        directory = tmp_path / name
        directory.mkdir()
        recipe["generations"] = generations
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
        original = json.dumps(record).encode("utf-8")
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
