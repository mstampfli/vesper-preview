import argparse
import copy
import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from conftest import small_raw
from vesper.column import set_scan
from vesper.config import InstructConfig, parse, parse_instruct_bytes
from vesper.decode import CapturedDecoder
from vesper.graphs import WARM_STEPS, GraphFamily
from vesper.instruct import (
    HALTING_PREFIX,
    ExampleLosses,
    Recipe,
    _on,  # pyright: ignore[reportPrivateUsage]
    evaluate_examples,
    final,
    generate,
    paired,
    recipe_for,
    train_step,
    tuning_optimizer,
)
from vesper.instruct_data import Example, Record, batches, collate, prompt_text, serialize
from vesper.instruct_panel import answer_of, load_panel, score_answer
from vesper.loop import DenseRule
from vesper.model import Vesper
from vesper.tokenizer import Tokenizer, train_from_texts

VOCAB = 320
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NO_HASH = "0" * 64
TINY = InstructConfig(
    dataset_name="test",
    dataset_revision="test",
    dataset_sha256=NO_HASH,
    split_key="test-split",
    train_below=0.9,
    validation_below=0.95,
    max_inputs=64,
    buckets=(32, 64),
    batch_tokens=128,
    peak_lr=3e-5,
    warmup_steps=1,
    final_lr_fraction=0.1,
    epochs=3,
    seed=0,
    eval_depth=2,
    resamples=200,
    max_new_tokens=6,
    days_micro=8,
    pin_sha256=NO_HASH,
    days_view_sha256=NO_HASH,
)
FRUIT: dict[str, object] = {"count": 3, "allowed": ["apple", "banana", "pear"]}


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    texts = [f"Instruction:\nsay {i}\n\nResponse:\nanswer {i}\n<end>" for i in range(80)]
    return train_from_texts(texts, vocab=VOCAB)


def tiny(seed: int = 0) -> Vesper:
    raw = small_raw(VOCAB)
    raw["model"]["t_max"] = 3
    return Vesper(parse(raw, "test"), torch.Generator().manual_seed(seed))


def recipe() -> Recipe:
    # the tiny model's setting; no clip, so a gradient compares exactly
    return dataclasses.replace(
        recipe_for(parse(small_raw(VOCAB), "test"), TINY), train_passes=3, grad_clip=1e9
    )


def ints(t: Tensor) -> list[int]:
    return cast("list[int]", t.tolist())  # pyright: ignore[reportUnknownMemberType]


def example(tok: Tokenizer, index: int, response: str, instruction: str = "say") -> Example:
    r = Record(index, f"{instruction} {index}", "", response, "open_qa")
    return serialize(tok, r, index, "train")


def frozen_opt(model: Vesper) -> torch.optim.AdamW:
    # the tuning optimizer at rate 0: nothing moves, and each parameter's .grad stays readable
    # after the step (the next step clears it first), so a step's gradient can be compared
    return tuning_optimizer(model, recipe(), capturable=False)


def grads(model: Vesper) -> dict[str, Tensor]:
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def delta(before: Vesper, after: Vesper) -> dict[str, Tensor]:
    b = dict(before.named_parameters())
    return {n: p.detach() - b[n].detach() for n, p in after.named_parameters()}


def test_the_optimizer_leaves_the_halting_head_out_and_unchanged(tok: Tokenizer) -> None:
    model = tiny()
    before = copy.deepcopy(model)
    opt = tuning_optimizer(model, recipe(), capturable=False)
    groups = cast("list[dict[str, object]]", opt.param_groups)
    in_groups = {id(p) for g in groups for p in cast("list[Tensor]", g["params"])}
    for name, p in model.named_parameters():
        assert (id(p) in in_groups) == (not name.startswith(HALTING_PREFIX))
        assert p.requires_grad == (not name.startswith(HALTING_PREFIX))
    batch = collate([example(tok, 0, "answer 0")], 32, 2)
    for _ in range(3):
        _ = train_step(model, opt, batch, 1e-2, recipe(), None)
    moved = delta(before, model)
    assert all(not bool(moved[n].any()) for n in moved if n.startswith(HALTING_PREFIX))
    assert any(bool(moved[n].any()) for n in moved if not n.startswith(HALTING_PREFIX))


def test_a_batch_is_its_rows_alone_pooled_over_their_labels(tok: Tokenizer) -> None:
    one, two = example(tok, 0, "answer 0"), example(tok, 1, "answer 1 answer 1 answer 1")
    runs: dict[str, tuple[float, dict[str, Tensor], int]] = {}
    for name, batch in (
        ("both", collate([one, two], 32, 2)),
        ("one", collate([one], 32, 1)),
        ("two", collate([two], 32, 1)),
        ("one-padded", collate([one], 64, 3)),  # longer rows and a filler row change nothing
    ):
        model = tiny()
        rec = train_step(model, frozen_opt(model), batch, 0.0, recipe(), None)
        runs[name] = (rec.loss, grads(model), rec.labels)
    (l1, d1, n1), (l2, d2, n2) = runs["one"], runs["two"]
    lb, db, nb = runs["both"]
    assert nb == n1 + n2
    assert lb == pytest.approx((n1 * l1 + n2 * l2) / nb, rel=1e-5)
    for name in db:
        expected = (n1 * d1[name] + n2 * d2[name]) / nb
        assert torch.allclose(db[name], expected, rtol=1e-4, atol=1e-6), name
    lp, dp, _ = runs["one-padded"]
    assert lp == pytest.approx(l1, rel=1e-5)
    assert all(torch.allclose(dp[n], d1[n], rtol=1e-4, atol=1e-6) for n in dp)


def test_no_state_crosses_from_one_update_to_the_next(tok: Tokenizer) -> None:
    first = collate([example(tok, 0, "answer 0 answer 0")], 32, 1)
    second = collate([example(tok, 1, "answer 1")], 32, 1)
    model = tiny()
    opt = frozen_opt(model)
    _ = train_step(model, opt, first, 0.0, recipe(), None)  # rate 0: the weights stay
    after = train_step(model, opt, second, 0.0, recipe(), None)
    after_grads = grads(model)
    fresh = tiny()
    alone = train_step(fresh, frozen_opt(fresh), second, 0.0, recipe(), None)
    assert after.loss == alone.loss
    assert after.pass_loss == alone.pass_loss
    fresh_grads = grads(fresh)
    assert all(torch.equal(after_grads[n], fresh_grads[n]) for n in fresh_grads)


def isolated(model: Vesper, e: Example, depth: int) -> tuple[float, int, float, int]:
    ids = e.ids
    n = e.inputs
    states = model.init_states(1, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        result = model.run(
            ids[None, :n], torch.arange(n)[None], states, passes=depth, exit_rule=DenseRule()
        )
        losses = model.position_losses(result.y, ids[None, 1:]).double()[0]
    p, b = int(e.prefix.shape[0]), int(e.body.shape[0])
    body = losses[p - 1 : p - 1 + b]
    term = losses[p - 1 + b :]
    return float(body.sum()), b, float(term.sum()), int(term.shape[0])


def test_evaluation_scores_every_example_as_if_alone(tok: Tokenizer) -> None:
    model = tiny()
    examples = [example(tok, i, "answer " * (1 + i % 5), "say") for i in range(9)]
    examples.append(example(tok, 9, ""))  # an empty body: only the terminator is scored
    plan = batches(examples, epoch=0, seed=0, batch_tokens=128, buckets=(32, 64))
    got = evaluate_examples(model, plan, depth=2, graphs=None)
    assert ints(got.index) == list(range(10))
    for row, e in enumerate(examples):
        body_sum, body_n, term_sum, term_n = isolated(model, e, 2)
        assert int(got.body_count[row]) == body_n
        assert int(got.term_count[row]) == term_n
        assert float(got.body_sum[row]) == pytest.approx(body_sum, rel=1e-5, abs=1e-6)
        assert float(got.term_sum[row]) == pytest.approx(term_sum, rel=1e-5)
    assert int(got.body_count[9]) == 0


def test_evaluation_scores_the_head_in_row_groups_with_the_same_losses(
    tok: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The 2026-10-25 preflights failed beside a 1 GiB float32 [8192, 32768] buffer: the
    # evaluation scored a whole batch through the head at once.
    model = tiny()
    examples = [example(tok, i, "answer " * (1 + i % 5), "say") for i in range(12)]
    plan = batches(examples, epoch=0, seed=0, batch_tokens=128, buckets=(32, 64))
    whole = evaluate_examples(model, plan, depth=2, graphs=None)
    rows: list[int] = []
    real = Vesper.position_losses

    def counted(self: Vesper, y: Tensor, targets: Tensor) -> Tensor:
        rows.append(int(y.shape[0]))
        return real(self, y, targets)

    monkeypatch.setattr(Vesper, "position_losses", counted)
    grouped = evaluate_examples(model, plan, depth=2, graphs=None, slices=3)
    widest = max(int(b.tokens.shape[0]) for b in plan)
    assert len(rows) > len(plan)  # more calls than batches
    assert max(rows) <= -(-widest // 3)  # no call scores more than a third of a batch's rows
    assert sum(rows) == sum(int(b.tokens.shape[0]) for b in plan)  # every row once
    assert torch.equal(grouped.index, whole.index)
    assert torch.equal(grouped.body_count, whole.body_count)
    assert torch.allclose(grouped.body_sum, whole.body_sum, rtol=1e-6, atol=1e-6)
    assert torch.allclose(grouped.term_sum, whole.term_sum, rtol=1e-6, atol=1e-6)
    rows.clear()
    _ = evaluate_examples(model, plan, depth=2, graphs=None, slices=1000)  # more than rows
    assert min(rows) >= 1
    with pytest.raises(ValueError, match="at least 1"):
        _ = evaluate_examples(model, plan, depth=2, graphs=None, slices=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
def test_cuda_evaluation_captures_each_length_once_for_the_whole_run(tok: Tokenizer) -> None:
    # The GPU preflight of f5fe880 recorded eval_captures 0: one graph shared by every length
    # was dropped at each length switch and re-warmed at every epoch, so evaluation ran eager.
    # One graph per length held across evaluations: each length warms once, then replays.
    short = [example(tok, i, "answer", "say") for i in range(24)]  # bucket 32
    long = [example(tok, 24 + i, "answer " * 18, "say") for i in range(12)]  # bucket 64
    plan = batches(short + long, epoch=0, seed=0, batch_tokens=128, buckets=(32, 64))
    lengths = sorted({int(b.tokens.shape[1]) for b in plan})
    per_length = {n: sum(int(b.tokens.shape[1]) == n for b in plan) for n in lengths}
    assert lengths == [32, 64]
    assert all(count > WARM_STEPS for count in per_length.values())
    cpu = evaluate_examples(tiny(), plan, depth=2, graphs=None)
    model = tiny().cuda()
    _ = set_scan(model, "fused-fp32")
    graphs = GraphFamily()
    first = evaluate_examples(model, plan, depth=2, graphs=graphs)
    second = evaluate_examples(model, plan, depth=2, graphs=graphs)
    keys = [("eval", n) for n in lengths]
    assert sorted(graphs.members, key=str) == keys
    assert {k: graphs.members[k].captures for k in keys} == dict.fromkeys(keys, 1)
    # every call after a length's first WARM_STEPS replays, across both evaluations
    assert {k: graphs.members[k].replays for k in keys} == {
        ("eval", n): 2 * per_length[n] - WARM_STEPS for n in lengths
    }
    for got in (first, second):
        assert torch.equal(got.index, cpu.index)
        assert torch.equal(got.body_count, cpu.body_count)
        assert torch.allclose(got.body_sum, cpu.body_sum, rtol=1e-4, atol=1e-5)
        assert torch.allclose(got.term_sum, cpu.term_sum, rtol=1e-4, atol=1e-5)
    assert torch.equal(first.body_sum, second.body_sum)


def losses(body: list[float], counts: list[int], groups: list[int]) -> ExampleLosses:
    n = len(body)
    return ExampleLosses(
        index=torch.arange(n),
        group=torch.tensor(groups),
        body_sum=torch.tensor(body, dtype=torch.float64),
        body_count=torch.tensor(counts),
        term_sum=torch.ones(n, dtype=torch.float64),
        term_count=torch.ones(n, dtype=torch.int64),
    )


def test_paired_pools_tokens_and_resamples_whole_groups() -> None:
    base = losses([2.0, 10.0, 3.0], [1, 10, 3], [0, 1, 1])
    tuned = losses([0.0, 10.0, 3.0], [1, 10, 3], [0, 1, 1])
    report = paired(base, tuned, resamples=500, seed=0)
    body = cast("dict[str, object]", report["body"])
    estimate = cast("float", body["delta"])
    assert estimate == pytest.approx(-2.0 / 14.0)  # pooled; a mean of example means is -2/3
    assert report["groups"] == 2
    low, high = cast("list[float]", body["interval_95"])
    assert low <= estimate <= high
    same = paired(base, base, resamples=100, seed=0)
    assert cast("dict[str, object]", same["body"])["interval_95"] == [0.0, 0.0]
    # one group: every resample draws it whole, so the interval collapses onto the estimate
    whole = paired(
        losses([2.0, 10.0, 3.0], [1, 10, 3], [5, 5, 5]),
        losses([0.0, 10.0, 3.0], [1, 10, 3], [5, 5, 5]),
        resamples=50,
        seed=0,
    )
    lone = cast("dict[str, object]", whole["body"])
    low_one, high_one = cast("list[float]", lone["interval_95"])
    assert low_one == pytest.approx(-2.0 / 14.0)
    assert high_one == pytest.approx(-2.0 / 14.0)
    with pytest.raises(ValueError, match="differ"):
        _ = paired(base, losses([0.0, 10.0, 3.0], [1, 9, 3], [0, 1, 1]), resamples=5, seed=0)
    with pytest.raises(ValueError, match="differ"):
        _ = paired(base, losses([0.0, 10.0, 3.0], [1, 10, 3], [0, 1, 2]), resamples=5, seed=0)


def test_paired_never_reports_a_nan_for_groups_without_targets() -> None:
    # group 0 is an empty response (no body target), group 1 is scored: maintainer B's e178dab case
    base = losses([0.0, 4.0], [0, 2], [0, 1])
    tuned = losses([0.0, 2.0], [0, 2], [0, 1])
    body = cast("dict[str, object]", paired(base, tuned, resamples=300, seed=0)["body"])
    assert body["status"] == "assessed"
    assert body["delta"] == pytest.approx(-1.0)
    assert (body["groups_contributing"], body["groups_without_targets"]) == (1, 1)
    interval = cast("list[float]", body["interval_95"])
    assert all(math.isfinite(v) for v in interval)
    assert interval[0] == pytest.approx(-1.0)
    assert interval[1] == pytest.approx(-1.0)
    # no body target anywhere: unassessed, no estimate, no interval
    none = paired(losses([0.0], [0], [0]), losses([0.0], [0], [0]), resamples=10, seed=0)
    nothing = cast("dict[str, object]", none["body"])
    assert nothing == {"status": "unassessed", "targets": 0, "groups_contributing": 0}
    assert cast("dict[str, object]", none["term"])["status"] == "assessed"


def test_answer_of_finds_the_marker_wherever_it_lands() -> None:
    assert answer_of("Paris\n<end>") == "Paris"
    assert answer_of("Paris<end> and more") == "Paris"
    assert answer_of("a\n\n<end>") == "a\n"
    assert answer_of("still going <en") is None


def test_generation_is_the_full_forwards_greedy_choice(tok: Tokenizer) -> None:
    model = tiny(3)
    prompt = prompt_text("say 4", "")
    g = generate(model, tok, prompt, depth=2, max_new=6)
    ids = torch.cat((tok.encode(prompt).long(), torch.tensor(g.ids, dtype=torch.int64)))
    n = int(ids.shape[0]) - 1
    states = model.init_states(1, torch.float32, torch.device("cpu"))
    with torch.no_grad():
        result = model.run(
            ids[None, :n], torch.arange(n)[None], states, passes=2, exit_rule=DenseRule()
        )
        choice = model.head(result.y).argmax(-1)[0]
    start = int(tok.encode(prompt).shape[0]) - 1
    assert ints(choice[start:]) == list(g.ids)
    assert g.stop in {"length", "terminator"}
    if g.stop == "length":
        assert len(g.ids) == 6


def test_generation_refuses_a_prompt_with_no_ids(tok: Tokenizer) -> None:
    # nothing would be computed; the captured path emitted id 0 (maintainer B's review of 0e342c3)
    with pytest.raises(ValueError, match="no ids"):
        _ = generate(tiny(3), tok, "", depth=2, max_new=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_decoder_made_for_other_settings_is_refused_before_any_step(
    tok: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda = torch.device("cuda")
    model, other = _on(tiny(3), cuda), _on(tiny(4), cuda)
    steps: list[int] = []
    real = CapturedDecoder.step

    def counted(
        self: CapturedDecoder, token: Tensor, position: Tensor, flat: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        steps.append(1)
        return real(self, token, position, flat)

    monkeypatch.setattr(CapturedDecoder, "step", counted)
    prompt = prompt_text("say 4", "")
    wrong = [
        CapturedDecoder(other, 2, tok.vocab_size),  # another model
        CapturedDecoder(model, 1, tok.vocab_size),  # another depth
        CapturedDecoder(model, 2, tok.vocab_size - 1),  # another vocabulary
    ]
    for decoder in wrong:
        with pytest.raises(ValueError, match="decoder was made for"):
            _ = generate(model, tok, prompt, depth=2, max_new=4, decoder=decoder)
    assert steps == []
    right = CapturedDecoder(model, 2, tok.vocab_size)
    _ = generate(model, tok, prompt, depth=2, max_new=4, decoder=right)
    assert steps  # the valid reuse control does step


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_captured_generation_gives_the_cpu_reference_ids(tok: Tokenizer) -> None:
    prompts = [prompt_text(f"say {i}", "") for i in range(4)]
    cpu = [generate(tiny(3), tok, p, depth=2, max_new=6) for p in prompts]
    model = _on(tiny(3), torch.device("cuda"))
    decoder = CapturedDecoder(model, 2, tok.vocab_size)
    gpu = [generate(model, tok, p, depth=2, max_new=6, decoder=decoder) for p in prompts]
    for c, g in zip(cpu, gpu, strict=True):
        assert (g.ids, g.stop, g.answer) == (c.ids, c.stop, c.answer)
    assert decoder.graph.captures == 1


@pytest.mark.parametrize(
    ("checks", "answer", "expected"),
    [
        ([{"kind": "exact", "value": "Oslo"}], " oslo. ", True),
        ([{"kind": "exact", "value": "Oslo"}], "Oslo is the city", False),
        ([{"kind": "exact", "value": "Oslo"}], "Oslo..", False),  # one period only
        ([{"kind": "verbatim", "value": "HELLO"}], " HELLO ", True),
        ([{"kind": "verbatim", "value": "HELLO"}], "HELLO.", False),
        ([{"kind": "verbatim", "value": "HELLO"}], "hello", False),
        ([{"kind": "one_of", "value": ["yes", "no"]}], "No.", True),
        ([{"kind": "one_of", "value": ["yes", "no"]}], "maybe", False),
        ([{"kind": "lines_in", "value": FRUIT}], "apple\n\nbanana\npear", True),
        ([{"kind": "lines_in", "value": FRUIT}], "stone\nstone\nstone", False),
        ([{"kind": "lines_in", "value": FRUIT}], "apple\napple\npear", False),
        ([{"kind": "lines_in", "value": FRUIT}], "apple\nbanana", False),
        ([{"kind": "regex", "value": r"1\s*,\s*2"}], " 1, 2 ", True),
        ([{"kind": "regex", "value": r"1\s*,\s*2"}], "1, 2, 3", False),
        # every check must pass
        ([{"kind": "exact", "value": "a"}, {"kind": "verbatim", "value": "A"}], "a", False),
        # the v2 keyword rule's survivors (maintainer B eb4d9f9): no explanation is scored any more
        ([{"kind": "contains_any", "value": ["rain"]}], "rain", "refused"),
        ([{"kind": "words", "value": 30}], "rain", "refused"),
        ([{"kind": "sentences", "value": 1}], "rain", "refused"),
    ],
)
def test_score_answer(
    checks: list[dict[str, object]], answer: str, *, expected: bool | str
) -> None:
    if expected == "refused":
        with pytest.raises(ValueError, match="unknown check kind"):
            _ = score_answer(checks, answer)
    else:
        assert score_answer(checks, answer) is expected


def test_score_answer_refuses_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown check kind"):
        _ = score_answer([{"kind": "vibes", "value": 1}], "x")


def test_the_panel_file_loads_with_every_counterexample_scored_as_declared() -> None:
    panel = load_panel((CONFIGS / "instruct-panel.json").read_bytes())
    assert len(panel.scored) == 24
    assert len(panel.illustrations) == 21
    assert panel.version == "instruct-panel-v3"
    # Maintainer B's eb4d9f9 survivors of format-08 are fail examples of its replacement or gone
    eight = next(item for item in panel.scored if item["id"] == "format-08")
    assert not score_answer(cast("list[dict[str, object]]", eight["checks"]), "rain")
    # every item has a wrong-content or extra-text answer that must fail (load_panel checked it)
    assert all(cast("list[str]", item["fail"]) for item in panel.scored)


def panel_with(**item: object) -> bytes:
    base: dict[str, object] = {
        "id": "x",
        "instruction": "Say red.",
        "context": "",
        "checks": [{"kind": "exact", "value": "red"}],
        "pass": ["red"],
        "fail": ["blue"],
    }
    base.update(item)
    none: list[object] = []
    panel: dict[str, object] = {"version": "t", "scored": [base], "illustrations": none}
    return json.dumps(panel).encode()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fail": ["Red."]}, "fail example"),  # a rubric that accepts what it should refuse
        ({"pass": ["the red one"]}, "pass example"),
        ({"fail": []}, "at least one"),
        ({"checks": [{"kind": "vibes", "value": 1}]}, "unknown check kind 'vibes'"),
        # after a check the pass example passes: still reached, still refused
        ({"checks": [{"kind": "exact", "value": "red"}, {"kind": "vibes", "value": 1}]}, "vibes"),
        ({"checks": []}, "checks missing"),
    ],
)
def test_a_panel_whose_rubric_does_not_separate_its_examples_is_refused(
    change: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = load_panel(panel_with(**change))


def test_a_malformed_panel_is_refused() -> None:
    bare: dict[str, object] = {"version": "t", "scored": [{"id": "x"}], "illustrations": []}
    with pytest.raises(ValueError, match="malformed"):
        _ = load_panel(json.dumps(bare).encode())  # an item without its fields
    two = cast("dict[str, list[object]]", json.loads(panel_with()))
    two["scored"] = two["scored"] * 2
    with pytest.raises(ValueError, match="duplicate id"):
        _ = load_panel(json.dumps(two).encode())


def test_final_never_reads_twice(tmp_path: Path) -> None:
    (tmp_path / "final.attempt").mkdir()
    assert final(argparse.Namespace(out=tmp_path)) == 3


def test_recipe_is_the_contracts_setting() -> None:
    raw = small_raw(VOCAB)
    raw["model"]["t_max"] = 8
    cfg = parse(raw, "test")
    shipped = parse_instruct_bytes((CONFIGS / "instruct-dolly.yaml").read_bytes(), "shipped")
    r = recipe_for(cfg, shipped)
    assert (r.peak_lr, r.warmup_steps, r.final_lr_fraction, r.epochs) == (3e-5, 100, 0.1, 3)
    assert (r.train_passes, r.eval_depth) == (8, 5)
    assert (r.max_inputs, r.buckets, r.batch_tokens) == (1024, (128, 256, 512, 1024), 8192)
    assert math.isclose(r.weight_decay, cfg.optimizer.weight_decay)
    assert r.grad_clip == cfg.optimizer.grad_clip
    # a depth the model cannot run is refused, not clipped
    with pytest.raises(ValueError, match="outside the model's passes"):
        _ = recipe_for(parse(small_raw(VOCAB), "test"), shipped)
    with pytest.raises(ValueError, match="slices"):
        _ = recipe_for(cfg, shipped, slices=0)
