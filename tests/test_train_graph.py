"""The trainer's step on CUDA is one captured graph, equal to the same step run eagerly.

The project owner 2026-10-21: CUDA graphs are a hard requirement for anything on the GPU. These
tests run the real ``vesper.train.step`` twice from identical starts: captured (``StepGraphs()``)
and as the eager oracle (``StepGraphs(warm=...)`` larger than the run, so it never captures), inside
``graphs.deterministic_algorithms``, and require every step's record, every parameter, every
optimizer tensor and every carried state to be bitwise equal, AND that the captured run
actually captured and replayed (a run that silently stayed eager would pass the equality).
"""

import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from test_train import Builder, run_fixture, training_config
from vesper.column import set_scan
from vesper.config import VesperConfig
from vesper.data import Document, Stream
from vesper.graphs import deterministic_algorithms
from vesper.model import Vesper
from vesper.train import (
    Group,
    Phase,
    StepGraphs,
    StepResult,
    optimizer,
    phase_schedule,
    run,
    step,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
CUDA = torch.device("cuda")
NEVER = 10**9  # a warm count no test reaches: the eager oracle


def _docs(seed: int, count: int, length: int) -> list[Document]:
    return [
        Document(
            f"s{seed}-d{i}",
            torch.randint(
                1, 64, (length,), generator=torch.Generator().manual_seed(seed * 100 + i)
            ).to(torch.int32),
        )
        for i in range(count)
    ]


def _start(
    cfg: VesperConfig, docs: list[list[Document]]
) -> tuple[Vesper, torch.optim.AdamW, list[Group]]:
    model = Vesper(cfg, torch.Generator().manual_seed(0)).to(CUDA)
    _ = set_scan(model, "fused-fp32")  # the reference scan cannot be captured
    opt = optimizer(model, cfg, capturable=True)
    groups = [
        Group(
            Stream(iter(group_docs), slots=2, chunk=cfg.training.chunk_tokens),
            model.init_states(2, torch.float32, CUDA),
        )
        for group_docs in docs
    ]
    return model, opt, groups


def _trace(
    cfg: VesperConfig,
    docs: list[list[Document]],
    phases: Callable[[int], Phase],
    steps: int,
    objective: str,
    graphs: StepGraphs,
) -> tuple[list[StepResult], list[Tensor]]:
    """``steps`` real steps; every record and the end state (parameters, Adam, carried)."""
    model, opt, groups = _start(cfg, docs)
    results: list[StepResult] = []
    with deterministic_algorithms():
        for index in range(steps):
            result = step(
                model,
                opt,
                groups,
                phases(index),
                cfg,
                CUDA,
                objective=objective,
                graphs=graphs,
            )
            assert result is not None
            results.append(result)
    torch.cuda.synchronize()
    end: list[Tensor] = [p.detach().clone() for p in model.parameters()]
    adam = cast("dict[Tensor, dict[str, Tensor]]", opt.state)
    for state in adam.values():
        end += [state[name].clone() for name in sorted(state)]
    for group in groups:
        for column in group.states:
            end += [*column.h, *column.key_history]
            end += [t for c in column.kv for t in (c.k, c.v, c.pos)]
    return results, end


def _same(a: StepResult, b: StepResult) -> bool:
    return (
        a.loss == b.loss
        and a.ntp == b.ntp
        and a.ponder == b.ponder
        and a.grad_norm == b.grad_norm
        and a.histogram == b.histogram
        and a.pass_ntp == b.pass_ntp
        and a.tokens == b.tokens
        and a.groups_active == b.groups_active
    )


def _assert_equal_runs(
    cfg: VesperConfig,
    docs: list[list[Document]],
    phases: Callable[[int], Phase],
    steps: int,
    objective: str,
    captures: int,
) -> list[StepResult]:
    captured = StepGraphs()
    eager = StepGraphs(warm=NEVER)
    got, got_end = _trace(cfg, docs, phases, steps, objective, captured)
    want, want_end = _trace(cfg, docs, phases, steps, objective, eager)
    assert captured.captures == captures  # it really captured (and the oracle never did)
    assert eager.captures == 0
    for index, (a, b) in enumerate(zip(got, want, strict=True)):
        assert _same(a, b), f"step {index}: {a} != {b}"
    assert len(got_end) == len(want_end)
    assert all(torch.equal(x, y) for x, y in zip(got_end, want_end, strict=True))
    return got


def test_a_captured_fixed_objective_run_equals_the_eager_one_through_the_tail() -> None:
    # depth-first training (the text pilot's objective) over two groups; the second group runs
    # dry part-way (a new signature: one group), so the run warms and captures twice after the
    # first step's empty attention windows; every record and the end state bitwise equal
    cfg = training_config()
    long, short = _docs(1, 12, 64), _docs(2, 5, 64)
    loop = phase_schedule(int(cfg.training.total_tokens * 0.5), 20, cfg)

    def phases(index: int) -> Phase:
        return phase_schedule(int(cfg.training.total_tokens * 0.5), 20 + index, cfg)

    assert loop.t_max == cfg.model.t_max
    results = _assert_equal_runs(cfg, [long, short], phases, 16, "fixed", captures=2)
    assert results[0].groups_active == 2
    assert results[-1].groups_active == 1  # the tail was exercised
    assert all(r.pass_ntp is not None for r in results)


def test_a_captured_act_run_reads_the_ramping_halting_weight_and_survives_a_phase_switch() -> None:
    # the ACT objective with lambda_h ramping every step across the loop phase (a float frozen
    # into the graph would differ from the eager oracle at once), and a switch from the frozen
    # warm head to the free one at step 6 (a new signature: re-warmed and captured again)
    cfg = training_config()
    total = cfg.training.total_tokens
    warm_end, loop_end = cfg.training.phase_warm_end, cfg.training.phase_loop_end

    def phases(index: int) -> Phase:
        if index < 6:  # late warm: frozen head, the full pass count
            return phase_schedule(int(total * warm_end * 0.95), index, cfg)
        fraction = warm_end + (loop_end - warm_end) * (index - 5) / 12
        return phase_schedule(int(total * fraction), index, cfg)

    ramp = [phases(i).lambda_h for i in range(6, 14)]
    assert len(set(ramp)) == len(ramp)  # a different weight every step
    assert phases(5).halting_frozen
    assert not phases(6).halting_frozen
    assert phases(5).t_max == phases(6).t_max
    docs = [_docs(3, 10, 80)]
    ramped = _assert_equal_runs(cfg, docs, phases, 14, "act", captures=2)
    # the oracle reads the same device scalar, so the equality alone cannot show the weight
    # arrives: a run whose weight is zero throughout must differ exactly where it is nonzero
    zero, _ = _trace(cfg, docs, lambda i: replace(phases(i), lambda_h=0.0), 14, "act", StepGraphs())
    assert all(a.loss == b.loss for a, b in zip(ramped[:6], zero[:6], strict=True))  # warm: 0
    assert all(a.loss != b.loss for a, b in zip(ramped[6:], zero[6:], strict=True))


def test_a_group_that_gains_its_first_loss_position_is_captured_anew() -> None:
    # one-token documents carry no loss (each target opens the next document), so the second
    # group skips the head while they last; a graph captured then must not be replayed once
    # that group carries a loss, or its loss would never train
    cfg = training_config()
    singles = [
        Document(f"one{i}", torch.full((1,), 7 + i % 50, dtype=torch.int32)) for i in range(200)
    ]
    results = _assert_equal_runs(
        cfg, [_docs(5, 12, 64), singles + _docs(6, 8, 64)], _loop_phase(cfg), 14, "fixed", 2
    )
    first, last = results[1].tokens_with_loss, results[-1].tokens_with_loss
    assert last > first  # the second group's loss arrived within the run


def _loop_phase(cfg: VesperConfig) -> Callable[[int], Phase]:
    def phases(index: int) -> Phase:
        return phase_schedule(int(cfg.training.total_tokens * 0.5), 20 + index, cfg)

    return phases


def test_a_cuda_step_without_graphs_is_refused_before_it_draws() -> None:
    cfg = training_config()
    model, opt, groups = _start(cfg, [_docs(4, 2, 40)])
    with pytest.raises(ValueError, match="captured CUDA graph"):
        _ = step(model, opt, groups, phase_schedule(0, 0, cfg), cfg, CUDA)
    # nothing was drawn: the next (captured) step reads the stream's first chunk
    first = step(model, opt, groups, phase_schedule(0, 0, cfg), cfg, CUDA, graphs=StepGraphs())
    assert first is not None
    assert first.tokens == 2 * cfg.training.chunk_tokens


def test_a_cuda_run_trains_and_evaluates_captured_and_says_so_in_its_log(
    tmp_path: Path, book_builder: Builder
) -> None:
    # the whole driver on CUDA: every step after the warm-up replays a graph, the checkpoint
    # evaluations run captured, and the end line records the captures and replays (the proof a
    # run trained captured); the reference scan is refused before anything is written
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, 10 * 2 * 16, 2
    )

    def train(out: Path, scan: str) -> Path:
        return run(
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
            device=CUDA,
            scan=scan,
            objective="fixed",
        )

    log_path = train(tmp_path / "run", "fused-fp32")
    lines = [cast("dict[str, object]", json.loads(x)) for x in log_path.read_text().splitlines()]
    end = lines[-1]
    assert end["kind"] == "end"
    graph = cast("dict[str, int]", end["step_graph"])
    assert graph["captures"] >= 1
    assert graph["replays"] >= 3
    checkpoints = [x for x in lines if x["kind"] == "checkpoint"]
    assert checkpoints
    for line in checkpoints:
        static = cast("list[float]", line["held_out_static_loss"])
        assert all(math.isfinite(v) for v in static)
        assert math.isfinite(cast("float", line["held_out_perplexity"]))
    with pytest.raises(ValueError, match="captured CUDA graph"):
        _ = train(tmp_path / "reference", "reference")
    assert not (tmp_path / "reference").exists()
