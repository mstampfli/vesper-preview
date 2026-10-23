"""Staggered cold restarts of the recurrent memory in training (training.state_reset_chunks).

The 16x d512 run of 2026-10-21 never reset its carried states: from its own training states the
final model reached held-out perplexity 36.4, from cold ones 351. These tests pin the schedule
(column.reset_recurrent, train.reset_keep), the config default, the trainer's use of it against a
manual cold restart, and the captured step against the eager one.
"""

from dataclasses import replace

import pytest
import torch

from conftest import small_raw
from test_train import training_config
from test_train_graph import (
    _assert_equal_runs,  # pyright: ignore[reportPrivateUsage]
    _docs,  # pyright: ignore[reportPrivateUsage]
)
from vesper.column import reset_recurrent
from vesper.config import ConfigError, VesperConfig, parse
from vesper.contracts import ContractError
from vesper.data import Document, Stream
from vesper.model import Vesper
from vesper.train import (
    RESET_LEVELS,
    RESET_RATIO,
    Group,
    Phase,
    optimizer,
    phase_schedule,
    reset_keep,
    step,
)

CPU = torch.device("cpu")
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_slots_have_mixed_lifetimes_staggered_never_restarting_at_the_start() -> None:
    # b = 12: two slots per level; periods 1, 4, 16, 64, 256 chunks and never (shortest 1)
    n, b, shortest, chunks = 4, 12, 1, 600
    restarts: dict[int, list[int]] = {s: [] for s in range(b)}
    for chunk in range(chunks):
        pos = (torch.arange(n) + chunk * n).expand(b, n)
        keep = reset_keep(pos, n, shortest, 0, b)
        assert keep.dtype == torch.float32
        for s in range(b):
            if keep[s] == 0:
                restarts[s].append(chunk)
    for s in range(b):
        level = s % RESET_LEVELS
        assert 0 not in restarts[s]  # a stream's first chunk is cold already
        if level == RESET_LEVELS - 1:
            assert restarts[s] == [], s  # the never-cleared lifetime
            continue
        period = shortest
        for _ in range(level):
            period *= RESET_RATIO
        gaps = [b - a for a, b in zip(restarts[s], restarts[s][1:], strict=False)]
        assert len(restarts[s]) >= chunks // period - 1, (s, len(restarts[s]))
        assert all(gap == period for gap in gaps), (s, gaps[:5])
    for level in range(1, RESET_LEVELS - 1):  # the two slots of a level are staggered
        assert restarts[level] != restarts[level + RESET_LEVELS], level


def test_a_slots_lifetime_does_not_depend_on_how_the_batch_is_grouped() -> None:
    # Maintainer B 9e3066c: numbered within a micro-group, six groups of two slots all had levels 0
    # and 1 (no slot ever kept its memory); numbered across the batch, every grouping of the same 12
    # slots gives every slot the same restarts as one group of 12
    n, total, shortest = 4, 12, 1
    for sizes in ([12], [6, 6], [2] * 6, [4, 8], [5, 7]):
        for chunk in range(300):
            pos = (torch.arange(n) + chunk * n).expand(total, n)
            whole = reset_keep(pos, n, shortest, 0, total)
            parts: list[torch.Tensor] = []
            first = 0
            for size in sizes:
                parts.append(reset_keep(pos[first : first + size], n, shortest, first, total))
                first += size
            assert torch.equal(torch.cat(parts), whole), (sizes, chunk)
    never = [s for s in range(total) if s % RESET_LEVELS == RESET_LEVELS - 1]
    pos = (torch.arange(n) + 300 * n).expand(2, n)
    assert reset_keep(pos, n, shortest, never[0] - 1, total)[1] == 1  # in a group of two


def test_rows_outside_the_batch_are_refused() -> None:
    pos = torch.zeros(4, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="not inside"):
        _ = reset_keep(pos, 2, 1, 10, 12)
    with pytest.raises(ValueError, match="not inside"):
        _ = reset_keep(pos, 2, 1, -1, 12)


def test_a_keep_that_is_not_one_value_per_row_is_refused() -> None:
    # Maintainer B 9e3066c: keep [1] against three rows broadcast and silently cleared every row
    model = Vesper(training_config(), torch.Generator().manual_seed(0))
    state = model.init_states(3, torch.float32, CPU)[0]
    for keep in (torch.zeros(1), torch.ones(3, 1), torch.ones(4)):
        with pytest.raises(ContractError, match="keep"):
            _ = reset_recurrent(state, keep)


def test_a_restart_zeroes_exactly_the_recurrent_memory_of_its_rows() -> None:
    model = Vesper(training_config(), torch.Generator().manual_seed(0))
    state = model.init_states(3, torch.float32, CPU)[0]
    g = torch.Generator().manual_seed(1)
    h = tuple(torch.randn(t.shape, generator=g) for t in state.h)
    z = tuple(torch.randn(t.shape, generator=g) for t in state.key_history)
    h[0][1, 0, 0, 0] = float("nan")  # a non-finite value in a restarted row cannot survive
    filled = type(state)(h=h, kv=state.kv, key_history=z)
    out = reset_recurrent(filled, (torch.arange(3) != 1).to(torch.float32))  # row 1 restarts
    for new, old in zip(out.h + out.key_history, h + z, strict=True):
        assert torch.equal(new[1], torch.zeros_like(new[1]))
        assert torch.equal(new[0], old[0])
        assert torch.equal(new[2], old[2])
    assert out.kv is filled.kv  # the attention windows are untouched


def test_the_config_reads_absent_as_never_and_refuses_a_negative_period() -> None:
    assert training_config().training.state_reset_chunks == 0
    raw = small_raw()
    raw["training"]["state_reset_chunks"] = 16
    assert parse(raw, "test").training.state_reset_chunks == 16
    raw["training"]["state_reset_chunks"] = -1
    with pytest.raises(ConfigError, match="state_reset_chunks"):
        _ = parse(raw, "test")


def _with_reset(cfg: VesperConfig, every: int) -> VesperConfig:
    return replace(cfg, training=replace(cfg.training, state_reset_chunks=every))


def test_a_restarting_step_equals_a_step_from_manually_cold_memory() -> None:
    # shortest = 1 with two slots: slot 0 (period 1) restarts at every chunk after its first,
    # slot 1 (period 4) not before its fourth chunk; each of three steps must equal the
    # never-resetting trainer's step after zeroing slot 0's h and key_history by hand
    docs = [
        Document(
            f"d{i}",
            torch.randint(1, 64, (40,), generator=torch.Generator().manual_seed(i)).to(torch.int32),
        )
        for i in range(8)
    ]
    base = training_config()

    def run(cfg: VesperConfig, *, manual: bool) -> list[float]:
        model = Vesper(cfg, torch.Generator().manual_seed(0))
        opt = optimizer(model, cfg)
        stream = Stream(iter(docs), slots=2, chunk=cfg.training.chunk_tokens)
        groups = [Group(stream, model.init_states(2, torch.float32, CPU))]
        losses: list[float] = []
        for index in range(3):
            if manual and index > 0:
                slot1_only = torch.arange(2).ne(0).to(torch.float32)
                groups[0].states = [reset_recurrent(s, slot1_only) for s in groups[0].states]
            result = step(model, opt, groups, phase_schedule(0, index, cfg), cfg, CPU)
            assert result is not None
            losses.append(result.loss)
        return losses

    restarting = run(_with_reset(base, 1), manual=False)
    assert restarting == run(base, manual=True)
    assert restarting != run(base, manual=False)  # the restart changes something


def test_the_trainer_numbers_slots_across_its_groups() -> None:
    # two groups of two slots, shortest 1: the batch's slots 0..3 have periods 1, 4, 16, 64, so
    # in three steps only group 0's first slot restarts (at steps 2 and 3); numbered within each
    # group, group 1's first slot would restart too
    base = training_config()

    def docs(seed: int) -> list[Document]:
        g = torch.Generator().manual_seed(seed)
        return [
            Document(f"d{seed}-{i}", torch.randint(1, 64, (40,), generator=g).to(torch.int32))
            for i in range(6)
        ]

    def run(cfg: VesperConfig, *, manual: bool) -> list[float]:
        model = Vesper(cfg, torch.Generator().manual_seed(0))
        opt = optimizer(model, cfg)
        groups = [
            Group(
                Stream(iter(docs(s)), slots=2, chunk=cfg.training.chunk_tokens),
                model.init_states(2, torch.float32, CPU),
            )
            for s in (1, 2)
        ]
        losses: list[float] = []
        for index in range(3):
            if manual and index > 0:
                first_cold = torch.arange(2).ne(0).to(torch.float32)
                groups[0].states = [
                    reset_recurrent(state, first_cold) for state in groups[0].states
                ]
            result = step(model, opt, groups, phase_schedule(0, index, cfg), cfg, CPU)
            assert result is not None
            losses.append(result.loss)
        return losses

    assert run(_with_reset(base, 1), manual=False) == run(base, manual=True)


def test_a_dry_group_keeps_its_slots_place() -> None:
    # group 0 has no documents (it never draws); group 1's slots stay the batch's slots 2 and 3
    # (periods 16 and 64), so in three steps nothing restarts. Numbered among the groups that
    # drew, they would be slots 0 and 1, and slot 0 would restart at every chunk.
    base = training_config()
    g = torch.Generator().manual_seed(3)
    docs = [
        Document(f"e{i}", torch.randint(1, 64, (40,), generator=g).to(torch.int32))
        for i in range(6)
    ]

    def run(cfg: VesperConfig) -> list[float]:
        model = Vesper(cfg, torch.Generator().manual_seed(0))
        opt = optimizer(model, cfg)
        n = cfg.training.chunk_tokens
        groups = [
            Group(Stream(iter([]), slots=2, chunk=n), model.init_states(2, torch.float32, CPU)),
            Group(Stream(iter(docs), slots=2, chunk=n), model.init_states(2, torch.float32, CPU)),
        ]
        losses: list[float] = []
        for index in range(3):
            result = step(model, opt, groups, phase_schedule(0, index, cfg), cfg, CPU)
            assert result is not None
            losses.append(result.loss)
        return losses

    assert run(_with_reset(base, 1)) == run(base)


@needs_cuda
def test_a_captured_restarting_run_equals_the_eager_one() -> None:
    cfg = _with_reset(training_config(), 2)

    def phases(index: int) -> Phase:
        return phase_schedule(int(cfg.training.total_tokens * 0.5), 20 + index, cfg)

    docs = [_docs(1, 12, 64), _docs(2, 12, 64)]
    _ = _assert_equal_runs(cfg, docs, phases, 8, "fixed", captures=1)
