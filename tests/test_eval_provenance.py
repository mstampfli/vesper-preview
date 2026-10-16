"""Checkpoint metrics must name the forward that actually measured them."""

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch

from test_train import read_log, run_fixture
from vesper import train
from vesper.config import VesperConfig
from vesper.loop import HaltingHead
from vesper.model import Vesper


def test_checkpoint_separates_pre_update_training_and_post_update_evaluation(
    tmp_path: Path,
    book_builder: Callable[[Path, str, dict[str, str]], Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg, config_path, parts, receipt, tokenizer_path = run_fixture(
        tmp_path, book_builder, total_tokens=320, checkpoints=10
    )
    schedule = train.phase_schedule
    make_optimizer = train.optimizer

    def two_pass_warm(tokens_seen: int, step: int, cfg: VesperConfig) -> train.Phase:
        return replace(schedule(tokens_seen, step, cfg), t_max=2, halting_frozen=True)

    def optimizer_that_changes_depth(model: Vesper, cfg: VesperConfig) -> torch.optim.AdamW:
        opt = make_optimizer(model, cfg)
        update = cast("Callable[[], float | None]", opt.step)

        def update_then_halt() -> None:
            _ = update()
            # A controlled post-update head change: the real training forward
            # used two passes; the checkpoint and its real evaluation use one.
            head = cast("HaltingHead", model.loop.halting)
            with torch.no_grad():
                _ = head.w_h.zero_()
                _ = head.b_h.fill_(10000.0)

        monkeypatch.setattr(opt, "step", update_then_halt)
        return opt

    monkeypatch.setattr(train, "phase_schedule", two_pass_warm)
    monkeypatch.setattr(train, "optimizer", optimizer_that_changes_depth)
    log = train.run(
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
        device=torch.device("cpu"),
    )
    rows = read_log(log)
    training = next(row for row in rows if row["kind"] == "step")
    saved = next(row for row in rows if row["kind"] == "checkpoint")
    assert training["histogram"] == [0.0, 1.0]
    assert saved["training_iteration_histogram"] == [0.0, 1.0]
    assert saved["training_tokens_with_loss"] == training["tokens_with_loss"]
    assert saved["training_measurement"] == "last-step-pre-update"
    assert saved["training_t_max"] == 2
    assert saved["held_out_iteration_histogram"] == [1.0, 0.0]
    positions = cast("int", saved["held_out_tokens_with_loss"])
    assert positions > 0
    assert saved["held_out_iteration_counts"] == [positions, 0]
    assert saved["held_out_measurement"] == "checkpoint-post-update"
    assert saved["held_out_precision"] == "float32"
    assert saved["held_out_t_max"] == 2
    assert saved["held_out_perplexity"] == pytest.approx(
        math.exp(cast("float", saved["held_out_loss_sum"]) / positions)
    )
    assert "iteration_histogram" not in saved
