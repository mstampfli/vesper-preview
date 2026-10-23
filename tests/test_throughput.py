"""Tests for vesper.throughput: the sweep's grid, its log reading and its outcomes."""

import json
import subprocess
from pathlib import Path
from typing import TextIO, cast

import pytest
import yaml

from conftest import small_raw
from vesper import throughput
from vesper.throughput import Inputs, Measured, Setting, read_log, settings, table


def _log(
    path: Path, seconds: list[float], tokens: int = 1000, peaks: list[int] | None = None
) -> Path:
    gib = peaks if peaks is not None else [i + 1 for i in range(len(seconds))]
    steps: list[dict[str, object]] = [
        {"kind": "step", "seconds": s, "tokens": tokens, "peak_reserved_bytes": g * 2**30}
        for s, g in zip(seconds, gib, strict=True)
    ]
    first: dict[str, object] = {"kind": "run"}
    last: dict[str, object] = {"kind": "checkpoint"}
    lines = [first, *steps, last]
    _ = path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_the_grid_keeps_only_micro_batches_that_divide_the_batch() -> None:
    grid = settings([False, True], [8, 16, 12], [4], [16, 32])
    assert {(s.batch, s.micro) for s in grid} == {(16, 8), (16, 16), (32, 8), (32, 16)}
    assert len(grid) == 8  # both recompute modes for each
    assert len({s.name() for s in grid}) == len(grid)


def test_the_log_is_read_after_the_warm_up_and_capture(tmp_path: Path) -> None:
    # the first SKIP steps (first chunk, warm-up, capture) are slow and must not count
    path = _log(tmp_path / "log.jsonl", [9.0] * throughput.SKIP + [1.0, 2.0, 3.0, 2.0])
    median, rate, peak, count = read_log(path)
    assert (median, count) == (2.0, 4)
    assert rate == pytest.approx(4000 / 8.0)
    assert peak == pytest.approx(throughput.SKIP + 4)
    short = _log(tmp_path / "short.jsonl", [1.0] * (throughput.SKIP + 2))
    with pytest.raises(ValueError, match="measured steps"):
        _ = read_log(short)


def test_the_memory_peak_counts_the_warm_up_and_the_capture(tmp_path: Path) -> None:
    # the capture step holds the most (its pool beside the warm-up's tensors); a sweep that
    # read the measured steps alone would call a setting fitting that cannot be captured
    peaks = [3] * throughput.SKIP + [2] * 4
    peaks[throughput.SKIP - 1] = 9
    path = _log(tmp_path / "log.jsonl", [1.0] * (throughput.SKIP + 4), peaks=peaks)
    _, _, peak, _ = read_log(path)
    assert peak == pytest.approx(9.0)


def test_the_table_puts_the_fastest_fitting_setting_first() -> None:
    a, b, c = Setting(True, 8, 4, 16), Setting(False, 8, 4, 16), Setting(False, 16, 4, 16)
    results = [
        Measured(a, 2.0, 8000.0, 2.8, 10, None),
        Measured(c, None, None, None, 0, "out of memory"),
        Measured(b, 1.8, 9000.0, 8.5, 10, None),
    ]
    lines = table(results)
    assert lines[0].startswith(b.name())
    assert lines[1].startswith(a.name())
    assert lines[2].startswith(c.name())
    assert "out of memory" in lines[2]


def test_a_setting_that_runs_out_of_memory_is_recorded_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "base.yaml"
    _ = config.write_text(yaml.safe_dump(small_raw()))
    inputs = Inputs(config, tmp_path, tmp_path / "f.json", tmp_path / "t.json", 30, "x", "fixed")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append(command)
        handle = cast("TextIO", kwargs["stdout"])
        _ = handle.write("torch.OutOfMemoryError: CUDA out of memory.\n")
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    setting = Setting(recompute=False, micro=16, slices=4, batch=32)
    out = tmp_path / "out"
    out.mkdir()
    result = throughput.measure(setting, inputs, out)
    assert result.error == "out of memory"
    assert len(calls) == 1
    command = calls[0]
    assert "--recompute" not in command
    assert command[command.index("--micro-chunks") + 1] == "16"
    written = cast(
        "dict[str, dict[str, object]]",
        yaml.safe_load((out / setting.name() / "config.yaml").read_text()),
    )
    assert written["training"]["batch_chunks"] == 32  # the batch is the setting's, not the base's
