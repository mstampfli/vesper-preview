"""Repeatable GPU training for the halting calibration (vesper.graphs.deterministic_algorithms).

Measured 2026-10-19 (torch 2.14.1+cu130, RTX 5060 Ti): CUDA calibration runs with bitwise-equal
inputs differed from the first updates (test size: from update 6 in one process; real source:
from update 0 across processes), eager and captured, fused or reference scan; with torch's strict
deterministic algorithms they are bitwise equal, with or without a cuBLAS workspace preset. Two
runs are the truth here; a nondeterministic failure is not required to recur on every machine,
so the guard itself is pinned by contract tests and mutants, and the repeat by equality.
"""

import hashlib
import itertools
import json
import os
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.graphs import CUBLAS_WORKSPACES, deterministic_algorithms, fill_uninitialized_memory
from vesper.halting_calibration import run
from vesper.halting_experiment import run as experiment_run
from vesper.kernels.delta_scan import fused_scan_available

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "halting-stage1.yaml"
PROBE = ROOT / "tests" / "calibration_repeat_probe.py"
needs_fused = pytest.mark.skipif(
    not fused_scan_available(), reason="the calibration's CUDA training needs the fused scan"
)


def _flags() -> tuple[bool, bool]:
    return (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )


@contextmanager
def _keeping_flags() -> Generator[None]:
    """Whatever a test sets, the session's flags afterwards are what they were before it."""
    before = _flags()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(before[0], warn_only=before[1])


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("source") / "run"
    _ = experiment_run(CONFIG, "fixed", 0, out, steps=2, eval_batches=1, eval_size=2)
    return out / "model.pt"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_switch_is_strict_inside_and_restores_both_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with _keeping_flags():
        for enabled, warn_only in itertools.product((False, True), repeat=2):
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
            assert _flags() == (enabled, warn_only)  # all four combinations are observable
            with deterministic_algorithms() as settings:
                assert _flags() == (True, False)
                inside = (settings["deterministic_algorithms"], settings["warn_only"])
                assert inside == (True, False)
                with deterministic_algorithms():  # nested
                    assert _flags() == (True, False)
                assert _flags() == (True, False)
            assert _flags() == (enabled, warn_only)
            boom = "boom"
            with pytest.raises(ValueError, match=boom), deterministic_algorithms():
                raise ValueError(boom)
            assert _flags() == (enabled, warn_only)  # an exception inside restores both


def test_uninitialized_memory_is_not_filled_inside_and_the_setting_is_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 2026-10-21, 5090: torch fills every new tensor under deterministic mode (~2700 kernels of a
    # ~6000-kernel step); the scope turns that off, records it, and restores the caller's value
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    before = fill_uninitialized_memory()
    try:
        for outside in (True, False):
            torch.utils.deterministic.fill_uninitialized_memory = outside  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
            with _keeping_flags(), deterministic_algorithms() as settings:
                assert fill_uninitialized_memory() is False
                assert settings["fill_uninitialized_memory"] is False
                with deterministic_algorithms():
                    assert fill_uninitialized_memory() is False
                assert fill_uninitialized_memory() is False
            assert fill_uninitialized_memory() is outside
            with (
                _keeping_flags(),
                pytest.raises(ValueError, match="boom"),
                deterministic_algorithms(),
            ):
                raise ValueError("boom")  # noqa: EM101
            assert fill_uninitialized_memory() is outside
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = before  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


def test_a_cublas_preset_outside_the_reproducible_ones_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _keeping_flags():
        before = _flags()
        for bad in (":0:0", ":4096:2", "4096:8"):
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", bad)
            with (
                pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"),
                deterministic_algorithms(),
            ):
                pass
            assert _flags() == before  # refused before anything changed
        for good in CUBLAS_WORKSPACES:
            monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", good)
            with deterministic_algorithms() as settings:
                assert settings["cublas_workspace_config"] == good
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
        with deterministic_algorithms() as settings:
            assert settings["cublas_workspace_config"] is None


@needs_fused
def test_two_cuda_runs_of_the_driver_are_bitwise_equal_and_record_the_settings(
    source: Path, tmp_path: Path
) -> None:
    before = _flags()
    results: list[tuple[list[str], dict[str, Tensor], dict[str, object]]] = []
    for name in ("a", "b"):
        out = tmp_path / name
        result = run(
            source,
            _sha(source),
            0,
            "none",
            out,
            device="cuda",
            updates=12,
            eval_batches=1,
            save_at=(0, 6, 12),
            required_accuracy=-1.0,
            train="joint",
            dense_weight=0.1,
        )
        weights = cast(
            "dict[str, Tensor]", torch.load(out / "weights" / "after-00012.pt", weights_only=True)
        )
        results.append(((out / "train.jsonl").read_text("utf-8").splitlines(), weights, result))
    (lines_a, weights_a, result_a), (lines_b, weights_b, _) = results
    assert len(lines_a) == 12
    for update, (a, b) in enumerate(zip(lines_a, lines_b, strict=True)):
        record_a = cast("dict[str, object]", json.loads(a))
        record_b = cast("dict[str, object]", json.loads(b))
        assert record_a == record_b, update
    assert weights_a.keys() == weights_b.keys()
    for name, tensor in weights_a.items():
        assert torch.equal(tensor, weights_b[name]), name
    determinism = cast("dict[str, object]", result_a["determinism"])
    assert (determinism["deterministic_algorithms"], determinism["warn_only"]) == (True, False)
    assert cast("int", determinism["cublas_workspace_size"]) > 0
    assert _flags() == before  # the driver's training scope left nothing behind


@needs_fused
def test_two_fresh_processes_repeat_every_parameter_and_adam_field(
    source: Path, tmp_path: Path
) -> None:
    # independent of this session: the variable removed, a new process each, 120 updates across
    # the capture (update 3) and the end of the warmup (100), the evaluation copy at update 60
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env["PYTHONPATH"] = str(ROOT)
    outputs: list[dict[str, object]] = []
    for name in ("a", "b"):
        out = tmp_path / f"{name}.json"
        done = subprocess.run(  # noqa: S603  (our own interpreter and a fixed script)
            [sys.executable, str(PROBE), str(source), _sha(source), str(out)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr[-2000:]
        outputs.append(cast("dict[str, object]", json.loads(out.read_text("utf-8"))))
    a, b = outputs
    states_a = cast("list[str]", a["states"])
    assert len(states_a) == 120
    assert (a["captures"], a["replays"]) == (1, 117)
    for update, (x, y) in enumerate(zip(states_a, cast("list[str]", b["states"]), strict=True)):
        assert x == y, update
    assert a["records"] == b["records"]
    assert a["evaluation"] == b["evaluation"] != ""
    settings = cast("dict[str, object]", a["settings"])
    assert (settings["deterministic_algorithms"], settings["warn_only"]) == (True, False)
    assert settings["cublas_workspace_config"] is None
