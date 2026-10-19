"""The crossed-seed diagnostic's endpoint (vesper/halting_escape.py) and the run's static checks.

Maintainer B's endpoint (channel 5dba6af): the first of TWO consecutive checks whose hop2 accuracy,
maximized over the static depths, exceeds 50 %; never crossing is censored, a lone final
crossing is unconfirmed. The checks themselves must be the static profile of the very weights
saved after their update, on the same first validation episodes, and must not disturb training.
"""

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from vesper.config import load
from vesper.halting_escape import (
    ESCAPE_CLASS,
    Endpoint,
    best_depth,
    escape_endpoint,
    main,
    read_run,
    read_runs,
    table,
)
from vesper.halting_experiment import (
    STATIC_CHECKS,
    _static_check,  # pyright: ignore[reportPrivateUsage]
    _static_profile,  # pyright: ignore[reportPrivateUsage]
    build,
    held_out,
    run,
)
from vesper.synthetic import CLASSES

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"
E = 1000
HOP2 = CLASSES.index(ESCAPE_CLASS)
T = 8


def _checks(*accuracies: float) -> list[tuple[int, float]]:
    return [(E * (i + 1), a) for i, a in enumerate(accuracies)]


def test_the_endpoint_is_the_first_of_two_consecutive_crossings() -> None:
    # an isolated crossing (2000) is not the endpoint; 4000 and 5000 are the first pair
    end = escape_endpoint(_checks(0.1, 0.6, 0.4, 0.55, 0.7, 0.3), E)
    assert end == Endpoint("escaped", 4000, 6000, (2000, 4000, 5000))


def test_a_lone_final_crossing_is_unconfirmed_and_never_crossing_is_censored() -> None:
    assert escape_endpoint(_checks(0.1, 0.6, 0.2, 0.7), E) == Endpoint(
        "unconfirmed", None, 4000, (2000, 4000)
    )
    assert escape_endpoint(_checks(0.1, 0.2), E) == Endpoint("censored", None, 2000, ())
    # a pair ending at the last check is an escape, not an unconfirmed crossing
    assert escape_endpoint(_checks(0.1, 0.6, 0.7), E) == Endpoint(
        "escaped", 2000, 3000, (2000, 3000)
    )


def test_exactly_the_threshold_is_not_a_crossing() -> None:
    assert escape_endpoint(_checks(0.5, 0.5, 0.5), E) == Endpoint("censored", None, 3000, ())
    assert escape_endpoint(_checks(0.5, 0.5000001, 0.5000001), E).escape == 2000


@pytest.mark.parametrize(
    "checks",
    [
        [],
        [(1000, 0.9), (3000, 0.9)],  # a missing check would join two non-consecutive ones
        [(2000, 0.9), (3000, 0.9)],  # the first check is after the first interval
        [(1000, 0.9), (1000, 0.9)],  # a repeated check
        [(2000, 0.9), (1000, 0.9)],  # out of order
    ],
)
def test_checks_must_be_exactly_every_interval(checks: list[tuple[int, float]]) -> None:
    with pytest.raises(ValueError, match="static checks"):
        _ = escape_endpoint(checks, E)


def _counts(hop2: list[tuple[float, float]]) -> list[list[list[float]]]:
    """[T, C, 2] counts with hop2 as given per depth and every other class at 1 of 2.

    Floats, as the driver writes them (``plain``): an int literal here would violate the
    ``Counts`` hint wherever beartype happens to sample it.
    """
    return [
        [[float(x) for x in hop2[d]] if c == HOP2 else [1.0, 2.0] for c in range(len(CLASSES))]
        for d in range(len(hop2))
    ]


def test_best_depth_is_the_maximum_over_depths_and_needs_labels_at_each() -> None:
    assert best_depth(_counts([(1, 10), (6, 10), (6, 10), (2, 10)])) == (0.6, 2)
    with pytest.raises(ValueError, match="no labels at depth 2"):
        _ = best_depth(_counts([(1, 10), (0, 0)]))


def _fake_run(
    root: Path,
    name: str,
    seed: int,
    data_seed: int,
    hop2: list[float],
    *,
    every: int = E,
    recipe: dict[str, object] | None = None,
) -> Path:
    """A finished run's two files as the driver writes them: 3000 updates, checks as given."""
    out = root / name
    out.mkdir()
    lines = [
        {"after": every * (i + 1), "static": _counts([(100 * a, 100.0)] * T)}
        for i, a in enumerate(hop2)
    ]
    _ = (out / STATIC_CHECKS).write_text(
        "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
    )
    result: dict[str, object] = {
        "condition": "fixed",
        "seed": seed,
        "data_seed": data_seed,
        "config_sha256": "c" * 64,
        "steps": 3000,
        "mix_schedule": "v1",
        "device": "cuda",
        "scan": "fused",
        "cuda_graph": True,
        "static_every": every,
        "static_batches": 8,
        "static_size": 32,
        "static_split": "validation",
        "eval_split": "validation",
        "eval_sequences": 1024,
        "backend": {"device": "cuda"},
        "counts": {"static": _counts([(1.0, 4.0)] * T)},
    }
    result |= recipe or {}
    _ = (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return out


def test_runs_are_read_by_their_pair_with_one_endpoint_each(tmp_path: Path) -> None:
    dirs = [
        _fake_run(tmp_path, "b", 4, 1, [0.1, 0.6, 0.7]),
        _fake_run(tmp_path, "a", 1, 4, [0.1, 0.2, 0.3]),
        _fake_run(tmp_path, "c", 1, 1, [0.1, 0.2, 0.9]),
    ]
    runs = read_runs(dirs)
    assert [(r["seed"], r["data_seed"]) for r in runs] == [(1, 1), (1, 4), (4, 1)]
    ends = [cast("dict[str, object]", r["endpoint"]) for r in runs]
    assert [(e["status"], e["escape"]) for e in ends] == [
        ("unconfirmed", None),
        ("censored", None),
        ("escaped", 2000),
    ]
    text = table(runs)
    assert "| 4 | 1 | escaped | 2000 | 2000, 3000 | 70.0 at depth 1 |" in text
    assert "| 1 | 4 | censored | - | - | 30.0 at depth 1 |" in text
    assert f"| {ESCAPE_CLASS} | 25.0 at 1 | 25.0 at 1 | 25.0 at 1 |" in text
    out = tmp_path / "escape.json"
    assert main([str(d) for d in dirs] + ["--out", str(out)]) == 0
    written = cast("dict[str, object]", json.loads(out.read_text("utf-8")))
    assert written["threshold"] == 0.5
    assert len(cast("list[object]", written["runs"])) == 3


def test_runs_of_another_recipe_a_repeated_pair_or_a_cut_file_are_refused(tmp_path: Path) -> None:
    a = _fake_run(tmp_path, "a", 1, 1, [0.1, 0.2, 0.3])
    changes: tuple[tuple[str, object], ...] = (
        ("mix_schedule", "v2-slow"),
        ("scan", "reference"),
        ("static_size", 2),
    )
    for key, value in changes:
        other = _fake_run(tmp_path, f"x-{key}", 4, 4, [0.1, 0.2, 0.3], recipe={key: value})
        with pytest.raises(ValueError, match=f"disagree on {key}"):
            _ = read_runs([a, other])
    twin = _fake_run(tmp_path, "twin", 1, 1, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="pair is repeated"):
        _ = read_runs([a, twin])
    cut = _fake_run(tmp_path, "cut", 4, 1, [0.1, 0.2])  # 3000 steps, checks end at 2000
    with pytest.raises(ValueError, match="last static check is not at update 3000"):
        _ = read_run(cut)
    off = _fake_run(tmp_path, "off", 4, 4, [], recipe={"static_every": 0})
    with pytest.raises(ValueError, match="without static checks"):
        _ = read_run(off)
    # a warm start escaped in its source run: it has no escape of its own; a fresh run's null
    # reads as before
    warm = _fake_run(tmp_path, "warm", 4, 4, [0.6, 0.7, 0.8], recipe={"init_weights": {}})
    with pytest.raises(ValueError, match="started from another run's weights"):
        _ = read_run(warm)
    fresh = _fake_run(tmp_path, "fresh", 4, 4, [0.6, 0.7, 0.8], recipe={"init_weights": None})
    assert cast("dict[str, object]", read_run(fresh)["endpoint"])["status"] == "escaped"
    with pytest.raises(ValueError, match="no runs"):
        _ = read_runs([])


def test_static_checks_are_the_profile_of_the_weights_saved_after_their_update(
    tmp_path: Path,
) -> None:
    out = tmp_path / "checked"
    result = run(
        CONFIG,
        "fixed",
        0,
        out,
        steps=3,
        eval_batches=1,
        eval_size=2,
        weights_every=1,
        static_every=1,
        static_batches=2,
    )
    lines = [
        cast("dict[str, object]", json.loads(line))
        for line in (out / STATIC_CHECKS).read_text("utf-8").splitlines()
    ]
    assert [line["after"] for line in lines] == [1, 2, 3]
    static_set = held_out(2, 2, "validation")
    cfg = load(CONFIG)
    for line in lines:
        model = build(cfg, "fixed", 0)
        path = out / "weights" / f"after-{cast('int', line['after']):05d}.pt"
        state = cast("dict[str, torch.Tensor]", torch.load(path, weights_only=True))
        _ = cast("object", model.load_state_dict(state))  # its result is untyped in torch
        _ = model.eval()
        with torch.no_grad():
            want = torch.stack([_static_profile(model, ep) for ep in static_set]).sum(0)
        got = torch.tensor(cast("list[list[list[float]]]", line["static"]), dtype=want.dtype)
        assert torch.equal(got, want), line["after"]
    assert (result["static_every"], result["static_batches"], result["static_size"]) == (1, 2, 2)
    assert result["static_split"] == "validation"
    # the checks read the FIRST episodes of the split: the same ones as a longer held-out set's
    longer = held_out(3, 2, "validation")
    for first_ep, longer_ep in zip(static_set, longer, strict=False):
        assert torch.equal(first_ep.chunk.tokens, longer_ep.chunk.tokens)
    # and they do not disturb training: the same run without checks ends on the same weights
    plain = tmp_path / "plain"
    _ = run(CONFIG, "fixed", 0, plain, steps=3, eval_batches=1, eval_size=2, weights_every=1)
    assert not (plain / STATIC_CHECKS).exists()
    a = cast("dict[str, torch.Tensor]", torch.load(out / "model.pt", weights_only=True))
    b = cast("dict[str, torch.Tensor]", torch.load(plain / "model.pt", weights_only=True))
    assert all(torch.equal(a[k], b[k]) for k in a)
    ran = read_run(out)
    assert cast("dict[str, object]", ran["endpoint"])["last"] == 3


def test_a_static_check_leaves_a_training_model_in_training_mode() -> None:
    # on the CPU the check reads the training model itself (no copy), so it must restore the mode
    model = build(load(CONFIG), "fixed", 0)
    _ = model.train()
    _ = _static_check(model, held_out(1, 2, "validation"))
    assert model.training
    _ = model.eval()
    _ = _static_check(model, held_out(1, 2, "validation"))
    assert not model.training


def test_bad_static_settings_are_refused_before_anything_is_written(tmp_path: Path) -> None:
    for every, batches in ((-1, 8), (1000, 0)):
        out = tmp_path / f"bad-{every}-{batches}"
        with pytest.raises(ValueError, match="static checks need"):
            _ = run(CONFIG, "fixed", 0, out, steps=1, static_every=every, static_batches=batches)
        assert not out.exists()
