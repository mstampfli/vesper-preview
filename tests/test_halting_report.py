"""The stage-1 report (vesper/halting_report.py) over records built by the driver itself."""

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import cast

import pytest

from vesper.config import load
from vesper.halting_experiment import SNAPSHOT, build, evaluate
from vesper.halting_report import Run, load_results, pooled_summary, report
from vesper.synthetic import CLASSES, episodes, generator_for

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"
T = 8


def _record(condition: str, seed: int) -> dict[str, object]:
    model = build(load(CONFIG), condition, seed)
    evaluation = evaluate(model, condition, [episodes(generator_for("test", seed), 2, "t")])
    return {
        "condition": condition,
        "seed": seed,
        "steps": 0,
        "eval_sequences": 2,
        "config": str(CONFIG),
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "environment": {"vesper_commit": "0" * 40, "vesper_dirty": "false"},
        **evaluation,
    }


def _report(records: list[dict[str, object]]) -> str:
    return report([Run(record=r, config=load(CONFIG)) for r in records])


@pytest.fixture(scope="module")
def records() -> list[dict[str, object]]:
    return [_record("fixed", 0), _record("fixed", 1), _record("local", 0)]


def test_pooling_sums_the_counts_of_the_seeds(records: list[dict[str, object]]) -> None:
    a, b = records[0], records[1]
    pooled = pooled_summary([a, b], T, 2)
    live = cast("dict[str, float]", pooled["live/0.5"])
    counts = [cast("dict[str, list[list[float]]]", r["counts"])["live/0.5"] for r in (a, b)]
    correct = sum(row[0] for c in counts for row in c)
    total = sum(row[1] for c in counts for row in c)
    assert live["overall"] == pytest.approx(correct / total)
    # pooling one record twice changes no ratio
    twice = pooled_summary([a, a], T, 2)
    once = pooled_summary([a], T, 2)
    assert twice["live/0.5"] == once["live/0.5"]


def test_the_report_has_every_section_and_a_row_per_condition(
    records: list[dict[str, object]],
) -> None:
    text = _report(records)
    for heading in range(1, 9):
        assert f"\n## {heading}. " in text
    assert "| B fixed |" in text
    assert "| C local |" in text
    assert "B fixed 2, C local 1" in text
    assert "-0.000" not in text


def test_the_acceptance_verdict_reads_the_live_against_the_dense_counts(
    records: list[dict[str, object]],
) -> None:
    def verdict(live_correct: float) -> str:
        record = copy.deepcopy(records[0])
        counts = cast("dict[str, list[list[float]]]", record["counts"])
        counts["dense_exit/0.5"] = [[500.0, 1000.0] for _ in CLASSES]
        counts["trace/0.5"] = [[500.0, 1000.0] for _ in CLASSES]
        counts["live/0.5"] = [[500.0, 1000.0] for _ in CLASSES]
        counts["live/0.5"][CLASSES.index("read2")] = [live_correct, 1000.0]
        return next(
            line for line in _report([record]).splitlines() if line.startswith("| B fixed | r")
        )

    assert verdict(490.0).endswith("| read2 | 1.0 | 0.0 | pass |")  # exactly 1 point: allowed
    assert verdict(489.0).endswith("| read2 | 1.1 | 0.0 | FAIL |")


def test_records_that_disagree_are_refused(records: list[dict[str, object]]) -> None:
    other = copy.deepcopy(records[0])
    other["steps"] = 5
    with pytest.raises(ValueError, match="disagree on steps"):
        _ = _report([records[0], other])
    with pytest.raises(ValueError, match="no result"):
        _ = report([])
    # a repeated condition and seed would count as another seed; different betas are not one
    # condition (maintainer B's review of c7d25c5)
    with pytest.raises(ValueError, match="duplicate record for condition fixed seed 0"):
        _ = _report([records[0], copy.deepcopy(records[0])])
    other_beta = copy.deepcopy(records[1])
    other_beta["beta"] = 0.5
    with pytest.raises(ValueError, match="different beta"):
        _ = _report([records[0], other_beta])


def _saved_run(root: Path, record: dict[str, object], original: Path) -> Path:
    """A run directory as the driver writes it: the snapshot beside the record, named relative
    to it, and the original config path recorded."""
    _ = original.write_bytes(CONFIG.read_bytes())
    run = root / "fixed-s0"
    run.mkdir(parents=True)
    _ = (run / SNAPSHOT).write_bytes(CONFIG.read_bytes())
    saved = copy.deepcopy(record) | {"config": str(original), "config_snapshot": SNAPSHOT}
    _ = (run / "result.json").write_text(json.dumps(saved), encoding="utf-8")
    return run


def test_a_saved_run_reads_its_snapshot_and_never_the_original(
    records: list[dict[str, object]], tmp_path: Path
) -> None:
    # Maintainer B's re-review of 9044238: the report read the original path, so a valid saved run
    # failed once the original changed or vanished, and an altered snapshot passed
    original = tmp_path / "source.yaml"
    run = _saved_run(tmp_path / "runs", records[0], original)
    _ = original.write_bytes(CONFIG.read_bytes().replace(b"window: 8", b"window: 7"))
    [loaded] = load_results(tmp_path / "runs")
    assert loaded.config.model.window == 8  # the snapshot's, not the edited original's
    original.unlink()
    # copied elsewhere with the original gone, the run directory still reads on its own
    copied = tmp_path / "export"
    copied.mkdir()
    _ = run.rename(copied / run.name)
    [moved] = load_results(copied)
    assert moved.config == loaded.config
    assert "## 9. " in report([moved])
    # an altered snapshot is refused, whatever the original says
    _ = original.write_bytes(CONFIG.read_bytes())
    _ = (copied / run.name / SNAPSHOT).write_bytes(original.read_bytes() + b"\n# edited\n")
    with pytest.raises(ValueError, match="changed since the run"):
        _ = load_results(copied)


def test_a_snapshot_must_stay_inside_its_run_directory(
    records: list[dict[str, object]], tmp_path: Path
) -> None:
    for name in (str(CONFIG), "../config.yaml"):
        root = tmp_path / str(len(name))
        run = _saved_run(root, records[0], tmp_path / "source.yaml")
        text = (run / "result.json").read_text(encoding="utf-8")
        saved = cast("dict[str, object]", json.loads(text)) | {"config_snapshot": name}
        _ = (run / "result.json").write_text(json.dumps(saved), encoding="utf-8")
        with pytest.raises(ValueError, match="must be relative to its run directory"):
            _ = load_results(root)


def test_a_record_without_snapshot_is_checked_against_its_original(
    records: list[dict[str, object]], tmp_path: Path
) -> None:
    run = tmp_path / "fixed-s0"
    run.mkdir()
    original = tmp_path / "source.yaml"
    _ = original.write_bytes(CONFIG.read_bytes())
    legacy = copy.deepcopy(records[0]) | {"config": str(original)}
    assert "config_snapshot" not in legacy
    _ = (run / "result.json").write_text(json.dumps(legacy), encoding="utf-8")
    [loaded] = load_results(tmp_path)
    assert loaded.config == load(CONFIG)
    _ = original.write_bytes(CONFIG.read_bytes() + b"\n# edited\n")
    with pytest.raises(ValueError, match="changed since the run"):
        _ = load_results(tmp_path)
    original.unlink()
    with pytest.raises(ValueError, match="without a snapshot needs its config"):
        _ = load_results(tmp_path)


def test_the_depth_need_is_the_first_pass_near_the_best(records: list[dict[str, object]]) -> None:
    learned = copy.deepcopy(records[0])
    counts = cast("dict[str, list[list[list[float]]]]", learned["counts"])
    # hop2: 10 % at passes 1-2, exactly 90 % at 3-4, 91 % from 5: need 3, one point included
    index = CLASSES.index("hop2")
    for t in range(T):
        right = 910.0 if t >= 4 else 900.0 if t >= 2 else 100.0
        counts["dense_pass"][t][index] = [right, 1000.0]
        counts["dense_pass"][t][CLASSES.index("hop1")] = [400.0, 1000.0]  # below the floor
    text = _report([learned])
    row = next(line for line in text.splitlines() if line.startswith("| hop2 |"))
    assert row.endswith("| 3 |")
    # a class below the learned floor has no need
    hop1 = next(line for line in text.splitlines() if line.startswith("| hop1 |"))
    assert hop1.endswith("| not learned |")


def _with_static(record: dict[str, object], curves: dict[str, list[float]]) -> dict[str, object]:
    """``record`` with its fixed-depth counts set: 1000 labels per class and depth, every class
    at 0.6 except those in ``curves`` (accuracy per depth; None as total 0)."""
    # the probe protocol's identity (seeds 0 and 1 come from the fixture's records)
    out = copy.deepcopy(record) | {
        "steps": 12000,
        "eval_split": "validation",
        "eval_batches": 32,
        "eval_size": 32,
        "eval_sequences": 1024,
    }
    counts = cast("dict[str, object]", out["counts"])
    counts["static"] = [
        [
            [1000.0 * curves.get(name, [0.6] * T)[j], 1000.0]
            if curves.get(name, [0.6] * T)[j] >= 0
            else [0.0, 0.0]
            for name in CLASSES
        ]
        for j in range(T)
    ]
    return out


def _gate_lines(records: list[dict[str, object]]) -> tuple[str, str]:
    text = _report(records)
    gate1 = next(line for line in text.splitlines() if line.startswith("Gate 1"))
    gate2 = next(line for line in text.splitlines() if line.startswith("Gate 2"))
    return gate1, gate2


def test_the_probe_gates_read_every_seed(records: list[dict[str, object]]) -> None:
    deeper = [0.4, 0.4, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9]  # gains 50 points, best at depth 4
    flat = [0.6] * T
    s0, s1 = records[0], records[1]
    gate1, gate2 = _gate_lines(
        [_with_static(s0, {"hop3": deeper}), _with_static(s1, {"hop3": deeper})]
    )
    assert ": pass (not learned: none)" in gate1
    assert gate2.endswith("pass (classes: hop3).")
    # the gain must hold in EVERY seed
    _, gate2 = _gate_lines([_with_static(s0, {"hop3": deeper}), _with_static(s1, {"hop3": flat})])
    assert "FAIL (classes: none)" in gate2
    # exactly 50 % is not learned; exactly 5 points is a gain (0.55 - 0.50 is not 0.05 in floats)
    edge = [0.50, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]
    half = [0.5] * T
    gate1, gate2 = _gate_lines(
        [_with_static(s0, {"read4": half, "hop2": edge}), _with_static(s1, {"hop2": edge})]
    )
    assert "FAIL (not learned: read4)" in gate1
    assert gate2.endswith("pass (classes: hop2).")
    # a class without labels fails, it does not pass by default
    missing = [-1.0] * T
    gate1, _ = _gate_lines([_with_static(s0, {"read1": missing}), _with_static(s1, {})])
    assert "FAIL (not learned: read1)" in gate1
    assert "no labels" in _report([_with_static(s0, {"read1": missing})])


def test_gates_outside_the_probe_protocol_are_not_assessed(
    records: list[dict[str, object]],
) -> None:
    # Maintainer B's review of 1678173: each of these passed both gates on qualifying curves
    deeper = [0.4, 0.4, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9]
    good = [_with_static(r, {"hop3": deeper}) for r in records[:2]]
    cases: dict[str, list[dict[str, object]]] = {
        "probe-v1: B seeds [0], it has [0, 1]": good[:1],
        "probe-v1: B seeds [8, 9], it has [0, 1]": [
            r | {"seed": 8 + i} for i, r in enumerate(good)
        ],
        "probe-v1: [1000] updates, it has 12000": [r | {"steps": 1000} for r in good],
        "probe-v1: evaluation split ['final'], it uses validation": [
            r | {"eval_split": "final"} for r in good
        ],
        "probe-v1: evaluation of 2 x 2 sequences, it uses 32 x 32": [
            r | {"eval_batches": 2, "eval_size": 2, "eval_sequences": 4} for r in good
        ],
        "probe-v1: mix ['v2-slow'], it uses ['v1']": [
            r | {"mix_schedule": "v2-slow"} for r in good
        ],
    }
    for reason, case in cases.items():
        text = _report(case)
        for line in text.splitlines():
            if line.startswith(("Gate 1", "Gate 2")):
                assert "not assessed (outside every gate protocol" in line, reason
                assert reason in line, line
        # the tables stay: the numbers are reported, only the verdict is withheld
        assert "| hop3 | " in text.split("\n## 10. ")[1]
    gate1, gate2 = _gate_lines(good)
    assert "under probe-v1: pass" in gate1
    assert "under probe-v1: pass" in gate2
    # a record from before the population was recorded is outside the protocol too
    legacy = [{k: v for k, v in r.items() if k not in {"eval_batches", "eval_size"}} for r in good]
    assert "evaluation of None x None sequences" in _gate_lines(legacy)[0]


C1_CONFIG = CONFIG.with_name("halting-stage1-c1.yaml")


def _c1_gate_lines(records: list[dict[str, object]], config: Path) -> tuple[str, str]:
    text = report([Run(record=r, config=load(config)) for r in records])
    lines = text.splitlines()
    return next(x for x in lines if x.startswith("Gate 1")), next(
        x for x in lines if x.startswith("Gate 2")
    )


def test_the_one_column_recipe_has_its_own_protocol(records: list[dict[str, object]]) -> None:
    # Maintainer B, 2026-10-16: one column, seeds 0 and 1, 36000 updates, mix v2-slow, the same
    # gates
    deeper = [0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    c1 = [
        _with_static(r, {"hop2": deeper}) | {"steps": 36000, "mix_schedule": "v2-slow"}
        for r in records[:2]
    ]
    gate1, gate2 = _c1_gate_lines(c1, C1_CONFIG)
    assert "under c1-slow-v2: pass (not learned: none)" in gate1
    assert gate2.endswith("under c1-slow-v2: pass (classes: hop2).")
    # each part of the recipe is required: the mix, the update count, the column count
    for reason, case, config in (
        (
            "c1-slow-v2: mix ['v1'], it uses ['v2-slow']",
            [r | {"mix_schedule": "v1"} for r in c1],
            C1_CONFIG,
        ),
        (
            "c1-slow-v2: [12000] updates, it has 36000",
            [r | {"steps": 12000} for r in c1],
            C1_CONFIG,
        ),
        ("c1-slow-v2: 2 columns, it has 1", c1, CONFIG),
    ):
        gate1, gate2 = _c1_gate_lines(case, config)
        for line in (gate1, gate2):
            assert "not assessed (outside every gate protocol" in line, reason
            assert reason in line, line
    # an old record (no mix field) never counts as the new recipe
    legacy = [{k: v for k, v in r.items() if k != "mix_schedule"} for r in c1]
    assert "c1-slow-v2: mix ['None'], it uses ['v2-slow']" in _c1_gate_lines(legacy, C1_CONFIG)[0]
    # the backend is part of the recipe: a GPU run with the fused scan is other arithmetic, and
    # a record that names the CPU reference explicitly is the same recipe as one written before
    # the field existed
    gpu = [r | {"device": "cuda", "scan": "fused", "cuda_graph": True} for r in c1]
    for line in _c1_gate_lines(gpu, C1_CONFIG):
        assert "c1-slow-v2: backend cuda/fused, it trains on cpu/reference" in line
    named = [r | {"device": "cpu", "scan": "reference"} for r in c1]
    assert "under c1-slow-v2: pass" in _c1_gate_lines(named, C1_CONFIG)[0]


def test_records_on_different_backends_are_refused(records: list[dict[str, object]]) -> None:
    a = copy.deepcopy(records[0])  # no backend fields: the CPU reference
    b = copy.deepcopy(records[1]) | {"device": "cuda", "scan": "fused"}
    with pytest.raises(ValueError, match="disagree on the training backend"):
        _ = _report([a, b])
    c = copy.deepcopy(records[1]) | {"device": "cpu", "scan": "reference"}
    _ = _report([a, c])  # explicit and implicit CPU reference are one backend


def test_records_on_different_mix_recipes_are_refused(records: list[dict[str, object]]) -> None:
    a = copy.deepcopy(records[0]) | {"mix_schedule": "v1"}
    b = copy.deepcopy(records[1]) | {"mix_schedule": "v2-slow"}
    with pytest.raises(ValueError, match="disagree on mix_schedule"):
        _ = _report([a, b])


def test_records_on_different_evaluation_splits_are_refused(
    records: list[dict[str, object]],
) -> None:
    validation = copy.deepcopy(records[0]) | {"eval_split": "validation"}
    final = copy.deepcopy(records[1]) | {"eval_split": "final"}
    with pytest.raises(ValueError, match="disagree on eval_split"):
        _ = _report([validation, final])


def test_maxima_are_pooled_by_max_not_sum(records: list[dict[str, object]]) -> None:
    a, b = copy.deepcopy(records[0]), copy.deepcopy(records[1])
    cast("dict[str, object]", a["counts"])["deployed_max_output_difference/decode"] = 0.125
    cast("dict[str, object]", b["counts"])["deployed_max_output_difference/decode"] = 0.25
    cast("dict[str, object]", a["counts"])["ablated_max_output_difference"] = 0.5
    cast("dict[str, object]", b["counts"])["ablated_max_output_difference"] = 0.25
    pooled = pooled_summary([a, b], T, 2)
    assert pooled["deployed_max_output_difference/decode"] == 0.25  # a sum gave 0.375
    assert pooled["ablated_max_output_difference"] == 0.5


def test_information_is_per_model_and_the_pooled_mixture_is_not_reported_as_it(
    records: list[dict[str, object]],
) -> None:
    # Maintainer B's fixture: two token-independent deterministic models, one exiting at pass 1 and
    # the other at pass 2; each carries zero information, their pooled mixture log 2
    def deterministic(record: dict[str, object], exit_pass: int) -> dict[str, object]:
        out = copy.deepcopy(record)
        counts = cast("dict[str, object]", out["counts"])
        p = [0.0] * T
        p[exit_pass] = 100.0
        counts["soft/all/p"] = p
        counts["soft/all/plogp"] = [0.0]  # p = 1 at one exit: p log p = 0
        counts["soft/all/count"] = [100.0]
        return out

    first, second = deterministic(records[0], 0), deterministic(records[1], 1)
    mixture = pooled_summary([first, second], T, 2)["soft_information/all"]
    assert mixture == pytest.approx(math.log(2))
    text = _report([first, second])
    section = text.split("\n## 6. ")[1].split("\n## 7. ")[0]
    row = next(line for line in section.splitlines() if line.startswith("| B fixed | "))
    soft_all = row.split("|")[5].strip()  # condition, H, I(role), I(class), soft all, soft loss
    assert soft_all == "0.000 [0.000, 0.000]"
    assert "0.693" not in text


def test_an_unlearned_register_reader_is_inconclusive_and_the_ablation_is_paired(
    records: list[dict[str, object]],
) -> None:
    text = _report(records)
    register = next(line for line in text.splitlines() if line.startswith("| B fixed | inconcl"))
    assert "inconclusive (" in register
    section = text.split("\n## 9. Recurrence ablation")[1]
    overall = next(line for line in section.splitlines() if line.startswith("| overall | "))
    # B's paired reference is its deployed run: the first cell repeats deployed/prefill
    summary = pooled_summary([records[0], records[1]], T, 2)
    deployed = cast("dict[str, float]", summary["deployed/prefill"])["overall"]
    assert overall.split("|")[2].strip().startswith(f"{100 * deployed:.1f} / ")
    assert "Largest output change at the replayed exits: B fixed " in section
    # records written before the paired reference existed say so instead of failing
    old = copy.deepcopy(records[0])
    for key in [k for k in cast("dict[str, object]", old["counts"]) if k.startswith("ablated")]:
        del cast("dict[str, object]", old["counts"])[key]
    assert "Not measured with a paired reference in these records." in _report([old])
