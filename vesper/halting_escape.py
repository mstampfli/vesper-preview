"""When a run's two-hop lookups leave the plateau: the crossed-seed diagnostic's endpoint.

The diagnostic (maintainer B, channel 5dba6af; docs/reviews/2026-10-16-crossed-seed-diagnostic.md)
trains condition B with the initialization seed and the data-order seed crossed, every run with
``halting_experiment.run``'s static checks on. This module reads the finished runs, keyed by
their (initialization seed, data-order seed) pair, and applies one endpoint to each:

    python -m vesper.halting_escape RUN_DIR [RUN_DIR ...] [--out FILE]

Endpoint: the FIRST of two consecutive static checks (``static_every`` updates apart) whose
``ESCAPE_CLASS`` accuracy, maximized over the static depths 1..T, EXCEEDS ``ESCAPE_THRESHOLD``
on the run's static set (the same first validation episodes in every check and run). No such
pair: right-censored at the last check, and a crossing at the last check alone is "unconfirmed",
never an escape. Every check's counts are kept, and each run's terminal static profile from its
full evaluation (every class, every depth). The runs must share one recipe (config, steps, mix,
static set, backend) and differ only in the pair; anything else is refused before a row is read.
Not a population escape distribution, and not a comparison of mixes: four selected cells.
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

from vesper.halting_experiment import STATIC_CHECKS
from vesper.provenance import write_json
from vesper.synthetic import CLASSES

ESCAPE_CLASS = "hop2"
ESCAPE_THRESHOLD = 0.5  # exceeded strictly: exactly half is not an escape
# what every run of one diagnostic must share besides the evaluation it is read with
RECIPE = (
    "condition",
    "config_sha256",
    "steps",
    "mix_schedule",
    "device",
    "scan",
    "cuda_graph",
    "static_every",
    "static_batches",
    "static_size",
    "static_split",
    "eval_split",
    "eval_sequences",
)

Counts = list[list[list[float]]]  # [T, C, 2]: correct and total per depth and class


@dataclass(frozen=True, slots=True)
class Endpoint:
    """``status`` "escaped" (``escape`` is the first of the pair), "censored" or "unconfirmed"."""

    status: str
    escape: int | None
    last: int
    crossings: tuple[int, ...]


def escape_endpoint(
    checks: Sequence[tuple[int, float]], every: int, threshold: float = ESCAPE_THRESHOLD
) -> Endpoint:
    """The endpoint of one run from its checks ``(updates, accuracy)``, in order.

    Raises:
        ValueError: on no checks, or checks not exactly at every, 2 every, ... (a missing or
            extra check would move the endpoint).
    """
    if not checks:
        msg = "no static checks"
        raise ValueError(msg)
    afters = [after for after, _ in checks]
    if afters != [every * (i + 1) for i in range(len(checks))]:
        msg = f"static checks must be every {every} updates from {every} on, got {afters}"
        raise ValueError(msg)
    crossings = tuple(after for after, accuracy in checks if accuracy > threshold)
    for (after, first), (_, second) in pairwise(checks):
        if first > threshold and second > threshold:
            return Endpoint("escaped", after, afters[-1], crossings)
    status = "unconfirmed" if checks[-1][1] > threshold else "censored"
    return Endpoint(status, None, afters[-1], crossings)


def best_depth(counts: Counts, cls: str = ESCAPE_CLASS) -> tuple[float, int]:
    """``cls``'s accuracy maximized over the depths, and the first depth (1-based) reaching it.

    Raises:
        ValueError: if ``cls`` has no labels at some depth (an accuracy there is undefined).
    """
    index = CLASSES.index(cls)
    accuracies: list[float] = []
    for depth, row in enumerate(counts, start=1):
        correct, total = row[index]
        if total <= 0:
            msg = f"class {cls} has no labels at depth {depth}"
            raise ValueError(msg)
        accuracies.append(correct / total)
    best = max(accuracies)
    return best, accuracies.index(best) + 1


def _profile(counts: Counts) -> dict[str, list[float | None]]:
    """Accuracy per class and depth (None where a class has no labels)."""
    return {
        cls: [(row[i][0] / row[i][1]) if row[i][1] > 0 else None for row in counts]
        for i, cls in enumerate(CLASSES)
    }


def read_run(run_dir: Path) -> dict[str, object]:
    """One finished run: its pair, its checks, its endpoint and its terminal static profile.

    Raises:
        ValueError: on a run without static checks, whose checks do not end at its last
            update (an unfinished or cut file), or that started from another run's weights
            (``init_weights``: its escape began in that run, not at its own first update).
    """
    result = cast("dict[str, object]", json.loads((run_dir / "result.json").read_text("utf-8")))
    if result.get("init_weights") is not None:
        msg = f"{run_dir} started from another run's weights: no escape of its own to measure"
        raise ValueError(msg)
    every = cast("int", result.get("static_every") or 0)
    if every <= 0:
        msg = f"{run_dir} ran without static checks"
        raise ValueError(msg)
    lines = (run_dir / STATIC_CHECKS).read_text("utf-8").splitlines()
    rows = [cast("dict[str, object]", json.loads(line)) for line in lines if line.strip()]
    checks = [(cast("int", r["after"]), best_depth(cast("Counts", r["static"]))) for r in rows]
    steps = cast("int", result["steps"])
    if not checks or checks[-1][0] != steps - steps % every:
        msg = f"{run_dir}: the last static check is not at update {steps - steps % every}"
        raise ValueError(msg)
    endpoint = escape_endpoint([(after, acc) for after, (acc, _) in checks], every)
    terminal = cast("Counts", cast("dict[str, object]", result["counts"])["static"])
    return {
        "run": str(run_dir),
        "seed": result["seed"],
        "data_seed": result.get("data_seed", result["seed"]),
        "recipe": {key: result.get(key) for key in RECIPE},
        "backend": result.get("backend"),
        "endpoint": {
            "status": endpoint.status,
            "escape": endpoint.escape,
            "last": endpoint.last,
            "crossings": list(endpoint.crossings),
        },
        "checks": [
            {"after": after, ESCAPE_CLASS: accuracy, "depth": depth}
            for after, (accuracy, depth) in checks
        ],
        "check_counts": {cast("int", r["after"]): r["static"] for r in rows},
        "terminal": _profile(terminal),
    }


def read_runs(run_dirs: Sequence[Path]) -> list[dict[str, object]]:
    """Every run, checked to share one recipe, one row per (seed, data seed) pair, in order.

    Raises:
        ValueError: on runs that disagree on any ``RECIPE`` key, or a repeated pair.
    """
    runs = [read_run(d) for d in run_dirs]
    if not runs:
        msg = "no runs"
        raise ValueError(msg)
    first = cast("dict[str, object]", runs[0]["recipe"])
    for run in runs[1:]:
        recipe = cast("dict[str, object]", run["recipe"])
        for key in RECIPE:
            if recipe[key] != first[key]:
                msg = f"runs disagree on {key}: {first[key]!r} and {recipe[key]!r}"
                raise ValueError(msg)
    pairs = [(r["seed"], r["data_seed"]) for r in runs]
    if len(set(pairs)) != len(pairs):
        msg = f"a (seed, data seed) pair is repeated: {pairs}"
        raise ValueError(msg)
    return sorted(runs, key=_pair)


def _pair(run: dict[str, object]) -> tuple[int, int]:
    return cast("int", run["seed"]), cast("int", run["data_seed"])


def table(runs: Sequence[dict[str, object]]) -> str:
    """The endpoint per pair, then each run's terminal best depth per class, as Markdown."""
    headings = (
        "init seed",
        "data seed",
        "endpoint",
        f"{ESCAPE_CLASS} escape (first of two > {ESCAPE_THRESHOLD:.0%})",
        "crossings",
        f"{ESCAPE_CLASS} at the last check",
    )
    lines = ["| " + " | ".join(headings) + " |", "|---|---|---|---|---|---|"]
    for run in runs:
        end = cast("dict[str, object]", run["endpoint"])
        last = cast("list[dict[str, object]]", run["checks"])[-1]
        crossings = ", ".join(str(c) for c in cast("list[int]", end["crossings"])) or "-"
        escape = "-" if end["escape"] is None else str(end["escape"])
        accuracy = f"{100 * cast('float', last[ESCAPE_CLASS]):.1f} at depth {last['depth']}"
        row = (run["seed"], run["data_seed"], end["status"], escape, crossings, accuracy)
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines += [
        "",
        "Terminal static profile (full evaluation), best depth and accuracy per class:",
        "",
    ]
    lines.append("| class | " + " | ".join(f"{r['seed']}/{r['data_seed']}" for r in runs) + " |")
    lines.append("|---|" + "---|" * len(runs))
    for cls in CLASSES:
        cells: list[str] = []
        for run in runs:
            accs = cast("dict[str, list[float | None]]", run["terminal"])[cls]
            known = [a for a in accs if a is not None]
            if not known:
                cells.append("-")
                continue
            best = max(known)
            cells.append(f"{100 * best:.1f} at {accs.index(best) + 1}")
        lines.append(f"| {cls} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the module docstring: the table on stdout, everything in ``--out``."""
    parser = argparse.ArgumentParser(description="The crossed-seed diagnostic's endpoint.")
    _ = parser.add_argument("runs", type=Path, nargs="+")
    _ = parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    runs = read_runs(cast("list[Path]", args.runs))
    print(table(runs), end="")  # noqa: T201
    out = cast("Path | None", args.out)
    if out is not None:
        write_json(out, {"threshold": ESCAPE_THRESHOLD, "class": ESCAPE_CLASS, "runs": runs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
