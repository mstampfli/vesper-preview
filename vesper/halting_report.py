"""The stage-1 report: the runs' result records pooled per condition, as one Markdown page.

    python -m vesper.halting_report RUN_ROOT [--out REPORT.md]

Reads every ``RUN_ROOT/*/result.json`` written by ``vesper.halting_experiment``. Counts are
summed over the seeds of a condition and summarized by ``halting_experiment.summarize``, the
one place accuracies, costs and entropies are computed; per-seed values give the spread. The
plan's acceptance criterion (docs/reviews/2026-10-15-halting-experiment-plan.md: the live gates
under copy-forward lose at most 1 point of accuracy on every class against the dense readout at
the same exit) is evaluated at ``Q_REFERENCE`` for B-D. A class's empirical depth need is the
first pass of B's dense readout within ``NEED_TOLERANCE`` of its best pass, for a class B
learned above ``LEARNED_FLOOR`` only. Section 10 evaluates the learnability probe's two gates per
seed from B's fixed-depth profile (``_gates``; docs/reviews/2026-10-15-retrieval-binding.md, 5.2).
"""

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from vesper.config import VesperConfig, parse_bytes
from vesper.halting_experiment import CONDITIONS, Q_GRID, Q_REFERENCE, summarize
from vesper.synthetic import CLASSES

NEED_TOLERANCE = 0.01
LEARNED_FLOOR = 0.5  # chance is 1/48 on a node answer, 1/8 among the cycle's nodes
ACCEPTANCE_DROP = 0.01
DEPTH_GAIN = 0.05  # Gate 2: a learned class gains this much from pass 1 to its best depth


@dataclass(frozen=True)
class Protocol:
    """One frozen recipe the gates may be read under.

    B's seeds, the update count, the evaluation split and population (batches, sequences per
    batch), the training mix recipe(s), the column count and the training backend (device and
    recurrent scan, ``_backend``: another backend is other arithmetic, so another recipe).
    ``None`` among ``mixes`` admits records written before the mix was recorded.
    """

    name: str
    seeds: tuple[int, ...]
    steps: int
    split: str
    population: tuple[int, int]
    mixes: tuple[str | None, ...]
    columns: int
    backend: tuple[str, str]


CPU_REFERENCE = ("cpu", "reference")
# the recipes the gates belong to; any other set of records keeps its tables and gets no verdict
PROTOCOLS = (
    # the learnability probe, docs/reviews/2026-10-15-retrieval-binding.md 5.2. Its records predate
    # the mix field: every record with an eval_split (1678173 on) trained under v1's schedule.
    Protocol("probe-v1", (0, 1), 12000, "validation", (32, 32), (None, "v1"), 2, CPU_REFERENCE),
    # the one-column calibration, docs/reviews/2026-10-16-stage1-c1-recipe.md (maintainer B,
    # 2026-10-16)
    Protocol("c1-slow-v2", (0, 1), 36000, "validation", (32, 32), ("v2-slow",), 1, CPU_REFERENCE),
)


def _backend(record: dict[str, object]) -> tuple[str, str]:
    """``(device, scan)`` a record trained with.

    Records written before 2026-10-16 carry neither field and trained on the CPU with the
    reference scan, the only backend the driver had.
    """
    return str(record.get("device", "cpu")), str(record.get("scan", "reference"))


# "at most 1 point" includes 1 point: 0.500 - 0.490 is 0.010000000000000009 in floating point
COMPARISON_SLACK = 1e-12
NATS_RESOLUTION = 5e-4  # below the three printed decimals
NAMES = {"act": "A act", "fixed": "B fixed", "local": "C local", "aggregate": "D aggregate"}

Record = dict[str, object]
Summary = dict[str, object]
Scores = dict[str, float | None]


@dataclass(frozen=True)
class Run:
    """One result record and the config it was trained from.

    The config is parsed from bytes whose sha256 is the record's ``config_sha256``.
    """

    record: Record
    config: VesperConfig


def load_results(root: Path) -> list[Run]:
    """Every ``ROOT/*/result.json`` with its verified recipe.

    The recipe is the run's snapshot (``config_snapshot``, a path relative to the record's
    directory, so a copied run directory reads anywhere); the original config path is not
    consulted then. Records written before the snapshot existed (legacy) name only the original
    path, read as recorded (relative to the current directory), and must still hash to the
    record. The bytes hashed are the bytes parsed.

    Raises:
        ValueError: when a recipe's bytes no longer have the recorded hash, a snapshot path is
            absolute or leaves its run directory, or a legacy record's config file is gone.
        FileNotFoundError: when a record's snapshot is missing.
    """
    runs: list[Run] = []
    for path in sorted(root.glob("*/result.json")):
        record = cast("Record", json.loads(path.read_text(encoding="utf-8")))
        snapshot = record.get("config_snapshot")
        if snapshot is None:
            recipe = Path(cast("str", record["config"]))
            if not recipe.is_file():
                msg = f"{path}: a record without a snapshot needs its config {recipe}, missing"
                raise ValueError(msg)
        else:
            name = Path(cast("str", snapshot))
            if name.is_absolute() or ".." in name.parts:
                msg = f"{path}: config_snapshot {name} must be relative to its run directory"
                raise ValueError(msg)
            recipe = path.parent / name
        data = recipe.read_bytes()
        if hashlib.sha256(data).hexdigest() != cast("str", record["config_sha256"]):
            msg = f"{path}: {recipe} changed since the run"
            raise ValueError(msg)
        runs.append(Run(record=record, config=parse_bytes(data, str(recipe))))
    return runs


def pooled_summary(records: Sequence[Record], t_max: int, columns: int) -> Summary:
    """``summarize`` over the counts of ``records`` pooled key by key (the seeds pooled).

    Counts are summed; a maximum (``*_max_*``) is reduced by max (maintainer B's review of c7d25c5:
    a sum turned maxima of 0.125 and 0.25 into 0.375). Entropies and information computed from
    pooled histograms describe the MIXTURE of the models, not any one of them: per-model
    values come from ``pooled_summary([record])`` (the report's section 6).
    """
    totals: dict[str, Tensor] = {}
    for record in records:
        for key, value in cast("dict[str, object]", record["counts"]).items():
            tensor = torch.tensor(value, dtype=torch.float64)
            if key not in totals:
                totals[key] = tensor
            elif "_max_" in key:
                totals[key] = torch.maximum(totals[key], tensor)
            else:
                totals[key] = totals[key] + tensor
    return summarize(totals, t_max, columns)


def report(runs: Sequence[Run]) -> str:
    """The Markdown page of the module docstring.

    Raises:
        ValueError: when there is no record, or records disagree on config, steps or held-out size.
    """
    if not runs:
        msg = "no result records"
        raise ValueError(msg)
    records = [run.record for run in runs]
    first = records[0]
    _check_identities(records)
    environments = [cast("dict[str, str]", r["environment"]) for r in records]
    commits = ", ".join(sorted({e["vesper_commit"][:7] for e in environments}))
    dirty = ", ".join(sorted({e["vesper_dirty"] for e in environments}))
    # one config_sha256 for every record (_check_identities): one recipe
    cfg = runs[0].config
    t_max, columns = cfg.model.t_max, cfg.model.columns
    by_condition = {c: [r for r in records if r["condition"] == c] for c in CONDITIONS}
    present = {c: rs for c, rs in by_condition.items() if rs}
    pooled = {c: pooled_summary(rs, t_max, columns) for c, rs in present.items()}
    seeds = {c: [pooled_summary([r], t_max, columns) for r in rs] for c, rs in present.items()}
    counts = ", ".join(f"{NAMES[c]} {len(rs)}" for c, rs in present.items())
    sha = cast("str", first["config_sha256"])[:12]
    header = " ".join(
        [
            f"Commit {commits} (dirty: {dirty}), config {first['config']} (sha256 {sha},",
            "verified per record against its snapshot, or for older records its original),",
            f"{first['steps']} steps, {first['eval_sequences']} held-out sequences per run",
            f"({first.get('eval_split', 'eval, before the split was recorded')}),",
            f"T = {t_max}, {columns} columns. Seeds: {counts}.",
            "Accuracies pooled over seeds; brackets are the per-seed range.",
        ]
    )
    need = _need(pooled.get("fixed"), t_max)
    lines = ["# Halting experiment, stage 1", "", header, ""]
    lines += _learning(pooled.get("fixed"), need, t_max)
    lines += _deployed(pooled, seeds, t_max)
    lines += _per_class(pooled, need)
    lines += _acceptance(pooled)
    lines += _curve(pooled)
    lines += _information(seeds)
    lines += _register(pooled, t_max)
    lines += _modes(pooled)
    lines += _ablation(pooled)
    lines += _gates(by_condition["fixed"], t_max, columns)
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """The command line of the module docstring."""
    parser = argparse.ArgumentParser(description="Pool stage-1 result records into a report.")
    _ = parser.add_argument("root", type=Path)
    _ = parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = report(load_results(cast("Path", args.root)))
    out = cast("Path | None", args.out)
    if out is None:
        print(text, end="")  # noqa: T201
    else:
        _ = out.write_text(text, encoding="utf-8")
    return 0


SHARED = ("config_sha256", "steps", "eval_sequences", "eval_split", "q_reference", "mix_schedule")


def _check_identities(records: Sequence[Record]) -> None:
    """One recipe for all records, one record per condition and seed, one objective per condition.

    Raises:
        ValueError: on a disagreement in ``SHARED``, a repeated condition and seed (a duplicate
            would count as another seed), or a condition whose records differ in an objective
            setting (``beta``: pooling different betas is not one condition; maintainer B's review
            of c7d25c5).
    """
    first = records[0]
    for key in SHARED:
        if any(r.get(key) != first.get(key) for r in records):
            msg = f"records disagree on {key}"
            raise ValueError(msg)
    if len({_backend(r) for r in records}) != 1:
        msg = "records disagree on the training backend (device, scan)"
        raise ValueError(msg)
    seen: set[tuple[object, object]] = set()
    betas: dict[object, object] = {}
    for record in records:
        identity = (record["condition"], record["seed"])
        if identity in seen:
            msg = f"duplicate record for condition {identity[0]} seed {identity[1]}"
            raise ValueError(msg)
        seen.add(identity)
        beta = betas.setdefault(record["condition"], record.get("beta"))
        if beta != record.get("beta"):
            msg = f"condition {record['condition']} pools different beta values"
            raise ValueError(msg)


def _row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def _table(*headings: str) -> list[str]:
    return [_row(*headings), "|" + "---|" * len(headings)]


def _scores(summary: Summary, key: str) -> Scores:
    return cast("Scores", summary[key])


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def _within(value: float, best: float) -> bool:
    """``value`` is within ``NEED_TOLERANCE`` of ``best``, the tolerance itself included."""
    return value >= best - NEED_TOLERANCE - COMPARISON_SLACK


def _nats(value: float) -> str:
    """Three decimals; a rounding residue below their resolution prints as 0.000, not -0.000."""
    return f"{0.0 if abs(value) < NATS_RESOLUTION else value:.3f}"


def _need(fixed: Summary | None, t_max: int) -> dict[str, int | None]:
    """Each class's first dense pass within ``NEED_TOLERANCE`` of its best, if B learned it."""
    out: dict[str, int | None] = {}
    for name in CLASSES:
        out[name] = None
    if fixed is None:
        return out
    curve = cast("list[Scores]", fixed["dense_pass"])
    for name in CLASSES:
        values = [row[name] or 0.0 for row in curve]
        best = max(values)
        if best >= LEARNED_FLOOR:
            out[name] = next(t + 1 for t in range(t_max) if _within(values[t], best))
    return out


def _learning(fixed: Summary | None, need: dict[str, int | None], t_max: int) -> list[str]:
    lines = ["## 1. What depth each class needs (B, dense readout of each pass)", ""]
    if fixed is None:
        return [*lines, "No B run.", ""]
    curve = cast("list[Scores]", fixed["dense_pass"])
    lines += _table("class", *(f"pass {t + 1}" for t in range(t_max)), "need")
    for name in CLASSES:
        cells = [_pct(curve[t][name]) for t in range(t_max)]
        lines.append(_row(name, *cells, need[name] or "not learned"))
    return [*lines, ""]


def _matched_static(summary: Summary, passes: float, t_max: int) -> float:
    """The static copy-forward curve's overall accuracy interpolated at ``passes``."""
    static = [row["overall"] or 0.0 for row in cast("list[Scores]", summary["static"])]
    low = min(max(int(passes), 1), t_max)
    high = min(low + 1, t_max)
    fraction = passes - low if high > low else 0.0
    return static[low - 1] + fraction * (static[high - 1] - static[low - 1])


def _deployed(pooled: dict[str, Summary], seeds: dict[str, list[Summary]], t_max: int) -> list[str]:
    lines = [
        f"## 2. Deployed policies (B-D at q = {Q_REFERENCE}, A its own ACT), against matched cost",
        "",
        *_table(
            "condition",
            "readout",
            "accuracy",
            "passes (all / loss)",
            "replayed writes",
            "random, same cost (one-hot)",
            "static at the same passes (one-hot)",
        ),
    ]
    for condition, summary in pooled.items():
        deployed = _scores(summary, "deployed/prefill")
        spread = [_scores(s, "deployed/prefill")["overall"] or 0.0 for s in seeds[condition]]
        passes = deployed["passes"] or 0.0
        lines.append(
            _row(
                NAMES[condition],
                "ACT mixture" if condition == "act" else "one-hot exit",
                f"{_pct(deployed['overall'])} [{_pct(min(spread))}, {_pct(max(spread))}]",
                f"{passes:.2f} / {deployed['loss_passes'] or 0.0:.2f}",
                f"{deployed['replayed_writes'] or 0.0:.2f}",
                _pct(_scores(summary, "random")["overall"]),
                _pct(_matched_static(summary, passes, t_max)),
            )
        )
    note = [  # one Markdown paragraph
        "For A the comparators differ in readout as well as in policy: A emits ACT's",
        "weighted mixture of its passes, the random and static comparators each token's own",
        "exit pass (maintainer B's review of c7d25c5).",
    ]
    return [*lines, "", *note, ""]


def _per_class(pooled: dict[str, Summary], need: dict[str, int | None]) -> list[str]:
    lines = [
        "## 3. Per class, deployed: accuracy at the mean passes the class's answers took",
        "",
        *_table("class", "need", *(NAMES[c] for c in pooled)),
    ]
    for i, name in enumerate(CLASSES):
        cells: list[str] = []
        for summary in pooled.values():
            histogram = cast("dict[str, list[list[int]]]", summary["depth/class"])["histogram"][i]
            total = sum(histogram)
            depth = sum((t + 1) * h for t, h in enumerate(histogram)) / total if total else 0.0
            cells.append(f"{_pct(_scores(summary, 'deployed/prefill')[name])} at {depth:.2f}")
        lines.append(_row(name, need[name] or "-", *cells))
    return [*lines, ""]


def _acceptance(pooled: dict[str, Summary]) -> list[str]:
    heading = " ".join(
        [
            f"## 4. Acceptance at q = {Q_REFERENCE}: (iii) live gates with copy-forward at most",
            f"{100 * ACCEPTANCE_DROP:.0f} point below (i) dense at the same exit, every class",
        ]
    )
    lines = [
        heading,
        "",
        *_table("condition", "worst class", "drop (points)", "(ii) trace drop", "verdict"),
    ]
    for condition, summary in pooled.items():
        if condition == "act":
            continue
        dense = _scores(summary, f"dense_exit/{Q_REFERENCE}")
        live = _scores(summary, f"live/{Q_REFERENCE}")
        trace = _scores(summary, f"trace/{Q_REFERENCE}")
        seen = [n for n in CLASSES if dense[n] is not None]
        drops = {n: (dense[n] or 0.0) - (live[n] or 0.0) for n in seen}
        worst = max(seen, key=drops.__getitem__)
        trace_drop = max((dense[n] or 0.0) - (trace[n] or 0.0) for n in seen)
        verdict = "pass" if drops[worst] <= ACCEPTANCE_DROP + COMPARISON_SLACK else "FAIL"
        lines.append(
            _row(
                NAMES[condition],
                worst,
                f"{100 * drops[worst]:.1f}",
                f"{100 * trace_drop:.1f}",
                verdict,
            )
        )
    return [*lines, ""]


def _curve(pooled: dict[str, Summary]) -> list[str]:
    learned = [c for c in pooled if c != "act"]
    lines = [
        "## 5. Cost and quality over q: overall accuracy (i) / (ii) / (iii), passes of (iii)",
        "",
        *_table("q", *(NAMES[c] for c in learned)),
    ]
    for q in Q_GRID:
        cells: list[str] = []
        for condition in learned:
            summary = pooled[condition]
            live = _scores(summary, f"live/{q}")
            dense = _pct(_scores(summary, f"dense_exit/{q}")["overall"])
            trace = _pct(_scores(summary, f"trace/{q}")["overall"])
            cells.append(
                f"{dense} / {trace} / {_pct(live['overall'])}, {live['passes'] or 0.0:.2f}"
            )
        lines.append(_row(q, *cells))
    return [*lines, ""]


def _per_model(summaries: list[Summary], read: Callable[[Summary], float | None]) -> str:
    """A per-model statistic as mean [min-max] over the seeds' own summaries."""
    values = [v for v in (read(s) for s in summaries) if v is not None]
    if not values:
        return "-"
    mean = sum(values) / len(values)
    return f"{_nats(mean)} [{_nats(min(values))}, {_nats(max(values))}]"


def _information(seeds: dict[str, list[Summary]]) -> list[str]:
    # per model: a statistic of the pooled histograms would describe the mixture of the
    # seeds' models, so two token-independent models with different constant depths would
    # show log 2 of "information" that neither has (maintainer B's review of c7d25c5)
    lines = [
        "## 6. Is depth input-specific? Deployed depth entropy and information (nats), per",
        "model: mean [range] over the seeds",
        "",
        *_table(
            "condition",
            "H(depth)",
            "I(depth; role)",
            "I(depth; class), loss tokens",
            "soft I(token; exit), all",
            "soft I(token; exit), loss",
        ),
    ]

    def histogram(name: str, field: str) -> Callable[[Summary], float | None]:
        return lambda s: cast("dict[str, float]", s[f"depth/{name}"])[field]

    def soft(scope: str) -> Callable[[Summary], float | None]:
        return lambda s: cast("float | None", s.get(f"soft_information/{scope}"))

    for condition, summaries in seeds.items():
        lines.append(
            _row(
                NAMES[condition],
                _per_model(summaries, histogram("role", "entropy")),
                _per_model(summaries, histogram("role", "information")),
                _per_model(summaries, histogram("class", "information")),
                _per_model(summaries, soft("all")),
                _per_model(summaries, soft("loss")),
            )
        )
    return [*lines, ""]


def _register(pooled: dict[str, Summary], t_max: int) -> list[str]:
    lines = [
        "## 7. Register: reader at fixed depth r* (shallowest within 1 point, deep register)",
        "",
        *_table("condition", "r*", "register at its own depth", "forced 1", "forced T"),
    ]
    for condition, summary in pooled.items():
        grid = cast("list[list[float]]", summary["register"])
        deep = [row[t_max] for row in grid]
        reader = next(r for r in range(t_max) if _within(deep[r], max(deep)))
        row = grid[reader]
        # until the reader is learned with a deep register, r* and the intervention say
        # nothing (maintainer B's review of c7d25c5)
        name = f"{reader + 1}" if max(deep) >= LEARNED_FLOOR else f"inconclusive ({reader + 1})"
        lines.append(_row(NAMES[condition], name, _pct(row[0]), _pct(row[1]), _pct(row[t_max])))
    return [*lines, ""]


def _ablation(pooled: dict[str, Summary]) -> list[str]:
    lines = [
        "## 9. Recurrence ablation: the task is hybrid (stacked attentions relay across the",
        "gap), so the same trained model with its recurrent read removed shows how much each",
        "class depends on the recurrent state. Each pair differs only in that read: the deployed",
        "exit trace replayed with one exit per token (for A without its ACT mixture; A's deployed",
        "readout is in section 2), and the dense last pass. An intervention on these models, not",
        "a separately trained model without recurrence.",
        "",
    ]
    # records before the paired reference (9044238) compared against A's mixture readout
    measured = {c: s for c, s in pooled.items() if "ablated/reference" in s}
    if not measured:
        return [*lines, "Not measured with a paired reference in these records.", ""]
    lines += _table(
        "class", *(f"{NAMES[c]}: trace / ablated, dense last / ablated" for c in measured)
    )
    for name in [*CLASSES, "overall"]:
        cells = [
            " / ".join(
                _pct(_scores(summary, key)[name]) for key in ("ablated/reference", "ablated/trace")
            )
            + ", "
            + " / ".join(
                _pct(value[name])
                for value in (
                    cast("list[Scores]", summary["dense_pass"])[-1],
                    _scores(summary, "ablated/dense_last"),
                )
            )
            for summary in measured.values()
        ]
        lines.append(_row(name, *cells))
    differences = ", ".join(
        f"{NAMES[c]} {float(cast('float', s['ablated_max_output_difference'])):.3g}"
        for c, s in measured.items()
    )
    return [*lines, "", f"Largest output change at the replayed exits: {differences}.", ""]


def _protocol_breaches(records: Sequence[Record], protocol: Protocol, columns: int) -> list[str]:
    """Why ``records`` (B's) are not ``protocol``; empty if they are.

    Seeds exactly the protocol's, each after its update count, evaluated on its split and
    population (batches, sequences per batch), trained under one of its mix recipes, with its
    column count. One recipe for all records is already enforced (``SHARED``). Anything else keeps
    its tables and gets no verdict (maintainer B's review of 1678173: a pass for one seed, other
    seeds, 1000 updates, the final split or four sequences was a pass no protocol defined).
    """
    breaches: list[str] = []
    seeds = sorted(cast("int", r["seed"]) for r in records)
    if tuple(seeds) != protocol.seeds:
        breaches.append(f"B seeds {seeds}, it has {list(protocol.seeds)}")
    steps = sorted({cast("int", r["steps"]) for r in records})
    if steps != [protocol.steps]:
        breaches.append(f"{steps} updates, it has {protocol.steps}")
    splits = sorted({str(r.get("eval_split")) for r in records})
    if splits != [protocol.split]:
        breaches.append(f"evaluation split {splits}, it uses {protocol.split}")
    population = sorted({(str(r.get("eval_batches")), str(r.get("eval_size"))) for r in records})
    batches, size = protocol.population
    if population != [(str(batches), str(size))]:
        shown = ", ".join(f"{b} x {s}" for b, s in population)
        breaches.append(f"evaluation of {shown} sequences, it uses {batches} x {size}")
    mixes = sorted({str(r.get("mix_schedule")) for r in records})
    if any(r.get("mix_schedule") not in protocol.mixes for r in records):
        allowed = [m for m in protocol.mixes if m is not None]
        breaches.append(f"mix {mixes}, it uses {allowed}")
    if columns != protocol.columns:
        breaches.append(f"{columns} columns, it has {protocol.columns}")
    backends = sorted({_backend(r) for r in records})
    if backends != [protocol.backend]:
        shown = ", ".join(f"{d}/{s}" for d, s in backends)
        breaches.append(f"backend {shown}, it trains on {'/'.join(protocol.backend)}")
    return breaches


def _protocol_of(records: Sequence[Record], columns: int) -> tuple[Protocol | None, str]:
    """The one protocol ``records`` match exactly, or None with every protocol's breaches."""
    reasons: list[str] = []
    for protocol in PROTOCOLS:
        breaches = _protocol_breaches(records, protocol, columns)
        if not breaches:
            return protocol, ""
        reasons.append(f"{protocol.name}: {'; '.join(breaches)}")
    return None, " | ".join(reasons)


def _gates(fixed: Sequence[Record], t_max: int, columns: int) -> list[str]:
    """The learnability probe's gates (docs/reviews/2026-10-15-retrieval-binding.md, 5.2).

    ``A(seed, class, j)`` is B's ``static`` readout: every token forced to depth j (a constant
    ``TraceRule``), that pass's own output. Gate 1: every hop and read class above
    ``LEARNED_FLOOR`` at its best depth, in every seed. Gate 2: one and the same class gains at
    least ``DEPTH_GAIN`` from pass 1 to its best depth in every seed. A class without labels has no
    accuracy and fails both. The verdicts are given only for records that match one frozen
    protocol of ``PROTOCOLS`` exactly, and name it; any other records keep the tables and read
    "not assessed" with every protocol's breaches.
    """
    lines = [
        "## 10. Probe gates (B: every token forced to depth j, per seed)",
        "",
    ]
    if not fixed:
        return [*lines, "No B run.", ""]
    protocol, breaches = _protocol_of(fixed, columns)
    by_seed = {cast("int", r["seed"]): r for r in fixed}  # one record per seed (_check_identities)
    summaries = [
        (seed, pooled_summary([by_seed[seed]], t_max, columns)) for seed in sorted(by_seed)
    ]
    lines += _table(
        "class", *(f"seed {seed}: best j, A, gain over pass 1" for seed, _ in summaries)
    )
    learned: dict[str, bool] = {}
    deeper: dict[str, bool] = {}
    for name in CLASSES:
        cells: list[str] = []
        learned[name] = deeper[name] = True
        for _, summary in summaries:
            curve = [row[name] for row in cast("list[Scores]", summary["static"])]
            if any(a is None for a in curve):
                cells.append("no labels")
                learned[name] = deeper[name] = False
                continue
            values = [cast("float", a) for a in curve]
            best = max(range(t_max), key=values.__getitem__)
            gain = values[best] - values[0]
            cells.append(f"{best + 1}, {_pct(values[best])}, {100 * gain:+.1f}")
            learned[name] = learned[name] and values[best] > LEARNED_FLOOR
            deeper[name] = deeper[name] and gain >= DEPTH_GAIN - COMPARISON_SLACK
        lines.append(_row(name, *cells))
    scored = [n for n in CLASSES if n.startswith(("hop", "read"))]
    gate1 = all(learned[n] for n in scored)
    gaining = [n for n in CLASSES if learned[n] and deeper[n]]
    failing = ", ".join(n for n in scored if not learned[n]) or "none"
    floor, gain_points = 100 * LEARNED_FLOOR, 100 * DEPTH_GAIN
    verdict1 = f"{'pass' if gate1 else 'FAIL'} (not learned: {failing})."
    verdict2 = f"{'pass' if gaining else 'FAIL'} (classes: {', '.join(gaining) or 'none'})."
    if protocol is None:
        verdict1 = verdict2 = f"not assessed (outside every gate protocol: {breaches})."
        under = ""
    else:
        under = f" under {protocol.name}"
    rule2 = f"one learned class gaining at least {gain_points:.0f} points over pass 1"
    return [
        *lines,
        "",
        f"Gate 1 (every hop and read class above {floor:.0f} % in every seed){under}: {verdict1}",
        f"Gate 2 ({rule2} in every seed){under}: {verdict2}",
        "",
    ]


def _modes(pooled: dict[str, Summary]) -> list[str]:
    lines = [
        "## 8. Prefill, token-by-token decode and carried chunks of the deployed policy",
        "",
        *_table(
            "condition",
            "prefill",
            "decode",
            "carried",
            "same depth (decode / carried)",
            "same answer",
            "max output difference",
        ),
    ]
    for condition, summary in pooled.items():
        chunked = ["decode", "carried"]
        accuracy = [
            _pct(_scores(summary, f"deployed/{m}")["overall"]) for m in ["prefill", *chunked]
        ]
        agree = [cast("dict[str, float]", summary[f"deployed_agreement/{m}"]) for m in chunked]
        difference = max(
            cast("float", summary[f"deployed_max_output_difference/{m}"]) for m in chunked
        )
        lines.append(
            _row(
                NAMES[condition],
                *accuracy,
                f"{_pct(agree[0]['depth'])} / {_pct(agree[1]['depth'])}",
                _pct(min(a["answer"] for a in agree)),
                f"{difference:.1e}",
            )
        )
    measured = {c: s for c, s in pooled.items() if "dense_mode/prefill" in s}
    if not measured:
        return [*lines, "", "(i) and (ii) over the modes: not measured in these records.", ""]
    lines += [
        "",
        "(i) dense at its own exits and (ii) those exits replayed with copy-forward,",
        f"at q = {Q_REFERENCE}, over the same modes; dense exits' agreement with prefill's",
        "",
        *_table(
            "condition", *(f"{m}: (i) / (ii) / exits" for m in ("prefill", "decode", "carried"))
        ),
    ]
    for condition, summary in measured.items():
        cells = [
            " / ".join(
                [
                    _pct(_scores(summary, f"dense_mode/{m}")["overall"]),
                    _pct(_scores(summary, f"trace_mode/{m}")["overall"]),
                    _pct(cast("float", summary[f"dense_mode_exits/{m}"])),
                ]
            )
            for m in ("prefill", "decode", "carried")
        ]
        lines.append(_row(NAMES[condition], *cells))
    return [*lines, ""]


if __name__ == "__main__":
    raise SystemExit(main())
