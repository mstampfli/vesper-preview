"""The trainer's throughput on this card: a sweep of the memory and batching options.

The project owner 2026-10-21: before money is spent on a faster card, the trainer must run there at
full speed with as little overhead as possible, and that card has more memory. This sweep measures,
on whatever card it runs, the real ``vesper.train`` path (captured CUDA graphs) for each
combination of the options that trade memory for speed: activation recomputation on or off,
the micro-batch per group (one group of the whole batch, or several in turn), the loss slice
count, and optionally the batch itself. Each combination runs in its OWN process (a CUDA
out-of-memory error leaves a process's allocator fragmented; it must not decide the next
combination), for ``steps`` steps from fresh weights; its log's per-step seconds after the
warm-up and capture give the median step time, tokens per second and the peak memory. A
combination that does not fit is recorded as such, never retried smaller in place.

The batch changes the recipe (gradient noise, learning rate): batch sizes are compared for
throughput only, the choice between them is a recipe decision, not this tool's.
"""

import argparse
import itertools
import json
import logging
import statistics
import subprocess  # nosec B404  (measure: our own interpreter and module, an argv list, no shell)
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

log = logging.getLogger(__name__)
SKIP = 8  # steps left out of the median: the first chunk's signature, the warm-up, the capture
MIN_MEASURED = 3  # a median of fewer steps is no measurement


@dataclass(frozen=True)
class Setting:
    """One combination of the swept options."""

    recompute: bool
    micro: int
    slices: int
    batch: int

    def name(self) -> str:
        """A directory name for the setting's run."""
        rc = "rc" if self.recompute else "norc"
        return f"b{self.batch}-m{self.micro}-{rc}-s{self.slices}"


@dataclass(frozen=True)
class Inputs:
    """What every setting's run reads: the config, the data and the fixed options."""

    config: Path
    data: Path
    finalization: Path
    tokenizer: Path
    steps: int
    scan: str
    objective: str


@dataclass(frozen=True)
class Measured:
    """A setting's outcome: its median step, tokens per second and peak memory, or why not."""

    setting: Setting
    median_step_s: float | None
    tokens_per_s: float | None
    peak_reserved_gib: float | None
    steps_measured: int
    error: str | None

    def record(self) -> dict[str, object]:
        """The outcome as JSON-ready fields."""
        s = self.setting
        return {
            "recompute": s.recompute,
            "micro": s.micro,
            "slices": s.slices,
            "batch": s.batch,
            "fits": self.error is None,
            "median_step_s": self.median_step_s,
            "tokens_per_s": self.tokens_per_s,
            "peak_reserved_gib": self.peak_reserved_gib,
            "steps_measured": self.steps_measured,
            "error": self.error,
        }


def settings(
    recompute: list[bool], micros: list[int], slices: list[int], batches: list[int]
) -> list[Setting]:
    """Every combination whose micro-batch divides its batch, in a stable order."""
    return [
        Setting(recompute=r, micro=m, slices=s, batch=b)
        for b, m, r, s in itertools.product(batches, micros, recompute, slices)
        if b % m == 0
    ]


def read_log(path: Path, skip: int = SKIP) -> tuple[float, float, float, int]:
    """``(median step seconds, tokens per second, peak GiB, steps measured)`` from a run log.

    The median over the steps after ``skip`` (warm-up and capture excluded); tokens per second
    is the measured steps' tokens over their summed seconds. The peak is the reserved peak of
    EVERY step, the warm-up and the capture included: they must fit on the card too.

    Raises:
        ValueError: when fewer than ``MIN_MEASURED`` steps remain after ``skip``.
    """
    seconds: list[float] = []
    tokens = 0
    peak = 0
    for text in path.read_text().splitlines():
        line = cast("dict[str, object]", json.loads(text))
        if line.get("kind") != "step":
            continue
        peak = max(peak, int(cast("int", line["peak_reserved_bytes"])))
        if skip > 0:
            skip -= 1
            continue
        seconds.append(float(cast("float", line["seconds"])))
        tokens += int(cast("int", line["tokens"]))
    if len(seconds) < MIN_MEASURED:
        msg = f"{path}: {len(seconds)} measured steps after the warm-up; run more steps"
        raise ValueError(msg)
    return statistics.median(seconds), tokens / sum(seconds), peak / 2**30, len(seconds)


def measure(setting: Setting, inputs: Inputs, out: Path) -> Measured:
    """One setting in its own ``vesper.train`` process; its measurement or why it did not fit."""
    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(inputs.config.read_text()))
    raw["training"]["batch_chunks"] = setting.batch
    where = out / setting.name()
    where.mkdir(parents=True, exist_ok=False)
    config = where / "config.yaml"
    _ = config.write_text(yaml.safe_dump(raw))
    command = [
        sys.executable, "-m", "vesper.train",
        "--config", str(config), "--data", str(inputs.data),
        "--finalization", str(inputs.finalization), "--tokenizer", str(inputs.tokenizer),
        "--out", str(where / "run"), "--micro-chunks", str(setting.micro),
        "--steps", str(inputs.steps), "--seed", "0", "--device", "cuda",
        "--scan", inputs.scan, "--objective", inputs.objective,
        "--loss-slices", str(setting.slices),
    ]  # fmt: skip
    if setting.recompute:
        command.append("--recompute")
    with (where / "train.log").open("w") as handle:
        done = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)  # noqa: S603  # nosec B603
    if done.returncode != 0:
        text = (where / "train.log").read_text()
        oom = "OutOfMemoryError" in text or "out of memory" in text
        return Measured(setting, None, None, None, 0, "out of memory" if oom else text[-500:])
    median, rate, peak, count = read_log(where / "run" / "log.jsonl")
    return Measured(setting, median, rate, peak, count, None)


def table(results: list[Measured]) -> list[str]:
    """One line per setting, the fastest fitting first, then those that did not fit."""
    fitting = sorted((r for r in results if r.error is None), key=_slowest_last)
    lines: list[str] = []
    for r in fitting + [r for r in results if r.error is not None]:
        if r.tokens_per_s is None or r.median_step_s is None or r.peak_reserved_gib is None:
            lines.append(f"{r.setting.name():24s} no fit: {r.error}")
            continue
        rate = f"{r.tokens_per_s:9.0f} tok/s"
        lines.append(
            " ".join(
                [
                    f"{r.setting.name():24s}",
                    rate,
                    f"{r.median_step_s:6.3f} s",
                    f"{r.peak_reserved_gib:5.2f} GiB reserved",
                ]
            )
        )
    return lines


def _slowest_last(result: Measured) -> float:
    return -(result.tokens_per_s or 0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    _ = parser.add_argument("--config", type=Path, required=True)
    _ = parser.add_argument("--data", type=Path, required=True)
    _ = parser.add_argument("--finalization", type=Path, required=True)
    _ = parser.add_argument("--tokenizer", type=Path, required=True)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--steps", type=int, default=30)
    _ = parser.add_argument("--scan", default="fused-fp32")
    _ = parser.add_argument("--objective", default="fixed")
    modes: list[str] = ["on", "off", "both"]
    default_slices: list[int] = [4]
    _ = parser.add_argument("--recompute", choices=modes, default="both")
    _ = parser.add_argument("--micro", type=int, nargs="+", required=True)
    _ = parser.add_argument("--slices", type=int, nargs="+", default=default_slices)
    _ = parser.add_argument("--batch", type=int, nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sweep; write ``throughput.json`` and print the table, fastest fitting first."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    inputs = Inputs(
        config=cast("Path", args.config),
        data=cast("Path", args.data),
        finalization=cast("Path", args.finalization),
        tokenizer=cast("Path", args.tokenizer),
        steps=cast("int", args.steps),
        scan=cast("str", args.scan),
        objective=cast("str", args.objective),
    )
    modes = {"on": [True], "off": [False], "both": [False, True]}
    out = cast("Path", args.out)
    out.mkdir(parents=True, exist_ok=False)
    grid = settings(
        modes[cast("str", args.recompute)],
        cast("list[int]", args.micro),
        cast("list[int]", args.slices),
        cast("list[int]", args.batch),
    )
    results: list[Measured] = []
    for setting in grid:
        result = measure(setting, inputs, out)
        log.info("%s: %s", setting.name(), result)
        results.append(result)
        records: list[dict[str, object]] = [r.record() for r in results]
        _ = (out / "throughput.json").write_text(json.dumps(records, indent=2))
    lines = table(results)
    _ = (out / "throughput.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        log.info("%s", line)
    return 0 if any(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
