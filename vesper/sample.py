"""The tokenizer's training sample and the frozen tokenizer (spec, Data: a 10 GB sample of the mix).

    python -m vesper.sample --web ~/data/fineweb-edu/sample/10BT --mix ~/data/mix --out DIR
        [--config configs/vesper-s.yaml] [--budget-bytes 10737418240] [--seed 0]
        [--label frozen-candidate]

Components and shares (docs/reviews/2026-10-15-mixture-sources.md): web 80%, code 15%,
encyclopedia 4%, books 1% of the budget's UTF-8 bytes. Web is every parquet file under ``--web``;
code, wiki and books are the collected files under ``--mix/<component>``. The texts are the
training split, deduplicated across components and interleaved by byte share
(``corpus.iter_mixed_texts``). The receipt beside the artifact carries the input manifest (every
file with its bytes and sha256), the seed, the split fraction, the budget and the rules, the
exact bytes and documents per component, the duration, the environment and the artifact's
sha256; the label says what the run is (a pilot, a candidate, the frozen tokenizer).
"""

import argparse
import logging
import sys
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from vesper.config import load
from vesper.corpus import (
    Component,
    Split,
    budgets,
    check_inputs,
    file_manifest,
    finalize,
    iter_mixed_texts,
    load_finalization,
)
from vesper.generations import SELECTION, pinned, selection
from vesper.keys import SequenceDigest, file_sha256
from vesper.provenance import environment, write_json
from vesper.tokenizer import DEFAULT_PRE_TOKENIZER, PRE_TOKENIZERS, Tokenizer, train_from_texts

SHARES: tuple[tuple[str, float], ...] = (
    ("web", 0.80),
    ("code", 0.15),
    ("wiki", 0.04),
    ("books", 0.01),
)
_GIB = 1 << 30  # a shift, not 2**30: the type checkers widen a large literal power to Any
DEFAULT_BUDGET = 10 * _GIB
ROUND_TRIP = "The delta rule erases what is stored under the key. foo123bar baz 2026"
ORDER = (
    "corpus.iter_raw_documents(paths sorted by name, seed): block shuffle, (file, row group) pairs"
    " permuted by blake2b rank of 'groups:<seed>:<index>', rows inside each group by"
    " 'rows:<seed>:<file>:<group>:<index>'"
)
FINALIZE = (
    "corpus.finalize over all inputs (file order, two passes): for every dedup key seen more than"
    " once the document with the smallest id survives and the others are dropped; then"
    " corpus.select_documents skips the dropped ids and data.is_held_out(group_key, fraction)"
    " keeps the training side; membership is a property of the ids, the same under every seed"
)
MIXING = (
    "corpus.iter_mixed_texts: budgets share * bytes (floored, remainder to the last component);"
    " at each step the component with the smallest emitted/budget yields its next document whole;"
    " a component stops at its budget or when exhausted"
)
TOKENIZER = (
    "byte-level BPE (HuggingFace tokenizers BpeTrainer), initial alphabet the 256 ByteLevel"
    " symbols, no special tokens; pre-tokenized under the named rule recorded beside this field"
    " (vesper.tokenizer.PRE_TOKENIZERS, vesper.tokenizer.train_from_texts)"
)
_PROGRESS_EVERY = 100_000
log = logging.getLogger("vesper.sample")


def components(web: Path, mix: Path) -> list[Component]:
    """The explicit legacy layout: the four components from flat directories, files by name.

    A run uses this only when told to (``--legacy-layout``); the selection is the default
    (``selected_components``). Nothing here pins a generation.

    Raises:
        ValueError: when a component has no parquet file.
    """
    out: list[Component] = []
    for name, share in SHARES:
        directory = web if name == "web" else mix / name
        paths = tuple(sorted(directory.glob("*.parquet")))
        if not paths:
            msg = f"component {name!r} has no parquet file under {directory}"
            raise ValueError(msg)
        out.append(Component(name, paths, share))
    return out


def selected_components(data: Path) -> list[Component]:
    """The four components through the data root's selection: each kind's pinned generation.

    The selection names one generation per kind; each is validated (``generations.pinned``) and
    its inputs are the component's files in manifest order; the component carries the
    generation's pin, so the finalization receipt and the training recipe name it.

    Raises:
        ValueError: when the selection is missing, names no generation for a kind, or a pinned
            generation fails validation.
    """
    chosen = selection(data)
    by_kind = {ref.kind: ref for ref in chosen.references}
    out: list[Component] = []
    for name, share in SHARES:
        ref = by_kind.get(name)
        if ref is None:
            msg = f"the selection {data / SELECTION} names no generation for {name!r}"
            raise ValueError(msg)
        generation = pinned(data, ref)
        out.append(Component(name, generation.inputs, share, generation=ref.pin))
    return out


def components_for(
    *, data: Path | None, legacy_layout: bool, web: Path | None, mix: Path | None
) -> list[Component]:
    """The inputs a command line names, one way only.

    The selection under ``--data``, or the explicit legacy layout under ``--legacy-layout --web
    --mix``; anything else is refused, never guessed.

    Raises:
        ValueError: when neither or both modes are named, or the legacy mode lacks a directory.
    """
    if data is not None and not legacy_layout and web is None and mix is None:
        return selected_components(data)
    if data is None and legacy_layout and web is not None and mix is not None:
        return components(web, mix)
    msg = (
        "name the inputs one way: --data <root> (the selection) or --legacy-layout --web <dir>"
        " --mix <dir> (today's flat directories, explicitly)"
    )
    raise ValueError(msg)


def _save(tokenizer: Tokenizer, artifact: Path) -> tuple[Path, str, int, bool]:
    """Save the tokenizer with its checksum file.

    Returns the artifact, its hash, the round-trip token count and whether the round trip held.
    """
    tokenizer.save(artifact)
    digest = file_sha256(artifact)
    _ = (artifact.parent / f"{artifact.name}.sha256").write_text(f"{digest}  {artifact.name}\n")
    ids = tokenizer.encode(ROUND_TRIP)
    return artifact, digest, int(ids.shape[0]), tokenizer.decode(ids) == ROUND_TRIP


def _per_component(
    parts: Sequence[Component], limits: list[int], documents: dict[str, int], used: dict[str, int]
) -> dict[str, object]:
    """Each component's files, bytes, share, budget and what the training pass used of it."""
    return {
        c.name: {
            "files": len(c.paths),
            "input_bytes": sum(p.stat().st_size for p in c.paths),
            "share": c.share,
            "budget_bytes": budget,
            "documents_used": documents[c.name],
            "utf8_bytes_used": used[c.name],
        }
        for c, budget in zip(parts, limits, strict=True)
    }


def _eligibility(
    label: str, *, trained_short: list[str], round_trip: bool, agrees: bool
) -> tuple[bool, str]:
    """Whether the trained artifact may carry ``label``, and the classification it gets.

    Eligibility is a property of what was trained on, not of the plan: every component's trained
    bytes met its budget, the round trip held, and the trained counts equal the plan's.
    """
    reasons: list[str] = []
    if trained_short:
        reasons.append(f"quota unmet for {trained_short}")
    if not round_trip:
        reasons.append("round trip failed")
    if not agrees:
        reasons.append("the trained sample differs from the plan")
    if reasons:
        return False, f"candidate; {'; '.join(reasons)}; not freezable"
    return True, label


def freeze(  # noqa: PLR0913
    parts: Sequence[Component],
    *,
    vocab: int,
    split: Split,
    seed: int,
    byte_budget: int,
    out: Path,
    label: str,
    rule: str = DEFAULT_PRE_TOKENIZER,
) -> Path:
    """Train the tokenizer on the mixed sample, save it with its checksum and receipt.

    The split's finalization must have been taken over exactly ``parts``'s files
    (``check_inputs``). Returns the receipt's path; the artifact is
    ``vesper-s-<vocab>-<stamp>.json`` beside it with a ``.sha256`` file.

    Raises:
        ValueError: when the finalization does not cover these files.
    """
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(UTC)
    finalization = split.finalization
    check_inputs(finalization, parts)
    files = file_manifest(parts)
    log.info("manifest: %d files hashed in %.0f s", len(files), time.monotonic() - started)
    documents = {c.name: 0 for c in parts}
    used = {c.name: 0 for c in parts}
    total_docs = 0

    limits = budgets(parts, byte_budget)

    consumed = SequenceDigest()

    def texts() -> Iterator[str]:
        nonlocal total_docs
        for name, text in iter_mixed_texts(parts, seed, split, byte_budget):
            encoded = text.encode("utf-8")
            documents[name] += 1
            used[name] += len(encoded)
            consumed.add(name.encode("utf-8"))
            consumed.add(encoded)
            total_docs += 1
            if total_docs % _PROGRESS_EVERY == 0:
                log.info(
                    "%8.0f s %9d documents %7.3f GiB %s",
                    time.monotonic() - started,
                    total_docs,
                    sum(used.values()) / _GIB,
                    {k: round(v / _GIB, 3) for k, v in used.items()},
                )
            yield text

    # the plan: one read of the sample before the expensive training, so a component that
    # cannot meet its budget is known first and the receipt can say so
    for _ in texts():
        pass
    plan_digest = consumed.hexdigest()
    consumed = SequenceDigest()
    plan = {c.name: {"documents": documents[c.name], "utf8_bytes": used[c.name]} for c in parts}
    short = [c.name for c, limit in zip(parts, limits, strict=True) if used[c.name] < limit]
    log.info("plan: %s; short of budget: %s", plan, short or "none")
    for c in parts:
        documents[c.name] = 0
        used[c.name] = 0
    total_docs = 0
    tokenizer = train_from_texts(texts(), vocab, rule=rule)
    # the inputs are read twice over many minutes; the artifact is only about the files the
    # manifest names if they are still the same files now
    if file_manifest(parts) != files:
        msg = "an input file changed while the sample was read; nothing is saved"
        raise ValueError(msg)
    stamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    artifact, digest, ids, round_trip = _save(tokenizer, out / f"vesper-s-{vocab}-{stamp}.json")
    trained = {c.name: {"documents": documents[c.name], "utf8_bytes": used[c.name]} for c in parts}
    trained_short = [c.name for c, limit in zip(parts, limits, strict=True) if used[c.name] < limit]
    trained_digest = consumed.hexdigest()
    # the bytes the trainer consumed are bound to the plan by a digest over every consumed text
    # with its component, in order: counts alone cannot tell a replaced source of equal length
    eligible, classification = _eligibility(
        label,
        trained_short=trained_short,
        round_trip=round_trip,
        agrees=trained == plan and trained_digest == plan_digest,
    )
    per_component = _per_component(parts, limits, documents, used)
    record: dict[str, object] = {
        "artifact": artifact.name,
        "artifact_sha256": digest,
        "classification": classification,
        "freeze_eligible": eligible,
        "plan": plan,
        "trained": trained,
        "plan_consumed_sha256": plan_digest,
        "trained_consumed_sha256": trained_digest,
        "plan_short_of_budget": short,
        "trained_short_of_budget": trained_short,
        "trained_at_utc": started_at.isoformat(),
        "duration_seconds": round(time.monotonic() - started, 1),
        "environment": environment(),
        "input": {
            "manifest": files,
            "components": per_component,
            "seed": seed,
            "held_out_fraction": split.held_out_fraction,
            "side": "held-out" if split.held_out else "training",
            "byte_budget": byte_budget,
            "documents_used": total_docs,
            "utf8_bytes_used": sum(used.values()),
            "order": ORDER,
            "finalization": {
                "rule": FINALIZE,
                "receipt": str(finalization.receipt),
                "dropped": len(finalization.dropped),
            },
            "mixing": MIXING,
        },
        "tokenizer": {
            "algorithm": TOKENIZER,
            "pre_tokenizer": tokenizer.rule,
            "pre_tokenizer_description": PRE_TOKENIZERS[tokenizer.rule],
            "vocab_requested": vocab,
            "vocab_size_actual": tokenizer.vocab_size,
            "round_trip_check": {
                "text": ROUND_TRIP,
                "ids": ids,
                "identical": round_trip,
            },
        },
    }
    receipt = out / f"{artifact.stem}.receipt.json"
    write_json(receipt, record)
    log.info(
        "done in %.1f s: %d documents, %.3f GiB, vocab %d, %s, sha256 %s, round trip %s",
        time.monotonic() - started,
        total_docs,
        sum(used.values()) / _GIB,
        tokenizer.vocab_size,
        artifact,
        digest,
        round_trip,
    )
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vesper.sample", description=(__doc__ or "").split("\n")[0]
    )
    _ = parser.add_argument("--data", type=Path, default=None, help="the data root (selection)")
    _ = parser.add_argument(
        "--legacy-layout", action="store_true", help="read today's flat --web and --mix dirs"
    )
    _ = parser.add_argument("--web", type=Path, default=None)
    _ = parser.add_argument("--mix", type=Path, default=None)
    _ = parser.add_argument("--out", type=Path, required=True)
    _ = parser.add_argument("--config", type=Path, default=Path("configs/vesper-s.yaml"))
    _ = parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument("--label", default="frozen-candidate")
    rules: list[str] = sorted(PRE_TOKENIZERS)
    _ = parser.add_argument(
        "--pretokenizer",
        default=DEFAULT_PRE_TOKENIZER,
        choices=rules,
        help="the pre-tokenization rule (vesper.tokenizer.PRE_TOKENIZERS); recorded in the receipt",
    )
    _ = parser.add_argument(
        "--finalization",
        type=Path,
        default=None,
        help="an existing finalize-<stamp>.json receipt; without it finalize runs first",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The command line: 0 when the artifact and its receipt were written, 1 otherwise."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = _parser().parse_args(argv)
    cfg = load(cast("Path", args.config))
    try:
        parts = components_for(
            data=cast("Path | None", args.data),
            legacy_layout=cast("bool", args.legacy_layout),
            web=cast("Path | None", args.web),
            mix=cast("Path | None", args.mix),
        )
        given = cast("Path | None", args.finalization)
        out = cast("Path", args.out)
        if given is None:
            started = time.monotonic()
            finalization = finalize(parts, out)
            log.info(
                "finalized in %.0f s: %d dropped, receipt %s",
                time.monotonic() - started,
                len(finalization.dropped),
                finalization.receipt,
            )
        else:
            finalization = load_finalization(given)
        _ = freeze(
            parts,
            vocab=cfg.model.vocab,
            split=Split(finalization, cfg.training.held_out_fraction),
            seed=cast("int", args.seed),
            byte_budget=cast("int", args.budget_bytes),
            out=out,
            label=cast("str", args.label),
            rule=cast("str", args.pretokenizer),
        )
    except (RuntimeError, ValueError):
        log.exception("stopped")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
