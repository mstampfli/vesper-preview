"""How a generated answer is read and scored in the instruction-tuning test.

``answer_of`` finds where a generation ends (the terminator marker in the decoded text);
``score_answer`` applies a panel item's checks; ``load_panel`` parses a panel file and refuses
it unless every item's own counterexamples score as declared. The panel's items, rules and
examples were fixed before any model output existed (docs/reviews/2026-10-25-instruction-
tuning.md); ``SCORING`` names this scorer's version and is bound into every run, so a change
here is a new version, never a silent change of what a reported pass count means.

Version 3. Version 1 accepted "the red team" for a colour, "the mango" for one word, "HELLO."
for nothing else, three stones as fruits and "banana" as an explanation (maintainer B's review of
e178dab); version 2's keyword rule for an explanation still accepted "rain" (maintainer B's review
of eb4d9f9), so explanations are no longer scored at all, and the kinds only a free-text item used
are gone. The kinds: ``exact`` and ``one_of`` compare the stripped, case-folded answer with ONE
trailing period dropped (a sentence end, not extra text); ``verbatim`` compares the stripped
answer exactly (no period, no case change); ``lines_in`` wants exactly ``count`` non-empty
lines, each a distinct member of ``allowed`` (stripped, case-folded, nothing else on the line);
``regex`` a full match on the stripped answer.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from vesper.instruct_data import MARKER

SCORING = "instruct-scoring-v3"


def answer_of(text: str) -> str | None:
    """The answer in a generated ``text``, or None while no ``MARKER`` has been generated.

    Everything before the first marker with one line feed dropped (``instruct_data.TERMINATOR``).
    The text is the decoding of every generated id together, so a marker split over ids, or
    merged with what surrounds it, is found the same way.
    """
    if MARKER not in text:
        return None
    return text[: text.index(MARKER)].removesuffix("\n")


def _folded(answer: str) -> str:
    return answer.strip().removesuffix(".").strip().casefold()


def _passes(kind: object, value: object, answer: str) -> bool:
    if kind == "exact":
        return _folded(answer) == str(value).casefold()
    if kind == "verbatim":
        return answer.strip() == str(value)
    if kind == "one_of":
        return _folded(answer) in {str(o).casefold() for o in cast("list[object]", value)}
    if kind == "lines_in":
        spec = cast("dict[str, object]", value)
        allowed = {str(a).casefold() for a in cast("list[object]", spec["allowed"])}
        lines = [line.strip().casefold() for line in answer.split("\n") if line.strip()]
        return (
            len(lines) == cast("int", spec["count"])
            and len(set(lines)) == len(lines)
            and all(line in allowed for line in lines)
        )
    if kind == "regex":
        return re.fullmatch(str(value), answer.strip()) is not None
    msg = f"unknown check kind {kind!r}"
    raise ValueError(msg)


def score_answer(checks: Sequence[dict[str, object]], answer: str) -> bool:
    """Whether ``answer`` passes every check of a panel item (the kinds: module docstring).

    Raises:
        ValueError: on an unknown check kind.
    """
    return all(_passes(check["kind"], check["value"], answer) for check in checks)


@dataclass(frozen=True)
class Panel:
    """A parsed, self-checked panel: scored items and illustrations."""

    version: str
    scored: tuple[dict[str, object], ...]
    illustrations: tuple[dict[str, object], ...]


def load_panel(data: bytes) -> Panel:
    """The panel in ``data`` (JSON), refused unless every scored item is consistent.

    Each scored item has an id, an instruction, a context, its checks and at least one ``pass``
    and one ``fail`` example; every pass example must score True and every fail example False
    under ``score_answer``, so the rubric is shown to separate what the prompt asks from what
    it does not before any model answers. Ids are unique.

    Raises:
        ValueError: on malformed JSON, a missing field, an unknown kind, a duplicate id or an
            example that scores against its declaration.
    """
    raw = cast("object", json.loads(data))
    if not isinstance(raw, dict):
        msg = "panel: not an object"
        raise ValueError(msg)  # noqa: TRY004  # a malformed file is a value error
    try:
        return _checked(cast("dict[str, object]", raw))
    except (KeyError, TypeError) as exc:
        msg = f"panel: malformed ({exc!r})"
        raise ValueError(msg) from exc


def _checked(fields: dict[str, object]) -> Panel:
    scored = cast("list[dict[str, object]]", fields["scored"])
    illustrations = cast("list[dict[str, object]]", fields["illustrations"])
    seen: set[str] = set()
    for item in scored:
        name = str(item["id"])
        if name in seen:
            msg = f"panel: duplicate id {name}"
            raise ValueError(msg)
        seen.add(name)
        _ = str(item["instruction"]), str(item["context"])
        checks = cast("list[dict[str, object]]", item["checks"])
        if not checks:
            msg = f"panel {name}: checks missing"
            raise ValueError(msg)
        # an unknown kind raises in score_answer: a pass example must pass every check, so
        # scoring it reaches each one (round-7 mutant m54: a second kind list beside _passes
        # changed only the message)
        good = cast("list[str]", item["pass"])
        bad = cast("list[str]", item["fail"])
        if not good or not bad:
            msg = f"panel {name}: at least one pass and one fail example"
            raise ValueError(msg)
        for example in good:
            if not score_answer(checks, example):
                msg = f"panel {name}: pass example {example!r} fails its checks"
                raise ValueError(msg)
        for example in bad:
            if score_answer(checks, example):
                msg = f"panel {name}: fail example {example!r} passes its checks"
                raise ValueError(msg)
    for item in illustrations:
        _ = str(item["instruction"]), str(item["context"])
    return Panel(str(fields["version"]), tuple(scored), tuple(illustrations))
