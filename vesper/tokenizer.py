"""Tokenizer (spec 1): byte-level BPE, ``V = 32,768``, frozen after training.

A thin, typed wrapper around the ``tokenizers`` library so that nothing else in the project
touches that library directly (spec: "nothing else in the project may depend on another
tokenizer"). Pre-tokenization follows a named rule (``PRE_TOKENIZERS``), recorded in the saved
file and read back by ``Tokenizer.rule``: ``words-v1``, the default since the project owner's
decision of 2026-10-15, splits into runs of letters and digits, runs of other non-space characters
and whitespace runs, a single leading space attached to the run after it (``foo(bar); baz`` gives
``foo``, ``(``, ``bar``, ``);``, `` baz``; ``foo123bar`` stays one piece); ``spaces-v1``, the
original rule, splits on spaces only with the space attached to the piece that follows, and
stays selectable so the choice can be reversed (docs/reviews/2026-10-15-pretokenization.md
has the memory and compression measurements behind the switch). Every byte of the UTF-8 text
is a token of the 256-byte alphabet, the spec's "byte fallback", so any valid Unicode string
round-trips exactly.

Domain: ``str`` values that are valid Unicode, i.e. UTF-8 encodable; a string with an unpaired
surrogate raises ``ValueError``. The spec's "any byte string" is narrowed to text here: the
corpus is text and the wrapper takes ``str``; raw bytes would need a byte-level API.

``vocab`` is an upper bound: BPE stops when no merge is left, so a small corpus yields fewer
ids than requested and ``vocab_size`` reports the actual count. Ids are ``int32`` on the way
out, as storage; the model widens them at its boundary (``check_ids``), and ``decode`` holds
its input to the same contract.
"""

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import cast

import torch
from tokenizers import Regex, decoders, models, pre_tokenizers, trainers
from tokenizers import Tokenizer as _HfTokenizer
from torch import Tensor

from vesper.contracts import check_ids

_BYTE_ALPHABET = 256


def _checked_text(text: str) -> str:
    """Return ``text`` when it is valid Unicode; an unpaired surrogate raises ``ValueError``."""
    try:
        _ = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        msg = "text must be valid Unicode (UTF-8 encodable); an unpaired surrogate is not"
        raise ValueError(msg) from exc
    return text


def _checked_texts(texts: Iterable[str]) -> Iterator[str]:
    for text in texts:
        yield _checked_text(text)


class Tokenizer:
    """``encode(str) -> int32[n]`` and ``decode(int32[n]) -> str``, exact round trip."""

    def __init__(self, inner: _HfTokenizer) -> None:
        """Wrap a trained byte-level BPE; use ``train_from_texts`` or ``load`` instead."""
        self._inner: _HfTokenizer = inner

    @property
    def vocab_size(self) -> int:
        """Number of ids actually in the vocabulary, the 256 byte tokens included."""
        return int(self._inner.get_vocab_size(with_added_tokens=True))

    def encode(self, text: str) -> Tensor:
        """Text to ``int32[n]``; the empty string gives an empty tensor.

        Raises:
            ValueError: if ``text`` is not valid Unicode.
        """
        ids: list[int] = self._inner.encode(_checked_text(text), add_special_tokens=False).ids
        return torch.tensor(ids, dtype=torch.int32) if ids else torch.zeros(0, dtype=torch.int32)

    def decode(self, ids: Tensor) -> str:
        """``int32|int64[n]`` back to the exact text.

        Raises:
            ContractError: (a ``ValueError``) on a rank other than 1, a dtype other than int32 or
                int64, or an id outside the vocabulary.
        """
        checked = check_ids("ids", ids, ("n",), {}, self.vocab_size)
        return self._inner.decode(
            [int(t.item()) for t in checked.unbind()], skip_special_tokens=False
        )

    def pre_tokens(self, text: str) -> list[str]:
        """The pieces BPE merges within, as substrings of ``text``, under this tokenizer's rule.

        Under ``spaces-v1`` a space belongs to the piece after it (``"foo(bar); baz"`` gives
        ``["foo(bar);", " baz"]``); under ``words-v1`` letter-and-digit runs and punctuation runs
        are apart (``["foo", "(", "bar", ");", " baz"]``). The pieces concatenate to the text.

        Raises:
            ValueError: if ``text`` is not valid Unicode, or the tokenizer has no pre-tokenizer.
        """
        pre = self._inner.pre_tokenizer
        if pre is None:
            msg = "this tokenizer has no pre-tokenizer"
            raise ValueError(msg)
        checked = _checked_text(text)
        return [checked[start:end] for _, (start, end) in pre.pre_tokenize_str(checked)]

    @property
    def rule(self) -> str:
        """The name of this tokenizer's pre-tokenization rule, read from its own configuration.

        The saved file carries the rule's configuration, not a label; the name is found by
        comparing that configuration with the known rules, so a receipt names what the file does.
        ``"unknown"`` for a configuration none of the rules produces.
        """
        saved = cast("dict[str, object]", json.loads(self._inner.to_str()))["pre_tokenizer"]
        for name in PRE_TOKENIZERS:
            probe = _HfTokenizer(models.BPE())
            probe.pre_tokenizer = _pre_tokenizer(name)
            if cast("dict[str, object]", json.loads(probe.to_str()))["pre_tokenizer"] == saved:
                return name
        return "unknown"

    def save(self, path: Path) -> None:
        """Write the tokenizer as JSON; ``load`` restores it exactly."""
        self._inner.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        """Read a tokenizer written by ``save``."""
        return cls(_HfTokenizer.from_file(str(path)))

    @classmethod
    def parse(cls, data: bytes) -> "Tokenizer":
        """The tokenizer of the bytes of a file written by ``save``; ``load`` of the same file.

        For a tokenizer whose file hash is checked first (``keys.verified_bytes``): the bytes
        that were checked are the bytes parsed, never the path read again.
        """
        return cls(_HfTokenizer.from_buffer(data))


PRE_TOKENIZERS: dict[str, str] = {
    "spaces-v1": (
        "split on ' ' with the space merged into the following piece, then ByteLevel without the"
        " GPT-2 regex; the original rule. Its trainer's word table grows with every distinct"
        " punctuation-attached piece: 17.6 M on 2.5 GiB of the mix, out of memory at 10 GiB on a"
        " 64 GB machine (docs/reviews/2026-10-15-pretokenization.md)"
    ),
    "words-v1": (
        "split into runs of letters and digits, runs of other non-space characters, and"
        " whitespace runs, a single leading space attached to the following run (regex"
        " ' ?\\w+| ?[^\\w\\s]+|\\s+(?!\\S)|\\s+', pieces isolated), then ByteLevel without GPT-2's"
        " regex; three to four times fewer distinct pieces than spaces-v1, so the 10 GiB sample"
        " trains on a 64 GB machine; a merge never crosses a letter-to-punctuation boundary"
    ),
}
DEFAULT_PRE_TOKENIZER = "words-v1"  # the owner's choice, 2026-10-15; spaces-v1 stays selectable


def _pre_tokenizer(rule: str) -> pre_tokenizers.PreTokenizer:
    """The configured pre-tokenizer for a named rule.

    Raises:
        ValueError: for a name outside ``PRE_TOKENIZERS``.
    """
    if rule == "spaces-v1":
        first: pre_tokenizers.PreTokenizer = pre_tokenizers.Split(" ", "merged_with_next")
    elif rule == "words-v1":
        # GPT-2's whitespace handling: a run of spaces leaves its last space to the following
        # word, so "a  b" gives "a", " ", " b" and " b" is the same piece wherever it stands
        first = pre_tokenizers.Split(Regex(r" ?\w+| ?[^\w\s]+|\s+(?!\S)|\s+"), "isolated")
    else:
        msg = f"pre-tokenizer rule must be one of {sorted(PRE_TOKENIZERS)}, got {rule!r}"
        raise ValueError(msg)
    return pre_tokenizers.Sequence(
        [first, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )


def train_from_texts(
    texts: Iterable[str], vocab: int, *, rule: str = DEFAULT_PRE_TOKENIZER
) -> Tokenizer:
    """Train a byte-level BPE with at most ``vocab`` ids on an iterable of texts.

    Deterministic for the same texts in the same order. ``vocab`` is an upper bound that a
    small corpus undershoots (BPE stops when no merge is left); it must leave room for the 256
    byte tokens. ``rule`` names the pre-tokenization (``PRE_TOKENIZERS``); the trained file
    carries its configuration and ``Tokenizer.rule`` reads the name back.

    Raises:
        ValueError: if ``vocab`` is not larger than 256, the rule is unknown, or a text is not
            valid Unicode.
    """
    if vocab <= _BYTE_ALPHABET:
        msg = f"vocab must exceed the 256 byte tokens, got {vocab}"
        raise ValueError(msg)
    inner = _HfTokenizer(models.BPE())
    inner.pre_tokenizer = _pre_tokenizer(rule)
    inner.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=False,
    )
    inner.train_from_iterator(_checked_texts(texts), trainer=trainer)
    return Tokenizer(inner)


def train_from_files(
    paths: list[Path], vocab: int, *, rule: str = DEFAULT_PRE_TOKENIZER
) -> Tokenizer:
    """Train on text files, one document per line, the way the 10 GB sample is dumped."""

    def lines() -> Iterable[str]:
        for p in paths:
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    yield line.rstrip("\n")

    return train_from_texts(lines(), vocab, rule=rule)
