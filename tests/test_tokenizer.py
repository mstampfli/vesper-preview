from pathlib import Path

import pytest
import torch

from vesper.contracts import ContractError
from vesper.tokenizer import Tokenizer, train_from_texts

CORPUS = [
    "The delta rule erases what is stored under the key, then writes the value.",
    "Documents are kept whole and streamed in order; no shuffling within a document.",
    "Zürich, Genève and Bellinzona: three cities, three languages, one timetable.",
    "def forward(self, x):\n    return self.w2 @ torch.relu(self.w1 @ x) ** 2\n",
    "Numbers like 4096, 32768 and 0.01 must survive, as must   runs of spaces and\ttabs.",
    "Emoji and symbols: 🚀 ∑ √ 日本語 текст ✓",
] * 20
V = 400


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_from_texts(CORPUS, vocab=V)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "plain ascii text",
        " leading space and trailing space ",
        "line one\nline two\n\n\tindented",
        "Zürich 日本語 🚀 ∑ ✓ текст",
        "unseen words: quokka xylophone 0xDEADBEEF",
        "foo123bar baz",
        "\N{NO-BREAK SPACE}non-breaking\N{EM SPACE}spaces\N{ZERO WIDTH SPACE}",
    ],
)
def test_round_trip_is_exact(tok: Tokenizer, text: str) -> None:
    ids = tok.encode(text)
    assert ids.dtype == torch.int32
    assert ids.dim() == 1
    assert tok.decode(ids) == text


def test_vocabulary_size_and_id_range(tok: Tokenizer) -> None:
    assert 256 < tok.vocab_size <= V
    ids = torch.cat([tok.encode(t) for t in CORPUS[:6]])
    assert ids.min().item() >= 0
    assert ids.max().item() < tok.vocab_size


def test_vocab_is_an_upper_bound_a_small_corpus_undershoots() -> None:
    small = train_from_texts(["aa bb cc 123 test"] * 5, vocab=300)
    assert 256 < small.vocab_size < 300
    assert small.decode(small.encode("aa bb")) == "aa bb"


def test_the_spaces_rule_splits_on_spaces_only() -> None:
    tok = train_from_texts(CORPUS, vocab=V, rule="spaces-v1")
    assert tok.rule == "spaces-v1"
    assert tok.pre_tokens("foo123bar baz") == ["foo123bar", " baz"]
    assert tok.pre_tokens("a  b") == ["a", " ", " b"]
    assert tok.pre_tokens("tabs\tand\nnewlines stay") == ["tabs\tand\nnewlines", " stay"]
    assert tok.pre_tokens("foo(bar); baz") == ["foo(bar);", " baz"]
    assert tok.pre_tokens("") == []
    # the pieces are the text, nothing added or lost
    for text in ("x 1.5e-3 y", " lead", "trail "):
        assert "".join(tok.pre_tokens(text)) == text


def test_the_words_rule_keeps_letters_and_punctuation_apart(tok: Tokenizer) -> None:
    # the default (the project owner's decision of 2026-10-15): a merge never crosses a
    # letter-to-punctuation boundary, so the trainer's word table stays small enough for the 10 GiB
    # sample
    assert tok.rule == "words-v1"
    assert tok.pre_tokens("foo(bar); baz") == ["foo", "(", "bar", ");", " baz"]
    assert tok.pre_tokens("foo123bar baz") == ["foo123bar", " baz"]
    assert tok.pre_tokens("a  b") == ["a", " ", " b"]
    assert tok.pre_tokens("end.\nNext") == ["end", ".", "\n", "Next"]
    assert tok.pre_tokens("x 1.5e-3 y") == ["x", " 1", ".", "5e", "-", "3", " y"]
    assert tok.pre_tokens("") == []
    for text in ("x 1.5e-3 y", " lead", "trail ", "Zürich 日本語 🚀", "a\t\tb"):
        assert "".join(tok.pre_tokens(text)) == text
    with pytest.raises(ValueError, match="rule must be one of"):
        _ = train_from_texts(CORPUS, vocab=V, rule="gpt2")


def test_the_rule_survives_save_and_load_for_both_rules(tmp_path: Path) -> None:
    for rule in ("spaces-v1", "words-v1"):
        tok = train_from_texts(CORPUS, vocab=V, rule=rule)
        path = tmp_path / f"{rule}.json"
        tok.save(path)
        again = Tokenizer.load(path)
        assert again.rule == rule
        for text in ("foo(bar); baz", "Zürich 日本語 🚀 ∑", ""):
            assert again.decode(again.encode(text)) == text
            assert torch.equal(again.encode(text), tok.encode(text))


def test_training_is_deterministic() -> None:
    a = train_from_texts(CORPUS, vocab=V)
    b = train_from_texts(CORPUS, vocab=V)
    for text in CORPUS[:6]:
        assert torch.equal(a.encode(text), b.encode(text))


def test_save_and_load_give_identical_encodings(tok: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    parsed = Tokenizer.parse(path.read_bytes())  # the bytes of the same file
    for again in (loaded, parsed):
        assert again.vocab_size == tok.vocab_size
        assert again.rule == tok.rule
        for text in CORPUS[:6]:
            assert torch.equal(again.encode(text), tok.encode(text))
        assert again.pre_tokens("foo123bar baz") == ["foo123bar", " baz"]


def test_decode_accepts_int64_and_rejects_other_dtypes(tok: Tokenizer) -> None:
    ids = tok.encode("plain ascii text")
    assert tok.decode(ids.long()) == "plain ascii text"
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = tok.decode(ids.to(torch.float32) + 0.9)
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = tok.decode(torch.ones(2, dtype=torch.bool))
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = tok.decode(torch.full((2,), float("nan")))


def test_decode_rejects_bad_ids(tok: Tokenizer) -> None:
    with pytest.raises(ValueError, match="vocab"):
        _ = tok.decode(torch.full((3,), tok.vocab_size, dtype=torch.int32))
    with pytest.raises(ValueError, match="rank"):
        _ = tok.decode(torch.zeros(1, 3, dtype=torch.int32))


def test_text_must_be_valid_unicode(tok: Tokenizer) -> None:
    with pytest.raises(ValueError, match="surrogate"):
        _ = tok.encode("\ud800")
    with pytest.raises(ValueError, match="surrogate"):
        _ = tok.pre_tokens("ok \udfff")
    with pytest.raises(ValueError, match="surrogate"):
        _ = train_from_texts(["fine", "\ud800"], vocab=300)


def test_vocab_must_cover_the_byte_alphabet() -> None:
    with pytest.raises(ValueError, match="256"):
        _ = train_from_texts(CORPUS, vocab=200)
