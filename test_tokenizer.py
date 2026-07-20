"""Tests for the from-scratch BPE tokenizer.  Run: python -m pytest -q"""

import pytest

from tokenizer import SPLIT_PATTERN, BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick and the dog is lazy. "
) * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer().train(CORPUS, vocab_size=350)


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "the quick brown fox",
        "",
        " ",
        "   leading and trailing   ",
        "double  spaces",
        "\n\nnewlines\tand\ttabs\n",
        "unicode: café, naïve, 日本語",
        "emoji: 🔥🚀",
        "MiXeD CaSe 12345 !@#$%",
    ],
)
def test_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_on_training_corpus(tok):
    assert tok.decode(tok.encode(CORPUS)) == CORPUS


@pytest.mark.parametrize(
    "text",
    [
        "hello world",
        "",
        "   ",
        "don't can't we've I'll",
        "line one\nline two\n\n\ttabbed",
        "punctuation!!! ...yes? (parens) [brackets]",
        "numbers 42 and 3.14159 and 007",
        "unicode café 日本語 🔥 mixed",
        "ROMEO:\nWhat, ho! 'tis I.",
    ],
)
def test_presplit_is_lossless(text):
    """Every character must land in exactly one piece — a dropped character here
    would silently corrupt every encode/decode roundtrip."""
    assert "".join(SPLIT_PATTERN.findall(text)) == text


def test_presplit_separates_punctuation_and_newlines():
    pieces = SPLIT_PATTERN.findall("dog.\nThe cat")
    assert " cat" in pieces
    # the word must not be glued to the punctuation or the newline
    assert not any("dog." in p for p in pieces)


def test_untrained_tokenizer_is_still_lossless():
    """With zero merges it degenerates to raw bytes — must still roundtrip."""
    raw = BPETokenizer()
    text = "anything at all 日本語 🔥"
    assert raw.decode(raw.encode(text)) == text


def test_compression_beats_raw_bytes(tok):
    """The whole point of BPE: fewer tokens than bytes."""
    n_bytes = len(CORPUS.encode("utf-8"))
    n_tokens = len(tok.encode(CORPUS))
    assert n_tokens < n_bytes / 2


def test_vocab_size_respected():
    t = BPETokenizer().train(CORPUS, vocab_size=300)
    assert t.vocab_size <= 300
    assert max(t.encode(CORPUS)) < t.vocab_size


def test_ids_are_in_range(tok):
    ids = tok.encode("the quick brown fox 🔥")
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    text = "the quick brown fox jumps"
    assert loaded.encode(text) == tok.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
    assert loaded.vocab_size == tok.vocab_size


def test_learns_whole_words(tok):
    """A frequent word in the corpus should collapse to very few tokens."""
    assert len(tok.encode(" the")) <= 2


def test_cache_does_not_change_results(tok):
    text = "the quick brown fox the quick brown fox"
    first = tok.encode(text)
    second = tok.encode(text)  # served from cache
    assert first == second
    assert tok.decode(second) == text
