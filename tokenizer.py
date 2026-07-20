"""A byte-level Byte-Pair Encoding tokenizer, written from scratch.

No `tiktoken`, no `tokenizers` — the merge training, encoding and decoding
are all here. Byte-level means the base vocabulary is the 256 possible bytes,
so *any* UTF-8 text encodes with zero unknown tokens; BPE then learns to merge
frequent byte pairs into single tokens (whole words and word-chunks), which is
what lets a small model spend its context on meaning instead of on letters.

Train:   tok = BPETokenizer(); tok.train(text, vocab_size=1024); tok.save(path)
Use:     ids = tok.encode("hello");  text = tok.decode(ids)
"""

import json
import os
import pickle
from collections import Counter


def _get_pair_counts(ids_seq, counts=None, weights=None):
    """Count adjacent id pairs across token sequences, optionally frequency-weighted."""
    counts = Counter() if counts is None else counts
    for i, ids in enumerate(ids_seq):
        w = 1 if weights is None else weights[i]
        for pair in zip(ids, ids[1:]):
            counts[pair] += w
    return counts


def _merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    out = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self):
        # merges: (id_a, id_b) -> new_id, in the order they were learned
        self.merges = {}
        # vocab: id -> bytes  (the raw byte string each id expands to)
        self.vocab = {i: bytes([i]) for i in range(256)}
        # encoding the same word repeatedly is the hot path; memoize it
        self._cache = {}

    def train(self, text, vocab_size, verbose=False):
        """Learn merges from `text` until the vocab reaches `vocab_size`."""
        assert vocab_size >= 256, "vocab_size must leave room for the 256 byte tokens"
        num_merges = vocab_size - 256

        # Pre-split on whitespace so merges never span across a space boundary.
        # Each word keeps a leading-space marker by being encoded with its space,
        # which is how GPT-2-style tokenizers represent word boundaries.
        #
        # Work on *unique* words weighted by frequency, not on every occurrence:
        # a corpus with 200k words typically has only ~25k distinct ones, and
        # merging is identical for every copy of the same word.
        word_freq = Counter()
        for i, w in enumerate(text.split(" ")):
            piece = w if i == 0 else " " + w
            if piece:
                word_freq[piece] += 1

        chunks = [list(w.encode("utf-8")) for w in word_freq]
        weights = list(word_freq.values())

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._cache = {}

        for m in range(num_merges):
            counts = _get_pair_counts(chunks, weights=weights)
            if not counts:
                break
            pair = max(counts, key=counts.get)
            if counts[pair] < 2:
                break  # nothing repeats anymore; stop early
            new_id = 256 + m
            # only sequences actually containing the pair need rewriting
            for i, ids in enumerate(chunks):
                if pair[0] in ids:
                    chunks[i] = _merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (m < 5 or (m + 1) % 100 == 0):
                merged = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"  merge {m + 1}/{num_merges}: {pair} -> {new_id} ({merged!r}) x{counts[pair]}")

        return self

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _encode_piece(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = list(piece.encode("utf-8"))
        # Apply merges greedily by learned order: repeatedly merge the pair
        # with the lowest merge index present, until none apply.
        while len(ids) >= 2:
            counts = _get_pair_counts([ids])
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = _merge(ids, pair, self.merges[pair])
        self._cache[piece] = ids
        return ids

    def encode(self, text):
        """Encode a string into a list of token ids."""
        out = []
        for i, w in enumerate(text.split(" ")):
            piece = w if i == 0 else " " + w
            if piece:
                out.extend(self._encode_piece(piece))
        return out

    def decode(self, ids):
        """Decode a list of token ids back into a string."""
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        # merge keys are tuples -> serialize as "a,b" strings for JSON
        payload = {
            "merges": {f"{a},{b}": nid for (a, b), nid in self.merges.items()},
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        tok.merges = {}
        tok._cache = {}
        for key, nid in payload["merges"].items():
            a, b = key.split(",")
            tok.merges[(int(a), int(b))] = nid
        # rebuild vocab by replaying merges in id order
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), nid in sorted(tok.merges.items(), key=lambda kv: kv[1]):
            tok.vocab[nid] = tok.vocab[a] + tok.vocab[b]
        return tok


class CharTokenizer:
    """The Phase 1 baseline: one token per character. Kept for comparison."""

    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @classmethod
    def train(cls, text):
        chars = sorted(set(text))
        return cls({c: i for i, c in enumerate(chars)}, {i: c for i, c in enumerate(chars)})

    @property
    def vocab_size(self):
        return len(self.stoi)

    def encode(self, text):
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_codec(data_dir):
    """Load whichever tokenizer data/prepare.py last built."""
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    if meta["tokenizer"] == "bpe":
        return BPETokenizer.load(os.path.join(data_dir, "tokenizer.json")), meta
    return CharTokenizer(meta["stoi"], meta["itos"]), meta
