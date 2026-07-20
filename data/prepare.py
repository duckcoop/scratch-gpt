"""Download a training corpus and encode it to token ids.

Produces train.bin / val.bin (uint16 token ids) and meta.pkl (tokenizer info).
BPE also writes tokenizer.json (the learned merges).

    python data/prepare.py                          # BPE, 1024-token vocab
    python data/prepare.py --tokenizer char         # Phase 1 baseline
    python data/prepare.py --input TinyStories-train.txt --vocab-size 4096 \
                           --sample-mb 50           # large corpus

To train on your own corpus, drop a plain-text file in data/ and pass --input,
or place it at data/input.txt (the default, downloaded if missing).

Corpora larger than memory are handled by training the tokenizer on a sample
and encoding the rest in streamed chunks.
"""

import argparse
import os
import pickle
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import BPETokenizer, CharTokenizer  # noqa: E402

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))
CHUNK_BYTES = 8 << 20  # 8 MB of text per encode call

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
parser.add_argument("--vocab-size", type=int, default=1024, help="BPE only")
parser.add_argument("--input", default="input.txt", help="corpus filename inside data/")
parser.add_argument(
    "--sample-mb",
    type=float,
    default=0,
    help="train the tokenizer on only the first N MB (0 = whole corpus). "
    "Merge frequencies converge quickly, so a sample is enough for a big corpus.",
)
parser.add_argument("--val-frac", type=float, default=0.1)
args = parser.parse_args()

input_path = os.path.join(HERE, args.input)
if not os.path.exists(input_path):
    if args.input != "input.txt":
        raise SystemExit(f"no such corpus: {input_path}")
    print(f"downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, input_path)

n_bytes = os.path.getsize(input_path)
print(f"corpus: {os.path.basename(input_path)} ({n_bytes / 1e6:,.1f} MB)")


def read_text(limit=None):
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read() if limit is None else f.read(limit)


def iter_chunks():
    """Yield the corpus in chunks that always end on a line boundary."""
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        buf = ""
        while True:
            block = f.read(CHUNK_BYTES)
            if not block:
                break
            buf += block
            cut = buf.rfind("\n")
            if cut == -1:
                continue
            yield buf[: cut + 1]
            buf = buf[cut + 1 :]
        if buf:
            yield buf


# ---- train the tokenizer -------------------------------------------------
sample_chars = int(args.sample_mb * 1e6) if args.sample_mb else None
if args.tokenizer == "bpe":
    sample = read_text(sample_chars)
    label = f"{len(sample) / 1e6:.1f} MB sample" if sample_chars else "whole corpus"
    print(f"training BPE to a {args.vocab_size:,}-token vocab on the {label} ...")
    t0 = time.time()
    tok = BPETokenizer().train(sample, vocab_size=args.vocab_size, verbose=True)
    print(f"  learned {len(tok.merges):,} merges in {time.time() - t0:.1f}s")
    tok.save(os.path.join(HERE, "tokenizer.json"))
    meta = {"tokenizer": "bpe", "vocab_size": tok.vocab_size}
    del sample
else:
    tok = CharTokenizer.train(read_text(sample_chars))
    meta = {"tokenizer": "char", "vocab_size": tok.vocab_size, "stoi": tok.stoi, "itos": tok.itos}

assert tok.vocab_size <= 65536, "uint16 bins cannot hold a vocab this large"

# ---- verify the tokenizer roundtrips before writing anything ------------
probe = read_text(200_000)
assert tok.decode(tok.encode(probe)) == probe, \
    "tokenizer roundtrip failed — refusing to write a corrupt dataset"
del probe

# ---- encode, streaming so a large corpus never lands in memory at once ---
print("encoding corpus ...")
t0 = time.time()
parts, n_tokens, done_bytes = [], 0, 0
for chunk in iter_chunks():
    part = np.array(tok.encode(chunk), dtype=np.uint16)
    parts.append(part)
    n_tokens += len(part)
    done_bytes += len(chunk.encode("utf-8"))
    if n_bytes > 50e6:  # only worth reporting progress on a big corpus
        pct = 100 * done_bytes / n_bytes
        print(f"  {pct:5.1f}%  {n_tokens:,} tokens  ({time.time() - t0:.0f}s)", flush=True)

ids = np.concatenate(parts) if len(parts) > 1 else parts[0]
del parts
print(f"  encoded in {time.time() - t0:.1f}s")

meta["bytes_per_token"] = n_bytes / len(ids)

n = int(len(ids) * (1 - args.val_frac))
ids[:n].tofile(os.path.join(HERE, "train.bin"))
ids[n:].tofile(os.path.join(HERE, "val.bin"))
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"\ntokenizer : {meta['tokenizer']} (vocab {meta['vocab_size']:,})")
print(f"tokens    : {len(ids):,}  ({meta['bytes_per_token']:.2f} bytes/token)")
print(f"train.bin : {n:,} tokens | val.bin: {len(ids) - n:,} tokens")
