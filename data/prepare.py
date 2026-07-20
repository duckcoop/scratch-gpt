"""Download a training corpus and encode it to token ids.

Produces train.bin / val.bin (uint16 token ids) and meta.pkl (tokenizer info).
BPE also writes tokenizer.json (the learned merges).

    python data/prepare.py                          # BPE, 1024-token vocab
    python data/prepare.py --tokenizer char         # Phase 1 baseline
    python data/prepare.py --vocab-size 4096

To train on your own corpus, drop any plain-text file at data/input.txt
before running this — the download is skipped if the file exists.
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

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer", choices=["bpe", "char"], default="bpe")
parser.add_argument("--vocab-size", type=int, default=1024, help="BPE only")
args = parser.parse_args()

input_path = os.path.join(HERE, "input.txt")
if not os.path.exists(input_path):
    print(f"downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, input_path)

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()
n_bytes = len(text.encode("utf-8"))
print(f"corpus: {len(text):,} characters ({n_bytes:,} bytes)")

if args.tokenizer == "bpe":
    print(f"training BPE to a {args.vocab_size}-token vocab ...")
    t0 = time.time()
    tok = BPETokenizer().train(text, vocab_size=args.vocab_size, verbose=True)
    print(f"  learned {len(tok.merges):,} merges in {time.time() - t0:.1f}s")
    tok.save(os.path.join(HERE, "tokenizer.json"))
    meta = {"tokenizer": "bpe", "vocab_size": tok.vocab_size}
else:
    tok = CharTokenizer.train(text)
    meta = {"tokenizer": "char", "vocab_size": tok.vocab_size, "stoi": tok.stoi, "itos": tok.itos}

print("encoding corpus ...")
t0 = time.time()
ids = np.array(tok.encode(text), dtype=np.uint16)
print(f"  encoded in {time.time() - t0:.1f}s")

assert tok.vocab_size <= 65536, "uint16 bins cannot hold a vocab this large"
assert tok.decode(ids[:5000].tolist()) == text[: len(tok.decode(ids[:5000].tolist()))], \
    "tokenizer roundtrip failed — refusing to write a corrupt dataset"

meta["bytes_per_token"] = n_bytes / len(ids)

n = int(len(ids) * 0.9)
ids[:n].tofile(os.path.join(HERE, "train.bin"))
ids[n:].tofile(os.path.join(HERE, "val.bin"))
with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"\ntokenizer : {meta['tokenizer']} (vocab {meta['vocab_size']:,})")
print(f"tokens    : {len(ids):,}  ({n_bytes / len(ids):.2f} bytes/token)")
print(f"train.bin : {n:,} tokens | val.bin: {len(ids) - n:,} tokens")
