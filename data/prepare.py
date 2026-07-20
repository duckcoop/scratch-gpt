"""Download a training corpus and encode it at character level.

Produces train.bin / val.bin (uint16 token ids) and meta.pkl (the vocab).
To train on your own corpus, drop any plain-text file at data/input.txt
before running this — the download is skipped if the file exists.
"""

import os
import pickle
import urllib.request

import numpy as np

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
HERE = os.path.dirname(os.path.abspath(__file__))

input_path = os.path.join(HERE, "input.txt")
if not os.path.exists(input_path):
    print(f"downloading {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, input_path)

with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()
print(f"corpus: {len(text):,} characters")

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
print(f"vocab: {len(chars)} unique characters")

ids = np.array([stoi[c] for c in text], dtype=np.uint16)
n = int(len(ids) * 0.9)
ids[:n].tofile(os.path.join(HERE, "train.bin"))
ids[n:].tofile(os.path.join(HERE, "val.bin"))

with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)

print(f"train.bin: {n:,} tokens | val.bin: {len(ids) - n:,} tokens")
