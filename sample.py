"""Generate text from a trained checkpoint.

Usage:  python sample.py --prompt "ROMEO:" --tokens 500
"""

import argparse
import os
import pickle

import torch

from model import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="\n", help="text to continue from")
parser.add_argument("--tokens", type=int, default=500, help="tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--ckpt", default=os.path.join("out", "ckpt.pt"))
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location=device)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().to(device)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi, itos = meta["stoi"], meta["itos"]

ids = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=device)
out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
print("".join(itos[i] for i in out[0].tolist()))
