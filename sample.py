"""Generate text from a trained checkpoint.

Usage:  python sample.py --prompt "ROMEO:" --tokens 500
"""

import argparse
import os

import torch

from model import GPT, GPTConfig
from tokenizer import load_codec

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="\n", help="text to continue from")
parser.add_argument("--tokens", type=int, default=500, help="tokens to generate")
parser.add_argument("--temperature", type=float, default=0.8)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--ckpt", default=os.path.join("out", "ckpt.pt"))
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict(ckpt["model"])
model.eval().to(device)

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
codec, meta = load_codec(data_dir)

trained_with = ckpt.get("tokenizer")
if trained_with is not None and trained_with != meta["tokenizer"]:
    raise SystemExit(
        f"checkpoint was trained with the '{trained_with}' tokenizer but data/ currently "
        f"holds '{meta['tokenizer']}'. Re-run: python data/prepare.py --tokenizer {trained_with}"
    )

ids = torch.tensor([codec.encode(args.prompt)], dtype=torch.long, device=device)
if ids.numel() == 0:  # empty prompt — seed with a newline
    ids = torch.tensor([codec.encode("\n")], dtype=torch.long, device=device)

out = model.generate(ids, args.tokens, temperature=args.temperature, top_k=args.top_k)
print(codec.decode(out[0].tolist()))
