"""Train the GPT on the prepared corpus.

Run data/prepare.py first, then:  python train.py
Checkpoints go to out/ckpt.pt whenever validation loss improves.
"""

import math
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

# ---------------- config (edit here) ----------------
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 250
eval_iters = 100
learning_rate = 3e-4
warmup_iters = 100
min_lr = 3e-5
weight_decay = 0.1
grad_clip = 1.0
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
out_dir = "out"
seed = 1337
# ----------------------------------------------------

torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"
# bf16 autocast is a free ~2x speedup on Ampere+ GPUs
autocast = (
    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cuda" and torch.cuda.is_bf16_supported()
    else torch.autocast(device_type="cpu", enabled=False)
)
print(f"device: {device}")

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
vocab_size = meta["vocab_size"]

train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(it):
    """Linear warmup, then cosine decay to min_lr."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = {}
    for split in ("train", "val"):
        acc = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with autocast:
                _, loss = model(x, y)
            acc[k] = loss.item()
        losses[split] = acc.mean().item()
    model.train()
    return losses


def main():
    model_args = dict(
        block_size=block_size, vocab_size=vocab_size,
        n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
    )
    model = GPT(GPTConfig(**model_args)).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay
    )

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for it in range(max_iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(it)

        if it % eval_interval == 0:
            losses = estimate_loss(model)
            dt = time.time() - t0
            print(
                f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                f"| lr {get_lr(it):.2e} | {dt:.0f}s elapsed"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_args": model_args,
                        "iter": it,
                        "best_val": best_val,
                    },
                    os.path.join(out_dir, "ckpt.pt"),
                )
                print(f"        saved checkpoint (val {best_val:.4f})")

        if it == max_iters:
            break

        x, y = get_batch("train")
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    print(f"done. best val loss {best_val:.4f} | checkpoint: {out_dir}/ckpt.pt")


if __name__ == "__main__":
    main()
