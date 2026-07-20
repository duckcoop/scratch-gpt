# scratch-gpt

A GPT language model trained **from scratch** — no pretrained weights, no
fine-tuning, no safety layer. Every parameter starts as random noise and
learns only from the corpus in `data/`. What you train on is what you get.

Built in plain PyTorch, ~300 lines of model code. Runs on a single consumer
GPU (developed on an RTX 4070 SUPER) or CPU.

## Quickstart

```bash
pip install torch numpy          # CUDA build recommended: pytorch.org/get-started
python data/prepare.py           # download + tokenize the corpus
python train.py                  # ~10 min on a modern GPU
python sample.py --prompt "ROMEO:" --tokens 500
```

## What's in here

| File | What it does |
| --- | --- |
| `model.py` | The transformer: causal self-attention, pre-norm blocks, weight-tied head |
| `data/prepare.py` | Downloads the corpus, builds a char-level vocab, writes `train.bin`/`val.bin` |
| `train.py` | Training loop: AdamW, cosine LR schedule with warmup, bf16 autocast, grad clipping, best-val checkpointing |
| `sample.py` | Autoregressive generation with temperature and top-k sampling |

## Training on your own data

Drop any plain-text file at `data/input.txt` before running `prepare.py` —
the downloader is skipped if the file exists. The model learns whatever
distribution you feed it; there is no filtering at any stage.

## Architecture

Decoder-only transformer (GPT-2 style):

- Learned token + position embeddings
- N pre-norm blocks: LayerNorm → multi-head causal self-attention → residual,
  LayerNorm → 4x MLP with GELU → residual
- Final LayerNorm, output head tied to the token embedding matrix
- Flash attention via `F.scaled_dot_product_attention`

Default config: 6 layers, 6 heads, 384-dim embeddings, 256-token context
(~10.7M parameters). Scale it up by editing the config block in `train.py`.

## Roadmap

- [x] Phase 1 — char-level GPT, tiny-shakespeare, single GPU
- [ ] Phase 2 — byte-pair encoding tokenizer written from scratch (replaces char-level)
- [ ] Phase 3 — larger corpus (TinyStories / FineWeb sample), scale to ~50–124M params
- [ ] Phase 4 — instruction fine-tuning on top of the pretrained base → chat REPL

## Results

First full run — 10.75M params, char-level, tiny-shakespeare, RTX 4070 SUPER:

- 5000 iterations in 4m 22s (batch 64 × 256 context, bf16 autocast)
- val loss 4.28 → **1.48** (best, iter 2000; later iterations overfit the 1MB corpus)

Sample (`python sample.py --prompt "ROMEO:"`, temperature 0.8):

```text
ROMEO:
Come, that not the house of my heart's son:
Be not on the hell of the other's womb.

HERMIONE:
Be so the gods.

LEONTES:
Ay, ay, a more dead! What rather gawds
Than hand done thee, but these words and suit
The son in a dreams.
```

Not Shakespeare — but unmistakably *trying* to be, and every parameter
started as random noise a few minutes earlier.
