# scratch-gpt

A GPT language model trained **from scratch** — no pretrained weights, no
fine-tuning, no safety layer. Every parameter starts as random noise and
learns only from the corpus in `data/`. What you train on is what you get.

Built in plain PyTorch, ~300 lines of model code. Runs on a single consumer
GPU (developed on an RTX 4070 SUPER) or CPU.

## Quickstart

```bash
pip install torch numpy          # CUDA build recommended: pytorch.org/get-started
python data/prepare.py           # download corpus + train the BPE tokenizer (~1 min)
python train.py                  # a few minutes on a modern GPU
python sample.py --prompt "ROMEO:" --tokens 500
python -m pytest -q              # tokenizer test suite
```

## What's in here

| File | What it does |
| --- | --- |
| `model.py` | The transformer: causal self-attention, pre-norm blocks, weight-tied head |
| `tokenizer.py` | Byte-level BPE, written from scratch — merge training, encode, decode, save/load |
| `data/prepare.py` | Downloads the corpus, trains the tokenizer, writes `train.bin`/`val.bin` |
| `train.py` | Training loop: AdamW, cosine LR schedule with warmup, bf16 autocast, grad clipping, best-val checkpointing |
| `sample.py` | Autoregressive generation with temperature and top-k sampling |
| `test_tokenizer.py` | Tokenizer test suite — roundtrip on unicode, emoji, whitespace edge cases |

## Tokenizers

Two are implemented, and `prepare.py` chooses which the model trains on:

```bash
python data/prepare.py                       # byte-level BPE, 1024-token vocab (default)
python data/prepare.py --vocab-size 4096     # bigger vocab = fewer, denser tokens
python data/prepare.py --tokenizer char      # the Phase 1 character-level baseline
```

The BPE implementation in `tokenizer.py` is written from scratch — no `tiktoken`,
no `tokenizers` library. It is *byte-level*, so the base vocabulary is the 256
possible bytes and **any** UTF-8 input encodes losslessly with no unknown tokens
(emoji and non-Latin scripts included — see `test_tokenizer.py`).

### Comparing runs honestly

Validation loss is **not** comparable across tokenizers: a BPE token carries
several bytes, so a model with higher per-token loss can still be the better
model. Training therefore also reports **bits per byte**, which normalises for
token size and is directly comparable between runs.

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
- [x] Phase 2 — byte-level BPE tokenizer written from scratch, with a measured comparison
- [ ] Phase 3 — larger corpus (TinyStories / FineWeb sample), scale to ~50–124M params
- [ ] Phase 4 — instruction fine-tuning on top of the pretrained base → chat REPL

## Results

Both runs: ~11M params, 6 layers, 6 heads, 384-dim, 256-token context,
5000 iterations on one RTX 4070 SUPER, identical seed.

| Tokenizer | Vocab | Corpus tokens | Bytes/token | Best val bpb | Best at iter |
| --- | --- | --- | --- | --- | --- |
| char | 65 | 1,115,394 | 1.00 | **2.138** | 2000 |
| BPE | 1,024 | 416,758 | 2.68 | 2.144 | 750 |

**BPE did not win.** It matched the character baseline and got there in a
quarter of the steps, then overfit much harder — final train loss 0.13 against
a val loss of 6.44. That is the honest result and it is the useful one: BPE cut
the corpus from 1.1M tokens to 417k, so an 11M-parameter model simply runs out
of new text and starts memorising. The tokenizer was never the bottleneck.
**Data is.** Hence Phase 3.

What BPE *does* buy, even here: 2.68x more text fits in the same 256-token
context window, and generations contain far more well-formed English words,
because the model spends its capacity on word structure instead of spelling.

Character-level sample (`--prompt "ROMEO:"`, temperature 0.8):

```text
ROMEO:
Come, that not the house of my heart's son:
Be not on the hell of the other's womb.

HERMIONE:
Be so the gods.
```

BPE sample, same prompt and temperature:

```text
ROMEO: and now, no manne merites but that vock
Blues his framonment of his sugdden signs?
Have we cared to doubt not a pengggue for a visit
From such a foul morning in lament;
And I have gotted by my country's groan,
```

Neither is Shakespeare. Both started as random noise minutes earlier.

### What the tokenizer learned

The first merges are the most frequent byte pairs in English, and later ones
are recognisably Shakespearean:

```text
merge   1: (32, 116)   -> ' t'        x23837
merge   2: (104, 101)  -> 'he'        x18203
merge 100: (289, 270)  -> ' his'      x1415
merge 600: (350, 400)  -> ' stand'    x178
merge 700: (954, 438)  -> 'ORIOLAN'   x150     <- from CORIOLANUS
```

Training 768 merges over the 1MB corpus takes ~50s in pure Python. The naive
implementation — rescanning every word occurrence each merge — took ~20 minutes;
operating on unique words weighted by frequency (as in the original BPE paper)
is what makes it practical.
