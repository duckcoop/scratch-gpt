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
| `test_tokenizer.py` | Roundtrip on unicode, emoji, contractions, whitespace edge cases |
| `test_model.py` | Shapes, weight tying, gradient flow, and that attention is really causal |

39 tests, all CPU-only, run in under two seconds:

```bash
python -m pytest -q
```

The one worth reading is `test_attention_is_causal`. If the causal mask breaks,
the model can see the token it is trying to predict — training loss collapses,
generations turn to noise, and *nothing raises an error*. The test perturbs the
last token and asserts every earlier position's logits are bit-identical.

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

The first merges are the most frequent byte pairs in English, and later ones are
recognisably Shakespearean:

```text
merge   1: (32, 116)  -> ' t'      x23837
merge   2: (104, 101) -> 'he'      x18203
merge 100: (309, 293) -> ' have'   x1325
merge 200: (39, 273)  -> "'ll"     x580
merge 500: (309, 295) -> ' hast'   x205
merge 600: (260, 710) -> ' soul'   x163
```

Two things make this practical. Merging **unique pieces weighted by frequency**
rather than every occurrence (as in the original BPE paper) took training from
~20 minutes to ~50s. Adding **GPT-2's regex pre-tokenization** — which splits
letters, digits, punctuation and whitespace apart before merging, so no merge
can ever span `"dog.\nThe"` — took it to **8.7s**.

Pre-tokenization is a real trade, not a free win: raw compression *fell* from
2.68 to 2.43 bytes/token, because the tokenizer is no longer allowed to glue
punctuation and newlines into words. The tokens it does learn are cleaner and
generalise better, which is why GPT-2 through GPT-4 all do this — but whether
that improves *modelling* on this corpus is unmeasured. See below.

### Measured vs unmeasured

Being explicit, because the distinction matters:

| Claim | Status |
| --- | --- |
| BPE ties char-level at ~2.14 bpb, overfits sooner | Measured (table above) |
| Frequency-weighted merges: ~20 min → 50s | Measured |
| Regex pre-tokenization: 50s → 8.7s, 2.68 → 2.43 bytes/token | Measured |
| Regex pre-tokenization improves *model* quality | **Not yet measured** |

The results table was produced with the earlier space-splitting pre-tokenizer.
Re-running it against the regex version needs another GPU run, and until that
happens the last row stays honest.
