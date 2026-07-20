# scratch-gpt

A GPT language model trained **from scratch** — no pretrained weights, no
fine-tuning, no safety layer. Every parameter starts as random noise and
learns only from the corpus in `data/`. What you train on is what you get.

Built in plain PyTorch, ~300 lines of model code. Runs on a single consumer
GPU (developed on an RTX 4070 SUPER) or CPU.

The 52M-parameter model, after four hours of training on one desktop GPU,
prompted with *"The dragon looked at the tiny mouse and said,"*:

```text
The dragon looked at the tiny mouse and said, "Hello there. Would you like
to come to the castle and have some fun?"
The mouse said, "Yes, please!"
The dragon opened its giant hand, and off they went to the castle.
...
At the end of the day, the dragon and the mouse said goodbye. They promised
to come back another day. Then they both went home with smiles on their faces.
```

Random noise four hours earlier. Nothing was fine-tuned; this is what the
architecture in `model.py` learns on its own from raw text.

## Quickstart

```bash
pip install torch numpy          # CUDA build recommended: pytorch.org/get-started
python data/prepare.py           # download corpus + train the BPE tokenizer (~1 min)
python train.py                  # a few minutes on a modern GPU
python sample.py --prompt "ROMEO:" --tokens 500
python -m pytest -q              # 39 tests, CPU-only, ~2s
```

To reproduce the story model from the top of this README (~4 h on a 12 GB GPU),
download [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
to `data/TinyStories-train.txt`, then:

```bash
python data/prepare.py --input TinyStories-train.txt --vocab-size 4096 --sample-mb 50
python train.py --n-layer 10 --n-head 10 --n-embd 640 --block-size 512 \
                --batch-size 32 --lr 6e-4 --dropout 0.0 --max-iters 100000 \
                --eval-interval 2000 --eval-iters 50 --out-dir out-tinystories
python sample.py --ckpt out-tinystories/ckpt.pt --tokens 300 \
                 --prompt "Once upon a time, there was a little robot named Bolt."
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
(~10.7M parameters). Scale it up with `--n-layer/--n-head/--n-embd/--block-size`;
the story model above is `10 / 10 / 640 / 512`, which is 52M parameters and fits
in about 4 GB of VRAM at batch size 32.

## Roadmap

- [x] Phase 1 — char-level GPT, tiny-shakespeare, single GPU
- [x] Phase 2 — byte-level BPE tokenizer written from scratch, with a measured comparison
- [x] Phase 3 — TinyStories, 52M params, coherent English
- [ ] Phase 4 — instruction fine-tuning on top of the pretrained base → chat REPL

## Results — Phase 3, coherent English

| | |
| --- | --- |
| Corpus | TinyStories, 1.92 GB → **487M tokens** (4,096-token BPE, 3.95 bytes/token) |
| Model | **52.13M params** — 10 layers, 10 heads, 640-dim, 512-token context |
| Training | 100,000 iters, batch 32 — **1.64B tokens seen** (~3.7 epochs) |
| Hardware | one RTX 4070 SUPER, **4h 8m** |
| Best val loss | **1.1549** (0.4218 bits/byte), at the final iteration |

Phase 2 ended by concluding the bottleneck was data, not tokenization. Phase 3
is that prediction tested: same `model.py`, same `tokenizer.py`, same training
loop — 1,700x more text and 4.7x the parameters.

```text
Sara found a strange key in the garden. She didn't know what it was for, so
she asked her mom.
"Mom, what is this?" asked Sara.
"That's a key, Sara. It can unlock anything," said Mom.
Sara was so excited to try it out. She took the key and tried it. The door
opened, and she was so happy to see what was inside.
```

Compare that to the best the 11M model managed on Shakespeare — *"O, by thou
wert wit'st to bed, but crusp thy head"* — and the difference is entirely the
data. The model learns dialogue punctuation, attribution (`asked Sara` /
`said Mom`), character consistency across paragraphs, and story structure with
a beginning and an ending. It even learned to emit `<|endoftext|>` and start a
fresh story.

It is not flawless. Sentences like *"He could not turn the game back that
right"* still appear, and plots drift. At 52M parameters that is the expected
ceiling — TinyStories exists precisely because small models can learn *this*
distribution well, and general web text not at all.

**Data volume, not model size, was the whole story.** The 11M Shakespeare model
hit its best validation loss at iteration 1,000 of 5,000 and then memorised for
the remaining 80% of the run. The 52M model improved for all 100,000 iterations
and was still improving when the schedule ended — with 438M training tokens
there is simply nothing to memorise. Validation loss tracked training loss
within 0.03 for the first half of the run.

## Results — Phase 2, the tokenizer comparison

All runs: ~11M params, 6 layers, 6 heads, 384-dim, 256-token context, 5000
iterations, identical seed, one RTX 4070 SUPER. Lower bits-per-byte is better.

| Tokenizer | Vocab | Corpus tokens | Bytes/token | Best val bpb | Best at iter | Wall clock |
| --- | --- | --- | --- | --- | --- | --- |
| char | 65 | 1,115,394 | 1.00 | 2.1335 | 2000 | 254s |
| BPE, space-split | 1,024 | 416,758 | 2.68 | 2.1435 | 750 | — |
| BPE, regex pre-split | 1,024 | 459,760 | 2.43 | **2.0681** | 1000 | 246s |

**The pre-tokenizer mattered more than BPE itself.** Naive space-splitting BPE
*lost* to plain characters (2.1435 vs 2.1335). Adding GPT-2's regex pre-split —
so a merge can never span a word, punctuation or newline boundary — moved it to
2.0681, beating the character baseline by 3.1% and reaching that score in half
the iterations.

The counterintuitive part, and the most useful thing measured here:

> **Compression got worse while the model got better.** The regex version
> encodes the corpus *less* densely (2.43 vs 2.68 bytes/token) yet models it
> better. Bytes-per-token is a tempting proxy for tokenizer quality and it is
> the wrong one — space-splitting achieves higher compression by inventing
> tokens like `"dog.\nThe"` that carry real information density but generalise
> to almost nothing.

Every configuration still overfits hard — the best BPE checkpoint is at
iteration 1000 of 5000, ending at train loss 0.15 against val loss 5.53. An
11M-parameter model exhausts a 1MB corpus. The tokenizer is no longer the
bottleneck; **data is.** Hence Phase 3.

Character-level sample (`--prompt "ROMEO:"`, temperature 0.8):

```text
ROMEO:
What is much sickness? where you fair to Marcius have
To save the field-from that you have longed did
The gods common for the sun of that strivice
That straight a traitors of the crown.
```

BPE sample, same prompt and temperature:

```text
ROMEO:
O, by thou wert wit'st to bed, but crusp thy head;
For 'twas the redress of thy sorrow,
But buckle-winger, that thou hast not lived,
To given upon thy shadowing face:
Why should I not piece of thine own bed,
```

Neither is Shakespeare. The BPE one holds metre and produces far more
well-formed words, which is the qualitative shape of that 3.1% bpb gap. Both
started as random noise four minutes earlier.

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

Pre-tokenization also lowers raw compression — 2.68 to 2.43 bytes/token — since
punctuation and newlines can no longer be glued into words. It improves the
model anyway; see the results table above for why that is the interesting part.
