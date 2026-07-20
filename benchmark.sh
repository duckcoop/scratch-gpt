#!/usr/bin/env bash
# Clean head-to-head: character-level vs BPE, identical model and schedule.
# Early stopping is disabled so both produce a full comparable curve.
set -e
cd "$(dirname "$0")"

echo "=============================================================="
echo "RUN 1/2 — character-level baseline"
echo "=============================================================="
python -u data/prepare.py --tokenizer char
python -u train.py --out-dir out-char --patience 0

echo
echo "=============================================================="
echo "RUN 2/2 — BPE with GPT-2 regex pre-tokenization"
echo "=============================================================="
python -u data/prepare.py --tokenizer bpe --vocab-size 1024
python -u train.py --out-dir out-bpe --patience 0

echo
echo "=============================================================="
echo "SAMPLES"
echo "=============================================================="
echo "--- BPE (data/ currently holds bpe) ---"
python -u sample.py --ckpt out-bpe/ckpt.pt --prompt "ROMEO:" --tokens 200

python -u data/prepare.py --tokenizer char >/dev/null
echo
echo "--- character-level ---"
python -u sample.py --ckpt out-char/ckpt.pt --prompt "ROMEO:" --tokens 200

# leave the working tree on bpe, which is the project default
python -u data/prepare.py --tokenizer bpe --vocab-size 1024 >/dev/null
echo
echo "benchmark complete"
