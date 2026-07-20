"""Tests for the GPT model.  Run: python -m pytest -q

These run on CPU in a few seconds — no GPU required.
"""

import pytest
import torch

from model import GPT, GPTConfig

CFG = GPTConfig(block_size=32, vocab_size=50, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_forward_shapes(model):
    idx = torch.randint(0, CFG.vocab_size, (4, 16))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, 16, CFG.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_inference_returns_only_last_position(model):
    """Generation only needs the final step's logits; forward skips the rest."""
    logits, loss = model(torch.randint(0, CFG.vocab_size, (2, 16)))
    assert logits.shape == (2, 1, CFG.vocab_size)
    assert loss is None


def test_initial_loss_is_near_uniform():
    """An untrained model should be about as good as guessing: loss ~ ln(vocab)."""
    torch.manual_seed(0)
    m = GPT(CFG).eval()
    idx = torch.randint(0, CFG.vocab_size, (8, 32))
    _, loss = m(idx, idx)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(float(CFG.vocab_size))), abs=0.6)


def test_attention_is_causal(model):
    """The load-bearing property: position t must not see position t+1.

    Changing a *later* token must leave every earlier position's output
    untouched. If the causal mask breaks, the model trivially cheats by reading
    the answer, training loss collapses, and generation is garbage — with no
    error anywhere. This is the test that catches it.
    """
    idx = torch.randint(0, CFG.vocab_size, (1, 16))
    logits_a, _ = model(idx, idx)

    changed = idx.clone()
    changed[0, -1] = (changed[0, -1] + 1) % CFG.vocab_size  # perturb the last token only
    logits_b, _ = model(changed, changed)

    # every position before the change must be bit-identical
    assert torch.equal(logits_a[:, :-1, :], logits_b[:, :-1, :])
    # and the changed position itself must actually differ (test is not vacuous)
    assert not torch.equal(logits_a[:, -1, :], logits_b[:, -1, :])


def test_generate_extends_sequence(model):
    idx = torch.zeros((2, 3), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=7)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :3], idx)  # prompt is preserved verbatim
    assert (out >= 0).all() and (out < CFG.vocab_size).all()


def test_generate_respects_block_size(model):
    """A prompt longer than the context window must crop, not crash."""
    idx = torch.zeros((1, CFG.block_size + 10), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, CFG.block_size + 13)


def test_greedy_generation_is_deterministic(model):
    """temperature -> 0 with top_k=1 is argmax: same input, same output."""
    idx = torch.zeros((1, 4), dtype=torch.long)
    a = model.generate(idx, max_new_tokens=8, temperature=1e-9, top_k=1)
    b = model.generate(idx, max_new_tokens=8, temperature=1e-9, top_k=1)
    assert torch.equal(a, b)


def test_top_k_restricts_sampled_tokens(model):
    """With top_k=1 every sampled token must be the argmax token."""
    idx = torch.zeros((1, 4), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5, top_k=1)
    for step in range(4, out.shape[1] - 1):
        logits, _ = model(out[:, : step + 1])
        assert out[0, step + 1].item() == logits[0, -1].argmax().item()


def test_weights_are_tied(model):
    """Output head shares the token embedding matrix — one tensor, not a copy."""
    assert model.head.weight is model.tok_emb.weight


def test_backward_populates_all_grads(model):
    """Every parameter must receive gradient; an unused one signals dead code."""
    model.train()
    idx = torch.randint(0, CFG.vocab_size, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_rejects_sequence_longer_than_block_size(model):
    with pytest.raises(AssertionError):
        model(torch.zeros((1, CFG.block_size + 1), dtype=torch.long))
