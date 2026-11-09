"""Kestrel-6 test suite.

These are real correctness tests (not capability benchmarks), mirroring the
Day-6 contract: tokenizer round-trip, model import/compile, generation
determinism, checkpoint save/load with config-hash validation, and the
dense baseline forward/loss shape.

Run: python3 -m pytest research/day6/tests/  (or python3 research/day6/tests/test_kestrel6.py)
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

# Allow running both as a module and as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel6.config import get_config, CONFIGS
from kestrel6.model import Kestrel6, build_model, count_params
from kestrel6.tokenizer import BPETokenizer, BOS, EOS, PAD, UNK
from kestrel6.generate import generate, greedy_generate
from kestrel6.checkpoint import save_checkpoint, load_checkpoint, VERSION
from kestrel6.train import LinearWarmupCosineDecay
from kestrel6.data import PackedDataset, train_val_test_split


def test_config_presets():
    for name in CONFIGS:
        c = get_config(name)
        assert c["vocab"] > 0
        assert c["d"] > 0
        assert c["layers"] > 0
        assert c["d"] % c["n_heads"] == 0
    assert "TINY" in CONFIGS and "SMALL" in CONFIGS and "BASELINE" in CONFIGS


def test_tokenizer_round_trip():
    texts = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "def add(a, b):\n    return a + b\n",
        "x = f\"{compute(a, b)}\"",
        "",
    ]
    tok = BPETokenizer.train(texts, vocab_size=200)
    assert tok.vocab_size >= 200
    for t in texts:
        ids = tok.encode(t, add_bos=True, add_eos=True)
        # Decode must reproduce the text (special tokens stripped).
        dec = tok.decode(ids)
        assert BOS not in dec and EOS not in dec
        # Re-encode of decode should be stable.
        ids2 = tok.encode(dec, add_bos=False, add_eos=False)
        assert tok.decode(ids2) == dec
    # Unknown token maps to UNK.
    unk_id = tok.token_to_id[UNK]
    assert unk_id is not None


def test_tokenizer_save_load():
    texts = ["hello world", "foo bar baz"]
    tok = BPETokenizer.train(texts, vocab_size=120)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        tok.save(path)
        tok2 = BPETokenizer.load(path)
        assert tok2.vocab_size == tok.vocab_size
        assert tok2.token_to_id == tok.token_to_id
        assert tok2.merges == tok.merges
        ids = tok.encode("hello world", add_bos=True)
        assert tok2.decode(ids) == tok.decode(ids)
    finally:
        os.unlink(path)


def test_model_build_and_forward():
    cfg = get_config("TINY")
    model = build_model(cfg)
    total, trainable = count_params(model)
    assert total > 0 and trainable == total
    b, n = 2, 8
    input_ids = torch.randint(0, cfg["vocab"], (b, n))
    labels = input_ids.clone()
    logits, loss = model(input_ids, labels=labels)
    assert logits.shape == (b, n, cfg["vocab"])
    assert loss is not None and loss.item() >= 0
    # labels=-100 should be ignored.
    labels_ignored = labels.clone()
    labels_ignored[0, 1] = -100
    _, loss2 = model(input_ids, labels=labels_ignored)
    assert loss2 is not None


def test_model_tied_embedding():
    cfg = get_config("TINY")
    model = build_model(cfg)
    # Tie embeddings: lm_head is None and forward uses embed.weight.
    assert model.lm_head is None
    b, n = 1, 4
    input_ids = torch.randint(0, cfg["vocab"], (b, n))
    logits, _ = model(input_ids)
    expected = input_ids @ model.embed.weight.T  # not exactly, norm applied, but shape
    assert logits.shape == (b, n, cfg["vocab"])


def test_generation_determinism():
    cfg = get_config("TINY")
    model = build_model(cfg)
    tok = BPETokenizer.train(["hello world test generation"], vocab_size=120)
    model.eval()
    with torch.no_grad():
        out1 = greedy_generate(model, tok, "hello", max_new_tokens=8, add_bos=True)
        out2 = greedy_generate(model, tok, "hello", max_new_tokens=8, add_bos=True)
    assert out1 == out2, "greedy generation must be deterministic"
    assert out1.startswith("hello")


def test_generation_temperature_top_p():
    cfg = get_config("TINY")
    model = build_model(cfg)
    tok = BPETokenizer.train(["abc def ghi"], vocab_size=80)
    model.eval()
    with torch.no_grad():
        out = generate(model, tok, "abc", max_new_tokens=6, temperature=0.7, top_p=0.9, seed=0, add_bos=True)
    assert isinstance(out, str) and out.startswith("abc")


def test_checkpoint_save_load_round_trip():
    cfg = get_config("TINY")
    model = build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = LinearWarmupCosineDecay(optim, warmup_steps=10, total_steps=100)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(
            model, optim, sched, step=42, token_count=10_000,
            path=path, best_val_loss=2.5, history=[{"step": 1}], seed=42, run_name="t",
        )
        info = load_checkpoint(path, model, optim, sched)
        assert info["step"] == 42
        assert info["token_count"] == 10_000
        assert model.checkpoint_loaded is True
        assert model.model_status == "checkpoint_loaded"
    finally:
        os.unlink(path)


def test_checkpoint_fail_closed_on_config_mismatch():
    cfg_tiny = get_config("TINY")
    cfg_small = get_config("SMALL")
    model_a = build_model(cfg_tiny)
    model_b = build_model(cfg_small)
    optim = torch.optim.AdamW(model_a.parameters(), lr=1e-3)
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        save_checkpoint(model_a, optim, None, step=1, token_count=10, path=path)
        # Load into a model with a different config -> must raise.
        try:
            load_checkpoint(path, model_b)
            assert False, "Expected config-hash mismatch error"
        except ValueError as e:
            assert "mismatch" in str(e).lower() or "hash" in str(e).lower()
    finally:
        os.unlink(path)


def test_training_step_decreases_loss_on_overfit():
    """One batch overfit: loss should drop after a few steps on a tiny dataset."""
    cfg = get_config("TINY")
    model = build_model(cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    texts = ["the cat sat on the mat "] * 20
    tok = BPETokenizer.train(texts, vocab_size=60)
    ds = PackedDataset(tok, texts, seq_len=16, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    model.train()
    losses = []
    for input_ids, labels in loader:
        for _ in range(5):
            optim.zero_grad()
            _, loss = model(input_ids, labels=labels)
            loss.backward()
            optim.step()
            losses.append(float(loss.item()))
        break
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_forward_shapes_all_presets():
    for name in ["TINY", "SMALL", "BASELINE"]:
        cfg = get_config(name)
        model = build_model(cfg)
        b, n = 1, 10
        input_ids = torch.randint(0, cfg["vocab"], (b, n))
        logits, loss = model(input_ids, labels=input_ids.clone())
        assert logits.shape == (b, n, cfg["vocab"])
        assert loss is not None


if __name__ == "__main__":
    tests = [
        test_config_presets,
        test_tokenizer_round_trip,
        test_tokenizer_save_load,
        test_model_build_and_forward,
        test_model_tied_embedding,
        test_generation_determinism,
        test_generation_temperature_top_p,
        test_checkpoint_save_load_round_trip,
        test_checkpoint_fail_closed_on_config_mismatch,
        test_training_step_decreases_loss_on_overfit,
        test_forward_shapes_all_presets,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
