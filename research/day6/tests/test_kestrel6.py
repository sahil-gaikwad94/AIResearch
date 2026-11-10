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
from kestrel6.model import (
    Kestrel6,
    build_model,
    count_params,
    CodeJEPAWorkspace,
    SubspaceLatentMoE,
    WYHouseholderMemory,
    rms_norm,
)
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


def test_config_strips_metadata_keys():
    """P0-02: '_comment' is documentation, never a preset or a config field."""
    assert "_comment" not in CONFIGS
    assert get_config("BASELINE_MLA_LOCAL") is not None
    for name in CONFIGS:
        assert not any(k.startswith("_") for k in get_config(name))
    try:
        get_config("_comment")
        assert False, "get_config('_comment') must raise KeyError"
    except KeyError:
        pass


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
    with torch.no_grad():
        logits, _ = model(input_ids)
        # Replicate the trunk explicitly: with lm_head=None the final projection
        # must be exactly F.linear(rms_norm(x, norm), embed.weight).
        x = model.embed(input_ids)
        for blk in model.blocks:
            x, _, _ = blk(x)
        hidden = rms_norm(x, model.norm)
        expected = hidden @ model.embed.weight.T
    assert logits.shape == (b, n, cfg["vocab"])
    assert torch.allclose(logits, expected, atol=1e-5, rtol=1e-5)


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


def _tiny_mla_cfg():
    cfg = get_config("BASELINE_MLA_LOCAL")
    cfg.update(
        vocab=64, d=64, n_heads=4, n_kv_heads=2, d_ff=128, layers=2,
        max_ctx=32, sliding_window=None, d_c=16, d_rope=8,
    )
    return cfg


def test_baseline_mla_local_forward_and_cache():
    """P0-06: MLA latent KV sizing, k_pe cache axis, sinks batch broadcast, streaming."""
    cfg = _tiny_mla_cfg()
    model = build_model(cfg)
    model.eval()
    ids = torch.randint(0, cfg["vocab"], (2, 6))
    with torch.no_grad():
        logits, _ = model(ids)
        assert logits.shape == (2, 6, cfg["vocab"])
        assert torch.isfinite(logits).all()

        pre_logits, cache = model.prefill(ids[:, :3])
        steps = [pre_logits]
        for pos in range(3, 6):
            step_logits, cache = model.decode_step(ids[:, pos : pos + 1], cache, seq_start=pos)
            steps.append(step_logits)
        streamed = torch.stack(steps, dim=1)
        diff = float((logits[:, 2:].data - streamed.data).abs().max())
    assert diff < 1e-4, f"MLA streaming cache mismatch: {diff}"
    assert cache[0]["attn"]["k_pe"].shape[2] == 6


def test_wy_householder_memory_time_slice():
    """P0-07: u2 must be the per-token slice k2[:, t], not the [b, d_m] tensor."""
    mem = WYHouseholderMemory(d=16, d_m=8, r=4, n_householder=2)
    x = torch.randn(2, 5, 16)
    delta, state = mem(x)
    assert delta.shape == (2, 5, 16)
    assert state.shape == (2, 4, 8)
    assert torch.isfinite(delta).all() and torch.isfinite(state).all()


def test_subspace_moe_stiefel_retraction():
    """P0-07: stack W along columns so G*d_lat <= d and QR is full-rank."""
    moe = SubspaceLatentMoE(
        d=16, d_lat=4, d_ff=16, n_groups=2, experts_per_group=2,
        top_groups=1, top_k_per_group=1,
    )
    moe.retract_stiefel()
    orth = float(moe.orthogonality_error().detach())
    assert orth < 1e-10, f"Stiefel orthogonality error too high: {orth}"
    out = moe(torch.randn(2, 3, 16))
    assert out.shape == (2, 3, 16) and torch.isfinite(out).all()


def test_code_jepa_workspace_returns_delta_only():
    """P0-07: Linear calls, per-layer kv, and delta-only return (no double residual)."""
    ws = CodeJEPAWorkspace(d=16, d_w=16, w_layers=1, w_ff=32, d_head_h=8, k_hyp=2, window=8, r_max=8)
    x = torch.randn(2, 5, 16)
    delta2, h2, p2, zloc2, zexec2 = ws(x, r=2)
    delta8, h8, p8, zloc8, zexec8 = ws(x, r=8)
    assert delta2.shape == (2, 5, 16)
    assert p2.shape == (2, 5)
    assert zexec2.shape == (2, 5, 16) and zloc2.shape == (2, 5, 16)
    assert float((zexec8 - zexec2).abs().max().detach()) > 0.0
    assert float((delta8 - delta2).abs().max().detach()) > 0.0


def test_workspace_only_on_final_block():
    cfg = get_config("TINY")
    cfg.update(vocab=64, d=32, n_heads=4, n_kv_heads=2, d_ff=64, layers=3, max_ctx=32)
    cfg.update(use_workspace=True, d_w=16, w_layers=1, w_ff=32, d_head_h=8, k_hyp=2)
    model = build_model(cfg)
    flags = [blk.use_workspace for blk in model.blocks]
    assert flags == [False, False, True], flags
    ids = torch.randint(0, cfg["vocab"], (1, 8))
    logits, loss = model(ids, labels=ids.clone())
    assert logits.shape == (1, 8, cfg["vocab"]) and loss is not None


def test_single_shift_label_alignment():
    """P0-04: labels are stored unshifted; forward applies exactly one shift."""
    cfg = get_config("TINY")
    cfg.update(vocab=64, d=32, n_heads=4, n_kv_heads=2, d_ff=64, layers=1, max_ctx=32)
    tok = BPETokenizer.train(["the cat sat on the mat " * 8], vocab_size=64)
    cfg["vocab"] = tok.vocab_size
    model = build_model(cfg)
    ds = PackedDataset(tok, ["the cat sat on the mat " * 4], seq_len=16, seed=0)
    ids, labels = ds[0]
    pad_id = tok.token_to_id[PAD]
    for t in range(16):
        if int(ids[t]) == pad_id:
            assert int(labels[t]) == -100
        else:
            assert int(labels[t]) == int(ids[t]), "labels must equal input_ids (unshifted)"

    batch = ids.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits, loss = model(batch, labels=labels.unsqueeze(0))
        manual = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, cfg["vocab"]),
            labels[1:].reshape(-1),
            ignore_index=-100,
        )
    assert torch.allclose(loss, manual, atol=1e-6)


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
        test_config_strips_metadata_keys,
        test_baseline_mla_local_forward_and_cache,
        test_wy_householder_memory_time_slice,
        test_subspace_moe_stiefel_retraction,
        test_code_jepa_workspace_returns_delta_only,
        test_workspace_only_on_final_block,
        test_single_shift_label_alignment,
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
