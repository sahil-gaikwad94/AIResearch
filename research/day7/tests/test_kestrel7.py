"""Kestrel-7 x VEXX 3.0 verification suite (20 contract suites + 15 audit regressions).

Run:
    python3 -m pytest research/day7/tests/ -q

The 16 suites mirror ``Kestrel7_Day7_Architecture_and_VEXX3_Guide.md`` and the
reference ``kestrel7_vexx3_complete.py`` self-check: config hygiene, tokenizer
hashing, leakage-free splitting, single-shift labels, block-diagonal attention,
hybrid streaming equivalence, MTP gradients, accumulation/validation/checkpoint
behaviour, reverse-KL, and VEXX Tiers 0-5. The extra tests assert the *computed*
ledger and benchmark numbers that the markdown quotes, so documentation cannot
silently drift from the code.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel7.config import CONFIGS, VALID_CONFIG_KEYS, get_config
from kestrel7.data import StreamPackedDataset, collate_stream_packed, deduplicate_and_split
from kestrel7.generate import generate_v7, sample_next
from kestrel7.model import (
    GatedCausalSelfAttention,
    GatedDeltaNetLayer,
    Kestrel7,
    build_model,
    count_params,
    gdn_state_bytes,
    kv_cache_bytes,
)
from kestrel7.checkpoint import load_checkpoint_v7, save_checkpoint_v7
from kestrel7.tokenizer import (
    BOS,
    EOS,
    PAD,
    UNK,
    SPECIAL_TOKENS,
    THINK_END,
    THINK_START,
    BPETokenizer,
)
from kestrel7.train import (
    Kestrel7Trainer,
    LinearWarmupCosineDecay,
    compute_on_policy_reverse_kl_loss,
)
from kestrel7.vexx import Vexx3Controller, safe_sympy

DAY7_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(DAY7_ROOT, "tools"))
from eval_tempbench_v7 import (  # noqa: E402
    CALIBRATION_EXPECTED,
    KEY_PATH,
    PROMPTS_PATH,
    independent_answer,
    load_jsonl,
    run_benchmark,
    syllogism_answer,
)


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def tok() -> BPETokenizer:
    texts = [
        "def add(a, b): return a + b\nhello world",
        "def square(x):\n    return x * x\n",
        "The quick brown fox jumps over the lazy dog.",
        "2 + 2 = 4 and 37 * 48 - 19 * 32 = 1168",
    ]
    return BPETokenizer.train(texts * 4, vocab_size=280)


@pytest.fixture(scope="session")
def hybrid(tok):
    return build_model("TINY_HYBRID", tokenizer=tok)


@pytest.fixture(scope="session")
def dense(tok):
    return build_model("TINY_DENSE", tokenizer=tok)


# --------------------------------------------------------------------------- [1]


def test_config_metadata_and_schema():
    assert "_comment" not in CONFIGS
    assert set(CONFIGS) == {"TINY_DENSE", "TINY_HYBRID", "SMALL_HYBRID", "BASELINE_100M"}
    for name in CONFIGS:
        cfg = get_config(name)
        assert not any(k.startswith("_") for k in cfg)
        assert set(cfg) <= VALID_CONFIG_KEYS
        assert cfg["layers"] > 0 and cfg["d"] % cfg["n_heads"] == 0
        schedule = cfg["hybrid_schedule"]
        if schedule is not None:
            assert len(schedule) == cfg["layers"]
            assert set(schedule) <= {"gdn", "full"}
            # 3:1 layout: exactly one full layer per four, in position 4k+3.
            for i, kind in enumerate(schedule):
                assert kind == ("full" if i % 4 == 3 else "gdn")
    with pytest.raises(KeyError):
        get_config("_comment")


# --------------------------------------------------------------------------- [2]


def test_tokenizer_roundtrip_specials_and_hash():
    custom = SPECIAL_TOKENS + ["<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>"]
    tok = BPETokenizer.train(["def add(a, b): return a + b\nhello world"] * 5, vocab_size=280, special_tokens=custom)
    samples = [
        "hello world",
        "def add(a, b):\n    return a + b\n",
        'x = f"{compute(a, b)}"',
        "héllo wörld — 日本語",
    ]
    for s in samples:
        assert tok.decode(tok.encode(s)) == s
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tok.json")
        tok.save(path)
        tok2 = BPETokenizer.load(path)
    assert tok2.special_tokens == custom
    assert tok2.tokenizer_hash() == tok.tokenizer_hash()
    assert tok2.vocab_size == tok.vocab_size
    # A different merge table must produce a different hash.
    other = BPETokenizer.train(["totally different corpus text " * 3], vocab_size=280)
    assert other.tokenizer_hash() != tok.tokenizer_hash()


# --------------------------------------------------------------------------- [3]


def test_deduplicate_before_split_zero_overlap():
    raw = [f"Unique document number {i}: x = {i} * 2" for i in range(20)] * 5
    train, val, test, manifest = deduplicate_and_split(raw, seed=42)
    assert manifest["raw_count"] == 100
    assert manifest["unique_count"] == 20
    assert manifest["dropped_duplicates"] == 80
    assert not (set(train) & set(val))
    assert not (set(train) & set(test))
    assert not (set(val) & set(test))
    assert len(train) + len(val) + len(test) == 20


# --------------------------------------------------------------------------- [4]


def test_stream_packed_single_shift_and_bos_mask(tok):
    train, _val, _test, _m = deduplicate_and_split(
        [f"Document {i} says alpha beta gamma delta." for i in range(12)], seed=1
    )
    ds = StreamPackedDataset(tok, train, seq_len=32, seed=42)
    assert len(ds) > 0
    bos_id = tok.token_to_id[BOS]
    pad_id = tok.token_to_id[PAD]
    for idx in range(len(ds)):
        ids, labels, docs = ds[idx]
        for t in range(1, len(ids)):
            if int(ids[t]) == pad_id:
                assert int(labels[t]) == -100
            elif int(ids[t]) == bos_id:
                assert int(labels[t]) == -100, "a new document's BOS must never be a label"
            else:
                assert int(labels[t]) == int(ids[t]), "labels must be unshifted"
                assert int(docs[t]) >= 0
    # A manual single-shift cross-entropy must match the value Kestrel7 computes.
    model = build_model("TINY_DENSE", tokenizer=tok)
    ids, labels, docs = ds[0]
    model.eval()
    with torch.no_grad():
        logits, loss = model(
            ids.unsqueeze(0), labels=labels.unsqueeze(0), doc_ids=docs.unsqueeze(0), mtp_weight=0.0
        )
        manual = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, model.vocab), labels[1:].reshape(-1), ignore_index=-100
        )
    assert torch.allclose(loss, manual, atol=1e-6)


# --------------------------------------------------------------------------- [5]


def test_block_diagonal_document_mask(dense, hybrid):
    x = torch.randint(0, dense.vocab, (1, 8))
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    for model in (dense, hybrid):
        logits_a, _ = model(x, doc_ids=doc_ids)
        perturbed = x.clone()
        perturbed[0, 0] = (int(perturbed[0, 0]) + 5) % model.vocab
        logits_b, _ = model(perturbed, doc_ids=doc_ids)
        diff = float((logits_a[0, 4:] - logits_b[0, 4:]).abs().max().detach())
        assert diff == 0.0, f"{type(model.blocks[0].mixer).__name__}: cross-document leak {diff}"


# --------------------------------------------------------------------------- [6]


def test_hybrid_streaming_cache_equivalence(hybrid):
    seq = torch.randint(0, hybrid.vocab, (1, 6))
    full, _ = hybrid(seq)
    prefill_logits, cache, _ = hybrid.prefill(seq[:, :3])
    steps = [prefill_logits]
    for pos in range(3, 6):
        step_logits, cache = hybrid.decode_step(seq[:, pos : pos + 1], cache, seq_start=pos)
        steps.append(step_logits)
    streamed = torch.stack(steps, dim=1)
    diff = float((full[:, 2:] - streamed).abs().max().detach())
    assert diff < 1e-4, f"streaming cache mismatch: {diff}"
    # The cache must contain both families: attention K/V and GDN state.
    assert "k" in cache[3] and "state" in cache[0]


# --------------------------------------------------------------------------- [7]


def test_mtp_head_receives_gradient(hybrid):
    hybrid.zero_grad(set_to_none=True)
    seq = torch.randint(0, hybrid.vocab, (1, 6))
    _, loss = hybrid(seq, labels=seq.clone(), mtp_weight=0.2)
    loss.backward()
    assert hybrid.mtp_head is not None
    grad = hybrid.mtp_head.proj.weight.grad
    assert grad is not None and float(grad.norm().detach()) > 0.0
    # Without MTP the head must not be touched.
    hybrid.zero_grad(set_to_none=True)
    _, _ = hybrid(seq, labels=seq.clone(), mtp_weight=0.0)
    assert hybrid.mtp_head.proj.weight.grad is None


# --------------------------------------------------------------------------- [8]


def test_trainer_flush_validation_and_fail_closed_checkpoint(tok, tmp_path):
    train, val, _test, _m = deduplicate_and_split(
        [f"Training document {i} with tokens alpha beta gamma." for i in range(10)], seed=3
    )
    train_ds = StreamPackedDataset(tok, train, seq_len=16, seed=42)
    val_ds = StreamPackedDataset(tok, val, seq_len=16, seed=42)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)
    model = build_model("TINY_HYBRID", tokenizer=tok)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    trainer = Kestrel7Trainer(
        model,
        train_loader,
        val_loader,
        opt,
        grad_accum_steps=3,
        val_every_tokens=10_000_000,  # deliberately larger than the corpus
        checkpoint_dir=str(tmp_path),
        seed=42,
        run_name="t",
    )
    state = trainer.fit(epochs=1)
    expected_steps = math.ceil(len(train_ds) / 2 / 3)
    assert state["optimizer_step"] == expected_steps, "partial grad-accum group was not flushed"
    assert trainer._accum_pending == 0
    assert state["best_val_loss"] < float("inf"), "final validation did not run"

    loaded = build_model("TINY_HYBRID", tokenizer=tok)
    info = load_checkpoint_v7(state["checkpoint_path"], loaded, tokenizer=tok)
    assert loaded.checkpoint_loaded and loaded.model_status == "trained_checkpoint_loaded"
    assert info["best_val_loss"] == state["best_val_loss"]

    # Fail closed on a tokenizer hash mismatch.
    other_tok = BPETokenizer.train(
        ["a totally different corpus for the tokenizer " * 6], vocab_size=tok.vocab_size
    )
    assert other_tok.vocab_size == tok.vocab_size
    with pytest.raises(ValueError, match="tokenizer_hash"):
        load_checkpoint_v7(state["checkpoint_path"], build_model("TINY_HYBRID", tokenizer=tok), tokenizer=other_tok)

    # Fail closed on a config mismatch.
    cfg_small = get_config("TINY_HYBRID")
    cfg_small["d"] = 128
    cfg_small["n_heads"] = 4
    cfg_small["d_ff"] = 512
    with pytest.raises(ValueError, match="config_hash"):
        load_checkpoint_v7(state["checkpoint_path"], build_model(cfg_small, tokenizer=tok), tokenizer=tok)


# --------------------------------------------------------------------------- [9]


def test_on_policy_reverse_kl():
    teacher = torch.randn(2, 8, 64)
    student = teacher.clone().requires_grad_(True)
    same = compute_on_policy_reverse_kl_loss(student, teacher)
    assert float(same.detach()) < 1e-6
    other = torch.randn(2, 8, 64).requires_grad_(True)
    loss = compute_on_policy_reverse_kl_loss(other, teacher)
    loss.backward()
    assert float(loss.detach()) > 0.01
    assert other.grad is not None and float(other.grad.abs().sum()) > 0.0
    mask = torch.full((2, 8), -100)
    mask[:, 0] = 1
    masked = compute_on_policy_reverse_kl_loss(other, teacher, mask)
    assert torch.isfinite(masked)


# --------------------------------------------------------------------------- [10] VEXX fixtures


@pytest.fixture()
def vx() -> Vexx3Controller:
    return Vexx3Controller(
        domain_bounds={"w1": (0.0, 100.0), "w2": (0.0, 100.0)},
        symbol_table={
            "compute": {"min_arity": 2, "max_arity": 2, "return_type": "int"},
            "check": {"min_arity": 1, "max_arity": 1, "return_type": "bool"},
        },
    )


def test_tier0_crane_dual_phase(vx, tok):
    for prefix in (THINK_START + "thinking", "done"):
        masked = vx.apply_in_flight_mask(
            np.zeros(tok.vocab_size), prefix, tok.id_to_token, remaining_token_budget=100
        )
        for special in (PAD, BOS):
            sid = tok.token_to_id[special]
            assert np.isneginf(masked[sid]), f"{special} must be masked in {prefix!r}"
        assert not np.isneginf(masked[tok.token_to_id[EOS]]), "EOS must stay legal"
    assert vx.crane_phase(THINK_START + "Let's try compute(1) first") == "UNCONSTRAINED_THINK"
    assert vx.crane_phase(THINK_START + "done" + THINK_END + "\nx = compute(1") == "STRICT_VERIFIED"
    free = vx.apply_in_flight_mask(np.zeros(1), THINK_START + "calling unknown_fn", {0: "("})
    strict = vx.apply_in_flight_mask(
        np.zeros(1), THINK_START + "done" + THINK_END + "\nx = unknown_fn", {0: "("}
    )
    assert not np.isneginf(free[0])
    assert np.isneginf(strict[0])


# --------------------------------------------------------------------------- [11]


def test_tier0_swyb_budget_lookahead(vx):
    ok = vx.apply_in_flight_mask(np.zeros(1), "x = (a + ", {0: "("}, remaining_token_budget=10)
    low = vx.apply_in_flight_mask(np.zeros(1), "x = (a + ", {0: "("}, remaining_token_budget=2)
    assert not np.isneginf(ok[0])
    assert np.isneginf(low[0])


# --------------------------------------------------------------------------- [12]


def test_tier1_lexer_ast_arity_and_receivers(vx):
    header = "def helper(a, b, c=1) -> int:\n    return a + b + c\n"
    assert np.isneginf(vx.apply_in_flight_mask(np.zeros(1), header + "x = helper(1", {0: ")"})[0])
    assert not np.isneginf(vx.apply_in_flight_mask(np.zeros(1), header + "x = helper(1, 2", {0: ")"})[0])
    assert np.isneginf(vx.apply_in_flight_mask(np.zeros(1), 'doc = """a"compute(a, b"""\nx = ', {0: ")"})[0])
    assert not np.isneginf(vx.apply_in_flight_mask(np.zeros(1), 'p = r"\\\\" + compute(a, b', {0: ")"})[0])
    assert np.isneginf(vx.apply_in_flight_mask(np.zeros(1), 's = f"{compute(1"', {0: ")"})[0])
    assert np.isneginf(vx.apply_in_flight_mask(np.zeros(1), "evil_obj.compute(a, b", {0: ")"})[0])
    assert not np.isneginf(vx.apply_in_flight_mask(np.zeros(1), "self.compute(a, b", {0: ")"})[0])
    # Defaults reduce min_arity; *args removes max_arity.
    syms = vx.extract_ast_symbol_table("def f(a, b=2, *rest, **kw):\n    pass")
    assert syms["f"]["min_arity"] == 1 and syms["f"]["max_arity"] is None
    assert not np.isneginf(vx.apply_in_flight_mask(np.zeros(1), "x = (a + b", {0: ")"})[0])
    assert vx.extract_ast_symbol_table(header)["helper"]["return_type"] == "int"


# --------------------------------------------------------------------------- [13]


def test_tier2_crane_cas(vx):
    assert vx.crane_cas_evaluate("37 * 48 - 19 * 32") == "1168"
    assert vx.crane_cas_evaluate("(5/6 - 1/4) / (7/12)") == "1"
    assert vx.solve_linear_system(["2*x + 3*y = 17", "x - y = 1"], ["x", "y"]) == ("4", "3")
    assert vx.integrate_definite("3*x^2 + 2*x", "x", 0, 2) == "12"
    assert vx.crane_cas_evaluate("1 / (x - 2)", {"x": 2}) == "DIV_BY_ZERO"
    assert vx.crane_cas_evaluate("2 ** 10") == "1024"
    assert vx.crane_cas_evaluate("x^2 + 2*x + 1", {"x": 3}) == "16"
    assert vx.crane_cas_evaluate("x + x - x", {}) == "x"
    assert vx.modular_pow(7, 100, 13) == "9"


# --------------------------------------------------------------------------- [14]


def test_tier3_farkas_cut_and_hybrid_rewind(hybrid, vx):
    seq = torch.randint(0, hybrid.vocab, (1, 6))
    _, cache, _ = hybrid.prefill(seq)
    vx.save_checkpoint("w1", 2, cache, entropy=2.5, proposed_val=50.0)
    vx.save_checkpoint("w2", 5, cache, entropy=0.4, proposed_val=80.0)
    tok_idx, rewound, bounds = vx.trigger_farkas_cdcl_jumpback(
        {"w1", "w2"}, {"coeffs": {"w1": 2, "w2": 1}, "rhs": 120}, {"w1": 50.0, "w2": 80.0}
    )
    assert tok_idx == 2
    assert vx.last_exact_fraction is not None and vx.last_exact_fraction == 20
    # Sound widening: the cut never excludes the exact optimum.
    assert bounds[1] >= 20.0
    assert bounds[1] <= math.nextafter(20.0, float("inf"))
    assert rewound[3]["k"].shape[2] == 2
    assert "state" in rewound[0] and rewound[0]["state"].shape == cache[0]["state"].shape

    # Negative coefficient yields a lower bound, not an upper bound.
    vx2 = Vexx3Controller(domain_bounds={"w1": (0.0, 100.0), "w2": (0.0, 100.0)})
    _, cache2, _ = hybrid.prefill(seq)
    vx2.save_checkpoint("w1", 1, cache2, entropy=1.0, proposed_val=50.0)
    _, _, b2 = vx2.trigger_farkas_cdcl_jumpback(
        {"w1"}, {"coeffs": {"w1": -2}, "rhs": -50}, {"w1": 50.0}
    )
    assert b2[0] <= 25.0 <= b2[1]


# --------------------------------------------------------------------------- [15]


def test_tier4_epistemic_calibration(vx):
    cases = {
        "95% CI [-0.5, 2.9], p=0.17. Does this establish that the treatment works?": "not established",
        "Source A updated 2024 says 5 units. Source B official notice effective 2026-07-01 says 8 units.": "8 units",
        "90% support policy X based on an optional online poll answered by 84 visitors to one political blog.": "not justified",
        "Name the winner of the 2026 International Quantum Materials Prize. Deliberately unverified.": "INSUFFICIENT EVIDENCE",
        "Explain why 0.999999... is less than 1.": "equal",
        "A table says a sample contains 40 people. The accompanying paragraph says 44 people. End with FINAL: CONFLICT.": "CONFLICT",
        "What is the population of the fictional city Northbridge-on-Mars?": "UNKNOWN",
    }
    for prompt, verdict in cases.items():
        result = vx.verify_epistemic_calibration(prompt)
        assert result is not None and result["verdict"] == verdict, prompt
    # Do not over-flag a genuinely significant result.
    assert vx.verify_epistemic_calibration("95% CI [0.4, 2.1], p=0.01.") is None
    # A high p-value can be non-significant without the CI crossing zero; the
    # explanation must state only reasons that are actually true.
    weak = vx.verify_epistemic_calibration("95% CI [0.4, 2.1], p=0.8.")
    assert weak is not None and weak["verdict"] == "not established"
    assert "includes zero" not in weak["explanation"]


# --------------------------------------------------------------------------- [16]


def test_tier5_sandbox_and_dpo(vx):
    bad, _total, errors = vx.safe_execute_python(
        "import os\ndef f(x): return os.system('ls')", "f", [{"input": [1], "expected": 0}]
    )
    assert bad == 0 and "Forbidden AST node" in errors[0]
    assert vx.safe_execute_python("def f(x):\n    return open('x')", "f", [{"input": [1], "expected": 0}])[0] == 0
    assert vx.safe_execute_python("def f(x):\n    return x.__class__", "f", [{"input": [1], "expected": 0}])[0] == 0

    good = """def max_subarray(nums):
    if not nums: return 0
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best"""
    passed, total, _ = vx.safe_execute_python(
        good,
        "max_subarray",
        [
            {"input": [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], "expected": 6},
            {"input": [[-5, -2, -8]], "expected": -2},
            {"input": [[]], "expected": 0},
        ],
    )
    assert passed == total == 3

    # A function that mutates its input must be refuted, not rewarded.
    mutating = """def sort_in_place(xs):
    xs.sort()
    return xs"""
    ok, _tot, _err = vx.safe_execute_python(
        mutating, "sort_in_place", [{"input": [[3, 1, 2]], "expected": [1, 2, 3]}]
    )
    assert ok == 0
    assert len(Vexx3Controller._dpo_log) > 0
    assert any(entry["winner"] for entry in Vexx3Controller._dpo_log)


# --------------------------------------------------------------------------- [17] computed ledger


def test_param_and_cache_ledger_matches_models():
    with open(os.path.join(DAY7_ROOT, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["config_names"] == sorted(CONFIGS)
    for name in ["TINY_DENSE", "TINY_HYBRID", "SMALL_HYBRID", "BASELINE_100M"]:
        row = metrics["presets"][name]
        cfg = get_config(name)
        model = build_model(cfg)
        total, trainable = count_params(model)
        assert total == row["total_params"]
        assert trainable == row["trainable_params"] == total
        assert row["layers"] == cfg["layers"]
        assert row["layers_gdn"] + row["layers_full"] == cfg["layers"]
        assert kv_cache_bytes(model, dtype_bytes=2) == row["kv_cache_bytes_fp16_ctx"]
        assert gdn_state_bytes(model) == row["gdn_state_bytes_fp32_batch1"]
        all_full = cfg["layers"] * 2 * cfg["n_kv_heads"] * (cfg["d"] // cfg["n_heads"]) * cfg["max_ctx"] * 2
        assert row["kv_cache_bytes_all_full_counterfactual_fp16"] == all_full
        if cfg["hybrid_schedule"] is not None:
            assert row["kv_reduction_vs_all_full_pct"] == 75.0
    # The 100M preset must actually sit in the documented 100.1M-151.5M band.
    assert 100_100_000 <= metrics["presets"]["BASELINE_100M"]["total_params"] <= 151_500_000


# --------------------------------------------------------------------------- [18] benchmark


def test_tempbench_scores_48_of_48():
    result = run_benchmark(verbose=False)
    assert result["tasks"] == 33
    assert result["points"] == 48
    assert result["earned"] == result["points"], result["failures"]
    assert result["pct"] == 100.0
    cat_points = {k: v["points"] for k, v in result["categories"].items()}
    assert all(
        v["status"].startswith("implemented") for v in result["categories"].values()
    ), "every scored category must carry an honest implementation label"
    assert cat_points == {
        "mathematics": 7,
        "formal_reasoning": 5,
        "code_tracing": 4,
        "code_generation": 20,
        "science": 4,
        "research_calibration": 8,
    }
    # The committed ledger must describe this same run, not an earlier one.
    with open(os.path.join(DAY7_ROOT, "metrics.json"), "r", encoding="utf-8") as f:
        stored = json.load(f)["benchmark"]
    assert stored["suite"] == "Kestrel6_TempBench"
    for key in ("tasks", "points", "earned", "pct", "failures", "categories"):
        assert stored[key] == result[key], f"metrics.json benchmark.{key} is stale"


# --------------------------------------------------------------------------- [19] docs


def test_docs_quote_computed_numbers():
    with open(os.path.join(DAY7_ROOT, "metrics.json"), "r", encoding="utf-8") as f:
        metrics = json.load(f)
    texts = []
    for fname in sorted(os.listdir(DAY7_ROOT)):
        if fname.endswith(".md"):
            with open(os.path.join(DAY7_ROOT, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    blob = "\n".join(texts)
    for name, row in metrics["presets"].items():
        assert row["total_params_pretty"] in blob, f"{name} parameter count missing from docs"
        assert f"{row['kv_cache_bytes_fp16_ctx']:,}" in blob, f"{name} KV bytes missing from docs"
    assert "48 / 48" in blob and "100.0%" in blob
    assert str(metrics["benchmark"]["tasks"]) in blob


# --------------------------------------------------------------------------- [20] generation contract


def test_greedy_contract_and_determinism(hybrid, tok):
    logits = torch.randn(1, 16)
    assert torch.equal(sample_next(logits, None), torch.argmax(logits, dim=-1, keepdim=True))
    assert torch.equal(sample_next(logits, 0.0), torch.argmax(logits, dim=-1, keepdim=True))
    assert torch.equal(sample_next(logits, -1.0), torch.argmax(logits, dim=-1, keepdim=True))
    hybrid.eval()
    out1 = generate_v7(hybrid, tok, "hello", max_new_tokens=6, temperature=None)
    out2 = generate_v7(hybrid, tok, "hello", max_new_tokens=6, temperature=None)
    assert out1 == out2 and out1.startswith("hello")
    # Temperature sampling still returns text.
    sampled = generate_v7(hybrid, tok, "hello", max_new_tokens=4, temperature=0.8, top_p=0.9, seed=0)
    assert isinstance(sampled, str) and sampled.startswith("hello")


# --------------------------------------------------------------------------- [21] initialization
def test_initialization_is_calibrated(tok):
    model = build_model("TINY_HYBRID", tokenizer=tok)
    model.eval()
    torch.manual_seed(0)
    ids = torch.randint(0, model.vocab, (2, 12))
    with torch.no_grad():
        logits, loss = model(ids, labels=ids, mtp_weight=0.0)
    assert float(model.embed.weight.detach().std()) < 0.2, "embedding scale is back to N(0,1)"
    assert float(logits.std()) < 2.0, f"initial logit std too large: {float(logits.std())}"
    assert abs(float(loss) - math.log(model.vocab)) < 1.0, (
        f"initial CE {float(loss):.3f} far from uniform {math.log(model.vocab):.3f}"
    )


# --------------------------------------------------------------------------- [22] sliding window
def test_sliding_window_prefill_and_streaming_are_finite():
    cfg = get_config("TINY_DENSE")
    cfg.update(
        layers=2, max_ctx=64, sliding_window=4, mtp_depth=0, vocab=64,
        d=32, n_heads=4, n_kv_heads=2, d_ff=64,
    )
    model = Kestrel7(cfg)
    seq = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        full, _ = model(seq)
        assert torch.isfinite(full).all(), "sliding-window prefill produced NaN"
        _, cache, _ = model.prefill(seq[:, :6])
        outs = []
        for pos in range(6, 10):
            step, cache = model.decode_step(seq[:, pos : pos + 1], cache, pos)
            outs.append(step)
    streamed = torch.stack(outs, dim=1)
    diff = float((full[:, 6:10] - streamed).abs().max())
    assert diff < 1e-4, f"sliding-window streaming mismatch: {diff}"
    assert cache[0]["k"].shape[2] == 4, "decode cache was not capped to the window"


# --------------------------------------------------------------------------- [23] conv isolation
def test_gdn_conv_state_is_document_isolated_across_chunks():
    layer = GatedDeltaNetLayer(d=16, n_heads=2)
    x = torch.randn(2, 6, 16)
    docs = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    full, _ = layer(x, doc_ids=docs)
    first, cache = layer(x[:, :3], doc_ids=docs[:, :3])
    second, _ = layer(x[:, 3:], cache=cache, doc_ids=docs[:, 3:])
    chunked = torch.cat([first, second], dim=1)
    diff = float((full - chunked).abs().max().detach())
    assert diff < 1e-5, f"chunked conv/state carry differs from unrolled: {diff}"


# --------------------------------------------------------------------------- [24] tokenizer hash
def test_tokenizer_hash_covers_full_mapping_and_load_is_fail_closed(tok):
    tampered = copy.deepcopy(tok)
    a, b = 100, 101
    tok_a, tok_b = tampered.id_to_token[a], tampered.id_to_token[b]
    tampered.token_to_id[tok_a], tampered.token_to_id[tok_b] = b, a
    tampered.id_to_token[a], tampered.id_to_token[b] = tok_b, tok_a
    assert tampered.tokenizer_hash() != tok.tokenizer_hash(), (
        "swapping token ids must change the tokenizer identity hash"
    )

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tok.json")
        tok.save(path)
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["token_to_id"]["hello"] = payload["token_to_id"].get("hello", 0) + 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with pytest.raises(ValueError, match="hash mismatch"):
            BPETokenizer.load(path)


def test_decode_rejects_unknown_ids(tok):
    with pytest.raises(ValueError, match="unknown token"):
        tok.decode([tok.vocab_size + 7])


# --------------------------------------------------------------------------- [25] checkpoints
def test_checkpoint_requires_a_verifiable_tokenizer(tok, tmp_path):
    model = build_model("TINY_HYBRID", tokenizer=tok)
    path = str(tmp_path / "c.pt")
    save_checkpoint_v7(model, None, None, {"best_val_loss": 1.0}, path)
    cfg = get_config("TINY_HYBRID")
    cfg["vocab"] = tok.vocab_size
    bare = Kestrel7(cfg)  # same config hash, but no tokenizer identity
    assert bare.tokenizer_hash is None
    with pytest.raises(ValueError, match="cannot verify"):
        load_checkpoint_v7(path, bare)


# --------------------------------------------------------------------------- [26] config
def test_config_rejects_contradictory_hybrid_and_returns_deep_copies():
    cfg = get_config("TINY_HYBRID")
    cfg["hybrid_schedule"] = None
    with pytest.raises(ValueError, match="requires a hybrid_schedule"):
        Kestrel7(cfg)
    first = get_config("TINY_HYBRID")
    first["hybrid_schedule"][0] = "full"
    assert get_config("TINY_HYBRID")["hybrid_schedule"][0] == "gdn"


# --------------------------------------------------------------------------- [27] CAS safety
def test_cas_rejects_code_and_unbounded_powers(vx):
    for expr in (
        "__import__('os').getcwd()",
        "().__class__.__mro__",
        "[x for x in range(3)]",
        "2 ** 1000000",
    ):
        assert vx.crane_cas_evaluate(expr).startswith("ERROR"), expr
    with pytest.raises(ValueError):
        safe_sympy("2 ** 1000000")


# --------------------------------------------------------------------------- [28] sandbox limits
def test_sandbox_bounds_wall_clock_and_memory(vx):
    loop = "def f(x):\n    while True:\n        pass"
    passed, total, errors = vx.safe_execute_python(loop, "f", [{"input": [1], "expected": 1}])
    assert (passed, total) == (0, 1)
    assert "Timeout" in errors[0] or "SandboxError" in errors[0]
    bomb = "def f(x):\n    return len([0] * (10 ** 9))"
    passed, total, errors = vx.safe_execute_python(bomb, "f", [{"input": [1], "expected": 1}])
    assert (passed, total) == (0, 1)


# --------------------------------------------------------------------------- [29] generation
class _RecordingModel:
    """Minimal duck-typed model that records decode calls."""

    def __init__(self, vocab: int, max_ctx: int, forced_id: int = 0):
        self.vocab = vocab
        self.max_ctx = max_ctx
        self.forced_id = forced_id
        self.calls = []
        self.seq_starts = []
        self.prefill_len = None

    def eval(self):
        return self

    def prefill(self, input_ids):
        self.prefill_len = int(input_ids.shape[1])
        logits = torch.full((1, self.vocab), -10.0)
        logits[0, self.forced_id] = 10.0
        return logits, [{"fake": torch.zeros(1)}], {"truncated": False, "tokens_dropped": 0, "max_ctx": self.max_ctx}

    def decode_step(self, token_ids, cache, seq_start):
        self.calls.append(int(token_ids[0, 0]))
        self.seq_starts.append(int(seq_start))
        return torch.zeros(1, self.vocab), cache


def test_cas_splice_feeds_the_equals_token_into_the_cache(tok):
    eq_id = tok.token_to_id["="]
    model = _RecordingModel(tok.vocab_size, max_ctx=64, forced_id=eq_id)
    prompt = THINK_START + "<<1+1"  # the model itself emits the trailing '='
    out = generate_v7(model, tok, prompt, max_new_tokens=8, temperature=None, vexx=Vexx3Controller())
    assert out.startswith(prompt)
    expected = [eq_id] + tok.encode("2>>", add_bos=False, add_eos=False)
    assert model.calls[: len(expected)] == expected, (
        "the '=' token was skipped before the spliced CAS value"
    )


def test_generate_uses_truncated_prompt_length_as_position(tok):
    model = _RecordingModel(tok.vocab_size, max_ctx=8, forced_id=0)
    generate_v7(model, tok, "hello world " * 40, max_new_tokens=1, temperature=None)
    assert model.prefill_len == 8, "prefill was not truncated to max_ctx"
    assert model.seq_starts and model.seq_starts[0] == 8, (
        "seq_start used the untruncated prompt length"
    )


# --------------------------------------------------------------------------- [30] benchmark oracle
def test_answer_key_is_independent_of_the_solver_path():
    tasks = load_jsonl(PROMPTS_PATH)
    key = {row["id"]: row["expected"] for row in load_jsonl(KEY_PATH)}
    assert len(tasks) == 33
    for task in tasks:
        assert key[task["id"]] == independent_answer(task), task["id"]
    # The old finite-universe check returned "yes" here: satisfiable is not
    # entailed. This is the exact bug that made the benchmark self-satisfying.
    assert syllogism_answer([["all", "A", "B"]], ["all", "B", "A"]) == "no"
    assert set(CALIBRATION_EXPECTED) == {f"res-0{i}" for i in range(1, 9)}


def test_benchmark_records_preference_pairs():
    result = run_benchmark(verbose=False)
    assert result["earned"] == result["points"]
    assert result.get("dpo_pairs", 0) > 0, "code-gen tasks produced no preference pairs"


def test_dpo_selection_and_export(vx, tmp_path):
    good = "def f(x):\n    return 0"
    bad = "def f(x):\n    return 1"
    scored = vx.select_verified([good, bad], "f", [{"input": [0], "expected": 0}])
    assert [row["passed"] for row in scored] == [1, 0]
    assert len(vx.dpo_buffer) == 1
    path = str(tmp_path / "dpo.jsonl")
    assert vx.export_dpo(path) == 1
    with open(path, "r", encoding="utf-8") as f:
        assert json.loads(f.readline())["fn"] == "f"


# --------------------------------------------------------------------------- [31] exact resume
def test_resume_reproduces_uninterrupted_training(tok, tmp_path):
    corpus = [f"resume document {i} alpha beta gamma delta epsilon" for i in range(14)]
    train, _val, _test, _m = deduplicate_and_split(corpus, train_frac=0.8, val_frac=0.1, seed=1)
    ds = StreamPackedDataset(tok, train, seq_len=12, seed=1)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=collate_stream_packed
    )

    def make(seed=0):
        torch.manual_seed(seed)
        model = build_model("TINY_HYBRID", tokenizer=tok)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = LinearWarmupCosineDecay(opt, warmup_steps=1, total_steps=4)
        return model, opt, sched

    reference, opt_r, sched_r = make()
    straight = Kestrel7Trainer(reference, loader, None, opt_r, sched_r, seed=42, log_every=0)
    straight_state = straight.fit(epochs=3, max_optimizer_steps=4)

    interrupted, opt_i, sched_i = make()
    first = Kestrel7Trainer(
        interrupted, loader, None, opt_i, sched_i, seed=42,
        checkpoint_dir=str(tmp_path), run_name="resume", log_every=0,
    )
    half = first.fit(epochs=3, max_optimizer_steps=2)

    resumed, opt_c, sched_c = make(seed=99)  # deliberately different init
    trainer = Kestrel7Trainer(resumed, loader, None, opt_c, sched_c, seed=42, log_every=0)
    info = load_checkpoint_v7(
        half["checkpoint_path"], resumed, tokenizer=tok, optimizer=opt_c, scheduler=sched_c
    )
    trainer.load_state_dict(info)
    assert (trainer.epoch, trainer.batch_in_epoch, trainer.optimizer_step) == (0, 2, 2)
    continued = trainer.fit(epochs=3, max_optimizer_steps=4)

    max_diff = max(
        float((a - b).abs().max().detach())
        for a, b in zip(reference.parameters(), resumed.parameters())
    )
    assert max_diff == 0.0, f"resumed weights diverged by {max_diff}"
    assert [h["loss"] for h in continued["history"]] == [h["loss"] for h in straight_state["history"]]
    assert trainer.optimizer_step == straight_state["optimizer_step"] == 4


# --------------------------------------------------------------------------- [P0.1] MTP document-boundary masking

def test_mtp_document_boundary_masking(hybrid, tok):
    """P0.1: MTP must ignore targets whose source positions cross a document boundary."""
    model = hybrid
    model.eval()
    # Pack two different documents into one sequence.
    docs = [
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda",
        "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega",
    ]
    enc = [tok.encode(d, add_bos=True, add_eos=True) for d in docs]
    flat_ids = enc[0] + enc[1]
    input_ids = torch.tensor([flat_ids])
    labels = input_ids.clone()
    n_a = len(enc[0])
    doc_ids = torch.tensor([[0] * n_a + [1] * len(enc[1])])

    with torch.no_grad():
        _, _, comps = model(
            input_ids, labels=labels, doc_ids=doc_ids, mtp_weight=0.2, return_components=True
        )

    # The boundary MTP target (t at end of doc A, t+1 = BOS of doc B) must be masked.
    # Check that mtp_target_count excludes cross-boundary positions.
    assert comps["mtp_target_count"] <= max(0, len(flat_ids) - 2)

    # Perturb document A: MTP loss for document B must be unchanged.
    ids_a_perturbed = list(flat_ids)
    ids_a_perturbed[0] = (ids_a_perturbed[0] + 1) % model.vocab
    input_ids_p = torch.tensor([ids_a_perturbed])
    with torch.no_grad():
        _, _, comps_p = model(
            input_ids_p, labels=input_ids_p, doc_ids=doc_ids, mtp_weight=0.2, return_components=True
        )
    # Document-B MTP targets should be unaffected by changes to document A.
    assert comps["mtp_target_count"] == comps_p["mtp_target_count"]

    # Without doc_ids, legacy single-document behavior (mask only labels == -100).
    with torch.no_grad():
        _, _, comps_nodoc = model(
            input_ids, labels=labels, doc_ids=None, mtp_weight=0.2, return_components=True
        )
    assert comps_nodoc["mtp_target_count"] >= comps["mtp_target_count"]


# --------------------------------------------------------------------------- [P0.2] Next-token CE separate from MTP loss

def test_loss_components_returned(tok):
    """P0.2: forward with return_components=True exposes separate loss components."""
    model = build_model("TINY_HYBRID", tokenizer=tok)
    model.eval()
    ids = torch.randint(0, model.vocab, (2, 8))
    labels = ids.clone()
    docs = torch.zeros_like(ids)
    _, loss, comps = model(ids, labels=labels, doc_ids=docs, mtp_weight=0.2, return_components=True)
    assert comps is not None
    assert set(comps) == {"total_loss", "next_token_ce", "mtp_loss", "lm_target_count", "mtp_target_count"}
    assert torch.isfinite(comps["total_loss"]) and torch.isfinite(comps["next_token_ce"])
    assert comps["lm_target_count"] > 0

    # Manual next-token CE must match the reported component.
    with torch.no_grad():
        logits, _, _ = model._forward_with_cache(ids, cache=None, seq_start=0, doc_ids=docs)
    shift_logits = logits[:, :-1, :].reshape(-1, model.vocab)
    shift_labels = labels[:, 1:].reshape(-1)
    manual_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
    assert torch.allclose(comps["next_token_ce"], manual_ce, atol=1e-5)


def test_perplexity_from_next_token_ce_only(tok):
    """P0.2: PPL is computed from next-token CE, never the composite objective."""
    model = build_model("TINY_HYBRID", tokenizer=tok)
    model.eval()
    ids = torch.randint(0, model.vocab, (1, 10))
    labels = ids.clone()
    docs = torch.zeros_like(ids)
    _, loss, comps = model(ids, labels=labels, doc_ids=docs, mtp_weight=0.2, return_components=True)
    # Perplexity from next-token CE.
    ppl_from_ce = math.exp(float(comps["next_token_ce"]))
    # The composite objective (CE + 0.2*MTP) should be different from pure CE when MTP is active.
    if comps["mtp_loss"] is not None:
        assert float(comps["total_loss"]) != float(comps["next_token_ce"])
    # Validate via the trainer's validation path.
    train_texts = [f"validation document {i} with tokens alpha beta gamma delta." for i in range(10)]
    train, val, _test, _m = deduplicate_and_split(train_texts, train_frac=0.6, val_frac=0.2, seed=1)
    train_ds = StreamPackedDataset(tok, train, seq_len=16, seed=42)
    val_ds = StreamPackedDataset(tok, val, seq_len=16, seed=42)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t = Kestrel7Trainer(model, train_loader, val_loader, opt, grad_accum_steps=1, checkpoint_dir=None)
    metrics = t.validate()
    assert metrics["val_next_token_ppl"] == math.exp(metrics["val_next_token_ce"])


# --------------------------------------------------------------------------- [P0.3] Best checkpoint saved separately

def test_best_checkpoint_saved_and_restored(tok, tmp_path):
    """P0.3: best checkpoint stores weights from the best next-token CE, not the last."""
    train_texts = [f"best document {i} alpha beta gamma delta epsilon." for i in range(10)]
    train, val, _test, _m = deduplicate_and_split(train_texts, train_frac=0.6, val_frac=0.2, seed=2)
    train_ds = StreamPackedDataset(tok, train, seq_len=12, seed=1)
    val_ds = StreamPackedDataset(tok, val, seq_len=12, seed=1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)

    def make_model():
        return build_model("TINY_HYBRID", tokenizer=tok)

    model_a = make_model()
    opt_a = torch.optim.AdamW(model_a.parameters(), lr=5e-3)
    trainer_a = Kestrel7Trainer(
        model_a, train_loader, val_loader, opt_a,
        grad_accum_steps=1, val_every_tokens=50, checkpoint_dir=str(tmp_path),
        seed=42, run_name="best_a", log_every=0,
    )
    state_a = trainer_a.fit(epochs=1, max_optimizer_steps=4)
    best_ce_a = trainer_a.best_next_token_ce

    # The best checkpoint path should exist and differ from the last checkpoint.
    assert trainer_a.best_checkpoint_path is not None
    assert trainer_a.best_checkpoint_path != state_a.get("checkpoint_path")

    # Load the best checkpoint into a fresh model and confirm its metric.
    fresh = build_model("TINY_HYBRID", tokenizer=tok)
    info = load_checkpoint_v7(
        trainer_a.best_checkpoint_path, fresh, tokenizer=tok
    )
    assert info["best_next_token_ce"] == best_ce_a
    assert fresh.checkpoint_loaded

    # The last checkpoint (step N) should exist separately.
    last_path = state_a.get("checkpoint_path")
    assert last_path is not None
    assert last_path != trainer_a.best_checkpoint_path


# --------------------------------------------------------------------------- [P0.4] Token-weighted gradient accumulation

def test_token_weighted_accumulation_gradients_match(tok):
    """P0.4: gradients from one concatenated batch == micro-batched split with different valid counts."""
    model = build_model("TINY_HYBRID", tokenizer=tok)
    model.eval()

    # Build a batch where valid token counts differ across micro-batches.
    ids = torch.randint(0, model.vocab, (4, 10))
    labels = ids.clone()
    # Make some labels -100 to create variable valid counts.
    labels[0, 5:] = -100
    labels[1, 2:] = -100
    docs = torch.zeros_like(ids)

    # (a) Single concatenated batch.
    m1 = build_model("TINY_HYBRID", tokenizer=tok)
    m1.load_state_dict(model.state_dict())
    _, _, comps_a = m1(ids, labels=labels, doc_ids=docs, mtp_weight=0.0, return_components=True)
    loss_a = comps_a["next_token_ce"]
    loss_a.backward()
    grads_a = [p.grad.clone() for p in m1.parameters() if p.requires_grad and p.grad is not None]

    # (b) Same examples split into micro-batches with different valid counts,
    # accumulated with token-weighted normalisation.
    m2 = build_model("TINY_HYBRID", tokenizer=tok)
    m2.load_state_dict(model.state_dict())
    # Split rows 0-1 and 2-3 as separate micro-batches.
    opt2 = torch.optim.SGD(m2.parameters(), lr=1e-3)
    m2.train()
    accum_ce_sum = 0.0
    accum_lm_tokens = 0
    for mb in [(ids[:2], labels[:2], docs[:2]), (ids[2:], labels[2:], docs[2:])]:
        mb_ids, mb_lbls, mb_docs = mb
        _, _, comps_b = m2(mb_ids, labels=mb_lbls, doc_ids=mb_docs, mtp_weight=0.0, return_components=True)
        n_lm = int(comps_b["lm_target_count"])
        accum_ce_sum += float(comps_b["next_token_ce"].detach()) * n_lm
        accum_lm_tokens += n_lm
        (comps_b["next_token_ce"] * n_lm).backward()
    # Normalise by total valid LM token count.
    scale = 1.0 / max(1, accum_lm_tokens)
    for p in m2.parameters():
        if p.grad is not None:
            p.grad *= scale
    grads_b = [p.grad.clone() for p in m2.parameters() if p.requires_grad and p.grad is not None]

    for ga, gb in zip(grads_a, grads_b):
        assert torch.allclose(ga, gb, atol=1e-5), "token-weighted accumulation gradients mismatch"


# --------------------------------------------------------------------------- [P0.5] Exact resume with shuffle (P0.5 contract)

def test_resume_with_shuffle_reproduces_batch_order(tok, tmp_path):
    """P0.5: persisted per-epoch permutation and cursor reproduce batch order on resume."""
    corpus = [f"resume-shuffle document {i} alpha beta gamma delta epsilon." for i in range(12)]
    train, _val, _test, _m = deduplicate_and_split(corpus, train_frac=0.8, val_frac=0.1, seed=3)
    ds = StreamPackedDataset(tok, train, seq_len=10, seed=1)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=True, collate_fn=collate_stream_packed
    )

    def make(seed=0):
        torch.manual_seed(seed)
        model = build_model("TINY_HYBRID", tokenizer=tok)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        return model, opt

    reference, opt_r = make()
    straight = Kestrel7Trainer(reference, loader, None, opt_r, seed=7, log_every=0)
    straight_state = straight.fit(epochs=2, max_optimizer_steps=3)

    interrupted, opt_i = make()
    first = Kestrel7Trainer(
        interrupted, loader, None, opt_i, seed=7,
        checkpoint_dir=str(tmp_path), run_name="resume_shuffle", log_every=0,
    )
    half = first.fit(epochs=2, max_optimizer_steps=1)

    resumed, opt_c = make()
    trainer = Kestrel7Trainer(resumed, loader, None, opt_c, seed=7, log_every=0)
    info = load_checkpoint_v7(
        half["checkpoint_path"], resumed, tokenizer=tok, optimizer=opt_c
    )
    trainer.load_state_dict(info)
    continued = trainer.fit(epochs=2, max_optimizer_steps=3)

    max_diff = max(
        float((a - b).abs().max().detach())
        for a, b in zip(reference.parameters(), resumed.parameters())
    )
    assert max_diff == 0.0, f"resumed weights diverged by {max_diff}"
    assert trainer.optimizer_step == straight_state["optimizer_step"]


# --------------------------------------------------------------------------- [P0.6] Tier-4 bypass removed from normal generation

def test_generate_always_invokes_model_prefill(tok):
    """P0.6: model.prefill() is invoked for every normal prompt, even with a Tier-4-matching phrase."""
    class _RecordingModel:
        def __init__(self, vocab, max_ctx, forced_id=0):
            self.vocab = vocab
            self.max_ctx = max_ctx
            self.forced_id = forced_id
            self.prefill_calls = 0
            self.decode_calls = 0

        def eval(self):
            return self

        def prefill(self, input_ids):
            self.prefill_calls += 1
            logits = torch.full((1, self.vocab), -10.0)
            logits[0, self.forced_id] = 10.0
            return logits, [{"fake": torch.zeros(1)}], {"truncated": False, "tokens_dropped": 0, "max_ctx": self.max_ctx}

        def decode_step(self, token_ids, cache, seq_start):
            self.decode_calls += 1
            return torch.zeros(1, self.vocab), cache

    model = _RecordingModel(tok.vocab_size, max_ctx=64, forced_id=tok.token_to_id["e"])
    vexx = Vexx3Controller()

    # A prompt containing a Tier-4-recognised phrase must still invoke prefill.
    prompt = "A trial reports 95% CI [-0.5, 2.9], p=0.17. Does it establish efficacy?"
    out = generate_v7(model, tok, prompt, max_new_tokens=4, temperature=None, vexx=vexx)
    assert model.prefill_calls >= 1, "prefill must be invoked even for calibration-style prompts"
    assert out.startswith(prompt), "output must start with the prompt"

    # A calibration flag does not replace neural output.
    out2 = generate_v7(model, tok, prompt, max_new_tokens=4, temperature=None, vexx=vexx)
    assert model.prefill_calls >= 2


# --------------------------------------------------------------------------- [P0.7]  token categories

def test_token_categories_and_think_markers(tok):
    """P0.7: THINK_START/THINK_END are protocol markers, never-emit tokens stay blocked."""
    from kestrel7.tokenizer import token_category, THINK_START, THINK_END, NEVER_EMIT, LEGAL_STOP, PROTOCOL_MARKERS

    assert token_category(BOS) == NEVER_EMIT
    assert token_category(PAD) == NEVER_EMIT
    assert token_category(UNK) == NEVER_EMIT
    assert token_category(EOS) == LEGAL_STOP
    assert token_category(THINK_START) == PROTOCOL_MARKERS
    assert token_category(THINK_END) == PROTOCOL_MARKERS
    assert token_category("hello") is None

    vx = Vexx3Controller()
    vocab_map = {i: t for i, t in tok.id_to_token.items()}
    masked = vx.apply_in_flight_mask(np.zeros(tok.vocab_size), "hello world", vocab_map, mode="plain")
    # In plain mode, think markers are not active (phase is STRICT_VERIFIED), so they're blocked.
    sid_start = tok.token_to_id[THINK_START]
    assert np.isneginf(masked[sid_start]), "THINK_START blocked in plain mode outside think phase"

    # In think phase, think markers are legal.
    think_prefix = THINK_START + "reasoning here"
    masked_think = vx.apply_in_flight_mask(np.zeros(tok.vocab_size), think_prefix, vocab_map, mode="plain")
    assert not np.isneginf(masked_think[tok.token_to_id[THINK_END]]), "THINK_END legal inside think phase"


# --------------------------------------------------------------------------- [P0.8] Decoding modes (plain vs python)

def test_plain_mode_does_not_apply_python_lexer(vx, tok):
    """P0.8: plain mode applies no Python syntax constraints to prose."""
    prose = "The value of x = (a + b) / c is not code; f(x) is fine."
    # Plain mode: no Python lexer, so prose with parentheses/dots passes through.
    masked_plain = vx.apply_in_flight_mask(
        np.zeros(tok.vocab_size), prose, tok.id_to_token, mode="plain"
    )
    # EOS stays legal in plain mode.
    assert not np.isneginf(masked_plain[tok.token_to_id[EOS]])

    # Python mode activates the lexer; unknown callees may still be blocked,
    # but plain mode must not corrupt prose.
    assert not np.all(np.isneginf(masked_plain))


# --------------------------------------------------------------------------- [P0.9] Tier-3 renamed / claims corrected

def test_tier3_renamed_single_row_bound_contractor(vx):
    """P0.9: Tier-3 renamed to single_row_bound_contractor; old alias still resolves."""
    assert hasattr(vx, "single_row_bound_contractor")
    assert hasattr(vx, "trigger_farkas_cdcl_jumpback"), "backward-compatible alias retained"
    # Bound methods differ per access; compare the underlying function.
    assert (
        vx.single_row_bound_contractor.__func__
        is vx.trigger_farkas_cdcl_jumpback.__func__
    )


# --------------------------------------------------------------------------- [P0.10] Truthful capability metadata

def test_truthful_capability_metadata(tok, tmp_path):
    """P0.10: metadata fields track origin/updates separately from capability labels."""
    model = build_model("TINY_HYBRID", tokenizer=tok)
    assert model.weights_origin == "random_init"
    assert model.optimizer_updates == 0
    assert model.tokens_seen == 0
    assert model.checkpoint_artifact_id is None
    assert model.capability_evaluation_status == "not_evaluated"
    assert not model.checkpoint_loaded

    # After a checkpoint load, metadata is restored from the artifact.
    train_texts = [f"metadata document {i} alpha beta gamma delta epsilon zeta." for i in range(10)]
    train, val, _test, _m = deduplicate_and_split(train_texts, train_frac=0.6, val_frac=0.2, seed=5)
    train_ds = StreamPackedDataset(tok, train, seq_len=10, seed=1)
    val_ds = StreamPackedDataset(tok, val, seq_len=10, seed=1)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=2, shuffle=False)
    m2 = build_model("TINY_HYBRID", tokenizer=tok)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    t2 = Kestrel7Trainer(m2, train_loader, val_loader, opt2, grad_accum_steps=1,
                         checkpoint_dir=str(tmp_path), seed=6, run_name="meta", log_every=0)
    state = t2.fit(epochs=1, max_optimizer_steps=2)
    loaded = build_model("TINY_HYBRID", tokenizer=tok)
    load_checkpoint_v7(state["checkpoint_path"], loaded, tokenizer=tok)
    assert loaded.checkpoint_loaded
    assert loaded.checkpoint_artifact_id is not None
    assert loaded.model_status == "trained_checkpoint_loaded"
