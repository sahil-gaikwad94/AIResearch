#!/usr/bin/env python3
"""Kestrel-7 end-to-end run: train, validate, save a verified checkpoint, generate with VEXX.

Usage:
    python3 research/day7/run_kestrel7.py --preset TINY_HYBRID --steps 50

The corpus is deliberately tiny and self-contained; this is a *mechanism* run
that proves the pipeline (dedupe -> split -> tokenizer -> model -> training ->
validation -> hash-verified checkpoint -> VEXX-guided decode) is wired end to
end. It is not a capability claim: 50 optimizer steps on a synthetic corpus
cannot teach a language model anything useful, and the script prints the
validation loss so the gap is visible rather than implied.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kestrel7.config import get_config
from kestrel7.data import StreamPackedDataset, collate_stream_packed, deduplicate_and_split
from kestrel7.generate import generate_v7
from kestrel7.model import build_model, count_params, gdn_state_bytes, kv_cache_bytes
from kestrel7.tokenizer import BPETokenizer
from kestrel7.train import Kestrel7Trainer, LinearWarmupCosineDecay
from kestrel7.checkpoint import load_checkpoint_v7
from kestrel7.vexx import Vexx3Controller


def synthetic_corpus() -> List[str]:
    """A small, deduplicated-by-construction code/math/reasoning corpus."""
    return [
        "def add(a, b):\n    return a + b\n",
        "def square(x):\n    return x * x\n",
        "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n",
        "def is_even(n):\n    return n % 2 == 0\n",
        "def max_of_two(a, b):\n    return a if a > b else b\n",
        "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
        "def mean(xs):\n    return sum(xs) / len(xs)\n",
        "def unique(xs):\n    return list(set(xs))\n",
        "def reverse(s):\n    return s[::-1]\n",
        "def factorial(n):\n    out = 1\n    for i in range(2, n + 1):\n        out *= i\n    return out\n",
        "2 + 2 = 4",
        "37 * 48 - 19 * 32 = 1168",
        "(5/6 - 1/4) / (7/12) = 1",
        "7 ** 100 mod 13 = 9",
        "the integral of 3*x^2 + 2*x from 0 to 2 is 12",
        "the variance of [0, 0, 2, 4, 4] is 3.2",
        "all A are B and no B are C implies no A are C",
        "if P implies Q and not Q then not P",
        "0.999999... is equal to 1",
        "a 95% confidence interval that crosses zero does not establish an effect",
        "source B is newer than source A and supersedes it",
        "an optional online poll is not a representative sample",
        "an unverified award cannot be named",
        "a fictional city has an unknown population",
        "unresolved contradictory figures are reported as a conflict",
        "a correlation with a confounder does not establish causation",
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "The mitochondria is the powerhouse of the cell.",
        "E = m c^2 in natural units.",
        "Breadth-first search visits nodes in order of distance.",
        "Kadane's algorithm tracks the best subarray sum ending here.",
        "Merging intervals sorts by start and extends the current end.",
        "A stack validates bracket nesting in linear time.",
        "DeltaNet updates a fixed-size state with a delta rule.",
        "RoPE rotates query and key pairs by position-dependent angles.",
        "Grouped-query attention shares key and value heads.",
        "A zero-centered RMSNorm starts at the identity.",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="TINY_HYBRID")
    ap.add_argument("--steps", type=int, default=50, help="optimizer steps")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = get_config(args.preset)

    # 1. Deduplicate -> split -> train the tokenizer on the TRAIN split only.
    corpus = synthetic_corpus()
    train_texts, val_texts, test_texts, manifest = deduplicate_and_split(
        corpus, train_frac=0.7, val_frac=0.2, seed=args.seed
    )
    print(
        f"[data] raw={manifest['raw_count']} unique={manifest['unique_count']} "
        f"train={manifest['n_train']} val={manifest['n_val']} test={manifest['n_test']} "
        f"(zero overlap asserted)"
    )
    tok = BPETokenizer.train(train_texts, vocab_size=min(cfg["vocab"], 384))

    # 2. Build the model with vocab == tokenizer.vocab_size (strict contract).
    cfg["vocab"] = tok.vocab_size
    model = build_model(cfg, tokenizer=tok)
    total, trainable = count_params(model)
    print(
        f"[model] preset={args.preset} total={total:,} trainable={trainable:,} "
        f"vocab={model.vocab} layers={model.layers_n} d={model.d}"
    )
    print(
        f"[cache] full-attn KV bytes @ctx={model.max_ctx}: {kv_cache_bytes(model):,} | "
        f"GDN state bytes: {gdn_state_bytes(model):,}"
    )

    # 3. Data loaders.
    train_ds = StreamPackedDataset(tok, train_texts, seq_len=args.seq_len, seed=args.seed)
    val_ds = StreamPackedDataset(tok, val_texts, seq_len=args.seq_len, seed=args.seed)
    test_ds = StreamPackedDataset(tok, test_texts, seq_len=args.seq_len, seed=args.seed)
    if len(val_ds) == 0:
        print("[warn] empty validation split; disabling validation is NOT silent - it is reported")
    if len(test_ds) == 0:
        print("[warn] empty test split; the held-out number will not be reported as a result")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_stream_packed
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_stream_packed)
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_stream_packed)
        if len(test_ds) > 0
        else None
    )
    print(
        f"[data] packed examples: train={len(train_ds)} val={len(val_ds)} "
        f"test={len(test_ds)} seq_len={args.seq_len}"
    )

    # Initialization diagnostics: logit scale and initial CE versus the uniform
    # baseline. The audit found logit std 21.55 and a 251 CE at init; these
    # numbers make a regression visible immediately rather than at step 10.
    model.eval()
    with torch.no_grad():
        probe_ids = torch.randint(0, model.vocab, (2, 8))
        probe_logits, probe_loss = model(probe_ids, labels=probe_ids, mtp_weight=0.0)
    print(
        f"[init] embed_std={float(model.embed.weight.detach().std()):.4f} "
        f"logit_std={float(probe_logits.std()):.4f} "
        f"ce={float(probe_loss):.3f} uniform={math.log(model.vocab):.3f}"
    )

    # 4. Optimizer / schedule / trainer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    scheduler = LinearWarmupCosineDecay(
        optimizer, warmup_steps=5, total_steps=max(1, args.steps)
    )
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    trainer = Kestrel7Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=1.0,
        grad_accum_steps=1,
        val_every_tokens=2_000,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        run_name=f"kestrel7_{args.preset.lower()}",
        log_every=10,
    )
    trainer.dataset_manifest = manifest
    steps_per_epoch = max(1, math.ceil(len(train_ds) / args.batch))
    epochs = max(1, math.ceil(args.steps / steps_per_epoch)) + 1
    state = trainer.fit(epochs=epochs, max_optimizer_steps=args.steps)
    print(
        f"[train] optimizer_steps={state['optimizer_step']} tokens={state['token_count']} "
        f"best_val_loss={state['best_val_loss']:.4f} "
        f"val={state.get('val_metrics', {})}"
    )
    if not state.get("checkpoint_path"):
        print("[error] no checkpoint was written")
        return 1

    # 5. Reload into a fresh model and verify both hashes fail closed.
    fresh = build_model(cfg, tokenizer=tok)
    loaded_state = load_checkpoint_v7(
        state["checkpoint_path"], fresh, tokenizer=tok
    )
    print(
        f"[checkpoint] {state['checkpoint_path']}\n"
        f"[checkpoint] status={fresh.model_status} loaded={fresh.checkpoint_loaded} "
        f"best_val_loss={loaded_state['best_val_loss']:.4f} "
        f"config_hash={fresh.config_hash()[:12]}... tokenizer_hash={fresh.tokenizer_hash[:12]}..."
    )
    if not fresh.checkpoint_loaded or loaded_state["best_val_loss"] != state["best_val_loss"]:
        print("[error] checkpoint reload did not reproduce the trained state")
        return 1

    # 5b. Held-out test split: reported separately and never trained on.
    if test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        print(
            f"[test] loss={test_metrics['loss']:.4f} ppl={test_metrics['ppl']:.4f} "
            f"tokens={test_metrics['tokens']} (held-out; never trained)"
        )

    # 6. VEXX-guided generation: a strict code prompt, a plain prompt, and a
    #    Tier-4 calibration prompt that bypasses the weights entirely.
    vexx = Vexx3Controller(
        symbol_table={
            "add": {"min_arity": 2, "max_arity": 2, "return_type": "any"},
            "square": {"min_arity": 1, "max_arity": 1, "return_type": "any"},
            "fib": {"min_arity": 1, "max_arity": 1, "return_type": "any"},
        }
    )
    prompts = [
        "def add(a, b):",
        "The capital of France is",
        "A trial reports 95% CI [-0.5, 2.9], p=0.17. Does it establish efficacy?",
    ]
    print("[generate] VEXX-guided (greedy; temperature=None contract)")
    for prompt in prompts:
        result = generate_v7(
            fresh, tok, prompt, max_new_tokens=args.max_new, temperature=None, vexx=vexx
        )
        print(f"  PROMPT: {prompt!r}\n  OUTPUT: {result.continuation!r}")

    print(
        "[done] trained, validated, checkpoint verified, and VEXX-guided generation "
        "completed without error"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
