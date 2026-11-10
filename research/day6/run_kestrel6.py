#!/usr/bin/env python3
"""Kestrel-6 end-to-end run script.

Smoke-train the dense BASELINE on a tiny synthetic corpus, then generate.
Intended to be run once Python + PyTorch are available in this environment.

Usage:
    python3 research/day6/run_kestrel6.py [--preset TINY] [--steps 50]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel6.config import get_config
from kestrel6.model import build_model, count_params
from kestrel6.tokenizer import BPETokenizer
from kestrel6.data import PackedDataset, train_val_test_split
from kestrel6.train import Trainer, LinearWarmupCosineDecay
from kestrel6.generate import greedy_generate
from kestrel6.checkpoint import save_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="TINY")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = get_config(args.preset)

    # Tiny synthetic corpus so the run is self-contained and fast.
    corpus = [
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "2 + 2 = 4.",
        "def add(a, b): return a + b",
        "The quick brown fox jumps over the lazy dog.",
        "To be, or not to be, that is the question.",
        "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        "The mitochondria is the powerhouse of the cell.",
        "E = m c^2.",
        "Once upon a time there was a small model named Kestrel.",
    ] * 50

    # P1-01: deduplicate by SHA-256 content hash BEFORE splitting, then train
    # the BPE tokenizer on the train split only. Repeating the corpus 50x and
    # splitting afterwards leaked every validation document into training.
    seen = set()
    unique_corpus = []
    for text in corpus:
        h = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_corpus.append(text)
    train_texts, val_texts, test_texts = train_val_test_split(
        unique_corpus, train_frac=0.7, val_frac=0.2, seed=args.seed
    )
    print(
        f"[kestrel6] data: raw={len(corpus)} unique={len(unique_corpus)} "
        f"train={len(train_texts)} val={len(val_texts)} test={len(test_texts)}"
    )

    tok = BPETokenizer.train(train_texts, vocab_size=min(cfg["vocab"], 512))
    # P0-05: the model's logit space must match the tokenizer exactly.
    cfg["vocab"] = tok.vocab_size
    model = build_model(cfg)
    total, trainable = count_params(model)
    print(
        f"[kestrel6] preset={args.preset} total={total:,} trainable={trainable:,} "
        f"vocab={cfg['vocab']} (tokenizer={tok.vocab_size})"
    )

    train_ds = PackedDataset(tok, train_texts, seq_len=args.seq_len, seed=args.seed)
    val_ds = PackedDataset(tok, val_texts, seq_len=args.seq_len, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = LinearWarmupCosineDecay(optim, warmup_steps=10, total_steps=args.steps)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optim,
        scheduler=sched,
        device=device,
        grad_clip=1.0,
        grad_accum_steps=1,
        log_every=10,
        val_every_tokens=10_000_000,
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),
        seed=args.seed,
        run_name="kestrel6",
    )

    trainer.fit(epochs=1, max_steps=args.steps)

    model.eval()
    with torch.no_grad():
        for prompt in ["The capital of France is", "def add(", "2 + 2 ="]:
            out = greedy_generate(model, tok, prompt, max_new_tokens=12, add_bos=True)
            print(f"PROMPT: {prompt!r} -> {out!r}")


if __name__ == "__main__":
    main()
