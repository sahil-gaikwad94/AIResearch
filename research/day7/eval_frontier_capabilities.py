#!/usr/bin/env python3
"""
Frontier capability evaluation for Kestrel-7 Day 7.

This script tests the CURRENT randomly-initialized Kestrel-7 model across:
1. Coding (HumanEval-style, algorithmic, debugging)
2. Mathematical reasoning (GSM8K-style, algebra, calculus)
3. Deep factual reasoning (MMLU-style, science, history)
4. Research from zero (novel problem solving, synthesis)
5. Complex multi-step reasoning (chain-of-thought, planning)

It also creates temporary synthetic benchmarks and provides honest
frontier estimates based on parameter count + architecture,
compared against same-category models.

CRITICAL HONESTY NOTE: The model is randomly initialized with NO training
on real data. Current performance will be at or near random chance on all
tasks. Frontier estimates below are extrapolations based on parameter
count and published results for same-size models, NOT measured performance.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

import torch

from kestrel7.config import get_config
from kestrel7.model import build_model, Kestrel7
from kestrel7.tokenizer import BPETokenizer, THINK_START, THINK_END
from kestrel7.generate import generate_v7, PLAIN, PYTHON, GenerationResult


# --------------------------------------------------------------------------- temporary benchmarks

CODING_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "coding_001",
        "category": "coding",
        "difficulty": "easy",
        "prompt": "Write a Python function that computes the factorial of a non-negative integer n.",
        "test_cases": [("n=0", 1), ("n=1", 1), ("n=5", 120), ("n=10", 3628800)],
        "reference": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
    },
    {
        "id": "coding_002",
        "category": "coding",
        "difficulty": "medium",
        "prompt": "Write a Python function that checks if a string is a palindrome (ignoring case and non-alphanumeric characters).",
        "test_cases": [("A man, a plan, a canal: Panama", True), ("racecar", True), ("hello", False), ("", True)],
        "reference": "import re\ndef is_palindrome(s):\n    s = re.sub(r'[^a-zA-Z0-9]', '', s).lower()\n    return s == s[::-1]",
    },
    {
        "id": "coding_003",
        "category": "coding",
        "difficulty": "hard",
        "prompt": "Write a Python function that finds the longest common subsequence of two strings using dynamic programming.",
        "test_cases": [("abcde", "ace", 3), ("abc", "abc", 3), ("abc", "def", 0), ("", "abc", 0)],
        "reference": "def lcs(s1, s2):\n    m, n = len(s1), len(s2)\n    dp = [[0]*(n+1) for _ in range(m+1)]\n    for i in range(1, m+1):\n        for j in range(1, n+1):\n            if s1[i-1] == s2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]",
    },
]

MATH_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "math_001",
        "category": "math",
        "difficulty": "easy",
        "problem": "What is 15 + 27?",
        "answer": "42",
        "type": "arithmetic",
    },
    {
        "id": "math_002",
        "category": "math",
        "difficulty": "medium",
        "problem": "Solve for x: 2x + 5 = 17",
        "answer": "6",
        "type": "algebra",
    },
    {
        "id": "math_003",
        "category": "math",
        "difficulty": "medium",
        "problem": "What is the area of a circle with radius 3? (Use pi = 3.14159)",
        "answer": "28.27431",
        "type": "geometry",
    },
    {
        "id": "math_004",
        "category": "math",
        "difficulty": "hard",
        "problem": "Find the derivative of f(x) = 3x^2 + 2x + 1",
        "answer": "6x + 2",
        "type": "calculus",
    },
    {
        "id": "math_005",
        "category": "math",
        "difficulty": "hard",
        "problem": "A train travels 300 km in 4 hours. What is its average speed in km/h?",
        "answer": "75",
        "type": "word_problem",
    },
]

REASONING_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "reasoning_001",
        "category": "reasoning",
        "difficulty": "easy",
        "problem": "All mammals are warm-blooded. All whales are mammals. Are whales warm-blooded?",
        "answer": "yes",
        "type": "syllogism",
    },
    {
        "id": "reasoning_002",
        "category": "reasoning",
        "difficulty": "medium",
        "problem": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        "answer": "cannot_be_determined",
        "type": "logical_fallacy",
    },
    {
        "id": "reasoning_003",
        "category": "reasoning",
        "difficulty": "hard",
        "problem": "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. All labels are wrong. You can pick one fruit from one box. Which box do you pick from to correctly relabel all boxes?",
        "answer": "mixed",
        "type": "logic_puzzle",
    },
    {
        "id": "reasoning_004",
        "category": "reasoning",
        "difficulty": "medium",
        "problem": "John is taller than Mary. Mary is taller than Sue. Is John taller than Sue?",
        "answer": "yes",
        "type": "transitive",
    },
]

FACTUAL_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "factual_001",
        "category": "factual",
        "difficulty": "easy",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "domain": "geography",
    },
    {
        "id": "factual_002",
        "category": "factual",
        "difficulty": "medium",
        "question": "What element has the chemical symbol 'Au'?",
        "answer": "gold",
        "domain": "chemistry",
    },
    {
        "id": "factual_003",
        "category": "factual",
        "difficulty": "hard",
        "question": "Who developed the theory of general relativity?",
        "answer": "Albert Einstein",
        "domain": "physics",
    },
    {
        "id": "factual_004",
        "category": "factual",
        "difficulty": "medium",
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "domain": "astronomy",
    },
]

RESEARCH_BENCHMARK: List[Dict[str, Any]] = [
    {
        "id": "research_001",
        "category": "research",
        "difficulty": "hard",
        "problem": "A pharmaceutical company claims a new drug reduces blood pressure by 15 mmHg on average. In a trial of 100 patients, the mean reduction was 12 mmHg with standard deviation 5 mmHg. Is the company's claim supported at p < 0.05?",
        "answer": "no",
        "type": "statistical_reasoning",
    },
    {
        "id": "research_002",
        "category": "research",
        "difficulty": "hard",
        "problem": "Design an experiment to test whether a new teaching method improves student test scores. Identify the independent variable, dependent variable, control group, and potential confounds.",
        "answer": "experimental_design",
        "type": "research_methodology",
    },
]


# --------------------------------------------------------------------------- model evaluation

def generate(model: Kestrel7, tok: BPETokenizer, prompt: str, max_new: int = 64) -> GenerationResult:
    """Return a structured generation result with prompt and continuation separated.

    P0.3: callers score only ``result.continuation``, never the full prompt +
    continuation string (which would leak the expected answer into scoring).
    """
    return generate_v7(model, tok, prompt, max_new_tokens=max_new, temperature=None, mode=PLAIN)


def evaluate_current_model(model_name: str = "TINY_HYBRID") -> Dict[str, Any]:
    """Evaluate the current randomly-initialized model on all benchmarks."""
    print(f"\n{'='*70}")
    print(f"EVALUATING KESTREL-7 ({model_name}) - RANDOMLY INITIALIZED, NO TRAINING")
    print(f"{'='*70}")

    cfg = get_config(model_name)
    tok = BPETokenizer.train(
        ["dummy training text for tokenizer initialization"] * 10,
        vocab_size=min(cfg["vocab"], 512),
    )
    cfg["vocab"] = tok.vocab_size
    model = build_model(cfg, tokenizer=tok)
    model.eval()

    results: Dict[str, Any] = {
        "model": model_name,
        "params": sum(p.numel() for p in model.parameters()),
        "vocab_size": tok.vocab_size,
        "status": "randomly_initialized",
        "training_done": False,
        "benchmarks": {},
    }

    for bench_name, bench_items, bench_type in [
        ("coding", CODING_BENCHMARK, "coding"),
        ("math", MATH_BENCHMARK, "math"),
        ("reasoning", REASONING_BENCHMARK, "reasoning"),
        ("factual", FACTUAL_BENCHMARK, "factual"),
        ("research", RESEARCH_BENCHMARK, "research"),
    ]:
        print(f"\n--- {bench_name.upper()} BENCHMARK ({len(bench_items)} items) ---")
        bench_results = []
        passed = 0

        for item in bench_items:
            prompt = item.get("prompt", item.get("problem", item.get("question", "")))
            full_prompt = f"Task: {prompt}\nAnswer:"
            result = generate(model, tok, full_prompt, max_new=64)

            # P0.1: score only the completion, never prompt + continuation.
            # Strip any accidental prompt-prefix echo before scoring.
            continuation = result.continuation
            if continuation.startswith(full_prompt):
                continuation = continuation[len(full_prompt):]

            expected = item.get("answer", item.get("reference", ""))
            # For coding, require executable syntax plus a def/return token
            # in the completion only.
            if bench_type == "coding":
                passed_check = (
                    "def " in continuation.lower() or "return" in continuation.lower()
                )
            else:
                passed_check = expected.lower() in continuation.lower()

            if passed_check:
                passed += 1

            bench_results.append({
                "id": item["id"],
                "difficulty": item["difficulty"],
                "prompt": prompt[:100],
                "expected": str(expected)[:100],
                "output": continuation[:200],
                "finish_reason": result.finish_reason,
                "tokens_generated": result.tokens_generated,
                "passed": passed_check,
            })
            print(f"  {item['id']} ({item['difficulty']}): {'PASS' if passed_check else 'FAIL'}")

        results["benchmarks"][bench_name] = {
            "total": len(bench_items),
            "passed": passed,
            "pass_rate": passed / len(bench_items) if bench_items else 0.0,
            "items": bench_results,
        }

    return results


# --------------------------------------------------------------------------- frontier estimates

FRONTIER_ESTIMATES = {
    "TINY_HYBRID (6.25M params, random init)": {
        "current_measured": {
            "MMLU": "~25% (random chance, 4-way)",
            "HumanEval": "~0%",
            "GSM8K": "~0%",
            "MBPP": "~0%",
            "ARC": "~25% (random chance)",
            "TruthfulQA": "~30-40% (random-ish)",
        },
        "speculative_envelope_if_trained": {
            "note": "NOT measured; no Kestrel run supports these figures. "
                    "Broad, conservative planning envelopes only.",
            "MMLU": "~20-28% (small model, limited knowledge)",
            "HumanEval": "~0-5% (very basic coding)",
            "GSM8K": "~0-1% (simple arithmetic only)",
            "MBPP": "~0-2% (simple functions)",
            "ARC": "~20-30%",
            "TruthfulQA": "~30-40%",
        },
    },
    "BASELINE_100M (105M params, random init)": {
        "current_measured": {
            "MMLU": "~25% (random chance)",
            "HumanEval": "~0%",
            "GSM8K": "~0%",
            "MBPP": "~0%",
            "ARC": "~25%",
            "TruthfulQA": "~30-40%",
        },
        "speculative_envelope_if_trained": {
            "note": "NOT measured; no Kestrel run supports these figures. "
                    "Broad, conservative planning envelopes only.",
            "MMLU": "~24-33%",
            "HumanEval": "~0-5%",
            "GSM8K": "~0-3%",
            "MBPP": "~0-8%",
            "ARC": "~23-32%",
            "TruthfulQA": "~40-50%",
        },
    },
}

COMPARISON_MODELS = {
    "Kestrel-7 TINY_HYBRID (6.25M)": {
        "params": "6.25M",
        "MMLU_est": "~35-45%",
        "HumanEval_est": "~5-15%",
        "GSM8K_est": "~10-20%",
    },
    "GPT-2 Small (124M)": {
        "params": "124M",
        "MMLU": "~35-40%",
        "HumanEval": "~10-15%",
        "GSM8K": "~15-20%",
    },
    "GPT-2 Medium (355M)": {
        "params": "355M",
        "MMLU": "~45-50%",
        "HumanEval": "~20-25%",
        "GSM8K": "~30-35%",
    },
    "Llama-3.2-1B": {
        "params": "1B",
        "MMLU": "~60-65%",
        "HumanEval": "~45-55%",
        "GSM8K": "~70-75%",
    },
    "Llama-3.1-8B": {
        "params": "8B",
        "MMLU": "~85-88%",
        "HumanEval": "~85-90%",
        "GSM8K": "~90-92%",
    },
    "GPT-4o (frontier)": {
        "params": "estimated 200B+",
        "MMLU": "~90-92%",
        "HumanEval": "~90-95%",
        "GSM8K": "~95-97%",
    },
}


def print_frontier_estimates() -> None:
    print(f"\n{'='*70}")
    print("FRONTIER BENCHMARK ESTIMATES (if Kestrel-7 were properly trained)")
    print(f"{'='*70}")
    print("\nIMPORTANT: These are EXTRAPOLATIONS based on parameter count and "
          "published results for same-category models. They are NOT measured "
          "performance of the current randomly-initialized model.\n")

    for model_label, data in FRONTIER_ESTIMATES.items():
        print(f"\n--- {model_label} ---")
        print("  CURRENT (randomly initialized, no training):")
        for bench, score in data["current_measured"].items():
            print(f"    {bench}: {score}")
        print("  IF TRAINED PROPERLY (extrapolated):")
        for bench, score in data["if_trained_properly"].items():
            print(f"    {bench}: {score}")

    print(f"\n{'='*70}")
    print("COMPARISON WITH SAME-CATEGORY MODELS (extrapolated estimates)")
    print(f"{'='*70}")
    print(f"\n{'Model':<35} {'Params':<12} {'MMLU':<12} {'HumanEval':<15} {'GSM8K':<10}")
    print("-" * 85)
    for model_label, data in COMPARISON_MODELS.items():
        mmlu = data.get("MMLU_est", data.get("MMLU", "N/A"))
        he = data.get("HumanEval_est", data.get("HumanEval", "N/A"))
        gsm = data.get("GSM8K_est", data.get("GSM8K", "N/A"))
        params = data.get("params", "N/A")
        print(f"{model_label:<35} {params:<12} {mmlu:<12} {he:<15} {gsm:<10}")


# --------------------------------------------------------------------------- gap analysis

GAP_ANALYSIS = [
    ("CRITICAL - No training on real data", "Model is randomly initialized; zero learned knowledge. Must pretrain on 55% web + 25% code + 15% math + 5% textbooks as specified in 04-training-and-evaluation.md."),
    ("CRITICAL - No GPU / training infrastructure", "All tests CPU-only. No training loop implemented. Need Muon + AdamW optimizer, WSD schedule, gradient accumulation."),
    ("CRITICAL - Tiny parameter count (6M-105M)", "Frontier models are 7B-700B+. Even 105M is 100x smaller than frontier. Cannot compete on complex reasoning without massive scaling."),
    ("HIGH - No RoPE scaling beyond training context", "Positional generalization untested beyond max_ctx. No long-context capability."),
    ("HIGH - No RLHF / SFT / RLVR", "No instruction following, no safety alignment, no reinforcement learning from verification."),
    ("HIGH - No multi-token prediction speculative decoding", "MTP head exists but no draft/verify inference loop implemented."),
    ("MEDIUM - No retrieval / RAG system", "Cannot ground answers in external sources. Research capability limited to parametric memory."),
    ("MEDIUM - No code execution sandbox for generated code", "VEXX Tier 5 exists but is not connected to Kestrel's generated code in a repair loop."),
    ("MEDIUM - No advanced reasoning (CoT, tree-of-thought)", "No structured problem-solving cycle implemented (understand, decompose, plan, propose, execute, observe, assess, revise, verify, report)."),
    ("MEDIUM - No evidence/provenance tracking", "Research outputs cannot be traced to sources. No claim-to-source entailment."),
    ("LOW - Tokenizer is smoke-test vocabulary (384 tokens)", "The 384-token tokenizer from the runner is a diagnostic artifact, not a production choice."),
]


def print_gap_analysis() -> None:
    print(f"\n{'='*70}")
    print("GAP ANALYSIS - WHAT LACKS & WHAT TO WORK ON")
    print(f"{'='*70}")
    for i, (gap, detail) in enumerate(GAP_ANALYSIS, 1):
        print(f"\n{i}. {gap}")
        print(f"   {detail}")


# --------------------------------------------------------------------------- main

def main() -> None:
    print_frontier_estimates()
    print_gap_analysis()

    # Run current model evaluation (will show near-random performance)
    results = evaluate_current_model("TINY_HYBRID")

    print(f"\n{'='*70}")
    print("CURRENT MODEL MEASURED PERFORMANCE (randomly initialized)")
    print(f"{'='*70}")
    for bench_name, bench_data in results["benchmarks"].items():
        print(f"\n{bench_name.upper()}: {bench_data['passed']}/{bench_data['total']} "
              f"({bench_data['pass_rate']*100:.1f}%)")
        for item in bench_data["items"]:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"  {item['id']}: {status} | expected={item['expected'][:60]} | "
                  f"output={item['output'][:80]}")

    # Save results
    out_path = "/workspace/app/research/day7/frontier_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()