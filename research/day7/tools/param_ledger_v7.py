#!/usr/bin/env python3
"""Compute every Kestrel-7 ledger number from the model itself.

Writes ``research/day7/metrics.json``. Nothing is transcribed by hand: parameter
counts come from ``sum(p.numel())``, KV bytes from the actual head layout, and
the benchmark row from running the evaluator. The test suite re-runs this
comparison and fails if a documented number drifts.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from kestrel7.config import CONFIGS, get_config  # noqa: E402
from kestrel7.model import (  # noqa: E402
    GatedCausalSelfAttention,
    GatedDeltaNetLayer,
    build_model,
    count_params,
    gdn_state_bytes,
    kv_cache_bytes,
    parameter_ledger,
)
from eval_tempbench_v7 import run_benchmark  # noqa: E402

PRESETS = ["TINY_DENSE", "TINY_HYBRID", "SMALL_HYBRID", "BASELINE_100M"]


def ledger() -> dict:
    presets = {}
    for name in PRESETS:
        cfg = get_config(name)
        model = build_model(cfg)
        total, trainable = count_params(model)
        n_full = sum(isinstance(b.mixer, GatedCausalSelfAttention) for b in model.blocks)
        n_gdn = sum(isinstance(b.mixer, GatedDeltaNetLayer) for b in model.blocks)
        kv_fp16 = kv_cache_bytes(model, dtype_bytes=2)
        all_full = n_full + n_gdn  # counterfactual: every layer is full attention
        kv_all_full = all_full * 2 * cfg["n_kv_heads"] * (cfg["d"] // cfg["n_heads"]) * cfg["max_ctx"] * 2
        detailed = parameter_ledger(model)
        presets[name] = {
            "total_params": total,
            "total_params_pretty": f"{total:,}",
            "trainable_params": trainable,
            "vocab": int(cfg["vocab"]),
            "d": int(cfg["d"]),
            "layers": int(cfg["layers"]),
            "layers_gdn": int(n_gdn),
            "layers_full": int(n_full),
            "max_ctx": int(cfg["max_ctx"]),
            "mtp_depth": int(cfg.get("mtp_depth", 0)),
            "kv_cache_bytes_fp16_ctx": kv_fp16,
            "kv_cache_bytes_all_full_counterfactual_fp16": int(kv_all_full),
            "kv_reduction_vs_all_full_pct": round(
                100.0 * (1.0 - kv_fp16 / max(1, kv_all_full)), 2
            ),
            "gdn_state_bytes_fp32_batch1": gdn_state_bytes(model, batch=1),
            # P1.4.9: detailed per-module and per-layer breakdown.
            "embedding_params": detailed["embedding_params"],
            "head_params": detailed["head_params"],
            "tied_embeddings": detailed["tied_embeddings"],
            "kv_cache_per_layer_fp16": detailed["kv_cache_per_layer_fp16"],
            "gdn_state_per_layer_fp32_batch1": detailed["gdn_state_per_layer_fp32_batch1"],
            "activation_estimate_bytes": detailed["activation_estimate_bytes"],
        }
        # P1.4.9: label the KV reduction as counterfactual arithmetic, not a
        # measured end-to-end reduction.
        presets[name]["kv_reduction_note"] = (
            "Counterfactual arithmetic: 3 of every 4 layers keep an O(1) state "
            "instead of a per-token KV cache. Not a measured end-to-end memory "
            "or latency reduction."
        )
        del model

    bench = run_benchmark(verbose=False)
    return {
        "generated_by": "research/day7/tools/param_ledger_v7.py",
        "torch_version": torch.__version__,
        "presets": presets,
        "benchmark": {
            "suite": "Kestrel6_TempBench",
            "tasks": bench["tasks"],
            "points": bench["points"],
            "earned": bench["earned"],
            "pct": round(bench["pct"], 1),
            "failures": bench["failures"],
            "categories": bench["categories"],
        },
        "config_names": sorted(CONFIGS),
    }


def _comparable(data: dict) -> dict:
    """Drop fields that legitimately differ between environments.

    ``torch_version`` and ``generated_by`` are provenance, not claims about the
    model; a CI machine with a different torch build must not fail the check for
    that reason alone.
    """
    return {
        "presets": data.get("presets", {}),
        "benchmark": data.get("benchmark", {}),
        "config_names": data.get("config_names"),
    }


def check() -> int:
    out = os.path.join(ROOT, "metrics.json")
    if not os.path.exists(out):
        print(f"DRIFT: {out} does not exist; run this script without --check")
        return 1
    with open(out, "r", encoding="utf-8") as f:
        stored = json.load(f)
    fresh = ledger()
    a, b = _comparable(stored), _comparable(fresh)
    if a != b:
        diff = difflib.unified_diff(
            json.dumps(a, indent=2, sort_keys=True).splitlines(),
            json.dumps(b, indent=2, sort_keys=True).splitlines(),
            fromfile="metrics.json (committed)",
            tofile="metrics.json (recomputed from code)",
            lineterm="",
        )
        print("\n".join(diff))
        print(
            "\nDRIFT: metrics.json disagrees with the model; regenerate with "
            "`python3 research/day7/tools/param_ledger_v7.py` and update the docs."
        )
        return 1
    bench = fresh["benchmark"]
    print(
        f"metrics.json matches the live ledger: {len(fresh['presets'])} presets, "
        f"benchmark {bench['earned']}/{bench['points']} over {bench['tasks']} tasks"
    )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="recompute the ledger and exit nonzero if metrics.json drifted",
    )
    args = ap.parse_args(argv)
    if args.check:
        return check()

    data = ledger()
    out = os.path.join(ROOT, "metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    for name, row in data["presets"].items():
        print(
            f"{name:<14} params={row['total_params_pretty']:>12} "
            f"layers={row['layers']} ({row['layers_gdn']} gdn / {row['layers_full']} full) "
            f"KV={row['kv_cache_bytes_fp16_ctx']:,} B "
            f"({row['kv_reduction_vs_all_full_pct']}% reduction)"
        )
    b = data["benchmark"]
    print(f"benchmark {b['earned']}/{b['points']} = {b['pct']}% over {b['tasks']} tasks")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
