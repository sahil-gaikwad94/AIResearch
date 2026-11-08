#!/usr/bin/env python3
"""Long-context KV cost model for Kestrel-4.

Every number about the KV cache in `research/day4/*` is produced here from
`kestrel4.model.kv_sizes`, the same function assertion [14] checks.  Nothing in
this file is a remembered figure.

Run: python research/day4/tools/kv_model.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kestrel4.model import FULL, kv_sizes, total_params  # noqa: E402

BW = 3.35e12          # HBM3e, bytes/s
PCIE = 25e9           # Gen4 x16 effective, bytes/s
CTX = (4_096, 65_536, 262_144, 1 << 20)

EMBED = FULL["vocab"] * FULL["d"]
ROUTED_ALL = FULL["n_experts"] * 3 * FULL["d_lat"] * FULL["d_ff"] * FULL["layers"]
SUBSPACE_INACTIVE = ((FULL["n_groups"] - FULL["top_groups"])
                     * (2 * FULL["d"] * FULL["d_lat"] + FULL["d_lat"] * FULL["experts_per_group"]
                        + FULL["experts_per_group"]) * FULL["layers"])
ALWAYS_PARAMS = total_params(FULL) - EMBED - ROUTED_ALL - SUBSPACE_INACTIVE
ALWAYS_GB = ALWAYS_PARAMS * 0.55 / 1e9
EXPERT_GB = 3 * FULL["d_lat"] * FULL["d_ff"] * 0.55 / 1e9


def gb(x):
    return x / 1e9


def mb(x):
    return x / 1e6


def weight_gb(batch):
    distinct = FULL["n_experts"] * (
        1 - (1 - 1 / FULL["n_experts"]) ** (batch * FULL["top_k"]))
    return ALWAYS_GB + FULL["layers"] * distinct * EXPERT_GB


def kv_read_bytes(ctx, batch):
    """Bytes read from the cache to attend one step: local window, global
    window+selected blocks, and one summary per sealed block.  No double count:
    the global window is part of `read_keys`, not in addition to it."""
    s = kv_sizes(FULL, ctx, bits=8, batch=batch)
    local = (FULL["local_layers"] * min(ctx, FULL["window"])
             * (FULL["d_c"] + 2 * FULL["d_rope"]) * batch)
    glob = (FULL["global_nope_layers"] * min(ctx, s["read_keys"])
            * FULL["d_c"] * batch)
    return local + glob + s["summary_bytes"]


def table_residency():
    print("--- Residency per sequence ---")
    print(f"{'context':>9} {'local fp16':>10} {'global fp16':>11} {'int8 cold':>9} "
          f"{'on-chip':>8} {'HBM total':>9} {'x always':>8}")
    for ctx in CTX:
        fp16 = kv_sizes(FULL, ctx, bits=16)
        i8 = kv_sizes(FULL, ctx, bits=8)
        local = mb(fp16["local_hot_bytes"])
        glob = mb(fp16["global_hot_bytes"] + fp16["cold_bytes_fp16"])
        cold = mb(i8["cold_bytes"])
        onchip = mb(i8["global_hot_bytes"] + i8["summary_bytes"])
        total = mb(i8["local_hot_bytes"] + cold + i8["global_hot_bytes"] + i8["summary_bytes"])
        print(f"{ctx:>9,} {local:>10.1f} {glob:>11.1f} {cold:>9.1f} "
              f"{onchip:>8.1f} {total:>9.1f} {total/1e3/ALWAYS_GB:>7.2f}x")
    print("  on-chip = hot window + block summaries; int8 cold is the rest of the sequence.")


def table_step():
    print("\n--- Per-step KV traffic, int8 cold ---")
    print(f"{'ctx':>9} {'batch':>5} {'local MB':>9} {'global MB':>10} {'index MB':>9} "
          f"{'KV GB':>7} {'weights GB':>10} {'KV share':>8}")
    for ctx in (4_096, 1 << 20):
        for b in (1, 16, 64):
            s = kv_sizes(FULL, ctx, bits=8, batch=b)
            local = FULL["local_layers"] * min(ctx, FULL["window"]) * (FULL["d_c"] + 2 * FULL["d_rope"]) * b
            glob = FULL["global_nope_layers"] * min(ctx, s["read_keys"]) * FULL["d_c"] * b
            idx = s["summary_bytes"]
            kv = local + glob + idx
            w = weight_gb(b) * 1e9
            print(f"{ctx:>9,} {b:>5} {mb(local):>9.1f} {mb(glob):>10.1f} {mb(idx):>9.1f} "
                  f"{gb(kv):>7.2f} {gb(w):>10.2f} {kv/(kv+w):>7.1%}")


def table_serving():
    print("\n--- Sustained decode with KV traffic included ---")
    print(f"{'ctx':>9} {'batch':>5} {'step GB':>8} {'step ms':>8} {'tok/s':>8} "
          f"{'09 tok/s':>9} {'delta':>7}")
    documented = {(4096, 1): 5080, (4096, 16): 17633, (4096, 64): 29627}
    for ctx in (4_096, 65_536, 1 << 20):
        for b in (1, 16, 64):
            kv = kv_read_bytes(ctx, b)
            step = weight_gb(b) * 1e9 + kv
            ms = step / BW * 1e3
            tok = b / (ms / 1e3)
            doc = documented.get((ctx, b))
            d = f"{tok / doc - 1:+.1%}" if doc else "-"
            print(f"{ctx:>9,} {b:>5} {gb(step):>8.2f} {ms:>8.3f} {tok:>8,.0f} "
                  f"{(f'{doc:,}' if doc else '-'):>9} {d:>7}")


def table_index():
    print("\n--- Indexer: sealed vs recomputed, batch 1 ---")
    print(f"{'ctx':>9} {'unsealed GB/step':>17} {'unsealed ms':>12} {'sealed MB':>10} {'sealed us':>10}")
    for ctx in CTX:
        u = kv_sizes(FULL, ctx, bits=8)
        unsealed = gb(u["index_bytes_unsealed"])
        sealed = mb(u["summary_bytes"])
        print(f"{ctx:>9,} {unsealed:>17.2f} {u['index_bytes_unsealed']/BW*1e3:>12.2f} "
              f"{sealed:>10.1f} {sealed*1e6/BW*1e6:>10.1f}")


def table_tier():
    print("\n--- Cold tier over PCIe at 1M: does host-resident KV fit the step? ---")
    print(f"{'batch':>5} {'stream MB':>9} {'pcie ms':>8} {'step ms':>8} {'share':>6} {'verdict':>9}")
    for b in (1, 2, 8, 16, 64):
        s = kv_sizes(FULL, 1 << 20, bits=8, batch=b)
        stream = s["stream_bytes"]
        pcie_ms = stream / PCIE * 1e3
        step_ms = weight_gb(b) * 1e9 / BW * 1e3
        share = pcie_ms / step_ms
        print(f"{b:>5} {mb(stream):>9.2f} {pcie_ms:>8.2f} {step_ms:>8.2f} {share:>6.0%} "
              f"{('fits' if share < 0.25 else 'paging-bound'):>9}")


def table_choice():
    print("\n--- What each mechanism buys at 1M, batch 1 ---")
    fp16 = kv_sizes(FULL, 1 << 20, bits=16)
    i8 = kv_sizes(FULL, 1 << 20, bits=8)
    base = gb(fp16["global_hot_bytes"] + fp16["cold_bytes_fp16"])
    q8 = gb(i8["global_hot_bytes"] + i8["cold_bytes"])
    print(f"  fp16 all-resident       {base:6.2f} GB")
    print(f"  int8 per-channel        {q8:6.2f} GB   ({base/q8:.2f}x)")
    print(f"  + sealed index          index {gb(i8['index_bytes_unsealed']):.2f} GB/step -> "
          f"{mb(i8['summary_bytes']):.1f} MB/step ({i8['index_bytes_unsealed']/i8['summary_bytes']:.0f}x)")
    print(f"  + cold tier (DRAM)      on-chip {mb(i8['global_hot_bytes'] + i8['summary_bytes']):.1f} MB, "
          f"stream {mb(i8['stream_bytes']):.2f} MB/step")


def main():
    print("=" * 78)
    print("KESTREL-4 LONG-CONTEXT KV MODEL")
    print("=" * 78)
    table_residency()
    table_step()
    table_serving()
    table_index()
    table_tier()
    table_choice()
    print("=" * 78)


if __name__ == "__main__":
    main()
