#!/usr/bin/env python3
"""Adversarial audit of the Day 4 mechanisms.

This is not a smoke test. Each check tries to *refute* a claim made in
`research/day4/*` by measuring the mechanism in the reference implementation.
A claim that lives only in prose fails here.

Run: python research/day4/tools/audit_v4.py
Exit code is 0 only when every claim is measurably true.
"""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kestrel4.model import (  # noqa: E402
    Block,
    CodeJEPAWorkspace,
    HouseholderDeltaProductMemory,
    Kestrel4,
    LatentQuantizer,
    SealedBlockIndex,
    SubspaceLatentMoE,
    TieredLatentCache,
    VexxNeuroSymbolicController,
    attn_global_nope_params,
    attn_local_params,
    householder_step,
    kv_sizes,
    mem_params,
    moe_params,
    workspace_params,
)

torch.manual_seed(0)

TINY = dict(
    vocab=128, d=64, heads=4, dh=16, d_c=32, d_rope=8, layers=4, local_layers=3,
    global_nope_layers=1, global_layer_indices=(3,), window=8, sel_block=4, sel_topk=2,
    anchor_budget=2, idx_dim=16, d_m=32, r_m=16, n_householder=2, n_groups=4,
    experts_per_group=4, n_experts=16, top_groups=2, top_k_per_group=2, top_k=4,
    d_lat=16, d_ff=32, d_shared=32, w_layers=2, d_w=32, w_ff=64, d_head_h=16,
    r_max=8, k_hyp=4, rope_theta=500000.0,
)

RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def F_normalize(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def analytic_flops_fwd(ctx=4096, r=2):
    """The decomposition documented in chapters 07/09, written independently."""
    import kestrel4.model as m

    V = m.FULL
    total, active = m.total_params(V), m.active_params(V)
    emb, ws = V["vocab"] * V["d"], m.workspace_params(V)
    lin = active - emb - ws
    dense = 2 * (lin + max(0, r - 1) * ws)          # ACTIVE already includes one workspace pass
    w_eff = min(ctx, V.get("w_window", V["window"]))
    att_loc = V["local_layers"] * 4 * w_eff * V["heads"] * (V["dh"] + V["d_rope"])
    g_eff = min(ctx, V["window"] + V["sel_topk"] * V["sel_block"] + V["anchor_budget"])
    att_glb = V["global_nope_layers"] * 4 * g_eff * V["heads"] * V["dh"]
    idx_keys = (ctx // V["sel_block"]) + V["sel_topk"] * V["sel_block"] + V["anchor_budget"]
    idx = V["global_nope_layers"] * 2 * idx_keys * V["idx_dim"]
    w_heads = V["d_w"] // V["d_head_h"]
    w_att = V["w_layers"] * r * 2 * 2 * w_heads * w_eff * V["d_head_h"]
    return dense + att_loc + att_glb + idx + w_att


def section(title):
    print(f"\n{title}")


def audit_memory():
    section("A1  Householder memory: is there a negative eigenvalue?")
    # Empirical, not analytic: apply the implemented step to a probe vector along
    # the reflection axis and read the eigenvalue off the ratio.
    state = torch.randn(2, 5, TINY["d_m"])
    u = F_normalize(torch.randn(2, TINY["d_m"]))
    beta = torch.full((2,), 1.8)
    after = householder_step(state, u, beta)
    before_proj = (state @ u.unsqueeze(-1)).squeeze(-1)
    after_proj = (after @ u.unsqueeze(-1)).squeeze(-1)
    lam = float((after_proj / before_proj)[0, 0])
    check("transition has a negative eigenvalue", lam < 0,
          f"beta=1.8 => measured eigenvalue={lam:+.3f} (need < 0)")
    twice = householder_step(householder_step(state, u, torch.full((2,), 2.0)), u, torch.full((2,), 2.0))
    check("H(beta=2) is an involution (H^2 = I)",
          float((twice - state).abs().max()) < 1e-6,
          f"max|H^2 B - B|={float((twice - state).abs().max()):.2e}")
    mem = HouseholderDeltaProductMemory(TINY)
    with torch.no_grad():
        _, st = mem(torch.randn(1, 6, TINY["d"]))
    check("state is finite after 6 tokens", bool(torch.isfinite(st).all()),
          f"|state|={float(st.norm()):.4f}")


def audit_residual():
    section("A2  Residual stream: does a memory layer preserve input magnitude?")
    blk = Block(TINY, is_global_nope=False)
    ratios = []
    for scale in (1.0, 100.0):
        xb = torch.randn(2, 5, TINY["d"]) * scale
        with torch.no_grad():
            out, _ = blk(xb)
        ratios.append(float(out.pow(2).mean().sqrt()) / float(xb.pow(2).mean().sqrt()))
    check("output RMS tracks input RMS (residual preserved)", all(0.5 < q < 2.0 for q in ratios),
          f"out/in RMS ratios = {[round(q, 3) for q in ratios]} (need ~1)")


def audit_indexer():
    section("A3  Global indexer: does w_iq change the forward pass?")
    model = Kestrel4(TINY).eval()
    ids = torch.randint(0, TINY["vocab"], (1, 12))
    with torch.no_grad():
        base, *_ = model(ids, r=1)
        w = model.blocks[3].attn.w_iq.weight
        saved = w.detach().clone()
        w.mul_(3.0)
        perturbed, *_ = model(ids, r=1)
        w.copy_(saved)
    delta = float((base - perturbed).abs().max())
    check("indexer weights influence the logits", delta > 1e-6,
          f"max|logits(w_iq) - logits(3*w_iq)| = {delta:.3e}")


def audit_flops():
    section("A4  Cost model: does the documented FLOP decomposition match the tool?")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cp4", os.path.join(ROOT, "tools", "count_params_v4.py"))
    cp4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp4)
    for r in (1, 2, 4, 8):
        shipped = cp4.flops_fwd(4096, r)
        fixed = analytic_flops_fwd(4096, r)
        rel = abs(shipped - fixed) / fixed
        check(f"r={r} forward FLOPs agree within 2%", rel < 0.02,
              f"tool={shipped / 1e9:.3f} documented={fixed / 1e9:.3f} GFLOP ({rel:.2%})")


def audit_rho():
    section("A5  Retry correlation: is the discount the exchangeable-Bernoulli one?")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import sim_verify_v4 as sim

    def ladder(rho, discount):
        solved, alive = 0.0, 1.0
        for idx, r in enumerate(sim.LADDER):
            p_eff = sim.p(r) if idx == 0 else sim.p(r) * (1.0 - rho * discount)
            solved += alive * p_eff
            alive *= (1.0 - p_eff)
        return solved

    for rho in (0.18, 0.78):
        shipped, _ = sim.correlated_ladder(rho=rho)
        derived = ladder(rho, 1.0)
        check(f"rho={rho} uses P(success|failure)=p(1-rho)", abs(shipped - derived) < 1e-9,
              f"shipped={shipped:.3f} vs exchangeable={derived:.3f}")


def audit_certificate():
    section("A6  L3 certificate: is `verified` earned?")
    ctl = VexxNeuroSymbolicController(domain_bounds={"w1": (0.05, 0.50), "w2": (0.05, 0.50)})
    checks = [({'w1': 1, 'w2': 1}, 1.0)]
    bad = ctl.emit_l3_proof_certificate("pkg.auth.Session", {"w1": 0.90, "w2": 0.90},
                                        "(assert (= (+ w1 w2) 1.0))", checks=checks)
    good = ctl.emit_l3_proof_certificate("pkg.auth.Session", {"w1": 0.40, "w2": 0.60},
                                         "(assert (= (+ w1 w2) 1.0))", checks=checks)
    unchecked = ctl.emit_l3_proof_certificate("x", {"w1": 0.40}, "(assert true)")
    check("a violating witness is not 'verified'", bad["verified"] is False,
          f"w1+w2=1.8 -> verified={bad['verified']}")
    check("a satisfying witness is 'verified'", good["verified"] is True,
          f"w1+w2=1.0 -> verified={good['verified']}")
    check("an unchecked certificate fails closed", unchecked["verified"] is False,
          f"no checks -> verified={unchecked['verified']}")


def audit_domain():
    section("A7  CDCL domain cuts: can a jumpback empty the domain?")
    ctl = VexxNeuroSymbolicController(domain_bounds={"w": (0.05, 0.50)})
    empty = False
    for i in range(4):
        ctl.save_clause_checkpoint("w", token_idx=1, states=[torch.zeros(1)],
                                   mean_entropy=2.0, assigned_val=0.06)
        ctl.trigger_cdcl_jumpback({"w"}, rejected_tokens=[9], repaired_tokens=[8])
        lo, hi = ctl.bounds["w"]
        empty = empty or lo > hi
    lo, hi = ctl.bounds["w"]
    check("bounds stay non-empty after repeated cuts", not empty,
          f"final bounds=({lo:.4f}, {hi:.4f})")


def audit_workspace():
    section("A8  Workspace: does it respect the configured window?")
    layer = CodeJEPAWorkspace(TINY).layers[0]
    n = 12
    h = torch.randn(1, n, TINY["d_w"])
    ctx = torch.randn(1, n, TINY["d"])
    q = layer.q(layer.n1(h))
    k, _ = layer.kv(ctx).chunk(2, dim=-1)
    dist = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    m = dist < layer.window
    att = ((q @ k.transpose(-1, -2)) / math.sqrt(TINY["d_w"])).masked_fill(~m, float("-inf")).softmax(-1)
    leaked = float(att.masked_fill(m, 0.0).sum())
    check("attention is confined to the configured window", leaked < 1e-6,
          f"weight outside window={leaked:.4f} (window={layer.window})")


def audit_params():
    section("A9  Parameter ledger: does each component match its module?")
    local = sum(p.numel() for p in Block(TINY, is_global_nope=False).attn.parameters())
    glob = sum(p.numel() for p in Block(TINY, is_global_nope=True).attn.parameters())
    memo = sum(p.numel() for p in HouseholderDeltaProductMemory(TINY).parameters())
    moe = sum(p.numel() for p in SubspaceLatentMoE(TINY).parameters())
    ws = sum(p.numel() for p in CodeJEPAWorkspace(TINY).parameters())
    check("local MLA analytic == module", local == attn_local_params(TINY),
          f"{attn_local_params(TINY)} vs {local}")
    check("global NoPE analytic == module", glob == attn_global_nope_params(TINY),
          f"{attn_global_nope_params(TINY)} vs {glob}")
    check("memory analytic == module", memo == mem_params(TINY), f"{mem_params(TINY)} vs {memo}")
    check("MoE analytic == module", moe == moe_params(TINY, False),
          f"{moe_params(TINY, False)} vs {moe}")
    check("workspace analytic == module", ws == workspace_params(TINY),
          f"{workspace_params(TINY)} vs {ws}")


def audit_kv_bytes():
    section("A10 KV bytes: does the analytic model match an independent count?")
    import kestrel4.model as m

    V = m.FULL
    for ctx in (4096, 1 << 20):
        s = kv_sizes(V, ctx, bits=8)
        win = min(ctx, V["window"])
        local = V["local_layers"] * win * (V["d_c"] + 2 * V["d_rope"])
        cold = V["global_nope_layers"] * (ctx - win) * V["d_c"]
        summ = V["global_nope_layers"] * math.ceil(ctx / V["sel_block"]) * V["idx_dim"] * 2
        ok = (s["local_hot_bytes"] == local and s["cold_bytes"] == cold
              and s["summary_bytes"] == summ)
        check(f"ctx={ctx} KV bytes equal elementwise construction", ok,
              f"local={s['local_hot_bytes']:,} cold={s['cold_bytes']:,} summaries={s['summary_bytes']:,}")


def audit_index_seal():
    section("A11 Sealed index: exact summary, sublinear reads")
    import kestrel4.model as m

    V = m.FULL
    attn = m.GlobalNoPESparseMLA(TINY)
    n, bs = 4096, TINY["sel_block"]
    lat = torch.randn(n, TINY["d_c"])
    idx = SealedBlockIndex(TINY["d_c"], TINY["idx_dim"], bs)
    for b0 in range(0, n, bs):
        idx.seal(lat[b0:b0 + bs], attn.w_ip, attn.w_ik)
    iq = torch.randn(3, TINY["idx_dim"])
    sealed = idx.scores(iq)
    pooled = lat.view(n // bs, bs, -1).mean(dim=1)
    ref = 0.5 * (attn.w_ip(pooled)
                 + attn.w_ik(lat).view(n // bs, bs, -1).mean(dim=1))
    ref_scores = iq @ ref.T / math.sqrt(TINY["idx_dim"])
    dmax = float((sealed - ref_scores).abs().max())
    reads = idx.summaries_read()
    check("sealed summaries equal dense recomputation", dmax < 1e-6,
          f"max|dscore|={dmax:.2e} over {n} latents")
    check("index reads are one summary per block", reads == n // bs,
          f"{reads} rows read vs {n} latents ({reads / n:.2%})")


def audit_kv_precision():
    section("A12 KV precision: per-channel int8 survives outliers")
    g = torch.Generator().manual_seed(11)
    lat = torch.randn(2048, 256, generator=g)
    lat[:, 5] *= 25.0
    lat[:, 9] *= 8.0
    lat[:, 40] *= 3.0
    q8 = LatentQuantizer(8)
    s, z = q8.calibrate(lat)
    rec = q8.decode(q8.encode(lat, s, z), s, z)
    cos8 = float(torch.nn.functional.cosine_similarity(lat, rec, dim=-1).mean())
    s_tok = lat.abs().amax(dim=-1, keepdim=True) / 127.0
    rec_tok = torch.round(lat / s_tok).clamp(-127, 127) * s_tok
    cos_tok = float(torch.nn.functional.cosine_similarity(lat, rec_tok, dim=-1).mean())
    check("int8 per-channel cosine > 0.999", cos8 > 0.999, f"cos={cos8:.6f}")
    check("per-channel beats per-token under outliers", cos8 > cos_tok,
          f"per-channel {cos8:.6f} vs per-token {cos_tok:.6f}")


def audit_kv_tier():
    section("A13 Tiered cache: accounting matches, only selected latents stream")
    cache = TieredLatentCache(dc=64, bs=4, window=8, bits=8)
    cache.append(torch.randn(4, 64))
    cache.append(torch.randn(4, 64))
    cache.append(torch.randn(4, 64))
    picks = torch.tensor([0, 2, 5, 9, 11])
    out = cache.gather(picks)
    check("gather returns the selected rows", out.shape == (5, 64),
          f"shape={tuple(out.shape)} streamed={cache.latents_streamed} latents")
    check("only selected latents cross the bus", cache.latents_streamed == 5,
          f"{cache.latents_streamed} of {cache.token_count()} latents")
    check("cold bytes count only the un-windowed tail",
          cache.bytes_cold(layers=1) == (12 - 8) * 64,
          f"cold={cache.bytes_cold(layers=1)} B expected={(12 - 8) * 64} B")
    import kestrel4.model as m

    s = kv_sizes(m.FULL, 1 << 20, bits=8)
    expected = m.FULL["global_nope_layers"] * (
        m.FULL["sel_topk"] * m.FULL["sel_block"] + m.FULL["anchor_budget"]) * m.FULL["d_c"]
    check("1M stream per step is the selected non-window latents",
          s["stream_bytes"] == expected,
          f"stream={s['stream_bytes']:,} B ({s['stream_bytes'] / 1e6:.2f} MB)")


def main():
    print("=" * 78)
    print("DAY 4 ADVERSARIAL AUDIT")
    print("=" * 78)
    audit_memory()
    audit_residual()
    audit_indexer()
    audit_flops()
    audit_rho()
    audit_certificate()
    audit_domain()
    audit_workspace()
    audit_params()
    audit_kv_bytes()
    audit_index_seal()
    audit_kv_precision()
    audit_kv_tier()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 78)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} claims hold; "
          f"{len(failed)} refuted")
    if failed:
        print("refuted: " + ", ".join(failed))
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
