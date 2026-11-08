#!/usr/bin/env python3
"""Kestrel-4 x VEXX 1.5 verification, CDCL jumpback, and poison-rate simulator.

Every figure here is arithmetic over *stated assumptions*, labelled as such.
The measured inputs are the False-Positive Rates (Day 1-3 published; Day 4 gate
kill rates stated by their papers); everything else is derived.  The solver that
lives behind the Tier-1/Tier-2 masking is VEXX 1.5 (python/vexx/), which is
verified separately from this arithmetic.
"""

P1 = 0.30
PINF = 0.62
DECAY = 0.55
PASS_COST = 0.240
LADDER = (2, 4, 6, 8)

# False-positive rate of each verification regime.  These are inputs, not
# outputs: blind judge 0.719 (measured, day 3.1); commit-first 0.012 (measured);
# the Day 4 gates are stated as kill rates against the blind-judge residual.
FPR_BLIND = 0.719
FPR_COMMIT_FIRST = 0.012
FPR_MUTANT_GATE = 0.00216
FPR_VEXX_PLUS_MUTANT = 0.000324

# Per-conflict cost from the Kestrel-3.1 cost model (2.91 GFLOP per CoT step).
COT_STEP_GFLOP = 2.91
RETRY_TOKENS = 600
JUMPBACK_TOKENS = 18


def p(r):
    return PINF - (PINF - P1) * (DECAY ** (r - 1))


def correlated_ladder(ladder=LADDER, rho=0.78):
    """Depth ladder 2->4->6->8 over Kestrel-4's recurrent workspace.

    rho is the retry correlation across deterministic unrolls (day 3.1: 0.78;
    Kestrel-4's hypothesis codebook decorrelates it to 0.18).  A branch the
    model has already failed on is not expected to succeed again: conditional
    passes are discounted by (1 - rho * 0.85).
    """
    solved, cost, alive = 0.0, 0.0, 1.0
    for idx, r in enumerate(ladder):
        cost += alive * r * PASS_COST
        p_eff = p(r) if idx == 0 else p(r) * (1.0 - rho * 0.85)
        solved += alive * p_eff
        alive *= (1.0 - p_eff)
    return solved, cost


def poison_rate(p_solve, fpr):
    """Probability a newly accepted sample is a false positive.

    fp = fpr * (1 - p_solve) accepted false positives against p_solve accepted
    true positives; the rate is the former over the union.
    """
    fp = fpr * (1.0 - p_solve)
    return fp / (fp + p_solve)


def main():
    print("=" * 78)
    print("KESTREL-4 x VEXX 1.5 VERIFICATION, CDCL JUMPBACK & POISON SIMULATOR")
    print("=" * 78)

    s_ind, c_ind = correlated_ladder(rho=0.0)
    s_det, c_det = correlated_ladder(rho=0.78)
    s_hyp, c_hyp = correlated_ladder(rho=0.18)
    print("\n--- 1. Effect of Retry Correlation on the Depth Ladder (2->4->6->8) ---")
    print(f"  Day 3.1 assumed independent (rho=0.00) : solved={s_ind:.3f}  cost={c_ind:.2f} GFLOP")
    print(f"  Unconditioned deterministic (rho=0.78) : solved={s_det:.3f}  cost={c_det:.2f} GFLOP")
    print(f"  Kestrel-4 Hypothesis Codebook (rho=0.18): solved={s_hyp:.3f}  cost={c_hyp:.2f} GFLOP")

    print("\n--- 2. Self-Training Poison Rate Across Verification Regimes ---")
    print("  FPR regimes are the stated gate measurements: blind 0.719, commit-first")
    print("  0.012, differential-mutant-gate 0.00216, VEXX 1.5 + mutant gate 0.000324.")
    regimes = [
        ("Blind (0.719)", FPR_BLIND),
        ("Commit-1st (0.012)", FPR_COMMIT_FIRST),
        ("Mutant Gate (0.00216)", FPR_MUTANT_GATE),
        ("VEXX 1.5 + Mutant", FPR_VEXX_PLUS_MUTANT),
    ]
    print(f"  {'p_solve':>7s} | {'Blind (0.719)':>14s} | {'Commit-1st (0.012)':>18s} | "
          f"{'Mutant Gate':>13s} | {'VEXX 1.5 + Mutant':>18s}")
    for ps in (0.10, 0.20, 0.40, 0.60):
        rates = " | ".join(f"{poison_rate(ps, fpr):13.3%}" for _, fpr in regimes)
        print(f"  {ps:7.2f} | {rates}")
    print(f"  -> at p_solve=0.40: {poison_rate(0.40, FPR_COMMIT_FIRST):.3%} (commit-first) "
          f"vs {poison_rate(0.40, FPR_VEXX_PLUS_MUTANT):.3%} (VEXX 1.5), "
          f"{poison_rate(0.40, FPR_COMMIT_FIRST) / poison_rate(0.40, FPR_VEXX_PLUS_MUTANT):.0f}x lower")

    print("\n--- 3. VEXX 1.5 CDCL Jumpback vs Whole-Patch Retry (600 tokens) ---")
    full_retry_gflop = RETRY_TOKENS * COT_STEP_GFLOP
    cdcl_rewind_gflop = JUMPBACK_TOKENS * COT_STEP_GFLOP
    print(f"  Whole-patch retry (FC-SWE Day 3.1) : {full_retry_gflop:7.1f} GFLOP (rewinds 600 tokens)")
    print(f"  VEXX 1.5 CDCL Culprit Jumpback     : {cdcl_rewind_gflop:7.1f} GFLOP "
          f"(rewinds ~18 tokens -> {full_retry_gflop / cdcl_rewind_gflop:.1f}x cheaper)")
    print("=" * 78)


if __name__ == "__main__":
    main()