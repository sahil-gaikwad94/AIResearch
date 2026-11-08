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

# False-positive rate of each verification regime. The first two are measured:
# blind judge 0.719 and commit-first 0.012 (day 3.1). The Day 4 rows are
# DERIVED, not measured: each is commit-first's residual multiplied by the
# stated kill rate of the gate added on top of it. Printing them without that
# arithmetic is how a modelled number becomes a claimed measurement.
FPR_BLIND = 0.719
FPR_COMMIT_FIRST = 0.012
KILL_RATE_MUTANT = 0.18          # mutant gate removes 82% of residual false accepts
KILL_RATE_VEXX = 0.027           # VEXX tiers 0-2 remove 97.3% of what is left
FPR_MUTANT_GATE = FPR_COMMIT_FIRST * KILL_RATE_MUTANT
FPR_VEXX_PLUS_MUTANT = FPR_COMMIT_FIRST * KILL_RATE_VEXX

# Per-conflict cost from the Kestrel-3.1 cost model (2.91 GFLOP per CoT step).
COT_STEP_GFLOP = 2.91
RETRY_TOKENS = 600
JUMPBACK_TOKENS = 18


def p(r):
    return PINF - (PINF - P1) * (DECAY ** (r - 1))


def correlated_ladder(ladder=LADDER, rho=0.78):
    """Depth ladder 2->4->6->8 over Kestrel-4's recurrent workspace.

    rho is the retry correlation across deterministic unrolls (day 3.1: 0.78;
    Kestrel-4's hypothesis codebook decorrelates it to 0.18). For exchangeable
    Bernoulli attempts with correlation rho, P(success | previous failure) =
    p * (1 - rho) exactly: the covariance is rho * p * (1 - p), so subtracting
    rho * p from p is the whole of it. An earlier version discounted by
    (1 - 0.85 * rho), which is a fudge factor with no derivation and made a
    0.78-correlated retry look 8 points better than it is.
    """
    solved, cost, alive = 0.0, 0.0, 1.0
    for idx, r in enumerate(ladder):
        cost += alive * r * PASS_COST
        p_eff = p(r) if idx == 0 else p(r) * (1.0 - rho)
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
    print("  Regimes: blind 0.719 and commit-first 0.012 are measured; the gate rows")
    print(f"  are derived as 0.012 x kill rate: mutant 0.012 x {KILL_RATE_MUTANT} = "
          f"{FPR_MUTANT_GATE:.5f},")
    print(f"  VEXX+mutant 0.012 x {KILL_RATE_VEXX} = {FPR_VEXX_PLUS_MUTANT:.6f}.")
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