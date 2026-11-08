#!/usr/bin/env python3
"""The arithmetic behind verification-conditioned depth, search, and distillation.

THIS IS ARITHMETIC ON STATED ASSUMPTIONS, NOT EVIDENCE.

Every claim in doc 08 that depends on a probability model is computed here,
with the assumptions as named parameters so they can be argued with. The two
things that are *not* assumed are the verifier false-positive rates, which are
measured values taken from 2607.05904 (see doc 08).
"""
import math

# ---- stated assumptions ---------------------------------------------------
P1 = 0.30          # P(solve) with one workspace pass, on the hard eval slice
PINF = 0.62        # P(solve) as depth -> infinity
DECAY = 0.55       # p(r) = PINF - (PINF - P1) * DECAY**(r-1)
PASS_COST = 0.236  # GFLOP, one workspace pass
LADDER = (2, 4, 6, 8)


def p(r):
    return PINF - (PINF - P1) * DECAY ** (r - 1)


def fixed_depth(r, k_repeat=1):
    """Retry k times at a fixed depth, independent attempts."""
    return 1 - (1 - p(r)) ** k_repeat, k_repeat * r * PASS_COST


def ladder_policy(ladder=LADDER):
    """Escalate only when the verifier rejects. Cost is in passes."""
    solved, cost, alive = 0.0, 0.0, 1.0
    for r in ladder:
        cost += alive * r * PASS_COST
        solved += alive * p(r)
        alive *= 1 - p(r)
    return solved, cost


def rerank(k, p_one, verifier_acc):
    """Best-of-k with a verifier that ranks a correct sample above an
    incorrect one with probability `verifier_acc`.

    With acc=1 the answer is pass@k. With acc<1 the verifier can pick a wrong
    sample even when a right one exists, so accuracy can fall below greedy.
    """
    any_correct = 1 - (1 - p_one) ** k
    return any_correct * verifier_acc + (1 - any_correct) * (1 - verifier_acc) * 0.0


def poison_rate(p_success, fpr):
    """Share of 'verified successes' that are actually failures, if the
    verifier accepts a wrong solution with probability fpr."""
    fp = fpr * (1 - p_success)
    return fp / (fp + p_success)


def main():
    print("=" * 74)
    print("  ARITHMETIC ON STATED ASSUMPTIONS -- NOT A MEASUREMENT")
    print(f"  p(r) = {PINF} - ({PINF}-{P1}) * {DECAY}^(r-1)")
    print("=" * 74)

    print("\n--- 1. what depth buys, before any verification ---")
    for r in (1, 2, 4, 6, 8, 16):
        print(f"  r={r:2d}   p={p(r):.3f}   cost={r * PASS_COST:.2f} GFLOP")

    print("\n--- 2. fixed depth vs escalation on verifier rejection ---")
    ps, cs = ladder_policy()
    best_fixed = max(((fixed_depth(r)[0], r) for r in (2, 4, 6, 8)))
    print(f"  ladder {LADDER}: solved={ps:.3f}  cost={cs:.2f} GFLOP")
    for r in (2, 4, 6, 8):
        s, c = fixed_depth(r)
        print(f"  fixed r={r}: solved={s:.3f}  cost={c:.2f} GFLOP")
    print(f"  best fixed depth is r={best_fixed[1]} at {best_fixed[0]:.3f} "
          f"for {best_fixed[1] * PASS_COST:.2f} GFLOP")
    print(f"  -> the ladder reaches {ps:.3f} (vs {p(8):.3f} at always-r=8) "
          f"for {cs / (8 * PASS_COST):.0%} of its cost")

    print("\n--- 3. a bad verifier makes search worse, not better ---")
    print("  verifier FPR is measured in 2607.05904: 0.719 without a committed")
    print("  answer, 0.012 with one. acc = 1 - FPR.")
    for label, acc in (("commit-first  (FPR 0.012)", 0.988),
                       ("blind judge   (FPR 0.719)", 0.281)):
        print(f"\n  {label}")
        for k in (1, 4, 16, 64, 256):
            a = rerank(k, P1, acc)
            print(f"    k={k:3d}: accuracy={a:.3f}  "
                  f"(pass@k ceiling {1 - (1 - P1) ** k:.3f})")

    print("\n--- 4. poison rate of a self-training set ---")
    print(f"  {'p_solve':>8} {'FPR 0.719':>12} {'FPR 0.012':>12}")
    for ps_ in (0.10, 0.20, 0.40, 0.60):
        print(f"  {ps_:8.2f} {poison_rate(ps_, 0.719):12.1%} "
              f"{poison_rate(ps_, 0.012):12.1%}")
    print("\n  Self-training on 'verified' successes is safe only if the")
    print("  verifier's errors are independent of the policy's errors. Below")
    print("  ~1% poison, four rounds stay inside the noise of the eval.")
    print("=" * 74)


if __name__ == "__main__":
    main()
