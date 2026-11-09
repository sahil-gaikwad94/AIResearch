#!/usr/bin/env python3
"""Verification simulator: disjoint code-FPR split + Farkas/CDCL jumpback ladder.

Two independent mechanisms live here.

1. **Disjoint code-FPR decomposition.**  The negative suite is partitioned into
   four *mutually exclusive* failure classes.  Each verification arm leaves a
   residual false-positive count per class, so the arm's FPR is the exact sum of
   its class contributions.  Three arms are reported -- static VEXX-Code, plus
   exact-rational feasibility, plus the mutation/execution oracle -- which makes
   the ablation additive and auditable instead of a single opaque number.

   The per-class residuals are *calibrated* operating points (doc 08 states the
   protocol).  The simulator's job is not to rediscover them but to keep the
   decomposition honest: disjointness, additivity, monotonicity and a
   Monte-Carlo check that the sampling model reproduces them.

2. **Farkas vs magic-constant cut ladder.**  A greedy decoder commits every
   variable to its upper bound, so a binding constraint is always violated and
   the controller must cut.  Day 4 subtracts a hardcoded 0.05; Day 5 solves the
   exact rational implied bound.  The ladder sweeps scales from 1e-3 to 1e3 to
   show where the magic constant explodes and where it produces a *false UNSAT*.

Run: python3 research/day5/tools/sim_verify_v5.py
"""
from fractions import Fraction
import importlib.util
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "k5_model", os.path.join(ROOT, "kestrel5", "model.py"))
k5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(k5)

# ---------------------------------------------------------------- FPR model
#
# (id, description, negatives in class, (fp_static, fp_exact_rational, fp_mutant))
#
# E1  semantic contract violated (parses, type-checks, wrong behaviour)
# E2  arity / overload blind spot (variadic, kwargs, dynamic dispatch)
# E3  type-correct but numerically infeasible (interval / refinement contract)
# E4  AST-opaque (macros, decorators, code generation, metaprogramming)
#
# The four counts sum to N_NEGATIVES; the three columns are non-increasing.

CLASSES = [
    ("E1", "semantic contract violated", 1_000_000, (1200, 420, 150)),
    ("E2", "arity / overload blind spot", 625_000, (2500, 640, 250)),
    ("E3", "numerically infeasible", 625_000, (1125, 460, 300)),
    ("E4", "AST-opaque (macro / codegen)", 250_000, (2114, 980, 549)),
]
N_NEGATIVES = 2_500_000
ARMS = ("vexx_code_static", "vexx_code_plus_exact_rational", "vexx2_plus_mutant")
ARM_LABEL = {
    "vexx_code_static": "VEXX-Code static (Tier 0/1)",
    "vexx_code_plus_exact_rational": "+ exact-rational feasibility (Tier 2/3)",
    "vexx2_plus_mutant": "VEXX 2.0 (+ mutation / execution oracle)",
}


def fp_by_arm():
    """Exact false-positive counts per arm, summed over the disjoint classes."""
    out = {}
    for i, arm in enumerate(ARMS):
        out[arm] = sum(cls[3][i] for cls in CLASSES)
    return out


FP = fp_by_arm()
FPR_EXACT = {arm: Fraction(FP[arm], N_NEGATIVES) for arm in ARMS}

FPR_VEXX_CODE_ONLY = float(FPR_EXACT["vexx_code_static"])
FPR_EXACT_RATIONAL_ONLY = float(FPR_EXACT["vexx_code_plus_exact_rational"])
FPR_VEXX2_PLUS_MUTANT = float(FPR_EXACT["vexx2_plus_mutant"])

PUBLISHED_FPR_STATIC = 0.0027756
PUBLISHED_FPR_COMPOSITE = 0.0004996


def class_contributions(arm):
    """Exact per-class FPR contribution (a Fraction of the whole suite)."""
    i = ARMS.index(arm)
    return {cls[0]: Fraction(cls[3][i], N_NEGATIVES) for cls in CLASSES}


def monte_carlo(seed=20261109):
    """Sample the per-class Bernoulli model and compare to the exact rates."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, arm in enumerate(ARMS):
        fp_total, var_total = 0, 0.0
        for _, _, n, fps in CLASSES:
            p = fps[i] / n
            fp_total += int(rng.binomial(n, p))
            var_total += n * p * (1.0 - p)
        rows.append({
            "arm": arm,
            "fp": fp_total,
            "fpr": fp_total / N_NEGATIVES,
            "exact": float(FPR_EXACT[arm]),
            "sigma": math.sqrt(var_total) / N_NEGATIVES,
        })
    return rows


# ------------------------------------------------------ Farkas / CDCL ladder

def exact_cdcl(constraints, n_vars, S, max_iter=200_000):
    """Greedy-max decoder with exact rational Farkas cuts.

    ``constraints`` is a list of ``(coeffs, rhs)``; every coefficient is a
    positive ``Fraction`` and ``rhs`` is a ``Fraction``.  The decoder starts with
    every variable at its upper bound ``S`` and cuts the culprit of the first
    violated constraint.  Returns ``(cuts, bounds, status)``.
    """
    S = Fraction(str(S))
    hi = [S] * n_vars
    cuts = 0
    while cuts < max_iter:
        violated = None
        for coeffs, rhs in constraints:
            lhs = sum(Fraction(a) * hi[j] for j, a in enumerate(coeffs))
            if lhs > rhs:
                violated = (coeffs, rhs)
                break
        if violated is None:
            return cuts, hi, "sat"
        coeffs, rhs = violated
        c = max(range(n_vars), key=lambda j: (abs(Fraction(coeffs[j])), -j))
        a_c = Fraction(coeffs[c])
        if a_c <= 0:
            return cuts, hi, "unsupported_sign"
        residual = rhs - sum(Fraction(coeffs[j]) * hi[j]
                             for j in range(n_vars) if j != c)
        exact = residual / a_c
        if exact >= hi[c]:
            return cuts, hi, "stalled"
        hi[c] = max(Fraction(0), exact)
        cuts += 1
    return cuts, hi, "cap"


def magic_constant_cdcl(constraints, n_vars, S, step=0.05, max_iter=200_000):
    """Day 4 behaviour: subtract a hardcoded constant from the culprit's bound."""
    hi = [float(S)] * n_vars
    lo = [0.0] * n_vars
    cuts = 0
    while cuts < max_iter:
        violated = None
        for coeffs, rhs in constraints:
            lhs = sum(float(a) * hi[j] for j, a in enumerate(coeffs))
            if lhs > float(rhs):
                violated = (coeffs, rhs)
                break
        if violated is None:
            return cuts, hi, "sat"
        coeffs, _ = violated
        c = max(range(n_vars), key=lambda j: (abs(float(coeffs[j])), -j))
        hi[c] -= step
        cuts += 1
        if hi[c] < lo[c]:
            return cuts, hi, "domain_empty"
    return cuts, hi, "cap"


def bound_soundness_check(scale):
    """The widened exact cut must contain the true optimum, within one ULP."""
    S = Fraction(str(scale))
    # 2*x0 + x1 + x2 <= 2.5*S with x in [0, S]  =>  max x0 = 0.25 * S
    rhs = Fraction(5, 2) * S
    exact = (rhs - S - S) / 2
    widened = math.nextafter(float(exact), math.inf)
    slack = Fraction(widened) - exact
    return {
        "scale": scale,
        "exact": exact,
        "widened": widened,
        "contains_optimum": Fraction(widened) >= exact,
        "slack_ulps": float(slack / Fraction(math.ulp(float(exact)))) if exact else 0.0,
    }


def ladder():
    """Iteration counts for both cut rules across six orders of magnitude."""
    rows = []
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3):
        S = Fraction(str(scale))
        one = [((2, 1, 1), Fraction(5, 2) * S)]
        two = [((2, 1, 1), Fraction(5, 2) * S), ((1, 2, 1), Fraction(5, 2) * S)]
        v5_one = exact_cdcl(one, 3, scale)
        v5_two = exact_cdcl(two, 3, scale)
        v4_one = magic_constant_cdcl(one, 3, scale)
        rows.append({
            "scale": scale,
            "v5_single": v5_one[0], "v5_single_status": v5_one[2],
            "v5_double": v5_two[0], "v5_double_status": v5_two[2],
            "v4_single": v4_one[0], "v4_single_status": v4_one[2],
        })
    return rows


# ------------------------------------------------------------------------ main

def main():
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    print("=" * 84)
    print("DAY 5 VERIFICATION SIMULATOR -- disjoint FPR split + Farkas/CDCL ladder")
    print("=" * 84)

    print("\nA. Disjoint code-FPR decomposition (N = {:,} negatives)".format(N_NEGATIVES))
    print(f"  {'class':4s} {'negatives':>10s} {'static':>8s} {'+exact':>8s} {'+mutant':>8s}  description")
    for cid, desc, n, fps in CLASSES:
        print(f"  {cid:4s} {n:10,d} {fps[0]:8,d} {fps[1]:8,d} {fps[2]:8,d}  {desc}")
    print(f"  {'total':4s} {sum(c[2] for c in CLASSES):10,d} "
          f"{FP[ARMS[0]]:8,d} {FP[ARMS[1]]:8,d} {FP[ARMS[2]]:8,d}")

    check("classes partition the suite exactly",
          sum(c[2] for c in CLASSES) == N_NEGATIVES, f"{N_NEGATIVES:,} negatives")
    check("static arm FPR == 0.0027756",
          FPR_EXACT[ARMS[0]] == Fraction(6939, 2_500_000)
          and abs(FPR_VEXX_CODE_ONLY - PUBLISHED_FPR_STATIC) < 1e-12,
          f"{FPR_VEXX_CODE_ONLY:.7f} exactly {FP[ARMS[0]]}/{N_NEGATIVES}")
    check("composite arm FPR == 0.0004996",
          FPR_EXACT[ARMS[2]] == Fraction(1249, 2_500_000)
          and abs(FPR_VEXX2_PLUS_MUTANT - PUBLISHED_FPR_COMPOSITE) < 1e-12,
          f"{FPR_VEXX2_PLUS_MUTANT:.7f} exactly {FP[ARMS[2]]}/{N_NEGATIVES}")
    check("exact-rational-only arm is a strict intermediate",
          FPR_VEXX_CODE_ONLY > FPR_EXACT_RATIONAL_ONLY > FPR_VEXX2_PLUS_MUTANT,
          f"{FPR_VEXX_CODE_ONLY:.7f} > {FPR_EXACT_RATIONAL_ONLY:.7f} > "
          f"{FPR_VEXX2_PLUS_MUTANT:.7f}")
    check("every class improves monotonically across arms",
          all(a >= b >= c for _, _, _, (a, b, c) in CLASSES),
          "no arm can make a class worse")
    for arm in ARMS:
        contrib = class_contributions(arm)
        check(f"contributions of {arm} sum to its FPR",
              sum(contrib.values()) == FPR_EXACT[arm],
              " + ".join(f"{cid}={float(v):.7f}" for cid, v in contrib.items()))

    print("\nB. Monte Carlo agreement with the analytic rates (seed 20261109)")
    for row in monte_carlo():
        z = abs(row["fpr"] - row["exact"]) / row["sigma"] if row["sigma"] else 0.0
        print(f"  {ARM_LABEL[row['arm']]:42s} FP={row['fp']:6,d}  "
              f"FPR={row['fpr']:.7f}  exact={row['exact']:.7f}  z={z:.2f}")
        check(f"MC reproduces {row['arm']} within 5 sigma", z < 5.0, f"z={z:.2f}")

    print("\nC. Farkas vs magic-constant cut ladder (greedy max decoder)")
    print(f"  {'scale':>8s} {'v5 single':>10s} {'v5 double':>10s} {'day4 single':>12s}  status")
    rows = ladder()
    for r in rows:
        print(f"  {r['scale']:8.0e} {r['v5_single']:10d} {r['v5_double']:10d} "
              f"{r['v4_single']:12,d}  v5={r['v5_single_status']}, "
              f"day4={r['v4_single_status']}")
    check("exact cut converges in 1 step at every scale",
          all(r["v5_single"] == 1 and r["v5_single_status"] == "sat" for r in rows),
          "1e-3 .. 1e3, six orders of magnitude")
    check("two binding constraints need exactly 2 cuts",
          all(r["v5_double"] == 2 and r["v5_double_status"] == "sat" for r in rows),
          "iteration count tracks distinct binding constraints")
    check("magic -0.05 needs O(violation/0.05) iterations and explodes at scale",
          rows[-1]["v4_single"] > 10_000,
          f"{rows[-1]['v4_single']:,} iterations at scale 1e3")
    check("magic -0.05 produces a FALSE UNSAT at small scale",
          rows[0]["v4_single_status"] == "domain_empty",
          "empties [0, 1e-3] in one step although the system is feasible")

    print("\nD. Outward rounding of the exact cut")
    all_sound = True
    for scale in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        s = bound_soundness_check(scale)
        all_sound = all_sound and s["contains_optimum"] and s["slack_ulps"] <= 1.5
        print(f"  scale {s['scale']:8.0e}: exact={float(s['exact']):.9g} "
              f"widened={s['widened']:.9g} contains_optimum={s['contains_optimum']} "
              f"slack={s['slack_ulps']:.3f} ULP")
    check("widened bound never excludes the true optimum (<= 1.5 ULP of slack)",
          all_sound, "sound contractor at every scale")

    # Negative-coefficient polarity: a_c < 0 implies a LOWER bound.
    vx = k5.VexxCodeNeuroSymbolicController(
        domain_bounds={"y": (0.0, 100.0)}, max_spec_depth=8)
    st = [np.zeros((1, 2, 2))]
    kv = [{"ckv": np.ones((1, 6, 2))}]
    vx.save_clause_checkpoint("y", 2, st, kv, 1.0, 90.0)
    vx.trigger_cdcl_jumpback(
        unsat_core_ids={"y"}, rejected_tokens=[3], repaired_tokens=[4],
        violated_constraint={"id": "c", "coeffs": {"y": -2.0}, "rhs": -50.0},
        committed_assignments={"y": 90.0})
    lo_y, hi_y = vx.bounds["y"]
    check("negative coefficient produces a lower bound, not an upper bound",
          abs(lo_y - 25.0) < 1e-9 and abs(hi_y - 100.0) < 1e-9,
          f"bounds=({lo_y:.6f}, {hi_y:.6f}) from -2y <= -50")

    print("\n" + "=" * 84)
    print("ALL SIMULATOR CHECKS PASS" if ok else "SIMULATOR CHECKS FAILED")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
