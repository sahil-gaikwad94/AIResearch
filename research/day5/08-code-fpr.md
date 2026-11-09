# 08 — The disjoint code false-positive-rate split

The target: a composite FPR of **0.0004996** for VEXX 2.0 (static VEXX-Code:
0.0027756). A single number like that is useless without a decomposition, so
the negative suite is partitioned into four **mutually exclusive** failure
classes and each verification arm leaves a residual count per class. The arm's
FPR is then the *exact sum* of four rational contributions.

## The decomposition (N = 2,500,000 negatives)

| class | negatives | static | +exact-rational | +mutation |
|---|---|---|---|---|
| E1 semantic contract violated | 1,000,000 | 1,200 | 420 | 150 |
| E2 arity / overload blind spot | 625,000 | 2,500 | 640 | 250 |
| E3 numerically infeasible | 625,000 | 1,125 | 460 | 300 |
| E4 AST-opaque (macro/codegen) | 250,000 | 2,114 | 980 | 549 |
| **total** | **2,500,000** | **6,939** | **2,500** | **1,249** |

Arm FPRs as exact fractions:

| arm | FPR |
|---|---|
| VEXX-Code static (Tier 0/1) | `6939/2500000 = 0.0027756` |
| + exact-rational feasibility (Tier 2/3) | `2500/2500000 = 0.0010000` |
| VEXX 2.0 (+ mutation/execution oracle) | `1249/2500000 = 0.0004996` |

The per-class contributions sum to the arm FPR exactly (verified as rational
arithmetic, not as floats), and no class is made worse by adding a gate —
the ablation is monotone in every row, which is the property a layered
verifier must have.

## What the numbers say

- Exact-rational feasibility buys more on E3 (numeric infeasibility) than
  anything else — 1,125 → 460 — because that is the failure class it exists to
  catch.
- The mutation oracle buys the most on E2 (2,500 → 250): it actually executes,
  and execution is the ground truth for arity.
- E4 (AST-opaque) is the hard core in every arm: exact rational reasoning
  cannot see through macros and code generation, and even the composite arm
  leaves 549 of 2,114. If you are looking for where the 0.0004996 comes from,
  it is 44% this class.

## Monte Carlo agreement

The simulator samples the per-class Bernoulli model (seed 20261109) and
compares the empirical rate to the analytic one:

| arm | FP | empirical FPR | exact | z |
|---|---|---|---|---|
| static | 6,908 | 0.0027632 | 0.0027756 | 0.37 |
| +exact-rational | 2,541 | 0.0010164 | 0.0010000 | 0.82 |
| composite | 1,257 | 0.0005028 | 0.0004996 | 0.23 |

All three arms land within 5σ. This proves the *sampling model is internally
consistent with the analytic rates* — it does not prove the rates are right.

## Calibration protocol (what has to happen to make these numbers real)

The per-class residuals above are **calibrated operating points, not
measurements**. The simulator's job is to keep the decomposition honest
(disjointness, additivity, monotonicity, MC agreement); producing the residuals
requires measurement, which is a GPU-and-infra job. The protocol:

1. Build a held-out negative suite of 2.5M code fragments drawn from the four
   classes, labelled by *which* class, with the labelling procedure published.
2. Run each arm in isolation and record per-class residual FPs.
3. Reject any configuration where an added gate makes a class worse
   (non-monotonicity ⇒ the gate is broken, not "noisy").
4. Re-run the Monte Carlo check; only then quote 0.0004996.

Until step 2 has run, `FPR_VEXX2_PLUS_MUTANT = 0.0004996` is a *design target
with an exact arithmetic structure*, and this directory says so. The constant
is exported under that name precisely so the audit can pin it — pinning a
constant is not the same as measuring it.
