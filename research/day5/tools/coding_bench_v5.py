#!/usr/bin/env python3
"""Coding benchmark for the Kestrel-5 reference implementation.

This runs 10 problems of increasing difficulty through the model's *own*
machinery -- the VEXX 2.0 controller's tier-0/1 masks, exact Farkas cuts,
the checkpoint/rewind cycle -- and scores the *logic problems* on what the
pipeline can mechanically do:

  * constraint enforcement (delimiter, arity, identifier, return type)
  * branch filtering (MTP draft pruning)
  * non-syntax: rational expression evaluation (CRANE-CAS)

Each problem states its intended failure class and what a passing score
means.  Scores are about the *mechanism's* ability to constrain generation,
not about final code quality (the reference runs on random weights).

Run: python3 research/day5/tools/coding_bench_v5.py

The harness is defined first; BENCH is defined at module bottom.
"""
import importlib.util
import os
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


k5 = _load("k5model", "kestrel5/model.py")
TINY = k5.tiny_config()

_GRADE_LETTER = "ABCDEFGHIJ"


def _vocab(tokens):
    return {i: t for i, t in enumerate(tokens)}


def run_problem(p):
    """Run one problem spec; return a score dict (score = pass fraction)."""
    vexx = k5.VexxCodeNeuroSymbolicController(
        domain_bounds=p.get("bounds", {}),
        symbol_table=p.get("symbol_table", {}),
        linear_constraints=p.get("linear_constraints", []),
        max_spec_depth=p.get("max_spec_depth", 8),
    )
    t0 = time.perf_counter()
    parts = []
    for sub in p.get("mask_checks", []):
        kind = sub[0]
        if kind == "ast_guard":
            _, prefix, vocab, expect_masked, expect_ok, expected_return_type = sub
            m = vexx.apply_tier0_tier1_ast_and_scope_mask(
                np.zeros(len(vocab)), prefix, _vocab(vocab),
                expected_return_type=expected_return_type)
            for tok_id in expect_masked:
                m0 = bool(np.isneginf(m[tok_id]))
                parts.append(m0)
            for tok_id in expect_ok:
                m0 = not bool(np.isneginf(m[tok_id]))
                parts.append(m0)
        elif kind == "interval":
            _, var, cands, cap, expect_masked, expect_ok = sub
            m = vexx.apply_tier1_volume_twisted_mask(
                np.zeros(len(cands)), var, {i: v for i, v in cands}, cap)
            for tok_id in expect_masked:
                parts.append(bool(np.isneginf(m[tok_id])))
            for tok_id in expect_ok:
                parts.append(not bool(np.isneginf(m[tok_id])))
        elif kind == "shell":
            _, prefix, vocab, expect_masked, expect_ok = sub
            m = vexx.apply_vexx_shell_cli_mask(np.zeros(len(vocab)), prefix, _vocab(vocab))
            for tok_id in expect_masked:
                parts.append(bool(np.isneginf(m[tok_id])))
            for tok_id in expect_ok:
                parts.append(not bool(np.isneginf(m[tok_id])))
        elif kind == "cas":
            _, expr, env, want = sub
            try:
                got = vexx.crane_cas_rational_splice(expr, (env or None))
            except Exception:
                got = "__RAISED__"
            parts.append(got == want)
        elif kind == "branch":
            _, prefix, branches, want_survivors, expected_return_type = sub
            got = vexx.prune_mtp_speculative_tree(
                prefix_code=prefix, candidate_branches=branches,
                expected_return_type=expected_return_type)
            parts.append(got == sorted(want_survivors))
        elif kind == "farkas":
            _, coeffs, committed, want_bound, want_exact, clause_id, want_progress = sub
            # save a checkpoint for the clause so jumpback has somewhere to go
            st = [np.zeros((1, 2, 2))]
            kvc = [{"ckv": np.ones((1, 6, 2))}]
            vexx.save_clause_checkpoint(clause_id, 1, st, kvc, 1.0, committed.get(clause_id, 0.0))
            vexx.trigger_cdcl_jumpback(
                unsat_core_ids={clause_id}, rejected_tokens=[0], repaired_tokens=[1],
                violated_constraint={"id": "c", "coeffs": coeffs, "rhs": 100.0},
                committed_assignments=committed)
            lo, hi = vexx.bounds[clause_id]
            got_bound = hi if next(iter(coeffs.values())) > 0 else lo
            parts.append(abs(got_bound - want_bound) < 1e-9)
            parts.append(vexx.last_exact_fraction == Fraction(str(want_exact)))
            parts.append(vexx.last_jumpback["progress"] == want_progress)
        elif kind == "cert":
            _, key, witness, smt2, checks, want_verified = sub
            cert = vexx.emit_l3_proof_certificate(key, witness, smt2, checks=checks)
            parts.append(cert["verified"] == want_verified)
    dt = time.perf_counter() - t0
    total = len(parts)
    return {
        "id": p["id"],
        "difficulty": p["difficulty"],
        "title": p["title"],
        "intended": p["intended"],
        "score": (sum(parts) / total) if total else 1.0,
        "checks_passed": sum(parts),
        "checks_total": total,
        "seconds": dt,
    }


def benchmark(problems):
    rows = [run_problem(p) for p in problems]
    return rows


def report(rows):
    print("=" * 84)
    print("KESTREL-5 CODING BENCHMARK (Day 5 reference, random weights)")
    print("=" * 84)
    print(f"  {'id':28s} {'score':>7s} {'checks':>9s}  intended failure class")
    tot_score, tot_checks = 0.0, 0
    for r in rows:
        print(f"  {r['id']:28s} {r['score']:7.2f} {r['checks_passed']:4d}/{r['checks_total']:<4d}  {r['intended']}")
        tot_score += r["score"]
        tot_checks += r["checks_total"]
    mean = tot_score / len(rows) if rows else 0.0
    print(f"  {'MEAN':28s} {mean:7.2f} {tot_checks:9d} checks")
    print("=" * 84)
    return mean



# ================================================================ problems

SYM = {"compute": {"arity": 2, "return_type": "int"},
       "solve": {"arity": 1, "return_type": "int"},
       "check": {"arity": 1, "return_type": "bool"},
       "go": {"arity": 2, "return_type": "int"}}

PROBLEMS = [
    # ------------------------------------------------------------------ 01
    dict(id="01 balanced-lex",
         difficulty="easy", title="Balanced delimiters under string smuggling",
         intended="E1 semantic contract violated",
         symbol_table=SYM,
         mask_checks=[
             ("ast_guard", 's = "(y" + f(g', [")", "[", "{", "x", "y"],
              [0], [3, 4], None),   # ')' masked even though string held '('
             ("ast_guard", 's = "x" + g', ["}", "]", "y"], [0, 1], [2], None),
         ]),
    # ------------------------------------------------------------------ 02
    dict(id="02 arity-minimum",
         difficulty="easy", title="Minimum arity enforcement",
         intended="E2 arity / overload blind spot",
         symbol_table=SYM,
         mask_checks=[
             ("ast_guard", "compute(a", [")", ",", "z", "b"],
              [0], [1, 2, 3], None),      # too few args: ')' forbidden
         ]),
    # ------------------------------------------------------------------ 03
    dict(id="03 arity-maximum",
         difficulty="medium", title="Maximum arity enforcement",
         intended="E2 arity / overload blind spot",
         symbol_table=SYM,
         mask_checks=[
             ("ast_guard", "compute(a, b", [",", ")", "z"],
              [0], [1, 2], None),         # too many args: ',' forbidden
             ("ast_guard", "compute(a,", [",", ")", "z"],
              [0, 1], [2], None),         # trailing comma is a hard stop
         ]),
    # ------------------------------------------------------------------ 04
    dict(id="04 type-contract",
         difficulty="medium", title="Return-type contract (int vs bool)",
         intended="E1 semantic contract violated",
         symbol_table=SYM,
         mask_checks=[
             ("ast_guard", "res = ", ["compute(", "check(",
              ], [], [0], "int"),           # check() returns bool -> masked
             ("ast_guard", "res = ", ["compute(", "check("],
              [], [1], "bool"),
         ]),
    # ------------------------------------------------------------------ 05
    dict(id="05 unknown-callee",
         difficulty="medium", title="Undeclared identifier blocking",
         intended="E1 semantic contract violated",
         symbol_table=SYM,
         mask_checks=[
             ("ast_guard", "x = ", ["compute(", "evil_fn("],
              [1], [0], None),
             ("ast_guard", "x = ", ["compute(", "undeclared("],
              [1], [0], None),
         ]),
    # ------------------------------------------------------------------ 06
    dict(id="06 interval-bounds",
         difficulty="hard", title="Interval bounds with outward rounding",
         intended="E3 numerically infeasible",
         bounds={"a": (0.0, 1000.0)},
         mask_checks=[
             ("interval", "a", [(0, 0.0), (1, 1000.0), (2, -1.0), (3, 1001.0)],
              1000.0, [2, 3], [0, 1]),
             ("interval", "a", [(0, 500.0), (1, 999.0)], 999.0,
              [], [0, 1]),
         ]),
    # ------------------------------------------------------------------ 07
    dict(id="07 farkas-cut",
         difficulty="hard", title="Exact Farkas cut (upper bound)",
         intended="E3 numerically infeasible",
         bounds={"a": (0.0, 1000.0)},
         mask_checks=[
             # 2a <= rhs 100 with a committed at 240 => cut = 100/2 = 50
             ("farkas", {"a": 2.0}, {"a": 240.0}, 50.0, "50", "a", True),
         ]),
    # ------------------------------------------------------------------ 08
    dict(id="08 mtp-prune",
         difficulty="medium", title="MTP draft branch filtering",
         intended="E2 arity / overload blind spot",
         symbol_table=SYM,
         max_spec_depth=8,
         mask_checks=[
             ("branch", "res = ", {
                 "good":     ["compute(", "x", ",", "y", ")"],
                 "bad_arity": ["compute(", "x", ")"],
                 "unknown":  ["evil_fn(", "x", ")"],
                 "unbalanced": [")", "x"],
                 "deep":     ["x"] * 9,
             }, ["good"], "int"),
         ]),
    # ------------------------------------------------------------------ 09
    dict(id="09 cas-eval",
         difficulty="hard", title="Exact rational expression evaluation",
         intended="E4 AST-opaque (macro / codegen)",
         mask_checks=[
             ("cas", "3/7 + 2/5 * x - 1/35", {"x": "3/2"}, "1"),
             ("cas", "1/3 + 1/6", None, "1/2"),
             ("cas", "-2 + -3 * -4", None, "10"),
             ("cas", "x / 3 + y / 7", {"x": "3/2", "y": "2/3"},
              str(Fraction(1, 2) + Fraction(2, 21))),
         ]),
    # ------------------------------------------------------------------ 10
    dict(id="10 l3-certificate",
         difficulty="extremely hard", title="L3 proof certificate soundness",
         intended="E4 AST-opaque (macro / codegen)",
         mask_checks=[
             ("cert", "pkg.auth.Session", {"w1": 200, "w2": 100},
              "(assert (<= (+ (* 2 w1) w2) 500))",
              [({"w1": 2, "w2": 1}, 500)], True),
             ("cert", "pkg.auth.Session", {"w1": 240, "w2": 100},
              "(assert (<= (+ (* 2 w1) w2) 500))",
              [({"w1": 2, "w2": 1}, 500)], False),
             ("cert", "k", {"x": 3}, "(assert (= x 3))", [], False),
         ]),
]


def main():
    rows = benchmark(PROBLEMS)
    mean = report(rows)
    # Honest performance summary of what was actually measured.
    solved = sum(1 for r in rows if r["score"] == 1.0)
    print(f"  problems at perfect score : {solved}/10")
    print(f"  total constraint checks   : {sum(r['checks_total'] for r in rows)} "
          f"({sum(r['checks_passed'] for r in rows)} passed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
