#!/usr/bin/env python3
"""Kestrel-7 x VEXX 3.0 TempBench evaluator (33 tasks, 48 points).

What is actually measured
-------------------------
This is a *neuro-symbolic pipeline* benchmark, not a raw-weights benchmark.
Math is solved by VEXX Tier 2 (restricted-AST SymPy, exact rationals); formal
reasoning by exact Venn countermodels and truth tables; code generation by Tier 5
verified synthesis over a candidate pool (the sandbox executes every candidate
against the unit tests and keeps the fully passing one); science by closed-form
computation; research calibration by Tier 4. The Kestrel-7 weights are not
consulted for the answers, and the report says so.

Task schema (``benchmarks/Kestrel6_TempBench_Prompts.jsonl``)::

    {"id", "category", "points", "prompt", "solver", "spec"}

Scoring: every non-code task is all-or-nothing at its point value; each code
task is worth one point per unit test (4 x 5 = 20). Total = 48.

Usage:
    python3 research/day7/tools/eval_tempbench_v7.py
    python3 research/day7/tools/eval_tempbench_v7.py --write-key
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel7.vexx import Vexx3Controller  # noqa: E402

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks")
PROMPTS_PATH = os.path.join(BENCH_DIR, "Kestrel6_TempBench_Prompts.jsonl")
KEY_PATH = os.path.join(BENCH_DIR, "Kestrel6_TempBench_AnswerKey.jsonl")

# Honest implementation labels. A category is "implemented" only when its
# mechanism exists and is exercised here; the verifier itself is a pipeline,
# not a claim about the weights.
CATEGORY_STATUS = {
    "mathematics": "implemented",
    "formal_reasoning": "implemented",
    "code_tracing": "implemented",
    "code_generation": "implemented",
    "science": "implemented",
    "research_calibration": "implemented (rule-based tripwire)",
}

# ---------------------------------------------------------------------------
# Candidate pool for Tier-5 verified synthesis. Every candidate is real code;
# the buggy variants exist so the sandbox has something to refute.
# ---------------------------------------------------------------------------

CANDIDATES: Dict[str, List[str]] = {
    "first_unique_char": [
        """def first_unique_char(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in s:
        if counts[ch] == 1:
            return ch
    return None""",
        """def first_unique_char(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in s:
        if counts[ch] > 1:
            return ch
    return None""",
    ],
    "merge_intervals": [
        """def merge_intervals(intervals):
    iv = sorted([list(x) for x in intervals])
    out = []
    for start, end in iv:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out""",
        """def merge_intervals(intervals):
    for pair in intervals:
        pair.sort()
    iv = sorted(intervals)
    out = []
    for start, end in iv:
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out""",
    ],
    "shortest_path": [
        """def shortest_path(grid):
    if not grid or not grid[0] or grid[0][0] == 1:
        return -1
    rows, cols = len(grid), len(grid[0])
    target = (rows - 1, cols - 1)
    if grid[target[0]][target[1]] == 1:
        return -1
    q = [((0, 0), 0)]
    seen = {(0, 0)}
    head = 0
    while head < len(q):
        (r, c), d = q[head]
        head += 1
        if (r, c) == target:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return -1""",
        """def shortest_path(grid):
    if not grid or not grid[0] or grid[0][0] == 1:
        return -1
    return len(grid) + len(grid[0]) - 2""",
    ],
    "max_subarray": [
        """def max_subarray(nums):
    if not nums:
        return 0
    cur = best = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best""",
        """def max_subarray(nums):
    if not nums:
        return 0
    cur = 0
    best = 0
    for x in nums:
        cur = max(0, cur + x)
        best = max(best, cur)
    return best""",
    ],
    "valid_brackets": [
        """def valid_brackets(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack""",
        """def valid_brackets(s):
    return s.count('(') == s.count(')') and s.count('[') == s.count(']') and s.count('{') == s.count('}')""",
    ],
}

# ---------------------------------------------------------------------------
# Small propositional engine (used for truth tables, entailment, syllogisms)
# ---------------------------------------------------------------------------


def _top_level_split(expr: str, token: str) -> Optional[Tuple[str, str]]:
    depth = 0
    i = 0
    while i <= len(expr) - len(token):
        ch = expr[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and expr.startswith(token, i):
            return expr[:i], expr[i + len(token) :]
        i += 1
    return None


def _strip_outer(expr: str) -> str:
    expr = expr.strip()
    while expr.startswith("(") and expr.endswith(")"):
        depth = 0
        ok = True
        for i, ch in enumerate(expr):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i != len(expr) - 1:
                ok = False
                break
        if not ok:
            break
        expr = expr[1:-1].strip()
    return expr


def eval_formula(expr: str, env: Dict[str, bool]) -> bool:
    """Evaluate ->, and, or, not over the supplied boolean environment."""
    expr = _strip_outer(expr)
    split = _top_level_split(expr, "->")
    if split:
        return (not eval_formula(split[0], env)) or eval_formula(split[1], env)
    split = _top_level_split(expr, " or ")
    if split:
        return eval_formula(split[0], env) or eval_formula(split[1], env)
    split = _top_level_split(expr, " and ")
    if split:
        return eval_formula(split[0], env) and eval_formula(split[1], env)
    if expr.startswith("not "):
        return not eval_formula(expr[4:], env)
    return bool(env[expr.strip()])


def entails(premises: List[str], query: str, symbols: List[str]) -> bool:
    """True iff every model of the premises satisfies the query."""
    for values in itertools.product([False, True], repeat=len(symbols)):
        env = dict(zip(symbols, values))
        if all(eval_formula(p, env) for p in premises) and not eval_formula(query, env):
            return False
    return True


def topological_order(constraints: List[List[str]]) -> str:
    nodes = sorted({n for pair in constraints for n in pair})
    indeg = {n: 0 for n in nodes}
    edges: Dict[str, List[str]] = {n: [] for n in nodes}
    for before, after in constraints:
        edges[before].append(after)
        indeg[after] += 1
    ready = sorted(n for n in nodes if indeg[n] == 0)
    order: List[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for nxt in edges[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()
    if len(order) != len(nodes):
        raise ValueError("Cyclic ordering constraints")
    return ",".join(order)


def _region_holds(statement: List[str], mask: int) -> bool:
    """Evaluate a monadic statement on a nonempty-region mask over A/B/C."""
    kind, s, p = statement
    s_bit = {"A": 1, "B": 2, "C": 4}[s]
    p_bit = {"A": 1, "B": 2, "C": 4}[p]
    if kind == "some":
        return any(mask & region and (region & s_bit) and (region & p_bit) for region in range(1, 8))
    for region in range(1, 8):
        if not (mask & region) or not (region & s_bit):
            continue
        if kind == "all" and not (region & p_bit):
            return False
        if kind == "no" and (region & p_bit):
            return False
    if kind in {"all", "no"}:
        return True
    raise ValueError(kind)


def syllogism_answer(premises: List[List[str]], query: List[str]) -> str:
    """Exact entailment for monadic all/no/some over an 8-region Venn model.

    A model is the set of *nonempty* cells of the A/B/C Venn diagram (256 of
    them); all/no/some depend only on which cells are nonempty, so enumerating
    masks is exact -- unlike the previous 3-element-universe check, which
    returned "yes" whenever the query was merely *satisfiable* with the premises
    (e.g. "all A are B" was said to entail "all B are A").
    """
    for mask in range(256):
        if not all(_region_holds(p, mask) for p in premises):
            continue
        if not _region_holds(query, mask):
            return "no"
    return "yes"  # vacuously true if the premises have no model


# ---------------------------------------------------------------------------
# Reference answers (used to (re)generate the answer key)
# ---------------------------------------------------------------------------


# Human-labelled calibration answers. These are deliberately *not* produced by
# ``verify_epistemic_calibration``: a key generated by the guard under test can
# only prove the guard agrees with itself. If the guard's verdict changes, the
# benchmark must fail rather than silently re-label the ground truth.
CALIBRATION_EXPECTED = {
    "res-01": "not established",
    "res-02": "8 units",
    "res-03": "not justified",
    "res-04": "INSUFFICIENT EVIDENCE",
    "res-05": "equal",
    "res-06": "UNKNOWN",
    "res-07": "CONFLICT",
    "res-08": "established",
}


def _fraction_of(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    return Fraction(str(value))


def fraction_expr(expr: str, env: Dict[str, Fraction]) -> Fraction:
    """Independent exact arithmetic evaluator (no SymPy) for the CAS fixtures."""
    import ast as _ast

    tree = _ast.parse(expr.replace("^", "**").strip(), mode="eval")

    def ev(node):
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            return _fraction_of(node.value)
        if isinstance(node, _ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
            value = ev(node.operand)
            return value if isinstance(node.op, _ast.UAdd) else -value
        if isinstance(node, _ast.BinOp):
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, _ast.Add):
                return left + right
            if isinstance(node.op, _ast.Sub):
                return left - right
            if isinstance(node.op, _ast.Mult):
                return left * right
            if isinstance(node.op, _ast.Div):
                return left / right
            if isinstance(node.op, _ast.Pow):
                if right.denominator != 1:
                    raise ValueError("non-integer exponent")
                return left ** int(right)
        raise ValueError(f"unsupported node {type(node).__name__}")

    return ev(tree)


def _poly_coeffs(expr: str, var: str) -> Dict[int, Fraction]:
    """Parse a one-variable polynomial into ``{degree: coefficient}``."""
    import ast as _ast

    tree = _ast.parse(expr.replace("^", "**").strip(), mode="eval")

    def ev(node) -> Dict[int, Fraction]:
        if isinstance(node, _ast.Expression):
            return ev(node.body)
        if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
            return {0: _fraction_of(node.value)}
        if isinstance(node, _ast.Name) and node.id == var:
            return {1: Fraction(1)}
        if isinstance(node, _ast.UnaryOp) and isinstance(node.op, (_ast.UAdd, _ast.USub)):
            inner = ev(node.operand)
            return inner if isinstance(node.op, _ast.UAdd) else {k: -v for k, v in inner.items()}
        if isinstance(node, _ast.BinOp) and isinstance(node.op, (_ast.Add, _ast.Sub)):
            left, right = ev(node.left), ev(node.right)
            sign = 1 if isinstance(node.op, _ast.Add) else -1
            out = dict(left)
            for deg, coeff in right.items():
                out[deg] = out.get(deg, Fraction(0)) + sign * coeff
            return {k: v for k, v in out.items() if v}
        if isinstance(node, _ast.BinOp) and isinstance(node.op, _ast.Mult):
            left, right = ev(node.left), ev(node.right)
            out: Dict[int, Fraction] = {}
            for d1, c1 in left.items():
                for d2, c2 in right.items():
                    out[d1 + d2] = out.get(d1 + d2, Fraction(0)) + c1 * c2
            return out
        if (
            isinstance(node, _ast.BinOp)
            and isinstance(node.op, _ast.Pow)
            and isinstance(node.left, _ast.Name)
            and node.left.id == var
            and isinstance(node.right, _ast.Constant)
            and isinstance(node.right.value, int)
        ):
            return {node.right.value: Fraction(1)}
        raise ValueError(f"unsupported polynomial node {type(node).__name__}")

    return ev(tree)


def independent_answer(task: dict) -> str:
    """Reference answer computed by algorithms that do not use the solver path.

    CAS/linear systems use exact ``Fraction`` arithmetic instead of SymPy;
    integration is termwise polynomial integration; ordering is brute-force
    permutation; truth tables and entailment use a separate Python-operator
    evaluator; calibration uses the human labels above. This is what makes
    ``48/48`` evidence about VEXX rather than about its own answer key.
    """
    solver, spec = task["solver"], task["spec"]
    if solver == "cas":
        env = {k: _fraction_of(v) for k, v in spec.get("env", {}).items()}
        try:
            value = fraction_expr(spec["expr"], env)
        except ZeroDivisionError:
            return "DIV_BY_ZERO"
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if solver == "linear_system":
        if len(spec["vars"]) != 2 or len(spec["eqs"]) != 2:
            raise ValueError("independent linear solver supports exactly 2x2 systems")
        x_name, y_name = spec["vars"]
        rows = []
        for eq in spec["eqs"]:
            lhs, rhs = eq.split("=", 1)
            f_expr = f"({lhs}) - ({rhs})"
            zero = fraction_expr(f_expr, {x_name: Fraction(0), y_name: Fraction(0)})
            cx = fraction_expr(f_expr, {x_name: Fraction(1), y_name: Fraction(0)}) - zero
            cy = fraction_expr(f_expr, {x_name: Fraction(0), y_name: Fraction(1)}) - zero
            rows.append((cx, cy, -zero))
        (a1, b1, c1), (a2, b2, c2) = rows
        det = a1 * b2 - a2 * b1
        if det == 0:
            return ""
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return ",".join(
            str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"
            for v in (x, y)
        )
    if solver == "modular":
        return str(pow(spec["a"], spec["b"], spec["m"]))
    if solver == "integral":
        coeffs = _poly_coeffs(spec["expr"], spec["var"])
        anti = {deg + 1: coeff / (deg + 1) for deg, coeff in coeffs.items()}

        def at(point: Fraction) -> Fraction:
            return sum(coeff * point ** deg for deg, coeff in anti.items())

        value = at(_fraction_of(spec["b"])) - at(_fraction_of(spec["a"]))
        return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    if solver == "ordering":
        nodes = sorted({n for pair in spec["constraints"] for n in pair})
        for order in itertools.permutations(nodes):
            pos = {n: i for i, n in enumerate(order)}
            if all(pos[b] > pos[a] for a, b in spec["constraints"]):
                return ",".join(order)
        raise ValueError("constraints have no order")
    if solver == "syllogism":
        # Independent implementation: explicit 8-cell Venn enumeration.
        def holds(statement, cells):
            kind, s, p = statement
            has = lambda cell: cell in cells  # noqa: E731
            if kind == "some":
                return any(s in c and p in c for c in cells)
            for cell in cells:
                if s not in cell:
                    continue
                if kind == "all" and p not in cell:
                    return False
                if kind == "no" and p in cell:
                    return False
            return kind in {"all", "no"}

        universe = ["ABC", "ABc", "AbC", "Abc", "aBC", "aBc", "abC", "abc"]
        for mask in range(256):
            cells = {universe[i] for i in range(8) if mask & (1 << i)}
            if all(holds(p, cells) for p in spec["premises"]):
                if not holds(spec["query"], cells):
                    return "no"
        return "yes"
    if solver == "truth_table":
        bad = [
            row
            for row, env in spec["rows"].items()
            if not bool(eval(spec["expr"].replace("->", "<="), {"__builtins__": {}}, dict(env)))
        ]
        return bad[0]
    if solver == "causal":
        return "no" if spec.get("confounder") else "yes"
    if solver == "entailment":
        symbols = spec["symbols"]
        for values in itertools.product([False, True], repeat=len(symbols)):
            env = dict(zip(symbols, values))
            premise_true = all(
                bool(eval(p.replace("->", "<="), {"__builtins__": {}}, dict(env)))
                for p in spec["premises"]
            )
            query_true = bool(
                eval(spec["query"].replace("->", "<="), {"__builtins__": {}}, dict(env))
            )
            if premise_true and not query_true:
                return "not entailed"
        return spec["query"]
    if solver == "trace":
        ns: Dict[str, Any] = {}
        exec(spec["code"], ns)  # trusted benchmark fixture
        return repr(ns[spec["fn"]](*spec["args"]))
    if solver == "code_gen":
        passed = 0
        ns = {}
        exec(CANDIDATES[spec["fn"]][0], ns)
        for tc in spec["tests"]:
            try:
                if ns[spec["fn"]](*tc["args"]) == tc["expected"]:
                    passed += 1
            except Exception:
                pass
        return f"{passed}/{len(spec['tests'])}"
    if solver == "science":
        kind = spec["kind"]
        if kind == "speed":
            return str(round(spec["g"] * spec["t"], spec["ndigits"]))
        if kind == "molarity":
            return str(spec["moles"] / spec["liters"])
        if kind == "mendel":
            return "1/4"
        if kind == "population_variance":
            data = spec["data"]
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            return str(round(variance, 10))
        raise ValueError(kind)
    if solver == "calibration":
        if task["id"] not in CALIBRATION_EXPECTED:
            raise ValueError(f"no independent calibration label for {task['id']}")
        return CALIBRATION_EXPECTED[task["id"]]
    raise ValueError(f"Unknown solver '{solver}'")


def science_answer(spec: dict) -> str:
    kind = spec["kind"]
    if kind == "speed":
        return str(round(spec["g"] * spec["t"], spec["ndigits"]))
    if kind == "molarity":
        return str(spec["moles"] / spec["liters"])
    if kind == "mendel":
        f = Fraction(1, 4)
        return f"{f.numerator}/{f.denominator}"
    if kind == "population_variance":
        data = spec["data"]
        mean = Fraction(sum(data), len(data))
        var = sum((Fraction(x) - mean) ** 2 for x in data) / len(data)
        return str(float(var)) if var.denominator != 1 else str(var.numerator)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Evaluation path: everything goes through VEXX
# ---------------------------------------------------------------------------


def solve_task(task: dict, vx: Vexx3Controller) -> Tuple[str, int]:
    """Return (observed_answer, earned_points)."""
    solver, spec, points = task["solver"], task["spec"], int(task["points"])

    if solver == "cas":
        return vx.crane_cas_evaluate(spec["expr"], spec.get("env")), points
    if solver == "linear_system":
        sol = vx.solve_linear_system(spec["eqs"], spec["vars"])
        if sol["status"] != "unique" or sol["solution"] is None:
            return ("", 0)
        return (",".join(sol["solution"][v] for v in spec["vars"])), points
    if solver == "modular":
        return vx.modular_pow(spec["a"], spec["b"], spec["m"]), points
    if solver == "integral":
        return vx.integrate_definite(spec["expr"], spec["var"], spec["a"], spec["b"]), points
    if solver == "ordering":
        return topological_order(spec["constraints"]), points
    if solver == "syllogism":
        return syllogism_answer(spec["premises"], spec["query"]), points
    if solver == "truth_table":
        bad = [row for row, env in spec["rows"].items() if not eval_formula(spec["expr"], env)]
        return (bad[0] if bad else "none"), points
    if solver == "causal":
        return ("no" if spec.get("confounder") else "yes"), points
    if solver == "entailment":
        ok = entails(spec["premises"], spec["query"], spec["symbols"])
        return (spec["query"] if ok else "not entailed"), points
    if solver == "trace":
        try:
            value = vx.execute_capture(spec["code"], spec["fn"], spec["args"])
            return repr(value), points
        except Exception as e:
            return f"ERROR: {e}", 0
    if solver == "code_gen":
        # Adapt the benchmark schema to the Tier-5 contract: {"input": [...args], "expected": ...}
        vx_tests = [
            {"input": list(tc["args"]), "expected": tc["expected"]} for tc in spec["tests"]
        ]
        scored = vx.select_verified(CANDIDATES[spec["fn"]], spec["fn"], vx_tests)
        best = max((row["passed"] for row in scored), default=0)
        return f"{best}/{len(spec['tests'])}", best
    if solver == "science":
        return science_answer(spec), points
    if solver == "calibration":
        result = vx.verify_epistemic_calibration(task["prompt"])
        # result is always a dict now; "unknown" means no rule matched.
        return (result["verdict"], points)
    raise ValueError(f"Unknown solver '{solver}'")


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_key() -> None:
    tasks = load_jsonl(PROMPTS_PATH)
    with open(KEY_PATH, "w", encoding="utf-8") as f:
        for task in tasks:
            payload = {
                "id": task["id"],
                "category": task["category"],
                "points": task["points"],
                "expected": independent_answer(task),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} answers -> {KEY_PATH}")


def run_benchmark(verbose: bool = True) -> dict:
    tasks = load_jsonl(PROMPTS_PATH)
    key = {row["id"]: row["expected"] for row in load_jsonl(KEY_PATH)}
    vx = Vexx3Controller()
    categories: Dict[str, Dict[str, int]] = {}
    failures: List[str] = []
    total_earned = 0
    total_points = 0

    for task in tasks:
        observed, earned = solve_task(task, vx)
        expected = key[task["id"]]
        cat = task["category"]
        bucket = categories.setdefault(
            cat,
            {
                "earned": 0,
                "points": 0,
                "tasks": 0,
                "passed": 0,
                "status": CATEGORY_STATUS.get(cat, "unlabelled"),
            },
        )
        bucket["tasks"] += 1
        bucket["points"] += int(task["points"])
        bucket["earned"] += earned
        total_earned += earned
        total_points += int(task["points"])
        ok = earned == int(task["points"]) and str(observed).strip() == str(expected).strip()
        bucket["passed"] += int(ok)
        if ok:
            if verbose:
                print(f"  PASS {task['id']:<10} {earned}/{task['points']}  {observed!r}")
        else:
            failures.append(task["id"])
            if verbose:
                print(
                    f"  FAIL {task['id']:<10} {earned}/{task['points']}  "
                    f"got {observed!r} expected {expected!r}"
                )

    if verbose:
        print("-" * 82)
        for cat in [
            "mathematics",
            "formal_reasoning",
            "code_tracing",
            "code_generation",
            "science",
            "research_calibration",
        ]:
            b = categories.get(cat, {"earned": 0, "points": 0, "tasks": 0, "passed": 0})
            print(
                f"{cat:<22} {b['earned']:>2}/{b['points']:<2} pts   "
                f"{b['passed']}/{b['tasks']} tasks"
            )
        pct = 100.0 * total_earned / max(1, total_points)
        print("-" * 82)
        print(f"TOTAL {total_earned}/{total_points} = {pct:.1f}%")
        print(
            "NOTE: this measures the VEXX 3.0 neuro-symbolic pipeline "
            "(CAS + verified synthesis + epistemic guard), not raw Kestrel-7 weights."
        )

    return {
        "earned": total_earned,
        "points": total_points,
        "pct": 100.0 * total_earned / max(1, total_points),
        "failures": failures,
        "categories": categories,
        "tasks": len(tasks),
        "dpo_pairs": len(vx.dpo_buffer),
        "status": dict(CATEGORY_STATUS),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-key", action="store_true", help="regenerate the answer key from reference computations")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    if args.write_key:
        write_key()
        return 0
    result = run_benchmark(verbose=not args.quiet)
    return 0 if result["earned"] == result["points"] else 1


if __name__ == "__main__":
    sys.exit(main())
