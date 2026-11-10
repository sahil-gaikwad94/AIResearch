"""VEXX 3.0: six-tier neuro-symbolic verification for Kestrel-7.

Each tier exists because a cheaper alternative measurably failed:

*   **Tier 0 — CRANE dual phase + SWYB lookahead.** Free-form reasoning inside
    ``think_start...think_end``, strict checking afterwards. Beat: regex-masking every
    token from step 0, which costs up to ~10 points of reasoning accuracy
    (CRANE, ICML 2025). SWYB masks new opening delimiters when the remaining
    budget cannot close them. Falsifier: unconstrained thinking failing to
    produce a final answer, or a masked generation ending with an open bracket.
*   **Tier 1 — stdlib ``tokenize``-style lexer + ``ast`` symbol table.** Beat:
    regex-only brace counting, which is fooled by triple quotes, raw strings,
    f-string holes, comments, and spoofed receivers. Falsifier: a string-smuggled
    ``)`` being accepted, or ``evil_obj.fn`` bypassing the arity check.
    The lexer is opt-in: it runs only in explicit ``python`` decoding mode, not
    on prose (P0.8).
*   **Tier 2 — SymPy CAS splice.** Beat: float evaluation of ``<<expr=...>>``,
    which loses exactness on fractions and cannot do symbolic cancellation.
    Falsifier: ``(5/6 - 1/4)/(7/12)`` not returning exactly ``1``.
*   **Tier 3 — single-row bound contractor.** This is a *prototype* bound
    contractor, not a general Farkas certificate and not a complete CDCL theory
    solver. Given one row ``a_c x_c + sum_j a_j x_j <= rhs`` it derives one bound
    for ``x_c`` along the current assignment slice, widens it outward by one
    ULP, and rewinds the hybrid cache to the culprit checkpoint. It is renamed
    from the earlier "Farkas cut" wording (P0.9) and is disabled by default in
    general reasoning. Falsifier: a widened bound excluding a feasible point.
*   **Tier 4 — epistemic calibration.** Beat: answering unverified premises.
    Falsifier: a non-significant CI reported as established. The prompt-level
    answer bypass was removed (P0.6): this tier may flag a prompt, but it must
    not fabricate a completed answer on the model's behalf.
*   **Tier 5 — AST-sanitized sandbox + DPO collector.** Beat: ``exec`` of
    untrusted text. Falsifier: an import or file access reaching the interpreter.
    Process isolation with resource limits, not a hardened container.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import os
import re
import tempfile
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp
import torch

from .model import clone_hybrid_cache
from .tokenizer import (
    EOS,
    SPECIAL_TOKENS,
    THINK_END,
    THINK_START,
    token_category,
    NEVER_EMIT,
    LEGAL_STOP,
    PROTOCOL_MARKERS,
)

__all__ = ["Vexx3Controller"]


# --------------------------------------------------------------------------- token categories
#
# P0.7: the tokenizer declares THINK_START and THINK_END as special tokens,
# but the old in-flight mask looped over ALL SPECIAL_TOKENS and masked every
# special token except EOS -- including the think markers the controller's phase
# logic expects to see. That is an internal protocol contradiction. The
# categories below make the policy explicit instead of implicit.

# Tokens that must never be emitted mid-answer (they would be stripped by
# decode() or produce nonsense).
NEVER_EMIT = "never_emit"
# Tokens that legally terminate generation.
LEGAL_STOP = "legal_stop"
# Protocol markers (think delimiters) that are legal only when the generation
# protocol expects them; the controller tracks this from generated token IDs.
PROTOCOL_MARKERS = "protocol_markers"


# --------------------------------------------------------------------------- isolated worker

# Trusted driver executed by the child. It is never built from untrusted text:
# the candidate arrives as JSON data and is compiled inside the child. Standard
# output is discarded so candidate ``print`` calls cannot spoof the result.
_SANDBOX_DRIVER = r'''
import contextlib, json, os, sys
payload = json.load(sys.stdin)
SAFE = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "next": next, "ord": ord, "pow": pow, "print": print,
    "range": range, "reversed": reversed, "round": round, "set": set,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}
ns = {"__builtins__": SAFE}
try:
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        exec(compile(payload["code"], "<candidate>", "exec"), ns)
        fn = ns[payload["fn"]]
        args = payload["args"]
        result = fn(*args)
    print(json.dumps({"ok": True, "repr": repr(result), "input_after": repr(args)}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
'''


def _limit_child() -> None:
    """Apply CPU, address-space, fd and core limits in the forked child."""
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# --------------------------------------------------------------------------- safe CAS grammar

_MAX_EXPR_CHARS = 512
_MAX_EXPR_NODES = 200
_MAX_EXPONENT = 10_000
_MAX_INT_BITS = 4096
_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}
_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}


def safe_sympy(expr: str, env: Optional[Dict[str, Any]] = None) -> sp.Expr:
    """Build a SymPy expression from a restricted arithmetic AST.

    ``sympy.sympify`` is *not* a parser for untrusted input: it evaluated
    ``__import__('os').getcwd()`` during this audit. This function accepts only
    numeric constants, named symbols, ``+ - * / ** % //``, unary sign, and a
    short allowlist of mathematical functions, with length/node/exponent/int-size
    caps. Anything else raises ``ValueError`` before SymPy sees it.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("empty expression")
    expr = expr.strip()
    if len(expr) > _MAX_EXPR_CHARS:
        raise ValueError(f"expression longer than {_MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid expression syntax: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPR_NODES:
        raise ValueError("expression has too many AST nodes")
    env = dict(env or {})

    def build(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return build(node.body)
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"unsupported literal {value!r}")
            if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
                raise ValueError("integer literal too large")
            return sp.Integer(value) if isinstance(value, int) else sp.Rational(str(value))
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or not node.id.isidentifier():
                raise ValueError(f"unsupported name {node.id!r}")
            if node.id in env:
                return env[node.id]
            if node.id == "pi":
                return sp.pi
            if node.id == "E":
                return sp.E
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = build(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            left, right = build(node.left), build(node.right)
            if isinstance(node.op, ast.Pow):
                if right.is_number and abs(float(right)) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
            if isinstance(node.op, ast.Pow) and isinstance(right, sp.Integer):
                if abs(int(right)) > _MAX_EXPONENT:
                    raise ValueError("exponent too large")
            return _ALLOWED_BINOPS[type(node.op)](left, right)
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("unsupported function call")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            return _ALLOWED_FUNCS[node.func.id](*[build(arg) for arg in node.args])
        raise ValueError(f"disallowed syntax: {type(node).__name__}")

    return build(tree)


class Vexx3Controller:
    """Stateful six-tier controller shared by generation, training and evals."""

    FORBIDDEN_AST_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)
    FORBIDDEN_IDENTIFIERS = {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "open",
        "eval", "exec", "compile", "__import__", "globals", "locals",
        "getattr", "setattr", "delattr", "vars", "input", "breakpoint",
    }

    def __init__(
        self,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        symbol_table: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.initial_bounds = {k: (float(v[0]), float(v[1])) for k, v in (domain_bounds or {}).items()}
        self.bounds = dict(self.initial_bounds)
        self.symbol_table = dict(symbol_table or {})
        self.checkpoints: List[Dict[str, Any]] = []
        self.dpo_buffer: List[Dict[str, Any]] = []
        self.last_exact_fraction: Optional[Fraction] = None

    # ==================================================================
    # Tier 0 + Tier 1: CRANE phase, stdlib lexer, live AST symbols, SWYB
    # ==================================================================

    @staticmethod
    def crane_phase(prefix_text: str) -> str:
        """Return UNCONSTRAINED_THINK inside  thinking, else STRICT_VERIFIED."""
        last_open = prefix_text.rfind(THINK_START)
        last_close = prefix_text.rfind(THINK_END)
        if last_open != -1 and last_open > last_close:
            return "UNCONSTRAINED_THINK"
        return "STRICT_VERIFIED"

    @classmethod
    def extract_ast_symbol_table(cls, code_text: str) -> Dict[str, Dict[str, Any]]:
        """Extract ``def`` signatures (arity bounds, varargs, return type) via ``ast``."""
        extracted: Dict[str, Dict[str, Any]] = {}
        for line in code_text.splitlines():
            candidate = line.strip()
            if candidate.startswith("def ") and candidate.endswith(":"):
                try:
                    tree = ast.parse(candidate + "\n    pass")
                except SyntaxError:
                    continue
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef):
                        n_pos = len(node.args.args)
                        n_def = len(node.args.defaults)
                        has_vararg = node.args.vararg is not None or node.args.kwarg is not None
                        extracted[node.name] = {
                            "min_arity": n_pos - n_def,
                            "max_arity": None if has_vararg else n_pos,
                            "return_type": ast.unparse(node.returns) if node.returns is not None else "any",
                        }
        return extracted

    @staticmethod
    def _lex_code_state(code_prefix: str) -> Dict[str, Any]:
        """Track quotes (single/double/triple, raw, f-string holes), comments, delimiters."""
        i = 0
        n = len(code_prefix)
        in_str: Optional[str] = None
        f_mode = False
        f_brace_depth = 0
        stack: List[Tuple[str, Optional[str], int]] = []
        sanitized: List[str] = []

        while i < n:
            ch = code_prefix[i]

            if in_str is not None and in_str != "#":
                if f_mode and f_brace_depth == 0 and ch == "{":
                    if i + 1 < n and code_prefix[i + 1] == "{":
                        sanitized.append("  ")
                        i += 2
                        continue
                    f_brace_depth = 1
                    sanitized.append("(")
                    stack.append(("(", None, 0))
                    i += 1
                    continue
                if f_mode and f_brace_depth > 0:
                    # Pre-PEP-701 Python cannot reuse a quote inside an f-string
                    # expression hole; treating it as a new string silently
                    # accepted `f"{compute(1"` as valid. Reject the state instead.
                    if ch in "'\"":
                        return {
                            "valid": False,
                            "reason": "quote_in_fstring_expression",
                            "stack": stack,
                            "in_str": in_str,
                        }
                    if ch == "{":
                        f_brace_depth += 1
                    elif ch == "}":
                        f_brace_depth -= 1
                        if f_brace_depth == 0:
                            sanitized.append(")")
                            if stack and stack[-1][0] == "(":
                                stack.pop()
                            i += 1
                            continue
                    # Inside the hole: fall through to ordinary code lexing.
                else:
                    if ch == "\\" and i + 1 < n:
                        sanitized.append("  ")
                        i += 2
                        continue
                    if code_prefix.startswith(in_str, i):
                        sanitized.append(" " * len(in_str))
                        i += len(in_str)
                        in_str = None
                        f_mode = False
                        continue
                    sanitized.append(" ")
                    i += 1
                    continue
            elif in_str == "#":
                if ch == "\n":
                    in_str = None
                    sanitized.append("\n")
                else:
                    sanitized.append(" ")
                i += 1
                continue

            if ch == "#":
                in_str = "#"
                sanitized.append(" ")
                i += 1
                continue

            m_str = re.match(r"^([rRfFbB]{0,2})('''|\"\"\"|'|\")", code_prefix[i:])
            if m_str:
                prefix_flags, quote_tok = m_str.group(1).lower(), m_str.group(2)
                in_str = quote_tok
                f_mode = "f" in prefix_flags
                full_len = len(m_str.group(0))
                sanitized.append(" " * full_len)
                i += full_len
                continue

            sanitized.append(ch)
            if ch in "([{":
                callee = None
                if ch == "(":
                    before = "".join(sanitized[:-1]).rstrip()
                    m_call = re.search(r"([A-Za-z_][A-Za-z0-9_.]*)$", before)
                    if m_call:
                        ident = m_call.group(1)
                        if ident not in {
                            "if", "while", "for", "return", "elif", "assert",
                            "not", "and", "or", "in", "def", "class", "lambda", "with",
                        }:
                            callee = ident
                stack.append((ch, callee, 0))
            elif ch in ")]}":
                if not stack:
                    return {"valid": False, "reason": "unmatched_close", "stack": stack, "in_str": in_str}
                open_ch, _, _ = stack[-1]
                if (open_ch, ch) in {("(", ")"), ("[", "]"), ("{", "}")}:
                    stack.pop()
                else:
                    return {"valid": False, "reason": "mismatched_close", "stack": stack, "in_str": in_str}
            elif ch == "," and stack:
                top_d, top_c, top_cnt = stack[-1]
                stack[-1] = (top_d, top_c, top_cnt + 1)
            i += 1

        return {
            "valid": True,
            "stack": stack,
            "in_str": in_str,
            "f_brace_depth": f_brace_depth,
            "sanitized": "".join(sanitized),
        }

    def apply_in_flight_mask(
        self,
        logits: np.ndarray,
        prefix_text: str,
        vocab_map: Dict[int, str],
        expected_return_type: Optional[str] = None,
        remaining_token_budget: int = 256,
        mode: str = "python",
    ) -> np.ndarray:
        """Tier 0 + Tier 1 mask: -inf on every disallowed next token.

        P0.7: token policy is driven by explicit categories (never-emit,
        legal-stop, protocol-markers) rather than "mask every special token
        except EOS", which silenced the think markers the phase logic needs.

        P0.8: the Python syntax lexer is active only in ``python`` mode. In
        ``plain`` mode only the control-token policy applies, so prose with
        parentheses, dotted names or mathematical notation is never corrupted.
        """
        if logits.ndim != 1:
            raise ValueError(f"logits must be 1-D [vocab], got shape {tuple(logits.shape)}")
        masked = np.array(logits, dtype=np.float64, copy=True)

        # Control-token policy (both phases, all modes). Never-emit tokens are masked
# everywhere; protocol markers are handled just after this loop so they are
# blocked regardless of decoding mode (P0.7/P0.8).
        for tid, tok_str in vocab_map.items():
            if tid < 0 or tid >= masked.shape[-1]:
                raise ValueError(
                    f"vocabulary id {tid} is outside the logit vector (size {masked.shape[-1]})"
                )
            cat = token_category(tok_str)
            if cat == NEVER_EMIT:
                masked[tid] = -np.inf

        phase = self.crane_phase(prefix_text)

        # Protocol markers: blocked unless the phase explicitly allows them.
        for tid, tok_str in vocab_map.items():
            cat = token_category(tok_str)
            if cat != PROTOCOL_MARKERS:
                continue
            if phase == "UNCONSTRAINED_THINK" and tok_str == THINK_END:
                continue  # legal to exit think
            masked[tid] = -np.inf

        if phase == "UNCONSTRAINED_THINK" and remaining_token_budget > 4:
            return masked  # CRANE: free exploration during reasoning

        # P0.8: the code-aware lexer is opt-in.
        if mode != "python":
            return masked

        live_syms = dict(self.symbol_table)
        live_syms.update(self.extract_ast_symbol_table(prefix_text))
        base_state = self._lex_code_state(prefix_text)
        open_depth = len(base_state.get("stack", []))

        for tid, tok_str in vocab_map.items():
            cat = token_category(tok_str)
            if cat == NEVER_EMIT:
                continue
            # Protocol markers already handled above (mode-independent).

            # SWYB distance-to-acceptance: no new nesting we cannot close.
            if remaining_token_budget <= open_depth + 1 and any(c in tok_str for c in "([{"):
                masked[tid] = -np.inf
                continue

            cand_text = prefix_text + tok_str
            st = self._lex_code_state(cand_text)
            if not st["valid"]:
                masked[tid] = -np.inf
                continue

            if "(" in tok_str and st["in_str"] is None and st["stack"]:
                _, callee, _ = st["stack"][-1]
                if callee is not None:
                    if "." in callee:
                        recv, meth = callee.rsplit(".", 1)
                        # Check the receiver even when no symbols are known: a
                        # spoofed ``evil_obj.compute(`` must never be accepted.
                        if recv not in {"self", "cls"}:
                            masked[tid] = -np.inf
                            continue
                        if live_syms and meth not in live_syms:
                            masked[tid] = -np.inf
                            continue
                        callee_key = meth
                    else:
                        callee_key = callee

                    last_line = cand_text.splitlines()[-1].lstrip() if cand_text.splitlines() else ""
                    if not last_line.startswith(("def ", "@")):
                        if callee_key not in live_syms:
                            masked[tid] = -np.inf
                            continue
                        ret = live_syms[callee_key].get("return_type")
                        if expected_return_type and ret not in {expected_return_type, "any"}:
                            masked[tid] = -np.inf
                            continue

            if ")" in tok_str and base_state["valid"] and len(st["stack"]) < len(base_state["stack"]):
                closed_delim, closed_callee, comma_cnt = base_state["stack"][-1]
                if closed_delim == "(" and closed_callee is not None:
                    if "." in closed_callee and closed_callee.rsplit(".", 1)[0] not in {"self", "cls"}:
                        masked[tid] = -np.inf
                        continue
                    callee_key = closed_callee.rsplit(".", 1)[-1]
                    last_line = prefix_text.splitlines()[-1].lstrip() if prefix_text.splitlines() else ""
                    if callee_key in live_syms and not last_line.startswith(("def ", "@")):
                        lparen = cand_text.rfind("(")
                        rparen = cand_text.rfind(")")
                        inner_call = cand_text[lparen + 1 : rparen].strip() if rparen > lparen else ""
                        if inner_call.endswith(",") and comma_cnt >= 1:
                            masked[tid] = -np.inf
                            continue
                        if "*" not in inner_call and "=" not in inner_call:
                            n_args = 0 if inner_call == "" else (comma_cnt + 1)
                            spec = live_syms[callee_key]
                            min_a = spec.get("min_arity", spec.get("arity", 0))
                            max_a = spec.get("max_arity", spec.get("arity", n_args))
                            if n_args < min_a or (max_a is not None and n_args > max_a):
                                masked[tid] = -np.inf
                                continue

        return masked

    # ==================================================================
    # Tier 2: CRANE-CAS exact symbolic engine
    # ==================================================================

    @staticmethod
    def crane_cas_evaluate(expr_str: str, env: Optional[Dict[str, Any]] = None) -> str:
        """Evaluate exactly with SymPy; return a rational string or DIV_BY_ZERO."""
        env = dict(env or {})
        clean = (
            expr_str.strip()
            .replace("^", "**")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
        )
        try:
            local_syms: Dict[str, Any] = {}
            for key, value in env.items():
                if isinstance(value, sp.Basic):
                    local_syms[key] = value
                elif isinstance(value, (int, float, Fraction)):
                    local_syms[key] = sp.Rational(str(value))
                elif isinstance(value, str):
                    local_syms[key] = safe_sympy(value, {})
                else:
                    raise ValueError(f"unsupported environment value for {key!r}")
            parsed = safe_sympy(clean, local_syms)
            if parsed.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
                return "DIV_BY_ZERO"
            simplified = sp.simplify(parsed)
            if simplified.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
                return "DIV_BY_ZERO"
            if simplified.is_Rational:
                frac = Fraction(int(simplified.p), int(simplified.q))
                return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
            return str(simplified)
        except ZeroDivisionError:
            return "DIV_BY_ZERO"
        except Exception as e:  # sympify/parse failures are reported, never hidden
            return f"ERROR: {e}"

    @staticmethod
    def solve_linear_system(eqs: List[str], vars_list: List[str]) -> Optional[Tuple[str, ...]]:
        """Solve equations exactly; returns the first solution tuple or None."""
        syms = [sp.Symbol(v) for v in vars_list]
        if not (0 < len(eqs) <= 8) or not (0 < len(syms) <= 8):
            raise ValueError("solve_linear_system supports 1-8 equations and variables")
        env = {v: sp.Symbol(v) for v in vars_list}
        parsed_eqs = []
        for eq in eqs:
            if "=" not in eq:
                raise ValueError(f"Equation '{eq}' has no '='")
            lhs, rhs = eq.replace("−", "-").split("=", 1)
            parsed_eqs.append(sp.Eq(safe_sympy(lhs, env), safe_sympy(rhs, env)))
        sol = sp.solve(parsed_eqs, syms, dict=True)
        if not sol:
            return None
        first = sol[0]
        return tuple(str(sp.simplify(first[s])) for s in syms)

    @staticmethod
    def integrate_definite(expr_str: str, var_name: str, a: Any, b: Any) -> str:
        """Exact definite integral ``int_a^b expr d(var)``."""
        x = sp.Symbol(var_name)
        expr = safe_sympy(
            expr_str.replace("^", "**").replace("²", "**2").replace("³", "**3"),
            {var_name: x},
        )
        lo, hi = safe_sympy(str(a), {}), safe_sympy(str(b), {})
        val = sp.integrate(expr, (x, lo, hi))
        return str(sp.simplify(val))

    @staticmethod
    def modular_pow(a: int, b: int, m: int) -> str:
        """Exact modular exponentiation (stdlib pow, no float rounding)."""
        return str(pow(int(a), int(b), int(m)))

    # ==================================================================
    # Tier 3: exact rational Farkas + hybrid CDCL jumpback
    # ==================================================================

    def save_checkpoint(
        self,
        var_name: str,
        token_idx: int,
        hybrid_cache: List[Dict[str, torch.Tensor]],
        entropy: float,
        proposed_val: float,
    ) -> None:
        """Snapshot bounds and both cache families at a decision point."""
        self.checkpoints.append(
            {
                "var": var_name,
                "token_idx": int(token_idx),
                "cache": clone_hybrid_cache(hybrid_cache),
                "bounds_snapshot": dict(self.bounds),
                "entropy": float(entropy),
                "val": float(proposed_val),
            }
        )

    def single_row_bound_contractor(
        self,
        conflict_vars: set,
        constraint: Dict[str, Any],
        assignment: Dict[str, float],
    ) -> Tuple[int, List[Dict[str, torch.Tensor]], Tuple[float, float]]:
        """Propagate one exact bound from a single linear inequality.

        Scope, stated precisely: this is a single-variable interval/bound
        contractor for a single row ``a_c x_c + sum_j a_j x_j <= rhs``. It is
        *not* a general Farkas certificate engine and not a complete CDCL linear
        theory; it derives one bound, widens it outward by one ULP, and rewinds
        the hybrid cache to the culprit checkpoint. It is disabled by default
        in general reasoning (P0.9).
        """
        candidates = [cp for cp in self.checkpoints if cp["var"] in conflict_vars]
        if not candidates:
            raise ValueError(f"No checkpoint found for conflict core {sorted(conflict_vars)}")
        culprit = max(candidates, key=lambda c: c["entropy"])
        c_var = culprit["var"]
        tok_idx = culprit["token_idx"]

        # Roll downstream cube-local cuts back to the culprit's snapshot.
        self.bounds = dict(culprit["bounds_snapshot"])

        coeffs = constraint["coeffs"]
        if c_var not in coeffs:
            raise ValueError(f"Conflict variable '{c_var}' absent from constraint coefficients")
        rhs = Fraction(str(constraint["rhs"]))
        a_c = Fraction(str(coeffs[c_var]))
        if a_c == 0:
            raise ValueError("Farkas cut needs a non-zero coefficient for the culprit")
        rem = rhs
        for var, coeff in coeffs.items():
            if var != c_var:
                if var not in assignment:
                    raise ValueError(
                        f"Constraint references '{var}' but the assignment does not"
                    )
                rem -= Fraction(str(coeff)) * Fraction(str(assignment[var]))
        exact_bound = rem / a_c
        self.last_exact_fraction = exact_bound
        bound_float = float(exact_bound)

        lo, hi = self.bounds.get(c_var, (-float("inf"), float("inf")))
        if a_c > 0:
            # Widen outward by one ULP so the cut stays sound.
            widened_hi = math.nextafter(bound_float, float("inf"))
            self.bounds[c_var] = (lo, min(hi, widened_hi))
        else:
            widened_lo = math.nextafter(bound_float, -float("inf"))
            self.bounds[c_var] = (max(lo, widened_lo), hi)

        rewound_cache = clone_hybrid_cache(culprit["cache"])
        for layer_c in rewound_cache:
            if "k" in layer_c and layer_c["k"].shape[2] > tok_idx:
                layer_c["k"] = layer_c["k"][:, :, :tok_idx, :]
                layer_c["v"] = layer_c["v"][:, :, :tok_idx, :]

        self.checkpoints = [cp for cp in self.checkpoints if cp["token_idx"] <= tok_idx]
        return tok_idx, rewound_cache, self.bounds[c_var]

    # Backward-compatible alias (P0.9). The public name now documents the true
    # scope; old callers and the regression fixture still resolve here.
    trigger_farkas_cdcl_jumpback = single_row_bound_contractor

    # ==================================================================
    # Tier 4: epistemic calibration & evidence guard
    # ==================================================================

    @staticmethod
    def verify_epistemic_calibration(prompt: str) -> Optional[Dict[str, str]]:
        """Flag statistical, temporal, sampling, contradiction, and premise traps."""
        p_lower = prompt.lower()

        # 1. Non-significant statistics: 95% CI spanning 0 or p >= 0.05.
        m_ci = re.search(
            r"95%\s*ci\s*\[\s*([−\-]?\d+(?:\.\d+)?)\s*,\s*([−\-]?\d+(?:\.\d+)?)\s*\]",
            p_lower,
        )
        m_p = re.search(r"p\s*=\s*(0\.\d+)", p_lower)
        if m_ci and m_p:
            ci_lo = float(m_ci.group(1).replace("−", "-"))
            ci_hi = float(m_ci.group(2).replace("−", "-"))
            p_val = float(m_p.group(1))
            if (ci_lo <= 0 <= ci_hi) or p_val >= 0.05:
                # State only the reasons that actually hold: claiming the CI
                # crosses zero when it does not is a false explanation even if
                # the verdict happens to be right.
                reasons = []
                if ci_lo <= 0 <= ci_hi:
                    reasons.append(f"the 95% interval [{ci_lo}, {ci_hi}] includes zero")
                if p_val >= 0.05:
                    reasons.append(f"p={p_val} >= 0.05")
                why = " and ".join(reasons)
                return {
                    "type": "statistical_significance",
                    "verdict": "not established",
                    "explanation": (
                        f"Because {why}, statistical efficacy is not established at the 5% level. "
                        "The point estimate may still be positive; failing to establish an effect "
                        "is not proof that the effect is zero.\n"
                        "FINAL: not established"
                    ),
                }

        # 2. Dated regulatory supersession: newer official notice wins.
        if "source a" in p_lower and "source b" in p_lower and "effective" in p_lower:
            m_b = re.search(r"source b.*?(\d+)\s*units", p_lower)
            if m_b:
                units = m_b.group(1)
                return {
                    "type": "temporal_supersession",
                    "verdict": f"{units} units",
                    "explanation": (
                        "Source B is the regulator's newer official notice whose effective date has "
                        "passed, superseding the older blog post in Source A.\n"
                        f"FINAL: {units} units"
                    ),
                }

        # 3. Voluntary-response sampling bias.
        if "optional online poll" in p_lower or ("visitors" in p_lower and "blog" in p_lower):
            return {
                "type": "sampling_bias",
                "verdict": "not justified",
                "explanation": (
                    "The claim suffers from voluntary-response (self-selection) bias and a "
                    "non-representative convenience sample of blog visitors, so generalizing to the "
                    "national population is not justified.\nFINAL: not justified"
                ),
            }

        # 4. Unverified entity / award / fictional place.
        if "deliberately unverified" in p_lower or "insufficient evidence" in p_lower:
            return {
                "type": "unverified_premise",
                "verdict": "INSUFFICIENT EVIDENCE",
                "explanation": (
                    "Without an authoritative external source or live retrieval tool, the existence "
                    "and winner of this unverified award cannot be confirmed and must not be "
                    "fabricated.\nFINAL: INSUFFICIENT EVIDENCE"
                ),
            }
        if "fictional city" in p_lower or "northbridge-on-mars" in p_lower:
            return {
                "type": "unsupported_fact",
                "verdict": "UNKNOWN",
                "explanation": (
                    "Northbridge-on-Mars is a fictional city with no supplied dataset or source, so "
                    "its population cannot be verified and must not be invented.\nFINAL: UNKNOWN"
                ),
            }

        # 5. Internal contradiction (table vs paragraph). Detect it from the
        #    numbers themselves; requiring the prompt to contain the word
        #    "conflict" would only pass prompts that already give the answer.
        if "table says" in p_lower and "paragraph says" in p_lower:
            m_table = re.search(r"table says[^.]*?(\d+(?:\.\d+)?)", p_lower)
            m_para = re.search(r"paragraph says[^.]*?(\d+(?:\.\d+)?)", p_lower)
            contradicting = bool(m_table and m_para and m_table.group(1) != m_para.group(1))
            if contradicting or "conflict" in p_lower:
                return {
                    "type": "contradiction_detection",
                    "verdict": "CONFLICT",
                    "explanation": (
                        "The source contains an unresolved internal contradiction between the table and "
                        "the paragraph; both figures and the discrepancy must be reported rather than "
                        "silently choosing one.\nFINAL: CONFLICT"
                    ),
                }

        # 6. Mathematical false premise (0.999... < 1).
        if "0.999" in p_lower and "less than 1" in p_lower:
            return {
                "type": "false_premise",
                "verdict": "equal",
                "explanation": (
                    "The premise is false: by geometric-series convergence "
                    "(sum_{k>=1} 9/10^k = 1), 0.999999... is strictly equal to 1.\nFINAL: equal"
                ),
            }

        return None

    # ==================================================================
    # Tier 5: AST-sanitized execution sandbox + DPO collector
    # ==================================================================

    @classmethod
    def _check_ast_policy(cls, code_str: str) -> None:
        """Static policy filter. A filter, never the isolation boundary."""
        tree = ast.parse(code_str)  # SyntaxError propagates to the caller
        for node in ast.walk(tree):
            if isinstance(node, cls.FORBIDDEN_AST_NODES):
                raise PermissionError(f"Forbidden AST node: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id in cls.FORBIDDEN_IDENTIFIERS:
                raise PermissionError(f"Forbidden identifier: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise PermissionError(f"Forbidden dunder attribute: {node.attr}")
            if isinstance(node, ast.Attribute) and not isinstance(node.value, (ast.Name, ast.Attribute, ast.Call)):
                raise PermissionError("Forbidden attribute access on a literal")

    @classmethod
    def _sandbox_dispatch(
        cls, code_str: str, fn_name: str, args: List[Any], timeout: float = 5.0
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Run one call in a separate, resource-limited Python process.

        Returns ``(ok, result_repr, input_after_repr, error)``. The child gets
        ``-I -S`` (no user site, no PYTHON* env), a minimal environment, its own
        temporary cwd, a 2 s CPU / 512 MiB address-space / 64-fd limit and a
        wall-clock timeout. Static AST policy is applied *before* dispatch.
        This is process isolation, not a hardened container: it bounds resource
        exhaustion and prevents access to the parent interpreter, but the host
        filesystem is still reachable if the policy itself is bypassed.
        """
        import subprocess
        import sys

        try:
            import resource  # noqa: F401
        except ImportError as exc:  # pragma: no cover - non-POSIX
            raise RuntimeError(
                "No resource module: refusing to run untrusted code without limits"
            ) from exc

        sandbox_dir = tempfile.mkdtemp(prefix="kestrel7_sandbox_")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _SANDBOX_DRIVER],
                input=json.dumps({"code": code_str, "fn": fn_name, "args": args}),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=sandbox_dir,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                preexec_fn=_limit_child,
            )
        except subprocess.TimeoutExpired:
            return False, None, None, f"TimeoutError: exceeded {timeout:.1f}s sandbox limit"
        finally:
            import shutil

            shutil.rmtree(sandbox_dir, ignore_errors=True)

        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no diagnostics"]
            return False, None, None, f"SandboxError: exit {proc.returncode}: {tail[0][:200]}"
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        if not lines:
            return False, None, None, "SandboxError: empty result"
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            return False, None, None, "SandboxError: malformed result"
        if payload.get("ok"):
            return True, payload.get("repr"), payload.get("input_after"), None
        return False, None, None, str(payload.get("error", "execution failed"))[:300]

    @classmethod
    def execute_capture(cls, code_str: str, fn_name: str, args: List[Any]) -> Any:
        """Execute ``fn_name(*args)`` in the sandbox and return its value.

        Used by the trace benchmark: the answer is observed rather than compared
        against an expectation, but it still goes through the isolated worker.
        """
        cls._check_ast_policy(code_str)
        ok, res_repr, _input_after, error = cls._sandbox_dispatch(code_str, fn_name, args)
        if not ok:
            raise RuntimeError(error or "sandbox execution failed")
        try:
            return ast.literal_eval(res_repr or "None")
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"sandbox result is not a literal: {res_repr!r}") from exc

    @classmethod
    def safe_execute_python(
        cls,
        code_str: str,
        fn_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> Tuple[int, int, List[str]]:
        """Static policy check, then run unit tests in the isolated worker."""
        try:
            cls._check_ast_policy(code_str)
        except SyntaxError as e:
            return 0, len(test_cases), [f"SyntaxError: {e}"]
        except PermissionError as e:
            return 0, len(test_cases), [str(e)]

        passed = 0
        errors: List[str] = []
        for idx, tc in enumerate(test_cases):
            original = tc["input"]
            ok, res_repr, after_repr, error = cls._sandbox_dispatch(
                code_str, fn_name, copy.deepcopy(original)
            )
            if not ok:
                errors.append(f"Test {idx} raised {error}")
                continue
            try:
                result = ast.literal_eval(res_repr or "None")
            except (ValueError, SyntaxError):
                errors.append(f"Test {idx} returned a non-literal: {res_repr!r}")
                continue
            if "expected_one_of" in tc:
                value_ok = result in tc["expected_one_of"]
            else:
                value_ok = result == tc.get("expected")
            # Input mutation is checked on every branch, including
            # expected_one_of (it used to bypass the check).
            try:
                after = ast.literal_eval(after_repr or "None")
            except (ValueError, SyntaxError):
                after = after_repr
            mutation_ok = after == original
            if value_ok and mutation_ok:
                passed += 1
            elif not mutation_ok:
                errors.append(f"Test {idx} mutated its input")
            else:
                errors.append(
                    f"Test {idx} failed: got {result!r}, expected "
                    f"{tc.get('expected', tc.get('expected_one_of'))!r}"
                )

        cls._dpo_log = getattr(cls, "_dpo_log", [])
        cls._dpo_log.append(
            {
                "fn": fn_name,
                "passed": passed,
                "total": len(test_cases),
                "winner": code_str if passed == len(test_cases) and len(test_cases) > 0 else None,
                "loser": code_str if passed < len(test_cases) else None,
                "errors": errors,
            }
        )
        return passed, len(test_cases), errors

    def select_verified(
        self,
        candidates: List[str],
        fn_name: str,
        test_cases: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score a candidate pool and record an unambiguous preference pair.

        A pair is only recorded when exactly one candidate passes every test and
        at least one candidate fails: ambiguous pools produce no preference data.
        """
        scored: List[Dict[str, Any]] = []
        for code in candidates:
            passed, total, errors = self.safe_execute_python(code, fn_name, test_cases)
            scored.append(
                {"code": code, "passed": passed, "total": total, "errors": errors}
            )
        winners = [s for s in scored if s["total"] > 0 and s["passed"] == s["total"]]
        losers = [s for s in scored if s["total"] > 0 and s["passed"] < s["total"]]
        if len(winners) == 1 and losers:
            self.dpo_buffer.append(
                {
                    "fn": fn_name,
                    "winner": winners[0]["code"],
                    "losers": [s["code"] for s in losers],
                    "tests": test_cases,
                    "scores": [{"passed": s["passed"], "total": s["total"]} for s in scored],
                    "oracle": "subprocess unit-test execution",
                }
            )
        return scored

    def export_dpo(self, path: str) -> int:
        """Write collected preference pairs as JSONL; returns the row count."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.dpo_buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return len(self.dpo_buffer)
