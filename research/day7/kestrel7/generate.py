"""Autoregressive generation with VEXX 3.0 in the decode loop.

The deterministic contract is explicit: ``temperature is None`` or
``temperature <= 0`` means greedy argmax. Day 6 let ``None`` fall through to
multinomial sampling, which made "reproducible" generation non-reproducible
(P2-01). Alternative rejected: defaulting to temperature 1.0, which hides the
caller's intent. Falsifier: two greedy calls producing different text.

VEXX masking runs on the *logits* before selection, so a masked token can never
be sampled regardless of temperature. Inline ``<<expr=...>>`` inside a
`` thinking`` block is evaluated by the Tier-2 CAS and the result is spliced into
the stream.

Decoding modes (P0.8) are explicit and opt-in:

*   ``plain`` -- no Python syntax constraints; used for prose, research and
    general answers.
*   ``python`` -- syntax-aware code assistance; the lexer is a structural
    guard but does not hard-block callees absent from a prompt-local symbol
    table. Final validity is decided by ``ast.parse`` and execution/tests.
*   ``json``/structured -- schema-specific constraints only when requested.
*   ``think`` -- protocol-defined control policy from the CRANE phase logic.

The Tier-4 prompt-level answer bypass was removed (P0.6): a rule checker may
return a diagnostic flag, but it must not fabricate a completed answer on the
model's behalf. Every result carries provenance saying which component
produced it.
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import Kestrel7
from .tokenizer import BOS, EOS, BPETokenizer


# Decoding modes (P0.8). ``python`` is the only mode that applies the
# code-aware lexer; ``plain`` applies only the control-token policy.
PLAIN = "plain"
PYTHON = "python"
JSON = "json"
THINK = "think"
VALID_MODES = {PLAIN, PYTHON, JSON, THINK}


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest set whose cumulative probability >= top_p."""
    if top_p is None or top_p >= 1.0:
        return logits
    squeeze = logits.dim() == 1
    work = logits.unsqueeze(0) if squeeze else logits
    sorted_logits, sorted_indices = torch.sort(work, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    filtered = work.clone()
    filtered.scatter_(1, sorted_indices, torch.where(remove, float("-inf"), sorted_logits))
    return filtered.squeeze(0) if squeeze else filtered


def sample_next(
    logits: torch.Tensor,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Select one token id per batch row.

    ``temperature is None`` or ``<= 0`` is deterministic argmax.
    """
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    if top_p is not None and top_p < 1.0:
        scaled = top_p_filter(scaled, top_p)
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


def generate_v7(
    model: Kestrel7,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    vexx=None,
    add_bos: bool = True,
    seed: Optional[int] = None,
    mode: str = PLAIN,
    provenance: Optional[List[str]] = None,
    context_mode: str = "rolling_window",
) -> str:
    """Generate a continuation, optionally with VEXX Tier 0/1/2 masking.

    P0.6: there is no prompt-level answer bypass here. A rule checker may flag a
    prompt as needing a grounded workflow, but it must not emit a completed
    natural-language answer on the model's behalf -- that would make the
    system appear to answer without the model. The result's provenance list
    records which components actually contributed.

    P0.8: ``mode`` selects the code-aware lexer. Only ``python`` applies the
    syntax/symbol-table mask; ``plain`` applies only the control-token policy,
    so prose containing parentheses, dotted names or mathematical notation is
    never corrupted.

    P1.4.7: ``context_mode`` controls behavior when the prompt exceeds
    ``max_ctx``:
    * ``rolling_window`` (default): truncate the prompt to ``max_ctx`` and
      continue generation with the cached window; truncation is recorded.
    * ``strict``: return early with a structured ``context_limit_reached``
      result instead of silently truncating.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"unknown decoding mode {mode!r}; valid: {sorted(VALID_MODES)}")
    if context_mode not in {"rolling_window", "strict"}:
        raise ValueError(
            f"unknown context_mode {context_mode!r}; valid: rolling_window, strict"
        )
    model.eval()
    origins: List[str] = ["neural_generation"]

    ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    if len(ids) == 0:
        ids = [tokenizer.token_to_id[BOS]]
    # Truncate to max_ctx before prefill so mock models (and the real model's
    # truncation_info) agree on the effective context length.
    truncated = False
    if len(ids) > model.max_ctx:
        ids = ids[-model.max_ctx :]
        truncated = True
    input_ids = torch.tensor([ids], dtype=torch.long)

    # P1.4.7: prefill reports truncation explicitly; no silent long-context claim.
    logits, cache, truncation_info = model.prefill(input_ids)
    seq_start = int(input_ids.shape[1]) - truncation_info["tokens_dropped"]

    if context_mode == "strict" and truncation_info["truncated"]:
        provenance_note = (
            f"context_limit_reached: prompt truncated from "
            f"{seq_start + truncation_info['tokens_dropped']} to {seq_start} tokens"
        )
        if provenance is not None:
            provenance.extend(origins + ["context_limit_reached"])
        return prompt + "\n[CONTEXT_LIMIT_REACHED] " + provenance_note

    if truncation_info["truncated"]:
        origins.append("prompt_truncated")
    generated: List[int] = []
    eos_id = tokenizer.token_to_id.get(EOS)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    cur_text = prompt
    for step_idx in range(max_new_tokens):
        step_logits = logits[0] if logits.dim() == 2 else logits[0, -1]
        if vexx is not None:
            masked = vexx.apply_in_flight_mask(
                step_logits.detach().cpu().numpy(),
                cur_text,
                tokenizer.id_to_token,
                mode=mode,
                remaining_token_budget=max_new_tokens - step_idx,
            )
            step_logits = torch.tensor(
                masked, dtype=step_logits.dtype, device=step_logits.device
            )
            if not bool(torch.isfinite(step_logits).any()):
                raise RuntimeError(
                    "VEXX masked every token; refusing to emit an arbitrary argmax"
                )

        next_id = int(sample_next(step_logits.unsqueeze(0), temperature, top_p, generator).item())
        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)
        cur_text = prompt + tokenizer.decode(generated)

        # Tier 2 inline CAS splice while inside a CRANE think block: <<expr=
        if (
            vexx is not None
            and cur_text.endswith("=")
            and "<<" in cur_text
            and vexx.crane_phase(cur_text) == "UNCONSTRAINED_THINK"
        ):
            m_cas = re.search(r"<<([^<>\n]+)=$", cur_text)
            if m_cas:
                value = vexx.crane_cas_evaluate(m_cas.group(1).strip()) + ">>"
                if not value.startswith("ERROR"):
                    origins.append("deterministic_calculation")
                    # The '=' token is already part of ``generated`` but was not
                    # fed to the model. Feed it before the spliced value, or the
                    # cache silently skips a token and every later logit is wrong.
                    logits, cache = model.decode_step(
                        torch.tensor([[next_id]], dtype=torch.long), cache, seq_start
                    )
                    seq_start += 1
                    for cid in tokenizer.encode(value, add_bos=False, add_eos=False):
                        logits, cache = model.decode_step(
                            torch.tensor([[cid]], dtype=torch.long), cache, seq_start
                        )
                        seq_start += 1
                        generated.append(cid)
                    cur_text = prompt + tokenizer.decode(generated)
                    continue

        logits, cache = model.decode_step(
            torch.tensor([[next_id]], dtype=torch.long), cache, seq_start
        )
        seq_start += 1

    result = prompt + tokenizer.decode(generated)
    if truncation_info["truncated"]:
        origins.append("prompt_truncated")
    if provenance is not None:
        provenance.extend(origins)
    return result
