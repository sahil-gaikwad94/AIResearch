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
"""

from __future__ import annotations

import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from .model import Kestrel7
from .tokenizer import BOS, EOS, BPETokenizer


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
) -> str:
    """Generate a continuation, optionally with VEXX Tier 0/1/2 and Tier 4 guards."""
    model.eval()

    # Tier 4 fast path: for research/factuality prompts the calibrated answer is
    # returned directly, because weights are not evidence.
    if vexx is not None:
        calibration = vexx.verify_epistemic_calibration(prompt)
        if calibration is not None:
            return prompt + "\n" + calibration["explanation"]

    ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=False)
    if len(ids) == 0:
        ids = [tokenizer.token_to_id[BOS]]
    # Prefill truncates to the context window. Track the *truncated* length:
    # using the original prompt length as seq_start would place RoPE positions
    # beyond the cached keys and corrupt every later decode step.
    if len(ids) > model.max_ctx:
        ids = ids[-model.max_ctx :]
    input_ids = torch.tensor([ids], dtype=torch.long)

    logits, cache = model.prefill(input_ids)
    seq_start = int(input_ids.shape[1])
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

    return prompt + tokenizer.decode(generated)
