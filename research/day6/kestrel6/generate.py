"""Kestrel-6 autoregressive generation.

Greedy, temperature, and top-p sampling with stop tokens. Built on top of the
model's prefill() / decode_step() KV-cache interface for streaming decode.

A deterministic greedy mode is provided (temperature=0) for reproducible
smoke tests and debugging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering in-place on a copy of logits."""
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above top_p.
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift right to keep the first token above the threshold.
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    filtered = logits.clone()
    filtered[indices_to_remove] = float("-inf")
    return filtered


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_p: Optional[float] = None, rng=None):
    """Sample the next token id from logits.

    temperature == 0 (or None) => deterministic argmax (greedy).
    """
    # P2-01: temperature None or <= 0 is the deterministic greedy contract.
    # The old code let temperature=None fall through to multinomial sampling.
    if temperature is None or temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature != 1.0:
        logits = logits / temperature
    if top_p is not None and top_p < 1.0:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    if rng is not None:
        return torch.multinomial(probs, num_samples=1, generator=rng)
    return torch.multinomial(probs, num_samples=1)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    stop_tokens=None,
    add_bos: bool = True,
    add_eos: bool = False,
    seed: Optional[int] = None,
):
    """Generate text autoregressively from a prompt.

    Returns the generated text (prompt + continuation, with stop tokens stripped).
    """
    device = next(model.parameters()).device
    rng = torch.Generator(device=device)
    if seed is not None:
        rng.manual_seed(seed)

    stop_ids = set()
    if stop_tokens is not None:
        for st in stop_tokens:
            tid = tokenizer.token_to_id.get(st)
            if tid is not None:
                stop_ids.add(tid)
    eos_id = tokenizer.token_to_id.get("<|eos|>")

    ids = tokenizer.encode(prompt, add_bos=add_bos, add_eos=add_eos)
    input_ids = torch.tensor([ids], device=device, dtype=torch.long)

    logits, cache = model.prefill(input_ids)
    seq_start = input_ids.shape[1]
    generated = []

    for _ in range(max_new_tokens):
        next_id = sample_next(logits[:, -1, :] if logits.dim() == 3 else logits, temperature, top_p, rng)
        next_id_val = int(next_id.item())
        generated.append(next_id_val)

        if next_id_val in stop_ids or (eos_id is not None and next_id_val == eos_id):
            break

        tok_tensor = torch.tensor([[next_id_val]], device=device, dtype=torch.long)
        logits, cache = model.decode_step(tok_tensor, cache, seq_start)
        seq_start += 1

    continuation = tokenizer.decode(generated)
    return prompt + continuation


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 64, stop_tokens=None, add_bos: bool = True):
    """Deterministic greedy (argmax) generation."""
    return generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, temperature=0.0, stop_tokens=stop_tokens, add_bos=add_bos)
