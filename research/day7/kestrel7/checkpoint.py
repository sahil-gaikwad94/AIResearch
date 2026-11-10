"""Kestrel-7 checkpoints: hash-verified, fail-closed, resumable.

A checkpoint is only useful if it is the *same* model and tokenizer it claims to
be. Three checks enforce that:

*   ``version`` — a checkpoint from a different schema is rejected, not guessed at.
*   ``config_hash`` — SHA-256 of the sorted config JSON (also checked against the
    config embedded in the file, so a hand-edited config cannot ride along).
*   ``tokenizer_hash`` — SHA-256 of the complete token mapping, merges, specials
    and pre-tokenizer.

All are validated before any weights are loaded. A missing or unverifiable
tokenizer hash is an error, not a warning: "we could not check" and "it matches"
must not have the same outcome. RNG states are saved with the weights so a
resumed run reproduces its trajectory; ``model_status``/``checkpoint_loaded``
only change after every load step has succeeded.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from .model import Kestrel7, hash_config

VERSION = 7


def capture_rng_state() -> dict:
    """Snapshot Python, NumPy, CPU and (if present) CUDA RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a snapshot from :func:`capture_rng_state` (missing keys are ignored)."""
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_v7(
    model: Kestrel7,
    optimizer,
    scheduler,
    trainer_state: dict,
    path: str,
) -> None:
    """Persist model, optimizer, scheduler, trainer and RNG state plus hashes."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "version": VERSION,
        "config": model.config,
        "config_hash": hash_config(model.config),
        "tokenizer_hash": model.tokenizer_hash,
        "model_status": model.model_status,
        "trainer_state": trainer_state,
        "rng": capture_rng_state(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": {
            "step_num": getattr(scheduler, "step_num", 0),
            "base_lrs": getattr(scheduler, "base_lrs", None),
        }
        if scheduler is not None
        else None,
    }
    torch.save(payload, path)


def load_checkpoint_v7(
    path: str,
    model: Kestrel7,
    tokenizer=None,
    optimizer=None,
    scheduler=None,
    restore_rng: bool = True,
) -> dict:
    """Load a checkpoint, validating version, config and tokenizer first.

    Raises:
        ValueError: wrong version, missing required field, config mismatch,
            missing/unverifiable tokenizer hash, or tokenizer mismatch.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for req in ("version", "config", "config_hash", "tokenizer_hash", "model"):
        if req not in payload:
            raise ValueError(f"Checkpoint missing required field '{req}'")
    if int(payload["version"]) != VERSION:
        raise ValueError(
            f"Checkpoint version {payload['version']} != supported version {VERSION}"
        )

    # The embedded config must hash to its own claimed digest; otherwise the
    # model-hash comparison below could succeed against a corrupted file.
    if hash_config(payload["config"]) != payload["config_hash"]:
        raise ValueError("Checkpoint is internally inconsistent: config does not match config_hash")

    if payload["config_hash"] != model.config_hash():
        raise ValueError(
            "Checkpoint config_hash mismatch (fail-closed): "
            f"checkpoint={str(payload['config_hash'])[:12]}... "
            f"model={model.config_hash()[:12]}..."
        )

    ckpt_tok_hash = payload.get("tokenizer_hash")
    if not ckpt_tok_hash:
        raise ValueError(
            "Checkpoint has no tokenizer_hash (fail-closed): a checkpoint that cannot "
            "be tied to a tokenizer is not loadable"
        )
    if tokenizer is not None:
        if model.vocab != tokenizer.vocab_size:
            raise ValueError(
                f"Vocab mismatch (fail-closed): model={model.vocab} tokenizer={tokenizer.vocab_size}"
            )
        actual_tok_hash = tokenizer.tokenizer_hash()
    elif model.tokenizer_hash is not None:
        actual_tok_hash = model.tokenizer_hash
    else:
        raise ValueError(
            "No tokenizer or model.tokenizer_hash supplied (fail-closed): cannot verify "
            "the checkpoint's tokenizer contract"
        )
    if ckpt_tok_hash != actual_tok_hash:
        raise ValueError(
            "Checkpoint tokenizer_hash mismatch (fail-closed): "
            f"checkpoint={str(ckpt_tok_hash)[:12]}... tokenizer={actual_tok_hash[:12]}..."
        )

    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.step_num = payload["scheduler"].get("step_num", 0)
        if payload["scheduler"].get("base_lrs") is not None:
            scheduler.base_lrs = payload["scheduler"]["base_lrs"]
    if restore_rng:
        restore_rng_state(payload.get("rng") or {})

    # Truthful metadata only after every check and restoration succeeded.
    model.checkpoint_loaded = True
    model.model_status = "trained_checkpoint_loaded"
    return payload.get("trainer_state", {})
