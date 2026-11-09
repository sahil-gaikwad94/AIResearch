"""Kestrel-6 checkpoint save/load with config-hash validation.

A checkpoint stores:
  * model state_dict
  * optimizer state_dict
  * scheduler state (lr, step_num)
  * step, token_count, best_val_loss, history
  * config dict + config_hash (validated on load)
  * VERSION, source_commit (if available), seed, run_name

Loading fails closed: if the config hash doesn't match the current model's
config, or the checkpoint is missing required fields, loading raises.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import torch

VERSION = 6


def _try_get_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    token_count: int,
    path: str,
    best_val_loss: Optional[float] = None,
    history: Optional[list] = None,
    seed: Optional[int] = None,
    run_name: str = "kestrel6",
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "version": VERSION,
        "config": model.config,
        "config_hash": model.config_hash(),
        "source_commit": _try_get_commit(),
        "step": step,
        "token_count": token_count,
        "best_val_loss": best_val_loss if best_val_loss is not None else float("inf"),
        "history": history or [],
        "seed": seed,
        "run_name": run_name,
        "model_status": getattr(model, "model_status", "unknown"),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {
            "step_num": getattr(scheduler, "step_num", 0),
            "base_lrs": getattr(scheduler, "base_lrs", None),
        } if scheduler is not None else None,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, strict: bool = True):
    """Load a checkpoint, validating config hash first. Fails closed on mismatch."""
    payload = torch.load(path, map_location="cpu", weights_only=False)

    required = ["version", "config", "config_hash", "model"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Checkpoint missing required field: {key}")

    # Validate config hash against the model's current config.
    if payload["config_hash"] != model.config_hash():
        raise ValueError(
            f"Checkpoint config hash mismatch: "
            f"checkpoint={payload['config_hash'][:12]}... "
            f"model={model.config_hash()[:12]}... "
            f"Refusing to load (fail-closed)."
        )

    model.load_state_dict(payload["model"], strict=strict)
    model.checkpoint_loaded = True
    model.model_status = "checkpoint_loaded"

    if optimizer is not None and "optimizer" in payload and payload["optimizer"] is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        try:
            scheduler.step_num = payload["scheduler"].get("step_num", 0)
            if payload["scheduler"].get("base_lrs") is not None:
                scheduler.base_lrs = payload["scheduler"]["base_lrs"]
        except Exception:
            pass

    info = {
        "version": payload.get("version"),
        "step": payload.get("step"),
        "token_count": payload.get("token_count"),
        "best_val_loss": payload.get("best_val_loss"),
        "config_hash": payload.get("config_hash"),
        "source_commit": payload.get("source_commit"),
        "seed": payload.get("seed"),
        "run_name": payload.get("run_name"),
    }
    return info
