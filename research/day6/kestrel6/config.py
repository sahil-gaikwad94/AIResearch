"""Config loading for Kestrel-6.

Loads ``config.json`` and exposes ``CONFIGS`` (name -> dict) plus a helper to
build a config dict for a named preset with defaults filled in.
"""

from __future__ import annotations

import json
import os
from typing import Dict

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    CONFIGS: Dict[str, dict] = json.load(_f)


def get_config(name: str) -> dict:
    """Return a copy of the named config preset with defaults filled in."""
    if name not in CONFIGS:
        raise KeyError(f"Unknown config preset '{name}'. Available: {sorted(CONFIGS)}")
    c = dict(CONFIGS[name])
    c.setdefault("attn_impl", "dense")
    c.setdefault("sliding_window", None)
    c.setdefault("global_layers", [])
    c.setdefault("use_memory", False)
    c.setdefault("use_moe", False)
    c.setdefault("use_workspace", False)
    c.setdefault("norm_eps", 1e-5)
    c.setdefault("rope_theta", 500000.0)
    c.setdefault("tie_embeddings", True)
    return c
