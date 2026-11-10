"""Config loading for Kestrel-6.

Loads ``config.json`` and exposes ``CONFIGS`` (name -> dict) plus a helper to
build a config dict for a named preset with defaults filled in.

Metadata keys (anything starting with ``_``, e.g. the root-level ``_comment``)
are documentation, not presets: they are filtered out at load time and again in
``get_config`` so a stale ``_comment`` can never crash preset lookup.
"""

from __future__ import annotations

import json
import os
from typing import Dict

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _RAW_CONFIGS: Dict[str, dict] = json.load(_f)

# Only real, dict-valued presets survive; every "_"-prefixed metadata key is
# stripped here (P0-02: a root-level "_comment" must never reach get_config).
CONFIGS: Dict[str, dict] = {
    k: v
    for k, v in _RAW_CONFIGS.items()
    if not k.startswith("_") and isinstance(v, dict)
}


def get_config(name: str) -> dict:
    """Return a copy of the named config preset with defaults filled in."""
    if name not in CONFIGS:
        raise KeyError(f"Unknown config preset '{name}'. Available: {sorted(CONFIGS)}")
    c = {k: v for k, v in CONFIGS[name].items() if not k.startswith("_")}
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
