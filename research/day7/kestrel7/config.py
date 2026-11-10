"""Config loading and schema validation for Kestrel-7.

Two invariants, both enforced here rather than in prose:

1.  Metadata keys (anything starting with ``_``, e.g. the root ``_comment``)
    are documentation, not presets. They are stripped when ``config.json`` is
    loaded and again inside :func:`get_config`, so a future comment can never
    become a lookup key or leak into a hash.
2.  A preset may only contain fields in ``VALID_CONFIG_KEYS``. Unknown fields
    are a hard error: an ignored typo (``sliding_widow``) silently changes the
    architecture and the config hash, which is how checkpoint contracts rot.

Every default is explicit. ``get_config`` returns a fresh dict, so callers
(e.g. ``run_kestrel7.py``) may mutate ``vocab`` to match a trained tokenizer
without corrupting the preset registry.
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

with open(_CONFIG_PATH, "r", encoding="utf-8") as _f:
    _RAW_CONFIGS: Dict[str, object] = json.load(_f)

#: name -> preset dict. Only real, dict-valued, non-metadata keys survive.
CONFIGS: Dict[str, dict] = {
    k: v
    for k, v in _RAW_CONFIGS.items()
    if not k.startswith("_") and isinstance(v, dict)
}

#: The complete public schema for a Kestrel-7 preset.
VALID_CONFIG_KEYS = {
    "vocab",
    "d",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "layers",
    "max_ctx",
    "rope_theta",
    "norm_eps",
    "tie_embeddings",
    "attn_impl",
    "hybrid_schedule",
    "sliding_window",
    "qk_norm",
    "attn_output_gate",
    "zero_centered_norm",
    "mtp_depth",
    "conv_kernel",
}

_DEFAULTS = {
    "attn_impl": "dense",
    "hybrid_schedule": None,
    "sliding_window": None,
    "norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "tie_embeddings": True,
    "qk_norm": True,
    "attn_output_gate": True,
    "zero_centered_norm": True,
    "mtp_depth": 0,
    "conv_kernel": 4,
}


def validate_config(c: dict, name: str = "config") -> dict:
    """Validate a config dict, filling optional defaults; returns the same dict.

    Used by :func:`get_config` *and* by ``Kestrel7.__init__``, so a hand-built
    dict cannot bypass the invariants (an unknown field or a ``hybrid_gdn``
    preset without a schedule is an error everywhere, not only in JSON).
    """
    unknown = set(c) - VALID_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config fields in '{name}': {sorted(unknown)}")
    for key, value in _DEFAULTS.items():
        c.setdefault(key, value)

    if (
        int(c["layers"]) < 1
        or int(c["d"]) < 1
        or int(c["vocab"]) < 1
        or int(c["n_heads"]) < 1
        or int(c["n_kv_heads"]) < 1
        or int(c["d_ff"]) < 1
        or int(c["max_ctx"]) < 1
        or int(c["conv_kernel"]) < 1
        or float(c["rope_theta"]) <= 0
        or float(c["norm_eps"]) <= 0
    ):
        raise ValueError(f"'{name}': dimensions, context, kernel and eps must be positive")
    if int(c["d"]) % int(c["n_heads"]) != 0:
        raise ValueError(f"'{name}': d must be divisible by n_heads")
    head_dim = int(c["d"]) // int(c["n_heads"])
    if head_dim % 2 != 0:
        raise ValueError(f"'{name}': head_dim must be even for RoPE, got {head_dim}")
    if int(c["n_heads"]) % int(c["n_kv_heads"]) != 0:
        raise ValueError(f"'{name}': n_heads must be a multiple of n_kv_heads")
    if int(c["mtp_depth"]) not in (0, 1):
        raise ValueError(f"'{name}': mtp_depth must be 0 or 1, got {c['mtp_depth']}")

    schedule = c.get("hybrid_schedule")
    if schedule is not None:
        if len(schedule) != int(c["layers"]):
            raise ValueError(
                f"'{name}': hybrid_schedule length {len(schedule)} != layers {c['layers']}"
            )
        bad = set(schedule) - {"gdn", "full"}
        if bad:
            raise ValueError(f"'{name}': unknown mixer types {sorted(bad)}")
    # Reject contradictory configs instead of silently building a dense stack:
    # ``attn_impl: hybrid_gdn`` with no schedule used to ignore every GDN layer.
    if c["attn_impl"] == "hybrid_gdn" and schedule is None:
        raise ValueError(f"'{name}': attn_impl='hybrid_gdn' requires a hybrid_schedule")
    if c["attn_impl"] == "dense" and schedule is not None:
        raise ValueError(f"'{name}': attn_impl='dense' cannot declare a hybrid_schedule")
    if c["attn_impl"] not in {"dense", "hybrid_gdn"}:
        raise ValueError(f"'{name}': unsupported attn_impl {c['attn_impl']!r}")

    window = c.get("sliding_window")
    if window is not None and not (0 < int(window) <= int(c["max_ctx"])):
        raise ValueError(
            f"'{name}': sliding_window must be in (0, max_ctx], got {window}"
        )
    return c


def get_config(name: str) -> dict:
    """Return a validated copy of a named preset with defaults filled in.

    Raises:
        KeyError: unknown preset name.
        ValueError: unknown fields or an inconsistent hybrid schedule.
    """
    if name not in CONFIGS:
        raise KeyError(f"Unknown Kestrel-7 config preset '{name}'. Valid presets: {sorted(CONFIGS)}")
    # Deep copy: a shallow copy shares the schedule list with the preset
    # registry, so one caller mutating it would silently change every later run.
    c = copy.deepcopy({k: v for k, v in CONFIGS[name].items() if not k.startswith("_")})
    return validate_config(c, name)
