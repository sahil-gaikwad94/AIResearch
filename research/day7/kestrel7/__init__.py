"""kestrel7 — Kestrel-7 hybrid architecture plus the VEXX 3.0 verifier.

Public surface:

*   ``Kestrel7`` / ``build_model`` / ``count_params`` — the 3:1 hybrid LM.
*   ``BPETokenizer`` — versioned byte-level BPE.
*   ``deduplicate_and_split`` / ``StreamPackedDataset`` — leakage-free data.
*   ``Vexx3Controller`` — six-tier neuro-symbolic verification.
*   ``Kestrel7Trainer`` / ``save_checkpoint_v7`` / ``load_checkpoint_v7``.
*   ``generate_v7`` — decode with optional VEXX masking.

No claim here exceeds what the code does: a freshly built model reports
``model_status="randomly_initialized"`` and ``checkpoint_loaded=False``, and
checkpoint loading fails closed on a config or tokenizer hash mismatch.
"""

from .checkpoint import load_checkpoint_v7, save_checkpoint_v7
from .config import CONFIGS, VALID_CONFIG_KEYS, get_config
from .data import StreamPackedDataset, collate_stream_packed, deduplicate_and_split
from .generate import generate_v7, sample_next
from .model import (
    GatedCausalSelfAttention,
    GatedDeltaNetLayer,
    Kestrel7,
    MTPDraftHead,
    build_model,
    clone_hybrid_cache,
    count_params,
    gdn_state_bytes,
    kv_cache_bytes,
)
from .tokenizer import BPETokenizer
from .train import Kestrel7Trainer, LinearWarmupCosineDecay, compute_on_policy_reverse_kl_loss
from .vexx import Vexx3Controller

__all__ = [
    "BPETokenizer",
    "CONFIGS",
    "VALID_CONFIG_KEYS",
    "GatedCausalSelfAttention",
    "GatedDeltaNetLayer",
    "Kestrel7",
    "Kestrel7Trainer",
    "LinearWarmupCosineDecay",
    "MTPDraftHead",
    "StreamPackedDataset",
    "Vexx3Controller",
    "build_model",
    "clone_hybrid_cache",
    "collate_stream_packed",
    "compute_on_policy_reverse_kl_loss",
    "count_params",
    "deduplicate_and_split",
    "gdn_state_bytes",
    "generate_v7",
    "get_config",
    "kv_cache_bytes",
    "load_checkpoint_v7",
    "sample_next",
    "save_checkpoint_v7",
]
