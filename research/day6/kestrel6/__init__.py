"""kestrel6 -- the Kestrel-6 trainable baseline.

Kestrel-6 is a small, honest, end-to-end trainable language model: a dense
decoder-only Transformer baseline with a real tokenizer, next-token loss,
AdamW optimizer, checkpoint save/load, and autoregressive generation. Day-5
mechanisms (local MLA, global NoPE landmark attention, WY fast-weight memory,
subspace MoE, recurrent workspace) are offered as *toggleable ablations* that
must each beat the dense baseline at equal parameter count, tokens, and compute
before being promoted.

Everything here is real, trainable code. There are no false checkpoint or
training flags. Every public number should be tagged measured / derived /
simulated / assumed / target.
"""

from .model import Kestrel6, build_model, count_params  # noqa: F401
from .tokenizer import BPETokenizer  # noqa: F401
from .config import CONFIGS  # noqa: F401

__all__ = ["Kestrel6", "build_model", "count_params", "BPETokenizer", "CONFIGS"]
