"""Versioned byte-level BPE tokenizer for Kestrel-7.

Design choices, and what each beat:

*   **Byte-level first, merges second.** Every one of the 256 bytes is a token
    before training starts, so arbitrary UTF-8/code input round-trips without an
    ``<|unk|>`` fallback. Alternative rejected: a word-level vocabulary with
    ``<|unk|>`` for OOV, which silently destroys identifiers the model needs.
    Falsifier: any input whose decode does not reproduce the original text.
*   **Word-frequency-grouped merge counting.** Pre-tokens are counted once and
    merged as ``(word_tuple, frequency)`` groups, so a corpus containing one word
    a million times costs one scan per merge, not a million. Alternative
    rejected: re-scanning the raw corpus every merge (the Day-6 O(V*N) bug).
    Falsifier: training time growing linearly in corpus repetition count.
*   **Ordered, versioned hash.** ``tokenizer_hash`` hashes the specials, merges
    and vocab size. Checkpoints store it and refuse to load against a different
    tokenizer. Alternative rejected: trusting ``vocab_size`` alone, which cannot
    distinguish two 8k vocabularies with different merges. Falsifier: two
    tokenizers with equal size but different merges producing the same hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
UNK = "<|unk|>"
THINK_START = "think_start"
THINK_END = "think_end"

SPECIAL_TOKENS = [BOS, EOS, PAD, UNK, THINK_START, THINK_END]

# P0.7: explicit token categories. The old code masked every special token
# except EOS, which silenced the think markers the controller's phase logic
# depends on -- an internal protocol contradiction. These categories make
# the policy explicit and testable.
NEVER_EMIT = "never_emit"
LEGAL_STOP = "legal_stop"
PROTOCOL_MARKERS = "protocol_markers"

_TOKEN_CATEGORY = {
    BOS: NEVER_EMIT,
    PAD: NEVER_EMIT,
    UNK: NEVER_EMIT,
    EOS: LEGAL_STOP,
    THINK_START: PROTOCOL_MARKERS,
    THINK_END: PROTOCOL_MARKERS,
}


def token_category(token_str: str) -> Optional[str]:
    """Classify a token string for the generation protocol.

    Returns ``NEVER_EMIT``, ``LEGAL_STOP`` or ``PROTOCOL_MARKERS`` for special
    tokens, or ``None`` for ordinary text tokens.
    """
    return _TOKEN_CATEGORY.get(token_str)

# GPT-2-style pre-tokenizer: contractions, words, numbers, punctuation, spaces.
_PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+")


class BPETokenizer:
    """Byte-level BPE with explicit special tokens and a stable hash."""

    VERSION = 7

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_rank: Dict[Tuple[int, int], int] = {}

        for st in self.special_tokens:
            self._add_token(st)
        if vocab is not None:
            for token, tid in vocab.items():
                self.token_to_id[token] = int(tid)
                self.id_to_token[int(tid)] = token
        self._ensure_byte_tokens()
        if merges is not None:
            self.merges = [tuple(m) for m in merges]  # type: ignore[misc]
            for rank, (a, b) in enumerate(self.merges):
                la = self.token_to_id.get(a)
                rb = self.token_to_id.get(b)
                if la is not None and rb is not None:
                    self.merge_rank[(la, rb)] = rank
        self.vocab_size = len(self.token_to_id)

    # ------------------------------------------------------------------ vocab

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        tid = len(self.token_to_id)
        self.token_to_id[token] = tid
        self.id_to_token[tid] = token
        return tid

    def _ensure_byte_tokens(self) -> None:
        # Latin-1 code points 0..255 map one-to-one onto the 256 bytes.
        for i in range(256):
            ch = chr(i)
            if ch not in self.token_to_id:
                self._add_token(ch)

    def tokenizer_hash(self) -> str:
        """Canonical hash of the *complete* tokenizer identity.

        Includes the sorted ``(id, token)`` mapping, the merge order, the special
        spellings and the pre-tokenization pattern. Hashing only size + merges
        let two tokenizers with different integer assignments (a hand-edited
        ``token_to_id``) share a hash, so a checkpoint would accept text that
        decodes to different tokens under the same ids.
        """
        payload = {
            "version": self.VERSION,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
            "vocab": [[int(tid), tok] for tok, tid in sorted(
                self.token_to_id.items(), key=lambda kv: kv[1]
            )],
            "vocab_size": self.vocab_size,
            "pretokenizer": _PAT.pattern,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------ train

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        """Learn merges from ``texts`` until ``vocab_size`` is reached.

        Only the train split should be passed here; the caller owns the
        train/val/test split so the tokenizer never sees held-out text.
        """
        tok = cls(special_tokens=special_tokens)
        word_freqs: Counter = Counter()
        for text in texts:
            for pre in _PAT.findall(text):
                w = tuple(tok.token_to_id[chr(b)] for b in pre.encode("utf-8"))
                word_freqs[w] += 1

        while len(tok.token_to_id) < vocab_size:
            pair_counts: Counter = Counter()
            for w, freq in word_freqs.items():
                for i in range(len(w) - 1):
                    pair_counts[(w[i], w[i + 1])] += freq
            candidates = [(c, p) for p, c in pair_counts.items() if c >= min_pair_freq]
            if not candidates:
                break
            # A merge whose surface form equals a special token would make that
            # text encode to the special id, and ``decode`` strips specials, so
            # the round-trip would silently lose characters. Skip such pairs.
            specials = set(tok.special_tokens)
            safe = [cp for cp in candidates if tok.id_to_token[cp[1][0]] + tok.id_to_token[cp[1][1]] not in specials]
            if not safe:
                break
            # Deterministic tie-break: highest count, then lowest pair ids.
            _, (a, b) = max(safe, key=lambda cp: (cp[0], -cp[1][0], -cp[1][1]))
            left_tok, right_tok = tok.id_to_token[a], tok.id_to_token[b]
            merged_id = tok._add_token(left_tok + right_tok)
            tok.merges.append((left_tok, right_tok))
            tok.merge_rank[(a, b)] = len(tok.merges) - 1

            new_word_freqs: Counter = Counter()
            for w, freq in word_freqs.items():
                out: List[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                        out.append(merged_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_word_freqs[tuple(out)] += freq
            word_freqs = new_word_freqs

        tok.vocab_size = len(tok.token_to_id)
        return tok

    # ------------------------------------------------------------------ encode / decode

    def _encode_bytes(self, byte_ids: List[int]) -> List[int]:
        seq = list(byte_ids)
        while len(seq) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(seq) - 1):
                r = self.merge_rank.get((seq[i], seq[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_rank is None:
                break
            merged_tok = self.id_to_token[seq[best_i]] + self.id_to_token[seq[best_i + 1]]
            seq = seq[:best_i] + [self.token_to_id[merged_tok]] + seq[best_i + 2 :]
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id[BOS])
        for pre in _PAT.findall(text):
            byte_ids = [self.token_to_id[chr(b)] for b in pre.encode("utf-8")]
            ids.extend(self._encode_bytes(byte_ids))
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, ids: List[int], preserve_protocol: bool = False) -> str:
        unknown = [int(tid) for tid in ids if int(tid) not in self.id_to_token]
        if unknown:
            raise ValueError(
                f"Cannot decode unknown token ids {unknown[:5]}; "
                "the vocabulary and checkpoint do not match"
            )
        chunks = [self.id_to_token[int(tid)] for tid in ids]
        if preserve_protocol:
            # Keep protocol markers (think_start/think_end) so the controller
            # can track the CRANE phase from generated token IDs.
            chunks = [c for c in chunks if c not in self.special_tokens or c in (THINK_START, THINK_END)]
        else:
            chunks = [c for c in chunks if c not in self.special_tokens]
        raw = "".join(chunks)
        try:
            return raw.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            return raw

    def validate(self) -> None:
        """P1.4.8: explicitly validate every tokenizer invariant.

        Raises ValueError if any invariant is violated (fail-closed).
        """
        # Token IDs must be unique, dense and contiguous from zero.
        ids = list(self.id_to_token.keys())
        if ids != list(range(len(ids))):
            raise ValueError(f"token IDs are not dense/contiguous from zero: {ids[:5]}...")
        # vocab_size must equal number of IDs and max_id + 1.
        if self.vocab_size != len(self.token_to_id):
            raise ValueError(
                f"vocab_size {self.vocab_size} != len(token_to_id) {len(self.token_to_id)}"
            )
        if self.vocab_size != max(ids) + 1 if ids else self.vocab_size != 0:
            raise ValueError(
                f"vocab_size {self.vocab_size} != max_id + 1 {max(ids) + 1 if ids else 0}"
            )
        # Every merge operand and result must exist in the declared vocabulary.
        for a, b in self.merges:
            if a not in self.token_to_id:
                raise ValueError(f"merge operand {a!r} not in vocabulary")
            if b not in self.token_to_id:
                raise ValueError(f"merge operand {b!r} not in vocabulary")
            merged = a + b
            if merged not in self.token_to_id:
                raise ValueError(f"merge result {merged!r} not in vocabulary")
        # All special token names and IDs must be unique.
        if len(self.special_tokens) != len(set(self.special_tokens)):
            raise ValueError("special token names are not unique")
        special_ids = [self.token_to_id[st] for st in self.special_tokens]
        if len(special_ids) != len(set(special_ids)):
            raise ValueError("special token IDs are not unique")
        # Merge order must be deterministic: merge_rank must match merges order.
        for rank, (a, b) in enumerate(self.merges):
            la = self.token_to_id.get(a)
            rb = self.token_to_id.get(b)
            if la is not None and rb is not None:
                if self.merge_rank.get((la, rb)) != rank:
                    raise ValueError(
                        f"merge_rank mismatch at rank {rank} for ({a!r}, {b!r})"
                    )

    # ------------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        payload = {
            "version": self.VERSION,
            "special_tokens": self.special_tokens,
            "merges": [[a, b] for a, b in self.merges],
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "tokenizer_hash": self.tokenizer_hash(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls(
            vocab=payload.get("token_to_id"),
            merges=[tuple(m) for m in payload.get("merges", [])],
            # Read both spellings: Day-6 ``save`` wrote "specials" while
            # ``load`` read only "special_tokens" (P2-01 compatibility).
            special_tokens=payload.get("special_tokens") or payload.get("specials") or SPECIAL_TOKENS,
        )
        tok.vocab_size = payload.get("vocab_size", len(tok.token_to_id))
        claimed = payload.get("tokenizer_hash")
        if claimed is not None and claimed != tok.tokenizer_hash():
            raise ValueError(
                "Tokenizer file hash mismatch (fail-closed): the stored merges/vocab "
                "do not reproduce the stored tokenizer_hash"
            )
        return tok
