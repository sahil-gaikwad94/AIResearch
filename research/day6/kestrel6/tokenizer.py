"""Byte-level BPE tokenizer for Kestrel-6.

A small, dependency-free (stdlib only) byte-pair-encoding tokenizer.

Vocabulary layout:
  * special tokens (<|bos|>, <|eos|>, <|pad|>, <|unk|>) first
  * 256 raw byte tokens (chr(0)..chr(255))
  * merge tokens, one per learned BPE merge (the concatenation of two byte
    sequences, represented as a Python string of latin-1 characters)

encode/decode round-trip is exact for any input text: text -> utf-8 bytes ->
byte tokens -> BPE merges -> ids; decode joins token strings, encodes latin-1,
decodes utf-8.

Save/load is JSON-based so the tokenizer is fully versioned and reproducible.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

BOS = "<|bos|>"
EOS = "<|eos|>"
PAD = "<|pad|>"
UNK = "<|unk|>"

SPECIAL_TOKENS = [BOS, EOS, PAD, UNK]

# GPT-2-style pre-tokenization regex.
_PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+""")


class BPETokenizer:
    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        # merges as (left_token_str, right_token_str) in rank order.
        self.merges: List[Tuple[str, str]] = []
        # (left_id, right_id) -> rank
        self.merge_rank: Dict[Tuple[int, int], int] = {}

        for st in self.special_tokens:
            self._add_token(st)

        if vocab is not None:
            for token, tid in vocab.items():
                self.token_to_id[token] = tid
                self.id_to_token[tid] = token
        if merges is not None:
            self.merges = list(merges)
            for rank, (a, b) in enumerate(self.merges):
                la = self.token_to_id.get(a)
                rb = self.token_to_id.get(b)
                if la is not None and rb is not None:
                    self.merge_rank[(la, rb)] = rank

        self._ensure_byte_tokens()
        self.vocab_size = len(self.token_to_id)

    def _add_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        tid = len(self.token_to_id)
        self.token_to_id[token] = tid
        self.id_to_token[tid] = token
        return tid

    def _ensure_byte_tokens(self) -> None:
        for i in range(256):
            token = chr(i)
            if token not in self.token_to_id:
                self._add_token(token)

    # ------------------------------------------------------------------ training

    @classmethod
    def train(
        cls,
        texts: List[str],
        vocab_size: int,
        special_tokens: Optional[List[str]] = None,
        min_pair_freq: int = 2,
    ) -> "BPETokenizer":
        tok = cls(special_tokens=special_tokens)

        # Pre-tokenize each text into byte-id sequences (one id per byte).
        words: List[Tuple[int, ...]] = []
        for text in texts:
            for pre in _PAT.findall(text):
                words.append(tuple(tok.token_to_id[chr(b)] for b in pre.encode("utf-8")))

        pair_counts: Counter = Counter()
        for w in words:
            for i in range(len(w) - 1):
                pair_counts[(w[i], w[i + 1])] += 1

        while len(tok.token_to_id) < vocab_size:
            candidates = [(c, p) for p, c in pair_counts.items() if c >= min_pair_freq]
            if not candidates:
                break
            # Most frequent pair wins; ties broken by (left_id, right_id) for determinism.
            count, pair = max(candidates, key=lambda cp: (cp[0], cp[1][0], cp[1][1]))
            a, b = pair
            left_tok = tok.id_to_token[a]
            right_tok = tok.id_to_token[b]
            merged_tok = left_tok + right_tok
            merged_id = tok._add_token(merged_tok)

            tok.merges.append((left_tok, right_tok))
            tok.merge_rank[(a, b)] = len(tok.merges) - 1

            # Rebuild all words applying this merge greedily, left to right.
            new_words: List[Tuple[int, ...]] = []
            for w in words:
                out: List[int] = []
                i = 0
                while i < len(w):
                    if i + 1 < len(w) and (w[i], w[i + 1]) == (a, b):
                        out.append(merged_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_words.append(tuple(out))
            words = new_words

            pair_counts = Counter()
            for w in words:
                for i in range(len(w) - 1):
                    pair_counts[(w[i], w[i + 1])] += 1

        tok.vocab_size = len(tok.token_to_id)
        return tok

    # ------------------------------------------------------------------ encode / decode

    def _encode_bytes(self, byte_ids: List[int]) -> List[int]:
        """Apply BPE merges (in rank order, greedily) to a sequence of byte ids."""
        seq = list(byte_ids)
        if len(seq) <= 1:
            return seq
        changed = True
        while changed and len(seq) > 1:
            changed = False
            best_rank = None
            best_i = -1
            for i in range(len(seq) - 1):
                key = (seq[i], seq[i + 1])
                if key in self.merge_rank:
                    r = self.merge_rank[key]
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_i = i
            if best_rank is not None:
                i = best_i
                merged_tok = self.id_to_token[seq[i]] + self.id_to_token[seq[i + 1]]
                merged_id = self.token_to_id[merged_tok]
                seq = seq[:i] + [merged_id] + seq[i + 2:]
                changed = True
        return seq

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id[BOS])
        unk_id = self.token_to_id[UNK]
        for pre in _PAT.findall(text):
            byte_ids = [self.token_to_id[chr(b)] for b in pre.encode("utf-8")]
            ids.extend(self._encode_bytes(byte_ids))
        if add_eos:
            ids.append(self.token_to_id[EOS])
        return ids

    def decode(self, ids: List[int]) -> str:
        chunks: List[str] = []
        for tid in ids:
            tok = self.id_to_token.get(tid, UNK)
            if tok in self.special_tokens:
                continue
            chunks.append(tok)
        raw = "".join(chunks)
        try:
            return raw.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            return raw

    # ------------------------------------------------------------------ save / load

    def save(self, path: str) -> None:
        payload = {
            "specials": self.special_tokens,
            "merges": [[a, b] for (a, b) in self.merges],
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
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
            special_tokens=payload.get("special_tokens", SPECIAL_TOKENS),
        )
        tok.vocab_size = payload.get("vocab_size", len(tok.token_to_id))
        return tok
