"""Deduplicated splitting and true stream-packed batching for Kestrel-7.

Two P1 defects from Day 6 are fixed structurally, not patched:

*   **Leakage (P1-01).** ``deduplicate_and_split`` hashes the *stripped* text
    with SHA-256 and removes duplicates *before* any random split, then asserts
    that train/val/test share no document. Day 6 repeated its corpus 50x and
    split afterwards, so every validation document also appeared in training.
    Alternative rejected: splitting first and de-duplicating within each split,
    which still lets the same document straddle splits. Falsifier: any non-empty
    intersection between splits (asserted, not assumed).
*   **Cross-document leakage (P1-02).** ``StreamPackedDataset`` concatenates
    documents into one token stream and records a ``doc_id`` per position. The
    attention mask uses ``doc_ids[q] == doc_ids[k]``, so a packed sequence is
    block-diagonal by construction. Labels are stored **unshifted** (with BOS
    and PAD masked to -100) because ``Kestrel7.forward`` performs the single
    next-token shift. Alternative rejected: per-document padding (Day 6), which
    wastes tokens and still lets the EOS predict the next document's BOS.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokenizer import BOS, PAD, BPETokenizer


def deduplicate_and_split(
    texts: List[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], dict]:
    """De-duplicate by SHA-256 content hash, then split deterministically.

    Returns ``(train, val, test, manifest)``. The manifest records the raw and
    unique counts so a run can report exactly how much data was thrown away.
    """
    if not (0.0 < train_frac < 1.0) or not (0.0 <= val_frac < 1.0):
        raise ValueError("train_frac must be in (0,1) and val_frac in [0,1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave a non-empty test split")

    seen = set()
    unique_texts: List[str] = []
    for t in texts:
        h = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_texts.append(t)

    rng = random.Random(seed)
    order = list(range(len(unique_texts)))
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, int(n * train_frac))
    n_val = int(n * val_frac)
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)
    if n_train >= n:
        n_train = max(1, n - 1)

    train = [unique_texts[i] for i in order[:n_train]]
    val = [unique_texts[i] for i in order[n_train : n_train + n_val]]
    test = [unique_texts[i] for i in order[n_train + n_val :]]

    # Fail closed on leakage: this is the whole point of hashing before
    # splitting, and a bare assert disappears under ``python -O``.
    if set(train) & set(val):
        raise RuntimeError("Train/Val leakage detected")
    if set(train) & set(test):
        raise RuntimeError("Train/Test leakage detected")
    if set(val) & set(test):
        raise RuntimeError("Val/Test leakage detected")

    manifest = {
        "raw_count": len(texts),
        "unique_count": len(unique_texts),
        "dropped_duplicates": len(texts) - len(unique_texts),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "seed": seed,
        "dedup": "sha256(strip(text))",
        # Dataset identity: lets a checkpoint record *which* corpus produced it,
        # not just how many documents it had.
        "corpus_sha256": hashlib.sha256(
            "\n".join(sorted(text.strip() for text in unique_texts)).encode("utf-8")
        ).hexdigest(),
    }
    return train, val, test, manifest


class StreamPackedDataset(Dataset):
    """Pack documents into fixed-length streams with document IDs.

    Each item is ``(input_ids, labels, doc_ids)``, all length ``seq_len``:

    * ``input_ids`` – concatenated documents, each wrapped ``BOS ... EOS``.
    * ``labels``    – **unshifted** copy of ``input_ids``; the first token of
      every document and every padding position is ``-100``. ``Kestrel7.forward``
      applies the one and only shift (``logits[..., :-1]`` vs ``labels[..., 1:]``).
    * ``doc_ids``   – the document index for every real token, ``-1`` for PAD.
    """

    def __init__(self, tokenizer: BPETokenizer, texts: List[str], seq_len: int, seed: int = 42):
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.examples: List[Tuple[List[int], List[int], List[int]]] = []

        rng = random.Random(seed)
        order = list(range(len(texts)))
        rng.shuffle(order)

        pad_id = tokenizer.token_to_id[PAD]
        stream_ids: List[int] = []
        stream_lbls: List[int] = []
        stream_docs: List[int] = []

        for doc_idx, di in enumerate(order):
            ids = tokenizer.encode(texts[di], add_bos=True, add_eos=True)
            if len(ids) < 2:
                continue
            lbls = list(ids)
            lbls[0] = -100  # never predict a BOS from the previous document's EOS
            stream_ids.extend(ids)
            stream_lbls.extend(lbls)
            stream_docs.extend([doc_idx] * len(ids))

        pos = 0
        while pos < len(stream_ids):
            chunk_ids = stream_ids[pos : pos + seq_len]
            chunk_lbls = stream_lbls[pos : pos + seq_len]
            chunk_docs = stream_docs[pos : pos + seq_len]
            pos += seq_len
            if len(chunk_ids) < 2:
                break
            pad_len = seq_len - len(chunk_ids)
            if pad_len > 0:
                chunk_ids = chunk_ids + [pad_id] * pad_len
                chunk_lbls = chunk_lbls + [-100] * pad_len
                chunk_docs = chunk_docs + [-1] * pad_len
            self.examples.append((chunk_ids, chunk_lbls, chunk_docs))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, lbls, docs = self.examples[idx]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(lbls, dtype=torch.long),
            torch.tensor(docs, dtype=torch.long),
        )


def collate_stream_packed(batch):
    ids = torch.stack([b[0] for b in batch], dim=0)
    lbls = torch.stack([b[1] for b in batch], dim=0)
    docs = torch.stack([b[2] for b in batch], dim=0)
    return ids, lbls, docs
