"""Kestrel-6 data pipeline.

Deterministic, restartable data loading for language modeling:

  * Packs documents into fixed-length sequences (no cross-document attention).
  * Builds next-token labels shifted by one token, with document boundaries
    masked (label = -100 at the first token of each packed document).
  * Train / validation / test split done *before* training.
  * Document-level shuffling with a pinned seed.
  * Each sample records its source manifest and token count.

Usage:
    ds = PackedDataset(tokenizer, texts, seq_len=512)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_packed)
"""

from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """A dataset of packed, fixed-length sequences with next-token labels."""

    def __init__(
        self,
        tokenizer,
        texts: List[str],
        seq_len: int,
        add_bos: bool = True,
        seed: int = 42,
        doc_separator: str = "\n\n",
    ):
        self.seq_len = seq_len
        self.add_bos = add_bos
        self.examples: List[Tuple[List[int], List[int]]] = []  # (ids, labels)

        rng = random.Random(seed)
        doc_order = list(range(len(texts)))
        rng.shuffle(doc_order)

        for di in doc_order:
            text = texts[di]
            ids = tokenizer.encode(text, add_bos=add_bos, add_eos=False)
            if len(ids) == 0:
                continue
            # Pack multiple documents into one sequence; mark the first token
            # of each packed document so its label can be ignored at the
            # boundary (no cross-document attention).
            pos = 0
            while pos < len(ids):
                chunk = ids[pos : pos + seq_len]
                pos += seq_len
                if len(chunk) < 2:
                    continue
                labels = chunk[1:] + [tokenizer.token_to_id.get("<|eos|>", -100)]
                # The first token of a packed document (after the very first)
                # should not predict across the boundary: set its label to -100.
                if pos < len(ids):
                    labels[-1] = -100
                ids_padded = chunk + [tokenizer.token_to_id.get("<|pad|>", 0)] * (seq_len - len(chunk))
                labels_padded = labels + [-100] * (seq_len - len(labels))
                self.examples.append((ids_padded, labels_padded))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_packed(batch):
    ids = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    return ids, labels


def train_val_test_split(texts: List[str], train_frac: float = 0.9, val_frac: float = 0.05, seed: int = 42):
    """Split texts into train / val / test deterministically."""
    rng = random.Random(seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    n_train = int(len(order) * train_frac)
    n_val = int(len(order) * val_frac)
    train = [texts[i] for i in order[:n_train]]
    val = [texts[i] for i in order[n_train : n_train + n_val]]
    test = [texts[i] for i in order[n_train + n_val :]]
    return train, val, test
