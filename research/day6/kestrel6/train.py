"""Kestrel-6 training loop.

A small, honest training loop: next-token cross-entropy, AdamW, linear warmup
+ cosine decay, gradient clipping, gradient accumulation, validation at fixed
token intervals, and checkpoint save/resume.

Everything is deterministic given a seed. Checkpoints store the config hash so a
loaded checkpoint is always validated against the exact config that produced it.
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint, load_checkpoint
from .data import PackedDataset, collate_packed
from .model import Kestrel6, count_params


class LinearWarmupCosineDecay:
    """Linear warmup then cosine decay to a minimum LR multiplier."""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_multiplier=0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_multiplier = min_lr_multiplier
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self):
        self.step_num += 1
        for i, base_lr in enumerate(self.base_lrs):
            if self.step_num <= self.warmup_steps:
                mult = self.step_num / max(1, self.warmup_steps)
            else:
                progress = (self.step_num - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
                mult = self.min_lr_multiplier + 0.5 * (1.0 - self.min_lr_multiplier) * (
                    1.0 + math.cos(math.pi * progress)
                )
            self.optimizer.param_groups[i]["lr"] = base_lr * mult
        return self.optimizer.param_groups[0]["lr"]


class Trainer:
    def __init__(
        self,
        model: Kestrel6,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[LinearWarmupCosineDecay],
        device: torch.device,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        log_every: int = 10,
        val_every_tokens: int = 1_000_000,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
        run_name: str = "kestrel6",
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.log_every = log_every
        self.val_every_tokens = val_every_tokens
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.run_name = run_name

        self.step = 0
        self.token_count = 0
        self.last_val_tokens = 0
        self.best_val_loss = float("inf")
        self.history: list = []
        self.nan_count = 0
        self._accum_pending = 0

        total, trainable = count_params(model)
        print(f"[kestrel6] model: total={total:,} trainable={trainable:,}")
        print(f"[kestrel6] device={device} seed={seed} run={run_name}")

    def train_batch(self, input_ids: torch.Tensor, labels: torch.Tensor) -> float:
        self.model.train()
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        logits, loss = self.model(input_ids, labels=labels)
        if not torch.isfinite(loss):
            self.nan_count += 1
            raise FloatingPointError(f"Non-finite loss at step {self.step}")
        loss = loss / self.grad_accum_steps
        loss.backward()

        self.step += 1
        self._accum_pending += 1
        if self._accum_pending >= self.grad_accum_steps:
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self._accum_pending = 0

        n_tokens = int((labels != -100).sum().item())
        self.token_count += n_tokens
        return float(loss.item() * self.grad_accum_steps)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for input_ids, labels in self.val_loader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            _, loss = self.model(input_ids, labels=labels)
            n_tokens = int((labels != -100).sum().item())
            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens
        val_loss = total_loss / max(1, total_tokens)
        self.best_val_loss = min(self.best_val_loss, val_loss)
        return {"val_loss": val_loss, "val_ppl": math.exp(min(val_loss, 20.0))}

    def fit(
        self,
        epochs: int = 1,
        max_steps: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, float]:
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        start = time.time()
        for epoch in range(epochs):
            for input_ids, labels in self.train_loader:
                loss = self.train_batch(input_ids, labels)

                if self.step % self.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start
                    toks_per_s = self.token_count / max(elapsed, 1e-9)
                    print(
                        f"[step {self.step:6d}] loss={loss:.4f} lr={lr:.2e} "
                        f"tokens={self.token_count:,} tok/s={toks_per_s:,.0f} "
                        f"elapsed={elapsed:.1f}s"
                    )
                    self.history.append(
                        {"step": self.step, "loss": loss, "lr": lr, "tokens": self.token_count}
                    )

                if (
                    self.val_loader is not None
                    and self.token_count - self.last_val_tokens >= self.val_every_tokens
                ):
                    self.last_val_tokens = self.token_count
                    metrics = self.validate()
                    print(f"[val step {self.step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

                if max_steps is not None and self.step >= max_steps:
                    break
                if max_tokens is not None and self.token_count >= max_tokens:
                    break
            if max_steps is not None and self.step >= max_steps:
                break
            if max_tokens is not None and self.token_count >= max_tokens:
                break

        # P1-03: flush the final partial gradient-accumulation group instead of
        # silently dropping it. Gradients were scaled by 1/grad_accum_steps, so
        # rescale by grad_accum_steps / pending before the optimizer step.
        if self._accum_pending > 0:
            scale_correction = self.grad_accum_steps / float(self._accum_pending)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad *= scale_correction
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()
            self._accum_pending = 0

        # P1-03: always validate before saving, even when val_every_tokens is
        # larger than the corpus; otherwise the checkpoint records best=inf.
        final = {"step": self.step, "tokens": self.token_count, "best_val_loss": self.best_val_loss}
        if self.val_loader is not None:
            metrics = self.validate()
            final.update(metrics)
            final["best_val_loss"] = self.best_val_loss
            print(f"[val final step {self.step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_step{self.step}.pt")
            save_checkpoint(
                self.model,
                self.optimizer,
                self.scheduler,
                self.step,
                self.token_count,
                ckpt_path,
                best_val_loss=self.best_val_loss,
                history=self.history,
                seed=self.seed,
                run_name=self.run_name,
            )
            final["checkpoint_path"] = ckpt_path
            print(f"[kestrel6] checkpoint saved -> {ckpt_path}")
        return final
