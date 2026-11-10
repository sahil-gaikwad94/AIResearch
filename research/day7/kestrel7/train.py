"""Kestrel-7 training: WSD-style schedule, accumulation flush, reverse-KL distillation.

Fixes inherited from the Day-6 audit:

*   **Partial accumulation flush (P1-03).** A trailing group smaller than
    ``grad_accum_steps`` is rescaled (``grad_accum_steps / pending``) and stepped
    instead of being dropped. Beat: discarding the tail, which silently loses
    up to ``grad_accum_steps - 1`` batches per epoch. Falsifier: optimizer step
    count not increasing on a run shorter than one accumulation group.
*   **Forced final validation (P1-03).** Validation always runs before the
    checkpoint is written, so ``best_val_loss`` is never ``inf`` because
    ``val_every_tokens`` exceeded the corpus. Falsifier: a saved checkpoint whose
    ``best_val_loss`` is infinite while a validation loader exists.
*   **On-policy reverse KL (Thinking Machines Lab, 2025).** ``KL(pi_student ||
    pi_teacher)`` on student-visited states gives dense per-token supervision at
    O(N) bits per episode. Beat: sparse RLVR, which gives O(1) bits per episode
    at 10-30x the GPU cost. Implemented natively in torch so gradients reach the
    student logits (the reference's NumPy bridge detached them). Falsifier:
    KL(pi || pi) not exactly 0, or a non-zero student gradient missing.
"""

from __future__ import annotations

import math
import os
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint_v7
from .model import Kestrel7


class LinearWarmupCosineDecay:
    """Linear warmup then cosine decay to ``min_lr_multiplier`` of the base LR."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_multiplier: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_multiplier = float(min_lr_multiplier)
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> float:
        self.step_num += 1
        for i, base_lr in enumerate(self.base_lrs):
            if self.step_num <= self.warmup_steps:
                mult = self.step_num / max(1, self.warmup_steps)
            else:
                progress = (self.step_num - self.warmup_steps) / max(
                    1, self.total_steps - self.warmup_steps
                )
                mult = self.min_lr_multiplier + 0.5 * (1.0 - self.min_lr_multiplier) * (
                    1.0 + math.cos(math.pi * min(1.0, progress))
                )
            self.optimizer.param_groups[i]["lr"] = base_lr * mult
        return self.optimizer.param_groups[0]["lr"]


def compute_on_policy_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Per-token ``KL(pi_student || pi_teacher)`` on student-visited states.

    ``KL(s||t) = sum_v pi_s(v) * (log pi_s(v) - log pi_t(v))``, with both
    distributions softened by ``temperature`` and the result scaled by ``T^2``
    (Hinton et al.), so gradients do not grow as the temperature rises.

    The teacher is detached (it is a fixed target); the student is not. Unlike
    the reference's NumPy round-trip, ``log_softmax`` keeps the student graph
    intact, so the gradient is dense and non-zero on every supervised token.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    log_s = F.log_softmax(student_logits / temperature, dim=-1)
    log_t = F.log_softmax(teacher_logits.detach() / temperature, dim=-1)
    token_kl = (log_s.exp() * (log_s - log_t)).sum(dim=-1)  # [b, n]
    if mask is not None:
        valid = (mask != -100).to(token_kl.dtype)
        loss = (token_kl * valid).sum() / valid.sum().clamp(min=1.0)
    else:
        loss = token_kl.mean()
    return loss * (temperature ** 2)


class Kestrel7Trainer:
    """Minimal honest trainer: CE (+MTP), gradient accumulation, validation, checkpoints."""

    def __init__(
        self,
        model: Kestrel7,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        val_every_tokens: int = 100_000,
        checkpoint_dir: Optional[str] = None,
        seed: int = 42,
        run_name: str = "kestrel7",
        log_every: int = 10,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.val_every_tokens = val_every_tokens
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        self.run_name = run_name
        self.log_every = log_every

        self.micro_step = 0
        self.optimizer_step = 0
        self.token_count = 0
        self.last_val_tokens = 0
        self.best_val_loss = float("inf")
        self.history: List[dict] = []
        self._accum_pending = 0
        self.last_val_metrics: Dict[str, float] = {}
        # Resume cursor: the next epoch and the number of batches already
        # consumed in it. Meaningful for deterministic (shuffle=False) loaders;
        # with a shuffled loader the order is regenerated from the restored RNG.
        self.epoch = 0
        self.batch_in_epoch = 0
        self.dataset_manifest: Dict[str, object] = {}

    # ------------------------------------------------------------------ internals

    def _step_optimizer(self) -> None:
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer_step += 1
        self._accum_pending = 0
        self.model.model_status = "trained"

    def _unpack(self, batch):
        ids, lbls = batch[0], batch[1]
        docs = batch[2] if len(batch) == 3 else None
        return ids, lbls, docs

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.val_loader:
            ids, lbls, docs = self._unpack(batch)
            _, loss = self.model(ids, labels=lbls, doc_ids=docs)
            n_tok = int((lbls[:, 1:] != -100).sum().item())
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok
        if total_tokens == 0:
            raise RuntimeError("Validation set contains no valid language-model targets")
        val_loss = total_loss / total_tokens
        if not math.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss: {val_loss}")
        # Only a computed, finite metric may set the best-checkpoint value.
        self.best_val_loss = min(self.best_val_loss, val_loss)
        # A clamped perplexity would contradict its own unreported loss; above the
        # float range report inf explicitly instead of a finite-looking number.
        self.last_val_metrics = {
            "val_loss": val_loss,
            "val_ppl": math.exp(val_loss) if val_loss < 20.0 else float("inf"),
            "val_tokens": total_tokens,
        }
        self.model.train()
        return self.last_val_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Loss on an arbitrary split (e.g. the held-out test set), token-weighted."""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in loader:
            ids, lbls, docs = self._unpack(batch)
            _, loss = self.model(ids, labels=lbls, doc_ids=docs)
            n_tok = int((lbls[:, 1:] != -100).sum().item())
            total_loss += float(loss.item()) * n_tok
            total_tokens += n_tok
        if total_tokens == 0:
            raise RuntimeError("Evaluation split contains no valid language-model targets")
        value = total_loss / total_tokens
        if not math.isfinite(value):
            raise RuntimeError(f"Non-finite evaluation loss: {value}")
        return {
            "loss": value,
            "ppl": math.exp(value) if value < 20.0 else float("inf"),
            "tokens": total_tokens,
        }

    def load_state_dict(self, trainer_state: Dict[str, object]) -> None:
        """Restore counters so a loaded checkpoint continues rather than restarts."""
        self.micro_step = int(trainer_state.get("micro_step", 0))
        self.optimizer_step = int(trainer_state.get("optimizer_step", 0))
        self.token_count = int(trainer_state.get("token_count", 0))
        self.last_val_tokens = int(trainer_state.get("last_val_tokens", self.token_count))
        self.best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
        self._accum_pending = int(trainer_state.get("accum_pending", 0))
        self.history = list(trainer_state.get("history", []))
        self.epoch = int(trainer_state.get("epoch", 0))
        self.batch_in_epoch = int(trainer_state.get("batch_in_epoch", 0))

    # ------------------------------------------------------------------ public

    def fit(
        self,
        epochs: int = 1,
        max_optimizer_steps: Optional[int] = None,
        start_epoch: Optional[int] = None,
        start_batch: Optional[int] = None,
    ) -> dict:
        """Train. Passing the restored cursor continues an interrupted run."""
        if start_epoch is None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
        self.optimizer.zero_grad()

        begin_epoch = self.epoch if start_epoch is None else int(start_epoch)
        begin_batch = self.batch_in_epoch if start_batch is None else int(start_batch)

        stop = False
        for _epoch in range(begin_epoch, max(1, epochs)):
            self.epoch = _epoch
            for batch_idx, batch in enumerate(self.train_loader):
                if _epoch == begin_epoch and batch_idx < begin_batch:
                    continue
                self.batch_in_epoch = batch_idx + 1
                self.model.train()
                ids, lbls, docs = self._unpack(batch)
                _, loss = self.model(ids, labels=lbls, doc_ids=docs)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at micro_step {self.micro_step}")
                (loss / self.grad_accum_steps).backward()

                self.micro_step += 1
                self._accum_pending += 1
                self.token_count += int((lbls[:, 1:] != -100).sum().item())

                if self._accum_pending >= self.grad_accum_steps:
                    self._step_optimizer()
                    self.history.append(
                        {
                            "micro_step": self.micro_step,
                            "optimizer_step": self.optimizer_step,
                            "loss": float(loss.item()),
                            "tokens": self.token_count,
                        }
                    )
                    if self.log_every and self.optimizer_step % self.log_every == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(
                            f"[kestrel7 step {self.optimizer_step:4d}] loss={float(loss.item()):.4f} "
                            f"lr={lr:.2e} tokens={self.token_count}"
                        )

                if (
                    self.val_loader is not None
                    and (self.token_count - self.last_val_tokens) >= self.val_every_tokens
                ):
                    self.last_val_tokens = self.token_count
                    metrics = self.validate()
                    print(
                        f"[kestrel7 val step {self.optimizer_step}] "
                        + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    )

                if max_optimizer_steps is not None and self.optimizer_step >= max_optimizer_steps:
                    stop = True
                    break
            if stop:
                break

        # Flush the tail accumulation group with exact rescaling.
        if self._accum_pending > 0:
            scale_correction = self.grad_accum_steps / float(self._accum_pending)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad *= scale_correction
            self._step_optimizer()

        # Always validate before writing the checkpoint.
        val_metrics = self.validate() if self.val_loader is not None else {}

        state = {
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "token_count": self.token_count,
            "last_val_tokens": self.last_val_tokens,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "accum_pending": self._accum_pending,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "seed": self.seed,
            "run_name": self.run_name,
            "val_metrics": val_metrics,
            "dataset_manifest": self.dataset_manifest,
        }
        if self.checkpoint_dir:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"{self.run_name}_step{self.optimizer_step}.pt"
            )
            save_checkpoint_v7(self.model, self.optimizer, self.scheduler, state, ckpt_path)
            state["checkpoint_path"] = ckpt_path
        return state
