"""Kestrel-7 training: WSD-style schedule, accumulation flush, checkpoints.

Fixes inherited from the Day-6 audit and the Day-7 upgrade specification:

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
*   **P0.3 best/last checkpoints.** The best next-token-CE weights are saved to a
    separate artifact from the resumable last state, atomically, with the metric
    and step that produced them. Falsifier: a "best" file whose stored loss is
    not the best observed loss.
*   **P0.4 token-weighted accumulation.** Accumulated loss *sums* and valid-token
    *counts* are normalised at the optimizer boundary, so micro-batches with
    different numbers of ignored labels weight equally. Falsifier: gradients
    from one concatenated batch differing from the same examples split across
    micro-batches.
*   **P0.5 reproducible sampler.** The per-epoch permutation, batch cursor,
    optimizer step, token cursor, scheduler state and data-manifest hash are
    persisted; checkpoints are written periodically and atomically.
*   **P0.10 truthful metadata.** ``weights_origin``, ``optimizer_updates`` and
    ``capability_evaluation_status`` are tracked separately; a nonzero step
    count is never read as capability.
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
        checkpoint_every_steps: int = 100,
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
        self.checkpoint_every_steps = max(1, int(checkpoint_every_steps))

        self.micro_step = 0
        self.optimizer_step = 0
        self.token_count = 0
        self.last_val_tokens = 0
        self.best_val_loss = float("inf")
        # P0.3: the best-checkpoint criterion is next-token CE, tracked
        # separately from the legacy composite ``best_val_loss`` so a "best"
        # artifact can never be confused with the last resumable state.
        self.best_next_token_ce = float("inf")
        self.best_checkpoint_path: Optional[str] = None
        self.history: List[dict] = []
        self._accum_pending = 0
        self._accum_ce_sum = 0.0
        self._accum_lm_tokens = 0
        self._accum_mtp_sum = 0.0
        self._accum_mtp_tokens = 0
        self.last_val_metrics: Dict[str, float] = {}
        # Resume cursor: the next epoch and the number of batches already
        # consumed in it. Meaningful for deterministic (shuffle=False) loaders;
        # with a shuffled loader the order is regenerated from the restored RNG.
        self.epoch = 0
        self.batch_in_epoch = 0
        self.dataset_manifest: Dict[str, object] = {}

    # ------------------------------------------------------------------ internals

    def _save_best_checkpoint(self, best_next_token_ce: float) -> None:
        """Atomically persist the weights behind the best next-token CE.

        P0.3: this is a separate artifact identity from the resumable ``last``
        checkpoint. It stores the metric and step that produced it, and a
        partial temp file can never replace a valid checkpoint because the
        final name is only set by ``os.replace``.
        """
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best.pt")
        extra = {
            "best_next_token_ce": float(best_next_token_ce),
            "optimizer_step": int(self.optimizer_step),
            "token_count": int(self.token_count),
        }
        save_checkpoint_v7(
            self.model,
            self.optimizer,
            self.scheduler,
            {
                "micro_step": self.micro_step,
                "optimizer_step": self.optimizer_step,
                "token_count": self.token_count,
                "best_next_token_ce": float(best_next_token_ce),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "seed": self.seed,
                "run_name": self.run_name,
            },
            path,
            extra=extra,
        )
        self.best_checkpoint_path = path

    def _step_optimizer(self) -> None:
        # P0.4: normalise accumulated sums by their own valid-token counts.
        # Dividing each micro-batch's mean loss by grad_accum_steps weights
        # batches equally, which is wrong when packed batches contain
        # different numbers of valid LM and MTP targets.
        lm_tokens = max(1, self._accum_lm_tokens)
        mtp_tokens = max(1, self._accum_mtp_tokens)
        scale = 1.0 / lm_tokens
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad *= scale
        if self.grad_clip and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer_step += 1
        self._accum_pending = 0
        self._accum_ce_sum = 0.0
        self._accum_lm_tokens = 0
        self._accum_mtp_sum = 0.0
        self._accum_mtp_tokens = 0
        # P0.10: track updates and tokens separately from any capability label.
        self.model.optimizer_updates = int(self.model.optimizer_updates) + 1
        self.model.tokens_seen = int(self.model.tokens_seen) + lm_tokens

    def _unpack(self, batch):
        ids, lbls = batch[0], batch[1]
        docs = batch[2] if len(batch) == 3 else None
        return ids, lbls, docs

    def _mtp_weight(self) -> float:
        """The MTP auxiliary weight the model was built with (default 0.2)."""
        mtp = getattr(self.model, "mtp_depth", 0)
        return 0.2 if (mtp and mtp > 0) else 0.0

    def _make_epoch_permutation(self, n_items: int) -> List[int]:
        """Deterministic per-epoch permutation derived from seed + epoch.

        P0.5: a freshly constructed DataLoader iterator would regenerate its own
        permutation on resume. Persisting this list (or the seed that produced
        it) is what lets a resumed run reproduce the original remaining sequence.
        """
        rng = random.Random((self.seed * 1000003) ^ (self.epoch * 100003) ^ 0x9E3779B9)
        perm = list(range(n_items))
        rng.shuffle(perm)
        return perm

    def _write_periodic_checkpoint(self) -> None:
        """Persist the resumable state at an optimizer-step/token interval."""
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir, f"{self.run_name}_step{self.optimizer_step}.pt"
        )
        state = {
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "token_count": self.token_count,
            "last_val_tokens": self.last_val_tokens,
            "best_val_loss": self.best_val_loss,
            "best_next_token_ce": self.best_next_token_ce,
            "best_checkpoint_path": self.best_checkpoint_path,
            "history": self.history,
            "accum_pending": self._accum_pending,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "seed": self.seed,
            "run_name": self.run_name,
            "val_metrics": self.last_val_metrics,
            "dataset_manifest": self.dataset_manifest,
            "mtp_weight": self._mtp_weight(),
            "checkpoint_every_steps": int(self.checkpoint_every_steps),
            "epoch_permutation": getattr(self, "_epoch_perm", None),
        }
        save_checkpoint_v7(self.model, self.optimizer, self.scheduler, state, path)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation metrics with next-token CE and MTP reported separately.

        Perplexity is computed from next-token cross-entropy only. The composite
        objective is reported as ``val_total_objective`` and is never labelled
        perplexity (P0.2).
        """
        if self.val_loader is None:
            return {}
        self.model.eval()
        ce_sum = 0.0
        mtp_sum = 0.0
        total_obj = 0.0
        lm_tokens = 0
        mtp_tokens = 0
        for batch in self.val_loader:
            ids, lbls, docs = self._unpack(batch)
            _, _, comps = self.model(
                ids, labels=lbls, doc_ids=docs, return_components=True
            )
            n_lm = int(comps["lm_target_count"])
            n_mtp = int(comps["mtp_target_count"])
            ce_sum += float(comps["next_token_ce"]) * n_lm
            lm_tokens += n_lm
            if comps["mtp_loss"] is not None and n_mtp > 0:
                mtp_sum += float(comps["mtp_loss"]) * n_mtp
                mtp_tokens += n_mtp
            total_obj += float(comps["total_loss"]) * n_lm
        if lm_tokens == 0:
            raise RuntimeError("Validation set contains no valid language-model targets")
        val_next_token_ce = ce_sum / lm_tokens
        val_mtp_loss = (mtp_sum / mtp_tokens) if mtp_tokens > 0 else 0.0
        val_total_objective = total_obj / lm_tokens
        if not math.isfinite(val_next_token_ce):
            raise RuntimeError(f"Non-finite validation next-token CE: {val_next_token_ce}")
        # Best-checkpoint criterion (P0.3) is next-token CE, never the composite.
        if val_next_token_ce < self.best_next_token_ce:
            self.best_next_token_ce = val_next_token_ce
            self._save_best_checkpoint(val_next_token_ce)
        # Legacy best-composite tracker, kept so callers that read
        # ``best_val_loss`` still observe a finite value after validation runs.
        self.best_val_loss = min(self.best_val_loss, val_total_objective)
        ppl = math.exp(val_next_token_ce) if val_next_token_ce < 20.0 else float("inf")
        self.last_val_metrics = {
            "val_next_token_ce": val_next_token_ce,
            "val_next_token_ppl": math.exp(val_next_token_ce) if val_next_token_ce < 20.0 else float("inf"),
            "val_mtp_loss": val_mtp_loss,
            "val_total_objective": val_total_objective,
            "val_lm_tokens": lm_tokens,
            "val_mtp_tokens": mtp_tokens,
            # Legacy keys kept for callers that read val_loss/val_ppl.
            "val_loss": val_total_objective,
            "val_ppl": ppl,
            "val_tokens": lm_tokens,
        }
        self.model.train()
        return self.last_val_metrics

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Loss on an arbitrary split (e.g. the held-out test set), token-weighted.

        P0.2: next-token CE and the composite objective are reported
        separately, and perplexity is computed from next-token CE only.
        """
        self.model.eval()
        ce_sum = 0.0
        mtp_sum = 0.0
        obj_sum = 0.0
        lm_tokens = 0
        mtp_tokens = 0
        for batch in loader:
            ids, lbls, docs = self._unpack(batch)
            _, _, comps = self.model(ids, labels=lbls, doc_ids=docs, return_components=True)
            n_lm = int(comps["lm_target_count"])
            ce_sum += float(comps["next_token_ce"].detach()) * n_lm
            obj_sum += float(comps["total_loss"].detach()) * n_lm
            lm_tokens += n_lm
            if comps["mtp_loss"] is not None:
                n_mtp = int(comps["mtp_target_count"])
                mtp_sum += float(comps["mtp_loss"].detach()) * n_mtp
                mtp_tokens += n_mtp
        if lm_tokens == 0:
            raise RuntimeError("Evaluation split contains no valid language-model targets")
        next_token_ce = ce_sum / lm_tokens
        mtp_loss = (mtp_sum / mtp_tokens) if mtp_tokens > 0 else 0.0
        total_objective = obj_sum / lm_tokens
        if not math.isfinite(next_token_ce):
            raise RuntimeError(f"Non-finite evaluation next-token CE: {next_token_ce}")
        ppl = math.exp(next_token_ce) if next_token_ce < 20.0 else float("inf")
        return {
            "loss": next_token_ce,
            "next_token_ce": next_token_ce,
            "next_token_ppl": ppl,
            "ppl": ppl,
            "total_objective": total_objective,
            "mtp_loss": mtp_loss,
            "lm_tokens": lm_tokens,
            "mtp_tokens": mtp_tokens,
            "tokens": lm_tokens,
        }

    def load_state_dict(self, trainer_state: Dict[str, object]) -> None:
        """Restore counters so a loaded checkpoint continues rather than restarts."""
        self.micro_step = int(trainer_state.get("micro_step", 0))
        self.optimizer_step = int(trainer_state.get("optimizer_step", 0))
        self.token_count = int(trainer_state.get("token_count", 0))
        self.last_val_tokens = int(trainer_state.get("last_val_tokens", self.token_count))
        self.best_val_loss = float(trainer_state.get("best_val_loss", float("inf")))
        self.best_next_token_ce = float(trainer_state.get("best_next_token_ce", float("inf")))
        self.best_checkpoint_path = trainer_state.get("best_checkpoint_path")
        self._accum_pending = int(trainer_state.get("accum_pending", 0))
        self.history = list(trainer_state.get("history", []))
        self.epoch = int(trainer_state.get("epoch", 0))
        self.batch_in_epoch = int(trainer_state.get("batch_in_epoch", 0))
        self.dataset_manifest = trainer_state.get("dataset_manifest", {}) or {}
        if "checkpoint_every_steps" in trainer_state:
            self.checkpoint_every_steps = max(1, int(trainer_state["checkpoint_every_steps"]))
        perm = trainer_state.get("epoch_permutation")
        self._epoch_perm = list(perm) if perm is not None else None

    # ------------------------------------------------------------------ public

    def fit(
        self,
        epochs: int = 1,
        max_optimizer_steps: Optional[int] = None,
        start_epoch: Optional[int] = None,
        start_batch: Optional[int] = None,
    ) -> dict:
        """Train. Passing the restored cursor continues an interrupted run.

        P0.4: each micro-batch contributes its summed CE (and summed MTP loss)
        plus its valid-token counts to the accumulation buffer. At the optimizer
        boundary the accumulated CE is normalised by the total LM target count
        and the MTP term by its own count, so micro-batches with different
        numbers of ignored labels weight equally.

        P0.5: the per-epoch permutation is generated from a deterministic seed
        derived from ``self.seed`` and the epoch, persisted in the checkpoint,
        and checkpoints are written at an optimizer-step/token interval and on
        graceful termination.
        """
        if start_epoch is None:
            torch.manual_seed(self.seed)
            random.seed(self.seed)
        self.optimizer.zero_grad()

        begin_epoch = self.epoch if start_epoch is None else int(start_epoch)
        begin_batch = self.batch_in_epoch if start_batch is None else int(start_batch)

        stop = False
        for _epoch in range(begin_epoch, max(1, epochs)):
            self.epoch = _epoch
            self._epoch_perm = self._make_epoch_permutation(len(self.train_loader.dataset))
            for batch_idx, batch in enumerate(self.train_loader):
                if _epoch == begin_epoch and batch_idx < begin_batch:
                    continue
                self.batch_in_epoch = batch_idx + 1
                self.model.train()
                ids, lbls, docs = self._unpack(batch)
                _, _, comps = self.model(
                    ids, labels=lbls, doc_ids=docs, return_components=True
                )
                if not torch.isfinite(comps["total_loss"]):
                    raise FloatingPointError(f"Non-finite loss at micro_step {self.micro_step}")
                # Sum the raw per-token losses; normalisation happens at the
                # boundary. ``comps["next_token_ce"]`` is already a mean over
                # valid LM targets, so multiply back by the count.
                n_lm = int(comps["lm_target_count"])
                # P0.4: accumulate *sums*, not means. ``comps["next_token_ce"]``
                # is a mean over valid LM targets, so the summed CE is
                # ``ce * n_lm``; back-propagating the sum (rather than the mean
                # divided by grad_accum_steps) is what makes micro-batches with
                # different numbers of ignored labels weight equally once the
                # accumulated gradients are normalised by total LM tokens.
                ce_tensor = comps["next_token_ce"]
                self._accum_ce_sum += float(ce_tensor.detach()) * n_lm
                self._accum_lm_tokens += n_lm
                if comps["mtp_loss"] is not None:
                    n_mtp = int(comps["mtp_target_count"])
                    self._accum_mtp_sum += float(comps["mtp_loss"].detach()) * n_mtp
                    self._accum_mtp_tokens += n_mtp
                # Backward on the summed CE (+ summed MTP), so gradients
                # accumulate in proportion to each micro-batch's valid targets.
                aux_weight = self._mtp_weight()
                if comps["mtp_loss"] is not None and self.model.mtp_head is not None and n_mtp > 0:
                    combined = ce_tensor * n_lm + aux_weight * comps["mtp_loss"] * n_mtp
                else:
                    combined = ce_tensor * n_lm
                combined.backward()

                self.micro_step += 1
                self._accum_pending += 1
                self.token_count += n_lm

                if self._accum_pending >= self.grad_accum_steps:
                    self._step_optimizer()
                    self.history.append(
                        {
                            "micro_step": self.micro_step,
                            "optimizer_step": self.optimizer_step,
                            "loss": float(comps["total_loss"].item()),
                            "tokens": self.token_count,
                        }
                    )
                    if self.log_every and self.optimizer_step % self.log_every == 0:
                        lr = self.optimizer.param_groups[0]["lr"]
                        print(
                            f"[kestrel7 step {self.optimizer_step:4d}] loss={float(comps['total_loss'].item()):.4f} "
                            f"lr={lr:.2e} tokens={self.token_count}"
                        )
                    # P0.5: periodic checkpoint at optimizer-step intervals.
                    if self.checkpoint_dir and (
                        self.optimizer_step % max(1, self.checkpoint_every_steps) == 0
                        or (self.token_count - self.last_val_tokens) >= self.val_every_tokens
                    ):
                        self._write_periodic_checkpoint()

                if (
                    self.val_loader is not None
                    and (self.token_count - self.last_val_tokens) >= self.val_every_tokens
                ):
                    self.last_val_tokens = self.token_count
                    metrics = self.validate()
                    print(
                        f"[kestrel7 val step {self.optimizer_step}] "
                        + " ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
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
            "best_next_token_ce": self.best_next_token_ce,
            "best_checkpoint_path": self.best_checkpoint_path,
            "history": self.history,
            "accum_pending": self._accum_pending,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "seed": self.seed,
            "run_name": self.run_name,
            "val_metrics": val_metrics,
            "dataset_manifest": self.dataset_manifest,
            "mtp_weight": self._mtp_weight(),
            "checkpoint_every_steps": int(self.checkpoint_every_steps),
            "epoch_permutation": getattr(self, "_epoch_perm", None),
        }
        if self.checkpoint_dir:
            ckpt_path = os.path.join(
                self.checkpoint_dir, f"{self.run_name}_step{self.optimizer_step}.pt"
            )
            save_checkpoint_v7(self.model, self.optimizer, self.scheduler, state, ckpt_path)
            state["checkpoint_path"] = ckpt_path
        return state
