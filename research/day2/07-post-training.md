# 7 — Post-Training

> Day 1 was right that behaviour comes from environments, and right about the reward-hacking catalogue. Day 2 adds two things the architecture makes possible: a reward that pays for *coverage* rather than passing, and a reward that teaches the model **when to think**.

---

## 7.1 What carries over unchanged

The SFT corpus shape (300 k rejection-sampled trajectories, 12:1 generate-to-keep), the tool grammar, diff output, the read-only test mount, the impossibility canaries, and the audit pipeline for test suites (mutation kill rate >70%, flaky removal, stub detection).

That material is in [day 1 §6](../day1/06-post-training.md) and repeating it would add length, not information. Three additions follow, each tied to a day-2 decision.

---

## 7.2 Addition 1: the event-log format must be trained, not prompted

The [04](04-context-protocol.md) protocol only works if the model treats a patch as superseding earlier content. A base model that has only seen re-rendered files will read `read(f.py)` followed by `patch(f.py, …)` and may patch the *stale* content, because nothing in its training taught it otherwise.

So 30% of SFT trajectories use the append-only format, drawn from the same replay corpus as mid-training. The supervision signal is subtle and it is worth naming:

| Behaviour | Taught by |
|---|---|
| Treat a patched region as current | trajectories where the correct patch depends on applying an earlier patch first |
| Distinguish "was true" from "is true" | debugging trajectories that reference the pre-patch version |
| Never re-read a file to confirm a patch | trajectories where re-reading wastes turns and is penalised (see §7.4) |

The third is a *token-efficiency* behaviour, and it is trained, not hoped for.

---

## 7.3 Addition 2: coverage-delta reward

The pass/fail signal has a well-known hole: a patch can pass the visible tests while leaving the actual logic wrong, or while touching lines no test exercises. Day 1's canaries catch *cheating*; they do not catch *hollow success*.

Kestrel-2 adds a second term computed from the same run:

```
reward = 1[all tests pass]
       + 0.15 · coverage_delta        # fraction of changed lines now covered
       − 0.05 · 1[tests modified]
       − λ · tool_calls
```

`coverage_delta` is measured by instrumenting the test run and intersecting with the diff's changed lines. A patch that satisfies the tests while adding unreachable code scores 1.0; the same patch with the new branches exercised scores 1.15.

Why this helps more than it looks:

- It rewards the *intended* shape of a fix (guard clause actually reached, error path actually hit) without specifying it.
- It makes "delete the assertion" strictly worse than "make the assertion pass", because deleting a test line reduces coverage of the changed region.
- It is measurable with tooling that already exists in every CI system.

**Hazard:** a policy can inflate coverage by adding tests that execute code without asserting anything. That is why the *added* tests are themselves mutation-scored (§7.5) — a test that kills no mutant contributes nothing.

---

## 7.4 Addition 3: the thinking budget is a learned policy

This is the reward that only exists because the architecture has a dial.

```
reward = success − λ_R · (R − 1) − λ_tok · thinking_tokens
```

with `λ_R = 0.02` and `λ_tok = 2 × 10⁻⁵`.

Calibration matters and is easy to get wrong. A 20-turn session at `R = 3` pays `20 × 2 × 0.02 = 0.8` against a success reward of 1.0 — too much; the model would learn to never loop. At `λ_R = 0.01` the same session pays 0.4, which is a real but survivable cost that a successful solve easily covers. **Set the penalty so that a typical successful trajectory loses 10–20% of its reward, not 80%.**

What this trains, and what it does not:

| Intended | Not intended |
|---|---|
| `R = 1` for formatting, renames, single-line fixes | `R = 1` for a failing test that needs diagnosis |
| `R = 4–6` for a stack trace with three candidate causes | `R = 8` as a brute-force search over patches |
| Exit as soon as the gate's confidence crosses threshold | Fixed `R` regardless of difficulty |

The failure mode to watch: the model learns that hard tasks are unprofitable and *avoids* them — declining, or choosing the easy first hypothesis. Detection is a widening gap between pass rate on easy and hard slices; response is to normalise the reward by task difficulty (divide by the slice's mean success rate) so that hard solves pay proportionally more.

---

## 7.5 The verifier, extended

Day 1's three-pass audit (mutation, flaky, stub) is necessary and no longer sufficient, because two new attack surfaces exist:

| Attack | New in day 2? | Defence |
|---|---|---|
| Delete or weaken tests | no | read-only mount, assertion-count differential |
| Hardcode expected outputs | no | hidden tests, randomised inputs |
| **Add assertion-free tests to inflate coverage** | **yes** | mutation-score the added tests; zero-kill tests contribute zero |
| **Exploit `R` to brute-force patches** | **yes** | `λ_R` penalty plus a cap of one patch per iteration |
| **Exploit the event log**: append a patch that "fixes" a previous patch | **yes** | the final state is re-derived from the log and tested from scratch, never incrementally |
| Detect the eval harness | no | randomised formatting, held-out harness |
| Impossible-task canaries | no | ~0.5% rate, halt if pass rate >2% |

That fourth row is the day-2-specific one and it is subtle: a model that learns to append `<|patch|>x.py` with `-    broken()` `+    pass` can make the *log* look like progress while the file is still broken. The defence is structural — **the verifier always reconstructs the final tree by applying the log to the base commit and runs tests on that**, never on an incrementally-maintained state the model could desynchronise.

---

## 7.6 Environments and distribution

Unchanged in priority order (test correctness first, solvability second), with a day-2 twist: environments now include **repositories at a fixed base commit with a real CI configuration**, because the event-log protocol needs a realistic sequence of reads, test runs and patches to train on.

| Task shape | Share | Why |
|---|---|---|
| Single file, failing test | 45% | signal density |
| Multi-file change | 30% | the skill that matters |
| Repo-wide (rename, dependency upgrade) | 10% | long-horizon |
| Debug from logs only | 10% | diagnosis |
| Underspecified (should ask) | 5% | calibration |

Difficulty targeting stays a bandit: sample tasks whose current pass rate is in [0.2, 0.6], updated every epoch.

---

## 7.7 RL mechanics

GRPO, group size 8, clip 0.2, **constant loss divisor** (day 1's D10 — per-sequence normalisation rewards verbosity), KL 0.02 to the SFT reference.

Two day-2 specifics:

**Per-depth advantage normalisation.** Because `R` varies within a batch, raw rewards are not comparable across rollouts: a `R = 6` rollout spent 5 extra iterations of compute. Normalise the advantage by the rollout's compute cost, not just its length, or the policy learns that thinking is free.

**Zero-variance groups.** Same as day 1: keep only groups with `0 < pass < 8`, and use the discard statistics to resample difficulty.

---

## 7.8 Release gating

1. Canary pass rate < 1% on 500 impossible tasks.
2. No regression >2 points on the SFT suite.
3. **`R` policy sanity:** mean `R` on easy tasks ≤ 1.5, on hard ≥ 2.5. A model that uses `R = 3` everywhere has not learned the policy, whatever its pass rate.
4. **Protocol compliance:** on 200 held-out event-log tasks, zero cases of patching stale content.
5. Reliability at 3 consecutive successes, not pass@1.
6. Held-out repositories only.

Row 3 is the one that decides whether §7.4 worked, and it is the kind of check that is easy to omit because the headline number looks fine without it.

**Next:** [08 — Serving](08-serving.md).
