# 05 — Execution-driven learning

The training signal is execution, not opinion. This chapter is about what that buys, what it
costs, and the one place where day 3.1's arithmetic was optimistic in a way that matters more
than any parameter count: **retries are not independent**.

## The signal hierarchy

| Signal | Cost per candidate | FPR | Can be argued with? |
|---|---|---|---|
| LLM judge, blind | ~0 | 0.719 | yes — that is the failure |
| LLM judge, commit-first | ~1 extra generation | 0.012 | yes, at 1.8 % poison |
| Differential mutant gate | 1 execution + `k` mutants | 0.00216 | no |
| Execution against held-out tests | 1 execution | ~0 (test flakiness aside) | no |
| VEXX tier 0–2 rejection | ~µs per token | 0.000324 | no |

Execution is the only row that is both cheap and unarguable, which is why the corpus is built
from it and why the runtime is built to make rejection cheaper than regeneration. The judge
does not disappear; it becomes a *filter* whose output is never the last word.

## Retries are correlated, and day 3.1 assumed they were not

Day 3.1's depth ladder (attempt at `r = 2`, then 4, 6, 8) was priced as if each retry were an
independent draw:

| Assumption | Correlation `ρ` | Solved | Cost |
|---|---:|---:|---:|
| Independent retries | 0.00 | 0.963 | 1.54 GFLOP |
| Deterministic, unconditioned | 0.78 | 0.716 | 2.35 GFLOP |
| Kestrel-4 hypothesis codebook | 0.18 | **0.932** | **1.70 GFLOP** |

`ρ` is the correlation between successive attempt outcomes: with `ρ = 0.78`, four retries behave
like ~1.2 independent draws, so the ladder buys 25 points of solve rate less than the
independent model predicted while costing 53 % more (the failed attempts are paid for and
return nothing). This is the single largest economic error in day 3.1, and it is not a coding
bug: it is an assumption that was never written down as an assumption.

The fix is not "retry more"; it is **make the retry a different question**. The hypothesis
codebook (`K = 8` orthogonal embeddings `z_k`, 8,192 parameters) conditions the workspace on an
explicit hypothesis vector, so attempt 2 explores a different branch by construction. Assertion
[7] checks the codebook is not degenerate: the workspace at `r = 8` drifts 1.18e-04 from its
`ε`-identity initialization (it starts as a no-op), and two different hypothesis vectors
produce hidden states 0.042 apart.

## Where the data comes from

| Stage | Tokens | Source | Verified by |
|---|---:|---|---|
| Clean-PR + CWM traces | 80 B | merged PRs with tests, CWM-style execution traces | execution |
| SFT + JEPA + VEXX specs | 4 B | curated specs, L3 memory entries, JEPA targets | schema + proof |
| On-policy GKD distillation | 30 B | student samples filtered by the stack | tier 0–2 + mutant |
| VEXX RL + 8× self-improvement | 2.9 B | trajectories with CDCL DPO pairs | execution + verifier |

The 30 B-token distillation stage is only affordable because of §04's poison rate: at
commit-first's 9.7 % poison at `p_solve = 0.10` the stage would have to be filtered by a human
or abandoned; at 0.291 % it runs unattended.

## Preference pairs fall out of conflict, not out of a reward model

Every CDCL jumpback exports one triple: the culprit clause, the rejected branch, the repaired
branch, and the token index to rewind to. Assertion [8] exercises exactly this on the
reference controller — `cdcl_rewind_tok = 4` (the higher-entropy clause at token 4, not the
later one at token 9), `new_bound = 0.43` (a half-space cut below the offending 0.48), and
`dpo_pairs = 1`.

That is a DPO pair with a *reason* attached, which is more than a preference model gives you:
the rejected branch is rejected because it makes a constraint system infeasible, and the
repair is the minimal edit that restores feasibility. The pair is process-level (a step) rather
than outcome-level (a whole trajectory), so 2.9 B tokens of RL data yields on the order of 10⁸
pairs rather than 10⁵ trajectories.

## The workspace is rented, not owned

`CodeJEPAWorkspace` is 4 layers at `d_w = 1024` (120,084,482 parameters, 5.8 % of active), with:

- `r` recurrent iterations, `r` a runtime dial (1–8);
- an AST localization head (`r = 1`) and a latent execution-trace/outcome predictor (`r = 2`);
- query-attentive pooling for patch confidence;
- `K = 8` hypothesis embeddings.

Cost per extra iteration is `2 × (active workspace) + attention`, i.e. 0.38 GFLOP/token at
`r` from 1 to 2, against 5.06 GFLOP/token for the trunk at `r = 2`. Day 3.1's rent argument —
a workspace iteration costs 12.3× less than a decoded token — survives; what day 4 adds is
that the iteration is *conditioned*, which is what makes it worth renting.

## Rejected alternatives

- **Judge-scored self-play.** 86.6 % poison at `p_solve = 0.10`, 51.9 % at 0.40. Rejected as a
  training-signal source; retained only as a cheap pre-filter ahead of execution.
- **Whole-patch retry.** Correct but 33.3× more expensive per conflict, and it cannot reuse the
  partial patch that was already valid.
- **Distillation from a stronger model only.** No execution signal, so the student inherits the
  teacher's false accepts at the teacher's rate; and it cannot improve past the teacher.
- **Outcome-level RL (RLHF/PPO on final reward).** One bit per trajectory, credit assignment
  over hundreds of tokens, and a reward model that reintroduces exactly the 0.719 failure.
- **Independent-retry assumption.** Kept as the *baseline* in the table above, not as a design
  assumption, because the gap between 0.963 and 0.716 is the cost of pretending `ρ = 0`.

## Falsifier

**C6 — hypothesis codebook.** Fix the token budget and the depth ladder; compare `K = 8`
against `K = 1` (no conditioning) at equal GFLOP. Kill the codebook if the solved-rate gain is
below 5 points, or if measured `ρ` does not fall below 0.4. The simulation predicts 0.932 vs
0.716 solved at 1.70 vs 2.35 GFLOP; if the real gain is under 5 points, 8,192 parameters and a
conditioning path are not worth the complexity, and the ladder should be re-priced honestly at
`ρ = 0.78` instead.
