# 10 — Day 6 preview

Day 5 proves the mechanisms. Day 6 must turn them into a model. What has to
be built, in order of how badly it is currently missing.

## 1. The O(n) sparse gather kernel (blocking)

The reference implementation materialises the exact sparse mask and runs dense
matmuls. That is correct and wasteful — it computes attention over the full
`n × n` matrix and throws away 99% of it with `-inf`. The global layers' whole
cost model assumes the O(n) gather over the *selected* set (window + 8 blocks
× 64 + 256 landmarks). Without it, the 6.107 GFLOP/token figure is not
realisable. This is the single largest gap between "verified mechanism" and
"serving model".

## 2. Neural MTP draft heads (declared gap)

The draft-tree pruning machinery is implemented and tested, and
`mtp_spec_depth = 3` is asserted — but there are no MTP draft weights in the
Day 5 ledger. Day 6 adds them and must put them *in* the ledger, which will
move the numbers in doc 02. Quoting the Day 5 parameter counts after that
would be stale.

## 3. Real calibration of the code-FPR split

Doc 08's per-class residuals are calibrated operating points. Day 6 runs the
calibration protocol (held-out 2.5M negatives, labelled by class, per-arm
in isolation, monotonicity as a reject condition) and replaces the pinned
constants with measured ones — or revises them.

## 4. Real kernel timings and a real serving run

Every serving number here is bandwidth arithmetic. Day 6 measures: the gather
kernel's throughput, the WY chunkwise kernel's fusion, and an end-to-end
batch-1 decode. `12,216 tok/s` either survives measurement or it does not; the
label "target" is retired only by hardware.

## 5. Training the indexer and the workspace

The reference runs on random weights. The KL indexer and the Code-JEPA heads
are *architecturally* present and *mechanistically* verified (gradients flow
to unselected blocks; the hypothesis codebook moves the output), but no
training has happened. Day 6 trains them and must show:

- the indexer KL actually decreases (a dead indexer is detectable: it would
  still pass the Day 5 gradient-flow check, because that check only proves
  the gradient *can* flow),
- the JEPA heads predict rather than plateau.

## 6. The remaining honest carry-overs

- The window-complete session footprint (161.48 MB) is what serving must
  allocate; the specified 144.70 MB ledger stays for conformance.
- The `0.307 GFLOP` spec figure is still unreproduced; if Day 6 finds the
  decomposition that yields it, doc 02 should be updated with the derivation.
- The Tier-0 lexer is a sketch (strings/comments at identical indices). The
  production lexer is a Day 6 item; the soundness property it must preserve
  is the one the audit asserts.

## What Day 6 must not do

- Claim `12,216 tok/s` without hardware.
- Claim a validation FPR without running the calibration protocol.
- Move the parameter numbers and keep the old doc.
- Call the dense-mask implementation "the sparse kernel".

The Day 5 contract is small and specific: every number is computed, every
mechanism is probed, and every known gap is written down in the file it
belongs to.
