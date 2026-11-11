# Original-model distillation anchor

Owner: maintainer B. Reviewer: maintainer A. Exploratory A3, November 3, 2026.

Joint text training regressed at every fixed depth in J1/J2. Maintainer A's later A0
control reports substantial regression even with stopping losses removed. That
motivates retention against the original predictions, without establishing the
cause of the drift or that this remedy will work. Experiments proceed under the
existing lane; independent review remains required before a main merge.

## Objective

For original teacher q and student p, use forward KL(q_t || p_t) at each matching
dense depth t, temperature one. Both read the same cold training chunk. Pool over
scored label positions, then average across depths. Add lambda times this mean
to the existing objective, without normalizing by 1 + lambda. The existing dense
CE coefficient is independent. Only the student backbone receives anchor credit;
the teacher and halting head do not. This preserves each depth's original target,
not a depth-five teacher at every student depth or a stopped-trajectory objective.

The implementation accepts arbitrary nonnegative label weights for the shared
core; the text adapter's b/N weights make its row mean the pooled label mean.
No new data view, token selection, temperature search or empirical success is
implied. Zero anchor weight leaves the old training path unchanged.

## Ownership, memory and execution

`Anchor` makes one independent frozen copy of the model loaded from the checked
source bytes, before head initialization. Source/config hashes and the versioned
objective are recorded before training. There is no second mutable-path load and
no teacher optimizer, gradient or recurrent state carried between chunks.

The student dense forward is shared by CE and KL, including when dense CE is off.
The teacher runs once per unique chunk, with no gradient. Both use the selected
scan/device. Logits are computed in position slices PER depth: at most
ceil(b*n/slices) rows per projection, independent of the number of depths. The
existing non-reentrant checkpoint primitive rebuilds each slice's head work in
backward; no vocabulary activation is retained between slices. Masked outputs are
zeroed before norms, projections and softmax, with static shapes for capture.
This retains teacher weights and O(T*b*n*d) hidden outputs, and adds teacher
forward/projection work. It is not a measured full-size fit or throughput claim.

The first implementation uses PyTorch log-softmax and its backward, without a
new Triton kernel. Matmuls remain batched, vocabulary storage is bounded, and the
existing capture path includes the entire combined update. A later fused row
kernel would need measured benefit and its own numerical/capture checks.

## Measurements and command line

`--anchor-weight LAMBDA` defaults to zero; finite/nonnegative is required, and a
positive weight with `--backbone frozen` refuses before source reads. It works
with joint, pathwise and the dense-only control. Training logs keep sampled cost,
score, pathwise and dense values, and add the weighted `anchor` contribution.
The sampled `cost` does not silently become the combined optimized surrogate.

Positive weight also records unweighted validation KL at every saved checkpoint.
`--anchor-eval` enables the same measurement with weight zero, for A0. It uses the
same pinned validation chunks, pools by label count, starts cold and reports
one distance per depth in eval.jsonl and result.json. Ordinary NLL/control graphs
are released before the distance graph is built. CUDA distance evaluation uses a
signature graph binding models, scans, slices and input metadata, with actual
eager/capture/replay counts recorded. Validation-only measurement must not alter
updates, input/action streams or saved weights.

For the next comparison, keep A0's setup fixed and add only the anchor coefficient;
measure both NLL and KL, actual execution time and peak memory. A coefficient is
an experimental choice, not a constant selected by this implementation. Existing
A0 saves can be measured separately against their checked original; these changes
do not rewrite any completed run or label a reused validation set fresh.

## Checks and remaining scope

Tests compare complete distributions and all affected gradients with an explicit
full-vocabulary calculation, unequal masks and uneven slices. They exercise zero
weight, independent teacher storage, no teacher gradient, no saved vocabulary
activations, masked NaNs, empty positions, invalid settings, a real small run,
per-save distance and validation-only preservation. The supplied CUDA tests compare
seven changing updates in joint/pathwise/dense modes, with and without loop
recomputation, checking every parameter and Adam state against same-device eager
execution. They also cross capture and model-identity changes in validation.

CPU validation and the normal full gate are recorded in the review handoff.
CUDA execution and full-size cost are pending the shared card window. Passing
small checks is not a learning result. The working tree incorporates maintainer A's A0
format-patch from 1b51953 on eae3a2d; its provenance remains separate from the
new anchor, and the combined tree is the gate/review target.
