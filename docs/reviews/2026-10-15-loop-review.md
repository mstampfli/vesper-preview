# Independent loop review

Base: 38baeb6f8408a8bd97de7b88c55bd96e273cef15. Reviewer: maintainer B.
Time: 2026-10-15T00:52:35+00:00. Status: changes requested.
CPU runtime: Torch 2.10.0+cpu. Tests: 214 pass, one CUDA test skipped.
The full gate FAILS in the dependency audit; it is not green on this snapshot.

## Ponder objective

The literal raw-probability objective in the spec is implemented, but does not
implement the claimed incentive. With a column returning the same output at
every pass and T_max=3:

| Constant probability | Iterations | Current penalty | d penalty / d p |
| --- | --- | --- | --- |
| 0 | 3 | 0 | 6 |
| 0.1 | 3 | 0.6 | 6 |
| 0.995 | 1 | 0.995 | 1 |
| 1 | 1 | 1 | 1 |

Decreasing p approaches zero cost at maximum depth. This is a specification
and objective defect, faithfully reflected in the current tests. Proposal:
expected depth sum(a_t * t), with a_t the actual allocation weight p before
halting and the remainder at it. This gives cost T_max at p=0 and 1 at an
immediate halt, with a decreasing cost as early-pass allocation increases.
Alternatively define the standard ACT step-count/remainder penalty explicitly.
Update spec, registry and tests together; a test of current arithmetic alone
cannot show that the objective encourages earlier halting.

## Inactive outputs and work

Column.step explicitly leaves an inactive output unspecified. A conforming
stand-in that returns NaN only there makes Loop.y NaN for a token already
halted after one pass: multiplying by zero does not mask NaN. Avoid using
inactive values in the head/output/gradient path; test nonfinite inactive
outputs, mixed masks and gradients. Preserve state-only copied writes.

The dense reference limitation is disclosed. Instrumented linear work is the
same when every token uses one pass or three: 33 calls / 132 input rows for
four tokens and one column at T_max=3. Therefore n_iters is not a compute-saved
measurement for this implementation. Active projection/FFN filtering is still
needed to satisfy the spec's no-column-compute claim; until then, label this
as a dense correctness reference in metrics and contracts.

## Normalization and shape boundaries

The large-finite normalization failures are fixed for the checked fp32/bf16/
fp64 rows, and the independent prior column scalar oracle still passes. A new
fp16 small-row regression remains: x=0.001 -> 0, expected about 0.7071; the zero
row gradient is 0 instead of 1000. The scalar reciprocal of scale^2 overflows
in half precision. Evaluate the norm in adequate accumulator precision and
form the regularizer safely, preserving the formula and the activation dtype.

RMSNorm(8) accepts [1,2,1] and returns [1,2,8]; Loop(d=8) likewise broadcasts
an input width of 1 to 8. Bind the last dimension before arithmetic. Also,
Column.step accepts a carried rec_v of n=1 with an active mask and fresh
projections of n=3, broadcasting one value across three positions. Validate
all carried projection dimensions against the current batch/token/head axes.
These are invalid-input contracts, not claims of ordinary valid-input errors.

## State precision and budgets

Float32 persistent recurrent state is now explicit, and the native bf16 path
has a declared wider scan. Update the old bf16 recurrent-memory figures in the
spec: per stream the float32 H allocation is 6 MiB for S, 33.75 MiB for M and
144 MiB for L, twice the earlier bf16 allocation. These figures exclude KV,
activations, gradients and batch multiplication. Never infer actual storage
from the activation dtype alone.

## Positive schedule and copy-forward evidence

Two real columns, three passes, two streams and seven tokens exercise mixed
2/3-pass halting with nonzero conditioning and initial states. Chunk versus
token execution agrees: output max error 1.38e-14, recurrent state 8.66e-15,
KV 4.89e-15, gradients 3.78e-14 across 38 tensors. The fixed scalar copy-forward
cases and gradient checks also pass in the existing suite. This supports the
checked execution-order/copy behavior, not the ponder incentive or efficiency.

Clarify conditioning: this loop adds u_t once before column 0, while the spec's
"column input" wording and local-target note leave per-column addition open.
Keep the chosen once-per-pass convention explicit in the spec and in a
nonzero-conditioning multi-column example.

## Full gate and GPU dependency graph

Pinning Torch made the upstream versions equal but did not equalize dependency
graphs: the Linux GPU export has 18 packages absent from CPU. My earlier
suggestion that a common pin could suffice was incomplete. Audit both resolved
exports (or their complete union), not only CPU.

Fresh full CPU gate and a separate full GPU-export audit both exit 1, reporting
PYSEC-2026-139 and PYSEC-2025-194 / GHSA-rrmf-rvhw-rf47 on torch 2.10.0. Current
[PyPI 2.10.0 metadata](https://pypi.org/pypi/torch/2.10.0/json) reports the two;
[2.14.1 metadata](https://pypi.org/pypi/torch/2.14.1/json) reports none. This is
a scanner result and a failed required gate, not evidence of exploitation.
No advisory was suppressed. The reported 215-test GPU gate is maintainer A's earlier
report; maintainer B's fresh result differs at the audit stage.

The official CPU, cu130 and cu132 indexes list 2.14.1 cp312 Linux wheels. The
host reports RTX 5060 Ti, driver 580.126.18, capability 12.0. PyTorch's
[CUDA 13 migration note](https://pytorch.org/blog/pytorch-2-12-release-blog/)
states a 580.65.06 Linux minimum for CUDA 13.0. This gives a candidate upgrade
path, still requiring actual CUDA verification and both dependency audits.
Maintainer B has changed no shared driver or another account's environment.

## Pilot artifact and collection contracts

The selected pilot JSON matches receipt and sidecar SHA256
2a9a31b3ad6a8feb7b688931a9de41ee60f21185fd26986e5a07466a1f290304, loads at
32768 vocabulary entries, round-trips seven independent Unicode/whitespace/
code cases and reproduces the 20-token receipt example. Accepted as the
web-only pilot artifact, not as the final mixed-corpus tokenizer. Actual source
files and exact sampled-byte count were not independently established.

The versioned exact-text code dedup key in data 56c2138 addresses the earlier
whitespace concern. The collector remains a documented contract, not acquired
or independently checked data. No source edits, Vesper commits or merges by maintainer B.

## Outcome, appended 2026-10-15 by the other maintainer

Applied on dev/loop in the commit that adds this file. The ponder cost is the expected depth
under the allocation weights, `sum_t a_t t`, defined so in the spec with the first formula's
failure named; tests pin 3 for never halting at T_max 3, 1 for halting at once, 2.7 at a
constant p of 0.1 and the gradient -3 there. Inactive positions are zeroed before anything
reads them (head, output, per-pass record, next input) and the probability is zeroed where a
token is inactive, so no 0 times NaN reaches a value or a gradient; a stand-in returning NaN at
every inactive position reproduces the section B table with finite gradients. RMSNorm computes
its statistics in float32 or wider and casts back: the fp16 row of 1e-3 gives 0.7071 and the fp16
zero row's gradient 1000, tested with the bf16 and float64 cases kept. RMSNorm binds its last
axis and the loop binds x0 to its width; Column.step validates the carried projections' shapes
as well as the active mask; all three reject the broadcast with a ContractError, tested. The
spec names the conditioning convention (u_t added once, to column 1's input) and the float32
state figures per stream in MiB. The gate exports both extras and audits their union; torch is
2.14.1 on the CPU and cu130 indexes (the box's driver 580.126 meets CUDA 13.0's minimum), the
GPU dependency graph audited in that union and the CUDA tests run on the box. The dense
inactive work of this reference stays visible as an unfinished performance contract in the
registry: n_iters is not saved compute here, the kernel path is where it becomes one. Process
finding, in the owner's ledger: the pre-commit hook runs the fast gate without the audit, and
the commits after the 2.10.0 pin were called green on that basis.
