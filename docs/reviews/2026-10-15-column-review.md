# Independent column review

Base: e72ecb2dc8d5e205e2c95b432183b6ddc06c8566. Reviewer: maintainer B.
Time: 2026-10-15T00:28:40+00:00. Status: changes requested. Full CPU gate: 197 tests green.

## Positive independent evidence

A one-token, two-dimensional scalar oracle with nonzero initial H, explicit
projection matrices, RMS arithmetic, window-one attention and squared-ReLU FFN
predicts output (14.740768583677692, 15.336307522274701). The column matches
within 2e-12 and its new H matches the independently calculated matrix. The
inactive pass retains the same objects. Expectations call no tested sublayer.

Two distinct columns, three passes and six tokens with nonzero per-pass initial
states: pass-major versus token-major maximum output error 8.66e-15, recurrent
state error 2.22e-15, maximum gradient error 4.44e-14 across 35 tensors (input,
parameters and initial states). This supports schedule equivalence in the
checked float64, no-halting, no-retrieval case, including gradients.

The new delta_update extraction retains the independent recurrence oracle in
the full gate. Comparing the step with its extracted helper alone is not an
independent equation check; the existing rational/overwrite expectations are.

## Missing sublayer-order regression

In a separate archived copy, swapping attention and FFN passes all 12 existing
column tests. The scalar oracle above rejects it (logs attached). Add the
independent order-sensitive expectation to the regression suite. Cross-schedule
agreement, all-parameter gradient presence and zero-projection identity do not
determine the specified sublayer order.

## Precision contract and normalization

Native bf16 and fp16 columns/states fail at the blocked scan's inverse because
that operation rejects those dtypes. CPU bf16 autocast with float32 weights
succeeds with finite gradients. Output follows input dtype in the checked
float32/bf16 cases; recurrent state and cache follow the initialized state
choice. This is not proof that mixed-precision training is broken.

Specify and test the intended activation, accumulation, persistent-state and
output dtypes, including the actual CUDA path. If autocast is required, make
that part of the contract; if native low-precision inputs are supported, the
scan needs an explicit stable compute policy. Do not silently change the state
precision underlying the memory budget. Both forward and backward, a carried
chunk and a step/chunk comparison need the chosen tolerance and dtype checks.

RMSNorm outputs zero for finite constant rows of 1e20 in float32/bfloat16 and
1e200 in float64, while x/sqrt(mean(x^2)+eps) is approximately one at gain=1.
The squaring overflows. Use scale-safe evaluation of that same formula and
check zero/small/ordinary/extreme finite inputs and gradients. The registry's
formula is correct; eps is a variance offset, not a clamped floor of the RMS.
At x=1e-4 the current result is 0.099503719, while an RMS floor of 1e-6 would
give 1. Fix the comment/docstring wording without changing the chosen formula.

## Dependency/gate mismatch

The current lock resolves CPU Torch 2.14.1+cpu and CUDA Torch 2.10.0+cu128.
Maintainer B's runtime confirms the first; maintainer A's pilot receipt reports the second.
gate.sh says the releases are equal except for the index and audits only the
CPU resolution. That premise is false. Align supported upstream versions or
separately audit and test the resolved GPU dependency set, with the distinction
explicit. This identifies missing coverage, not a known vulnerability.

## Pilot and mixture follow-up

The shared 14-file manifest's SHA256 matches the pilot receipt. Maintainer B has not
read the private data files or the tokenizer artifact, so input-file hashes,
reported vocabulary and round trips remain attributed to maintainer A. Requested a
shared copy of the small tokenizer JSON for independent artifact checks.
Exact pilot UTF-8 bytes remain unrecorded; the budget and whole-document rule
are stated. That is an explicit receipt limitation, not fabricated precision.

Deduplication needs a versioned key that preserves code indentation and string
literal spaces; do not rewrite training text. Source and byte/token quota
choices otherwise stand as proposals recorded by maintainer A. The loop's halting,
copy-forward numeric cases, retrieval and local objective remain outside this
column review. No production source edits or Vesper commits/pushes by maintainer B.

## Outcome, appended 2026-10-15 by the other maintainer

Applied on dev/loop in the commit that adds this file. RMSNorm computes its formula
scale-safely (the row divided by its largest magnitude, floored at 1e-3 and held constant,
before squaring), so rows at 1e20 in float32 and 1e200 in float64 normalize to the gain; the
value and gradient equal the formula's in the interior, and the constant inside the root is
described as a regularizer, not a floor (a constant row of 1e-4 gives 0.0995). The precision
contract is chosen and written: projections in the activation dtype, the scan in
promote_types(dtype, float32) with autocast disabled, the recurrent state kept in that dtype
(init_column_state starts it there), reads cast back for the output projection; tested as a
native bf16 column on CPU, CPU autocast bf16 against float32, and CUDA autocast bf16 against
float32 on the box's GPU. An expectation computed outside the module, one token through all
three sublayers in order with hand-set weights, is a test now; swapping attention and the FFN
fails it. The torch release is pinned to 2.10.0 on both indexes so the cpu audit speaks for the
gpu set, with the gate's comment corrected. The loop, with your section B table as its fixed
expectation, is on the same branch.
