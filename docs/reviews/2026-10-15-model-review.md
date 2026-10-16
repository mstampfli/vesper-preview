# Model and loop revision review

Model: 49fff6b52df4a41da09471650e4a1e54f72df9fc.
Loop incorporated unchanged: 222a7007f1d95b7a551ff3219b2b686d73d57b05.
Reviewer: maintainer B. Time: 2026-10-15T01:09:47+00:00. Status: three changes requested below.

## Verified corrections and model evidence

The fresh full CPU gate is green: union dependency audit reports no known
vulnerabilities, 225 tests pass and two CUDA cases are skipped. Maintainer A's selected
shared full GPU log reports the audit plus 227 passing tests; maintainer B read that
receipt but did not independently run CUDA. Torch is now 2.14.1 on both extras,
and the gate exports both graphs, normalizes local wheel labels and audits
their union. The earlier dependency coverage finding is closed.

Original loop counterexamples now pass: expected-depth limits and d/dp=-3 at
p=.1; single-column inactive NaN masking; fp16 small-row values and the
zero-row derivative of 1000; rejected width broadcasts. Carried projection
shape checks and documented conditioning/state-memory changes are present.
Dense inactive work is explicitly an unfinished performance contract.

The assembled-model scalar oracle uses explicit E, conditioning, unit-depth
allocation, zero column output projections, nonunit final-normalization gains
and tied logits. It agrees with hand calculations to 2e-12; the masked objective
is 0.016279275725065493 versus expected 0.01627927572506515. The fixture has
61 parameters counted once. Vesper-S's implemented count is 59,429,377, excluding
the still-absent retrieval parameters; this is not the full 60,221,953 model.

A separate fixed-weight, two-column/three-pass model check compares a whole
chunk with two consecutive subchunks, retaining the graph across the artificial
cut. Logits and parameter/state gradients agree within the attached tolerances;
changing only future token IDs leaves prefix logits unchanged. This checks
causal composition and continuity, not the driver's deliberate chunk-boundary
BPTT detachment or training quality. Numeric results are in the JSON receipt.

## 1. Mask inactive values between columns

Loop currently sanitizes h after all columns have run. With two ColumnLike
objects, first returning NaN only for inactive outputs and second computing
w*x, an immediately halted token has finite y=2 but d(y)/dw=NaN instead of 1.
The next column reads an explicitly undefined inactive value before the mask.

Sanitize the boundary after each column, before another consumes h, and pin
the multi-column parameter/input gradients. The one-column PoisonColumn test
does not cover this composition. Internal Column.step implementations must
also keep ignored computation from generating nonfinite parameter gradients;
an outer output mask cannot retroactively undo invalid backward arithmetic.

## 2. Select loss positions before cross-entropy

wake_loss runs cross-entropy at all positions then indexes the result. With
logits [[[0,0],[NaN,NaN]]], targets [[0,0]], mask [[true,false]], the forward loss
is finite log(2), but the excluded logits' gradient is [NaN,NaN], not zero.

Validate the boundaries, select the live logits/targets, then evaluate the
nonlinear loss. Preserve empty-mask rejection and exact zero gradients for
excluded positions. The current finite-bump test checks the scalar loss only;
add a masked-nonfinite backward test. ntp_loss's unmasked contract can remain
separate for callers that need all per-token values.

## 3. State the mixed-dtype norm behavior accurately

RMSNorm's docstring promises input dtype. With default float32 gain and fp16 or
bf16 input it returns float32, because the gain multiplication follows the cast
back. Apply the gain in the intended accumulator and cast the final result if
input dtype is the contract, or explicitly choose/test the alternate behavior.
Cover mixed gain/input dtypes and gradients, not only module.to(dtype).

## Test claim and remaining scope

The optimizer test named one-step compares losses[2] < losses[0], which checks
two intervening updates. Rename it or assert the first post-update loss if the
one-step claim is intended. This is a verification-description gap, not evidence
that the optimizer fails to lower the first-step loss on the current fixture.

Retrieval, ledger, dusk, local mode and the collector are still absent or future
work. A full passing gate does not cover the new counterexamples above. No
production source edits, Vesper commits, pushes or main merge by maintainer B.

## Outcome, appended 2026-10-15 by the other maintainer

Applied on dev/model. The three remaining changes from the masking review are in 3e8aec5:
inactive positions are zeroed after every column, so a second column never reads an undefined
value and its gradient stays finite (the two-column poison case is a test with its gradient of
one); wake_loss selects the live positions before the cross-entropy, so a non-finite logit at a
masked position has no path into the value or the gradient (test: NaN at the masked position,
loss log 2, gradient exactly zero there); RMSNorm applies its gain in the accumulator and
returns the input dtype under a float32 gain (test for fp16 and bf16). The optimizer test now
asserts the first update, as its name says. The scalar assembled-model agreement, the chunk
continuity and the prefix independence results are kept as the model's independent evidence.
