# Embedding and wake-loss review, 2026-10-14

Owner: maintainer B. Reviewed tip: `d69512aefeeca698fc583d9ed4c80c06113a24b9`.
Verdict: the checked nonempty e2e computations are correct; request the boundary,
test and contract changes below. Keeping local mode explicitly unavailable is
appropriate until its target is defined. No Vesper commit or merge by this review.

## A. Independent expectations

The original embedding/learning suites pass all 11 tests. Additional checks use
fixed values rather than the implementation to construct its own expectation:

- With embedding rows `(1,2)`, `(3,4)`, `(5,6)`, lookup IDs `(0,2,0)` return the
  first, third and first rows. The gradient of their sum is rows `(2,2)`, `(0,0)`,
  `(1,1)`. This isolates the lookup path.
- Projecting hidden value `(2,-1)` through that same matrix gives logits `(0,2,4)`.
- For logits `(log(2),0)` and target 0, cross-entropy is `log(3/2)` and its logit
  gradient is `(-1/3,1/3)`.
- Ponder 2 at weight 0.2 and gate 3 at weight 0.4 give total
  `log(3/2)+1.6 = 2.0054651081081643`. Their gradients are 0.2 and 0.4 respectively.

All of these agree on the reviewed code. The per-token objective belongs in
`learning.py`; the modulator should consume losses without owning vocabulary or
logit mechanics.

## B. Changes required

### 1. Empty aggregate loss produces NaN

For logits `[1,0,2]` and int64 targets `[1,0]`, `ntp_loss` returns an empty array
as expected, but `wake_loss(..., mode="e2e")` returns a non-finite total through
`.mean()` of that array. No positive-length precondition is checked or documented.

Recommendation: reject an empty aggregate objective with a clear contract error.
The training driver can then skip a batch deliberately; it should not receive a
silent NaN. Keeping the unreduced helper's empty result is a separate valid choice.
Test both zero sequence length and zero batch size, plus the normal case.

### 2. The tied-gradient test does not establish both paths

In an isolated copy, change only the lookup return to:

```
return F.embedding(tokens, self.weight).detach()
```

All five original embedding tests still pass. The output projection touches all
rows of the shared parameter, so a nonzero aggregate gradient on every row does
not prove the lookup path exists. The independent lookup-only gradient check
fails immediately on this mutant.

The implementation currently has the correct path; this is a demonstrated test
blind spot, not an existing detach bug. Add a lookup-only gradient expectation,
separately from the output projection's gradient. When the full model exists,
also verify global weight sharing across input, output and any local heads;
counting parameters of one embedding instance cannot establish that model-wide.

### 3. The registry has two incompatible embedding interfaces

The old `embed` rows remain specified with int32 inputs, while the new `embedding`
rows describe tested `TiedEmbedding` methods requiring int64. Reconcile the
module name and token-dtype boundary rather than leaving both as current contracts.
The tokenizer/data-to-model conversion should be explicit.

### 4. Loss-weight defaults duplicate configuration

The hard-coded `lambda_h=0.01` and `lambda_g=0.01` defaults are another source of
truth beside the configuration. These are objective weights, not numerical floors.
Under the repository rule, callers should provide the configured weights or a
typed configuration object. Avoid silently enabling a different objective when
a caller omits a keyword.

## C. The local target question is real, but its wraparound argument is wrong

Write `y_c^t` for column `c`'s output at pass `t`, and `u_c^t` for its input before
any separately specified iteration conditioning.

```
u_(c+1)^t = y_c^t                 for c < C
u_1^(t+1) = y_C^t
```

Thus a same-pass next-column target admits the identity solution. Its loss is
not identically zero for arbitrary `Q_c`; it can become zero when the predictor
learns identity. Iteration conditioning may add a known offset, which must be
named in the input convention.

For the proposed next-pass/next-column reading, `u_(c+1)^(t+1) = y_c^(t+1)` for
`c < C`, but the wraparound case is still `u_1^(t+1) = y_C^t`. The last column's
target is its current output, not its next-pass output. The parenthetical in the
local-mode note states the first identity and then draws the opposite conclusion.

Before local mode is implemented, choose whether to retain that special case or
change the target (for example, a same-column next-pass input or a next-column
output). Spell out the pass/column indices, conditioning, missing next-pass
targets, and halted-token masks. Do not call the term nontrivial for every column
until the wraparound case is resolved. The current explicit refusal prevents
training against an accidentally chosen interpretation.

## D. Arrival-time surprise needs an alignment contract

`ntp_loss` aligns each loss with the target supplied in the same array position;
it does not shift anything itself. In a usual shifted training batch, logits at
input position `i` predict token `i+1`, so that loss belongs to the next token.

For arrival-time modulation of token `i`, use its prediction from the previous
position, including the carried boundary prediction at a chunk boundary. Feeding
a shifted next-token loss back into the same input token's fast-state write would
use the wrong token's surprise and could expose a future target. This is an
integration requirement, not a reason to move cross-entropy into the modulator.

The probes were run on CPython 3.12.14 and PyTorch 2.14.1+cpu. They establish these
helper computations and the listed failure cases, not a complete local-learning
algorithm or an end-to-end training result.

## Outcome, appended 2026-10-14 by the other maintainer

All four changes and the wraparound correction were accepted and applied on dev/embedding-loss
in the commit that adds this file: `wake_loss` raises `ContractError` on an empty aggregate
(zero sequence length and zero batch tested; `ntp_loss` still returns an empty result); the
lookup-only gradient check with the rows `(1,2), (3,4), (5,6)` and ids `(0, 2, 0)` is in
tests/test_embedding.py, and a detached lookup was run against the suite to confirm that only
that test fails; the integer boundary is one primitive, `check_ids`, which accepts the data
stream's int32 or int64, checks the range and widens, with the old `embed` registry rows removed;
`wake_loss` takes `lambda_h` and `lambda_g` as keyword-only arguments without defaults; the
loss-to-arrival alignment is `ntp_loss`'s documented contract; and
docs/reviews/2026-10-14-local-mode-question.md is rewritten with the indices, the corrected wrap,
the missing targets and the halted-token masks, proposing that every column predicts its own
output at the next pass. Local mode stays refused until the maintainers agree.
