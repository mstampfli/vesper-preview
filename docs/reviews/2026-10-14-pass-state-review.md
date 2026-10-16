# Pass-state and halting review, 2026-10-14

Owner: maintainer B. Base: `52fe933` (state specification unchanged from `c0210ad`).
Status: review and recommendations for discussion; no loop implementation or
Vesper commit. Companion numerical probe and results accompany the channel review.

Keep one recurrent state and KV window per column per pass. Keep copying the
last active pass's projection data as the default. Setting beta to zero is a
different policy, even though it can be implemented consistently in both orders.
The implementation needs explicit rules for state-only updates and gradients;
the inference-time modulation rule also needs a bound before integration.

## A. Why the two execution orders can agree

Consider nodes `(token i, pass t, column c)`. With fixed slow weights during a
chunk, an active node depends on:

- Its previous column in the same pass, or the previous pass's last column.
- Earlier tokens' state/cache updates for this same `(t,c)`.
- Its fixed per-token inputs, conditioning, retrieval result and modulator value.

A copied node depends on its own last active `(k,v,beta,K,V)` for this column and
the preceding token's state/cache at `(t,c)`. It does not depend on a future token.

Pass-major training (pass, then column, with a causal sequence scan) and
token-major inference (token, then pass, then column) are both topological orders
of this graph. Thus they compute the same nodes if the state identities, copy
policy, inputs and arithmetic agree. If randomness is introduced, use the same
per-node randomness for an equivalence check. Slow-weight or modulator updates
must not occur at different points between the two compared executions.

This is a conditional dependency argument, not proof that the unimplemented
loop already follows it. State sharing across passes introduces extra edges
and destroys the argument.

## B. Copy projections, not the entire lower-pass state

One column, two tokens, two passes; scalar unit keys and queries; beta is `1/2`.
The column writes its input value and returns the state read. Token inputs are
`(2,4)`. Token 1 halts after pass 1; token 2 uses both passes. Initial states are
`H^(1)=0`, `H^(2)=10`.

At pass 1, token 1 produces state/output 1 and token 2 produces `5/2`. For the
second pass, token 1's saved value is 2, not its output/state 1.

| Inactive-token policy | Pass-2 state after token 1 | Token 2's pass-2 output |
| --- | --- | --- |
| Apply saved `(k,v,beta)` to the existing pass-2 state | 6 | `17/4` |
| Beta zero: leave pass-2 recurrent state unchanged | 10 | `25/4` |
| Replace pass-2 state with token 1's lower-pass state snapshot | 1 | `7/4` |

The exact-fraction probe executes both schedules and obtains the same answer
within each policy, but the policies give different answers. Therefore passing
a training/inference equality check cannot choose the intended policy for us.

With a single shared state initialized at zero, the same copied-update example
instead gives `19/8` in pass-major order and `11/4` in token-major order. This
is an explicit counterexample to sharing the state across passes.

## C. KV copies and gradient copies are separate obligations

The attention window still includes the halted token. With two zero attention
logits and values `(2,4)`, retaining its KV gives output 3; dropping it gives 4.
An ablation must say separately whether it skips recurrent writes, KV insertion,
or both. Copying only a recurrent beta mask does not define attention behavior.

In the scalar recurrence example above, call token 1's input value `a` and token
2's `b`. The later output under copy-forward is:

```
H1_after_token1 = a/2
H1_after_token2 = a/4 + b/2
H2_after_token1 = 5 + a/2
H2_after_token2 = 5/2 + 3*a/8 + b/4
```

Its derivative with respect to `a` is `3/8`. Detaching the saved copy leaves the
forward value unchanged at `17/4` for `(a,b)=(2,4)`, but changes that derivative
to `1/8`. The PyTorch probe checks both numbers.

Consequences:

- In e2e mode, reuse the saved tensors without detaching their within-chunk graph.
- In local mode, enforce the specified stop-gradient at pass boundaries, including
  saved projection data that crosses a pass boundary. Separate state identities
  alone do not stop gradients through activations or copied tensors.
- Detaching carried history at a sequence-chunk boundary is a separate operation.

The local-mode gradient policy should be supplied by the learning interface so
the same forward computation serves both modes. The next-pass predictive target
and its missing-target cases still need the separate local-mode decision.

## D. Halting removes projections, not every state update

For each inactive token and each later pass, copy-forward still applies a delta
update to that pass's existing state and inserts its KV at the correct token
position. It does not simply copy one matrix into another. An empty active-token
batch therefore does not imply an empty causal sequence update.

The recurrence has `C * T_max * n` state updates under dense copy-forward, even
when expensive active column work scales with the observed pass count. Outputs
and query reads for inactive positions can be omitted. An update-only primitive
should share the delta update with the ordinary step path; do not duplicate its
equation in the loop. FLOPs and wall-clock comparisons must count the residual
copy/update work rather than assuming every operation vanishes on halting.

Keep original token positions for cache insertion and rotary indexing. Compacting
the active batch must not compact the logical sequence seen by later tokens.

## E. Inference modulation can break the recurrence contract

The spec multiplies beta by `(1 + alpha*m)` at inference. Even if the learning
rule's clipping `m in [-1,3]` is also applied here, `alpha=1/2`, `m=3`, and
`beta=9/10` give effective beta `9/4`.

For scalar unit key, initial state 1 and zero value, the updated state is `-5/4`.
Its magnitude grows despite a zero write. Repeating that update grows as
`(5/4)^n`. Thus the advertised contraction argument does not survive this scaling.
The example stays inside the specified clipped modulator range.

Recommendation: explicitly clamp the effective write strength to `[0,1]` after
modulation, preserving the recurrent contract. A bounded logit-shift gate is
another possible design, but is a different mapping and should be chosen
deliberately. Neither change has been implemented by this review.

## F. Acceptance checks for the future loop

1. Compare both schedules using the same initial per-pass states, several token
   halt depths and column-specific saved projections. Use an independent fixed
   expectation such as section B, not only agreement between two implementations.
2. Verify later-token recurrence reads, attention-window contents and original
   positions after an earlier token halts. Include an all-inactive later pass.
3. In e2e mode, check a later-token loss gradient through an earlier halted token's
   copied projections. In local mode, check the corresponding pass-boundary stop.
4. Verify returned persistent state/cache contents, including the last token's
   inactive later-pass updates, before the next chunk starts.
5. Exercise the effective-beta endpoints and the clipped-surprise maximum before
   invoking the recurrence. Verify zero writes cannot amplify a carried state.

The probe verifies the exact examples and the gradient distinction. It does not
test a full Vesper column, ACT weighting, a GPU kernel, a training run, or the
future loop implementation. Those claims remain open until their own tests run.

## Outcome, appended 2026-10-14 by the other maintainer

Sections A to D and F were accepted as written and are the loop's acceptance checks; the
section B table is the loop's fixed expectation. For section E the clamp above remained a
historical proposal: the chosen policy scales the write odds, `beta_eff = sigmoid(logit(beta)
+ ln(1 + alpha m))`, bounded by construction, implemented in `write_strength` and stated in
docs/spec.md (Fast weights during wake) on branch dev/modulation. It changes the interior
mapping as well as the bound (at `beta = 1/2, alpha = 1/2, m = 1` the plain product gives 3/4,
the odds rule 3/5); a zero odds multiplier gives a zero write; a negative one is rejected by
the configuration. The channel record (messages of 2026-10-14, 23:07Z to 23:16Z) holds the
exchange that led here.
