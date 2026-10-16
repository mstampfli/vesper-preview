# Halting diagnosis: why the depth did not adapt

Maintainer A, 2026-10-15. Question: why did every token take the same number of passes in both real
runs (smoke-b797837: all 8; short-31ffc2c: all 1 from step 22)? Maintainer B's independent review of the
halting code (`<evidence>/reviews/halting-003d92b/`) reached the same mechanism on
CPU fixtures and corrected the timing; this file holds the measurements on the real run.
Evidence, scripts and SHA256SUMS: `<evidence>/halting-diagnosis/`.

## Method

- The four checkpoints of short-31ffc2c (steps 50..200), and a replay of steps 1..22 from seed 0
  on the same selection with checkpoints at 20, 21 and 22 (the run kept none before 50). Replay
  fidelity: step 1 loss identical, steps 2..22 within 5.1e-4, every logged histogram identical
  (step 22 again 0.99389 at pass 1). Close, not bitwise (bf16 on the GPU).
- Held-out: the run's own 16 evaluation chunks, states carried; 130,998 positions with a loss.
  Harness check: natural-mode perplexity 1,637.99 / 1,160.89 / 966.26 / 891.71 against the
  logged 1,638 / 1,161 / 966 / 892.
- Per pass: the head's logit split into the common-direction part `w . mean(h)` and the
  per-position remainder; the head input's L1 and L2. "Forced" runs report the real head's logit
  while the loop is told `p = 0`, so every pass runs and the output is the last pass alone: a
  last-pass-only readout, not a test of what depth is worth (maintainer B: at step 20 natural and forced
  both run 8 passes and still differ, 3,010 against 3,058).
- Gradients: one held-out chunk, the trainer's next-token and ponder losses separately.
- All measurements here run in float32; training ran under bf16 autocast.
- Histograms: the one a checkpoint's log line carries beside `held_out_perplexity` is the last
  TRAINING batch's, collected before its update (maintainer B); the held-out histograms here come from
  this harness's own forwards.
- The first update: Adam's state after one step holds `exp_avg = (1 - beta1) g`, so the clipped
  step-21 gradient is recovered from the step-21 checkpoint (to the checkpoint's float32
  rounding).

## Measured

| | step 20 | step 21 | step 22 | step 50 | step 200 |
|---|---|---|---|---|---|
| pass-1 logit mean (std) | -1.95 (0) | 6.76 (0.58) | 13.29 (0.99) | 45.5 (10.3) | 27.0 (7.4) |
| common-direction part | 0 | 8.7 | 15.2 | 47.5 | 29.0 |
| share halting at pass 1 | 0 | 0.989 | 1.0 | 1.0 | 1.0 |
| head input L1 at pass 1 | 29,440 | 30,260 | 31,578 | 30,977 | 31,587 |
| head gradient on one held-out chunk (ntp, ponder) | frozen | | | 0.0, 0.0 | 0.0, 0.0 |

- The first free update (step 21, `lambda_h = 0`): every one of the 768 coordinates of `w_h`
  moved by the learning rate to within the reconstruction's rounding (|dw| from 2.98125e-4 to
  2.98139e-4, lr 2.98139e-4), opposite the gradient's sign on 768 of 768, the bias by +lr: Adam's
  first step, `lr * g / (|g| + eps)`. Step 22's update has cosine 0.9997 with it.
- Training did not stay entirely at pass 1: after step 22, 15 training batches sent some
  positions past pass 1 (step 22: 0.61 %, step 50: 0.20 %, steps 133..149: 0.01 % to 0.18 %).
  Those positions give the head a gradient; Adam's first moment of `w_h` grows from step 100 to
  step 150 (maintainer B), which an all-zero history would have decayed to about 1e-7.
- The head input grows with the pass: L2 about 1,400 at pass 1 to 8,600..10,300 at pass 8 in the
  trained checkpoints. At initialization (seed 0) the embedding rows have L2 1.00 and the head
  input after pass 1 already 354, growing about 250 per pass.
- The head's gradient is exactly zero from both losses at steps 50, 100, 150 and 200, while the
  rest of the model gets 1.39 / 1.39 / 1.99 / 1.45 from the next-token loss.
- Read out from the last of 8 passes, the trained model is worse than at its own one pass (1,586
  against 892 at step 200); at step 21 the two were close (2,996 against 3,033). Both are
  last-pass-only readouts (Method), so neither says what a learned depth would be worth.
- The smoke run's log shows the same event with the other sign: the head freed at step 5 at
  `p = 1/6`, every token at all 8 passes from step 6 to 40 under the full penalty.

## Mechanism (what the measurements support)

1. The head reads the raw residual state, whose scale is set by the residual branches, not by
   the embedding. Adam's first step on a freshly freed parameter is `lr * g / (|g| + eps)`,
   `lr * sign(-g)` up to `eps`, on every coordinate, so it moves a logit by up to `lr * |h|_1`,
   about 9 here.
2. That shift is common to all positions (8.7 against a per-position std of 0.46): one step
   moved 98.9137 % of the held-out positions past `1 - epsilon` at pass 1 together, and the
   next step the rest. The decision is per token in the
   code (maintainer B's oracle: depths [1, 8] for two positions; the global-mean mutant fails it); the
   collapse is a shared offset, not a reduction across positions.
3. ACT's allocation gives the halting pass the remainder `1 - P`. A token halting at pass 1 gets
   `a_1 = 1` whatever `p_1` is, and its depth is 1 whatever `p_1` is: no gradient reaches the head
   from either loss. The region is nearly absorbing for the head: only the rare positions that
   pass the first pass (at most 0.61 % of a batch after step 22) give it a gradient, while the
   rest of the model trains as a one-pass model.

Not established: whether the head would ever have recovered through those rare gradients,
changed inputs or Adam moments (maintainer B's caveat); it did drift (|w| spread 0.025 to 0.37, pass-1
minimum held-out logit 14.3 to 6.5) without approaching the threshold. What the residual scale at initialization costs the
model beyond the head is not measured.

## Remedies, proposed for review, none implemented

1. The head reads a normalized state, `p = sigmoid(w . RMSNorm(h) + b)` with its own gain,
   reusing the column's RMSNorm: with unit gain `|RMSNorm(h)|_1 <= d`, so the first update moves
   a logit by at most `lr * (d + 1)`, 0.2293 here (maintainer B), at every pass alike; a learned gain and
   later updates change that bound. Test, red first: maintainer B's fixture shape (raw input 30 per
   coordinate, one fresh AdamW step from the warm head moves depth 8 to 1), also under bf16.
   Avoiding the collapse does not by itself show useful adaptive depth: that is the small CPU
   task's job, then a short run.
2. The pass-1 flat region needs a change of the halting rule; ACT's remainder makes the halting
   pass's own `p` gradient-free by construction. A separate design and experiment from 1, after
   1 has been tried alone (maintainer B). Options:
   - (a) Stick-breaking weights over all passes in training, `a_t = p_t prod_{j<t}(1 - p_j)`, the
     last pass the rest; inference stops a token when the remaining mass falls below epsilon (the
     truncation error bounded like ACT's epsilon). Every `p_t` gets a gradient, also for a token
     that stops at pass 1, but only if training keeps the post-stop passes' outputs in the
     mixture; those outputs must be computed from states into which the token wrote its carried
     projections, so the states stay equal to inference. The reference zeroes inactive outputs
     today, so this needs a separate activation continuation while the state writes stay frozen
     (maintainer B). First test: state equality, same token and later tokens, between the dense training
     forward and token-major inference with mixed stop passes, separately from the output
     approximation (tail mass `r` returned to the stop pass bounds the hidden-output error by
     `r * max |h_t - h_tau|` and the ponder error by `r * (T - tau)`). This is not PonderNet's
     objective, which is the expected per-step loss plus a KL to a prior
     (https://arxiv.org/html/2107.05407).
   - (b) Keep ACT and check halting only from pass 2: `p_1` then always has a gradient through
     `a_1 = p_1` and `a_2 = 1 - p_1`; the flat region moves to the halting pass, and every token
     pays two passes.
   Either changes spec section 4's halting rule and is decided with the project owner before code.
3. Separate item: the residual scale at initialization (measured above, cost unknown).
4. Instruments: per-pass halting logit mean/std/min and the head's gradient norm in every step's
   log line, and the pass histogram on the dashboard's run view. Either would have shown this at
   step 22.
