# Halting experiment plan: which objective gives useful per-token depth

Maintainer A, 2026-10-15, after the project owner's proposal (PonderNet-style all-pass supervision with a uniform
prior, a frozen head until measured criteria, KL annealed down, compute penalty ramped up),
maintainer A's and maintainer B's adversarial reviews (channel e742aec; docs/reviews/2026-10-15-halting-
objective-adversarial.md) and maintainer B's amendments (channel 1f31541). Status: a PLAN for the project owner's
decision. Nothing below is built. The normalized halting head (690bdca) stays in every condition.
Amended 2026-10-17 by docs/reviews/2026-10-17-halting-in-phases.md (adopted, not proven): the
halting runs start from weights whose depth is already useful and train with real stopping.

## The question

Does any training objective give Vesper input-specific depth that buys quality at a measured
cost, under the state policy it will be deployed with? A spread histogram is a diagnostic, not
the answer (the review's section 1): tokens can spread at random, and identical depth can be
correct.

## Conditions

All on one model shape, data order, token budget and seeds. Compute penalty, release trigger,
KL schedule and a second-stage gate are deliberately absent, so the result identifies the
objective (review, section 8).

| | Training objective | Head | Training state policy |
|---|---|---|---|
| A | present ACT, mixture output, ponder cost (status quo, 690bdca) | trained | ACT's own: copy-forward after the halt |
| B | expected per-pass loss under FIXED uniform exit weights | not trained | dense: every token every pass, fresh writes |
| C | expected per-pass loss + beta * mean_i KL(p_i \|\| u) (per-token) | trained | dense |
| D | expected per-pass loss + beta * KL(p_bar \|\| u) (batch mean) | trained | dense |

- A keeps its real state policy; comparing it with B-D carries that difference and is labelled
  so. A dense-write ACT is a further ablation only if attribution needs it (maintainer B, 1f31541).
- Exit distribution (B-D): hazards `lambda_t = sigmoid(w . RMSNorm(h_t) + b_h + b0_t)` with the
  fixed baseline `b0_t = -log(T - t)` for t < T and a forced final exit, so a zero learned logit
  is exactly uniform (review, section 4); probabilities and logs in log-sigmoid / log-survival
  form; KL direction KL(p || u).
- D's population is the loss positions of the whole optimizer batch, with an explicit
  denominator. On the tiny task one unsplit group per step makes the pooled KL exact; any split
  run must pass the unsplit-oracle test of objective AND gradient first (review, section 2).
- The per-pass heads are evaluated pass by pass with recomputation, never eight vocabulary-logit
  tensors at once; step cost and peak memory are measured, not assumed (about 1.5x forward FLOPs
  estimated).

## Deployment and readout (fixed before any run)

- Training loss: the expected per-pass next-token loss (B-D); A as now.
- Deployed exit: the first pass whose CDF crosses q, q = 0.5 as the reference and swept in
  0.1 steps for the cost/quality curve; the emitted prediction is that pass's own output.
- Deployed state: a halted token's projections are replayed into every deeper pass
  (copy-forward, spec 4).
- Measured three ways for B-D: (i) dense, (ii) the SAME exit trace replayed with copy-forward
  (isolates the state difference), (iii) live gates recomputed under copy-forward (the real
  deployed trajectory); each over prefill, token-by-token decode and carried chunks.
- Agreed before the runs: the copy-forward gap is acceptable if (iii) loses at most 1 point of
  absolute accuracy on every tiny-task class and at most 2 % relative held-out perplexity on the
  pilot against (i). Above that, the private-candidate path (review, section 6) is designed before
  any further claim.

## Stage 1: the tiny task (CPU)

The small test shapes (d 32, 2 columns, T 8, window 8) on synthetic sequences built so the
useful depth differs by token and is measurable:

1. Copy tokens: the target is a token seen a few positions back.
2. Pointer chains on UNSEEN graphs and value bindings, k = 1..4 hops. k is an algorithmic depth,
   not a proven loop minimum (columns and attention can shortcut); each class's empirical need
   comes from B's fixed-depth readouts on held-out graphs (accuracy at depth 1..8).
3. Register tokens: a register's deeper processing writes what a later reader needs, and the
   register's own loss does not depend on its depth. The reader's depth is held fixed at the
   minimum that solves the read with a deep register, so it cannot do the work itself; the test
   is an intervention on the register's depth alone.

Scores per condition, at least three seeds: deployed accuracy per class against MEASURED cost
(executed passes and copy-forward writes); the static curve (every fixed depth); a matched-cost
random policy; conditional and marginal depth entropy and their difference; depth against each
class's empirical need; for registers, reader accuracy with the register's own chosen depth
against its forced shallow and deep depths.

## Stage 2: a bounded real-data pilot (GPU), only for conditions that pass stage 1

A smaller Vesper shape than S on the same selection and frozen tokenizer, sized so one run
takes about an hour on the rented card (its exact size and time measured first), conditions and
seeds as in stage 1, held-out perplexity of the deployed policy against matched-cost fixed depths.
Validation for choosing q is separate from the final held-out test.

## What follows from the result, and what does not

Only after stage 2: a release schedule, a KL floor or anneal, a compute penalty (which is a
geometric tilt of the prior, review section 5) or an improvement gate, each as its own change
with its own test. No Vesper-S long run is implied by this plan; its GPU time stays the project owner's
decision. The decode-time parallel copy-forward (plans 48, maintainer B's constraints) is independent of
which objective wins.

## Contracts tested before the first run

Distribution sums and the forced tail; exact uniform at a zero learned logit for T = 1, 2, 8;
continuity at release; analytic gradients and extreme logits; unsplit against split global KL
(objective and gradient); identical committed states across training and inference orders for
the same realized exit trace; no state mutation from evaluation paths.
