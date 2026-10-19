# Halting in phases: depth first, then stopping

Maintainer A with the project owner, 2026-10-17, in conversation after reading LT2 and the sleep paper
(docs/related-work.md). Status: an AMENDMENT to docs/reviews/2026-10-15-halting-experiment-
plan.md, ADOPTED by the project owner on 2026-10-17 as the plan for the halting runs, and NOT PROVEN: no
phase has been run; the collapse it is meant to avoid is documented, the cure is not. Not
reviewed by maintainer B. Nothing below is built; the eight-pair exact-scan runs
(docs/reviews/2026-10-17-exact-scan-seeds.md) were unchanged by it.

## Why

Per-token halting trained from scratch collapses: every token stops after one pass (Vesper's
short run of 2026-10-15; the head's gradient was exactly zero after the collapse,
docs/reviews/2026-10-15-halting-diagnosis.md), or, with a weaker compute penalty, no token stops.
LT2 (appendix A) reports both regimes for ACT in pre-training and dropped it; Ouro reported the
same. The cause is a chicken and egg: at the start extra passes do not help yet, so stopping
early wins, and once every token stops early the deeper passes are never trained and never
become useful.

## The phases

1. **Depth first.** Every token runs every pass, each pass supervised (condition B of the plan),
   until depth is useful: on the stage-1 task, until the multi-hop classes have escaped and the
   static profile shows k-hop classes answered at their own depth.
2. **Real stopping, with exploration.** From those weights, the halting head decides, tokens
   really stop, and a stopped token's write is copied forward into every deeper pass (spec 4),
   so training sees the same notebooks as deployment (the state gap of the 2026-10-15 review,
   section 6, does not arise). A floor on the probability of continuing (the project owner's proposal) keeps
   the deeper passes trained and their benefit visible even where the head would stop.
3. **Exploration off.** The floor is lowered to zero on a schedule, then a short stretch trains
   with none, so the model adapts to its own deterministic stopping (other tokens' writes change
   when their stops stop being random).
4. **Measure the deterministic policy**, the one that would be deployed.

## What counts as success (unchanged from the plan, made explicit)

- Depth follows need: per class, mean passes rise with k, and each class is answered at about
  the depth its static profile needs.
- Quality holds: deployed accuracy per class within the plan's copy-forward bound of the
  all-pass model, at a measured cost below it.
- Against a matched-cost random policy and the fixed depths; a spread histogram alone is not
  success (the review, section 1).

## Open, to settle before the runs

- The form of exploration: a floor on the continue probability, or sampling the stop from the
  head's own exit distribution (PonderNet; ALoDLM's latent schedule is a further option).
- What the head is trained on in phase 2: the existing objectives A-D, or a supervised target
  from the measured improvement of the next pass (Ouro's second stage), which is greedy: an
  improvement two passes on is invisible to it (the review, section 7).
- Floor value, schedule and the length of the final stretch, from the stage-1 numbers.
- Which escaped checkpoints phase 2 starts from: the eight-pair runs will say how many there are.
