# Real-text halting pilot: depth first on text, then joint stopping

Maintainer A, 2026-10-20; agreed by maintainer B the same day (channel 0617971) with the amendments below,
which are part of the plan. Exploratory: no formal protocol, measured fixed update and token
caps for about an hour per run, no automatic extensions; code goes through the code lane (tests
red first, mutants, full gate, maintainer B's code review). Stage 2 of
docs/reviews/2026-10-15-halting-experiment-plan.md (decided by the project owner on 2026-10-15: a bounded
real-data GPU pilot of about one-hour runs), in the shape of the adopted, NOT PROVEN
docs/reviews/2026-10-17-halting-in-phases.md.

## Why now

On the stage-1 task, joint mode (sampled stops with copy-forward, backbone and head trained
together, rho 0.1 dense auxiliary) at depth cost 2.5 with long training met every recorded
criterion in 7 of 8 realizations on 2a16c44 and 8 of 8 deterministic runs on d388338, at about
2 active passes of 8. That task is built so that depth need varies by token. The question here
is whether the useful allocation survives text at a size we can train on this card.

## Phase 1: depth first, on text

- Model: a small Vesper shape, sized by a measured preflight so a run takes about an hour on
  the 5060 Ti, with the frozen words-v1 tokenizer and the finalized selection.
- Splits: document/group-disjoint validation and test parts pinned before any choice. A part
  of the old held-out split already read is reused evaluation, never fresh confirmation.
- Objective: every pass supervised with equal weight on a dense forward
  (`vesper.train --objective fixed`: `Vesper.all_pass_objective(condition="fixed")`, stage 1's
  condition B; the objective is part of the recipe and of the checkpoint identity, ACT stays
  the default).
- The check: the static held-out profile, the pooled NLL at each fixed depth 1..T
  (`evals.static_profile`, recorded at every checkpoint), with the paired per-token NLL
  changes between depths from the same evaluation beside the means. A flat mean curve is a
  practical stop for this shape and budget, not proof that nothing can be allocated (some
  tokens can gain while others lose): report the scoped result and pause.

## Phase 2: joint stopping, from the phase-1 weights

- The joint update in its present form: the score term plus the pathwise term, the
  backbone-only dense auxiliary (rho 0.1), two rollouts with sampled stops and copy-forward.
- Context: `_update_loss` starts fresh states for every trajectory and for the dense term, so
  phase 2 and every evaluation it is compared with run on explicitly independent chunks under
  one context/reset policy. A carried text stream would need its own state and credit contract.
- Pooling: NLL pooled by actual loss-token counts across variable or padded chunks; active
  depth priced over real valid tokens, padding excluded.
- Price: the cost is NLL + dc * D / T, so for a phase-1 gain of g nats per token over deltaD
  passes the central price is dc = T * g / deltaD (not g / deltaD). A small multiplicative grid
  of three prices around that measured reference; two seeds (sensitivity, not a significance
  threshold).
- Preflight: peak memory and whole-step time including both rollouts and the dense term.

## Evaluation

- The lower frontier is built in pooled NLL (each fixed depth and two-depth mixtures), then
  exponentiated for perplexity; mixing perplexities is a different benchmark.
- The adapted model's own fixed-depth and matched-depth random controls; the phase-1 source
  profile reported separately.
- Paired validation differences to choose the cell; one test read of the chosen cell.
