# Stage 1: levels with time, second batch, pairs (22,22)..(29,29) to 144000 updates

Date: 2026-10-17, frozen 19:59Z before launch. Decision and execution: maintainer A, under maintainer B's "yes"
of 2026-10-17 09:20Z (channel 5b84635) and the project owner's wish to run something overnight (19:58Z). NOT
REVIEWED BY MAINTAINER B.

## Why

A replication of docs/reviews/2026-10-17-levels-with-time.md on eight further new pairs, so the
question "do the harder levels arrive with time, and how often do runs stay stuck" rests on 16
pairs instead of 8, and so the halting runs (halting in phases, adopted, not proven) have more
runs that learned every level to start from.

## The recipe

Exactly docs/reviews/2026-10-17-levels-with-time.md (C1/T8/K4, B, v1, batch 32 x 88, constant
learning rate after the 100-step warmup, fused-fp32, captured step, --steps 144000, static checks
every 1000 on 8 x 32 validation episodes, CPU/reference evaluation, snapshots every 1000, the
worktree pinned at 0e887f7), except the pairs: (22,22) (23,23) (24,24) (25,25) (26,26) (27,27)
(28,28) (29,29), none run before. It starts when halting-exact144-0e887f7 writes "all done", so
the two batches never share the card.

## Reading

The reading rules of docs/reviews/2026-10-17-levels-with-time.md, applied to this batch alone
and then to both batches together (16 pairs). Descriptive counts only.
