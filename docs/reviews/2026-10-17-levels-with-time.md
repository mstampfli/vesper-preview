# Stage 1: do the harder levels arrive with time? Eight new pairs to 144000 updates

Date: 2026-10-17, frozen 18:26Z before launch. Decision and execution: maintainer A, under maintainer B's "yes"
of 2026-10-17 09:20Z (channel 5b84635). NOT REVIEWED BY MAINTAINER B. Committed later as
docs/reviews/2026-10-17-levels-with-time.md (same text).

## Why

In the 72000-update runs (halting-exact72-0e887f7) learning looked like a staircase of plateaus:
hop2 in 6 of 8, hop3 later in 2 (after 36000 and after 54000), hop4 in none, one run rising at the
end. Whether the higher levels arrive with more time, and how long each wait is, decides how much
training the halting runs need before they start (the halting-in-phases plan, adopted, not
proven: depth must be useful first).

## The recipe

Exactly the recipe of docs/reviews/2026-10-17-escape-slowness-or-luck.md (C1/T8/K4, B, v1, batch
32 x 88, constant learning rate after the 100-step warmup, fused-fp32, captured step, static
checks every 1000 on 8 x 32 validation episodes, CPU/reference evaluation, snapshots every
1000), except:

- pairs (14,14) (15,15) (16,16) (17,17) (18,18) (19,19) (20,20) (21,21), none run before;
- `--steps 144000`;
- the same pinned worktree at 0e887f7 (vesper/ and configs/ identical to 9ce2453).

Runs finish inside maintainer A's quiet hours (bounded job, allowed); read at 06:00Z.

## Reading, fixed before launch

- Per pair and per level (hop2, hop3, hop4, read1-read4): the first of two consecutive checks
  above 50 % (best static depth, 256 episodes), or censored at 144000; counted by 36000, by
  72000 and by 144000.
- "Levels arrive with time": at least half of the pairs whose hop2 escaped also escape hop3 by
  144000, and at least one escapes hop4.
- "Levels mostly stall": at most one pair escapes hop3 by 144000.
- Between: report as observed. Eight pairs: descriptive counts only.
