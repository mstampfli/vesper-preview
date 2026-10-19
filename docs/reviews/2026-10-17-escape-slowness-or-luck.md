# Stage 1: slowness or luck, eight new pairs to 72000 updates

Date: 2026-10-17. Decision and execution: maintainer A, under maintainer B's "yes" of 2026-10-17 09:20Z (channel
5b84635), and the project owner's go ("yes please run it", 15:29Z). NOT REVIEWED BY MAINTAINER B. Status: frozen
before launch; the record of what ran is the run directories and this file.

## Why

Under the exact fused scan, 1 of 8 pairs escaped by 36000 updates (halting-exact-9ce2453); pair
(4,1) was rising at the end (hop2 29 and 30 % at the last two checks). Either most runs are slow
and escape later (then longer training is the cure), or escape is rare however long one trains
(then something early decides it). Doubling the budget on new pairs separates the two.

## The recipe

| Item | Value |
|---|---|
| Model | configs/halting-stage1-c1.yaml (C1 / T8 / K4), unchanged |
| Condition | B (`fixed`), fresh starts |
| Pairs (init, data) | (6,6) (7,7) (8,8) (9,9) (10,10) (11,11) (12,12) (13,13): new, none run before |
| Batch, updates | 32 x 88, `--steps 72000`; the learning rate is constant after the 100-step warmup (`final_lr_fraction: 1.0`), so updates 0..35999 are exactly those of a 36000-update run |
| Mix | `--mix v1` |
| Backend | `--device cuda --scan fused-fp32 --cuda-graph`, as in halting-exact-9ce2453 |
| Identity | `initial_state_digest` and `consumed_digest` per run |
| Evaluation | CPU copy with the reference scan, static checks and the final evaluation, as before |
| Endpoint | `--static-every 1000 --static-batches 8`: the first of two consecutive checks with hop2 above 50 % (vesper.halting_escape), censored at 72000; terminal 1024-episode profiles; snapshots every 1000 |

## Reading, fixed before launch

- Escapes by 36000 are counted separately from escapes in 36001..72000.
- "Mostly slowness": at least half of the eight escape by 72000, most of them after 36000.
- "Mostly luck": at most two of the eight escape by 72000.
- Between the two: report as observed, no label.
- Eight pairs is a small sample: the counts are descriptive, not a rate with a confidence claim.

## Rules

- No pooling with older runs beyond quoting them for context; no final split.
- Everything goes to maintainer B for review after their reset (Thursday 06:13Z).
