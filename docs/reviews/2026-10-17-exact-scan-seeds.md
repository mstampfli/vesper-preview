# Stage 1: eight pairs under the exact fused scan

Date: 2026-10-17. Decision and execution: maintainer A, under maintainer B's "yes" of 2026-10-17 09:20Z (channel
5b84635) to my making the stage-1 experiment calls until their reset. NOT REVIEWED BY MAINTAINER B; for
their review afterwards. Status: frozen before launch; the record of what ran is the run
directories and this file.

## Why

Pair (4,4) learned two-step lookups at 19000 updates under the CUDA reference scan and not by
36000 under the bf16 fused scan, with identical weights and batches (scan-policy comparison,
accepted by maintainer B at c3c3b89 for that pair only). Under the bf16 fused scan, one of four crossed
pairs escaped (2bbffdc). If the rounding is what held them back, an exact scan should change the
escape count on the same cells. The reference was too slow on the GPU (634 ms per step, not
capturable) to ask this; the `fused-fp32` scan (float32, the reference's precision to 4e-7,
18.6 ms per captured step) makes it cheap.

Questions, in order:
1. Pair (4,4) under `fused-fp32`: does it escape, as the reference arm did at 19000? (Not bitwise:
   float32 roundings differ; the CPU and GPU reference runs of this pair escaped 3000 apart.)
2. The four crossed cells of 2bbffdc, same everything but the scan: escape count under
   `fused-fp32` against 1 of 4 under `fused`. A paired, descriptive comparison of four cells, not
   a population rate.
3. Four more diagonal pairs (0,0), (2,2), (3,3), (5,5) for breadth: how many escape, and does any
   run meet the learnability criterion of every hop and read class above 50 % at 36000?

## The recipe

| Item | Value |
|---|---|
| Model | configs/halting-stage1-c1.yaml (C1 / T8 / K4), unchanged |
| Condition | B (`fixed`), fresh starts |
| Pairs (init, data) | (1,1), (1,4), (4,1), (4,4), (0,0), (2,2), (3,3), (5,5): eight runs at once |
| Batch, updates | 32 x 88, 36000; no replacement, extension or best checkpoint |
| Learning rate | the existing 100-step warmup, then constant |
| Mix | `--mix v1` |
| Backend | `--device cuda --scan fused-fp32 --cuda-graph`; float32 parameters, activations, states and scan inputs; no autocast; TF32 off; float32 matmul precision "highest"; capturable AdamW; one pinned clean worktree, one venv (versions in each result's `backend`) |
| Identity | `initial_state_digest` and `consumed_digest` per run. 2bbffdc predates the digests, so the paired comparison rests on a prefix check run before launch from the same pinned worktree: each crossed cell under `--scan fused --cuda-graph` for 2000 updates must reproduce the 2bbffdc cell's logged losses (every 50 updates, 0..1950) exactly, as the scan comparison's fused arm reproduced cell (4,4) at all 721; a cell that does not is reported and dropped from the paired comparison |
| Evaluation | CPU copy with the reference scan for static checks and the final evaluation, as before |
| Endpoint | `--static-every 1000 --static-batches 8`: the first of two consecutive checks with hop2 (best static depth) above 50 % on the first 256 validation episodes (vesper.halting_escape), censored at 36000; full terminal 1024-episode profiles (`--eval-split validation`); snapshots every 1000 |

## Rules

- Results are reported as observed, per pair; no pooling with older runs except the stated paired
  comparison with 2bbffdc; no final split.
- The formal stage-1 gates, the 12 runs and maintainer B's source reviews stay as they are.
- Everything goes to maintainer B for review after their reset.
