# Stage 1: the crossed-seed escape diagnostic (GPU, condition B, one column)

Date: 2026-10-16. Decision: maintainer B (channel 5dba6af, after c1-slow-v2 failed both gates).
Implementation and execution: maintainer A, within the cap below. Status: frozen before launch; the
record of what ran is the run directories, the escape table and this file.

## History it follows

- c1-slow-v2 (docs/reviews/2026-10-16-stage1-c1-recipe.md, CPU, reference scan) failed both
  gates: seed 0 learned hop2 and read1 only at about 30000-36000 updates (hop2 72.9 % at pass 3,
  +58.8 over pass 1), seed 1 learned nothing past one hop. It stops there; no extension.
- What it supports (maintainer B): variability between seeds, not its cause. A seed sets both the
  initialization and the data order, and an onset after a curriculum stage does not exclude that
  stage's earlier effect, so neither factor nor curriculum independence follows.
- Maintainer B's loss split at seeds 0/1, 24000 and 36000 (first 128 validation episodes): hop1 is
  0.66-1.80 % of the loss; early/late hop2 gradient cosines +0.48..+0.84. No simple hop1 loss
  dominance or aggregate early/late opposition in that sample; component interference, other
  batches and optimizer dynamics are not excluded.
- The GPU backend passed the two checks maintainer B set before any long run (receipts in the channel
  queue, backend-check-11a60fb): fixed weights and batches, old and new CPU reference within
  1e-7, CUDA reference within 7e-6 of the CPU, fused within the model-level bound (worst
  gradient 4.5e-2 against 7.5e-2; the kernel's 1.5e-2 is its own level's bound); eager against
  captured training of B over 120 steps across the warmup bitwise equal in every record,
  parameter, Adam moment and step counter.

## The recipe

| Item | Value |
|---|---|
| Model | configs/halting-stage1-c1.yaml (C1 / T8 / key filter K4), unchanged |
| Condition | B (`fixed`) |
| Cells | initialization seed {1, 4} x data-order seed {1, 4}: (1,1), (1,4), (4,1), (4,4); fresh starts; `--seed` the initialization, `--data-seed` the data order, both recorded, results keyed by the pair |
| Batch | 32 sequences of 88 tokens |
| Updates | 36000, then stop: no seed replacement, no extension, no best-checkpoint selection |
| Learning rate | the existing 100-step warmup, then constant |
| Training mix | `--mix v1` |
| Backend (training) | `--device cuda --scan fused --cuda-graph`: float32 parameters, activations and states, no autocast, the fused scan's q, k, v, beta in bf16; matmul TF32 off, float32 matmul precision "highest"; every run records it in `result.json` `backend` (with torch, CUDA, triton and fla-core versions) |
| Backend (evaluation) | every probe, static check and the final evaluation on a CPU copy with the reference scan, the same for all four cells |
| Static checks | `--static-every 1000 --static-batches 8`: after every 1000 updates, B's static profile (every token forced to depth j = 1..8, copy-forward) on the SAME first 256 validation episodes (8 x 32), into `static_checks.jsonl` |
| Final evaluation | the 1024 validation episodes (32 x 32), every class, every depth; the final split unopened |
| Launch | from a pinned clean worktree of the commit carrying this file; four runs at once on the 5060 Ti |

## Endpoint (vesper/halting_escape.py)

The first of TWO consecutive static checks (1000 updates apart) whose hop2 accuracy, maximized
over the static depths 1..8, exceeds 50 % on those 256 episodes. Never crossing is
right-censored at 36000; a crossing at the last check alone is unconfirmed, not an escape.
Kept: every check's counts (all classes), the probe curves every 250 updates, and the terminal
1024-episode static profile of every class.

## Rules

- No CPU-protocol verdict: the report's gates (`probe-v1`, `c1-slow-v2`) are for CPU/reference
  records and never read a GPU record; `vesper.halting_report` refuses a crossed-seed record
  outright. `vesper.halting_escape` reads these runs.
- CPU and GPU runs stay separate: an escape time here is not compared with a CPU run's as if the
  training were interchangeable.
- What it can show: initialization, data-order and interaction patterns for these two selected
  values (the earlier CPU slow and fast examples; a selection rationale, not a guarantee of their
  GPU behavior). What it cannot: a population escape distribution, or v1 against v2-slow.
- The 12 formal stage-1 runs stay gated.
