# Stage 1: the one-column calibration recipe (c1-slow-v2)

Date: 2026-10-16. Decision: maintainer B (channel 4f97b71, the stage-1 design call). Implementation and
execution: maintainer A, within the cap below. Status: frozen before launch; the record of what ran is
the run directories and this file.

## History it follows

- The 12000-update learnability probe at f4c9279 (two columns, mix v1) failed both gates
  (~/data runs halting-probe-f4c9279; report section 10). It stays the failed probe; it is not
  re-read under any later protocol.
- Two calibration extensions on the validation split, not stage-1 runs: the probe extended to
  36000 updates (seeds 0 1 3 4) and the same model with ONE column (24000 updates). With two
  columns every class at or below two hops was flat over depth; with one column the learned
  classes showed a staircase (seed 4: hop2 17.6 -> 93.7 at pass 2, hop3 20.5 -> 77.6 at pass 3,
  hop4 jumping at pass 4 to 36.7; read1 at pass 2 in seeds 3 and 4). Maintainer B verified the counts,
  reloaded the weights and recomputed every profile.
- What this does and does not establish (maintainer B): an empirical profile of those learned weights,
  not a structural depth bound; hop4 stayed below the learnability criterion and seed 3 learned
  read1 while its hop2 stayed at 22 %, so neither k-pass necessity in general nor learning time
  as the only remaining bottleneck follows. Optimization, supervision and interference remain
  open alternatives.

## The recipe

| Item | Value |
|---|---|
| Model | configs/halting-stage1-c1.yaml: the stage-1 model with `columns: 1` (C1 / T8 / key filter K4) |
| Condition | B (`fixed`): uniform all-pass objective, no halting head trained |
| Seeds | 0 and 1, fresh starts |
| Batch | 32 sequences of 88 tokens |
| Updates | 36000 (the config's `total_tokens` 101,376,000), then stop: no best-checkpoint selection, no extension |
| Learning rate | 100-step warmup, then constant (`final_lr_fraction 1.0`), the existing optimizer recipe |
| Training mix | `--mix v2-slow` (vesper/synthetic.py `MIX_SCHEDULES`): free hop queries (.7,.1,.1,.1) for updates 0..7999, (.4,.3,.15,.15) for 8000..15999, uniform 16000..35999; the mandatory one-of-each-k queries, balanced registers, layout and objective unchanged |
| Evaluation | the same 1024 validation episodes (32 x 32) as the probe; final split unopened |
| Gates | report section 10, protocol `c1-slow-v2` (vesper/halting_report.py `PROTOCOLS`): Gate 1 every hop AND read class above 50 % at its best depth in both seeds; Gate 2 one and the same learned class gaining at least 5 points over pass 1 in both seeds |

The schedule's stages are a candidate motivated by the delayed onsets measured in the
extensions (hop1 3500-6500, hop2 8000-10000 updates), not a measured optimum.

## Rules

- The report gives verdicts only for records that match `c1-slow-v2` exactly (seeds, updates,
  split, population, mix, columns); anything else keeps its tables and reads "not assessed".
- Failing read2-4 cannot be set aside because the hop curves look encouraging.
- If this candidate fails: stop and diagnose before proposing more time.
- The 12 formal stage-1 runs stay gated. A later A-D comparison shares one frozen curriculum and
  budget; no per-condition adaptation.
