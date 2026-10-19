# Stage 1: one controlled scan-policy comparison for pair (4,4)

Date: 2026-10-16. Decision: maintainer B (channel 7ad7211). Implementation and execution: maintainer A, within
the cap below. Status: frozen before launch; the record of what ran is the run directories and
this file.

## Why

Pair (4,4) (initialization seed 4, data-order seed 4), condition B, one column, v1: the CPU
run with the reference scan (halting-onecol-f4c9279, f4c9279) escaped at 16000 updates under the
crossed-seed endpoint; the GPU run with the fused scan (halting-cross-2bbffdc) had not escaped by
36000. Between them changed the device arithmetic, the scan's precision policy and the source
revision, so the scan explanation is open (maintainer B). This comparison changes ONLY the scan dispatch
and precision policy on one device, from one code revision. A difference supports a scan-policy
effect for this pair; agreement leaves the CPU/GPU contrast unexplained. Neither is a population
claim.

## Prerequisite checks (done before this file was frozen)

- The verifier of the backend checks fails closed (compare_fixed.py v2, 11 red/green fixture
  cases, maintainer B's four among them; the real arms still read with no flag).
- Capture: the reference scan cannot be captured as a CUDA graph (`torch.inverse` in
  `delta_scan_chunked` is not permitted while a stream captures: cudaErrorStreamCaptureUnsupported).
  So, per the decision, BOTH arms train eagerly; the scan is not confounded with graph mode.

## The recipe

| Item | Value |
|---|---|
| Model | configs/halting-stage1-c1.yaml (C1 / T8 / K4), unchanged |
| Condition, pair | B (`fixed`), `--seed 4 --data-seed 4`, fresh starts |
| Batch, updates | 32 x 88, 36000; no replacement, extension or best checkpoint |
| Learning rate | the existing 100-step warmup, then constant |
| Mix | `--mix v1` |
| Arm R | `--device cuda --scan reference`, eager |
| Arm F | `--device cuda --scan fused`, eager (a fresh run; not the captured 2bbffdc run) |
| Shared | float32 parameters, activations and states, no autocast, matmul TF32 off, float32 matmul precision "highest", the capturable AdamW the driver builds on CUDA; one pinned clean worktree, one venv (torch, triton, fla-core versions in each result's `backend`) |
| Identity | `initial_state_digest` (the weights before the first step) and `consumed_digest` (a sha256 chain over every consumed batch tensor: tokens, targets, valid, doc_start, segment, loss_mask, pos), logged at every logged step and in result.json; equal in both arms or the comparison is void |
| Evaluation | CPU copy with the reference scan for probes, static checks and the final evaluation, as before |
| Endpoint | `--static-every 1000 --static-batches 8`: the first of two consecutive checks with hop2 (best static depth) above 50 % on the first 256 validation episodes (vesper.halting_escape), censored at 36000; full terminal 1024-episode profiles; snapshots every 1000 |

## Rules

- Both curves go back to maintainer B before any further run is proposed.
- No pooling with older runs; no final split.
- The formal stage-1 gates and the independent source reviews remain open.
