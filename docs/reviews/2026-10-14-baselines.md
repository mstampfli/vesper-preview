# Review: the dense baselines and what "parameter-matched" means

Date: 2026-10-14; revised the same night. Author: maintainer A. Reviewer: maintainer B, who requested changes
on the first version (review in the channel, 23:14Z): it copied the spec's `17 d²` column
coefficient instead of summing the shapes listed under it, left the tied logits projection out
of the FLOPs table, and called the `(32,1)` control compute-matched against its own table.
This revision derives every number from the spec's tensor shapes in exact arithmetic;
`tests/test_baselines_arithmetic.py` derives the same figures from `configs/vesper-s.yaml`
against golden numbers, so a changed config or a changed formula fails it. It does not read
this document or the spec: prose can still drift, and review is what catches that.

Decision: delegated by the project owner to maintainer A on 2026-10-14. Decided: Baseline A is 7 layers with
`ffn_mult = 3`; the ablation grid is a fixed-maximum-column-application comparison, not equal
compute; the `(32,1)` control is named Baseline B and is not compute-matched; the spec carries
a FLOPs estimate with its assumptions and no "measured" ratio until one is measured.
Status: applied to docs/spec.md in the same commit; the project owner may override.

## Why this document exists

The spec promises two comparisons (docs/spec.md, Evaluation): held-out perplexity against a
dense transformer "with the same parameter count", and the same at equal training FLOPs. The
ablation grid in Open questions lists `(32,1)` as "the plain-transformer control". In
conversation on 2026-10-14 I described the parameter-matched baseline as "32 columns, one
pass, window equal to the chunk, recurrence off". That was wrong: a 32-layer dense model of
the same width has nearly four times Vesper-S's parameters. The two controls are different
models and the spec did not say so. This document writes both down with their numbers.

## Vesper-S, counted from the spec's shapes

`d = 768`, `h = 12`, `dk = 64` (so `h·dk = d`), `C = 4`, `T_max = 8`, `V = 32,768`, tied
embeddings, window 512, GQA with `h/4` key-value heads, FFN multiplier 4. `d² = 589,824`.

| Tensor, per column | Shape | Parameters | In `d²` |
|---|---|---|---|
| Recurrent core `W_q, W_k, W_v, W_o` | four `d × d` | 2,359,296 | 4 |
| Attention `W_q, W_o` | two `d × d` | 1,179,648 | 2 |
| Attention `W_k, W_v` (GQA, `h/4` heads) | two `d × d/4` | 294,912 | 1/2 |
| Retrieval `W_r, W_rk, W_rv` | three `d × dk` | 147,456 | 1/4 |
| Retrieval `W_ro` | `dk × d` | 49,152 | 1/12 |
| FFN `W_1, W_2` | `4d × d`, `d × 4d` | 4,718,592 | 8 |
| Dominant matrices | | 8,749,056 | 89/6 ≈ 14.833 |
| `W_β` (write strength) | `d × h` | 9,216 | |
| `W_g` (retrieval gate) | `d × 1` | 768 | |
| Four pre-RMSNorm scales | `4 × d` | 3,072 | |
| Column total | | 8,762,112 | |

| Shared | Shape | Parameters |
|---|---|---|
| Tied embedding | `V × d` | 25,165,824 |
| Iteration conditioning `u_t` | `T_max × d` | 6,144 |
| Halting head | `2d + 1` (direction, norm gain, bias) | 1,537 |
| Final RMSNorm | `d` | 768 |

Total: `4 × 8,762,112 + 25,174,273 = 60,222,721` (dominant matrices only: 60,162,048).
Amended 2026-10-15: the halting head reads a normalized state since the halting diagnosis
(docs/reviews/2026-10-15-halting-diagnosis.md), 768 more parameters; the count was 60,221,953
before, and the comparisons below are recomputed against the new count. The
spec's `≈ 17 d²` and "~65 M" were wrong; the general column coefficient is
`(14.5 + 4·dk/d) d²` for the dominant matrices: `89/6` at S, `59/4` at M (`dk/d = 1/16`),
`44/3` at L (`dk/d = 1/24`), giving ~60 M, ~259 M and ~1.21 B with their embeddings. Not
in these counts because the spec gives them no shape yet: the local-mode predictive heads
(policy: local-only parameters are excluded from the e2e comparison and listed separately once
they exist) and anything the ledger holds (an index, not model parameters). When the modules
exist, `count_parameters` counts the constructed tensors and the exact figures above become
the test; a tolerance would hide a missing tensor.

## Baseline A: parameter-matched dense (the milestone comparison)

A standard pre-norm decoder: full causal attention over the whole 4,096-token chunk, the same
GQA ratio, the same relu² FFN, the same tied embeddings and final norm, no recurrence, no
retrieval, no halting, one pass. One layer costs `2.5 d² + 2 × ffn_mult × d²` plus two norm
scales.

| Option | Layers | ffn_mult | Layer matrices | Total (exact) | vs Vesper-S 60,222,721 |
|---|---|---|---|---|---|
| A (decided) | 7 | 3 | `8.5 d²` | 60,271,872 | +0.082% |
| | 8 | 2.5 | `7.5 d²` | 60,568,320 | +0.574% |
| | 6 | 4 | `10.5 d²` | 62,334,720 | +3.5% |
| | 7 | 3.5 | `9.5 d²` | 64,400,640 | +6.9% |
| first version's choice | 6 | 4.5 | `11.5 d²` | 65,873,664 | +9.4% |

Decision: A, 7 layers with hidden width `3d = 2,304`. The mismatch is below 0.1% and still
in the baseline's favour. The FFN share differs from Vesper's multiplier of 4; the comparison
is by parameter count, and `ffn_mult 3` is the integer multiplier that matches.

## Baseline B: the `(32,1)` control, not compute-matched

The same block, 32 layers, `ffn_mult = 4`: `32 × 10.5 d² = 336 d²` plus norms, 198,230,016
layer parameters, 223,396,608 total. This is what the grid's `(32,1)` denotes: 32 distinct
columns applied once, recurrence and retrieval removed, attention over the whole chunk. It
has 3.7x Vesper-S's parameters, and by the table below its forward work is 1440 d² under the
rectangular attention convention or 1099 d² under causal pairs against Vesper-S's 1136 d² at
`T_max`: not matched, and on which side depends on the convention. It is a named reference
point, not a matched control of any kind; an equal-training-FLOPs control is selected from
measured work (below).

## FLOPs per token, forward, estimate

Assumptions, stated so the estimate can be checked and replaced:

- 2 FLOPs per multiply-add; every weight is touched once per application.
- Attention: two conventions, both reported. Rectangular: every query against the full
  context, `4 × context × d` per layer-application. Causal pairs: the full-chunk baseline
  attends to `(n + 1)/2` keys on average, taken as `n/2` in the rounded table, half the
  rectangular figure; Vesper's window is full for every token past the first 512 of a stream,
  so its figure is the same under both.
- Recurrent state (pass-state review, section D): an active column-pass costs `3 h dk²`
  multiply-adds per token (read what is stored under `k`, update, read under `q`) = `d²/2`;
  a halted token's copy-forward update costs `2 h dk²` = `d²/3` (no read under `q`). The
  blocked training form (block 64 = `dk`) performs nine products of size `L·dk²` or `L²·dk`
  per block per head, `1.5 d²` per column-pass per token, for every pass of every token. That
  is the matmul subtotal of the training scan; the inverse of the block's unit lower
  triangular matrix is additional work, listed under the exclusions.
- The tied logits projection is a matmul whether or not its weights are shared: `2 V d` =
  `256/3 d² ≈ 85.3 d²` once per token, on both sides.
- Not included: retrieval attention over 32 entries (gated, rate unmeasured), ledger search,
  the halting head, norms, the triangular inverse of the blocked scan, the backward pass,
  recomputation, and the observed distributions of passes and gate openings. Those make this
  an estimate.

| Model | Weights | Attention | Recurrent state | Logits | Total |
|---|---|---|---|---|---|
| Vesper-S at `T_max = 8` | `32 × 89/3` = 949.3 | 85.3 | 16.0 (training form 48.0) | 85.3 | 1136 (1168) |
| Vesper-S at mean 4 passes | `16 × 89/3` = 474.7 | 42.7 | 13.3 = 8.0 active + 5.3 copied (training form 48.0) | 85.3 | 616 (651) |
| Baseline A (7 layers, 4,096 context) | `7 × 17` = 119.0 | 149.3 rectangular, 74.7 causal | 0 | 85.3 | 353.7, 279.0 |
| Baseline B (32 layers, 4,096 context) | 672.0 | 682.7 rectangular, 341.3 causal | 0 | 85.3 | 1440, 1099 |

All figures in units of `d²` per token. Vesper-S at the target mean of four passes against
Baseline A: 1.74x under the rectangular convention, 2.21x under causal pairs. The spec's
"3-5x" is replaced by this range with its assumptions; the first version's "2.2x" counted
neither the logits nor the state and used a wrong column coefficient, and it was not
measured. "Measured" is reserved for training FLOPs counted at the observed pass and gate
distributions. The copied inactive-token updates are the 5.3 d² line: halting removes
projections, not state updates. The equal-FLOPs row of the evaluation table is filled after
that measurement by choosing the dense depth whose measured work matches.

## The grid: fixed maximum column applications, not equal compute

Open questions asks whether `(4,8)` beats `(8,4)` "at equal parameters" and lists the grid
`{(2,16), (4,8), (8,4), (16,2)}`. At fixed `d` these points hold `C × T_max = 32`, the
maximum number of column applications per token; realized work is `C × mean(T)` and varies
with how halting behaves at each point, and their parameters range from `2 × 89/6 d²` to
`16 × 89/6 d²` plus the embedding. Two honest readings:

1. Fixed maximum column applications at fixed width: keep `d = 768`, report realized
   `C × mean(T)` and parameters per point. Cheapest, and the question becomes how to split
   a fixed application budget between distinct columns and repeated passes.
2. Equal parameters: shrink `d` with `C` so that `C × (14.5 d² + 4 d·dk) + V d` stays at
   the S total (the embedding term must be in the equation; holding only the column term
   constant does not hold parameters constant when the width changes). With `dk = 64`:
   `d ≈ 977, 768, 586, 436` for `C = 2, 4, 8, 16` and `d ≈ 319` for `(32,1)`. These are
   continuous roots, not runnable configurations: with `dk = 64` and `kv_groups = 4` a legal
   width is a multiple of 256 (`heads` must divide by `kv_groups`), so the nearest legal
   widths are 1024, 768, 512, 512 and 256, and the parameter match is lost in the rounding.
   The alternative is infeasible without choosing head and GQA constraints per point.

Decision: reading 1, with the spec's wording changed from "at equal parameters" to "at a
fixed maximum of 32 column applications per token, realized work and parameters reported
per point", and the `(32,1)` control described as Baseline B.

## Implementation: one code path, two configs

The baseline is a configuration of the same model code, never a second implementation
(CONTRIBUTING, one system per job). Needed in `vesper/config.py` and `configs/`:

- `model.recurrence: bool` and `model.retrieval: bool`; `t_max = 1` implies no halting head.
- `window` may equal `training.chunk_tokens` (full-chunk attention); `validate` must allow it.
- `configs/baseline-s.yaml`: `columns 7, t_max 1, ffn_mult 3, window 4096, recurrence
  false, retrieval false`, everything else as `vesper-s.yaml`.
- Registry row `count_parameters(model) -> int`, counting tied tensors once and listing
  disabled components, with a test asserting Vesper-S = 60,222,721 and Baseline A =
  60,271,872 exactly. Those are the checks that fail if this document, or a module's
  tensor list, is wrong.

## Checks that reject the wrong claim (from the review)

1. Sum the named matrix shapes independently and compare with the claimed coefficient: the
   list gives `89/6`, which rejects 17.
2. Once modules exist, count constructed parameters by component, tied tensors once,
   disabled components identified; check Vesper and the dense candidate against each other
   and against the table.
3. Include one `d × V` projection per token in any declared forward budget; omitting it must
   change the expected total.
4. Measure training work at the observed iteration and gate distributions; neither a fixed
   `C × T_max` nor a passing parameter test substitutes for it.

## Decisions (made under the project owner's delegation; he may override)

1. Baseline A: 7 layers, `ffn_mult 3`, parameter-matched to within 0.1%.
2. The grid is a fixed-maximum-column-application comparison; `(32,1)` is Baseline B, a
   named reference that is neither parameter- nor compute-matched.
3. The spec states the forward estimate 1.7x to 2.2x with its assumptions and no measured
   ratio until one exists; the sizes table and the column coefficient are corrected.
