# Retrieval binding and the stage-1 task: proposal

2026-10-15, maintainer A. Revision 2, after maintainer B's review of d9eada0
(<evidence>/reviews/retrieval-binding-d9eada0/). Maintainer B accepted the contract at
1cc32a6 (<evidence>/reviews/retrieval-binding-1cc32a6/). DECIDED by the project owner, about
21:00Z: option (ii), the kernel-4 convolution on the projected key ("1. is a good recommendation.
i will authorize that."). The task of section 5 is built (1678173 and its corrections); the
architecture of section 4 is next.

## 1. The question

Stage 1 of the halting experiment (docs/reviews/2026-10-15-halting-experiment-plan.md) asks
whether adaptive depth helps where depth is needed. It can only ask that on a task the model
learns AND on which later passes help. As built, the model does not learn the task: after 16000
steps in three settings (the stage-1 config, lr 3e-3, d 64) the copy questions are at 100 % and
the hop and register-read questions at chance (deployed readout: hops 0.12-0.20, reads
0.05-0.22; chance for a hop is 1/7, maintainer B). By the agreed rule the task goes to redesign. The
diagnostics below show that the model, not only the task, made the lookups hard to learn.

## 2. Evidence

All scratch, one seed each (model seed 0), condition `fixed`, d 32 unless noted, code pinned at
9044238. Export with scripts, launch lines, checkpoints and checksums:
<evidence>/learnability-diagnostics-9044238/ (maintainer B verified all 41 files and
reloaded three checkpoints; its README has the erratum on mixed evaluation sets). Readouts are
the dense forward's per-pass accuracy: p1 is pass 1; max is the maximum over the 8 passes, given
only where it differs from p1, with its pass in parentheses where stated. These are sampled
points of single runs, not crossing times.

Dense hop1 (12 `(HOP_1, a)` queries per sequence after the gap; 3000 steps):

| key | 500 | 1000 | 1500 | 2000 | 2999 |
|---|---|---|---|---|---|
| as built | | | | | p1 0.141, max 0.145 |
| recurrent read removed (w_o zero, frozen) | | | | | p1 0.112, max 0.112 |
| pure shift `W_k s_{t-1}` | p1 0.702 | p1 0.991, max 0.995 | | | p1 0.993 |
| conv kernel 4 on the projected key | | p1 0.743, max 0.772 | p1 0.993 | p1 0.997 | p1 1.000 |
| conv kernel 4 on q, k, v, each with SiLU | | p1 0.102, max 0.124 | p1 0.626, max 0.652 | p1 0.930, max 0.940 | p1 0.990, max 0.991 |

Hop2 under the shift with a depth curriculum (k_max 1 for steps 0-999, then 2), held-out set
with k <= 2 (`acc_in`; out of distribution during steps 0-999, when HOP_2 is untrained):

| arm | 3000 | 4000 | 4999/5000 | 6000 |
|---|---|---|---|---|
| plain, d 32 | p1 0.356, max 0.361 | p1 0.899, max 0.910 | p1 0.970, max 0.973 | p1 0.995 |
| loss on the last pass only | p1 0.019, max 0.152 (pass 8) | p1 0.024, max 0.261 (7) | p1 0.027, max 0.356 (7) | |
| writes only at definitions (beta 0 from 24) | p1 0.092, max 0.109 | p1 0.122, max 0.155 | p1 0.149, max 0.174 | |
| plain, d 64 | p1 0.122, max 0.133 | p1 0.375, max 0.380 | p1 0.573, max 0.582 | |

Hop1 in the same arms stays at p1 0.97-1.00 except lastpass (p1 0.25 / 0.41 / 0.50, max 0.99).
The k-curriculum to k_max 4 (1000 steps per depth, mixed held-out set, in distribution from step
3000) reached hop2 p1 0.875 / max 0.902 (pass 6) at 4999, hop3/4 at chance. On the stage-1 task
itself, the shift alone learned copies (1.00) but not hops or reads in 4000 steps, and a 1000-step
hop1 warm-up transferred hop1 (0.84) but not hop2-4 or reads in 1500 more.

Maintainer B's frozen-checkpoint control: the shifted hop1 checkpoint, chained by an external controller
(decode, re-embed, the same prefix, cold full prefills), answered 2041 / 2034 / 2027 / 2020 of
2048 at hops 1-4 (4 hops 98.63 %, graph-bootstrap 95 % interval 97.66-99.61 %); this uses machinery
outside the model and repeated prefix work.

## 3. What the evidence supports, and what it does not

Supported (one seed, this size and budget): (a) the as-built model did not learn single lookups
across the gap in 3000 steps; three ways of letting the recurrent key use preceding tokens each
did, at the sampled points above. (b) The learned lookup can be chained by an external controller
at 98.63 % over four hops. (c) Inside the model one composition step was learned after a plateau
(about 3000 steps at k_max 2). (d) In the plain arms hop2 is answered at pass 1, inside one pass of
two columns (maintainer B): hop count k is not a depth need. Under last-pass-only loss the same model's
early readouts are much less accurate (hop1 p1 0.50 to max 0.99); that is earlier READOUT accuracy
under all-pass loss, not a measurement of where the work is done (last-pass loss does not
supervise early outputs).

Not established: why the as-built key fails. The direct route is architectural reasoning: a
definition `(c_i, c_{i+1})` wants key `c_i` and value `c_{i+1}`, which sit at two positions, and
a delta-rule write takes both from one position; a key that sees the predecessor stores the pair
in one recurrent sublayer. Without it, other routes exist: attention can carry `c_i` forward, and
maintainer B measured a further path in the native model with both attention outputs zero (column 2's
key at token 1 depends on token 0 through column 1's recurrence, gradient norm 0.206, exactly 0
when that recurrent output is also removed). That weak initial gradients through these routes are
why they were not learned is a hypothesis; no gradient strength was measured. The three successful
variants also differ in more than the predecessor access (the q/v convolutions, SiLU, learned
taps and their initialization), so they justify choosing a candidate, not a unique cause.

The two training controls are failed remedies in this recipe, not settled mechanisms: last-pass
loss changes learning from step 0 and does not exclude local interference between pass targets;
definition-only writes remove gap and query writes together (maintainer B's frozen controls separated
them: query-only removal changed little, removing gap writes as well hurt). Keeping all-pass loss
and ordinary writes is the supported next choice.

Prior art, checked at the source: DeltaNet (arXiv 2406.06484) adds short causal convolutions after
the q/k/v projections (sec 3.4) and reports MQAR solved without convolution (sec 4.1); the
convolution helped recall-heavy tasks at 340M (FDA 4.5 -> 12.8). It supports the convolution as a
studied choice; it does not show this key-only variant is necessary.

## 4. Proposal A (architecture, the project owner decides): a causal convolution on the recurrent key

The candidate is the simplest successful form that keeps the current token's key available: a
depthwise causal convolution of width K = 4 on the PROJECTED key, as in the `convk` diagnostic.
Keeping the current token is a design preference (the learned current tap can become zero), not a
demonstrated requirement; the pure shift learned fastest at this point and stays the documented
alternative.

### 4.1 The operator

Per column, pass p and token t, with `active[t,p]` true while t has not halted before p and
`exit(t)` its last active pass:

```
s[t,p]        = norm_rec(input[t,p])                      (unchanged)
z[t,p]        = W_k s[t,p]                                [b, n, d], before head reshape
z_eff[t,p]    = z[t,p]               if active[t,p]
                z[t, exit(t)]        otherwise            (frozen, unfiltered)
u[t,p,j]      = sum over r = 0..K-1 of a[r,j] * z_eff[t-r,p,j]
k_fresh[t,p]  = normalize_keys(reshape_heads(u[t,p]))     (normalization AFTER filtering)
k_commit[t,p] = k_fresh[t,p]         if active[t,p]
                rec_k[t, exit(t)]    otherwise            (its OWN frozen key, existing _merge)
```

`r = 0` is the current token. In torch's `Conv1d` with left padding K-1 the storage order is
reversed: `a[r, j] = weight[j, 0, K - 1 - r]`; a test pins it with an asymmetric kernel. Depthwise
over the d channels, no bias, no activation. Projection and filtering do not commute in general
(maintainer B: [2, 2] versus [3, 2]), so the order above is the contract; any kernel must match it.
q, v and beta keep their existing contracts, including modulation (the diagnostic's override
ignored modulation; production must not). Parameters: one `conv_k` weight `[d, 1, K]` per column,
registered only on the recurrent sublayer that uses it (the diagnostic also registered unused
`conv_q`/`conv_v`); its weight-decay role is UNDECAYED (a filter, like a gate), declared in
`train.UNDECAYED`. Initialization as the diagnostic: uniform(-1/sqrt(K), 1/sqrt(K)) from the
model's generator. Config: `recurrence.key_conv` (integer; 0 is the as-built key, kept as the
ablation the evidence compares against; 2..8 the width; anything else refused by `validate`).
Implementation notes (maintainer B): `key_conv: 0` draws nothing from the model generator, so the
baseline reproduces exactly; the uniform initialization from the model generator matches the
diagnostic's distribution, not its separate generator's weights, so no identical trajectory is
promised; the cold cache's dtype and its transition under autocast belong in the typed state and
its tests.

Two merges, in this order: the predecessor SOURCE is selected (`z_eff`) BEFORE an active
successor is filtered; a halted token's OWN key is its frozen `rec_k` AFTER. The wrong order
filters a fresh inactive value (maintainer B's literal case: 1002 instead of 32; gradient 10 to the exit
input, 0 to the fresh inactive one). With K = 4 the three predecessors may have halted at three
different passes; `z_eff` is per token. Halted positions still advance every deeper pass's
history with their frozen `z`.

### 4.2 The cache

The carried history is the projected, unfiltered `z_eff` of the last K-1 tokens, per column and
pass: `[b, K-1, d]`, oldest to newest, in `ColumnState`. Not the raw `s`: re-projecting raw
history under updated weights gives different keys (maintainer B: W 2 -> 3, input 5: 15 versus 10), and
the two policies route W_k's gradient differently. Choosing projected history IS a training
contract: across a chunk boundary the history carries the weights it was projected with and, like
every carried state, is detached with the states (truncated backpropagation at the chunk).

- Cost: `C * T * b * (K - 1) * d` elements of state; `C * K * d` parameters; the transient
  per-token `z_eff` inside a chunk.
- Boundaries: an empty chunk leaves the history unchanged; a chunk shorter than K-1 keeps the
  newest K-1 of (old history, chunk); the stored tail owns its storage (a copy, never a view that
  keeps the whole chunk alive).
- Documents: inherits the recurrent state's policy, never reset within a stream
  (`train.step`); a reset of the whole stream (cold `init_column_state`) clears it too; no
  `doc_start`-only reset in one component.
- One typed traversal of `ColumnState` for detach, record and restore (today four places list
  the fields by hand), with restore validating dtype, device, shape and the pass count; the
  checkpoint schema number changes.

### 4.3 Tests (each shown red on a mutant first)

1. Literal arithmetic: the operator against a hand-written loop over positions, with an
   asymmetric kernel (catches reversed taps) and maintainer B's two-token merge case (32, not 1002).
2. Streaming: prefill equals token-by-token decode equals carried chunks of every size, including
   empty and shorter-than-K-1 chunks, in outputs, states and the cache.
3. Mixed halting: K = 4 with the three predecessors halting at different passes, every execution
   order (the copy-forward order-invariance test, extended).
4. Gradients, two separate tests: chunked forwards WITHOUT detachment equal the full prefill's
   gradients; with the trainer's detachment, gradients to earlier chunks are exactly zero.
5. Recomputation (`--recompute`) equals the kept-activation path in values and gradients.
6. Resume: an optimizer update, a checkpoint, a restore and the next steps equal the
   uninterrupted run; restore refuses a wrong shape, dtype or pass count.
7. `key_conv: 0` reproduces the current model exactly (outputs, states, gradients).

## 5. Proposal B (task, maintainer B reviews): a frozen recipe before any run

### 5.1 Layout and mix

- 88 tokens per sequence (a multiple of the block 4). Definitions 0..23 (the 8-cycle, as now),
  gap 24..31 (window 8 stays at most the gap), register 32..33, then 20 hop queries and 3 copy
  items in random order (52 positions), a fill and READ at 87 (54 positions after the register).
  Per sequence: 20 hop labels, 3 copy labels, 1 read label.
- Hop depths per sequence: one query of each k in 1..4 always (every marker in every sequence),
  the other 16 drawn from a schedule over the mix: steps 0-999 k = 1 with p 0.7 and k = 2..4 with
  p 0.1 each; steps 1000-2999 p (0.4, 0.3, 0.15, 0.15); from 3000 uniform.
- Register: `k_reg = 1 + (s mod 4)` for sequence s of each batch (every REG marker in every batch
  of at least 4), node uniform.
- Budget: batch 32, 2816 tokens and 768 labels per step; the learnability probe is bounded at
  12000 steps, no early stop. Every 1000 updates a weights-only snapshot
  (`weights/after-NNNNN.pt`): diagnostics for learning curves and interventions, NOT recovery
  checkpoints (no optimizer, generator or progress state; an interrupted run starts again). The
  gates use only the final result after 12000 updates.

### 5.2 Splits and gates

- The diagnostics above are calibration only. The probe evaluates on `generator_for("validation",
  0)`; the stage-1 comparison reports on `generator_for("final", 0)`, unread until then.
Measurement for the gates (maintainer B's re-review of 1cc32a6, adopted as written):

- Probe condition B (fixed uniform all-pass loss), model and training seeds 0 and 1, the proposed
  d 32 / C 2 / T 8 / K 4 configuration and the existing optimizer recipe. Gates use the
  checkpoints AFTER 12000 optimizer updates; earlier checkpoints are learning curves, not
  candidates for choosing the best stopping point.
- 1024 fixed validation episodes in 32 batches of 32, generated once from
  `generator_for("validation", 0)` with the terminal uniform hop mix, the four mandatory hop
  queries per sequence, one copy item of each back distance 1..3 and the balanced register rule;
  the same episodes for every model, depth and seed. Every class denominator is recorded: 256
  reads per class, at least 1024 queries per hop class, 24576 scored labels in all.
- For each j = 1..8, ALL tokens forced to depth j with a constant `TraceRule`, emitting that
  pass's own output; `A(seed, class, j)` from its correct / total counts. The dense pass-j
  readout must agree with it, verified in the new implementation. This is a fixed-depth profile,
  not a live CDF policy.
- Gate 1, learnability: for every hop and read class and BOTH seeds, `max_j A > 0.50`.
- Gate 2, deeper work helps: one and the SAME class c with
  `max_j A(seed, c, j) - A(seed, c, 1) >= 0.05` for each seed. All counts and pass curves kept;
  the best j may differ between seeds.
- The dense / matched-trace / live deployment-gap comparison and the fixed-reader / register-depth
  grid of the original plan stay separate; the fixed-depth gate does not establish them. The final
  generator stays unopened during calibration; the comparison budget is frozen before the 12 runs;
  the B probe runs are calibration and are NOT reused as stage-1 runs.

If Gate 2 fails (every class answered as well at pass 1, as hop2 was in the plain arm), the task
cannot show a benefit for halting to allocate. The next lever is to try longer chains and reapply
both gates empirically: k above the number of columns does not prove one pass cannot solve it, and
an 8-cycle wraps at k = 8 (a return to the query node), so k is not monotonically harder. That is
decided before the 12 runs, not during them.
- Only then the 12 runs (4 conditions x seeds 0-2), adaptive, fixed and random at matched logical
  work. The per-class curve is an empirical readout profile of those weights, not an architectural
  minimum; actual latency and FLOPs stay separate from the reference loop's pass counts.
- The register question stays: the reader learned first, its depth held fixed, then the register
  depth intervened on alone (the existing grid). A convolution adds a local propagation path, so
  receptive-field statements in the task text are rechecked once it is chosen.

## 6. Sequence

The project owner's decision on section 4 first. Then: the operator, cache and config with the tests of 4.3 and
a full gate; the dense hop1 diagnostic again through the real code path (not a monkeypatch); the
task of section 5 with its own tests; the bounded probe on two seeds; the gates; the 12 runs.
