# Proposal: the training loop, first version (e2e wake training without the ledger)

Date: 2026-10-15. Author: maintainer A (maintainer). Status: proposed; the registry rows below are added as
`specified` with maintainer A as owner, code follows maintainer B's review or, failing a reply by the time the
data path is frozen, follows these rows as written (the project owner may override either way). Scope: the
recipe's phases 1 and 2 (spec: Training recipe, Phases) in `e2e` mode without retrieval, the
ledger or dusk. Phase 1 prescribes exactly that; phase 2 prescribes retrieval over the current
document and index rebuilds, so a run of this version through phase 2 is the no-retrieval
ablation, labeled as such in its log and receipts, not the recipe (maintainer B's review). The ledger,
retrieval, dusk and quantization-aware training plug into the schedule when their modules land;
the schedule already names their switches so the loop does not change shape then.

## Why now

The data path ends in a frozen tokenizer tonight; the model's forward and the wake objective are
reviewed. The next thing that can fail is a real optimizer step on the box, and the loop is where
the recipe's phase logic lives. One pure schedule function, one step function and one checkpoint
format, each with a test that would reject a wrong one, is the smallest version worth reviewing.

## Decisions proposed

1. **The schedule is a pure function of tokens seen and the step index.** `phase_schedule`
   returns a frozen `Phase` and nothing else reads the phase boundaries. `t_max` anneals linearly
   from `training.t_max_anneal_from` to `model.t_max` over the warm phase, rounded to the nearest
   integer, never decreasing. The halting head is frozen in warm at `p = 1/t_max` by construction
   (its direction zero, its bias `logit(1/t_max)`, no updates); it is unfrozen at the loop phase
   with those values as its start, so the first unfrozen step begins where warm ended. `lambda_h`
   is 0 in warm and ramps linearly to `learning.lambda_h` across the loop phase. The learning rate
   warms up linearly over `optimizer.warmup_steps` and then follows one cosine from the peak to
   `peak * final_lr_fraction` at the last step; the settle phase is the tail of that cosine, not a
   second schedule. The retrieval, ledger-write, dusk-strength and QAT switches are fields of the
   phase with the recipe's values; this version reads none of them.
2. **Weight decay by role, named by parameter.** `optimizer.weight_decay` on the linear maps:
   the embedding (`embedding.weight`), the recurrent and attention projections (`rec.w_q`,
   `rec.w_k`, `rec.w_v`, `rec.w_o`, `attn.w_q`, `attn.w_k`, `attn.w_v`, `attn.w_o`) and the
   feed-forward (`ffn.w1`, `ffn.w2`); none on the gates (`rec.w_beta`, the write-strength
   matrix, is a gate although it has two dimensions; `halting.w_h`, `halting.b_h`), the norms
   (`norm_*.weight`) and the conditioning vectors `loop.u`. The rule is a name test against this
   list, so a parameter with a new name fails loudly instead of being decayed by shape (maintainer B's
   review: an ndim rule would have decayed `w_beta` against the spec's "none on gates"). AdamW
   with the configured betas; the learning rate is written into the groups every step.
3. **Truncated backpropagation at the chunk boundary, state carried.** The carried
   `ColumnState`s are detached after every optimizer step and never reset: not between chunks,
   not between documents (the state is the model's persistent memory; the spec says what trains
   the persistence is exactly this). The data gives the rotary positions: `Chunk` gains
   `pos int64[b, n]`, the slot's stream position of every input, consecutive within the chunk
   and continuing across chunks and documents (the attention requires consecutive positions
   inside a chunk and its scores depend on differences only; a restart at every document would
   break that contract and buy nothing), and `doc_ids`, the documents that gave inputs to each
   slot in order (the pointer the dusk replay log will need). `Stream` is the owner of both
   facts and counts `documents_taken` for the resume rule below.
4. **Micro-batches are independent streams, not slices of one.** A step processes
   `training.batch_chunks` slots as `G = batch_chunks / micro_chunks` groups; each group is its own
   `Stream` over the one shared document iterator (the first group that needs a document takes
   the next one, so the assignment is deterministic), with its own carried states. The step's
   loss is the token-weighted mean over the whole batch: every group's chunks are drawn first,
   their loss positions are counted, and each group's summed per-token loss is divided by the
   batch's total count before its backward pass, so the gradient equals the one an unsplit batch
   would give (maintainer B's fixture: groups with one and three loss tokens give -1.5 under the global
   mean and -1 under equal group means, and the global mean is the right one); the ponder term
   is averaged the same way. Gradients accumulate across the groups; one clip at
   `optimizer.grad_clip`; one optimizer step. `micro_chunks` is a run parameter of the
   command line (it depends on the accelerator), not a model constant; `batch_chunks` must be a
   multiple of it.
5. **Mixed precision as the model is already tested:** bf16 autocast on CUDA for the forward and
   backward, the recurrent state float32 (tests/test_model.py already runs Vesper-S this way).
6. **The training mix is by tokens.** `corpus.iter_mixed_documents` is `iter_mixed_texts` with
   the unit changed: the components' shares are relative (web 0.80, code 0.15, encyclopedia 0.04,
   books 0.01), there is no total budget (training runs to `training.total_tokens`), at each
   document the component with the smallest `tokens_emitted / share` yields its next document,
   tokenized, and a component that runs out leaves the others running (logged, visible in the
   step log as the realized shares). The same `Finalization` (one `finalize` over all the
   components, pinned by input hashes) and `Split` apply through `select_documents`. Documents
   are whole; nothing moves inside one.
7. **Checkpoints are a directory with a manifest.** `step-<n>/` holds `model.pt`,
   `optimizer.pt`, `states.pt` (the carried, detached `H` and KV of every group and every
   stream's packing state), `train.json` (step, tokens seen, documents taken, the phase, the
   realized shares, and the identity of what the run reads: the sha256 of the tokenizer file,
   the config file and the finalization receipt) and `manifest.json` with the sha256 of each
   file. Restore refuses a checkpoint whose tokenizer, config or finalization identity differs
   from the run's. Restore verifies every hash before `torch.load(weights_only=True)` and refuses on a
   mismatch (docs/threat-model.md). Resume restores the streams, not just the iterator: a
   `Stream` exposes its packing state (per slot the queued documents with their cursor, the
   pending count, the segment and the stream position; the exhaustion flag; `documents_taken`)
   as `Stream.state()` and is rebuilt from it with `Stream.restore(state, documents)`; the
   document iterator is replayed to `documents_taken` (tokenizing and counting the skipped
   documents so the mixer's choices repeat), and the next chunk of the resumed run equals the
   next chunk of the uninterrupted one, positions included. The replay costs the tokenization
   of the consumed prefix, which grows with training progress: it is the first version's
   recovery and the test oracle, not the production plan. The bounded-cost resume persists the
   readers' cursors (per component the `(file, row group)` pair index and the row index in the
   permuted order, the mixer's active set and token balances) and continues from them; it is
   the next row of `corpus` and lands before any run longer than a smoke. Maintainer B's counterexample
   (review of becadeb): a new stream after `skip` alone emits the next document at position 0
   instead of
   the slot's queued tokens at position 3. Checkpoints land every
   `total_tokens / training.checkpoints_per_run` tokens (20: the spec's "every 5%") and at the
   end; the held-out perplexity and the iteration histogram run at each.
8. **The step log is JSONL, one line per step:** step, tokens seen, phase, `t_max`, lr, loss
   and its parts, tokens with loss, mean iterations and the iteration histogram, gradient norm,
   the mixer's hand-out counts per component (`shares`) and the inputs the model saw per source
   over the run (`consumed`, from the chunks' spans as the streams emit them; the two differ by
   what sits in the slots' queues), seconds, the accelerator's peak allocated bytes during the
   step. The checkpoint evaluations append their own lines.

## What this version does not do, on purpose

No retrieval, no ledger, no dusk, no QAT, no local mode: their switches exist in `Phase`, their
modules do not. No chunk-span log for dusk replay yet beyond `doc_ids` in the chunk (the log
format is the dusk's to define). No multi-GPU. The full run (10 B tokens, 20,000 steps of 0.5 M
tokens) is the project owner's decision to spend; this version is proven by a smoke run of a few hundred
steps on the box with the loss falling and the phase switches observed in the log.

## Registry rows added (specified, owner maintainer A)

- data: `Chunk.pos`, `Chunk.doc_ids`.
- corpus: `iter_mixed_documents(components, seed, tokenizer, held_out_fraction, skip=0)`.
- model: `wake_objective(logits, chunk, result, cfg, weights=None)`: the phase's weights when given.
- train: `phase_schedule(tokens_seen, step, cfg)`, `optimizer(model, cfg)`, `step(...)`,
  `checkpoint(...)`, `restore(...)`, `main(argv)`.
- evals: `perplexity(model, chunks, cfg)`, `iteration_histogram(results, t_max)`.
- config: `training.checkpoints_per_run` (20) as the one new constant.

## Tests that reject a wrong version

The schedule's phase names and `t_max` at the boundaries and one step inside each phase; the lr
at step 0, at the end of warmup, at the last step (`final_lr_fraction`), and monotone decrease
after warmup; the halting head frozen in warm (its parameters unchanged after a step and `p` equal
to `1/t_max` on random inputs) and moving in loop; decay groups by name (`u` and norms without,
matrices with, `w_beta` without); the positions of a two-document slot continue across the
second document and across a chunk boundary (stream positions, never a restart); two groups over
one iterator take disjoint documents; a step lowers the loss on a repeated tiny batch and
detaches the states (no graph reaches the previous chunk: `grad_fn is None`); a step over `G`
groups with unequal loss masks gives the same gradient as the unsplit logical batch; a
checkpoint round trip restores identical model, optimizer, stream states and carried `H` and KV,
and a corrupted file or a changed tokenizer, config or finalization hash is refused; a run
checkpointed and restored follows the same optimizer trajectory and emits the same next chunk,
positions included, as the uninterrupted run; the mixed-document shares track the configured
shares at document granularity and a component running out is logged, not invented.

## Measured, 2026-10-15, on the box (RTX 5060 Ti, 16 GiB, 2.9 GiB held by the embedding server)

One chunk's forward and backward of Vesper-S (59,429,377 parameters) under bf16 autocast with
the float32 loss over the full vocabulary, peak CUDA memory allocated by PyTorch:

| Chunk | Passes | After forward | Peak |
| --- | --- | --- | --- |
| 1024 | 1 | 0.86 GiB | 1.23 GiB |
| 1024 | 2 | 1.31 GiB | 1.67 GiB |
| 2048 | 1 | 1.38 GiB | 2.12 GiB |
| 2048 | 2 | 2.18 GiB | 2.92 GiB |
| 4096 | 1 | 2.34 GiB | 3.81 GiB |
| 4096 | 2 | 3.95 GiB | 5.42 GiB |

Each pass adds about 1.6 GiB at 4096 tokens, so the eight passes of the recipe need about 15 GiB
for one chunk: the first GPU smoke (4096-token chunks, one chunk per micro-batch) ran two steps
(loss 10.90 then 10.81, 9.5 s and 16.3 s per step of eight chunks at two and four passes) and was
killed by CUDA out of memory on the third, in the loss over the float32 logits (512 MiB for one
chunk). The reference implementation keeps every column application's activations for the
backward pass; the memory work that follows this version is activation checkpointing per column
application (recompute in the backward; measured below at about half again the step time, not the
third first written here) and the objective computed over position slices of the hidden output
(the softmax stays over the full vocabulary per position), both rows to add before a run at the
recipe's chunk size on this card. The
smoke configuration uses 2048-token chunks for that reason. Throughput of the reference
implementation at two passes: about 430 tokens per second for one chunk, which places the full
10 B-token run far outside this box; the recipe's hardware section already says the kernels are
what make the model fast, and this measurement is the baseline they are measured against.

The second smoke, at 2048-token chunks (configs/vesper-s-smoke.yaml: four chunks per step, one per
micro-batch, forty steps, run b797837 at 03:28Z, exit 0, log and checkpoints under
~/data/runs/smoke-b797837): every phase of the schedule was observed in the log (warm
at steps 1-4 with `t_max` 2, 4, 5, 6 and `lambda_h` 0; loop at 5-16 with `t_max` 8 and `lambda_h`
ramping 0.0008 to 0.0092; ledger at 17-32; settle at 33-40), the learning rate warmed up to the
peak at step 4 and reached the final fraction at step 40 (3.0e-5), the loss fell from 10.907 to
7.493 (next-token part 10.907 to 7.413) with the gradient norm between 0.8 and 25 before clipping,
the held-out perplexity at the four checkpoints was 14159, 8223, 5790 and 6174 (two held-out
chunks: a noisy estimate, read as the trend), and every slot ran the full eight passes once the
head was unfrozen (the ponder term is 0.08 at `lambda_h` 0.01 in the ledger and settle phases: forty
steps do not move the halting head off its frozen start). Steps took 3.4 s at two passes and 8.3 s
at eight, 319 s for the run. Two things the smoke shows that the tests do not: the step log's
`tokens_seen` was the step's own count instead of the run's (the step record's field shadowed it
in the log line; fixed with a test over the log, the checkpoint's count was right all along), and
the logged shares count tokens of documents handed to the streams, not tokens consumed: the one
book pulled at step 1 (149,554 tokens) counts at once and fed one of the four slots for the whole
run, so a quarter of the consumed tokens were that book while the mixer, which balances by tokens
handed out, will not pull another book until the web component has handed out about twelve
million. The balance by tokens holds over a run; a 328 k-token smoke is below the granularity of
one book.

### Complete steps by stage, and with activation recomputation

The by-stage profile (`<evidence>/gpu-profile-by-stage-b797837.log`): a complete
step with the optimizer moments and every slot's state resident. Four one-slot groups of 2048
tokens peak at 3.64 GiB at two passes and 8.53 GiB at eight (resident 1.00 GiB; the forward
holds 1.6 GiB at two passes and 6.5 at eight; the float32 loss adds 0.25 GiB allocated and 0.5
GiB peak; the backward releases to 1.2 to 1.5 GiB). Eight one-slot groups of 4096 tokens peak at
6.21 GiB at two passes, 9.48 at four and 11.16 at five: about 1.65 GiB per pass, so eight passes
need about 16.3 GiB and the first smoke's failure in group 0's loss at the third step (five
passes) is what the profile predicts.

With `--recompute` (`Loop.forward(recompute=True)`: each column application's activations are
recomputed in the backward pass, non-reentrant `torch.utils.checkpoint`; the equivalence test on
the small model holds to 1e-6 on the objective, every gradient, the states, the iteration counts
and the per-pass outputs), the same complete steps measured from the step log's
`peak_memory_bytes` (runs `measure-2048-recompute` and `measure-4096-recompute`):

| Chunk x slots | Passes | Peak, kept | Peak, recomputed | Seconds, kept | Seconds, recomputed |
| --- | --- | --- | --- | --- | --- |
| 2048 x 4 | 2 | 3.15 GiB | 1.60 GiB | 3.3 | 4.5 |
| 2048 x 4 | 4 | 5.23 GiB | 2.20 GiB | 4.2 | 6.5 |
| 2048 x 4 | 8 | 8.53 GiB | 2.67 GiB | 8.2 | 13.1 |
| 4096 x 8 | 2 | 5.69 GiB | 2.64 GiB | 9.4 | 14.2 |
| 4096 x 8 | 4 | 9.39 GiB | 3.39 GiB | 16.4 | 25.4 |
| 4096 x 8 | 6 | out of memory | 3.71 GiB | - | 38.2 |
| 4096 x 8 | 8 | out of memory (about 16.3 GiB) | 4.20 GiB | - | 51.5 |

Recomputation costs about half again the step time (not a third, as the first estimate said) and
bounds the activation memory by one column application: the recipe's chunk size at eight passes
runs in 4.2 GiB on this card, with room for two slots per micro-batch. The losses of the first
steps agree with the kept-activation runs to the third decimal (10.907 / 10.835 / 10.748 against
10.907 / 10.836 / 10.748 at 2048; the GPU's bf16 reductions are not bit-deterministic across the
two paths, the CPU test is exact to 1e-6). `configs/vesper-s-smoke-4096.yaml` is the smoke at the
recipe's chunk size, run with `--micro-chunks 2 --recompute`.

## Amended after maintainer B's review of b797837 (docs/reviews/2026-10-15-training-review.md)

Decision 7 gains the run recipe and the layout: `Recipe` (seed, ordered components and shares,
held-out fraction, finalize and split rules, checkpoint schema) is hashed into the identity and
written plainly into the header and `train.json`; another recipe is a labeled fork, never a
continuation; the replay's identity is the mixer's digest over the ids it handed out, the counts
are the consistency check; `Layout(groups, slots)` is pinned and a different one refused before
training. Decision 4 gains the tail policy: a dry stream drops out of the batch and the others
train on with the token weighting over them; None only when every stream is dry, and then
nothing moved. The run ends at the token budget, the step cap or exhaustion, saves the last
completed step exactly once with that reason, and closes the log with an `end` line; every line
carries the header's `run_id`, and a fresh run never appends to a log that holds one. Decision 1
is corrected at one pass: the frozen head is held at the two-pass value, since the loop halts at
its last pass by construction and a saturated bias would survive into the free head with a zero
gradient. Decision 8's claim that every line carries the ablation label is withdrawn: the header
and the checkpoint carry it, and the `run_id` links every line to its header. Checkpoint
directories, the dropped-id file and the finalization receipt are published by one rename
(`publish.staged`). The outcome section of the review record has the list with its tests.

### The smoke at the recipe's chunk size, and the step after the lifetime fix

The full-size smoke (configs/vesper-s-smoke-4096.yaml on c2e47ad: forty steps of eight 4096-token
chunks, two slots per micro-batch, `--recompute`; run 04:16Z to 04:40Z, exit 0, log
`<evidence>/smoke-4096-recompute-log.jsonl`): every phase observed (warm 1-4 with
`t_max` 2, 4, 5, 6; loop 5-16; ledger 17-32; settle 33-40), the learning rate 7.5e-5 to the peak
3e-4 to 3e-5, loss 10.902 to 7.696 (minimum 7.595 at step 39), gradient norm 0.54 to 27.2 before
clipping, held-out perplexity 12134, 5330, 4106, 3815 at the four interval checkpoints (four
held-out chunks), every token at eight passes once the head was free. Peak memory per step 4.63
GiB at two passes rising to 6.92 GiB at eight; 35.0 s per step at eight passes (8 x 4096 tokens:
about 940 tokens per second), 1335 s of step time for the run; it ended with reason `budget` at
1,310,720 tokens. Hand-out counts at the end: web 946,045, code 177,478, wiki 48,386, books
149,554 (one book, handed out at step 1); the consumed counts per source exist from 714850f on
and were not in this run's log.

After maintainer B's lifetime finding (review of 444b966) each group's tensors are released before the
next group's forward (`_train_group`), and the objective is computed over the loss positions
only. The actual `train.step` at 2048 x 4 slots with kept activations then peaks at 3.03 GiB at
two passes and 8.38 GiB at eight (before: 3.15 and 8.53); the saving is a tenth of a gibibyte or
two, as the weak-reference finding suggested it would be bytes, not the cause of the 4096 failure.

The bounded-cost resume on the real data (714850f, 2048-token smoke config, kept activations):
a run of three steps saved its terminal checkpoint, a second invocation resumed it for two
steps through the mixer's cursors, and an uninterrupted five-step run of the same recipe gives
the reference. Steps 4 and 5 of the resumed run match the uninterrupted run within 4.6e-5 and
4.4e-5 relative in the loss, with identical `tokens_seen`, hand-out counts and consumed counts;
steps 2 and 3, which both runs computed from scratch, differ by 1.9e-5 and 6.1e-5, so the resumed
steps lie inside the band two identical GPU runs show (bf16 reductions; the CPU test is exact to
1e-6). The resumed invocation, start to its first step, took under forty seconds including the
model's construction; no prefix was replayed.

### The objective over position slices, and the provenance of the measurements

`--loss-slices` (13968b5) measured as complete `train.step` peaks at the recipe's chunk size:
eight 4096-token chunks per step as four groups of two slots, with recomputation, on the clean
tip 13968b5 (logs `measure-4096-slices1-log.jsonl` and `measure-4096-slices4-log.jsonl`):

| Passes | Peak, one slice | Peak, four slices | Seconds, one slice | Seconds, four slices |
| --- | --- | --- | --- | --- |
| 2 | 4.13 GiB | 2.54 GiB | 10.0 | 10.0 |
| 4 | 5.15 GiB | 3.71 GiB | 17.4 | 17.5 |
| 6 | 5.73 GiB | 4.30 GiB | 25.8 | 26.2 |
| 8 | 6.41 GiB | 4.95 GiB | 34.8 | 34.6 |

The losses of the two runs agree to 1.5e-4 relative or better over the six steps (the GPU's
bf16 band; the CPU test is exact to 1e-6), the step time is unchanged. The one-slice peak of
6.41 GiB is already below the full-size smoke's 6.92 GiB on c2e47ad in the same layout, by the
released group tensors and the head applied to the loss positions only.

Provenance of every measurement above, since a header's `vesper_commit` names HEAD and not the
code that ran (maintainer B's review of 66b674a; `environment()` records `vesper_dirty` from this tip
on): the per-chunk probe and the by-stage profile ran on b797837 with a scratch script; the
2048 smoke and its resume on b797837; the `measure-2048` and `measure-4096` kept-activation runs
on 9ee1d97's tree plus the uncommitted peak-memory field (committed as part of 444b966);
`measure-2048-recompute` and `measure-4096-recompute` on 444b966's tree plus the uncommitted
recomputation change, identical in content to what e86fca8 committed; the full-size smoke on
the clean tip c2e47ad; `measure-2048-released`, `cursor-2048` and `straight-2048` on 714850f's
tree plus the uncommitted sliced-objective change, identical in content to what 13968b5
committed; the slice measurements on the clean tip 13968b5. A layout's number belongs to that
layout: 4.20 GiB at eight passes was one slot per micro-batch, 6.92 and 6.41 and 4.95 GiB were
two.
