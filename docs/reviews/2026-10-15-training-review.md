# Training lifecycle review and collection follow-up

Date: 2026-10-15. Pinned tree: b7978373710e86cd2330eb74a06d31316ca87d7a.
Status: changes requested for the trainer. The prior collection counterexamples
below close; the shared generation publication and bounded-cost cursor work
remain explicitly unfinished, as acknowledged in channel 2e431c9.

Evidence: <evidence>/reviews/training-b797837/. Run probes.py with
PYTHONPATH=.:tests in the reviewed checkout, and collection-checks.py with
PYTHONPATH=., using its locked Python environment. The test configuration helper
supplies only model geometry; the lifecycle comparisons and expected results
are independent. All source data and runs are tiny owned temporary fixtures.

The full independent CPU gate passes 302 tests with three CUDA skips and a clean
union dependency audit. The source and HEAD remained unchanged through it.
Maintainer A reports 305 passes in the full GPU gate. Passing those tests does not
establish that the actual 4096-token training step fits the device; the supplied
smoke log records an OOM, discussed below.

## Positive evidence

- Exclusive receipt reservation rejects the previous overlapping finalization
  fixture; the outer operation completes and its receipt reloads correctly.
- A hash-read failure after closing ParquetOutput now removes the unpublished
  temporary and preserves the previous destination.
- The changed-then-restored input fixture has equal plan/trained counts and
  endpoint hashes, but distinct consumed-stream digests. It is now correctly
  non-freezable. This establishes plan/training text agreement detection; it does
  not replace the acknowledged immutable input-generation work.
- The inline phrase is now `I really agree\nYes`; a noteref's number is removed
  and its ` tail` remains. The shared-buffer traversal closes those failures.
- An independent end-to-end run of seven steps and a same-seed run stopped at
  step three, checkpointed, and resumed for four more steps have identical final
  parameter tensors (maximum difference 0) and identical final four losses.
  Packing-state restoration is doing useful work; the old lost-buffer problem
  is fixed for this unchanged recipe.

## 1. Resume does not identify the data order or actual micro-batch layout

Identity contains tokenizer, config and finalization hashes, but not the seed.
The seed is also absent from the run header. run accepts a new --seed, replays
that order, and checks only aggregate token counts. Equal-length documents make
that check unable to distinguish orders.

The fixture resumes the same step-3 checkpoint with seeds 0 and 1. Both are
accepted with the same replay totals; their saved queues initially agree, then
new documents arrive in different orders. After four more steps, the final
parameter difference is 0.00042301416397094727 and the last losses are
5.5151379108428955 versus 5.328225612640381. In contrast, unchanged-seed resume
matches the uninterrupted run exactly.

Pin an explicit run recipe including shuffle seed, ordered component identities
and shares, finalization/split policy, tokenizer/config identity and schema.
Validate it before mutating a model or consuming input on restore. Aggregate
counts are consistency checks, not identity. A deliberately changed recipe
starts a labeled fork, not an equivalent continuation.

Micro-batch layout has a separate issue: a checkpoint with two one-slot groups
resumes with micro=2, logs micro_chunks=2, but retains two one-slot groups in its
next checkpoint. Either explicitly repartition logical slot states/streams into
the requested grouping or refuse an incompatible request before training. Log
the grouping actually executed. Preserve logical slot order and test a supported
regrouping against the unchanged optimizer trajectory.

## 2. Exhaustion consumes an untrained chunk and can leave no checkpoint

step calls next on every group, then returns None if any group returned None.
In the fixture, one stream has no further inputs while another still has them.
The latter advances its emitted position from 16 to 32, but step returns None
and its model H remains at position 16. That chunk was consumed without a model
update. This is different from the documented component-exhaustion policy.

The actual run driver has a second failure. With a short corpus that completes
one step before exhaustion, its log is run, step, exhausted and there are no
checkpoint directories. Thus even that completed optimizer step is not saved,
despite the checkpoint-at-end contract. Saving the already-advanced stream state
after the failed draw would not be a correct fix for the first problem.

Prefer processing all non-exhausted groups, using the existing token-count
weighting for the effective batch, until every stream is exhausted or the run
budget is reached. Distinguish missing groups from groups with zero loss that
still contain inputs and need their states advanced. If a different tail policy
is deliberately chosen, it must retain a coherent data/model checkpoint and
account for omitted inputs explicitly. Prepare a usable batch before changing
the halting head; a no-step return should not silently advance its schedule.

On normal termination, save the last completed optimizer/data boundary exactly
once, including termination reason and actual processed tokens. Separate token
budget, optional optimizer-step limit and data exhaustion; fixed nominal batch
arithmetic does not establish that a partially filled run consumed its budget.
Test unequal remaining stream lengths, empty groups, a final partial batch and
exhaustion before/between checkpoint intervals, then resume the terminal save.

## 3. Step logs overwrite the cumulative token counter

run creates a line with cumulative tokens_seen, then line.update(_plain(result))
overwrites it with StepResult.tokens_seen, the current step's count. Three
32-token steps log [32, 32, 32] while the checkpoint correctly records 96.
This makes external phase/throughput comparisons misleading even though the
internal schedule uses the correct cumulative variable.

Give the two quantities distinct fields, for example step_tokens and
tokens_seen_total, and test them against the checkpoint and the emitted masks.
The ablation label currently appears in the run header and checkpoint record,
not every log line as claimed. A stable run id linking lines to the complete
header is sufficient if that is the chosen documented log contract; include
the actual seed and layout there. Avoid silently mixing fresh runs in one log.

## 4. A one-pass warm phase can permanently saturate the learned head

A valid small-budget configuration begins warm at t_max=1 and reaches the free
phase before a later warm step changes the bias. set_halting writes 10000 for
the one-pass case; unfreezing only enables requires_grad. The fixture then runs
with t_max=2, but bias and direction gradients are exactly zero, the bias stays
10000 and every token halts at pass one. The head cannot learn out of that state.

Treat the single-pass warm rule as a forced-halt override, or establish a finite
learnable bias at the transition to multiple free passes. Pin the transition
state across save/restore. Test a short warm phase that really executes only the
single-pass setting, then verify a nonzero head update on a discriminating loss.
The existing test manually sets p=1/2 before unfreezing, so it misses this case.
This is a valid short-budget boundary, not a claim that the current S smoke
configuration necessarily takes that path.

## 5. Full-size execution and acknowledged unfinished work

The selected <evidence>/smoke-b797837-4096-oom.log records failure
at masked next-token cross-entropy while allocating 512 MiB. It reports 11.95
GiB allocated by PyTorch in the training process and another process using
2.83 GiB. This is maintainer A's exported run evidence, not an independently repeated
GPU run. It does not identify the micro-group or optimizer-step index, so the
review does not infer which one failed.

Profile allocated/live/peak memory by micro-group at forward, loss, backward and
optimizer stages, for both the warm and maximum pass counts. Include every
logical slot's persistent state, optimizer moments after creation, and the
shared device occupancy. The separate forward-memory table is not a complete
step measurement. Any memory-saving change must preserve the masked objective,
gradients, carried-state semantics and intended chunk size; test that equivalence
and measure the resulting complete step. No shared service or driver was changed
by this review, and no additional GPU run was launched here.

The shared generation transaction remains needed for collection, tokenizer,
finalization and checkpoints. The new checkpoint function also writes directly
into its final directory, so a failed save leaves a partial directory that a
retry refuses. Apply the same publication primitive rather than declaring the
new checkpoint path exempt. Finalization's exclusive writer reservation closes
its overlap fixture, but its empty reserved JSON exists before publication;
existence of both files is not a reader-ready marker as its comment claims.

Bounded-cost upstream cursor restoration is also still open. Prefix replay can
serve the correctness oracle; it is not the completed production recovery path.
The reported no-retrieval smoke remains a partial-scope ablation using the pilot
tokenizer. Neither the full recipe nor full-size device fit is established.

## Real-corpus receipt scope

The explicitly shared historical becadeb receipt lists 20 inputs (14 web, four
code, one wiki, one books), 10,097,165 documents, 382,042 repeated keys and 404,078
dropped ids (404,018 web and 60 wiki). These are recorded run results from the
old validator, not an independent scan of the private source files. The current
validator's rerun and receipt remain pending. A single code shard's equal row
and distinct-key counts establish that shard's internal uniqueness; they alone
do not establish a corpus-wide upstream deduplication guarantee.

Maintainer A owns trainer fixes, memory work, generation publication and cursor resume.
Maintainer B re-reviews pinned results. No production edits, Vesper commits, pushes or
merges were made by this review.

---

## Outcome (maintainer A, 2026-10-15, taken on a8c22f4..the tip named in the channel)

Every requested trainer change is taken, each reproduced as a test first:

1. **Recipe and layout.** `Recipe(seed, ordered (name, share) components, held_out_fraction,
   finalize_rule, split_rule, schema)` is hashed into `Identity` and written plainly into the run
   header and `train.json`; a resume under another seed is refused as another recipe (test: seed 1
   from a seed-0 checkpoint). The replay's identity is the `Mixer` digest over the ids it handed
   out (`SequenceDigest`), compared with the checkpoint's before anything is consumed; the counts
   stay as the consistency check (test: two mixers with equal counts and different seeds have
   different digests; a replay reproduces the uninterrupted digest). `Layout(groups, slots)` is
   written into `train.json` and the header; a checkpoint restored under another layout is refused
   before training ("regrouping is not supported"), so the layout logged is the one executed.
2. **Exhaustion.** `step` draws first and processes every group that has a chunk, with the token
   weighting over those; `groups_active` is in the result and the step line; None only when every
   stream is dry, and then nothing was drawn, no state moved and the head was not touched (the
   halting freeze now happens after the draw). The run logs an `exhausted` line when the active
   count drops, ends with reason `exhausted`, `budget` (tokens seen against `total_tokens`, not
   nominal step arithmetic) or `steps`, and saves the last completed step exactly once with that
   reason unless an interval checkpoint already holds it; an `end` line closes the log. Tests: one
   dry stream trains the live one; both dry moves nothing; a run over a short corpus with one long
   document trains on with one group, saves once at the end, and resumes to an immediate end.
3. **Counts.** `StepResult.tokens` is the step's count, `tokens_seen` the run's, written after the
   record; the test checks the running sum and the checkpoint's count (a8c22f4, found by the
   smoke before this review arrived). Every log line carries `run_id`; a fresh run refuses an `out`
   whose log holds a run; the header carries the seed, the recipe and the layout.
4. **One-pass warm.** The frozen head is held at the two-pass value (`p = 1/2`, logit 0) when the
   phase runs one pass: the loop halts at its last pass by construction, so the head's value is
   irrelevant there and no saturated parameter survives the transition. Test: a warm step at
   `t_max = 1`, a checkpoint round trip of that head, then a free step at `t_max = 2` with nonzero
   gradients on both head parameters, a moved bias and tokens reaching the second pass.
5. **Memory.** `StepResult.peak_memory_bytes` is in every step line. Measured on the box (docs/
   reviews/2026-10-15-training-loop.md, Measured): a complete step with four one-slot groups of
   2048 tokens, optimizer moments and every slot's state resident, peaks at 3.64 GiB at two passes
   and 8.53 GiB at eight; eight one-slot groups of 4096 tokens peak at 6.21 GiB at two passes, 9.48
   at four and 11.16 at five, about 1.65 GiB per pass, so eight passes need about 16.3 GiB against
   the card's 12.5 GiB beside the other tenant: the full-size step does not fit this card without
   the memory rows (activation checkpointing per column application, the loss over vocabulary
   slices), which are specified in docs/registry.md and ARCHITECTURE.md with the equivalence tests
   they must pass (masked objective, gradients, carried states, intended chunk size).

Also taken: `publish.staged` (PRIMITIVE) publishes a file or a directory by one rename; the
checkpoint directory, the dropped-id file and the finalization receipt use it (a failed save
leaves no `step-n`; the reserved receipt stays empty until the complete one lands, and
`load_finalization` refuses an empty receipt as reserved, not published). The generation
transaction and the bounded-cost cursor are registry rows marked `specified` (accepted plans,
no code), distinct from the tested rows.

Not taken: nothing. Open as named: the generation transaction, the cursor resume, the memory rows,
the frozen tokenizer (the project owner's pre-tokenization decision).
