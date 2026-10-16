# Recipe binding, staging ownership and smoke evidence

Date: 2026-10-15. Pinned tree: 444b966d9e3ae98db37d4ffcfcb671127baebc3c.
The earlier trainer lifecycle findings are closed in the tested cases. Changes
are requested for two narrower boundaries: operation-owned staging and the
association between components and their files. The previously acknowledged
generation transaction, cursor resume and full-size memory work remain open.

Evidence: <evidence>/reviews/training-444b966/. Run probes.py with
PYTHONPATH=.:tests and the reviewed checkout's locked Python environment. All
inputs are tiny owned fixtures. Full independent CPU gate: 310 passed, three
CUDA skips, clean union dependency audit; HEAD and tracked source unchanged.
Maintainer A's selected full GPU gate log records 313 passes; not repeated here.

## Earlier lifecycle fixes verified

- The equal-length changed-seed resume is rejected as another recipe before
  training. A changed micro layout is explicitly refused. An unchanged recipe
  resumed after step three still matches the uninterrupted seven-step run's
  final parameters exactly (maximum difference 0).
- Three steps log tokens=[32,32,32], tokens_seen=[32,64,96], checkpoint total 96.
  The header contains the seed and every line has the matching run_id.
- The uneven short/long corpus processes groups_active=[2,1,1,1], then writes
  one terminal checkpoint at step 4, the last completed step. Resuming that save
  adds no steps or checkpoints and terminates exhausted.
- A one-pass warm setting now starts with finite bias 0. The free two-pass update
  gives bias gradient 0.0020963477 and direction gradient magnitude 0.01809084,
  moves the bias to -0.00029999856 and reaches pass two. The previous saturated
  head case is closed.

## 1. publish.staged is process-named, not operation-owned

Every invocation for one destination in one process uses the same
.<name>.<pid>.partial. Two nested or overlapping calls therefore share a path.
The fixture writes the outer call's data, enters a second staged call for the
same destination, and raises inside the inner body. Its cleanup deletes the
outer call's file. The outer publication then fails because nothing remains.

A second fixture pre-creates that predictable partial path and enters staged
without writing anything. The helper publishes the preexisting content instead
of raising the documented nothing-written error. It never established ownership
of that content. These failures are inside the new primitive; they do not rely
on the still-unimplemented multi-artifact generation transaction.

Exclusively allocate per-invocation staging, and remove only that invocation's
owned files. One option is a uniquely allocated temporary parent on the same
filesystem with an initially absent child path for the caller's file/directory.
Publish that child and clean up its owned parent in finally. This preserves
the empty-body check and supports both output kinds without a shared predictable
partial. Other exclusive-allocation designs can satisfy the same contract.
Tests should pin ownership, cleanup and publication behavior rather than require
the pid-only filename. Include overlap, preexisting remnants, inner failure,
rename failure and retry, for both files and directories.

## 2. Recipe checks omit file-to-component membership

The recipe hashes ordered (name, share) pairs; check_inputs compares only the
flattened (path, hash) sequence. Neither check binds each file to its component,
although the finalization receipt already records that association.

Fixture, all five file contents unchanged and in the same flattened order:

    original: web=[A], code=[B], wiki=[C,D], books=[E]
    changed:  web=[A], code=[B], wiki=[C],   books=[D,E]

Names and shares remain web .80 / code .15 / wiki .04 / books .01. Finalization input checks
pass, recipes compare equal, and the step-3 checkpoint resumes under the changed
mapping. Its already-handed-out prefix is A,B in both cases, so the new replay
digest also agrees. After four further steps, final parameter tensors differ
by up to 0.000438004732131958 from the unchanged-mapping continuation.

This is a different mixture despite unchanged file bytes. Check the association
against the canonical input manifest, including component name, file order and
content identity; alternatively put a canonical component-input-manifest digest
in the recipe. Reuse one definition rather than hashing a second drifting
representation. Refuse changes to as-yet-unseen files' component membership
before restore/training. The replay digest validates the consumed prefix; it
cannot establish that the unconsumed suffix follows the same recipe.

## Memory evidence and tensor lifetimes

The selected stage profile records complete measured steps with moments and slot
states resident: 2048 tokens / four one-slot groups peak at 3.64 GiB for two passes
and 8.53 GiB for eight; 4096 tokens / eight one-slot groups at 6.21, 9.48 and
11.16 GiB for two, four and five passes. The estimate for eight passes at 4096 is
an extrapolation, not a successful full-size run. The actual 4096 smoke still
failed; the 2048 config is explicitly a smoke configuration, not the full recipe.

The profile explicitly releases group outputs. The actual train.step retains
logits, result.y and result.per_pass when it begins the next group's forward.
CPU weak-reference probes establish those lifetimes. This does not quantify GPU
bytes or prove the cause of the earlier OOM. Release unneeded group outputs after
backward/state detachment, and compare the actual step with the profiling driver
under the same inputs, passes, optimizer state and shared device occupancy.
Use actual train.step peaks to evaluate the final memory change.

Activation checkpointing and vocabulary-sliced loss are described in the map and
proposal. At this pinned tip, I did not find their separate API/owner/test rows
in docs/registry.md: its explicit unfinished rows are generation publication and
Cursor. Add the memory rows before implementation. For the sliced loss, preserve
the global vocabulary normalization, tied-embedding and hidden-state gradients,
mask behavior (including the earlier masked-nonfinite case), and carried states.
A per-slice softmax would change the objective. Measure the whole forward/backward
step rather than only the smaller forward allocation.

Attribution correction: the last review explicitly did not infer the failed
micro-group or step from the OOM trace. The identification of group 0 at step 3,
t_max=5, is maintainer A's later report/profile context, not an earlier maintainer B inference.

## Independent checks of shared run artifacts

The supplied 2048 smoke log has 40 steps covering warm/loop/ledger/settle. Logged
step durations sum 318.9678s; that is a sum of step timings, not separately measured
end-to-end wall time. Loss moves 10.9067893 to 7.4932595. Checkpoint perplexities are
14158.9640, 8223.4300, 5790.2503, 6174.4460. The local 2048 smoke config file hash
matches the original run header exactly. These are checks of maintainer A's exported
logs, not a rerun or an independent evaluation of the private training inputs.

The supplied resume log covers steps 31..40. Step 31 loss matches exactly; maximum
relative loss difference across the ten steps is 7.975884855177101e-5. Phase,
learning rate and handed-out token counts agree. This supports the reported
bf16 smoke continuation, not a bitwise-GPU-determinism claim. All free-phase
histograms allocate every token to pass 8; adaptive compute savings are not shown.

The long-book observation makes consumption attribution worth keeping explicit.
The field shares contains tokens handed to streams, as now documented. Its
149554 book tokens are not all consumed within this smoke, and the claimed
roughly one-quarter book occupancy is maintainer A's observation. Do not present
hand-out ratios as measured optimizer input ratios. Count source spans as streams
emit them before making a prescribed-mixture comparison.

The new finalization receipt names 20 inputs with exactly the same hashes as the
old receipt and the same totals: 10097165 documents, 382042 repeated keys, 404078
dropped ids. I verified the shared dropped parquet SHA256 against its receipt:
ea94a6485ecb14057c94d1ecbf102a5f287203a98f6135f0498fc1afaad3d5cf.
Loading it gives 404078 unique ids, 60 with wiki prefixes, and 20 recorded inputs.
The shared run log records 1758 s and exit 0 under b797837. This validates the
published artifact/count relationship; private input hashes and the scan itself
were not recomputed by maintainer B.

Maintainer A owns the two boundary fixes and the remaining generation/cursor/memory
work. The tokenizer decision remains with the project owner; no tokenizer mutation or large
training run was made by maintainer B. No Vesper commit, push, merge, shared-service or
driver change was made in this review.

---

## Outcome (maintainer A, 2026-10-15, taken on the tip named in the channel)

1. **Operation-owned staging.** `publish.staged` allocates a fresh staging directory per call
   beside the destination (`tempfile.mkdtemp`, `.<name>.<random>.staging`) and yields the
   initially absent child inside it; the child is renamed onto the destination on exit and only
   this call's staging directory is removed, in `finally`. Tests pin the behaviour, not a name:
   a nested call's failure leaves the outer call's file and the outer publication succeeds; a
   leftover staging directory of a crashed run is never published and never touched; a body
   that writes nothing is refused; two overlapping calls on one destination each publish their
   own content; files and directories; the non-empty directory destination refused; a failed
   directory save leaves no checkpoint and the retry succeeds (tests/test_publish.py,
   tests/test_train.py).
2. **File-to-component assignment.** Bound at the canonical manifest: the finalization receipt's
   input rows already carried the component, and `Finalization.inputs` now keeps it as
   `(component, path, sha256)`, so `check_inputs` refuses the same files under another
   assignment before any restore ("the files' assignment to components is pinned"). One
   definition, no second digest. Test: your fixture in two components (a=[P0], b=[P1, extra]
   against a=[P0, P1], b=[extra]), refused; the loaded receipt lists the components.
3. **Tensor lifetimes in `train.step`.** Each group's forward, objective, backward and state
   hand-off run in `_train_group`, whose frame dies before the next group's forward, so the
   previous group's logits, `y` and `per_pass` are released. Measured as actual `train.step`
   peaks after the change (2048 x 4 slots, kept activations): 3.03 GiB at two passes and 8.38
   GiB at eight, against 3.15 and 8.53 before; a tenth of a gibibyte or two, as the finding
   said: bytes, not the cause of the 4096 failure (the proposal's Measured section).
4. **Memory rows.** The recomputation row landed in e86fca8 with its registry rows (loop,
   model, train) and tests; the sliced objective is a registry row marked specified
   (`Vesper.objective(chunk, result, cfg, weights, *, slices)`, position slices of the hidden
   output, the softmax over the full vocabulary per position, equality with `wake_objective`
   on objective, gradients and masked behaviour as the test, a complete step as the measure).
   The earlier wording "vocabulary slices" was wrong and is corrected: a per-slice softmax over
   vocabulary pieces would change the objective, as you say.
5. **Bounded-cost resume (the open cursor row).** `RawReader(paths, seed)` with `cursor()` and
   `seek()`, `Mixer.state()` and `Mixer.restore()` (readers' cursors, active set, balances,
   count, digest), the digest a sha256 chain that continues from its stored hex
   (`SequenceDigest(state, count)`), the mixer state in `states.pt`, and `run` resumes by
   seeking instead of replaying; the replay stays as the test oracle. Tests: a seeked reader
   yields exactly what the uninterrupted one yields next; a restored mixer hands out the same
   remaining documents with the same balances and digest; the run's resumed fifth step equals an
   uninterrupted run's fifth step; a checkpoint whose mixer state, progress and streams disagree
   is refused; checkpoint schema 3.
6. **Attribution.** Corrected: the identification of group 0, step 3, `t_max` 5 is from my
   profile, not an inference of yours; the channel message said otherwise and this record
   says so.

Open as named: the generation transaction (proposal docs/reviews/2026-10-15-generations.md,
the project owner's layout decision), the sliced objective, the frozen tokenizer. The hand-out versus
consumed distinction stays explicit in the registry; source-span counters as the streams emit
them are the next instrumentation before any mixture comparison.
