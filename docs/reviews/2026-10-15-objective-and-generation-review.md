# Objective compatibility and generation boundaries

Date: 2026-10-15. Loss/counter checks pinned to
629559f76eb160d4a1b980c9dcea19b423d8b413. Generation checks pinned to
5726c87a329d1ffa4e147b8048a87aa94d9429bf (the implementation of 4accce1 plus its
documentation note). The loss/counter code is unchanged between those tips.

Changes requested: preserve unsupported-mode refusal and checkpoint format
identity; correct selection concurrency/snapshot binding, path controls and
parent validation before integrating the generation drivers or moving live data.

Evidence folders under <evidence>/reviews/:
objective-629559f and generations-5726c87. Runnable fixtures, outputs and full CPU
gate logs accompany this review. The old-schema test creates a genuine checkpoint
using a temporary git archive of 66b674a, then tries the new reader. All data and
outside-path sentinels are owned temporary files; no private peer data was read
or modified. This review changes documentation only; no production code, merge
or live data move is included.

Independent full gates: 629559f has 317 CPU passes and three CUDA skips;
5726c87 has 322 CPU passes and three CUDA skips. Both union dependency audits
are clean and tracked source/HEAD stayed unchanged. Selected maintainer A GPU logs
record 320 and 325 passes respectively; those GPU tests were not repeated here.

## 1. The sliced loss bypasses the unsupported local-mode guard

Vesper.objective calls wake_total directly. The old wake_loss path refuses
cfg.learning.mode=local because its predictive-coding objective is not built;
the new path never checks the mode. The same fixture still raises through
wake_objective, but returns loss 4.8005595 through Vesper.objective and completes
a training step. There are 31 parameter gradients and a non-halting/embedding
parameter update of 1.788e-7. The run is performing the e2e objective despite
the local configuration.

Restore one shared mode validation used by both objective paths. Reject an
unsupported run before streams or training state advance, and retain the direct
objective refusal as well. Factoring the arithmetic into wake_total is useful;
it must not remove the precondition enforced by wake_loss. A test selecting
local mode must fail on both paths until that mode is actually implemented.

## 2. Changed checkpoint records still claim schema 3

Progress now requires tokens_consumed and queued documents now serialize
(id, tokens, source) instead of (id, tokens). CHECKPOINT_SCHEMA remains 3.

schema-check.py creates a real schema-3 checkpoint with archived 66b674a code.
Its identity equals the current reader's computed identity, so it passes the
format/recipe gate and then fails with KeyError: tokens_consumed. The old queued
tuple format is another incompatible field behind that first error.

Bump the schema and reject old formats explicitly before loading weights/state,
or provide a tested versioned conversion. Historical consumed-source counts
must not silently become zeros. Test an actual previous-format fixture as well
as current round trips; testing only a writer against its matching reader misses
the compatibility break.

## Slicing and consumed-source evidence

The e2e formula keeps the full-vocabulary softmax per position and selects masked
positions before applying the head. A separate token-value oracle checks 92
slot-chunks across empty, singleton, short and long documents: doc_ids, source
spans and valid counts agree. Seven-step interrupted/uninterrupted run counters
also agree, sum to tokens_seen, and do not exceed hand-out totals. Example final
consumption is web 112 / code 98 / books 14 versus handed-out 136 / 98 / 98 at 224 inputs.

The sliced numerical comparison uses a full-head reference at the declared
float32-logit precision, with slices 1 / 3 / 100 and column recomputation enabled:

- Float32 gradients pass atol 1e-6 plus rtol 1e-5; maximum absolute delta 1.13e-6,
  relative L2 delta 5.78e-7. The combined tolerance is not a universal absolute
  1e-6 bound.
- Float64 model/history with float32 logits gives gradient differences at about
  1e-16 and maximum scalar loss delta 4.77e-7 from accumulation.
- CPU bfloat16 autocast gives equal tested scalar losses but maximum parameter
  gradient delta 0.001953125. Relative L2 deltas are 5.36e-4 and 6.76e-4 for 3 and 100
  slices; these do not meet the float32 test tolerance.

The formulas agree. Numerically unchanged gradients require dtype-specific
tolerances and a production CUDA bfloat16 comparison, including tied weights
and hidden-state gradients. These measurements do not establish a wrong
normalization or invalidate the memory approach; they limit the equality claim.

The selected 13968b5 GPU logs show eight-pass peaks 6.413 GiB at one slice and
4.952 GiB at four, for the same four groups of two slots at 4096 tokens. Maximum
six-step loss difference is 1.45105e-4 relative. The real-data cursor comparison's
resumed steps 4 / 5 differ by 4.6057e-5 / 4.3785e-5; its fresh steps 2 / 3 already differ
by 1.8894e-5 / 6.0713e-5. Input counters agree exactly. These are checks of exported
run evidence, not independent GPU executions. The new dirty marker prevents
silently treating HEAD as a clean source identity; dirty-run patch equality
remains maintainer A's explicit provenance declaration unless its source hash is saved.

## 3. Selection comparison and publication are not one transaction

select checks the predecessor hash, then constructs and renames a new file
without serializing that comparison with publication. The probe interleaves a
second select after the first has passed its check. Both return successfully
with the same previous_sha256; the first then overwrites the second's choice.
The existing sequential stale-update test does not exercise this window.

Use a shared per-data-root writer lock or an equivalent true compare-and-publish
primitive. Under it, check the expected predecessor and publish the new selection
as one operation; one of two contenders based on the same predecessor must fail
explicitly. Return the selection that this call committed. Atomic rename alone
only prevents partial-file visibility; it does not enforce the predecessor rule.

There is an independent read-side race. selection reads and parses the old JSON,
validates its generations, then reopens the path to hash it. A valid concurrent
replacement during validation makes the returned Selection contain old references
and the new file's hash. Read the bytes once, parse and hash that same snapshot,
then validate the referenced immutable generations. Apply the same rule to
MANIFEST.json so a digest always identifies the bytes actually interpreted.

## 4. Control paths and control files bypass the validation boundary

The listed data-file checks work, but not all path-controlling values go through
them before use. Reproduced cases:

- publish_generation(kind='../outside-kind', ...) publishes outside the supplied
  data directory. The publisher never validates kind before forming its path.
- A MANIFEST.json symlink in staging points to an outside sentinel. write_json
  overwrites that sentinel before validation, and the resulting generation is
  accepted with a symlink manifest. Validating after the write is too late.
- pinned follows a generation-directory symlink outside data. The file checks
  use the redirected generation as their root, so all of them pass; they do not
  establish containment under the selected data root.
- validate_generation accepts a manifest identity different from its published
  directory name. That returned identity cannot correctly address the same
  generation through reference/pinned.

Validate kind and identity before any path construction or mutation. Reject
unintended symlinks for staging/generation directories and control files, including
the manifest and selection. Bind the canonical generation path to the canonical
data/kind root, not merely to itself after redirection. Bind a published
directory's identity to its manifest; staging validation can take the expected
future identity explicitly. Use one typed schema/name/containment boundary on
both writer and reader paths, including repeated-kind and malformed-record
validation. Preserve the outside sentinel in refusal tests.

These are observed failures in owned fixtures, not reports of an attack or
corruption of the actual acquired corpus.

## 5. Inheritance trusts a stale parent object

_inherit uses the supplied Generation object and hard-links its paths without
revalidating its pinned manifest/content. The probe changes a parent data file.
validate_generation correctly rejects that parent, but publish_generation with
the previously returned parent object still succeeds: it hashes the changed
inherited bytes into a new child manifest while recording the old parent
manifest hash. The child validates, despite its inherited content disagreeing
with the pinned parent record.

Revalidate the parent against its pinned identity/hash before inheritance and
preserve the parent's verified file records for inherited content. A corrupt
parent must not be relabeled as verified child content. Test parent data changes
and parent-manifest replacement after the original parent object was created.

A second fixture inherits a wiki parent into a code child. It is accepted even
though the stored parent tuple contains no kind and therefore cannot describe
that cross-kind relationship. Require the same kind for extension, or design an
explicit cross-kind fork with an unambiguous reference; the current extension
contract should not accept it accidentally. Reject before adding links or
rewriting the caller's staging manifest.

## Integration boundary

The six amended generation principles are taken correctly into the proposal.
The isolated code and tests can continue while the live layout remains the project owner's
decision. The driver integration, pinned generation references in finalization /
training, selection-based components, explicit legacy mode and migration plan
are still unwritten. Keep that distinction visible. Fix the publication and
validation boundary above before basing those consumers on it; do not move live
data as part of this review.

The schema/mode findings were sent early in channel 715a682 so maintainer A could fix
them while the generation review continued. Maintainer A owns all corrections and the
CUDA dtype test; maintainer B re-reviews pinned results. Earlier accepted cursor and
recomputation findings are not reopened by these new paths.

---

## Outcome (maintainer A, 2026-10-15, taken on the tips named in the channel)

Items 1 and 2 were taken on 4a4c233 before this record arrived: `learning.check_mode` is the one
mode validator on both loss paths and in `run`; the checkpoint schema is 4 and is refused before
the identity and before any weight is loaded; a CUDA bf16 test states tolerances for the sliced
objective (loss within 2e-3 relative, every gradient within 1e-2 in relative L2) instead of
claiming equality, and the float32 equality is stated as atol 1e-6 with rtol 1e-5.

Items 3 and 4, each reproduced as a test first:

- **Selection as one transaction.** `select` holds the data root's writer lock
  (`.selection.lock`, `flock`, exposed as `writer_lock`) across its compare and publish, and
  returns the selection it committed, read back under the lock. Test: a contender blocks while
  another writer holds the lock, nothing is published meanwhile, and the stale contender based on
  the same predecessor is refused afterwards.
- **Snapshot binding.** A control file is read once; the hash is of the bytes parsed
  (`_snapshot` for the selection, the same rule for `MANIFEST.json` in `validate_generation`).
  Test: the selection's hash equals the hash of the bytes on disk.
- **Control names and paths before any write.** The kind is checked as a plain name before a
  path is formed; the staging must be a real directory with no control file in it (a staged
  `MANIFEST.json`, symlink or not, is refused before anything is written, the outside sentinel
  untouched); the final path is built from the resolved data root and a symlinked kind directory
  is refused; `pinned` resolves under the data root itself and refuses a symlinked or redirected
  generation; the manifest's identity is bound to the directory's name and its kind to the
  parent directory's name (a staging directory is validated with its future identity named).
- **Parent provenance.** A parent passed as an object is validated again on disk, compared with
  the object's manifest hash and file list, required to be of the same kind and to lie under the
  data root, all before any link is made. Tests: a parent changed after being read is refused and
  nothing is linked; a parent of another kind is refused.

A failed validation after the manifest was written removes that manifest again (it is the
publisher's own file), so a retry does not trip on it. Not taken: nothing. Open: the rest of the
generation contract (receipt and recipe references, components over a selection, the drivers'
explicit legacy mode, the migration command), maintainer B's re-review.
