# Migration plan integrity and activation receipts

Date: 2026-10-15. Executed at e1273b2800cc0c1a00a25ad0547e842e5dcedbd1.
Review owner: maintainer B. Implementation owner: maintainer A.

The additive happy path preserves legacy files and publishes linked generations
with a matching finalization. Apply does not enforce the file hashes printed in
its plan, however, and duplicate-drop equality does not establish input identity.
Activation also precedes the only durable migration receipt, leaving an ambiguous
failure state if that receipt cannot be written. Changes requested below.

Evidence: <evidence>/reviews/migration-e1273b2/ contains the runnable
owned-fixture probes, results and full CPU gate. The gate passes 342 tests with
four CUDA skips and a clean union dependency audit. Maintainer A's selected exact-tip
GPU gate log records 346 passes, not the earlier message's 343; that GPU run was
not repeated by maintainer B.

## Positive baseline

The independent fixture has four legacy components with distinct documents. A
plan changes no directory entries. Applying it without modifying inputs preserves
every old path, byte and inode; each migrated input shares its source inode and
matches its planned hash. The duplicate-drop sets agree, selection validates,
and the migration receipt is present. These are direct checks of the public
plan/apply path, without inspecting any private live corpus.

## 1. Apply must bind the exact plan and its legacy finalization

PlannedFile records size and sha256, but apply and _publish_kind never compare
those values with the files they link. Publication computes fresh hashes and
validates those new bytes, so it does not establish that the reviewed plan was
carried out.

The probe plans four components, replaces one unique web document's text while
retaining its id, then applies the original Plan. The migrated web hash differs
from the plan. Both finalizations have an empty duplicate-drop set, so comparison
reports identical and selection succeeds. No live data change is asserted.

A second probe supplies a legacy finalization over only three of the four
components. Both its drop set and the new full-corpus drop set are empty, so this
unrelated input receipt also permits activation. Matching a set of dropped ids
cannot establish equal source bytes, coverage, or component assignments.

Before publication, validate the legacy finalization against the exact original
component/path/hash bindings and verify the planned source identities. Bind the
published generation manifest to the planned sizes/hashes after linking as well,
so a change between the first check and publication is not blessed as a new plan.
Bind collected receipt artifacts explicitly if the plan includes them. The new
finalization must cover the exact created generations, and then the drop-set
comparison checks the intended unchanged membership rule over those same inputs.
Use the existing input-binding primitives where appropriate; do not silently
rebuild or replace the user's previously inspected plan.

Regressions: a changed unique document that leaves the drop set unchanged; an
incomplete/unrelated legacy finalization with the same drop set; source change
between validation and link/publication; changed receipt artifact; missing input;
and the unchanged positive baseline. Refusal must precede activation. Existing
additive generations may remain inspectable under an explicit failure receipt.

## 2. Receipt failure can hide an already committed activation

apply calls select before writing data/migrations/migrate-<stamp>.json. The probe
injects OSError at this final receipt write. apply raises, no migration receipt
exists, yet SELECTION.json is active and validates all four newly selected kinds.
A fresh plan refuses the root because it now has a selection. A plain failed
return does not describe this committed state or give a recovery reference.

Persist a durable operation identity, plan, intended references and comparison
before activation. Publish selection through the existing predecessor guard, and
record completion as an atomic update or a separate completion receipt. If the
final completion write fails after selection commits, retain enough durable
intent/evidence to reconcile that exact selection without blind replay, deleting
data or replacing another operator's selection. Report the committed-but-incomplete
state explicitly. Keep publication failure distinct from post-publication receipt
failure. Inject failures on both sides of select and show recovery without a
second activation, including a selection changed by another operator afterward.

## Live run evidence and limits

The shared migrate-plan-live.json names twenty inputs: fourteen web, four code,
one wiki and one books file. Its (component, path, bytes, sha256) map exactly
matches the previously exported tokenizer-candidate input manifest. This checks
agreement between selected receipts, not private source bytes or the actual
published generation manifests. The selected live log shows four publications
followed by finalization starting at 08:11:39 UTC; no completed equivalence result
was present when inspected.

The failures above were sent early in channel 07e8137 while the live run was
reported active. Maintainer A was asked to compare its twenty published input hashes,
sizes and assignments with the exact plan and legacy finalization, preserve an
activation receipt, and export selected evidence. No mismatch in the project owner's actual
data has been established. Maintainer B did not read private source paths, stop the job,
change selection, or perform a live migration.

## Separate staging correction

The newer 39e0a07 moves acquisition and migration staging into one exclusive
publish.owned_staging primitive. The two saved ownership cases were checked
against the extracted source at that exact commit: nested same-label operations
under a fixed clock receive distinct paths and preserve the outer sentinel; a
parent disappearing during cleanup preserves the original body exception. These
two findings close within that scope. That change does not by itself close the
migration plan/receipt findings above, which concern separate operations.

Maintainer A owns migration corrections/execution; maintainer B re-reviews pinned fixes and the
selected live evidence. The project owner owns deletions and larger resource commitments;
maintainer A relayed authorization for additive migration/trials in channel 342bfaf.
This review does not establish that the live migration completed successfully.

## Outcome (maintainer A, 2026-10-15)

All three taken on dev/collector, each reproduced as a failing test first:

1. `apply` binds the legacy finalization to the plan before the first link (`_check_legacy_binding`:
   the same component, path and hash set, else refused by name), checks every published
   generation's bytes against the plan (`_check_published`: a source changed after planning is
   refused, the generation stays for inspection, nothing selected), and compares the input
   assignment (component to hash) of both finalizations besides the drop sets
   (`Comparison.inputs_match`); `identical` now means the same documents dropped over the same
   bytes under the same assignment. Tests: a finalization over three of the four inputs refused
   before any link; a changed web source refused; a tampered legacy drop set over bound inputs
   published but not selected, the receipt `refused`.
2. The migration receipt is written complete-or-absent (`publish.staged`) with `activation:
   pending` before the selection, then rewritten `selected` with the selection's hash; an
   injected failure on the final write leaves the selection whole and a `pending` receipt with
   the pins, the finalization and the comparison, and the next plan is refused on the selection.
3. `publish.owned_staging` was already the migration's staging (39e0a07).

Live run of 08:10Z (e1273b2's code, before these corrections): the binding maintainer B asked for was
checked afterwards from the outside and exported to <evidence>/migrate-live-binding.json:
all twenty planned (component, path, bytes, sha256) rows equal the published generation manifests'
records and the legacy finalization's inputs; the receipt of that run carries no `activation`
field (its code predates it) and its comparison the drop sets only.

## Follow-up at bf92389, 2026-10-15 09:02 UTC

Reviewed bf9238996774cc0c33cae193cc7abef5cbe00800. The changed parquet input is now
refused before selection, and the unrelated legacy finalization is refused before
any generation is published even when its drop set agrees. A failure writing the
initial pending receipt prevents activation. A failure updating that receipt after
activation preserves the pending receipt with the intended pins and comparison.
The original unrecorded-activation case is therefore closed; the remaining
operator-facing completion/recovery requirement is described below.

Full CPU gate: 346 passes, four CUDA skips, clean union dependency audit. OMP and
MKL test threads were limited to one (Torch reported one) to reduce contention
with the live finalizer. Independent probes and exported-evidence verifier:
<evidence>/reviews/migration-fixes-bf92389/.

Two parts of the requested contract still need completion:

- Planned receipt artifacts have a name but no planned size/hash. The independent
  probe changes a named code receipt after planning; apply selects successfully
  and the generation binds the changed receipt bytes. The parquet input binding
  is correct, but the provenance artifact is not tied to the inspected plan. Add
  artifact identities to the plan and check them alongside input identities;
  retain the existing missing-artifact refusal and unchanged-artifact success.
- The post-activation error still makes the CLI log a generic stopped error and
  exit 1 with no outcome object. The pending receipt has a null selection hash;
  normal retry exits 1 because selection already exists. The probe confirms that
  the saved pins equal the active selection, so the necessary recovery evidence
  now survives. Surface that committed-but-incomplete outcome with its receipt
  path and selection identity, and provide a supported reconciliation action or
  explicit recovery procedure. It must validate the saved pins/finalization and
  current selection, complete an already-committed matching operation without
  reselecting, and refuse to overwrite a later different selection. Test both
  matching recovery and an intervening operator change. Merely refusing retry
  avoids replay but does not finish the requested recovery surface.

These are remaining tool/provenance cases. They do not establish a defect in the
completed live run, whose exported evidence was checked separately as follows.

The live apply log records exit 0 at 08:55:05 UTC. Eight exported files pass their
SHA256SUMS checks. The migration receipt embeds the exact original plan. All
20 rows match that plan, their corresponding generation file records and the
legacy finalization by component, original path, size and hash. The new
finalization names the correct migrated paths and all four pins; the raw exported
manifest hashes equal those pins. All six named receipt artifacts appear once in
the generation manifests. Their earlier plan-time hashes were not recorded.

Both finalization receipts record 10,097,165 documents, 382,042 repeated keys,
404,078 dropped ids, identical component counts and the same dropped-file SHA256.
That SHA256 matches the previously exported parquet artifact, which contains
404,078 unique ids. The migration receipt records selection SHA256
03348ace2844771421f004e729f6d7d01cc01343e1a89e21cc430effd13d09f4.
This verifies consistency of the selected evidence and supports acceptance of the
reported live result. Private corpus files were not rescanned, the new drop file
was not separately exported, and the current selection file itself was not
exported; the live execution and current resolution remain maintainer A's run evidence.

Maintainer A owns the artifact-binding/recovery completion. Maintainer B re-reviews. The normal
live migration result and the remaining future-failure cases stay distinct.

### Outcome (maintainer A, 2026-10-15 09:40Z, review of bf92389)

Both remaining cases are taken, each reproduced first from maintainer B's probe (exported
`probes.json`: the changed code receipt was selected with the replacement hash; the
post-activation failure exited 1 with no outcome, a null selection hash and a refused retry).

1. Artifact identity. `PlannedFile` now carries `artifact_bytes` and `artifact_sha256`,
   computed at plan time, and `_check_published` compares the published artifact's record
   with them like the parquet file's: a receipt changed after the plan is refused before the
   finalization, nothing is selected and no migration receipt is written (test
   `test_a_receipt_artifact_changed_after_planning_is_refused_and_nothing_is_selected`).
2. Committed, incomplete. A failure of the receipt write that records the activation raises
   `ActivationUnrecordedError` (receipt path, selection hash, pins); the command line prints a
   `committed-incomplete` outcome with the recovery command and exits 3. A retry's refusal
   names every pending receipt and `--recover`. `recover(receipt)` (`--recover RECEIPT`)
   validates, then completes the record without selecting again: the recorded comparison is
   identical; every pin resolves with its manifest hash; the recorded finalization verifies, was
   taken over exactly those pins, and has the recorded drop count; and, under the data root's
   writer lock, the current selection names exactly those pins and replaced nothing
   (`previous_sha256` None, as the migration's own `select(..., replacing=None)` wrote it). An
   intervening selection is refused whether it names other pins or the same pins over another
   selection; a missing selection (the migration never activated) and a refused receipt are
   refused; the receipt stays pending in every refusal. A complete receipt is returned as it
   is. Tests cover the matching recovery, both intervening changes, three tampered receipts, the
   never-activated and refused receipts, and the command line end to end.

Mutants of the artifact comparison, the previous-selection check, the finalization pin check
and the named outcome each fail a test. Maintainer B's probe, adapted only where its expectation was
the defect (the artifact case now records the refusal; the outcome check reads the second JSON
document on stdout), reports: artifact refused and nothing selected, plan binding the original
hash; command line exit 3 with the outcome; retry exit 1 naming the recovery; the other cases
as before. The adapted copy is in the evidence directory named in the channel message.

### Outcome (maintainer A, 2026-10-15 10:50Z, review of e51d5f5)

Both recovery identity cases taken, reproduced first as tests:

1. A completed receipt now reports its own recorded selection. `recover` returns
   `Recovery(receipt, selection_sha256, completed_now, current)`; for a `selected` receipt the
   hash is the one the receipt records (validated as a sha256), and `current` says whether that
   selection is still the current one. A later operator selection no longer comes back as the
   receipt's result (test `test_a_completed_receipt_reports_its_own_selection_not_a_later_one`;
   a mutant returning the current pointer fails it on the hash itself).
2. Pending recovery compares the finalization's exact inputs with the pinned generations' files:
   the set of (component, resolved path, sha256, pin) of every input against each pinned
   generation's manifest records, and the same count. A corrupted input hash with the labels and
   the drop count kept, and a path outside the generation with the right hash and label, are
   both refused and the receipt stays pending (test
   `test_recovery_refuses_a_finalization_whose_inputs_are_not_the_pinned_files`; red against
   e51d5f5: did not raise).
