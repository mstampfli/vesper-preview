# Proposal: acquisition outputs as immutable generations with an active manifest

Date: 2026-10-15. Author: maintainer A (maintainer). Status: proposed, no code; the registry row
`acquire.publish_generation` is `specified`. The project owner decides the layout and the migration (it moves
the data directories he has on disk and costs one finalization, about thirty minutes); maintainer B
reviews the contract. The functions are listed before any code, as the process asks.

## The problem, as maintainer B stated it over seven rounds

An acquisition writes two files at two moments: the parquet (published by one rename through
`ParquetOutput`) and its receipt (written after). A reader lists whatever parquet files sit in
`data/mix/<kind>/` (`sample.components` globs `*.parquet`), so a parquet without its receipt is
read, a second acquisition into the same directory adds files beside the first (the repeated
document id of the fifth round came from exactly that), and the active set of inputs is implicit:
"what is in the directory now". The finalization pins paths and hashes, so a changed directory is
refused afterwards, but nothing says which set was meant. Each caller got its own check for this
(round seven); the shared contract was named as the next row instead.

## The contract proposed

1. **A generation is a directory that is complete or absent.** `data/<kind>/gen-<stamp>/` holds
   that kind's parquet files, their receipts and one `MANIFEST.json`: the file list in reading
   order with each file's bytes and sha256, the provenance (`vesper_commit`, the acquisition's
   arguments, the environment record `acquire.environment()` already writes into receipts) and,
   for a generation that extends another, the parent generation's name and manifest hash. The
   directory is assembled in this run's staging (`data/staging/<kind>-<stamp>/`, which already
   exists per run) and published by one rename (`publish.staged` on the directory), so no reader
   can see a generation without its manifest or a manifest without its files.
2. **A generation is never modified.** More shards of a kind make a new generation whose
   manifest lists the inherited files; the inherited parquet files are hard-linked into the new
   directory (same filesystem, no copy, the inode is shared and the file is never written again),
   the new ones are added. Deleting a generation is an operator action on a directory nothing
   points at.
3. **The active generation is a deliberate operator choice.** `data/<kind>/ACTIVE.json` names
   the generation a reader uses (`{"generation": "gen-<stamp>", "manifest_sha256": ...,
   "activated_at_utc": ..., "vesper_commit": ...}`), written only by
   `python -m vesper.acquire activate <kind> <generation>` through `publish.staged`, never by an
   acquisition or a training run. `activate` refuses a directory without a manifest or whose files
   do not match it. A reader that finds no `ACTIVE.json` stops: no generation is not "the files
   that happen to be there".
4. **Readers pin the generation they read.** `sample.components` resolves each kind through
   `ACTIVE.json` to its generation and lists the manifest's files in manifest order. The sample's
   receipt, the finalization receipt and (through the finalization's hash in `Identity`) every
   checkpoint record the generation path and manifest hash per kind. A run under another active
   generation is refused by `check_inputs` as today, and the receipt now says which generation it
   was taken over.
5. **The web component joins the scheme.** `data/web/gen-<stamp>/` with a manifest over the
   downloaded fineweb-edu files (one acquisition writes it from the existing download: no second
   copy, hard links), so all four kinds resolve the same way and the `--web` argument becomes the
   data root like `--mix`. Until the project owner decides, a plain directory is accepted with a warning in the
   receipt (`"generation": null`); the proposal is to drop that path once the migration is done.

## Functions (registry rows, all `specified` until written)

- `acquire.Generation(kind, path, manifest)`: a read generation; `files` in manifest order.
- `acquire.publish_generation(kind, staging, data, *, parent=None) -> Path`: writes
  `MANIFEST.json` into the staging directory (hashing every file there, linking the parent's
  files first when given), publishes the directory by one rename to `data/<kind>/gen-<stamp>`;
  the stamp is reserved like the finalization receipt (exclusive), a failure removes the staged
  directory and publishes nothing.
- `acquire.activate(kind, generation, data) -> Path`: verifies the manifest against the files,
  writes `ACTIVE.json` by rename; refuses a mismatch; the operator's command.
- `acquire.active_generation(kind, data) -> Generation`: reads `ACTIVE.json`, verifies the
  manifest hash it names, returns the generation; `ValueError` when absent or mismatched.
- `sample.components(data) -> list[Component]`: the four kinds through `active_generation`.
- `corpus.finalize` and `sample.freeze` receipts gain `"generations": {kind: {"path", "manifest_sha256"}}`.
- `acquire.main`: `acquire shards|books` publish a generation and print its path (they do not
  activate); `acquire activate <kind> <generation>`; `acquire extend <kind> <parent> ...` for
  more shards.

## Threats (STRIDE over the new boundary: the data root is written by acquisitions and read by runs)

- Tampering: a file changed inside a generation is caught by the manifest hash at `activate` and
  by `check_inputs` against the finalization at every run (as today); the manifest adds the check
  at activation, before any run depends on it.
- Spoofing: a directory planted under `data/<kind>/` is not a generation until `ACTIVE.json`
  names it, and `activate` verifies the manifest; a planted `ACTIVE.json` pointing at a
  consistent planted generation is the operator's own mistake, recorded with time and commit.
- Repudiation: `ACTIVE.json` records when and under which commit a generation was activated;
  every receipt downstream names the generation.
- Denial of service: staging under `data/staging/` is released on failure (today's rule), the
  free-space floor (`ensure_free`) runs before an acquisition, hard links keep extensions from
  doubling the disk.
- Information disclosure, elevation of privilege: nothing new crosses a trust boundary; the data
  root is local and single-user.

## Migration (the project owner's call)

Move today's `data/mix/{code,wiki,books}/*` into one generation each with a manifest written from
the present files (their receipts stay beside them), write the three `ACTIVE.json`, write the web
generation over the fineweb-edu download, then finalize once more (about thirty minutes; the
dropped set is expected identical to the two receipts of 2026-10-15, since the files do not
change) and sample from that finalization. The `old-*` directories under `data/mix/` are left as
they are. Nothing is deleted.

## Tests that reject a wrong version

A generation whose publication fails midway leaves no directory and no `ACTIVE.json` change; a
reader with no `ACTIVE.json` refuses; `activate` refuses a manifest mismatch and a directory
without a manifest; an acquisition never changes `ACTIVE.json`; an extension lists the parent's
files first and shares their inodes; `components` lists the manifest's files in manifest order
and ignores a parquet file in the directory that the manifest does not name; the finalization
receipt names the generations and `check_inputs` refuses a run under another active generation.

## Amended after maintainer B's review of 66b674a (docs/reviews/2026-10-15-continuation-review-66b674a.md)

Maintainer B supports the structure and amends the contract before any migration is called ready; the
amendments are taken into the proposal as follows, and they shape the code, which is still absent.

1. **Selection is pinned independently of the current pointer.** The active selection chooses the
   inputs of a *new* run. A resume resolves the exact generation references stored in its
   finalization and checkpoint, whatever the selection says now; a missing or corrupted pinned
   generation is refused, and an explicit change of inputs is a new recipe, a fork. There is one
   atomic corpus-selection manifest naming all four components (`data/SELECTION.json`, published
   by `publish.staged`), not four `ACTIVE.json` files that could be read mid-switch; a stale
   concurrent activation is detected (the manifest records the selection it replaces) rather than
   silently overwritten. Test: two runs pinned to an old and a new selection while the selection
   changes, both resumed without touching it.
2. **One versioned manifest schema.** `MANIFEST.json` separates the ordered parquet inputs (what
   the reader opens) from the complete artifact list (receipts, listings, hashed and bound, never
   passed to the reader); it names the kind, the generation identity, the logical acquisition
   units (shard indices, title selections), ordered data descriptors with bytes and sha256,
   receipt descriptors, the parent identity and hash when extending, and the source provenance
   (`environment()` with its dirty flag). The manifest does not hash itself; the selector and the
   downstream receipts record its hash.
3. **Append and replacement are defined per logical unit.** Recollecting a shard the parent holds
   cannot add a second active copy; a larger Standard Ebooks selection replaces the books snapshot
   explicitly (its titles overlap the parent's) rather than inheriting and duplicating ids. Tests:
   a repeated shard, overlapping title selections, a replaced revision, a file-name collision, all
   refused or merged as specified before a finalization meets them.
4. **Immutability is a precondition of hard links.** Hard links only between already immutable
   generations; a legacy import quiesces and retires the writable paths first or copies (reflink
   where the filesystem offers it); the same-filesystem condition is checked and a failure is
   explicit, never a silent large copy. No writer opens an inherited link for in-place replacement.
5. **One validation boundary.** Generation references and manifest paths are validated once, at
   publication and activation and again when a reader pins a manifest: relative names, canonical
   containment under the generation, no traversal, no symlink escape, no duplicate or conflicting
   entries, expected file types, receipt and data hashes as recorded (`books._contained` is the
   containment primitive to reuse, not a second one).
6. **Migration is additive and reviewable.** All four generations are prepared without removing
   today's paths, validated, finalized, and the resulting drop set compared with the historical
   receipt (expected identical) before the selection is published; today's readers and retained
   checkpoints keep their paths; the migration deletes nothing, and any later deletion is a
   separate deliberate operation whose retention considers active selections, parents,
   finalizations and checkpoints. Legacy input is an explicit mode with a recorded selection, not
   a fallback when the selection manifest is missing (the web exception of item 5 above is
   withdrawn in favour of this). The migration command prints its source and destination
   manifests, link or copy policy, required free space, writer precondition, validation,
   activation and rollback before the operator confirms. Tests fail each publication and
   activation boundary and prove the previous selection still reads.

Functions, revised: `acquire.publish_generation(kind, staging, data, *, parent=None, replaces=())`,
`acquire.select(data, references)` (writes the selection manifest), `acquire.selection(data)`,
`acquire.pinned(reference) -> Generation` (what a finalization or checkpoint stores),
`acquire.validate_generation(path)`, `sample.components(selection)`; the finalization receipt
and `train.Recipe` carry the four generation references and manifest hashes. Code and fixtures
can be prepared in isolation; the live switch stays the project owner's, as does the layout.

## First unit landed (maintainer A, 2026-10-15, after the amendments)

The module is `vesper/generations.py` (not `acquire`, which is at its function limit and stays the
driver): `Unit(name, inputs, artifacts)`, `FileRecord`, `Generation` (kind, identity, path,
units, files, parent as identity and manifest hash, provenance with the dirty flag, manifest
sha256, `inputs` in reading order), `publish_generation(kind, staging, data, *, units, parent,
replaces)` and `validate_generation(path, *, expected_kind)`, with `MANIFEST_SCHEMA = 1`
separating each unit's parquet inputs from its artifacts. Publication writes the manifest into
the acquisition's own staging directory, validates it as a generation, and renames the directory
onto `data/<kind>/gen-<stamp>` in one step (`publish.publish`); an extension hard-links the
parent's units first (collisions refused unless named in `replaces`, a replaced unit the parent
lacks refused), so the inode is shared and no copy ever starts (a cross-filesystem link raises).
Validation is the one boundary: plain names, regular files, no symlinks, containment
(`publish.contained`, moved from books so both share it), sizes and hashes, units named once,
every file listed and nothing unlisted. Tests: tests/test_generations.py (round trip, tampering
in four forms, refused units, extension by hard link, collisions and replacement).

Not yet written, in order: the corpus-selection manifest (`select`, `selection`, `pinned`), the
generation references in the finalization receipt and `train.Recipe`, `sample.components` over a
selection, the drivers publishing generations with the explicit legacy mode, the migration
command with its printed plan. The live switch and the layout remain the project owner's.

## Second unit landed (maintainer A, 2026-10-15)

`select(data, references, *, replacing)`, `selection(data)`, `pinned(data, ref)`,
`reference(generation)`, `Reference`, `Selection`: one atomic selection over all kinds
(`data/SELECTION.json`, schema 1, written by one rename), every referenced generation validated
first, a repeated kind refused, a stale switch refused by comparing `replacing` with the current
selection's hash, the predecessor's hash recorded; a resume resolves its stored references through
`pinned`, whatever the selection says now, and a missing selection is an error rather than a
fallback to the files present. Tests cover the first selection, a switch and a stale switch, the
old generation pinned after the switch, tampering seen through both `pinned` and `selection`,
missing and mismatched references and a traversal name. Still to write, in order: the references
in the finalization receipt and `train.Recipe`, `sample.components` over a selection, the drivers
publishing generations with the explicit legacy mode, the migration command with its printed plan.

## Third unit (maintainer A, 2026-10-15, after maintainer B's review of 5726c87)

The boundaries maintainer B found are closed: compare and publish under the data root's writer lock,
control files read once with the hash of the bytes parsed, every path-controlling value checked
before a path is formed or anything written, the manifest's identity and kind bound to the
directory, the parent validated again on disk and required to be of the same kind. Details and
tests in docs/reviews/2026-10-15-objective-and-generation-review.md, Outcome.

## Fourth unit (maintainer A, 2026-10-15): the pins travel

`Component.generation` carries the selected generation's pin (`kind/identity@manifest-sha256`);
`sample.selected_components(data)` resolves the four kinds through the selection and `pinned`
(inputs in manifest order); `sample.components_for` makes the command line name its inputs one
way only (`--data ROOT`, or `--legacy-layout --web --mix` explicitly; anything else refused, never
a fallback); the finalization receipt's rows and `Finalization.inputs` carry the pin and
`check_inputs` refuses the same files under another pin; `train.Recipe.generations` carries the
pins into the checkpoint identity (schema 5). The drivers' generation mode and the migration
command remain.

## Fifth unit (maintainer A, 2026-10-15): the drivers publish generations; the legacy layout is explicit

`acquire_shards` and `acquire_books` take `layout="generation"` (the default) or `"legacy"` and
return `AcquireResult(receipts, generation)`. Under the generation layout every output of a run is
collected into the run's own output staging (`data/staging/<kind>-publish-<stamp>`) and published
together by `publish_generation` as `data/<kind>/gen-<stamp>`: a shard run extends a `parent` by
hard link (its units first, in its order), a books run replaces the parent's unit, since a larger
selection repeats its titles; a failure publishes nothing and releases the staging, and after a
publication the staging parent is removed only when empty (a parent another run refilled between
the check and the removal is left alone). Under `--legacy-layout` the outputs land in `data/<kind>/`
as before and a parent is refused. Publication is not selection: the command line prints the pin
and `python -m vesper.generations select --data ROOT --ref kind/identity@sha256 ...
[--replacing SHA]` is the operator's separate step, `show --data ROOT` prints the current selection
with every generation validated. `environment()` and `write_json` moved to `vesper/provenance.py`,
imported by acquire, generations, sample and train, so the receipt environment is one function and
generations no longer imports the driver. Remaining: the migration command (additive, a printed plan,
the drop-set comparison, no cleanup; the project owner confirms the plan).

## Amendment 1 implemented (maintainer A, 2026-10-15, after maintainer B's review of 1725ca6)

The command line now reads its inputs by the kind of run: a new run under `--data` resolves the
selection (`sample.selected_components`); a resume under `--data` resolves the checkpoint's own
pins (`train.resumed_components`: the stored recipe's order and shares, the record checked against
the checkpoint manifest first, each pin through `pinned`, the files in manifest order), so a run
pinned to an old selection resumes while the pointer moves on, and with the selection file absent;
a missing, changed or re-hashed generation, a missing pin and an inconsistent or tampered record
are refused by name, never answered with today's selection, and the pointer is never rewritten.
Maintainer B's probe (a one-step checkpoint, the web generation switched, the same resume command) is the
regression test. The explicit legacy layout is unchanged.

## Migration command written (maintainer A, 2026-10-15)

`vesper/migrate.py` (`python -m vesper.migrate --data ROOT --web DIR --mix DIR
--legacy-finalization RECEIPT --out DIR [--apply]`) implements the Migration section as amended:
the plan is the exact file set a legacy run reads (`sample.components`), with sizes, hashes, the
unit each file becomes and the receipt bound beside it; printed first, it writes nothing. `--apply`
hard-links every kind's files into a staging of its own and publishes one generation per kind (one
inode, the old path untouched; a link that fails publishes nothing for that kind), finalizes once
more over the generations and compares the dropped set with the legacy finalization's: identical,
the selection is written; different, nothing is selected and the counts are reported. A root that
already holds a selection is refused. No cleanup. The live run on the project owner's directories is the next
step: the printed plan goes to the channel before `--apply`.
