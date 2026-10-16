# Resume corrections and acquisition lifetime review

Date: 2026-10-15. Executed at f55bfb3bba9e70b456152ea9597991ed81eec0f8.
Review owner: maintainer B. Implementation owner: maintainer A.

The two prior behavioral counterexamples close: unsupported direct steps refuse
without advancing state, and CLI resume follows its saved pins after the active
selection changes. The new metadata resolver needs a single validated snapshot
and schema check before field access. Book acquisition still has cleanup gaps
before collection and after collection. These are changes requested, with scoped
positive evidence below.

Shared evidence: <evidence>/reviews/resume-driver-f55bfb3/. The prior
probe is rerun unchanged; new boundaries.py exercises only owned temporary data.
The old-schema builder executes archived 66b674a to create a genuine checkpoint.
Full CPU gate: 335 passed, four CUDA skips, clean union dependency audit. Maintainer A's
selected full GPU log at this exact tip reports 339 passes; not a maintainer B GPU run.
No live corpus, tokenizer selection or shared service was changed by this review.

## Prior findings closed

The same direct-step probe now reports documents_taken=0, original carried-state
objects retained, and no changed parameters. It still raises NotImplementedError
for local mode, through the shared validator at the start of step. The normal-run
and both objective-path refusals continue to work. The new repository regression
also checks live/dry streams, requires_grad flags and gradient absence.

The same real one-step CLI resume now exits 0 both before and after switching
only the active web generation. All original pins validate, and the current
selection stays at its new hash. The implementation resolves stored components
through pinned instead of selected_components. The repository regression adds
absent selection, missing/corrupted generations, malformed references and conflicting
metadata. This closes the original selection-dependent resume failure.

The tokenizer module's documentation now correctly describes named rules and
the words-v1 default. The accepted candidate bytes/rule/vocabulary/round trips
remain unchanged. No retraining claim is added here.

## 1. Checkpoint metadata needs one snapshot and an early version boundary

_checked_record hashes train.json with file_sha256 and then reads the path again
for JSON parsing. The public resumed_components therefore does not necessarily
return the record whose bytes matched manifest.json.

An independent fixture creates valid generation data and hash-matching checkpoint
metadata with a component share of 1.0. Its interleaving replaces train.json
immediately after the real hash function reads the original bytes. The replacement
has share 0.5. resumed_components returns 0.5 even though those bytes do not match
the manifest. This is an observed metadata-integrity failure in the public resolver,
not evidence of a training update using altered weights: the later restore can
still reject the changed checkpoint. It repeats the hash/parse split already
corrected for generation selection.

The same boundary also omits the schema check that restore performs later. With
a genuine schema-3 checkpoint produced by 66b674a, ordinary run/restore explicitly
refuses schema 3 versus 5. resumed_components instead raises KeyError 'generations'
before reaching that version check. The new CLI path calls this resolver first.
Thus the explicit old-format refusal is not preserved at every resume entry.

Use a shared checkpoint metadata reader that reads train.json bytes once, hashes
and parses exactly those bytes against a captured manifest, validates the supported
schema before accessing its fields, and returns the validated record for both
restore and resumed_components. Keep required-field/shape errors explicit rather
than allowing KeyError/IndexError to escape from cast-only parsing. Do not add a
silent conversion or fallback. Regressions should schedule the replacement
between hashing and parsing, and feed the real old-format checkpoint through the
resolver/CLI as well as ordinary restore. Preserve normal same-schema resume and
tampered-record rejection.

## 2. Book acquisition does not own staging across its whole lifetime

acquire_books creates the generation output staging, calls listing, writes the
listing, and chooses titles before entering its collection try/finally. After
collection it constructs and writes the receipt outside the publication cleanup
scope. Exceptions in these windows leave output staging behind.

Two local injected failures demonstrate the gap:

- A RuntimeError from listing leaves an empty books-publish directory.
- An OSError while writing the final receipt leaves the listing JSON and a
  3,617-byte parquet file under books-publish. There is no published generation.

The latter can retain a much larger completed book collection in a real run;
only the small fixture size was measured. This is retained staging/disk use,
not an observed partial generation activation or corruption of old data.

Give each acquisition operation one ownership scope covering staging allocation,
listing, collection, receipt/provenance construction, publication, and release.
Prefer a reusable context manager for the acquisition staging lifetime, used by
both drivers, so setup errors and later errors get the same cleanup. A renamed
published generation must survive scope exit, existing generations and other
operations' staging must remain untouched, and the original exception must not
be masked by a cleanup race. Keep the explicit legacy output ownership separate.
Test failure at allocation, listing, listing write, collection, receipt write,
provenance, and publication, plus successful publication and an interleaved other
operation. The existing per-loop collection guards cover only part of that scope.

## Driver checks that pass

A two-run local shard acquisition first publishes shard 0, then extends it with
shard 1. The child preserves unit order, the inherited parquet has the same inode,
and every byte in the original parent directory remains unchanged. Returned
receipts and manifest validate; staging is gone, and no selection is implicitly
created. An injected failure on the second shard of a later run leaves the
previous generations unchanged, publishes no additional generation, and removes
that run's staging.

A local books run selects one title, then a second run selects two with the first
generation as parent. The child has one replacement unit, two unique document ids,
a recorded parent pin, and no leftover staging. It does not append the earlier
one-title output as another input. These checks support the stated replacement
semantics; the failure windows above still need closure.

The migration command remains unimplemented at this reviewed tip. Maintainer A owns the
metadata/lifetime corrections and migration preparation; maintainer B re-reviews their
pinned evidence. The project owner owns a live migration and merges. This review does not
activate a new corpus or rewrite an existing selection.

## Outcome (maintainer A, 2026-10-15)

Both corrections taken on dev/collector, each reproduced from the fixtures in
<evidence>/reviews/resume-driver-f55bfb3/ (checksums verified) as a failing test first:

1. `train._checked_record` is one snapshot: the bytes of `train.json` are read once, hashed and
   parsed as the same bytes against the manifest, the record and its `recipe` must be mappings,
   and the schema is checked before any field is read; `restore` and `resumed_components` both
   read through it, and `resumed_components` refuses malformed `components`/`generations` by
   name (lists of `[name, share]` with a pin or null) instead of letting a `KeyError` or
   `IndexError` escape. Regression
   `test_the_resolver_parses_exactly_the_bytes_it_hashed_and_checks_the_schema_first`: the
   replacement scheduled right after the hash (maintainer B's interleaving) plus a count of the reads
   (exactly one), a schema-3 record refused by its number, no recipe, a non-list, a short entry
   and non-JSON bytes each refused as malformed. The ordinary run/restore refusal is unchanged.
2. `acquire._owned(data, label)` is the staging's lifetime for the whole operation (created on
   entry, released on exit however the body ends, another operation's staging untouched, the
   body's exception the one that propagates) and `_outputs(data, kind, layout)` yields either the
   run's own output staging or the legacy destination; both drivers run their listing,
   collection, receipt, provenance and publication inside it. Regression
   `test_a_run_owns_its_staging_through_every_failure_window`: a listing failure, a listing-write
   failure, a receipt-write failure (books and shards), a provenance failure, a publication
   failure, each leaving nothing of the run's and the neighbouring operation's staging intact;
   a success leaving nothing either; an allocation failure (the data root is a file) propagating
   as itself.
