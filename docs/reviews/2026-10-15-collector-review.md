# Collector review and model revision acceptance

Collector: f93840c43f3021d527e29954840616a2b7a5fd8a.
Model included unchanged: 1e0211ae428ec14470e1e091da0d7f72a76688b0.
Reviewer: maintainer B. Time: 2026-10-15T01:29:53+00:00.
Model accepted for its stated scope. Collector: changes requested.

## Model and gate evidence

The exact former counterexamples now give zero excluded-logit gradients,
finite two-column output and weight derivative 1, and half/bf16 RMS outputs
in the input dtype with float32 gains. The first-update loss assertion now
matches its name. Whole/split model logits agree within 3.11e-15, 39 gradients
within 1.78e-15; future-token mutation has zero effect on prefix logits.
The model source files match 1e0211a exactly in this collector checkout.

Full CPU gate: union audit reports no known vulnerabilities; 234 tests pass,
two CUDA cases skip. Maintainer A's selected shared GPU full-gate log reports 236
passes. Maintainer B read that receipt, not another maintainer's private files.

## 1. Staging is not yet a safe training input

The collected parquet contains group_key and dedup_key, but iter_raw_documents
reads only id/text and iter_documents splits on id. Two identical code files
in the same repository produce one training document and one held-out document
at fraction 0.5. Both rows have identical group and dedup keys in the output.

Enforce group/dedup finalization before training-split iteration, or have the
reader reject unfinalized collected artifacts. Do not freeze the mixed tokenizer
through the existing ID-only path. The current output-reader test only proves
rows can be decoded, not that the split/dedup contracts survive composition.
This fixture proves an unsafe API path, not that a final real corpus has already
leaked. Cross-shard/source duplicates and group relationships need one explicit
policy in the finalizer; per-shard keys alone do not apply that policy.

## 2. Publish only complete output and close on errors

_Output opens/truncates the final destination before input validation. A source
with missing columns overwrites a previously valid output with a four-byte
header; keeping the exception alive keeps the output descriptor open. A separate
same-input/output fixture also truncates the input and then fails.

Use a temporary sibling and atomic publication after successful completion;
close and remove temporary artifacts on every failure path; preserve a prior
valid output; reject input/output aliases. Pin failure before the first group,
a later-group error, early cancellation and input/output aliasing. Do not rely
on destructor timing to close a writer.

## 3. Keep hashing bounded

close() calls Path.read_bytes on the completed output, allocating its entire
contents. The instrumented probe observes a bytes allocation exactly equal to
the output length. This defeats the claimed row-group working-set bound even
though the table processing streams. Use a reusable bounded file-hash helper,
for example a chunked digest over an opened file. This is an algorithmic memory
finding, not a measured full-corpus throughput result.

## 4. Apply the label filter before validating rejected payloads

A GPL-labelled row with content=None aborts the whole shard, before it can be
counted as rejected. The function documents TypeError on non-string fields in
kept rows. Filter labels first, then validate only retained payloads, or state
and deliberately test a different fail-closed contract. A rejected row's
unneeded content need not become a Python string at all.

## 5. Exact policy and provenance boundaries

An MIT repository label plus explicit file header
SPDX-License-Identifier: GPL-3.0-only is currently kept. The function is a
repository-label filter, not a file-license audit; this example does not show
such a conflict exists in the real outputs. Before calling the final subset
permissive-only, implement the promised rejection of explicit conflicting
evidence, or keep these outputs labeled candidates for that later check.
Retain headers, labels and origin paths; do not assert blanket file clearance.

normalize_text_v1 turns CRLF into LF despite saying line endings stay, and
strips leading spaces rather than merely collapsing them. Define the exact
newline/whitespace policy and version/rebuild keys if its behavior changes.
Training text itself is preserved, which the passing tests check.

The selected real receipts report row/byte counts and output hashes; input
content hashes, collector code/policy version and the actual allowlist are not
included. Add those to the final provenance, especially while the allowlist is
being revised. Keep full source-relative shard identity, not just a basename.
Maintainer B checked the receipts' arithmetic but did not read their private output files.

## Allowlist proposal

Maintainer B supports adding cc0-1.0 and unlicense to the documented policy and code
at the same time, retaining the existing provenance/conflict qualifications.
Primary terms: [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en)
and [Unlicense](https://unlicense.org/). This is the proposed dataset-label
policy, not an assessment of every individual file. Record the used policy
in each receipt so the five-label and seven-label runs remain distinguishable.

The current first-shard counts are internally consistent: 102209 code rows =
66303 kept + 35906 rejected; Wikipedia reports 156289 kept and none rejected.
Those are selected receipts supplied by maintainer A, not an independent reprocessing
of the real shards. No production source edits or Vesper commits/pushes by maintainer B.

## Outcome, appended 2026-10-15 by the other maintainer

All five findings reproduced with maintainer B's fixtures as tests before the fixes, then applied on
dev/collector. (1) The reader finalizes: `corpus.select_documents` drops duplicates (the first in
the seeded order survives; a shared seen-set across components) and splits on the group key; every
split iterator goes through it, and the two-identical-files fixture now yields one document on the
group's side (test). (2) `_Output` writes a temporary sibling and publishes by rename in
`close()`; `abort()` or an exception closes the writer and removes the temporary file; input and
output are refused when they are one file (tests: failure before the first group, in a later
group, aliasing; a prior output byte-identical afterwards, no open handle, no temporary file).
(3) `keys.file_sha256` hashes in one-mebibyte chunks; the collector's hashes are computed with
`Path.read_bytes` forbidden in the test. (4) The label filter runs before any other field is read;
a GPL row with a null content is rejected and counted (test). (5) `spdx_header` names the licenses
an explicit header mentions and a file naming one outside the allowlist is rejected as
`header:<id>`, the expression kept in a `header` column (tests: GPL header under an MIT label
rejected, `MIT OR Apache-2.0` kept, `MIT AND GPL-2.0-or-later` rejected). `normalize_text_v1` is
pinned exactly as it behaves (leading and trailing whitespace per line removed, CRLF to LF) in
its docstring, the registry and the mixture document; the keys did not change. Receipts carry the
rule that ran (the allowlist in words), the input's sha256, the output's sha256, the
source-relative shard path (driver), the interpreter and library versions and the vesper commit.
The allowlist gains `cc0-1.0` and `unlicense` in the document and the code together. The collected
first shards are re-collected under this revision; the new receipts replace the old ones.

## Outcome, second to fourth rounds (channel 1b9b3fd, adbc48d, 89c737e), appended 2026-10-15

Finalization. The first-seen rule is gone. `corpus.finalize` decides membership once against the
pinned inputs: two reads in file order, every dedup key occurring more than once, the smallest id
survives, the others are the dropped set, written with a receipt (inputs by hash, rule, counts);
every reader goes through `select_documents` with a `Split` of that finalization, and the drivers
check the inputs' hashes before reading. Maintainer B's two-group fixture gives one membership under eight
seeds; the same-seed different-view case cannot occur because the finalization is one map over
all components, not a traversal's set. A duplicate across groups on different sides: the
survivor's group decides, the other copies are in no split (the conservative policy). The
cross-source claim is narrowed in the code and the documents: text keys deduplicate against each
other, code among code, exact-byte identity across key versions is two documents by design.
Keys: text-v1 is retired for text-v2 (`str.strip` both ends), because its two implementations
differed on a leading no-break space; every text key is recomputed. Books: the package and every
spine file must resolve (symlinks followed) inside the checkout; nested blocks are containers, so
minified `<li><p>One</p><p>Two</p></li>` gives two lines; the selection docstring and registry
row say what the ranking does and does not promise, and the receipt pins the listing. Writers:
each `ParquetOutput` owns a unique temporary (`.<name>.<pid>-<n>.partial`), a failed publication
removes it, and `abort` is idempotent (tests: two interleaved writers on one destination, a
refused rename). Drivers: a run's staging is released in `finally`, each title's clone after the
consumer has read it (test: the next fetch sees the previous clone gone). Hook: every `GIT_*`
name is unset through `"${!GIT_@}"`; the commit script pushes only after the commit returned
zero, HEAD moved and the tree is clean. SPDX: a header that is present but unreadable (`***`,
markup after the identifier, a `WITH` clause, `MIT/GPL-3.0-only` now splits on `/`) is
`unsupported` and rejects the file; absent evidence keeps the file on its label. Sample: the
driver reads the sample once as a plan before training, names the components short of their
budget, and a short run is labeled a candidate that is not freezable. The frozen tokenizer is not
promoted: the 10 GiB mixed sample's BPE training exceeded the box's 64 GB (one cgroup OOM kill);
the measured cause and the decision it needs are in the channel and the plans.

## Outcome, fifth and sixth rounds (channel 4352e5f, c0e6dfb), appended 2026-10-15

Finalize refuses a document id that occurs more than once across the inputs (an occurrence is its
id; two acquisition runs in one directory or two sources claiming one id are an input error) and
never overwrites a receipt with an existing stamp. The book walker treats any element with a block
anywhere below it as a container, so an inner list under an intervening `ul` and an inline element
directly inside a container keep their text as lines (tests with the fixtures). The sample driver
hashes the inputs again after training and refuses an artifact whose inputs changed under it;
eligibility follows the trained pass (its counts against the budgets, its agreement with the plan,
the round trip), not the plan. `ParquetOutput.close` hashes its own temporary before the rename, so
a writer's hash is its content even when another publication lands between the rename and the hash
(test with the race reproduced). The iteration histogram accumulates on the CPU whatever device
the counts come from (a GPU test, skipped without one); the perplexity is pinned against a hand
computation over the masked positions and leaves the model untouched. The training proposal:
resume restores the streams' packing state and the detached `H` and KV, not an iterator skip; the
step's loss is the token-weighted mean over the batch, equal to the unsplit batch's gradient; weight
decay is by role and name (`w_beta` is a gate); the no-retrieval run through phase 2 is labeled the
ablation it is; checkpoints pin the tokenizer, config and finalization identity; the positions
paragraph says continue, not restart. The pre-tokenization decision and the memory evidence are in
docs/reviews/2026-10-15-pretokenization.md for the project owner; the candidate is not promoted.

## Outcome, seventh round (channel a383583, review of 04beaf4), appended 2026-10-15

Taken: finalize reserves its receipt path exclusively before any work (`touch(exist_ok=False)`),
writes the dropped set to a temporary it owns, publishes it by rename and fills the reserved
receipt last; a failure releases the reservation and the published file (tests: a reserved stamp
refused and left untouched; the repeated-id refusal leaves no reservation). `ParquetOutput.close`
removes the unpublished temporary on any failure after the handle closed (test: a failing hash).
The sample driver binds the consumed bytes: a digest over every consumed text with its component,
taken in the plan pass and in the training pass, must agree for eligibility, beside the counts,
the round trip and the endpoint hashes (test with maintainer B's replace-and-restore fixture: counts and
hashes agree, the digests do not, the artifact is not freezable). The book walker is one recursive
traversal with a shared inline buffer flushed at structural boundaries; inline runs stay one
line, `br` breaks a line, a `noteref` is skipped wherever it sits with its tail kept (tests with
the fixtures, with and without source whitespace). The pre-tokenization proposal marks "within the
box" and the smaller-sample fit as estimates pending the measured peak RSS and lists the
measurements agreed. The perplexity test is named a reference comparison and a known-probability
check through the model's forward is added. Not yet taken, and said so: the shared publication
contract (exclusively owned generations for every acquisition output, data and receipt completed
and hashed together, one atomic publication, a deliberate manifest selecting the active
generations, readers pinning what they read) and the bounded-cost cursor resume. Both are the
next collection and corpus rows; until the first lands, the acquisition drivers are single-operator
tools and the finalization is the only pinned input.
