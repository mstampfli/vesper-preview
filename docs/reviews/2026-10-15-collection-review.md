# Collection, books and sampler review

Current tip: d57b48ea8aaedbd93635055c19918e5b9d104297, header follow-up to
base ddcc4be06873ff081e07c6d3bc87fb86df582a59. Reviewer: maintainer B.
Time: 2026-10-15T02:36:41+00:00. Status: changes requested; do not promote the frozen candidate.

## Evidence that passed

Original same-repository duplicate fixture now retains one document on the
group's side; malformed input preserves the prior output with no retained
writer; hashing performs no whole-file read_bytes; a GPL row with null content
is rejected and counted; explicit GPL SPDX under repo MIT is rejected.
The deterministic, single-writer cases requested previously are corrected.

Full CPU gates: ddcc4be 261 passed / two CUDA skips; d57b48e 262 passed / two
CUDA skips. Both union audits pass. HEAD and tracked tree remained unchanged
through the Git-fixture tests. The header follow-up's focused tests also pass.
These tests do not establish the properties contradicted below.

## 1. Finalization must define one stable partition

A duplicate's first encounter currently chooses its group, hence its split.
With the same two-row parquet manifest, seed 0 puts the content in held-out;
seed 1 puts the identical content in training. This breaks a fixed evaluation
set when training reshuffles across epochs.

There is also a same-seed, different-view case. A balanced 20-byte training mix
emits web 'duplicate!' and wiki 'different!'; a held-out view of wiki emits
'duplicate!'. Both use seed 0. The shared seen-set only exists within one
traversal; independently finalizing component views does not preserve global
membership. This is a synthetic composed-API counterexample, not evidence of
an already-finalized real corpus leaking.

Compile a stable membership/canonicalization map from a pinned full manifest,
reconcile group/duplicate constraints, then shuffle or select components within
that map. A fixed shuffle seed alone does not fix the view problem. An explicit
component-union or conservative cross-split duplicate-exclusion policy can work;
first encounter in a consumer's traversal cannot define evaluation membership.

Identical code-v1/text-v1 bytes are also both retained because the version
prefix differs. If cross-source dedup includes exact byte identity, represent
that relation separately from source-specific normalized equality, or narrow
the claim. The current shared set proves neither kind universally.

## 2. Bound book paths and preserve block boundaries

A valid package href text/chapter.xhtml can be a symlink outside the checkout;
book_text reads its target. The fixture's outside sentinel is recovered as book
text. No private file was read. Defused XML and lexical '..' checks address
different boundaries. Resolve and enforce containment of the package and spine
paths, including symlink targets, before opening them.

Nested supported blocks in minified valid XHTML produce 'ParentChild' and
'OneTwo' from a nested list and paragraphs inside a table cell. Source formatting
must not supply the separators that the structural extraction promises. Walk
actual leaf blocks or emit structural boundaries, and pin minified nested cases.

Hash rank preserves relative ordering, not top-count membership under universe
growth: select one from [a_one,b_two] gives a_one; adding x_0 replaces it.
Keep the ranking if desired, but correct that invariance claim and preserve the
exact universe/selected list used for the run.

## 3. Publication and scratch ownership

The temporary filename .<output>.partial is shared. Two interleaved writers
use the same file; the first publishes, the second alters it then fails to
rename, and the first returned hash no longer matches. Use unique owned
temporaries and an explicit destination concurrency policy.

When replace() fails after writer close, the temporary remains because context
cleanup checks only writer-open. Track unpublished ownership separately from
an open handle. Compute the staged digest before publication, and clean up
unpublished temporaries even when the writer already closed.

acquire_shards leaves staging after a fetch exception. acquire_books retains
all earlier clones until the entire batch ends (fixture shows 0, 1, 2 existing
clones before three fetches). Both use the same data/staging directory. Give
each invocation its own scratch, clean it in finally, and release each title
once consumed; one invocation must not remove another's files. Preserve receipt
publication consistency as well as parquet atomicity.

## 4. Key versions and unsupported license evidence

The text-v1 implementation changed despite the unchanged-key claim. Executing
the historical function from f93840c on a leading NBSP retains it; the new
strip() removes it, yielding a different SHA256 under text-v1. Restore old v1
semantics or bump/rebuild keys. Preserved training text allows key rebuilding
without downloading source text again.

At d57b48e, repo-MIT rows with headers '***', 'MIT WITH Unknown-Exception' and
'MIT/GPL-3.0-only' are kept. The latter is truncated to MIT; the unknown exception
is discarded; an unparsed header produces no identifiers and is accepted.
Distinguish absent evidence from parsed evidence and unsupported/malformed
expressions. Validate the supported expression syntax and exception policy,
and reject unsupported explicit evidence rather than treating it as no conflict.
A bounded header check remains distinct from a full file-license audit.

## 5. Sampling eligibility and provenance

The documented iterator may end when a component is exhausted. That is useful
for a pilot. The freeze driver currently retains a caller-supplied 'frozen'
label for a requested 100-byte sample that supplies 16 bytes and zero from one
component. Add explicit quota/round-trip eligibility status and require the
recipe to be satisfied before final freezing; do not infer eligibility from an
artifact being written. This is a freeze-readiness requirement, not a claim
that partial iteration violates its own documented contract.

A finalized sample plan can establish per-component available/selected bytes
and exact rows before the expensive BPE run. Bind the training inputs to that
plan, with immutable or verified snapshots, so a manifest hashed before a run
cannot silently attest to files replaced while they are later read.

## Git incident and isolation

Maintainer A reported temporary remote collector commit 861af8d and recovery. Maintainer B's
collector ref was still f93840c before fetching; 861af8d is absent locally, and
the corrected branch descends from f93840c. No reset or force push by maintainer B.
The child git_output filter and test helper remove all GIT_* names correctly.

The hook's regex leaves names containing digits: GIT_TRACE2_EVENT,
GIT_CONFIG_KEY_0 and GIT_CONFIG_VALUE_0 remain in the isolated gate fixture.
Clear names using Bash's prefix enumeration rather than a partial character
class. Stop on a failed commit; a subject and clean tree alone are not sufficient
publication evidence when a hook can move HEAD. Check the intended resulting
commit/parent/tree before pushing.

All incident reproduction used temporary fixture repositories, and book-path
probes used an outside file within the same owned temporary directory. No
production source edits, Vesper commits/pushes, shared-driver changes or private
maintainer A file reads by maintainer B. The accepted model remains unchanged in scope; data
finalization and the new drivers remain unaccepted.

## Outcome, appended 2026-10-15 by the other maintainer

Taken in full; the point-by-point record is the outcome appended to
docs/reviews/2026-10-15-collector-review.md (second to fourth rounds), since this document and
the channel messages 1b9b3fd, adbc48d and 89c737e are one review. The candidate was not promoted.
