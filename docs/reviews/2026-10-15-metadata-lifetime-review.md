# Checkpoint metadata closure and exclusive acquisition ownership

Date: 2026-10-15. Executed at e39edd3fad2cefa2084b3f11c3e40d1aaede5069.
Review owner: maintainer B. Implementation owner: maintainer A.

The prior checkpoint snapshot/version failures and book listing/receipt cleanup
failures close. The new operation lifetime is the right scope; it still needs
exclusive allocation and cleanup that preserves the body's exception across a
parent-directory race. These remaining counterexamples use only owned fixtures.

Evidence: <evidence>/reviews/metadata-lifetime-e39edd3/. Full CPU gate:
337 passes, four CUDA skips, clean union dependency audit. The source receipt
from maintainer A reports 341 GPU passes; that GPU execution was not repeated by maintainer B.
The review also corrects docs/REPORTS.md to the measured batch size, partial model
scope, and CPU-exact versus observed GPU numerical continuation evidence.

## Closed cases

The public resumed_components reader now hashes and parses the same train.json
bytes. The independent interleaving replaces the file after the read returns its
bytes: the resolver returns the original share 1.0, performs exactly one metadata
read, and never returns replacement share 0.5 under the old hash. The changed
on-disk file no longer matches the manifest, but the returned snapshot does.

A genuine schema-3 checkpoint created by executing archived 66b674a is refused
explicitly as schema 3 versus 5 through both the ordinary run path and the new
resolver, before reading generation fields. The prior KeyError is closed.

The original injected listing and receipt-write failures now leave no staged
output. Successful shard extension, parent bytes, receipt hashes, hard-link
identity, book replacement and mid-fetch failure cleanup continue to pass. The
repository regression additionally exercises listing write, provenance and
publication failure while preserving a neighboring operation's staging.

## Remaining ownership cases

### 1. Timestamp names do not reserve exclusive ownership

_staging names a directory with label plus wall-clock timestamp. _owned then uses
mkdir(parents=True, exist_ok=True). Two same-label calls that receive the same
timestamp adopt the same directory instead of allocating separate owned paths.

With a fixed clock, nested _owned(data, "books") calls yield the same path. The
inner scope sees the outer scope's sentinel, and its exit deletes that sentinel
before the outer scope finishes. This is a deterministic collision test, not a
claim that real acquisitions collided on this host.

Use an exclusive allocation primitive, such as tempfile.mkdtemp under the staging
parent, for every operation. Preserve the operation's reserved root for its full
lifetime; releasing per-shard downloads must not release another operation's
reservation. Human-readable labels can remain prefixes. A frozen clock and
concurrent/nested operations should still receive different paths and leave each
other's sentinels intact. Merely adding more timestamp precision is insufficient.

### 2. Parent enumeration can mask the original failure

_owned documents that the body exception propagates because _release raises
nothing. However, only parent.rmdir is covered by OSError suppression. If another
operation removes the empty staging parent after is_dir returns true and before
iterdir enumerates it, _release raises FileNotFoundError.

The probe raises an original RuntimeError inside _owned and injects exactly that
parent-removal interleaving during scope exit. The observed exception is the
cleanup FileNotFoundError, not the original object. No other operation's data is
removed in this fixture; the defect is failure reporting and cleanup ownership.

Avoid an unguarded check/enumeration/removal sequence. Handle expected absent or
nonempty parent cases around the actual removal, preserving the body exception
and other operation directories. Keep unexpected cleanup failures visible without
misreporting a successful operation or replacing its primary failure. Exercise
both a disappearing parent and a concurrent new child, on normal and error exit.

## Scope and next ownership

Maintainer A owns these primitive corrections; maintainer B re-reviews. Maintainer A reported the newer
migration command and an additive apply in the channel while this review was
running; neither its implementation nor its live result is accepted by this
older pinned review. The selected plan/receipt will be reviewed separately.
No live corpus, selection, tokenizer, or service was changed by maintainer B.
