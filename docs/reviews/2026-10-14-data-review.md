# Independent data/tokenizer review

Base: 24212a00e21ca2e7a5b965f33dfa930c05bdf374. Reviewer: maintainer B, 2026-10-14.
Status: changes requested. Source unchanged; the existing 24 tests pass.

## Reproduced contract gaps

1. Tokenizer.decode documents int32[n] but accepts floating and boolean IDs.
   A tokenizer trained on ["aa bb cc 123 test"] with vocab=300 decodes float32
   [0.9, 1.9] to the same text as integer [0, 1], silently truncating IDs.
   bool IDs also decode. Use the public tensor contract primitive for rank and
   dtype, retaining the range check; reject fractional, bool and NaN inputs
   through a deliberate contract rather than accidental conversion exceptions.
2. is_held_out hardcodes 5 per mille while training.held_out_fraction already
   exists in configuration. The configured split currently cannot reach the
   predicate. Accept an explicit validated fraction (or a typed split policy),
   with a stable threshold definition. Test 0, 1 and a nondefault interior
   fraction as permitted by that policy, plus fixed golden IDs and process
   stability. The existing 0.5% distribution check is good evidence for the
   current default, not for respecting configuration. This is not an observed
   train/evaluation leak; corpus filtering has not yet been integrated.
3. Default ByteLevel pre-tokenization uses its regex: "foo123bar baz" splits
   into foo / 123 / bar / space+baz. That does not implement the spec's stated
   spaces-only pre-tokenization with no digit special-casing. Choose and
   document the intended boundary rule, then test the pre-token boundaries
   independently; round-trip tests cannot distinguish these tokenizers.
4. Narrow or implement the tokenizer's vocabulary and text contracts. The tiny
   corpus above requests 300 but yields 268 IDs; the trainer's target is a cap,
   not a guarantee. Explicitly allow undershoot or reject it at the training
   boundary if exactly V IDs is required. Byte alphabet coverage gives valid
   Unicode text coverage, not an arbitrary-byte API: the wrapper takes str,
   and an unpaired surrogate is rejected by the backend. The spec's "any byte
   string" and the wrapper's "any Unicode string" need a precise domain. If
   arbitrary raw bytes are required, they need an actual byte-preserving API.

## Scaling issue

_Slot.append repeatedly concatenates all pending tokens, start flags and
segment IDs for each document. Instrumenting torch.cat while filling one
chunk from one-token documents gives:

| Chunk positions | Concats in slot accumulation | Input bytes copied |
| --- | --- | --- |
| 64 | 195 | 27,885 |
| 128 | 387 | 109,005 |
| 256 | 771 | 430,989 |

This is quadratic copying with chunk length for short-document packing, not
just an interpretation of a timing result. Use queued document spans with
cursors and fill a fixed chunk buffer once, or another bounded linear-copy
primitive. Retain the exact boundary and successor-token contracts. Add an
allocation/copy-work assertion that distinguishes the two designs; elapsed
wall time alone is too noisy for the regression. A corpus-scale stream should
not do repeated pending-buffer copies per document.

## What passed and what these checks miss

The existing exact one/two-slot examples, boundary masks, segment tracking,
empty source, short documents, process-stable split, Unicode round trips and
save/load checks all pass (24 tests). The short-document probe preserves
current outputs and measures the repeated copies; it does not claim a measured
training bottleneck. The decode counterexample is a contract failure on invalid
input; valid int32 decoding has not been shown wrong. Split integration,
checkpoint/resume, replay exclusion and the real corpus reader remain outside
this branch and were not validated by these unit tests.

## Outcome, appended 2026-10-14 by the other maintainer

All four contract gaps and the scaling issue were reproduced on 24212a0 and fixed on dev/data
in the commit that adds this file. `decode` holds its ids to `check_ids`, the model's one
integer boundary (rank, int32 or int64, range), so float, bool and NaN ids are contract errors.
`is_held_out(doc_id, fraction)` takes the configured fraction, compares the first 64 bits of
blake2b against `fraction * 2^64` (nested sets for nested fractions), accepts 0 and 1 and rejects
anything outside. Pre-tokenization is a split on spaces with the space joined to the following
piece and the GPT-2 regex switched off, exposed as `pre_tokens` and tested on boundaries, not
only on round trips. `vocab` is documented as an upper bound with a test that a small corpus
undershoots it; the text domain is narrowed to valid Unicode in the spec and the wrapper, with
unpaired surrogates rejected by a deliberate check. The packer keeps a queue of whole documents
with a cursor and writes each token once into the chunk buffers; a test wraps `torch.cat` and
`torch.stack` and asserts neither is called while packing 256 one-token documents.
