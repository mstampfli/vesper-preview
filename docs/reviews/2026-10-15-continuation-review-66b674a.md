# Continuation, recomputation and generation contract review

Date: 2026-10-15. Pinned tree: 66b674a94ef99723c2b5be05f267637867e038f9.
The inspected staging, component binding, cursor continuation and recomputation
implementation is accepted for the scope exercised below. The generation concept
is supported, with contract amendments before implementation/migration is called
complete. This is not acceptance of the full training recipe or a main merge.

Evidence: <evidence>/reviews/continuation-66b674a/. Run probes.py
with PYTHONPATH=.:tests and the reviewed checkout's locked Python environment.
All independently executed tests use owned local fixtures, not private peer data.
Full CPU gate: 315 passed, three CUDA skips, clean union audit. The selected
maintainer A GPU gate log records 318 passes. HEAD and tracked source were unchanged.

## Implementation checks

The nested staging failure preserves and publishes the outer operation's data;
an unrelated leftover remains untouched; an empty body is refused. File-to-
component reassignment is rejected against the canonical finalization manifest.
The earlier two boundary findings are closed.

RawReader continuation matches the uninterrupted sequence at 52 cut points over
four seeds, multiple files, row-group boundaries and an empty file. Each seek
loads at most one row group. Mixer restore matches the remaining documents,
token balances and final digest at 24 cut points across three seeds, with split
filtering and empty text. Restoring two components reads at most two row groups
and performs no tokenization. Restoring an exhausted mixer remains exhausted.

An end-to-end seven-step run equals a run stopped after three steps and resumed
for four with recomputation enabled: final parameter maximum difference 0.
The resume runs with positive prefix replay forbidden. Evaluation's replay(0)
is allowed because it does no prefix work. My first instrumentation also rejected
that zero-count setup call; correcting that test guard was necessary and was not
a production-code defect.

The recomputation comparison includes nonzero carried history from a prior
chunk, mixed halting at passes 1/2/3, differentiable modulation, a masked document
boundary, every model parameter and the initial H/K/V leaf gradients. Both
float64 and CPU bfloat16 autocast give zero difference in the objective, outputs,
returned states and compared parameter/history/modulation gradients. Previous
micro-group logits, y and per_pass are now dead at the next forward. That closes
the lifetime finding; this CPU observation does not itself measure GPU savings.

Cursor restoration cost is independent of the tokenized prefix. Whole-run
startup still reads input metadata and validates file hashes; do not describe
the entire startup as literally one row-group read. Schema 3 and the recipe hash
make the new cursor/digest format a deliberate checkpoint compatibility change.

## Full-chunk smoke and evidence limits

The newly shared smoke-4096-recompute-log.jsonl ends normally at step 40 with
reason budget and 1,310,720 processed tokens. It uses four groups of two slots,
4096-token chunks, and recomputation. Its c2e47ad header's configuration hash
matches configs/vesper-s-smoke-4096.yaml in this checkout.

- Loss: 10.9015878072 to 7.6956393800; all logged losses finite.
- All four phases observed, checkpoints at steps 10/20/30/40.
- Held-out perplexities: 12134.2296, 5329.9494, 4106.3627, 3814.7727.
- Peak allocated memory: 6.916044235 GiB; median eight-pass step: 34.8354 s.
- Sum of step timings: 1334.6323 s. The outer log reports start 04:16:39Z,
  exit 0 and end 04:40:07Z; these are different timing quantities.

This establishes the reported completed full-chunk smoke at eight logical slots,
not the default 128-slot training batch or a GPU execution of the later cursor
schema. The full recipe also still lacks its final tokenizer and other modules.
No GPU run was repeated by maintainer B. The independent CPU checks above cover the
newer 66b674a code. Record which source version supplied each result.

Two documentation/evidence corrections remain:

1. CONTRIBUTING's --micro-chunks 2 example says eight passes stay under 5 GiB;
   the selected run for that layout peaks at 6.92 GiB. The approximately 4.20 GiB
   measurement was for one slot per micro-batch. State the measured layout.
   Recomputation retains boundary tensors and carried states across applications;
   internal activations are reconstructed on demand. Avoid an unqualified claim
   that all activation memory is bounded by one column independent of C and T.
2. The earlier measure-2048-recompute and measure-4096-recompute headers record
   commit 444b966 with recompute=true, but that commit has no recompute option.
   They ran a working tree ahead of its recorded HEAD. Preserve them as such;
   identify the actual source tree/patch or provide a measurement at a pinned tip.
   Extend run provenance with dirty status and a source-tree/patch identity so
   HEAD is not mistaken for the code executed. The complete c2e47ad smoke is
   separate evidence and already contains the recomputation implementation.

Moving memory and cursor work ahead of the data migration was sensible: those
were independently testable while the live layout decision remained open. The
reason for changing that order is now recorded, as the collaboration protocol asks.

## Generation proposal: supported structure

One complete directory per generation, immutable published content, explicit
selection, manifest order, source/receipt provenance and one common reader path
are the right structure. The implementation can be prepared and tested with
isolated fixtures while the live migration remains an operator action. The
following amendments make the proposed ownership and recovery contract concrete.

### Pin selection independently of the current ACTIVE pointer

ACTIVE selects inputs for a new run. A resume must resolve the exact generation
references stored in its finalization/checkpoint, even after ACTIVE selects a
newer release. Requiring the operator to roll ACTIVE back to resume an old run
would defeat the purpose of pinning. Refuse missing or corrupted pinned content;
an explicit change of inputs is a new recipe/fork. Test two runs pinned to old
and new releases while activation changes, and resume both without changing it.

Prefer one atomic corpus-selection manifest containing all four component
references. Four independently read ACTIVE files can otherwise select an
unintended mixture during a multi-component switch. Component generation
publication remains independent; activation publishes one chosen four-reference
set. Detect a stale concurrent activation rather than silently losing another
operator's update.

### Define the manifest and extension unit

Use one versioned schema interpreted by all readers/publishers. Separate the
ordered parquet input list from the complete artifact list: receipts must be
hashed and bound, but must never be passed to the parquet reader. Include kind,
generation identity, logical acquisition-unit identity, ordered data descriptors,
receipt descriptors, parent identity/hash when present, and source provenance.
The manifest does not hash itself; its hash is recorded by the selector and
downstream pinned records.

Specify append versus replacement conflicts. Recollecting a shard already in
the parent cannot silently add another active copy. A larger Standard Ebooks
selection normally includes previously selected titles: inheriting the old book
snapshot and adding the new whole snapshot repeats ids. Either merge by defined
logical units or replace that component snapshot explicitly. Test repeated shard
acquisition, overlapping title selections, replaced revisions and file-name
collisions before an expensive finalization discovers the ambiguity.

### Preserve immutability through hard-link imports and path validation

Hard-linking files between already immutable generations is appropriate. Linking
legacy files that still have writable aliases does not make them immutable.
Retire/quiesce legacy writers and define how those aliases become immutable, or
use a copy/reflink when that guarantee cannot be made. No writer may open an
inherited hard link for in-place replacement. State the same-filesystem condition
and fail explicitly when it is not met; do not silently start a large copy.

Validate generation references and manifest paths: relative names, canonical
containment under the selected kind/generation, no traversal or unintended
symlink escape, no duplicate/conflicting entries, expected file types, and
receipt/data hash agreement. Reuse a single containment/manifest boundary.
Activation validates the complete generation; readers verify the pinned manifest
identity. This is the proposal's actual input boundary even in a single-user root.

### Make migration additive and rollback reviewable

Prepare all four generations without removing today's paths, validate their
manifests and receipts, finalize the selected set, and compare the resulting
drop-set artifact with the historical one before activation. Existing readers
must not lose their paths midway through a run. Publish the new selection only
when it is complete. Preserve old paths/receipts until their readers and retained
checkpoints no longer require them; migration itself should not perform cleanup.

Make legacy use an explicit mode with a recorded selection. A missing ACTIVE
file must not silently trigger a fallback to a directory glob. The proposed
temporary web exception and the general fail-closed rule need that distinction.

The concrete migration command should show its source/destination manifest,
link/copy policy, required free space, existing-writer precondition, validation,
activation and rollback before the operator runs it. Retention considers active
selections, parent references, finalizations and checkpoints, not just which
generation is ACTIVE today. Any later deletion remains a separate deliberate
operation. Test failures at each publication/activation boundary and prove the
previous selection still reads correctly.

These are contract amendments, not claims about implemented generation code;
that code is still absent in the reviewed tip. Maintainer A owns implementation and
quantitative follow-up; the project owner owns the live layout/migration and tokenizer
decisions. No production edits, Vesper commits/pushes/merges, live data moves or
shared-service changes were made by this review.

---

## Outcome (maintainer A, 2026-10-15, taken on the tip named in the channel)

Accepted scope noted with thanks; the two corrections and the amendments are taken:

1. **CONTRIBUTING** states the measured layout with its number (two slots per micro-batch with
   four loss slices: 4.95 GiB at eight passes; 6.41 with one slice) and qualifies recomputation
   (internal activations rebuilt on demand; boundary tensors and carried states still scale with
   the columns and passes); the same qualification is in `Loop.forward`'s docstring and the
   registry row.
2. **Provenance.** `environment()` records `vesper_dirty` beside `vesper_commit` from this tip
   on, and the proposal's Measured section lists, for every measurement of the night, the tree
   it ran on (clean tip or a named tip plus the uncommitted change identical in content to the
   commit that followed). The two `measure-*-recompute` runs are recorded as 444b966 plus the
   recomputation change later committed as e86fca8; the full-size smoke is clean c2e47ad; the
   slice measurements are clean 13968b5.
3. **The sliced objective** (13968b5) measured at the full-size layout: 6.41 GiB with one slice,
   4.95 GiB with four, at eight passes, same step time, losses within 1.5e-4 relative (the GPU's
   band; the CPU test is exact).
4. **Generation contract amendments** are taken into docs/reviews/2026-10-15-generations.md
   (selection pinned independently of the pointer, one atomic four-component selection manifest,
   a versioned manifest schema separating inputs from artifacts, append and replacement per
   logical unit, immutability before hard links, one validation boundary, additive migration with
   the drop-set comparison and an explicit legacy mode). Code and fixtures are the next
   implementation block; the live switch and the layout are the project owner's.

Not taken: nothing. Open: the generation implementation, the project owner's tokenizer and layout decisions.
