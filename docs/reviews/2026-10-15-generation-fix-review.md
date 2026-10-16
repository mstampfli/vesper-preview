# Generation fixes, resume selection, and tokenizer candidate

Date: 2026-10-15. Executed at 1725ca60e79a82ef562e18fc20ff8c2154068d5d.
Review owner: maintainer B. Implementation owner: maintainer A.

The eight previous generation counterexamples now pass independently. Both loss
paths and the run driver reject unsupported local mode, and a real schema-3
checkpoint is explicitly refused as incompatible with schema 5. Two boundary
issues remain: direct step rejection happens after state changes, and CLI resume
resolves today's selection instead of its stored generation pins.

Evidence: <evidence>/reviews/generation-fixes-1725ca6/ contains the
runnable probe, JSON results, expected resume-error log, old-checkpoint probe,
and full CPU gate log. All data fixtures are owned temporary files. The gate
passes 330 tests with four CUDA skips, including the new bf16 comparison; the
union dependency audit is clean. The selected maintainer A GPU log reports 334 passes
at this exact commit. That GPU run was not repeated by maintainer B.

## Closed original cases

- Two threads attempt selection updates from the same predecessor. The first
  pauses inside the locked transaction; the second waits, then refuses its stale
  predecessor after the first commits. Exactly one succeeds.
- Replacing the selection file during reference validation returns the old
  parsed references with the hash of those same old bytes, not the new file hash.
- A traversing publication kind is refused before publishing outside data.
- A staged MANIFEST symlink is refused and the owned outside sentinel is intact.
- A pinned generation-root symlink outside data is refused.
- A parent corrupted after its Generation object was returned is refused before
  inheritance; its changed bytes are not republished under the old parent pin.
- Cross-kind inheritance is refused.
- A manifest identity differing from its published directory is refused.

The new learning.check_mode validator rejects both wake_objective and the sliced
Vesper.objective. Calling run with local mode and deliberately nonexistent input
paths raises NotImplementedError first and creates no output directory. This
closes the prior silent local-mode training through the normal run driver.

The checkpoint probe archives 66b674a and executes its trainer to produce an
actual schema-3 checkpoint. At the reviewed tip, resume raises ValueError naming
schema 3 versus 5 and the absence of a conversion, instead of a later missing
field. Inspection and the new regression place this check before torch.load.

The added CUDA sliced-loss comparison declares relative loss tolerance 2e-3 and
per-parameter relative gradient L2 tolerance 1e-2 (zero reference norm uses 1).
It compares one versus four slices in the stated fixture. Accept this as scoped
numerical comparison, not bitwise equality or all possible training layouts.

## 1. Direct step changes state before refusing local mode

The run driver now validates before work, but the exported step entry point does
not. The independent probe calls step with local configuration, a fresh Stream,
and saved model parameters/states. It raises the intended NotImplementedError
only after taking one document, replacing the group's recurrent states, and
zeroing loop.halting.w_h through the warm-phase setup. No optimizer update is
claimed. Catching the error and reusing these objects has already lost the
original stream/state boundary.

Call the existing shared check_mode at the beginning of step, before drawing,
GPU counter reset, halting setup or any other state change. Add a direct-step
regression that compares the stream cursor, carried states, parameters and
requires_grad flags before/after refusal, including a dry stream. Keep the
current normal-run and both-loss-path checks.

## 2. CLI resume uses the new-run selection

The agreed generation contract says an active selection chooses a new run, while
a resume resolves the exact references in its checkpoint/finalization without
changing that selection (2026-10-15-generations.md, amendment 1). The current
train.main calls components_for before examining the resume checkpoint. That
function always resolves --data through the active selection.

The probe creates real generations for all four components, finalizes them,
trains one step and resumes its checkpoint successfully through the CLI. It then
switches only the web generation in SELECTION.json, leaving the original files,
finalization and checkpoint intact. The identical resume command now exits 1:
check_inputs says the finalization was not taken over the newly selected web
file. All four original pins still validate. Calling run with the original
components resumes successfully, and the active selection remains at the new
hash. The failure is input resolution, not a broken checkpoint or missing data.

Provide one resume resolver that validates the checkpoint/finalization metadata
and resolves their stored references through pinned, retaining component order,
shares and manifest input order. New runs continue through selected_components.
Fail explicitly for absent/corrupt pinned generations or inconsistent stored
metadata, without falling back to the current selection or rewriting it. Test
old and new runs resuming while selection changes, and resume with the current
selection absent; neither should need to reset the global pointer. Preserve the
explicit legacy path for old layouts.

The 8175b35 bundle was also verified and its delta inspected. Its train.py and
sample.py changes only move environment/write_json imports to provenance.py;
the two paths above are unchanged. The executed reproductions remain pinned to
1725ca6. Driver and migration review at the newer tip remains separate.

## Tokenizer candidate checked

Shared artifact: vesper-s-32768-20261015T054353Z.json, 2,306,169 bytes.
SHA256: 65f546427412a22593e9148d4c0c606fb0c590d6f669e8ec24b78f79afd77d80.

The actual shared bytes match both checksum file and receipt. Tokenizer.rule
reads words-v1 from the serialized configuration; vocab_size is 32768. Ten
independent round trips include empty input, alphanumeric identifiers,
punctuation, leading/trailing/repeated whitespace, tabs and CRLF, combining
characters, multiple scripts, emoji, controls including NUL, all code points
0..255, long repeated input, code indentation, and the receipt's example. The
example produces 19 ids as recorded. Accepted as the identified mixed-corpus
candidate under the project owner's words-v1 decision, with the provenance limits below.

Receipt arithmetic agrees: 2,135,612 documents and 10,737,829,405 UTF-8 bytes,
411,165 bytes above the 10 GiB target under whole-document stopping. Every
component reaches its allocation; plan/trained counts and consumed digests match.
The receipt labels the source clean 107aba3 and duration 842.5 seconds. Those are
run-reported evidence, not an independent retraining. Private source files were
not read or rehashed. The selected compression report's combined holdout is
74,857,070 bytes; the configured target was 80,000,000. The reported 3.3% combined
token increase and 17.7% code decrease describe that sample, not model quality.

Documentation follow-up: tokenizer.py's module docstring still claims spaces-only
pre-tokenization. It must describe the named rules and current words-v1 default;
the older rule remains selectable. This is a stale contract description, not a
round-trip failure. No tokenizer switch or live corpus migration was performed
by this review. The project owner owns the live migration and merges; maintainer A owns the two
implementation corrections and documentation update, maintainer B their re-review.

## Outcome (maintainer A, 2026-10-15)

Both corrections taken on dev/collector, each reproduced as a failing test first from the probe
in <evidence>/reviews/generation-fixes-1725ca6/ (checksums verified):

1. `train.step` calls the shared `learning.check_mode` before the draw, the counter reset and the
   halting setup. The regression (`test_a_direct_step_refuses_an_unsupported_mode_before_touching_anything`)
   drives a live and a dry stream under local mode and checks the stream cursors, the carried
   states (identity and values), every parameter, every `requires_grad` flag, the halting head's
   values and the absence of gradients after the refusal.
2. `train.resumed_components(data, resume)` resolves a resume through the checkpoint's own pins:
   the record checked against the checkpoint manifest first, the stored recipe's order and shares,
   each pin validated by `generations.pinned`, the files in manifest order; a missing, changed or
   re-hashed generation, a missing pin and an inconsistent or tampered record are refused by name,
   the selection is neither read nor rewritten. `train.main` uses it for `--data --resume`; a new
   run still goes through `sample.selected_components`; `--legacy-layout --web --mix` is unchanged.
   The regression (`test_a_resume_resolves_its_checkpoints_pins_not_todays_selection`) is the probe:
   a one-step checkpoint resumed through the command line, the web generation switched, the same
   command resumes, the pointer stays at the new hash; then the selection absent, a pinned
   generation missing, a pinned file changed and restored, and crafted records (a pin of another
   kind, one pin short, no pins, a tampered record, no checkpoint), each refused by name.

The tokenizer module's docstring names the rules and the current default (`words-v1`, `spaces-v1`
selectable) and points at the pretokenization proposal. The candidate's acceptance and its
provenance limits are recorded as stated above; no live switch.
