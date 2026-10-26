# Instruction tuning: the first test (contract as built)

Status: contract drafted by maintainer A 2026-10-25 (CONTRACT.md, export 0ddce265), accepted with five
amendments by maintainer B (review of 6080029, b4e583f); the CPU code returned for changes by maintainer B's
early review (a62ff1e, e9d6d8e; review instruction-code-e178dab) and reworked. This file is the
amended contract the code implements (`vesper/instruct_data.py`, `vesper/instruct.py`,
`vesper/instruct_panel.py`, `vesper/config.py` `InstructConfig`, `configs/instruct-dolly.yaml`,
`configs/instruct-panel.json`). The GPU trial is a separate bounded go after a full-lifecycle
preflight on this box's card.

Every setting below is a field of `configs/instruct-dolly.yaml` (typed by `InstructConfig`);
the optimizer's betas, weight decay by role and gradient clip are the source run's own.

## Question

Does supervised instruction tuning make the days-run model (step 427000, 64 M parameters) follow
simple instructions measurably better than the unchanged checkpoint? The project owner's goal is
a model he can talk to and give tasks; this is the smallest honest test of that direction at this
size, not a claim that 64 M parameters suffice.

## Data

- databricks-dolly-15k, revision bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a, file
  `databricks-dolly-15k.jsonl` sha256 2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec,
  15,011 records. License CC BY-SA 3.0, attribution to Databricks and to the Wikipedia
  contributors whose text the contexts quote; any derived dataset keeps that license and this
  description of the preprocessing.
- The file is read once, hashed and parsed from the checked bytes; a record's id is its line.
- Groups: two records are linked when their canonical instructions are equal, or their canonical
  contexts are equal and not empty; a group is a connected component. Canonical form:
  `keys.normalize_text_v2`, leading and trailing line feeds removed, `casefold`. Exact-text
  grouping only: no semantic decontamination is claimed.
- Split: blake2b-64 of the group id keyed `dolly-split-v1`, below 0.90 train, below 0.95
  validation, else test. Whole groups only. The test split is reserved from evaluation and
  selection; deterministic parsing and filtering read its bytes, nothing else does before
  `final`.
- Membership, groups and counts per split and category before and after the filter are written
  with every run (`data.json`).

## Serialization

    Instruction:
    {instruction}

    Context:            (only when the record has a context)
    {context}

    Response:
    {response}
    <end>

- Three segments encoded separately and concatenated: the prompt prefix, the response body, the
  terminator `"\n<end>"` (four ordinary ids for the frozen tokenizer 65f54642, [198, 27, 450,
  29]; `<end>` alone is three; the tokenizer has no special tokens). Whole-string encoding would change the prefix's ids when a response starts with line
  feeds (maintainer B's boundary probe); the prefix trained on is exactly the prefix given at inference.
- A field containing the literal `<end>` is refused and counted (none in this revision).
- An example with more than 1024 inputs is dropped and counted (214 records), never truncated.

## Training

- Every parameter except the halting head, which is frozen, left out of the optimizer and
  checked unchanged after training.
- The existing fixed objective at T = 8: a dense forward, every pass's output scored with equal
  weight (`Vesper.all_pass_objective`, condition `fixed`), zero ponder, no halting gradient.
- Loss targets: the response body and the terminator, as SHIFTED targets (the last prefix
  position predicts the first response id); headers, prompts and padding carry no loss. The loss
  is pooled over each batch's real labels.
- Every row is one example from cold states created inside the training step, positions from 0;
  nothing is carried between examples, batches or graph replays.
- Batches: length buckets 128/256/512/1024, 8192 positions per batch (filler rows complete the
  last batch of a bucket), order a keyed function of (seed, epoch).
- One setting: a fresh AdamW (betas, weight decay by role and the gradient clip as the source
  run's configuration), peak 3e-5, 100 warm-up steps, cosine to 10 % (`learning.warmup_cosine`,
  the trainer's own schedule), three epochs. Captured as CUDA graphs on the GPU (one per batch
  length), deterministic algorithms on.

## Evaluation

- Every epoch, the unchanged checkpoint as epoch 0, on the validation split at fixed depth 5:
  token-pooled body NLL (primary) and terminator NLL (apart), per example sums kept.
- Selection: the lowest pooled validation body NLL among epochs 1..3, ties to the earlier. All
  epochs' weights are kept. Validation intervals after this choice are descriptive.
- Paired comparisons resample whole groups with their loss sums and counts (2000 resamples, 95 %
  percentile interval), so each resample recomputes a token-pooled NLL. Per part (body,
  terminator), only groups with targets of that part contribute and are resampled; a group
  without (an empty response, for the body) is counted apart; a part with no target at all is
  reported "unassessed" with no estimate or interval, never a NaN.
- Forgetting: the source and the selected weights on the days run's own held-out view with the
  same fixed-depth evaluator (`evals.static_profile`, depth 5), both recomputed (not the rounded
  25.26). The view is fixed before any training (`instruct.days_view`): the checkpoint's PINNED
  components (`train.resumed_components`; today's selection cannot move it), its finalization
  receipt checked against the checkpoint's identity, the first chunks of the seed-0 held-out
  stream at 8 slots, every document inside the accepted test pin 7a39ce6f
  (`halting_text.validation_chunks`), and the view's digest equal to the config's (7296447b:
  16 chunks, 140 documents, 130,940 labels, the count of the days run's own evaluations).
- Once, for the selected weights and the unchanged checkpoint together (`final`): the test split
  (paired as above); the scored panel; the 21 illustrations. Nothing is retuned afterwards.
  Before it claims its directory or reads a test example, `final` refuses (exit 2) a preflight
  run and any input that differs from the tuning run's: the source weights and identity, the
  config's bytes, the panel's bytes, the scorer's version; a missing or changed `data.json`, or
  one its own preparation of the dataset does not reproduce; selected weights that do not hash
  as recorded. The claim is exclusive (a second attempt exits 3) and records `complete` or the
  failure in its `status.json`. A preflight (`--limit-examples`) never reaches the test split.
- The checked source parameters and configuration are kept: every later model
  (the forgetting baseline, the reloaded epoch, the tuned model in `final`) is built from that
  checked snapshot and config, never by reading the files again; `final.json` carries the hash
  of exactly the result bytes it parsed; the checkpoint record the days view reads must still
  be the loaded one (maintainer B's review of eb4d9f9).

## What a result means

- Lower response NLL means a better fit to these answers. It need not change a greedy answer, so
  it is not instruction following by itself.
- The scored panel (`configs/instruct-panel.json` v3, 24 items: 12 grounded extractions, 12
  explicit output constraints, every one objectively checkable (no explanation is scored; the
  umbrella question is an illustration); expected answers, scoring rules and, per item,
  answers that must pass and answers that must fail (wrong content, extra text, wrong format)
  fixed before any output was seen; `instruct_panel.load_panel` refuses a panel whose rubric
  does not separate them; the scorer's version `instruct-scoring-v3` is bound into the run) is
  generated greedily at depth 5 from cold states, at most 160 new ids, stopping when the decoded
  text holds `<end>` (wherever BPE split or merged it); a length stop is reported apart. Paired
  pass counts (both, only base, only tuned, neither) and every answer are reported.
- A fit improvement with unchanged compliance is a fit-only result. A worse general-text loss or
  compliance is reported beside any gain, never netted against it.

## Not in scope

Chat formatting, multiple turns, preference tuning, adaptive stopping during tuning (fixed depth,
one change at a time), safety tuning, and a grid over settings.
