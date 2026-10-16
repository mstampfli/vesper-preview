# Architecture

## Theory of operation

Vesper is a language model that iterates a small core instead of stacking depth. A token enters
the loop, passes through `C` columns (distinct weights) up to `T_max` times, and halts when an
adaptive-computation head says its answer has settled. Each column holds a fast-weight state per
pass (a delta-rule matrix updated every token, never reset between chunks or turns), a bounded
attention window, a sparse retrieval path into an episodic store (the ledger), and a squared-ReLU
feed-forward. A consolidation pass (dusk) replays salient ledger entries into the slow weights
while the model is idle. One scalar per token, the modulator (normalized surprise), scales every
learning rate and decides what is salient. The design is docs/spec.md; the function-level
protocol is docs/registry.md.

## Codemap (what exists)

    vesper/
      contracts.py   check_shape, check_ids: the tensor-contract primitives every public function calls first
      config.py      typed, fail-closed loader; the only code that reads configs/*.yaml
      recurrence.py  the delta-rule scan, reference implementation (step form and blocked form)
      modulator.py   the single EMA pair and the normalized surprise per token
      quant.py       int4 packing of the fast state with per-row fp16 scales
      attention.py   the bounded-window attention with its rotary positions and KV window
      ffn.py         the squared-ReLU feed-forward
      embedding.py   tied input embedding and output head
      learning.py    the wake loss: next-token cross-entropy, ponder cost, loss weights
      column.py      one column: recurrent sublayer, attention, feed-forward, RMSNorm, its state
      exits.py       the halting experiment's exit distribution: hazard logits, exit log-probs,
                     expected per-pass loss, local/aggregate KL, the shared CDF exit accumulation
      loop.py        the pass loop with ACT halting and copy-forward over the columns, exit
                     rules (ACT, dense, trace, CDF); each column application's activations
                     kept or recomputed for the backward pass
      model.py       Vesper: embedding, loop, head; init_states, forward, wake_objective,
                     the all-pass objective of the halting experiment
      synthetic.py   the halting experiment's stage-1 task: per-sequence graphs whose useful
                     depth differs by token (hops, copies, a register read beyond the window)
      halting_experiment.py  the stage-1 driver (python -m vesper.halting_experiment): one
                     condition and seed trained on the CPU or on CUDA (fused scan, optionally
                     one captured step replayed), evaluated on the CPU under every state policy
      graphs.py      one training step captured as a CUDA graph and replayed (CapturedStep)
      halting_report.py  the stage-1 report (python -m vesper.halting_report): result records
                     pooled per condition into one Markdown page with the plan's verdicts
      tokenizer.py   byte-level BPE training under a named pre-tokenization rule (words-v1 the
                     frozen choice, spaces-v1 kept selectable), encode/decode, save/load
      data.py        Document, Chunk, Stream (slots of whole documents into chunks), is_held_out
      keys.py        normalize_text_v2, dedup_key, file_sha256, SequenceDigest: identity primitives
      publish.py     staged: a file or directory written beside its final name and published by
                     one rename, or removed; receipts, dropped-id files and checkpoints use it
      corpus.py      parquet to documents: finalize (membership decided once, by ids, against the
                     pinned inputs), Split and select_documents, block shuffle, the mixes by byte
                     share and by token share (Mixer: balances, pulled count, digest of the ids),
                     tokenization into Documents
      collect.py     the non-web components to compact parquet with provenance: code, Wikipedia,
                     books; published whole or not at all; receipts
      books.py       Standard Ebooks: title listing, hash-order selection, sparse clones, XHTML text
      acquire.py     the acquisition driver (python -m vesper.acquire): one unit at a time, receipts;
                     a run publishes one generation (the legacy layout is an explicit option)
      provenance.py  what every receipt, manifest and run header records about the run that wrote
                     it (environment with the dirty flag) and how a record is written
      generations.py acquisition outputs as immutable generations: the manifest schema, publication
                     of a staged directory by one rename, extension by hard link, the one
                     validation boundary, the selection manifest, pinned references and the
                     operator's select/show command line
      migrate.py     today's flat layout into generations (python -m vesper.migrate): the printed
                     plan, hard links, a second finalization compared with the legacy one, the
                     selection only when the dropped sets agree; nothing moved or deleted;
                     --recover completes a receipt whose selection is committed
      sample.py      the tokenizer's 10 GB mixed sample and the frozen tokenizer (python -m vesper.sample)
      evals.py       held-out perplexity and the iteration histogram, run at every checkpoint
      train.py       the training loop (python -m vesper.train): phase schedule, decay by role,
                     the token-weighted step over independent streams (a dry stream drops out,
                     the others train on), checkpoints with a manifest, recipe and layout, the
                     last completed step saved once at the end with its reason
    configs/         vesper-s.yaml: every constant, none in code
    stubs/           precise local stubs where a library's own hints degrade to Any
    tests/           one file per module; the scan equality and the delta property live here
    docs/            spec.md (design), registry.md (contracts and status), threat-model.md,
                     PRIMITIVES.md, reviews/ (dated review records with their outcomes)
    gate.sh          the build gate; CI runs exactly this file
    vesper/kernels/  delta_scan.py: the fused CUDA scan (fla-core's Triton kernels, gpu extra),
                     tested against vesper/recurrence.py; chosen per run by column.set_scan

Not yet written (rows marked "specified" in docs/registry.md): retrieval, ledger, dusk, the
rest of the evaluation suite (`eval/run_all.py`), the int4 state cast at inference, the
rest of the generation transaction (the corpus-selection manifest, pinned references in receipts
and recipes, the drivers' explicit legacy mode, the migration command; `generations.py` has the
publication and validation of a generation, docs/reviews/2026-10-15-generations.md). The memory rows
exist: activation recomputation per column application (`Loop.forward(recompute=True)`) and the
objective over position slices of the hidden output (`Vesper.objective(slices)`), as does the
bounded-cost resume (`Mixer.restore`); docs/reviews/2026-10-15-training-loop.md, Measured.

## Dependency direction

    config.py  ->  (nothing in vesper)        reads YAML, produces frozen dataclasses
    contracts.py, keys.py -> (nothing in vesper)
    recurrence.py, modulator.py, quant.py, attention.py, ffn.py, embedding.py -> contracts.py
    column.py -> recurrence, kernels, attention, ffn, contracts;  kernels -> contracts;
    graphs.py -> contracts;  exits.py -> contracts;  loop.py ->
    column, exits;  model.py -> loop, exits, embedding, learning, config (the one compute
    module that reads the config, to build itself)
    synthetic.py -> data;  halting_experiment.py -> column (scan), config, data, exits, graphs,
    learning, loop, model, provenance, synthetic, train (optimizer, schedule, head freeze);
    halting_report.py ->
    config, halting_experiment (summarize), synthetic
    tokenizer.py -> (nothing in vesper);  data.py -> contracts
    corpus.py -> data, keys, tokenizer;  books.py -> (nothing in vesper)
    collect.py -> books, corpus, keys;  acquire.py -> books, collect, generations, provenance
    books.py -> publish;  provenance.py -> books (git_output);  generations.py -> keys,
    provenance, publish
    sample.py -> config, corpus, generations (the selection), keys, provenance, tokenizer
    migrate.py -> corpus (finalize), generations, keys, provenance, sample (the legacy components)
    evals.py -> model, learning, data;  train.py -> model, corpus, data, evals, sample (the
    components), config, keys, provenance, publish, tokenizer
    tests -> everything

Compute modules never import config.py: constants arrive as arguments (`block`,
`half_life_tokens`), so a module is testable without a config file and a config value is read in
exactly one place; `model.py` is the assembler and reads the config once. The data path points one
way: `collect` writes what `corpus` reads, never the reverse. Kernels will depend on
recurrence.py (as the oracle in tests), never the reverse.

## Invariants and where they are enforced

- Every public tensor function checks its contract at the boundary with `check_shape`, and
  beartype checks the Python types at call time (`vesper/__init__.py` installs it on the package).
- No configuration constant appears in code (spec rule). Enforced by review and by the config
  loader's `done()`: a key that nothing reads fails the load, so the YAML and the code agree.
- The blocked scan equals the step scan applied `n` times (spec: 1e-4). Enforced by
  tests/test_recurrence.py in float64 (1e-10) and float32 (1e-4), gradients included.
- The recurrence is a delta rule: with `beta = 1` the read of `k` returns `v` exactly. Enforced by
  tests/test_recurrence.py, which also pins the reversed (wrong) erase order as a failing case.
- No `Any` reaches a value: mypy `disallow_any_expr` plus basedpyright `all` in gate.sh.
- Held-out documents never train, under any seed: membership is decided once by `finalize`
  against the pinned inputs (duplicates by the smallest id, the split on the group key) and every
  reader goes through `select_documents` with that `Split` (tests/test_corpus.py: the same
  membership under eight seeds, a duplicate on two sides read once, a group on one side).
- A collected output is published whole or not at all: `_Output` writes a temporary sibling and
  moves it onto the destination in `close()`; a failed run leaves a prior output untouched
  (tests/test_collect.py).
- The gate is the spec: a red check means the work is wrong, never the check.

## Key flows

- Config: `configs/vesper-s.yaml -> config.load -> _Section getters -> validate -> VesperConfig`.
- Scan (training form): `delta_scan_chunked -> per block: inverse(I + A) -> W, U -> delta ->
  outputs, state` with `check_shape` first.
- Scan (inference form): `delta_scan_step` once per token.
- Forward: `Vesper.forward -> TiedEmbedding -> Loop.forward (for t in 0..T_max: x + u[t]; for c:
  Column.step -> mask inactive; HaltingHead.probability -> allocation, depth) -> RMSNorm -> logits`;
  `wake_objective -> wake_loss (ntp_loss on the live positions, ponder)`.
- Data: `acquire.main -> fetch_shard / fetch_title -> collect_code / collect_wikipedia /
  collect_books -> parquet + receipt`; then `sample.main -> corpus.iter_mixed_texts
  (iter_raw_documents -> select_documents, by byte share) -> tokenizer.train_from_texts ->
  artifact + receipt`.
- Training: `train.run -> corpus.Mixer (one shared mixer: balances, digest of the ids handed
  out) -> G x data.Stream (micro_chunks slots each) -> per step: train.phase_schedule ->
  train.step (every live group's chunk drawn and its loss positions counted, Vesper.forward with
  the phase's pass count under autocast, wake_objective scaled by the group's share of the loss
  tokens, backward, states detached; one clip, one AdamW step; a dry stream drops out) -> log
  line`; every `total_tokens / checkpoints_per_run` tokens and once at the end (budget, step cap
  or exhaustion) `train.checkpoint -> publish.staged` (model, optimizer, states, stream packing
  state, identity with the recipe, layout, reason, manifest) and `evals.perplexity` on held-out
  chunks; `train.restore` checks the identity and the layout, then `Mixer.restore` seeks the
  readers to their cursors (bounded cost; `Mixer.replay` is the test oracle) and the run checks
  that mixer state, progress and streams agree before the first chunk.

Planned (spec, sections 3 and 4): retrieval into the ledger inside the column, `ledger.append`
after the chunk, `dusk.run` every `D` tokens, `train.py`, `eval/run_all.py`.

## Entry points

`python -m vesper.acquire` (collect one component unit at a time), `python -m vesper.sample`
(the mixed sample and the frozen tokenizer), `python -m vesper.train` (the training loop: e2e
wake training, no retrieval, ledger or dusk yet; `configs/vesper-s-smoke.yaml` crosses every
phase in forty steps), the test suite (`uv run pytest`) and the gate (`./gate.sh`).
`eval/run_all.py` is the planned entry point for the full suite (docs/registry.md).
