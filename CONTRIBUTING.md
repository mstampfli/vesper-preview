# Contributing

## Dev loop

    uv sync --extra cpu --group dev          # once; --extra gpu on a CUDA box (cu130 wheels)
    git config core.hooksPath .githooks      # once per clone; runs the fast gate before each commit
    uv run pytest                            # the inner loop
    ./gate.sh --fast                         # format, lint, both type checkers, bandit, secrets, tests
    ./gate.sh                                # the full gate: adds the dependency audit; CI runs this

Nothing is done until `./gate.sh` prints `GATE GREEN`. CI (.github/workflows/gate.yml) runs the
same file and uploads an SBOM.

The end-to-end check on a GPU is the smoke run: forty steps through every phase of the schedule
over the finalized data, with checkpoints, a resume and the held-out perplexity in its log.

    uv run python -m vesper.train --config configs/vesper-s-smoke.yaml --data ROOT \
        --finalization finalize-<stamp>.json --tokenizer vesper-s-32768-<stamp>.json \
        --out runs/smoke --micro-chunks 1                      # 2048-token chunks, 6 min on a 5060 Ti
    uv run python -m vesper.train --config configs/vesper-s-smoke-4096.yaml ... \
        --micro-chunks 2 --recompute --loss-slices 4           # the recipe's chunk size, 22 min;
                                                               # 4.95 GiB at eight passes in this
                                                               # layout (6.41 with one slice)

The first run on real data is the same command with `configs/vesper-s-short.yaml` (two hundred
steps of eight 4096-token chunks, about two hours, a checkpoint with evaluations every fifty
steps), the data root's selection (`--data ROOT`) and its finalization, and the frozen tokenizer;
its number is a first held-out perplexity on the finalized corpus at a small scale, recorded in
docs/REPORTS.md, not the recipe's.

Checkpoint log rows keep `held_out_perplexity` and also report `held_out_loss_sum`,
`held_out_tokens_with_loss`, `held_out_iteration_counts` and `held_out_iteration_histogram`.
These measure the checkpoint after its update in float32, from the same forwards and loss
positions. Arrays cover passes `1..model.t_max`, including unused passes. Aggregate loss sums
and integer counts before computing a pooled perplexity or histogram; do not average chunk
perplexities or percentages. No loss positions means zero sums/counts/fractions and infinite
perplexity. The row tags these as `held_out_measurement: checkpoint-post-update` and
`held_out_precision: float32`.

`training_iteration_histogram` and `training_tokens_with_loss` describe only the last step's
training forward BEFORE its update (`training_measurement: last-step-pre-update`), not the
held-out data. `step` identifies that update, `training_t_max` its pass limit, and
`held_out_t_max` the evaluation's pass limit. Per-step `histogram`
keeps this same training meaning. Historical checkpoint rows used the ambiguous key
`iteration_histogram` for that training histogram; new rows omit that key. Do not reinterpret
old rows as held-out measurements. This logging correction changes no checkpoint payload,
model parameters, objective or training schedule.

Every step line carries `peak_reserved_bytes` (`graphs.memory_peak`: the allocator's reservation,
which counts a captured graph's pool; allocated bytes do not, a replay allocates nothing). `--recompute` trades about half again the step time
for the internal activations of each column application, which are rebuilt on demand (the
boundary tensors and the carried states still scale with the columns and passes); `--loss-slices`
computes the head and the loss over position slices instead of `[b, n, V]` float32 logits at
once. Both leave the objective and the gradients unchanged. The measured layouts are in
docs/reviews/2026-10-15-training-loop.md, Measured; a number belongs to its layout.

`--scan fused` (CUDA, `--extra gpu`) runs the recurrent scan in fla-core's Triton kernels instead
of the PyTorch reference: the same equation, with bfloat16 intermediates (about 0.5 % relative,
vesper/kernels/delta_scan.py); the run line records which scan ran. The reference stays the
default and the truth every kernel test compares against.

The stage-1 driver takes the same choice: `python -m vesper.halting_experiment ... --device cuda
--scan fused --cuda-graph` trains any condition on the GPU with a captured step replayed, one
capture per phase for ACT (vesper/graphs.py; equal to the eager GPU step, tested); probes and the evaluation run on a CPU copy with the
reference scan. The result records `device`, `scan` and `cuda_graph`, and the report's gate
protocols name their backend: the existing ones are CPU and reference, so a GPU run is not
assessed under them until a protocol for it is agreed.

An acquisition publishes one immutable generation per run under `ROOT/<kind>/gen-<stamp>/` and
logs its pin; choosing it for the next run is a separate, explicit step, and the flat layout of
the first runs is an explicit option, never inferred from what lies in a directory:

    uv run python -m vesper.acquire --data ROOT code --shards 0 1            # a new code generation
    uv run python -m vesper.acquire --data ROOT --parent gen-<stamp> code --shards 2   # extends it
    uv run python -m vesper.acquire --data ROOT books --count 300            # replaces a books parent
    uv run python -m vesper.generations select --data ROOT \
        --ref code/gen-<stamp>@<manifest sha256> --ref wiki/... --ref books/... --ref web/... \
        --replacing <sha256 of the selection being replaced>            # omitted for the first
    uv run python -m vesper.generations show --data ROOT                    # every generation validated
    uv run python -m vesper.acquire --data ROOT --legacy-layout code --shards 0   # the flat layout

The one-time step from the flat layout to generations prints its plan and stops; `--apply`
hard-links the files into one generation per kind, finalizes again, and writes the selection
only when the new finalization drops exactly the documents the legacy one did. Nothing is moved
or deleted; the legacy directories keep working.

    uv run python -m vesper.migrate --data ROOT --web WEBDIR --mix MIXDIR \
        --legacy-finalization finalize-<stamp>.json --out ROOT/finalization            # the plan
    uv run python -m vesper.migrate ... --apply                              # about 30 min live
    uv run python -m vesper.migrate --recover ROOT/migrations/migrate-<stamp>.json

Exit 3 from `--apply` means the selection is written and its receipt could not be completed;
the printed outcome names the receipt and the selection. On a pending receipt, `--recover`
validates the pins, the finalization's exact inputs and the current selection, then completes
the receipt without selecting again; it refuses when another selection has been written since,
and the receipt stays pending. On a receipt that is already complete it changes nothing and
prints the selection that receipt recorded, with `current` saying whether that is still the
selection in force (a later operator selection is not an error there).

## Check finite training supply before launching

Count once explicitly for the selected corpus, finalization, tokenizer and training split:

    uv run python -m vesper.supply --config configs/vesper-s.yaml --data ROOT \
        --finalization FINALIZATION --tokenizer TOKENIZER --out SUPPLY --workers 4

Then use the same training command with `--supply SUPPLY --plan --out PLAN` to print warnings
and save `PLAN/plan.json` without constructing a model. Launch into a different output
directory with `--supply SUPPLY` and without `--plan`. The legacy layout uses the same
`--legacy-layout --web WEBDIR --mix MIXDIR` arguments as training. Counting and planning
require new output directories; completed caches and plans are never overwritten.

The report separates finite encoded tokens, nonempty eligible documents and potential
within-document loss positions. `share * total_tokens` is **nominal** whole-run demand, and its
ratio to supply is nominal corpus-equivalent exposure, not predicted completed passes. A
shortage prints a warning and is saved in the plan/run header. It does not reject an intentional
finite run or turn on cycling. Completion remains `unassessed`, even at exact nominal supply:
successors, buffers, ordering, batch rounding and prefetch affect actual consumed inputs.
Zero supply for a positive share and an entirely zero-label mix refuse when checked; a
zero-label component in an otherwise productive mix warns. The runtime no-label guard remains.

`--resume` plans compare the same configured whole-run budget with the same finite inventory;
they never subtract handed documents from supply. They check the checkpoint's metadata identity,
recipe and layout, record its progress/hash, and leave tensor restoration unassessed. An actual
resume still performs its normal full restoration. The run header's `supply_basis_progress`
comes from that restored committed state, not speculative prefetched chunks.

Without `--supply`, ordinary training records supply as unknown and warns; it does not count
implicitly. Explicit `--plan` requires a checked cache. Missing, partial, corrupt or mismatched
evidence refuses. Reuse checks actual input hashes (including legacy files), ordered component
assignment, generation pins, finalization/drop set, tokenizer/rule, split, counting source and
encoding dependencies. Shares, seed and budget can change without recounting; their assessment
is bound separately to the run recipe. A plan does not certify other training prerequisites.

Each count worker copies one file to private scratch disk, verifies that copy's digest, and
reads only the copy through the normal corpus reader/filter/tokenizer. Source edits cannot
alter that checked snapshot. Workers each retain a tokenizer/drop set and one row group plus
an encoded document. Temporary disk is bounded by concurrently counted file sizes; choose
`--workers` for available RAM and disk. Counting needs this copy/hash I/O as well as encoding.
No speed or full-corpus runtime is promised by small-fixture tests. Counts publish together
only when all required files finish; failed work cannot become a valid cache.

## Where things go

- After a milestone that changes what the model can do or how the system works, add a short
  plain-words entry at the top of docs/REPORTS.md (the project owner's request, 2026-10-15): what changed,
  what it means for him, what is next, where the details are. Maintainer A or maintainer B writes it in the
  same commit as the milestone or the one right after.
- One module per component (docs/spec.md, "Repository layout"), at most ten public functions
  each. Before writing a function, find its row in docs/registry.md; if there is none, add the
  row (contract, rule) first, then the code, then flip its status to "tested" naming the test.
- Every public tensor function starts with `check_shape` calls binding `b, n, h, dk` (see
  vesper/recurrence.py for the pattern) and documents its contract in the docstring in the same
  notation as the registry.
- Constants live in configs/*.yaml and arrive as function arguments. A new constant means a new
  field in vesper/config.py (typed getter, `validate` relation if any) and a test in
  tests/test_config.py. Numerical floors such as `1e-6` are not configuration.
- A new check is shown red before it is trusted green: write the test against the broken or
  missing behavior first. Kernels are tested against the pure-PyTorch reference, never the
  reverse.
- Recurring concerns become primitives (docs/PRIMITIVES.md), not inline checks.

## Ownership and review

The project owner sets the direction and may override any decision. Two maintainers share
technical decisions as equals. A change merges after the other maintainer has reviewed it; the
rules are the same for both:

- One owner per task. A task is a row of docs/registry.md, or a review document under
  docs/reviews/. The owner's name goes on the row while it is in progress; responsibility
  follows the row.
- The other maintainer reviews before the project owner merges. The reviewer's first job is to name the test
  that would have failed if the work were wrong, and to run it. A review that finds nothing
  says what it looked for.
- Spec changes are proposed as a dated review document under docs/reviews/ (see
  2026-10-14-spec-review.md for the shape); the project owner decides, and the spec and registry change in
  the same commit as the decision.
- Personal working notes stay outside this repo.

## Commits

Conventional prefixes (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`),
imperative subject under 72 characters, a body that says why. One logical change per commit.
Plain ASCII in prose. Docs that describe structure, flows, invariants or the dev loop change in
the same commit as the code they describe.
