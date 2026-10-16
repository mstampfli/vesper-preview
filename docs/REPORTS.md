# Vesper reports for the project owner

One short entry after every milestone that changes what the model can do or how the system
works, in plain words, newest first. Written by maintainer A or maintainer B (the project owner asked for this on
2026-10-15: "a small executive summary after every milestone that affects the model's
capabilities or changes how it works, in simple words to understand quickly"). The detailed
evidence stays where it is (docs/reviews, the registry, the channel); this file is the summary.
Each entry: what changed, what it means for you, what is next, where the details are.

## 2026-10-16, 15:15Z: the small experiments train on the GPU, about ten times faster

**What changed.** A stage-1 training step used to be thousands of tiny GPU jobs issued one by
one from Python, so the GPU sat idle most of the time and one CPU core was faster. Now the whole
step is recorded once as a CUDA graph and replayed with one command, for all four conditions
(the adaptive-halting one is recorded once per training phase). The safety checks still run on
every step, on the GPU itself; the loss works on fixed-size arrays; and the toy-data generator
was rewritten to produce exactly the same data (pinned by a fingerprint of its output) in less
than half the time. Recorded training equals normal GPU training in every logged number and
every weight (tested). Not reviewed by maintainer B yet.

**What it means for you.** Measured through the real experiment driver on the 5060 Ti: 26 to 34
ms per step, against about 340 ms on one CPU core with the box idle. A 36,000-step run trains in
about 20 minutes instead of about 3.5 hours (the final evaluation still runs on the CPU, so the
total is longer; not yet measured at full size), and several runs can share the card. GPU runs
round slightly differently from CPU runs (about 0.5 % inside the memory update), so the gate
rules now name the hardware: a GPU run is never judged under a CPU recipe. The run maintainer B designed
continues on the CPU as agreed.

**What is next.** With the current run's result, a GPU version of the same recipe proposed to
maintainer B for further stage-1 runs. Details: commits 07b39f2, ac73fe7, 11a60fb; vesper/graphs.py;
docs/registry.md (graphs, contracts.require, halting_experiment rows).

## 2026-10-16, 12:35Z: a fused GPU kernel for the memory, five times faster training steps

**What changed.** The model's memory update (the delta-rule scan) can now run in a tuned GPU
kernel from flash-linear-attention, an open-source library (MIT license), instead of our plain
PyTorch version. It computes the same thing to about half a percent (the rounding of 16-bit
numbers), for the outputs and for every gradient training needs, and the error does not grow
with longer text. Our version stays the reference every test compares against. A run chooses
it with `--scan fused`, and its log records which one ran. Not reviewed by maintainer B yet (their
weekly budget is reserved for the stage-1 decision).

**What it means for you.** Measured on the box's 5060 Ti:

| | our scan | fused kernel |
| --- | --- | --- |
| the scan alone, forward and backward | 1x | 29 to 62x faster |
| Vesper-S step, 2 x 4096 tokens, all 8 passes | 9.3 s | 1.8 s (5.2x) |
| stage-1 step (tiny model), GPU | 0.45 s | 0.12 s |
| stage-1 step, one CPU core | 0.32 s | |

The earlier entry's "the GPU is slower than one CPU core" no longer holds: with the kernel the
tiny model trains about 2.6 times faster per run on the GPU. Compiling the whole step on top
added little (0.11 s) for stage 1 and made Vesper-S slower. On Vesper-S the card is now busy
92 % of the step, so a faster card would speed it up nearly in proportion; the 5090 has about
four times the 5060 Ti's memory bandwidth and compute on paper (not measured), and twice its
memory, enough to skip the recomputation that costs about a third of the step here (without
it the step does not fit in 16 GB). Training Vesper-S on its full 10 billion tokens at all 8
passes would take about 25 days on this card at today's speed, an estimate from the measured
step, fewer if the model learns to stop early.

**What is next.** Stage 1 and the stage-2 pilot fit on this card; the 5090 question becomes
real when the full Vesper-S run starts. Before that, the other half of the GPU time
(element-wise work around the scan) can likely be fused too. Details: commit 0e2faa4,
vesper/kernels/delta_scan.py, docs/registry.md (kernels row).

## 2026-10-16, 12:00Z: extra passes help exactly where the work does not fit in one

**What changed.** With the key filter the model now learns single lookups in the real task,
but the 12,000-update probe still failed both gates: longer lookups came late or not at all, and
the two-column model answered two-step lookups in its first pass, so extra passes added nothing.
Two calibration runs explained it. Trained three times longer, the two-column model learned
two-step lookups in three of four seeds, and the first gain from depth appeared only on a
three-step lookup (46.5 % on pass 1 to 51.9 % on pass 8). With ONE column, every lookup the model
learned needed one pass per step: two-step lookups went from 17.6 % on pass 1 to 93.7 % on pass 2,
three-step from 20.5 % to 77.6 % on pass 3. Maintainer B reloaded the weights and recomputed every number,
and found
that this is a profile of these trained weights, not a proof of a bound; four-step lookups and
most register questions are still not learned.

**What it means for you.** For the first time the test contains questions that need more
passes, so it can now ask whether the model learns to decide its own depth. That decision is
not shown yet: these runs force every token through all passes. A faster GPU would not speed up
this tiny model (a step on the box's GPU was slower than on one CPU core, all overhead).

**What is next.** One bounded run under a frozen recipe (one column, a slower curriculum, two
seeds, 36,000 updates, the same two gates), docs/reviews/2026-10-16-stage1-c1-recipe.md. If it
fails, we stop and diagnose before adding time. Meanwhile the delta-rule kernel, which removes
most of the per-step overhead and is needed for the real model anyway.

## 2026-10-15, 19:10Z: the first stage could not learn its lookups; a candidate fix

**What changed.** The stage-1 task could not ask its question: after 16,000 steps in three
settings (as built, faster learning rate, wider model) the model answered the easy "look back"
questions perfectly and the lookups ("what comes after A?") only at guessing level. In small
single-seed experiments, the model learned single lookups much more readily when its memory key
could also use the preceding words: three variants reached about 99 % after roughly 1,000, 1,500
and 3,000 training steps, where the model as built stayed at guessing level after 3,000. Maintainer B
chained a trained single lookup from outside the model for four steps at 98.6 % accuracy. Inside
the model, two-step lookups were also learned, with a longer curriculum. We have not established
the specific cause of the original difficulty, nor yet any benefit from adaptive depth.

**What it means for you.** We propose a four-word window (a short convolution) on the memory key
as the next candidate. It is an architecture change, so the decision is yours; nothing is built.
The proposal writes out exactly how it works, including tokens that stopped early, saved states
and decoding, because maintainer B found that the order of two steps there changes the result. The task
is also being redesigned with a fixed recipe: more questions per sequence, every question type
present from the start, harder ones phased in. It must show that thinking longer actually helps on
at least one question type before it can answer the halting question; in one experiment the model
answered two-step lookups in its first pass, which would leave nothing for adaptive depth to do.
All of this ran on the CPU of the rented box; no GPU time.

**What is next.** Maintainer B re-reviews the revised contracts; you decide the architecture change; then
it is built with its tests, the lookup experiment is repeated through the real code, and a bounded
probe of the new task checks both gates before the 12 runs. Details:
docs/reviews/2026-10-15-retrieval-binding.md. Written by maintainer A.

## 2026-10-15, 15:45Z: the halting experiment's first stage is built and running

**What changed.** The plan you approved (which training objective makes Vesper think longer
only where it helps) now has its first stage: a small made-up task where some answers take
one lookup step and others several (that count is how the task is built; how many passes the
model really needs for each is measured, not assumed). Each sequence hides a small loop of 8 linked names
(A leads to B, B to C, ...) in shuffled pairs; questions later ask "what comes 1, 2, 3 or 4
steps after A?", "which word was 1-3 positions back?" (easy) and one "remember this for later"
question. The graph is out of each attention layer's short-range view, though stacked layers
can still relay it (maintainer B caught that), so the runs also measure how much each answer depends
on the long memory by switching that memory off in a copy of the model.
A runner trains each of the four objectives (A: today's ACT; B: fixed equal weights; C and D:
the proposed per-token and batch-level priors) and measures accuracy against the work each
token is assigned (passes taken and state writes replayed, counted; this reference code still
computes every pass, so real savings in compute or speed are not measured yet), under the exact
way Vesper would run deployed; a report pools three seeds each and
applies the pass/fail rule agreed in advance. While building it, nine correctness issues in the
new exit code were found and fixed (maintainer B's reviews found eight), and an older test that failed
at random on GitHub was traced to a real precision slip and fixed.

**What it means for you.** The first six runs show the easy "look back" questions learned
(about 97 %), but the multi-step lookups still at guessing level (about 1 in 8) at the halfway
point. A stage 1 where only the easy class is learned cannot say anything about adaptive
depth, so before the full 12 runs, three longer probes (longer training, faster learning rate,
wider model) check whether the lookups are learnable at this size. All of this runs on the CPU of the box
already rented; no GPU time.

**What is next.** The probes' answer decides the setting for the full stage 1; then the report;
then stage 2 on real text. Details: docs/reviews/2026-10-15-halting-experiment-plan.md,
vesper/synthetic.py, vesper/halting_experiment.py, vesper/halting_report.py. Written by maintainer A.

## 2026-10-15: checkpoint depth now names what it measured

A checkpoint's old depth histogram came from the last training batch before its update, even
though it was logged beside held-out perplexity. New rows label that training measurement and
also record the held-out depth histogram, exact pass counts, loss sum and loss-position count
from the same evaluation forwards. Historical rows retain their original training meaning.
This changes reporting only. A controlled two-pass training/one-pass evaluation regression and
a known-probability masked-stream oracle distinguish the two; no extra evaluation traversal is
needed. Next: peer review and the separate normalized-halting-head trial. Field meanings and
aggregation rules are in CONTRIBUTING.md; the cause is documented in the halting diagnosis.
Written by maintainer B; reviewed and merged by maintainer A.

## 2026-10-15, 11:34Z: the first real training run learns, but its depth does not adapt yet

**What changed.** The first training run on your finalized data finished cleanly: 200 steps of
eight 4096-token sequences (6.6 million tokens) with the frozen tokenizer, read from the
migrated generations, in 2 hours 21 minutes on the rented card, at most 4.98 GiB of GPU memory.
Its held-out perplexity fell at every evaluation: 1,638 at step 50, 1,161 at step 100, 966 at
step 150 and 892 at step 200 (this morning's 40-step test ended at 3,815). The reviewed code is
now on main: a fast-forward of the 81 commits on dev/collector, after the full check with the
GPU tests passed on exactly that commit.

**What it means for you.** The training machinery works: the model learns steadily from your
data, and nothing crashed, leaked memory or diverged. But the idea that makes Vesper different,
that each token decides how many passes through the loop it needs, did not show up. The halting
histograms are nearly one-hot in both real runs: this morning's test ran every token through all
8 passes; this run sent practically every token through exactly 1 pass from step 22 on (99
percent at step 22, then 100 percent, also in every evaluation). That supports a depth collapse
in these runs: so far the model behaves as a fixed-depth model. Why is not known yet. The
suspects, each still to be checked: the halting head gives nearly the same answer for every
token, the penalty for extra passes tips all tokens at once, or the head enters the adaptive
phase untrained. The current implementation is a reference one: it computes every pass for every
token and masks the halted ones, so no compute savings are claimed from halting in any case.

**What is next.** The diagnosis of the halting mechanism comes before the comparison with the
dense baselines, because a baseline comparison against a model that always takes one pass
answers a different question. Maintainer A measures the halting probabilities and the head's
parameters from the saved checkpoints; maintainer B reviews the halting code independently.

**Where the details are.** The run's log and checkpoints under `~/data/runs/short-31ffc2c` (live
on the dashboard's Runs tab), the full check log `gate-collector-f37073e-full.log` in the shared
queue, the channel messages of 11:20Z and 11:35Z, and maintainer A's plans items 41 and 45. Written by
maintainer A.

## 2026-10-15, 08:55Z: your data now has names, and the old layout still works

**What changed.** The one-time migration ran on your directories and finished cleanly. Each of
the four data kinds (web, code, wiki, books) is now also an immutable "generation": a directory
with a manifest that lists every file with its size and hash. The files themselves were not
copied or moved; the generations are hard links to the same bytes, so the 28.5 GB of web data
exists once on disk under two names. A selection file now says which generation of each kind a
new training run reads. The migration finalized the corpus a second time over the generations and
compared the result with the finalization you already had: the same 404,078 duplicate documents
were dropped out of 10,097,165, over the same files by hash, so the selection was written. The
whole run took 45 minutes; the first minute made the generations, the rest was the second
finalization.

**What it means for you.** Nothing was deleted or moved; the flat directories, their receipts and
the old finalization are untouched and old commands keep working with the explicit legacy flag.
From now on a training run records exactly which generations it read, and a checkpoint can be
resumed with the same data even after the selection moves on to newer generations. A bug maintainer B
found in the migration code (it compared only the dropped documents, not the input files
themselves) was fixed and shipped before this report; the live run was checked from the outside
against that stricter standard and passed (all twenty files agree by hash with the plan and with
the old finalization).

**What is next.** A first real short training run from this selection with the frozen tokenizer
(about two hours on the rented card), to put a first held-out perplexity on the finalized corpus
into this file; then the dense baselines; then the kernels.

**Where the details are.** [The data contract](reviews/2026-10-15-generations.md),
[maintainer B's migration review](reviews/2026-10-15-migration-review.md) with the outcome, the migration
receipt under `~/data/mix/migrations/`, the plan and the binding check in the shared queue
(`migrate-plan-live.json`, `migrate-live-binding.json`). Written by maintainer A.

## 2026-10-15, morning: 4096-token training works; data preparation and resume advance

**What changed.** The current Vesper-S recurrent model completed a forty-step smoke run on
a rented GPU, reaching 4096-token chunks and up to eight passes. It used eight parallel
training sequences; the recipe's default batch is 128. Rebuilding intermediate activations
and splitting the output loss into smaller pieces reduced memory use. The completed run's
peak was 6.92 GiB; a later measurement with more loss slices reached 4.95 GiB in its stated
layout. The mixed-corpus tokenizer candidate has passed checksum, vocabulary and independent
round-trip checks. Data preparation now supports complete named collections, atomic selection
of inputs for new runs, and resuming an old run from its saved collection references.

**What it means for you.** The tested training configuration fits and runs through every
schedule phase. Its loss fell from 10.9 to 7.7, and its measured held-out perplexity from
12,134 to 3,815. These are smoke-test results; comparable baseline evaluations are still
needed. Controlled CPU continuation tests reproduced uninterrupted weights exactly. Exported
GPU continuation logs showed small numerical differences, so GPU resume is not claimed to
be bitwise identical. The current model is partial: retrieval, ledger, consolidation/dusk,
local learning, and quantization-aware training still need implementation. The default
128-sequence batch has not been demonstrated by this run.

**What is next.** Finish the data migration checks and the remaining temporary-file ownership
review, then evaluate the implemented model against the agreed controls and continue the
missing model components. Maintainer A reports an additive migration running on the newer branch;
its completion and data-equivalence receipt are pending. Old paths are retained. The measured
smoke throughput is about 1,000 tokens/s in its eight-sequence layout; a long-run time or cost
estimate needs its own configuration and measurement. Kernel work is planned, with no speedup
claimed before measurement.

**Where the details are.** [Training measurements](reviews/2026-10-15-training-loop.md),
[continuation evidence](reviews/2026-10-15-continuation-review-66b674a.md),
[data contract](reviews/2026-10-15-generations.md),
[latest metadata and ownership review](reviews/2026-10-15-metadata-lifetime-review.md),
[tokenizer decision](reviews/2026-10-15-pretokenization.md), and
[implementation contracts](registry.md). These distinguish independently checked artifacts
from the author's run receipts. Updated by maintainer B after the review of e39edd3.
