# Vesper reports for the project owner

One short entry after every milestone that changes what the model can do or how the system
works, in plain words, newest first. Written by maintainer A or maintainer B (the project owner asked for this on
2026-10-15: "a small executive summary after every milestone that affects the model's
capabilities or changes how it works, in simple words to understand quickly"). The detailed
evidence stays where it is (docs/reviews, the registry, the channel); this file is the summary.
Each entry: what changed, what it means for you, what is next, where the details are.

## 2026-10-20, 14:25Z: the halting work moves to real text: depth-first training and a per-depth measurement

**What changed.** The adaptive-depth results so far come from a synthetic task built so that
some tokens need more passes than others. The next step is real text (the pilot you approved on
2026-10-15). Its first phase trains a small model with every pass learning equally ("depth
first"), and every checkpoint now records the held-out loss the model would have if every token
stopped after 1, 2, ... 8 passes, all measured in one run through the data. Maintainer B
reviewed the code, checked it independently and accepted it.

**What it means for you.** The first question the pilot answers is simple: on real text, at a
size this card can train in about an hour, does the loss go down when the model thinks longer?
If it does not, there is nothing for adaptive stopping to save, and the pilot stops and reports
that (for this size and budget only). If it does, the per-pass gain sets the price of an extra
pass in the second phase, which trains the stopping itself.

**What is next.** A short timing run picks the model size, then the first phase-1 runs. Nothing
has been trained on text with this yet.

**Details.** `vesper.train --objective fixed`, `vesper.evals.static_profile`; the plan with
maintainer B's amendments in docs/reviews/2026-10-20-real-data-halting-pilot.md.

## 2026-10-19, 17:45Z: repeatability checks pass; longer training recovers quality at lower depth

**What changed.** Two things. First, GPU training runs with the same data and random draws
diverged. Enabling PyTorch's strict deterministic mode made the tested repeats match bit for
bit, including every weight and optimizer value across 120 updates on this runtime and card.
The operation responsible for the earlier divergence is still unknown; the failure also
occurred with eager execution and the reference scan. The measured cost was a few percent per
captured training update under load. Maintainer B reviewed and accepted the change.

Second, with the cheaper stopping price (2.5), three of four runs met every recorded criterion
at 60000 updates: 97.3-98.2 % right at about 2.2 active passes per token, against 95.2 % for the
original model at all 8 passes. Their read4 accuracy was 66-93 %, against the original's 62 %.
All four had already passed at 5000 updates using 4.1-4.4 passes, then lost accuracy as depth
fell; three later recovered at the lower depth. The fourth still failed read4 retention.

**What it means for you.** Longer training recovered quality while using fewer active passes
in these runs. Learning was not monotone. The new setting passed the bounded repeatability
checks; entire 60000- or 120000-update trajectories have not yet been repeated. The four long
runs used the earlier setting, so each is one realization. Their seeds were deliberately
selected, and "three of four" is not a population success rate.

**What is next.** Four more 60000-update runs on the earlier setting, then two other trained
models and 120000-update runs on the accepted deterministic setting. Results in the morning.

**Details.** vesper/graphs.py (`deterministic_algorithms`), tests/test_determinism.py; run pages
dc25long-2a16c44 and dc25long-prefix-2a16c44; maintainer B's reviews in the channel (9bb3aa8, 774165e).

## 2026-10-19, 08:03Z: the whole model can now learn with its stopping, and its training updates are 3x faster

**What changed.** Two things. First, joint mode: until now only the small stopping head learned
while the rest of the model stayed frozen. In joint mode the whole model learns along with it,
so it can reshape what it computes to make stopping early easier. Its gradient was checked
against an exact hand-enumerated case (one of the checks caught a mistake in my own test first),
and maintainer B reviewed and accepted it (07:15Z). The first two joint runs started at 07:15Z. Second,
speed: the project owner asked why these runs did not use the GPU recording trick (CUDA graphs) that the
earlier experiments already used. They had never been given it, which was my miss. Each update
was about 10,000 tiny GPU jobs sent one at a time, and the waiting between them took about 70 %
of the time (counted once with the slower reference version of the recurrence; the runs use the
fast one, whose count was not taken). Now the update is recorded once and replayed. Measured on the real model, with
other runs sharing the card: 49 to 15 ms per update for the stopping head alone, 98 to 33 ms in
joint mode, 161 to 52 ms in joint mode with the extra all-depth training signal. About 3.1x
everywhere. The recorded updates give bit-for-bit the same results as the plain ones, and the
earlier experiment driver now uses the same shared mechanism instead of its own copy.

**What it means for you.** The training updates of these experiments are about three times
faster, so longer and bigger exploratory runs fit in a day or one night. A whole experiment also
generates its data (about 4 ms per update) and evaluates on the CPU, outside that speedup; the
throughput of a whole experiment is not measured yet. Runs from before this
change are not bit-for-bit repeatable under it (the GPU optimizer variant rounds differently);
each run records which one it used.

**What is next.** The results of the two joint runs (does the whole model learning stop the
overshoot?), then the overnight batch with the faster runs.

**Details.** vesper/graphs.py (`WarmedGraph`), vesper/halting_calibration.py (`train_update`,
`_update_step`), tests/test_calibration_graph.py; maintainer B's joint review: channel 4871b54.

## 2026-10-19, 07:05Z: the model learns to stop early and beats fixed depth, then overshoots

**What changed.** On 10-18 the model, already trained to answer every lookup class, was taught
to decide per token when to stop thinking (only its stopping head trained; the rest frozen).
Maintainer B's careful first trial learned nothing (every token still used all 8 passes). After the project owner
moved experiments to a faster, exploratory lane (code keeps full checks), eighteen exploratory
runs followed in four hours. With a stronger price on depth, the stopping policy for the first
time beat every fixed-depth setting at the same average number of passes, on 3 of the 5 trained
models (best: 92.3 % at 3.2 passes on average, where the best fixed-depth mixture gets 84.0 %),
and it spends depth where it is needed (one-step questions about 2 passes, four-step questions
about 4). The code now also has a joint mode, in which the whole model adapts while the stopping
head learns, checked against the exact gradient; it is pushed and waiting for maintainer B's review.

**What it means for you.** Adaptive depth works in the sense that matters first: at the same
average number of passes a token is counted as using (its mean active depth), choosing per token
beats choosing one depth for everything. The loop still executes every pass for every token, so
no compute is saved yet. Two limits stay
visible: the policy keeps cutting depth past the good region and then loses the rarest
questions (the "read" classes) or, on two models, collapses; and always running 4 passes
(95.2 %) is still better than anything it found below 4. Counting rare question types equally
did not keep them. All of this is exploratory: one seed per setting, 256 evaluation episodes,
logical depth, not measured speed.

**What is next.** Maintainer B's review of the joint mode, then its first two runs (one GPU hour or
less each), to see whether a model that adapts to its own stopping stops overshooting. A morning
batch checks whether the best result repeats with other seeds and on the other models. Then the
real-text halting pilot you approved on 10-15.

**Where the details are.** Exports with tables and notes under <evidence>/:
sweep-depthcost-4161f93p, sweep-sources-4161f93p, sweep-classweight-6998942, long-dc10-6998942,
joint-review; the calibration code and its options in [the registry](registry.md)
(halting_calibration); maintainer B's reviews in the channel (9f1eceb to 631bdb9).

## 2026-10-18, 06:00Z: with time, the harder levels come too; about half the runs stay stuck

**What changed.** Sixteen more stage-1 runs trained overnight, four times as long as the first
ones (144,000 updates). Nine learned the two-step lookups, six of those then the three-step
lookups, and four the four-step lookups; each level came after its own long wait, from 13,000 to
79,000 updates for the first level alone. One run, pair 20/20, learned every class including the
hardest register read: the second run ever, after pair 4/4. Seven of the sixteen never left the
first plateau.

**What it means for you.** The model can learn to chain lookups and to use more passes for
harder ones, and more training brings more levels. We now have five trained models that know all
four lookup levels, which is what the halting runs need to start from. Why half the runs stay
stuck is not known; the project owner's call this morning was not to spend days on that and to move to
halting, and I agree.

**What is next.** Two small pieces of code for the halting runs (start from trained weights; a
floor on taking another pass, lowered to zero), then the halting runs in the phases adopted on
2026-10-17 (not proven). Maintainer B decides their exact design.

**Details.** docs/reviews/2026-10-17-levels-with-time.md and -b.md (recipes and reading rules
fixed before launch), <evidence>/levels-with-time-0e887f7/PAGE.md. Not reviewed by
maintainer B yet.

## 2026-10-17, 18:26Z: most runs do learn the two-step lookups, given time; harder levels come later

**What changed.** Eight new stage-1 runs on the exact fast kernel trained twice as long, to
72,000 updates. Six of the eight learned the two-step lookups (four by 36,000, two right at it);
two never did, even at 72,000. This morning's batch of eight, with other seeds, had one in
eight. The harder lookups behave like separate steps on a staircase: three-step lookups were
learned later still, in two runs (one after 36,000, one after 54,000), four-step lookups in none
yet, though one run was rising at the end.

**What it means for you.** Learning to chain lookups is mostly slow rather than impossible: most
runs get there, each level takes its own long wait, and 36,000 updates was too short to see it.
Two runs in eight stayed stuck at the first level, so it is not only slowness. Only one run so
far (pair 4/4, this morning) has learned every level.

**What is next.** Longer runs on the cheap kernel, or a way to shorten the waits, before the
halting runs: halting needs models that have learned several levels, so that harder questions
can use more passes.

**Details.** docs/reviews/2026-10-17-escape-slowness-or-luck.md (the recipe and the reading rules
fixed before launch), <evidence>/halting-exact72-0e887f7/PAGE.md. Not reviewed by
maintainer B until Thursday.

## 2026-10-17, 11:21Z: an exact fast kernel, and one run that learns every lookup class

**What changed.** The GPU now has a scan that is as exact as our reference and as fast as the
rounded one: 18.6 ms per training step instead of 634 ms. With it, eight training runs of the
stage-1 task (one column, eight passes, every token running every pass) ran at once in under an
hour and a half. One of the eight, pair 4/4, learned the two-step lookups at 24,000 updates and
then everything else: three- and four-step lookups at 90 % and 82 %, and every register-reading
class above 50 %, the hardest at 65 %. No stage-1 run before met that bar. The other seven did
not learn the two-step lookups by 36,000.

**What it means for you.** This morning's hint was that the fast kernel's rounding was why
learning is so unreliable. That holds for pair 4/4 (the same weights and data learn with both
exact kernels and not with the rounded one), but not in general: on the four pairs from
yesterday, one learned with each kernel, and not the same one. So exactness is not the cure for
the unreliability; learning these lookups may hinge on small differences early in training (a
hypothesis, not shown). The good news is that a full run now costs minutes of GPU time, so
finding out what decides it is cheap.

**What is next.** Find what makes a run escape, using the cheap exact kernel and more seeds; then
the halting runs, started from weights that escaped, in the phases the project owner adopted today (marked not
proven, docs/reviews/2026-10-17-halting-in-phases.md). Not reviewed by maintainer B until Thursday.

**Details.** docs/reviews/2026-10-17-exact-scan-seeds.md (the recipe), the run export
<evidence>/halting-exact-9ce2453/PAGE.md, docs/registry.md (delta_scan_fused_fp32).

## 2026-10-17, 06:20Z: the fast GPU kernel's rounding stopped one pair from learning

**What changed.** Maintainer B's four-run test (two starting-weight seeds crossed with two data-order
seeds, 36,000 steps each on the GPU with the fast kernel) gave one escape in four: only pair
4/1 learned two-step lookups, at step 34,000. The same pair 4/4 had learned them at step 16,000
in an older CPU run, so maintainer B asked for one clean comparison: pair 4/4 twice on the GPU, the same
code, starting weights and training data (both checked by fingerprints over every batch), only
the memory update computed differently. With our exact version it learned two-step lookups at
step 19,000, and then three- and four-step lookups as well (95 % and 89 %, each answered after
as many passes as it has steps). With the fast kernel, which rounds its inputs to 16-bit numbers,
it learned none of them by step 36,000.

**What it means for you.** For this one pair the kernel's rounding decided whether the model
learned to chain lookups at all; small numerical differences can move the moment it learns by
thousands of steps. One pair is not a general rule, but it means a run with the fast kernel that
fails to learn cannot be counted as the model's failure until this is understood. The exact
version took about three times as long per step here (0.50 against 0.17 seconds; the two runs
shared the card during the fast one's 1.7 hours). Nothing needs a decision from you.

**What is next.** Maintainer B decides the next measurement; I sent both learning curves and proposed
nothing. Details: docs/reviews/2026-10-16-scan-policy-comparison.md, the result page in the
channel queue (halting-scan-4c2e590/PAGE.md), the crossed-seed page (halting-cross-2bbffdc). Not
reviewed by maintainer B yet.

## 2026-10-16, 18:55Z: maintainer B's slower teaching schedule failed; next, a four-run test on the GPU

**What changed.** The one-column run with maintainer B's slower teaching schedule (two seeds, 36,000
steps on the CPU) failed both of its pass marks, so by its own rule it stopped there. Seed 0
learned two-step lookups only in its last few thousand steps, and showed the depth effect the
project is after: 15 % after one pass, 73 % after three. Seed 1 never learned anything past
one-step lookups. Maintainer B checked every number and narrowed what it means: the two seeds behave
very differently, but a seed sets both the starting weights and the order of the training data,
so the result cannot say which one matters, and it cannot clear the schedule either. Maintainer B then
allowed training on the GPU as a new, separately recorded recipe, after two checks I ran this
evening: the GPU computes the same gradients as the CPU within the limits we had already set,
and the recorded (CUDA graph) training equals normal GPU training bit for bit over 120 steps.

**What it means for you.** A run that took five hours on the CPU should take about an hour on
the GPU, four at a time (measured per step; the full four-run time is measured by this test),
so questions that needed a week of runs fit into an evening. Nothing about the model's design
changed; nothing needs a decision from you.

**What is next.** Maintainer B's chosen test: four GPU runs that cross two starting-weight seeds with two
data-order seeds, measuring the step at which each first learns two-step lookups (checked every
1,000 steps). That separates "it depends on the starting weights" from "it depends on the data
order" for these two seeds; it is not yet a full picture over many seeds. The 12 formal runs stay
on hold. Details: docs/reviews/2026-10-16-crossed-seed-diagnostic.md, vesper/halting_escape.py,
the backend receipts in the channel queue (backend-check-11a60fb). The new code is not
reviewed by maintainer B yet.

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
