# Vesper reports for the project owner

One short entry after every milestone that changes what the model can do or how the system
works, in plain words, newest first. Written by maintainer A or maintainer B (the project owner asked for this on
2026-10-15: "a small executive summary after every milestone that affects the model's
capabilities or changes how it works, in simple words to understand quickly"). The detailed
evidence stays where it is (docs/reviews, the registry, the channel); this file is the summary.
Each entry: what changed, what it means for you, what is next, where the details are.

## 2026-10-26, 21:03Z: replaying general text during instruction tuning removes most of the forgetting

This entry's wording is not yet reviewed by maintainer B (their weekly review budget is nearly used up);
the numbers in it are the ones maintainer B reproduced independently when accepting the trial.

**What changed.** Instruction tuning made the model worse at ordinary text: on the days run's
held-out view, perplexity rose from 25.26 to 29.53 at a peak rate of 3e-5 and to 27.20 at 1e-5.
A third trial repeated the 1e-5 run exactly (same model, examples, order, schedule and selection
rule) but added, to every update, two rows of 1,024 positions of the source run's own training
text. The two losses are added with equal weight and make one optimizer step together. Every
replayed row is recorded and checked to come from the training side: none of its 2,378
documents or 2,341 groups is in the held-out set.

- Held-out general text: perplexity 25.26 before tuning and 25.45 after (+0.007 nats per token),
  against +0.074 at 1e-5 without replay and +0.156 at 3e-5. The measured increase is 90 %
  smaller than without replay.
- Response fit on the reused validation set: body loss 2.902 at the selected epoch (epoch 3),
  0.004 lower than without replay; the end-marker loss is 0.013 higher. The descriptive paired
  interval against the untuned model is [-0.221, -0.183].
- Cost: about 29 % more training time (33 minutes instead of 25), the same memory.

**What it means for you.** Mixing a little of the original text into tuning keeps the model's
general text ability almost intact while it learns the answer format. The limits: one run with
one setting; the held-out view is text of the same kind that was replayed, so this shows
retention of that kind of text, not of everything the model knew; the validation set was used
for selection; and lower losses do not yet show better written answers.

**What is next.** Answers from the replay model to the same twenty validation prompts, to see
whether the writing itself changes; it waits for maintainer B's review budget, which resets on
November 2. No reserved final test has been selected.

**Where the details are.** Exports instruction-preflight-replay-ee68f18 and
instruction-trial-replay-ee68f18, each with SHA256SUMS, and maintainer B's reviews replay-a771dee,
replayfix-01e2b3f, replaysnapshot-ee68f18, replay-preflight-1bb38a8 and replay-trial-b2fc2f8.

## 2026-10-26, 15:56Z: instruction tuning improves answer form, with a forgetting cost

**What changed.** The days run's final model (step 427,000) completed one instruction-tuning
trial on 13,296 Dolly examples: three epochs at a peak learning rate of 3e-5, with the planned
warmup and decay. The whole trial took about 25 minutes. After 12 eager warmup updates across four
batch lengths, the remaining 1,278 updates used captured CUDA graphs. Epoch 1 was selected
by the lowest pooled validation response-body loss among epochs 1, 2 and 3.

The 743 validation examples were split by connected groups of matching normalized instruction
text or nonempty context. This controls those exact-text matches, not semantic overlap. No
reserved test examples were evaluated by a model. The validation set was used for selection.

- Response-body loss in nats per token: 3.103 before tuning, 2.924 at epoch 1, 2.939 at epoch 2
  and 2.967 at epoch 3. The selected improvement is -0.179. Its descriptive paired 95% group
  resampling interval is [-0.205, -0.157] over 718 groups; this is not fresh confirmation after
  selecting on the same validation set.
- Terminator loss falls from 5.34 to 0.46 at epoch 1. On the days run's held-out general text,
  perplexity instead rises from 25.26 to 29.53 (+0.156 nats per token). The follow-up curve
  reproduces those values and reaches 32.55 at epoch 2 and 34.63 at epoch 3.
- Twenty validation prompts were chosen by a fixed category rule before inspecting outputs.
  The base model continues text or repeats headings and lists, reaching the length cap every
  time. The selected model emits its end marker in 13 of 20 cases and produces more recognizable
  answer forms, but often fails the request. It omits requested numbers already in the context,
  gives countries that do not start with the requested letter, changes Indiana to Iceland and
  repeats even in some answers that terminate. One answer matches its reference after trimming
  whitespace, and another extracts a relevant sentence. These are illustrations, not an accuracy
  estimate or evidence that a matching instrument-classification template establishes knowledge.
- A captured single-token GPU decoder uses a fixed-capacity, position-masked attention cache.
  In the selected measurement all 40 answer texts, stop reasons and generated-token counts
  match the CPU results. The generation calls, including prompt processing and host stopping
  checks, total 35.57 seconds versus 192.22 seconds on 12 CPU threads; model loading is excluded.
  A separate already-captured step loop reports a 4.48 ms median over 190 retained observations.
  Prompt processing and answer generation have not yet been timed separately. These timings
  compare the complete measured paths, not capture alone against an otherwise identical GPU path.
- A second bounded trial changed only the peak rate, to 1e-5, on the same code, data and
  reused validation set. Its response-body loss is 2.914, 2.906 and 2.908 at epochs 1 to 3;
  epoch 2 is selected (-0.196, descriptive interval [-0.220, -0.177]). Held-out general-text
  perplexity at the selected epoch rises from 25.26 to 27.20 (+0.074 nats per token), about
  half of the 3e-5 trial's rise. One run per rate. The 3e-5 follow-up measured all three saved
  epochs; for 1e-5, only the selected epoch's forgetting is measured.
- On the same twenty validation prompts, the lower-rate model stops in 15 cases versus 13 for
  the first trial. Content remains mixed: one requested single-sentence summary improves, while
  the previously correct instrument classification becomes wrong and several requests remain
  unanswered; this is not an accuracy estimate.

**What it means for you.** Tuning produces a visible improvement in answer format and
stopping, while substantial task failures remain and general-text prediction worsens on the
measured view. It does not isolate model size, data quality or forgetting as the cause. In
these two runs the lower rate fit the reference answers slightly better and reduced the
measured general-text loss increase by about 53%.

**What is next.** A decision on whether to test a further forgetting remedy (for example
mixing general text into tuning); no such run is selected yet. No reserved final test has been
selected. The decoder corrections and final-command integration are reviewed. On CUDA, final
now reuses one captured decoder per model across its panel and illustrations; an owned
synthetic lifecycle matches the CPU answers. The 35.57-second measurement above comes from the
separate validation-answer runner, not a measured full final-command run.

**Where the details are.** Exports instruction-trial-full-e8407f9,
instruction-forgetting-curve-e8407f9, instruction-pairs-epoch1-e8407f9,
instruction-trial-lr1e-5-e8407f9, instruction-answers-lr1e5-epoch2-b55852c,
review-evidence-decode-0e342c3 and review-evidence-decodefix-b55852c, each with SHA256SUMS,
and maintainer B's reviews instruction-trial-full-008fdfc, decode-0e342c3, decodefix-b55852c,
instruction-lr1e5-a3f71d9 and answers-lr1e5-b990094.

## 2026-10-26, 08:38Z: the stopping rule's training repeated with two more seeds

**What changed.** The small stopping head behind the 10-25 09:25Z result was trained twice more
from the same head initialization and a fresh optimizer. Training seeds 1 and 2 changed the data order and
action draws; the original was seed 0. Each new terminal head was evaluated with the fixed
Bounded23 rule on the same validation text as the original. The rule and terminal-checkpoint
choice were fixed before these repeats. Nothing was tuned between them, and they read no
test data.

- The model: the days run's final backbone (step 427,000), unchanged. Per seed, 20,000 updates
  of the stopping head at depth price 0.672, on this box's own card.
- Results on the validation text (loss in nats, lower is better; the benchmark is the best
  mixture of fixed-depth runs at no greater average number of passes; the random controls sample
  each token's stopping depth from the rule's observed depth proportions and replay the model;
  their realized mean depths differ slightly):

  | seed | passes | rule loss | benchmark | rule minus benchmark | ahead of the 4 random controls by |
  |------|--------|-----------|-----------|----------------------|-----------------------------------|
  | 0 (original) | 2.66 | 3.36258 | 3.36495 | -0.0024 | 0.0065 to 0.0072 |
  | 1 | 2.65 | 3.36364 | 3.36563 | -0.0020 | 0.0061 to 0.0068 |
  | 2 | 2.65 | 3.36417 | 3.36603 | -0.0019 | 0.0061 to 0.0068 |

**What it means for you.** The small validation advantage repeated in both new training runs,
by similar margins. The positive result is now observed in three training runs on this
backbone, at this price, on the reused validation view. The two prospective repetitions support
repeatability here without estimating reliability across training seeds or extending the result
to other backbones, prices or model sizes. Only the original head had the one held-out test. The saving is still
in passes, not in measured running time, and fixed 3 passes for every token still predicts better than the rule.

**What is next.** Complete the instruction-tuning GPU lifecycle check. The CPU functional
review is complete; the subsequent graph-pooling changes are under independent review.
The larger preflight has not fit the 16 GB card beside the embedding service. After code
review, the next bounded preflight will temporarily free that service's GPU memory. Any
further memory change gets its own checks. A full tuning run and final evaluation remain
unselected.

**Where the details are.** Export phase2-final-seeds (both seeds' training logs, weights,
evaluations, the runner and the reproduction control on seed 0) with SHA256SUMS, and maintainer B's
review seeds-9df6de0.

## 2026-10-25, 09:25Z: adaptive stopping on text passes a held-out test

**What changed.** This text pilot has its first positive held-out result for a rule that chooses
how many passes each token gets. It predicts better than the best mixture of fixed-depth runs
at the same average number of passes. The rule was frozen before one test on a bundle reserved
from the model evaluations in the accepted read inventory.

- The model: the days run's final backbone (step 427,000), unchanged. Only the small stopping
  head was trained: 20,000 updates at one depth price (0.672), one seed. This head training used
  the existing development card, with no additional rental.
- The rule (Bounded23): every token runs at least 2 passes and at most 3. It keeps the head's
  stopping-probability mass from both first passes: stop after pass 2 if their cumulative
  probability reaches 0.5; otherwise take a third pass and stop. Decisions use this execution's
  own states.
- The test: 1,048,576 tokens (981 documents) from the part of the held-out data that is
  untouched under the accepted read inventory. The rule, the trained head and the pass
  criterion were fixed before the run, and it ran exactly once.
- The result:
  - The rule averages 2.63 passes; 37 % of tokens stop at 2 and 63 % go to 3.
  - Its loss is 3.3369 nats against 3.3400 for the best fixed-depth mixture at the same average
    number of passes: better by 0.0031.
  - It also beats four random stopping controls sampled from its exit histogram, by 0.0076
    to 0.0080 nats. Their realized average depths differ slightly.
  - The predefined criterion was to beat that fixed-depth benchmark and all four random draws;
    both conditions held.

**How we got there.** These measurements were independently checked by maintainer B.

1. A fixed-weight diagnostic compared one and five passes on validation text. The tenth of
   scored positions with the largest observed gains accounts for about 69 % of the net loss
   reduction. That uses known next-token labels; it does not by itself show what the head can
   predict before seeing them.
2. The first trained rule, with exits from 1 to 8, did not beat the fixed-depth benchmark.
   Changing its original pass-2 exits to pass 3 improved mean loss at positions whose own
   exits stayed unchanged by 0.0078 nats. A reverse intervention from fixed pass 3 worsened
   those positions by 0.0073 nats on average. This establishes an effect on other tokens in
   those traces, not a unique explanation of the failure or a separate recurrence mechanism.
3. A hybrid trace beat the benchmark but borrowed its decisions from another execution. That
   motivated the live 2/3 rule, which beat the validation benchmark using its own states: by
   0.0024 for the 20,000-update head and 0.0019 for the 10,000-update head. The 20,000-update
   candidate was frozen before the one held-out test above.

**What it means for you.** Vesper's central idea of learning to allocate passes per token now
has its first positive held-out result on real text. The size is modest, and the scope is
narrow:

- One trained head from one training run, not shown across fresh runs.
- Saving is in passes, not measured running time.
- Fixed 3 passes for every token still predicts better (by 0.020 nats); the rule uses 12 % fewer
  passes than that.
- Against 5 passes for every token, the best fixed depth, the rule uses about half the passes
  for 0.031 nats more loss.

**What is next.** Not decided yet, each needs a choice with maintainer B:

- repeating the head training with other seeds and prices;
- testing whether training the whole model with the stopping rule reduces the cost that early
  stops impose on other tokens;
- or the instruction-tuning test.

A correction to the entry below: on 10-25 the final step's optimizer state was deleted with the
other finished runs' optimizer files (the box's disk was nearly full). The weights and memory
state are kept, but the retained artifacts no longer support exact training continuation
from step 427,000.

**Where the details are.** The evidence exports phase2-final-test-result (the test),
phase2-final-bounded and phase2-final-bounded-10k (validation), phase2-final-tracecross (the
cross-token effect), phase2-final-qreplay and phase2-final-pilot (the first rule), and stopA-427000
(the fixed-weight diagnostic), each with SHA256SUMS and maintainer B's reviews.

## 2026-10-24, 17:11Z: the days run finished: 427,000 steps, held-out perplexity 25.3

**What changed.** This Vesper run used d1024 and 64 M parameters,
recurrent memory restarted at mixed intervals per slot, 427,000 steps (about 7 billion tokens of
web text, code, Wikipedia and books) on one rented RTX 5090, from 10-22 02:42Z to 10-24 16:42Z,
with CUDA-graph training after warm-up. It was repaired once on the way: at step 226,650 the code
ran out (entry of 10-23, 11:58Z), and from step 225,362 it continued on the cycling recipe.
Held-out perplexity at a fixed 5 passes, the measurement that stays comparable across
checkpoints:

| step | repaired run | original run (code ran out) |
|---|---|---|
| 225,362 (before the run-out) | 32.88 | 32.88 |
| 237,223 | 31.94 | 43.77 |
| 249,084 | 31.07 | 43.83 |
| 260,945 | 31.13 | (stopped) |
| 272,806 | 30.03 | |
| 284,667 | 29.44 | |
| 296,528 | 29.06 | |
| 308,389 | 29.42 | |
| 320,250 | 28.38 | |
| 332,112 | 27.93 | |
| 343,973 | 27.47 | |
| 355,834 | 26.56 | |
| 367,695 | 26.59 | |
| 379,556 | 26.20 | |
| 391,417 | 25.71 | |
| 403,278 | 25.31 | |
| 415,139 | 25.39 | |
| 427,000 (final) | 25.26 | |

Per source at depth 5 (web, books, code and Wikipedia, measured at 237,223, 249,084, 260,945
and 272,806): at 237,223 all four were better than before the run-out, at 272,806 all four were
better than at 249,084, and 260,945 was a mixed move (code better, the other three slightly
worse). Five passes beat one by about 0.32 to 0.34 nats at every checkpoint of the repaired run,
and depth 5 is the best fixed depth at the final step too (3.2293 nats; 3.5590 at one pass). A
separate policy evaluation on the analysis box gives 25.25 when cold at the start and 25.40 when
initialized from the saved training state. These policy measurements are distinct from the
fixed-depth-5 score. Cost: about $55 for the whole rental up to the run's end, estimated as
hours times the rental's on-demand rate of $0.686/h: the card was rented on 10-21 at 08:31Z and
this run ended on 10-24 at 16:42Z, about 80.2 hours. The rental is longer than this run (about
62 hours) because the same card first ran the earlier text runs of 10-21 and the first days run
of 10-22; about $0.10 of failed starts on other cards comes on top.

**What it means for you.** The repaired trajectory improved overall through the remaining
201,638 updates, with four upward checkpoint moves (260,945, 308,389, 367,695 and 415,139).
The selected generation examples show recognizable English and answer/list/code formatting,
but also factual errors and repetition. These examples are not a systematic factuality or
instruction-following evaluation. This run does not establish a model-size requirement for
instruction following; whether instruction tuning helps this checkpoint remains untested.

**What is next.** Your call on the rented card (it stays until you say). The next jobs are not
ready to run: the instruction-tuning test on the final model has to be built first, and phase 2
of adaptive stopping on text needs a discussion with maintainer B. The final model, its memory state and
its optimizer are kept on the analysis box, so training can continue from here later.

**Where the details are.** The run's logs and every checkpoint of the repaired run (the final
one with its optimizer) on the analysis box, each copy checked against the run's manifest; the
evidence receipts with the component splits; the entry of 10-23, 11:58Z for the repair.

## 2026-10-23, 11:58Z: a data source that runs out starts over instead of dropping out

**What changed.** The days run trains on a mix of web text (80 %), code (15 %), Wikipedia (4 %)
and books (1 %). At about step 226,650 the code ran out: the run had used all of its roughly 0.56
billion code tokens, and the mixer is built to carry on with the other sources when one runs
dry. From then on the run trained on web and Wikipedia text only. The training loss stepped up
(about 3.42 to 3.68) at that point and the next checkpoint scored worse on the held-out text at
every depth (fixed depth 5: perplexity 32.9 to 43.8); split by source, all of it was code (code
perplexity on these labels 14.5 to 220, while web, books and Wikipedia improved). The run was
sized at 7 billion tokens without comparing that with what each source holds; that was a
planning miss. Now a run can cycle its sources (`--cycle-components`): a source that runs out
starts again from the beginning in a new, fixed order, so the mix stays at its shares for any
length of run. The first pass is exactly the old order, the counters and the record of what was
read carry on across passes, and a resume continues exactly where it stopped, also after several
restarts of a source (tested). Moving an existing run onto this recipe is one explicit, logged
step (`--from-finite-checkpoint`), allowed only from a checkpoint taken before any source ran
out (maintainer B's review: from a later one, cycling would silently leave that source out); without it
the old checkpoint is refused as another recipe.

**What it means for you.** A long run keeps learning from every source instead of quietly
dropping one. Code is then seen a second time, which is normal practice for small data
sources.

**What is next.** Done on 10-23 at 13:57Z, after maintainer B's acceptance: the days run continues from
step 225,362 (the last checkpoint before the code ran out) on this recipe; the original run and
its later checkpoints are kept as history. For later runs: download enough data first and check
supply against the budget before starting.

**Where the details are.** docs/registry.md (`corpus.Mixer`, `train.run`); tests in
tests/test_corpus.py and tests/test_train.py: 20 mutant variants red, the unchanged control green.

## 2026-10-23, 09:16Z: early stopping on text can now train with the model held still

**What changed.** The first longer test of early stopping on web text (3000 updates from the days
run's step 189778, exploratory, one seed) made the model stop after about 2.9 of 8 passes, but
the language model itself got worse at every depth while it learned (by 0.22 to 0.57 nats; 0.22
at depth 5), so the saving cost far more than it was worth; and on its own changed model the stopping choices beat
random stopping only slightly and lost to simply using 2-3 passes everywhere. Maintainer B's design for
the next step separates the two problems: train only the stopping head while the language model
stays exactly as it was. `vesper.halting_text` now has that mode (`--backbone frozen`): no update
reaches the language model, and the tests check it both ways (every weight and every fixed-depth
prediction unchanged after real updates, on the CPU and across captured GPU updates; the head's
gradient equal to an independent calculation; deliberate breakages all caught).

**What it means for you.** The next run asks one cleaner question: can the stopping head find
tokens that need fewer passes in a model that already works, without changing it? A positive
result gives a useful baseline with the model held still; a negative one, from one bounded run,
does not show that the model offers no such signal. Neither outcome alone explains why the
joint training damaged the model.

**What is next.** Maintainer B's code review, then the 3000-update head-only run on this box's GPU between
checkpoint evaluations. Details: docs/registry.md (halting_text row), the days-run science export
for maintainer B (NOTE addenda 7-11).

**Update, 10-23 afternoon.** Both head-only runs are done (exploratory, one seed each, checked by
maintainer B). The language model stayed exactly unchanged, as designed. At the original price the head
learned to stop after about 3.1 of 8 passes, at half the price after about 4.7. The original-price
run scored slightly better than its four random controls (by 0.0020-0.0035 nats per label); the
half-price run scored between its four controls. Both were worse than the fixed-depth benchmark
at the same mean depth. Average depth was still falling at the 3000-update cap. These runs show
learned depth reduction, without a useful allocation advantage over the fixed-depth controls.
No further run is planned on this line until it is discussed.

## 2026-10-22, 16:41Z (revised 17:3xZ after maintainer B's review): a crash costs minutes, not up to an hour and a half

**What changed.** On 10-22 around 16:22Z the rented 5090's host rebooted. The days run lost
the 8749 steps (about 68 minutes) since its last checkpoint, because a run saved its state only at
its 36 evaluated checkpoints, about 93 minutes apart, and everything between them lived only in
GPU memory. The run resumed from that checkpoint, optimizer and data position included. Now a run
can also make safety saves (`--safety-minutes M`): after the first step that ends M minutes or
more after the last save, the whole resumable state is written without an evaluation, and older
safety saves are removed (only the run's own; anything else in that folder is left alone). The
training itself does not change, and a resume from a safety save continues exactly like one from
a checkpoint (both tested). A failed save (a file-system error, or the writer's own error on a full or
limited disk) or a failed cleanup is logged and training goes on (maintainer B found that all three
stopped the run in the first version; fixed before it ever ran). The live analysis now
also copies the two newest checkpoints' optimizer states to the home box, checked by their hash,
so even a host that never comes back cannot take the resumable state with it.

**What it means for you.** A reboot or crash costs the training since the last successful save:
about M minutes plus one step when saving works (10 minutes for the days run, an estimated 3-5
seconds of saving each time, under 1 % of the training; measured once it runs). A dead host costs
the time since the last checkpoint whose optimizer reached the home box. The safety saves stay on
the rented card; the optimizer copies are extra network transfer (about 0.5 GB per checkpoint,
about 2 cents at the GPU provider's $0.039 per GB).

**What is next.** The days run switches to safety saves by restarting from a checkpoint right
after it is written, once maintainer B has checked the correction.

**Where the details are.** docs/registry.md (`train.run`); 14 tests in tests/test_train.py,
each mutant of the save, the cleanup and the interval red on its own test.

**Update, 10-23.** After maintainer B's acceptance, the switch was armed on 10-22 at 18:50Z and
executed at 19:37Z; the resumed training header is at 19:38Z. The selected 59 successful saves
took about 2.4 seconds each at a target interval of 10 minutes. That successful-row selection
does not establish that no other save failed. Scope
(maintainer B): recovery assumes the training log stays writable; a failure of the whole volume is not
shown survivable.

## 2026-10-21, 14:05Z: the next batch of text is prepared while the GPU trains on this one

**What changed.** Measured on the 5090 with the real corpus, not synthetic text: of a 274 ms
training step, 42 ms went to reading and tokenizing the next batch while the GPU waited. The
trainer now starts the step on the GPU first and prepares the next batch during it. The batches
and their order are exactly what they were; a checkpoint records the data position of the last
batch actually trained, so stopping and resuming gives the same run (tested, and seven deliberate
breakages of it were all caught).

**What it means for you.** Real training on the 5090 went from 274 to 224 ms per step: about
73,000 tokens per second at batch 16 (61,900 before this change, 31,000 this morning). The earlier
0.228 s figure was measured on synthetic text and did not include reading real data. Nothing has
been trained with it yet. Also found today: the settings sweep on the faster step (batch 16, one
micro-batch, no recomputation, one loss slice is the fastest that fits), and maintainer B found four
numerical problems in the new loss kernel, all fixed before any training used it.

**What is next.** Maintainer B's review, then your call on the longer runs. Details: docs/registry.md
(train, kernels rows).

## 2026-10-21, 12:50Z: the windowed attention runs as one tiled kernel on the GPU

**What changed.** On the GPU in bf16, the attention over the last 512 tokens now runs as
flash-linear-attention's windowed kernel (the same library as the scan) instead of PyTorch's
blocked attention with a mask. It skips the blocks outside the window instead of computing and
masking them. The CPU and float32 keep the old path, and the tests compare the two.

**What it means for you.** On the 5090 the d512 step went from 0.250 to 0.228 s (about 72,000
tokens per second, 2.3x this morning's 31,000; about 37 % of the card's peak arithmetic as
useful work). Nothing has been trained with it yet.

**What is next.** The settings sweep on the faster step, then the longer runs. Details:
docs/registry.md (kernels, attention rows).

## 2026-10-21, 11:55Z: training on the GPU is twice as fast, and every GPU path runs as a CUDA graph

**What changed.** You set a hard rule this morning: anything on the GPU runs as a captured CUDA
graph. The text trainer (`vesper.train`) did not; now every training step, every checkpoint
evaluation and the phase-2 evaluations are captured, and a CUDA run without a graph is refused.
Then the step itself was profiled on the rented RTX 5090 and made faster, one measured fix at a
time: the head's loss became one kernel that never makes float32 copies of the 32768-wide
scores (2.35x faster on its own); the layer norm, the key filter with its normalization and the
FFN's squared ReLU became one kernel each way; a float32 detour in the FFN went away; torch's
deterministic mode no longer fills every new tensor; the loop skips the halting bookkeeping when
every token runs every pass anyway; and without recomputation the head keeps its scores for the
backward pass. Each fix has tests against the old arithmetic (several bit for bit) and mutants.
Memory is now measured as what the card really holds (the old figure missed the graph's memory).

**What it means for you.** On the 5090, the d512 recipe went from 31,000 to about 65,000 tokens
per second (0.525 to 0.25 s per step, median of 40 captured steps; batch 16, one group, no
recomputation, about 20 GiB of the card's 32). The card does about 34 % of its peak arithmetic
as useful work, from about 16 %; a model twice as wide measured 38 %. The scan stays exact (your
call): the faster bf16 scan saves ~9 %, but in the 10-17 comparison of one pair the bf16 arm
missed the two-step escape within its 36,000 updates while the exact arm escaped at 19,000 (a
contrast between the two scan policies at that horizon, not a proven cause or a permanent block;
corrected after maintainer B's review of 571b5f9), and an exact parallel scan measured 3x slower on this
card. Nothing has been trained with the new code
yet.

**What is next.** A settings sweep on the faster step, more small fixes (each ~1 %), maintainer B's
review, then the longer runs. Details: docs/registry.md (kernels, graphs, train, evals rows).

## 2026-10-21, 07:20Z: the code that teaches the model to stop early on real text is built and reviewed

**What changed.** Phase 2 of the text pilot now has its code (`vesper/halting_text.py`): it
takes a model trained by phase 1 and trains it to decide, token by token, when to stop thinking,
with a price on every extra pass. Each row of text is read as its own fresh context, losses
are counted over real tokens, and the validation set is tied to the test set we fixed
yesterday, so no tuning ever reads the test part. Maintainer B reviewed it in full, with their
own numerical checks and full gate (both green), and found three faults that this entry's
commit fixes: (1) the price of extra passes averaged per row instead of over all real tokens, so
short rows counted as much as long ones; (2) the loader checked a file's fingerprint and then
read the file a second time, so a file swapped in between would have been used under the old
fingerprint (the weights, the run record, the config, the test pin, the tokenizer and the data
receipt now go through one shared read-once check, and so does the trainer's resume); (3) the
command line touched the source model before refusing a bad option. Each fix has a test that
fails on the old behaviour.

**What it means for you.** This is checked code, not a result. Nothing has been trained with
phase 2 yet, and its memory use and speed on the GPU are not measured. Separately, the phase-1
measurements it will start from improved overnight: with four times the training, extra passes
help much more (best at 3 passes, 0.15 nats better than 1, where the 1-hour runs gained 0.035,
measured from fresh contexts). That is a measurement of fixed depths, two runs, with the
training length and the learning-rate schedule changed together.

**What is next.** Maintainer B rechecks the three fixes; a short GPU preflight measures memory
and speed; then the first phase-2 runs at three prices around the overnight model's measured
gain.

**Details.** `vesper/halting_text.py`, `vesper/keys.py` (`hashed_bytes`, `verified_bytes`),
`vesper/joint_update.py` (`pooled_depth`), docs/registry.md; the plan in
docs/reviews/2026-10-20-real-data-halting-pilot.md. Written by maintainer A.

## 2026-10-20: refresh the TLS trust bundle after a new dependency advisory

**What changed.** The dependency audit began rejecting Certifi 2026.7.22 after advisory
GHSA-6q48-6c7w-9gw2 was published. The lock now selects Certifi 2026.10.19, and a transitive
version floor keeps later resolutions from selecting an affected release. The same constraint
applies to the CPU and GPU dependency sets.

**What it means for you.** The certificate bundle used by the project's HTTP clients removes
the affected emSign root. The model, data recipe and training code are unchanged. Existing
environments take the update when synchronized with the new lock.

**What is next.** Complete the normal dependency audit and code gate, then peer review before
merging. The running training jobs keep their recorded environments.

**Details.** `pyproject.toml`, `uv.lock`;
https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2.

## 2026-10-20, 14:25Z: the halting work moves to real text: depth-first training and a per-depth measurement

**What changed.** The adaptive-depth results so far come from a synthetic task built so that
some tokens need more passes than others. The next step is real text (the pilot you approved on
2026-10-15). Its first phase trains a small model with every pass learning equally ("depth
first"), and every checkpoint now records the held-out loss the model would have if every token
stopped after 1, 2, ... 8 passes, all measured in one run through the data. Maintainer B
reviewed the code, checked it independently and accepted it.

**What it means for you.** The first question the pilot answers is simple: on real text, at a
size this card can train in about an hour, does the loss go down when the model thinks longer?
A flat mean curve is a practical stopping point for this measured size and budget, not proof
that individual tokens cannot benefit; paired per-token changes remain part of the check.
When there is a useful gain, remeasure it under phase 2's independent-chunk, fresh-state
context before setting the price for learning to stop.

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
