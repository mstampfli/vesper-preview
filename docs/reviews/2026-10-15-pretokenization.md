# Proposal: the tokenizer's pre-tokenization, after the sample's trainer ran out of memory

Date: 2026-10-15. Author: maintainer A. Status: proposed; the project owner decides, maintainer B's view requested. Nothing
in the code changes until then; the tokenizer stays as reviewed (split on spaces only, the GPT-2
regex off).

## What happened

The 10 GiB mixed sample (web 80%, code 15%, encyclopedia 4%, books 1% by UTF-8 bytes) read
completely with the shares held, and the BPE trainer was then killed by the container's memory
limit: `memory.max` 64.6 GB, `oom_kill 1` in the cgroup counters. The web-only 10 GiB pilot of the
same night had fit.

## Measurements (this box, `tokenizers` 0.23.3, vocabulary 32,768)

Peak resident memory of the training process, 2.5 GiB samples, current pre-tokenizer:

| Sample | Documents | Distinct space-split words | Peak RSS |
| --- | --- | --- | --- |
| web only | 566,011 | 13.4 M | 17.6 GiB |
| the mix | 535,510 | 17.6 M | 29.0 GiB |

Distinct pre-tokens on the first 400 MB of one web file and one code shard, three rules:

| Rule | Web, distinct | Code, distinct | Web, pre-tokens | Code, pre-tokens |
| --- | --- | --- | --- | --- |
| spaces only (current) | 3.09 M | 5.70 M | 66.1 M | 83.8 M |
| GPT-2 regex (letters, digits, punctuation apart) | 0.82 M | 1.73 M | 82.5 M | 105.1 M |
| words and punctuation (letters and digits together, punctuation runs apart) | 0.86 M | 2.37 M | 82.5 M | 94.0 M |

The trainer's memory follows the number of distinct pre-tokens (each becomes a word with its
symbol list and pair counts). Code has about twice the distinct space-split words of prose per
megabyte (whole expressions without spaces are single words), and the mix at 10 GiB sits far past
the limit under the current rule.

## Options

1. **Change the pre-tokenizer to "words and punctuation"**: a run of letters and digits stays one
   piece (`foo123bar` stays whole, which is why the GPT-2 regex was switched off), a run of
   punctuation is its own piece, a space attaches to the piece that follows. Distinct pre-tokens
   fall 3.6 times on web and 2.4 times on code; whether that brings the 10 GiB mix within the
   box is an estimate until the trainer's peak RSS is measured under this rule (the measurement
   list below). Round trip stays exact (byte level). Cost: a documented tokenizer decision
   changes, the pilot tokenizer is superseded, and every number measured with the current rule is
   remeasured. The rule's boundary policy (Unicode letter and mark categories, digits, the
   underscore, whitespace classes) is written down with the change and pinned by tests before it
   is reviewed. This is my recommendation.
2. **Keep the rule and train on a smaller sample.** From the 2.5 GiB measurements the fitting
   budget is an estimate near 4 GiB, to be measured. The spec says 10 GB; the vocabulary would
   be learned from less of the mix.
3. **Keep the rule and the sample; aggregate words on disk first** (count distinct words in a
   first pass, drop singletons, feed the trainer counted words). The reduction is an estimate,
   not a measurement: the trainer's own structures (word representations, pair counts, occurrence
   sets) still scale with the distinct words that remain, and the merge statistics change
   slightly (singletons' pairs no longer count).

Measurements to take under the chosen rule before any freeze (agreed with maintainer B): the rule's exact
boundary expression and version, the inputs by hash, the command and its output, held-out
compression (tokens per byte per component), encode throughput, and the trainer's peak RSS under
the container's 60.2 GiB limit.

## If option 1 is taken

`tokenizer.train_from_texts` changes its pre-tokenizer sequence (one place), its docstring and
registry row say the new rule, the pilot receipt is marked superseded, a 400 MB measurement of the
pilot's and the new rule's compression (tokens per byte on each component) is recorded beside the
decision, and the frozen tokenizer is trained from the finalized mix with the plan pass and the
freeze-eligibility rule already in `vesper.sample`.

## Decision (the project owner, 2026-10-15, morning)

Option 1, the words-and-punctuation rule, now named `words-v1` in `vesper.tokenizer.PRE_TOKENIZERS`
and the default of `train_from_texts` and `python -m vesper.sample --pretokenizer`. The project owner's
condition: the change must not be forgotten, so that it can be reversed on a machine with the
memory for the original rule. Therefore:

- Both rules stay in the code with their tests (`spaces-v1` is the original); a rule is selected
  by name, never by editing the trainer.
- Every tokenizer file carries its rule's configuration, and `Tokenizer.rule` reads the name back
  from the file itself; every sample receipt records the name and its description beside the
  algorithm. A run's tokenizer therefore says which rule trained it.
- The spec's tokenizer section names the rule and the reason; this document holds the numbers.
- The way back: on a machine with enough memory, `python -m vesper.sample --pretokenizer
  spaces-v1` trains a new tokenizer; it is a new tokenizer and a new run (the recipe pins the
  tokenizer's hash), never a patch of an existing one. The expected memory is about 29 GiB per
  2.5 GiB of the mix under `spaces-v1` on the measurements above.
- What `words-v1` costs: a merge never crosses a letter-to-punctuation boundary, so punctuation-
  dense pieces of code take more tokens; measured as tokens per byte on held-out text per
  component in the section below, once the two tokenizers trained on the same sample exist.

## Measured: tokens per byte under both rules (2026-10-15, after the decision)

Two tokenizers of 32,768 ids trained on the same 400 MB mixed sample of the training side (clean
tip 2a712dd), measured on 75 MB of held-out text per component; the receipt is
`<evidence>/pretokenizer-compression-400MB.json`.

| Component | Held-out bytes | spaces-v1, tokens per KiB | words-v1, tokens per KiB | Change |
| --- | --- | --- | --- | --- |
| web | 64.0 MB | 210.2 | 225.2 | +7.1 % |
| code | 7.1 MB | 372.0 | 306.3 | -17.7 % |
| wiki | 3.2 MB | 232.5 | 246.7 | +6.1 % |
| books | 0.54 MB | 226.7 | 249.4 | +10.0 % |
| all | 74.9 MB | 226.6 | 234.0 | +3.3 % |

Training took 50.7 s under `spaces-v1` and 24.4 s under `words-v1` on this sample. The expectation
written above, that the words rule would cost a few percent on code, was wrong in direction: under
the space-only rule the vocabulary is spent on punctuation-attached pieces that rarely recur, so
code falls back to byte pieces; under the words rule the same ids cover the pieces that do recur
and code takes 18 % fewer tokens. Prose takes 6 to 10 % more, and the mix 3.3 % more tokens for the
same text. The frozen tokenizer is trained under `words-v1` on the 10 GiB sample.
