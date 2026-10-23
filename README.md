# Vesper

A language model that thinks in time instead of depth, learns while it runs, and sleeps to
remember. A small core (the column) is iterated until a token's answer settles; a fast-weight
state carries the session; an episodic store (the ledger) holds what fell out of the attention
window; a consolidation pass (dusk) writes what was learned into the slow weights while the
model is idle.

The design is in docs/spec.md; the model as it actually trains today, on one page, is in
docs/model-card.md. How the code is laid out and what it must never break is in
ARCHITECTURE.md; how to build, test and change it is in CONTRIBUTING.md. Every public
function and its tensor contract is listed in docs/registry.md. For readers who are not
machine-learning specialists, docs/explained.md walks through the architecture in plain words,
with diagrams. docs/related-work.md compares recent papers that build parts of the same design.

## Status

Research code, in progress. Implemented and tested in the pure-PyTorch reference: the column
(delta-rule recurrence with a causal key filter, windowed attention, FFN), the loop with per-token
halting and copy-forward of halted tokens, the model, the data pipeline (collection,
deduplicated finalization, generations, a 32k BPE tokenizer) and the training loop with activation
recomputation and resumable streams. On CUDA, the delta scan runs on kernels adopted from
flash-linear-attention, in bfloat16 or in float32 at the reference's precision, and the stage-1
training step is captured as a CUDA graph. Specified but not yet code: the episodic ledger and
its retrieval, the dusk consolidation, and the remaining kernels (sparse FFN, int8 and int4
inference).

Measured so far: a 200-step run of the roughly 60 M-parameter configuration reached held-out perplexity
892, but every token took the same number of passes, so the depth did not adapt. The current
work is a small synthetic experiment on halting (docs/reviews/2026-10-15-halting-experiment-plan.md
and docs/REPORTS.md). docs/registry.md lists what is implemented and what is only specified.

## Build and test

    uv sync --extra cpu --group dev     # CPU torch; use --extra gpu on a CUDA box
    ./gate.sh                           # the full gate, the same one CI runs
