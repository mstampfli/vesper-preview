# Vesper: model card (as it trains, 2026-10-22)

The model as built and trained in the days run (`days-mixed`, recipe of commit 65c21bf, config
`days-mixed-d1024.yaml`), checked line by line against the code on 2026-10-22. The design and its
goals are in docs/spec.md; parts of the spec that are not built or not switched on are listed at
the end. Plain-words version: docs/explained.md.

## Shape

| | |
| --- | --- |
| Parameters | 64.0 M: tied embedding 32768 x 1024 = 33.6 M, the rest 30.5 M |
| Width / heads | d = 1024, 16 query heads of 64, 4 key/value heads (grouped-query) |
| Columns | 2 distinct blocks, applied in order on every pass |
| Passes | up to 8 per token (t_max), the same weights each pass |
| Vocabulary | 32768, byte-level BPE, pre-tokenizer `words-v1` |

## One pass

Pass t: `h = x + u[t]` (one learned vector per pass), then column 0 and column 1, each with its
own per-pass state. A column is three pre-RMSNorm sublayers with residuals, in this order:

1. **Recurrent memory (delta rule, no decay gate).** `H_t = (I - beta k kT) H_{t-1} + beta k vT`,
   read `o_t = H_tT q_t` after the write. q, v and z = W_k x are bias-free linears; the key is a
   causal depthwise width-4 filter over z, then L2-normalized per head; `beta = sigmoid(W_beta
   x)` per head; q is not normalized; output W_o. State per head 64 x 64, per pass, per column;
   it carries across chunks (cost per token independent of sequence length). Scan: fla's fused
   recurrent kernel in fp32.
2. **Windowed attention.** Causal, the last 512 positions, rotary (base 10000, phases in
   float64) at absolute stream positions, scale 1/sqrt(64), no biases; a KV cache of the newest
   512 entries per pass carries across chunks.
3. **FFN.** `W2 relu(W1 x)^2` (squared ReLU), hidden 4096, no biases.

After the last pass: a final RMSNorm, then logits = x E^T with the tied embedding.

## Training objective

All 8 passes run for every token (no early exit while training: `--objective fixed`). The loss
is the mean over loss positions of `sum_t (1/8) NLL(head(pass t))`: every depth is trained to
predict the next token. The halting head gets no gradient in this run; adaptive stopping is a
separate second phase (vesper/halting_text.py), trained on top of a finished snapshot.

## Memory restarts

Each of the 16 slots (document streams) restarts its recurrent memory (h and key filter history,
not the attention window) on its own schedule: level = slot % 6, period 16 x 4^level chunks of
1024 tokens (16, 64, 256, 1024, 4096 chunks), the last level never; slots on one level staggered.
The days run (65c21bf) numbers slots within each group of 8 (levels 0-5,0,1 per group): all six
levels are present, but the two shortest lifetimes (16 and 64 chunks) hold 4 slots each and the
other four levels 2 each, where numbering across the batch gives 3,3,3,3,2,2. From ec107cf slots
are numbered across the whole batch, so the layout no longer changes the mix. Document boundaries
trigger no reset. Backprop is truncated at chunk boundaries (1024 tokens).

## Training setup

| | |
| --- | --- |
| Data | 7.0 B tokens planned (427,000 steps); mix 80 % web, 15 % code, 4 % wiki, 1 % books |
| Batch | 16 slots x 1024 tokens = 16,384 tokens per step, as 2 groups of 8 |
| Optimizer | AdamW, betas 0.9 / 0.95, weight decay 0.1 on matrices only, grad clip 1.0 |
| Learning rate | 6e-4, linear warmup 100 steps, one cosine down to 6e-5 at the last step |
| Precision | fp32 weights, bf16 autocast, the scan and norm statistics in fp32 |
| Execution | the whole step as one captured CUDA graph; RTX 5090, ~35k tokens/s |

## Evaluation and what each number means

All on the same held-out tokens: the first 16 held-out chunks of 8 slots (16 x 8 x 1024 tokens,
130,940 labels), fp32, no gradient. Comparisons below are paired on these tokens; absolute values
from a set this size are monitoring numbers, and the final model is to be scored on >= 1 M
held-out tokens before any number is quoted outside.

- `held_out_perplexity` (the trainer's checkpoint line): states cold at the start and carried
  from chunk to chunk; the default ACT rule with the frozen head (p = 1/8 per pass), so the head
  reads the AVERAGE of the 8 passes' outputs. No inference path uses that readout: it is a
  monitoring number, not the model's quality.
- `static_profile`: the loss at each FIXED depth 1-8 from one dense pass, states carried. This is
  the headline: the best fixed depth (5 of 8 so far) and depth 8.
- Cold vs warm (tools/warm_eval.py): cold = the stream above; warm = the same, but the first
  chunk starts from the run's own carried TRAINING states (memories of other documents). Warm
  close to cold is the healthy result (a never-reset run gave 351 cold vs 36 warm).
- Memory use (tools/memory_use.py): states carried from chunk to chunk vs the recurrent memory
  wiped before every chunk (attention window kept) vs everything wiped. The gap measures how
  much the model uses memory of the SAME text from earlier chunks.

Not converged: the learning rate decays by one cosine over all 427,000 steps, so at 20 % of the
run it is still about 90 % of its peak.

## In the spec, not in this model

- Ledger / sparse retrieval (out-of-window episodic store): not built.
- Dusk (sleep consolidation): not built.
- Block-local learning: `learning.mode` must be `e2e`; `local` raises.
- Surprise modulation of the write strength: implemented, not used in training.
- Quantization-aware training / int8: `training.qat_from` is parsed, nothing reads it.
- `recurrence.state_read_noise`: parsed and validated, applied nowhere.
- A decay gate in the recurrence ("gated delta rule" in recurrence.py's docstring): none exists.
