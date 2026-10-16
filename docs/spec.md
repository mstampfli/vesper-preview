# Vesper — Model Specification

Oct 14, 2026 · @the project owner

Revised 2026-10-14 after the first implementation review (docs/reviews/2026-10-14-spec-review.md). This file in the repo (docs/spec.md) is the authoritative copy.

## Identity

**Vesper** — a language model that thinks in time instead of depth, learns while it runs, and sleeps to remember.

The name is the evening star: the model's defining phase is the consolidation pass that runs when it is idle, the way memory is written to cortex during sleep. The design thesis in one sentence: **amortize.** A transformer pays for everything on every token — re-reads the whole past, runs 80 layers once, carries all knowledge as context. Vesper pays once: a compressed state carries the past, a small core iterates until the answer settles, and what it learns in a session is written to fast weights now and to slow weights later.

- Model family name: Vesper. Sizes: Vesper-S, Vesper-M, Vesper-L.
- Component names: the **column** (the iterated core block), the **loop** (adaptive iteration), the **ledger** (fast-weight episodic store), **dusk** (the consolidation pass), the **modulator** (global learning signal).
- Training is from raw text only. No teacher model, no distillation, no synthetic data from another LLM.
- Visual identity: deep indigo on off-white, a single thin horizontal line as the logo mark (the horizon at dusk); monospace for code, a humanist sans for everything else. No gradients, no glow.

## Design principles

Five pillars, each paying for a specific transformer cost. Everything else in the spec follows from them.

| Pillar | What it replaces | What it buys | What it costs |
| --- | --- | --- | --- |
| Persistent state | Re-reading the full context per token (quadratic attention) | Linear cost in sequence length; state survives across turns | Exact long-range lookup is harder; kept via a small sparse retrieval path |
| Learning during inference | Carrying all session knowledge as tokens | Session facts live in fast weights, not context; context stays short | A write per step; drift risk, handled by dusk consolidation |
| Iteration over depth | 80 layers run once | 4 column weights run up to 12 times; parameters \~5x smaller for the same effective depth | Wall-clock latency grows with difficulty |
| Block-local learning | End-to-end backprop through the whole stack | No global backward pass; each column learns from its own prediction error and a scalar modulator | Accuracy gap at scale is unproven; shipped as a switch |
| Hardware co-design | FP16 dense matmul over a memory bus | Int8 weights, int4 fast weights, sparse routing, column-sized working sets that fit on-chip | Custom kernels; standard GPU gets a smaller win |

Two constraints are not pillars but hold everywhere: **no distillation** (the only training signal is raw text and the model's own prediction errors), and **no duplication** (the column is one module applied repeatedly; fast weights and slow weights share one matrix shape and one update rule, differing only in learning rate and lifetime).

## Architecture overview

&#91;embedded content: Vesper architecture · the loop, the ledger, dusk and the modulator\]

A token enters the loop, passes through the C columns, and either halts or goes around again; the columns' recurrent state H persists across tokens and turns. Out-of-window memory lives in the ledger, which columns read by sparse retrieval and write to after each chunk. Dusk replays salient ledger entries into the slow weights while the model is idle, and the modulator (the final loss, normalized) scales every learning rate and decides what is salient.

## Component specifications

Seven components; every interface is a tensor contract. Notation: `n` tokens in the current chunk, `d` model width, `h` heads, `dk = d/h` head width, `C` distinct columns, `T` iterations, `w` attention window, `k` retrieval count, `V` vocabulary. All activations are `bf16` in training, `int8` at inference unless stated.

### 1. Tokenizer

Byte-level BPE trained on the pretraining corpus, `V = 32,768`, byte fallback for everything else, no special-casing of digits or whitespace beyond the pre-tokenization rule. The rule is named and recorded in every tokenizer file and sample receipt (`vesper.tokenizer.PRE_TOKENIZERS`): `words-v1` (runs of letters and digits, runs of punctuation and whitespace runs apart, a leading space attached to the following run) is the frozen choice of 2026-10-15, because the original `spaces-v1` (split on spaces only) needs more than 64 GB of memory to train on the 10 GB sample of the mix; `spaces-v1` stays selectable for a machine with the memory, at the cost of a new tokenizer and a new run (docs/reviews/2026-10-15-pretokenization.md, Decision). Contract: `encode(str) -> int32[n]`, `decode(int32[n]) -> str`, round-trip exact on any valid Unicode string (the tokenizer's domain is text, UTF-8 encodable; raw bytes would need a byte-level API). `V` is an upper bound: BPE stops when no merge is left, so a small corpus yields fewer ids; the 10 GB sample is expected to reach it, to be confirmed by the training receipt of the frozen tokenizer.

### 2. Embedding and head

One matrix `E ∈ R^{V×d}`, used for both input embedding and output logits (tied). Input: `x0 = E[tokens] + pos` where `pos` is rotary applied inside attention, not an added table. Output: `logits = x_final · Eᵀ`. One matrix, one contract: `embed(int32[n]) -> [n, d]`, `logits([n, d]) -> [n, V]`.

### 3. Column

The column is the only compute block in the model. There are `C` of them with distinct weights; the loop applies all `C` in order, `T` times. A column takes `x ∈ R^{n×d}` and its own persistent state and returns `x' ∈ R^{n×d}` plus updated state. Four sublayers, pre-RMSNorm, residual around each:

**3a. Recurrent core (the persistent state and the fast weights are the same object).** A gated delta-rule recurrence per head. State `H^{(t)} ∈ R^{h×dk×dk}`, one per loop pass `t` (`T_max` states per column, see the loop's state ordering; the pass index is dropped below). For each token:

```latex
H_t = (I - \beta_t k_t k_t^{\top})\,H_{t-1} + \beta_t\, k_t v_t^{\top}, \qquad o_t = H_t^{\top} q_t
```

where `q, v = W_q x, W_v x`, `β_t = σ(W_β x_t) ∈ (0,1)` is a learned per-token write strength, and the key passes a short causal filter before it is L2-normalized: `z_t = W_k x_t`, `u_t = Σ_{r=0}^{K-1} a_r ⊙ z_{t-r}` (depthwise, `K = 4`, no bias, no activation, `r = 0` the current token), `k_t = u_t / ‖u_t‖` per head. A definition "a is followed by b" sits at two positions, and a delta-rule write takes key and value from one; the filter lets one write store the pair (in single-seed diagnostics the as-built key did not learn single lookups at the stage-1 size, and with the filter it did; docs/reviews/2026-10-15-retrieval-binding.md, decided by the project owner 2026-10-15). Each pass carries its last `K-1` unfiltered key inputs across chunk boundaries in its state, never reset within a stream like `H`; a halted token's successors read its frozen exit-pass input, and its own key stays its frozen exit-pass key (section 4). `key_conv: 0` is the unfiltered key, kept as the ablation. The erase factor stands on the key side: `(I - βkkᵀ)H` removes whatever is stored under `k` before the write, so with `β = 1` the read `Hᵀk` returns exactly `v` (the reversed order `H(I - βkkᵀ)` does not; checked numerically in the review and in tests/test_recurrence.py). This is a Hebbian outer-product write with a delta-rule correction; it is exactly a fast-weight matrix updated online. `H` is **not** reset between chunks or turns: it is the session state. Cost per token is `O(h·dk²) = O(d·dk)`, independent of sequence length. Blocked parallel form (WY representation, block size 64) for training; scan form at inference. "Chunk" in this spec always means the 4,096-token sequence chunk; the 64-token unit of the parallel scan is a "block".

Implementation precision: `normalize_keys` computes `k / max(L2 norm, 1e-6)`. Keys below the floor, including zero, are subunit; the exact overwrite statement above requires a unit key. Floating-point sigmoid may saturate to 0 or 1, and both endpoints preserve the contraction bound. A scan over zero tokens returns an empty output and the unchanged carried state. See docs/reviews/2026-10-14-recurrence-review.md for the independent oracle, reproduced numerical failures, and proposed fixes.

**3b. Windowed attention.** Exact softmax attention over the last `w = 512` tokens, rotary positions (base `model.rotary_base = 10,000`; phases computed in float64 so that long sessions keep distinct positions in low-precision activations), `h` heads, GQA with `h/4` KV heads. Contract: `[n, d] -> [n, d]` with a KV cache of at most `w` entries per pass (`T_max` windows per column). This is the only quadratic component and it is bounded.

**3c. Sparse retrieval.** Each token issues one query `q_r = W_r x_t` against the ledger (component 5) and receives the top-`k = 32` entries by inner product, then attends over those 32 with one softmax; the result is projected back by `W_ro ∈ R^{dk×d}`. The entries a token writes to the ledger are `k_r = W_rk x_t`, `v_r = W_rv x_t` (both `dk`). Contract: `retrieve([n, dk]) -> keys [n, k, dk], values [n, k, dk]`. Only tokens whose retrieval gate `g_r = σ(W_g x_t) > 0.5` issue a query; the rest skip the sublayer entirely (sparse, event-driven). A query is issued once per token per column, at the token's first pass; the retrieved 32 entries are kept in `ColumnState` and re-attended with the current pass's query on every later pass, so a token costs at most `C` ledger searches, never `C×T`. The gate carries a budget `L_gate = λ_g · mean(g_r)` (`λ_g = 0.01`), and the open rate is a first-class metric with a target of `ρ_r ≤ 10%` of tokens: without a cost a learned gate opens whenever retrieval helps even slightly, and training throughput dies on index searches.

**3d. Feed-forward.** `W_2 · relu(W_1 x)²`, hidden width `4d`. Squared ReLU is chosen for activation sparsity: measured on comparable models, 85-95% of hidden units are exactly zero per token, which the hardware section exploits.

**Iteration conditioning.** At pass `t` the loop adds `u_t ∈ R^d`, a learned per-pass vector (`T_max` of them, shared across columns), once: to the input of column 1. Columns `2..C` receive the previous column's output unchanged within the pass. This lets the same weights behave differently early and late in the loop. (The convention matters for the local-mode target, docs/reviews/2026-10-14-local-mode-question.md.)

Column parameter count: `(14.5 + 4·dk/d) d²` in matrices (4d² recurrence, 2.5d² attention with GQA, `4·d·dk` retrieval from four `d×dk`-sized matrices, 8d² FFN), which is `89/6 d² ≈ 14.83 d²` at Vesper-S, plus `W_β` (`d×h`), the retrieval gate (`d`) and four RMSNorm scales; plus `T_max×d` conditioning vectors shared across columns. Exact counts per tensor: docs/reviews/2026-10-14-baselines.md.

### 4. Loop and halting controller

One pass = columns `1..C` in order. After each pass `t`, a halting head computes `p_t = σ(w_h · RMSNorm_h(x_t) + b_h) ∈ (0,1)` per token, through its own RMSNorm so the logit does not carry the residual state's scale, which grows with every pass (the first version read `x_t` raw; one optimizer step then shifted every token's logit by about the same amount and halted 98.9 % of them at pass 1, all of them a step later: docs/reviews/2026-10-15-halting-diagnosis.md). Adaptive Computation Time rule: accumulate `P_t = Σ_{i≤t} p_i`; a token halts at the first `t` where `P_t ≥ 1 - ε` (`ε = 0.01`) or `t = T_max`. Its output is the halting-weighted mean of its per-pass outputs, so the gradient reaches every pass it used. Halted tokens leave the active batch (no column compute on them); their last-pass `(k, v, β)` and attention `(K, V)` are copied forward into every later pass, so the per-pass states stay dense and later tokens still see them. Ponder cost `L_ponder = Σ_t a_t·t`, the expected number of passes under the halting-weighted allocation (`a_t = p_t` before the halting pass, the remainder `1 - P` at it; it is 1 for a token that halts at once and `T_max` for one that never does, and it falls as `p` rises), weighted by `λ_h`, is what makes easy tokens halt early. The first version of this sentence summed the raw probabilities, `Σ_t p_t·t`, which rewards never halting: `p = 0` uses every pass at zero cost and the gradient pushes toward maximum depth (found in review, 2026-10-15).

Contract: `loop(x0 [n,d], states) -> (x_final [n,d], states', n_iters int32[n])`. `n_iters` is logged; its distribution is a first-class metric.

**State ordering (training = inference).** Pass `t` of column `c` reads and writes its own recurrent state `H_c^{(t)}` and its own KV window. In training, passes run in order `1..T_max` over the whole chunk (pass-major); within a pass, the recurrence is a causal scan over all `n` positions from the carried `H_c^{(t)}`. At inference a token runs its passes depth-first and touches `H_c^{(t)}` at pass `t`, so every state sees the identical sequence of updates in both orders. A single shared `H` per column cannot do this: pass-major would let pass 2 of token 1 read tokens 2..n from pass 1 (future leakage), and depth-first would serialize the chunk and lose the parallel scan. State cost `C × T_max × h × dk²` entries per stream, kept in float32 while learning (the scan never runs below float32): 6.0 MiB at S, 33.8 MiB at M, 144 MiB at L, excluding the KV windows, activations and the batch; int4 at inference (`quant`): 0.75 MiB, 4.2 MiB, 18 MiB.

### 5. Ledger (episodic store)

An append-only key-value store outside the model. After a chunk is processed, for each column the final-pass `(k_t, v_t)` pairs of its retrieval sublayer are appended, tagged with the token's salience `s_t` (the modulator value at that step, component 6 of the learning section). Capacity `M` entries per column (`M = 2^20` for Vesper-S). Eviction: FIFO among the lowest-salience quartile. Index: IVF with product-quantized keys (8-bit codes), rebuilt during dusk. Contract: `append(keys [n,dk], values [n,dk], salience [n])`, `search(queries [m,dk], k) -> ids [m,k], keys [m,k,dk], values [m,k,dk]`, `mark_consolidated(ids)`.

The ledger is what lets context stay short: a fact mentioned 200,000 tokens ago is reachable through one retrieval, not through attention over 200,000 keys.

### 6. Dusk (consolidation)

A scheduled pass with no user input. Steps: (1) sample ledger entries by salience, (2) reconstruct their source chunks from the raw-text log, (3) run the model over those chunks in **replay mode** (learning on, retrieval gate forced open), applying slow-weight updates at `η_slow = η_wake / 10`, (4) mark replayed entries consolidated and drop them from the index, (5) decay every column's `H` toward zero by factor `γ = 0.5`. Dusk runs every `D` wake tokens counted over the whole batch (`D = 2^22` for Vesper-S: every 8 steps at the 0.5 M batch) during training, and on an idle timer at inference; the `γ` decay applies to every stream's states. Contract: `dusk(model, ledger, text_log, budget_tokens) -> model'`.

### 7. Modulator

A scalar per token, broadcast to every learning rule. `m_t = (ℓ_t - μ) / σ`, where `ℓ_t` is the loss of predicting token `t` from the tokens before it (known the moment token `t` arrives, at inference too; for a generated token it is `-log p` of the sampled token) and `μ, σ` are exponential moving averages with a half-life of 10,000 tokens per stream. One pair is shared by all streams: a chunk's tokens are normalized with the pre-chunk `μ, σ`, and the pair is then updated once with the chunk's statistics. Positive `m_t` means surprise. It scales learning rates (section Learning) and is stored as the ledger salience. Contract: `modulate(loss [n]) -> m [n]`; one EMA state pair.

## Learning

There are two learning modes selected by one config flag, `learning.mode = "e2e" | "local"`, and both run on the same forward code. Train `e2e` first as the baseline; `local` is the research target and is measured against it.

### Wake objective

```latex
L = L_{\text{ntp}} + \lambda_{pc} \sum_{c=1}^{C} L_{pc,c} + \lambda_h L_{\text{ponder}} + \lambda_g L_{\text{gate}}
```

`L_ntp` is cross-entropy of the final (halting-weighted) output against the next token. `L_pc,c` is the column-local predictive loss below. `L_ponder` is from the halting controller. `L_gate` is the retrieval-gate budget (component 3c). Defaults: `λ_pc = 0.1` (0 in `e2e` mode), `λ_h = 0.01`, `λ_g = 0.01`.

### Mode e2e (baseline)

Standard backpropagation through all columns and through the loop (backprop through time over `T` passes, full, not truncated); across 4,096-token chunks the carried state is detached (truncated BPTT at the chunk boundary), so activation memory is bounded by one chunk. AdamW, cosine schedule, gradient clipping at 1.0. This is a normal, if unusual, transformer-class model and should train like one. It establishes the accuracy ceiling for the architecture.

### Mode local (block-local predictive learning)

No gradient crosses a column boundary and no gradient crosses a loop iteration. Each column `c` learns from three signals, all available at its own boundary:

1. **Local next-token head.** A small linear head `P_c` on the column's output predicts the next token: `L_tok,c = CE(P_c(x_c), y)`. Gradient flows into column `c` only. This is greedy layerwise training, extended to a loop.
2. **Predictive-coding term.** Column `c` also predicts the *input* of column `c+1` on the next pass: `L_pc,c = ‖Q_c(x_c) − sg(x_{c+1})‖²`, where `sg` is stop-gradient. Column `c` learns to produce representations the next column can build on without any signal from the next column's loss. Column `C` predicts column `1`'s input at the next iteration.
3. **Modulated learning rate.** Every parameter update in the model is scaled by the modulator: `η_eff,t = η · (1 + α·m_t)`, `α = 0.5`, with `m_t` clipped to `[-1, 3]`. Surprising tokens teach more. This is the one global signal and it is a scalar, not a gradient.

Within a column, backprop over its four sublayers is allowed (it is \~10 matmuls deep, which is local in any meaningful sense). The `P_c` and `Q_c` heads are `d×V` (tied to `E`, no new parameters) and `d×d` respectively.

The halting controller in local mode is trained from `L_ponder` plus the final `L_ntp` through its own head only; the columns never see `L_ntp` except through the modulator.

### Fast weights during wake

The recurrent state `H` updates every token by the delta rule in the forward pass; this is learning without any loss or optimizer and happens identically in both modes. The write strength `β_t` is a learned function of the input, so the model decides what to remember. At inference the write *odds* are additionally scaled by `(1 + α·m_t)`: `β_eff = σ(logit(β_t) + ln(1 + α·m_t)) = β_t(1 + α·m_t) / (1 + α·β_t·m_t)`, so surprising tokens are written harder. This equals the plain product `β_t(1 + α·m_t)` to first order in `β_t` and never leaves `[0, 1]`, which the plain product does (`β = 0.9, m = 3` gives 2.25 and an erase eigenvalue of -1.25; the contraction argument of 3a would not survive it). It is a different interior mapping, not a clamp of the product: at `β = 1/2, α = 1/2, m = 1` the product gives 3/4 and the odds rule 3/5. A zero multiplier (`α = 1` at `m = -1`, both in the ablation ranges) is a zero write; a negative one is rejected by the configuration (`1 + α·m_clip_low ≥ 0`). `write_strength` is the one place this is computed; the scan never sees an unbounded `β`. Decided 2026-10-14 (docs/reviews/2026-10-14-pass-state-review.md, section E and the appended outcome).

### Dusk (slow-weight consolidation)

Replay runs the wake objective in the current mode over salience-sampled ledger chunks at `η_slow = η/10`, with the retrieval gate forced open so the model learns to use its own ledger. After each dusk, `H ← γH` with `γ = 0.5`. Dusk is where session-specific fast weights become general slow weights; it is also the only phase where the ledger index is rebuilt.

### Optimizer and schedule (both modes)

AdamW, `β = (0.9, 0.95)`, weight decay 0.1 on matrices, 0 on gates, norms and `u_t`. Peak learning rate `3e-4` (S), `2e-4` (M), `1.5e-4` (L); 2,000-step warmup, cosine to 10% of peak. Batch `0.5M` tokens (S), `2M` (M), `4M` (L). Sequence chunk 4,096 tokens, chunks of one document processed in order with state carried across them (this is what trains the persistence). `T_max` is annealed from 2 to its final value over the first 10% of training so early training resembles a shallow model.

## Hardware co-design

The architecture is shaped so that the working set per step is one column, not the whole model, and so that most of the arithmetic can be skipped or done at low precision. These are constraints the model is trained under, not post-hoc optimizations.

| Constraint | Rule | Why it holds |
| --- | --- | --- |
| Working set | One column's weights (`≈ 14.8 d²` int8 ≈ 8.7 MB at d=768, `≈ 14.7 d²` ≈ 138 MB at d=3072) must fit in on-chip or near memory; the loop streams columns 1..C, T times | Weight traffic per token drops from whole-model to one-column, and columns are reused T times once loaded |
| Weight precision | Slow weights int8 (per-channel), served from memory; fast state `H` int4 with one fp16 scale per state row | `H` is rewritten every token, so it must be cheap to write; int4 outer-product updates are the neuromorphic analog of synaptic precision |
| Activation sparsity | FFN hidden units below 0 are never materialized; kernels use the relu² mask to skip rows of `W_2` | 85-95% of FFN work vanishes; this is the largest single saving |
| Event-driven retrieval | Only gated tokens issue ledger queries; the ledger lives on CPU/NVMe, not GPU | Retrieval is rare and latency-tolerant; it must not occupy the accelerator |
| Bounded attention | KV cache ≤ `T_max × w` entries per column (one window per pass; ≈ 25 MB at S) | Memory is constant in sequence length; no paging |
| Halting | Halted tokens drop out of the batch | Average iterations per token is the real throughput, not `T_max` |
| Noise tolerance | Training adds Gaussian noise (σ = 0.01) to `H` reads and int8 fake-quantization to weights from 50% of training onward | The model is robust to analog or low-precision substrates; required for any in-memory-compute target |

Target substrates, in order of readiness: (1) a single consumer GPU with custom sparse-FFN and chunked-scan kernels (Triton), (2) CPU inference with AVX-512 VNNI, where one-column working sets fit in L2 and the architecture's low arithmetic intensity stops hurting, (3) in-memory-compute or neuromorphic hardware, for which the int4 `H`, noise tolerance and event-driven retrieval are the prerequisites. The spec commits to (1) for training and (2) as the first efficiency demonstration; (3) is a port, not a redesign.

**Deployment (what learning at inference needs on the serving machine).** fp master weights and optimizer state (the int8 weights are the serving copy, re-quantized after every dusk), the ledger, and the raw-text log for replay. Everything that learns at inference is per tenant: `H`, the KV windows, the ledger, the raw-text log and the slow-weight updates. The base weights are shared and read-only; dusk writes a per-tenant delta `ΔW` of the same shape, served as `W + ΔW`. No tenant's text reaches another tenant's ledger or weights. The raw-text log is a deployment option, off by default; when on, it is encrypted at rest and an entry is deleted once its ledger entries are consolidated. The threat model is docs/threat-model.md.

Efficiency target: Vesper-M at inference should match a dense transformer of equal `L_ntp` at ≤ 25% of its energy per token on substrate (2), measured as joules per token over a held-out set, with latency allowed to be up to 4x higher.

## Model sizes

Three sizes. Vesper-S exists to validate the architecture on one GPU; Vesper-M is the first model that should be good; Vesper-L is the scaling check. Parameter counts are from the `(14.5 + 4·dk/d) d²` column formula plus the small tensors and tied embeddings (docs/reviews/2026-10-14-baselines.md); figures rounded.

|  | Vesper-S | Vesper-M | Vesper-L |
| --- | --- | --- | --- |
| Width `d` | 768 | 1,536 | 3,072 |
| Heads `h` / head width `dk` | 12 / 64 | 16 / 96 | 24 / 128 |
| Distinct columns `C` | 4 | 6 | 8 |
| Max iterations `T_max` | 8 | 10 | 12 |
| Effective depth `C×T_max` | 32 | 60 | 96 |
| Parameters | \~60 M | \~259 M | \~1.21 B |
| Attention window `w` | 512 | 512 | 1,024 |
| Retrieval `k` | 32 | 32 | 64 |
| Ledger capacity `M` per column | 2^20 | 2^22 | 2^24 |
| Training tokens | 10 B | 100 B | 1 T |
| Hardware | 1× 24-32 GB consumer GPU | 8× 80 GB | 64× 80 GB |
| Wall clock, e2e mode | \~5-7 days | \~7-10 days | \~3-4 weeks |

The wall-clock estimates assume full `T_max` passes in early training (before halting is learned) and average 40% of `T_max` once it is. Local mode trains roughly 1.3x faster per token (no backward pass through the loop) but may need more tokens for the same loss; that ratio is the key measurement of the whole project.

## Training recipe (from scratch)

Everything the model knows comes from raw text and its own prediction errors. The recipe is one wake/dusk cycle repeated; nothing is pre-trained elsewhere and copied in.

### Data

- Pretraining: a public web-text corpus with a quality filter (FineWeb-Edu or equivalent; no filtering by another LLM's judgments beyond what the dataset already applied), plus a code slice (The Stack or equivalent, permissive licenses only) at 15% of tokens, plus Wikipedia and books at 5%.
- Documents are kept whole and streamed in order. **No shuffling within a document.** Chunks of 4,096 tokens are processed sequentially with state carried across them. Documents are shuffled relative to each other.
- A raw-text log of every chunk's source span is kept for dusk replay; it is a pointer into the dataset, not a copy.
- Held-out: 0.5% of documents, never replayed, never in the ledger.

### Tokenizer

Train the BPE on a 10 GB sample of the mix before anything else. Vocabulary 32,768. Freeze. Nothing else in the project may depend on another tokenizer.

### Phases

1. **Warm (first 10% of tokens).** `T_max` annealed 2 → final, halting head frozen at `p = 1/T_max` (every token runs all passes), retrieval disabled, ledger not written. The model learns to be a shallow recurrent transformer.
2. **Loop (10-40%).** Halting head unfrozen, `λ_h` ramped 0 → 0.01. Retrieval enabled but ledger only holds the current document's out-of-window tokens. Dusk runs every `D` tokens but at `η_slow = 0` (index rebuild only), to validate the pipeline.
3. **Ledger (40-80%).** Ledger capacity opened to `M`, cross-document. Dusk learning on. Quantization-aware training and `H` read noise switched on at 50%.
4. **Settle (80-100%).** Learning rate decays to 10%; `T_max` fixed; final dusk cadence.

In local mode, the same four phases apply; the only difference is that phase 1 trains each column from its own `P_c` head, which means columns 2..C start learning only once column 1 produces stable outputs (it does within the first 1,000 steps).

### Hyperparameters (Vesper-S)

| Parameter | Value |
| --- | --- |
| Chunk length | 4,096 tokens |
| Batch | 128 chunks = 0.5 M tokens |
| Peak LR / warmup | 3e-4 / 2,000 steps |
| `λ_pc` / `λ_h` / `α` | 0.1 / 0.01 / 0.5 |
| `λ_g` / `ρ_r` (gate budget / target open rate) | 0.01 / 10% |
| `ε` halting threshold | 0.01 |
| Dusk cadence `D` / dusk budget | 2^22 tokens / 5% of `D` |
| `η_slow` / `γ` | `η/10` / 0.5 |
| Modulator EMA half-life | 10,000 tokens |
| Quantization-aware from | 50% of training |

### Evaluation suite

Standard perplexity on held-out text, plus five metrics that exist because of the architecture. Run all at every 5% checkpoint.

| Metric | What it measures | Pass criterion (S) |
| --- | --- | --- |
| Held-out perplexity | Accuracy, vs a dense transformer trained on the same tokens with the same parameter count | Within 5% of baseline (e2e), within 15% (local) |
| Held-out perplexity at equal FLOPs | The same, vs a dense transformer with the same training FLOPs per token (forward estimate: Vesper-S at the target mean of four passes spends 1.7x to 2.2x the forward FLOPs of Baseline A, the range set by the attention convention, with the logits projection and the recurrent state counted on both sides; the estimate and its assumptions are in docs/reviews/2026-10-14-baselines.md, and it is replaced by training FLOPs measured at the observed pass and gate distributions once those exist) | Reported, no pass criterion: the claim is parameter-matched on purpose (the thesis is memory traffic, not FLOPs); joules per token is the efficiency number |
| Retrieval open rate | Fraction of tokens whose gate opens, per column | ≤ 10% with the needle test still passing |
| Iteration histogram | Fraction of tokens halting at each `t`; easy tokens must halt early | Mean iterations ≤ 0.5·`T_max`; mean on code tokens > mean on prose |
| Needle at distance | Retrieve a fact planted 100k tokens back, via ledger only | ≥ 90% at 100k; a 512-window model without ledger scores \~0 |
| Session learning | Define a nonsense term once, query it 50k tokens later with the ledger disabled (fast weights only) | ≥ 60% accuracy |
| Dusk gain | Held-out loss on replayed-domain text before vs after a dusk | Measurable decrease; no increase on unrelated held-out (forgetting check) |
| Joules per token | Energy on the CPU target vs the dense baseline at equal perplexity | ≤ 50% for S, ≤ 25% for M |

## Repository layout and implementation protocol

One module per component, each with a fixed public surface of at most ten functions, every function documented with its tensor contract. The contracts in this section are the protocol; implementation starts only after they are agreed. PyTorch + Triton; nothing else on the hot path.

```
vesper/
  config.py      load (typed, fail-closed; the only reader of configs/)
  contracts.py   check_shape (the tensor-contract primitive every module uses at its boundary)
  tokenizer.py   encode, decode, train
  embed.py       embed, logits
  column.py      forward, init_state, state_dtype_cast
  recurrence.py  delta_scan_chunked, delta_scan_step, write_strength
  attention.py   windowed_forward, kv_cache_push
  retrieval.py   gate, query, attend
  ffn.py         forward (sparse relu² kernel)
  loop.py        run, halt_step, ponder_loss
  ledger.py      append, search, mark_consolidated, evict, rebuild_index, sample_for_dusk
  modulator.py   update, normalize
  learning.py    wake_loss, local_heads, modulated_step, mode
  dusk.py        run, replay_chunk, decay_state
  quant.py       fake_quant_weights, quant_state_int4, dequant_state
  train.py       phase_schedule, step, checkpoint
  eval/          perplexity, iteration_histogram, needle, session_learning, dusk_gain, joules
  kernels/       triton: delta_scan, sparse_ffn, windowed_attn
  configs/       vesper-s.yaml, vesper-m.yaml, vesper-l.yaml
```

### Interface contracts

The full registry (every public function with its tensor contract and its status) is docs/registry.md; the rows below are the load-bearing ones. All contracts carry a leading batch dimension `b`, omitted here and in the component section for readability.

| Module | Function | In | Out | Rule |
| --- | --- | --- | --- | --- |
| column | `forward(x, state, iter_idx)` | `[n,d]`, `ColumnState`, int | `[n,d]`, `ColumnState` | Pure w.r.t. weights; all mutation is in the returned state |
| recurrence | `delta_scan_chunked(q,k,v,β,H0)` | `[n,h,dk]`×3, `[n,h]`, `[h,dk,dk]` | `o [n,h,dk]`, `H_n` | Must equal `delta_scan_step` applied n times, tested to 1e-4 |
| loop | `run(x0, states, T_max)` | `[n,d]`, `List[ColumnState]`, int | `[n,d]`, states, `n_iters [n]` | Halted tokens are removed from the active batch, not masked |
| ledger | `search(q, k)` | `[m,dk]`, int | `ids [m,k]`, `keys [m,k,dk]`, `vals [m,k,dk]` | Runs off-accelerator; async with a one-chunk lag |
| modulator | `update(state, loss)` | `ModulatorState`, `[n]` | `m [n]`, `ModulatorState` | Single EMA pair; no other state; pure, the new state is returned |
| learning | `wake_loss(outputs, targets, mode)` | per-column outputs, `[n]`, str | scalar, dict of parts | `mode="local"` asserts no grad crosses a column boundary (test) |
| dusk | `run(model, ledger, log, budget)` | — | model | Idempotent on an empty ledger |
| quant | `quant_state_int4(H)` | `[h,dk,dk] fp` | codes `[h,dk,dk/2]` uint8 (two int4 per byte), scale `[h,dk]` fp16 | Round-trip error bound is measured, not assumed: see docs/registry.md (int4 cannot reach 2% on a Gaussian state; QAT is what makes the model tolerate the real figure) |

### Rules

- `ColumnState` is one dataclass: `H` (one per pass), KV cache (one window per pass), the retrieved entries per token, the last-pass `(k, v, β)` and `(K, V)` for copy-forward, retrieval gate stats. It is the only mutable object in the forward pass.
- The chunked and stepwise scans share one reference implementation in pure PyTorch; kernels are tested against it, never the other way around.
- `local` and `e2e` share every line of forward code. The mode flag touches `learning.py` and nothing else.
- Before implementing any function, check the table above and the existing module for an equivalent. The table is the registry.
- No configuration constant appears in code; all are in `configs/`.
- Every eval is a function `(model, data) -> float` and is run by one `eval/run_all.py`.

## Risks, open questions, ablation plan

The honest ranking: local learning is the bet, everything else is engineering with known-good precedents (DeltaNet-class recurrences, sliding-window + retrieval hybrids, ACT halting, replay-based consolidation, relu² sparsity, QAT).

| Risk | Likelihood | Mitigation in the spec |
| --- | --- | --- |
| Local mode lags e2e by more than 15% perplexity | High | The mode switch: ship e2e, keep local as the research track; the 15% gap is the measurement that decides |
| Halting collapses (all tokens use `T_max`, or all halt at 1) | Medium | `T_max` annealing, `λ_h` ramp, iteration histogram as a first-class metric with alerts |
| Fast state `H` drifts in long sessions (hours of tokens) | Medium | Dusk decay `γ`; `β_t` learned; session-learning eval catches it |
| Ledger retrieval is learned as noise and ignored (gate stays closed) | Medium | Retrieval gate forced open during dusk; needle eval |
| Retrieval gate opens everywhere and training throughput dies on index searches | Medium | Gate budget `λ_g`, open rate as a first-class metric, one query per token per column |
| Consolidation causes forgetting on unrelated domains | Medium | Dusk-gain eval includes an unrelated held-out set; `η_slow` is tunable down to 0 |
| Int4 `H` loses too much (measured 2026-10-14: 10.7% relative RMS round-trip error with per-row scales, 15.6% per-head, on Gaussian and on delta-rule states; int8 per-head is 0.9%) | Medium | QAT with noise from 50%; int8 fallback for `H` is a one-line config change |
| One-column working set does not fit on-chip at L | Low | L is the only size where this binds; it is a scaling check, not a product |

**Open questions** (each becomes an ablation on Vesper-S, e2e mode unless noted):

- [ ] Does the predictive-coding term `L_pc` help in e2e mode, or only in local mode? Run `λ_pc ∈ {0, 0.1}` in both.
- [ ] Is `C = 4, T = 8` better than `C = 8, T = 4` at a fixed maximum of 32 column applications per token? At fixed `d` the grid `{(2,16), (4,8), (8,4), (16,2)}` holds `C × T_max = 32`, the maximum applications per token; realized work is `C × mean(T)` and is reported per point together with the parameters, which range from `2 × 89/6 d²` to `16 × 89/6 d²` plus the embedding. `(32,1)` with recurrence and retrieval removed is Baseline B, a named dense reference that is neither parameter- nor compute-matched; the parameter-matched dense baseline is Baseline A (7 layers, `ffn_mult` 3), both in docs/reviews/2026-10-14-baselines.md.
- [ ] Does the modulator help, or is it a learning-rate schedule in disguise? Run `α ∈ {0, 0.5, 1.0}`.
- [ ] How much does carrying `H` across documents help vs resetting it? Run both; this decides whether "persistent" means session or document.
- [ ] Does dusk replay beat simply training on those tokens again at low LR? Run replay-by-salience vs replay-uniform vs no replay.
- [ ] Halted tokens in later passes: copy their last `(k, v, β)` forward into `H^{(t)}` (the default) or stop writing them (`β = 0`)? Run both; the difference shows whether later passes need to see easy tokens at all.
- [ ] Per-tenant `ΔW` at M and L: a full int8 delta (one model copy per tenant) or low-rank? S ships full; the answer decides whether multi-tenant serving is viable at L.
- [ ] What is the actual joules-per-token on the CPU target at equal perplexity? This is the number that justifies the project or doesn't.

First milestone: Vesper-S in e2e mode within 5% of a dense transformer of equal parameter count (≈ 60 M) on held-out perplexity, with mean iterations ≤ 4 and the needle test passing. Nothing below that line is worth optimizing.
