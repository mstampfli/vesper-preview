# Vesper spec review, 2026-10-14

Read: "Vesper — Model Specification.md" (290 lines) in full. No Vesper code or notes
exist yet anywhere on this machine. The spec's own protocol says implementation starts
once the interface contracts are agreed, so this review lists what an implementation
hits in week one. Checked numerically where a number could be checked.

## A. Bugs in the text (fix, no decision needed)

1. Delta-rule order. Written: `H_t = H_{t-1}(I - b k k^T) + b k v^T`. Correct:
   `H_t = (I - b k k^T) H_{t-1} + b k v^T` (erase the value currently stored under k,
   then write). Numeric check (dk=8, unit k, b=1): reading k back after the write
   returns v with error 0.0 for the correct order and 2.03 for the written order; the
   written order is also not idempotent on a repeated write (change 3.43 vs 0.0).
2. Halted tokens: section 4 says "masked out", the interface table says "removed, not
   masked". Pick: removed from column compute; their last-pass k, v and attention K, V
   are copied forward so later tokens still see them.
3. Iteration vectors u_t: the text says T vectors shared across columns; the parameter
   line says C x T x d.
4. Retrieval parameter count: as written (one head, dk-dim query, keys and values), the
   sublayer is W_r, W_rv, W_ro, W_g = about 0.25 d^2, not 2 d^2. Column is then
   about 17 d^2 and Vesper-S about 65 M, not 70 M. Either accept the numbers or widen
   retrieval (multi-head keys multiply ledger size by h).
5. KV cache size: each pass produces its own K, V for a token, so the cache is T x w
   per column, not w (about 25 MB at S; fine, but the hardware table should say so).
6. "Chunk" means two things: the 64-token block of the parallel scan and the 4,096-token
   sequence chunk. Rename one (suggest "block" for the scan).
7. `search` has two return contracts (component 5 vs interface table).
8. Modulator at inference: define l_t as the loss of predicting token t (known when
   token t arrives), so m_t exists when token t is written to H.
9. BPTT across sequence chunks is unspecified. State: the state is detached at chunk
   boundaries (truncated BPTT at 4,096); full BPTT only over the T passes within a chunk.

## B. The core gap: the loop and the recurrent state do not compose as written

With one H per column and pass-major training (pass 1 over all n tokens, then pass 2),
token 1 at pass 2 reads an H that already contains tokens 2..n from pass 1: future
leakage. Depth-first (all T passes of token i before token i+1) is causal but serializes
the whole 4,096-token chunk and kills the chunked parallel scan.

Proposed fix: T states per column, H_c^(t), one per pass. Pass t scans its own state
from its own carried value; at inference, token i at pass t reads and writes
H_c^(t). Training and inference then do identical state updates, which is what the
"scan equals step applied n times" test is meant to guarantee. Cost of the extra states:

  S: C*T = 32 states of 49k floats = 3 MB bf16, 1 MB int4
  M: 60 states of 147k = 18 MB bf16, 4 MB int4
  L: 96 states of 393k = 75 MB bf16, 19 MB int4

Everything else (dusk decay, int4 state, "H is the session state") holds unchanged;
"one H per column" becomes "T per column". In local mode this also makes "no gradient
crosses a loop iteration" true by construction.

## C. Decisions (each with a recommendation)

1. Retrieval frequency: once per token per column (issued at pass 1, result reused on
   later passes) or once per pass. Recommend once per token per column. The ledger is on
   CPU; a GPU-CPU-GPU round trip per pass is up to C*T = 32 per token, and at training
   it means millions of index queries per step.
2. Gate budget: nothing pushes g_r toward closed. A learned gate opens whenever retrieval
   helps even slightly, then training throughput dies on ledger searches. Recommend a
   target rate (<= 10% of tokens) via a penalty lambda_g * mean(g_r), or a hard top-k
   per chunk. Either is one config line.
3. Dusk cadence D = 2^22 "wake tokens": counted over the whole batch that is every 8
   steps (budget 5% of D = 0.4 steps of replay, 5% overhead); counted per sequence slot
   it is every 1,024 steps. Factor 128. Recommend total wake tokens, the natural
   reading, and state that the gamma decay applies to every slot's states.
4. Inference-time dusk needs fp master weights plus optimizer state on the deployed
   machine (int8 is the serving copy, re-quantized after each dusk) and a raw-text log
   of every conversation for replay. Both are consequences the spec should state; the
   log is the main item for the STRIDE pass before building.
5. Baseline fairness: "within 5% at equal parameters" favors a looped model, which spends
   about 3-5x the FLOPs per token. Recommend stating that the claim is parameter-matched
   on purpose (the thesis is memory traffic, and joules per token is the honest
   number), and adding an equal-FLOPs column to the eval table.

## D. Project, not spec

The spec trains S on a 24-32 GB GPU. The laptop has 8 GB, cp1 has none. Vesper-S
training is a rental (or the development box, 16 GB, between other jobs).

## Next

The project owner decides B and C. Then: back up the spec, patch A plus the decisions in, write the
interface table as the registry (every function with its tensor contract), STRIDE pass,
then Vesper-S skeleton starting with the pure-PyTorch reference scan and its 1e-4 test.
