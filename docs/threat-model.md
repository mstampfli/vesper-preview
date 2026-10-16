# Threat model (STRIDE), first pass 2026-10-14

Scope: the training pipeline (data to checkpoint) and the serving deployment with learning at
inference (fast state, ledger, raw-text log, dusk). Re-run when a trust boundary moves, when the
serving design lands, and after any incident. Status per mitigation: **enforced** names the code
or test; **design rule** names the spec section; **open** means not yet built.

## 1. What we are working on (data-flow diagram, as text)

External entities
- E1 Public corpora (FineWeb-Edu, The Stack, Wikipedia): untrusted text.
- E2 Tenants (users or applications) at inference: untrusted text, untrusted identity claims.
- E3 Operator: runs training and serving, owns configs and checkpoints.
- E4 Host application: authenticates tenants and hands the serving process an opaque tenant handle.

Processes
- P1 Tokenizer training and encoding.
- P2 Wake training step (train.py, loop, learning).
- P3 Dusk (dusk.py): replay from ledger plus raw-text log into slow weights.
- P4 Serving loop: per-token inference with fast-state writes and ledger reads.
- P5 Ledger index build (IVF-PQ), in dusk only.
- P6 Eval suite.

Data stores
- D1 configs/*.yaml.
- D2 Raw-text log: pointers into the dataset at training; tenant text at inference (optional).
- D3 Ledger per column: keys, values, salience, consolidated flags; per tenant at inference.
- D4 Checkpoints: weights, optimizer state, config hash, dataset manifest.
- D5 Per-tenant slow-weight delta `dW`, fast states `H`, KV windows.
- D6 Held-out set (never replayed, never in the ledger).

Flows and trust boundaries
- B1 Internet to training box: E1 -> P1/P2 (dataset download, text only).
- B2 Tenant to serving process: E2 -> P4 (text), E4 -> P4 (tenant handle).
- B3 Between tenants: D3/D5/D2 of tenant A vs tenant B inside one serving process.
- B4 Disk to process: D1, D2, D3, D4 loaded by P2/P3/P4.
- B5 Model to slow weights: P3 writes what P4 absorbed (learning from untrusted text).

## 2. What can go wrong, and what we do about it

| # | Element | STRIDE | Threat | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | E2, B2 | S | A tenant claims another tenant's identity and reads their ledger or inherits their `dW` | Vesper never derives a tenant from request content; the host (E4) authenticates and passes an opaque handle, which is the only key for D2/D3/D5. ASVS V4.1 (access control at a trusted layer) | design rule: spec "Deployment" |
| 2 | E1, B1 | T | Poisoned public text shapes the model | Accepted for pretraining as for every LM; no LLM-generated data (spec); held-out eval and dusk-gain forgetting check detect gross damage; dataset manifest (hashes) pinned in the checkpoint | design rule; manifest open |
| 3 | D1, B4 | T | A wrong or hostile config silently changes behavior (typo, `yes` for 1, `3e-4` as a string) | Typed fail-closed loader: unknown keys, missing keys, bool-as-number, string-as-number and inconsistent relations all raise before anything runs | enforced: vesper/config.py, tests/test_config.py |
| 4 | D4, B4 | T, E | A tampered checkpoint executes code on load (pickle) or swaps weights | `torch.load(..., weights_only=True)` only; a sha256 manifest of every file is written at save and verified before load; mismatch refuses to load | open (train.checkpoint) |
| 5 | D3, B3 | T, I | One tenant's text lands in another tenant's ledger, state or weights (cross-tenant leakage or poisoning) | Everything that learns at inference is per tenant: `H`, KV windows, ledger, raw-text log, `dW`; base weights are shared and read-only | design rule: spec "Deployment"; test open: a two-tenant run shows zero cross-tenant retrieval hits |
| 6 | B5 | T | A tenant feeds adversarial text so dusk steers the model (training-data injection) | Damage is bounded to that tenant's own `dW`; `eta_slow` is tunable to 0 per deployment; dusk-gain eval on an unrelated held-out set is the detector | design rule; accepted residual: a tenant can degrade only their own model |
| 7 | P3, D3 | R | No record of what dusk wrote into the weights or which entries were replayed | Ledger is append-only with salience and consolidated flags; dusk logs the ids it replayed and the loss before/after; training logs the config hash and manifest | design rule; logging open |
| 8 | D2 | I | The raw-text log holds user conversations in clear | Off by default; per tenant; encrypted at rest when on; an entry is deleted once its ledger entries are consolidated (data minimization). ASVS V8.1, V8.3 | design rule: spec "Deployment"; implementation open |
| 9 | D3 | I | Ledger values are learned representations of tenant text and can be inverted | Treated as tenant data: same storage protections as D2, per tenant, evicted by capacity, never exported | design rule |
| 10 | P2, P4, P6 | I | Metrics or logs leak text | Logs carry losses, iteration histograms, open rates and ids, never token strings; eval needles live only in the eval set | rule for every logger; review item |
| 11 | P4 | D | The retrieval gate opens on every token and the CPU index saturates | Gate budget `lambda_g`, open rate as a first-class metric, one query per token per column, and a hard cap on queries per chunk as a runtime limit | design rule: spec 3c; runtime cap open |
| 12 | D3 | D | A tenant floods the ledger | Per-tenant capacity `M`, eviction FIFO among the lowest-salience quartile | design rule: spec component 5 |
| 13 | P3 | D | Dusk at inference consumes the machine while a tenant waits | Idle timer only, `budget_tokens` bound, interruptible | design rule: spec component 6 |
| 14 | D1 | D | A config that cannot run (`block` not dividing the chunk, `t_max_anneal_from > t_max`, a cadence shorter than a step) wastes a run | `validate` rejects these relations before training starts | enforced: vesper/config.py `validate`, tests/test_config.py |
| 15 | P2, P4 | E | Dataset or tenant text is executed (code corpora, YAML, pickle) | Text is only ever tokenized; `yaml.safe_load` only (bandit B506 in the gate); no pickle except `weights_only` checkpoints | enforced for YAML: bandit in gate.sh; checkpoints open |
| 16 | kernels | E | Triton kernels compile at runtime; shapes come from data | Kernels take shapes only from the config and the tensor contract (`check_shape` at every boundary); no tenant-controlled kernel parameter | design rule; contract primitive enforced: vesper/contracts.py |

## 3. Did we do a good job

Every row names a mitigation or an explicit acceptance (rows 2 and 6 carry accepted residual
risk). Enforced today: rows 3, 14 (config), 15 (YAML), 16 (contracts). The rest become code with
the components they protect; each "open" item is a line in the registry's rule column or a test
to write before that component is called done. Rows 5 and 8 are the two that change the serving
design and were folded into the spec on 2026-10-14.
