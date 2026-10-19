# Related work

Recent papers that build parts of what Vesper builds, read from the papers themselves (PDF text),
with what they mean for Vesper. Read 2026-10-17; each entry names the parts read. This page is a
living one: newer entries go on top, and an entry is corrected, not deleted, when a fuller reading
changes it.

| Paper | Date | Scale | Loops | Per-token depth | Delta-rule fast weights | Consolidation |
| --- | --- | --- | --- | --- | --- | --- |
| Vesper (this repo) | - | stage-1 toy; Vesper-S about 60 M (spec) | up to 8 passes per token | yes (ACT halting, not yet stable) | yes, one state per pass | dusk into slow weights (designed) |
| Do Language Models Need Sleep? | 2026-05/06 | 4-10 layers from scratch; 1.4-2 B fine-tuned | N offline passes before each eviction | no | yes (Gated DeltaNet blocks) | yes, into fast weights |
| LT2: Linear-Time Looped Transformers | 2026-05 | 0.6 and 1.3 B on 100 B tokens; a 1.4 B conversion | fixed T in pre-training | tried and dropped (appendix A) | yes (GDN among the mixers) | no |
| ALoDLM | 2026-10-03 | 1.7 and 8 B | yes, up to 4 | yes, sampled exits in training, a threshold at inference | no (a transformer core) | no |
| Continuous depth batching | 2026-08/09 | serving Ouro 1.4 B, Huginn 3.5 B | the served models' | yes, at inference | no (attention KV only) | no |

## Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference

Lee, McLeish, Goldstein, Fanti (Carnegie Mellon, University of Maryland); arXiv 2605.26099 v3,
2026-06-05. Read: pages 1 to 9 (introduction, method, all three experiments).

- What it is: an attention plus Gated DeltaNet hybrid whose attention cache is cleared every L
  tokens. Before each clearing the model makes N extra passes over the window ("sleep"), and
  those passes update the fast weights in the delta-rule blocks; prediction afterwards is one
  ordinary pass. Trained end to end through all the passes.
- Results: on Rule 110 at 32 steps, no extra pass stays near 10 % after about 5 B tokens, 2
  passes reach about 20 %, 3 and 4 passes above 30 %. On Depo, a k-hop lookup over a shuffled
  directed cycle (k from 1 to 16), the 1-pass model makes little progress at 4 hops and beyond,
  2 passes stall at 8 and beyond, and only 4 passes start on 16 hops. Fine-tuning Ouro 1.4 B on
  GSM-Infinite with 4 passes lifts 6-operation problems from 0.419 to 0.615.
- Their central claim: a fast-weight memory that has room to store the facts still fails as the
  required reasoning depth grows; the limit is computation, not capacity.
- Overlap with Vesper: delta-rule fast weights, extra passes for depth, and a multi-hop task on a
  shuffled cycle that is the same family as our stage-1 task (8-node cycles, 1 to 4 hops).
- Difference: their extra passes run over a whole window at a fixed count before the cache is
  cleared, not per token; there is no halting; and their "sleep" writes into the fast weights,
  where Vesper's dusk is designed to write into the slow weights.
- For Vesper: the strongest outside support so far for the stage-1 hypothesis that multi-hop
  lookups over a fast-weight memory need recurrent depth, and Depo is a public benchmark we
  could run for comparison.

## LT2: Linear-Time Looped Transformers

Deng, Zhang, Zhu, Xu, Liu, Ng, Chen (Rice, Apple, UC Santa Cruz, Carnegie Mellon); arXiv
2605.20670 v2, 2026-05-22. Read: pages 1 to 3 (introduction, architecture) and appendix A
(adaptive computation).

- What it is: a looped transformer whose attention is replaced by linear attention (including
  DeltaNet and Gated DeltaNet) or sparse attention, plus hybrids. Pre-trained at 0.6 and 1.3 B
  parameters on 100 B tokens of FineWeb-Edu, the corpus Vesper-S is planned on. The hybrid with
  a few full-attention layers is reported 2.1 points above the plain looped transformer at about
  5 times the decoding speed at 8k context; a converted Ouro-hybrid-1.4B is reported competitive
  with 4 B open models.
- Per-token halting: appendix A describes the same rule Vesper uses (ACT: halt when the summed
  halting probabilities pass 1 - 0.01) and explains why they dropped it in pre-training: the
  halting rule is unstable, and small changes of the ponder weight move the model between
  "halting after a single iteration on every token" and "never halting". They report that Ouro
  saw the same and that their own preliminary runs reproduced it, and leave a stable variant for
  future work.
- Overlap with Vesper: the core is the same shape, a looped stack with a delta-rule memory, at
  about 20 times the size of Vesper-S.
- Difference: a fixed number of loops; no per-token depth in the main model.
- For Vesper: an outside report of exactly the collapse Vesper hit on 2026-10-15 (every token at
  one pass), at 1 B scale. Making per-token depth trainable is not a solved part that others
  have; it is the open problem stage 1 aims at.

## ALoDLM: Adaptively Looped Diffusion Language Models

Fang, Li, Kim and others (University of Illinois Chicago, Amazon AGI, Korea University); arXiv
2610.04198, 2026-10-03. Read: abstract and introduction (page 1) on 2026-10-17; the whole paper
but the reference list on 2026-10-18 (method, experiments, appendices B to D). Code and an 8 B
checkpoint are public.

- What it is: a diffusion language model (it fills in many tokens in parallel) whose tokens get
  different numbers of recurrent passes: ready tokens are committed, unresolved ones keep
  refining. Each token's computation schedule is treated as a latent variable and learned with a
  variational objective (a conditional NELBO). At 1.7 and 8 B it is reported above every
  evaluated diffusion and autoregressive baseline on the average of eleven benchmarks.
- Overlap with Vesper: per-token adaptive depth, trained end to end, at scale.
- Difference: diffusion, not token by token; no recurrent memory described in the part read.
- For Vesper: a working way to train per-token depth at 8 B, with depth as a latent variable in
  a probabilistic objective, close in spirit to the PonderNet-style objective proposed for Vesper
  on 2026-10-15 (docs/reviews/2026-10-15-halting-objective-adversarial.md). Worth reading in full
  before Vesper's halting objective is redesigned.
- How the halting is trained (full reading, 2026-10-18). Tokens really stop during training:
  each rollout SAMPLES its exit decisions from the gate's hazards (the sigmoid of an exit-gate
  logit read from the pass's readout through a stop-gradient, so the gate's loss never trains the
  backbone; a forced exit at K = 4), and
  stopped tokens change what the others see from then on. Exploration is that sampling; there is
  no floor or temperature schedule. At inference the stop is deterministic: the loop ends when
  the mean cumulative halt probability of the open tokens reaches a threshold q (q is their
  test-time compute knob: raising it from 0.1 to 0.4 moves 1.6 to 2.34 passes per token).
- The gate learns by a score-function (REINFORCE) estimator on the whole sequence's cost, the
  sampled-exit losses plus a depth regularizer, with the first-pass loss as its baseline (it
  precedes every sampled exit, so it needs no extra pass). The backbone learns from every pass up
  to the sampled exit, weighted by the detached exit probabilities with the remaining mass on the
  sampled exit; without that, gradient variance was 1.4 to 1.8 times higher.
- The regularizer is a KL to a geometric prior over depths (c = 0.4 at K = 4, prior mean about
  2.0), split into a per-token part (weight 0.1) and an aggregate part over the batch's mean depth
  distribution (weight 1): the same local-versus-aggregate distinction as Vesper's conditions C and
  D, weighted toward the aggregate.
- They start from a pretrained Qwen3 and fine-tune on 5 B tokens: a warm start, as in Vesper's
  adopted halting-in-phases plan (docs/reviews/2026-10-17-halting-in-phases.md, NOT PROVEN).
- For Vesper's open "form of exploration" (that plan): ALoDLM is a scaled example of the
  sampling option, not of a floor; a floor could still be layered on the sampled hazard. Their
  stopped tokens become discrete context, where Vesper's write is copied forward into the deeper
  passes' states; the estimator does not depend on which.

## Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching

Schwethelm, Rueckert, Kaissis; arXiv 2608.09444 v2, 2026-09-25. Read: the scheduling challenges
and the depth-aware KV cache (sections 2 and 4.2).

- What it is: a serving system for looped models whose tokens stop at different depths,
  regrouping tokens between loop steps; up to 99 % of the estimated maximum speedup on Ouro
  1.4 B and Huginn 3.5 B.
- The "missing state" problem: a token that stops early writes nothing for the deeper loop
  steps, but later tokens that go deeper need something to attend to. Their two fixes are for
  attention caches only: copy the token's last keys and values into the skipped slots, or keep
  one shared slot per token that every loop step overwrites.
- Overlap with Vesper: the same problem Vesper's copy-forward solves.
- Difference: no recurrent state. Copying cannot work for a delta-rule state, because each write
  reads the state it updates; Vesper replays the stopped token's write into every deeper pass's
  own state (docs/reviews/2026-10-15-halting-diagnosis.md and the 2026-10-15 notes on Ragged KV
  and CHASE).
- For Vesper: the copy-forward of recurrent writes still looks unaddressed in what was read.

## What this means, in short

Each of Vesper's parts has active, larger work behind it: loops with delta-rule memory (LT2),
extra passes to turn fast weights into reasoning (the sleep paper), per-token depth at scale
(ALoDLM), serving ragged depth (continuous depth batching). None of the four combines per-token
depth with a delta-rule memory per pass and exact copy-forward, and LT2 reports that per-token
halting in a looped model is still unstable in pre-training. Vesper's open question is therefore
a real one, and the multi-hop cycle task it uses is shared with the sleep paper's Depo.
