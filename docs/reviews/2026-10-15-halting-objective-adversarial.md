# Adversarial review of the proposed halting objective

Maintainer B, 2026-10-15. Proposal received through maintainer A in channel e742aec, at the project owner's request,
before implementation. Source inspected: 4ad6be2. This is a design review, not an adopted
architecture or a long-run authorization. Runnable owned counterexamples and numeric output:
`<evidence>/reviews/halting-design-20261015/`.

Recommendation: keep normalized head input and investigate all-pass supervision, but do not
implement the proposed frozen-prior/release/KL/compute/gate sequence as one package. It leaves
the distribution, deployed output and state trajectory under-specified. In particular, a
strong per-token uniform KL can oppose the input-specific routing that the proposal wants.
First pin the contracts below; use a small controlled experiment to distinguish them.

## 1. Uniform per-token entropy is not useful depth diversity

Let p_i(t) be token i's distribution over exit depths, u a uniform prior, and p_bar their
mean over an explicitly chosen token population. The identity

    mean_i KL(p_i || u) = KL(p_bar || u) + mean_i KL(p_i || p_bar)

separates marginal-prior matching from dependence between token identity and exit depth.
The last term is that joint distribution's mutual information. At a fixed marginal, local
KL penalizes input-specific routing. This does not forbid specialization: sufficient loss
gain can pay that penalty. It does mean increasing beta is not a neutral way to demand spread.

Exact fixture, T=8: p_i=u for all eight tokens gives local KL=0 and a CDF threshold q=0.5
exits every token at pass 4. Assigning those tokens deterministically to depths 1..8 keeps
the SAME uniform marginal but costs local KL=log(8)=2.07944154. Aggregate KL is zero in both
cases. Randomly sampling depths from the first policy would also make a broad observed
histogram, with no input-conditioned allocation at all.

Compare local KL against aggregate KL as an explicit ablation. Aggregate KL removes the
extra penalty but does not establish useful routing or protect every depth from starvation
on every token. Arbitrary token partitions can satisfy it. A spread histogram is a diagnostic,
not a success criterion: compare held-out quality at equal measured cost with fixed-depth
policies and a matched-cost random policy. Also report conditional depth entropy, marginal
entropy, their difference, and loss benefit versus the selected depth. Identical optimal
depth can be a correct outcome on a particular task.

The fixed-loss local optimum is p_i(t) proportional to exp(-(loss_i(t)+alpha*c(t))/beta).
Different gain profiles CAN give different optima; neither uniformity nor diversity is
mathematically guaranteed. This expression treats losses/costs as fixed, not as an exact
solution of joint recurrent training.

## 2. The global KL has a micro-batch trap in this trainer

Vesper currently weights each micro-group's loss by its loss-position share, backpropagates,
and frees that group's graph before the next group. That works for separable mean losses.
It does not make mean_g KL(p_bar_g || u) equal KL(p_bar_full || u).

Two nearly deterministic, opposite two-depth micro-groups give pooled KL=0 with zero logit
gradient, but averaging their KLs gives 0.69306731 with nonzero gradients. Changing micro-chunk
layout would then change the supposed global objective. Padding, document boundaries, final
short groups and any future device reduction must use an explicit population and denominator.
Decide whether prior matching covers loss positions or all real tokens; never include padding
silently, and do not conflate these two populations.

A real aggregate implementation must keep the connected probabilities until the global
statistic is available, or recompute from identical inputs/parameters/states. For an exact
first-order two-pass implementation, first collect p_bar without updating anything; replay
with detached coefficient beta/N * (log(p_bar/u)+1) multiplying each connected p_i. Log the
actual global KL, not the linear surrogate's value. Replay must not draw streams twice or
advance persistent states twice, and must reproduce stochastic decisions. This costs extra
forwards. An EMA or detached moving target is another objective, not an exact shortcut.
Require both objective and gradient equality against an unsplit oracle before scaling.

## 3. Full losses remove ACT's structural flat region, not every zero gradient

Define conditional hazards lambda_t and survival S_t. For t<T use p_t=S_t*lambda_t; at T
assign all remaining survival to p_T. Then the final forced exit has no independent hazard
parameter. With fixed two-pass losses (2,1), beta=0.1 and first logit 6, p_1=0.997527 and
its expected-loss-plus-KL gradient is 0.0039464. ACT would already stop and lose its local
head gradient; this full-loss construction retains a useful path.

But at equal losses and the uniform optimum, zero gradient is correct. With a first logit
of 40, stable float32 log-space arithmetic retains only about 2.12e-17 gradient in the
fixture; at 1000 it is zero through underflow. A blanket promise that every pass/gate gets
a nonzero gradient, or that collapse is impossible, is false. During a frozen-head phase
the head intentionally receives none; the model's all-pass loss paths are a separate claim.

Specify KL direction, reduction and tail handling. KL(p||u) and KL(u||p) have different
boundary behavior. Compute hazards/log-survival/log-probabilities in a deliberate precision
using log-sigmoid operations, rather than subtracting rounded probabilities from one or
clamping logs without stating the changed objective. Test T=1, forced-tail mass, extreme
logits, padding, unequal valid counts and backward finiteness. A loss-of-weighted-hidden-state,
a weighted loss, and a loss-of-mixture-probabilities are different objectives.

## 4. Uniform warm-up need not add T learned biases

Maintainer A is right that a zero direction and ONE constant bias produces geometric hazards,
not uniform exit probabilities. But learned pass biases are optional. A deterministic
baseline is lambda_t=1/(T-t+1), equivalently b_t=-log(T-t) for t<T, with forced final exit.
A zero learned residual logit on top reproduces uniform without adding learned parameters.
The fixture checks T=1,2,8. Use the actual allowed pass count, including annealing, not a
silently different configured maximum. Test continuity from fixed uniform weights into the
released parameterization; no probability jump or inherited optimizer moments at release.

Also note that frozen uniform EXIT weights and a fixed CDF threshold do not mean full-depth
state writes. At q=0.5 they commit around the midpoint. If the warm-up goal is to teach every
pass from dense histories, explicitly separate the supervision weights from the state policy.

## 5. The compute penalty changes the prior, and q changes the deployed objective

For beta>0, the regularizer and expected cost combine exactly as

    beta KL(p||u) + alpha E_p[c]
      = beta KL(p||u_alpha) - beta log Z,
    u_alpha(t) = u(t) exp(-alpha c(t)/beta) / Z.

If c(t)=t, this is a truncated geometric tilt toward shallow depths. Increasing alpha while
reducing beta strengthens that tilt; the combined preference is no longer uniform. This
can be intentional, but do not describe it as merely removing exploration pressure. A
nonzero beta floor is an experimental setting, not a guarantee against synchronized exits.

Expected stochastic depth is also not deterministic CDF-exit cost. For T=8 and q=0.5,
p=(0.49,0,...,0.51) has expected depth 4.57 and exits at 8; p=(0.51,0,...,0.49) has expected
depth 4.43 and exits at 1. The smooth cost barely changes while deployed depth jumps seven.
Choose sampling, CDF exit, or another policy explicitly and evaluate the one deployed.

Choose the emitted prediction too. For target probabilities (0.9,0.1) and equal exit weights,
expected NLL is 1.20397; mixture-probability NLL is 0.69315; exiting at the first or second
pass gives 0.10536 or 2.30259. Vesper's current hidden mixture followed by RMSNorm/tied head is
yet another function. Keep expected training loss and actual deployed-policy perplexity
separate. Neither their names nor the PonderNet label establishes equivalence.

## 6. State equality and gate credit are two different problems

I agree with maintainer A that dense training and copy-forward inference need not expose the same
history. A scalar fixture using Vesper's actual delta_update makes this exact: key=1,
beta=1 and initial state=0; an earlier token's fresh deep value 2 leaves state 2, while its
frozen shallow value 1 leaves state 1. The next token writes beta=0 and reads query=1. It
observes 2 versus 1. Every future gate can then also see different inputs.

My earlier "continue activations while freezing writes" suggestion needs a further contract.
If that current token's candidate deeper prediction reads its own frozen write, it sees 1,
where a real deeper candidate would see 2. A genuine current-token counterfactual needs a
private fresh candidate write/read against the same committed prefix, while only the chosen
write reaches later tokens. In the fixture the candidate reads 2 but the later token still
reads committed 1. Do not leak the private candidate into the carried state or evaluate it
under an unexplained mixed state. Recurrent and attention paths both need this separation.

Even a correct private-candidate construction misses delayed gate credit if hard commits
are detached. Let the early token's own loss be equal at both depths, and let the next
prediction need state 2: its squared loss is 1 after shallow commit and 0 after deep commit.
A local loss-improvement target sees no benefit to continuing the early token. Compute
pressure favors the harmful early exit. Backbone gradients through committed values do not
supply a derivative through a hard exit decision. Do not call local expected losses an
unbiased objective over complete stochastic state trajectories.

Recommendation: pin a simple reference deployment state policy and measure its trajectory
mismatch first. Dense diagnostic training can establish whether depth has value, provided
its scope stays explicit. Before claiming deployable adaptive depth, include mixed-exit
multi-token/chunk tests and an inference-state training/evaluation path. For the small task,
include a token whose extra processing benefits a LATER token; a purely local task cannot
expose this credit problem. Exact score-function credit or other estimators are separate
cost/variance design choices, not required additions to the first experiment.

## 7. Release criteria and the optional improvement gate can select the wrong evidence

A mean deeper-pass gain of 0.05-0.1 nats is a hypothesis to test, not a universal maturity
threshold. The per-token quantity L1-min_t Lt selects the best of several noisy observations;
it can look positive and skewed when the gain is not predictable from the gate's available
features. It is an oracle diagnostic, not an executable policy. Include a held-out test of
predictability and a fixed-depth reference. Keep controller-development validation distinct
from the final test set; repeated release/threshold choices use validation information.

Specify token budget rather than relying on step count, data identity, state mode, gain
statistic, confidence/stability checks, minimum sample count, evaluation cadence, release
latch and resume state. If the deadline arrives without the criterion, record a failed
criterion and revisit the experiment, rather than silently calling a forced release success.
The 1,000-3,000-step window has not been established by the 40/200-step runs.

One-step improvement supervision is greedy. Losses (2,2.1,0.1) have no gain on the next step
but a large gain one step later. A gate that stops at the first non-improvement misses it.
Also train the continuation decision at t from the improvement t -> t+1, not from improvement
already obtained on arriving at t. The two-pass test must update the first gate to continue
when only pass 2 improves; the forced final gate is not the target. Ouro's equation/prose
indexing is not internally uniform, so pin the intended behavior rather than copying a
symbol. Detached future losses can supervise training; future target tokens cannot be gate
inputs at deployment.

## 8. Cost and a minimal discriminating experiment

Maintainer A's illustrative forward arithmetic is plausible: with columns costing 131 d^2 per
pass, a vocab head costs 2*d*32768 = 85.333 d^2 at d=768. Eight passes change the forward
estimate from 8*131+85.333 to 8*(131+85.333), about 1.527 times. That is not measured whole-step
FLOPs, memory or wall time, and an exact global-KL replay strategy adds its own work.
Eight head evaluations do NOT require eight simultaneous vocabulary-logit allocations:
score passes/slices sequentially with recomputation, preserving gradient equivalence.
Measure activation/recompute cost and actual peak allocation in the complete step.
Inference still owes copy-forward work to every destination; mean active depth alone is not
a wall-clock speedup. No large GPU budget follows from this review.

Start with an explicit readout/state/distribution reference and three controlled conditions:
fixed uniform all-pass supervision, trainable local uniform-KL, and trainable aggregate-KL.
Match data order, model, training tokens and measured cost; use multiple seeds at a small
scale. Keep compute pressure and the proposed release trigger out of the first comparison
so the outcome identifies the objective. Test depth gains on a tiny task with known
heterogeneous useful depth and the later-token dependency above, then a bounded real-data
pilot. Compare deployed-policy quality with static and random policies at matched cost.
Only then choose a release schedule, KL floor/anneal, compute penalty or auxiliary gate.

Acceptance checks: distribution sums/forced tail/uniform release; the analytic gradients
and extreme-logit cases; unsplit versus micro-batch global-KL gradients; no state mutation
from a private candidate; identical committed states across training/inference execution
orders for the same realized trace; actual deployed-policy quality/cost, not just spread.

## Sources and evidence limits

[PonderNet, sections 2.3-2.5](https://arxiv.org/html/2107.05407) distinguishes expected per-step
loss, finite-horizon tail handling and sampled exit predictions from ACT's mixture.
[Ouro, sections 3.3-3.4](https://arxiv.org/html/2510.25741v1#S3.SS3) uses local entropy
regularization and separate gate supervision from detached improvements. Its CDF exit and
objective motivate the contract checks above; they are not Vesper validation.

[Ragged KV, sections 4-5 and Appendix C](https://arxiv.org/html/2610.09013v1#S4) explicitly
compares local with aggregate prior enforcement and studies inference-state adaptation,
including private counterfactuals and delayed cache credit. It reports difficulties with
an early fixed beta and a ragged-prefill failure for one model. These observations motivate
ablations; they do not establish a Vesper schedule or delta-state guarantee.
[CHASE](https://arxiv.org/abs/2607.10110v2) studies adapting looped state-space models to
skipped-state inference; Vesper's copy-forward rule is a different state transition.

The numerical fixtures here are independent CPU calculations with a real delta_update call;
they are not trained-model results. The selected normalized trial log and command/exit artifacts
are now read and match their supplied checksums: 40 steps, 1,310,720 tokens, exit 0, final
training loss 7.66240353. The reported trend toward two passes is supported, with corrections:
step 34 splits 29.66%/70.34% over passes 2/3 (not a maximum split of 93/7); step 35 is 91.27%
at pass 2, rising to 97.20% at step 40. No logged token uses pass 1. This is synchronized drift
with some spread, not an observed return to ACT's first-pass flat region, and not evidence
that the spread is useful. The historical checkpoint histogram still means training.

Normalization correction 4ad6be2 is inspected; targeted checks: 13 passed, six CUDA skips.
The selected full correction gate and merged-provenance 8a7f76a gate report green; the latter
has 379 passes, no skips, exit 0 and dirty=0. The correction's full-gate file was not listed
in the supplied checksum manifest at this check; the run log, command, exit and commit-hook
files were. CPU full gate on this review base: 366 passed, ten CUDA skips, clean audit.
Actual CUDA execution remains maintainer A's selected evidence. The all-pass redesign is not built.
