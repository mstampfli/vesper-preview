# Recurrence review, 2026-10-14

Owner: maintainer B. Base: `c0210ad`. Status: implementation and tests are an uncommitted
proposal for maintainer A's review and the project owner's acceptance. No Vesper commit or push was made.

Scope: `vesper/recurrence.py`, its registry/spec contracts, and the original
recurrence tests. The per-pass state/halting decision and baseline arithmetic
remain separate reviews.

The blocked formula implements the stated delta rule. Independent checks agree
with its outputs and carried state. The original tests nevertheless allow a
shared partial-write error, and they miss numerical failures in key normalization
and the zero-length scan. The proposed patch fixes those boundary failures.

## A. Independent derivation

Work with one head and one block. Let `d_i` be a row vector:

```
d_i = beta_i (v_i^T - k_i^T H_{i-1})
H_i = H_0 + sum_{j<=i} k_j d_j

d_i + beta_i sum_{j<i} (k_i^T k_j) d_j
    = beta_i (v_i^T - k_i^T H_0)
```

Stack `K`, `V`, and `Q` as rows and put `B = diag(beta)` and
`L = tril(B K K^T, -1)`. Then:

```
(I + L) D = B (V - K H_0)
T = (I + L)^-1 B
D = T V - T K H_0
H_end = H_0 + K^T D
O = Q H_0 + tril(Q K^T) D
```

These are the expressions in the implementation. The output includes the diagonal
of `Q K^T`: each token reads after its own write. The block carries `H_end` into
the next block. A unit lower-triangular matrix is invertible; that fact alone is
not a condition-number bound or a universal floating-point error guarantee.

The algebra does not require normalized keys. The contraction argument does:
with `||k|| <= 1` and `beta` in `[0,1]`, the erase factor has eigenvalues in `[0,1]`.
The exact-overwrite property at `beta = 1` requires `||k|| = 1` specifically.

## B. Independent two-token oracle

The expected numbers below were calculated by scalar arithmetic, not by calling
either scan to generate the other's expectation. Keys are nonorthogonal, queries
differ from keys, write strengths differ and are partial, and `H_0` is nonsymmetric.

```
H_0 = [[1, 2], [3, 4]]

token  k             v        q        beta
1      (3/5, 4/5)    (5,-2)   (2,-1)   1/2
2      (0,1)         (-3,1)   (-1,3)   3/4

H_0^T k_1 = (3, 22/5)
d_1 = (1, -16/5)
H_1 = [[8/5, 2/25], [19/5, 36/25]]
o_1 = (-3/5, -32/25)

d_2 = (-51/10, -33/100)
H_2 = [[8/5, 2/25], [-13/10, 111/100]]
o_2 = (-11/2, 13/4)
```

`test_two_token_rational_oracle` checks both intermediate step states, both
outputs, and the blocked final state. Block sizes 1, 2, 3, and 64 exercise a
block boundary, one complete block, and partial blocks.

## C. The shared-error question, demonstrated

In an isolated copy of `c0210ad`, insert `beta = beta.square()` in BOTH scan
functions before they use beta. This changes the specified partial-write strength.

| Check | Result on the deliberately wrong pair |
| --- | --- |
| All original recurrence tests | 9 passed |
| New rational oracle, four block sizes | 4 failed |

Why the old checks miss it:

| Original check | What it establishes | What it can miss |
| --- | --- | --- |
| Step versus blocked outputs/state | Agreement of two implementations on the sampled inputs | A shared wrong update, including beta squared |
| Block size one | One-token blocking agrees with step | The same shared update error |
| Beta-one overwrite | The unit-key overwrite property of the step form | Wrong behavior for partial beta; beta squared still equals one at one |
| Reversed erase-order counterexample | The specific old right-erase formula is wrong | Other mistakes that preserve the overwrite endpoint |
| Gradient comparison | Both implementations differentiate the same sampled computation | Derivatives of a shared wrong computation |
| Norm bound | The sampled state stays under a write-budget bound | Smaller wrong writes; beta squared is at most beta on `[0,1]` |
| Shape/block rejection | The named invalid inputs are rejected | Valid edge cases not exercised |

This does not mean the old tests lack value. It identifies the independent
expectation they needed. Mutation logs accompany the review artifact.

## D. Reproduced defects and proposed fixes

### 1. Key normalization can violate its own stability premise

The old `sqrt(sum(k.square()))` has three observed failures:

- Zero keys give the documented zero output but NaN gradients in float32,
  float64, and bfloat16. Clamping after the square root does not repair its
  undefined derivative at zero.
- The float16 key `(1e-4, 0)` becomes `(98.6875, 0)` after its square underflows.
  The erase factor can therefore amplify the state instead of contracting it.
- Finite float32 `(1e20, 0)` and float64 `(1e200, 0)` become zero after squaring
  overflows. The latter also loses the nonzero, representable derivative.

The fix retains the existing denominator floor, `eps = 1e-6`:

```
s = max(max(abs(k)), eps)
z = k / s
result = z / sqrt(max(sum(z*z), 1))
```

If `s > eps`, a component of `z` has magnitude one, so `||z|| >= 1`. Otherwise
`s = eps`. In either case this is exactly `k / max(||k||, eps)` in real arithmetic.
The square root's input is at least one, and scaled components cannot overflow
when squared. The sum uses float32 accumulation except for float64 inputs;
a 65,536-component float16 row tests the accumulation overflow case.

The auxiliary positive scale is detached from autograd because it cancels from
the mathematical function. Independent derivatives at ordinary and extreme
magnitudes, plus the floored zero-key derivative, check that this retains the
intended first derivative. Empty key arrays retain their shape and gradient path.

Limit: the zero-key derivative is `1/eps = 1e6`, which exceeds float16's finite
range. This patch does not promise representable gradients in every dtype.
The finite zero-gradient checks cover float32, float64, and bfloat16. No GPU or
higher-order-gradient claim is made.

### 2. A zero-token scan is not defined by the implementation

The input shapes accept `n = 0`, but `torch.cat([])` raises. The fix returns the
empty query tensor and the unchanged carried state, with both gradient paths
preserved. Shape and block validation still run first; an empty input does not
make an invalid block valid.

### 3. Clarify the actual numerical contracts

- Floating-point sigmoid can return 0 or 1. The stability argument permits both.
- A nonzero key below the norm floor is subunit, not unit. For such a key the
  effective write toward `v` when reading the same key is `beta * ||k||^2`.
- The exact beta-one overwrite statement applies to unit keys, excluding zero
  and floor-limited keys.

The accompanying spec and registry wording is proposed clarification of these
existing numerical choices, not a new recurrence or halting policy.

## E. Validation and remaining limits

- Pinned original recurrence suite: 9 passed.
- Final new review checks against unmodified `c0210ad`: 11 failed, 10 passed.
- Patched recurrence suites: 30 passed.
- Full project gate: format, lint, mypy, basedpyright, security scans, dependency
  audit, and 54 tests passed; `GATE GREEN`.
- Runtime: CPython 3.12.14, PyTorch 2.14.1+cpu, locked dependencies.

The deterministic oracle and mutation address shared correctness errors; the
existing random multi-head tests remain useful for indexing and block layouts.
These CPU checks do not establish GPU kernel correctness, every possible rounding
error, a model-training result, or the unimplemented pass-major halting contract.
Maintainer A's independent review and the project owner's merge decision remain outstanding.
