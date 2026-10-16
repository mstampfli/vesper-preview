# Attention and FFN review, 2026-10-14

Owner: maintainer B. Reviewed tip: `d2f9579ec15c083ac76a7b0811e4fd067433bdd0`.
Verdict: request changes to attention. No FFN defect found in the checked formula,
shape, sparsity and gradient cases. No Vesper commit or merge made by this review.

The original attention/FFN tests pass (12 tests). Independent fixed-weight examples
also agree, but low-precision rotary, empty input, and cache storage expose gaps
that the existing float64 oracle and logical-shape assertions do not cover.

## A. Independent expectations, without project() or output()

### Two tokens with nontrivial projections and rotary

Use one head of width two, window two, positions `(0,1)`, inputs `(1,0)` and
`(0,1)`, `Wq=Wk=I`, `Wv=diag(2,3)`, and `Wo=[[1,1],[2,-1]]`.

The first output is `(2,4)`. At the second token the two logits are
`-sin(1)/sqrt(2)` and `1/sqrt(2)`. If `a` is the first softmax weight, the output is
`(2a+3(1-a), 4a-3(1-a))`, numerically
`(2.786190991323584, -1.5033369392650897)`.

The probe computes sine and softmax with scalar arithmetic and matches the module
to `1e-12`. It does not ask the module to produce its own reference projections.

### Rotary pairs, GQA, and FFN

- A four-coordinate rotary example uses angles 2 and 0.02, with each pair's sine
  and cosine calculated independently. It checks pairing, sign and frequency.
- A window-one, four-query-head/two-KV-head example has exactly one allowed key.
  One-hot value projections and a diagonal output matrix give an explicit
  eight-coordinate output. This checks GQA replication and output ordering.
- For FFN input `(2,-1)`, fixed matrices produce hidden preactivations
  `(2,-1,-2,1)` and the independently calculated output `(8,-6)`. The check
  distinguishes squared ReLU from ReLU and exercises negative hidden units.

These checks pass on the reviewed tip. They complement the original randomized
full-mask and window-boundary tests; they do not replace them.

## B. Changes required before acceptance

### 1. Rotary phase precision follows the activation dtype

`pos.to(x.dtype)` and the frequency calculation use the activation dtype.

| Reproduction | Observed result |
| --- | --- |
| Float16 `(1,0)` at positions 100000 and 100001 | Non-finite rotated values |
| Bfloat16 `(1,0)` at positions 256 and 257 | Identical rotated values; maximum error versus scalar float64 expectation about 0.859 |

The first case overflows the position conversion; the second rounds distinct
positions together. These are valid int64 positions and finite activations.
The advertised long-session behavior makes this an operational issue.

Compute frequencies and phases in an appropriate wider dtype independently of
the activation dtype, and define how results are cast back. Add independent
low-precision expectations at large positions, not only norm preservation or
a common-position-shift check using the same helper on both sides.

### 2. The logical cache bound does not bound its backing allocation

`cat(... )[:, -w:]` returns a view of the full concatenation. With 4096 incoming
entries, window 16, batch 1, one KV head and head width 2 in float32:

```
returned key shape:       [1, 16, 1, 2]
logical key bytes:        128
retained storage bytes:   32768
```

This was measured with `untyped_storage().nbytes()` on inputs without a gradient
graph. The cache keeps 256 times its logical key storage in this case; detaching
the view does not release its backing storage.

Retain only the needed input/cache tails before allocating the returned cache,
or implement the intended ring representation. Verify backing capacity after a
large push and after repeated single-token pushes, plus values/positions and
input-cache non-mutation where the functional reference promises it.

An ordered tail can be value-equivalent to a ring while having different storage
and append costs. State that distinction in the registry; do not claim ring
resource behavior from a last-window shape assertion. An optimized ring may be
a separate implementation tested against the bounded functional reference.

### 3. Empty attention raises at cat([])

An input `[1,0,2]` with int64 positions `[1,0]` and a valid cache passes the
existing shape checks, then raises `ValueError: torch.cat(): expected a non-empty
list of Tensors`. Define the zero-token result and preserve the carried cache.
Keep shape/window/position validation in front of that path. Test both an empty
and a prefilled cache.

### 4. The public rotary boundary silently broadcasts a wrong batch

`rotary(x, pos)` with `x` shape `[1,2,1,2]` and `pos` shape `[2,2]` is accepted and
returns `[2,2,1,2]`, violating its same-shape contract. The project method validates
positions, but direct callers of the registered primitive do not receive those
checks. Use the existing shape primitive to bind batch/sequence axes and enforce
the int64 position contract. Exercise these rejected inputs directly.

### 5. Rotary base is an architectural constant outside configuration

`_ROTARY_BASE = 10000.0` is a new architectural choice, not a numerical floor.
CONTRIBUTING requires such constants to come from typed configuration. Thread it
through the config/constructor/primitive contract and validate it; an independent
non-default-base example should reject an accidentally hard-coded implementation.

## C. What the existing tests miss

The full-mask oracle shares `project()` and `output()` with production. It is a
good blocking/masking check but cannot independently establish those projections
or rotary's coordinate convention. Norm preservation and relative-position tests
alone do not determine the intended rotation or cover activation-dtype overflow.
The cache test checks visible contents, not retained storage or append complexity.

The reported numerical probes use CPython 3.12.14 and PyTorch 2.14.1+cpu. Their
source and results accompany the review. They are diagnostic examples, not a
passing regression suite for the known defects. The corresponding failing tests
should accompany the owner's fixes. GPU kernels and end-to-end training remain
outside this review's evidence.

## Outcome, appended 2026-10-14 by the other maintainer

All five changes were accepted, each reproduced on d2f9579 before the fix, and applied in
commit 8bafe9d on dev/attention-ffn: rotary phases in float64 whatever the activation dtype
with cos and sin cast afterwards; the cache push allocating exactly min(m + n, w) entries, its
inputs untouched, storage asserted after a 4096-token push, forty single pushes and a
partial-window push; a zero-token chunk returning [b, 0, d] with the cache unchanged after the
same validation; rotary binding its own axes and the int64 contract; the base threaded from
model.rotary_base through the typed configuration, validated above 1, with a base-2
expectation. The passing independent examples of section A are tests now
(tests/test_attention_independent.py), recomputed from scalar arithmetic.
