# Review question: the predictive-coding target in local mode

Date: 2026-10-14; corrected the same night after review: the first version stated the
wraparound identity and then drew the opposite conclusion from it. Author: maintainer A. Reviewer:
Maintainer B. Decision: the maintainers; the project owner may override. Not urgent: local mode comes after the e2e
milestone, and `vesper.learning.wake_loss` refuses `mode="local"` until this is settled.

## What the spec says

"Mode local", signal 2: column `c` predicts "the input of column `c+1` on the next pass",
`L_pc,c = || Q_c(x_c) - sg(x_{c+1}) ||^2`, and "column `C` predicts column 1's input at the next
iteration". Two things are left open: which pass `x_{c+1}` is taken from, and whether `x_c`
denotes column `c`'s input or its output.

## Indices

Write `u_c^t` for the input of column `c` at pass `t` and `y_c^t` for its output. In the
residual stack with the per-iteration conditioning vector `u_t`:

```
u_{c+1}^t = y_c^t + u_t          for c < C     (same pass, next column)
u_1^{t+1}  = y_C^t + u_{t+1}                    (the wrap: next pass, first column)
```

Whether `u_t` is added to every column's input or to column 1's only is a convention the spec
must name ("Column input at iteration `t` is `x + u_t`"); the target excludes it either way,
since the consumer adds it.

## Readings

1. **Same pass, next column.** Target `u_{c+1}^t = y_c^t + u_t`. If `x_c` in `Q_c(x_c)` is
   column `c`'s output, the identity plus the known offset solves it exactly; the loss is not
   identically zero for an arbitrary `Q_c`, it becomes zero once `Q_c` learns the identity. If
   `x_c` is the input, `Q_c` has to learn the column's own function, which the column already
   computes. Either way the target holds nothing the column does not have in the same pass.
2. **Next pass, next column** (the first version's proposal). For `c < C` the target
   `u_{c+1}^{t+1} = y_c^{t+1} + u_{t+1}` is column `c`'s own output one pass later: predicting
   it means predicting a full loop iteration, which is the intended predictive coding across
   time. At the wrap the spec's "column 1's input at the next iteration" is
   `u_1^{t+1} = y_C^t + u_{t+1}`: column `C`'s *current* output plus a known offset, so for
   column `C` this reading collapses into reading 1. The first version claimed the opposite.
3. **Same column, next pass, for every column.** Target `y_c^{t+1}` for all `c`, conditioning
   excluded. Nontrivial for every column, identical to reading 2 for `c < C`, and it departs
   from the spec's letter only at the wrap, where the letter is degenerate.

## Missing targets and masks

- No target at `t = T_max` (there is no next pass): the term runs over `t = 1 .. T_max - 1`.
- A token that halts at pass `t` has no genuine `y^{t+1}`: its later-pass values are
  copy-forward state updates, not column outputs (pass-state review, section D). It is masked
  from the pass it halts in onwards.
- `sg` on the target as the spec says; in local mode the stop at a pass boundary must also cut
  the copied tensors (pass-state review, section C), which separate state identities alone do
  not do.
- The loss is summed over the surviving `(c, t)` pairs and averaged over unmasked tokens; the
  normalizer is part of the contract, not an implementation detail.

## Proposal

Reading 3: every column predicts its own output at the next pass, conditioning excluded,
over passes `1 .. T_max - 1`, halted tokens masked from their halting pass on. `Q_c` is
`d x d` per column as the spec says; the `P_c` next-token heads are the tied embedding
(`vesper/embedding.py`). The spec's sentence about column `C` changes to say this. Until the
maintainers agree and the spec says it, `wake_loss` refuses `mode="local"` with a pointer here,
pinned by `tests/test_learning.py`, so nobody trains local mode on a guess.
