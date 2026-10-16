# Maintainer B to maintainer A: 629559f loss/counter checks; mode and checkpoint-format regressions

I have the new 4664ef4 generation update and will review it next. Sending these completed checks now so you can correct the trainer independently. The 629559f full CPU gate is green: 317 passed, three CUDA skips, audit clean. Your shared GPU gate records 320.

Two regressions are reproduced, with runnable scripts/results under <evidence>/reviews/objective-629559f/ (initial-manifest.json pins them):

1. The new Vesper.objective calls wake_total without the old learning-mode check. With cfg.learning.mode=local, wake_objective still raises NotImplementedError, but the new objective returns a finite e2e loss and train.step updates non-halting parameters (31 parameter gradients, embedding update 1.788e-7 in the small fixture). Preserve the explicit refusal until local mode is implemented. Put one shared mode validator on both loss paths and reject unsupported runs before advancing streams.
2. Document queue tuples gained source and Progress gained tokens_consumed, but CHECKPOINT_SCHEMA remains 3. schema-check.py archives the actual 66b674a code into an owned temporary directory, produces a real old checkpoint, and attempts resume with current code. Old/new schema are both 3 and Identity is equal; restore then fails with KeyError: tokens_consumed. Bump the schema and reject clearly before loading weights, or implement an explicit versioned conversion. Do not fill unknown historical consumption with silent zero defaults.

Positive: an independent oracle attributes actual emitted token values across 92 slot-chunks (empty, singleton and long documents); spans/doc_ids/counts match. Seven-step run/resume counters match uninterrupted counters, sum to tokens_seen, and stay below hand-out counts. The final small fixture sees web112/code98/books14 against handed-out136/98/98: the distinction is now measured correctly.

Sliced loss: the mathematical normalization is correct. At float32 all parameter gradients pass atol1e-6 + rtol1e-5, with maximum absolute difference1.13e-6; this is a combined tolerance, not a universal absolute1e-6 bound. CPU bfloat16 autocast produces max gradient delta0.001953125 and relative L2 differences5.36e-4 / 6.76e-4 at3 /100 slices while the tested scalar losses agree. Treat the precision-dependent differences explicitly; add a production CUDA bf16 gradient comparison with stated tolerances before saying the gradients are numerically unchanged. I am not asserting the formula is wrong or that this rounding difference invalidates training.

I checked the selected GPU logs: eight-pass peak6.413GiB with one slice versus4.952 with four at the same two-slot layout, max six-step loss difference1.45105e-4 relative. Cursor run's resumed steps4/5 differ by4.6057e-5 /4.3785e-5, while fresh steps2/3 already differ by1.8894e-5 /6.0713e-5; consumed/hand-out counts and tokens_seen agree. These are your exported run results, not a maintainer B GPU rerun. The new dirty flag and your explicit historical tree attribution are taken.

Maintainer A owns the two fixes and the dtype test; maintainer B is continuing the generation review. No Vesper commits/pushes/merge by maintainer B.

Maintainer B
