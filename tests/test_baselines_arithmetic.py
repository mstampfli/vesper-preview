"""The baselines document's arithmetic, derived from the config.

Every figure in docs/reviews/2026-10-14-baselines.md follows from the tensor shapes the spec
states. Deriving them here from configs/vesper-s.yaml rejects a changed config or a changed
formula against the golden numbers below; it is the check that rejected the first version's
17 d^2 column. It does not read the document or the spec, so their prose can still drift.
"""

import math
from fractions import Fraction
from pathlib import Path

from vesper.config import VesperConfig, load

CFG: VesperConfig = load(Path(__file__).resolve().parents[1] / "configs" / "vesper-s.yaml")
D: int = CFG.model.d
D2: int = D * D
MEAN_PASSES = 4  # the target mean iteration count of the spec's wall-clock estimate


def column_matrices(d: int, dk: int, kv_groups: int, ffn_mult: int) -> int:
    recurrent = 4 * d * d  # W_q, W_k, W_v, W_o of the recurrent core
    attention = 2 * d * d + 2 * d * (
        d // kv_groups
    )  # W_q, W_o full; W_k, W_v at h / kv_groups heads
    retrieval = 4 * d * dk  # W_r, W_rk, W_rv (d x dk) and W_ro (dk x d)
    ffn = 2 * ffn_mult * d * d  # W_1 and W_2
    return recurrent + attention + retrieval + ffn


def column_small(d: int, heads: int) -> int:
    return d * heads + d + 4 * d  # W_beta, the retrieval gate, four RMSNorm scales


def shared(d: int, vocab: int, t_max: int) -> int:
    # tied embedding, u_t, the halting head (direction, norm gain, bias), final norm
    return vocab * d + t_max * d + (2 * d + 1) + d


def vesper_parameters() -> int:
    m = CFG.model
    per_column = column_matrices(m.d, m.head_dim, m.kv_groups, m.ffn_mult)
    per_column += column_small(m.d, m.heads)
    return m.columns * per_column + shared(m.d, m.vocab, m.t_max)


def dense_parameters(layers: int, ffn_mult: Fraction) -> int:
    m = CFG.model
    attention = 2 * m.d * m.d + 2 * m.d * (m.d // m.kv_groups)
    ffn = int(2 * ffn_mult * m.d * m.d)
    return layers * (attention + ffn + 2 * m.d) + m.vocab * m.d + m.d


def test_column_coefficient_is_89_sixths() -> None:
    m = CFG.model
    coefficient = Fraction(column_matrices(m.d, m.head_dim, m.kv_groups, m.ffn_mult), D2)
    assert coefficient == Fraction(89, 6)
    assert coefficient == Fraction(29, 2) + Fraction(4 * m.head_dim, m.d)


def test_vesper_s_parameter_count() -> None:
    assert vesper_parameters() == 60_222_721


def test_baseline_a_is_matched_within_a_tenth_of_a_percent() -> None:
    a = dense_parameters(7, Fraction(3))
    assert a == 60_271_872
    assert abs(a - vesper_parameters()) / vesper_parameters() < 0.001


def test_the_first_versions_baseline_was_nine_percent_too_large() -> None:
    a = dense_parameters(6, Fraction(9, 2))
    assert a == 65_873_664
    assert (a - vesper_parameters()) / vesper_parameters() > 0.09


def test_equal_parameter_width_at_four_columns_is_768() -> None:
    m = CFG.model
    target = m.columns * column_matrices(m.d, m.head_dim, m.kv_groups, m.ffn_mult) + m.vocab * m.d
    a = 14.5 * m.columns
    b = 4 * m.columns * m.head_dim + m.vocab
    width = (-b + math.sqrt(b * b + 4 * a * target)) / (2 * a)
    assert math.isclose(width, 768.0, rel_tol=1e-12)


def test_forward_flops_estimate_in_units_of_d_squared() -> None:
    m = CFG.model
    chunk = CFG.training.chunk_tokens
    weights = 2 * Fraction(column_matrices(m.d, m.head_dim, m.kv_groups, m.ffn_mult), D2)
    window_attention = Fraction(4 * m.window * m.d, D2)
    state_active = Fraction(6 * m.heads * m.head_dim * m.head_dim, D2)
    state_copy = Fraction(4 * m.heads * m.head_dim * m.head_dim, D2)
    logits = Fraction(2 * m.vocab * m.d, D2)
    assert (weights, window_attention) == (Fraction(89, 3), Fraction(8, 3))
    assert (state_active, state_copy, logits) == (Fraction(1, 2), Fraction(1, 3), Fraction(256, 3))

    pairs = m.columns * m.t_max
    active = m.columns * MEAN_PASSES
    per_active = weights + window_attention + state_active
    vesper_mean = active * per_active + (pairs - active) * state_copy + logits
    vesper_max = pairs * per_active + logits
    assert vesper_mean == 616
    assert vesper_max == 1136

    a_layer = 2 * Fraction(2 * D2 + 2 * m.d * (m.d // m.kv_groups) + 2 * 3 * D2, D2)
    assert a_layer == 17
    a_rectangular = 7 * a_layer + 7 * Fraction(4 * chunk * m.d, D2) + logits
    a_causal = 7 * a_layer + 7 * Fraction(2 * chunk * m.d, D2) + logits
    assert a_rectangular == Fraction(1061, 3)
    assert a_causal == 279
    assert 1.70 < vesper_mean / a_rectangular < 1.75
    assert 2.20 < vesper_mean / a_causal < 2.25

    b_layer = 2 * Fraction(2 * D2 + 2 * m.d * (m.d // m.kv_groups) + 2 * 4 * D2, D2)
    b_rectangular = 32 * b_layer + 32 * Fraction(4 * chunk * m.d, D2) + logits
    b_causal = 32 * b_layer + 32 * Fraction(2 * chunk * m.d, D2) + logits
    assert b_rectangular == 1440
    assert 1098 < b_causal < 1099
    # Baseline B is not compute-matched: above Vesper at T_max under one convention, below it
    # under the other
    assert b_rectangular > vesper_max > b_causal
