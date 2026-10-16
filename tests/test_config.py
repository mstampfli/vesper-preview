from pathlib import Path
from typing import cast

import pytest
import yaml

from vesper.config import ConfigError, load, parse

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _raw() -> dict[str, dict[str, object]]:
    with (CONFIGS / "vesper-s.yaml").open(encoding="utf-8") as fh:
        return cast("dict[str, dict[str, object]]", yaml.safe_load(fh))


def test_vesper_s_matches_the_spec() -> None:
    cfg = load(CONFIGS / "vesper-s.yaml")
    assert (cfg.model.d, cfg.model.heads, cfg.model.head_dim) == (768, 12, 64)
    assert (cfg.model.columns, cfg.model.t_max, cfg.model.vocab) == (4, 8, 32768)
    assert (cfg.model.window, cfg.model.retrieval_k) == (512, 32)
    assert cfg.ledger.capacity_per_column == 2**20
    assert cfg.dusk.cadence_tokens == 2**22
    assert cfg.optimizer.peak_lr == 3e-4
    assert cfg.training.chunk_tokens * cfg.training.batch_chunks == 524288
    assert cfg.learning.mode == "e2e"
    assert cfg.modulator.half_life_tokens == 10000
    assert cfg.model.rotary_base == 10000.0


@pytest.mark.parametrize("base", [1.0, 0.5, float("nan"), float("inf")])
def test_rotary_base_must_be_finite_and_above_one(base: float) -> None:
    raw = _raw()
    raw["model"]["rotary_base"] = base
    with pytest.raises(ConfigError, match="rotary_base"):
        _ = parse(raw, "test")


def test_missing_key_fails() -> None:
    raw = _raw()
    del raw["model"]["d"]
    with pytest.raises(ConfigError, match="missing key 'd'"):
        _ = parse(raw, "test")


def test_unknown_key_fails() -> None:
    raw = _raw()
    raw["model"]["dd"] = 1
    with pytest.raises(ConfigError, match="unknown keys: dd"):
        _ = parse(raw, "test")


def test_bool_is_not_an_int() -> None:
    raw = _raw()
    raw["model"]["d"] = True
    with pytest.raises(ConfigError, match="expected an integer"):
        _ = parse(raw, "test")


def test_unquoted_scientific_notation_trap_is_caught() -> None:
    # PyYAML 1.1 reads "3e-4" as a string; the loader must refuse it, not coerce it.
    raw = _raw()
    raw["optimizer"]["peak_lr"] = "3e-4"
    with pytest.raises(ConfigError, match="expected a number"):
        _ = parse(raw, "test")


def test_inconsistent_shapes_fail() -> None:
    raw = _raw()
    raw["model"]["head_dim"] = 32
    with pytest.raises(ConfigError, match=r"heads \* head_dim"):
        _ = parse(raw, "test")


def test_bad_mode_fails() -> None:
    raw = _raw()
    raw["learning"]["mode"] = "both"
    with pytest.raises(ConfigError, match=r"learning\.mode"):
        _ = parse(raw, "test")


def test_negative_write_odds_multiplier_fails() -> None:
    # alpha 1 with the clip at -1 is a zero multiplier and allowed; below the clip it goes negative
    raw = _raw()
    raw["learning"]["alpha"] = 1.0
    raw["learning"]["m_clip_low"] = -1.0
    _ = parse(raw, "test")
    raw["learning"]["m_clip_low"] = -1.5
    with pytest.raises(ConfigError, match=r"alpha \* m_clip_low"):
        _ = parse(raw, "test")


def test_alpha_must_be_finite_and_non_negative() -> None:
    # alpha -0.5 passes the lower-clip product (1 + 0.5 = 1.5) but turns m = 3 into a factor of
    # -0.5 inside the allowed clip; the direction of the modulation is a contract, not a sign bit
    raw = _raw()
    raw["learning"]["alpha"] = -0.5
    with pytest.raises(ConfigError, match=r"alpha must be finite and >= 0"):
        _ = parse(raw, "test")
    raw["learning"]["alpha"] = 0.0
    _ = parse(raw, "test")


def test_root_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        _ = parse([1, 2], "test")


def test_checkpoint_and_evaluation_counts_are_read_and_bounded() -> None:
    cfg = load(CONFIGS / "vesper-s.yaml")
    assert cfg.training.checkpoints_per_run == 20  # every 5% of the tokens (spec: Evaluation suite)
    assert cfg.training.eval_chunks == 32
    for key in ("checkpoints_per_run", "eval_chunks"):
        raw = _raw()
        raw["training"][key] = 0
        with pytest.raises(ConfigError, match=key):
            _ = parse(raw, "test")
