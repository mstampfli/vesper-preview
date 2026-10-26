from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml

from vesper.config import (
    ConfigError,
    InstructConfig,
    ReplayConfig,
    load,
    parse,
    parse_instruct_bytes,
    validate_instruct,
)

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


def _instruct_raw() -> dict[str, dict[str, object]]:
    with (CONFIGS / "instruct-dolly.yaml").open(encoding="utf-8") as fh:
        return cast("dict[str, dict[str, object]]", yaml.safe_load(fh))


def _instruct(raw: object) -> InstructConfig:
    return parse_instruct_bytes(yaml.safe_dump(raw).encode(), "test")


def test_the_shipped_instruct_config_is_the_contracts_setting() -> None:
    cfg = parse_instruct_bytes((CONFIGS / "instruct-dolly.yaml").read_bytes(), "shipped")
    revision = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"  # pragma: allowlist secret
    assert cfg.dataset_revision == revision
    assert (cfg.train_below, cfg.validation_below, cfg.split_key) == (0.9, 0.95, "dolly-split-v1")
    assert (cfg.max_inputs, cfg.buckets, cfg.batch_tokens) == (1024, (128, 256, 512, 1024), 8192)
    assert (cfg.peak_lr, cfg.warmup_steps, cfg.final_lr_fraction, cfg.epochs) == (3e-5, 100, 0.1, 3)
    assert (cfg.eval_depth, cfg.days_micro) == (5, 8)
    assert cfg.pin_sha256.startswith("7a39ce6f")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("split", "train_below", 0.96, "train < validation"),
        ("split", "validation_below", 1.0, "train < validation"),
        ("examples", "buckets", [256, 128], "strictly increasing"),
        ("examples", "buckets", [128, 128], "strictly increasing"),
        ("examples", "buckets", [128, True], "list of integers"),
        ("examples", "buckets", [], "strictly increasing"),
        ("examples", "max_inputs", 2048, "fit the last bucket"),
        ("examples", "batch_tokens", 512, "last-bucket row"),
        ("schedule", "peak_lr", 0.0, "positive"),
        ("schedule", "epochs", 0, "at least 1"),
        ("schedule", "final_lr_fraction", 1.5, r"\[0, 1\]"),
        ("evaluation", "depth", 0, "at least 1"),
        ("forgetting", "pin_sha256", "7A39", "lowercase sha256"),
        ("dataset", "sha256", 5, "expected a string"),
    ],
)
def test_an_inconsistent_instruct_config_is_refused(
    section: str, key: str, value: object, message: str
) -> None:
    raw = _instruct_raw()
    raw[section][key] = value
    with pytest.raises(ConfigError, match=message):
        _ = _instruct(raw)


def test_the_shipped_replay_config_differs_only_in_rate_and_replay() -> None:
    plain = parse_instruct_bytes((CONFIGS / "instruct-dolly.yaml").read_bytes(), "plain")
    replay = parse_instruct_bytes((CONFIGS / "instruct-dolly-replay.yaml").read_bytes(), "replay")
    assert replay.replay is not None
    assert (replay.replay.weight, replay.replay.rows, replay.replay.length) == (1.0, 2, 1024)
    assert replay.replay.seed != replay.seed  # its own stream order
    assert replay.peak_lr == 1e-5
    assert replace(replay, peak_lr=plain.peak_lr, replay=None) == plain


def test_the_replay_section_is_optional_typed_and_checked() -> None:
    # replay of the source's training text beside each update (maintainer B 758b4fd); absent means
    # none
    assert _instruct(_instruct_raw()).replay is None
    raw = _instruct_raw()
    raw["replay"] = {"weight": 1.0, "rows": 2, "length": 1024, "seed": 7}
    replay = _instruct(raw).replay
    assert replay is not None
    assert (replay.weight, replay.rows, replay.length, replay.seed) == (1.0, 2, 1024, 7)
    raw["replay"]["weight"] = 0  # weight 0: the update without replay (no rows are read)
    assert _instruct(raw).replay == replace(replay, weight=0.0)
    for key, value, message in [
        ("weight", -0.5, "replay.weight"),
        ("weight", float("nan"), "replay.weight"),
        ("weight", float("inf"), "replay.weight"),
        ("rows", 0, "replay.rows"),
        ("length", 1, "replay.length"),
        ("seed", -1, "replay.seed"),
        ("rows", 2.0, "expected an integer"),
    ]:
        raw = _instruct_raw()
        raw["replay"] = {"weight": 1.0, "rows": 2, "length": 1024, "seed": 7, key: value}
        with pytest.raises(ConfigError, match=message):
            _ = _instruct(raw)
    # the relations hold for a typed config too, not only on the YAML path (maintainer B, a771dee)
    shipped = _instruct(_instruct_raw())
    for bad, message in [
        (ReplayConfig(-1.0, 2, 1024, 7), "replay.weight"),
        (ReplayConfig(float("nan"), 2, 1024, 7), "replay.weight"),
        (ReplayConfig(1.0, 0, 1024, 7), "replay.rows"),
        (ReplayConfig(1.0, 2, 1, 7), "replay.length"),
        (ReplayConfig(1.0, 2, 1024, -1), "replay.seed"),
    ]:
        with pytest.raises(ConfigError, match=message):
            validate_instruct(replace(shipped, replay=bad))
    validate_instruct(replace(shipped, replay=ReplayConfig(0.0, 1, 2, 0)))
    raw = _instruct_raw()
    raw["replay"] = {"weight": 1.0, "rows": 2, "length": 1024}
    with pytest.raises(ConfigError, match="missing key 'seed'"):
        _ = _instruct(raw)
    raw["replay"] = {"weight": 1.0, "rows": 2, "length": 1024, "seed": 7, "cycle": True}
    with pytest.raises(ConfigError, match="unknown keys: cycle"):
        _ = _instruct(raw)


def test_a_missing_or_unknown_instruct_key_is_refused() -> None:
    raw = _instruct_raw()
    del raw["schedule"]["seed"]
    with pytest.raises(ConfigError, match="missing key 'seed'"):
        _ = _instruct(raw)
    raw = _instruct_raw()
    raw["schedule"]["sead"] = 0
    with pytest.raises(ConfigError, match="unknown keys: sead"):
        _ = _instruct(raw)
    raw = _instruct_raw()
    raw["extra"] = {}
    with pytest.raises(ConfigError, match="unknown keys: extra"):
        _ = _instruct(raw)
