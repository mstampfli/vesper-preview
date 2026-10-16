"""Typed, fail-closed loader for configs/*.yaml: the only code that reads a config file.

Every constant the model, the optimizer and the schedule use is a field here (spec rule: no
configuration constant appears in code). The file is untrusted on-disk input: a missing key, an
unknown key, a wrong type or an inconsistent value raises ``ConfigError`` before anything runs.
``bool`` is never accepted where a number is expected (YAML's ``yes``/``no`` trap).

PRIMITIVE ``_Section``: typed extraction from parsed YAML. Each getter consumes its key, so
``done()`` can reject the keys nobody read, and each rejects ``bool`` and wrong types.
Use it for: configs/*.yaml. Not for: tensors, checkpoints or anything that is already typed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import yaml

LearningMode = Literal["e2e", "local"]
_LEARNING_MODES: frozenset[str] = frozenset({"e2e", "local"})
# the key filter's width bounds (docs/reviews/2026-10-15-retrieval-binding.md): 1 would only
# rescale the current key per channel, so the smallest filter that sees a predecessor is 2
MIN_KEY_CONV = 2
MAX_KEY_CONV = 8


class ConfigError(ValueError):
    """The config file is missing a key, has an unknown key, a wrong type or an impossible value."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Shapes and thresholds of the model (spec: Component specifications, Model sizes)."""

    d: int
    heads: int
    head_dim: int
    columns: int
    t_max: int
    vocab: int
    window: int
    kv_groups: int
    ffn_mult: int
    retrieval_k: int
    retrieval_gate_threshold: float
    halt_epsilon: float
    rotary_base: float


@dataclass(frozen=True, slots=True)
class RecurrenceConfig:
    """The delta-rule scan (spec 3a) and the noise on state reads (Hardware co-design)."""

    block: int
    state_read_noise: float
    key_conv: int  # width of the causal filter on the projected key; 0 = none (as built)


@dataclass(frozen=True, slots=True)
class LedgerConfig:
    """The episodic store (spec component 5)."""

    capacity_per_column: int
    pq_bits: int


@dataclass(frozen=True, slots=True)
class ModulatorConfig:
    """The global learning signal (spec component 7)."""

    half_life_tokens: int


@dataclass(frozen=True, slots=True)
class LearningConfig:
    """Loss weights and the mode switch (spec: Learning)."""

    mode: LearningMode
    lambda_pc: float
    lambda_h: float
    lambda_g: float
    alpha: float
    m_clip_low: float
    m_clip_high: float
    target_open_rate: float


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """AdamW and its schedule (spec: Optimizer and schedule)."""

    peak_lr: float
    warmup_steps: int
    beta1: float
    beta2: float
    weight_decay: float
    grad_clip: float
    final_lr_fraction: float


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Batch shape, token budget and the phase boundaries (spec: Training recipe)."""

    chunk_tokens: int
    batch_chunks: int
    total_tokens: int
    phase_warm_end: float
    phase_loop_end: float
    phase_ledger_end: float
    t_max_anneal_from: int
    qat_from: float
    held_out_fraction: float
    checkpoints_per_run: int
    eval_chunks: int


@dataclass(frozen=True, slots=True)
class DuskConfig:
    """Consolidation cadence and strength (spec component 6)."""

    cadence_tokens: int
    budget_fraction: float
    eta_slow_fraction: float
    gamma: float


@dataclass(frozen=True, slots=True)
class VesperConfig:
    """One config file, fully typed and validated."""

    model: ModelConfig
    recurrence: RecurrenceConfig
    ledger: LedgerConfig
    modulator: ModulatorConfig
    learning: LearningConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    dusk: DuskConfig


def _mapping(raw: object, where: str) -> dict[str, object]:
    """Narrow a parsed YAML node to a mapping with string keys, or fail."""
    if not isinstance(raw, dict):
        msg = f"{where}: expected a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    out: dict[str, object] = {}
    for key, value in cast("dict[object, object]", raw).items():
        if not isinstance(key, str):
            msg = f"{where}: key {key!r} is not a string"
            raise ConfigError(msg)
        out[key] = value
    return out


class _Section:
    """One mapping of the config; getters consume keys so leftovers are detectable."""

    def __init__(self, raw: object, where: str) -> None:
        self._where: str = where
        self._items: dict[str, object] = _mapping(raw, where)

    def _take(self, key: str) -> object:
        if key not in self._items:
            msg = f"{self._where}: missing key '{key}'"
            raise ConfigError(msg)
        return self._items.pop(key)

    def get_int(self, key: str) -> int:
        """Read an integer (not a bool, not a float)."""
        value = self._take(key)
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{self._where}.{key}: expected an integer, got {value!r}"
            raise ConfigError(msg)
        return value

    def get_float(self, key: str) -> float:
        """Read a float; an integer literal is accepted, a bool is not."""
        value = self._take(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = f"{self._where}.{key}: expected a number, got {value!r}"
            raise ConfigError(msg)
        return float(value)

    def get_str(self, key: str) -> str:
        """Read a string."""
        value = self._take(key)
        if not isinstance(value, str):
            msg = f"{self._where}.{key}: expected a string, got {value!r}"
            raise ConfigError(msg)
        return value

    def get_section(self, key: str) -> "_Section":
        """Descend into a nested mapping."""
        return _Section(self._take(key), f"{self._where}.{key}")

    def done(self) -> None:
        """Fail if any key was never read: a typo must not pass silently."""
        if self._items:
            unknown = ", ".join(sorted(self._items))
            msg = f"{self._where}: unknown keys: {unknown}"
            raise ConfigError(msg)


def _model(s: _Section) -> ModelConfig:
    cfg = ModelConfig(
        d=s.get_int("d"),
        heads=s.get_int("heads"),
        head_dim=s.get_int("head_dim"),
        columns=s.get_int("columns"),
        t_max=s.get_int("t_max"),
        vocab=s.get_int("vocab"),
        window=s.get_int("window"),
        kv_groups=s.get_int("kv_groups"),
        ffn_mult=s.get_int("ffn_mult"),
        retrieval_k=s.get_int("retrieval_k"),
        retrieval_gate_threshold=s.get_float("retrieval_gate_threshold"),
        halt_epsilon=s.get_float("halt_epsilon"),
        rotary_base=s.get_float("rotary_base"),
    )
    s.done()
    return cfg


def _recurrence(s: _Section) -> RecurrenceConfig:
    cfg = RecurrenceConfig(
        block=s.get_int("block"),
        state_read_noise=s.get_float("state_read_noise"),
        key_conv=s.get_int("key_conv"),
    )
    s.done()
    return cfg


def _ledger(s: _Section) -> LedgerConfig:
    cfg = LedgerConfig(
        capacity_per_column=s.get_int("capacity_per_column"), pq_bits=s.get_int("pq_bits")
    )
    s.done()
    return cfg


def _modulator(s: _Section) -> ModulatorConfig:
    cfg = ModulatorConfig(half_life_tokens=s.get_int("half_life_tokens"))
    s.done()
    return cfg


def _learning(s: _Section) -> LearningConfig:
    mode = s.get_str("mode")
    if mode not in _LEARNING_MODES:
        msg = f"learning.mode: expected one of {sorted(_LEARNING_MODES)}, got {mode!r}"
        raise ConfigError(msg)
    cfg = LearningConfig(
        mode=cast("LearningMode", mode),
        lambda_pc=s.get_float("lambda_pc"),
        lambda_h=s.get_float("lambda_h"),
        lambda_g=s.get_float("lambda_g"),
        alpha=s.get_float("alpha"),
        m_clip_low=s.get_float("m_clip_low"),
        m_clip_high=s.get_float("m_clip_high"),
        target_open_rate=s.get_float("target_open_rate"),
    )
    s.done()
    return cfg


def _optimizer(s: _Section) -> OptimizerConfig:
    cfg = OptimizerConfig(
        peak_lr=s.get_float("peak_lr"),
        warmup_steps=s.get_int("warmup_steps"),
        beta1=s.get_float("beta1"),
        beta2=s.get_float("beta2"),
        weight_decay=s.get_float("weight_decay"),
        grad_clip=s.get_float("grad_clip"),
        final_lr_fraction=s.get_float("final_lr_fraction"),
    )
    s.done()
    return cfg


def _training(s: _Section) -> TrainingConfig:
    cfg = TrainingConfig(
        chunk_tokens=s.get_int("chunk_tokens"),
        batch_chunks=s.get_int("batch_chunks"),
        total_tokens=s.get_int("total_tokens"),
        phase_warm_end=s.get_float("phase_warm_end"),
        phase_loop_end=s.get_float("phase_loop_end"),
        phase_ledger_end=s.get_float("phase_ledger_end"),
        t_max_anneal_from=s.get_int("t_max_anneal_from"),
        qat_from=s.get_float("qat_from"),
        held_out_fraction=s.get_float("held_out_fraction"),
        checkpoints_per_run=s.get_int("checkpoints_per_run"),
        eval_chunks=s.get_int("eval_chunks"),
    )
    s.done()
    return cfg


def _dusk(s: _Section) -> DuskConfig:
    cfg = DuskConfig(
        cadence_tokens=s.get_int("cadence_tokens"),
        budget_fraction=s.get_float("budget_fraction"),
        eta_slow_fraction=s.get_float("eta_slow_fraction"),
        gamma=s.get_float("gamma"),
    )
    s.done()
    return cfg


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise ConfigError(message)


def validate(cfg: VesperConfig) -> None:
    """Reject value combinations the architecture cannot run with.

    Args:
        cfg: a fully parsed config.

    Raises:
        ConfigError: naming the first violated relation.
    """
    m, t, d = cfg.model, cfg.training, cfg.dusk
    _require(m.heads * m.head_dim == m.d, "model: heads * head_dim must equal d")
    _require(m.heads % m.kv_groups == 0, "model: heads must be divisible by kv_groups")
    _require(m.head_dim % 2 == 0, "model: head_dim must be even (int4 state packs two per byte)")
    _require(m.columns >= 1 and m.t_max >= 1, "model: columns and t_max must be at least 1")
    _require(
        m.window >= 1 and m.retrieval_k >= 1, "model: window and retrieval_k must be at least 1"
    )
    _require(0.0 < m.halt_epsilon < 1.0, "model: halt_epsilon must be in (0, 1)")
    _require(0.0 < m.retrieval_gate_threshold < 1.0, "model: retrieval_gate_threshold in (0, 1)")
    _require(1.0 < m.rotary_base < float("inf"), "model: rotary_base must be finite and above 1")
    _require(cfg.recurrence.block >= 1, "recurrence: block must be at least 1")
    _require(cfg.recurrence.state_read_noise >= 0.0, "recurrence: state_read_noise must be >= 0")
    _require(
        cfg.recurrence.key_conv == 0 or MIN_KEY_CONV <= cfg.recurrence.key_conv <= MAX_KEY_CONV,
        f"recurrence: key_conv must be 0 (no filter) or {MIN_KEY_CONV}..{MAX_KEY_CONV}",
    )
    _require(cfg.modulator.half_life_tokens >= 1, "modulator: half_life_tokens must be at least 1")
    _require(
        cfg.learning.m_clip_low < cfg.learning.m_clip_high, "learning: m_clip_low < m_clip_high"
    )
    _require(0.0 <= cfg.learning.alpha < float("inf"), "learning: alpha must be finite and >= 0")
    _require(
        1.0 + cfg.learning.alpha * cfg.learning.m_clip_low >= 0.0,
        "learning: 1 + alpha * m_clip_low must be >= 0 (no negative write-odds multiplier)",
    )
    _require(0.0 < cfg.learning.target_open_rate <= 1.0, "learning: target_open_rate in (0, 1]")
    _require(cfg.optimizer.peak_lr > 0.0, "optimizer: peak_lr must be positive")
    _require(0.0 < cfg.optimizer.final_lr_fraction <= 1.0, "optimizer: final_lr_fraction in (0, 1]")
    _require(
        t.chunk_tokens % cfg.recurrence.block == 0,
        "training: chunk_tokens must be a multiple of block",
    )
    _require(
        0.0 < t.phase_warm_end < t.phase_loop_end < t.phase_ledger_end < 1.0,
        "training: phase ends must satisfy 0 < warm < loop < ledger < 1",
    )
    _require(1 <= t.t_max_anneal_from <= m.t_max, "training: t_max_anneal_from in [1, t_max]")
    _require(0.0 < t.qat_from < 1.0, "training: qat_from in (0, 1)")
    _require(0.0 < t.held_out_fraction < 1.0, "training: held_out_fraction in (0, 1)")
    _require(t.checkpoints_per_run >= 1, "training: checkpoints_per_run must be at least 1")
    _require(t.eval_chunks >= 1, "training: eval_chunks must be at least 1")
    _require(0.0 < d.gamma <= 1.0, "dusk: gamma in (0, 1]")
    _require(0.0 <= d.budget_fraction <= 1.0, "dusk: budget_fraction in [0, 1]")
    _require(0.0 <= d.eta_slow_fraction <= 1.0, "dusk: eta_slow_fraction in [0, 1]")
    _require(d.cadence_tokens >= t.chunk_tokens * t.batch_chunks, "dusk: cadence must cover a step")


def parse(raw: object, where: str) -> VesperConfig:
    """Build a validated config from an already-parsed YAML document.

    Args:
        raw: the parsed document (a mapping).
        where: a label for error messages, usually the file path.

    Returns:
        The validated config.

    Raises:
        ConfigError: on any missing, unknown, mistyped or inconsistent value.
    """
    root = _Section(raw, where)
    cfg = VesperConfig(
        model=_model(root.get_section("model")),
        recurrence=_recurrence(root.get_section("recurrence")),
        ledger=_ledger(root.get_section("ledger")),
        modulator=_modulator(root.get_section("modulator")),
        learning=_learning(root.get_section("learning")),
        optimizer=_optimizer(root.get_section("optimizer")),
        training=_training(root.get_section("training")),
        dusk=_dusk(root.get_section("dusk")),
    )
    root.done()
    validate(cfg)
    return cfg


def load(path: Path) -> VesperConfig:
    """Read and validate one configs/*.yaml file.

    Args:
        path: the YAML file.

    Returns:
        The validated config.

    Raises:
        ConfigError: see ``parse``.
    """
    return parse_bytes(path.read_bytes(), str(path))


def parse_bytes(data: bytes, where: str) -> VesperConfig:
    """Parse and validate a config file's bytes, already read.

    A caller that hashes those bytes (a run's recipe snapshot) validates exactly what it hashed.

    Args:
        data: the file's bytes, UTF-8 YAML.
        where: a label for error messages, usually the file path.

    Returns:
        The validated config.

    Raises:
        ConfigError: see ``parse``.
        UnicodeDecodeError: if ``data`` is not UTF-8.
    """
    return parse(cast("object", yaml.safe_load(data.decode("utf-8"))), where)
