"""Typed, fail-closed loader for configs/*.yaml: the only code that reads a config file.

Every constant the model, the optimizer and the schedule use is a field here (spec rule: no
configuration constant appears in code). The file is untrusted on-disk input: a missing key, an
unknown key, a wrong type or an inconsistent value raises ``ConfigError`` before anything runs.
``bool`` is never accepted where a number is expected (YAML's ``yes``/``no`` trap).

PRIMITIVE ``_Section``: typed extraction from parsed YAML. Each getter consumes its key, so
``done()`` can reject the keys nobody read, and each rejects ``bool`` and wrong types.
Use it for: configs/*.yaml. Not for: tensors, checkpoints or anything that is already typed.
"""

import math
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
    # the SHORTEST memory lifetime in chunks: slots restart their recurrent memory (h, key
    # history) every this, 4x, 16x, 64x, 256x many of their chunks or never, by slot (mixed
    # lifetimes, train.reset_keep); 0 = never for every slot, the state carried forever (absent in
    # a config = 0: the 16x d512 run of 2026-10-21 showed cold-start decay without restarts)
    state_reset_chunks: int = 0


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


@dataclass(frozen=True, slots=True)
class InstructConfig:
    """The instruction-tuning test's one setting (docs/reviews/2026-10-25-instruction-tuning.md).

    Its own file (configs/instruct-*.yaml), beside the source run's VesperConfig, which keeps the
    model and the optimizer's betas, decay and clip. Sections: ``dataset`` (the pinned source
    file), ``split`` (the keyed group split's key and cumulative bounds), ``examples`` (the input
    limit, the length buckets, positions per batch), ``schedule`` (rate, warm-up, final fraction,
    epochs, order seed), ``evaluation`` (fixed depth, group resamples, generated ids at most) and
    ``forgetting`` (the days run's held-out view: its width, the accepted test pin's hash and the
    view's own digest, both checked before any model reads it), and the optional ``replay``
    (``ReplayConfig``; absent: no replay).
    """

    dataset_name: str
    dataset_revision: str
    dataset_sha256: str
    split_key: str
    train_below: float
    validation_below: float
    max_inputs: int
    buckets: tuple[int, ...]
    batch_tokens: int
    peak_lr: float
    warmup_steps: int
    final_lr_fraction: float
    epochs: int
    seed: int
    eval_depth: int
    resamples: int
    max_new_tokens: int
    days_micro: int
    pin_sha256: str
    days_view_sha256: str
    replay: "ReplayConfig | None" = None


@dataclass(frozen=True)
class ReplayConfig:
    """General text from the source run's TRAINING side beside every tuning update.

    Maintainer B's design, 758b4fd.

    Each update adds ``weight`` times the mean loss of ``rows`` rows of ``length`` positions from
    a stream of that side, seeded by ``seed`` (independent of the example order), to the
    instruction loss; weight 0 is the update without replay, and no row is read.
    """

    weight: float
    rows: int
    length: int
    seed: int


def _replay(s: "_Section") -> ReplayConfig:
    cfg = ReplayConfig(
        weight=s.get_float("weight"),
        rows=s.get_int("rows"),
        length=s.get_int("length"),
        seed=s.get_int("seed"),
    )
    s.done()
    return cfg  # its relations are validate_instruct's, for a typed config as for a parsed one


def _instruct(root: "_Section") -> InstructConfig:
    dataset, split = root.get_section("dataset"), root.get_section("split")
    examples, schedule = root.get_section("examples"), root.get_section("schedule")
    evaluation, forgetting = root.get_section("evaluation"), root.get_section("forgetting")
    cfg = InstructConfig(
        dataset_name=dataset.get_str("name"),
        dataset_revision=dataset.get_str("revision"),
        dataset_sha256=dataset.get_str("sha256"),
        split_key=split.get_str("key"),
        train_below=split.get_float("train_below"),
        validation_below=split.get_float("validation_below"),
        max_inputs=examples.get_int("max_inputs"),
        buckets=examples.get_int_list("buckets"),
        batch_tokens=examples.get_int("batch_tokens"),
        peak_lr=schedule.get_float("peak_lr"),
        warmup_steps=schedule.get_int("warmup_steps"),
        final_lr_fraction=schedule.get_float("final_lr_fraction"),
        epochs=schedule.get_int("epochs"),
        seed=schedule.get_int("seed"),
        eval_depth=evaluation.get_int("depth"),
        resamples=evaluation.get_int("resamples"),
        max_new_tokens=evaluation.get_int("max_new_tokens"),
        days_micro=forgetting.get_int("micro"),
        pin_sha256=forgetting.get_str("pin_sha256"),
        days_view_sha256=forgetting.get_str("view_sha256"),
        replay=_replay(root.get_section("replay")) if root.has("replay") else None,
    )
    for section in (dataset, split, examples, schedule, evaluation, forgetting, root):
        section.done()
    return cfg


def validate_instruct(cfg: InstructConfig) -> None:
    """The relations an InstructConfig must hold; ``ConfigError`` names the first one broken."""
    _require(
        0.0 < cfg.train_below < cfg.validation_below < 1.0, "split: 0 < train < validation < 1"
    )
    _require(
        len(cfg.buckets) > 0
        and all(b > 0 for b in cfg.buckets)
        and list(cfg.buckets) == sorted(set(cfg.buckets)),
        "examples.buckets: positive and strictly increasing",
    )
    last = max(cfg.buckets, default=0)
    _require(cfg.max_inputs <= last, "examples: max_inputs must fit the last bucket")
    _require(cfg.batch_tokens >= last, "examples: batch_tokens holds a last-bucket row")
    _require(math.isfinite(cfg.peak_lr) and cfg.peak_lr > 0, "schedule.peak_lr: positive")
    _require(cfg.warmup_steps >= 1 and cfg.epochs >= 1, "schedule: warm-up and epochs at least 1")
    _require(0.0 <= cfg.final_lr_fraction <= 1.0, "schedule.final_lr_fraction: in [0, 1]")
    _require(cfg.eval_depth >= 1, "evaluation.depth: at least 1")
    _require(cfg.resamples >= 1 and cfg.max_new_tokens >= 1, "evaluation: counts at least 1")
    _require(cfg.days_micro >= 1, "forgetting.micro: at least 1")
    if (r := cfg.replay) is not None:
        _require(math.isfinite(r.weight) and r.weight >= 0, "replay.weight: finite, at least 0")
        _require(r.rows >= 1, "replay.rows: at least 1")
        _require(r.length >= 2, "replay.length: at least 2")  # noqa: PLR2004  # input and target
        _require(r.seed >= 0, "replay.seed: at least 0")
    for name, digest in (
        ("dataset.sha256", cfg.dataset_sha256),
        ("forgetting.pin_sha256", cfg.pin_sha256),
        ("forgetting.view_sha256", cfg.days_view_sha256),
    ):
        _require(
            len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),  # noqa: PLR2004
            f"{name}: a lowercase sha256",
        )


def parse_instruct_bytes(data: bytes, where: str) -> InstructConfig:
    """Parse and validate an instruction-tuning config's bytes, already read (and hashed).

    Raises:
        ConfigError: on any missing, unknown, mistyped or inconsistent value.
        UnicodeDecodeError: if ``data`` is not UTF-8.
    """
    cfg = _instruct(_Section(cast("object", yaml.safe_load(data.decode("utf-8"))), where))
    validate_instruct(cfg)
    return cfg


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

    def get_int_or(self, key: str, default: int) -> int:
        """Read an optional integer: ``default`` when the key is absent, else as ``get_int``.

        For a recipe knob added after configs and checkpoints exist whose absence has one
        meaning (the old behaviour); never for a value with no safe default.
        """
        return default if key not in self._items else self.get_int(key)

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

    def get_int_list(self, key: str) -> tuple[int, ...]:
        """Read a list of integers (no bools, no floats)."""
        value = self._take(key)
        if not isinstance(value, list):
            msg = f"{self._where}.{key}: expected a list of integers, got {value!r}"
            raise ConfigError(msg)
        items = cast("list[object]", value)
        ints = [v for v in items if isinstance(v, int) and not isinstance(v, bool)]
        if len(ints) != len(items):
            msg = f"{self._where}.{key}: expected a list of integers, got {value!r}"
            raise ConfigError(msg)
        return tuple(ints)

    def get_section(self, key: str) -> "_Section":
        """Descend into a nested mapping."""
        return _Section(self._take(key), f"{self._where}.{key}")

    def has(self, key: str) -> bool:
        """Whether ``key`` is still unread here: for an optional section whose absence means off."""
        return key in self._items

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
        state_reset_chunks=s.get_int_or("state_reset_chunks", 0),
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
    _require(t.state_reset_chunks >= 0, "training: state_reset_chunks must be 0 (never) or more")
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
