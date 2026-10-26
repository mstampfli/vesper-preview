"""One column (spec 3): the compute block the loop applies ``C`` times per pass.

Sublayers in order, each pre-RMSNorm with a residual: the recurrent core (3a), windowed attention
(3b), the feed-forward (3d). The sparse retrieval sublayer (3c) is not built yet: its slot in the
order is between attention and the FFN, its registry rows stay specified, and this module says so
rather than pretending. Per-pass state: pass ``t`` reads and writes ``H^(t)`` and KV window ``t``
only (spec: state ordering), which is what lets the pass-major training order and the token-major
inference order see the same updates (docs/reviews/2026-10-14-pass-state-review.md, A and B); the
tests run both orders and also show that one shared state breaks the equality. The iteration
conditioning ``x + u_t`` and halting belong to the loop: the column takes the conditioned input
and returns its output, the new state and what the chunk wrote into that state (``Projections``),
which the loop carries forward for tokens that have halted (spec 4: their last ``(k, v, beta)``
and ``(K, V)`` enter every later pass, so the per-pass states stay dense).
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, override

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vesper.attention import (
    KVCache,
    RingKVCache,
    WindowedAttention,
    kv_cache_empty,
    kv_ring_empty,
)
from vesper.config import VesperConfig
from vesper.contracts import ContractError, Dims, check_shape
from vesper.ffn import FFN
from vesper.kernels.delta_scan import (
    FP32_INPUT_DTYPE,
    INPUT_DTYPE,
    STATE_DTYPE,
    delta_scan_fused,
    delta_scan_fused_fp32,
    fused_scan_available,
)
from vesper.kernels.keys import fused_keys
from vesper.kernels.norm import fused_rms_norm
from vesper.recurrence import delta_scan_chunked, normalize_keys, write_strength

ScanImpl = Literal["reference", "fused", "fused-fp32"]
SCAN_IMPLS: tuple[ScanImpl, ...] = ("reference", "fused", "fused-fp32")
# what each fused scan takes its q, k, v, beta in; the reference takes promote_types(dtype, float32)
FUSED_INPUT_DTYPES: dict[ScanImpl, torch.dtype] = {
    "fused": INPUT_DTYPE,
    "fused-fp32": FP32_INPUT_DTYPE,
}

_RMS_EPS = 1e-6  # the regularizer inside the root of RMSNorm, not a tunable and not a floor
_RMS_SCALE_FLOOR = 1e-3  # sqrt(_RMS_EPS): keeps eps / scale^2 finite for rows near zero
_FUSED_NORM_DTYPES = (torch.float32, torch.bfloat16, torch.float16)  # float64 keeps the reference


@dataclass(frozen=True)
class ColumnShape:
    """The geometry of one column: the `model` and `recurrence` fields the column reads."""

    d: int
    heads: int
    kv_groups: int
    head_dim: int
    window: int
    rotary_base: float
    ffn_mult: int
    block: int
    key_conv: int  # width of the causal filter on the projected recurrent key; 0 = none

    @property
    def key_history(self) -> int:
        """Projected key inputs a pass carries across a chunk boundary: ``key_conv - 1``."""
        return max(self.key_conv - 1, 0)

    @property
    def kv_heads(self) -> int:
        """Key-value heads under GQA."""
        return self.heads // self.kv_groups


def column_shape(cfg: VesperConfig) -> ColumnShape:
    """The column's geometry from a validated configuration (one fact, one place)."""
    m = cfg.model
    return ColumnShape(
        d=m.d,
        heads=m.heads,
        kv_groups=m.kv_groups,
        head_dim=m.head_dim,
        window=m.window,
        rotary_base=m.rotary_base,
        ffn_mult=m.ffn_mult,
        block=cfg.recurrence.block,
        key_conv=cfg.recurrence.key_conv,
    )


@dataclass(frozen=True)
class Modulation:
    """The modulator's signal and its strength, which travel together (``write_strength``).

    ``m`` is ``[b, n]``, one surprise value per token, clipped by the modulator; ``alpha`` is
    ``learning.alpha``. One object, so a signal without its strength cannot be passed.
    """

    m: Tensor
    alpha: float


@dataclass(frozen=True)
class Projections:
    """What a chunk writes into a pass's states, per position; carried by the loop once halted.

    ``rec_k, rec_v`` are ``[b, n, h, dk]`` (keys normalized, after the key filter), ``rec_beta`` is
    ``[b, n, h]``, ``attn_k, attn_v`` are ``[b, n, hkv, dk]`` (keys rotated). ``rec_z`` is
    ``[b, n, d]``, the projected key input BEFORE the filter: what a later token's filter reads
    from this position (a halted token's frozen exit-pass value; docs/reviews/
    2026-10-15-retrieval-binding.md, 4.1), distinct from the token's own committed ``rec_k``.
    """

    rec_z: Tensor
    rec_k: Tensor
    rec_v: Tensor
    rec_beta: Tensor
    attn_k: Tensor
    attn_v: Tensor


@dataclass(frozen=True)
class Carry:
    """Which tokens are still active (``bool[b, n]``) and what the halted ones write instead."""

    active: Tensor
    projections: Projections


def _carry(active: Tensor, fresh: Tensor, carried: Tensor) -> Tensor:
    """Per position: the fresh projection where the token is active, the carried one elsewhere."""
    mask = active.reshape(*active.shape, *([1] * (fresh.dim() - active.dim())))
    return torch.where(mask, fresh, carried)


class RMSNorm(nn.Module):
    """PRIMITIVE ``RMSNorm``: the one pre-normalization of every sublayer, ``x / rms(x) * g``.

    Use it for: the input of each sublayer, with its own gain ``g`` (initialized to one).
    Not for: anything that needs a mean subtraction or a bias; the model has neither.
    """

    def __init__(self, d: int) -> None:
        """One gain vector of width ``d``."""
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(d))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """``[..., d] -> [..., d]``: ``x / sqrt(mean(x^2) + 1e-6) * g``, computed scale-safely.

        Each row is divided by its largest magnitude (floored at ``1e-3``, treated as a constant)
        before squaring, so rows near the dtype's range do not overflow to zero and rows near zero
        stay finite; in exact arithmetic the result is the formula above, and so is its gradient.
        The statistics and the gain are applied in float32 or wider whatever the input dtype,
        because in half precision the reciprocal of the floored scale squared overflows; the
        result returns in the input dtype, also when the gain is float32 under half-precision
        activations. The ``1e-6`` is a regularizer inside the root, not a floor of the RMS: a
        constant row of ``1e-4`` gives ``0.0995``, not one.

        Raises:
            ContractError: if the last axis is not the gain's width (no silent broadcast).
        """
        d = self.weight.shape[0]
        if x.shape[-1] != d:
            msg = f"RMSNorm: last axis must be {d}, got {x.shape[-1]} (shape {tuple(x.shape)})"
            raise ContractError(msg)
        if x.is_cuda and x.dtype in _FUSED_NORM_DTYPES:
            # the same arithmetic as one kernel per direction (kernels.norm); below, the reference
            return fused_rms_norm(x, self.weight, _RMS_SCALE_FLOOR, _RMS_EPS)
        compute = torch.promote_types(x.dtype, torch.float32)
        wide = x.to(compute)
        scale = wide.detach().abs().amax(dim=-1, keepdim=True).clamp_min(_RMS_SCALE_FLOOR)
        scaled = wide / scale
        rms = torch.sqrt(scaled.square().mean(dim=-1, keepdim=True) + _RMS_EPS / (scale * scale))
        # the gain is applied in the accumulator too, so the result returns in x's dtype even
        # when the gain is float32 under half-precision activations
        return (scaled / rms * self.weight.to(compute)).to(x.dtype)


class RecurrentSublayer(nn.Module):
    """Projections around the delta-rule scan: ``q, k, v``, the write strength and the output.

    ``4 d^2 + h d`` parameters: ``w_q, w_k, w_v, w_o`` of ``[d, d]`` and ``w_beta`` of ``[h, d]``.
    Keys are normalized and the write strength bounded here, so the scan never sees a key above
    unit norm or a ``beta`` outside ``[0, 1]``. Call ``forward`` directly.
    """

    def __init__(self, shape: ColumnShape, generator: torch.Generator | None = None) -> None:
        """Build the sublayer; ``generator`` makes the initialization reproducible.

        Raises:
            ValueError: if ``heads * head_dim != d`` or ``block < 1``.
        """
        super().__init__()
        d, heads, head_dim = shape.d, shape.heads, shape.head_dim
        if heads * head_dim != d:
            msg = f"heads ({heads}) * head_dim ({head_dim}) must equal d ({d})"
            raise ValueError(msg)
        if shape.block < 1:
            msg = f"block must be at least 1, got {shape.block}"
            raise ValueError(msg)
        self.heads: int = heads
        self.head_dim: int = head_dim
        self.block: int = shape.block
        # which implementation runs the scan: the reference or the fused CUDA kernel (set_scan);
        # module state like train/eval, never a parameter, never in a checkpoint
        self.scan_impl: ScanImpl = "reference"
        std = 1.0 / math.sqrt(d)
        self.w_q: nn.Parameter = nn.Parameter(torch.randn(d, d, generator=generator) * std)
        self.w_k: nn.Parameter = nn.Parameter(torch.randn(d, d, generator=generator) * std)
        self.w_v: nn.Parameter = nn.Parameter(torch.randn(d, d, generator=generator) * std)
        self.w_o: nn.Parameter = nn.Parameter(torch.randn(d, d, generator=generator) * std)
        self.w_beta: nn.Parameter = nn.Parameter(torch.randn(heads, d, generator=generator) * std)
        # the causal key filter (the project owner's decision of 2026-10-15): taps[r] weighs the
        # projected key input r tokens back, r = 0 the current one; uniform(-1/sqrt(K), 1/sqrt(K)),
        # drawn after every other parameter so key_conv 0 consumes no generator draws (the as-built
        # model)
        self.key_conv: int = shape.key_conv
        self.conv_k: nn.Parameter | None = None
        if shape.key_conv:
            bound = 1.0 / math.sqrt(shape.key_conv)
            taps = torch.rand(shape.key_conv, d, generator=generator) * (2 * bound) - bound
            self.conv_k = nn.Parameter(taps)

    def project(
        self, x: Tensor, modulation: Modulation | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``[b, n, d]`` to ``q``, the projected key input ``z``, ``v`` and ``beta``.

        ``q, v`` are ``[b,n,h,dk]``, ``beta [b,n,h]``; ``z = W_k x`` is ``[b, n, d]``, neither
        filtered nor normalized (``keys`` makes the key from it).
        """
        dims: Dims = {}
        check_shape("x", x, ("b", "n", "d"), dims)
        b, n = dims["b"], dims["n"]
        q = F.linear(x, self.w_q).view(b, n, self.heads, self.head_dim)
        z = F.linear(x, self.w_k)
        v = F.linear(x, self.w_v).view(b, n, self.heads, self.head_dim)
        logits = F.linear(x, self.w_beta)
        if modulation is None:
            return q, z, v, write_strength(logits)
        return q, z, v, write_strength(logits, modulation.m, modulation.alpha)

    def keys(self, z: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
        """Normalized keys ``[b, n, h, dk]`` from key inputs ``z [b, n, d]``, and the next history.

        PRIMITIVE causal key filter (docs/reviews/2026-10-15-retrieval-binding.md, 4.1):
        ``u[t, j] = sum_{r=0}^{K-1} conv_k[r, j] * z[t - r, j]``, depthwise, no bias, no
        activation, ``r = 0`` the current token; positions before the chunk come from ``history
        [b, K-1, d]`` (oldest first); the key is ``normalize_keys`` of ``u`` per head, AFTER the
        filter. The filter runs in float32 or wider, the history is kept in that dtype, and the
        key returns in ``z``'s dtype. The next history is the newest ``K-1`` inputs of
        (history, z): unchanged for an empty chunk, partly old for a chunk shorter than ``K-1``,
        always an owned copy. With ``key_conv`` 0 the key is ``normalize_keys(z)`` exactly, as
        before the filter existed, and the empty history passes through.

        For: the recurrent key of every token, active or (via ``z``) a halted predecessor's
        frozen input. Not for: a halted token's OWN key, which is its frozen ``rec_k``.
        """
        b, n, _ = z.shape
        if self.conv_k is None or n == 0:
            # no filter, or no token: the history object itself, as the state's other fields do
            return normalize_keys(z.view(b, n, self.heads, self.head_dim)), history
        width = self.key_conv
        compute = torch.promote_types(z.dtype, torch.float32)
        if z.is_cuda and z.dtype in _FUSED_NORM_DTYPES and self.head_dim & (self.head_dim - 1) == 0:
            # the same filter and normalization as one kernel per direction (kernels.keys);
            # the next history is the newest K - 1 inputs, an owned float32 copy as below
            k = fused_keys(z, history, self.conv_k, self.heads)
            tail = z[:, max(0, n - (width - 1)) :].to(compute)
            return k, torch.cat([history.to(compute)[:, n:], tail], dim=1)
        full = torch.cat([history.to(compute), z.to(compute)], dim=1)  # [b, K - 1 + n, d]
        taps = self.conv_k.to(compute)
        u = taps[0] * full[:, width - 1 : width - 1 + n]
        for r in range(1, width):
            u = u + taps[r] * full[:, width - 1 - r : width - 1 - r + n]
        k = normalize_keys(u.view(b, n, self.heads, self.head_dim)).to(z.dtype)
        return k, full[:, n:].clone()

    def scan(
        self, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, state: Tensor
    ) -> tuple[Tensor, Tensor]:
        """The blocked scan over the projections and the output projection: ``(y, state')``.

        Precision policy for the bf16 training path: projections arrive in the activation dtype,
        the scan runs in float32 (or wider, never narrower) with autocast disabled, the state is
        kept in that dtype across passes, and the reads return in the activation dtype for the
        output projection. The blocked form inverts a small triangular matrix, which torch does
        not do below float32 and which would lose the WY cancellation in half precision anyway.

        With ``scan_impl`` "fused" the scan is ``delta_scan_fused`` instead: ``q, k, v, beta``
        enter in bfloat16 (on the bf16 training path the projections already are), the state
        stays float32 across passes as above, and the kernel's bfloat16 intermediates are the
        precision it brings (vesper/kernels/delta_scan.py, measured there). With "fused-fp32"
        it is ``delta_scan_fused_fp32``: everything in float32, the reference's precision.
        """
        if self.scan_impl != "reference":
            kernel = delta_scan_fused if self.scan_impl == "fused" else delta_scan_fused_fp32
            dtype = FUSED_INPUT_DTYPES[self.scan_impl]
            with torch.autocast(device_type=q.device.type, enabled=False):
                o, new_state = kernel(
                    q.to(dtype), k.to(dtype), v.to(dtype), beta.to(dtype), state.to(STATE_DTYPE)
                )
            return F.linear(o.to(q.dtype).flatten(2), self.w_o), new_state
        compute = torch.promote_types(q.dtype, torch.float32)
        with torch.autocast(device_type=q.device.type, enabled=False):
            o, new_state = delta_scan_chunked(
                q.to(compute),
                k.to(compute),
                v.to(compute),
                beta.to(compute),
                state.to(compute),
                self.block,
            )
        return F.linear(o.to(q.dtype).flatten(2), self.w_o), new_state

    @override
    def forward(
        self,
        x: Tensor,
        state: Tensor,
        history: Tensor,
        modulation: Modulation | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Scan the chunk through this pass's state and project the reads back to ``d``.

        Args:
            x: ``[b, n, d]`` the normalized sublayer input.
            state: ``[b, h, dk, dk]`` this pass's state before the chunk.
            history: ``[b, K-1, d]`` this pass's key inputs before the chunk (``keys``).
            modulation: the modulator's signal and strength at inference, or None.

        Returns:
            ``(y [b, n, d], state', history')``; ``state`` and ``history`` are not modified.
        """
        q, z, v, beta = self.project(x, modulation)
        k, history_next = self.keys(z, history)
        y, state_next = self.scan(q, k, v, beta, state)
        return y, state_next, history_next


def set_scan(model: nn.Module, impl: str) -> int:
    """Choose the scan implementation of every ``RecurrentSublayer`` in ``model``.

    One switch for the whole model, so no run mixes two. "fused" and "fused-fp32" need a CUDA
    device and fla-core, and every parameter of ``model`` on CUDA: they are refused here, before
    any step, rather than failing at the first scan.

    Returns:
        The number of sublayers set.

    Raises:
        ValueError: if ``impl`` is not one of ``SCAN_IMPLS``, if ``model`` has no recurrent
            sublayer, or if a fused scan is asked where it cannot run.
    """
    if impl not in SCAN_IMPLS:
        msg = f"scan must be one of {SCAN_IMPLS}, got {impl!r}"
        raise ValueError(msg)
    sublayers = [m for m in model.modules() if isinstance(m, RecurrentSublayer)]
    if not sublayers:
        msg = "the model has no recurrent sublayer to set"
        raise ValueError(msg)
    chosen: ScanImpl = "reference"
    for device in sorted({p.device for p in model.parameters()}, key=str):
        chosen = check_scan(impl, device)
    for sublayer in sublayers:
        sublayer.scan_impl = chosen
    return len(sublayers)


def check_scan(impl: str, device: torch.device) -> ScanImpl:
    """The one check of a scan choice against the device it would run on; returns it typed.

    ``set_scan`` checks every parameter's device with it; a driver calls it before writing
    anything, with the device it is about to use. Whether the choice can run CAPTURED is a
    driver's question (``check_capturable_scan``): the reference scan on CUDA stays valid here,
    as the eager oracle the fused kernels' equivalence tests compare against.

    Raises:
        ValueError: if ``impl`` is not one of ``SCAN_IMPLS``, or a fused scan off CUDA or
            without fla-core.
    """
    if impl == "reference":
        return "reference"
    if impl not in FUSED_INPUT_DTYPES:
        msg = f"scan must be one of {SCAN_IMPLS}, got {impl!r}"
        raise ValueError(msg)
    if device.type != "cuda":
        msg = f"the fused scan needs the model on CUDA, got {device}"
        raise ValueError(msg)
    if not fused_scan_available():
        msg = "the fused scan needs a CUDA device and fla-core (the gpu extra)"
        raise ValueError(msg)
    return "fused" if impl == "fused" else "fused-fp32"


def check_capturable_scan(impl: str, device: torch.device) -> ScanImpl:
    """``check_scan``, and the choice must run as a captured CUDA graph on CUDA.

    For every driver that launches work (training, evaluation): CUDA graphs are a hard
    requirement for anything on the GPU (the project owner 2026-10-21), and the reference scan's
    block inverse (``torch.inverse``) cannot be captured, so on CUDA the choice is a fused scan.
    Not for: equivalence tests, whose eager CUDA reference is the oracle (``check_scan``).

    Raises:
        ValueError: as ``check_scan``, or the reference scan on CUDA.
    """
    chosen = check_scan(impl, device)
    if chosen == "reference" and device.type == "cuda":
        msg = (
            "the reference scan cannot run as a captured CUDA graph (torch.inverse); on CUDA"
            " use fused or fused-fp32"
        )
        raise ValueError(msg)
    return chosen


def scan_of(model: nn.Module) -> ScanImpl:
    """The scan implementation every ``RecurrentSublayer`` in ``model`` uses, read from them.

    Raises:
        ValueError: if the model has no recurrent sublayer or its sublayers disagree.
    """
    impls: set[ScanImpl] = {
        m.scan_impl for m in model.modules() if isinstance(m, RecurrentSublayer)
    }
    if len(impls) != 1:
        msg = f"recurrent sublayers use {sorted(impls)} scan implementations, expected exactly one"
        raise ValueError(msg)
    return impls.pop()


@dataclass(frozen=True)
class ColumnState:
    """Per-pass state of one column: recurrent states, attention windows and key histories.

    ``h[t]`` is ``[b, heads, dk, dk]``, ``kv[t]`` one window (a chronological ``KVCache``, or a
    ``RingKVCache`` for one-token decoding, whose shapes never change), ``key_history[t]`` the
    last ``K-1`` projected key inputs ``[b, K-1, d]`` (oldest first, the key filter's carry;
    ``[b, 0, d]`` without a filter). All three tuples have ``T_max`` entries. ``map_state`` is the
    one place that walks the fields; ``check_column_state`` validates a restored one. The
    retrieved entries per token come with retrieval; the carried projections of halted tokens
    live in the loop, which owns halting.
    """

    h: tuple[Tensor, ...]
    kv: tuple[KVCache | RingKVCache, ...]
    key_history: tuple[Tensor, ...]


def init_column_state(
    shape: ColumnShape,
    b: int,
    t_max: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    ring: bool = False,
) -> ColumnState:
    """Cold state: ``t_max`` zero recurrent states, empty windows and zero key histories.

    ``dtype`` is the activation dtype; the KV windows take it, the recurrent states and key
    histories take ``promote_types(dtype, float32)`` because the scan and the key filter never run
    below float32 (``RecurrentSublayer.scan``, ``RecurrentSublayer.keys``). ``ring`` gives every
    pass an unwritten ``RingKVCache`` instead of an empty ``KVCache`` (one-token decoding).

    Raises:
        ValueError: if ``t_max`` is below 1 (``kv_cache_empty`` rejects a bad ``window``).
    """
    if t_max < 1:
        msg = f"t_max must be at least 1, got {t_max}"
        raise ValueError(msg)
    dk = shape.head_dim
    state_dtype = torch.promote_types(dtype, torch.float32)
    empty = kv_ring_empty if ring else kv_cache_empty
    return ColumnState(
        h=tuple(
            torch.zeros(b, shape.heads, dk, dk, dtype=state_dtype, device=device)
            for _ in range(t_max)
        ),
        kv=tuple(empty(b, shape.kv_heads, dk, shape.window, dtype, device) for _ in range(t_max)),
        key_history=tuple(
            torch.zeros(b, shape.key_history, shape.d, dtype=state_dtype, device=device)
            for _ in range(t_max)
        ),
    )


def map_state(state: ColumnState, f: Callable[[Tensor], Tensor]) -> ColumnState:
    """``state`` with every tensor passed through ``f``, the typed structure and windows kept.

    PRIMITIVE: the one walk over a state's fields (detach at a chunk boundary, move to a device,
    copy); a new field is added here once. Not for: validation (``check_column_state``) or a
    serialized record (the trainer's, which keeps plain containers for ``weights_only`` loads).
    """

    def window(c: KVCache | RingKVCache) -> KVCache | RingKVCache:
        if isinstance(c, RingKVCache):
            return RingKVCache(k=f(c.k), v=f(c.v), pos=f(c.pos), valid=f(c.valid), window=c.window)
        return KVCache(k=f(c.k), v=f(c.v), pos=f(c.pos), window=c.window)

    return ColumnState(
        h=tuple(f(h) for h in state.h),
        kv=tuple(window(c) for c in state.kv),
        key_history=tuple(f(z) for z in state.key_history),
    )


def reset_recurrent(state: ColumnState, keep: Tensor) -> ColumnState:
    """PRIMITIVE ``state`` with the recurrent memory of the rows where ``keep`` is 0 made cold.

    ``keep`` is ``[b]`` (1 carries, 0 restarts): those rows' recurrent states ``h`` and key
    histories become exactly zero, what ``init_column_state`` gives (``torch.where``, so a value
    that is not finite cannot survive as ``0 * x``); the attention windows are left as they are
    (a cold row's window is the short context it would build within its first chunk anyway, and
    a window cannot be emptied for one row of a shared cache). Static shapes, capturable.

    A recurrent restart is not a full cold start: the attention windows keep their context.

    For: the trainer's staggered restarts (``train.reset_keep``). Not for: a full cold start of a
    whole batch (``init_column_state``).

    Raises:
        ContractError: when ``keep`` is not ``[b]`` for the state's ``b`` rows (maintainer B
        9e3066c: a
            ``[1]`` keep broadcast over three rows and cleared them all).
    """
    rows = next(iter(state.h + state.key_history)).shape[0]
    check_shape("keep", keep, ("b",), {"b": int(rows)})
    hold = keep.gt(0)

    def where(t: Tensor) -> Tensor:
        return torch.where(hold.view(-1, *([1] * (t.dim() - 1))), t, torch.zeros_like(t))

    return ColumnState(
        h=tuple(where(h) for h in state.h),
        kv=state.kv,
        key_history=tuple(where(z) for z in state.key_history),
    )


def state_tensors(states: Sequence[ColumnState]) -> list[Tensor]:
    """PRIMITIVE every tensor of ``states`` as a flat list, in ``map_state``'s fixed order.

    For: a captured CUDA graph's values and outputs, which are flat tensor lists (the trainer's
    step, the evaluations' chunks). ``with_state_tensors`` is its inverse.
    """
    out: list[Tensor] = []

    def collect(t: Tensor) -> Tensor:
        out.append(t)
        return t

    for state in states:
        _ = map_state(state, collect)
    return out


def with_state_tensors(
    templates: Sequence[ColumnState], tensors: Sequence[Tensor]
) -> list[ColumnState]:
    """PRIMITIVE ``templates``' structure (and windows) filled with ``tensors`` in order.

    The inverse of ``state_tensors``: only the templates' structure is read, never their values.

    Raises:
        ValueError: if the count differs from the templates' own.
    """
    expected = len(state_tensors(templates))
    if expected != len(tensors):
        msg = f"{len(tensors)} state tensors for a structure of {expected}"
        raise ValueError(msg)
    feed = iter(tensors)
    return [map_state(state, lambda _: next(feed)) for state in templates]


def check_column_state(state: ColumnState, shape: ColumnShape, t_max: int) -> None:
    """Refuse a state that this column geometry and pass count cannot run (a restored one).

    The dtype and device policy (maintainer B's review of f4c9279): recurrent states and key
    histories are real floating point, float32 or wider; attention windows are real floating point
    at any width (they take the activation dtype, bf16 under autocast) with ``int64`` positions;
    every tensor of the state lives on one device. Moving a valid state to another device is
    ``map_state(state, lambda t: t.to(device))``, which keeps it on one device.

    Raises:
        ValueError: on a pass count other than ``t_max`` in any field, a window other than the
            shape's, a recurrent state or key history that is not real floating point of
            float32 or wider, an attention window that is not real floating point, or a tensor
            on another device than ``h[0]``.
        ContractError: on a tensor shape that does not match the geometry, or batch sizes that
            differ between fields.
    """
    counts = {len(state.h), len(state.kv), len(state.key_history)}
    if counts != {t_max}:
        msg = f"state has {sorted(counts)} passes in its fields, the loop runs {t_max}"
        raise ValueError(msg)
    dims: Dims = {
        "h": shape.heads,
        "hkv": shape.kv_heads,
        "dk": shape.head_dim,
        "d": shape.d,
        "w": shape.key_history,
    }
    for t in range(t_max):
        h, cache, history = state.h[t], state.kv[t], state.key_history[t]
        check_shape(f"h[{t}]", h, ("b", "h", "dk", "dk"), dims)
        check_shape(f"key_history[{t}]", history, ("b", "w", "d"), dims)
        _check_window(t, cache, shape, dims, state.h[0].device)
        for name, tensor in ((f"h[{t}]", h), (f"key_history[{t}]", history)):
            # promote_types alone admits complex64 (maintainer B): a real floating dtype is required
            # first
            if not tensor.dtype.is_floating_point or (
                torch.promote_types(tensor.dtype, torch.float32) != tensor.dtype
            ):
                msg = f"{name} is {tensor.dtype}; carried states are real float32 or wider"
                raise ValueError(msg)
            if tensor.device != state.h[0].device:
                msg = f"{name} is on {tensor.device}, h[0] on {state.h[0].device}; one device"
                raise ValueError(msg)


def _check_window(
    t: int, cache: KVCache | RingKVCache, shape: ColumnShape, dims: Dims, device: torch.device
) -> None:
    """``check_column_state``'s part for pass ``t``'s window (either kind), same errors."""
    named: list[tuple[str, Tensor]] = [
        (f"kv[{t}].k", cache.k),
        (f"kv[{t}].v", cache.v),
        (f"kv[{t}].pos", cache.pos),
    ]
    if isinstance(cache, RingKVCache):
        dims[f"m{t}"] = shape.window  # a ring always holds exactly window slots
        check_shape(f"kv[{t}].valid", cache.valid, ("b", f"m{t}"), dims, dtype=torch.bool)
        named.append((f"kv[{t}].valid", cache.valid))
    check_shape(f"kv[{t}].k", cache.k, ("b", f"m{t}", "hkv", "dk"), dims)
    check_shape(f"kv[{t}].v", cache.v, ("b", f"m{t}", "hkv", "dk"), dims)
    check_shape(f"kv[{t}].pos", cache.pos, ("b", f"m{t}"), dims, dtype=torch.long)
    if cache.window != shape.window:
        msg = f"kv[{t}] has window {cache.window}, the column has {shape.window}"
        raise ValueError(msg)
    for name, tensor in named[:2]:
        if not tensor.dtype.is_floating_point:
            msg = f"{name} is {tensor.dtype}; attention windows are real floating point"
            raise ValueError(msg)
    for name, tensor in named:
        if tensor.device != device:
            msg = f"{name} is on {tensor.device}, h[0] on {device}; one device"
            raise ValueError(msg)


class Column(nn.Module):
    """Recurrent core, windowed attention and FFN, each pre-RMSNorm with a residual.

    Call ``forward`` or ``step`` directly (the generic module call is untyped).
    """

    def __init__(self, shape: ColumnShape, generator: torch.Generator | None = None) -> None:
        """Build the three sublayers; ``generator`` makes the initialization reproducible.

        Raises:
            ValueError: from the sublayers, on any inconsistent dimension.
        """
        super().__init__()
        self.norm_rec: RMSNorm = RMSNorm(shape.d)
        self.rec: RecurrentSublayer = RecurrentSublayer(shape, generator)
        self.norm_attn: RMSNorm = RMSNorm(shape.d)
        self.attn: WindowedAttention = WindowedAttention(
            shape.d,
            shape.heads,
            shape.kv_groups,
            shape.head_dim,
            shape.window,
            shape.rotary_base,
            generator,
        )
        self.norm_ffn: RMSNorm = RMSNorm(shape.d)
        self.ffn: FFN = FFN(shape.d, shape.ffn_mult, generator)

    def step(
        self,
        x: Tensor,
        pos: Tensor,
        state: ColumnState,
        pass_index: int,
        *,
        modulation: Modulation | None = None,
        carry: Carry | None = None,
    ) -> tuple[Tensor, ColumnState, Projections]:
        """One pass of this column over a chunk, through the state of pass ``pass_index``.

        Where ``carry.active`` is false the token has halted: its carried projections are written
        into this pass's states instead of fresh ones (spec 4, copy-forward), and its output
        position is not meaningful; the loop keeps such a token's value. The returned
        ``Projections`` are what was written, fresh or carried, so the loop can carry them again.

        Args:
            x: ``[b, n, d]`` the conditioned column input.
            pos: ``int64[b, n]`` absolute positions, consecutive within the chunk.
            state: this column's per-pass state.
            pass_index: ``t`` in ``[0, T_max)``; only ``h[t]`` and ``kv[t]`` are read and written.
            modulation: the modulator's signal and strength (inference), or None.
            carry: the active mask with the projections to write for inactive positions, or None
                when every token is active.

        Returns:
            ``(y [b, n, d], state', written)`` where ``state'`` holds the same objects as ``state``
            at every pass but ``t``; a zero-token chunk returns ``[b, 0, d]`` and the same state
            objects.

        Raises:
            ValueError: if ``pass_index`` is outside the state or the state's two tuples differ in
                length.
            ContractError: from the sublayers, a ``carry.active`` that is not ``bool[b, n]``, or
                carried projections whose shapes do not match this chunk and this column.
        """
        if not len(state.h) == len(state.kv) == len(state.key_history):
            msg = (
                f"state has {len(state.h)} recurrent states, {len(state.kv)} windows and"
                f" {len(state.key_history)} key histories"
            )
            raise ValueError(msg)
        if not 0 <= pass_index < len(state.h):
            msg = f"pass_index must lie in [0, {len(state.h)}), got {pass_index}"
            raise ValueError(msg)
        history = state.key_history[pass_index]
        check_shape(
            "state.key_history",
            history,
            ("b", "w", "d"),
            {"b": x.shape[0], "w": max(self.rec.key_conv - 1, 0), "d": x.shape[2]},
        )
        if carry is not None:
            dims: Dims = {
                "b": x.shape[0],
                "n": x.shape[1],
                "d": x.shape[2],
                "h": self.rec.heads,
                "hkv": self.attn.kv_heads,
                "dk": self.rec.head_dim,
            }
            check_shape("carry.active", carry.active, ("b", "n"), dims, dtype=torch.bool)
            check_shape("carry.rec_z", carry.projections.rec_z, ("b", "n", "d"), dims)
            check_shape("carry.rec_k", carry.projections.rec_k, ("b", "n", "h", "dk"), dims)
            check_shape("carry.rec_v", carry.projections.rec_v, ("b", "n", "h", "dk"), dims)
            check_shape("carry.rec_beta", carry.projections.rec_beta, ("b", "n", "h"), dims)
            check_shape("carry.attn_k", carry.projections.attn_k, ("b", "n", "hkv", "dk"), dims)
            check_shape("carry.attn_v", carry.projections.attn_v, ("b", "n", "hkv", "dk"), dims)

        q, z, v, beta = self.rec.project(self.norm_rec.forward(x), modulation)
        if carry is not None:
            # two merges in this order (maintainer B's review of d9eada0): the filter of an active
            # successor reads a halted token's FROZEN exit-pass input, selected before filtering;
            # the halted token's own key is its frozen rec_k, selected after
            z = _carry(carry.active, z, carry.projections.rec_z)
        k, history_next = self.rec.keys(z, history)
        if carry is not None:
            k = _carry(carry.active, k, carry.projections.rec_k)
            v = _carry(carry.active, v, carry.projections.rec_v)
            beta = _carry(carry.active, beta, carry.projections.rec_beta)
        r, h_new = self.rec.scan(q, k, v, beta, state.h[pass_index])
        x = x + r

        aq, ak, av = self.attn.project(self.norm_attn.forward(x), pos)
        if carry is not None:
            ak = _carry(carry.active, ak, carry.projections.attn_k)
            av = _carry(carry.active, av, carry.projections.attn_v)
        window = state.kv[pass_index]
        kv_new: KVCache | RingKVCache
        if isinstance(window, RingKVCache):
            a, kv_new = self.attn.attend_ring(aq, ak, av, pos, window)
        else:
            a, kv_new = self.attn.attend(aq, ak, av, pos, window)
        x = x + a

        x = x + self.ffn.forward(self.norm_ffn.forward(x))
        h = tuple(h_new if i == pass_index else s for i, s in enumerate(state.h))
        kv = tuple(kv_new if i == pass_index else c for i, c in enumerate(state.kv))
        key_history = tuple(
            history_next if i == pass_index else z_old for i, z_old in enumerate(state.key_history)
        )
        written = Projections(rec_z=z, rec_k=k, rec_v=v, rec_beta=beta, attn_k=ak, attn_v=av)
        return x, ColumnState(h=h, kv=kv, key_history=key_history), written

    @override
    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        state: ColumnState,
        pass_index: int,
        modulation: Modulation | None = None,
    ) -> tuple[Tensor, ColumnState]:
        """``step`` with every token active; see there."""
        y, new_state, _ = self.step(x, pos, state, pass_index, modulation=modulation)
        return y, new_state
