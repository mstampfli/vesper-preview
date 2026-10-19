"""The causal filter on the recurrent key (docs/reviews/2026-10-15-retrieval-binding.md, 4.1-4.3).

Every expected value here is computed by hand from the operator's definition, not by the module's
own filter: ``u[t, j] = sum_r conv_k[r, j] * z[t - r, j]`` with ``r = 0`` the current token,
earlier positions from the carried history, the key ``normalize_keys(u)`` per head; a halted
token's frozen unfiltered input is what its successors read, and its own key stays its frozen
``rec_k`` (maintainer B's review of d9eada0).
"""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor

from vesper.column import (
    Carry,
    Column,
    ColumnShape,
    ColumnState,
    RecurrentSublayer,
    check_column_state,
    init_column_state,
    map_state,
)
from vesper.config import ConfigError, load, parse
from vesper.contracts import ContractError
from vesper.loop import TraceRule
from vesper.model import Vesper
from vesper.recurrence import normalize_keys

F64 = torch.float64
CPU = torch.device("cpu")
K = 4
SHAPE = ColumnShape(
    d=8,
    heads=2,
    kv_groups=2,
    head_dim=4,
    window=4,
    rotary_base=10000.0,
    ffn_mult=2,
    block=2,
    key_conv=K,
)
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1.yaml"


def _sublayer(shape: ColumnShape = SHAPE, seed: int = 0) -> RecurrentSublayer:
    return RecurrentSublayer(shape, torch.Generator().manual_seed(seed)).to(F64)


def _asymmetric_taps() -> Tensor:
    """``[K, d]`` taps whose every tap and channel differ, so a reversed or shifted filter shows."""
    r = torch.arange(K, dtype=F64)[:, None]
    j = torch.arange(SHAPE.d, dtype=F64)[None, :]
    return (r + 1) * 0.5 - 0.3 * j + 0.1 * r * j


def _literal_keys(taps: Tensor, z: Tensor, history: Tensor) -> tuple[Tensor, Tensor]:
    """The operator by its definition, one position and tap at a time."""
    b, n, d = z.shape
    full = torch.cat([history, z], dim=1)  # position t of the chunk is row K - 1 + t
    u = torch.zeros(b, n, d, dtype=z.dtype)
    for t in range(n):
        for r in range(K):
            u[:, t] += taps[r] * full[:, K - 1 + t - r]
    keys = normalize_keys(u.view(b, n, SHAPE.heads, SHAPE.head_dim))
    return keys, full[:, full.shape[1] - (K - 1) :]


def test_the_filter_matches_its_definition_with_an_asymmetric_kernel() -> None:
    sub = _sublayer()
    assert sub.conv_k is not None
    with torch.no_grad():
        _ = sub.conv_k.copy_(_asymmetric_taps())
    g = torch.Generator().manual_seed(1)
    z = torch.randn(2, 6, SHAPE.d, generator=g, dtype=F64)
    history = torch.randn(2, K - 1, SHAPE.d, generator=g, dtype=F64)
    keys, after = sub.keys(z, history)
    expected, expected_after = _literal_keys(_asymmetric_taps(), z, history)
    assert torch.allclose(keys, expected, rtol=0, atol=1e-12)
    assert torch.equal(after, expected_after)
    # the same chunk with the taps in the other order is a different filter, not the same one
    reversed_keys, _ = _literal_keys(_asymmetric_taps().flip(0), z, history)
    assert not torch.allclose(keys, reversed_keys, rtol=0, atol=1e-6)


def test_the_history_carries_exactly_the_newest_inputs() -> None:
    sub = _sublayer()
    g = torch.Generator().manual_seed(2)
    history = torch.randn(1, K - 1, SHAPE.d, generator=g, dtype=F64)
    # an empty chunk leaves the history itself
    empty = torch.zeros(1, 0, SHAPE.d, dtype=F64)
    keys, after = sub.keys(empty, history)
    assert keys.shape == (1, 0, SHAPE.heads, SHAPE.head_dim)
    assert after is history
    # chunks shorter than K - 1 keep part of the old history, in order
    for n in (1, 2):
        z = torch.randn(1, n, SHAPE.d, generator=g, dtype=F64)
        _, after = sub.keys(z, history)
        assert torch.equal(after, torch.cat([history[:, n:], z], dim=1))
    # a long chunk keeps only its own newest inputs, in storage of exactly that size
    z = torch.randn(1, 50, SHAPE.d, generator=g, dtype=F64)
    _, after = sub.keys(z, history)
    assert torch.equal(after, z[:, -(K - 1) :])
    assert after.untyped_storage().nbytes() == (K - 1) * SHAPE.d * 8


def test_streaming_the_filter_equals_one_pass_over_the_whole_sequence() -> None:
    sub = _sublayer()
    g = torch.Generator().manual_seed(3)
    z = torch.randn(1, 11, SHAPE.d, generator=g, dtype=F64)
    cold = torch.zeros(1, K - 1, SHAPE.d, dtype=F64)
    whole, _ = sub.keys(z, cold)
    for sizes in ((1,) * 11, (2, 0, 3, 1, 5), (11,), (0, 4, 7)):
        history, start = cold, 0
        parts: list[Tensor] = []
        for size in sizes:
            keys, history = sub.keys(z[:, start : start + size], history)
            parts.append(keys)
            start += size
        assert torch.allclose(torch.cat(parts, dim=1), whole, rtol=0, atol=1e-12), sizes


def test_no_filter_is_the_unfiltered_key_and_draws_nothing() -> None:
    plain = replace(SHAPE, key_conv=0)
    sub = _sublayer(plain)
    assert sub.conv_k is None
    assert "conv_k" not in dict(sub.named_parameters())
    g = torch.Generator().manual_seed(4)
    z = torch.randn(2, 5, SHAPE.d, generator=g, dtype=F64)
    none = torch.zeros(2, 0, SHAPE.d, dtype=F64)
    keys, after = sub.keys(z, none)
    assert torch.equal(keys, normalize_keys(z.view(2, 5, SHAPE.heads, SHAPE.head_dim)))
    assert after is none
    # the filter's taps are drawn after every other parameter: the rest is identical
    filtered = _sublayer(SHAPE)
    for name, p in sub.named_parameters():
        assert torch.equal(p, dict(filtered.named_parameters())[name]), name


def _column_with_taps() -> Column:
    col = Column(SHAPE, torch.Generator().manual_seed(5)).to(F64)
    assert col.rec.conv_k is not None
    with torch.no_grad():
        _ = col.rec.conv_k.copy_(_asymmetric_taps())
    return col


def test_a_successor_reads_its_halted_predecessors_frozen_input() -> None:
    # Maintainer B's merge order: the source is selected BEFORE filtering, the halted token's own
    # key AFTER. Token 0 halted earlier with a frozen unfiltered input z0 and a frozen key k0; token
    # 1 is active.
    col = _column_with_taps()
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 2, SHAPE.d, generator=g, dtype=F64)
    pos = torch.arange(2).expand(1, 2)
    state = init_column_state(SHAPE, 1, 1, F64, CPU)
    history = torch.randn(1, K - 1, SHAPE.d, generator=g, dtype=F64)
    state = ColumnState(h=state.h, kv=state.kv, key_history=(history,))
    _, _, fresh = col.step(x, pos, state, 0)  # what both tokens would write if active
    frozen_z = torch.randn(1, 2, SHAPE.d, generator=g, dtype=F64)
    frozen_k = normalize_keys(torch.randn(1, 2, SHAPE.heads, SHAPE.head_dim, generator=g))
    carried = type(fresh)(
        rec_z=frozen_z,
        rec_k=frozen_k.to(F64),
        rec_v=fresh.rec_v,
        rec_beta=fresh.rec_beta,
        attn_k=fresh.attn_k,
        attn_v=fresh.attn_v,
    )
    active = torch.zeros(1, 2, dtype=torch.bool)
    active[0, 1] = True
    _, after, written = col.step(x, pos, state, 0, carry=Carry(active, carried))
    # token 1's key filters (history, token 0's FROZEN z, its own fresh z)
    source = torch.cat([frozen_z[:, :1], fresh.rec_z[:, 1:]], dim=1)
    expected, expected_history = _literal_keys(_asymmetric_taps(), source, history)
    assert torch.allclose(written.rec_k[:, 1], expected[:, 1], rtol=0, atol=1e-12)
    # filtering token 0's fresh value first and merging after would give another key
    wrong, _ = _literal_keys(_asymmetric_taps(), fresh.rec_z, history)
    assert not torch.allclose(wrong[:, 1], expected[:, 1], rtol=0, atol=1e-6)
    # token 0 keeps its own frozen key and passes its frozen input on, also into the history
    assert torch.equal(written.rec_k[:, 0], frozen_k.to(F64)[:, 0])
    assert torch.equal(written.rec_z[:, 0], frozen_z[:, 0])
    assert torch.equal(after.key_history[0], expected_history)


def _model() -> Vesper:
    return Vesper(load(CONFIG), torch.Generator().manual_seed(7)).to(F64).eval()


def _run_in_chunks(
    model: Vesper, tokens: Tensor, depths: Tensor, sizes: tuple[int, ...]
) -> tuple[Tensor, list[ColumnState]]:
    n = int(tokens.shape[1])
    pos = torch.arange(n).expand(tokens.shape[0], n)
    states = model.init_states(int(tokens.shape[0]), F64, CPU)
    outputs: list[Tensor] = []
    start = 0
    for size in sizes:
        end = start + size
        result = model.run(
            tokens[:, start:end],
            pos[:, start:end],
            states,
            exit_rule=TraceRule(depths[:, start:end]),
        )
        states = list(result.states)
        outputs.append(result.y)
        start = end
    assert start == n
    return torch.cat(outputs, dim=1), states


def test_prefill_decode_and_chunks_agree_under_mixed_halting() -> None:
    # every predecessor window of K - 1 tokens holds tokens that halted at different passes
    model = _model()
    g = torch.Generator().manual_seed(8)
    tokens = torch.randint(0, 64, (2, 13), generator=g, dtype=torch.int32)
    depths = torch.randint(1, model.t_max + 1, (2, 13), generator=g, dtype=torch.int64)
    with torch.no_grad():
        whole, final = _run_in_chunks(model, tokens, depths, (13,))
        for sizes in ((1,) * 13, (2, 0, 5, 1, 2, 3), (4, 9)):
            y, states = _run_in_chunks(model, tokens, depths, sizes)
            assert torch.allclose(y, whole, rtol=0, atol=1e-10), sizes
            for a, b in zip(states, final, strict=True):
                for ha, hb in zip(a.h, b.h, strict=True):
                    assert torch.allclose(ha, hb, rtol=0, atol=1e-10), sizes
                for za, zb in zip(a.key_history, b.key_history, strict=True):
                    assert torch.allclose(za, zb, rtol=0, atol=1e-10), sizes


def test_chunked_gradients_equal_the_whole_and_stop_at_a_detached_boundary() -> None:
    model = _model()
    loop = model.loop
    g = torch.Generator().manual_seed(9)
    n, cut = 9, 4
    x0 = torch.randn(1, n, 32, generator=g, dtype=F64, requires_grad=True)
    pos = torch.arange(n).expand(1, n)

    def run(x: Tensor, p: Tensor, states: list[ColumnState]) -> tuple[Tensor, list[ColumnState]]:
        result = loop.forward(x, p, states, exit_rule=TraceRule(torch.full((1, x.shape[1]), 3)))
        return result.y, list(result.states)

    cold = model.init_states(1, F64, CPU)
    whole, _ = run(x0, pos, cold)
    (grad_whole,) = torch.autograd.grad(whole.square().sum(), [x0])
    first, carried = run(x0[:, :cut], pos[:, :cut], cold)
    second, _ = run(x0[:, cut:], pos[:, cut:], carried)
    loss = first.square().sum() + second.square().sum()
    (grad_chunked,) = torch.autograd.grad(loss, [x0])
    assert torch.allclose(grad_chunked, grad_whole, rtol=1e-9, atol=1e-10)
    # the trainer detaches the carried states (map_state(..., detach)): no gradient crosses
    first, carried = run(x0[:, :cut], pos[:, :cut], cold)
    detached = [map_state(s, lambda t: t.detach()) for s in carried]
    second, _ = run(x0[:, cut:], pos[:, cut:], detached)
    (grad_second,) = torch.autograd.grad(second.square().sum(), [x0])
    assert float(grad_second[:, :cut].abs().max()) == 0.0
    assert float(grad_second[:, cut:].abs().max()) > 0.0
    # without the detach the second chunk's loss reaches the first chunk through the history
    first, carried = run(x0[:, :cut], pos[:, :cut], cold)
    second, _ = run(x0[:, cut:], pos[:, cut:], carried)
    (grad_through,) = torch.autograd.grad(second.square().sum(), [x0])
    assert float(grad_through[:, :cut].abs().max()) > 0.0


def test_a_restored_state_is_checked_against_the_geometry() -> None:
    good = init_column_state(SHAPE, 2, 3, F64, CPU)
    check_column_state(good, SHAPE, 3)
    with pytest.raises(ValueError, match="passes"):
        check_column_state(good, SHAPE, 2)
    short = ColumnState(h=good.h, kv=good.kv, key_history=good.key_history[:2])
    with pytest.raises(ValueError, match="passes"):
        check_column_state(short, SHAPE, 3)
    wide = map_state(good, lambda t: t)
    wide = ColumnState(
        h=wide.h,
        kv=wide.kv,
        key_history=tuple(torch.zeros(2, K, SHAPE.d, dtype=F64) for _ in range(3)),
    )
    with pytest.raises(ContractError, match="key_history"):
        check_column_state(wide, SHAPE, 3)
    half = map_state(good, lambda t: t.half() if t.is_floating_point() else t)
    with pytest.raises(ValueError, match="float32 or wider"):
        check_column_state(half, SHAPE, 3)
    other_batch = ColumnState(
        h=good.h,
        kv=good.kv,
        key_history=tuple(torch.zeros(5, K - 1, SHAPE.d, dtype=F64) for _ in range(3)),
    )
    with pytest.raises(ContractError):
        check_column_state(other_batch, SHAPE, 3)


def test_a_restored_state_refuses_complex_integer_and_mixed_device_tensors() -> None:
    # Maintainer B's review of f4c9279: promote_types admitted complex64, integer windows and a
    # tensor on another device; each passed _states_from_record, and a complex history then ran with
    # its imaginary part discarded
    from vesper.train import (  # noqa: PLC0415
        _state_record,  # pyright: ignore[reportPrivateUsage]
        _states_from_record,  # pyright: ignore[reportPrivateUsage]
    )

    model = _model()
    good = model.init_states(1, F64, CPU)
    first, last = good[0], model.t_max - 1
    meta = torch.device("meta")
    cases = {
        r"key_history\[0\] is torch.complex128": replace(
            first, key_history=tuple(z.to(torch.complex128) + 1j for z in first.key_history)
        ),
        r"h\[0\] is torch.complex64": replace(
            first, h=tuple(h.to(torch.complex64) for h in first.h)
        ),
        r"kv\[0\].k is torch.int64": replace(
            first, kv=tuple(replace(c, k=c.k.long()) for c in first.kv)
        ),
        r"kv\[0\].v is torch.int32": replace(
            first, kv=tuple(replace(c, v=c.v.int()) for c in first.kv)
        ),
        r"kv\[0\].k is torch.complex64": replace(
            first, kv=tuple(replace(c, k=c.k.to(torch.complex64)) for c in first.kv)
        ),
        r"key_history\[0\] is on meta": replace(
            first, key_history=(first.key_history[0].to(meta), *first.key_history[1:])
        ),
        rf"kv\[{last}\].pos is on meta": replace(
            first, kv=(*first.kv[:-1], replace(first.kv[-1], pos=first.kv[-1].pos.to(meta)))
        ),
    }
    for message, bad in cases.items():
        with pytest.raises(ValueError, match=message):
            _ = _states_from_record(_state_record([bad, *good[1:]]), model)
    # valid states stay accepted: float64 throughout, and the autocast layout (bf16 windows
    # beside float32 recurrent states and histories), as init_column_state builds it
    _ = _states_from_record(_state_record(good), model)
    check_column_state(init_column_state(SHAPE, 2, 3, torch.bfloat16, CPU), SHAPE, 3)


def test_the_config_bounds_the_filter_width() -> None:
    import yaml  # noqa: PLC0415

    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    assert load(CONFIG).recurrence.key_conv == K
    for width in (0, 2, 8):
        raw["recurrence"]["key_conv"] = width
        assert parse(raw, "test").recurrence.key_conv == width
    for width in (-1, 1, 9):
        raw["recurrence"]["key_conv"] = width
        with pytest.raises(ConfigError, match="key_conv"):
            _ = parse(raw, "test")
