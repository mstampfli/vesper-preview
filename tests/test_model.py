import copy
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor

from conftest import small_config, small_raw
from vesper.config import load, parse
from vesper.contracts import ContractError
from vesper.data import Chunk, Document, Stream
from vesper.learning import LossWeights
from vesper.loop import LoopResult
from vesper.model import Vesper, count_parameters, wake_objective
from vesper.tokenizer import train_from_texts

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CPU = torch.device("cpu")


def one_chunk(docs: list[Document], slots: int, chunk: int) -> Chunk:
    produced = Stream(iter(docs), slots=slots, chunk=chunk).next()
    assert produced is not None
    return produced


def test_forward_shapes_and_carried_states() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 64, (2, 16), generator=g).to(torch.int32)
    pos = torch.arange(16).expand(2, 16)
    logits, result = model.forward(tokens, pos, model.init_states(2, torch.float32, CPU))
    assert logits.shape == (2, 16, 64)
    assert bool(torch.isfinite(logits).all().item())
    assert len(result.states) == 2
    assert int(result.n_iters.min().item()) >= 1
    assert int(result.n_iters.max().item()) <= 2
    # the carried state changes the next chunk's answer
    carried, _ = model.forward(tokens.flip(1), pos + 16, result.states)
    fresh, _ = model.forward(tokens.flip(1), pos + 16, model.init_states(2, torch.float32, CPU))
    assert (carried - fresh).abs().max().item() > 1e-6


def test_parameter_count_small_and_vesper_s() -> None:
    cfg = small_config()
    m = cfg.model
    d = m.d
    column = 4 * d * d + m.heads * d  # recurrent core with its write-strength matrix
    column += cfg.recurrence.key_conv * d  # the key filter's taps (0 without a filter)
    column += 2 * d * d + 2 * d * (d // m.kv_groups)  # attention under GQA
    column += 2 * m.ffn_mult * d * d  # feed-forward
    column += 3 * d  # three RMSNorm gains
    halting = 2 * d + 1  # the head's direction, its norm gain and its bias
    expected = m.columns * column + m.vocab * d + m.t_max * d + halting + d
    assert count_parameters(Vesper(cfg)) == expected
    # the baselines document counts Vesper-S at 60,222,721 with retrieval; without the retrieval
    # sublayer (four matrices, its gate and its norm per column) the assembled model holds this,
    # plus the key filter's 4 x 768 taps per column (the project owner's decision of 2026-10-15)
    without_retrieval = 60_222_721 - 4 * (196_608 + 768 + 768)
    vesper_s = load(CONFIGS / "vesper-s.yaml")
    assert vesper_s.recurrence.key_conv == 4
    assert count_parameters(Vesper(vesper_s)) == without_retrieval + 4 * 4 * 768
    assert cfg.recurrence.key_conv > 0  # the small config exercises the filter


def test_one_optimizer_step_lowers_the_loss_on_the_same_chunk() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(2)
    tokens = torch.randint(0, 64, (2, 17), generator=g).to(torch.int32)
    chunk = one_chunk([Document("a", tokens[0]), Document("b", tokens[1])], slots=2, chunk=16)
    pos = torch.arange(16).expand(2, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses: list[float] = []
    for _ in range(2):
        logits, result = model.forward(chunk.tokens, pos, model.init_states(2, torch.float32, CPU))
        total, parts = wake_objective(logits, chunk, result, cfg)
        assert set(parts) == {"ntp", "ponder"}
        opt.zero_grad()
        grads = torch.autograd.grad(total, list(model.parameters()))
        for p, grad in zip(model.parameters(), grads, strict=True):
            p.grad = grad
        # the optimizer's step is untyped in torch; the gradients above are typed through autograd
        opt.step()  # pyright: ignore[reportUnknownMemberType, reportUnusedCallResult]
        losses.append(float(total.item()))
    # the loss recorded on the second pass is the one after the first update
    assert losses[1] < losses[0], losses


def test_wake_objective_skips_masked_positions() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [
        Document("a", torch.arange(1, 6, dtype=torch.int32)),
        Document("b", torch.arange(10, 15, dtype=torch.int32)),
    ]
    chunk = one_chunk(docs, slots=1, chunk=8)
    assert not bool(chunk.loss_mask.all().item())  # the boundary position carries no loss
    pos = torch.arange(8).expand(1, 8)
    logits, result = model.forward(chunk.tokens, pos, model.init_states(1, torch.float32, CPU))
    total, _ = wake_objective(logits, chunk, result, cfg)
    assert bool(torch.isfinite(total).item())
    # moving the logits at a masked position does not move the loss; at a live one it does
    masked = int((~chunk.loss_mask[0]).nonzero()[0].item())
    live = int(chunk.loss_mask[0].nonzero()[0].item())
    bumped = logits.clone()
    bumped[0, masked, 0] += 10.0  # one vocabulary entry, so the softmax does move
    assert abs(wake_objective(bumped, chunk, result, cfg)[0].item() - total.item()) < 1e-12
    bumped = logits.clone()
    bumped[0, live, 0] += 10.0
    assert abs(wake_objective(bumped, chunk, result, cfg)[0].item() - total.item()) > 1e-6


def test_wake_objective_refuses_a_chunk_without_a_loss_position() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    chunk = one_chunk([Document("a", torch.arange(1, 10, dtype=torch.int32))], slots=1, chunk=8)
    silenced = Chunk(
        tokens=chunk.tokens,
        targets=chunk.targets,
        valid=chunk.valid,
        doc_start=chunk.doc_start,
        segment=chunk.segment,
        loss_mask=torch.zeros_like(chunk.loss_mask),
        pos=chunk.pos,
        doc_ids=chunk.doc_ids,
        spans=chunk.spans,
    )
    pos = torch.arange(8).expand(1, 8)
    logits, result = model.forward(chunk.tokens, pos, model.init_states(1, torch.float32, CPU))
    with pytest.raises(ContractError, match="no position"):
        _ = wake_objective(logits, silenced, result, cfg)


def test_data_to_model_round_trip_with_a_trained_tokenizer() -> None:
    tok = train_from_texts(["the delta rule erases and writes"] * 20, vocab=300)
    cfg = small_config(vocab=tok.vocab_size)
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    docs = [Document(f"d{i}", tok.encode("the delta rule erases and writes")) for i in range(4)]
    chunk = one_chunk(docs, slots=2, chunk=16)
    pos = torch.arange(16).expand(2, 16)
    logits, result = model.forward(chunk.tokens, pos, model.init_states(2, torch.float32, CPU))
    assert logits.shape == (2, 16, tok.vocab_size)
    total, _ = wake_objective(logits, chunk, result, cfg)
    assert bool(torch.isfinite(total).item())
    predicted = logits.argmax(-1)[0, 0]
    assert 0 <= int(predicted.item()) < tok.vocab_size
    assert isinstance(tok.decode(predicted.reshape(1)), str)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Vesper-S forward and backward need a GPU"
)
def test_vesper_s_forward_and_backward_on_the_gpu_under_autocast() -> None:
    cfg = load(CONFIGS / "vesper-s.yaml")
    cuda = torch.device("cuda")
    model = Vesper(cfg, torch.Generator().manual_seed(0)).to(cuda)
    g = torch.Generator().manual_seed(3)
    tokens = torch.randint(0, cfg.model.vocab, (1, 129), generator=g).to(torch.int32)
    chunk = one_chunk([Document("a", tokens[0])], slots=1, chunk=128)
    pos = torch.arange(128, device=cuda).reshape(1, 128)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, result = model.forward(
            chunk.tokens.to(cuda), pos, model.init_states(1, torch.float32, cuda)
        )
        total, _ = wake_objective(logits.float(), chunk, result, cfg)
    assert bool(torch.isfinite(total).item())
    assert result.states[0].h[0].dtype == torch.float32
    (g_e,) = torch.autograd.grad(total, [model.embedding.weight])
    assert bool(torch.isfinite(g_e).all().item())
    assert int(result.n_iters.max().item()) <= cfg.model.t_max


def test_the_model_runs_only_the_requested_passes() -> None:
    # the recipe anneals T_max: a model built for two passes runs one on request, the pass run
    # is the forced halt, and the state of the pass not run stays untouched
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    g = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, 64, (1, 8), generator=g).to(torch.int32)
    pos = torch.arange(8).reshape(1, 8)
    _, one = model.forward(tokens, pos, model.init_states(1, torch.float32, CPU), passes=1)
    assert int(one.n_iters.max().item()) == 1
    assert one.per_pass.shape[0] == 1
    assert all(float(s.h[1].abs().sum().item()) == 0.0 for s in one.states)
    _, two = model.forward(tokens, pos, model.init_states(1, torch.float32, CPU), passes=2)
    assert two.per_pass.shape[0] == 2
    for bad in (0, 3):
        with pytest.raises(ValueError, match="passes"):
            _ = model.forward(tokens, pos, model.init_states(1, torch.float32, CPU), passes=bad)


def test_recompute_gives_the_same_objective_gradients_and_states() -> None:
    # activation recomputation per column application is an implementation of the same forward
    # and backward: the objective, every parameter's gradient, the carried states, the iteration
    # counts and the per-pass outputs agree with the plain forward
    cfg = small_config()
    kept = Vesper(cfg, torch.Generator().manual_seed(0))
    recomputed = copy.deepcopy(kept)
    docs = [Document("a", torch.arange(1, 40, dtype=torch.int32))]
    chunk = one_chunk(docs, slots=2, chunk=16)
    pos = chunk.pos
    weights = LossWeights(lambda_h=0.01, lambda_g=cfg.learning.lambda_g)
    results: list[tuple[float, LoopResult, list[Tensor]]] = []
    for model, recompute in ((kept, False), (recomputed, True)):
        states = model.init_states(2, torch.float32, CPU)
        logits, result = model.forward(chunk.tokens, pos, states, recompute=recompute)
        objective, _ = wake_objective(logits, chunk, result, cfg, weights=weights)
        torch.autograd.backward([objective])
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        assert len(grads) == len(list(model.parameters()))
        results.append((float(objective.item()), result, grads))
    (loss_a, a, grads_a), (loss_b, b, grads_b) = results
    assert loss_b == pytest.approx(loss_a, rel=1e-6, abs=1e-7)
    for ga, gb in zip(grads_a, grads_b, strict=True):
        assert torch.allclose(ga, gb, atol=1e-6, rtol=1e-5)
    assert torch.equal(a.n_iters, b.n_iters)
    assert torch.allclose(a.per_pass, b.per_pass, atol=1e-6, rtol=1e-5)
    for sa, sb in zip(a.states, b.states, strict=True):
        assert all(
            torch.allclose(x, y, atol=1e-6, rtol=1e-5) for x, y in zip(sa.h, sb.h, strict=True)
        )
        assert all(
            torch.allclose(x.k, y.k, atol=1e-6, rtol=1e-5) and torch.equal(x.pos, y.pos)
            for x, y in zip(sa.kv, sb.kv, strict=True)
        )


def test_the_sliced_objective_equals_the_full_one_and_never_reads_masked_positions() -> None:
    # the memory row: the head over the loss positions in slices, the softmax over the full
    # vocabulary per position, the same value and gradients as the objective over [b, n, V] logits
    cfg = small_config()
    weights = LossWeights(lambda_h=0.02, lambda_g=cfg.learning.lambda_g)
    docs = [
        Document("a", torch.arange(1, 6, dtype=torch.int32)),
        Document("b", torch.arange(20, 29, dtype=torch.int32)),
    ]
    chunk = one_chunk(docs, slots=2, chunk=8)
    assert not bool(chunk.loss_mask.all().item())  # the short slot's padding carries no loss
    outcomes: list[tuple[float, list[Tensor]]] = []
    # float64: the slicing changes summation order, and float32 rounding amplified by
    # cancellation in loop.u's gradient sat at 0.7 of a float32 bound and crossed it on CI
    for slices in (0, 1, 3, 100):
        model = Vesper(cfg, torch.Generator().manual_seed(0)).to(torch.float64)
        states = model.init_states(2, torch.float64, CPU)
        if slices == 0:
            logits, result = model.forward(chunk.tokens, chunk.pos, states)
            objective, parts = wake_objective(logits, chunk, result, cfg, weights=weights)
        else:
            result = model.run(chunk.tokens, chunk.pos, states)
            objective, parts = model.objective(chunk, result, cfg, weights, slices=slices)
        assert set(parts) == {"ntp", "ponder"}
        torch.autograd.backward([objective])
        grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        assert len(grads) == len(list(model.parameters()))
        outcomes.append((float(objective.item()), grads))
    reference, grads_ref = outcomes[0]
    for value, grads in outcomes[1:]:
        # measured in float64: values within 1e-15, gradients at 1.5e-3 of this bound
        assert value == pytest.approx(reference, rel=1e-12, abs=1e-14)
        for g, r in zip(grads, grads_ref, strict=True):
            assert torch.allclose(g, r, atol=1e-12, rtol=1e-10)
    # a masked position's output has no path into the value or the gradient
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    result = model.run(chunk.tokens, chunk.pos, model.init_states(2, torch.float32, CPU))
    poisoned = result.y.clone()
    poisoned[~chunk.loss_mask] = float("nan")
    poisoned_result = LoopResult(
        y=poisoned,
        states=result.states,
        n_iters=result.n_iters,
        ponder=result.ponder,
        per_pass=result.per_pass,
    )
    objective, _ = model.objective(chunk, poisoned_result, cfg, weights, slices=2)
    assert math.isfinite(float(objective.item()))
    torch.autograd.backward([objective])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="slices"):
        _ = model.objective(chunk, result, cfg, weights, slices=0)
    silenced = Chunk(
        tokens=chunk.tokens,
        targets=chunk.targets,
        valid=chunk.valid,
        doc_start=chunk.doc_start,
        segment=chunk.segment,
        loss_mask=torch.zeros_like(chunk.loss_mask),
        pos=chunk.pos,
        doc_ids=chunk.doc_ids,
        spans=chunk.spans,
    )
    with pytest.raises(ContractError, match="no position"):
        _ = model.objective(silenced, result, cfg, weights)


def test_both_loss_paths_refuse_the_local_mode_through_one_check() -> None:
    # Maintainer B's finding (review of 629559f): the sliced objective trained in local mode while
    # wake_objective refused it; one validator now sits on both paths
    raw = small_raw()
    raw["learning"]["mode"] = "local"
    cfg = parse(raw, "test")
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    chunk = one_chunk([Document("a", torch.arange(1, 18, dtype=torch.int32))], slots=1, chunk=16)
    states = model.init_states(1, torch.float32, CPU)
    logits, result = model.forward(chunk.tokens, chunk.pos, states)
    with pytest.raises(NotImplementedError, match="local"):
        _ = wake_objective(logits, chunk, result, cfg)
    with pytest.raises(NotImplementedError, match="local"):
        _ = model.objective(chunk, result, cfg, slices=2)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast needs the GPU")
def test_the_sliced_objective_under_bf16_autocast_differs_only_by_rounding() -> None:
    # under bf16 the slice boundaries change the rounding of the head's matmuls; the objective
    # is the same formula, so the differences are stated as tolerances, not as equality: loss
    # within 2e-3 relative, every parameter's gradient within 1e-2 in relative L2 (maintainer B
    # measured 5e-4 to 7e-4 on CPU bf16 at 3 and 100 slices)
    cfg = small_config()
    cuda = torch.device("cuda")
    weights = LossWeights(lambda_h=0.02, lambda_g=cfg.learning.lambda_g)
    docs = [
        Document("a", torch.arange(1, 6, dtype=torch.int32)),
        Document("b", torch.arange(20, 29, dtype=torch.int32)),
    ]
    chunk = one_chunk(docs, slots=2, chunk=8)
    outcomes: list[tuple[float, list[Tensor]]] = []
    for slices in (1, 4):
        model = Vesper(cfg, torch.Generator().manual_seed(0)).to(cuda)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = model.run(
                chunk.tokens.to(cuda), chunk.pos.to(cuda), model.init_states(2, torch.float32, cuda)
            )
            objective, _ = model.objective(chunk, result, cfg, weights, slices=slices)
        torch.autograd.backward([objective])
        grads = [p.grad.detach().float().clone() for p in model.parameters() if p.grad is not None]
        outcomes.append((float(objective.item()), grads))
    (loss_a, grads_a), (loss_b, grads_b) = outcomes
    assert loss_b == pytest.approx(loss_a, rel=2e-3)
    for ga, gb in zip(grads_a, grads_b, strict=True):
        scale = float(ga.square().sum().sqrt().item()) or 1.0
        assert float((ga - gb).square().sum().sqrt().item()) / scale < 1e-2


def test_wake_objective_takes_the_schedules_weights() -> None:
    cfg = small_config()
    model = Vesper(cfg, torch.Generator().manual_seed(0))
    chunk = one_chunk([Document("a", torch.arange(1, 18, dtype=torch.int32))], slots=1, chunk=16)
    pos = torch.arange(16).reshape(1, 16)
    logits, result = model.forward(chunk.tokens, pos, model.init_states(1, torch.float32, CPU))
    default, parts = wake_objective(logits, chunk, result, cfg)
    zero, zero_parts = wake_objective(logits, chunk, result, cfg, weights=LossWeights(0.0, 0.0))
    assert zero_parts["ntp"].item() == parts["ntp"].item()
    assert float(zero.item()) == pytest.approx(float(parts["ntp"].item()))
    assert float(default.item()) != float(zero.item())
