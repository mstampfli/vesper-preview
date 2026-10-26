"""The single-token decoding step and its captured form (vesper/decode.py).

References (maintainer B's review of 49e7b88): the chunked forward over the whole sequence is the
behavioural truth for the tokenwise ring step (CPU); the same step run eagerly on CUDA is the
truth for its capture (bitwise, every logit and every carried state at every step, across the
warm-up/capture boundary); a second generation on one decoder must equal a fresh decoder's.
"""

import pytest
import torch
from torch import Tensor

from conftest import small_config
from vesper.column import set_scan, state_tensors
from vesper.contracts import ContractError
from vesper.decode import CapturedDecoder, decode_step
from vesper.graphs import WARM_STEPS
from vesper.loop import DenseRule
from vesper.model import Vesper

CPU = torch.device("cpu")
CUDA = torch.device("cuda")
VOCAB = 64
DEPTH = 2
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _model(device: torch.device, *, scan: str | None = None) -> Vesper:
    """The small model on ``device``; on CUDA the capturable scan unless ``scan`` says otherwise,
    as ``instruct._on`` moves models."""
    model = Vesper(small_config(VOCAB), torch.Generator().manual_seed(0)).to(device).eval()
    chosen = scan or ("fused-fp32" if device.type == "cuda" else "reference")
    _ = set_scan(model, chosen)
    return model


def _sequence(n: int, seed: int = 3) -> Tensor:
    return torch.randint(0, VOCAB, (1, n), generator=torch.Generator().manual_seed(seed))


def _at(i: int, device: torch.device) -> Tensor:
    return torch.full((1, 1), i, dtype=torch.int64, device=device)


def test_tokenwise_ring_decoding_equals_the_chunked_forward() -> None:
    model = _model(CPU)
    n = 20  # the small config's window is 8: the ring wraps twice
    tokens = _sequence(n)
    with torch.no_grad():
        chunked = model.run(
            tokens,
            torch.arange(n)[None],
            model.init_states(1, torch.float32, CPU),
            passes=DEPTH,
            exit_rule=DenseRule(),
        )
        reference = model.head(chunked.y)[..., :VOCAB]
        templates = model.init_states(1, torch.float32, CPU, ring=True)
        flat = state_tensors(templates)
        for i in range(n):
            out = decode_step(
                model, DEPTH, VOCAB, templates, [tokens[:, i : i + 1], _at(i, CPU), *flat]
            )
            torch.testing.assert_close(out[1], reference[:, i : i + 1], rtol=1e-5, atol=1e-5)
            assert int(out[0]) == int(reference[0, i].argmax())
            flat = out[2:]
    # the carried recurrent states agree with the chunked run's (windows differ in kind)
    ring_h = [t for t in flat if t.dim() == 4 and t.shape[-1] == t.shape[-2]]
    chunk_h = [h for state in chunked.states for h in state.h]
    for got, want in zip(ring_h, chunk_h, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


@needs_cuda
def test_the_captured_step_equals_the_eager_step_bitwise_across_the_capture() -> None:
    model = _model(CUDA)
    decoder = CapturedDecoder(model, DEPTH, VOCAB)
    templates = model.init_states(1, torch.float32, CUDA, ring=True)
    tokens = _sequence(WARM_STEPS + 6).to(CUDA)
    eager_flat, captured_flat = decoder.start(), decoder.start()
    with torch.no_grad():
        for i in range(int(tokens.shape[1])):
            values = [tokens[:, i : i + 1], _at(i, CUDA)]
            eager = decode_step(model, DEPTH, VOCAB, templates, [*values, *eager_flat])
            nxt, logits, flat = decoder.step(values[0], values[1], captured_flat)
            assert torch.equal(nxt, eager[0]), i
            assert torch.equal(logits, eager[1]), i
            for got, want in zip(flat, eager[2:], strict=True):
                assert torch.equal(got, want), i
            eager_flat = eager[2:]
            captured_flat = [t.clone() for t in flat]  # the graph overwrites its outputs
    assert decoder.graph.captures == 1
    assert decoder.graph.replays == int(tokens.shape[1]) - WARM_STEPS


@needs_cuda
def test_a_second_generation_on_one_decoder_starts_truly_cold() -> None:
    # Maintainer B's review of 0e342c3: comparing two decoders after the SAME first prompt cannot
    # see retained history; the used decoder's second prompt is compared with a cold eager reference
    model = _model(CUDA)
    used = CapturedDecoder(model, DEPTH, VOCAB)
    templates = model.init_states(1, torch.float32, CUDA, ring=True)
    first, second = _sequence(7, seed=4).to(CUDA), _sequence(9, seed=5).to(CUDA)
    with torch.no_grad():
        flat = used.start()
        for i in range(int(first.shape[1])):  # warms and captures on a discarded prompt
            _, _, flat = used.step(first[:, i : i + 1], _at(i, CUDA), flat)
            flat = [t.clone() for t in flat]
        assert used.graph.captures == 1
        flat, cold = used.start(), state_tensors(templates)
        for i in range(int(second.shape[1])):
            values = [second[:, i : i + 1], _at(i, CUDA)]
            _, logits, flat = used.step(values[0], values[1], flat)
            reference = decode_step(model, DEPTH, VOCAB, templates, [*values, *cold])
            assert torch.equal(logits, reference[1]), i
            flat, cold = [t.clone() for t in flat], reference[2:]


def test_a_decoder_refuses_the_cpu_and_a_depth_outside_the_passes() -> None:
    with pytest.raises(ContractError, match="CUDA"):
        _ = CapturedDecoder(_model(CPU), DEPTH, VOCAB)
    if torch.cuda.is_available():
        with pytest.raises(ContractError, match="depth"):
            _ = CapturedDecoder(_model(CUDA), 3, VOCAB)
        with pytest.raises(ContractError, match="reference scan"):
            _ = CapturedDecoder(_model(CUDA, scan="reference"), DEPTH, VOCAB)
