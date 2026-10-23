"""The captured training step (vesper/graphs.py, halting_experiment.GraphedTraining).

The eager step on the same device is the truth: a captured run must train exactly as the eager
one does, step for step, from the first replay on (the warm steps are eager by construction).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
import torch
import yaml
from torch import Tensor

from vesper.column import scan_of, set_scan
from vesper.config import VesperConfig, load, parse
from vesper.contracts import ContractError
from vesper.graphs import (
    WARM_STEPS,
    CapturedStep,
    WarmedGraph,
    memory_peak,
    reset_memory_peak,
)
from vesper.halting_common import evaluation_model
from vesper.halting_experiment import (
    GraphedTraining,
    build,
    run,
    train_step,
)
from vesper.kernels.delta_scan import fused_scan_available
from vesper.synthetic import episodes, generator_for, hop_mix
from vesper.train import optimizer

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "halting-stage1-c1.yaml"
CUDA = torch.device("cuda")
needs_fused = pytest.mark.skipif(
    not fused_scan_available(), reason="captured steps need CUDA and the fused scan"
)


def test_a_captured_step_refuses_inputs_off_cuda() -> None:
    with pytest.raises(ContractError, match="not CUDA"):
        _ = CapturedStep(list, [torch.zeros(2)])


@pytest.mark.parametrize(
    ("device", "scan", "cuda_graph", "match"),
    [
        ("cpu", "reference", True, "cuda_graph needs device cuda"),
        ("cpu", "fused", False, "fused scan needs"),
        ("tpu", "reference", False, "device must be one of"),
        ("cpu", "triton", False, "scan must be one of"),
    ],
)
def test_backend_choices_are_refused_before_anything_is_written(
    tmp_path: Path, device: str, scan: str, *, cuda_graph: bool, match: str
) -> None:
    out = tmp_path / "run"
    with pytest.raises(ValueError, match=match):
        _ = run(CONFIG, "fixed", 0, out, steps=1, device=device, scan=scan, cuda_graph=cuda_graph)
    assert not out.exists()


def test_an_unknown_condition_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    out = tmp_path / "e"
    with pytest.raises(ValueError, match="condition must be one of"):
        _ = run(CONFIG, "e", 0, out, steps=1)
    assert not out.exists()


def test_a_cpu_run_records_its_backend(tmp_path: Path) -> None:
    result = run(CONFIG, "fixed", 0, tmp_path / "cpu", steps=1, eval_batches=1, eval_size=2)
    assert (result["device"], result["scan"], result["cuda_graph"]) == ("cpu", "reference", False)
    backend = cast("dict[str, object]", result["backend"])
    assert backend["device"] == "cpu"
    assert backend["device_name"] is None
    assert (backend["scan"], backend["scan_input_dtype"]) == ("reference", "torch.float32")
    assert backend["parameter_dtypes"] == ["torch.float32"]
    assert (backend["state_dtype"], backend["autocast"]) == ("torch.float32", None)
    assert backend["float32_matmul_precision"] == torch.get_float32_matmul_precision()
    assert backend["torch"] == torch.__version__
    assert backend["evaluation"] == {"device": "cpu", "scan": "reference"}


Trained = tuple[list[dict[str, object]], list[Tensor], GraphedTraining | None]


def _trained(
    *,
    graphed: bool,
    steps: int,
    condition: str = "aggregate",
    cfg: VesperConfig | None = None,
    scan: str = "fused",
) -> Trained:
    cfg = load(CONFIG) if cfg is None else cfg
    model = build(cfg, condition, 0).to(CUDA)
    _ = set_scan(model, scan)
    opt = optimizer(model, cfg, capturable=True)
    trainer = GraphedTraining(model, opt, cfg, condition, 0.1) if graphed else None
    data = generator_for("train", 0)
    records: list[dict[str, object]] = []
    for step in range(steps):
        ep = episodes(data, cfg.training.batch_chunks, f"t-{step}", hop_mix(step, "v2-slow"))
        if trainer is None:
            records.append(train_step(model, opt, cfg, condition, 0.1, ep, step))
        else:
            records.append(trainer.step(ep, step))
    return records, [p.detach().clone() for p in model.parameters()], trainer


def _assert_same_training(eager: Trained, graphed: Trained, keys: tuple[str, ...]) -> None:
    for step, (a, b) in enumerate(zip(eager[0], graphed[0], strict=True)):
        for key in keys:
            want, got = cast("float", a[key]), cast("float", b[key])
            assert got == pytest.approx(want, rel=1e-6, abs=1e-9), (step, key)
    for eager_p, graphed_p in zip(eager[1], graphed[1], strict=True):
        assert float((eager_p - graphed_p).abs().max()) < 1e-6


@needs_fused
def test_act_is_captured_once_per_phase_and_trains_as_the_eager_step() -> None:
    # a 40-step budget puts the end of the warm phase (head frozen, no ponder cost) at step 4:
    # steps 0-2 eager, 3 captured (frozen), 4-6 eager again (the head is freed: its optimizer
    # state is created outside a capture), 7 captured, 7-9 replayed with the ponder weight
    # ramping every step
    raw = cast("dict[str, dict[str, object]]", yaml.safe_load(CONFIG.read_text("utf-8")))
    raw["training"]["total_tokens"] = 40 * 32 * 88
    cfg = parse(raw, "test")
    eager = _trained(graphed=False, steps=10, condition="act", cfg=cfg)
    graphed = _trained(graphed=True, steps=10, condition="act", cfg=cfg)
    _assert_same_training(eager, graphed, ("loss", "grad_norm", "lr", "loss_depth", "ponder"))
    trainer = graphed[2]
    assert trainer is not None
    assert trainer.captures == 2
    ramp = [cast("float", r["ponder"]) for r in graphed[0]]
    assert ramp[:5] == [0.0] * 5  # the warm phase carries no ponder cost; the ramp starts at 0
    assert all(p > 0 for p in ramp[5:])  # and is read from the filled tensor in every replay


@needs_fused
@pytest.mark.parametrize("scan", ["fused", "fused-fp32"])
def test_the_captured_training_equals_the_eager_training(scan: str) -> None:
    # condition D (the learned head and the batch KL: every part of the update is exercised);
    # three warm steps, then three replays, each on a new batch and a new warmup learning rate
    steps = WARM_STEPS + 3
    eager = _trained(graphed=False, steps=steps, scan=scan)
    graphed = _trained(graphed=True, steps=steps, scan=scan)
    keys = ("loss", "grad_norm", "lr", "loss_depth", "expected_ntp", "kl")
    _assert_same_training(eager, graphed, keys)
    assert len({cast("float", r["loss"]) for r in graphed[0]}) == steps  # a new batch each
    assert graphed[2] is not None
    assert graphed[2].captures == 1


@needs_fused
def test_a_warmed_graph_runs_eagerly_then_captures_once_and_replays_on_its_own_buffers() -> None:
    calls: list[int] = []

    def step(values: Sequence[Tensor]) -> list[Tensor]:
        calls.append(len(calls))  # Python runs only while warming and capturing
        return [values[0] * 2.0 + values[1]]

    graph = WarmedGraph(warm=2)
    inputs = [
        (torch.full((3,), float(i), device=CUDA), torch.ones(3, device=CUDA)) for i in range(5)
    ]
    kept = [tuple(t.clone() for t in pair) for pair in inputs]
    for i, pair in enumerate(inputs):
        (out,) = graph(step, pair)
        assert torch.equal(out, torch.full((3,), 2.0 * i + 1.0, device=CUDA)), i
    assert (graph.captures, graph.replays, len(calls)) == (1, 3, 3)  # 2 eager + 1 capture
    for pair, copy in zip(inputs, kept, strict=True):  # no caller's tensor became a buffer
        assert all(torch.equal(a, b) for a, b in zip(pair, copy, strict=True))
    graph.reset()
    _ = graph(step, inputs[0])
    assert (graph.captures, len(calls)) == (1, 4)  # eager again after a reset
    with pytest.raises(ContractError, match="value 0 is on cpu"):
        _ = graph(step, [torch.zeros(3), torch.ones(3, device=CUDA)])


@needs_fused
def test_the_memory_peak_counts_a_replayed_graphs_pool() -> None:
    floats = 64 * 2**20
    size = 4 * floats  # bytes of an intermediate the step frees before it returns

    def step(values: Sequence[Tensor]) -> list[Tensor]:
        return [values[0].repeat(floats).sum().reshape(1)]

    graph = WarmedGraph(warm=1)
    x = torch.ones(1, device=CUDA)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_reserved(CUDA)
    _ = graph(step, [x])  # eager
    _ = graph(step, [x])  # capture: the graph's own pool keeps a block for the intermediate
    reset_memory_peak(CUDA)
    (out,) = graph(step, [x])  # a replay allocates nothing
    assert float(out) == floats
    assert memory_peak(CUDA) - base >= size  # yet the pool is held, and counted
    assert memory_peak(torch.device("cpu")) == 0


@needs_fused
def test_a_replay_refuses_a_batch_of_another_shape() -> None:
    cfg = load(CONFIG)
    model = build(cfg, "fixed", 0).to(CUDA)
    _ = set_scan(model, "fused")
    trainer = GraphedTraining(model, optimizer(model, cfg, capturable=True), cfg, "fixed", 0.1)
    data = generator_for("train", 0)
    for step in range(WARM_STEPS + 1):
        _ = trainer.step(episodes(data, cfg.training.batch_chunks, f"t-{step}", hop_mix(0)), step)
    smaller = episodes(data, cfg.training.batch_chunks - 1, "small", hop_mix(0))
    with pytest.raises(ContractError, match="captured input 0"):
        _ = trainer.step(smaller, WARM_STEPS + 1)


@needs_fused
@pytest.mark.parametrize(
    ("scan", "input_dtype"), [("fused", "torch.bfloat16"), ("fused-fp32", "torch.float32")]
)
def test_a_captured_run_records_its_backend_and_saves_cpu_weights(
    tmp_path: Path, scan: str, input_dtype: str
) -> None:
    out = tmp_path / "graphed"
    result = run(
        CONFIG,
        "fixed",
        0,
        out,
        steps=WARM_STEPS + 2,
        eval_batches=1,
        eval_size=2,
        weights_every=WARM_STEPS + 1,
        device="cuda",
        scan=scan,
        cuda_graph=True,
    )
    assert (result["device"], result["scan"], result["cuda_graph"]) == ("cuda", scan, True)
    backend = cast("dict[str, object]", result["backend"])
    assert backend["device"] == "cuda:0"
    assert backend["device_name"] == torch.cuda.get_device_name(0)
    assert (backend["scan"], backend["scan_input_dtype"]) == (scan, input_dtype)
    assert (backend["state_dtype"], backend["autocast"], backend["cuda_graph"]) == (
        "torch.float32",
        None,
        True,
    )
    assert backend["cuda"] == torch.version.cuda
    assert backend["fla-core"] == "0.5.2"
    assert isinstance(backend["triton"], str)
    assert backend["evaluation"] == {"device": "cpu", "scan": "reference"}
    for path in (out / "model.pt", out / "weights" / f"after-{WARM_STEPS + 1:05d}.pt"):
        state = cast("dict[str, Tensor]", torch.load(path, weights_only=True))
        assert all(t.device.type == "cpu" for t in state.values()), path
    assert "summary" in result  # evaluated, on the CPU copy


@needs_fused
def test_evaluation_sees_a_cpu_reference_copy_and_training_keeps_its_model() -> None:
    cfg = load(CONFIG)
    model = build(cfg, "fixed", 0).to(CUDA)
    _ = set_scan(model, "fused")
    seen = evaluation_model(model)
    assert seen is not model
    assert {p.device.type for p in seen.parameters()} == {"cpu"}
    assert scan_of(seen) == "reference"
    assert {p.device.type for p in model.parameters()} == {"cuda"}  # the training model as it was
    assert scan_of(model) == "fused"
    for a, b in zip(seen.parameters(), model.parameters(), strict=True):
        assert torch.equal(a, b.cpu())
    cpu_model = build(cfg, "fixed", 0)
    assert evaluation_model(cpu_model) is cpu_model  # a CPU run evaluates exactly as before
