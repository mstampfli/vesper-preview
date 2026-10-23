"""One CUDA calibration run for tests/test_determinism.py, in its own process (not a test file).

    python tests/calibration_repeat_probe.py SOURCE SHA256 OUT.json

120 joint updates (rho 0.1, class weights, depth cost 2.5, the taper floor, so the floor and the
warmup's rates change every update and the warmup ends at 100) through ``train_update`` with a
``WarmedGraph`` (three eager updates, then the capture and replays) and the capturable optimizer,
inside ``vesper.graphs.deterministic_algorithms``; at update 60 the CPU evaluation copy
(``evaluating``) runs one forward between two replays. Writes every record, a sha256 of every
parameter and every Adam field after every update, the evaluation output's digest, the
capture counts and the effective settings, as JSON.
"""

import json
import sys
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from vesper.column import set_scan
from vesper.graphs import WarmedGraph, deterministic_algorithms
from vesper.halting_calibration import (
    EPISODES,
    _floor_at,  # pyright: ignore[reportPrivateUsage]
    _source_model,  # pyright: ignore[reportPrivateUsage]
    train_update,
)
from vesper.halting_common import chained_digest, evaluating
from vesper.joint_update import ROLLOUTS, optimizer, release_backbone
from vesper.synthetic import TERMINAL_MIX, episodes, generator_for

UPDATES = 120
EVAL_AT = 60
# the driver's batch (EPISODES, imported): at 4 episodes two undeterministic processes still
# agreed over 120 updates on the test model, at 32 they differ from update 0 (measured 10-19)


def _digest(tensors: list[Tensor]) -> str:
    return chained_digest("", tensors)  # dtype, shape and bytes of each, in order


def main() -> None:
    source, sha, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    _, model, head = _source_model(source, sha)
    backbone = release_backbone(model)
    model = model.to(torch.device("cuda"))
    _ = set_scan(model, "fused-fp32")
    params = head + backbone
    opt = optimizer(head, backbone, capturable=True)
    graph = WarmedGraph()
    data, action = generator_for("repeat-train", 0), generator_for("repeat-action", 0)
    t_max = model.loop.t_max
    records: list[dict[str, object]] = []
    states: list[str] = []
    evaluation = ""
    with deterministic_algorithms() as settings:
        for update in range(UPDATES):
            ep = episodes(data, EPISODES, f"repeat-{update}", TERMINAL_MIX)
            n = int(ep.chunk.tokens.shape[1])
            uniforms = torch.rand(
                (t_max - 1, ROLLOUTS * EPISODES, n), generator=action, dtype=torch.float64
            )
            epsilon = _floor_at("taper", update)
            records.append(
                train_update(
                    model, opt, params, ep, uniforms, epsilon, update, 2.5, "class", 0.1, graph
                )
            )
            adam = cast("dict[Tensor, dict[str, Tensor]]", opt.state)
            fields = [adam[p][k] for p in params for k in ("exp_avg", "exp_avg_sq", "step")]
            states.append(_digest([*params, *fields]))
            if update == EVAL_AT:
                with evaluating(model) as cpu:
                    c = ep.chunk
                    fresh = cpu.init_states(
                        int(c.tokens.shape[0]), torch.float32, torch.device("cpu")
                    )
                    evaluation = _digest([cpu.run(c.tokens, c.pos, fresh).y])
    result = {
        "records": records,
        "states": states,
        "evaluation": evaluation,
        "captures": graph.captures,
        "replays": graph.replays,
        "settings": settings,
    }
    _ = out.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
