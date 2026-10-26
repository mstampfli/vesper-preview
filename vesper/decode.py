"""Greedy one-token decoding of a Vesper on CUDA as one captured CUDA graph.

PRIMITIVE ``CapturedDecoder``. ``instruct.generate`` on CUDA feeds the prompt and then every new
id through one single-token step: the model's dense forward at a fixed depth from carried states,
the head on the last position, the argmax over the decodable vocabulary. Every column keeps one
recurrent state, key history and ``RingKVCache`` per pass, so the step's shapes never change and
one ``WarmedGraph`` serves the whole prompt and answer (maintainer B's review of 49e7b88: the
growing ``KVCache`` would change the signature every token). The first ``WARM_STEPS`` calls of a
decoder run eagerly (real tokens, each consumed once), every later call replays; a new generation
starts from fresh cold states copied into the same captured buffers, so nothing of an earlier prompt
survives. One host read per token remains: the id, for the caller's marker check.

For: greedy generation on CUDA (``instruct.generate``). Not for: training, chunked evaluation, or
the CPU, whose path (``instruct.generate`` without a decoder) is the behavioural reference.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from vesper.column import ColumnState, scan_of, state_tensors, with_state_tensors
from vesper.contracts import ContractError
from vesper.graphs import WarmedGraph
from vesper.loop import DenseRule
from vesper.model import Vesper


def decode_step(
    model: Vesper,
    depth: int,
    vocab: int,
    templates: Sequence[ColumnState],
    values: Sequence[Tensor],
) -> list[Tensor]:
    """PRIMITIVE the single-token step itself, eager on any device: ``CapturedDecoder`` captures it.

    ``values`` are the token ``int64 [1, 1]``, its position ``int64 [1, 1]`` and the flat states
    shaped like ``templates`` (``state_tensors``); the result is the argmax id over the first
    ``vocab`` ids ``[1, 1]``, those logits ``[1, 1, vocab]`` and the new flat states.
    """
    token, position, *flat = values
    states = with_state_tensors(templates, flat)
    result = model.run(token, position, states, passes=depth, exit_rule=DenseRule())
    logits = model.head(result.y[:, -1:])[..., :vocab]
    return [logits.argmax(-1), logits, *state_tensors(result.states)]


class CapturedDecoder:
    """PRIMITIVE one model's greedy single-token step at ``depth`` passes, captured on CUDA.

    ``start()`` gives cold states; ``step(token, position, flat)`` advances them by one token and
    returns the argmax id over the first ``vocab`` ids (a ``[1, 1]`` device tensor), the last
    logits and the new flat states. Batch 1 only (the ring writes one slot per row; a batch would
    need its own decoding loop). The outputs are the graph's buffers after a replay: read or pass
    them on before the next step.

    Raises:
        ContractError: if the model is not on CUDA, its scan is the reference one (it cannot be
            captured: ``column.set_scan(model, "fused-fp32")``, as ``instruct._on`` does), or
            ``depth`` is outside ``1..t_max``.
    """

    def __init__(self, model: Vesper, depth: int, vocab: int) -> None:
        """Bind the model and the depth; nothing is captured before the first ``step``."""
        device = next(model.parameters()).device
        if device.type != "cuda":
            msg = f"CapturedDecoder runs on CUDA, the model is on {device}"
            raise ContractError(msg)
        if scan_of(model) == "reference":
            msg = "the reference scan cannot be captured: set_scan(model, 'fused-fp32') first"
            raise ContractError(msg)
        if not 1 <= depth <= model.t_max:
            msg = f"depth {depth} is outside 1..{model.t_max}"
            raise ContractError(msg)
        self.model: Vesper = model
        self.depth: int = depth
        self.vocab: int = vocab
        self.device: torch.device = device
        self.graph: WarmedGraph = WarmedGraph()
        self._templates: list[ColumnState] = model.init_states(1, torch.float32, device, ring=True)

    def start(self) -> list[Tensor]:
        """Cold states for a new generation, as the step's flat state tensors."""
        cold = self.model.init_states(1, torch.float32, self.device, ring=True)
        return state_tensors(cold)

    def _forward(self, values: Sequence[Tensor]) -> list[Tensor]:
        return decode_step(self.model, self.depth, self.vocab, self._templates, values)

    def step(
        self, token: Tensor, position: Tensor, flat: Sequence[Tensor]
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """One token (``int64 [1, 1]``) at ``position`` (``int64 [1, 1]``): id, logits, states."""
        with torch.no_grad():
            out = self.graph(self._forward, [token, position, *flat])
        return out[0], out[1], list(out[2:])
