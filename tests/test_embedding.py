import pytest
import torch

from vesper.contracts import ContractError
from vesper.embedding import TiedEmbedding

V, D = 50, 16


def _emb(seed: int = 0) -> TiedEmbedding:
    g = torch.Generator().manual_seed(seed)
    return TiedEmbedding(vocab=V, d=D, generator=g).to(torch.float64)


def test_shapes_and_single_matrix() -> None:
    emb = _emb()
    tokens = torch.arange(1, 7).view(2, 3)
    x = emb.embed(tokens)
    assert x.shape == (2, 3, D)
    logits = emb.logits(x)
    assert logits.shape == (2, 3, V)
    assert sum(p.numel() for p in emb.parameters()) == V * D


def test_embedding_rows_and_logits_use_the_same_matrix() -> None:
    emb = _emb()
    tokens = torch.zeros(1, 3, dtype=torch.long)
    tokens[0, 0] = 7
    tokens[0, 2] = V - 1
    x = emb.embed(tokens)
    assert torch.equal(x[0, 0], emb.weight[7])
    assert torch.equal(x[0, 2], emb.weight[V - 1])
    # logits(x) = x E^T, so the logit of a token against its own embedding is its squared norm.
    logits = emb.logits(x)
    assert (logits[0, 0, 7] - emb.weight[7].square().sum()).abs().item() < 1e-12


def test_gradient_reaches_the_matrix_from_both_sides() -> None:
    emb = _emb()
    tokens = torch.full((1, 2), 3, dtype=torch.long)
    tokens[0, 1] = 9
    (g,) = torch.autograd.grad(emb.logits(emb.embed(tokens)).square().sum(), [emb.weight])
    assert g.abs().sum().item() > 0
    # Every row takes part in logits, so all rows move; this alone cannot show the lookup path.
    assert (g.abs().sum(dim=1) > 0).all()


def mat(rows: list[list[float]]) -> torch.Tensor:
    out = torch.zeros(len(rows), len(rows[0]), dtype=torch.float64)
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            out[i, j] = value
    return out


def test_lookup_gradient_counts_each_row_by_its_uses() -> None:
    # The lookup path alone: a detached lookup passes every other test in this file and fails here.
    emb = TiedEmbedding(vocab=3, d=2).to(torch.float64)
    with torch.no_grad():
        _ = emb.weight.copy_(mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    ids = torch.zeros(1, 3, dtype=torch.long)
    ids[0, 1] = 2
    x = emb.embed(ids)
    assert torch.equal(x[0], mat([[1.0, 2.0], [5.0, 6.0], [1.0, 2.0]]))
    (g,) = torch.autograd.grad(x.sum(), [emb.weight])
    assert torch.equal(g, mat([[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]]))
    hidden = mat([[2.0, -1.0]]).reshape(1, 1, 2)
    assert torch.equal(emb.logits(hidden)[0, 0], mat([[0.0, 2.0, 4.0]])[0])


def test_int32_ids_are_widened_at_the_boundary() -> None:
    emb = _emb()
    ids64 = torch.arange(1, 7).view(2, 3)
    assert torch.equal(emb.embed(ids64.to(torch.int32)), emb.embed(ids64))
    with pytest.raises(ContractError, match="int32 or int64"):
        _ = emb.embed(ids64.to(torch.float32))


def test_rejects_token_ids_outside_the_vocabulary() -> None:
    emb = _emb()
    with pytest.raises(ContractError, match="vocab"):
        _ = emb.embed(torch.full((1, 2), V, dtype=torch.long))
    with pytest.raises(ContractError, match="vocab"):
        _ = emb.embed(torch.full((1, 2), -1, dtype=torch.long))


def test_rejects_wrong_width() -> None:
    emb = _emb()
    with pytest.raises(ContractError):
        _ = emb.logits(torch.zeros(1, 2, D + 1, dtype=torch.float64))
