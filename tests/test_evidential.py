"""Hermetic unit tests for Evidential AstroJev and Dirichlet Subjective Logic."""
import torch
import pytest

from celestrium.evidential_astrojev import (
    EvidentialAstroJev,
    evidential_brier_loss,
    kl_divergence_dirichlet,
)
from celestrium.astrojev import NUM_FEATURES, NUM_CLASSES


def test_kl_divergence_dirichlet_non_negative():
    # When alpha is all ones (uniform prior), KL must be 0
    alpha_ones = torch.ones(4, 4)
    kl_zero = kl_divergence_dirichlet(alpha_ones, num_classes=4)
    assert torch.allclose(kl_zero, torch.zeros(4), atol=1e-5)

    # When alpha departs from 1, KL must be strictly positive
    alpha_peaky = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    kl_pos = kl_divergence_dirichlet(alpha_peaky, num_classes=4)
    assert kl_pos.item() > 0.0


def test_evidential_brier_loss_computation():
    torch.manual_seed(42)
    alpha = torch.tensor([[5.0, 1.0, 1.0, 1.0]], requires_grad=True)
    labels = torch.tensor([0])

    loss, probs, brier_comp = evidential_brier_loss(alpha, labels, num_classes=4)
    loss.backward()

    assert not torch.isnan(loss)
    assert alpha.grad is not None
    assert probs.shape == (1, 4)
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_evidential_astrojev_forward_and_decomposition():
    model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=64, num_classes=NUM_CLASSES, n_iter=4)
    x = torch.randn(8, NUM_FEATURES)
    out = model(x)

    # Output schema verification
    assert "alpha" in out
    assert "evidence" in out
    assert "probs" in out
    assert "noul" in out
    assert "u_epi" in out
    assert "u_ale" in out
    assert "delta_eq" in out
    assert "sparsity" in out

    # Dirichlet properties
    assert (out["alpha"] >= 1.0).all()
    assert (out["evidence"] >= 0.0).all()
    assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(8), atol=1e-5)

    # Epistemic decomposition properties: u_epi + noul == 1.0
    assert torch.allclose(out["u_epi"] + out["noul"], torch.ones(8), atol=1e-5)
    assert (out["noul"] >= 0.0).all() and (out["noul"] <= 1.0).all()
    assert (out["u_ale"] >= 0.0).all()

    # Sparsity bound
    assert out["sparsity"] >= 0.40
