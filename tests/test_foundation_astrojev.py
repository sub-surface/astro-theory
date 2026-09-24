"""Tests for FoundationAstroJev and heteroscedastic error-calibrated phenomena streaming."""
import pytest
import numpy as np
import torch

from celestrium.foundation_astrojev import (
    FoundationAstroJev,
    heteroscedastic_evidential_loss,
    PHENOMENA_CLASSES,
    NUM_PHENOMENA_CLASSES,
    NUM_OBSERVABLES,
    NUM_UNCERTAINTIES,
    NUM_MASKS,
)
from celestrium.data_streamer import SyntheticPhenomenaGenerator, AstroStreamingDataset


def test_synthetic_phenomena_generator():
    gen = SyntheticPhenomenaGenerator(seed=101)
    for c_idx in range(NUM_PHENOMENA_CLASSES):
        rec = gen.generate_sample(class_idx=c_idx)
        assert rec.label == c_idx
        assert rec.class_name == PHENOMENA_CLASSES[c_idx]
        assert rec.mu.shape == (NUM_OBSERVABLES,)
        assert rec.sigma.shape == (NUM_UNCERTAINTIES,)
        assert rec.mask.shape == (NUM_MASKS,)
        assert np.all(np.isfinite(rec.mu))
        assert np.all(np.isfinite(rec.sigma))
        assert np.all(rec.sigma > 0)


def test_astro_streaming_dataset():
    ds = AstroStreamingDataset(total_samples=500, batch_size=64, seed=42)
    batches = list(ds)
    assert len(batches) > 0
    total_count = 0
    for mu, sigma, mask, labels in batches:
        b_sz = mu.size(0)
        total_count += b_sz
        assert mu.shape == (b_sz, NUM_OBSERVABLES)
        assert sigma.shape == (b_sz, NUM_UNCERTAINTIES)
        assert mask.shape == (b_sz, NUM_MASKS)
        assert labels.shape == (b_sz,)
        assert torch.all(labels >= 0) and torch.all(labels < NUM_PHENOMENA_CLASSES)
    assert total_count == 500


def test_foundation_astrojev_forward():
    model = FoundationAstroJev(num_classes=NUM_PHENOMENA_CLASSES, d_model=64, n_iter=5)
    model.eval()

    batch_size = 16
    mu = torch.randn(batch_size, NUM_OBSERVABLES)
    sigma = torch.rand(batch_size, NUM_UNCERTAINTIES) * 0.1 + 0.01
    mask = torch.ones(batch_size, NUM_MASKS)

    out = model(mu, sigma, mask=mask, apply_jitter=False)

    probs = out["probs"]
    alpha = out["alpha"]
    u_epi = out["u_epi"]
    u_ale = out["u_ale"]
    noul = out["noul"]
    sparsity = out["sparsity"]

    # Verify probability simplex
    assert probs.shape == (batch_size, NUM_PHENOMENA_CLASSES)
    assert torch.allclose(torch.sum(probs, dim=-1), torch.ones(batch_size), atol=1e-5)

    # Verify Dirichlet parameters
    assert torch.all(alpha >= 1.0)
    assert u_epi.shape == (batch_size,)
    assert torch.all(u_epi > 0.0) and torch.all(u_epi <= 1.0)
    assert u_ale.shape == (batch_size,)
    assert torch.all(u_ale >= 0.0)
    assert noul.shape == (batch_size,)
    assert torch.all(noul >= 0.0) and torch.all(noul <= 1.0)

    # Verify CReLU sparsity (~50% empirical zero activations)
    assert float(sparsity.item()) >= 0.40


def test_heteroscedastic_loss_and_backward():
    model = FoundationAstroJev(num_classes=NUM_PHENOMENA_CLASSES, d_model=64, n_iter=3)
    model.train()

    batch_size = 8
    mu = torch.randn(batch_size, NUM_OBSERVABLES)
    sigma = torch.rand(batch_size, NUM_UNCERTAINTIES) * 0.1 + 0.01
    labels = torch.randint(0, NUM_PHENOMENA_CLASSES, (batch_size,))

    out = model(mu, sigma, apply_jitter=True)
    loss = heteroscedastic_evidential_loss(out["alpha"], labels, num_classes=NUM_PHENOMENA_CLASSES)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    loss.backward()
    # Verify gradients flow to encoder and heads
    assert model.encoder.linear.weight.grad is not None
    assert model.evidence_head.weight.grad is not None


def test_local_convergence_small_scale():
    """Verify local small-scale training convergence on 600 sources over 4 epochs."""
    torch.manual_seed(42)
    model = FoundationAstroJev(num_classes=NUM_PHENOMENA_CLASSES, d_model=64, n_iter=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = AstroStreamingDataset(total_samples=600, batch_size=64, seed=42)
    initial_loss = None
    final_loss = None

    model.train()
    for epoch in range(4):
        epoch_losses = []
        for mu, sigma, mask, labels in ds:
            optimizer.zero_grad()
            out = model(mu, sigma, mask=mask, apply_jitter=True)
            loss = heteroscedastic_evidential_loss(out["alpha"], labels, num_classes=NUM_PHENOMENA_CLASSES)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        mean_loss = float(np.mean(epoch_losses))
        if epoch == 0:
            initial_loss = mean_loss
        final_loss = mean_loss

    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
