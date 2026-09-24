"""Hermetic unit tests for high-performance fused kernels."""
import torch
import torch.nn.functional as F
import pytest

from celestrium.kernels import (
    fused_crelu,
    fused_brier_loss,
    fused_rewarding_doubt_loss,
    fused_rlcd_loss,
)


def test_fused_crelu_gradients_and_sparsity():
    x = torch.tensor([-0.5, 0.2, 0.8, 1.5], requires_grad=True)
    out, sparsity = fused_crelu(x)

    # Clamping check
    assert torch.allclose(out, torch.tensor([0.0, 0.2, 0.8, 1.0]))
    assert sparsity == 0.25  # 1 out of 4 is 0.0

    # Backward gradient: active only in (0, 1)
    loss = out.sum()
    loss.backward()
    expected_grad = torch.tensor([0.0, 1.0, 1.0, 0.0])
    assert torch.allclose(x.grad, expected_grad)


def test_fused_brier_loss_exact_gradient():
    torch.manual_seed(42)
    logits = torch.randn(6, 4, requires_grad=True, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 3, 1, 2])

    loss, probs, rewards = fused_brier_loss(logits, labels)
    loss.backward()
    analytic_grad = logits.grad.clone()

    # Compare against numerical autograd of unfused formula
    logits_ref = logits.detach().clone().requires_grad_(True)
    probs_ref = F.softmax(logits_ref, dim=-1)
    one_hot = F.one_hot(labels, num_classes=4).to(torch.float64)
    loss_ref = torch.mean(torch.sum((probs_ref - one_hot) ** 2, dim=-1) / 4.0)
    loss_ref.backward()
    autograd_grad = logits_ref.grad

    max_diff = torch.max(torch.abs(analytic_grad - autograd_grad)).item()
    assert max_diff < 1e-12


def test_fused_rewarding_doubt_loss_gradient():
    torch.manual_seed(123)
    logits = torch.randn(8, 3, requires_grad=True, dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    loss, probs = fused_rewarding_doubt_loss(logits, labels)
    loss.backward()
    analytic_grad = logits.grad.clone()

    # Compare against standard autograd
    logits_ref = logits.detach().clone().requires_grad_(True)
    probs_ref = F.softmax(logits_ref, dim=-1)
    m = torch.argmax(probs_ref, dim=-1)
    is_correct = (m == labels)
    p_y = probs_ref.gather(1, labels.unsqueeze(1)).squeeze(1)
    p_m = probs_ref.gather(1, m.unsqueeze(1)).squeeze(1)
    loss_ref = torch.where(is_correct, -torch.log(p_y), -torch.log(1.0 - p_m)).mean()
    loss_ref.backward()
    autograd_grad = logits_ref.grad

    max_diff = torch.max(torch.abs(analytic_grad - autograd_grad)).item()
    assert max_diff < 1e-12


def test_fused_rlcd_loss_composite():
    torch.manual_seed(999)
    logits = torch.randn(5, 4, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3, 0])

    loss, probs, rewards = fused_rlcd_loss(logits, labels)
    loss.backward()

    assert not torch.isnan(loss)
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert rewards.shape == (5,)
