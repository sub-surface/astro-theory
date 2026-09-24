"""
=============================================================================
Evidential AstroJev: Dirichlet Subjective Logic on Krasnoselskii-Mann Manifolds
=============================================================================
An advanced epistemic decision model solving the extraction dilemma and
disentangling epistemic uncertainty (model ignorance / OOD) from aleatoric
uncertainty (astrophysical photon noise / measurement error).

Mathematical Foundations:
  1. Krasnoselskii-Mann Fixed-Point Equilibrium:
       h_{k+1} = (1 - gamma_k) h_k + gamma_k T_theta(h_k, x)
       Banach contractive mapping reaching fixed point h*.
  2. Dirichlet Subjective Logic (Evidential Deep Learning):
       e_k = softplus(z_k) >= 0 (Class Evidence)
       alpha_k = e_k + 1 (Dirichlet Concentration Parameter)
       S = sum_k alpha_k (Total Dirichlet Strength)
       p_bar_k = alpha_k / S (Expected Posterior Probability)
  3. Strict Epistemic-Aleatoric Decomposition:
       u_epi = K / S in (0, 1] (Epistemic Vacuity / Ignorance)
       Noul = 1 - u_epi = (S - K) / S in [0, 1) (Axiomatic Epistemic Credence)
       u_ale = - sum_k p_bar_k log p_bar_k (Aleatoric Data Entropy)
  4. Expected Brier Loss under Dirichlet Posterior:
       L_Brier = sum_k (p_bar_k - y_k)^2 + sum_k [ p_bar_k(1 - p_bar_k) / (S + 1) ]
       The second term directly penalizes having low evidence on confident choices.
  5. Information-Geometric KL Prior Regularization:
       KL(Dir(alpha_tilde) || Dir(1)) pushes misleading evidence to zero.
  6. Equilibrium Tension Damping:
       Evidence is dynamically dampened when recurrent contraction norm is large.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from celestrium.astrojev import (
    ContinuousFourierEncoder,
    KMContractiveBlock,
    NUM_FEATURES,
    NUM_CLASSES,
    CLASSES,
)
from celestrium.kernels import fused_crelu


# --------------------------------------------------------------------------- #
# 1. Evidential Dirichlet Loss Function
# --------------------------------------------------------------------------- #
def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL divergence KL(Dir(alpha) || Dir(1)) to uniform Dirichlet prior."""
    beta = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_b_alpha = (
        torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
        - torch.lgamma(sum_alpha)
    )
    ln_b_beta = (
        torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)
        - torch.lgamma(sum_beta)
    )

    dg_alpha = torch.digamma(alpha)
    dg_sum_alpha = torch.digamma(sum_alpha)

    kl = (
        torch.sum((alpha - beta) * (dg_alpha - dg_sum_alpha), dim=-1, keepdim=True)
        + ln_b_beta
        - ln_b_alpha
    )
    return kl.squeeze(-1)


def evidential_brier_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    kl_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes expected Brier loss under Dirichlet distribution + KL shrinkage."""
    S = torch.sum(alpha, dim=-1, keepdim=True)  # (B, 1)
    probs = alpha / S  # (B, K)

    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # 1. Mean prediction Brier error: sum (p_bar - y)^2
    err = torch.sum((probs - one_hot) ** 2, dim=-1)

    # 2. Dirichlet posterior variance: sum [ p_bar * (1 - p_bar) / (S + 1) ]
    var = torch.sum(probs * (1.0 - probs) / (S + 1.0), dim=-1)

    brier_loss = torch.mean(err + var)

    # 3. KL divergence penalty on misleading evidence:
    # Scale down evidence for incorrect classes
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    kl_penalty = torch.mean(kl_divergence_dirichlet(alpha_tilde, num_classes))

    total_loss = brier_loss + kl_weight * kl_penalty
    return total_loss, probs, brier_loss


def evidential_brier_carl_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    carl_weight: float = 0.2,
    kl_weight: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Computes Dirichlet Brier-CARL Loss (MIT ICLR 2026 RLCR + USC/AWS ACL 2026 CARL).

    When a prediction is incorrect (argmax p_bar != y), CARL forces the model towards
    the uniform barycenter centroid (1/K). In Dirichlet space, this pulls alpha -> 1
    (zero evidence), driving Dirichlet strength S -> K and epistemic vacuity u_epi -> 1.0!

    Loss = L_Brier + kl_weight * KL(alpha_tilde || 1) + carl_weight * 1_{wrong} * KL(alpha || 1)
    """
    S = torch.sum(alpha, dim=-1, keepdim=True)
    probs = alpha / S

    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # 1. Expected Brier loss (prediction error + Dirichlet posterior variance)
    err = torch.sum((probs - one_hot) ** 2, dim=-1)
    var = torch.sum(probs * (1.0 - probs) / (S + 1.0), dim=-1)
    brier_loss = torch.mean(err + var)

    # 2. Standard misleading evidence KL penalty
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    kl_misleading = torch.mean(kl_divergence_dirichlet(alpha_tilde, num_classes))

    # 3. CARL Simplex Centroid Regularization on incorrect classifications
    top1 = torch.argmax(probs, dim=-1)
    is_wrong = (top1 != labels).float()  # (B,)
    kl_to_uniform_prior = kl_divergence_dirichlet(alpha, num_classes)  # (B,)
    carl_penalty = torch.mean(is_wrong * kl_to_uniform_prior)

    total_loss = brier_loss + kl_weight * kl_misleading + carl_weight * carl_penalty

    metrics = {
        "brier_loss": float(brier_loss.item()),
        "kl_misleading": float(kl_misleading.item()),
        "carl_penalty": float(carl_penalty.item()),
        "wrong_fraction": float(is_wrong.mean().item()),
    }
    return total_loss, probs, metrics


# --------------------------------------------------------------------------- #

# 2. Evidential AstroJev Neural Architecture
# --------------------------------------------------------------------------- #
class EvidentialAstroJev(nn.Module):
    """AstroJev with Dirichlet Subjective Logic & Krasnoselskii-Mann Equilibrium."""

    def __init__(
        self,
        in_features: int = NUM_FEATURES,
        d_model: int = 64,
        num_classes: int = NUM_CLASSES,
        n_iter: int = 5,
        damping_beta: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.n_iter = n_iter
        self.damping_beta = damping_beta

        self.encoder = ContinuousFourierEncoder(in_features, d_model)
        self.km_block = KMContractiveBlock(d_model)

        # CReLU sparse accumulator
        self.accum_proj = nn.Linear(d_model, 128)
        self.accum_norm = nn.LayerNorm(128)

        # Evidential Head (predicts non-negative evidence e_k >= 0)
        self.evidence_head = nn.Linear(128, num_classes)

        # Temperature scaling on raw evidence projection
        self.log_evidence_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        heteroscedastic_noise: bool = False,
        noise_scale: float = 1.0,
    ) -> Dict[str, Any]:
        x_in = x
        if heteroscedastic_noise and self.training:
            pm_err = x[:, 6:7].clamp(min=0.01, max=5.0)
            snr = x[:, 9:10].clamp(min=1.0, max=100.0)
            phot_err = 1.0 / snr
            noise_sigma = torch.cat([
                phot_err, phot_err, phot_err, phot_err, phot_err,
                pm_err, torch.zeros_like(pm_err),
                torch.zeros_like(pm_err), torch.zeros_like(pm_err),
                torch.zeros_like(pm_err),
            ], dim=-1)
            eps = torch.randn_like(x) * noise_sigma * noise_scale
            x_in = x + eps

        x_ctx = self.encoder(x_in)
        h = x_ctx
        h_prev = h
        residuals = []

        # Krasnoselskii-Mann contractive equilibrium loop
        for k in range(self.n_iter):
            gamma_k = 1.0 / (1.0 + 0.2 * (k + 1))
            t_h = self.km_block(h, x_ctx)
            step_norm = torch.norm(t_h - h, p=2, dim=-1)
            residuals.append(step_norm.mean().item())
            h_prev = h
            h = (1.0 - gamma_k) * h + gamma_k * t_h

        # Final contraction stability residual ||h_final - h_prev||_2
        delta_eq = torch.norm(h - h_prev, p=2, dim=-1)

        # CReLU sparse accumulator [0, 1]
        accum = self.accum_norm(self.accum_proj(h))
        crelu_sparse, sparsity = fused_crelu(accum)

        # Compute raw evidence: e_k = softplus(z_k)
        scale = torch.exp(self.log_evidence_scale)
        raw_logits = self.evidence_head(crelu_sparse) * scale
        raw_evidence = F.softplus(raw_logits)

        # Equilibrium tension damping:
        # If recurrent deliberation did not converge, damp evidence
        tension = torch.tanh(self.damping_beta * delta_eq).unsqueeze(-1)
        evidence = raw_evidence * (1.0 - tension)

        # Dirichlet Concentration parameters: alpha_k = evidence_k + 1
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)  # Dirichlet strength
        probs = alpha / S

        # Exact Epistemic & Aleatoric Decomposition
        # Epistemic vacuity / doubt: u_epi = K / S in (0, 1]
        u_epi = (float(self.num_classes) / S).squeeze(-1).clamp(max=1.0)
        # Axiomatic Epistemic Noul: Noul = 1 - u_epi in [0, 1)
        noul = (1.0 - u_epi).clamp(min=0.0, max=1.0)

        # Aleatoric uncertainty: Entropy of expected categorical
        log_probs = torch.log(probs.clamp(min=1e-8))
        u_ale = -torch.sum(probs * log_probs, dim=-1) / math.log(self.num_classes)

        return {
            "alpha": alpha,
            "evidence": evidence,
            "probs": probs,
            "noul": noul,
            "u_epi": u_epi,
            "u_ale": u_ale,
            "delta_eq": delta_eq,
            "residuals": residuals,
            "sparsity": sparsity,
            "latent": crelu_sparse,
        }
