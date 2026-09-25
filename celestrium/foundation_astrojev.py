"""
=============================================================================
Foundation AstroJev: Heteroscedastic Error-Calibrated 12-Class Decision Model
=============================================================================
Extends AstroJev with:
1. Comprehensive 12-Class Astronomical Phenomena Taxonomy (Extragalactic, Galactic, Transients).
2. Heteroscedastic Observational Error Conditioning: pairs physical observable means
   with measurement uncertainties (photometric errors, astrometric covariances, RUWE).
3. In-flight Differentiable Error Jittering: samples x ~ N(mu, sigma^2) during training
   to instill physical noise invariance and SNR-stratified calibration.
4. Missing-Band Presence Masking: handles multi-survey dropout gracefully.
5. Dirichlet Subjective Evidential Head: outputs Dirichlet concentration alpha,
   epistemic vacuity u_epi, aleatoric entropy u_ale, and analytical Dirichlet BALD mutual info.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import fused_crelu

# --------------------------------------------------------------------------- #
# 1. 12-Class Astronomical Phenomena Taxonomy
# --------------------------------------------------------------------------- #
PHENOMENA_CLASSES = [
    # Extragalactic Cosmological Tracers
    "High_z_Quasar_AGN",        # 0: z > 2.0, point optical, red mid-IR W1-W2 > 0.8, zero PM/parallax
    "Low_z_Seyfert_AGN",         # 1: z < 1.0, host galaxy dilution, intermediate colors
    "Luminous_Red_Galaxy",       # 2: LRG, 4000Å break, BAO standard ruler, extended optical profile
    "Emission_Line_Galaxy",      # 3: ELG, starburst [O II], flat SED, low IR excess
    "Blazar_Relativistic_Jet",   # 4: Flat-spectrum radio/gamma-ray loud, stochastic optical variance
    # Galactic Stellar Populations
    "Main_Sequence_Dwarf",       # 5: Dominant stellar foreground, non-zero proper motions
    "Red_Giant_Branch",          # 6: High luminosity, asymptotic branch, IR excess, low PM
    "White_Dwarf",               # 7: High surface gravity, blue optical, high reduced PM H_G
    "Subdwarf_Halo_Star",        # 8: High-velocity halo kinematics, low metallicity offsets
    "Brown_Dwarf_LTY",           # 9: Ultracool sub-stellar, negligible optical flux, extreme W1-W2
    # Time-Domain & Multi-Messenger Transients
    "Explosive_Transient",       # 10: Type Ia SNe, CC-SNe, FBOTs, kilonova candidates (rapid rise dm/dt)
    "Variable_Star_Flarer",      # 11: Cataclysmic variables, RR Lyrae, M-dwarf coronal flares
]
PHENOMENA_CLASS_TO_IDX = {c: i for i, c in enumerate(PHENOMENA_CLASSES)}
NUM_PHENOMENA_CLASSES = len(PHENOMENA_CLASSES)

NUM_OBSERVABLES = 10     # Mean feature values mu
NUM_UNCERTAINTIES = 6    # Measured standard errors sigma
NUM_MASKS = 6            # Presence flags m (optical, infrared, astrometry, etc.)
TOTAL_INPUT_DIM = NUM_OBSERVABLES + NUM_UNCERTAINTIES + NUM_MASKS  # 22-D input


# --------------------------------------------------------------------------- #
# 2. Continuous Fourier Error Encoder
# --------------------------------------------------------------------------- #
class HeteroscedasticFourierEncoder(nn.Module):
    """Encodes physical observables, log-uncertainties, and presence masks into latent space."""
    def __init__(self, in_features: int = TOTAL_INPUT_DIM, d_model: int = 128):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(in_features, d_model)
        self.fourier_weight = nn.Parameter(torch.randn(in_features, d_model // 2) * 0.5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h_dir = self.linear(z)
        proj = 2.0 * math.pi * (z @ self.fourier_weight)
        h_fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.norm(h_dir + h_fourier)


# --------------------------------------------------------------------------- #
# 3. Krasnoselskii-Mann Contractive Loop Block
# --------------------------------------------------------------------------- #
class HeteroscedasticKMBlock(nn.Module):
    """Weight-tied contractive recurrence operator T_theta with Squeeze-and-Excitation gating."""
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.se_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, x_ctx: torch.Tensor) -> torch.Tensor:
        h_in = h + x_ctx
        gate = self.se_gate(h_in)
        update = self.mlp(h_in)
        return h + gate * (update - h)


# --------------------------------------------------------------------------- #
# 4. Foundation AstroJev Model
# --------------------------------------------------------------------------- #
class FoundationAstroJev(nn.Module):
    """Heteroscedastic Error-Calibrated Evidential Foundation Model for 12 Astronomical Phenomena."""
    def __init__(
        self,
        num_classes: int = NUM_PHENOMENA_CLASSES,
        d_model: int = 128,
        n_iter: int = 5,
        gamma: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_iter = n_iter
        self.gamma = gamma

        # Encoder: 22 inputs -> d_model
        self.encoder = HeteroscedasticFourierEncoder(in_features=TOTAL_INPUT_DIM, d_model=d_model)

        # Recurrent Contractive Deliberation Operator
        self.recurrence = HeteroscedasticKMBlock(d_model=d_model)
        self.norm = nn.LayerNorm(d_model)

        # Sparse CReLU Accumulator
        self.accum_proj = nn.Linear(d_model, 128)
        self.accum_norm = nn.LayerNorm(128)

        # Calibrated Evidential Decision Heads
        self.evidence_head = nn.Linear(128, num_classes)
        self.noul_head = nn.Linear(128, 1)      # Cosmological tracer probability
        self.score_head = nn.Linear(128, 5)     # Purity score (1-5)

        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_jitter: bool = True,
        jitter_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Forward pass with heteroscedastic error jittering and Dirichlet subjective logic."""
        batch_size = mu.size(0)
        device = mu.device

        # Defensive nan-to-num sanitization
        mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
        sigma = torch.nan_to_num(sigma, nan=1.0, posinf=50.0, neginf=1e-5).clamp(min=1e-5, max=50.0)

        # Default presence mask if not provided
        if mask is None:
            mask = torch.ones((batch_size, NUM_MASKS), dtype=mu.dtype, device=device)
        else:
            mask = torch.nan_to_num(mask, nan=0.0).clamp(min=0.0, max=1.0)

        # Physical In-Flight Error Jittering (differentiable augmentation)
        if apply_jitter and self.training:
            # sigma = [sig_g, sig_bprp, sig_w1, sig_w1w2, sig_pm, ruwe] maps onto
            # mu cols [g, bp_rp, w1, w1_w2, pm] = [0, 1, 3, 4, 5]; RUWE is a fit-quality
            # statistic, not an error bar, so it jitters nothing.
            jitter_sigma = torch.zeros_like(mu)
            jitter_sigma[:, [0, 1, 3, 4, 5]] = sigma[:, :5].clamp(min=1e-5, max=5.0)
            eps = torch.randn_like(mu)
            x_in = mu + eps * jitter_sigma * jitter_scale
        else:
            x_in = mu

        # Combine observables, log-uncertainties, and mask into composite packet z
        log_sigma = torch.log(sigma.clamp(min=1e-5))
        z = torch.cat([x_in, log_sigma, mask], dim=-1)
        z = torch.nan_to_num(z, nan=0.0)

        # Encode input packet
        x_ctx = self.encoder(z)

        # Krasnoselskii-Mann Contractive Deliberation Loop
        h = x_ctx
        residuals = []
        for _ in range(self.n_iter):
            h_prev = h
            t_h = self.recurrence(h, x_ctx)
            h = (1.0 - self.gamma) * h + self.gamma * t_h
            h = self.norm(h)
            with torch.no_grad():
                res = torch.norm(h - h_prev, p=2, dim=-1).mean().item()
                residuals.append(res)

        delta_eq = torch.norm(h - h_prev, p=2, dim=-1)

        # Sparse CReLU Accumulator
        z_accum = self.accum_norm(self.accum_proj(h))
        z_sparse, sparsity = fused_crelu(z_accum)

        # Evidential Dirichlet Readout
        evidence = F.softplus(self.evidence_head(z_sparse))
        alpha = evidence + 1.0
        dirichlet_strength = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / dirichlet_strength

        # Epistemic vs Aleatoric Uncertainty
        u_epi = (self.num_classes / dirichlet_strength).squeeze(-1).clamp(0.0, 1.0)
        u_ale = -torch.sum(probs * torch.log(probs.clamp(min=1e-7)), dim=-1)

        # Axiomatic Noul (Extragalactic Credence)
        noul_logit = self.noul_head(z_sparse) / torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
        noul = torch.sigmoid(noul_logit).squeeze(-1)

        return {
            "probs": probs,
            "alpha": alpha,
            "evidence": evidence,
            "dirichlet_strength": dirichlet_strength.squeeze(-1),
            "u_epi": u_epi,
            "u_ale": u_ale,
            "noul": noul,
            "delta_eq": delta_eq,
            "sparsity": sparsity,
            "residuals": residuals,
        }


# --------------------------------------------------------------------------- #
# 5. Evidential Dirichlet Loss with Heteroscedastic Error Regularization
# --------------------------------------------------------------------------- #
def kl_divergence_dirichlet(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL divergence KL(Dir(alpha) || Dir(1)) to uniform Dirichlet prior."""
    beta = torch.ones((1, num_classes), dtype=alpha.dtype, device=alpha.device)
    sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
    sum_beta = torch.sum(beta, dim=-1, keepdim=True)

    ln_b_alpha = torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True) - torch.lgamma(sum_alpha)
    ln_b_beta = torch.sum(torch.lgamma(beta), dim=-1, keepdim=True) - torch.lgamma(sum_beta)

    dg_alpha = torch.digamma(alpha)
    dg_sum_alpha = torch.digamma(sum_alpha)

    kl = torch.sum((alpha - beta) * (dg_alpha - dg_sum_alpha), dim=-1, keepdim=True) + ln_b_beta - ln_b_alpha
    return kl.squeeze(-1)


def heteroscedastic_evidential_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = NUM_PHENOMENA_CLASSES,
    kl_lambda: float = 0.05,
) -> torch.Tensor:
    """Expected Brier Loss under Dirichlet posterior with information-geometric KL prior."""
    dirichlet_strength = torch.sum(alpha, dim=-1, keepdim=True)
    probs = alpha / dirichlet_strength
    one_hot = F.one_hot(targets, num_classes=num_classes).float()

    # Expected Brier Score: E_{p ~ Dir(alpha)} [ sum_k (p_k - y_k)^2 ]
    # = sum_k (p_bar_k - y_k)^2 + sum_k [ p_bar_k (1 - p_bar_k) / (S + 1) ]
    sq_err = torch.sum((probs - one_hot) ** 2, dim=-1, keepdim=True)
    var_penalty = torch.sum(probs * (1.0 - probs) / (dirichlet_strength + 1.0), dim=-1, keepdim=True)
    brier_loss = sq_err + var_penalty

    # KL Prior on Misleading Evidence
    alpha_tilde = one_hot + (1.0 - one_hot) * alpha
    kl = kl_divergence_dirichlet(alpha_tilde, num_classes)

    total_loss = brier_loss.squeeze(-1) + kl_lambda * kl
    return total_loss.mean()
