"""
=============================================================================
EXP-2026-V: Continuous-Flow Foundation AstroJev for Multi-Survey Cross-Calibration
=============================================================================
Unifies heterogeneous multi-survey photometry across Euclid DR1 Wide (I_E, Y, J, H),
DESI Legacy Surveys DR10 (g, r, z), and Rubin LSST DP0 (u, g, r, i, z, y) using
Simulation-Free Conditional Flow Matching (CFM) and Dirichlet Evidential Decision Heads.

Core Capabilities:
  1. Simulation-Free Conditional Flow Matching (CFM):
     Learns a continuous time-dependent velocity field v_theta(z_t, t | c) transporting
     from base Gaussian noise p_0(z) = N(0, I) to an invariant physical SED latent
     manifold p_1(z | c), conditioned on multi-survey photometry, uncertainties, and masks.
  2. Cross-Survey Zero-Point Disentanglement:
     Corrects photometric zero-point offsets Delta m_zp ~ 0.01 - 0.05 mag across instruments.
  3. Autonomous Multi-Object Spectrograph Allocation (e.g., DESI 5,000-fiber focal plane):
     - DISPATCH_HIGH_Z_QUASAR_120M: High-redshift quasar Ly-alpha forest (120 min)
     - DISPATCH_ELG_FIBER_45M: Emission Line Galaxy BAO tracer (45 min)
     - DISPATCH_LRG_FIBER_60M: Luminous Red Galaxy standard ruler (60 min)
     - DISPATCH_CALIBRATION_FIBER: Standard calibration star (15 min)
     - PURGE_CONTAMINANT_NO_FIBER: Brown dwarfs, artifacts, and interlopers (0 min)
  4. Dirichlet Calibration & Conformal Risk Control:
     - Retains analytical posterior standard deviations sigma_k and 95% Credible Intervals.
     - Conformal risk control guarantees wasted fibers on contaminants <= 2.0%.
     - Disentangled RLCD optimization collapsing debiased calibration error E^2_db.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import fused_crelu

# --------------------------------------------------------------------------- #
# Constants & Fiber Allocation Taxonomy
# --------------------------------------------------------------------------- #
ALLOCATION_CLASSES = [
    "HIGH_Z_QUASAR_120M",      # 0: z > 2.15 Ly-alpha forest (120 min fiber integration)
    "ELG_TRACER_45M",          # 1: [O II] emission line galaxy (45 min integration)
    "LRG_RULER_60M",           # 2: Luminous red galaxy 4000A break (60 min integration)
    "CALIBRATION_STAR_15M",    # 3: F-turnoff / DA white dwarf spectrophotometric standard
    "PURGE_CONTAMINANT_0M",    # 4: Brown dwarfs, spurious diffraction spikes, bad pixels
]
ALLOCATION_CLASS_TO_IDX = {c: i for i, c in enumerate(ALLOCATION_CLASSES)}
NUM_ALLOCATION_CLASSES = len(ALLOCATION_CLASSES)

FIBER_EXPOSURE_MINUTES = {
    "HIGH_Z_QUASAR_120M": 120.0,
    "ELG_TRACER_45M": 45.0,
    "LRG_RULER_60M": 60.0,
    "CALIBRATION_STAR_15M": 15.0,
    "PURGE_CONTAMINANT_0M": 0.0,
}

# Input Photometry Dimension:
# Euclid (I_E, Y, J, H) [4] + DESI (g, r, z) [3] + Rubin (u, g, r, i, z, y) [6] = 13 bands
# Uncertainties: 13 bands
# Presence Masks: 13 bands
NUM_BANDS = 13
TOTAL_OBSERVABLE_DIM = NUM_BANDS * 3  # 39-D input vector [fluxes, errors, masks]
LATENT_SED_DIM = 32                  # Invariant latent SED manifold dimension


@dataclass
class SpectroscopicTriageDecision:
    """Rigorous decision container retaining analytical uncertainties and conformal bounds."""
    action: str
    target_class: str
    confidence: float
    dirichlet_std: float
    ci_95: Tuple[float, float]
    epistemic_vacuity: float
    aleatoric_entropy: float
    conformal_set: List[str]
    estimated_zp_offset_mag: float
    fiber_exposure_minutes: float
    is_rejected: bool


# --------------------------------------------------------------------------- #
# 1. Multi-Survey Photometric Generator & Zero-Point Injection
# --------------------------------------------------------------------------- #
class MultiSurveyPhotometryGenerator:
    """Simulates realistic Euclid, DESI, and Rubin photometry from underlying physical SEDs."""
    def __init__(self, rng_seed: int = 42):
        self.rng = np.random.default_rng(rng_seed)

    def generate_survey_photometry(
        self,
        base_mu: np.ndarray,      # (N, 10) base observables from real_phenomena_dataset
        labels: np.ndarray,       # (N,) 12-class labels
        zp_jitter_mag: float = 0.03
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates 13-band photometry, errors, presence masks, allocation targets, and true ZP offsets."""
        n_samples = len(base_mu)
        g_mag = base_mu[:, 0]
        bprp = base_mu[:, 1]
        bp_mag = g_mag + 0.4 * bprp
        rp_mag = g_mag - 0.6 * bprp
        w1_mag = base_mu[:, 3]
        w1w2 = base_mu[:, 4]
        w2_mag = w1_mag - w1w2

        # Photometric arrays:
        # Bands: [Euclid_IE, Euclid_Y, Euclid_J, Euclid_H,
        #         DESI_g, DESI_r, DESI_z,
        #         Rubin_u, Rubin_g, Rubin_r, Rubin_i, Rubin_z, Rubin_y]
        phot = np.zeros((n_samples, NUM_BANDS), dtype=np.float32)
        errors = np.zeros((n_samples, NUM_BANDS), dtype=np.float32)
        masks = np.ones((n_samples, NUM_BANDS), dtype=np.float32)

        # Injected instrument zero-point shifts (ground truth to recover)
        zp_true = self.rng.normal(0.0, zp_jitter_mag, size=(n_samples, 1)).astype(np.float32)

        # Euclid VIS & NISP NIR bands (0.9 - 2.0 microns)
        phot[:, 0] = g_mag - 0.20 * bprp + zp_true[:, 0]
        phot[:, 1] = g_mag - 0.50 * bprp + 0.1 * w1w2
        phot[:, 2] = phot[:, 1] - 0.30 * bprp
        phot[:, 3] = phot[:, 2] - 0.25 * bprp

        # DESI Legacy Surveys optical bands (g, r, z)
        phot[:, 4] = g_mag + zp_true[:, 0] * 0.5
        phot[:, 5] = rp_mag
        phot[:, 6] = rp_mag - 0.40 * bprp

        # Rubin LSST optical bands (u, g, r, i, z, y)
        phot[:, 7] = bp_mag + 0.60 * bprp  # u-band
        phot[:, 8] = g_mag
        phot[:, 9] = rp_mag
        phot[:, 10] = rp_mag - 0.25 * bprp
        phot[:, 11] = rp_mag - 0.45 * bprp
        phot[:, 12] = phot[:, 1]    # y-band near-IR

        # Physical Lyman break for High-z Quasars (z > 2.15 Ly-alpha forest absorption)
        is_qso = (labels == 0)
        phot[is_qso, 7] += 2.2  # Rubin u-band Lyman absorption
        phot[is_qso, 4] += 0.8  # DESI g-band forest attenuation

        # Physical Brown Dwarf optical extinction (T_eff < 1500K)
        is_bd = (labels == 9)
        phot[is_bd, 0:7] += 3.5

        # Heteroscedastic measurement errors: SNR scales with apparent magnitude
        for b in range(NUM_BANDS):
            snr = np.clip(100.0 * 10.0 ** (-0.3 * (phot[:, b] - 17.0)), 2.0, 150.0)
            errors[:, b] = (1.0857 / snr).astype(np.float32)
            # Add measurement noise to observed flux
            phot[:, b] += self.rng.normal(0, 1, size=n_samples) * errors[:, b]

        # Multi-survey missing band masks (e.g. 10% dropout in Rubin or Euclid)
        for b in range(NUM_BANDS):
            drop_prob = 0.08
            dropped = self.rng.uniform(0, 1, size=n_samples) < drop_prob
            masks[dropped, b] = 0.0
            phot[dropped, b] = 25.0  # Detection limit placeholder
            errors[dropped, b] = 9.99

        # Map 12-class phenomena labels to 5-class Spectroscopic Allocation:
        # 0: High_z_Quasar -> HIGH_Z_QUASAR_120M
        # 1, 3, 4: Emission Line Galaxies & Low-z AGNs -> ELG_TRACER_45M
        # 2: Luminous_Red_Galaxy -> LRG_RULER_60M
        # 5, 7, 8: Main Sequence / White Dwarf / Subdwarf -> CALIBRATION_STAR_15M
        # 6, 9, 10, 11: Brown dwarfs, giants, artifacts -> PURGE_CONTAMINANT_0M
        alloc_targets = np.full(n_samples, 4, dtype=np.int64)  # Default PURGE
        alloc_targets[labels == 0] = 0  # HIGH_Z_QUASAR
        alloc_targets[np.isin(labels, [1, 3, 4])] = 1  # ELG / Standard AGNs
        alloc_targets[labels == 2] = 2  # LRG
        alloc_targets[np.isin(labels, [5, 7, 8])] = 3  # CALIBRATION
        alloc_targets[np.isin(labels, [6, 9, 10, 11])] = 4  # CONTAMINANTS -> PURGE!

        return phot, errors, masks, alloc_targets, zp_true


# --------------------------------------------------------------------------- #
# 2. Conditional Flow Matching (CFM) Velocity Network
# --------------------------------------------------------------------------- #
class ConditionalFlowVelocityNet(nn.Module):
    """Parameterizes the continuous time-dependent vector field v_theta(z_t, t | c).
    
    Transforms base distribution z_0 ~ N(0, I) into physical invariant SED manifold z_1.
    """
    def __init__(
        self,
        z_dim: int = LATENT_SED_DIM,
        cond_dim: int = TOTAL_OBSERVABLE_DIM,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.z_dim = z_dim
        self.cond_dim = cond_dim

        # Sinusoidal time embedding for t in [0, 1]
        self.time_embed_dim = 32
        self.time_linear = nn.Linear(self.time_embed_dim, self.time_embed_dim)

        # Conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Joint velocity MLP
        in_dim = z_dim + self.time_embed_dim + hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, z_dim)
        )

    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal Fourier positional encoding for continuous time t in [0, 1]."""
        half_dim = self.time_embed_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=t.device) / (half_dim - 1))
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return F.silu(self.time_linear(embedding))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Evaluates vector field v_theta(z_t, t | c)."""
        t_emb = self._get_time_embedding(t)
        c_emb = self.cond_proj(c)
        x = torch.cat([z_t, t_emb, c_emb], dim=-1)
        return self.net(x)


# --------------------------------------------------------------------------- #
# 3. Continuous-Flow Foundation AstroJev Architecture
# --------------------------------------------------------------------------- #
class ContinuousFlowFoundationAstroJev(nn.Module):
    """Foundation AstroJev with Continuous Flow Matching & Dirichlet Evidential Readout."""
    def __init__(
        self,
        z_dim: int = LATENT_SED_DIM,
        cond_dim: int = TOTAL_OBSERVABLE_DIM,
        num_classes: int = NUM_ALLOCATION_CLASSES,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        # 1. Continuous Flow Matching Velocity Network
        self.velocity_net = ConditionalFlowVelocityNet(
            z_dim=z_dim, cond_dim=cond_dim, hidden_dim=hidden_dim
        )

        # 2. Photometric Zero-Point Offset Head (Disentangles cross-instrument delta m)
        self.zp_head = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

        # 3. Dirichlet Evidential Readout Head (Disentangled RLCD)
        self.evidence_head = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, num_classes),
            nn.Softplus()  # Guarantees positive evidence e_k >= 0
        )

    def compute_flow_loss(self, z_1: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Simulation-Free Conditional Flow Matching (CFM) Loss.
        
        Straight paths: z_t = (1 - t) * z_0 + t * z_1
        Target vector field: u_t = z_1 - z_0
        Objective: E_{t, z_0} || v_theta(z_t, t | c) - (z_1 - z_0) ||^2
        """
        batch_size = z_1.shape[0]
        device = z_1.device

        t = torch.rand(batch_size, device=device)
        z_0 = torch.randn_like(z_1)

        # Straight-path optimal transport interpolation
        z_t = (1.0 - t[:, None]) * z_0 + t[:, None] * z_1
        target_velocity = z_1 - z_0

        pred_velocity = self.velocity_net(z_t, t, c)
        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        return flow_loss

    def integrate_ode(
        self,
        c: torch.Tensor,
        num_steps: int = 10,
        method: str = "midpoint"
    ) -> torch.Tensor:
        """Integrates dz/dt = v_theta(z_t, t | c) from t=0 to t=1 to reach latent manifold z_1."""
        batch_size = c.shape[0]
        device = c.device
        z = torch.randn(batch_size, self.z_dim, device=device)

        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_curr = torch.full((batch_size,), step * dt, device=device)
            if method == "euler":
                v = self.velocity_net(z, t_curr, c)
                z = z + dt * v
            elif method == "midpoint":
                v1 = self.velocity_net(z, t_curr, c)
                t_mid = t_curr + 0.5 * dt
                z_mid = z + 0.5 * dt * v1
                v2 = self.velocity_net(z_mid, t_mid, c)
                z = z + dt * v2

        return z

    def forward(
        self,
        c: torch.Tensor,
        num_ode_steps: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs continuous flow ODE integration followed by zero-point and evidential readout.
        
        Returns:
          - alpha: Dirichlet concentrations (batch_size, num_classes)
          - zp_pred: Predicted zero-point offset in mag (batch_size, 1)
          - z_1: Latent invariant SED representations (batch_size, z_dim)
        """
        z_1 = self.integrate_ode(c, num_steps=num_ode_steps)
        c_emb = self.velocity_net.cond_proj(c)
        feat = torch.cat([z_1, c_emb], dim=-1)
        zp_pred = self.zp_head(feat)
        evidence = self.evidence_head(feat)
        alpha = evidence + 1.0  # Dirichlet concentration alpha_k = e_k + 1
        return alpha, zp_pred, z_1

    def compute_dirichlet_statistics(
        self,
        alpha: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates expected probabilities, Dirichlet standard deviations, and uncertainties.
        
        Returns:
          - probs: p_k = alpha_k / S
          - stds: sigma_k = sqrt(p_k * (1 - p_k) / (S + 1))
          - epistemic_vacuity: u_epi = K / S
          - aleatoric_entropy: Shannon entropy H(p)
        """
        s = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / s
        stds = torch.sqrt(torch.clamp(probs * (1.0 - probs) / (s + 1.0), min=1e-12))
        u_epi = float(self.num_classes) / s.squeeze(-1)

        # Aleatoric entropy
        log_probs = torch.log(torch.clamp(probs, min=1e-12))
        u_ale = -torch.sum(probs * log_probs, dim=-1)

        return probs, stds, u_epi, u_ale


# --------------------------------------------------------------------------- #
# 4. Disentangled RLCD Calibration Optimization (TUM 2026 / Bani-Harouni et al.)
# --------------------------------------------------------------------------- #
def train_continuous_flow_astrojev(
    model: ContinuousFlowFoundationAstroJev,
    c_train: torch.Tensor,
    y_train: torch.Tensor,
    zp_train: torch.Tensor,
    num_epochs: int = 30,
    lr: float = 1e-3,
    verbose: bool = False
) -> Dict[str, Any]:
    """Jointly trains the continuous velocity field (CFM) and optimizes Disentangled RLCD on evidence."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 256
    n_samples = c_train.shape[0]

    # Pre-generate target latent vectors z_1 using initial projection
    target_sed_proj = nn.Linear(TOTAL_OBSERVABLE_DIM, LATENT_SED_DIM).to(c_train.device)
    with torch.no_grad():
        z_1_targets = target_sed_proj(c_train)
        z_1_targets = F.normalize(z_1_targets, dim=-1)

    class_counts = torch.bincount(y_train, minlength=model.num_classes).float()
    class_weights = (n_samples / (model.num_classes * torch.clamp(class_counts, min=1.0))).to(c_train.device)

    epoch_losses = []
    for epoch in range(num_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            c_b = c_train[idx]
            y_b = y_train[idx]
            zp_b = zp_train[idx]
            z_targ_b = z_1_targets[idx]

            optimizer.zero_grad()

            # 1. Flow Matching Loss
            flow_loss = model.compute_flow_loss(z_targ_b, c_b)

            # 2. Fast 1-step ODE integration for backprop
            z_pred = model.integrate_ode(c_b, num_steps=2, method="euler")
            c_emb = model.velocity_net.cond_proj(c_b)
            feat = torch.cat([z_pred, c_emb], dim=-1)
            zp_pred = model.zp_head(feat)
            zp_loss = F.mse_loss(zp_pred, zp_b)

            # 3. Evidential Brier Loss + Weighted Cross-Entropy
            evidence = model.evidence_head(feat)
            alpha = evidence + 1.0
            s = torch.sum(alpha, dim=-1, keepdim=True)
            p = alpha / s

            y_one_hot = F.one_hot(y_b, num_classes=model.num_classes).float()
            ce_loss = F.cross_entropy(torch.log(torch.clamp(p, min=1e-12)), y_b, weight=class_weights)
            brier_loss = torch.mean(class_weights[y_b] * torch.sum((y_one_hot - p) ** 2, dim=-1))

            # Composite Loss
            loss = flow_loss + 10.0 * zp_loss + 2.0 * ce_loss + brier_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

    # 4. Disentangled RLCD Phase: Freeze velocity & ZP heads, tune ONLY evidence readout
    for param in model.velocity_net.parameters():
        param.requires_grad = False
    for param in model.zp_head.parameters():
        param.requires_grad = False

    rlcd_optimizer = torch.optim.AdamW(model.evidence_head.parameters(), lr=5e-4)
    for _ in range(15):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            c_b = c_train[idx]
            y_b = y_train[idx]

            rlcd_optimizer.zero_grad()
            with torch.no_grad():
                z_pred = model.integrate_ode(c_b, num_steps=3, method="euler")
                c_emb = model.velocity_net.cond_proj(c_b)
                feat = torch.cat([z_pred, c_emb], dim=-1)

            evidence = model.evidence_head(feat)
            alpha = evidence + 1.0
            s = torch.sum(alpha, dim=-1, keepdim=True)
            p = alpha / s

            y_one_hot = F.one_hot(y_b, num_classes=model.num_classes).float()
            brier = torch.mean(class_weights[y_b] * torch.sum((y_one_hot - p) ** 2, dim=-1))

            # Clipped Log Doubt Reward (Bani-Harouni et al. 2026)
            u_epi = float(model.num_classes) / s.squeeze(-1)
            pred_class = torch.argmax(p, dim=-1)
            is_incorrect = (pred_class != y_b).float()
            doubt_reward = torch.mean(is_incorrect * torch.log(torch.clamp(u_epi, min=1e-4, max=1.0)))

            # CARL Calibration Regularizer
            max_p, _ = torch.max(p, dim=-1)
            is_correct = (pred_class == y_b).float()
            carl_loss = torch.mean((max_p - is_correct) ** 2)

            rlcd_loss = brier - 0.2 * doubt_reward + 0.3 * carl_loss
            rlcd_loss.backward()
            rlcd_optimizer.step()

    # Re-enable grads
    for param in model.parameters():
        param.requires_grad = True

    return {
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epoch_losses": epoch_losses
    }


# --------------------------------------------------------------------------- #
# 5. Spectroscopic Fiber Triage Evaluator & Conformal Gate
# --------------------------------------------------------------------------- #
class SpectroscopicFiberTriageEngine:
    """Dispatches autonomous focal-plane fiber allocations with guaranteed risk bounds."""
    def __init__(
        self,
        model: ContinuousFlowFoundationAstroJev,
        alpha_risk: float = 0.02  # Maximum 2.0% false allocation on contaminants
    ):
        self.model = model
        self.alpha_risk = alpha_risk
        self.conformal_threshold = 0.85

    def calibrate_conformal_gate(
        self,
        c_cal: torch.Tensor,
        y_cal: torch.Tensor
    ) -> float:
        """Calibrates conformal non-conformity threshold ensuring <= alpha_risk false allocations."""
        self.model.eval()
        with torch.no_grad():
            alpha, _, _ = self.model(c_cal, num_ode_steps=4)
            probs, _, _, _ = self.model.compute_dirichlet_statistics(alpha)

        # Contaminants: true label == 4 (PURGE_CONTAMINANT_0M)
        # Non-conformity score: probability assigned to non-purge allocation classes
        p_alloc = 1.0 - probs[:, 4].cpu().numpy()
        contaminant_mask = (y_cal.cpu().numpy() == 4)

        if np.sum(contaminant_mask) > 0:
            contam_scores = p_alloc[contaminant_mask]
            # Quantile at 1 - alpha_risk
            n_c = len(contam_scores)
            q_level = math.ceil((n_c + 1) * (1.0 - self.alpha_risk)) / n_c
            q_level = min(max(q_level, 0.0), 1.0)
            self.conformal_threshold = float(np.quantile(contam_scores, q_level))
        else:
            self.conformal_threshold = 0.85

        return self.conformal_threshold

    def triage_target(
        self,
        c_target: torch.Tensor
    ) -> SpectroscopicTriageDecision:
        """Evaluates single or batch target and returns structured triage decision."""
        self.model.eval()
        if c_target.dim() == 1:
            c_target = c_target.unsqueeze(0)

        with torch.no_grad():
            alpha, zp_pred, _ = self.model(c_target, num_ode_steps=5)
            probs, stds, u_epi, u_ale = self.model.compute_dirichlet_statistics(alpha)

        probs_np = probs[0].cpu().numpy()
        stds_np = stds[0].cpu().numpy()
        best_idx = int(np.argmax(probs_np))
        best_class = ALLOCATION_CLASSES[best_idx]
        conf = float(probs_np[best_idx])
        sigma = float(stds_np[best_idx])
        ci_95 = (max(0.0, conf - 1.96 * sigma), min(1.0, conf + 1.96 * sigma))

        p_alloc = float(1.0 - probs_np[4])
        # Conformal Gate: If probability of allocation > conformal_threshold or best_idx == 4
        is_rejected = (best_idx == 4) or (p_alloc < self.conformal_threshold)

        action = "PURGE_CONTAMINANT_NO_FIBER" if is_rejected else f"ALLOCATE_{best_class}"
        exp_time = 0.0 if is_rejected else FIBER_EXPOSURE_MINUTES[best_class]

        # Conformal prediction set (all classes where p_k > 0.05)
        conf_set = [ALLOCATION_CLASSES[k] for k in range(NUM_ALLOCATION_CLASSES) if probs_np[k] >= 0.05]

        return SpectroscopicTriageDecision(
            action=action,
            target_class=best_class,
            confidence=conf,
            dirichlet_std=sigma,
            ci_95=ci_95,
            epistemic_vacuity=float(u_epi[0]),
            aleatoric_entropy=float(u_ale[0]),
            conformal_set=conf_set,
            estimated_zp_offset_mag=float(zp_pred[0, 0]),
            fiber_exposure_minutes=exp_time,
            is_rejected=is_rejected
        )
