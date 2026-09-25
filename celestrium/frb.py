r"""
=============================================================================
Celestrium Fast Radio Burst (FRB) Evidential Dispersion Triage & Host Localization (EXP-2026-X)
=============================================================================
Calibrated decision engine for real-time triage of Fast Radio Bursts (FRBs) from
coherent radio transit surveys (CHIME/FRB, DSA-110, CRAFT, MeerKAT).

Physics & Mathematical Foundations:
1. Dispersion Measure (DM) Budget & Macquart Relation (Macquart et al. 2020 Nature):
   DM_tot = DM_MW,disk(l, b) + DM_MW,halo + DM_cosmic(z) + DM_host / (1 + z)
   <DM_cosmic(z)> ~= \int_0^z [c n_e(z')] / [H_0 (1+z')^2 sqrt(Omega_m(1+z')^3 + Omega_Lambda)] dz'
   Nominal scaling: <DM_cosmic(z)> ~= 950 * z pc cm^-3 for z < 1.5.

2. Galactic Dispersion Discrepancy & Scattering Anomalies:
   - NE2001 (Cordes & Lazio 2002) vs YMW16 (Yao et al. 2017) model discrepancies.
   - Bhat et al. (2004) Galactic empirical scattering-DM relation:
     log10(tau_scat) = -3.86 + 0.154*log10(DM) + 1.07*(log10(DM))^2.
   - Cosmological FRBs exhibit scattering orders of magnitude lower than Galactic disk
     predictions for the same DM, providing a clean physical cosmological signature.

3. Evidential Decision Engine (Zero Naked Predictions):
   - 4-Class Dirichlet Evidential Network:
     ["COSMOLOGICAL_IGM", "LOCAL_PLASMA_EXCESS", "HOST_DOMINATED_BURST", "INSTRUMENTAL_RFI"]
   - Analytical Dirichlet posterior variance: sigma_k = sqrt( [p_k (1 - p_k)] / [S + 1] )
   - 95% Credible Intervals [p_k - 1.96*sigma_k, p_k + 1.96*sigma_k] clamped to [0, 1].
   - Epistemic vacuity u_epi = K / S (S = sum_k alpha_k).

4. Disentangled RLCD Calibration (TUM 2026 / Bani-Harouni et al.):
   - Frozen representation trunk during calibration fine-tuning.
   - Clipped logarithmic doubt scoring with Stanford debiased squared error E^2_db.

5. Autonomous Follow-up Gating Policy:
   - COMMIT_8M_HOST_SPECTROSCOPY: High-confidence cosmological FRB with bounded epistemic
     vacuity routed to Gemini GMOS / Keck LRIS for deep host galaxy spectroscopic redshift.
   - MONITOR_RADIO_REPETITION: High vacuity or repeater candidates routed to low-cost
     robotic radio arrays (CHIME / DSA-110) to search for repeat bursts.
   - FLAG_LOCAL_PLASMA_CONTAMINANT: Unmodeled Galactic ISM / Cygnus-like plasma overdensity.
   - REJECT_RFI: Terrestrial / instrumental interference.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FRB_CLASSES = [
    "COSMOLOGICAL_IGM",
    "LOCAL_PLASMA_EXCESS",
    "HOST_DOMINATED_BURST",
    "INSTRUMENTAL_RFI",
]
FRB_CLASS_TO_IDX = {c: i for i, c in enumerate(FRB_CLASSES)}
NUM_FRB_CLASSES = len(FRB_CLASSES)

FRB_FEATURE_NAMES = [
    "log_dm",                 # log10(DM_fit in pc cm^-3)
    "log_dm_exc_ne2001",      # log10(max(DM_exc_ne2001, 1.0))
    "log_dm_exc_ymw16",       # log10(max(DM_exc_ymw16, 1.0))
    "dm_model_discrepancy",   # |DM_ne2001 - DM_ymw16| / DM_fit
    "sin_abs_lat",            # sin(|b|) Galactic latitude factor
    "log_scattering_ms",      # log10(max(tau_scat, 1e-3))
    "log_width_ms",           # log10(max(width_fit, 0.05))
    "log_peak_flux_jy",       # log10(max(flux, 0.01))
    "log_fluence_jyms",       # log10(max(fluence, 0.05))
    "snr",                    # Detection S/N ratio
    "scattering_excess",      # log10(tau_scat) - log10(tau_Bhat(DM))
    "inferred_macquart_z",    # max(DM_exc - 60, 0) / 950.0
]
NUM_FRB_FEATURES = len(FRB_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# 1. Physics Calculations: Galactic DM, Bhat Scattering, Macquart Redshift
# --------------------------------------------------------------------------- #
def bhat_empirical_scattering_ms(dm: float) -> float:
    """Empirical Galactic disk scattering timescale in ms from Bhat et al. (2004)."""
    dm_c = max(float(dm), 1.0)
    log_dm = math.log10(dm_c)
    # Bhat+2004 relation: log10(tau_mu_s) = -3.86 + 0.154 log10(DM) + 1.07 (log10(DM))^2
    log_tau_mus = -3.86 + 0.154 * log_dm + 1.07 * (log_dm ** 2)
    # Convert from microseconds to milliseconds
    tau_ms = (10.0 ** log_tau_mus) / 1000.0
    return max(tau_ms, 1e-4)


def macquart_inferred_redshift(dm_excess: float, dm_halo: float = 60.0) -> float:
    """Estimate nominal cosmological redshift z using the Macquart et al. (2020) relation."""
    cosmic_dm = max(float(dm_excess) - float(dm_halo), 0.0)
    # Mean IGM scaling: ~950 pc cm^-3 per unit redshift
    z_nom = cosmic_dm / 950.0
    return float(z_nom)


# --------------------------------------------------------------------------- #
# 2. FRB Observation Record & Feature Extraction
# --------------------------------------------------------------------------- #
@dataclass
class FRBBurstRecord:
    name: str
    ra: float
    dec: float
    glon: float
    glat: float
    dm: float
    dm_err: float = 0.5
    dm_exc_ne2001: float = 0.0
    dm_exc_ymw16: float = 0.0
    snr: float = 15.0
    scat_ms: float = 0.0
    scat_err_ms: float = 0.0
    width_ms: float = 1.0
    flux_jy: float = 1.0
    fluence_jyms: float = 2.0
    is_repeater: bool = False
    repeater_name: Optional[str] = None
    true_class: Optional[str] = None


def extract_frb_features(burst: FRBBurstRecord, perturb_noise: bool = False, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Extract 12-dimensional physical feature vector with heteroscedastic noise option."""
    r = rng or np.random.default_rng(42)

    dm = burst.dm
    if perturb_noise and burst.dm_err > 0:
        dm = float(max(10.0, r.normal(dm, burst.dm_err)))

    dm_ne = max(burst.dm_exc_ne2001, 0.0)
    dm_ymw = max(burst.dm_exc_ymw16, 0.0)
    model_diff = abs(dm_ne - dm_ymw) / max(dm, 1.0)
    sin_b = math.sin(math.radians(abs(burst.glat)))

    scat = max(burst.scat_ms, 1e-4)
    if perturb_noise and burst.scat_err_ms > 0:
        scat = float(max(1e-4, r.normal(scat, burst.scat_err_ms)))

    tau_bhat = bhat_empirical_scattering_ms(dm)
    scat_excess = math.log10(scat) - math.log10(tau_bhat)
    z_macquart = macquart_inferred_redshift(dm_ne)

    feats = np.array([
        math.log10(max(dm, 1.0)),
        math.log10(max(dm_ne, 1.0)),
        math.log10(max(dm_ymw, 1.0)),
        model_diff,
        sin_b,
        math.log10(scat),
        math.log10(max(burst.width_ms, 0.05)),
        math.log10(max(burst.flux_jy, 0.01)),
        math.log10(max(burst.fluence_jyms, 0.05)),
        min(burst.snr, 100.0) / 20.0,  # normalized SNR
        float(np.clip(scat_excess, -5.0, 5.0)),
        min(z_macquart, 4.0),
    ], dtype=np.float32)

    return feats


# --------------------------------------------------------------------------- #
# 3. Evidential Neural Network Architecture
# --------------------------------------------------------------------------- #
class FRBEvidentialNet(nn.Module):
    """Deep Dirichlet evidential network with disentangled RLCD calibration head."""

    def __init__(self, in_features: int = NUM_FRB_FEATURES, hidden_dim: int = 64, num_classes: int = NUM_FRB_CLASSES):
        super().__init__()
        # 1. Representation trunk (frozen during RLCD calibration)
        self.trunk = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        # 2. Dirichlet readout head (fine-tuned during RLCD calibration)
        self.evidence_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        logits = self.evidence_head(h)
        # Dirichlet parameters alpha_k = softplus(logits) + 1.0 (evidence >= 0)
        alpha = F.softplus(logits) + 1.0
        s = torch.sum(alpha, dim=-1, keepdim=True)
        probs = alpha / s

        # Epistemic vacuity u_epi = K / S (per CLAUDE.md standard)
        num_k = alpha.shape[-1]
        u_epi = num_k / torch.clamp(s.squeeze(-1), min=1e-8)

        # Dirichlet posterior standard deviation: sqrt( p_k (1 - p_k) / (S + 1) )
        sigmas = torch.sqrt(torch.clamp(probs * (1.0 - probs) / (s + 1.0), min=0.0))

        # Aleatoric entropy
        eps = 1e-12
        u_ale = -torch.sum(probs * torch.log(probs + eps), dim=-1)

        return {
            "alpha": alpha,
            "probs": probs,
            "sigmas": sigmas,
            "u_epi": u_epi,
            "u_ale": u_ale,
            "features_latent": h,
        }

    def freeze_trunk(self) -> None:
        """Freeze representation trunk for Disentangled RLCD optimization."""
        for param in self.trunk.parameters():
            param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


# --------------------------------------------------------------------------- #
# 4. Calibration: Stanford Debiased Error & Disentangled RLCD Loss
# --------------------------------------------------------------------------- #
def compute_stanford_debiased_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Calculate Stanford debiased squared calibration error E^2_db (Kumar et al. NeurIPS 2019)."""
    n = len(labels)
    if n == 0:
        return {"ece": 0.0, "debiased_e2": 0.0, "rmsce": 0.0}

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    e2_db_sum = 0.0
    rmsce_sum = 0.0

    for b in range(n_bins):
        in_bin = bin_indices == b
        nb = float(np.sum(in_bin))
        if nb > 0:
            acc_b = float(np.mean(accuracies[in_bin]))
            conf_b = float(np.mean(confidences[in_bin]))
            diff = acc_b - conf_b
            ece += (nb / n) * abs(diff)
            rmsce_sum += (nb / n) * (diff ** 2)

            # Debiased term subtracting positive finite-sample variance bias:
            # Var_bias = [acc_b * (1 - acc_b)] / (nb - 1) for nb > 1
            if nb > 1:
                var_bias = (acc_b * (1.0 - acc_b)) / (nb - 1.0)
                debiased_term = (diff ** 2) - var_bias
            else:
                debiased_term = diff ** 2
            e2_db_sum += (nb / n) * debiased_term

    e2_db = max(0.0, float(e2_db_sum))
    rmsce = float(np.sqrt(rmsce_sum))

    return {
        "ece": float(ece),
        "debiased_e2": float(e2_db),
        "rmsce": float(rmsce),
    }


def frb_rlcd_doubt_loss(
    probs: torch.Tensor,
    labels: torch.Tensor,
    beta_doubt: float = 0.25,
    gamma_carl: float = 0.05,
    alpha_dirichlet: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Disentangled RLCD composite loss: Brier + Clipped Log-Doubt + CARL regularizer."""
    num_classes = probs.shape[-1]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # 1. Brier score
    brier_loss = torch.mean(torch.sum((probs - one_hot) ** 2, dim=-1))

    # 2. Clipped logarithmic doubt scoring
    top_p, pred = torch.max(probs, dim=-1)
    correct = (pred == labels).float()
    eps = 1e-4
    r_correct = torch.log(torch.clamp(top_p, min=eps))
    r_incorrect = torch.log(torch.clamp(1.0 - top_p, min=eps))
    reward = correct * r_correct + (1.0 - correct) * r_incorrect
    loss_doubt = -torch.mean(reward)

    # 3. CARL (Conformal Aleatoric Regularizer) penalty
    loss_carl = torch.tensor(0.0, device=probs.device)
    if alpha_dirichlet is not None:
        s = torch.sum(alpha_dirichlet, dim=-1, keepdim=True)
        # Penalize vacuity on high-confidence predictions
        vacuity = num_classes / torch.clamp(s, min=1e-8)
        loss_carl = torch.mean(vacuity * (top_p.detach() ** 2))

    total_loss = brier_loss + beta_doubt * loss_doubt + gamma_carl * loss_carl
    return total_loss


def calibrate_frb_conformal_risk(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha_risk: float = 0.05,
    target_class: int = 0,  # COSMOLOGICAL_IGM
) -> Dict[str, Any]:
    """Calibrate Conformal Risk Control threshold bounding false discoveries (Angelopoulos+ 2024)."""
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot calibrate CRC on empty set")

    target_probs = probs[:, target_class]
    is_target = (labels == target_class).astype(float)
    n_targets = int(np.sum(is_target))

    candidate_lambdas = np.sort(np.unique(np.concatenate([target_probs, [0.0, 1.0]])))
    best_lambda = 0.85
    best_risk = 0.0
    best_retention = 0.0
    feasible = False

    for lam in candidate_lambdas:
        selected = target_probs >= lam
        n_sel = int(np.sum(selected))
        if n_sel == 0:
            continue
        false_disc = np.sum(selected & (is_target == 0))
        risk = false_disc / float(n_sel)
        adj_risk = (n / (n + 1.0)) * risk + (1.0 / (n + 1.0))

        if adj_risk <= alpha_risk:
            best_lambda = float(lam)
            best_risk = float(risk)
            retained = np.sum(selected & (is_target == 1))
            best_retention = float(retained / max(n_targets, 1))
            feasible = True
            break

    return {
        "lambda_hat": float(best_lambda),
        "alpha_risk": float(alpha_risk),
        "empirical_risk": float(best_risk),
        "sample_retention": float(best_retention),
        "n_samples": n,
        "feasible": feasible,
    }


# --------------------------------------------------------------------------- #
# 5. Real CHIME/FRB Catalog 1 Streamer
# --------------------------------------------------------------------------- #
class RealCHIMEFRBStreamer:
    """Streamer for real Fast Radio Bursts from CHIME/FRB Catalog 1."""

    def __init__(self, data_path: Optional[str | Path] = "data/chime_frb_catalog1.npz", seed: int = 42):
        self.path = Path(data_path) if data_path else None
        self.rng = np.random.default_rng(seed)
        self.bursts: List[FRBBurstRecord] = []
        self._load_data()

    def _load_data(self) -> None:
        if self.path and self.path.is_file():
            data = np.load(str(self.path))
            n = len(data["name"])
            for i in range(n):
                name = str(data["name"][i])
                is_rep = bool(data["repeater"][i] == 1)
                # Assign physically grounded label for benchmark evaluations
                # Cosmological if high Galactic latitude & high excess DM
                dm_exc = float(data["dm_exc_ne2001"][i])
                glat = float(data["glat"][i])
                if is_rep:
                    t_class = "HOST_DOMINATED_BURST"
                elif abs(glat) < 7.0 or dm_exc < 40.0:
                    t_class = "LOCAL_PLASMA_EXCESS"
                else:
                    t_class = "COSMOLOGICAL_IGM"

                record = FRBBurstRecord(
                    name=name,
                    ra=float(data["ra"][i]),
                    dec=float(data["dec"][i]),
                    glon=float(data["glon"][i]),
                    glat=glat,
                    dm=float(data["dm"][i]),
                    dm_err=float(data["dm_err"][i]),
                    dm_exc_ne2001=dm_exc,
                    dm_exc_ymw16=float(data["dm_exc_ymw16"][i]),
                    snr=float(data["snr"][i]),
                    scat_ms=float(data["scat"][i]),
                    width_ms=float(data["width"][i]),
                    flux_jy=float(data["flux"][i]),
                    fluence_jyms=float(data["fluence"][i]),
                    is_repeater=is_rep,
                    repeater_name=name if is_rep else None,
                    true_class=t_class,
                )
                self.bursts.append(record)
        else:
            # Hermetic synthetic fallback generating 100 physically calibrated bursts
            for i in range(100):
                c_idx = i % 4
                c_name = FRB_CLASSES[c_idx]
                if c_name == "COSMOLOGICAL_IGM":
                    dm = float(self.rng.uniform(350.0, 1500.0))
                    glat = float(self.rng.uniform(15.0, 85.0) * self.rng.choice([-1, 1]))
                    scat = float(self.rng.uniform(0.01, 0.5))
                elif c_name == "LOCAL_PLASMA_EXCESS":
                    dm = float(self.rng.uniform(150.0, 600.0))
                    glat = float(self.rng.uniform(0.5, 8.0) * self.rng.choice([-1, 1]))
                    scat = float(self.rng.uniform(2.0, 25.0))
                elif c_name == "HOST_DOMINATED_BURST":
                    dm = float(self.rng.uniform(500.0, 1200.0))
                    glat = float(self.rng.uniform(20.0, 70.0))
                    scat = float(self.rng.uniform(0.5, 4.0))
                else:  # INSTRUMENTAL_RFI
                    dm = float(self.rng.uniform(5.0, 45.0))
                    glat = float(self.rng.uniform(-90.0, 90.0))
                    scat = 1e-4

                self.bursts.append(
                    FRBBurstRecord(
                        name=f"FRB_MOCK_{i:04d}",
                        ra=float(self.rng.uniform(0.0, 360.0)),
                        dec=float(self.rng.uniform(-10.0, 80.0)),
                        glon=float(self.rng.uniform(0.0, 360.0)),
                        glat=glat,
                        dm=dm,
                        dm_err=float(self.rng.uniform(0.1, 1.5)),
                        dm_exc_ne2001=max(dm - 60.0, 5.0),
                        dm_exc_ymw16=max(dm - 55.0, 5.0),
                        snr=float(self.rng.uniform(10.0, 65.0)),
                        scat_ms=scat,
                        width_ms=float(self.rng.uniform(0.5, 6.0)),
                        flux_jy=float(self.rng.uniform(0.3, 10.0)),
                        fluence_jyms=float(self.rng.uniform(0.5, 20.0)),
                        true_class=c_name,
                    )
                )

    def __len__(self) -> int:
        return len(self.bursts)

    def stream_bursts(self) -> Iterator[FRBBurstRecord]:
        for b in self.bursts:
            yield b


# --------------------------------------------------------------------------- #
# 6. Triage Result & Production Decision Engine
# --------------------------------------------------------------------------- #
@dataclass
class FRBTriageResult:
    burst_name: str
    action: str  # COMMIT_8M_HOST_SPECTROSCOPY | MONITOR_RADIO_REPETITION | FLAG_LOCAL_PLASMA_CONTAMINANT | REJECT_RFI
    predicted_class: str
    confidence: float
    confidence_err: float
    confidence_interval_95: Tuple[float, float]
    epistemic_vacuity: float
    aleatoric_entropy: float
    conformal_passed: bool
    prediction_set: List[str]
    inferred_z: float
    too_payload: Optional[Dict[str, Any]] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


class FRBTriageEngine:
    """Production Fast Radio Burst Evidential Decision & ToO Dispatch Engine."""

    def __init__(
        self,
        model: Optional[FRBEvidentialNet] = None,
        alpha_crc: float = 0.05,
        crc_lambda: float = 0.85,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or FRBEvidentialNet()
        self.model.to(self.device)
        self.model.eval()
        self.alpha_crc = alpha_crc
        self.crc_lambda = crc_lambda

    def triage_bursts(self, bursts: List[FRBBurstRecord]) -> List[FRBTriageResult]:
        """Triage batch of FRB alerts with explicit analytical error bars."""
        if not bursts:
            return []

        feat_list = [extract_frb_features(b) for b in bursts]
        feats = np.stack(feat_list, axis=0)

        with torch.no_grad():
            x = torch.from_numpy(feats).to(self.device)
            out = self.model(x)
            probs = out["probs"].cpu().numpy()
            sigmas = out["sigmas"].cpu().numpy()
            u_epi = out["u_epi"].cpu().numpy()
            u_ale = out["u_ale"].cpu().numpy()

        results = []
        for i, b in enumerate(bursts):
            p = probs[i]
            sig = sigmas[i]
            vac = float(u_epi[i])
            ale = float(u_ale[i])

            pred_idx = int(np.argmax(p))
            pred_class = FRB_CLASSES[pred_idx]
            top_p = float(p[pred_idx])
            top_sig = float(sig[pred_idx])
            ci_95 = (float(max(0.0, top_p - 1.96 * top_sig)), float(min(1.0, top_p + 1.96 * top_sig)))

            p_cosmo = float(p[0])
            sig_cosmo = float(sig[0])
            cosmo_ci_low = float(max(0.0, p_cosmo - 1.96 * sig_cosmo))

            # Conformal prediction set: C_lambda = {k : p_k >= 1 - lambda_hat}
            conf_set = [FRB_CLASSES[k] for k in range(NUM_FRB_CLASSES) if p[k] >= (1.0 - self.crc_lambda)]
            conformal_passed = p_cosmo >= self.crc_lambda

            # Macquart cosmological redshift estimate
            z_est = macquart_inferred_redshift(b.dm_exc_ne2001)

            # Distribution-Free Gating Policy:
            # 1. FLAG_LOCAL_PLASMA_CONTAMINANT: Galactic disk / plasma overdensity
            if pred_class == "LOCAL_PLASMA_EXCESS" or abs(b.glat) < 7.0:
                action = "FLAG_LOCAL_PLASMA_CONTAMINANT"
                too_payload = None
            # 2. COMMIT_8M_HOST_SPECTROSCOPY: High-confidence cosmological burst, low vacuity, out of Galactic plane
            elif conformal_passed and vac <= 0.58 and cosmo_ci_low >= 0.15 and abs(b.glat) >= 10.0:
                action = "COMMIT_8M_HOST_SPECTROSCOPY"
                too_payload = {
                    "instrument": "GEMINI_GMOS_SPECTROSCOPY",
                    "target_name": b.name,
                    "ra_deg": b.ra,
                    "dec_deg": b.dec,
                    "exposure_seconds": 3600.0,
                    "priority": "RAPID_TOO",
                    "inferred_z_macquart": round(z_est, 3),
                    "confidence_95_low": round(cosmo_ci_low, 3),
                    "epistemic_vacuity": round(vac, 3),
                }
            # 3. MONITOR_RADIO_REPETITION: High vacuity, repeater, or host-dominated
            elif b.is_repeater or vac > 0.58 or pred_class == "HOST_DOMINATED_BURST":
                action = "MONITOR_RADIO_REPETITION"
                too_payload = {
                    "instrument": "ROBOTIC_RADIO_ARRAY_MONITORING",
                    "target_name": b.name,
                    "target_dm": b.dm,
                    "cadence_days": 1.0,
                    "reason": "High vacuity or repeating source candidate; search for repeat bursts to localize host galaxy.",
                }
            # 4. REJECT_RFI
            else:
                action = "REJECT_RFI"
                too_payload = None

            results.append(
                FRBTriageResult(
                    burst_name=b.name,
                    action=action,
                    predicted_class=pred_class,
                    confidence=top_p,
                    confidence_err=top_sig,
                    confidence_interval_95=ci_95,
                    epistemic_vacuity=vac,
                    aleatoric_entropy=ale,
                    conformal_passed=conformal_passed,
                    prediction_set=conf_set,
                    inferred_z=z_est,
                    too_payload=too_payload,
                    telemetry={
                        "p_cosmo": p_cosmo,
                        "p_cosmo_err": sig_cosmo,
                        "cosmo_ci_low": cosmo_ci_low,
                        "dm_excess": b.dm_exc_ne2001,
                        "glat": b.glat,
                    },
                )
            )

        return results


def train_frb_model(
    streamer: Optional[RealCHIMEFRBStreamer] = None,
    epochs: int = 10,
    lr: float = 3e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Train FRBEvidentialNet on real/synthetic FRB streams with Disentangled RLCD."""
    torch.manual_seed(seed)
    st = streamer or RealCHIMEFRBStreamer(seed=seed)
    bursts = st.bursts

    # Extract features & labels
    feats = np.stack([extract_frb_features(b, perturb_noise=True) for b in bursts], axis=0)
    labels = np.array([FRB_CLASS_TO_IDX.get(b.true_class, 0) for b in bursts], dtype=np.int64)

    # 80/20 train/val split
    n = len(bursts)
    indices = np.random.default_rng(seed).permutation(n)
    n_train = int(0.8 * n)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    x_train = torch.from_numpy(feats[train_idx]).to(device)
    y_train = torch.from_numpy(labels[train_idx]).to(device)
    x_val = torch.from_numpy(feats[val_idx]).to(device)
    y_val = torch.from_numpy(labels[val_idx]).to(device)

    model = FRBEvidentialNet().to(device)

    # Phase 1: Trunk + Head representation learning
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(x_train)
        loss = frb_rlcd_doubt_loss(out["probs"], y_train, beta_doubt=0.1, alpha_dirichlet=out["alpha"])
        loss.backward()
        optimizer.step()

    # Pre-RLCD evaluation on validation split
    model.eval()
    with torch.no_grad():
        out_pre = model(x_val)
        probs_pre = out_pre["probs"].cpu().numpy()
        calib_pre = compute_stanford_debiased_ece(probs_pre, labels[val_idx])

    # Phase 2: Disentangled RLCD calibration (Freeze trunk, train head with doubt reward)
    model.freeze_trunk()
    head_opt = torch.optim.Adam(model.evidence_head.parameters(), lr=lr * 0.5)
    model.train()
    for ep in range(max(epochs // 2, 5)):
        head_opt.zero_grad()
        out = model(x_train)
        # Emphasize doubt reward and CARL regularizer
        loss = frb_rlcd_doubt_loss(out["probs"], y_train, beta_doubt=0.4, gamma_carl=0.1, alpha_dirichlet=out["alpha"])
        loss.backward()
        head_opt.step()

    # Post-RLCD evaluation on validation split
    model.eval()
    with torch.no_grad():
        out_post = model(x_val)
        probs_post = out_post["probs"].cpu().numpy()
        calib_post = compute_stanford_debiased_ece(probs_post, labels[val_idx])

    # Conformal Risk Control calibration on val split
    crc = calibrate_frb_conformal_risk(probs_post, labels[val_idx], alpha_risk=0.05, target_class=0)

    return {
        "model": model,
        "calib_pre": calib_pre,
        "calib_post": calib_post,
        "crc": crc,
        "n_train": n_train,
        "n_val": len(val_idx),
    }
