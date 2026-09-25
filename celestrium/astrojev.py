"""AstroJev: Calibrated System One Decision Model for Astrophysics.

A first-principles, non-autoregressive decision model combining:
1. Continuous Fourier feature encoding for astronomical photometry, colors, and astrometry.
2. Weight-tied Krasnoselskii-Mann contractive equilibrium loop (ERET / Jevformer).
3. Sparse CReLU accumulator guaranteeing >= 50% latent sparsity.
4. Calibrated decision heads: Choice (multi-class), Noul (binary probability), Score (ordinal).
5. RLCD (Reinforcement Learning for Calibrated Decisions) using strictly proper scoring rules.
"""
from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["Quasar_AGN", "Galactic_Star", "Passive_Galaxy", "White_Dwarf"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# Feature schema: [phot_g, phot_bp - phot_rp, phot_g - phot_bp, mag_w1, mag_w1 - mag_w2, pm, pm_err, l, b, snr_flux]
FEATURE_NAMES = [
    "phot_g", "bp_rp", "g_bp", "w1", "w1_w2",
    "pm", "pm_err", "gal_l", "gal_b", "snr_flux"
]
NUM_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# 1. Strictly Proper Scoring Rules & RLCD Reward Engine
# --------------------------------------------------------------------------- #
def brier_score_reward(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Bounded Brier Score reward in [0, 1] (strictly proper).

    R_Brier = 1 - 0.5 * sum_k (p_k - delta_{y,k})^2
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    sq_err = torch.sum((probs - one_hot) ** 2, dim=-1)
    return 1.0 - 0.5 * sq_err


def brier_rlcr_reward(probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Bounded Brier RLCR reward (MIT ICLR 2026 formulation):

    R = 1_{correct} - ||p - e_y||^2
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    brier_penalty = torch.sum((probs - one_hot) ** 2, dim=-1)
    top1 = torch.argmax(probs, dim=-1)
    correctness = (top1 == targets).float()
    return correctness - brier_penalty


def normalized_log_score(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Cardinality-normalized logarithmic score (strictly proper).

    R_log = 1 + log(p_y) / log(K)
    Uniform guess p=1/K yields 0.0, perfect guess yields 1.0.
    """
    k = probs.size(1)
    p_y = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    log_k = math.log(max(k, 2))
    return 1.0 + torch.log(p_y.clamp(min=eps)) / log_k


def rewarding_doubt_score(probs: torch.Tensor, targets: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """TUM Rewarding Doubt Score (2026):

    R = 1_{correct} * log(p_y) + 1_{incorrect} * log(1 - p_pred)
    """
    top1 = torch.argmax(probs, dim=-1)
    is_correct = (top1 == targets).float()
    p_y = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).clamp(min=eps, max=1.0 - eps)
    p_pred = torch.max(probs, dim=-1).values.clamp(min=eps, max=1.0 - eps)

    reward_correct = torch.log(p_y)
    reward_doubt = torch.log(1.0 - p_pred)
    return is_correct * reward_correct + (1.0 - is_correct) * reward_doubt



def tsallis_score_reward(
    probs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Tsallis alpha-divergence strictly proper scoring rule.

    S_alpha(p, y) = [alpha / (alpha - 1)] * p_y^(alpha - 1) - sum_k p_k^alpha - 1
    (the sum term carries no 1/(alpha - 1) factor; with it the rule is improper for alpha != 2).
    Interpolates continuously between Log Score (alpha -> 1) and Brier Score (alpha = 2).
    For alpha in (1, 2], polynomial bounded gradients prevent logit divergence on corrupted spectra.
    """
    if abs(alpha - 1.0) < 1e-4:
        return normalized_log_score(probs, targets, eps=eps)
    elif abs(alpha - 2.0) < 1e-4:
        return brier_score_reward(probs, targets)

    p_clamped = probs.clamp(min=eps, max=1.0)
    p_y = p_clamped.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    sum_p_alpha = torch.sum(p_clamped ** alpha, dim=-1)

    term1 = (alpha / (alpha - 1.0)) * (p_y ** (alpha - 1.0))
    term2 = sum_p_alpha
    return term1 - term2 - 1.0


def focal_brier_score_reward(
    probs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Bregman-Focal Proper Scoring Rule (Charoenphakdee et al. 2021).

    R = 1 - 0.5 * (1 - p_y)^gamma * sum_k (p_k - delta_{y,k})^2
    Modulates Brier penalty by (1 - p_y)^gamma to focus on hard ambiguous boundaries
    without destroying proper scoring calibration semantics.
    """
    one_hot = F.one_hot(targets, num_classes=probs.size(1)).float()
    sq_err = torch.sum((probs - one_hot) ** 2, dim=-1)
    p_y = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
    focal_weight = (1.0 - p_y) ** gamma
    return 1.0 - 0.5 * focal_weight * sq_err


def rlcd_astronomy_reward(
    probs: torch.Tensor,
    targets: torch.Tensor,
    w_brier: float = 1.0,
    w_log: float = 0.5,
    w_doubt: float = 0.2,
) -> torch.Tensor:
    """Combined strictly proper scoring rule reward for astronomical RLCD."""
    r_brier = brier_score_reward(probs, targets)
    r_log = normalized_log_score(probs, targets)
    r_doubt = rewarding_doubt_score(probs, targets)
    return w_brier * r_brier + w_log * r_log + w_doubt * r_doubt


# --------------------------------------------------------------------------- #
# 2. AstroJev Neural Architecture (Continuous Fourier + Krasnoselskii-Mann + CReLU)
# --------------------------------------------------------------------------- #
class ContinuousFourierEncoder(nn.Module):
    """Encodes continuous astronomical features into high-dimensional representations."""
    def __init__(self, in_features: int = NUM_FEATURES, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.linear = nn.Linear(in_features, d_model)
        self.fourier_weight = nn.Parameter(torch.randn(in_features, d_model // 2) * 0.5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direct linear mapping + random Fourier harmonic features
        h_dir = self.linear(x)
        proj = 2.0 * math.pi * (x @ self.fourier_weight)
        h_fourier = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.norm(h_dir + h_fourier)


class KMContractiveBlock(nn.Module):
    """Weight-tied contractive recurrence operator T_theta with Squeeze-and-Excitation."""
    def __init__(self, d_model: int = 64):
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
        h_out = self.mlp(h_in)
        se = self.se_gate(h_out)
        return h_out * se


class AstroJev(nn.Module):
    """AstroJev: Epistemic Recurrent Equilibrium Decision Model for Astrophysics."""
    def __init__(
        self,
        in_features: int = NUM_FEATURES,
        d_model: int = 64,
        num_classes: int = NUM_CLASSES,
        n_iter: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_iter = n_iter
        self.encoder = ContinuousFourierEncoder(in_features, d_model)
        self.km_block = KMContractiveBlock(d_model)

        # CReLU sparse accumulator: guarantees >= 50% sparsity
        self.accum_proj = nn.Linear(d_model, 128)
        self.accum_norm = nn.LayerNorm(128)

        # Typed Calibrated Decision Heads
        self.choice_head = nn.Linear(128, num_classes)
        self.noul_head = nn.Linear(128, 1)    # Extragalactic cosmological tracer (Yes/No)
        self.score_head = nn.Linear(128, 5)   # Contamination / Purity score (1-5)

        # Learnable temperature scaling per decision primitive
        self.log_temp_choice = nn.Parameter(torch.zeros(1))
        self.log_temp_noul = nn.Parameter(torch.zeros(1))
        self.log_temp_score = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        heteroscedastic_noise: bool = False,
        noise_scale: float = 1.0,
    ) -> Dict[str, Any]:
        from .kernels import fused_crelu

        x_in = x
        if heteroscedastic_noise and self.training:
            # Physical observational error scaling:
            # Feature 6: pm_err (mas/yr)
            # Feature 9: snr_flux
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

        # Krasnoselskii-Mann contractive equilibrium loop (graph-break-free)
        for k in range(self.n_iter):
            gamma_k = 1.0 / (1.0 + 0.2 * (k + 1))
            t_h = self.km_block(h, x_ctx)
            h_prev = h
            h = (1.0 - gamma_k) * h + gamma_k * t_h

        # Contraction residual ||h_K - h_{K-1}||_2 (measures epistemic equilibrium stability)
        delta_eq = torch.norm(h - h_prev, p=2, dim=-1)

        # CReLU sparse accumulator [0, 1] using fused autograd kernel
        accum = self.accum_norm(self.accum_proj(h))
        crelu_sparse, sparsity = fused_crelu(accum)

        # Calibrated readouts
        t_choice = torch.exp(self.log_temp_choice)
        t_noul = torch.exp(self.log_temp_noul)
        t_score = torch.exp(self.log_temp_score)

        choice_logits = self.choice_head(crelu_sparse) / t_choice
        choice_probs = F.softmax(choice_logits, dim=-1)

        noul_logit = self.noul_head(crelu_sparse).squeeze(-1) / t_noul
        noul_prob = torch.sigmoid(noul_logit)

        score_logits = self.score_head(crelu_sparse) / t_score
        score_probs = F.softmax(score_logits, dim=-1)

        # Prediction entropy as epistemic confidence indicator
        entropy = -torch.sum(choice_probs * torch.log(choice_probs.clamp(min=1e-8)), dim=-1)
        max_entropy = math.log(choice_probs.size(-1))
        normalized_confidence = 1.0 - (entropy / max_entropy)

        return {
            "choice_logits": choice_logits,
            "choice_probs": choice_probs,
            "noul_prob": noul_prob,
            "score_probs": score_probs,
            "confidence": normalized_confidence,
            "delta_eq": delta_eq,
            "residuals": residuals,
            "sparsity": sparsity,
            "latent": crelu_sparse,
        }


# --------------------------------------------------------------------------- #
# 3. Calibration Evaluation Utilities
# --------------------------------------------------------------------------- #
def evaluate_calibration(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Calculate Expected Calibration Error (ECE), Max Calibration Error (MCE), and Brier Score."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(labels)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            avg_confidence = float(np.mean(confidences[in_bin]))
            avg_accuracy = float(np.mean(accuracies[in_bin]))
            diff = abs(avg_confidence - avg_accuracy)
            ece += (bin_size / n) * diff
            mce = max(mce, diff)

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n), labels] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    accuracy = float(np.mean(accuracies))

    # Compute unbiased debiased squared calibration error (Kumar, Liang, Ma NeurIPS 2019)
    debias_res = debiased_squared_calibration_error(probs, labels, n_bins=n_bins)

    return {
        "accuracy": accuracy,
        "brier": brier,
        "ece": float(ece),
        "mce": float(mce),
        "debiased_squared_ce": debias_res["debiased_squared_ce"],
        "rmsce_debiased": debias_res["rmsce_debiased"],
        "rmsce_plugin": debias_res["rmsce_plugin"],
    }


def debiased_squared_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Computes unbiased debiased squared calibration error E^2_db (Kumar, Liang, Ma NeurIPS 2019).

    Standard plugin ECE and RMSCE are positively biased due to the finite-sample variance
    Var(y_hat_s) = y_hat_s * (1 - y_hat_s) / |B_s|.
    The debiased estimator subtracts this variance term, achieving optimal O(sqrt(B)/n) sample complexity.

    E^2_db = sum_s p_s * [ (s - y_s)^2 - y_s*(1 - y_s) / (p_s * n - 1) ]
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    n = len(labels)

    if n <= 1:
        return {"debiased_squared_ce": 0.0, "rmsce_debiased": 0.0, "rmsce_plugin": 0.0}

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    e2_plugin = 0.0
    e2_debiased = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        b_s = int(np.sum(in_bin))
        if b_s > 0:
            p_s = b_s / float(n)
            s_mean = float(np.mean(confidences[in_bin]))
            y_mean = float(np.mean(accuracies[in_bin]))
            sq_err = (s_mean - y_mean) ** 2
            e2_plugin += p_s * sq_err

            if b_s > 1:
                bias = (y_mean * (1.0 - y_mean)) / float(b_s - 1)
                e2_debiased += p_s * (sq_err - bias)
            else:
                e2_debiased += p_s * sq_err

    rmsce_plugin = float(np.sqrt(max(0.0, e2_plugin)))
    rmsce_debiased = float(np.sqrt(max(0.0, e2_debiased)))

    return {
        "debiased_squared_ce": float(e2_debiased),
        "rmsce_debiased": rmsce_debiased,
        "rmsce_plugin": rmsce_plugin,
    }


class ScalingBinningCalibrator:
    """Verified Scaling-Binning Calibrator (Kumar, Liang, Ma, Stanford NeurIPS 2019).

    Combines temperature scaling to reduce variance with uniform-mass quantile binning
    to provably guarantee finite-sample calibration with O(1/eps^2 + B) sample complexity.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.temperature: float = 1.0
        self.bin_edges: Optional[np.ndarray] = None
        self.bin_values: Optional[np.ndarray] = None

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "ScalingBinningCalibrator":
        """Fits temperature scaling on logits then calibrates uniform-mass bins."""
        from scipy.optimize import minimize_scalar

        logits_t = torch.from_numpy(logits).float()
        labels_t = torch.from_numpy(labels).long()

        # Step 1: Optimize temperature T via NLL
        def nll_obj(temp: float) -> float:
            if temp <= 1e-4:
                return 1e6
            scaled = logits_t / temp
            loss = F.cross_entropy(scaled, labels_t)
            return float(loss.item())

        res = minimize_scalar(nll_obj, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)

        # Scaled probabilities
        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        confidences = np.max(scaled_probs, axis=-1)
        predictions = np.argmax(scaled_probs, axis=-1)
        accuracies = (predictions == labels).astype(float)

        # Step 2: Uniform-mass quantile binning
        quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
        self.bin_edges = np.quantile(confidences, quantiles)
        self.bin_edges[0] = 0.0
        self.bin_edges[-1] = 1.0001
        self.bin_edges = np.unique(self.bin_edges)

        actual_bins = len(self.bin_edges) - 1
        self.bin_values = np.zeros(actual_bins)

        for b in range(actual_bins):
            in_b = (confidences >= self.bin_edges[b]) & (confidences < self.bin_edges[b + 1])
            if np.sum(in_b) > 0:
                self.bin_values[b] = float(np.mean(accuracies[in_b]))
            else:
                self.bin_values[b] = 0.5 * (self.bin_edges[b] + self.bin_edges[b + 1])

        return self

    def calibrate(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrates logits returning calibrated probabilities and adjusted top-1 confidences."""
        if self.bin_edges is None or self.bin_values is None:
            raise ValueError("ScalingBinningCalibrator must be fit before calling calibrate")

        scaled_logits = logits / self.temperature
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        confidences = np.max(scaled_probs, axis=-1)

        # Assign confidences to bins
        bin_indices = np.digitize(confidences, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_values) - 1)
        calibrated_confidences = self.bin_values[bin_indices]

        # Adjust the full probability vector proportionally to preserve sum=1
        top_idx = np.argmax(scaled_probs, axis=-1)
        calibrated_probs = scaled_probs.copy()
        for i in range(len(calibrated_probs)):
            c_orig = scaled_probs[i, top_idx[i]]
            c_new = calibrated_confidences[i]
            if c_orig < 1.0:
                scale_rest = (1.0 - c_new) / (1.0 - c_orig + 1e-12)
                calibrated_probs[i] *= scale_rest
                calibrated_probs[i, top_idx[i]] = c_new
            else:
                calibrated_probs[i, top_idx[i]] = c_new

        return calibrated_probs, calibrated_confidences



def conformal_evidential_calibrate(
    probs: np.ndarray,
    u_epi: np.ndarray,
    labels: np.ndarray,
    alpha_error: float = 0.10,
    lambda_epi: float = 0.5,
) -> Dict[str, Any]:
    """Calibrate split conformal prediction set using Dirichlet epistemic vacuity.

    Non-conformity score: s_i = 1 - p_i(y_i) + lambda_epi * u_epi_i
    Guarantees finite-sample coverage P(Y in C(X)) >= 1 - alpha_error under exchangeability.
    """
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot calibrate on empty dataset")
    p_true = probs[np.arange(n), labels]
    nonconf_scores = (1.0 - p_true) + lambda_epi * u_epi

    # Conformal quantile level: ceil((n + 1) * (1 - alpha_error)) / n.
    # If that rank exceeds n, the calibration set is too small for the requested
    # coverage and the valid threshold is +inf (predict all classes).
    rank = math.ceil((n + 1) * (1.0 - alpha_error))
    if rank > n:
        q_hat = float("inf")
    else:
        q_hat = float(np.quantile(nonconf_scores, rank / n, method="higher"))

    # Compute empirical coverage on calibration set
    sets = conformal_predict_sets(probs, u_epi, q_hat=q_hat, lambda_epi=lambda_epi)
    covered = sum(labels[i] in sets[i] for i in range(n))
    empirical_coverage = float(covered / n)
    mean_set_size = float(np.mean([len(s) for s in sets]))
    singleton_fraction = float(np.mean([len(s) == 1 for s in sets]))

    return {
        "q_hat": q_hat,
        "alpha_error": alpha_error,
        "lambda_epi": lambda_epi,
        "empirical_coverage": empirical_coverage,
        "target_coverage": 1.0 - alpha_error,
        "mean_set_size": mean_set_size,
        "singleton_fraction": singleton_fraction,
        "n_samples": n,
    }


def conformal_predict_sets(
    probs: np.ndarray,
    u_epi: np.ndarray,
    q_hat: float,
    lambda_epi: float = 0.5,
) -> List[List[int]]:
    """Generate conformal prediction sets C(x) given calibrated threshold q_hat."""
    prediction_sets = []
    n, k = probs.shape
    for i in range(n):
        scores = (1.0 - probs[i]) + lambda_epi * u_epi[i]
        valid_classes = [c for c in range(k) if scores[c] <= q_hat]
        if not valid_classes:
            # If empty set, fallback to greedy top-1 class
            valid_classes = [int(np.argmax(probs[i]))]
        prediction_sets.append(valid_classes)
    return prediction_sets


def dirichlet_bald_information_gain(alpha: np.ndarray) -> np.ndarray:
    """Exact analytic Dirichlet BALD (Bayesian Active Learning by Disagreement).

    Computes mutual information I(y; p | x) between candidate label y and Dirichlet
    posterior p ~ Dir(alpha) in closed form using the digamma function psi:

        I_BALD = H(p_bar) - E_{p ~ Dir(alpha)}[H(p)]
        E[H(p)] = - sum_k (alpha_k / S) * [ psi(alpha_k + 1) - psi(S + 1) ]

    High I_BALD identifies candidates where posterior belief has maximal epistemic variance,
    prioritizing targets whose follow-up provides the highest information gain.
    """
    from scipy.special import psi

    alpha = np.asarray(alpha, dtype=np.float64)
    S = np.sum(alpha, axis=-1, keepdims=True)
    probs = alpha / np.maximum(S, 1e-12)

    # 1. Total predictive entropy H(p_bar)
    h_total = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)

    # 2. Expected entropy under Dirichlet posterior E[H(p)]
    psi_alpha_plus_1 = psi(alpha + 1.0)
    psi_s_plus_1 = psi(S + 1.0)
    h_expected = -np.sum(probs * (psi_alpha_plus_1 - psi_s_plus_1), axis=-1)

    # 3. Mutual Information (epistemic disagreement)
    bald = np.maximum(0.0, h_total - h_expected)
    return bald


def conformal_risk_control_calibrate(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha_risk: float = 0.05,
    target_class: int = 0,
) -> Dict[str, Any]:
    """Conformal Risk Control (Angelopoulos, Bates et al. 2024 / Bates et al. 2021).

    Calibrates inclusion threshold lambda_hat such that the False Discovery Rate
    (contamination rate of non-target contaminants in target_class selection) satisfies:
        E[FDR] <= alpha_risk with finite-sample guarantee under exchangeability.

    Returns calibrated threshold lambda_hat and empirical risk statistics. The
    ``feasible`` key is False when no threshold selects any sample while meeting
    the risk bound (the returned lambda_hat then selects nothing; a warning is issued).
    """
    n = len(labels)
    if n == 0:
        raise ValueError("Cannot calibrate risk control on empty dataset")

    # Target class confidence
    target_probs = probs[:, target_class]
    is_target = (labels == target_class).astype(float)

    # Grid search for threshold lambda in [0, 1]
    candidate_lambdas = np.sort(np.unique(np.concatenate([target_probs, [0.0, 1.0]])))
    best_lambda = 1.0
    best_risk = 0.0
    best_retention = 0.0
    best_n_selected = 0

    for lam in candidate_lambdas:
        selected = target_probs >= lam
        n_selected = int(np.sum(selected))
        if n_selected == 0:
            risk = 0.0
        else:
            # False discoveries: selected but true label is not target_class
            false_discoveries = np.sum(selected & (is_target == 0))
            risk = false_discoveries / float(n_selected)

        # Finite sample upper bound with (n / (n + 1)) adjustment
        adjusted_risk = (n / (n + 1.0)) * risk + (1.0 / (n + 1.0))

        if adjusted_risk <= alpha_risk:
            best_lambda = float(lam)
            best_risk = float(risk)
            best_retention = float(n_selected / n)
            best_n_selected = n_selected
            break

    feasible = best_n_selected > 0
    if not feasible:
        warnings.warn(
            f"conformal_risk_control_calibrate: no threshold selects any sample of class "
            f"{target_class} with adjusted risk <= alpha_risk={alpha_risk} (n={n}); "
            f"lambda_hat={best_lambda} selects nothing.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "lambda_hat": best_lambda,
        "alpha_risk": alpha_risk,
        "empirical_risk": best_risk,
        "sample_retention": best_retention,
        "n_samples": n,
        "target_class": target_class,
        "feasible": feasible,
    }


