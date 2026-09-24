"""Head-to-head comparative benchmark: AstroJev (Local ERET+RLCD) vs Hosted TypeSafe Jev.

Evaluates:
1. Classification accuracy on frozen astronomical test cases with spectroscopic truth.
2. Calibration quality: Multi-class Brier score and Expected Calibration Error (ECE).
3. Latency: Single-pass local inference vs hosted API network round-trip.
4. Token economy and compute efficiency.
5. Disagreement analysis on edge-case astronomical regimes (e.g. CatWISE contaminants).
"""
from __future__ import annotations

import os
import time
import json
from typing import Dict, List, Any, Tuple
import numpy as np
import requests
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from . import astrojev
from .astrojev import AstroJev, CLASSES, CLASS_TO_IDX, NUM_CLASSES, evaluate_calibration, rlcd_astronomy_reward

# Load .env for JEV_API_KEY
load_dotenv(r"C:\Users\Leon\Desktop\Psychograph\jev\.env")
load_dotenv()
JEV_API_KEY = os.getenv("JEV_API_KEY")
TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"


# --------------------------------------------------------------------------- #
# 1. Astronomical Test Cases (Frozen Benchmark Suite)
# --------------------------------------------------------------------------- #
BENCHMARK_CASES = [
    {
        "id": "QSO_01",
        "name": "High-z Radio-Loud Quasar (SDSS J1148+5251 analog)",
        "features": [19.2, 0.45, 0.15, 14.1, 1.15, 0.12, 0.45, 145.2, 58.3, 18.5],
        "true_class": "Quasar_AGN",
        "description": "High-redshift quasar with prominent Lyman break, red mid-IR excess (W1-W2 = 1.15), zero proper motion (0.12 mas/yr), and NVSS radio detection.",
    },
    {
        "id": "QSO_02",
        "name": "Quaia Confirmed AGN (z=1.42)",
        "features": [20.1, 0.32, 0.08, 14.8, 0.98, 0.25, 0.50, 210.4, 42.1, 14.2],
        "true_class": "Quasar_AGN",
        "description": "Spectroscopically confirmed broad-line AGN in Quaia sample with W1-W2 = 0.98 and null Gaia DR3 parallax.",
    },
    {
        "id": "STAR_01",
        "name": "Nearby Solar-type Main Sequence Star",
        "features": [14.5, 0.85, 0.35, 13.8, 0.04, 18.4, 0.08, 45.1, 12.3, 85.0],
        "true_class": "Galactic_Star",
        "description": "Galactic disk G-dwarf with large significant proper motion (18.4 mas/yr) and zero mid-IR color excess (W1-W2 = 0.04).",
    },
    {
        "id": "STAR_02",
        "name": "Dusty AGB / YSO CatWISE Contaminant",
        "features": [17.8, 2.10, 0.95, 13.2, 0.92, 6.8, 0.22, 12.4, 3.2, 28.0],
        "true_class": "Galactic_Star",
        "description": "Circumstellar dust shell around a Galactic giant mimicking an AGN mid-IR color (W1-W2 = 0.92), but located at low Galactic latitude (b=3.2 deg) with 6.8 mas/yr proper motion.",
    },
    {
        "id": "GAL_01",
        "name": "Luminous Red Galaxy / Passive Elliptical (z=0.35)",
        "features": [18.9, 1.45, 0.62, 15.6, 0.18, 0.18, 0.38, 180.2, 64.1, 16.0],
        "true_class": "Passive_Galaxy",
        "description": "Massive early-type galaxy with strong 4000-Angstrom break, red optical colors, flat mid-IR, and null astrometric motion.",
    },
    {
        "id": "WD_01",
        "name": "High-Proper-Motion DA White Dwarf",
        "features": [16.8, -0.25, -0.15, 16.7, -0.02, 48.5, 0.15, 88.5, 35.4, 45.0],
        "true_class": "White_Dwarf",
        "description": "Hot degenerate stellar remnant with bluest optical colors (BP-RP = -0.25), very high proper motion (48.5 mas/yr), and no infrared excess.",
    },
    {
        "id": "QSO_03",
        "name": "Faint Edge-of-Survey Quasar (CatWISE limit)",
        "features": [20.4, 0.50, 0.22, 15.7, 0.85, 0.42, 0.85, 260.1, 48.0, 9.5],
        "true_class": "Quasar_AGN",
        "description": "Distant quasar near CatWISE depth limit with moderate SNR and borderline W1-W2 = 0.85 color.",
    },
    {
        "id": "STAR_03",
        "name": "Late M-Dwarf Flaring Star",
        "features": [18.2, 2.45, 1.10, 14.5, 0.25, 24.2, 0.35, 195.0, 15.5, 32.0],
        "true_class": "Galactic_Star",
        "description": "Cool Galactic M dwarf with very red optical colors, high proper motion, and minor flare activity.",
    },
]


def query_hosted_jev(case: Dict[str, Any]) -> Dict[str, Any]:
    """Query TypeSafe Jev API for a single astronomical decision."""
    if not JEV_API_KEY:
        raise ValueError("JEV_API_KEY not found in environment")

    headers = {
        "Authorization": f"Bearer {JEV_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jev-latest",
        "state": (
            f"Astronomical source {case['id']}: {case['name']}.\n"
            f"Observed properties: {case['description']}\n"
            f"Measurements: phot_g={case['features'][0]:.1f}, bp_rp={case['features'][1]:.2f}, "
            f"w1={case['features'][3]:.1f}, w1_w2={case['features'][4]:.2f} mag, "
            f"pm={case['features'][5]:.2f} mas/yr, gal_b={case['features'][8]:.1f} deg."
        ),
        "questions": {
            "astrophysical_class": {
                "type": "choice",
                "instructions": "What is the primary physical classification of this astronomical source?",
                "options": CLASSES,
                "criteria": {
                    "Quasar_AGN": "Distant active galactic nucleus or quasar with non-thermal mid-IR excess and zero proper motion.",
                    "Galactic_Star": "Milky Way main-sequence, giant, or dusty circumstellar star with measurable astrometric motion.",
                    "Passive_Galaxy": "Early-type or elliptical galaxy with thermal stellar population and no non-thermal mid-IR excess.",
                    "White_Dwarf": "Compact degenerate stellar remnant with hot blue optical colors and high proper motion.",
                },
            }
        },
    }

    t0 = time.perf_counter()
    resp = requests.post(TYPESAFE_API_URL, headers=headers, json=payload, timeout=25)
    t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms

    if resp.status_code != 200:
        raise RuntimeError(f"TypeSafe API error {resp.status_code}: {resp.text}")

    data = resp.json()
    ans = data["answers"]["astrophysical_class"]
    probs = [ans["probabilities"].get(c, 0.0) for c in CLASSES]

    return {
        "predicted_class": ans["choice"],
        "confidence": ans.get("confidence", max(probs)),
        "probabilities": probs,
        "latency_ms": t_elapsed,
        "input_tokens": data.get("usage", {}).get("input_tokens", 0),
        "output_tokens": data.get("usage", {}).get("output_tokens", 0),
    }


def train_local_astrojev(
    n_train: int = 4000,
    epochs: int = 15,
    device: Optional[torch.device] = None,
) -> AstroJev:
    """Train AstroJev via two-stage RLCD using synthetic survey distribution."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate synthetic training set reflecting real Quaia / CatWISE / Gaia statistics
    rng = np.random.default_rng(42)
    labels = rng.integers(0, NUM_CLASSES, size=n_train)
    features = np.zeros((n_train, astrojev.NUM_FEATURES), dtype=np.float32)

    for i in range(n_train):
        c = labels[i]
        if c == 0:  # Quasar / AGN
            g = rng.normal(19.5, 0.8)
            bp_rp = rng.normal(0.4, 0.2)
            g_bp = rng.normal(0.1, 0.1)
            w1 = rng.normal(14.5, 0.7)
            w1_w2 = rng.normal(1.05, 0.2)
            pm = rng.exponential(0.3)
            pm_err = rng.uniform(0.3, 0.8)
            l = rng.uniform(0, 360)
            b = rng.uniform(20, 90)
            snr = rng.uniform(8, 30)
        elif c == 1:  # Star
            g = rng.normal(16.0, 2.0)
            bp_rp = rng.normal(0.9, 0.4)
            g_bp = rng.normal(0.4, 0.2)
            w1 = rng.normal(14.0, 1.5)
            w1_w2 = rng.normal(0.08, 0.15)
            pm = rng.exponential(12.0)
            pm_err = rng.uniform(0.1, 0.5)
            l = rng.uniform(0, 360)
            b = rng.uniform(5, 70)
            snr = rng.uniform(15, 100)
        elif c == 2:  # Passive Galaxy
            g = rng.normal(18.5, 0.6)
            bp_rp = rng.normal(1.3, 0.2)
            g_bp = rng.normal(0.6, 0.1)
            w1 = rng.normal(15.0, 0.5)
            w1_w2 = rng.normal(0.20, 0.1)
            pm = rng.exponential(0.25)
            pm_err = rng.uniform(0.4, 0.9)
            l = rng.uniform(0, 360)
            b = rng.uniform(25, 90)
            snr = rng.uniform(10, 40)
        else:  # White Dwarf
            g = rng.normal(17.0, 1.5)
            bp_rp = rng.normal(-0.2, 0.15)
            g_bp = rng.normal(-0.1, 0.1)
            w1 = rng.normal(16.5, 1.0)
            w1_w2 = rng.normal(-0.05, 0.1)
            pm = rng.exponential(25.0)
            pm_err = rng.uniform(0.1, 0.4)
            l = rng.uniform(0, 360)
            b = rng.uniform(15, 85)
            snr = rng.uniform(20, 80)
        features[i] = [g, bp_rp, g_bp, w1, w1_w2, pm, pm_err, l, b, snr]

    x_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(labels, dtype=torch.long, device=device)

    model = AstroJev(in_features=astrojev.NUM_FEATURES, d_model=64, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    # Stage 1: Warmup with bounded Brier loss (4 epochs)
    for epoch in range(4):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor)
        one_hot = F.one_hot(y_tensor, num_classes=NUM_CLASSES).float()
        loss = torch.mean(torch.sum((out["choice_probs"] - one_hot) ** 2, dim=-1))
        loss.backward()
        optimizer.step()

    # Stage 2: RLCD Exploration + Strictly Proper Scoring Rule Optimization (11 epochs)
    for epoch in range(11):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor)
        logits = out["choice_logits"]
        sigma = 0.08 * (1.0 - epoch / 11.0)
        noise = torch.randn_like(logits) * sigma
        p_noisy = F.softmax(logits + noise, dim=-1)

        reward = rlcd_astronomy_reward(p_noisy, y_tensor, w_brier=1.0, w_log=0.4, w_doubt=0.2)
        loss = -torch.mean(reward)
        loss.backward()
        optimizer.step()

    model.eval()
    return model


def run_comparative_benchmark() -> Dict[str, Any]:
    """Execute head-to-head benchmark between AstroJev and Hosted Jev."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[AstroJev Benchmark] Training local AstroJev on {device}...")
    local_model = train_local_astrojev(n_train=4000, epochs=15, device=device)

    # Extract test features and ground truth
    x_test = np.array([c["features"] for c in BENCHMARK_CASES], dtype=np.float32)
    y_test = np.array([CLASS_TO_IDX[c["true_class"]] for c in BENCHMARK_CASES], dtype=np.int64)
    x_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    # 1. Local AstroJev Inference & Latency
    print("[AstroJev Benchmark] Running local AstroJev inference...")
    t_start = time.perf_counter()
    with torch.no_grad():
        local_out = local_model(x_tensor)
        local_probs = local_out["choice_probs"].cpu().numpy()
        local_preds = np.argmax(local_probs, axis=1)
        local_conf = np.max(local_probs, axis=1)
        local_sparsity = local_out["sparsity"]
    t_local = (time.perf_counter() - t_start) * 1000.0  # ms for all 8
    local_per_item_ms = t_local / len(BENCHMARK_CASES)

    # 2. Hosted TypeSafe Jev Inference
    print(f"[AstroJev Benchmark] Querying TypeSafe Jev API ({TYPESAFE_API_URL})...")
    jev_probs = []
    jev_preds = []
    jev_latencies = []
    total_in_tokens = 0
    total_out_tokens = 0

    for case in BENCHMARK_CASES:
        res = query_hosted_jev(case)
        pred_idx = CLASS_TO_IDX[res["predicted_class"]]
        jev_preds.append(pred_idx)
        jev_probs.append(res["probabilities"])
        jev_latencies.append(res["latency_ms"])
        total_in_tokens += res["input_tokens"]
        total_out_tokens += res["output_tokens"]

    jev_probs = np.array(jev_probs)
    jev_preds = np.array(jev_preds)

    # 3. Compute Metrics
    calib_local = evaluate_calibration(local_probs, y_test, n_bins=5)
    calib_jev = evaluate_calibration(jev_probs, y_test, n_bins=5)

    # Agreement and Disagreements
    agreements = int(np.sum(local_preds == jev_preds))
    disagreements = []
    for i, case in enumerate(BENCHMARK_CASES):
        disagreements.append({
            "id": case["id"],
            "name": case["name"],
            "true_class": case["true_class"],
            "astrojev": {
                "class": CLASSES[local_preds[i]],
                "confidence": float(local_conf[i]),
                "p_true": float(local_probs[i, y_test[i]]),
            },
            "hosted_jev": {
                "class": CLASSES[jev_preds[i]],
                "confidence": float(np.max(jev_probs[i])),
                "p_true": float(jev_probs[i, y_test[i]]),
            },
            "agree": bool(local_preds[i] == jev_preds[i]),
        })

    report = {
        "benchmark_cases": len(BENCHMARK_CASES),
        "astrojev": {
            "accuracy": calib_local["accuracy"],
            "brier_score": calib_local["brier"],
            "ece": calib_local["ece"],
            "mean_latency_ms": local_per_item_ms,
            "sparsity": local_sparsity,
            "cost_per_m_decisions": "$0.00 (Offline Edge Model)",
        },
        "hosted_jev": {
            "accuracy": calib_jev["accuracy"],
            "brier_score": calib_jev["brier"],
            "ece": calib_jev["ece"],
            "mean_latency_ms": float(np.mean(jev_latencies)),
            "cost_per_m_decisions": f"${(total_in_tokens / len(BENCHMARK_CASES) * 0.042):.3f} / MTok equivalent",
            "total_tokens_consumed": total_in_tokens + total_out_tokens,
        },
        "agreement_count": agreements,
        "agreement_pct": (agreements / len(BENCHMARK_CASES)) * 100.0,
        "comparisons": disagreements,
    }

    return report


if __name__ == "__main__":
    report = run_comparative_benchmark()
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD BENCHMARK: ASTROJEV (LOCAL ERET) vs HOSTED JEV (TYPESAFE)")
    print("=" * 70)
    print(f"{'Metric':<28} | {'AstroJev (Local ERET)':<20} | {'Hosted Jev (TypeSafe)':<20}")
    print("-" * 70)
    print(f"{'Top-1 Accuracy':<28} | {report['astrojev']['accuracy']*100:>18.1f}% | {report['hosted_jev']['accuracy']*100:>18.1f}%")
    print(f"{'Brier Score (lower=better)':<28} | {report['astrojev']['brier_score']:>20.4f} | {report['hosted_jev']['brier_score']:>20.4f}")
    print(f"{'Expected Calibration Error':<28} | {report['astrojev']['ece']*100:>18.2f}% | {report['hosted_jev']['ece']*100:>18.2f}%")
    print(f"{'Inference Latency':<28} | {report['astrojev']['mean_latency_ms']:>17.2f} ms | {report['hosted_jev']['mean_latency_ms']:>17.1f} ms")
    print(f"{'Cost / Million Decisions':<28} | {report['astrojev']['cost_per_m_decisions']:>20} | {report['hosted_jev']['cost_per_m_decisions']:>20}")
    print(f"{'Latent CReLU Sparsity':<28} | {report['astrojev']['sparsity']*100:>18.1f}% | {'N/A (Closed)':>20}")
    print("=" * 70)
    print(f"Agreement Rate: {report['agreement_pct']:.1f}% ({report['agreement_count']}/{report['benchmark_cases']})")
