"""Autonomous Active Inference & Multi-Wavelength Follow-up Agent.

Implements an active inference policy (Friston 2010, Angelopoulos & Bates 2023)
driven by AstroJev Dirichlet Evidential vacuity:
1. Sources with high evidence (Noul >= 0.85, u_epi <= 0.20) are automatically cataloged.
2. Sources with epistemic doubt (u_epi >= 0.50) trigger targeted multi-wavelength queries
   (MAST UV/optical, HEASARC X-ray, Gaia epoch astrometry) to maximize Bayesian information gain.
3. Sources with high contractive tension (delta_eq >= 0.15) trigger extended equilibrium deliberation.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from .astrojev import CLASSES, CLASS_TO_IDX, NUM_FEATURES, dirichlet_bald_information_gain
from .evidential_astrojev import EvidentialAstroJev


def triage_sources(
    features: np.ndarray,
    source_ids: Optional[List[str]] = None,
    coords: Optional[List[Tuple[float, float]]] = None,
    model: Optional[EvidentialAstroJev] = None,
    model_checkpoint: Optional[str | Path] = None,
    u_epi_threshold: float = 0.50,
    noul_min: float = 0.85,
    delta_eq_threshold: float = 0.15,
) -> Dict[str, Any]:
    """Triage astronomical sources into autonomous active inference action queues.

    Parameters:
        features: Array of shape (N, NUM_FEATURES).
        source_ids: Optional list of source identifiers.
        coords: Optional list of (ra, dec) coordinates.
        model: Optional pre-loaded EvidentialAstroJev instance.
        model_checkpoint: Optional checkpoint path.
        u_epi_threshold: Threshold above which epistemic follow-up is triggered.
        noul_min: Minimum epistemic confidence for auto-cataloging.
        delta_eq_threshold: Contractive equilibrium residual threshold for deliberation.

    Returns:
        Structured triage report containing action queues and information gain metrics.
    """
    n_sources = len(features)
    if n_sources == 0:
        return {
            "stats": {
                "total_triaged": 0, "auto_cataloged": 0,
                "followup_triggered": 0, "mast_queries": 0,
                "heasarc_queries": 0, "gaia_queries": 0,
                "deliberation_needed": 0,
            },
            "actions": [],
        }

    if source_ids is None:
        source_ids = [f"SRC_{i:06d}" for i in range(n_sources)]
    if coords is None:
        coords = [(float(features[i, 7] * 360.0), float(features[i, 8] * 180.0 - 90.0)) for i in range(n_sources)]

    # Load or initialize EvidentialAstroJev
    if model is None:
        model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=128, num_classes=4, n_iter=5)
        if model_checkpoint and Path(model_checkpoint).is_file():
            ckpt = torch.load(str(model_checkpoint), map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Batch forward pass
    feats_tensor = torch.from_numpy(features.astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(feats_tensor)
        probs = out["probs"].cpu().numpy()
        u_epi = out["u_epi"].cpu().numpy()
        noul = out["noul"].cpu().numpy()
        u_ale = out["u_ale"].cpu().numpy()
        delta_eq = out["delta_eq"].cpu().numpy()
        alpha = out["alpha"].cpu().numpy()
        bald_scores = dirichlet_bald_information_gain(alpha)

    auto_cataloged = []
    followup_triggered = []
    deliberations = []
    mast_count = 0
    heasarc_count = 0
    gaia_count = 0

    for i in range(n_sources):
        src_id = source_ids[i]
        ra, dec = coords[i]
        p_qso = float(probs[i, 0])
        top_cls_idx = int(np.argmax(probs[i]))
        top_cls = CLASSES[top_cls_idx]
        top_p = float(probs[i, top_cls_idx])
        ue = float(u_epi[i])
        nl = float(noul[i])
        ua = float(u_ale[i])
        de = float(delta_eq[i])

        # Feature diagnostics for targeted information gain:
        # Features: [phot_g, bp_rp, g_bp, w1, w1_w2, pm, pm_err, l, b, snr_flux]
        phot_g = float(features[i, 0])
        w1_w2 = float(features[i, 4])
        pm = float(features[i, 5])
        pm_err = float(features[i, 6])

        # 1. Action: AUTO_CATALOG
        # High evidence, high confidence cosmological tracer
        if top_cls == "Quasar_AGN" and top_p >= 0.85 and nl >= noul_min and ue <= 0.20:
            auto_cataloged.append({
                "source_id": src_id,
                "ra": ra,
                "dec": dec,
                "action": "AUTO_CATALOG",
                "class": top_cls,
                "p_quasar": p_qso,
                "noul": nl,
                "u_epi": ue,
                "reason": f"High evidential purity (Noul={nl:.2f}, p_qso={p_qso:.2f})",
            })
            continue

        # 2. Action: CONTRACTIVE_DELIBERATION
        # High equilibrium tension -> deliberative iteration needed
        if de >= delta_eq_threshold:
            deliberations.append({
                "source_id": src_id,
                "ra": ra,
                "dec": dec,
                "action": "CONTRACTIVE_DELIBERATION",
                "class_tentative": top_cls,
                "delta_eq": de,
                "recommendation": "Extend recurrence iterations K: 5 -> 15 to resolve fixed-point tension",
            })

        # 3. Action: MULTI_WAVELENGTH_QUERY (Active Inference Follow-up)
        # Model confesses high epistemic vacuity / doubt
        if ue >= u_epi_threshold:
            # Determine archive that maximizes Bayesian information gain:
            # Case A: Faint, ambiguous infrared color (W1 - W2 < 0.8) -> MAST UV/Optical spectroscopy
            # Case B: Strong IR color (W1 - W2 >= 0.8) but high PM error -> HEASARC X-ray detection
            # Case C: High PM error with stationary PM -> Gaia epoch astrometry
            if w1_w2 < 0.8 and phot_g <= 21.0:
                archive = "MAST"
                query_type = "Spectra / UV-Optical Cutout"
                expected_gain = 0.82
                mast_count += 1
            elif pm_err >= 1.5 and pm < 3.0:
                archive = "GAIA_EPOCH"
                query_type = "Astrometric Time-Series"
                expected_gain = 0.74
                gaia_count += 1
            else:
                archive = "HEASARC"
                query_type = "Chandra / XMM / eROSITA X-ray Cross-Match"
                expected_gain = 0.89
                heasarc_count += 1

            followup_triggered.append({
                "source_id": src_id,
                "ra": ra,
                "dec": dec,
                "action": "SCHEDULE_FOLLOWUP_QUERY",
                "archive": archive,
                "query_type": query_type,
                "u_epi": ue,
                "u_ale": ua,
                "noul": nl,
                "expected_info_gain": expected_gain,
                "bald_info_gain": float(bald_scores[i]),
                "tentative_class": top_cls,
                "phot_g": phot_g,
                "w1_w2": w1_w2,
            })

    stats = {
        "total_triaged": n_sources,
        "auto_cataloged": len(auto_cataloged),
        "followup_triggered": len(followup_triggered),
        "mast_queries": mast_count,
        "heasarc_queries": heasarc_count,
        "gaia_queries": gaia_count,
        "deliberation_needed": len(deliberations),
    }

    return {
        "stats": stats,
        "auto_cataloged": auto_cataloged,
        "followup_triggered": followup_triggered,
        "deliberations": deliberations,
    }


def triage_candidates_for_followup(
    candidates_ref: str,
    u_epi_threshold: float = 0.50,
    noul_min: float = 0.85,
    limit: int = 1000,
    model_checkpoint: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Triage candidate list or mock catalog using active inference agent."""
    from . import candidates

    cand_table = None
    if candidates_ref and candidates.exists(candidates_ref):
        cand_table = candidates.load(candidates_ref)

    if cand_table is not None:
        n = min(len(cand_table), limit)
        sub = cand_table[:n]
        src_ids = [str(r.get("source_id", r.get("id", f"CAND_{i}"))) for i, r in enumerate(sub)]
        ras = [float(r.get("ra", 0.0)) for r in sub]
        decs = [float(r.get("dec", 0.0)) for r in sub]
        coords = list(zip(ras, decs))

        # Build feature vector
        g = np.asarray([float(r.get("phot_g", 19.5)) for r in sub], dtype=np.float32)
        bp_rp = np.asarray([float(r.get("bp_rp", 0.8)) for r in sub], dtype=np.float32)
        w1 = np.asarray([float(r.get("w1", 14.5)) for r in sub], dtype=np.float32)
        w1_w2 = np.asarray([float(r.get("w1_w2", 0.7)) for r in sub], dtype=np.float32)
        pm = np.asarray([float(r.get("pm", 0.5)) for r in sub], dtype=np.float32)
        pm_err = np.asarray([float(r.get("pm_err", 0.2)) for r in sub], dtype=np.float32)

        features = np.column_stack([
            g, bp_rp, np.zeros_like(g), w1, w1_w2,
            pm, pm_err, np.full_like(g, 0.3), np.full_like(g, 0.7), np.full_like(g, 50.0)
        ])
    else:
        # Create synthetic evaluation batch representing the full ambiguity simplex
        rng = np.random.default_rng(42)
        n = min(limit, 200)
        src_ids = [f"SYNTH_{i:04d}" for i in range(n)]
        coords = [(float(rng.uniform(0, 360)), float(rng.uniform(-90, 90))) for _ in range(n)]

        # Group 1: 50% Clean Quasars (G ~ 19, W1-W2 ~ 1.0, PM ~ 0.1)
        # Group 2: 30% Ambiguous Faint (G ~ 21, W1-W2 ~ 0.5, PM ~ 1.5) -> High Epistemic Vacuity
        # Group 3: 20% Stars (G ~ 17, W1-W2 ~ 0.1, PM ~ 15.0)
        n_qso = int(n * 0.5)
        n_amb = int(n * 0.3)
        n_star = n - n_qso - n_amb

        f_qso = np.column_stack([
            rng.normal(19.0, 0.5, n_qso), rng.normal(0.6, 0.2, n_qso), np.zeros(n_qso),
            rng.normal(14.0, 0.5, n_qso), rng.normal(1.1, 0.15, n_qso),
            rng.exponential(0.2, n_qso), rng.uniform(0.1, 0.3, n_qso),
            rng.uniform(0, 1, n_qso), rng.uniform(0, 1, n_qso), np.full(n_qso, 60.0)
        ])
        f_amb = np.column_stack([
            rng.normal(20.8, 0.5, n_amb), rng.normal(1.2, 0.4, n_amb), np.zeros(n_amb),
            rng.normal(16.5, 0.8, n_amb), rng.normal(0.45, 0.2, n_amb),  # ambiguous color boundary
            rng.normal(1.8, 0.8, n_amb), rng.uniform(1.2, 2.5, n_amb),   # noisy PM
            rng.uniform(0, 1, n_amb), rng.uniform(0, 1, n_amb), np.full(n_amb, 15.0)
        ])
        f_star = np.column_stack([
            rng.normal(16.5, 1.0, n_star), rng.normal(1.5, 0.3, n_star), np.zeros(n_star),
            rng.normal(15.0, 1.0, n_star), rng.normal(0.05, 0.1, n_star), # stellar color
            rng.normal(12.0, 3.0, n_star), rng.uniform(0.2, 0.5, n_star), # significant PM
            rng.uniform(0, 1, n_star), rng.uniform(0, 1, n_star), np.full(n_star, 80.0)
        ])
        features = np.vstack([f_qso, f_amb, f_star])

    return triage_sources(
        features=features,
        source_ids=src_ids,
        coords=coords,
        model_checkpoint=model_checkpoint,
        u_epi_threshold=u_epi_threshold,
        noul_min=noul_min,
    )


def evaluate_truthrl_policy(
    triage_results: Dict[str, Any],
    ground_truth_labels: Dict[str, str],
    w_correct: float = 1.0,
    w_abstain: float = 0.0,
    w_hallucination: float = 2.0,
) -> Dict[str, Any]:
    """Evaluates TruthRL Ternary Decision Policy (Wei et al., Meta/UW 2025).

    Reward: R = w_correct * 1_{correct} + w_abstain * 1_{abstain} - w_hallucination * 1_{hallucination}
    Distinguishes:
      1. Correct Auto-Catalog: High confidence, correct cosmological tracer.
      2. Honest Abstention: Follow-up query triggered under high epistemic doubt (u_epi).
      3. Overconfident Hallucination: Wrong class asserted with high confidence without follow-up.
    """
    auto = triage_results.get("auto_cataloged", [])
    followup = triage_results.get("followup_triggered", [])
    delib = triage_results.get("deliberations", [])

    n_total = triage_results["stats"]["total_triaged"]
    if n_total == 0:
        return {"truthfulness_score": 0.0, "accuracy": 0.0, "abstention_rate": 0.0, "hallucination_rate": 0.0}

    # Sources that abstained and requested follow-up
    abstained_ids = {item["source_id"] for item in followup}.union({item["source_id"] for item in delib})

    n_correct = 0
    n_hallucination = 0
    n_abstain = len(abstained_ids)

    for item in auto:
        src_id = item["source_id"]
        true_cls = ground_truth_labels.get(src_id)
        if true_cls is not None:
            if item["class"] == true_cls:
                n_correct += 1
            else:
                n_hallucination += 1

    total_reward = (w_correct * n_correct) + (w_abstain * n_abstain) - (w_hallucination * n_hallucination)
    truthfulness_score = total_reward / float(n_total)

    return {
        "truthfulness_score": float(truthfulness_score),
        "n_total": n_total,
        "n_correct": n_correct,
        "n_abstain": n_abstain,
        "n_hallucination": n_hallucination,
        "accuracy_on_auto": float(n_correct / max(len(auto), 1)),
        "abstention_rate": float(n_abstain / n_total),
        "hallucination_rate": float(n_hallucination / n_total),
        "weights": {"w_correct": w_correct, "w_abstain": w_abstain, "w_hallucination": w_hallucination},
    }


class TelescopeQueueMDP:
    """Sequential Markov Decision Process for Autonomous Telescope Follow-up Scheduling.

    Formulation (Naghib et al. 2019, Neill et al. 2023):
      - State s: (current_pointing (ra, dec), remaining_time_min, current_airmass)
      - Action a: Slew to target i, configure instrument, integrate t_exp
      - Reward R(s, a): BALD_Information_Gain / (t_slew + t_exp)

    Maximizes cumulative Bayesian Information Gain within a fixed observing night budget.
    """

    def __init__(
        self,
        site_lat_deg: float = -24.627,
        time_budget_min: float = 360.0,
        base_exp_min: float = 5.0,
    ):
        self.site_lat_deg = site_lat_deg
        self.time_budget_min = time_budget_min
        self.base_exp_min = base_exp_min

    def estimate_exposure_time(self, g_mag: float) -> float:
        """Estimates required spectroscopic integration time based on apparent magnitude."""
        # Baseline 5 min at G=19; scaled exponentially, clamped between 2 and 45 min
        raw = self.base_exp_min * (10.0 ** (0.4 * (g_mag - 19.0)))
        return float(np.clip(raw, 2.0, 45.0))

    def angular_separation_deg(self, ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """Haversine angular separation between two sky coordinates."""
        d_ra = math.radians(ra2 - ra1)
        d_dec = math.radians(dec2 - dec1)
        a = (
            math.sin(d_dec / 2.0) ** 2
            + math.cos(math.radians(dec1)) * math.cos(math.radians(dec2)) * math.sin(d_ra / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(max(0.0, min(1.0, a))), math.sqrt(max(0.0, min(1.0, 1.0 - a))))
        return math.degrees(c)

    def schedule(
        self,
        candidates: List[Dict[str, Any]],
        initial_pointing: Tuple[float, float] = (0.0, -25.0),
    ) -> Dict[str, Any]:
        """Executes greedy knapsack policy to construct optimal observing sequence."""
        if not candidates:
            return {
                "schedule": [],
                "total_time_min": 0.0,
                "time_budget_min": self.time_budget_min,
                "total_bald_gain": 0.0,
                "targets_scheduled": 0,
            }

        remaining_time = self.time_budget_min
        current_ra, current_dec = initial_pointing
        pool = list(candidates)
        scheduled = []
        total_gain = 0.0

        while pool and remaining_time > 2.0:
            # Score each candidate by (BALD_gain + 0.5 * u_epi) / total_cost_min
            best_idx = -1
            best_efficiency = -1.0
            best_cost = 0.0
            best_t_exp = 0.0
            best_t_slew = 0.0

            for idx, cand in enumerate(pool):
                ra = cand.get("ra", 0.0)
                dec = cand.get("dec", 0.0)
                g_mag = cand.get("phot_g", 19.5)
                bald = cand.get("bald_info_gain", cand.get("expected_info_gain", 0.5))
                u_epi = cand.get("u_epi", 0.5)

                slew_deg = self.angular_separation_deg(current_ra, current_dec, ra, dec)
                t_slew = 0.5 + 0.03 * slew_deg  # 30 sec base + slew rate
                t_exp = self.estimate_exposure_time(g_mag)
                total_cost = t_slew + t_exp

                if total_cost <= remaining_time:
                    reward = bald + 0.5 * u_epi
                    efficiency = reward / total_cost
                    if efficiency > best_efficiency:
                        best_efficiency = efficiency
                        best_idx = idx
                        best_cost = total_cost
                        best_t_exp = t_exp
                        best_t_slew = t_slew

            if best_idx == -1:
                # No candidates fit in remaining time
                break

            chosen = pool.pop(best_idx)
            remaining_time -= best_cost
            bald_val = chosen.get("bald_info_gain", chosen.get("expected_info_gain", 0.5))
            total_gain += bald_val

            scheduled.append({
                "sequence": len(scheduled) + 1,
                "source_id": chosen.get("source_id", f"TARGET_{len(scheduled)}"),
                "ra": chosen.get("ra", 0.0),
                "dec": chosen.get("dec", 0.0),
                "phot_g": chosen.get("phot_g", 19.5),
                "archive": chosen.get("archive", "OPTICAL_SPEC"),
                "bald_gain": bald_val,
                "t_slew_min": round(best_t_slew, 2),
                "t_exp_min": round(best_t_exp, 2),
                "cumulative_time_min": round(self.time_budget_min - remaining_time, 2),
            })
            current_ra = chosen.get("ra", current_ra)
            current_dec = chosen.get("dec", current_dec)

        return {
            "schedule": scheduled,
            "total_time_min": round(self.time_budget_min - remaining_time, 2),
            "time_budget_min": self.time_budget_min,
            "remaining_budget_min": round(remaining_time, 2),
            "total_bald_gain": round(total_gain, 4),
            "targets_scheduled": len(scheduled),
            "targets_unobserved": len(pool),
        }


