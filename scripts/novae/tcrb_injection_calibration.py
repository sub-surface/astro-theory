"""Injection-recovery calibration of the T CrB accretion inversion (EXP-2026-Y).

Random synthetic accretion histories (quiescence + super-active phase + post-SAP level, random
reprocessing efficiencies eta_b) are pushed through the forward model with realistic sampling and
flicker noise, then inverted with the same pipeline. We record, for each accreted-mass window,
the standardised error z = (log M_true - mean log M_post) / sd log M_post from Laplace samples, and
the empirical coverage of the nominal 68/95% intervals. Output: data/tcrb/injection_calibration.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from celestrium.novae.tcrb import MSUN, YR, jd_to_year, year_to_jd  # noqa: E402
from celestrium.novae.inversion import (AccretionInversion, BandData, InversionConfig, WINDOWS_YR,  # noqa: E402
                                       accreted_mass, laplace_samples)

EPH = {"t0": 2455828.9, "period": 227.55}
BANDS = {"V": (700, 0.04), "B": (500, 0.08), "U": (120, 0.15), "UVM2": (150, 0.25)}


def make_truth(inv, rng):
    yrs = jd_to_year(inv.bin_start)
    q = 10 ** rng.uniform(-9.3, -8.5)
    sap0, sap1 = rng.uniform(2014.5, 2015.5), rng.uniform(2022.8, 2023.5)
    hi = 10 ** rng.uniform(-7.6, -6.7)
    post = 10 ** rng.uniform(math_log10(q), -7.5)
    shape = 1 + 0.5 * np.sin(2 * np.pi * (yrs - sap0) / rng.uniform(3, 9))
    tr = np.where((yrs > sap0) & (yrs < sap1), hi * shape, q)
    tr = np.where(yrs >= sap1 + 1.0, post, tr)
    tr = tr * np.exp(rng.normal(0, 0.15, len(tr)))
    return tr, q


def math_log10(x):
    return float(np.log10(x))


def run_one(seed, n_iter=700):
    rng = np.random.default_rng(seed)
    cfg = InversionConfig(n_iter=n_iter, seed=seed)
    stub = [BandData(b, np.array([year_to_jd(2010.0)]), np.array([10.0]), np.array([0.05])) for b in BANDS]
    gen = AccretionInversion(stub, EPH, cfg=cfg)
    truth, q = make_truth(gen, rng)
    g = MSUN / YR
    P = gen.init_params()
    with torch.no_grad():
        P["log_mdot"].copy_(torch.log(torch.tensor(truth * g)))
        P["log_mdot_q"].fill_(float(np.log(q * g)))
        P["ell"].copy_(torch.tensor([0.0, 0.0, 0.12, 0.0]))
        P["giant_v"].fill_(10.35 + rng.normal(0, 0.05))
        P["logit_eta"].fill_(float(rng.uniform(-6, 0)))
    data = []
    for b, (n, s) in BANDS.items():
        jd = np.sort(rng.uniform(year_to_jd(2005), year_to_jd(2026.7), n))
        m = gen.predict_mag(P, b, jd) + rng.normal(0, s, n)
        data.append(BandData(b, jd, m, np.full(n, 0.02)))
    true_m = accreted_mass(gen, P)
    inv = AccretionInversion(data, EPH, cfg=cfg)
    Pf, _ = inv.fit()
    samples = laplace_samples(inv, Pf, n_samples=150, seed=seed)
    post = [accreted_mass(inv, s) for s in samples]
    map_m = accreted_mass(inv, Pf)
    rec = {"seed": seed, "truth": true_m, "map": map_m, "z": {}, "ratio_map": {}}
    for k in list(WINDOWS_YR) + ["mdot_q"]:
        lp = np.log10(np.clip([p[k] for p in post], 1e-14, None))
        rec["z"][k] = float((np.log10(true_m[k]) - lp.mean()) / max(lp.std(), 1e-6))
        rec["ratio_map"][k] = float(map_m[k] / true_m[k])
        rec.setdefault("post_sd_dex", {})[k] = float(lp.std())
    return rec


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    out = ROOT / "data" / "tcrb" / "injection_calibration.json"
    recs = json.loads(out.read_text()) if out.exists() else []
    done = {r["seed"] for r in recs}
    for seed in range(n):
        if seed in done:
            continue
        t = time.time()
        recs.append(run_one(seed))
        out.write_text(json.dumps(recs, indent=1))
        r = recs[-1]
        print(seed, f"{time.time() - t:.0f}s", {k: round(v, 2) for k, v in r["ratio_map"].items()},
              "z:", {k: round(v, 2) for k, v in r["z"].items()}, flush=True)
    z = {k: np.array([r["z"][k] for r in recs]) for k in recs[0]["z"]}
    for k, v in z.items():
        print(f"{k:28s} cov68 {np.mean(np.abs(v) < 1):.2f}  cov95 {np.mean(np.abs(v) < 1.96):.2f}  "
              f"rms(z) {np.sqrt(np.mean(v ** 2)):.2f}  median MAP ratio {np.median([r['ratio_map'][k] for r in recs]):.2f}")
