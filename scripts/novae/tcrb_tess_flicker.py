"""TESS flickering of T CrB across accretion states (EXP-2026-Y).

Sectors (SPOC 120-s): S24+S25 (2020 Apr-Jun, super-active peak), S51 (2022 Apr-May, late SAP),
S78 (2024 May, recovery after the 2023 dip). For each sector:
  * SAP flux -> relative flux; slow trends (> 2 d) removed with a running median.
  * Absolute flicker amplitude: rms of the detrended flux on 10 min - 6 h timescales, in e-/s
    (the giant dominates the TESS band, so fractional amplitudes are diluted; absolute ones compare
    sectors directly because the giant's flux cancels).
  * Power spectrum (Lomb-Scargle, rms-normalised) fitted with a bending power law + white noise
    by maximum likelihood (Whittle); break frequency = characteristic inner-disk time scale.
  * Calibrated QPO test: the highest periodogram peak (and the power near the reported ~6000 s
    signal) is compared with 500 red-noise simulations drawn from the fitted continuum
    (Timmer & Koenig 1995), giving a global false-alarm probability across all trial frequencies.
Usage: python scripts/novae/tcrb_tess_flicker.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import median_filter  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TESS = ROOT / "data" / "tcrb" / "tess"
FIG = ROOT / "docs" / "figures" / "tcrb"
RESULTS = ROOT / "docs" / "results" / "experiment_y_tcrb_results.json"
warnings.filterwarnings("ignore")
STATE = {24: "2020 SAP peak", 25: "2020 SAP peak", 51: "2022 late SAP", 78: "2024 May post-dip recovery"}
COL = {24: "#c62828", 25: "#ef6c00", 51: "#6a1b9a", 78: "#1565c0"}


def load_sector(sector: int):
    from astropy.io import fits
    files = [p for p in TESS.rglob("*.fits") if f"s{sector:04d}" in p.name and "fast" not in p.name]
    with fits.open(files[0]) as h:
        d = h[1].data
        t, f, q = d["TIME"], d["SAP_FLUX"], d["QUALITY"]
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    return t[ok].astype(float), f[ok].astype(float)


def detrend(t, f, win_d=2.0):
    """Subtract a running median over win_d (on a regular 120-s grid, gaps filled by interpolation)."""
    dt = np.median(np.diff(t))
    n = max(3, int(round(win_d / dt)) | 1)
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    fi = np.interp(grid, t, f)
    trend = median_filter(fi, size=n, mode="nearest")
    return f - np.interp(t, grid, trend), np.interp(t, grid, trend)


def bandpass_rms(t, r, lo_s=600.0, hi_s=6 * 3600.0):
    """rms of r on time scales between lo_s and hi_s: smooth at lo_s, subtract a hi_s running mean."""
    dt = np.median(np.diff(t)) * 86400
    k_lo = max(1, int(lo_s / dt))
    k_hi = max(k_lo + 2, int(hi_s / dt))
    s_lo = np.convolve(r, np.ones(k_lo) / k_lo, mode="same")
    s_hi = np.convolve(r, np.ones(k_hi) / k_hi, mode="same")
    x = (s_lo - s_hi)[k_hi:-k_hi]
    return float(np.std(x))


def periodogram(t, r, fmin=1 / (2 * 86400), fmax=1 / 600.0, n=3000):
    from astropy.timeseries import LombScargle
    ts = (t - t[0]) * 86400
    freq = np.geomspace(fmin, fmax, n)
    p = LombScargle(ts, r, normalization="psd").power(freq)
    # convert to rms^2/Hz (approx; Lomb-Scargle 'psd' normalisation = |FT|^2 / N)
    T = ts[-1] - ts[0]
    psd = p * 2 * T / len(ts) ** 2 * len(ts) / T  # = 2 * power / N  (units flux^2 / Hz after * dt scaling)
    psd = psd * (np.median(np.diff(ts)))
    return freq, psd


def model(f, logA, logfb, a_hi, logC):
    return 10 ** logA / (1 + (f / 10 ** logfb) ** a_hi) + 10 ** logC


def fit_continuum(freq, psd):
    def nll(p):
        m = model(freq, *p)
        if p[2] < 0.5 or p[2] > 4.5 or not (np.log10(freq[0]) - 0.3 < p[1] < np.log10(freq[-1]) + 0.3):
            return 1e30  # break confined to (just beyond) the sampled band; at the edge = unconstrained
        return np.sum(np.log(m) + psd / m)
    p0 = [np.log10(np.median(psd[:50])), np.log10(3e-4), 2.0, np.log10(np.median(psd[-200:]))]
    best = None
    for fb in (-4.5, -4.0, -3.5, -3.0):
        p0[1] = fb
        r = minimize(nll, p0, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-5, "fatol": 1e-5})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def tk_sim(n, dt, pfun, rng):
    """Timmer & Koenig red-noise light curve with PSD pfun(f) on a regular grid (length n, step dt s)."""
    f = np.fft.rfftfreq(2 * n, dt)[1:]
    s = pfun(f)
    re = rng.normal(size=len(f)) * np.sqrt(s / 2)
    im = rng.normal(size=len(f)) * np.sqrt(s / 2)
    spec = np.concatenate([[0], re + 1j * im])
    x = np.fft.irfft(spec, n=2 * n)[:n]
    return x


def main(n_sim: int = 500):
    rng = np.random.default_rng(42)
    out = {}
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
    for sec in (24, 25, 51, 78):
        try:
            t, f = load_sector(sec)
        except IndexError:
            continue
        r, trend = detrend(t, f)
        rms = bandpass_rms(t, r)
        freq, psd = periodogram(t, r)
        p = fit_continuum(freq, psd)
        # calibrated max-peak test vs fitted continuum (global over all trial frequencies)
        ratio = psd / model(freq, *p)
        ratio /= np.median(ratio) / np.log(2)  # chi^2_2/2 has median ln 2
        k6000 = (freq > 1 / 7500) & (freq < 1 / 5000)
        obs_max, obs_6000 = ratio.max(), ratio[k6000].max()
        dt = np.median(np.diff(t)) * 86400
        grid = np.arange(t[0], t[-1], dt / 86400)
        sims_max, sims_6000 = [], []
        for _ in range(n_sim):
            x = tk_sim(len(grid), dt, lambda ff: model(ff, *p), rng)
            xs = np.interp(t, grid, x)  # same sampling and gaps as the data
            ff, ps = periodogram(t, xs - xs.mean())
            rr = ps / model(ff, *p)
            rr /= np.median(rr) / np.log(2)
            sims_max.append(rr.max())
            sims_6000.append(rr[k6000].max())
        sims_max, sims_6000 = np.array(sims_max), np.array(sims_6000)
        fap_global = float((sims_max >= obs_max).mean())
        fap_6000 = float((sims_6000 >= obs_6000).mean())
        fpk = float(freq[np.argmax(ratio)])
        out[str(sec)] = {"state": STATE[sec], "btjd": [float(t[0]), float(t[-1])], "median_flux_e_s": float(np.median(f)),
                         "flicker_rms_10min_6h_e_s": rms, "flicker_rms_frac": rms / float(np.median(f)),
                         "break_freq_hz": float(10 ** p[1]), "break_timescale_min": float(1 / 10 ** p[1] / 60),
                         "break_in_band": bool(np.log10(freq[0]) + 0.15 < p[1] < np.log10(freq[-1]) - 0.15),
                         "high_freq_slope": float(p[2]), "max_peak_period_s": 1 / fpk, "max_peak_ratio": float(obs_max),
                         "fap_global_max_peak": fap_global, "peak_ratio_5000_7500s": float(obs_6000),
                         "fap_5000_7500s": fap_6000}
        print(sec, json.dumps(out[str(sec)], default=float))
        c = COL[sec]
        ax = axs[0, 0] if sec in (24, 25) else axs[0, 1] if sec == 51 else axs[1, 0]
        tt = t - t[0]
        sel = tt < 6.0
        ax.plot(tt[sel], (r[sel]) / 1e3, ",", color=c, alpha=0.7)
        sm = np.convolve(r, np.ones(5) / 5, mode="same")
        ax.plot(tt[sel], sm[sel] / 1e3, "-", color=c, lw=0.5)
        ax.set_title(f"S{sec} ({STATE[sec]}): first 6 d, detrended", fontsize=9)
        ax.set_xlabel("days")
        ax.set_ylabel(r"$\Delta$flux (10$^3$ e$^-$/s)")
        ax.set_ylim(-4, 4)
        # PSD panel
        b = np.geomspace(freq[0], freq[-1], 60)
        idx = np.digitize(freq, b)
        fb_ = np.array([freq[idx == i].mean() for i in np.unique(idx)])
        pb = np.array([psd[idx == i].mean() for i in np.unique(idx)])
        axs[1, 1].loglog(fb_, pb * fb_, "o-", ms=2.5, lw=0.8, color=c, label=f"S{sec} {STATE[sec]}")
        axs[1, 1].loglog(freq, model(freq, *p) * freq, "--", lw=0.8, color=c)
    axs[1, 1].axvspan(1 / 7500, 1 / 5000, color="0.6", alpha=0.25, label="reported ~6000 s signal")
    axs[1, 1].set_xlabel("frequency (Hz)")
    axs[1, 1].set_ylabel(r"$f \times P(f)$ (e$^{-2}$ s$^{-2}$)")
    axs[1, 1].set_title("Flicker power spectra + bending-power-law fits", fontsize=9)
    axs[1, 1].legend(fontsize=7)
    fig.suptitle("T CrB flickering seen by TESS across accretion states", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "tcrb_tess_flicker.png", dpi=140)
    plt.close(fig)
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["tess_flicker"] = out
    RESULTS.write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
