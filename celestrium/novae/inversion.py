r"""
Accretion-history inversion for T CrB (EXP-2026-Y): light curves -> Mdot_tr(t) -> Mdot_acc(t).

Forward model (dereddening applied to the data, model unreddened), band b:
    F_b(t) = F_g,b E(phi) + F_disk,b[Sigma(t)] + kappa_b Mdot_acc(t)
  * F_g,b E(phi): M3III donor with ellipsoidal modulation (shared harmonic shape); its
    U/B fluxes follow fixed intrinsic M3III colours; UV red-leak giant terms are free.
  * F_disk,b: blackbody annuli of the linear viscous disk (celestrium.novae.tcrb.ViscousDisk).
  * kappa_b Mdot_acc: reprocessed light (nebular Balmer continuum, irradiation) assumed
    proportional to the accretion luminosity; kappa_b >= 0 free per band.
  * Sigma(t) = Mdot_q Sigma_ss + sum_k (Mdot_k - Mdot_q) G(t - t_k): LTI superposition
    of 60-d boxcar pulse responses, so one implicit solve serves every history.
Likelihood: Gaussian in magnitude, variance = sigma_obs^2 + s_b^2 with the per-band
flickering term s_b fitted (calibrated errors, not assumed).
Prior: Gaussian random walk on log Mdot_k (scale smooth_tau per bin).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from celestrium.novae.tcrb import (C_LIGHT, EXT_COEFF, F_LAMBDA_ZP, FIDUCIAL, GIANT_COLORS, H_PLANCK, K_B,
                             LAMBDA_EFF, MSUN, PC, SIGMA_SB, UVOT_FLUX_CONV, UVOT_ZP, YR, SystemParams,
                             ViscousDisk, jd_to_year, orbital_phase, year_to_jd)


@dataclass
class BandData:
    band: str
    jd: np.ndarray
    mag: np.ndarray
    err: np.ndarray


@dataclass
class InversionConfig:
    t_start_yr: float = 2003.0
    t_end_yr: float = 2026.80
    dt_d: float = 5.0
    bin_d: float = 60.0
    t_visc_out_yr: float = 3.0
    n_exp: float = 1.0
    smooth_tau: float = 0.35
    n_iter: int = 1500
    lr: float = 0.05
    seed: int = 0
    # Optional absolute anchor: (yr0, yr1, log10 Mdot_acc mean [Msun/yr], sd dex). Luna+ 2018
    # boundary-layer blackbody: Mdot >= 6.6e-8 (d/806 pc)^2 for M=1.2, R=5e8 cm -> 3.3e-8 for
    # our d = 887 pc, M = 1.35, R = 2.29e8 cm; we centre at 5e-8 with 0.3 dex.
    xray_anchor: tuple = None
    # v2 options -----------------------------------------------------------------
    # Red giant apparent V at ellipsoidal maximum, (mean, sd): Zamanov+ 2024 (2024AcA....74..265Z)
    # measured 10.17 +/- 0.06 from simultaneous photometry + spectroscopy. Breaks the
    # giant-vs-disk degeneracy that otherwise lets the giant absorb the disk's optical light.
    giant_v_prior: tuple = None
    # Inner-disk truncation: emission only from R > R_tr(t), R_tr = R_wd [1 + (Mdot_c / Mdot_acc)^a]
    # (evaporation-like: the inner disk recedes steeply as the accretion rate falls; a magnetospheric
    # Mdot^(-2/7) law only reaches a few R_wd, far inside the UV-emitting radii). Accretion inside
    # R_tr proceeds through an optically thin hot flow (the hard X-ray boundary layer), so the
    # quiescent disk can be UV-faint while still feeding the WD. Mdot_c and a are fitted.
    truncation: bool = False
    # v3: physical ranges that break the brightness <-> truncation degeneracy found in v2
    # (free index ran to a = 33, an on/off switch, with Mdot pinned at 7e-7 Msun/yr):
    # truncation index a ~ N(0.8, 0.35) (evaporation / magnetic laws give ~0.3-1.5) and
    # reprocessed fraction log10 eta ~ N(-1, 0.5), so accretion power cannot simply vanish.
    physical_priors: bool = False


T_REPROCESS = 1.5e4   # K, colour temperature of reprocessed / nebular light
ETA_MAX = 0.5         # max fraction of L_acc re-emitted as optical/UV reprocessed light


def kappa_unit(band: str, p: SystemParams) -> float:
    """F_lambda (erg/s/cm^2/A) per (g/s) accreted if ALL of L_acc = G M Mdot / R_wd were
    re-emitted as a T_REPROCESS blackbody, isotropically, at distance d."""
    lam = LAMBDA_EFF[band] * 1e-8
    x = H_PLANCK * C_LIGHT / (lam * K_B * T_REPROCESS)
    frac_per_A = math.pi * 2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / math.expm1(x) / (SIGMA_SB * T_REPROCESS ** 4) * 1e-8
    gm_r = 6.674e-8 * p.m_wd * MSUN / p.r_wd
    return gm_r / (4 * math.pi * (p.dist_pc * PC) ** 2) * frac_per_A


def _zp(band: str) -> float:
    """F_lambda of a 0-mag (Vega) source in this band."""
    return F_LAMBDA_ZP[band] if band in F_LAMBDA_ZP else UVOT_FLUX_CONV[band] * 10 ** (0.4 * UVOT_ZP[band])


class AccretionInversion:
    def __init__(self, data: Sequence[BandData], ephem: Dict[str, float], params: SystemParams = FIDUCIAL,
                 cfg: InversionConfig = InversionConfig()):
        self.p, self.cfg, self.ephem = params, cfg, ephem
        self.disk = ViscousDisk(params, cfg.t_visc_out_yr, cfg.n_exp)
        jd0, jd1 = float(year_to_jd(cfg.t_start_yr)), float(year_to_jd(cfg.t_end_yr))
        self.n_t = int(math.ceil((jd1 - jd0) / cfg.dt_d)) + 1
        self.t_grid = jd0 + cfg.dt_d * np.arange(self.n_t)
        self.n_bins = int(math.ceil((jd1 - jd0) / cfg.bin_d))
        self.bin_start = jd0 + cfg.bin_d * np.arange(self.n_bins)
        self.steps_per_bin = int(round(cfg.bin_d / cfg.dt_d))
        f64 = dict(dtype=torch.float64)
        self.G = torch.tensor(self.disk.pulse_response(cfg.dt_d, self.n_t, cfg.bin_d), **f64)
        self.sig_ss = torch.tensor(self.disk.steady_state(), **f64)
        self.nu = torch.tensor(self.disk.nu, **f64)
        self.om2 = torch.tensor(self.disk.omega ** 2, **f64)
        self.area = torch.tensor(self.disk.grid.area, **f64)
        self.r = torch.tensor(self.disk.grid.r, **f64)
        self.inner = torch.tensor(self.disk.inner_flux, **f64)
        self.data = [d for d in data if len(d.jd)]
        self.bands = [d.band for d in self.data]
        self._obs = []
        for d in self.data:
            x = (np.asarray(d.jd) - jd0) / cfg.dt_d
            ok = (x >= 0) & (x < self.n_t - 1)
            i0 = np.floor(x[ok]).astype(int)
            self._obs.append({
                "i0": torch.tensor(i0), "w": torch.tensor(x[ok] - i0, **f64),
                "ph": torch.tensor(orbital_phase(d.jd[ok], ephem["t0"], ephem["period"]), **f64),
                "mag": torch.tensor(d.mag[ok] - EXT_COEFF[d.band] * params.ebv, **f64),
                "err": torch.tensor(d.err[ok], **f64), "jd": np.asarray(d.jd)[ok]})

    # -- forward model ----------------------------------------------------------
    def sigma_t(self, mdot_bins, mdot_q):
        """Sigma(t, R) on the time grid via causal superposition (vectorised as a convolution)."""
        delta = mdot_bins - mdot_q
        # impulse train on the fine grid, convolved with G along time
        train = torch.zeros(self.n_t, dtype=torch.float64)
        idx = torch.arange(self.n_bins) * self.steps_per_bin
        keep = idx < self.n_t
        train = train.index_put((idx[keep],), delta[keep])
        n = 2 * self.n_t
        ft = torch.fft.rfft(train, n=n)
        fg = torch.fft.rfft(self.G, n=n, dim=0)
        conv = torch.fft.irfft(ft[:, None] * fg, n=n, dim=0)[: self.n_t]
        return mdot_q * self.sig_ss[None, :] + conv

    def band_flux(self, sig, lam_A: float, r_tr=None):
        t4 = (9.0 / 8.0) * self.nu * torch.clamp(sig, min=0) * self.om2 / SIGMA_SB
        T = torch.clamp(t4, min=1.0) ** 0.25
        lam = lam_A * 1e-8
        x = torch.clamp(H_PLANCK * C_LIGHT / (lam * K_B * T), 1e-6, 700.0)
        b = 2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / torch.expm1(x)
        if r_tr is not None:  # smooth inner cut-off of the emitting disk
            b = b / (1 + (r_tr[:, None] / self.r[None, :]) ** 8)
        d = self.p.dist_pc * PC
        return (b * self.area).sum(-1) * math.cos(math.radians(self.p.incl_deg)) / d ** 2 * 1e-8

    def r_trunc(self, P, macc):
        """Truncation radius (cm) on the time grid, or None when truncation is off."""
        if not self.cfg.truncation:
            return None
        a = 0.2 + torch.nn.functional.softplus(P["tr_a"])
        ratio = torch.exp(P["log_mdot_c"]) / torch.clamp(macc, min=1e-30)
        return self.p.r_wd * (1 + torch.clamp(ratio, max=1e6) ** a)

    def init_params(self, mdot_q0: float = 2e-9, giant_v: float = 10.35):
        g = MSUN / YR
        f64 = dict(dtype=torch.float64)
        P = {"log_mdot": torch.full((self.n_bins,), math.log(mdot_q0 * g), **f64),
             "log_mdot_q": torch.tensor(math.log(mdot_q0 * g), **f64),
             "giant_v": torch.tensor(giant_v, **f64),
             "ell": torch.zeros(4, **f64),
             # ONE reprocessing efficiency for all bands: reprocessed light has a single
             # spectrum. Per-band eta (tried first) can mimic the disk SED and makes the
             # absolute Mdot unidentifiable (injection seeds 2, 5 recovered at 0.15-0.21x).
             "logit_eta": torch.tensor(-3.0, **f64)}
        if self.cfg.truncation:
            P["log_mdot_c"] = torch.tensor(math.log(3e-8 * g), **f64)
            P["tr_a"] = torch.tensor(0.54, **f64)  # softplus(0.54) + 0.2 = 1.2
        for b in self.bands:
            P[f"log_jit_{b}"] = torch.tensor(math.log(0.1), **f64)
            if b.startswith("UV"):
                P[f"log_giant_{b}"] = torch.tensor(math.log(0.2 * _zp(b) * 10 ** (-0.4 * 15.0)), **f64)
        return {k: v.requires_grad_(True) for k, v in P.items()}

    def giant_flux(self, P, band: str, ph):
        a = 2 * math.pi * ph
        e = P["ell"]
        mod = 10 ** (-0.4 * (e[0] * torch.cos(a) + e[1] * torch.sin(a) + e[2] * torch.cos(2 * a) + e[3] * torch.sin(2 * a)))
        if band.startswith("UV"):
            return torch.exp(P[f"log_giant_{band}"]) * mod
        m = P["giant_v"] + {"V": 0.0, "B": GIANT_COLORS["B-V"], "U": GIANT_COLORS["B-V"] + GIANT_COLORS["U-B"]}[band]
        return F_LAMBDA_ZP[band] * 10 ** (-0.4 * m) * mod

    def kappa(self, P, band: str):
        return ETA_MAX * torch.sigmoid(P["logit_eta"]) * kappa_unit(band, self.p)

    def model_mag_at(self, P, band, i0, w, ph, sig=None, macc=None):
        if sig is None:
            sig = self.sigma_t(torch.exp(P["log_mdot"]), torch.exp(P["log_mdot_q"]))
        if macc is None:
            macc = sig @ self.inner
        fd = self.band_flux(sig, LAMBDA_EFF[band], self.r_trunc(P, macc))
        lerp = lambda y: y[i0] * (1 - w) + y[i0 + 1] * w  # noqa: E731
        f = self.giant_flux(P, band, ph) + lerp(fd) + self.kappa(P, band) * lerp(macc)
        return -2.5 * torch.log10(f / _zp(band))

    def neg_log_post(self, P):
        sig = self.sigma_t(torch.exp(P["log_mdot"]), torch.exp(P["log_mdot_q"]))
        macc = sig @ self.inner
        nll = 0.0
        for d, o in zip(self.data, self._obs):
            mm = self.model_mag_at(P, d.band, o["i0"], o["w"], o["ph"], sig, macc)
            var = o["err"] ** 2 + torch.exp(2 * P[f"log_jit_{d.band}"])
            nll = nll + 0.5 * torch.sum((o["mag"] - mm) ** 2 / var + torch.log(var))
        lm = P["log_mdot"]
        prior = 0.5 * torch.sum((lm[1:] - lm[:-1]) ** 2) / self.cfg.smooth_tau ** 2
        if self.cfg.xray_anchor is not None:
            y0, y1, mu, sd = self.cfg.xray_anchor
            tyr = jd_to_year(self.t_grid)
            w = torch.tensor((tyr >= y0) & (tyr < y1))
            mean = macc[w].mean() / (MSUN / YR)
            prior = prior + 0.5 * ((torch.log10(mean) - mu) / sd) ** 2
        prior = prior + 0.5 * ((lm[0] - P["log_mdot_q"]) / 0.5) ** 2 + 0.5 * torch.sum(P["ell"] ** 2) / 0.3 ** 2
        if self.cfg.giant_v_prior is not None:
            mu, sd = self.cfg.giant_v_prior
            prior = prior + 0.5 * ((self.giant_v_at_max(P) - mu) / sd) ** 2
        if self.cfg.physical_priors:
            prior = prior + 0.5 * ((torch.log10(ETA_MAX * torch.sigmoid(P["logit_eta"])) + 1.0) / 0.5) ** 2
            if self.cfg.truncation:
                a = 0.2 + torch.nn.functional.softplus(P["tr_a"])
                prior = prior + 0.5 * ((a - 0.8) / 0.35) ** 2
        if self.cfg.truncation:  # weak prior: Mdot_c within 1e-10 .. 1e-6 Msun/yr
            lc = (P["log_mdot_c"] - math.log(MSUN / YR)) / math.log(10)
            prior = prior + 0.5 * ((lc + 8.0) / 1.0) ** 2
        return nll + prior

    def giant_v_at_max(self, P):
        """Apparent (reddened) giant V at the brightest ellipsoidal phase."""
        a = torch.linspace(0, 2 * math.pi, 129, dtype=torch.float64)[:-1]
        e = P["ell"]
        mod = e[0] * torch.cos(a) + e[1] * torch.sin(a) + e[2] * torch.cos(2 * a) + e[3] * torch.sin(2 * a)
        return P["giant_v"] + mod.min() + EXT_COEFF["V"] * self.p.ebv

    def _data_init(self):
        """Giant V from the faint envelope of V (disk adds light, so the giant is the floor)."""
        for d, o in zip(self.data, self._obs):
            if d.band == "V":
                return float(np.percentile(o["mag"].numpy(), 97)) - 0.05 + EXT_COEFF["V"] * 0.0
        return 10.35

    def fit(self, P=None, verbose: bool = False, start_levels=(2e-9, 3e-8)):
        """Multi-start MAP: one start per constant Mdot level; returns the best posterior.

        A single flat low-Mdot start can stall in the faint-disk basin (found by the
        injection tests), so we always also start from a high-state level.
        """
        if P is not None:
            return self._fit_one(P, verbose)
        best = None
        for lvl in start_levels:
            Pi, L = self._fit_one(self.init_params(mdot_q0=lvl, giant_v=self._data_init()), verbose)
            if verbose:
                print(f"  start {lvl:.0e}: -logpost {L:.2f}")
            if best is None or L < best[1]:
                best = (Pi, L)
        return best

    def _fit_one(self, P, verbose: bool = False):
        """Stage 1: Adam with flicker terms frozen at 0.1 mag; stage 2: L-BFGS on everything."""
        torch.manual_seed(self.cfg.seed)
        jit = [k for k in P if k.startswith("log_jit")]
        free1 = [v for k, v in P.items() if k not in jit]
        opt = torch.optim.Adam(free1, lr=self.cfg.lr)
        for it in range(self.cfg.n_iter):
            opt.zero_grad()
            loss = self.neg_log_post(P)
            loss.backward()
            opt.step()
            if verbose and it % 250 == 0:
                print(f"  adam {it:5d}  -logpost {float(loss):.2f}")
        lb = torch.optim.LBFGS(list(P.values()), lr=1.0, max_iter=400, history_size=50,
                               line_search_fn="strong_wolfe", tolerance_grad=1e-7, tolerance_change=1e-10)

        def closure():
            lb.zero_grad()
            l = self.neg_log_post(P)
            l.backward()
            return l
        for rnd in range(3):
            loss = lb.step(closure)
            if verbose:
                print(f"  lbfgs round {rnd}  -logpost {float(loss):.2f}")
        L = float(self.neg_log_post(P))
        return P, (L if math.isfinite(L) else float("inf"))

    @torch.no_grad()
    def summarize(self, P) -> Dict[str, np.ndarray]:
        g = MSUN / YR
        sig = self.sigma_t(torch.exp(P["log_mdot"]), torch.exp(P["log_mdot_q"]))
        macc = (sig @ self.inner).numpy()
        out = {"t_yr": jd_to_year(self.t_grid), "mdot_acc": macc / g,
               "m_disk": (sig * self.area).sum(-1).numpy() / MSUN,
               "bin_yr": jd_to_year(self.bin_start), "mdot_tr": torch.exp(P["log_mdot"]).numpy() / g,
               "mdot_q": float(torch.exp(P["log_mdot_q"])) / g,
               "jitter": {b: float(torch.exp(P[f"log_jit_{b}"])) for b in self.bands},
               "eta": float(ETA_MAX * torch.sigmoid(P["logit_eta"])),
               "giant_v": float(P["giant_v"]), "ell": P["ell"].numpy().tolist()}
        rtr = self.r_trunc(P, sig @ self.inner)
        for b in ("U", "B", "V", "UVM2"):
            out[f"disk_flux_{b}"] = self.band_flux(sig, LAMBDA_EFF[b], rtr).numpy()
        out["r_trunc_cm"] = None if rtr is None else rtr.numpy()
        if self.cfg.truncation:
            out["mdot_c"] = float(torch.exp(P["log_mdot_c"])) / g
            out["tr_a"] = float(0.2 + torch.nn.functional.softplus(P["tr_a"]))
        out["giant_v_at_max_app"] = float(self.giant_v_at_max(P))
        out["sigma"] = sig.numpy()
        return out

    @torch.no_grad()
    def predict_mag(self, P, band: str, jd) -> np.ndarray:
        x = np.clip((np.asarray(jd, float) - self.t_grid[0]) / self.cfg.dt_d, 0, self.n_t - 1.001)
        i0 = torch.tensor(np.floor(x).astype(int))
        w = torch.tensor(x - np.floor(x), dtype=torch.float64)
        ph = torch.tensor(orbital_phase(jd, self.ephem["t0"], self.ephem["period"]), dtype=torch.float64)
        return self.model_mag_at(P, band, i0, w, ph).numpy() + EXT_COEFF[band] * self.p.ebv


# ---------------------------------------------------------------------------
# Laplace posterior & derived quantities
# ---------------------------------------------------------------------------
def _flatten(P):
    keys = list(P.keys())
    shapes = [P[k].shape for k in keys]
    vec = torch.cat([P[k].detach().reshape(-1) for k in keys])
    return keys, shapes, vec


def _unflatten(keys, shapes, vec):
    out, i = {}, 0
    for k, s in zip(keys, shapes):
        n = int(np.prod(s)) if len(s) else 1
        out[k] = vec[i:i + n].reshape(s)
        i += n
    return out


def laplace_samples(inv: "AccretionInversion", P, n_samples: int = 200, seed: int = 0, ridge: float = 1e-6):
    """Gaussian (Laplace) posterior around the MAP; returns list of parameter dicts."""
    keys, shapes, v0 = _flatten(P)
    f = lambda v: inv.neg_log_post(_unflatten(keys, shapes, v))  # noqa: E731
    H = torch.autograd.functional.hessian(f, v0)
    H = 0.5 * (H + H.T)
    evals, evecs = torch.linalg.eigh(H)
    evals = torch.clamp(evals, min=ridge * float(evals.abs().max()))
    cov_sqrt = evecs @ torch.diag(evals ** -0.5)
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n_samples, len(v0), generator=g, dtype=torch.float64)
    return [_unflatten(keys, shapes, v0 + cov_sqrt @ zi) for zi in z]


WINDOWS_YR = {"quiescence_2005_2015.0": (2005.0, 2015.0), "high_state_2015.0_2023.3": (2015.0, 2023.3),
              "post_2023.3_2026.7": (2023.3, 2026.7), "total_2005_2026.7": (2005.0, 2026.7)}


@torch.no_grad()
def accreted_mass(inv: "AccretionInversion", P, windows=WINDOWS_YR) -> Dict[str, float]:
    """Mass accreted onto the WD (Msun) within each window, from Mdot_acc(t)."""
    sig = inv.sigma_t(torch.exp(P["log_mdot"]), torch.exp(P["log_mdot_q"]))
    macc = (sig @ inv.inner).numpy() / (MSUN / YR)
    t = jd_to_year(inv.t_grid)
    out = {}
    for k, (a, b) in windows.items():
        s = (t >= a) & (t < b)
        out[k] = float(np.trapezoid(macc[s], t[s])) if s.sum() > 1 else float("nan")
    out["mdot_q"] = float(torch.exp(P["log_mdot_q"])) / (MSUN / YR)
    return out


@torch.no_grad()
def mdot_acc_curve(inv: "AccretionInversion", P) -> np.ndarray:
    sig = inv.sigma_t(torch.exp(P["log_mdot"]), torch.exp(P["log_mdot_q"]))
    return (sig @ inv.inner).numpy() / (MSUN / YR)
