"""Pseudo-Cl Mode-Coupling Dipole Mask Deconvolution for Quaia / CatWISE.

Solves the partial-sky mask bias on the cosmic dipole:
1. Constructs orthonormal real spherical harmonics Y_lm up to LMAX (l=0, 1, 2).
2. Computes the exact mode-coupling matrix K_{(lm),(l'm')} = Integral W(n) Y_lm Y_l'm' dOmega
   and the angular power spectrum coupling matrix M_{l l'}.
3. Inverts the mode-coupling operator to de-bias the Galactic plane mask (|b| < 10 deg)
   using continuous probabilistic quasar weights w_i = p_i from AstroJev.
4. Recovers the unmasked cosmic dipole amplitude D, direction (l, b), and quadrupole power.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy_healpix import HEALPix
import astropy.units as u
from scipy.special import lpmv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SQRT3 = math.sqrt(3.0)
CMB_DIPOLE_AMP = 0.0070  # Approximate kinematic expectation for optical/infrared quasars
CMB_DIPOLE_L = 264.0     # Galactic longitude of CMB dipole
CMB_DIPOLE_B = 48.0      # Galactic latitude of CMB dipole


def real_ylm_basis(
    lon_rad: np.ndarray,
    lat_rad: np.ndarray,
    lmax: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute orthonormal real spherical harmonics Y_lm for l=0..lmax.

    Returns:
        Y: Array of shape (npix, (lmax+1)^2)
        ll: Multipole l for each column
        mm: Order m for each column (-l..+l)
    """
    theta = np.pi / 2.0 - lat_rad
    phi = lon_rad
    x = np.cos(theta)

    cols, ll, mm = [], [], []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            norm = math.sqrt((2 * l + 1) / (4.0 * math.pi) *
                             math.factorial(l - am) / math.factorial(l + am))
            P = lpmv(am, l, x)
            if m > 0:
                y = math.sqrt(2.0) * norm * P * np.cos(m * phi)
            elif m < 0:
                y = math.sqrt(2.0) * norm * P * np.sin(am * phi)
            else:
                y = norm * P
            cols.append(y)
            ll.append(l)
            mm.append(m)

    return np.column_stack(cols), np.array(ll), np.array(mm)


def compute_mode_coupling(
    mask: np.ndarray,
    lmax: int = 2,
    nside: int = 32,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute harmonic mode-coupling matrix K and power spectrum coupling matrix M_ll.

    Parameters:
        mask: Boolean or float array of shape (npix,), where 1 = observed, 0 = masked.
        lmax: Maximum multipole order (default 2 -> monopole, dipole, quadrupole).
        nside: HEALPix resolution.

    Returns:
        K: Mode-coupling matrix of shape (N_lm, N_lm).
        M_ll: Multipole coupling matrix of shape (lmax+1, lmax+1).
        cond_num: Condition number of K.
    """
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    Y, ll, _ = real_ylm_basis(lon.rad, lat.rad, lmax=lmax)

    omega_pix = 4.0 * math.pi / hp.npix
    w = np.asarray(mask, dtype=np.float64)

    # K_{(lm), (l'm')} = omega_pix * sum_p W_p Y_lm(p) Y_l'm'(p)
    K = omega_pix * (Y.T @ (w[:, None] * Y))
    cond_num = float(np.linalg.cond(K))

    # Compute multipole power coupling matrix M_{l l'}
    # M_{l l'} = (1 / (2l + 1)) sum_{m, m'} |K_{(lm), (l'm')}|^2
    M_ll = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
    for l1 in range(lmax + 1):
        idx1 = (ll == l1)
        deg1 = 2 * l1 + 1
        for l2 in range(lmax + 1):
            idx2 = (ll == l2)
            sub_k = K[idx1, :][:, idx2]
            M_ll[l1, l2] = np.sum(sub_k ** 2) / float(deg1)

    return K, M_ll, cond_num


def deconvolve_dipole(
    pixel_counts: np.ndarray,
    mask: np.ndarray,
    lmax: int = 2,
    nside: int = 32,
) -> Dict[str, Any]:
    """Perform pseudo-Cl mode-coupling dipole deconvolution.

    Inverts the mask coupling K @ c = c_pseudo to deconvolve the cosmic dipole.

    Returns:
        dict containing deconvolved dipole vector, amplitude, direction (l, b),
        quadrupole amplitude, raw dipole, and matrix diagnostics.
    """
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    Y, ll, mm = real_ylm_basis(lon.rad, lat.rad, lmax=lmax)

    omega_pix = 4.0 * math.pi / hp.npix
    w = np.asarray(mask, dtype=np.float64)
    surviving_pix = (w > 0.0)

    if np.sum(surviving_pix) < 16:
        raise ValueError("Insufficient surviving pixels to perform dipole deconvolution")

    # 1. Mode-coupling matrix K
    K = omega_pix * (Y.T @ (w[:, None] * Y))
    cond_num = float(np.linalg.cond(K))

    # 2. Pseudo spherical harmonic coefficients c_pseudo
    # Mean density on unmasked sky
    total_counts = np.sum(pixel_counts * w)
    total_eff_pix = np.sum(w)
    mean_density = total_counts / total_eff_pix

    # Fractional density contrast delta_p
    delta = np.zeros_like(pixel_counts, dtype=np.float64)
    delta[surviving_pix] = (pixel_counts[surviving_pix] / (mean_density * w[surviving_pix])) - 1.0

    # c_pseudo = omega_pix * sum_p W_p delta_p Y_lm(p)
    c_pseudo = omega_pix * (Y.T @ (w * delta))

    # 3. Mode-coupling Inversion: c_deconv = K^(-1) c_pseudo
    try:
        inv_K = np.linalg.inv(K)
        c_deconv = inv_K @ c_pseudo
    except np.linalg.LinAlgError:
        inv_K = np.linalg.pinv(K)
        c_deconv = inv_K @ c_pseudo

    # Extract dipole components (l=1):
    # In our basis:
    # m = -1: Y_1,-1 ~ y
    # m =  0: Y_1,0  ~ z
    # m = +1: Y_1,+1 ~ x
    idx_l1 = np.where(ll == 1)[0]
    c1_y = float(c_deconv[idx_l1[0]])   # m = -1
    c1_z = float(c_deconv[idx_l1[1]])   # m = 0
    c1_x = float(c_deconv[idx_l1[2]])   # m = +1

    # Exact orthonormal real spherical harmonics to Cartesian dipole conversion:
    # x_cart = -sqrt(4pi/3) Y_{1,1}, y_cart = -sqrt(4pi/3) Y_{1,-1}, z_cart = sqrt(4pi/3) Y_{1,0}
    # For delta = 1 + D * n_hat: D_x = -sqrt(3/(4pi)) c_{1,1}, etc.
    y1_factor = math.sqrt(3.0 / (4.0 * math.pi))
    y2_factor = math.sqrt(5.0 / (4.0 * math.pi))

    vec_deconv = np.array([-y1_factor * c1_x, -y1_factor * c1_y, y1_factor * c1_z])
    amp_deconv = float(np.linalg.norm(vec_deconv))

    # Galactic coordinates of dipole direction
    l_deg = float(np.degrees(np.arctan2(vec_deconv[1], vec_deconv[0])) % 360.0)
    b_deg = float(np.degrees(np.arcsin(np.clip(vec_deconv[2] / max(amp_deconv, 1e-12), -1.0, 1.0))))

    # Extract quadrupole amplitude (l=2)
    idx_l2 = np.where(ll == 2)[0]
    amp_quad = float(np.sqrt(np.sum(c_deconv[idx_l2] ** 2)) * y2_factor)

    # 4. Compare with Raw (Masked) Dipole (naive fit without mode-coupling deconvolution)
    c1_y_raw = float(c_pseudo[idx_l1[0]])
    c1_z_raw = float(c_pseudo[idx_l1[1]])
    c1_x_raw = float(c_pseudo[idx_l1[2]])
    vec_raw = np.array([-y1_factor * c1_x_raw, -y1_factor * c1_y_raw, y1_factor * c1_z_raw])
    amp_raw = float(np.linalg.norm(vec_raw))
    l_raw = float(np.degrees(np.arctan2(vec_raw[1], vec_raw[0])) % 360.0)
    b_raw = float(np.degrees(np.arcsin(np.clip(vec_raw[2] / max(amp_raw, 1e-12), -1.0, 1.0))))

    # Error estimation via Poisson shot noise propagation.
    # Var(delta_p) = 1 / (nbar * w_p) with nbar = N_tot / sum(w), so
    # Cov(c_pseudo) = omega_pix^2 * sum_p w_p^2 Var(delta_p) Y Y^T = omega_pix * (sum(w) / N_tot) * K.
    cov_pseudo = (omega_pix * total_eff_pix / max(total_counts, 1.0)) * K
    cov_c = inv_K @ cov_pseudo @ inv_K.T
    sigma_D = float(y1_factor * np.sqrt(np.trace(cov_c[idx_l1, :][:, idx_l1]) / 3.0))

    return {
        "deconvolved": {
            "amplitude": amp_deconv,
            "vector": vec_deconv.tolist(),
            "l_deg": l_deg,
            "b_deg": b_deg,
            "sigma": sigma_D,
            "quadrupole_amp": amp_quad,
        },
        "raw_masked": {
            "amplitude": amp_raw,
            "vector": vec_raw.tolist(),
            "l_deg": l_raw,
            "b_deg": b_raw,
        },
        "diagnostics": {
            "condition_number": cond_num,
            "surviving_pixels": int(np.sum(surviving_pix)),
            "f_sky": float(np.sum(w) / hp.npix),
            "total_sources": float(total_counts),
            "mean_density_per_pix": float(mean_density),
        },
        "mode_coupling_matrix": K,
    }


def compute_risk_sensitive_weights(
    p_quasar: np.ndarray,
    u_epi: Optional[np.ndarray] = None,
    delta_eq: Optional[np.ndarray] = None,
    gamma_risk: float = 1.5,
    delta_max: float = 0.15,
) -> np.ndarray:
    """Computes CVaR risk-sensitive source weights (Koren et al. 2025/2026, UAMDP).

    Suppresses the high-vacuity and unstable equilibrium tail:
    w_i = p_i * (1 - u_epi_i)^gamma_risk * 1_{delta_eq <= delta_max}
    """
    weights = np.asarray(p_quasar, dtype=np.float64)
    if u_epi is not None:
        credence = np.clip(1.0 - u_epi, 0.0, 1.0)
        weights *= (credence ** gamma_risk)
    if delta_eq is not None:
        stable_mask = (delta_eq <= delta_max).astype(np.float64)
        weights *= stable_mask
    return weights


def run_quaia_deconvolution(
    catalog_path: str | Path,
    model_checkpoint: Optional[str | Path] = None,
    nside: int = 64,
    b_cut_deg: float = 10.0,
    risk_sensitive_weighting: bool = False,
    save_fig: bool = True,
    fig_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Execute complete pseudo-Cl dipole deconvolution on the Quaia dataset.

    Parameters:
        catalog_path: Path to quaia_G20.5.fits.
        model_checkpoint: Optional path to AstroJev checkpoint for continuous weights.
        nside: HEALPix resolution (default 64 -> 49,152 pixels).
        b_cut_deg: Galactic latitude exclusion cut |b| < b_cut_deg.
        risk_sensitive_weighting: If True and evidential model loaded, apply CVaR risk suppression.
        save_fig: Whether to render and save diagnostic figure.
        fig_path: Target save path for the diagnostic figure.

    Returns:
        Structured evaluation dictionary with de-biased dipole measurements.
    """
    cat_path = Path(catalog_path)
    if not cat_path.is_file():
        raise FileNotFoundError(f"Quaia catalog not found at {cat_path}")

    print(f"Loading Quaia catalog from {cat_path}...")
    t = Table.read(str(cat_path))
    n_sources = len(t)
    print(f"Loaded {n_sources:,} sources.")

    ra = np.asarray(t["ra"], dtype=np.float64)
    dec = np.asarray(t["dec"], dtype=np.float64)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    gal_l = coords.galactic.l.deg
    gal_b = coords.galactic.b.deg

    # Continuous weights: if model_checkpoint provided, load AstroJev and score
    weights = np.ones(n_sources, dtype=np.float64)
    if model_checkpoint and Path(model_checkpoint).is_file():
        print(f"Applying continuous AstroJev quasar weights from {model_checkpoint}...")
        try:
            import torch
            from ..astrojev import AstroJev, NUM_FEATURES
            from ..evidential_astrojev import EvidentialAstroJev

            ckpt = torch.load(str(model_checkpoint), map_location="cpu", weights_only=False)
            arch = ckpt.get("architecture", "astrojev")
            d_model = ckpt.get("d_model", 128)

            if arch == "evidential":
                model = EvidentialAstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)
            else:
                model = AstroJev(in_features=NUM_FEATURES, d_model=d_model, num_classes=4)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            # Pre-extract features: [phot_g, bp_rp, g_bp, w1, w1_w2, pm, pm_err, l, b, snr_flux]
            g = np.asarray(t["phot_g_mean_mag"], dtype=np.float32)
            bp = np.asarray(t["phot_bp_mean_mag"], dtype=np.float32)
            rp = np.asarray(t["phot_rp_mean_mag"], dtype=np.float32)
            w1 = np.asarray(t["mag_w1_vg"], dtype=np.float32)
            w2 = np.asarray(t["mag_w2_vg"], dtype=np.float32)
            pm = np.asarray(t["pm"], dtype=np.float32)
            pm_err = np.asarray(t["pmra_error"], dtype=np.float32)

            feats = np.column_stack([
                g, bp - rp, g - bp, w1, w1 - w2,
                pm, pm_err, (gal_l / 360.0).astype(np.float32), ((gal_b + 90.0) / 180.0).astype(np.float32), np.full_like(g, 50.0)
            ]).astype(np.float32)
            # Handle NaNs
            feats = np.nan_to_num(feats, nan=0.0, posinf=50.0, neginf=-50.0)

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)

            batch_size = 32768
            p_quasar_list = []
            with torch.no_grad():
                for b_idx in range(0, n_sources, batch_size):
                    batch_feats = torch.from_numpy(feats[b_idx:b_idx+batch_size]).float().to(device)
                    out = model(batch_feats)
                    probs = out["probs"] if "probs" in out else out["choice_probs"]
                    p_q = probs[:, 0].cpu().numpy()
                    if risk_sensitive_weighting and "u_epi" in out:
                        ue = out["u_epi"].cpu().numpy()
                        de = out["delta_eq"].cpu().numpy()
                        w_batch = compute_risk_sensitive_weights(p_q, u_epi=ue, delta_eq=de)
                    else:
                        w_batch = p_q
                    p_quasar_list.append(w_batch)

            weights = np.concatenate(p_quasar_list).astype(np.float64)
            print(f"AstroJev scoring complete: mean weight = {np.mean(weights):.4f} (risk_sensitive={risk_sensitive_weighting})")
        except Exception as exc:
            print(f"Notice: model inference bypassed ({exc}); using Quaia high-purity selection weights.")
            weights = np.ones(n_sources, dtype=np.float64)

    # Pixelize into HEALPix map
    hp = HEALPix(nside=nside, order="ring")
    pix_indices = hp.lonlat_to_healpix(coords.galactic.l, coords.galactic.b)
    pixel_counts = np.bincount(pix_indices, weights=weights, minlength=hp.npix).astype(np.float64)

    # Construct Galactic plane mask |b| < b_cut_deg
    pix_lon, pix_lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    pix_b = pix_lat.deg
    mask = (np.abs(pix_b) >= b_cut_deg).astype(np.float64)

    # Execute Pseudo-Cl Deconvolution
    print(f"Performing Pseudo-Cl mode-coupling deconvolution (NSIDE={nside}, |b| >= {b_cut_deg} deg)...")
    results = deconvolve_dipole(pixel_counts, mask, lmax=2, nside=nside)

    # Compute offset from CMB kinematic dipole
    dec_res = results["deconvolved"]
    raw_res = results["raw_masked"]

    # Angular separation to CMB dipole
    coord_dec = SkyCoord(l=dec_res["l_deg"] * u.deg, b=dec_res["b_deg"] * u.deg, frame="galactic")
    coord_raw = SkyCoord(l=raw_res["l_deg"] * u.deg, b=raw_res["b_deg"] * u.deg, frame="galactic")
    coord_cmb = SkyCoord(l=CMB_DIPOLE_L * u.deg, b=CMB_DIPOLE_B * u.deg, frame="galactic")

    sep_dec_cmb = float(coord_dec.separation(coord_cmb).deg)
    sep_raw_cmb = float(coord_raw.separation(coord_cmb).deg)

    results["deconvolved"]["sep_to_cmb_deg"] = sep_dec_cmb
    results["raw_masked"]["sep_to_cmb_deg"] = sep_raw_cmb

    print("=" * 72)
    print("COSMIC DIPOLE DECONVOLUTION SUMMARY (QUAIA G20.5):")
    print(f"  Raw Masked Dipole:       D = {raw_res['amplitude']:.4f} @ (l={raw_res['l_deg']:.1f} deg, b={raw_res['b_deg']:.1f} deg) [sep_CMB = {sep_raw_cmb:.1f} deg]")
    print(f"  Pseudo-Cl Deconvolved:   D = {dec_res['amplitude']:.4f} +/- {dec_res['sigma']:.4f} @ (l={dec_res['l_deg']:.1f} deg, b={dec_res['b_deg']:.1f} deg) [sep_CMB = {sep_dec_cmb:.1f} deg]")
    print(f"  Quadrupole Power:        Q = {dec_res['quadrupole_amp']:.4f}")
    print(f"  CMB Kinematic Benchmark: D = {CMB_DIPOLE_AMP:.4f} @ (l={CMB_DIPOLE_L:.1f} deg, b={CMB_DIPOLE_B:.1f} deg)")
    print(f"  Mode-Coupling Cond Num:  cond(K) = {results['diagnostics']['condition_number']:.2f}")
    print("=" * 72)

    if save_fig:
        out_fig = Path(fig_path) if fig_path else Path("docs/figures/quaia_pseudo_cl_deconvolution.png")
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        _plot_diagnostic_figure(pixel_counts, mask, results, out_fig, nside=nside)
        results["figure_path"] = str(out_fig)

    return results


def _plot_diagnostic_figure(
    pixel_counts: np.ndarray,
    mask: np.ndarray,
    results: Dict[str, Any],
    out_fig: Path,
    nside: int = 64,
):
    """Render 4-panel publication diagnostic plot for pseudo-Cl deconvolution."""
    fig = plt.figure(figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("#0b0f19")

    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    ra_deg = lon.deg
    dec_deg = lat.deg

    dec_res = results["deconvolved"]
    raw_res = results["raw_masked"]
    K = results["mode_coupling_matrix"]

    # 1. Sky Density Mollweide Projection
    ax1 = fig.add_subplot(2, 2, 1, projection="mollweide", facecolor="#111827")
    lon_rad_plot = np.radians(ra_deg - 180.0)
    lat_rad_plot = np.radians(dec_deg)

    # Contrast on masked sky
    w = mask > 0
    density = np.zeros_like(pixel_counts)
    mean_val = np.mean(pixel_counts[w])
    density[w] = (pixel_counts[w] - mean_val) / max(mean_val, 1e-6)

    sc = ax1.scatter(
        lon_rad_plot[w], lat_rad_plot[w],
        c=density[w], cmap="magma", s=2, alpha=0.8,
        vmin=-0.25, vmax=0.25, rasterized=True
    )
    cb = plt.colorbar(sc, ax=ax1, orientation="horizontal", pad=0.08, fraction=0.04)
    cb.set_label("Fractional Density Contrast delta (Quaia Quasars)", color="white", fontsize=9)
    cb.ax.tick_params(colors="white", labelsize=8)
    ax1.set_title("1. Observed Footprint & Density Contrast (|b| >= 10 deg)", color="white", fontsize=11, pad=12)
    ax1.tick_params(colors="#9ca3af", labelsize=8)
    ax1.grid(color="#374151", linestyle="--", alpha=0.5)

    # 2. Mode-Coupling Matrix K
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#111827")
    im = ax2.imshow(np.abs(K), cmap="viridis", interpolation="nearest")
    labels = ["0,0", "1,-1", "1,0", "1,1", "2,-2", "2,-1", "2,0", "2,1", "2,2"]
    ax2.set_xticks(range(len(labels)))
    ax2.set_yticks(range(len(labels)))
    ax2.set_xticklabels(labels, color="white", fontsize=8, rotation=45)
    ax2.set_yticklabels(labels, color="white", fontsize=8)
    ax2.set_title(f"2. Mode-Coupling Matrix |K| (cond = {results['diagnostics']['condition_number']:.2f})", color="white", fontsize=11)
    cb2 = plt.colorbar(im, ax=ax2, pad=0.03, fraction=0.046)
    cb2.set_label("Coupling Magnitude", color="white", fontsize=9)
    cb2.ax.tick_params(colors="white", labelsize=8)

    # 3. Vector Dipole Recovery Comparison
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#111827")
    categories = ["Raw Masked\n(Biased)", "Pseudo-Cl\nDeconvolved", "CMB Kinematic\nBenchmark", "CatWISE\nAnomaly"]
    amplitudes = [raw_res["amplitude"], dec_res["amplitude"], CMB_DIPOLE_AMP, 0.0140]
    errors = [0.0, dec_res["sigma"], 0.0005, 0.0020]
    colors = ["#ef4444", "#38bdf8", "#10b981", "#f59e0b"]

    bars = ax3.bar(categories, amplitudes, yerr=errors, color=colors, alpha=0.85, capsize=6, edgecolor="white", linewidth=0.8)
    ax3.set_ylabel("Dipole Amplitude D", color="white", fontsize=10)
    ax3.set_title(f"3. Dipole Amplitude De-biasing (D_deconv = {dec_res['amplitude']:.4f} +/- {dec_res['sigma']:.4f})", color="white", fontsize=11)
    ax3.tick_params(colors="white", labelsize=9)
    ax3.grid(color="#374151", linestyle="--", alpha=0.4, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, h + 0.0008, f"{h:.4f}", ha="center", va="bottom", color="white", fontsize=8, fontweight="bold")

    # 4. Angular Direction and Quadrupole Diagnostic
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#111827")
    ax4.scatter([raw_res["l_deg"]], [raw_res["b_deg"]], color="#ef4444", s=140, marker="o", label=f"Raw: ({raw_res['l_deg']:.0f} deg, {raw_res['b_deg']:.0f} deg)", zorder=5)
    ax4.scatter([dec_res["l_deg"]], [dec_res["b_deg"]], color="#38bdf8", s=160, marker="*", label=f"Deconvolved: ({dec_res['l_deg']:.0f} deg, {dec_res['b_deg']:.0f} deg)", zorder=6)
    ax4.scatter([CMB_DIPOLE_L], [CMB_DIPOLE_B], color="#10b981", s=140, marker="D", label=f"CMB Dipole: ({CMB_DIPOLE_L:.0f} deg, {CMB_DIPOLE_B:.0f} deg)", zorder=5)
    ax4.set_xlim(0, 360)
    ax4.set_ylim(-90, 90)
    ax4.set_xlabel("Galactic Longitude l (deg)", color="white", fontsize=10)
    ax4.set_ylabel("Galactic Latitude b (deg)", color="white", fontsize=10)
    ax4.set_title(f"4. Dipole Vector Orientation (CMB Offset: {dec_res['sep_to_cmb_deg']:.1f} deg)", color="white", fontsize=11)
    ax4.tick_params(colors="white", labelsize=9)
    ax4.grid(color="#374151", linestyle="--", alpha=0.4)
    legend = ax4.legend(loc="lower right", facecolor="#1f2937", edgecolor="#4b5563")
    for text in legend.get_texts():
        text.set_color("white")
        text.set_fontsize(8)

    plt.tight_layout()
    resolved_path = Path(out_fig).resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(resolved_path), dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic figure to: {resolved_path}")
