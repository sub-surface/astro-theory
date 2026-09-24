"""Euclid DR1 cosmic dipole forecast and partial-sky harmonic leakage.

Answers the gate question from docs/research/euclid-dr1-prep.md:
"Can DR1 measure the dipole or only rehearse the pipeline?"

Models the ~1,900 deg² DR1 wide-survey footprint across disjoint high-latitude
patches, computes exact partial-sky Fisher covariance for shot noise, evaluates
harmonic leakage from quadrupole clustering into the dipole estimator, and
determines signal-to-noise for distinguishing the CatWISE/Quaia anomaly
(D ~ 0.012) from the kinematic expectation (D_kin ~ 0.0047).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u

# Total sky area in square degrees (4pi sr * (180/pi)^2)
TOTAL_SKY_DEG2 = 41252.96125

# Reference Euclid DR1 wide survey patches (~1,900 deg2 across ~3 major fields)
# 1. EDF-North / North Galactic Patch: high-latitude, low-extinction
# 2. EDF-South / South Ecliptic Patch: Fornax / Eridanus / Horologium
# 3. EDF-Fornax / South Galactic Patch
DEFAULT_DR1_PATCHES = [
    {"name": "EDF-North", "ra": 269.75, "dec": 66.0, "fraction": 0.35},
    {"name": "EDF-South", "ra": 61.24, "dec": -48.4, "fraction": 0.45},
    {"name": "EDF-Fornax", "ra": 54.0, "dec": -28.0, "fraction": 0.20},
]


def build_dr1_footprint(
    nside: int = 32,
    area_deg2: float = 1900.0,
    patches: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build a realistic HEALPix mask for the Euclid DR1 wide footprint.

    Parameters
    ----------
    nside : int
        HEALPix resolution parameter (default 32 -> ~1.8 deg resolution).
    area_deg2 : float
        Target total survey area in square degrees (default 1900.0).
    patches : list of dict, optional
        Patch definitions with 'ra', 'dec', and 'fraction' of total area.

    Returns
    -------
    mask : np.ndarray of bool
        Boolean array of shape (npix,) where True = observed.
    meta : dict
        Footprint geometry metadata.
    """
    patches = patches or DEFAULT_DR1_PATCHES
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    pixel_coords = SkyCoord(lon, lat, frame="icrs")

    mask = np.zeros(hp.npix, dtype=bool)
    patch_details = []

    for p in patches:
        ra = p["ra"]
        dec = p["dec"]
        frac = p["fraction"]
        patch_area = area_deg2 * frac
        # Cap area: Omega = 2pi(1 - cos(theta_rad))
        patch_sr = patch_area * (math.pi / 180.0) ** 2
        cos_theta = max(-1.0, min(1.0, 1.0 - patch_sr / (2.0 * math.pi)))
        rad_deg = math.degrees(math.acos(cos_theta))

        patch_coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        sep = pixel_coords.separation(patch_coord).deg
        patch_mask = sep <= rad_deg
        mask |= patch_mask

        patch_details.append({
            "name": p.get("name", f"patch_{ra:.1f}_{dec:.1f}"),
            "ra": ra,
            "dec": dec,
            "target_area_deg2": patch_area,
            "radius_deg": rad_deg,
            "pixels": int(patch_mask.sum()),
        })

    pixel_area_deg2 = TOTAL_SKY_DEG2 / hp.npix
    actual_area_deg2 = float(mask.sum() * pixel_area_deg2)
    f_sky = actual_area_deg2 / TOTAL_SKY_DEG2

    meta = {
        "nside": nside,
        "npix": hp.npix,
        "surviving_pixels": int(mask.sum()),
        "target_area_deg2": area_deg2,
        "actual_area_deg2": actual_area_deg2,
        "f_sky": f_sky,
        "pixel_area_deg2": pixel_area_deg2,
        "patches": patch_details,
    }
    return mask, meta


def compute_dipole_fisher(
    mask: np.ndarray,
    n_sources: float,
    nside: int = 32,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the Fisher matrix and dipole covariance for a given sky mask.

    Parameters
    ----------
    mask : np.ndarray of bool
        HEALPix sky mask (True = observed).
    n_sources : float
        Total number of sources across the footprint.
    nside : int
        HEALPix resolution parameter.

    Returns
    -------
    cov_D : np.ndarray (3, 3)
        Covariance matrix for the dipole vector [Dx, Dy, Dz].
    sigma_D_shot : float
        Trace-averaged shot-noise uncertainty sqrt(Tr(Cov)/3).
    cond_num : float
        Condition number of the design matrix X^T X.
    """
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    x = np.cos(lat.rad) * np.cos(lon.rad)
    y = np.cos(lat.rad) * np.sin(lon.rad)
    z = np.sin(lat.rad)

    n_pix = int(mask.sum())
    if n_pix < 4:
        raise ValueError("Too few surviving pixels to compute Fisher matrix")

    x_mask = x[mask]
    y_mask = y[mask]
    z_mask = z[mask]

    X1 = np.column_stack([np.ones(n_pix), x_mask, y_mask, z_mask])
    XTX = X1.T @ X1
    cond_num = float(np.linalg.cond(XTX))

    inv_XTX = np.linalg.inv(XTX)
    a0 = n_sources / n_pix

    # Full covariance of (a0, a0*Dx, a0*Dy, a0*Dz) is a0 * (X^T X)^(-1)
    cov_beta = a0 * inv_XTX
    # For D = beta[1:] / a0, Cov(D) = cov_beta[1:, 1:] / a0^2
    cov_D = cov_beta[1:, 1:] / (a0 ** 2)

    sigma_D_shot = float(np.sqrt(np.trace(cov_D) / 3.0))
    return cov_D, sigma_D_shot, cond_num


def compute_harmonic_leakage(
    mask: np.ndarray,
    nside: int = 32,
    c2_clustering: float = 5e-5,
) -> Tuple[np.ndarray, float]:
    """Compute harmonic leakage covariance from quadrupole (l=2) into dipole (l=1).

    Parameters
    ----------
    mask : np.ndarray of bool
        HEALPix sky mask.
    nside : int
        HEALPix resolution.
    c2_clustering : float
        Quadrupole clustering power C_2. Typical optical/NIR galaxy clustering
        at z ~ 1 yields C_2 in range 2e-5 to 1e-4.

    Returns
    -------
    cov_leak : np.ndarray (3, 3)
        Covariance matrix of dipole induced by quadrupole leakage.
    sigma_D_leak : float
        Trace-averaged leakage uncertainty sqrt(Tr(Cov)/3).
    """
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    x = np.cos(lat.rad) * np.cos(lon.rad)
    y = np.cos(lat.rad) * np.sin(lon.rad)
    z = np.sin(lat.rad)

    n_pix = int(mask.sum())
    if n_pix < 9:
        raise ValueError("Too few surviving pixels to compute harmonic leakage")

    # Real spherical harmonics Y_2m (orthonormal on 4pi sphere)
    norm2 = math.sqrt(15.0 / (4.0 * math.pi))
    norm20 = math.sqrt(5.0 / (16.0 * math.pi))
    norm22 = math.sqrt(15.0 / (16.0 * math.pi))

    y20 = norm20 * (3.0 * z ** 2 - 1.0)
    y21 = norm2 * x * z
    y2m1 = norm2 * y * z
    y22 = norm22 * (x ** 2 - y ** 2)
    y2m2 = norm2 * x * y

    Y2 = np.column_stack([y20, y21, y2m1, y22, y2m2])[mask]
    X1 = np.column_stack([np.ones(n_pix), x[mask], y[mask], z[mask]])

    inv_XTX = np.linalg.inv(X1.T @ X1)
    # Projection matrix L = (X1^T X1)^(-1) X1^T Y2
    L = inv_XTX @ (X1.T @ Y2)
    L_dipole = L[1:, :]  # shape (3, 5)

    cov_leak = c2_clustering * (L_dipole @ L_dipole.T)
    sigma_D_leak = float(np.sqrt(np.trace(cov_leak) / 3.0))
    return cov_leak, sigma_D_leak


def forecast_dr1(
    area_deg2: float = 1900.0,
    density_arcmin2: float = 30.0,
    nside: int = 32,
    d_anom: float = 0.012,
    d_kin: float = 0.0047,
    c2_clustering: float = 5e-5,
) -> Dict[str, Any]:
    """Execute the full pre-DR1 cosmic dipole forecast.

    Answers: Can Euclid DR1 distinguish D_anom (~0.012) from D_kin (~0.0047)
    at >= 3 sigma, or is it a methods dress-rehearsal for DR2?

    Parameters
    ----------
    area_deg2 : float
        DR1 wide survey footprint area in square degrees (~1900).
    density_arcmin2 : float
        Source surface density in galaxies per square arcminute (default 30.0).
        For a clean quasar sample, pass e.g. 0.08.
    nside : int
        HEALPix grid resolution (default 32).
    d_anom : float
        CatWISE/Quaia anomaly dipole amplitude (default 0.012).
    d_kin : float
        Ellis-Baldwin kinematic CMB expectation for Euclid bands (default 0.0047).
    c2_clustering : float
        Cosmic galaxy clustering quadrupole power C_2 (default 5e-5).

    Returns
    -------
    dict
        Forecast report including shot noise, leakage, SNR, and verdict.
    """
    mask, meta = build_dr1_footprint(nside=nside, area_deg2=area_deg2)

    # Convert density from arcmin^-2 to total sources
    arcmin2_per_deg2 = 3600.0
    n_sources = density_arcmin2 * arcmin2_per_deg2 * meta["actual_area_deg2"]

    # Full-sky analytic benchmark
    full_sky_shot = math.sqrt(3.0 / n_sources)

    # Partial-sky Fisher shot-noise
    cov_shot, sigma_shot, cond_num = compute_dipole_fisher(mask, n_sources, nside=nside)

    # Partial-sky harmonic leakage from cosmic clustering
    cov_leak, sigma_leak = compute_harmonic_leakage(mask, nside=nside, c2_clustering=c2_clustering)

    # Total combined uncertainty
    sigma_total = math.sqrt(sigma_shot ** 2 + sigma_leak ** 2)

    # Signal-to-noise ratio for detecting the anomaly above kinematic
    delta_d = abs(d_anom - d_kin)
    snr = delta_d / sigma_total

    # Decision rule: >= 3.0 sigma -> MEASUREMENT, else DRESS REHEARSAL
    is_measurement = snr >= 3.0
    verdict = "MEASUREMENT" if is_measurement else "DRESS REHEARSAL"

    if is_measurement:
        verdict_reason = (
            f"DR1 can distinguish D={d_anom:.3f} from D_kin={d_kin:.4f} at {snr:.1f}σ "
            f"(>= 3σ threshold). Proceed with Day-One pipeline execution."
        )
    else:
        verdict_reason = (
            f"DR1 reaches {snr:.1f}σ (< 3σ threshold) due to partial-sky harmonic leakage "
            f"(σ_leak={sigma_leak:.4f} vs σ_shot={sigma_shot:.4f}). "
            f"DR1 is a methods dress-rehearsal & cross-catalogue audit; definitive "
            f"dipole requires full-sky DR2."
        )

    return {
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "is_measurement": is_measurement,
        "snr": snr,
        "delta_d": delta_d,
        "d_anom": d_anom,
        "d_kin": d_kin,
        "sigma_total": sigma_total,
        "sigma_shot": sigma_shot,
        "sigma_leak": sigma_leak,
        "full_sky_shot": full_sky_shot,
        "geometric_penalty": sigma_shot / full_sky_shot,
        "condition_number": cond_num,
        "n_sources": n_sources,
        "density_arcmin2": density_arcmin2,
        "area_deg2": meta["actual_area_deg2"],
        "f_sky": meta["f_sky"],
        "nside": nside,
        "patches": meta["patches"],
    }
