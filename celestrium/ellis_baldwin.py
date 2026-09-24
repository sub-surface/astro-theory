"""Ellis-Baldwin kinematic dipole predictions for Euclid photometric bands.

Pre-registers the expected CMB-frame kinematic dipole amplitude:
    D_kin = [2 + x(1 + alpha)] * beta,  beta = v_CMB / c

Replaces the Quaia/CatWISE empirical (x, alpha) values with pre-registered
expectations for Euclid VIS (I_E) and NISP (Y, J, H) based on galaxy and AGN
luminosity functions at Euclid DR1 depth.
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

V_CMB = 369.82  # km/s (Planck 2018 / CMB dipole velocity)
C_KMS = 299792.458
BETA = V_CMB / C_KMS  # ~ 1.23358e-3

# CMB dipole apex direction
CMB_APEX_L = 264.021
CMB_APEX_B = 48.253

# Pre-registered band models:
# x = 2.5 * d log10(N)/dm (integral number-count slope at survey limit)
# alpha = effective spectral index (S_nu ~ nu^-alpha)
EUCLID_BAND_MODELS: Dict[str, Dict[str, Any]] = {
    "VIS": {
        "band": "VIS (I_E)",
        "wavelength_nm": "550 - 900",
        "depth_mag": 24.5,
        "x_mean": 0.85,
        "x_std": 0.05,
        "alpha_mean": 1.10,
        "alpha_std": 0.15,
        "description": "Euclid VIS primary optical galaxy sample",
    },
    "NISP_Y": {
        "band": "NISP Y",
        "wavelength_nm": "920 - 1146",
        "depth_mag": 24.0,
        "x_mean": 0.80,
        "x_std": 0.05,
        "alpha_mean": 1.20,
        "alpha_std": 0.15,
        "description": "Euclid NISP near-infrared Y-band galaxies",
    },
    "NISP_J": {
        "band": "NISP J",
        "wavelength_nm": "1146 - 1372",
        "depth_mag": 24.0,
        "x_mean": 0.78,
        "x_std": 0.05,
        "alpha_mean": 1.25,
        "alpha_std": 0.15,
        "description": "Euclid NISP near-infrared J-band galaxies",
    },
    "NISP_H": {
        "band": "NISP H",
        "wavelength_nm": "1372 - 2000",
        "depth_mag": 24.0,
        "x_mean": 0.75,
        "x_std": 0.05,
        "alpha_mean": 1.30,
        "alpha_std": 0.15,
        "description": "Euclid NISP near-infrared H-band galaxies",
    },
    "GALAXY_COMBINED": {
        "band": "Combined Galaxies (VIS+NISP)",
        "wavelength_nm": "550 - 2000",
        "depth_mag": 24.5,
        "x_mean": 0.82,
        "x_std": 0.04,
        "alpha_mean": 1.15,
        "alpha_std": 0.10,
        "description": "Joint optical+NIR full galaxy sample at DR1 depth",
    },
    "AGN_QUASAR": {
        "band": "Euclid Quasars / AGN",
        "wavelength_nm": "550 - 2000",
        "depth_mag": 24.0,
        "x_mean": 1.60,
        "x_std": 0.10,
        "alpha_mean": 0.60,
        "alpha_std": 0.15,
        "description": "High-redshift AGN / QSO subset selected by colour & morphology",
    },
}


def get_cmb_apex_coords() -> Dict[str, float]:
    """Return CMB apex direction in both Galactic and ICRS frames."""
    coord = SkyCoord(CMB_APEX_L * u.deg, CMB_APEX_B * u.deg, frame="galactic")
    return {
        "l": CMB_APEX_L,
        "b": CMB_APEX_B,
        "ra": float(coord.icrs.ra.deg),
        "dec": float(coord.icrs.dec.deg),
    }


def predict_band(
    model: Dict[str, Any],
    n_samples: int = 50000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Monte Carlo propagation of x and alpha uncertainties into D_kin."""
    rng = np.random.default_rng(seed)
    xs = rng.normal(model["x_mean"], model["x_std"], n_samples)
    alphas = rng.normal(model["alpha_mean"], model["alpha_std"], n_samples)

    # Ellis & Baldwin (1984) relation
    d_kins = (2.0 + xs * (1.0 + alphas)) * BETA

    apex = get_cmb_apex_coords()
    mean_val = float(np.mean(d_kins))
    std_val = float(np.std(d_kins, ddof=1))
    p16 = float(np.percentile(d_kins, 16))
    p50 = float(np.percentile(d_kins, 50))
    p84 = float(np.percentile(d_kins, 84))

    return {
        "band": model["band"],
        "wavelength_nm": model["wavelength_nm"],
        "depth_mag": model["depth_mag"],
        "x": model["x_mean"],
        "x_std": model["x_std"],
        "alpha": model["alpha_mean"],
        "alpha_std": model["alpha_std"],
        "description": model["description"],
        "d_kin_mean": mean_val,
        "d_kin_std": std_val,
        "d_kin_p16": p16,
        "d_kin_p50": p50,
        "d_kin_p84": p84,
        "direction": apex,
        "beta": BETA,
        "v_cmb_kms": V_CMB,
    }


def predict_all_bands(
    n_samples: int = 50000,
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Compute pre-registered D_kin predictions for all Euclid bands."""
    results = {}
    for key, model in EUCLID_BAND_MODELS.items():
        results[key] = predict_band(model, n_samples=n_samples, seed=seed)
    return {
        "cmb_apex": get_cmb_apex_coords(),
        "beta": BETA,
        "v_cmb_kms": V_CMB,
        "predictions": results,
    }
