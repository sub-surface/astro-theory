"""
=============================================================================
EXP-2026-W: Unified Multi-Tracer Anisotropy & Cosmic Dipole Co-Inference
=============================================================================
Simultaneously infers the cosmic bulk velocity vector v_bulk and tests cosmological
isotropy across three independent all-sky surveys spanning optical, infrared, and radio:
  1. Quaia Quasars (Gaia DR3 x unWISE): 1.295M sources (optical/near-IR)
  2. CatWISE2020 AGNs: 1.354M sources (mid-IR W1/W2)
  3. NVSS Radio Galaxies: 212k sources (1.4 GHz continuum)

Core Capabilities:
  - Joint Hierarchical Bayesian Poisson Likelihood over all surviving HEALPix pixels.
  - Shared Cosmic Kinematic Bulk Flow Vector: beta = v_bulk / c.
  - Tracer-Specific Ellis-Baldwin Response Factors: k_i = 2 + x_i (1 + alpha_i).
  - Survey Systematic Marginalization: Non-linear selection exponents gamma_i and
    normalization scales N_{0, i}.
  - Bayesian Model Comparison on the Jeffreys Scale:
      * H_0: Strict CMB Kinematic Null (v_bulk = 369.82 km/s toward (264.02 deg, 48.25 deg))
      * H_1: Unified Cosmological Bulk Flow (shared beta in R^3)
      * H_2: Decoupled Independent Dipoles (separate uncoupled dipoles D_i)
  - Vectorized Goodman & Weare (2010) Affine-Invariant Ensemble MCMC Sampler.
  - Gelman-Rubin convergence R_hat < 1.05 and autocorrelation time tau_act.
  - Conformal Sky Residual Anomaly Filter bounding regional contamination.
"""
from __future__ import annotations

import json
import math
import os
import warnings
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy_healpix import HEALPix

# --------------------------------------------------------------------------- #
# Cosmological Constants & Kinematic Calibration
# --------------------------------------------------------------------------- #
SPEED_OF_LIGHT_KMS = 299792.458

# Planck 2018 CMB Kinematic Dipole
V_CMB_KMS = 369.82
BETA_CMB_MAGNITUDE = V_CMB_KMS / SPEED_OF_LIGHT_KMS  # ~ 0.0012336
CMB_APEX_L_DEG = 264.021
CMB_APEX_B_DEG = 48.253

# Nominal Ellis-Baldwin factors from literature (Storey-Fisher 2024, Secrest 2021, Blake & Wall 2002)
# k_i = 2 + x_i * (1 + alpha_i)
TRACER_DEFAULTS = {
    "quaia": {
        "x": 1.70,
        "alpha": 1.00,
        "kinematic_factor": 2.0 + 1.70 * (1.0 + 1.00),  # 5.40
        "expected_d_kin": (2.0 + 1.70 * (1.0 + 1.00)) * BETA_CMB_MAGNITUDE,  # ~ 0.00666
    },
    "catwise": {
        "x": 1.35,
        "alpha": 1.26,
        "kinematic_factor": 2.0 + 1.35 * (1.0 + 1.26),  # 5.051
        "expected_d_kin": (2.0 + 1.35 * (1.0 + 1.26)) * BETA_CMB_MAGNITUDE,  # ~ 0.00623
    },
    "nvss": {
        "x": 0.74,
        "alpha": 0.75,
        "kinematic_factor": 2.0 + 0.74 * (1.0 + 0.75),  # 3.295
        "expected_d_kin": (2.0 + 0.74 * (1.0 + 0.75)) * BETA_CMB_MAGNITUDE,  # ~ 0.00406
    }
}


def lb_to_unit_vector(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
    """Converts Galactic (l, b) in degrees to 3D Cartesian unit vectors."""
    cos_b = np.cos(np.radians(b_deg))
    sin_b = np.sin(np.radians(b_deg))
    cos_l = np.cos(np.radians(l_deg))
    sin_l = np.sin(np.radians(l_deg))
    return np.column_stack([cos_b * cos_l, cos_b * sin_l, sin_b])


def unit_vector_to_lb(vec: np.ndarray) -> Tuple[float, float]:
    """Converts 3D Cartesian vector to Galactic (l, b) in degrees."""
    norm = float(np.linalg.norm(vec))
    if norm < 1e-14:
        return 0.0, 0.0
    u_vec = vec / norm
    b = float(np.degrees(np.arcsin(np.clip(u_vec[2], -1.0, 1.0))))
    l = float(np.degrees(np.arctan2(u_vec[1], u_vec[0])) % 360.0)
    return l, b


def get_cmb_beta_vector() -> np.ndarray:
    """Returns the CMB velocity vector beta_CMB = (v_CMB / c) * n_hat_CMB in Galactic Cartesian."""
    u_cmb = lb_to_unit_vector(np.array([CMB_APEX_L_DEG]), np.array([CMB_APEX_B_DEG]))[0]
    return BETA_CMB_MAGNITUDE * u_cmb


# --------------------------------------------------------------------------- #
# Tracer Survey Container & HEALPix Geometry
# --------------------------------------------------------------------------- #
class TracerSurvey:
    """Encapsulates a single survey's binned sky counts, selection map, and physical response."""
    def __init__(
        self,
        name: str,
        nside: int,
        counts: np.ndarray,
        selection: np.ndarray,
        mask: np.ndarray,
        kinematic_factor: float,
        total_sources: int,
        x_slope: float,
        alpha_index: float,
    ):
        self.name = name
        self.nside = nside
        self.npix = 12 * nside * nside
        self.counts = np.asarray(counts, dtype=np.float64)
        self.selection = np.asarray(selection, dtype=np.float64)
        self.mask = np.asarray(mask, dtype=bool)
        self.kinematic_factor = float(kinematic_factor)
        self.total_sources = int(total_sources)
        self.x_slope = float(x_slope)
        self.alpha_index = float(alpha_index)

        # Precompute unit vectors for valid sky pixels
        hp = HEALPix(nside=nside, order="ring")
        lon, lat = hp.healpix_to_lonlat(np.arange(self.npix))
        # Ensure coordinates are in Galactic frame
        sc = SkyCoord(lon, lat, frame="icrs").galactic
        self.pixel_l = sc.l.deg
        self.pixel_b = sc.b.deg
        self.unit_vectors = lb_to_unit_vector(self.pixel_l, self.pixel_b)


class MultiTracerDataset:
    """Loads and compiles Quaia, CatWISE, and NVSS into unified HEALPix maps (Nside=64)."""
    def __init__(self, nside: int = 64, b_cut_deg: float = 30.0):
        self.nside = nside
        self.npix = 12 * nside * nside
        self.b_cut_deg = b_cut_deg
        self.surveys: Dict[str, TracerSurvey] = {}
        self.mock_surveys: List[str] = []

    @classmethod
    def load_from_cache_or_catalogs(
        cls,
        root_dir: Path,
        cache_path: Optional[Path] = None,
        nside: int = 64,
        b_cut_deg: float = 30.0,
        verbose: bool = True
    ) -> "MultiTracerDataset":
        """Loads cached HEALPix maps if available; otherwise aggregates raw catalogs."""
        dataset = cls(nside=nside, b_cut_deg=b_cut_deg)
        dataset.mock_surveys = []
        if cache_path is None:
            cache_path = root_dir / "data" / "cache" / f"multi_tracer_nside{nside}_bcut{int(b_cut_deg)}.npz"

        if cache_path.exists():
            if verbose:
                print(f"[MultiTracer] Loading cached survey maps from {cache_path}...")
            data = np.load(str(cache_path), allow_pickle=True)
            for key in ["quaia", "catwise", "nvss"]:
                prefix = f"{key}_"
                counts = data[f"{prefix}counts"]
                selection = data[f"{prefix}selection"]
                mask = data[f"{prefix}mask"]
                k_factor = float(data[f"{prefix}kinematic_factor"])
                n_src = int(data[f"{prefix}total_sources"])
                x_slope = float(data[f"{prefix}x_slope"])
                alpha = float(data[f"{prefix}alpha_index"])
                dataset.surveys[key] = TracerSurvey(
                    name=key,
                    nside=nside,
                    counts=counts,
                    selection=selection,
                    mask=mask,
                    kinematic_factor=k_factor,
                    total_sources=n_src,
                    x_slope=x_slope,
                    alpha_index=alpha
                )
            return dataset

        # Build from source catalogs
        if verbose:
            print("[MultiTracer] Building HEALPix maps from raw survey catalogs...")
        hp = HEALPix(nside=nside, order="ring")
        npix = 12 * nside * nside
        lon_all, lat_all = hp.healpix_to_lonlat(np.arange(npix))
        gal_all = SkyCoord(lon_all, lat_all, frame="icrs").galactic
        gal_b = gal_all.b.deg
        gal_l = gal_all.l.deg

        # Common galactic plane mask |b| > b_cut
        galactic_mask = np.abs(gal_b) > b_cut_deg

        # 1. Quaia G20.5
        quaia_fits = root_dir / "Archive" / "2026-06-G-dipole" / "data" / "quaia" / "quaia_G20.5.fits"
        quaia_selfunc = root_dir / "Archive" / "2026-06-G-dipole" / "data" / "quaia" / "selfunc_G20.5_nside64.fits"
        if quaia_fits.exists() and quaia_selfunc.exists():
            if verbose:
                print(f"  Ingesting Quaia from {quaia_fits.name}...")
            with fits.open(quaia_fits, memmap=True) as hdu:
                ra_q = np.asarray(hdu[1].data["ra"], dtype=np.float64)
                dec_q = np.asarray(hdu[1].data["dec"], dtype=np.float64)
            pix_q = hp.lonlat_to_healpix(ra_q * u.deg, dec_q * u.deg)
            counts_q = np.bincount(pix_q, minlength=npix).astype(np.float64)
            
            with fits.open(quaia_selfunc) as hdu_s:
                col_name = hdu_s[1].data.columns.names[0]
                s_data = np.asarray(hdu_s[1].data[col_name], dtype=np.float64).reshape(-1)
            s_map_q = np.zeros(npix, dtype=np.float64)
            s_map_q[:min(npix, len(s_data))] = s_data[:min(npix, len(s_data))]
            mask_q = galactic_mask & (s_map_q > 0.5)

            dataset.surveys["quaia"] = TracerSurvey(
                name="quaia",
                nside=nside,
                counts=counts_q,
                selection=s_map_q,
                mask=mask_q,
                kinematic_factor=TRACER_DEFAULTS["quaia"]["kinematic_factor"],
                total_sources=len(ra_q),
                x_slope=TRACER_DEFAULTS["quaia"]["x"],
                alpha_index=TRACER_DEFAULTS["quaia"]["alpha"]
            )
        else:
            dataset._warn_mock("quaia")
            dataset.surveys["quaia"] = dataset._create_mock_survey("quaia", nside, galactic_mask)

        # 2. CatWISE2020 AGNs
        cw_dir = root_dir / "Archive" / "2026-06-G-dipole" / "data" / "catwise"
        cw_files = sorted(cw_dir.glob("cw_ra*.csv")) if cw_dir.exists() else []
        if len(cw_files) > 0:
            import pandas as pd
            if verbose:
                print(f"  Ingesting CatWISE from {len(cw_files)} stripe CSVs via pandas...")
            counts_cw = np.zeros(npix, dtype=np.float64)
            n_cw = 0
            for cw_f in cw_files:
                df_cw = pd.read_csv(cw_f, usecols=["ra", "dec"])
                ra_cw = df_cw["ra"].to_numpy(dtype=np.float64)
                dec_cw = df_cw["dec"].to_numpy(dtype=np.float64)
                n_cw += len(ra_cw)
                pix_cw = hp.lonlat_to_healpix(ra_cw * u.deg, dec_cw * u.deg)
                counts_cw += np.bincount(pix_cw, minlength=npix)

            # CatWISE systematic template: ecliptic latitude coverage trend s_p = (a + b|elat_p|) / mean
            ecl_lat = SkyCoord(lon_all, lat_all, frame="icrs").barycentrictrueecliptic.lat.deg
            s_cw = (1.0 + 0.15 * np.abs(ecl_lat) / 90.0)
            s_cw = s_cw / np.mean(s_cw)

            # Mask LMC and SMC in CatWISE
            lmc_smc_mask = np.ones(npix, dtype=bool)
            for l0, b0, r0 in [(280.5, -32.9, 9.0), (302.8, -44.3, 7.0)]:
                sep = SkyCoord(gal_l * u.deg, gal_b * u.deg, frame="galactic").separation(
                    SkyCoord(l0 * u.deg, b0 * u.deg, frame="galactic")
                ).deg
                lmc_smc_mask &= (sep > r0)

            # Robust 6-sigma hot-pixel clip on counts to reject artifact stars
            base_cw = galactic_mask & lmc_smc_mask & (counts_cw > 0)
            good_cw = np.copy(base_cw)
            for _ in range(5):
                med = np.median(counts_cw[good_cw])
                mad = np.median(np.abs(counts_cw[good_cw] - med)) * 1.4826
                new_good = good_cw & (counts_cw <= med + 6.0 * mad)
                if np.sum(new_good) == np.sum(good_cw):
                    break
                good_cw = new_good

            mask_cw = good_cw
            dataset.surveys["catwise"] = TracerSurvey(
                name="catwise",
                nside=nside,
                counts=counts_cw,
                selection=s_cw,
                mask=mask_cw,
                kinematic_factor=TRACER_DEFAULTS["catwise"]["kinematic_factor"],
                total_sources=int(counts_cw[mask_cw].sum()),
                x_slope=TRACER_DEFAULTS["catwise"]["x"],
                alpha_index=TRACER_DEFAULTS["catwise"]["alpha"]
            )
        else:
            dataset._warn_mock("catwise")
            dataset.surveys["catwise"] = dataset._create_mock_survey("catwise", nside, galactic_mask)

        # 3. NVSS 1.4 GHz Radio
        nvss_csv = root_dir / "Archive" / "2026-06-G-dipole" / "data" / "nvss" / "nvss_S15.csv"
        if nvss_csv.exists():
            import pandas as pd
            if verbose:
                print(f"  Ingesting NVSS from {nvss_csv.name} via pandas...")
            df_nvss = pd.read_csv(nvss_csv)
            # Use sources with S1.4 > 15 mJy (standard homogenization cut)
            if "s14" in df_nvss.columns:
                df_nvss = df_nvss[df_nvss["s14"] >= 15.0]
            ra_col = "RAJ2000" if "RAJ2000" in df_nvss.columns else df_nvss.columns[0]
            dec_col = "DEJ2000" if "DEJ2000" in df_nvss.columns else df_nvss.columns[1]
            ra_nvss = df_nvss[ra_col].to_numpy(dtype=np.float64)
            dec_nvss = df_nvss[dec_col].to_numpy(dtype=np.float64)

            pix_nvss = hp.lonlat_to_healpix(ra_nvss * u.deg, dec_nvss * u.deg)
            counts_nvss = np.bincount(pix_nvss, minlength=npix).astype(np.float64)

            # NVSS sky coverage: Dec > -37 deg
            pix_dec = lat_all.deg
            mask_nvss = galactic_mask & (pix_dec > -37.0)
            s_nvss = np.ones(npix, dtype=np.float64)

            dataset.surveys["nvss"] = TracerSurvey(
                name="nvss",
                nside=nside,
                counts=counts_nvss,
                selection=s_nvss,
                mask=mask_nvss,
                kinematic_factor=TRACER_DEFAULTS["nvss"]["kinematic_factor"],
                total_sources=int(counts_nvss[mask_nvss].sum()),
                x_slope=TRACER_DEFAULTS["nvss"]["x"],
                alpha_index=TRACER_DEFAULTS["nvss"]["alpha"]
            )
        else:
            dataset._warn_mock("nvss")
            dataset.surveys["nvss"] = dataset._create_mock_survey("nvss", nside, galactic_mask)

        # Never persist synthetic mock maps to the real-data cache path.
        if dataset.mock_surveys:
            warnings.warn(
                f"[MultiTracer] Not writing cache {cache_path}: surveys {dataset.mock_surveys} "
                "are synthetic mocks, not real catalog data.",
                RuntimeWarning,
                stacklevel=2,
            )
            return dataset

        # Save cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {}
        for key, s in dataset.surveys.items():
            prefix = f"{key}_"
            save_dict[f"{prefix}counts"] = s.counts
            save_dict[f"{prefix}selection"] = s.selection
            save_dict[f"{prefix}mask"] = s.mask
            save_dict[f"{prefix}kinematic_factor"] = np.array(s.kinematic_factor)
            save_dict[f"{prefix}total_sources"] = np.array(s.total_sources)
            save_dict[f"{prefix}x_slope"] = np.array(s.x_slope)
            save_dict[f"{prefix}alpha_index"] = np.array(s.alpha_index)
        np.savez_compressed(str(cache_path), **save_dict)
        if verbose:
            print(f"[MultiTracer] Cache written to {cache_path}")

        return dataset

    def _warn_mock(self, name: str) -> None:
        """Record and loudly warn that a survey fell back to a synthetic mock."""
        self.mock_surveys.append(name)
        warnings.warn(
            f"[MultiTracer] Catalog for '{name}' not found; substituting a SYNTHETIC mock survey. "
            "Results using this dataset are not real-data results.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _create_mock_survey(self, name: str, nside: int, base_mask: np.ndarray) -> TracerSurvey:
        """Hermetic synthetic mock generator for rapid pytest test isolation."""
        npix = 12 * nside * nside
        # zlib.crc32 is stable across processes (built-in hash() of str is salted per process)
        rng = np.random.default_rng(zlib.crc32(name.encode("utf-8")) % 10000)
        u_cmb = lb_to_unit_vector(np.array([CMB_APEX_L_DEG]), np.array([CMB_APEX_B_DEG]))[0]
        
        hp = HEALPix(nside=nside, order="ring")
        lon, lat = hp.healpix_to_lonlat(np.arange(npix))
        gal = SkyCoord(lon, lat, frame="icrs").galactic
        unit_vecs = lb_to_unit_vector(gal.l.deg, gal.b.deg)

        k_factor = TRACER_DEFAULTS[name]["kinematic_factor"]
        # Expected kinematic dipole signal + Poisson noise
        dipole_term = 1.0 + k_factor * BETA_CMB_MAGNITUDE * (unit_vecs @ u_cmb)
        base_rate = 50.0 if name == "nvss" else 200.0
        counts = rng.poisson(np.clip(base_rate * dipole_term, 1.0, 1000.0)).astype(np.float64)
        selection = np.ones(npix, dtype=np.float64)
        mask = np.copy(base_mask)
        if name == "nvss":
            mask &= (lat.deg > -37.0)

        return TracerSurvey(
            name=name,
            nside=nside,
            counts=counts,
            selection=selection,
            mask=mask,
            kinematic_factor=k_factor,
            total_sources=int(counts[mask].sum()),
            x_slope=TRACER_DEFAULTS[name]["x"],
            alpha_index=TRACER_DEFAULTS[name]["alpha"]
        )


# --------------------------------------------------------------------------- #
# Joint Hierarchical Poisson Likelihood
# --------------------------------------------------------------------------- #
class MultiTracerLikelihood:
    """Joint Poisson likelihood across multiple surveys with selection function co-marginalization.
    
    Model Parametrizations:
      - 'unified' (H_1): 9 parameters
          theta = [beta_x, beta_y, beta_z, ln_N0_q, gamma_q, ln_N0_cw, gamma_cw, ln_N0_nvss, gamma_nvss]
          Common bulk velocity beta shared across all 3 surveys.
      - 'cmb_null' (H_0): 6 parameters
          theta = [ln_N0_q, gamma_q, ln_N0_cw, gamma_cw, ln_N0_nvss, gamma_nvss]
          Beta is strictly fixed to beta_CMB.
      - 'decoupled' (H_2): 15 parameters
          theta = [D_x_q, D_y_q, D_z_q, ln_N0_q, gamma_q,
                   D_x_cw, D_y_cw, D_z_cw, ln_N0_cw, gamma_cw,
                   D_x_nv, D_y_nv, D_z_nv, ln_N0_nv, gamma_nv]
          Independent uncoupled dipole vectors for each survey.
    """
    def __init__(self, dataset: MultiTracerDataset, model_type: str = "unified"):
        self.dataset = dataset
        self.model_type = model_type
        self.survey_keys = ["quaia", "catwise", "nvss"]

        # Cache masked slices for ultra-fast vectorized evaluation
        self.cached_slices = {}
        for key in self.survey_keys:
            if key in dataset.surveys:
                s = dataset.surveys[key]
                m = s.mask
                self.cached_slices[key] = {
                    "k": s.counts[m],
                    "s": s.selection[m],
                    "u": s.unit_vectors[m],
                    "k_factor": s.kinematic_factor,
                    "npix": int(np.sum(m)),
                }

        self.beta_cmb = get_cmb_beta_vector()

    def log_likelihood(self, theta: np.ndarray) -> float:
        """Computes the total joint Poisson log-likelihood across all active surveys."""
        total_ll = 0.0

        if self.model_type == "unified":
            beta_vec = theta[0:3]
            param_idx = 3
            for key in self.survey_keys:
                if key not in self.cached_slices:
                    continue
                ln_n0 = theta[param_idx]
                gamma = theta[param_idx + 1]
                param_idx += 2

                sl = self.cached_slices[key]
                k = sl["k"]
                s = sl["s"]
                u = sl["u"]
                k_fac = sl["k_factor"]

                # Expected mean counts: mu = exp(ln_N0) * (s ** gamma) * (1 + k_factor * beta . u)
                dipole_factor = 1.0 + k_fac * (u @ beta_vec)
                if np.any(dipole_factor <= 0.05):
                    return -1e12  # Unphysical negative or near-zero intensity

                mu = np.exp(ln_n0) * (s ** gamma) * dipole_factor
                ll = np.sum(k * np.log(mu) - mu)
                total_ll += float(ll)

        elif self.model_type == "cmb_null":
            param_idx = 0
            beta_vec = self.beta_cmb
            for key in self.survey_keys:
                if key not in self.cached_slices:
                    continue
                ln_n0 = theta[param_idx]
                gamma = theta[param_idx + 1]
                param_idx += 2

                sl = self.cached_slices[key]
                k = sl["k"]
                s = sl["s"]
                u = sl["u"]
                k_fac = sl["k_factor"]

                dipole_factor = 1.0 + k_fac * (u @ beta_vec)
                if np.any(dipole_factor <= 0.05):
                    return -1e12

                mu = np.exp(ln_n0) * (s ** gamma) * dipole_factor
                ll = np.sum(k * np.log(mu) - mu)
                total_ll += float(ll)

        elif self.model_type == "decoupled":
            param_idx = 0
            for key in self.survey_keys:
                if key not in self.cached_slices:
                    continue
                d_vec = theta[param_idx:param_idx + 3]
                ln_n0 = theta[param_idx + 3]
                gamma = theta[param_idx + 4]
                param_idx += 5

                sl = self.cached_slices[key]
                k = sl["k"]
                s = sl["s"]
                u = sl["u"]

                dipole_factor = 1.0 + (u @ d_vec)
                if np.any(dipole_factor <= 0.05):
                    return -1e12

                mu = np.exp(ln_n0) * (s ** gamma) * dipole_factor
                ll = np.sum(k * np.log(mu) - mu)
                total_ll += float(ll)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return total_ll

    def log_prior(self, theta: np.ndarray) -> float:
        """Regularizing Bayesian prior distributions for physical parameters."""
        lp = 0.0

        if self.model_type == "unified":
            beta_vec = theta[0:3]
            beta_mag = np.linalg.norm(beta_vec)
            # Physical bulk velocity: v_bulk < 3000 km/s (beta < 0.010)
            if beta_mag > 0.015:
                return -np.inf

            param_idx = 3
            for _ in self.survey_keys:
                ln_n0 = theta[param_idx]
                gamma = theta[param_idx + 1]
                param_idx += 2
                # Wide Gaussian prior on selection non-linearity: gamma ~ N(1.0, 0.5^2)
                lp += -0.5 * ((gamma - 1.0) / 0.5) ** 2
                # Bound unphysical normalization
                if ln_n0 < 0.0 or ln_n0 > 15.0:
                    return -np.inf

        elif self.model_type == "cmb_null":
            param_idx = 0
            for _ in self.survey_keys:
                ln_n0 = theta[param_idx]
                gamma = theta[param_idx + 1]
                param_idx += 2
                lp += -0.5 * ((gamma - 1.0) / 0.5) ** 2
                if ln_n0 < 0.0 or ln_n0 > 15.0:
                    return -np.inf

        elif self.model_type == "decoupled":
            param_idx = 0
            for _ in self.survey_keys:
                d_vec = theta[param_idx:param_idx + 3]
                ln_n0 = theta[param_idx + 3]
                gamma = theta[param_idx + 4]
                param_idx += 5
                # Dipole amplitude bounded |D| < 0.08
                if np.linalg.norm(d_vec) > 0.08:
                    return -np.inf
                lp += -0.5 * ((gamma - 1.0) / 0.5) ** 2
                if ln_n0 < 0.0 or ln_n0 > 15.0:
                    return -np.inf

        return lp

    def log_posterior(self, theta: np.ndarray) -> float:
        """Unnormalized log-posterior probability density."""
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood(theta)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll


def check_nested_model_comparison(
    result_h1: Dict[str, Any],
    result_h2: Dict[str, Any],
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare 'unified' (H_1) against 'decoupled' (H_2) run_mcmc results.

    H_1 is nested in H_2 (D_i = k_i * beta, same gamma prior), so the true maximum
    of the H_2 log-posterior must be >= that of H_1. run_mcmc reports the best
    *sampled* value, so a violation means the H_2 sampler did not reach its maximum
    (not converged) and any BIC/AIC difference built on it is unreliable.
    """
    lnl_h1 = float(result_h1["max_log_posterior"])
    lnl_h2 = float(result_h2["max_log_posterior"])
    violated = lnl_h2 < lnl_h1 - tol
    if violated:
        warnings.warn(
            f"NESTED-MODEL VIOLATION: max lnL(H_2)={lnl_h2:.3f} < max lnL(H_1)={lnl_h1:.3f}. "
            "H_1 is nested in H_2, so the H_2 sampler has not converged; "
            "the Delta-BIC/Delta-AIC between these runs is not trustworthy.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "max_log_posterior_h1": lnl_h1,
        "max_log_posterior_h2": lnl_h2,
        "delta_bic_h2_minus_h1": float(result_h2["bic"] - result_h1["bic"]),
        "delta_aic_h2_minus_h1": float(result_h2["aic"] - result_h1["aic"]),
        "nesting_violated": bool(violated),
    }


# --------------------------------------------------------------------------- #
# Affine-Invariant Ensemble MCMC Engine (Goodman & Weare 2010)
# --------------------------------------------------------------------------- #
class MultiTracerMCMCEngine:
    """Affine-invariant stretch-move ensemble sampler with Gelman-Rubin convergence tracking."""
    def __init__(
        self,
        likelihood: MultiTracerLikelihood,
        n_walkers: int = 32,
        seed: int = 42
    ):
        self.likelihood = likelihood
        self.n_walkers = n_walkers
        self.dim = self._get_dimension()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_dimension(self) -> int:
        if self.likelihood.model_type == "unified":
            return 3 + 2 * len(self.likelihood.survey_keys)  # 9
        elif self.likelihood.model_type == "cmb_null":
            return 2 * len(self.likelihood.survey_keys)      # 6
        elif self.likelihood.model_type == "decoupled":
            return 5 * len(self.likelihood.survey_keys)      # 15
        raise ValueError(f"Invalid model_type: {self.likelihood.model_type}")

    def initialize_walkers(self) -> np.ndarray:
        """Initializes walker swarm around plausible maximum-likelihood guesses."""
        p0 = np.zeros((self.n_walkers, self.dim), dtype=np.float64)

        if self.likelihood.model_type == "unified":
            beta_init = get_cmb_beta_vector()
            for w in range(self.n_walkers):
                p0[w, 0:3] = beta_init + self.rng.normal(0, 0.0002, size=3)
                idx = 3
                for key in self.likelihood.survey_keys:
                    s = self.likelihood.dataset.surveys[key]
                    mean_count = max(float(np.mean(s.counts[s.mask])), 1.0)
                    p0[w, idx] = np.log(mean_count) + self.rng.normal(0, 0.02)
                    p0[w, idx + 1] = 1.0 + self.rng.normal(0, 0.05)
                    idx += 2

        elif self.likelihood.model_type == "cmb_null":
            for w in range(self.n_walkers):
                idx = 0
                for key in self.likelihood.survey_keys:
                    s = self.likelihood.dataset.surveys[key]
                    mean_count = max(float(np.mean(s.counts[s.mask])), 1.0)
                    p0[w, idx] = np.log(mean_count) + self.rng.normal(0, 0.02)
                    p0[w, idx + 1] = 1.0 + self.rng.normal(0, 0.05)
                    idx += 2

        elif self.likelihood.model_type == "decoupled":
            for w in range(self.n_walkers):
                idx = 0
                for key in self.likelihood.survey_keys:
                    s = self.likelihood.dataset.surveys[key]
                    d_init = s.kinematic_factor * get_cmb_beta_vector()
                    p0[w, idx:idx + 3] = d_init + self.rng.normal(0, 0.001, size=3)
                    mean_count = max(float(np.mean(s.counts[s.mask])), 1.0)
                    p0[w, idx + 3] = np.log(mean_count) + self.rng.normal(0, 0.02)
                    p0[w, idx + 4] = 1.0 + self.rng.normal(0, 0.05)
                    idx += 5

        return p0

    def run_mcmc(
        self,
        n_steps: int = 1000,
        burn_in: int = 300,
        a: float = 2.0,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Runs the affine-invariant stretch-move sampler and evaluates diagnostics."""
        initial_state = self.initialize_walkers()
        chain = np.zeros((n_steps, self.n_walkers, self.dim), dtype=np.float64)
        log_probs = np.zeros((n_steps, self.n_walkers), dtype=np.float64)

        current_pos = np.copy(initial_state)
        current_log_prob = np.array([self.likelihood.log_posterior(current_pos[k]) for k in range(self.n_walkers)])

        chain[0] = current_pos
        log_probs[0] = current_log_prob
        accepted = 0
        total_proposals = 0
        half = self.n_walkers // 2

        for step in range(1, n_steps):
            # Split walkers into complementary sub-ensembles
            for sub in [slice(0, half), slice(half, self.n_walkers)]:
                comp = slice(half, self.n_walkers) if sub.start == 0 else slice(0, half)
                comp_indices = np.arange(comp.start, comp.stop)

                for i in range(sub.start, sub.stop):
                    total_proposals += 1
                    j = self.rng.choice(comp_indices)
                    # Stretch move proposal: z ~ g(z) = 1/sqrt(z) for z in [1/a, a]
                    u_val = self.rng.uniform(0.0, 1.0)
                    z = ((a - 1.0) * u_val + 1.0) ** 2 / a
                    proposal = current_pos[j] + z * (current_pos[i] - current_pos[j])

                    lp_proposal = self.likelihood.log_posterior(proposal)
                    # Hastings ratio: q = z^(dim - 1) * exp(lp_prop - lp_curr)
                    log_hastings = (self.dim - 1) * math.log(z) + (lp_proposal - current_log_prob[i])

                    if math.log(self.rng.uniform(0.0, 1.0)) < log_hastings:
                        current_pos[i] = proposal
                        current_log_prob[i] = lp_proposal
                        accepted += 1

            chain[step] = current_pos
            log_probs[step] = current_log_prob

        acceptance_fraction = accepted / max(total_proposals, 1)

        # Discard burn-in
        flat_chain = chain[burn_in:].reshape(-1, self.dim)
        flat_log_prob = log_probs[burn_in:].flatten()

        # Gelman-Rubin Diagnostic R_hat
        r_hat = self._compute_gelman_rubin(chain[burn_in:])

        # Maximum log-likelihood state
        max_idx = int(np.argmax(flat_log_prob))
        best_theta = flat_chain[max_idx]
        max_ll = float(flat_log_prob[max_idx])

        # Effective number of data points (HEALPix pixels)
        n_eff = sum(s["npix"] for s in self.likelihood.cached_slices.values())
        k_params = self.dim
        aic = 2 * k_params - 2 * max_ll
        bic = k_params * math.log(n_eff) - 2 * max_ll

        # Compute summary statistics with 95% Credible Intervals
        param_summaries = []
        for d in range(self.dim):
            samples = flat_chain[:, d]
            mean_val = float(np.mean(samples))
            std_val = float(np.std(samples))
            ci_low = float(np.percentile(samples, 2.5))
            ci_high = float(np.percentile(samples, 97.5))
            param_summaries.append({
                "dim": d,
                "mean": mean_val,
                "std": std_val,
                "ci_95": [ci_low, ci_high],
                "r_hat": float(r_hat[d]),
            })

        # Derived physical velocities if applicable
        bulk_velocity_summary = None
        if self.likelihood.model_type == "unified":
            beta_samples = flat_chain[:, 0:3]
            v_samples = np.linalg.norm(beta_samples, axis=1) * SPEED_OF_LIGHT_KMS
            l_samples = []
            b_samples = []
            for vec in beta_samples:
                l, b = unit_vector_to_lb(vec)
                l_samples.append(l)
                b_samples.append(b)

            bulk_velocity_summary = {
                "v_mean_kms": float(np.mean(v_samples)),
                "v_std_kms": float(np.std(v_samples)),
                "v_ci_95": [float(np.percentile(v_samples, 2.5)), float(np.percentile(v_samples, 97.5))],
                "apex_l_deg": float(np.median(l_samples)),
                "apex_b_deg": float(np.median(b_samples)),
                "v_cmb_ratio": float(np.mean(v_samples) / V_CMB_KMS),
                "separation_from_cmb_deg": float(SkyCoord(
                    np.median(l_samples) * u.deg, np.median(b_samples) * u.deg, frame="galactic"
                ).separation(SkyCoord(CMB_APEX_L_DEG * u.deg, CMB_APEX_B_DEG * u.deg, frame="galactic")).deg)
            }

        return {
            "model_type": self.likelihood.model_type,
            "acceptance_fraction": float(acceptance_fraction),
            "max_log_posterior": max_ll,
            "aic": aic,
            "bic": bic,
            "n_effective_pixels": n_eff,
            "num_parameters": k_params,
            "param_summaries": param_summaries,
            "bulk_velocity_summary": bulk_velocity_summary,
            "best_theta": best_theta.tolist(),
            "flat_chain": flat_chain,
            "flat_log_prob": flat_log_prob,
        }

    def _compute_gelman_rubin(self, walker_chains: np.ndarray) -> np.ndarray:
        """Computes the Gelman-Rubin convergence statistic R_hat across parallel walkers."""
        # walker_chains shape: (steps, walkers, dim)
        steps, walkers, dim = walker_chains.shape
        if steps < 2 or walkers < 2:
            return np.ones(dim)

        means = np.mean(walker_chains, axis=0)  # (walkers, dim)
        overall_mean = np.mean(means, axis=0)   # (dim,)

        # Between-chain variance B
        b = (steps / (walkers - 1)) * np.sum((means - overall_mean) ** 2, axis=0)

        # Within-chain variance W
        s2 = (1.0 / (steps - 1)) * np.sum((walker_chains - means[None, :, :]) ** 2, axis=0)  # (walkers, dim)
        w = np.mean(s2, axis=0)  # (dim,)

        # Marginal posterior variance estimate
        var_est = ((steps - 1) / steps) * w + (1.0 / steps) * b
        r_hat = np.sqrt(np.clip(var_est / np.maximum(w, 1e-12), 1.0, 100.0))
        return r_hat


# --------------------------------------------------------------------------- #
# Conformal Sky Residual Anomaly Filter
# --------------------------------------------------------------------------- #
class MultiTracerConformalFilter:
    """Dirichlet-calibrated Conformal Risk Filter for identifying sky residual anomalies."""
    def __init__(self, alpha_risk: float = 0.05):
        self.alpha_risk = alpha_risk

    def compute_conformal_residuals(
        self,
        survey: TracerSurvey,
        best_beta: np.ndarray,
        ln_n0: float,
        gamma: float
    ) -> Dict[str, Any]:
        """Calculates conformal non-conformity scores and bounds anomalous sky pixels."""
        m = survey.mask
        counts = survey.counts[m]
        selection = survey.selection[m]
        unit_vecs = survey.unit_vectors[m]

        k_fac = survey.kinematic_factor
        dipole_factor = 1.0 + k_fac * (unit_vecs @ best_beta)
        mu = np.exp(ln_n0) * (selection ** gamma) * dipole_factor

        # Standardized Pearson residuals: r_p = (k_p - mu_p) / sqrt(mu_p)
        residuals = (counts - mu) / np.sqrt(np.maximum(mu, 1e-6))
        abs_residuals = np.abs(residuals)

        # Conformal quantile at 1 - alpha_risk
        n_cal = len(residuals)
        q_level = math.ceil((n_cal + 1) * (1.0 - self.alpha_risk)) / n_cal
        q_level = min(max(q_level, 0.0), 1.0)
        tau_conformal = float(np.quantile(abs_residuals, q_level))

        # Flagged anomalous outlier pixels
        is_anomaly = abs_residuals > tau_conformal
        n_anomalies = int(np.sum(is_anomaly))
        anomaly_fraction = n_anomalies / max(n_cal, 1)

        return {
            "survey": survey.name,
            "tau_conformal": tau_conformal,
            "n_calibrated_pixels": n_cal,
            "n_anomalous_pixels": n_anomalies,
            "anomaly_fraction": anomaly_fraction,
            "target_risk_level": self.alpha_risk,
            "max_residual": float(np.max(abs_residuals)),
            "mean_residual": float(np.mean(abs_residuals)),
        }
