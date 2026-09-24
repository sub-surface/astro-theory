"""Hermetic mock catalogue and null suite generator for Euclid DR1.

Validates the analytic forecast against Monte Carlo Poisson realizations on
the realistic DR1 partial-sky footprint, calibrating empirical significance
thresholds and testing bias and variance.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Any
import numpy as np
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u

from . import forecast


def _unit_vectors(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype="float64"))
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


def _vector_to_lonlat(vec: np.ndarray) -> Tuple[float, float]:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return 0.0, 0.0
    uvec = vec / norm
    lat = float(np.rad2deg(np.arcsin(np.clip(uvec[2], -1.0, 1.0))))
    lon = float(np.rad2deg(np.arctan2(uvec[1], uvec[0])) % 360.0)
    return lon, lat


def fit_dipole(
    counts: np.ndarray,
    mask: np.ndarray,
    nside: int = 32,
    frame: str = "galactic",
) -> Dict[str, Any]:
    """Fit a dipole n = a0(1 + D·n̂) to pixel counts on the mask.

    Parameters
    ----------
    counts : np.ndarray
        Pixel counts array of shape (npix,).
    mask : np.ndarray
        Boolean mask array of shape (npix,), True = observed.
    nside : int
        HEALPix grid resolution.
    frame : str
        Coordinate frame ("galactic" or "icrs").
    """
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))
    coords = SkyCoord(lon, lat, frame="icrs")
    if frame == "galactic":
        px_lon = coords.galactic.l.deg
        px_lat = coords.galactic.b.deg
    else:
        px_lon = coords.icrs.ra.deg
        px_lat = coords.icrs.dec.deg

    keep = mask & (counts > 0)
    if keep.sum() < 4:
        raise ValueError("Too few non-zero pixels to fit dipole")

    cnt = counts[keep].astype(float)
    unit = _unit_vectors(px_lon[keep], px_lat[keep])
    design = np.column_stack([np.ones(len(unit)), unit])

    # Poisson weighted least squares (variance = count)
    weights = 1.0 / np.maximum(cnt, 1.0)
    xtw = design.T * weights
    normal = xtw @ design
    cov = np.linalg.inv(normal)
    beta = cov @ (xtw @ cnt)

    monopole = float(beta[0])
    if abs(monopole) < 1e-12:
        raise ValueError("Dipole fit found zero monopole")

    vector = beta[1:] / monopole
    amplitude = float(np.linalg.norm(vector))
    dir_lon, dir_lat = _vector_to_lonlat(vector)

    return {
        "amplitude": amplitude,
        "monopole": monopole,
        "vector": vector.tolist(),
        "dir_lon": dir_lon,
        "dir_lat": dir_lat,
        "frame": frame,
        "n_sources": float(cnt.sum()),
        "n_pixels": int(keep.sum()),
    }


def generate_pixel_mock(
    mask: np.ndarray,
    n_sources: float,
    amplitude: float = 0.0,
    lon: float = 264.021,
    lat: float = 48.253,
    nside: int = 32,
    frame: str = "galactic",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a single Poisson realisation with injected dipole modulation.

    Parameters
    ----------
    mask : np.ndarray
        HEALPix footprint mask (True = observed).
    n_sources : float
        Target total source count across the mask.
    amplitude : float
        Injected dipole amplitude (e.g. 0.0047 for kinematic, 0.0 for null).
    lon, lat : float
        Injected dipole apex direction in coordinates of `frame`.
    nside : int
        HEALPix grid resolution.
    frame : str
        Coordinate frame ("galactic" or "icrs").
    rng : np.random.Generator, optional
        Numpy random generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    hp = HEALPix(nside=nside, order="ring")
    lon_pix, lat_pix = hp.healpix_to_lonlat(np.arange(hp.npix))
    coords = SkyCoord(lon_pix, lat_pix, frame="icrs")
    if frame == "galactic":
        px_lon = coords.galactic.l.deg
        px_lat = coords.galactic.b.deg
    else:
        px_lon = coords.icrs.ra.deg
        px_lat = coords.icrs.dec.deg

    unit_pix = _unit_vectors(px_lon, px_lat)
    dir_unit = _unit_vectors(np.array([lon]), np.array([lat]))[0]

    modulation = np.clip(1.0 + amplitude * (unit_pix @ dir_unit), 0.0, None)
    surv_pixels = int(mask.sum())
    if surv_pixels == 0:
        raise ValueError("Mask has no surviving pixels")

    mean_per_pix = n_sources / surv_pixels
    expected = np.zeros(hp.npix)
    expected[mask] = mean_per_pix * modulation[mask]

    counts = rng.poisson(expected)
    return counts


def run_mock_suite(
    n_mocks: int = 50,
    area_deg2: float = 1900.0,
    n_sources: float = 100000.0,
    amplitude: float = 0.0047,
    lon: float = 264.021,
    lat: float = 48.253,
    nside: int = 32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run signal and null mock suites to validate variance and significance.

    Parameters
    ----------
    n_mocks : int
        Number of Monte Carlo realizations.
    area_deg2 : float
        Footprint area in square degrees.
    n_sources : float
        Total source count.
    amplitude : float
        Injected signal amplitude (e.g. kinematic CMB dipole).
    lon, lat : float
        Injected direction (galactic coordinates).
    nside : int
        HEALPix grid resolution.
    seed : int
        Random seed for hermetic repeatability.

    Returns
    -------
    dict
        Suite report with recovered signal and null distributions.
    """
    mask, meta = forecast.build_dr1_footprint(nside=nside, area_deg2=area_deg2)
    rng = np.random.default_rng(seed)

    signal_amps = []
    null_amps = []

    # 1. Injected signal mocks
    for _ in range(n_mocks):
        cnt = generate_pixel_mock(
            mask=mask,
            n_sources=n_sources,
            amplitude=amplitude,
            lon=lon,
            lat=lat,
            nside=nside,
            frame="galactic",
            rng=rng,
        )
        fit = fit_dipole(cnt, mask, nside=nside, frame="galactic")
        signal_amps.append(fit["amplitude"])

    # 2. Isotropic null mocks (amplitude = 0)
    for _ in range(n_mocks):
        cnt = generate_pixel_mock(
            mask=mask,
            n_sources=n_sources,
            amplitude=0.0,
            nside=nside,
            frame="galactic",
            rng=rng,
        )
        fit = fit_dipole(cnt, mask, nside=nside, frame="galactic")
        null_amps.append(fit["amplitude"])

    signal_amps = np.array(signal_amps)
    null_amps = np.array(null_amps)

    _, analytic_sigma_shot, _ = forecast.compute_dipole_fisher(
        mask, n_sources, nside=nside
    )

    return {
        "n_mocks": n_mocks,
        "n_sources": n_sources,
        "injected_amplitude": amplitude,
        "injected_l": lon,
        "injected_b": lat,
        "area_deg2": meta["actual_area_deg2"],
        "nside": nside,
        "analytic_sigma_shot": analytic_sigma_shot,
        "signal": {
            "mean": float(np.mean(signal_amps)),
            "std": float(np.std(signal_amps, ddof=1)),
            "p16": float(np.percentile(signal_amps, 16)),
            "p50": float(np.percentile(signal_amps, 50)),
            "p84": float(np.percentile(signal_amps, 84)),
        },
        "null": {
            "mean": float(np.mean(null_amps)),
            "std": float(np.std(null_amps, ddof=1)),
            "p95": float(np.percentile(null_amps, 95)),
            "p99": float(np.percentile(null_amps, 99)),
            "empirical_3sigma": float(np.percentile(null_amps, 99.73)),
        },
        "ratio_mock_to_analytic_sigma": float(np.std(signal_amps, ddof=1) / analytic_sigma_shot),
    }
