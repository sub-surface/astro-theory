"""DASCH DR7 (Harvard plates, 1885-1992) through the public StarGlass API.

Talks to the same endpoints as the `daschlab` package (querycat, lightcurve, queryexps, cutout) with plain
`requests`, so we avoid its heavy interactive dependencies. No key is needed for the public tier.
Responses are CSV lines inside JSON; we parse them into numpy column dicts and cache the raw text.

Quality flags follow daschlab 1.0 (`photometry.AFlags`, bits counted from 0).
"""
from __future__ import annotations

import base64
import gzip
import os
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import requests

API = "https://api.starglass.cfa.harvard.edu/public"
_HEAD = {"User-Agent": "daschlab", "Accept": "application/json"}

# AFLAGS bits (daschlab photometry.AFlags)
HIGH_BACKGROUND = 1 << 6
BAD_PLATE_QUALITY = 1 << 7
MULT_EXP_UNMATCHED = 1 << 8
UNCERTAIN_DATE = 1 << 9
MULT_EXP_BLEND = 1 << 10
LARGE_ISO_RMS = 1 << 11
LARGE_LOCAL_SMOOTH_RMS = 1 << 12
CLOSE_TO_LIMITING = 1 << 13
RADIAL_BIN_9 = 1 << 14
BIN_DRAD_UNKNOWN = 1 << 15
UNCERTAIN_CATALOG_MAG = 1 << 19
CASE_B_BLEND = 1 << 20
CASE_C_BLEND = 1 << 21
CASE_BC_BLEND = 1 << 22
LARGE_DRAD = 1 << 23
PICKERING_WEDGE = 1 << 24
# BFLAGS bits
B_SATURATED = 1 << 2
B_PSF_SATURATED = 1 << 10

BAD_AFLAGS = (HIGH_BACKGROUND | BAD_PLATE_QUALITY | MULT_EXP_UNMATCHED | UNCERTAIN_DATE | MULT_EXP_BLEND
              | BIN_DRAD_UNKNOWN | CASE_C_BLEND | CASE_BC_BLEND | PICKERING_WEDGE)
# Soft flags: kept with errors inflated. They fire on points far outside the plate's calibrated range, i.e. on
# exactly the bright outburst points we look for (26/27 of T CrB's Feb 1946 peak points carry them).
# CASE_B_BLEND marks detections that may belong to a fainter catalogue star nearby: a nova next to a neighbour
# (A4; position-based assembly in `assemble` decides attribution instead).
SOFT_AFLAGS = LARGE_ISO_RMS | LARGE_LOCAL_SMOOTH_RMS | LARGE_DRAD | CASE_B_BLEND
SOFT_ERR = 0.5
# plate-level problems that also void a non-detection limit
BAD_LIMIT_AFLAGS = HIGH_BACKGROUND | BAD_PLATE_QUALITY | UNCERTAIN_DATE | PICKERING_WEDGE


def _post(endpoint: str, payload: dict, retries: int = 4, timeout: float = 120):
    err = None
    for k in range(retries):
        try:
            r = requests.post(API + endpoint, json=payload, headers=_HEAD, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # network hiccup / lambda timeout
            err = repr(e)
        time.sleep(2 ** k)
    raise RuntimeError(f"DASCH {endpoint} failed: {err}")


def parse_csv_lines(lines: list[str]) -> dict[str, np.ndarray]:
    """CSV lines (header first) -> {column: array}; numeric where possible, NaN for blanks."""
    if not lines:
        return {}
    cols = lines[0].split(",")
    rows = [ln.split(",") for ln in lines[1:]]
    out = {}
    for j, c in enumerate(cols):
        vals = [r[j] if j < len(r) else "" for r in rows]
        try:
            out[c] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[c] = np.array(vals, dtype=object)
    return out


def querycat(ra: float, dec: float, radius_arcsec: float = 20.0, refcat: str = "atlas") -> list[str]:
    data = _post("/dasch/dr7/querycat", {"refcat": refcat, "ra_deg": float(ra), "dec_deg": float(dec),
                                         "radius_arcsec": float(radius_arcsec)})
    if not isinstance(data, list):
        raise RuntimeError(f"querycat failed: {data!r}")
    return data


def lightcurve(refcat: str, gsc_bin_index: int, ref_number: int) -> list[str]:
    data = _post("/dasch/dr7/lightcurve", {"refcat": refcat, "gsc_bin_index": int(gsc_bin_index),
                                           "ref_number": int(ref_number)})
    if not isinstance(data, list):
        raise RuntimeError(f"lightcurve failed: {data!r}")
    return data


def exposures(ra: float, dec: float) -> list[str]:
    data = _post("/dasch/dr7/queryexps", {"ra_deg": float(ra), "dec_deg": float(dec)})
    if not isinstance(data, list):
        raise RuntimeError(f"queryexps failed: {data!r}")
    return data


def cutout_fits(series: str, platenum: int, solnum: int, ra: float, dec: float) -> bytes:
    data = _post("/dasch/dr7/cutout", {"plate_id": f"{series}{int(platenum):05d}", "solution_number": int(solnum),
                                       "center_ra_deg": float(ra), "center_dec_deg": float(dec)})
    if not isinstance(data, str):
        raise RuntimeError(f"cutout failed: {data!r}")
    return gzip.decompress(base64.b64decode(data))


LC_KEEP = ("date_jd", "series", "plate_number", "solution_number", "mosaic_number", "limiting_mag_local",
           "magcal_magdep", "magcal_local_rms", "aflags", "bflags", "gsc_bin_index", "sextractor_number",
           "ra_deg", "dec_deg", "fwhm_deg", "ellipticity", "plate_center_dist_deg", "time_accuracy_days")


def reduce_lines(lines: list[str], keep=LC_KEEP) -> list[str]:
    """Keep only the named columns of CSV lines (cuts DASCH light curves ~8x)."""
    if not lines:
        return lines
    head = lines[0].split(",")
    idx = [head.index(c) for c in keep if c in head]
    return [",".join(r.split(",")[j] for j in idx) for r in lines]


def cached(path: Path, fetch, reduce: bool = False) -> list[str]:
    """Return cached CSV lines at `path` (.gz allowed), fetching (and writing) them on a miss."""
    if path.exists():
        raw = path.read_bytes()
        try:
            txt = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode("utf-8")
            return txt.split("\n") if txt else []
        except (OSError, EOFError, UnicodeDecodeError):  # truncated by an interrupted writer
            if fetch is None:
                warnings.warn(f"unreadable DASCH cache {path.name}; treated as empty")
                return []
    lines = fetch()
    if reduce:
        lines = reduce_lines(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "\n".join(lines).encode("utf-8")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(gzip.compress(data) if path.suffix == ".gz" else data)
    os.replace(tmp, path)  # atomic: readers never see a partial file
    return lines


def tidy(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reduce a DASCH light curve to what the detectors need.

    Returns arrays: t (Julian year), mag (calibrated B-like, NaN for non-detection), err, lim, det (bool),
    good (bool; quality cuts passed), near_lim (bool), sat (bool), series, plate.
    """
    if not cols:
        return {k: np.array([]) for k in ("t", "mag", "err", "lim", "det", "good", "near_lim", "sat")}
    jd = cols["date_jd"]
    t = 2000.0 + (jd - 2451545.0) / 365.25
    gsc = np.nan_to_num(cols.get("gsc_bin_index", np.zeros_like(jd)))
    sxt = np.nan_to_num(cols.get("sextractor_number", np.zeros_like(jd)))
    det = ~((gsc == 0) & (sxt == 0))
    mag = np.where(det, cols["magcal_magdep"], np.nan)
    det &= np.isfinite(mag)
    rms = cols.get("magcal_local_rms", np.full_like(jd, np.nan))
    err = np.where(np.isfinite(rms) & (rms > 0), rms, 0.25)
    err = np.clip(err, 0.08, 1.5)
    af = np.nan_to_num(cols.get("aflags", np.zeros_like(jd))).astype(np.int64)
    bf = np.nan_to_num(cols.get("bflags", np.zeros_like(jd))).astype(np.int64)
    lim = cols["limiting_mag_local"]
    good_det = det & ((af & BAD_AFLAGS) == 0)
    good_lim = (~det) & np.isfinite(lim) & ((af & BAD_LIMIT_AFLAGS) == 0) & (lim > 5)
    near = det & ((af & CLOSE_TO_LIMITING) != 0)
    err = np.where(near, np.hypot(err, 0.3), err)
    err = np.where(det & ((af & SOFT_AFLAGS) != 0), np.hypot(err, SOFT_ERR), err)
    sat = det & (((bf & B_SATURATED) != 0) | ((bf & B_PSF_SATURATED) != 0))
    o = np.argsort(t)
    out = dict(t=t, mag=mag, err=err, lim=lim, det=det, good=good_det | good_lim, near_lim=near, sat=sat,
               series=cols.get("series", np.array([""] * len(t), dtype=object)),
               plate=cols.get("plate_number", np.zeros_like(t)))
    return {k: np.asarray(v)[o] for k, v in out.items()}


def assemble(own: dict, neighbours: list, ra: float, dec: float, r_det: float = 6.0, r_own: float = 15.0,
             r_resolve: float = 8.0, excess: float = 1.0) -> dict:
    """Position-based light curve at (ra, dec), undoing DASCH source splitting.

    `own` is the column dict of the refcat source matched to the position; `neighbours` is a list of
    (column dict, nb_ra, nb_dec) for refcat sources nearby. A detection is kept if its measured position is
    within `r_det` arcsec of (ra, dec) (`r_own` for the matched source itself, whose bright images scatter) and closer to (ra, dec) than to its owner's catalogue position. On each
    plate exposure (series, plate, solution) the detection nearest the position wins; exposures with no kept
    detection contribute the own source's non-detection limit.
    """
    cosd = np.cos(np.radians(dec))

    def dist(c, r0, d0):
        return np.hypot((c["ra_deg"] - r0) * np.cos(np.radians(d0)), c["dec_deg"] - d0) * 3600

    def keys(c):
        return [f"{s}|{int(p)}|{int(np.nan_to_num(q))}" for s, p, q in
                zip(c["series"], np.nan_to_num(c["plate_number"]), c.get("solution_number", np.zeros(len(c["series"]))))]

    pieces = []  # (cols, row indices, distances)
    if own:
        det = np.isfinite(own["magcal_magdep"])
        dt = np.hypot((own["ra_deg"] - ra) * cosd, own["dec_deg"] - dec) * 3600
        pieces.append((own, np.where(det & (dt <= r_own))[0], dt))
    for cols, nra, ndec in neighbours:
        if not cols:
            continue
        det = np.isfinite(cols["magcal_magdep"])
        dt = np.hypot((cols["ra_deg"] - ra) * cosd, cols["dec_deg"] - dec) * 3600
        dn = dist(cols, nra, ndec)
        sep = float(np.hypot((nra - ra) * cosd, ndec - dec) * 3600)
        if sep < r_resolve:
            med = np.nanmedian(cols["magcal_magdep"]) if det.sum() >= 3 else np.inf
            ok = det & (dt <= r_det) & (cols["magcal_magdep"] <= med - excess)
        else:
            ok = det & (dt <= r_det) & (2 * dt < dn)
        pieces.append((cols, np.where(ok)[0], dt))
    best = {}
    for k, (cols, idx, dt) in enumerate(pieces):
        ks = keys(cols)
        for i in idx:
            if ks[i] not in best or dt[i] < best[ks[i]][2]:
                best[ks[i]] = (k, i, dt[i])
    names = list(own.keys()) if own else list(pieces[0][0].keys())
    rows = {n: [] for n in names}
    for k, i, _ in best.values():
        cols = pieces[k][0]
        for n in names:
            rows[n].append(cols[n][i] if n in cols else np.nan)
    if own:
        ks = keys(own)
        nd = ~np.isfinite(own["magcal_magdep"])
        for i in np.where(nd)[0]:
            if ks[i] not in best:
                for n in names:
                    rows[n].append(own[n][i])
    out = {}
    for n in names:
        try:
            out[n] = np.array(rows[n], dtype=float)
        except (ValueError, TypeError):
            out[n] = np.array(rows[n], dtype=object)
    return out
