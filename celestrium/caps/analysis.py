"""The validation wing's missing floor — sky statistics that produce *answers*.

Until now Celestrium could fetch rows and draw quick-look plots, but the actual
science (the shelved dipole pipeline in `Archive/2026-06-G-dipole/`, 1,586
lines) lived outside the instrument and imported none of it. These capabilities
are that pipeline, generalised: pixelise → mask → fit → and, crucially, the two
controls that decide whether a result is real (`bootstrap`, `null_shuffle`).

Everything here is pure and offline — it runs on an artifact produced upstream,
so a whole study is cached, deduped, and provenance-linked end to end.

Convention: a *density* artifact is one row per HEALPix pixel with columns
``ipix · count · lon · lat · coverage``. `coverage` is the fraction of the pixel
that survives the mask (0 = excluded), which is what lets masking stay a
separate, auditable step instead of being buried in the fit.
"""
from __future__ import annotations

import numpy as np

from ..core.capability import Param, capability

DENSITY_IN = Param("artifact", help="artifact id of a sky-density map")
FRAMES = ("icrs", "galactic")
C_KMS = 299792.458


# --------------------------------------------------------------------------- #
# shared maths
# --------------------------------------------------------------------------- #
def _unit_vectors(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lon_deg, dtype="float64"))
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def _vector_to_lonlat(vec: np.ndarray) -> tuple:
    vec = vec / np.linalg.norm(vec)
    lat = np.rad2deg(np.arcsin(np.clip(vec[2], -1.0, 1.0)))
    lon = np.rad2deg(np.arctan2(vec[1], vec[0])) % 360.0
    return float(lon), float(lat)


def _fit(unit: np.ndarray, counts: np.ndarray, coverage: np.ndarray) -> dict:
    """Weighted least squares of n_i = a0 (1 + D·n̂_i) over surviving pixels.

    Poisson weights (variance = counts, floored at 1) scaled by coverage², so a
    partially-covered pixel contributes proportionally rather than being
    all-or-nothing. Returns the dipole vector, amplitude, direction, and the
    amplitude uncertainty propagated from the fit covariance.
    """
    design = np.column_stack([np.ones(len(unit)), unit])
    density = counts / np.maximum(coverage, 1e-12)
    variance = np.maximum(counts, 1.0) / np.maximum(coverage, 1e-12) ** 2
    weights = 1.0 / variance

    xtw = design.T * weights
    normal = xtw @ design
    try:
        cov = np.linalg.inv(normal)
    except np.linalg.LinAlgError as exc:
        raise ValueError("dipole fit is singular — too few surviving pixels") from exc
    beta = cov @ (xtw @ density)

    monopole = float(beta[0])
    if abs(monopole) < 1e-12:
        raise ValueError("dipole fit found a zero monopole (empty map?)")
    vector = beta[1:] / monopole
    amplitude = float(np.linalg.norm(vector))

    if amplitude > 0:
        direction = vector / amplitude
        var_amp = float(direction @ cov[1:, 1:] @ direction) / monopole ** 2
    else:
        direction = np.array([1.0, 0.0, 0.0])
        var_amp = float(np.mean(np.diag(cov)[1:])) / monopole ** 2
    return {
        "vector": vector, "amplitude": amplitude,
        "sigma_amplitude": float(np.sqrt(max(var_amp, 0.0))),
        "monopole": monopole, "direction": direction,
    }


def _load_density(ctx, ref):
    table = ctx.load(ref)
    art = ctx.artifact(ref)
    frame = (art.meta.get("frame") if art else None) or "icrs"
    coverage = (np.asarray(table["coverage"], dtype="float64")
                if "coverage" in table.colnames else np.ones(len(table)))
    keep = coverage > 0
    if keep.sum() < 16:
        raise ValueError(f"only {int(keep.sum())} pixels survive the mask — "
                         "not enough to fit")
    return table, frame, coverage, keep


def _directions(direction: np.ndarray, frame: str) -> dict:
    """Report the fitted direction in the map frame *and* in galactic/icrs."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    lon, lat = _vector_to_lonlat(direction)
    coord = SkyCoord(lon * u.deg, lat * u.deg,
                     frame="galactic" if frame == "galactic" else "icrs")
    return {
        "lon": lon, "lat": lat, "frame": frame,
        "ra": float(coord.icrs.ra.deg), "dec": float(coord.icrs.dec.deg),
        "l": float(coord.galactic.l.deg), "b": float(coord.galactic.b.deg),
    }


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
@capability(
    name="analysis.synthetic_sky", kind="table", wing="validation", cost="free",
    params={"nsources": Param("int", 100000),
            "amplitude": Param("float", 0.0, help="injected dipole amplitude"),
            "lon": Param("float", 264.021, help="injection direction, galactic l"),
            "lat": Param("float", 48.253, help="injection direction, galactic b"),
            "sky_fraction": Param("float", 1.0, help="target footprint fraction"),
            "gal_lat_min": Param("float", 0.0), "ecl_lat_min": Param("float", 0.0),
            "seed": Param("int", 1), "nside_mask": Param("int", 16)},
    summary="Mock catalogue with a known dipole on a realistic footprint.",
)
def synthetic_sky(ctx, nsources, amplitude, lon, lat, sky_fraction, gal_lat_min,
                  ecl_lat_min, seed, nside_mask):
    """The forecast primitive: inject a dipole you know, measure what comes back.

    This is how "measure or rehearse?" gets answered *before* a survey lands —
    build the footprint, inject an amplitude, run the identical downstream
    pipeline, and read σ_D off the recovered value. Offline and free.

    The injected modulation is applied per footprint pixel (nside_mask), so it
    is piecewise-constant at ~3.7° for the default — negligible against an ℓ=1
    signal, and it keeps 10⁶-source draws instant.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy_healpix import HEALPix

    healpix = HEALPix(nside=nside_mask, order="ring")
    centre_lon, centre_lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))
    centres = SkyCoord(centre_lon, centre_lat, frame="galactic")

    allowed = np.ones(healpix.npix, dtype=bool)
    if gal_lat_min > 0:
        allowed &= np.abs(centres.b.deg) >= gal_lat_min
    if ecl_lat_min > 0:
        allowed &= np.abs(centres.barycentrictrueecliptic.lat.deg) >= ecl_lat_min

    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(allowed)
    if pool.size == 0:
        raise ValueError("footprint cuts leave no sky")
    wanted = int(round(np.clip(sky_fraction, 1e-4, 1.0) * healpix.npix))
    if 0 < wanted < pool.size:
        pool = np.sort(rng.choice(pool, size=wanted, replace=False))
    ctx.progress(f"footprint: {pool.size}/{healpix.npix} pixels "
                 f"({pool.size / healpix.npix * 100:.2f}% of sky)")

    direction = _unit_vectors(np.array([lon]), np.array([lat]))[0]
    unit = _unit_vectors(centre_lon.deg[pool], centre_lat.deg[pool])
    weights = np.clip(1.0 + amplitude * (unit @ direction), 0.0, None)
    total = weights.sum()
    if total <= 0:
        raise ValueError("injected amplitude leaves no positive density")
    weights = weights / total

    count = max(1, int(nsources))
    ctx.progress(f"drawing {count} sources (A={amplitude:g})")
    drawn = rng.choice(pool, size=count, p=weights)
    source_lon, source_lat = healpix.healpix_to_lonlat(
        drawn, dx=rng.random(count), dy=rng.random(count))
    icrs = SkyCoord(source_lon, source_lat, frame="galactic").icrs

    table = Table({"ra": icrs.ra.deg, "dec": icrs.dec.deg})
    return ctx.table(table, label=f"synthetic {count} (A={amplitude:g})",
                     injected_amplitude=amplitude, injected_l=lon, injected_b=lat,
                     sky_fraction=float(pool.size / healpix.npix),
                     nsources=count, seed=seed, synthetic=True)


@capability(
    name="analysis.sky_density", kind="table", wing="validation", cost="free",
    params={"table": Param("artifact", help="source catalogue artifact"),
            "nside": Param("int", 64), "frame": Param("enum", "galactic", FRAMES),
            "ra_col": Param("str", "auto"), "dec_col": Param("str", "auto"),
            "empty": Param("enum", "unobserved", ("unobserved", "observed"),
                           help="how to treat pixels containing no sources")},
    summary="Pixelise a source catalogue into a HEALPix number-count map.",
)
def sky_density(ctx, table, nside, frame, ra_col, dec_col, empty):
    """Count sources per HEALPix pixel.

    `empty` is the footprint assumption, and it is not a detail: a catalogue
    covering 5% of the sky produces ~95% empty pixels, and feeding those to a
    dipole fit as genuine zero-density measurements manufactures an enormous
    dipole pointing away from the footprint. The default (`unobserved`) marks
    empty pixels `coverage = 0` so they are excluded. Pass `observed` only for
    a genuinely all-sky catalogue where a zero really means zero — and for real
    survey data, prefer supplying a completeness map via `analysis.mask`.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy_healpix import HEALPix
    from .. import tables as table_utils

    source = ctx.load(table)
    if ra_col == "auto" or dec_col == "auto":
        found_ra, found_dec = table_utils.find_coord_columns(source)
        ra_col = found_ra if ra_col == "auto" else ra_col
        dec_col = found_dec if dec_col == "auto" else dec_col

    ra = np.asarray(source[ra_col], dtype="float64")
    dec = np.asarray(source[dec_col], dtype="float64")
    good = np.isfinite(ra) & np.isfinite(dec) & (np.abs(dec) <= 90.0)
    dropped = int((~good).size - good.sum()) if good.size else 0
    if dropped:
        ctx.note(f"dropped {dropped} rows with unusable coordinates")
    ra, dec = ra[good], dec[good]
    if ra.size == 0:
        raise ValueError("no usable coordinates in this table")

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    if frame == "galactic":
        lon = coord.galactic.l.deg
        lat = coord.galactic.b.deg
    else:
        lon, lat = ra, dec

    healpix = HEALPix(nside=nside, order="ring")
    ctx.progress(f"pixelising {ra.size} sources at nside={nside} ({frame})")
    ipix = healpix.lonlat_to_healpix(lon * u.deg, lat * u.deg)
    counts = np.bincount(np.asarray(ipix), minlength=healpix.npix)
    centre_lon, centre_lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))

    occupied = int((counts > 0).sum())
    coverage = (np.ones(healpix.npix, dtype="float64") if empty == "observed"
                else (counts > 0).astype("float64"))
    out = Table({
        "ipix": np.arange(healpix.npix, dtype="int64"),
        "count": counts.astype("int64"),
        "lon": centre_lon.deg, "lat": centre_lat.deg,
        "coverage": coverage,
    })
    sky_fraction = float(coverage.sum()) / healpix.npix
    ctx.progress(f"{occupied}/{healpix.npix} pixels occupied "
                 f"(footprint {sky_fraction * 100:.1f}% of sky, empty={empty})")
    if empty == "observed" and occupied < 0.5 * healpix.npix:
        ctx.note(f"empty='observed' with only {occupied}/{healpix.npix} pixels "
                 "occupied — empty sky is being fitted as real zero density, "
                 "which manufactures a footprint-shaped dipole")
    return ctx.table(out, label=f"density nside={nside} ({frame})",
                     nside=nside, frame=frame, nsources=int(ra.size),
                     npix=int(healpix.npix), occupied=occupied,
                     sky_fraction=sky_fraction, empty=empty,
                     mean_count=float(counts[counts > 0].mean()) if occupied else 0.0)


@capability(
    name="analysis.mask", kind="table", wing="validation", cost="free",
    params={"density": DENSITY_IN,
            "gal_lat_min": Param("float", 0.0, help="|b| cut in degrees"),
            "ecl_lat_min": Param("float", 0.0, help="|ecliptic lat| cut in degrees"),
            "dec_min": Param("float", -90.0), "dec_max": Param("float", 90.0),
            "min_count": Param("int", 0, help="drop pixels below this count")},
    summary="Zero a density map's coverage where a cut applies (auditable masking).",
)
def analysis_mask(ctx, density, gal_lat_min, ecl_lat_min, dec_min, dec_max, min_count):
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    table, frame, coverage, _ = _load_density(ctx, density)
    table = table.copy()
    coord = SkyCoord(np.asarray(table["lon"]) * u.deg,
                     np.asarray(table["lat"]) * u.deg,
                     frame="galactic" if frame == "galactic" else "icrs")

    keep = coverage > 0
    if gal_lat_min > 0:
        keep &= np.abs(coord.galactic.b.deg) >= gal_lat_min
    if ecl_lat_min > 0:
        keep &= np.abs(coord.barycentrictrueecliptic.lat.deg) >= ecl_lat_min
    if dec_min > -90.0 or dec_max < 90.0:
        dec = coord.icrs.dec.deg
        keep &= (dec >= dec_min) & (dec <= dec_max)
    if min_count > 0:
        # Note in the artifact: a count-based cut is *not* geometry, it removes
        # low-density sky and can itself imprint a dipole. Kept explicit.
        keep &= np.asarray(table["count"]) >= min_count
        ctx.note("min_count is a depth cut, not a footprint cut — vary it and "
                 "check the answer moves the way you expect")

    table["coverage"] = np.where(keep, coverage, 0.0)
    fraction = float(keep.sum()) / max(len(table), 1)
    ctx.progress(f"{int(keep.sum())}/{len(table)} pixels survive "
                 f"({fraction * 100:.1f}% of sky)")
    return ctx.table(table, label=f"mask |b|>{gal_lat_min:g}°", frame=frame,
                     nside=(ctx.artifact(density).meta.get("nside")
                            if ctx.artifact(density) else None),
                     sky_fraction=fraction, surviving_pixels=int(keep.sum()),
                     gal_lat_min=gal_lat_min, ecl_lat_min=ecl_lat_min,
                     min_count=min_count)


@capability(
    name="analysis.dipole_fit", kind="data", wing="validation", cost="free",
    params={"density": DENSITY_IN},
    summary="Fit n = a0(1 + D·n̂) over surviving pixels → amplitude + direction.",
)
def dipole_fit(ctx, density):
    table, frame, coverage, keep = _load_density(ctx, density)
    counts = np.asarray(table["count"], dtype="float64")[keep]
    unit = _unit_vectors(np.asarray(table["lon"])[keep],
                         np.asarray(table["lat"])[keep])
    result = _fit(unit, counts, coverage[keep])

    total = float(counts.sum())
    shot_noise = 3.0 / np.sqrt(total) if total > 0 else float("nan")
    payload = {
        "amplitude": result["amplitude"],
        "sigma_amplitude": result["sigma_amplitude"],
        "significance": (result["amplitude"] / result["sigma_amplitude"]
                         if result["sigma_amplitude"] > 0 else float("nan")),
        "shot_noise_amplitude": shot_noise,
        "direction": _directions(result["direction"], frame),
        "monopole": result["monopole"],
        "n_sources": total, "n_pixels": int(keep.sum()),
        "sky_fraction": float(keep.sum()) / max(len(table), 1),
        "frame": frame,
    }
    ctx.progress(f"D = {payload['amplitude']:.5f} ± {payload['sigma_amplitude']:.5f} "
                 f"toward (l={payload['direction']['l']:.1f}°, "
                 f"b={payload['direction']['b']:.1f}°)")
    return ctx.data(payload, label=f"dipole {payload['amplitude']:.5f}",
                    amplitude=payload["amplitude"],
                    significance=payload["significance"])


@capability(
    name="analysis.multipoles", kind="data", wing="validation", cost="free",
    params={"density": DENSITY_IN, "lmax": Param("int", 3)},
    summary="Power in ℓ = 0…lmax — is the dipole special, or is the map just lumpy?",
)
def multipoles(ctx, density, lmax):
    table, frame, coverage, keep = _load_density(ctx, density)
    if not 1 <= lmax <= 3:
        raise ValueError("lmax must be 1, 2 or 3")
    counts = np.asarray(table["count"], dtype="float64")[keep]
    unit = _unit_vectors(np.asarray(table["lon"])[keep],
                         np.asarray(table["lat"])[keep])
    x, y, z = unit[:, 0], unit[:, 1], unit[:, 2]

    basis = {0: [np.ones_like(x)], 1: [x, y, z]}
    if lmax >= 2:
        basis[2] = [x * y, y * z, z * x, x ** 2 - y ** 2, 3 * z ** 2 - 1]
    if lmax >= 3:
        basis[3] = [x * (x ** 2 - 3 * y ** 2), y * (3 * x ** 2 - y ** 2),
                    z * (x ** 2 - y ** 2), x * y * z,
                    x * (5 * z ** 2 - 1), y * (5 * z ** 2 - 1), z * (5 * z ** 2 - 3)]

    columns, degrees = [], []
    for degree in range(0, lmax + 1):
        for column in basis[degree]:
            # Normalise each column to unit RMS over surviving pixels so
            # coefficients are comparable across ℓ (the raw bases are not
            # orthonormal, and the mask breaks orthogonality anyway).
            rms = np.sqrt(np.mean(column ** 2)) or 1.0
            columns.append(column / rms)
            degrees.append(degree)

    design = np.column_stack(columns)
    weights = coverage[keep] ** 2 / np.maximum(counts, 1.0)
    xtw = design.T * weights
    beta = np.linalg.lstsq(xtw @ design, xtw @ (counts / coverage[keep]), rcond=None)[0]

    degrees = np.array(degrees)
    monopole = abs(beta[0]) or 1.0
    power = {int(degree): float(np.sum(beta[degrees == degree] ** 2)) / monopole ** 2
             for degree in range(0, lmax + 1)}
    ctx.progress(" · ".join(f"C{d}={v:.3e}" for d, v in power.items()))
    return ctx.data({"power": power, "lmax": lmax, "frame": frame,
                     "n_pixels": int(keep.sum())},
                    label=f"multipoles ℓ≤{lmax}")


@capability(
    name="analysis.bootstrap", kind="data", wing="validation", cost="free",
    params={"density": DENSITY_IN, "n": Param("int", 200), "seed": Param("int", 0)},
    summary="Resample pixels to get an empirical error on the dipole amplitude.",
)
def bootstrap(ctx, density, n, seed):
    table, frame, coverage, keep = _load_density(ctx, density)
    counts = np.asarray(table["count"], dtype="float64")[keep]
    unit = _unit_vectors(np.asarray(table["lon"])[keep],
                         np.asarray(table["lat"])[keep])
    cover = coverage[keep]
    observed = _fit(unit, counts, cover)

    rng = np.random.default_rng(seed)
    size = len(counts)
    amplitudes = []
    for draw in range(max(1, n)):
        if draw % 50 == 0:
            ctx.check_cancel()
            ctx.progress(f"bootstrap {draw}/{n}")
        pick = rng.integers(0, size, size)
        try:
            amplitudes.append(_fit(unit[pick], counts[pick], cover[pick])["amplitude"])
        except ValueError:
            continue
    amplitudes = np.array(amplitudes)
    payload = {
        "observed_amplitude": observed["amplitude"],
        "mean": float(amplitudes.mean()), "sd": float(amplitudes.std(ddof=1)),
        "p16": float(np.percentile(amplitudes, 16)),
        "p50": float(np.percentile(amplitudes, 50)),
        "p84": float(np.percentile(amplitudes, 84)),
        "draws": int(amplitudes.size), "seed": seed, "frame": frame,
    }
    return ctx.data(payload, label=f"bootstrap σ={payload['sd']:.5f}")


@capability(
    name="analysis.null_shuffle", kind="data", wing="validation", cost="free",
    params={"density": DENSITY_IN, "n": Param("int", 200), "seed": Param("int", 0)},
    summary="Shuffle counts across the same mask → the null the signal must beat.",
)
def null_shuffle(ctx, density, n, seed):
    """The control that decides whether an amplitude means anything: keep the
    footprint and the count distribution, destroy the sky positions. Any dipole
    that survives *this* is a property of the mask, not of the universe."""
    table, frame, coverage, keep = _load_density(ctx, density)
    counts = np.asarray(table["count"], dtype="float64")[keep]
    unit = _unit_vectors(np.asarray(table["lon"])[keep],
                         np.asarray(table["lat"])[keep])
    cover = coverage[keep]
    observed = _fit(unit, counts, cover)["amplitude"]

    rng = np.random.default_rng(seed)
    nulls = []
    for draw in range(max(1, n)):
        if draw % 50 == 0:
            ctx.check_cancel()
            ctx.progress(f"null {draw}/{n}")
        shuffled = rng.permutation(counts)
        try:
            nulls.append(_fit(unit, shuffled, cover)["amplitude"])
        except ValueError:
            continue
    nulls = np.array(nulls)
    exceed = int((nulls >= observed).sum())
    payload = {
        "observed_amplitude": float(observed),
        "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)),
        "null_p95": float(np.percentile(nulls, 95)),
        "p_value": (exceed + 1) / (nulls.size + 1),
        "draws": int(nulls.size), "seed": seed, "frame": frame,
    }
    ctx.progress(f"p = {payload['p_value']:.4f} "
                 f"(null mean {payload['null_mean']:.5f})")
    return ctx.data(payload, label=f"null p={payload['p_value']:.4f}")


@capability(
    name="analysis.kinematic_dipole", kind="data", wing="validation", cost="free",
    params={"velocity_kms": Param("float", 369.82), "alpha": Param("float", 0.75),
            "x_slope": Param("float", 1.7)},
    summary="Ellis–Baldwin kinematic prediction D = [2 + x(1+α)]·v/c.",
)
def kinematic_dipole(ctx, velocity_kms, alpha, x_slope):
    beta = velocity_kms / C_KMS
    amplitude = (2.0 + x_slope * (1.0 + alpha)) * beta
    return ctx.data({
        "amplitude": float(amplitude), "beta": float(beta),
        "velocity_kms": velocity_kms, "alpha": alpha, "x_slope": x_slope,
        "note": "CMB-frame expectation for number counts (Ellis & Baldwin 1984)",
    }, label=f"kinematic D={amplitude:.5f}", amplitude=float(amplitude))


@capability(
    name="analysis.compare", kind="data", wing="validation", cost="free",
    params={"measured": Param("artifact"), "expected": Param("artifact")},
    summary="Ratio and tension between a measured dipole and a prediction.",
)
def compare(ctx, measured, expected):
    got = ctx.load(measured)
    want = ctx.load(expected)
    amplitude = float(got.get("amplitude", float("nan")))
    sigma = float(got.get("sigma_amplitude", float("nan")))
    predicted = float(want.get("amplitude", float("nan")))
    ratio = amplitude / predicted if predicted else float("nan")
    tension = (amplitude - predicted) / sigma if sigma else float("nan")
    payload = {"measured": amplitude, "sigma": sigma, "expected": predicted,
               "ratio": ratio, "tension_sigma": tension}
    ctx.progress(f"ratio {ratio:.2f}× · tension {tension:.1f}σ")
    return ctx.data(payload, label=f"{ratio:.2f}× expected ({tension:.1f}σ)",
                    ratio=ratio, tension_sigma=tension)
