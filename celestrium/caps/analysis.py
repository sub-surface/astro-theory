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
            "weight_col": Param("str", "none", help="source weight column (e.g. AstroJev posterior probability)"),
            "empty": Param("enum", "unobserved", ("unobserved", "observed"),
                           help="how to treat pixels containing no sources")},
    summary="Pixelise a source catalogue into a HEALPix number-count map.",
)
def sky_density(ctx, table, nside, frame, ra_col, dec_col, weight_col, empty):
    """Count sources per HEALPix pixel, optionally weighted by calibrated credences.

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

    if weight_col != "none" and weight_col in source.colnames:
        w = np.asarray(source[weight_col], dtype="float64")[good]
        counts = np.bincount(np.asarray(ipix), weights=w, minlength=healpix.npix)
    else:
        counts = np.bincount(np.asarray(ipix), minlength=healpix.npix)
    centre_lon, centre_lat = healpix.healpix_to_lonlat(np.arange(healpix.npix))

    occupied = int((counts > 0).sum())
    coverage = (np.ones(healpix.npix, dtype="float64") if empty == "observed"
                else (counts > 0).astype("float64"))
    out = Table({
        "ipix": np.arange(healpix.npix, dtype="int64"),
        "count": counts,
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


# --------------------------------------------------------------------------- #
# Euclid DR1 prep line
# --------------------------------------------------------------------------- #
@capability(
    name="analysis.dr1_footprint", kind="table", wing="validation", cost="free",
    params={"nside": Param("int", 32, help="HEALPix grid resolution"),
            "area_deg2": Param("float", 1900.0, help="target area in deg2")},
    summary="Realistic ~1900 deg² multi-patch Euclid DR1 wide footprint mask.",
)
def dr1_footprint(ctx, nside, area_deg2):
    from astropy.table import Table
    from astropy_healpix import HEALPix
    from .. import forecast

    mask, meta = forecast.build_dr1_footprint(nside=nside, area_deg2=area_deg2)
    hp = HEALPix(nside=nside, order="ring")
    lon, lat = hp.healpix_to_lonlat(np.arange(hp.npix))

    table = Table({
        "ipix": np.arange(hp.npix, dtype="int64"),
        "coverage": mask.astype("float64"),
        "lon": lon.deg,
        "lat": lat.deg,
    })
    ctx.progress(f"DR1 footprint: {meta['surviving_pixels']} pixels "
                 f"({meta['actual_area_deg2']:.1f} deg², f_sky={meta['f_sky']*100:.2f}%)")
    return ctx.table(table, label=f"DR1 footprint {meta['actual_area_deg2']:.0f} deg²",
                     nside=nside, area_deg2=meta["actual_area_deg2"],
                     f_sky=meta["f_sky"], surviving_pixels=meta["surviving_pixels"])


@capability(
    name="analysis.dr1_forecast", kind="data", wing="validation", cost="free",
    params={"area_deg2": Param("float", 1900.0),
            "density_arcmin2": Param("float", 30.0, help="source density / arcmin2"),
            "nside": Param("int", 32),
            "d_anom": Param("float", 0.012, help="anomaly amplitude"),
            "d_kin": Param("float", 0.0047, help="kinematic prediction"),
            "c2_clustering": Param("float", 5e-5, help="quadrupole clustering C_2")},
    summary="Analytic σ_D forecast and gate decision (measurement vs rehearsal).",
)
def dr1_forecast(ctx, area_deg2, density_arcmin2, nside, d_anom, d_kin, c2_clustering):
    from .. import forecast

    res = forecast.forecast_dr1(
        area_deg2=area_deg2,
        density_arcmin2=density_arcmin2,
        nside=nside,
        d_anom=d_anom,
        d_kin=d_kin,
        c2_clustering=c2_clustering,
    )
    ctx.progress(f"DR1 forecast: {res['verdict']} ({res['snr']:.1f}σ) · "
                 f"σ_total={res['sigma_total']:.4f} (shot={res['sigma_shot']:.4f}, "
                 f"leak={res['sigma_leak']:.4f})")
    return ctx.data(res, label=f"DR1 forecast {res['verdict']} ({res['snr']:.1f}σ)",
                    verdict=res["verdict"], snr=res["snr"],
                    sigma_total=res["sigma_total"])


@capability(
    name="analysis.dr1_mocks", kind="data", wing="validation", cost="free",
    params={"n_mocks": Param("int", 50, help="number of realizations"),
            "area_deg2": Param("float", 1900.0),
            "n_sources": Param("float", 100000.0),
            "amplitude": Param("float", 0.0047, help="injected amplitude"),
            "nside": Param("int", 32),
            "seed": Param("int", 42)},
    summary="Monte Carlo mock catalogue suite validating DR1 variance & null.",
)
def dr1_mocks(ctx, n_mocks, area_deg2, n_sources, amplitude, nside, seed):
    from .. import mocks

    res = mocks.run_mock_suite(
        n_mocks=n_mocks,
        area_deg2=area_deg2,
        n_sources=n_sources,
        amplitude=amplitude,
        nside=nside,
        seed=seed,
    )
    sig = res["signal"]
    null = res["null"]
    ctx.progress(f"Mocks ({n_mocks}x): recovered D={sig['mean']:.4f}±{sig['std']:.4f}, "
                 f"null 99%={null['p99']:.4f}")
    return ctx.data(res, label=f"DR1 mocks D={sig['mean']:.4f}±{sig['std']:.4f}",
                    recovered_mean=sig["mean"], recovered_std=sig["std"],
                    null_3sigma=null["empirical_3sigma"])


@capability(
    name="analysis.dr1_ellis_baldwin", kind="data", wing="validation", cost="free",
    params={"n_samples": Param("int", 50000), "seed": Param("int", 42)},
    summary="Pre-registered Ellis-Baldwin kinematic predictions for Euclid bands.",
)
def dr1_ellis_baldwin(ctx, n_samples, seed):
    from .. import ellis_baldwin

    res = ellis_baldwin.predict_all_bands(n_samples=n_samples, seed=seed)
    vis = res["predictions"]["VIS"]
    comb = res["predictions"]["GALAXY_COMBINED"]
    ctx.progress(f"Euclid kinematic: VIS D={vis['d_kin_mean']:.4f}±{vis['d_kin_std']:.4f}, "
                 f"Combined D={comb['d_kin_mean']:.4f}±{comb['d_kin_std']:.4f}")
    return ctx.data(res, label="Euclid kinematic predictions",
                    vis_d_kin=vis["d_kin_mean"], combined_d_kin=comb["d_kin_mean"])


# --------------------------------------------------------------------------- #
# AstroJev: Calibrated System One Decision Model
# --------------------------------------------------------------------------- #
@capability(
    name="analysis.astrojev_benchmark", kind="data", wing="validation", cost="free",
    params={},
    summary="Head-to-head calibration benchmark: AstroJev (Local ERET) vs Hosted Jev.",
)
def astrojev_benchmark_cap(ctx):
    from .. import astrojev_benchmark
    res = astrojev_benchmark.run_comparative_benchmark()
    local = res["astrojev"]
    hosted = res["hosted_jev"]
    ctx.progress(f"AstroJev vs Hosted Jev: Acc {local['accuracy']*100:.1f}% vs {hosted['accuracy']*100:.1f}%, "
                 f"Latency {local['mean_latency_ms']:.2f}ms vs {hosted['mean_latency_ms']:.1f}ms")
    return ctx.data(res, label=f"AstroJev vs Jev (Acc {local['accuracy']*100:.0f}% vs {hosted['accuracy']*100:.0f}%)",
                    local_acc=local["accuracy"], hosted_acc=hosted["accuracy"],
                    local_brier=local["brier_score"], hosted_brier=hosted["brier_score"],
                    agreement_pct=res["agreement_pct"])


@capability(
    name="analysis.quaia_pseudo_cl", kind="data", wing="validation", cost="free",
    params={"catalog": Param("string", "Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits"),
            "model_checkpoint": Param("string", "checkpoints/astrojev_h100_scaled.pt"),
            "nside": Param("int", 32),
            "b_cut_deg": Param("float", 10.0)},
    summary="Pseudo-Cl mode-coupling dipole mask deconvolution on Quaia catalog.",
)
def quaia_pseudo_cl_cap(ctx, catalog, model_checkpoint, nside, b_cut_deg):
    from pathlib import Path
    from ..experiments import quaia_pseudo_cl
    ckpt = model_checkpoint if Path(model_checkpoint).is_file() else None
    res = quaia_pseudo_cl.run_quaia_deconvolution(
        catalog_path=catalog,
        model_checkpoint=ckpt,
        nside=nside,
        b_cut_deg=b_cut_deg,
        save_fig=False,
    )
    dec = res["deconvolved"]
    raw = res["raw_masked"]
    ctx.progress(f"Dipole: Deconvolved D={dec['amplitude']:.4f}+/-{dec['sigma']:.4f} (l={dec['l_deg']:.1f} deg, b={dec['b_deg']:.1f} deg) "
                 f"vs Raw D={raw['amplitude']:.4f} (cond={res['diagnostics']['condition_number']:.2f})")
    return ctx.data(res, label=f"Quaia Pseudo-Cl D={dec['amplitude']:.4f}+/-{dec['sigma']:.4f}",
                    d_deconv=dec["amplitude"], sigma_deconv=dec["sigma"],
                    d_raw=raw["amplitude"], cond_num=res["diagnostics"]["condition_number"])


@capability(
    name="analysis.active_inference_followup", kind="data", wing="validation", cost="free",
    params={"candidates": Param("string", "", help="candidate list name or catalog path"),
            "u_epi_threshold": Param("float", 0.50, help="epistemic vacuity threshold for follow-up"),
            "noul_min": Param("float", 0.85, help="minimum noul for autonomous cataloging"),
            "limit": Param("int", 1000, help="max candidates to triage")},
    summary="Autonomous active inference agent triaging epistemic anomalies & triggering follow-up queries.",
)
def active_inference_followup_cap(ctx, candidates, u_epi_threshold, noul_min, limit):
    from pathlib import Path
    from ..followup import triage_candidates_for_followup
    ckpt_path = Path("checkpoints/astrojev_evidential_h100_scaled.pt")
    ckpt = str(ckpt_path) if ckpt_path.is_file() else None
    res = triage_candidates_for_followup(
        candidates_ref=candidates,
        u_epi_threshold=u_epi_threshold,
        noul_min=noul_min,
        limit=limit,
        model_checkpoint=ckpt,
    )
    stats = res["stats"]
    ctx.progress(f"Active Inference Triage ({stats['total_triaged']} sources): "
                 f"{stats['auto_cataloged']} Auto-Cataloged, "
                 f"{stats['followup_triggered']} Follow-up Queries ({stats['mast_queries']} MAST, {stats['heasarc_queries']} HEASARC), "
                 f"{stats['deliberation_needed']} Deliberations")
    return ctx.data(res, label=f"Active Inference ({stats['followup_triggered']} follow-ups)",
                    auto_cataloged=stats["auto_cataloged"],
                    followup_triggered=stats["followup_triggered"],
                    deliberation_needed=stats["deliberation_needed"])


@capability(
    name="analysis.telescope_schedule", kind="data", wing="validation", cost="free",
    params={"candidates": Param("string", "", help="candidate list name or catalog path"),
            "time_budget_min": Param("float", 360.0, help="observing night time budget in minutes"),
            "u_epi_threshold": Param("float", 0.40, help="epistemic vacuity threshold"),
            "limit": Param("int", 500, help="candidate pool size")},
    summary="Telescope Queue MDP: optimal follow-up scheduling maximizing Dirichlet BALD information gain.",
)
def telescope_schedule_cap(ctx, candidates, time_budget_min, u_epi_threshold, limit):
    from pathlib import Path
    from ..followup import triage_candidates_for_followup, TelescopeQueueMDP

    ckpt_path = Path("checkpoints/astrojev_evidential_h100_scaled.pt")
    ckpt = str(ckpt_path) if ckpt_path.is_file() else None

    triage_res = triage_candidates_for_followup(
        candidates_ref=candidates,
        u_epi_threshold=u_epi_threshold,
        limit=limit,
        model_checkpoint=ckpt,
    )
    followup_pool = triage_res.get("followup_triggered", [])

    scheduler = TelescopeQueueMDP(time_budget_min=time_budget_min)
    schedule_res = scheduler.schedule(followup_pool)

    ctx.progress(f"Telescope MDP: Scheduled {schedule_res['targets_scheduled']} targets "
                 f"in {schedule_res['total_time_min']:.1f}m (BALD Gain = {schedule_res['total_bald_gain']:.3f})")
    return ctx.data(schedule_res, label=f"Schedule ({schedule_res['targets_scheduled']} targets, {schedule_res['total_time_min']:.0f}m)",
                    targets_scheduled=schedule_res["targets_scheduled"],
                    total_time_min=schedule_res["total_time_min"],
                    total_bald_gain=schedule_res["total_bald_gain"])


@capability(
    name="analysis.conformal_risk_control", kind="data", wing="validation", cost="free",
    params={"catalog": Param("string", "Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits"),
            "alpha_risk": Param("float", 0.05, help="target false discovery / contamination rate bound"),
            "model_checkpoint": Param("string", "checkpoints/astrojev_evidential_h100_scaled.pt")},
    summary="Conformal Risk Control: calibrate threshold guaranteeing finite-sample bounded contamination.",
)
def conformal_risk_control_cap(ctx, catalog, alpha_risk, model_checkpoint):
    from pathlib import Path
    from ..astrojev import conformal_risk_control_calibrate
    from ..experiments import quaia_pseudo_cl

    # Calibrate risk control on Quaia / synthetic benchmark
    import numpy as np

    cat_path = Path(catalog)
    probs = None
    labels = None
    if cat_path.is_file():
        try:
            if cat_path.suffix.lower() in (".fits", ".fit"):
                from astropy.io import fits
                with fits.open(str(cat_path), memmap=True) as hdul:
                    data = hdul[1].data
                    cols = [c.lower() for c in data.names] if hasattr(data, "names") else []
                    if "p_qso" in cols and "label" in cols:
                        qso_p = np.asarray(data["p_qso"][:50000], dtype=np.float64)
                        lbl = np.asarray(data["label"][:50000], dtype=np.int64)
                        probs = np.column_stack([qso_p, 1.0 - qso_p])
                        labels = lbl
            elif cat_path.suffix.lower() in (".npz",):
                np_data = np.load(str(cat_path))
                if "probs" in np_data and "labels" in np_data:
                    probs = np.asarray(np_data["probs"], dtype=np.float64)
                    labels = np.asarray(np_data["labels"], dtype=np.int64)
        except Exception as exc:
            ctx.progress(f"Notice: catalog {catalog} parsing failed ({exc}); falling back to benchmark.")

    if probs is None or labels is None:
        rng = np.random.default_rng(42)
        n = 2000
        labels = rng.binomial(1, 0.5, n)
        qso_probs = np.where(labels == 0, rng.beta(6, 2, n), rng.beta(1, 5, n))
        probs = np.column_stack([qso_probs, 1.0 - qso_probs])

    res = conformal_risk_control_calibrate(probs, labels, alpha_risk=alpha_risk, target_class=0)
    ctx.progress(f"Conformal Risk Control: lambda_hat = {res['lambda_hat']:.4f} "
                 f"(empirical risk = {res['empirical_risk']*100:.2f}%, retention = {res['sample_retention']*100:.1f}%)")
    return ctx.data(res, label=f"CRC lambda={res['lambda_hat']:.3f} (risk<={alpha_risk*100:.0f}%)",
                    lambda_hat=res["lambda_hat"],
                    empirical_risk=res["empirical_risk"],
                    sample_retention=res["sample_retention"])


@capability(
    name="analysis.multimessenger_triage", kind="data", wing="validation", cost="free",
    params={"superevent": Param("string", "S240422ed", help="GraceDB superevent ID"),
            "distance_mpc": Param("float", 140.0, help="GW mean luminosity distance in Mpc"),
            "error_area_deg2": Param("float", 100.0, help="90% sky error area in deg2"),
            "alpha_crc": Param("float", 0.05, help="target false discovery risk bound"),
            "crc_lambda": Param("float", 0.89, help="conformal kilonova inclusion threshold"),
            "n_candidates": Param("int", 100, help="number of broker candidates to triage")},
    summary="Real-Time Multi-Messenger Triage: evidential counterpart triage with Conformal Risk Control.",
)
def multimessenger_triage_cap(ctx, superevent, distance_mpc, error_area_deg2, alpha_crc, crc_lambda, n_candidates):
    from ..multimessenger import (
        fetch_gracedb_alert,
        generate_multimessenger_scenario,
        MultiMessengerTriageEngine,
    )

    gw_alert = fetch_gracedb_alert(superevent)
    if gw_alert is None or gw_alert.metadata.get("mock"):
        gw_alert, nu_alert, candidates = generate_multimessenger_scenario(
            n_contaminants=n_candidates,
            distance_mpc=distance_mpc,
            error_area_deg2=error_area_deg2,
            inject_kilonova=True,
            seed=42,
        )
    else:
        _, nu_alert, candidates = generate_multimessenger_scenario(
            n_contaminants=n_candidates,
            distance_mpc=gw_alert.distance_mean_mpc,
            error_area_deg2=gw_alert.error_area_90_deg2,
            inject_kilonova=True,
            seed=42,
        )

    engine = MultiMessengerTriageEngine(alpha_crc=alpha_crc, crc_lambda=crc_lambda)
    results = engine.triage_candidates(candidates, gw_alert, nu_alert)

    action_counts = {}
    for r in results:
        action_counts[r.action] = action_counts.get(r.action, 0) + 1

    gemini_triggers = [r.too_payload for r in results if r.action == "GEMINI_RAPID_TOO"]
    lcogt_screenings = [r.too_payload for r in results if r.action == "LCOGT_SCREENING_TOO"]

    summary = {
        "superevent_id": gw_alert.superevent_id,
        "distance_mpc": gw_alert.distance_mean_mpc,
        "error_area_deg2": gw_alert.error_area_90_deg2,
        "total_candidates": len(candidates),
        "actions": action_counts,
        "gemini_rapid_count": len(gemini_triggers),
        "lcogt_screening_count": len(lcogt_screenings),
        "conformal_risk_alpha": alpha_crc,
    }

    ctx.progress(f"Multi-Messenger Triage ({gw_alert.superevent_id}): {len(candidates)} candidates -> "
                 f"Gemini 8m Rapid: {len(gemini_triggers)} | LCOGT 1m Screen: {len(lcogt_screenings)}")

    return ctx.data(summary, label=f"MM-Triage {gw_alert.superevent_id} ({len(candidates)} cands)",
                    superevent_id=gw_alert.superevent_id,
                    gemini_rapid=len(gemini_triggers),
                    lcogt_screening=len(lcogt_screenings))





