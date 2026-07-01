"""Product executor registry shared by presenter surfaces.

The planner ranks products; this module executes a chosen product key. Keeping
archive-specific routing here prevents the TUI from becoming a long list of
service-specific branches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
import importlib

from astropy.table import Table

from . import cache, cutouts, mast, packets, planner, sdss, spectra

Emit = Callable[[str], None]


@dataclass(frozen=True)
class ProductResult:
    kind: str
    label: str
    target_name: str
    table: Optional[Table] = None
    path: Optional[Path] = None
    fov: Optional[float] = None
    source: Optional[str] = None
    summary: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return 0 if self.table is None else len(self.table)


@dataclass(frozen=True)
class ProductExecutor:
    key: str
    run: Callable[[planner.ResolvedTarget, planner.ObservationPlan, dict[str, Any], Emit], Optional[ProductResult]]
    description: str


def _emit(emit: Emit | None, msg: str) -> None:
    if emit:
        emit(msg)


def _fov(target: planner.ResolvedTarget, settings: dict[str, Any],
         point_default: float = 3.0, extended_default: float = 8.0) -> float:
    raw = settings.get("fov", "auto")
    if isinstance(raw, (int, float)):
        return float(raw)
    return packets.default_fov(target.otype or "", extended_default, point_default)


def _pix(settings: dict[str, Any]) -> int:
    try:
        return max(64, min(4096, int(settings.get("pix", 512))))
    except (TypeError, ValueError):
        return 512


def _survey(settings: dict[str, Any]) -> str | None:
    value = str(settings.get("survey", "auto"))
    return value if value in cutouts.COLOR_SURVEYS else None


def product_cache_query(target: planner.ResolvedTarget, plan: planner.ObservationPlan,
                        fetch_func: str, args: tuple[Any, ...]) -> str:
    arg_text = ",".join(str(a) for a in args)
    return (
        f"{plan.product.key}:{fetch_func}|target={target.display_name}|"
        f"ra={target.ra:.8f}|dec={target.dec:.8f}|args={arg_text}"
    )


def _table_result(target: planner.ResolvedTarget, plan: planner.ObservationPlan,
                  module_name: str, fetch_func: str, args: tuple[Any, ...],
                  emit: Emit | None = None) -> ProductResult:
    mod = importlib.import_module(module_name)
    func = getattr(mod, fetch_func)
    archive = f"product-{plan.product.key}"
    query = product_cache_query(target, plan, fetch_func, args)
    _emit(emit, f"table fetch {archive}; executor={module_name}.{fetch_func}; cache=normal")

    def _fetch():
        result = func(*args)
        return result if result is not None else Table()

    tab = cache.cached_query(archive, query, _fetch)
    _emit(emit, f"{plan.product.label}: table returned {len(tab)} rows")
    return ProductResult(
        "table",
        plan.product.label,
        target.display_name,
        table=tab,
        provenance={"archive": archive, "query": query, "executor": f"{module_name}.{fetch_func}"},
    )


def _colour_image(target, plan, settings, emit):
    fov = _fov(target, settings)
    pix = _pix(settings)
    survey = _survey(settings)
    _emit(emit, f"image fetch {target.display_name}; fov={fov}' pix={pix} survey={survey or 'auto'}")
    path, label = cutouts.color_auto(
        target.ra, target.dec, fov_arcmin=fov, pix=pix, survey=survey)
    return ProductResult(
        "image", label, target.display_name, path=Path(path), fov=fov,
        provenance={"executor": "celestrium.cutouts.color_auto", "survey": survey or "auto"},
    )


def _multi_panel(target, plan, settings, emit):
    fov = _fov(target, settings)
    _emit(emit, f"panel fetch {target.display_name}; fov={fov}' datatype=multi-wavelength image")
    path = cutouts.panel(target.ra, target.dec, fov_arcmin=fov)
    return ProductResult(
        "panel", plan.product.label, target.display_name, path=Path(path), fov=fov,
        provenance={"executor": "celestrium.cutouts.panel"},
    )


def _ned_spectrum(target, plan, settings, emit):
    _emit(emit, f"spectrum fetch {target.display_name}; source=NED datatype=1D spectrum")
    result = spectra.fetch_ned_spectrum(target.display_name)
    if result is None:
        return None
    return ProductResult(
        "spectrum", plan.product.label, target.display_name,
        path=Path(result.path), source=result.source, summary=result.summary,
        provenance=result.provenance,
    )


def _sdss_package(target, plan, settings, emit):
    fov = _fov(target, settings, point_default=3.0, extended_default=6.0)
    _emit(emit, f"SDSS search {target.display_name}; spectra -> optical image -> photometry fallback")
    result = sdss.fetch_spectrum(target.display_name, target.ra, target.dec)
    if result is not None:
        return ProductResult(
            "spectrum", plan.product.label, target.display_name,
            path=Path(result.path), source=result.source, summary=result.summary,
            provenance=result.provenance,
        )

    path = sdss.fetch_optical_image(target.ra, target.dec, fov_arcmin=fov)
    if path is not None:
        return ProductResult(
            "image", "SDSS optical image", target.display_name,
            path=Path(path), fov=fov, source="SDSS",
            summary="SDSS optical field image",
            provenance={"executor": "celestrium.sdss.fetch_optical_image"},
        )

    return _table_result(
        target, plan, "celestrium.sdss", "fetch_photometry",
        (target.ra, target.dec), emit)


def _mast_uv_optical(target, plan, settings, emit):
    radius = float(settings.get("mast_radius_deg", 0.02) or 0.02)
    _emit(emit, f"MAST UV/optical observation search; radius={radius} deg")
    return _table_result(
        target, plan, "celestrium.mast", "fetch_uv_optical_observations",
        (target.ra, target.dec, radius), emit)


def _lightcurve(target, plan, settings, emit):
    _emit(emit, "MAST time-series metadata search; missions=TESS,Kepler,K2")
    return _table_result(
        target, plan, "celestrium.mast", "fetch_lightcurves",
        (target.display_name,), emit)


def _vizier(target, plan, settings, emit):
    return _table_result(
        target, plan, "celestrium.vizier", "cone_pull",
        (target.ra, target.dec), emit)


def _heasarc(target, plan, settings, emit):
    return _table_result(
        target, plan, "celestrium.heasarc", "query_region",
        (target.ra, target.dec), emit)


def _exoplanet(target, plan, settings, emit):
    return _table_result(
        target, plan, "celestrium.exoplanet", "query_target",
        (target.display_name,), emit)


def _transit(target, plan, settings, emit):
    return _table_result(
        target, plan, "celestrium.exoplanet", "predict_transits",
        (target.display_name,), emit)


def _ephemeris(target, plan, settings, emit):
    return _table_result(
        target, plan, "celestrium.solarsystem", "fetch_ephemeris",
        (target.display_name,), emit)


EXECUTORS: dict[str, ProductExecutor] = {
    "colour-image": ProductExecutor("colour-image", _colour_image, "HiPS colour cutout"),
    "multi-panel": ProductExecutor("multi-panel", _multi_panel, "SkyView multi-wavelength panel"),
    "ned": ProductExecutor("ned", _ned_spectrum, "NED spectrum renderer"),
    "sdss": ProductExecutor("sdss", _sdss_package, "SDSS spectra, imaging, and photometry"),
    "mast": ProductExecutor("mast", _mast_uv_optical, "MAST UV/optical observation metadata"),
    "lightcurve": ProductExecutor("lightcurve", _lightcurve, "MAST TESS/Kepler/K2 lightcurve metadata"),
    "vizier": ProductExecutor("vizier", _vizier, "VizieR cone pull"),
    "heasarc": ProductExecutor("heasarc", _heasarc, "HEASARC region query"),
    "exoplanet-archive": ProductExecutor("exoplanet-archive", _exoplanet, "NASA Exoplanet Archive query"),
    "transit": ProductExecutor("transit", _transit, "Exoplanet transit prediction"),
    "ephemeris": ProductExecutor("ephemeris", _ephemeris, "JPL Horizons ephemeris"),
}


def executor_for(key: str) -> ProductExecutor | None:
    return EXECUTORS.get(key)


def executable_keys() -> set[str]:
    return set(EXECUTORS)


def execute_product(target: planner.ResolvedTarget, plan: planner.ObservationPlan,
                    settings: Optional[dict[str, Any]] = None,
                    emit: Emit | None = None) -> Optional[ProductResult]:
    executor = executor_for(plan.product.key)
    if executor is None:
        raise NotImplementedError(f"no product executor for {plan.product.key}")
    return executor.run(target, plan, settings or {}, emit or (lambda msg: None))
