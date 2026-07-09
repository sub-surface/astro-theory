from astropy.table import Table

from celestrium import planner, products, spectra


def _target(object_class: str = "galaxy_agn") -> planner.ResolvedTarget:
    return planner.ResolvedTarget(
        display_name="3C 273",
        aliases=(),
        ra=187.2779,
        dec=2.0524,
        otype="QSO" if object_class == "galaxy_agn" else "*",
        object_class=object_class,
        confidence=1.0,
        match_kind="exact",
    )


def _plan(target: planner.ResolvedTarget, key: str) -> planner.ObservationPlan:
    return next(p for p in planner.recommend_plans(target)
                if p.product.key == key)


def test_all_executable_target_products_have_executor():
    missing = {
        cap.key for cap in planner.SOURCE_CAPABILITIES
        if cap.status == "executable" and cap.scope != "global"
    } - products.executable_keys()

    assert missing == set()


def test_colour_image_executor_routes_through_color_auto(monkeypatch, tmp_path):
    target = _target()
    plan = _plan(target, "colour-image")
    path = tmp_path / "color.jpg"
    path.write_text("img", encoding="utf-8")
    seen = {}

    def fake_color_auto(ra, dec, fov_arcmin, pix, survey):
        seen.update(ra=ra, dec=dec, fov=fov_arcmin, pix=pix, survey=survey)
        return path, "Legacy Surveys DR10 (deep)"

    monkeypatch.setattr(products.cutouts, "color_auto", fake_color_auto)

    result = products.execute_product(
        target, plan, {"fov": 5.0, "pix": 256, "survey": "auto"},
        emit=lambda msg: None)

    assert result.kind == "image"
    assert result.path == path
    assert result.label == "Legacy Surveys DR10 (deep)"
    assert seen == {"ra": target.ra, "dec": target.dec,
                    "fov": 5.0, "pix": 256, "survey": None}


def test_ned_spectrum_executor_routes_through_spectra_backend(monkeypatch, tmp_path):
    target = _target()
    plan = _plan(target, "ned")
    path = tmp_path / "spec.png"
    path.write_text("fake", encoding="utf-8")
    seen = {}

    def fake_fetch(name):
        seen["name"] = name
        return spectra.SpectrumResult(
            "spectrum", path, "NED", ("wave", "flux"), {}, "fake spectrum")

    monkeypatch.setattr(products.spectra, "fetch_ned_spectrum", fake_fetch)

    result = products.execute_product(target, plan, {}, emit=lambda msg: None)

    assert seen["name"] == "3C 273"
    assert result.kind == "spectrum"
    assert result.path == path
    assert result.source == "NED"


def test_ned_spectrum_executor_returns_none_when_no_records(monkeypatch):
    target = _target()
    plan = _plan(target, "ned")
    monkeypatch.setattr(products.spectra, "fetch_ned_spectrum", lambda name: None)

    assert products.execute_product(target, plan, {}, emit=lambda msg: None) is None


def test_sdss_executor_prefers_rendered_spectrum(monkeypatch, tmp_path):
    target = _target()
    plan = _plan(target, "sdss")
    path = tmp_path / "sdss.png"
    path.write_text("fake", encoding="utf-8")

    monkeypatch.setattr(
        products.sdss,
        "fetch_spectrum",
        lambda *args, **kwargs: spectra.SpectrumResult(
            "spectrum", path, "SDSS", ("loglam", "flux"), {}, "SDSS spectrum"),
    )

    result = products.execute_product(target, plan, {}, emit=lambda msg: None)

    assert result.kind == "spectrum"
    assert result.source == "SDSS"
    assert result.path == path


def test_lightcurve_executor_uses_cache(monkeypatch):
    target = _target("star")
    plan = _plan(target, "lightcurve")
    seen = {}

    monkeypatch.setattr(
        products.mast,
        "fetch_lightcurves",
        lambda name: Table({"obs_collection": ["TESS"], "target": [name]}),
    )

    def fake_cached_query(archive, query, fetch, refresh=False):
        seen["archive"] = archive
        seen["query"] = query
        seen["refresh"] = refresh
        return fetch()

    monkeypatch.setattr(products.cache, "cached_query", fake_cached_query)

    result = products.execute_product(target, plan, {}, emit=lambda msg: None)

    assert result.kind == "table"
    assert result.rows == 1
    assert seen["archive"] == "product-lightcurve"
    assert "fetch_lightcurves" in seen["query"]
    assert seen["refresh"] is False


def test_vizier_executor_uses_product_cache(monkeypatch):
    target = _target()
    plan = _plan(target, "vizier")
    seen = {}

    monkeypatch.setattr(
        products.importlib.import_module("celestrium.vizier"),
        "cone_pull",
        lambda ra, dec: Table({"ra": [ra], "dec": [dec]}),
    )

    def fake_cached_query(archive, query, fetch, refresh=False):
        seen["archive"] = archive
        seen["query"] = query
        seen["refresh"] = refresh
        return fetch()

    monkeypatch.setattr(products.cache, "cached_query", fake_cached_query)

    result = products.execute_product(target, plan, {}, emit=lambda msg: None)

    assert result.rows == 1
    assert seen["archive"] == "product-vizier"
    assert "3C 273" in seen["query"]
    assert seen["refresh"] is False
