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
