from celestrium import planner


def test_parse_request_extracts_modality_and_target():
    req = planner.parse_request("spectrum of 3C 273")
    assert req.raw == "spectrum of 3C 273"
    assert req.target_text == "3C 273"
    assert req.modality_hint == "spectrum"


def test_parse_request_accepts_coordinates():
    req = planner.parse_request("187.7059 +12.3911")
    assert req.target_text == "187.7059 +12.3911"
    assert req.modality_hint is None
    assert req.coordinates == (187.7059, 12.3911)


def test_classify_object_type_groups_agn_and_blank():
    assert planner.classify_otype("QSO") == "galaxy_agn"
    assert planner.classify_otype("Rad") == "galaxy_agn"
    assert planner.classify_otype("G") == "galaxy_agn"
    assert planner.classify_otype("") == "unknown"
    assert planner.classify_otype(None) == "unknown"


def test_source_capabilities_include_first_wave_sources():
    keys = {src.key for src in planner.SOURCE_CAPABILITIES}
    assert {"ned", "sdss", "mast", "exoplanet-archive", "heasarc", "vizier"} <= keys


def test_recommendations_for_agn_include_spectra_and_high_energy():
    target = planner.ResolvedTarget(
        display_name="3C 273",
        aliases=(),
        ra=187.2779,
        dec=2.0524,
        otype="QSO",
        object_class="galaxy_agn",
        confidence=1.0,
        match_kind="exact",
    )
    plans = planner.recommend_plans(target)
    labels = [p.product.label for p in plans]
    assert any("NED spectra" in label or "SDSS spectra" in label for label in labels)
    assert any("HEASARC" in label for label in labels)
    assert all(p.product.status in {"executable", "metadata", "planned"} for p in plans)
