from astropy.table import Table

from celestrium import planner


def _simbad_table(rows):
    """Build a SIMBAD-shaped Table (main_id/ra/dec/otype) from row dicts."""
    cols = {k: [r[k] for r in rows] for k in ("main_id", "ra", "dec", "otype")}
    return Table(cols)


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


def test_source_capabilities_do_not_reference_missing_sources_package():
    assert all(
        src.service_module is None
        or not src.service_module.startswith("celestrium.sources.")
        for src in planner.SOURCE_CAPABILITIES
    )


def test_unknown_spectrum_request_gets_inspectable_fallback():
    target = planner.ResolvedTarget(
        display_name="Mystery source",
        aliases=(),
        ra=10.0,
        dec=-5.0,
        otype="",
        object_class="unknown",
        confidence=0.2,
        match_kind="coordinate",
    )
    plans = planner.recommend_plans(target, modality="spectrum")
    assert plans
    assert any(p.product.status in {"metadata", "planned"} for p in plans)
    assert all(p.next_action == "inspect" for p in plans)


def test_observation_plan_to_dict_includes_product_metadata():
    target = planner.ResolvedTarget(
        display_name="3C 273",
        aliases=("PKS 1226+023",),
        ra=187.2779,
        dec=2.0524,
        otype="QSO",
        object_class="galaxy_agn",
        confidence=1.0,
        match_kind="exact",
    )
    plan = planner.recommend_plans(target, modality="spectrum")[0].to_dict()
    assert plan["target"]["display_name"] == "3C 273"
    assert plan["product"]["key"]
    assert plan["product"]["label"]
    assert plan["product"]["status"] in {"executable", "metadata", "planned"}


def test_coverage_note_mentions_forced_survey_and_fallback():
    note = planner.image_coverage_note(dec=41.27, survey="panstarrs")
    assert "Pan-STARRS" in note
    assert "fallback" in note.lower()


def test_coverage_note_unknown_survey_uses_auto_candidates():
    note = planner.image_coverage_note(dec=41.27, survey="not-a-survey")
    lowered = note.lower()
    assert "unknown" in lowered or "unavailable" in lowered
    assert "auto candidates" in lowered
    assert "fallback" in lowered
    assert "selected colour survey: not-a-survey" not in lowered


# --------------------------------------------------------------------------- #
# Layered resolution (hermetic — backends injected)
# --------------------------------------------------------------------------- #
def test_resolve_target_exact_match():
    table = _simbad_table([{"main_id": "M  87", "ra": 187.7, "dec": 12.4, "otype": "G"}])
    tgt = planner.resolve_target("M87", identify=lambda name: table, search=lambda name: None)
    assert tgt is not None
    assert tgt.match_kind == "exact"
    assert tgt.confidence == 1.0
    assert tgt.display_name == "M  87"
    assert tgt.object_class == "galaxy_agn"
    assert tgt.alternatives == ()


def test_resolve_target_ambiguous_uses_relaxed_search():
    hits = _simbad_table([
        {"main_id": "NGC 1", "ra": 1.0, "dec": 2.0, "otype": "G"},
        {"main_id": "NGC 2", "ra": 1.1, "dec": 2.1, "otype": "G"},
    ])
    tgt = planner.resolve_target(
        "NGC", identify=lambda name: None, search=lambda name: hits)
    assert tgt.match_kind == "ambiguous"
    assert tgt.display_name == "NGC 1"
    assert len(tgt.alternatives) == 1
    assert tgt.alternatives[0]["name"] == "NGC 2"


def test_resolve_target_relaxed_single_hit():
    hit = _simbad_table([{"main_id": "Vega", "ra": 279.2, "dec": 38.8, "otype": "*"}])
    tgt = planner.resolve_target(
        "vegaa", identify=lambda name: None, search=lambda name: hit)
    assert tgt.match_kind == "relaxed"
    assert tgt.object_class == "star"


def test_resolve_target_coordinates_attach_nearby_identity():
    near = _simbad_table([{"main_id": "M  87", "ra": 187.70, "dec": 12.39, "otype": "G"}])
    tgt = planner.resolve_target("187.7059 12.3911",
                                 identify=lambda name: None,
                                 nearby=lambda ra, dec: near)
    assert tgt.match_kind == "nearby"
    # keeps the *requested* position, names it from the cone search
    assert tgt.ra == 187.7059
    assert tgt.display_name == "M  87"


def test_resolve_target_blank_field_when_no_nearby():
    tgt = planner.resolve_target("10.0 -5.0",
                                 identify=lambda name: None,
                                 nearby=lambda ra, dec: None)
    assert tgt.match_kind == "blank"
    assert tgt.ra == 10.0 and tgt.dec == -5.0
    assert tgt.object_class == "unknown"


def test_resolve_target_returns_none_when_nothing_matches():
    assert planner.resolve_target(
        "zzz-nope", identify=lambda name: None, search=lambda name: None) is None


def test_resolve_target_survives_backend_errors():
    def boom(name):
        raise RuntimeError("network down")

    assert planner.resolve_target("M87", identify=boom, search=lambda name: None) is None


# --------------------------------------------------------------------------- #
# Executable products + ranking
# --------------------------------------------------------------------------- #
def test_executable_image_products_exist():
    by_key = {c.key: c for c in planner.SOURCE_CAPABILITIES}
    assert by_key["colour-image"].status == "executable"
    assert by_key["multi-panel"].status == "executable"
    # image products apply to any object class (and to blank fields)
    assert by_key["colour-image"].object_classes == ("*",)


def test_recommendations_rank_executable_first():
    target = planner.ResolvedTarget(
        display_name="3C 273", aliases=(), ra=187.2779, dec=2.0524,
        otype="QSO", object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    plans = planner.recommend_plans(target)
    statuses = [p.product.status for p in plans]
    # executable band leads; planned band trails
    assert statuses == sorted(statuses, key=lambda s: {"executable": 0, "metadata": 1, "planned": 2}[s])
    assert plans[0].product.status == "executable"
    assert plans[0].next_action == "fetch"


def test_blank_field_still_gets_image_and_catalogue_actions():
    target = planner.ResolvedTarget(
        display_name="field", aliases=(), ra=10.0, dec=-5.0, otype="",
        object_class="unknown", confidence=0.3, match_kind="blank")
    keys = {p.product.key for p in planner.recommend_plans(target)}
    assert "colour-image" in keys
    assert "multi-panel" in keys
    assert "vizier" in keys
    assert {"neo", "satellite", "transient"}.isdisjoint(keys)


def test_global_feeds_are_not_target_products_by_default():
    target = planner.ResolvedTarget(
        display_name="field", aliases=(), ra=10.0, dec=-5.0, otype="",
        object_class="unknown", confidence=0.3, match_kind="blank")
    plans = planner.recommend_plans(target)
    assert all(p.product.scope != "global" for p in plans)
