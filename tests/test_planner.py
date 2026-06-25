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
