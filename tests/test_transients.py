"""transients.py — real ALeRCE-backed implementation (hermetic: canned JSON).

The live endpoint/schema was verified against
https://api.alerce.online/ztf/v1/swagger.json during implementation (see
transients.py's module docstring); these tests pin the shape with a
fake fetcher so they never touch the network.
"""
from celestrium import transients

_ITEM = {
    "oid": "ZTF25aaaa", "meanra": 187.7059, "meandec": 12.3911,
    "class": "SNIa", "classifier": "lc_classifier", "probability": 0.87,
    "ndet": 12, "firstmjd": 60800.1, "lastmjd": 60801.3,
}


def test_fetch_latest_transients_maps_alerce_fields():
    def fake_fetcher(params):
        assert "firstmjd" in params and "page_size" in params
        return {"items": [_ITEM]}

    tab = transients.fetch_latest_transients(days=2.0, fetcher=fake_fetcher)
    assert tab is not None and len(tab) == 1
    assert tab["main_id"][0] == "ZTF25aaaa"
    assert tab["ra"][0] == 187.7059
    assert tab["dec"][0] == 12.3911
    assert tab["class"][0] == "SNIa"


def test_fetch_latest_transients_empty_is_none():
    tab = transients.fetch_latest_transients(fetcher=lambda params: {"items": []})
    assert tab is None


def test_fetch_transients_near_passes_conesearch_params():
    seen = {}

    def fake_fetcher(params):
        seen.update(params)
        return {"items": [_ITEM]}

    tab = transients.fetch_transients_near(187.7059, 12.3911, radius_arcmin=2.0,
                                           fetcher=fake_fetcher)
    assert seen["ra"] == 187.7059 and seen["dec"] == 12.3911
    assert seen["radius"] == 120.0  # arcmin -> arcsec
    assert len(tab) == 1


def test_fetch_transients_near_without_days_omits_date_filter():
    seen = {}
    transients.fetch_transients_near(10.0, -5.0, fetcher=lambda p: seen.update(p) or {"items": []})
    assert "firstmjd" not in seen


def test_fetch_transients_near_with_days_adds_date_range():
    seen = {}
    transients.fetch_transients_near(10.0, -5.0, days=7.0,
                                     fetcher=lambda p: seen.update(p) or {"items": []})
    assert "firstmjd" in seen and len(seen["firstmjd"]) == 2


def test_missing_optional_fields_become_nan_or_empty_not_a_crash():
    sparse = {"oid": "ZTF25zzzz", "meanra": 1.0, "meandec": 2.0}  # no class/probability/etc.
    tab = transients.fetch_latest_transients(fetcher=lambda p: {"items": [sparse]})
    assert tab is not None
    assert tab["class"][0] == ""
    assert tab["probability"][0] != tab["probability"][0]  # NaN


def test_network_failure_raises_runtime_error_not_bare_exception():
    def boom(params):
        raise ConnectionError("no route")

    import pytest
    with pytest.raises(RuntimeError, match="ALeRCE fetch failed"):
        transients.fetch_latest_transients(fetcher=boom)
    with pytest.raises(RuntimeError, match="ALeRCE cone search failed"):
        transients.fetch_transients_near(1.0, 2.0, fetcher=boom)
