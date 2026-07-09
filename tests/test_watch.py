"""Wave 4 wiring: transient capability status, the transient-alerts product
executor, and packets.build_watch_packet — all hermetic (fake fetch/execute).
"""
import pytest
from astropy.table import Table

from celestrium import candidates, packets, planner, products


# --------------------------------------------------------------------------- #
# planner capabilities
# --------------------------------------------------------------------------- #
def test_transient_global_feed_is_now_executable():
    by_key = {c.key: c for c in planner.SOURCE_CAPABILITIES}
    assert by_key["transient"].status == "executable"
    assert by_key["transient"].scope == "global"


def test_transient_alerts_target_capability_matches_any_object_class():
    by_key = {c.key: c for c in planner.SOURCE_CAPABILITIES}
    cap = by_key["transient-alerts"]
    assert cap.status == "executable"
    assert cap.scope == "target"
    assert cap.object_classes == ("*",)


def test_transient_alerts_appears_in_recommendations_for_any_target():
    target = planner.ResolvedTarget(
        display_name="3C 273", aliases=(), ra=187.2779, dec=2.0524, otype="QSO",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    keys = {p.product.key for p in planner.recommend_plans(target)}
    assert "transient-alerts" in keys


# --------------------------------------------------------------------------- #
# products.py executor
# --------------------------------------------------------------------------- #
def test_transient_alerts_executor_registered():
    executor = products.executor_for("transient-alerts")
    assert executor is not None


def test_transient_alerts_executor_calls_fetch_transients_near(monkeypatch):
    from celestrium import transients
    seen = {}
    monkeypatch.setattr(transients, "fetch_transients_near",
                        lambda ra, dec, radius, days: seen.update(
                            ra=ra, dec=dec, radius=radius, days=days) or
                            Table({"main_id": ["ZTF1"], "ra": [ra], "dec": [dec]}))

    def fake_cached_query(archive, query, fetch, refresh=False):
        return fetch()  # bypass the on-disk cache entirely (test isolation)

    monkeypatch.setattr(products.cache, "cached_query", fake_cached_query)

    target = planner.ResolvedTarget(
        display_name="M87", aliases=(), ra=187.7059, dec=12.3911, otype="G",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    cap = next(c for c in planner.SOURCE_CAPABILITIES if c.key == "transient-alerts")
    plan = planner.ObservationPlan(target=target, product=cap,
                                   parameters={}, recommendation="test")
    result = products.execute_product(target, plan,
                                      {"transient_radius_arcmin": 5.0, "transient_days": 3.0})
    assert seen == {"ra": 187.7059, "dec": 12.3911, "radius": 5.0, "days": 3.0}
    assert result.kind == "table" and result.rows == 1


# --------------------------------------------------------------------------- #
# packets.build_watch_packet
# --------------------------------------------------------------------------- #
def _fake_fetch_hit(ra, dec, radius_arcmin=2.0, days=7.0):
    return Table({"main_id": ["ZTF1"], "ra": [ra], "dec": [dec], "class": ["SNIa"]})


def _fake_fetch_empty(ra, dec, radius_arcmin=2.0, days=7.0):
    return None


def test_watch_packet_for_a_resolved_target():
    target = planner.ResolvedTarget(
        display_name="M87", aliases=(), ra=187.7059, dec=12.3911, otype="G",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    pkt = packets.build_watch_packet(target=target, fetch=_fake_fetch_hit)
    assert pkt.kind == "target" and pkt.label == "M87"
    assert pkt.total_alerts == 1
    assert pkt.entries[0].status == "hit"
    assert "Watch: M87" in pkt.to_markdown()


def test_watch_packet_resolves_raw_target_text(monkeypatch):
    target = planner.ResolvedTarget(
        display_name="3C 273", aliases=(), ra=187.2779, dec=2.0524, otype="QSO",
        object_class="galaxy_agn", confidence=1.0, match_kind="exact")
    monkeypatch.setattr(planner, "resolve_target", lambda text: target)
    pkt = packets.build_watch_packet(target="3C 273", fetch=_fake_fetch_hit)
    assert pkt.label == "3C 273"


def test_watch_packet_unresolvable_target_raises(monkeypatch):
    monkeypatch.setattr(planner, "resolve_target", lambda text: None)
    with pytest.raises(RuntimeError, match="could not resolve"):
        packets.build_watch_packet(target="definitely-not-a-real-object-xyz",
                                   fetch=_fake_fetch_hit)


def test_watch_packet_for_a_bare_field_position():
    pkt = packets.build_watch_packet(ra=10.0, dec=-5.0, fetch=_fake_fetch_empty)
    assert pkt.kind == "field"
    assert pkt.entries[0].status == "empty"
    assert pkt.total_alerts == 0


def test_watch_packet_over_a_candidate_list(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates, "CAND_DIR", tmp_path)
    monkeypatch.setattr(candidates, "INDEX", tmp_path / "index.jsonl")
    monkeypatch.setattr(candidates, "_REPO", tmp_path)
    tab = Table({"main_id": ["A", "B"], "ra": [1.0, 2.0], "dec": [3.0, 4.0]})
    candidates.save("watchlist", tab, origin="test")

    pkt = packets.build_watch_packet(candidate_list="watchlist", fetch=_fake_fetch_hit)
    assert pkt.kind == "list" and pkt.label == "watchlist"
    assert len(pkt.entries) == 2
    assert pkt.entries[0].label == "A" and pkt.entries[1].label == "B"
    assert pkt.total_alerts == 2


def test_watch_packet_candidate_list_row_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates, "CAND_DIR", tmp_path)
    monkeypatch.setattr(candidates, "INDEX", tmp_path / "index.jsonl")
    monkeypatch.setattr(candidates, "_REPO", tmp_path)
    tab = Table({"ra": [float(i) for i in range(5)], "dec": [0.0] * 5})
    candidates.save("many", tab, origin="test")

    pkt = packets.build_watch_packet(candidate_list="many", row_limit=2,
                                     fetch=_fake_fetch_hit)
    assert len(pkt.entries) == 2
    assert pkt.entries[0].label == "row 0"  # no name column -> falls back to index


def test_watch_packet_fetch_error_becomes_an_entry_not_an_exception():
    def boom(ra, dec, radius_arcmin=2.0, days=7.0):
        raise ConnectionError("down")

    pkt = packets.build_watch_packet(ra=1.0, dec=2.0, fetch=boom)
    assert pkt.entries[0].status == "error"
    assert "ConnectionError" in pkt.entries[0].note


def test_watch_packet_needs_exactly_one_source():
    with pytest.raises(ValueError):
        packets.build_watch_packet()
