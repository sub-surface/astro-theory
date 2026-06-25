"""Hermetic tests for the candidate-list service layer (no network, no real disk).

Phase 3's output side: a crossmatch / row selection persisted as a named list with
provenance. Mirrors the cache.py contract, so it's tested the same way — monkeypatch
the storage paths into a tmp dir and exercise save/load/index/latest/drop.
"""
from astropy.table import Table

from celestrium import candidates


def _redirect(tmp_path, monkeypatch):
    monkeypatch.setattr(candidates, "CAND_DIR", tmp_path)
    monkeypatch.setattr(candidates, "INDEX", tmp_path / "index.jsonl")
    monkeypatch.setattr(candidates, "_REPO", tmp_path)


def test_save_load_round_trip(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    tab = Table({"id": [1, 2, 3], "ra": [10.0, 20.0, 30.0], "dec": [-5.0, 0.0, 5.0]})
    path = candidates.save("Euclid AGN", tab, origin="xmatch gaia x nvss", note="r<5\"")
    assert path.exists()

    back = candidates.load("Euclid AGN")
    assert len(back) == 3
    assert list(back["id"]) == [1, 2, 3]
    # name is slugged for the filename but recoverable by name OR slug
    assert candidates.load("euclid-agn") is not None

    rec = candidates.find_record("Euclid AGN")
    assert rec["nrows"] == 3
    assert rec["origin"] == "xmatch gaia x nvss"
    assert rec["columns"] == ["id", "ra", "dec"]


def test_latest_dedupes_by_slug(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    candidates.save("listA", Table({"x": [1]}), origin="v1")
    candidates.save("listA", Table({"x": [1, 2]}), origin="v2")  # overwrite
    candidates.save("listB", Table({"x": [9]}))

    latest = candidates.latest()
    assert len(latest) == 2  # one row per slug, not three index lines
    a = next(r for r in latest if r["slug"] == "lista")
    assert a["nrows"] == 2 and a["origin"] == "v2"


def test_drop_removes_table_keeps_history(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    candidates.save("gone", Table({"x": [1]}))
    assert candidates.drop("gone") is True
    assert candidates.drop("gone") is False  # already gone
    # index history is append-only and preserved
    assert any(r["slug"] == "gone" for r in candidates.index())


def test_load_missing_raises(tmp_path, monkeypatch):
    _redirect(tmp_path, monkeypatch)
    try:
        candidates.load("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass
