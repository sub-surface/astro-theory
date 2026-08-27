"""Hermetic tests for object-scoped capabilities (dossier, field, sweep, watch)."""
from pathlib import Path
import pytest
from astropy.table import Table

from celestrium.core.kernel import Kernel
from celestrium.core.ledger import Ledger
from celestrium import resolvers
import celestrium.caps  # noqa: F401


def _fake_resolve(target):
    return resolvers.ResolvedTarget(
        display_name="M87",
        aliases=("Messier 87",),
        ra=187.7059,
        dec=12.3911,
        otype="G",
        object_class="galaxy_agn",
        confidence=1.0,
        match_kind="exact",
    )


def test_object_dossier_generates_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(resolvers, "resolve_target", _fake_resolve)
    
    k = Kernel(Ledger(tmp_path / "test.db"))
    art = k.run("object.dossier", {"target": "M87", "rows": 3, "images": False})
    
    assert art.kind == "document"
    doc_text = k.load(art.id)
    assert "# Dossier: M87" in doc_text
    assert "RA = 187.70590°" in doc_text
    assert "(mocked)" not in doc_text


def test_object_field_generates_markdown(monkeypatch, tmp_path):
    monkeypatch.setattr(resolvers, "resolve_target", _fake_resolve)
    
    k = Kernel(Ledger(tmp_path / "test.db"))
    art = k.run("object.field", {"target": "187.7059 12.3911", "fov": 5.0, "images": False})
    
    assert art.kind == "document"
    doc_text = k.load(art.id)
    assert "# Field Packet: 187.70590°, +12.39110°" in doc_text
    assert "(mocked)" not in doc_text


def test_object_sweep_produces_table(monkeypatch, tmp_path):
    monkeypatch.setattr(resolvers, "resolve_target", _fake_resolve)
    
    k = Kernel(Ledger(tmp_path / "test.db"))
    art = k.run("object.sweep", {"target": "M87"})
    
    assert art.kind == "table"
    tab = k.load(art.id)
    assert len(tab) == 3
    assert "product" in tab.colnames
    assert "status" in tab.colnames


def test_object_watch_produces_table(monkeypatch, tmp_path):
    monkeypatch.setattr(resolvers, "resolve_target", _fake_resolve)
    
    from celestrium import transients
    monkeypatch.setattr(transients, "fetch_transients_near",
                        lambda ra, dec, radius_arcmin, days: Table({
                            "main_id": ["ZTF20abc"], "ra": [187.70], "dec": [12.39],
                            "class": ["SN Ia"], "probability": [0.95]
                        }))
    
    k = Kernel(Ledger(tmp_path / "test.db"))
    art = k.run("object.watch", {"target": "M87", "radius_arcmin": 2.0, "days": 7.0})
    
    assert art.kind == "table"
    tab = k.load(art.id)
    assert len(tab) == 1
    assert tab["main_id"][0] == "ZTF20abc"
