"""Hermetic tests for unified TAP client and archive fallbacks."""
import pytest
from astropy.table import Table

from celestrium import tap


class DummyTableEntry:
    def __init__(self, name):
        self.name = name


class DummyService:
    def __init__(self, endpoint):
        self.endpoint = endpoint
        self.tables = [
            DummyTableEntry("gaiadr3.gaia_source"),
            DummyTableEntry("allwise_p3as_psd"),
            DummyTableEntry("catalogue.mer_catalogue"),
        ]

    def search(self, adql):
        class DummyResult:
            def to_table(self):
                return Table({"source_id": [12345], "ra": [10.0], "dec": [20.0]})
        return DummyResult()


def test_tap_client_query(monkeypatch):
    monkeypatch.setattr(tap.pyvo.dal, "TAPService", DummyService)
    
    client = tap.TAPClient("http://fake.tap")
    tab = client.query("SELECT * FROM foo")
    assert len(tab) == 1
    assert "source_id" in tab.colnames


def test_tap_client_discover(monkeypatch):
    monkeypatch.setattr(tap.pyvo.dal, "TAPService", DummyService)
    
    client = tap.TAPClient("http://fake.tap")
    tables = client.discover()
    assert len(tables) == 3
    assert "gaiadr3.gaia_source" in tables

    wise = client.discover(substr="wise")
    assert len(wise) == 1
    assert wise[0] == "allwise_p3as_psd"


def test_gaia_fallback_to_tap_client(monkeypatch):
    monkeypatch.setattr(tap.pyvo.dal, "TAPService", DummyService)
    
    def raise_err(*args, **kwargs):
        raise RuntimeError("astroquery connection failed")
    
    import astroquery.gaia
    monkeypatch.setattr(astroquery.gaia.Gaia, "launch_job_async", raise_err)

    tab = tap.query("gaia", "SELECT top 1 source_id FROM gaiadr3.gaia_source")
    assert len(tab) == 1
    assert tab["source_id"][0] == 12345


def test_irsa_fallback_to_tap_client(monkeypatch):
    monkeypatch.setattr(tap.pyvo.dal, "TAPService", DummyService)
    
    def raise_err(*args, **kwargs):
        raise RuntimeError("astroquery irsa failed")
    
    import astroquery.ipac.irsa
    monkeypatch.setattr(astroquery.ipac.irsa.Irsa, "query_tap", raise_err)

    tab = tap.query("irsa", "SELECT top 1 * FROM allwise_p3as_psd")
    assert len(tab) == 1
    assert tab["source_id"][0] == 12345


def test_euclid_fallback_to_tap_client(monkeypatch):
    monkeypatch.setattr(tap.pyvo.dal, "TAPService", DummyService)
    
    def raise_err(*args, **kwargs):
        raise RuntimeError("astroquery euclid failed")
    
    import astroquery.esa.euclid
    monkeypatch.setattr(astroquery.esa.euclid.Euclid, "launch_job", raise_err)

    tab = tap.query("euclid", "SELECT top 1 * FROM catalogue.mer_catalogue")
    assert len(tab) == 1
    assert tab["source_id"][0] == 12345
