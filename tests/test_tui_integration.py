"""Hermetic integration tests for TUI resolution and plan execution."""
from pathlib import Path
import pytest
from astropy.table import Table

from celestrium import resolvers, config
from celestrium.core.artifact import Artifact
from celestrium.core.kernel import Kernel


def _fake_target(name="M87"):
    return resolvers.ResolvedTarget(
        display_name=name,
        aliases=("Messier 87", "NGC 4486"),
        ra=187.7059,
        dec=12.3911,
        otype="G",
        object_class="galaxy_agn",
        confidence=1.0,
        match_kind="exact",
    )


def test_recommend_plans_ranks_capabilities_for_target():
    tgt = _fake_target("M87")
    plans = resolvers.recommend_plans(tgt)
    assert len(plans) > 0
    keys = [p.product.key for p in plans]
    assert "object.spectrum" in keys or "imaging.cutout" in keys
    for p in plans:
        assert p.next_action == "fetch"
        assert p.product.status == "executable"


def test_execute_product_runs_via_kernel(monkeypatch):
    tgt = _fake_target("M87")
    plans = resolvers.recommend_plans(tgt)
    plan = next(p for p in plans if p.product.key == "object.spectrum")

    def fake_run(self, cap_name, params=None, **kwargs):
        assert params["target"] == "M87"
        return Artifact(id="spec_art", kind="spectrum", cap=cap_name, path="data/artifacts/spec.png", label="M87 spectrum")

    def fake_load(self, art_id):
        return {"spectrum": "ok"}

    monkeypatch.setattr(Kernel, "run", fake_run)
    monkeypatch.setattr(Kernel, "load", fake_load)

    res = resolvers.execute_product(tgt, plan)
    assert res.kind == "spectrum"
    assert "M87 spectrum" in res.label


def test_tui_resolve_target_integration(monkeypatch):
    """Verify that resolvers.resolve_target produces valid structures for the TUI."""
    monkeypatch.setattr(resolvers, "identify", lambda name: Table({
        "main_id": ["M87"], "otype": ["Gal"], "ra": [187.7059], "dec": [12.3911]
    }))

    target = resolvers.resolve_target("M87")
    assert target is not None
    assert target.display_name == "M87"
    assert target.object_class == "galaxy_agn"

    plans = resolvers.recommend_plans(target)
    assert len(plans) > 0
    assert plans[0].product.label
