"""Tests for Celestrium Observatory Dashboard & Telemetry Cockpit."""
from typer.testing import CliRunner
from celestrium.cli import app
from celestrium.dashboard import build_dashboard_data, render_ascii_sky_map


def test_build_dashboard_data():
    data = build_dashboard_data()
    assert isinstance(data, dict)
    assert "gpu_name" in data
    assert "has_cuda" in data
    assert "euclid_dr1" in data
    assert "days_remaining" in data["euclid_dr1"]
    assert data["euclid_dr1"]["days_remaining"] >= 0
    assert "budget" in data
    assert data["budget"]["remaining_usd"] > 0
    assert "checkpoints" in data
    assert "datasets" in data


def test_render_ascii_sky_map():
    sky = render_ascii_sky_map()
    assert isinstance(sky, str)
    assert "CMB Apex" in sky
    assert "Quaia Apex" in sky
    assert "Galactic Plane" in sky


def test_cli_dashboard_smoke():
    runner = CliRunner()
    res = runner.invoke(app, ["dashboard"])
    assert res.exit_code == 0
    assert "CELESTRIUM OBSERVATORY" in res.output or "Observatory Telemetry" in res.output


def test_cli_lab_alias_smoke():
    runner = CliRunner()
    res = runner.invoke(app, ["lab"])
    assert res.exit_code == 0
    assert "CELESTRIUM OBSERVATORY" in res.output or "Observatory Telemetry" in res.output
