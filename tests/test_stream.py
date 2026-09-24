"""Tests for real-time transient alert stream ingestion and triage engine."""
from pathlib import Path
import json
import pytest
import numpy as np

from celestrium.stream import (
    AlertRecord,
    TriageDecision,
    RubinBurstSimulator,
    FinkLiveAlertSource,
    StreamingTriageEngine,
)
from celestrium.astrojev import NUM_FEATURES


def test_rubin_burst_simulator():
    sim = RubinBurstSimulator(seed=123)
    alerts = list(sim.stream(limit=10))
    assert len(alerts) == 10
    for a in alerts:
        assert isinstance(a, AlertRecord)
        assert a.features.shape == (NUM_FEATURES,)
        assert np.all(np.isfinite(a.features))
        assert 0.0 <= a.ra <= 360.0
        assert -90.0 <= a.dec <= 90.0
        assert a.filter_band in ("u", "g", "r", "i", "z", "y")
        d = a.to_dict()
        assert "features" in d
        assert isinstance(d["features"], list)


def test_fink_live_alert_source_fallback():
    # Should fall back gracefully even with unreachable or invalid URL
    src = FinkLiveAlertSource(fink_url="https://invalid-nonexistent-domain.test/api", timeout_sec=0.5)
    alerts = list(src.stream(limit=5))
    assert len(alerts) == 5
    for a in alerts:
        assert isinstance(a, AlertRecord)
        assert a.features.shape == (NUM_FEATURES,)


def test_streaming_triage_engine():
    engine = StreamingTriageEngine(device="cpu")
    sim = RubinBurstSimulator(seed=999)
    alerts = list(sim.stream(limit=5))

    decisions = []
    for a in alerts:
        dec = engine.triage_record(a)
        assert isinstance(dec, TriageDecision)
        assert dec.action in ("URGENT_FOLLOWUP", "AUTO_CATALOG", "EXTEND_DELIBERATION", "PASS_DEFER")
        assert 0.0 <= dec.confidence <= 1.0
        assert 0.0 <= dec.epistemic_vacuity <= 1.0
        assert dec.latency_ms > 0.0
        decisions.append(dec)

    summary = engine.get_summary()
    assert summary["total_processed"] == 5
    assert sum(summary["actions"].values()) == 5
    assert summary["average_latency_ms"] > 0.0


def test_process_stream_with_sink(tmp_path: Path):
    sink_file = tmp_path / "stream_sink.jsonl"
    engine = StreamingTriageEngine(device="cpu")
    sim = RubinBurstSimulator(seed=42)

    decisions = list(engine.process_stream(source=sim, limit=8, output_sink=sink_file))
    assert len(decisions) == 8
    assert sink_file.is_file()

    lines = sink_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 8
    first_dec = json.loads(lines[0])
    assert "alert_id" in first_dec
    assert "action" in first_dec
    assert "bald_information_gain" in first_dec


def test_cli_stream_smoke():
    from typer.testing import CliRunner
    from celestrium.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["jev", "stream", "--broker", "rubin-sim", "--limit", "5"])
    assert res.exit_code == 0
    assert "Stream Performance Summary" in res.output or "Alert ID" in res.output
