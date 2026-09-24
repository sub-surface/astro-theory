"""
Tests for Celestrium Target-of-Opportunity (ToO) API Serializers and Pipeline Validator.
"""
import pytest
from celestrium.too_protocol import (
    build_lcogt_too_request,
    build_gemini_too_request,
    build_voevent_packet,
    ToOPipelineValidator,
)


def test_build_lcogt_too_request():
    payload = build_lcogt_too_request(
        target_name="TEST_TRANS_01",
        ra_deg=180.25,
        dec_deg=12.50,
        filter_bands=["gp", "rp"],
        exposure_time_per_filter_sec=60.0,
    )
    assert payload["proposal"] == "CELESTRIUM-2026B-001"
    assert payload["observation_type"] == "RAPID_TOO"
    assert len(payload["requests"]) == 1
    req = payload["requests"][0]
    assert req["target"]["name"] == "TEST_TRANS_01"
    assert req["target"]["ra"] == 180.25
    assert req["target"]["dec"] == 12.50
    assert len(req["configurations"]) == 2
    assert req["configurations"][0]["optical_elements"]["filter"] == "gp"
    assert req["configurations"][1]["optical_elements"]["filter"] == "rp"

    # Validate against schema validator
    is_valid, errors = ToOPipelineValidator.validate_lcogt_payload(payload)
    assert is_valid is True
    assert len(errors) == 0


def test_build_gemini_too_request():
    payload = build_gemini_too_request(
        target_name="KN_CAND_02",
        ra_deg=150.0,
        dec_deg=-15.0,
        r_mag=20.5,
        facility="Gemini-South",
        too_type="Rapid",
    )
    assert payload["facility"] == "Gemini-South"
    assert payload["too_type"] == "Rapid"
    obs = payload["observation"]
    assert obs["instrument"]["name"] == "GMOS-S"
    assert obs["sequence"]["exposure_count"] == 4
    assert obs["target"]["ra_deg"] == 150.0
    assert obs["target"]["dec_deg"] == -15.0
    assert obs["target"]["ra_sexagesimal"] == "10:00:00.00"
    assert obs["target"]["dec_sexagesimal"] == "-15:00:00.0"

    # Validate against schema validator
    is_valid, errors = ToOPipelineValidator.validate_gemini_payload(payload)
    assert is_valid is True
    assert len(errors) == 0


def test_build_voevent_packet():
    packet = build_voevent_packet(
        alert_id="ALERT_9999",
        ra_deg=200.0,
        dec_deg=35.0,
        mag=19.2,
        mag_err=0.05,
        filter_band="g",
        classification="Kilonova",
        confidence=0.91,
        epistemic_uncertainty=0.12,
        action_dispatched="GEMINI_GMOS_RAPID_SPECTROSCOPY",
        facility_dispatched="Gemini-North",
    )
    assert "voevent" in packet
    assert packet["voevent"]["version"] == "2.0"
    params = {p["name"]: p["value"] for p in packet["voevent"]["what"]["params"]}
    assert params["predicted_class"] == "Kilonova"
    assert params["confidence"] == 0.91
    assert params["action_dispatched"] == "GEMINI_GMOS_RAPID_SPECTROSCOPY"


def test_pipeline_validator_quarantine():
    # Valid alert
    valid_alert = {
        "alert_id": "AL_001",
        "ra": 120.0,
        "dec": 45.0,
        "mag": 18.5,
        "mag_err": 0.02,
        "filter_band": "r",
    }
    is_valid, errors = ToOPipelineValidator.validate_alert_packet(valid_alert)
    assert is_valid is True
    assert len(errors) == 0

    # Corrupted alert (missing coordinates, negative mag)
    corrupt_alert = {
        "alert_id": "AL_CORRUPT",
        "ra": 450.0,  # Out of range
        "mag": -5.0,  # Non-physical
    }
    is_valid, errors = ToOPipelineValidator.validate_alert_packet(corrupt_alert)
    assert is_valid is False
    assert len(errors) >= 3
