"""
=============================================================================
Celestrium Target-of-Opportunity (ToO) Protocols & Observatory API Serializers
=============================================================================
Provides production-grade serializers and schema validators for autonomous
Target-of-Opportunity (ToO) follow-up dispatch to premier astronomical facilities:

1. Las Cumbres Observatory (LCOGT) Observation Portal API:
   - Tier 1 Robotic 1m Network (Sinistro Imager, gp/rp/ip filters)
   - Schema compliant with LCOGT RequestGroup / Configuration JSON-RPC API.

2. Gemini Observatory Phase II / Observation Tool (OT) Protocol:
   - Tier 3 8m-class Giant Telescope (GMOS-N / GMOS-S Longslit Spectroscopy)
   - Rapid ToO & Standard ToO observation payloads with grating & slit configs.

3. IVOA VOEvent 2.0 & TNS (Transient Name Server) Notice Schema:
   - Standardized international astronomical alert format with full provenance.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# 1. Las Cumbres Observatory (LCOGT) Protocol
# --------------------------------------------------------------------------- #
@dataclass
class LCOGTTarget:
    name: str
    ra_deg: float
    dec_deg: float
    epoch: float = 2000.0


@dataclass
class LCOGTInstrumentConfig:
    instrument_type: str = "1M0-SCICAM-SINISTRO"
    optical_elements: Dict[str, str] = field(default_factory=lambda: {"filter": "rp"})
    exposure_time_sec: float = 120.0
    exposure_count: int = 2
    bin_x: int = 1
    bin_y: int = 1


@dataclass
class LCOGTConstraints:
    max_airmass: float = 1.8
    min_lunar_distance_deg: float = 30.0
    max_lunar_phase: float = 1.0


def build_lcogt_too_request(
    target_name: str,
    ra_deg: float,
    dec_deg: float,
    filter_bands: List[str] = ("gp", "rp", "ip"),
    exposure_time_per_filter_sec: float = 90.0,
    proposal_id: str = "CELESTRIUM-2026B-001",
    ipp_value: float = 1.05,  # Intra-Proposal Priority
    reason: str = "Epistemic RLCD Alert Screening",
) -> Dict[str, Any]:
    """Builds a fully compliant LCOGT Observation Portal RequestGroup payload.
    
    Reference: LCO Observation Portal API v2 (https://developers.lco.global/)
    """
    configurations = []
    for f in filter_bands:
        configurations.append({
            "type": "EXPOSE",
            "instrument_type": "1M0-SCICAM-SINISTRO",
            "optical_elements": {"filter": f},
            "exposure_time": float(exposure_time_per_filter_sec),
            "exposure_count": 2,
            "bin_x": 1,
            "bin_y": 1,
            "extra_params": {
                "defocus": 0.0,
            }
        })

    payload = {
        "name": f"ToO_{target_name}_{int(time.time())}",
        "proposal": proposal_id,
        "ipp_value": float(ipp_value),
        "operator": "SINGLE",
        "observation_type": "RAPID_TOO",
        "requests": [
            {
                "acceptability_threshold": 80.0,
                "target": {
                    "name": target_name,
                    "type": "ICRS",
                    "ra": float(ra_deg),
                    "dec": float(dec_deg),
                    "epoch": 2000.0,
                },
                "location": {
                    "telescope_class": "1m0",
                },
                "constraints": {
                    "max_airmass": 1.8,
                    "min_lunar_distance": 30.0,
                    "max_lunar_phase": 1.0,
                },
                "configurations": configurations,
                "windows": [
                    {
                        "start": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                        "end": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 86400 * 2)),
                    }
                ],
            }
        ],
        "metadata": {
            "origin": "Celestrium Autonomous Follow-Up Engine",
            "algorithm": "Epistemic RLCD Constrained MDP",
            "reason": reason,
        }
    }
    return payload


# --------------------------------------------------------------------------- #
# 2. Gemini Observatory Phase II / GMOS Spectroscopy Protocol
# --------------------------------------------------------------------------- #
def build_gemini_too_request(
    target_name: str,
    ra_deg: float,
    dec_deg: float,
    r_mag: float,
    program_id: str = "GS-2026B-Q-104",
    facility: str = "Gemini-South",
    grating: str = "B600+_G5323",
    central_wavelength_nm: float = 600.0,
    slit_width_arcsec: float = 1.0,
    exposure_time_sec: float = 900.0,
    n_exposures: int = 4,
    too_type: str = "Rapid",  # Rapid (< 24h) or Standard (< 72h)
    reason: str = "High-Impact Rare Transient Candidate Confirmation",
) -> Dict[str, Any]:
    """Builds a compliant Gemini Observatory Phase II GMOS Longslit Spectroscopy payload.
    
    Reference: Gemini Observatory Phase II Observing Tool (OT) Data Dictionary.
    """
    # Coordinate sexagesimal conversion
    ra_hours = ra_deg / 15.0
    ra_h = int(ra_hours)
    ra_m = int((ra_hours - ra_h) * 60.0)
    ra_s = (ra_hours - ra_h - ra_m / 60.0) * 3600.0

    dec_sign = "+" if dec_deg >= 0 else "-"
    abs_dec = abs(dec_deg)
    dec_d = int(abs_dec)
    dec_m = int((abs_dec - dec_d) * 60.0)
    dec_s = (abs_dec - dec_d - dec_m / 60.0) * 3600.0

    ra_sexagesimal = f"{ra_h:02d}:{ra_m:02d}:{ra_s:05.2f}"
    dec_sexagesimal = f"{dec_sign}{dec_d:02d}:{dec_m:02d}:{dec_s:04.1f}"

    payload = {
        "program_id": program_id,
        "facility": facility,
        "too_type": too_type,
        "observation": {
            "title": f"ToO Spectroscopy of {target_name} (r={r_mag:.1f})",
            "priority": "HIGH" if too_type == "Rapid" else "MEDIUM",
            "target": {
                "name": target_name,
                "ra_deg": float(ra_deg),
                "dec_deg": float(dec_deg),
                "ra_sexagesimal": ra_sexagesimal,
                "dec_sexagesimal": dec_sexagesimal,
                "epoch": "J2000",
                "magnitude": float(r_mag),
                "band": "r",
            },
            "instrument": {
                "name": "GMOS-S" if "South" in facility else "GMOS-N",
                "mode": "Longslit",
                "grating": grating,
                "central_wavelength_nm": float(central_wavelength_nm),
                "filter": "OG515",
                "focal_plane_unit": f"Longslit {slit_width_arcsec} arcsec",
                "ccd_binning": "2x2",
                "gain_setting": "Low",
            },
            "observing_conditions": {
                "image_quality": "IQ85",
                "cloud_cover": "CC70",
                "sky_background": "SB80",
                "airmass_limit": 1.75,
            },
            "sequence": {
                "exposure_time_sec": float(exposure_time_sec),
                "exposure_count": int(n_exposures),
                "total_time_hours": float((exposure_time_sec * n_exposures + 1200) / 3600.0),  # Includes acquisition overhead
                "dither_spatial_arcsec": [0.0, 5.0, 0.0, -5.0],
                "dither_spectral_nm": [0.0, 5.0, 0.0, 5.0],  # GMOS CCD chip gap dithers
            }
        },
        "trigger_metadata": {
            "origin": "Celestrium Autonomous Decision Engine",
            "policy": "Epistemic RLCD with Conformal Safety Gate",
            "urgency": "CRITICAL" if too_type == "Rapid" else "NORMAL",
            "justification": reason,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }
    return payload


# --------------------------------------------------------------------------- #
# 3. IVOA VOEvent 2.0 / TNS Alert Notification Protocol
# --------------------------------------------------------------------------- #
def build_voevent_packet(
    alert_id: str,
    ra_deg: float,
    dec_deg: float,
    mag: float,
    mag_err: float,
    filter_band: str,
    classification: str,
    confidence: float,
    epistemic_uncertainty: float,
    action_dispatched: str,
    facility_dispatched: str,
) -> Dict[str, Any]:
    """Generates an IVOA VOEvent 2.0 compliant JSON packet for astronomical alert broadcast."""
    ivorn = f"ivo://celestrium.org/too/{alert_id}#{int(time.time())}"
    return {
        "voevent": {
            "version": "2.0",
            "ivorn": ivorn,
            "role": "observation",
            "who": {
                "author_ivorn": "ivo://celestrium.org",
                "contact_name": "Celestrium Autonomous Follow-Up Project",
                "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "wherewhen": {
                "coord_system": "UTC-ICRS-TOPO",
                "ra_deg": float(ra_deg),
                "dec_deg": float(dec_deg),
                "pos_error_arcsec": 0.15,
                "time_mjd": float(time.time() / 86400.0 + 40587.0),
            },
            "what": {
                "params": [
                    {"name": "observed_mag", "value": float(mag), "unit": "mag"},
                    {"name": "mag_error", "value": float(mag_err), "unit": "mag"},
                    {"name": "filter", "value": filter_band, "unit": ""},
                    {"name": "predicted_class", "value": classification, "unit": ""},
                    {"name": "confidence", "value": float(confidence), "unit": ""},
                    {"name": "epistemic_vacuity", "value": float(epistemic_uncertainty), "unit": ""},
                    {"name": "action_dispatched", "value": action_dispatched, "unit": ""},
                    {"name": "facility_target", "value": facility_dispatched, "unit": ""},
                ]
            },
            "how": {
                "description": "Triggered via Celestrium Epistemic RLCD Decision Policy under constrained aperture budget."
            }
        }
    }


# --------------------------------------------------------------------------- #
# 4. Pipeline Validation & Schema Validator
# --------------------------------------------------------------------------- #
class ToOPipelineValidator:
    """Validates data pipeline integrity and schema conformance for alert-to-ToO routing."""

    @staticmethod
    def validate_alert_packet(alert: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates incoming alert packet schema."""
        errors = []
        required_fields = ["alert_id", "ra", "dec", "mag", "mag_err", "filter_band"]
        for f in required_fields:
            if f not in alert:
                errors.append(f"Missing required alert field: '{f}'")

        if "ra" in alert:
            ra = alert["ra"]
            if not isinstance(ra, (int, float)) or not (0.0 <= ra <= 360.0):
                errors.append(f"Invalid RA value: {ra} (must be in [0, 360])")

        if "dec" in alert:
            dec = alert["dec"]
            if not isinstance(dec, (int, float)) or not (-90.0 <= dec <= 90.0):
                errors.append(f"Invalid Dec value: {dec} (must be in [-90, 90])")

        if "mag" in alert:
            mag = alert["mag"]
            if not isinstance(mag, (int, float)) or math.isnan(mag) or not (0.0 < mag < 35.0):
                errors.append(f"Non-physical magnitude: {mag}")

        return (len(errors) == 0, errors)

    @staticmethod
    def validate_lcogt_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates LCOGT RequestGroup payload schema."""
        errors = []
        if "name" not in payload or not payload["name"].startswith("ToO_"):
            errors.append("Invalid or missing ToO name")
        if "proposal" not in payload or not payload["proposal"]:
            errors.append("Missing proposal identifier")
        if "requests" not in payload or len(payload["requests"]) == 0:
            errors.append("Missing requests block")
        else:
            req = payload["requests"][0]
            if "target" not in req or "ra" not in req["target"] or "dec" not in req["target"]:
                errors.append("Missing target coordinates in request")
            if "configurations" not in req or len(req["configurations"]) == 0:
                errors.append("Missing optical configurations")
        return (len(errors) == 0, errors)

    @staticmethod
    def validate_gemini_payload(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validates Gemini GMOS Phase II payload schema."""
        errors = []
        if "program_id" not in payload:
            errors.append("Missing Gemini program_id")
        if "observation" not in payload:
            errors.append("Missing observation block")
        else:
            obs = payload["observation"]
            if "target" not in obs or "ra_deg" not in obs["target"]:
                errors.append("Missing target block in Gemini observation")
            if "instrument" not in obs or "grating" not in obs["instrument"]:
                errors.append("Missing instrument / grating specification")
            if "sequence" not in obs or "exposure_time_sec" not in obs["sequence"]:
                errors.append("Missing sequence / exposure time specification")
        return (len(errors) == 0, errors)
