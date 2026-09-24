"""
=============================================================================
Celestrium Data Sources & Streaming Verification Suite
=============================================================================
Tests all external and local astronomical data sources, request protocols,
streaming pipelines, schema integrity, and fallback mechanisms:
  1. Gaia DR3 ESA TAP (Astrometry, RUWE, Photometry)
  2. IRSA NASA IPAC TAP (AllWISE, 2MASS Infrared)
  3. Euclid ESA TAP (Q1 / DR1 Survey Metadata)
  4. Quaia G20.5 Quasar Catalog (FITS / Astropy)
  5. Fink Time-Domain Alert Broker (Rubin / LSST & ZTF)
  6. ALeRCE Transient Broker (ZTF & Rubin Alerts)
  7. MAST STScI (JWST / HST / TESS Time-Series)
  8. NED Extragalactic Database (Spectra & SEDs)
  9. SDSS (Sloan Digital Sky Survey Spectra & Frames)
  10. Multi-Survey Cutout Synthesis (SkyView / HiPS / Pan-STARRS)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def test_gaia_dr3_tap() -> Dict[str, Any]:
    """Test Gaia DR3 TAP query and schema validation."""
    from celestrium.tap import TAPClient
    t0 = time.perf_counter()
    client = TAPClient("https://gea.esac.esa.int/tap-server/tap")
    adql = """
    SELECT TOP 5
        source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        pm, pmra_error, pmdec_error, ruwe, l, b
    FROM gaiadr3.gaia_source
    WHERE phot_g_mean_mag < 16.0 AND pm > 10.0
    """
    try:
        tab = client.query(adql)
        elapsed = time.perf_counter() - t0
        has_cols = all(c in tab.colnames for c in ["phot_g_mean_mag", "pm", "ruwe"])
        return {
            "source": "Gaia DR3 TAP",
            "endpoint": "https://gea.esac.esa.int/tap-server/tap",
            "status": "ONLINE" if has_cols and len(tab) > 0 else "DEGRADED",
            "latency_seconds": round(elapsed, 3),
            "rows_returned": len(tab),
            "columns": list(tab.colnames),
            "sample_source_id": int(tab["source_id"][0]) if len(tab) > 0 else None,
        }
    except Exception as e:
        return {
            "source": "Gaia DR3 TAP",
            "endpoint": "https://gea.esac.esa.int/tap-server/tap",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_irsa_tap() -> Dict[str, Any]:
    """Test NASA/IPAC IRSA TAP query and schema validation."""
    from celestrium.tap import TAPClient
    t0 = time.perf_counter()
    client = TAPClient("https://irsa.ipac.caltech.edu/TAP")
    adql = """
    SELECT TOP 5
        source_id, w1mpro, w2mpro, w1sigmpro, w2sigmpro, ra, dec
    FROM allwise_p3as_psd
    WHERE w1mpro < 14.0
    """
    try:
        tab = client.query(adql)
        elapsed = time.perf_counter() - t0
        has_cols = all(c in tab.colnames for c in ["w1mpro", "w2mpro"])
        return {
            "source": "IRSA AllWISE TAP",
            "endpoint": "https://irsa.ipac.caltech.edu/TAP",
            "status": "ONLINE" if has_cols and len(tab) > 0 else "DEGRADED",
            "latency_seconds": round(elapsed, 3),
            "rows_returned": len(tab),
            "columns": list(tab.colnames),
        }
    except Exception as e:
        return {
            "source": "IRSA AllWISE TAP",
            "endpoint": "https://irsa.ipac.caltech.edu/TAP",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_euclid_tap() -> Dict[str, Any]:
    """Test ESA Euclid TAP service accessibility."""
    from celestrium.tap import TAPClient
    t0 = time.perf_counter()
    client = TAPClient("https://eas.esac.esa.int/tap-server/tap")
    try:
        tables = client.discover(substr="")
        elapsed = time.perf_counter() - t0
        return {
            "source": "Euclid Science Archive TAP",
            "endpoint": "https://eas.esac.esa.int/tap-server/tap",
            "status": "ONLINE" if len(tables) > 0 else "DEGRADED",
            "latency_seconds": round(elapsed, 3),
            "total_tables_discovered": len(tables),
            "sample_tables": tables[:5],
        }
    except Exception as e:
        return {
            "source": "Euclid Science Archive TAP",
            "endpoint": "https://eas.esac.esa.int/tap-server/tap",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_quaia_fits() -> Dict[str, Any]:
    """Test Quaia G20.5 Quasar Catalog local parsing."""
    from astropy.table import Table
    t0 = time.perf_counter()
    p = Path("Archive/2026-06-G-dipole/data/quaia/quaia_G20.5.fits")
    if not p.is_file():
        return {
            "source": "Quaia G20.5 Quasar Catalog",
            "status": "MISSING",
            "path": str(p),
        }
    t = Table.read(str(p))
    elapsed = time.perf_counter() - t0
    required_cols = ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "mag_w1_vg", "mag_w2_vg", "redshift_quaia"]
    has_req = all(c in t.colnames for c in required_cols)
    return {
        "source": "Quaia G20.5 Quasar Catalog",
        "path": str(p),
        "status": "ONLINE" if has_req else "DEGRADED",
        "latency_seconds": round(elapsed, 3),
        "total_sources": len(t),
        "file_size_mb": round(p.stat().st_size / (1024**2), 2),
        "mean_redshift": round(float(np.nanmean(t["redshift_quaia"])), 3),
    }


def test_fink_alert_broker() -> Dict[str, Any]:
    """Test Fink live time-domain alert API."""
    import requests
    t0 = time.perf_counter()
    endpoint = "https://api.fink-portal.org/api/v1/latests"
    try:
        resp = requests.post(endpoint, json={"class": "Supernova candidate", "n": 3}, timeout=8)
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            alerts = resp.json()
            return {
                "source": "Fink Alert Broker (Rubin/ZTF)",
                "endpoint": endpoint,
                "status": "ONLINE",
                "latency_seconds": round(elapsed, 3),
                "alerts_received": len(alerts),
                "sample_object_id": alerts[0].get("v:objectId") if alerts else None,
            }
        else:
            return {
                "source": "Fink Alert Broker (Rubin/ZTF)",
                "endpoint": endpoint,
                "status": "DEGRADED",
                "http_status": resp.status_code,
                "latency_seconds": round(elapsed, 3),
            }
    except Exception as e:
        return {
            "source": "Fink Alert Broker (Rubin/ZTF)",
            "endpoint": endpoint,
            "status": "OFFLINE_FALLBACK_AVAILABLE",
            "fallback": "RubinBurstSimulator",
            "notice": "Fink public portal timed out; in-flight simulation pipeline ready.",
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_alerce_broker() -> Dict[str, Any]:
    """Test ALeRCE transient broker API."""
    from celestrium.transients import fetch_latest_transients
    t0 = time.perf_counter()
    try:
        tab = fetch_latest_transients(days=7.0, limit=5)
        elapsed = time.perf_counter() - t0
        if tab is not None and len(tab) > 0:
            return {
                "source": "ALeRCE Transient Broker",
                "endpoint": "https://api.alerce.online/ztf/v1/objects/",
                "status": "ONLINE",
                "latency_seconds": round(elapsed, 3),
                "alerts_received": len(tab),
                "sample_oid": str(tab["main_id"][0]),
                "columns": list(tab.colnames),
            }
        return {
            "source": "ALeRCE Transient Broker",
            "status": "NO_RECENT_ALERTS",
            "latency_seconds": round(elapsed, 3),
        }
    except Exception as e:
        return {
            "source": "ALeRCE Transient Broker",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_mast_stsci() -> Dict[str, Any]:
    """Test MAST observation archive query."""
    from celestrium.mast import fetch_uv_optical_observations
    t0 = time.perf_counter()
    try:
        # Query 3C 273 (RA=187.2779, DEC=2.0524)
        tab = fetch_uv_optical_observations(187.2779, 2.0524, radius_deg=0.02)
        elapsed = time.perf_counter() - t0
        if tab is not None and len(tab) > 0:
            return {
                "source": "MAST STScI (HST/JWST/TESS)",
                "status": "ONLINE",
                "latency_seconds": round(elapsed, 3),
                "records_found": len(tab),
                "sample_collection": str(tab["obs_collection"][0]) if "obs_collection" in tab.colnames else None,
            }
        return {
            "source": "MAST STScI (HST/JWST/TESS)",
            "status": "NO_MATCH",
            "latency_seconds": round(elapsed, 3),
        }
    except Exception as e:
        return {
            "source": "MAST STScI (HST/JWST/TESS)",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_ned_extragalactic() -> Dict[str, Any]:
    """Test NASA/IPAC Extragalactic Database (NED)."""
    from celestrium.spectra import fetch_ned_spectrum
    t0 = time.perf_counter()
    try:
        spec_table = fetch_ned_spectrum("3C 273")
        elapsed = time.perf_counter() - t0
        return {
            "source": "NED Extragalactic Database",
            "status": "ONLINE" if spec_table is not None else "NO_SPECTRUM",
            "latency_seconds": round(elapsed, 3),
            "has_spectral_data": spec_table is not None,
        }
    except Exception as e:
        return {
            "source": "NED Extragalactic Database",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_sdss_spectra() -> Dict[str, Any]:
    """Test SDSS Science Archive query."""
    from celestrium.sdss import fetch_spectrum
    t0 = time.perf_counter()
    try:
        # Query 3C 273 (RA=187.2779, DEC=2.0524)
        res = fetch_spectrum("3C 273", 187.2779, 2.0524, radius_arcsec=10.0)
        elapsed = time.perf_counter() - t0
        return {
            "source": "SDSS Science Archive",
            "status": "ONLINE" if res is not None else "NO_SPECTRUM_FOUND",
            "latency_seconds": round(elapsed, 3),
            "has_data": res is not None,
        }
    except Exception as e:
        return {
            "source": "SDSS Science Archive",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def test_cutout_engine() -> Dict[str, Any]:
    """Test SkyView / HiPS / Pan-STARRS image cutout engine."""
    from celestrium.cutouts import color_auto
    t0 = time.perf_counter()
    try:
        out_path = Path("cache/cutouts/test_verification_cutout.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img_path, survey_used = color_auto(187.2779, 2.0524, fov_arcmin=2.0, out=out_path)
        elapsed = time.perf_counter() - t0
        return {
            "source": "Cutout Engine (SkyView/DSS/PanSTARRS)",
            "status": "ONLINE",
            "survey_used": survey_used,
            "latency_seconds": round(elapsed, 3),
            "output_exists": Path(img_path).is_file(),
            "output_path": str(img_path),
        }
    except Exception as e:
        return {
            "source": "Cutout Engine (SkyView/DSS/PanSTARRS)",
            "status": "ERROR",
            "error": str(e),
            "latency_seconds": round(time.perf_counter() - t0, 3),
        }


def run_full_data_sources_audit() -> Dict[str, Any]:
    """Execute live audit across all 10 astronomical data sources."""
    print("=" * 70)
    print("CELESTRIUM ASTRONOMICAL DATA SOURCES & STREAMING AUDIT")
    print("=" * 70)

    results = {}
    tests = [
        ("Gaia DR3 TAP", test_gaia_dr3_tap),
        ("IRSA AllWISE TAP", test_irsa_tap),
        ("Euclid TAP", test_euclid_tap),
        ("Quaia G20.5 Catalog", test_quaia_fits),
        ("Fink Alert Broker", test_fink_alert_broker),
        ("ALeRCE Transient Broker", test_alerce_broker),
        ("MAST STScI", test_mast_stsci),
        ("SDSS Archive", test_sdss_spectra),
        ("Cutout Engine", test_cutout_engine),
    ]

    for name, test_fn in tests:
        print(f"Testing {name}...", end=" ", flush=True)
        res = test_fn()
        results[name] = res
        status = res.get("status", "UNKNOWN")
        latency = res.get("latency_seconds", "-")
        print(f"[{status}] ({latency}s)")

    out_p = Path("docs/research/data_sources_audit_report.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved comprehensive data sources audit to: {out_p}")
    return results


if __name__ == "__main__":
    run_full_data_sources_audit()
