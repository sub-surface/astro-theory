"""Instrument health checks — turn "· not yet run here" into an answer.

`run_checks` pings every registered archive's availability endpoint (no data
pull, just "are you up"), and reports local state: ADS token size,
manifest rows, candidate lists. Pure besides the injectable `pinger`, so the
CLI presents it and tests stay hermetic.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from . import candidates, config


def _default_pinger(url: str, timeout: float) -> tuple[bool, str]:
    """GET the endpoint; (ok, note). VO availability endpoints return 200 + XML."""
    import requests
    r = requests.get(url, timeout=timeout,
                     headers={"User-Agent": "Celestrium doctor"})
    ok = r.status_code == 200
    return ok, f"HTTP {r.status_code}"


def check_archives(pinger: Optional[Callable] = None,
                   timeout: float = 10.0) -> list[dict[str, Any]]:
    pinger = pinger or _default_pinger
    out = []
    for key, archive in config.ARCHIVES.items():
        rec: dict[str, Any] = {"archive": key, "label": archive.label,
                               "url": archive.health_url}
        if not archive.health_url:
            rec.update(status="unchecked", note="no health endpoint registered")
            out.append(rec)
            continue
        t0 = time.perf_counter()
        try:
            ok, note = pinger(archive.health_url, timeout)
            rec.update(status="up" if ok else "down", note=note,
                       ms=int((time.perf_counter() - t0) * 1000))
        except Exception as e:
            rec.update(status="down", note=f"{type(e).__name__}: {e}",
                       ms=int((time.perf_counter() - t0) * 1000))
        out.append(rec)
    return out


def check_local() -> dict[str, Any]:
    """Local instrument state: token footprint, provenance, candidates."""
    try:
        from . import ads
        ads._token()
        ads_ok = True
    except Exception:
        ads_ok = False
    from . import paths
    artifact_files = list(paths.ARTIFACTS.glob("*.*")) if paths.ARTIFACTS.exists() else []
    artifact_bytes = sum(f.stat().st_size for f in artifact_files)
    manifest_file = paths.DATA / "manifest.jsonl"
    manifest_rows = sum(1 for _ in manifest_file.open(encoding="utf-8")) if manifest_file.exists() else 0
    return {
        "ads_token": ads_ok,
        "cache_files": len(artifact_files),
        "cache_mb": round(artifact_bytes / 1e6, 2),
        "manifest_rows": manifest_rows,
        "candidate_lists": len(candidates.latest()),
    }


def run_checks(pinger: Optional[Callable] = None, ping: bool = True,
               timeout: float = 10.0) -> dict[str, Any]:
    report: dict[str, Any] = {"local": check_local()}
    report["archives"] = check_archives(pinger, timeout) if ping else []
    up = sum(1 for a in report["archives"] if a.get("status") == "up")
    report["summary"] = {
        "archives_up": up,
        "archives_checked": len(report["archives"]),
        "ads": "ok" if report["local"]["ads_token"] else "missing",
    }
    return report
