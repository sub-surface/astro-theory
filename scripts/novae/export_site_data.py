"""Export the T CrB data the public site plots (site/src/data/tcrb.json).

Reads data/tcrb/artifact_data.json (written by tcrb_analysis.py) and keeps only the light curves and
the headline numbers the /tcrb page shows, so the site builds without the git-ignored data/ tree.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "tcrb" / "artifact_data.json"
OUT = ROOT / "site" / "src" / "data" / "tcrb.json"


def main() -> None:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    r = d["results"]
    fam = r["srn_family"]["uv_luminosity_vs_tcrb_quiescence"]
    out = {
        "uvm2": d["uvm2"],
        "hard": d["hard"],
        "V": d["V"],
        "overlay": d["overlay"],
        "n": {
            "uvx_pairs": r["uv_xray_track"]["n_pairs"],
            "uvx_rho": r["uv_xray_track"]["spearman_rho"],
            "hard_now": r["uv_xray_track"]["latest"][0]["hard_rate"],
            "uv_last_bright": r["uv_dips"]["uvm2_2026_last_bright_mag"],
            "uv_latest": r["uv_dips"]["uvm2_2026_latest_mag"],
            "fade_days": r["viscous_drain_test"]["observed_2026_interval_days"],
            "drain_2mag_days": r["viscous_drain_test"]["fastest_2mag_fade_days"],
            "drain_30d_mag": r["viscous_drain_test"]["max_fade_in_30d_mag"],
            "period_d": r["ephemeris"]["period_d"],
            "period_sd_d": r["ephemeris"]["period_sd_d"],
            "ell_amp": r["ephemeris"]["incl_from_ellipsoidal"]["quiescent_amp_half_period_mag"],
            "ell_amp_sd": r["ephemeris"]["incl_from_ellipsoidal"]["quiescent_amp_sd"],
            "incl_lo": r["ephemeris"]["incl_from_ellipsoidal"]["lo"],
            "rec_median": r["forecast"]["recurrence"]["median"],
            "rec_q16": r["forecast"]["recurrence"]["q16"],
            "rec_q84": r["forecast"]["recurrence"]["q84"],
            "p_before": {k[9:]: v for k, v in r["forecast"]["recurrence"].items() if k.startswith("p_before_")},
            "intervals_yr": r["forecast"]["intervals_yr"],
            "analogue_eruption": r["dip_overlay"]["implied_eruption_alignment_B"],
            "lit": {k: r["literature"][k] for k in ("n_focused", "n_fulltext", "n_refereed", "n_circulars", "papers_2023_2026")},
            "forecasts_published": r["forecast_scoreboard"]["n_published"],
            "forecasts_lapsed": r["forecast_scoreboard"]["n_lapsed"],
        },
        "family": [
            {"name": k, "log_ratio": v["log10_ratio"], "sys_dex": v["sys_dex"]} for k, v in fam.items()
        ],
        "tess": {
            k: {f: v[f] for f in ("state", "flicker_rms_10min_6h_e_s", "high_freq_slope", "fap_global_max_peak", "fap_5000_7500s")}
            for k, v in r["tess_flicker"].items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
