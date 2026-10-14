"""EXP-2026-Z follow-up: adversarial audit of the 1943 eruption of V347 Nor (= Hen 2-147) on the Harvard plates.

Each gate tests one alternative explanation with data; the numbers go to docs/results/v347nor_audit.json.
  G1 astrometry      the transient is at V347 Nor, not a neighbour (mean offset of all bright detections)
  G2 uniqueness      no comparable brightening on the deep plates of 1890-1942 and 1947-1990
  G3 rise            last pre-eruption limit vs first detection
  G4 luminosity      peak absolute magnitude (distance from the Mira P-L relation and Gaia DR3, SFD extinction)
  G5 decline         t2, t3 from a power-law fit to the detections (bootstrap)
  G6 coincidence     probability that an unrelated nova falls within 2" of any of the 1,211 targets in a century
  G7 alternatives    Mira maximum, Z And-type outburst, plate defect: amplitude/rise/series arguments
  G8 nebula          consistency with the HST expansion of the ring (Santander-Garcia et al. 2007): the deceleration
                     index n (R ~ t^n) that a 1943 origin requires, and the prediction for a third imaging epoch
Usage: python scripts/novae/v347nor_audit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "novae"))
from celestrium.archives import dasch  # noqa: E402
from celestrium.novae import plates as P  # noqa: E402
from rn_search import load  # noqa: E402

RN = ROOT / "data" / "rn"
OUT = ROOT / "docs" / "results" / "v347nor_audit.json"
SITE = ROOT / "site" / "src" / "data" / "v347nor.json"
TID = "T1274"
T_FIRST = 1943.153  # rb12036, 1943 Feb 26 01:32 UT, B = 7.99 (flagged saturated)
NB_6 = (243.501752, -56.992055)  # the 6.5" catalogue neighbour


def main():
    rng = np.random.default_rng(2026)
    T = pd.read_csv(RN / "targets.csv").set_index("tid", drop=False)
    row = T.loc[TID]
    ra0, de0 = row.ra, row.dec
    cd = np.cos(np.radians(de0))
    NB = pd.read_csv(RN / "neighbours.csv").drop_duplicates()
    nb = NB[(NB.tid == TID) & (NB.role == "t")]
    lc = load(TID, "t", nb.to_dict("records"), (ra0, de0))
    g = lc["good"]
    t, det, mag, lim, ser = lc["t"][g], lc["det"][g], lc["mag"][g], lc["lim"][g], lc["series"][g]
    out = {"object": "V347 Nor = Hen 2-147 = Wray 16-208 (D-type symbiotic, Mira P = 373 d)",
           "first_detection": {"plate": "rb12036", "date": "1943-02-26 01:32 UT", "saturated_flag": True, "year": T_FIRST, "B": 7.99}}

    # G1 astrometry: every raw detection brighter than 13.5 in 1943-46, from all files that hold them
    rows = []
    for f in [RN / "lc" / f"{TID}_t.csv.gz"] + [RN / "lc" / "nb" / f"{x.refcat}_{x.gsc}_{x.ref}.csv.gz"
                                                for x in nb.itertuples()]:
        c = dasch.parse_csv_lines(dasch.cached(f, None))
        yr = 2000 + (c["date_jd"] - 2451545.0) / 365.25
        w = np.isfinite(c["magcal_magdep"]) & (yr > 1942.9) & (yr < 1946.5) & (c["magcal_magdep"] < 13.5)
        for i in np.where(w)[0]:
            rows.append(((c["ra_deg"][i] - ra0) * cd * 3600, (c["dec_deg"][i] - de0) * 3600, str(c["series"][i])))
    D = pd.DataFrame(rows, columns=["dE", "dN", "series"])
    sem = D[["dE", "dN"]].std() / np.sqrt(len(D))
    nbE, nbN = (NB_6[0] - ra0) * cd * 3600, (NB_6[1] - de0) * 3600
    z_nb = np.hypot((D.dE.mean() - nbE) / sem.dE, (D.dN.mean() - nbN) / sem.dN)
    out["G1_astrometry"] = {"n": int(len(D)), "mean_offset_arcsec": [round(D.dE.mean(), 2), round(D.dN.mean(), 2)],
                            "sem_arcsec": [round(sem.dE, 2), round(sem.dN, 2)],
                            "median_offset_arcsec": [round(D.dE.median(), 2), round(D.dN.median(), 2)],
                            "neighbour_6.5as_offset": [round(nbE, 2), round(nbN, 2)],
                            "neighbour_rejected_sigma": round(float(z_nb), 1),
                            "series": sorted(D.series.unique().tolist())}

    # G2 uniqueness
    deep = (~det) & (lim > 13)
    bright = det & (mag < 13.5)
    out["G2_uniqueness"] = {"deep_plates_1890_1942": int((deep & (t < 1942.9)).sum()),
                            "deep_plates_1947_1990": int((deep & (t > 1946.5)).sum()),
                            "bright_detections_outside_1943_46": int((bright & ((t < 1942.9) | (t > 1946.5))).sum()),
                            "eruption_nights": P.n_nights(t[det & (t > 1943.1) & (t < 1946.5)])}

    # G3 rise
    pre = (~det) & (t < T_FIRST - 0.5 / 365.25) & (t > T_FIRST - 30 / 365.25)  # earlier nights only
    k = np.argmax(np.where(pre, lim, -np.inf))
    last = np.where(pre)[0][np.argmax(t[pre])]
    deep_last = (~det) & (t < T_FIRST - 0.5 / 365.25) & (lim > 12.9)
    kl = np.where(deep_last)[0][np.argmax(t[deep_last])]
    out["G3_rise"] = {"deepest_limit_in_prior_30_days": {"year": round(float(t[k]), 3), "series": str(ser[k]),
                                                          "limit": round(float(lim[k]), 2)},
                      "last_limit_before": {"year": round(float(t[last]), 3), "series": str(ser[last]),
                                            "limit": round(float(lim[last]), 2)},
                      "last_limit_fainter_than_12.9": {"year": round(float(t[kl]), 3), "series": str(ser[kl]),
                                                       "limit": round(float(lim[kl]), 2)},
                      "rise_mag_min": round(float(lim[kl] - 7.99), 2),
                      "rise_days_max": round(float((T_FIRST - t[kl]) * 365.25), 1),
                      "superseded_by": "G3_onset_from_cutouts: the Feb 15 plate shows the star; its catalogue limit "
                                       "is a pipeline miss, so this catalogue-only rise is too conservative",
                      "same_night_mf": "an mf plate the same night measured 10.77 at 12 arcsec offset (saturated; "
                                       "deep plates read bright stars too faint)"}

    # G4 luminosity (Monte Carlo over distance, extinction, photographic calibration at saturation)
    N = 200_000
    d = rng.normal(3.0, 0.4, N)  # kpc: Mira P-L 3.0 +- 0.4 (Santander-Garcia+ 2007); Gaia DR3 parallax 0.34 +- 0.11 mas
    E = rng.normal(0.47, 0.08, N)  # SFD / S&F total E(B-V) at l = 327.9, b = -4.3
    B = 7.99 + rng.normal(0, 0.3, N)
    MB = B - 5 * np.log10(d * 100) - 4.1 * E
    MBc = MB + (9.65 - 7.99)  # without rb12036: first unsaturated detection, bi00897, 1943 May 10 (day 74)
    out["G4_luminosity"] = {"M_B_at_first_detection": round(float(np.median(MB)), 2),
                            "M_B_68": [round(float(x), 2) for x in np.percentile(MB, [16, 84])],
                            "M_B_day74": round(float(np.median(MBc)), 2),
                            "M_B_day74_68": [round(float(x), 2) for x in np.percentile(MBc, [16, 84])],
                            "note": "true peak at least this bright; Z And-type outbursts peak at M ~ -2 to -4; "
                                    "RR Tel (1944) ~ -6; classical novae -5 to -9"}

    # G5 decline: m = a + b log10(dt + 1) on detections 1943.1-1947
    w = det & (t > 1943.1) & (t < 1947)
    dt, m = (t[w] - T_FIRST) * 365.25, mag[w]
    A = np.vstack([np.ones_like(dt), np.log10(dt + 1)]).T

    def t_at(dm, p):
        return 10 ** ((7.99 + dm - p[0]) / p[1]) - 1

    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(dt), len(dt))
        p = np.linalg.lstsq(A[i], m[i], rcond=None)[0]
        boot.append((t_at(2, p), t_at(3, p)))
    boot = np.array(boot)
    out["G5_decline"] = {"n_detections": int(w.sum()),
                         "t2_days": [round(float(x)) for x in np.percentile(boot[:, 0], [50, 16, 84])],
                         "t3_days": [round(float(x)) for x in np.percentile(boot[:, 1], [50, 16, 84])],
                         "note": "from the first detection; upper limits if the true peak was earlier and brighter"}

    # G6 chance coincidence with an unrelated nova
    from astropy.coordinates import SkyCoord
    S = T[T.src.str.contains("Schaefer22")]
    c0 = SkyCoord(ra0, de0, unit="deg")
    n20 = int((SkyCoord(S.ra.values, S.dec.values, unit="deg").separation(c0).deg < 20).sum())
    area = 2 * np.pi * (1 - np.cos(np.radians(20))) * (180 / np.pi) ** 2
    rho = n20 / area / 125.0  # catalogued novae per deg^2 per yr near this position (catalogue span ~125 yr)
    p_one = rho * 10 * np.pi * (2.0 / 3600) ** 2 * 100  # x10 for uncatalogued novae, r = 2", 100-yr window
    out["G6_coincidence"] = {"catalogued_novae_within_20deg": n20, "p_one_target": float(f"{p_one:.2e}"),
                             "p_any_of_1211_targets": float(f"{p_one * 1211:.2e}")}

    # G7 alternatives
    out["G7_alternatives"] = {
        "mira_maximum": "excluded: quiescent B ~ 17-20 (POSS-II J 17.3, SPM4 B 19.8); no maximum brighter than the "
                        "plate limits in 50 yr of annual cycles before 1943; a Mira cannot rise >= 5 mag in <= 11 d",
        "z_and_outburst": "disfavoured: amplitude >= 9 mag above quiescent B (B_q ~ 17-19.8; >= 7.4 mag from the first "
                          "unsaturated plate, day 74) and M_B ~ -6.3 at 3 kpc, vs 2-4 mag and M ~ -2 to -4; "
                          "absent on Feb 9 (B > 12.9) and Feb 10 (> 11.7), strong on Feb 11 on two cameras: a ~1-day rise vs "
                          "weeks-months",
        "plate_defect_or_asteroid": f"excluded: {len(D.series.unique())} plate series, "
                                    f"{out['G2_uniqueness']['eruption_nights']} nights over 3 years, stationary",
        "dasch_misses": "56 eruption-window plates logged as non-detections show the star in image stamps "
                        "(docs/figures/rn/v347nor_dasch_misses.png): pipeline misses, not contradictions"}

    # G8 nebula: HST [N II] magnification 2001.75 -> 2004.73 (Santander-Garcia+ 2007, Fig. 8)
    dt_hst, t_2001 = 2.99, 2001.75 - T_FIRST
    regions = {"N": 1.013, "S": 1.015, "E": 1.017, "W": 1.029}
    g8 = {}
    for r, M in regions.items():
        ballistic_age = dt_hst / (M - 1)
        n = t_2001 * (M - 1) / dt_hst  # deceleration index a 1943 origin needs (R ~ t^n)
        # ratio of the ring size between 2001.75 and 2027.0 under each hypothesis
        r_ball = (ballistic_age + 25.25) / ballistic_age
        r_1943 = ((t_2001 + 25.25) / t_2001) ** n
        g8[r] = {"magnification": M, "ballistic_age_yr": round(ballistic_age), "n_needed_for_1943": round(n, 2),
                 "growth_2001_to_2027_ballistic": round(r_ball, 3), "growth_2001_to_2027_1943": round(r_1943, 3)}
    out["G8_nebula"] = {"regions": g8, "t_since_1943_at_2001": round(t_2001, 1),
                        "mean_speed_kms_at_3kpc": round(7800 / t_2001 * 1.496e8 / 3.156e7),
                        "physical_n": "0.25 (momentum, uniform), 0.4 (Sedov, uniform), 0.5 (momentum, r^-2 wind), "
                                      "0.67 (energy, r^-2 wind)",
                        "note": "a 1943 origin needs n = 0.25-0.57, the decelerating-shock regime that Santander-Garcia "
                                "et al. (2007) independently inferred; a ballistic ring needs an unrecorded outburst "
                                "~100-230 yr ago. A third [N II] epoch separates the two (growth rates differ by ~1-3%)."}
    # G3 from the cutouts: the catalogue misses the first two weeks (v347nor_stamps.py)
    sj = ROOT / "docs" / "results" / "v347nor_stamps.json"
    if sj.exists():
        st = json.loads(sj.read_text(encoding="utf-8"))
        pre = st["pre_eruption"]
        on = [p for p in pre if p["S"] > st["control_S_p99"] and p["lim"] > 11.5]
        before = [p for p in pre if p["S"] <= st["control_S_p99"] and p["date_utc"] < on[0]["date_utc"]]
        out["G3_onset_from_cutouts"] = {
            "last_absent": before[-1], "deepest_absent_within_3d": max(
                (p for p in before if p["date_utc"] >= "1943-02-08"), key=lambda p: p["lim"]),
            "first_present": on[0], "same_night": [p for p in on if p["date_utc"][:10] == on[0]["date_utc"][:10]],
            "control_S_p99": st["control_S_p99"], "control_S_max": st["control_S_max"],
            "note": "cutout significance S (10 arcsec core vs 100-200 arcsec annulus); the DASCH catalogue logs these "
                    "plates as non-detections. Feb 11 image strength matches the B = 8.0 plate of Feb 26."}
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    # the light curve the public page plots (site/src/data/ is tracked)
    stamps = ROOT / "docs" / "results" / "v347nor_stamps.json"  # from v347nor_stamps.py, if run
    site = {"audit": out, "stamps": json.loads(stamps.read_text(encoding="utf-8")) if stamps.exists() else None,
            "det": [[round(float(a), 4), round(float(b), 2)] for a, b in zip(t[det], mag[det])],
            "lim": [[round(float(a), 4), round(float(b), 2)] for a, b in zip(t[~det], lim[~det])
                    if np.isfinite(b)]}
    sc = RN / "v347nor" / "stamps_S.csv"
    if sc.exists() and site["stamps"]:
        C = pd.read_csv(sc)
        C = C[(C.S > site["stamps"]["control_S_p99"]) & C.set.isin(["eruption", "pre"])]
        jd = pd.to_datetime(C.date_utc).map(lambda d: d.to_julian_date()).values
        site["cut"] = sorted(round(float(2000 + (j - 2451545.0) / 365.25), 4) for j in jd)
    SITE.write_text(json.dumps(site, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
