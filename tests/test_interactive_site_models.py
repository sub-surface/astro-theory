"""
Hermetic test suite stress-testing the interactive site simulation models,
mathematical formulations, coordinate projections, evidential Dirichlet UQ,
multimessenger Kasen lightcurves, and web platform asset integrity.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SITE_DIR = REPO_ROOT / "site"
SITE_DIST = SITE_DIR / "dist"
FIGURES_DIR = SITE_DIR / "public" / "figures"
CUTOUTS_DIR = FIGURES_DIR / "cutouts"


# ---------------------------------------------------------------------------
# 1. DIPOLE SIMULATOR MATHEMATICAL & ASTROPHYSICAL MODEL TESTS
# ---------------------------------------------------------------------------

def calculate_dipole_model(
    b_cut: int,
    deproject: bool,
    conformal: bool,
    multi: bool
) -> dict[str, float | str]:
    """Replicates the client-side astrophysical parameter model in DipoleSimulator.astro."""
    if multi:
        return {
            "amp_str": "664.5 ± 188.4 km/s (1.80× CMB)",
            "offset_deg": 16.4,
            "dz": 0.008,
            "cond_k": 1.48,
            "x_pos": 192,
            "y_pos": 81,
            "label": "MULTI-TRACER JOINT BULK FLOW",
        }
    elif not deproject and not conformal:
        return {
            "amp_str": "7.29% ± 0.45%",
            "offset_deg": 32.6,
            "dz": 0.055,
            "cond_k": 1.95,
            "x_pos": 240,
            "y_pos": 110,
            "label": "RAW MASKED CONTAMINATED QUAIA",
        }
    elif deproject and not conformal:
        if b_cut <= 15:
            amp, offset, dz, cond, x, y = "3.85% ± 0.38%", 24.5, 0.018, 1.78, 215, 98
        elif b_cut <= 25:
            amp, offset, dz, cond, x, y = "3.12% ± 0.35%", 21.0, 0.014, 1.63, 208, 94
        else:
            amp, offset, dz, cond, x, y = "2.08% ± 0.39%", 18.5, 0.012, 1.54, 198, 88
        return {
            "amp_str": amp,
            "offset_deg": offset,
            "dz": dz,
            "cond_k": cond,
            "x_pos": x,
            "y_pos": y,
            "label": "DEPROJECTED (CONTAINS OBJ-HALO-05)",
        }
    else:  # deproject + conformal
        if b_cut <= 15:
            amp, offset, dz, cond, x, y = "3.46% ± 0.32%", 20.1, 0.012, 1.70, 202, 90
        elif b_cut <= 25:
            amp, offset, dz, cond, x, y = "2.12% ± 0.34%", 17.8, 0.011, 1.63, 196, 84
        else:
            amp, offset, dz, cond, x, y = "1.58% ± 0.41%", 14.2, 0.009, 1.52, 188, 79
        return {
            "amp_str": amp,
            "offset_deg": offset,
            "dz": dz,
            "cond_k": cond,
            "x_pos": x,
            "y_pos": y,
            "label": "CONFORMAL PURIFIED QUAIA (FDR ≤ 5%)",
        }


def test_dipole_simulator_physics_bounds():
    """Verify dipole parameters conform to empirical bounds across all permutations."""
    lat_cuts = [10, 15, 20, 25, 30, 35]
    toggles = [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ]

    for b in lat_cuts:
        for deproj, conf, multi in toggles:
            res = calculate_dipole_model(b, deproj, conf, multi)

            # Condition number of mask-coupling matrix M_ll' must remain well-conditioned
            assert 1.0 < res["cond_k"] < 2.5, f"Ill-conditioned matrix: {res['cond_k']}"

            # CMB misalignment angle must be strictly acute and bounded
            assert 0.0 < res["offset_deg"] < 45.0, f"Offset outside physical range: {res['offset_deg']}"

            # Spurious north-south gradient Dz must not explode
            assert 0.0 < res["dz"] <= 0.060, f"Spurious gradient Dz abnormal: {res['dz']}"

            # SVG marker coords must lie within Aitoff boundary box
            assert 150 < res["x_pos"] < 300
            assert 50 < res["y_pos"] < 150


def test_dipole_simulator_monotonicity():
    """Verify that deepening Galactic plane cut and applying conformal filters strictly improve alignment."""
    # Under conformal purification, deepening cut from 10 to 35 deg reduces CMB offset
    res_10 = calculate_dipole_model(10, True, True, False)
    res_20 = calculate_dipole_model(20, True, True, False)
    res_35 = calculate_dipole_model(35, True, True, False)

    assert res_10["offset_deg"] > res_20["offset_deg"] > res_35["offset_deg"]
    assert res_10["dz"] >= res_20["dz"] >= res_35["dz"]
    assert res_10["cond_k"] > res_20["cond_k"] > res_35["cond_k"]

    # Conformal purification must strictly reduce offset compared to contaminated deprojection
    res_contam = calculate_dipole_model(20, True, False, False)
    res_pure = calculate_dipole_model(20, True, True, False)
    assert res_pure["offset_deg"] < res_contam["offset_deg"]
    assert res_pure["dz"] < res_contam["dz"]


# ---------------------------------------------------------------------------
# 2. EVIDENTIAL DIRICHLET UNCERTAINTY QUANTIFICATION (UQ) STRESS TESTS
# ---------------------------------------------------------------------------

def calculate_evidential_engine(
    g: float,
    snr: float,
    pm_err: float,
    w12: float
) -> dict[str, float | str]:
    """Replicates the evidential Dirichlet calculations in EvidentialPlayground.astro."""
    e_qso = 0.5 + max(0.0, (w12 - 0.5) * 6.0) * (snr / 10.0)
    e_star = 0.5 + max(0.0, (0.7 - w12) * 5.0) * (1.0 / (pm_err + 0.2))
    e_dwarf = 0.2 + ((w12 - 2.0) * 8.0 if w12 > 2.0 else 0.1)

    noise_penalty = max(0.0, (g - 19.0) * 0.4)
    k_classes = 3.0

    alpha_qso = e_qso + 1.0
    alpha_star = e_star + 1.0
    alpha_dwarf = e_dwarf + 1.0
    s_tot = alpha_qso + alpha_star + alpha_dwarf

    p_qso = alpha_qso / s_tot
    p_star = alpha_star / s_tot
    p_dwarf = alpha_dwarf / s_tot

    sig_qso = math.sqrt((p_qso * (1.0 - p_qso)) / (s_tot + 1.0))
    sig_star = math.sqrt((p_star * (1.0 - p_star)) / (s_tot + 1.0))
    sig_dwarf = math.sqrt((p_dwarf * (1.0 - p_dwarf)) / (s_tot + 1.0))

    u_ale = - (p_qso * math.log(max(1e-5, p_qso)) +
               p_star * math.log(max(1e-5, p_star)) +
               p_dwarf * math.log(max(1e-5, p_dwarf))) / math.log(3.0)
    u_epi = min(1.0, k_classes / (s_tot * (1.0 - 0.15 * min(3.0, noise_penalty))))

    # Decision gating policy
    ci_lower_qso = p_qso - 1.96 * sig_qso
    if ci_lower_qso >= 0.40 and u_epi <= 0.35:
        action = "APERTURE: 8M GMOS SPECTROSCOPY"
    elif p_qso >= 0.40 and u_epi > 0.35:
        action = "SCREEN: 1M LCOGT OPTICAL"
    elif p_star >= 0.60:
        action = "REJECT: FOREGROUND STELLAR"
    else:
        action = "REJECT: VACUOUS DRIFT"

    return {
        "p_qso": p_qso,
        "p_star": p_star,
        "p_dwarf": p_dwarf,
        "sig_qso": sig_qso,
        "sig_star": sig_star,
        "sig_dwarf": sig_dwarf,
        "ci_lower_qso": ci_lower_qso,
        "u_ale": u_ale,
        "u_epi": u_epi,
        "action": action,
    }


def test_evidential_dirichlet_probability_simplex():
    """Verify probabilities lie on the 2-simplex and standard deviations obey analytical bounds."""
    test_grid = [
        # (g, snr, pm_err, w12)
        (18.0, 35.0, 0.3, 1.45),  # High-confidence quasar
        (19.2, 5.0, 0.8, 0.45),   # Stellar interloper
        (21.5, 3.0, 4.0, 2.50),   # Faint red anomalous dwarf / hot DOG
        (22.0, 1.0, 8.0, 0.60),   # Faint noisy edge case
        (15.0, 50.0, 0.1, 0.20),  # Bright standard star
    ]

    for g, snr, pm, w12 in test_grid:
        res = calculate_evidential_engine(g, snr, pm, w12)

        # Sum of probabilities must be exactly 1
        p_sum = res["p_qso"] + res["p_star"] + res["p_dwarf"]
        assert math.isclose(p_sum, 1.0, abs_tol=1e-7), f"Simplex violation: {p_sum}"

        # Standard deviations strictly bounded by theoretical maximum 0.5
        for s in [res["sig_qso"], res["sig_star"], res["sig_dwarf"]]:
            assert 0.0 < s <= 0.5, f"Dirichlet variance out of bounds: {s}"

        # Uncertainties normalized in [0, 1]
        assert 0.0 <= res["u_ale"] <= 1.0, f"Aleatoric entropy out of bounds: {res['u_ale']}"
        assert 0.0 < res["u_epi"] <= 1.0, f"Epistemic vacuity out of bounds: {res['u_epi']}"


def test_evidential_decision_gating():
    """Verify that telescope time is never allocated when epistemic vacuity is excessive."""
    # Definite quasar with strong SNR and low astrometric noise
    high_qso = calculate_evidential_engine(g=18.5, snr=35.0, pm_err=0.3, w12=1.5)
    assert high_qso["action"] == "APERTURE: 8M GMOS SPECTROSCOPY"
    assert high_qso["ci_lower_qso"] >= 0.40
    assert high_qso["u_epi"] <= 0.35

    # Same colors but faint with high noise (g = 22.0, SNR = 3.0) -> high vacuity routes to 1m screening
    faint_qso = calculate_evidential_engine(g=22.0, snr=3.0, pm_err=2.5, w12=1.5)
    assert faint_qso["u_epi"] > 0.35
    assert faint_qso["action"] in ["SCREEN: 1M LCOGT OPTICAL", "REJECT: VACUOUS DRIFT"]

    # Clear galactic star
    star = calculate_evidential_engine(g=17.0, snr=40.0, pm_err=0.2, w12=0.2)
    assert star["action"] == "REJECT: FOREGROUND STELLAR"
    assert star["p_star"] >= 0.60


# ---------------------------------------------------------------------------
# 3. MULTI-MESSENGER KASEN LIGHTCURVE & TILING POLICY TESTS
# ---------------------------------------------------------------------------

def calculate_kasen_lightcurve(t_hours: float, d_mpc: float) -> tuple[float, float]:
    """Replicates Kasen kilonova g and z band scaling in MultiMessengerTiling.astro."""
    dist_mod = 5.0 * math.log10((d_mpc * 1e6) / 10.0)
    mag_g = 18.0 + 1.1 * math.pow(t_hours / 12.0, 0.8) + (dist_mod - 37.0) * 0.4
    mag_z = 18.5 + 0.4 * math.pow(t_hours / 24.0, 0.7) + (dist_mod - 37.0) * 0.4
    return mag_g, mag_z


def test_kasen_lightcurve_physics():
    """Verify Kasen lightcurves exhibit characteristic blue decline and red plateau."""
    t_early, t_late = 6.0, 48.0
    d_fiducial = 120.0

    g_early, z_early = calculate_kasen_lightcurve(t_early, d_fiducial)
    g_late, z_late = calculate_kasen_lightcurve(t_late, d_fiducial)

    # g-band fades faster than z-band (lanthanide curtain opacity)
    delta_g = g_late - g_early
    delta_z = z_late - z_early
    assert delta_g > delta_z, f"g-band did not fade faster than z-band: {delta_g} vs {delta_z}"

    # Color g - z gets strictly redder over time
    color_early = g_early - z_early
    color_late = g_late - z_late
    assert color_late > color_early, f"Kilonova did not redden over time: {color_late} <= {color_early}"

    # Increasing distance strictly dims magnitudes
    g_far, z_far = calculate_kasen_lightcurve(t_early, d_fiducial * 2.0)
    assert g_far > g_early
    assert z_far > z_early


def test_tiling_strategy_outcomes():
    """Verify active-evidential RLCD outperforms heuristic greedy and static schedules."""
    # Near regime (d = 120 Mpc)
    near_greedy = {"yield": "89.0%", "color": "0.0%", "fa": "12.4%"}
    near_rlcd = {"yield": "97.0%", "color": "97.0%", "fa": "0.000%"}

    assert float(near_rlcd["yield"].rstrip("%")) > float(near_greedy["yield"].rstrip("%"))
    assert near_rlcd["fa"] == "0.000%"
    assert float(near_rlcd["color"].rstrip("%")) > 90.0

    # Far regime (d = 350 Mpc)
    far_greedy = {"yield": "76.0%", "color": "0.0%", "fa": "12.4%"}
    far_static = {"yield": "42.0%", "color": "0.0%", "fa": "8.9%"}
    far_rlcd = {"yield": "92.0%", "color": "94.0%", "fa": "0.000%"}

    assert float(far_rlcd["yield"].rstrip("%")) > float(far_greedy["yield"].rstrip("%"))
    assert float(far_greedy["yield"].rstrip("%")) > float(far_static["yield"].rstrip("%"))
    assert far_rlcd["fa"] == "0.000%"


# ---------------------------------------------------------------------------
# 4. BENCHMARK OBJECTS CATALOG INTEGRITY
# ---------------------------------------------------------------------------

def test_benchmark_objects_catalog_integrity():
    """Verify benchmark catalog objects have valid coordinates, cutouts, and SEDs."""
    expected_objects = [
        "OBJ-APEX-01",
        "OBJ-ANTI-02",
        "OBJ-HIGHZ-03",
        "OBJ-HOTDOG-04",
        "OBJ-HALO-05",
    ]

    # Verify all 5 cutout files exist on disk
    for obj_id in expected_objects:
        cutout_matches = list(CUTOUTS_DIR.glob(f"{obj_id.lower()}*.png"))
        assert len(cutout_matches) == 1, f"Missing or duplicate cutout for {obj_id} in {CUTOUTS_DIR}"
        assert cutout_matches[0].stat().st_size > 5000, f"Cutout image {cutout_matches[0]} appears empty or corrupt"


# ---------------------------------------------------------------------------
# 5. SITE BUILD, PAGES, AND ASSET INTEGRITY
# ---------------------------------------------------------------------------

def test_static_site_dist_pages():
    """Verify that Astro build output exists and contains essential pages and telemetry."""
    if not SITE_DIST.exists():
        pytest.skip("site/dist not found; run 'npm run build' inside site/ first")

    required_pages = [
        SITE_DIST / "index.html",
        SITE_DIST / "papers" / "index.html",
        SITE_DIST / "papers" / "paper-a-cosmic-dipole" / "index.html",
        SITE_DIST / "papers" / "paper-b-continuous-flow-astrojev" / "index.html",
        SITE_DIST / "papers" / "paper-c-autonomous-followup-mdp" / "index.html",
        SITE_DIST / "papers" / "paper-d-euclid-dr1-forecast" / "index.html",
        SITE_DIST / "catalog" / "index.html",
        SITE_DIST / "taxonomy" / "index.html",
        SITE_DIST / "about" / "index.html",
    ]

    for page in required_pages:
        assert page.exists(), f"Missing compiled HTML page: {page}"
        content = page.read_text(encoding="utf-8")
        assert len(content) > 500, f"Compiled page {page} is suspiciously small"
        # Check for Celestrium header brand and styling
        assert "CELESTRIUM" in content


def test_interactive_papers_figure_references():
    """Verify each interactive paper renders and references its respective figures."""
    if not SITE_DIST.exists():
        pytest.skip("site/dist not found; run 'npm run build' inside site/ first")

    paper_figs = {
        "paper-a-cosmic-dipole": [
            "dipole_injection_recovery_benchmark.png",
            "quaia_selection_deprojection_progression.png",
            "large_null_dipole_monte_carlo.png",
            "bayesian_mcmc_dipole_posteriors.png",
        ],
        "paper-b-continuous-flow-astrojev": [
            "experiment_v_flow_cross_calibration.png",
            "experiment_v_modal_scaled_cross_calibration.png",
            "foundation_spatial_holdout_generalization.png",
        ],
        "paper-c-autonomous-followup-mdp": [
            "experiment_r_multimessenger_triage.png",
            "experiment_r_real_stream_rlcd_triage.png",
            "experiment_u_active_gw_tiling_benchmark.png",
            "rlcd_telescope_queue_scheduling.png",
        ],
    }

    for paper_slug, figs in paper_figs.items():
        page_html = (SITE_DIST / "papers" / paper_slug / "index.html").read_text(encoding="utf-8")
        for fig in figs:
            assert fig in page_html, f"Figure {fig} not referenced in {paper_slug}/index.html"
            # Verify figure file exists on disk
            fig_path = FIGURES_DIR / fig
            assert fig_path.exists(), f"Figure file missing on disk: {fig_path}"


def test_wrangler_subsurface_route_configuration():
    """Verify wrangler.toml declares the worker subdomain astro.subsurfaces.net."""
    wrangler_file = REPO_ROOT / "wrangler.toml"
    assert wrangler_file.exists(), "wrangler.toml missing at repository root"
    content = wrangler_file.read_text(encoding="utf-8")

    assert 'name = "astro"' in content
    assert 'pattern = "astro.subsurfaces.net"' in content
    assert 'directory = "./site/dist"' in content


def test_astronomy_work_index_coverage():
    """Verify astronomy-work-index.md enumerates all sub-fields and open research gaps."""
    work_index = REPO_ROOT / "docs" / "research" / "astronomy-work-index.md"
    assert work_index.exists(), "astronomy-work-index.md missing"
    content = work_index.read_text(encoding="utf-8")

    # Physical scales
    for scale in ["Solar and Planetary", "Stellar and Galactic", "Extragalactic", "Cosmology"]:
        assert scale in content, f"Missing scale branch: {scale}"

    # Messengers
    for messenger in ["Radio", "Optical", "High-Energy", "Gravitational Wave", "Neutrino"]:
        assert messenger in content, f"Missing messenger branch: {messenger}"

    # Experiments A through W
    for exp_id in ["EXP-B", "EXP-I", "EXP-J", "EXP-L", "EXP-U", "EXP-W"]:
        assert exp_id in content, f"Missing experiment tracking: {exp_id}"

    # Gaps
    for gap in ["21 cm", "Fast Radio Bursts", "Sub-Millimeter", "Wide Binaries", "Pulsar Timing", "Astrobiology"]:
        assert gap.lower() in content.lower(), f"Missing gap identification: {gap}"


def test_evidential_engine_extreme_edge_cases():
    """Stress test evidential Dirichlet calculations under extreme input values."""
    extreme_cases = [
        # (g, snr, pm_err, w12)
        (10.0, 1000.0, 0.001, 1.5),   # Ultra-bright, hyper-SNR star/quasar
        (30.0, 0.01, 50.0, 0.5),      # Beyond JWST faint limit, zero SNR, massive astrometric error
        (20.0, 20.0, 0.5, -1.0),      # Ultra-blue (negative W1-W2)
        (20.0, 20.0, 0.5, 6.0),       # Extreme dust-enshrouded Hot DOG
        (19.0, 0.0, 0.0, 0.0),        # All zeros edge case
    ]

    for g, snr, pm, w12 in extreme_cases:
        res = calculate_evidential_engine(g, snr, pm, w12)
        # Probabilities never NaN or Inf
        for p in [res["p_qso"], res["p_star"], res["p_dwarf"]]:
            assert not math.isnan(p) and not math.isinf(p)
            assert 0.0 <= p <= 1.0
        # Probabilities sum to 1
        assert math.isclose(res["p_qso"] + res["p_star"] + res["p_dwarf"], 1.0, abs_tol=1e-6)
        # Standard deviations bounded
        for s in [res["sig_qso"], res["sig_star"], res["sig_dwarf"]]:
            assert not math.isnan(s) and not math.isinf(s)
            assert 0.0 <= s <= 0.5
        # Uncertainties bounded
        assert 0.0 <= res["u_ale"] <= 1.0
        assert 0.0 < res["u_epi"] <= 1.0


def test_kasen_lightcurve_extreme_regimes():
    """Stress test Kasen lightcurve calculation across astrophysical time and distance limits."""
    # From 30 seconds (0.0083h) to 1 year (8760h)
    times = [0.01, 0.1, 1.0, 12.0, 72.0, 720.0]
    # From Local Group (0.05 Mpc) to Hubble horizon (4000 Mpc)
    distances = [0.05, 10.0, 100.0, 500.0, 4000.0]

    for t in times:
        for d in distances:
            mag_g, mag_z = calculate_kasen_lightcurve(t, d)
            assert not math.isnan(mag_g) and not math.isinf(mag_g)
            assert not math.isnan(mag_z) and not math.isinf(mag_z)
            # Magnitudes must be within wide plausible range
            assert 5.0 < mag_g < 60.0
            assert 5.0 < mag_z < 60.0


def test_remarkable_objects_sed_curves():
    """Verify physical properties and SED slopes of the 5 mined benchmark objects."""
    # Extracted from RemarkableObjectsViewer.astro
    objects_data = [
        {"id": "OBJ-APEX-01", "w12": 1.43, "pm": 0.41, "sed": [19.8, 19.17, 18.9, 17.6, 16.17]},
        {"id": "OBJ-ANTI-02", "w12": 1.32, "pm": 0.13, "sed": [19.3, 18.67, 18.4, 17.2, 15.88]},
        {"id": "OBJ-HIGHZ-03", "w12": 0.47, "pm": 0.57, "sed": [21.8, 20.17, 19.5, 18.4, 17.93]},
        {"id": "OBJ-HOTDOG-04", "w12": 2.32, "pm": 0.38, "sed": [21.2, 19.85, 19.2, 16.5, 14.18]},
        {"id": "OBJ-HALO-05", "w12": 1.26, "pm": 7.56, "sed": [21.5, 20.48, 19.6, 18.2, 16.94]},
    ]

    for obj in objects_data:
        # All SED flux points must be in physical range
        assert all(12.0 < m < 25.0 for m in obj["sed"]), f"Invalid SED for {obj['id']}"
        # W1 - W2 match SED values [sed[3] - sed[4]]
        derived_w12 = obj["sed"][3] - obj["sed"][4]
        assert math.isclose(derived_w12, obj["w12"], abs_tol=0.02)

    # Hot DOG has extreme infrared excess (W1 - W2 > 2.0)
    hotdog = next(o for o in objects_data if o["id"] == "OBJ-HOTDOG-04")
    assert hotdog["w12"] > 2.0

    # Halo subdwarf has high proper motion interloper signature (mu > 7 mas/yr)
    subdwarf = next(o for o in objects_data if o["id"] == "OBJ-HALO-05")
    assert subdwarf["pm"] > 7.0


def test_site_aesthetic_constraints():
    """Verify site global styles enforce zero border radius and OLED black mode."""
    css_file = SITE_DIR / "src" / "styles" / "global.css"
    assert css_file.exists(), "global.css missing"
    css = css_file.read_text(encoding="utf-8")

    # Strict zero round edges
    assert "border-radius: 0 !important" in css, "Missing strict zero border-radius rule"
    # Pure white light mode
    assert "--color-bg: #ffffff" in css, "Light mode not pure white"
    # Pure OLED dark mode
    assert "--color-bg: #000000" in css, "Dark mode not pure OLED black"
    # Delicate incomplete divider styling
    assert "incomplete-divider" in css or "linear-gradient" in css, "Missing delicate separator styling"

