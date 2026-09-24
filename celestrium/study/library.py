"""Built-in studies — the pipelines. The *claims* live in the vault.

Each id here matches an `Astronomy/Observables/*.md` note's `id:` frontmatter,
so `celestrium study list` shows both halves of the same research object and
`vault sync` can write results back to the note that motivated them.

A study with an empty pipeline is legitimate: it's an idea that isn't scoped
yet. That's the vault's own discipline — "an observable you cannot refute is
not an observable" — and it shows up here as `status: idea`.
"""
from __future__ import annotations

from .model import P, Ref, Study, step

# CatWISE-style colour-selected AGN. Row-capped by design: the desk version
# asks "does the pipeline reproduce the published amplitude on the sky I can
# afford to pull", and `row_limit` is the honest cost knob.
_CATWISE_ADQL = (
    "SELECT TOP {row_limit} ra, dec, w1mpro, w2mpro "
    "FROM catwise_2020 "
    "WHERE w1mpro < {flux_limit} AND w1mpro - w2mpro > 0.8 "
    "AND w1sigmpro < 0.2"
)

STUDIES: dict = {}


def _register(study: Study) -> Study:
    STUDIES[study.id] = study
    return study


_register(Study(
    id="quasar-number-count-dipole",
    title="The quasar number-count dipole",
    claim="Number counts of distant sources show a dipole roughly twice the "
          "kinematic prediction from the CMB dipole, aligned in direction.",
    signature="A number-count dipole significantly larger than the kinematic "
              "prediction, aligned with it, and robust to flux-limit and "
              "masking choices.",
    refutation="The excess disappearing under a properly modelled selection "
               "function, flux calibration gradient, or Galactic-plane mask.",
    archives=("CatWISE", "Quaia", "NVSS"),
    rows="~1500000", leverage="high", status="scoped",
    pipeline=(
        step("archive.query", archive="irsa", adql=_CATWISE_ADQL),
        step("analysis.sky_density", table=Ref("prev"), nside=64, frame="galactic"),
        step("analysis.mask", density=Ref("prev"),
             gal_lat_min=P("mask_width"), min_count=0),
        step("analysis.dipole_fit", density=Ref("prev"), into="fit"),
    ),
    grid={
        "row_limit": [200000],
        "flux_limit": [15.0, 15.5, 16.0],
        "mask_width": [10.0, 15.0, 20.0],
    },
    metric="amplitude",
    metrics=("sigma_amplitude", "significance", "shot_noise_amplitude",
             "direction.l", "direction.b", "n_sources", "sky_fraction"),
    note="Report the amplitude as a function of the cuts, not as a number. "
         "Pair every run with analysis.null_shuffle before believing it.",
))


_register(Study(
    id="euclid-dr1-dipole-forecast",
    title="Euclid DR1 σ_D forecast — measure or rehearse?",
    claim="A ~1900 deg² Euclid DR1 wide footprint yields a number-count dipole "
          "uncertainty small enough to test the CatWISE/Quaia excess.",
    signature="Recovered σ_D comfortably below the ~2× excess amplitude, on a "
              "footprint with a selection function independent of WISE/Gaia.",
    refutation="σ_D at or above the claimed excess — in which case DR1 is a "
               "rehearsal, not a measurement, and we say so before it lands.",
    archives=("Euclid DR1 (synthetic footprint)",),
    rows="synthetic", leverage="high", status="scoped",
    pipeline=(
        step("analysis.synthetic_sky", nsources=P("nsources"),
             amplitude=P("injected"), sky_fraction=P("sky_fraction"),
             gal_lat_min=25.0, seed=P("seed")),
        step("analysis.sky_density", table=Ref("prev"), nside=32, frame="galactic"),
        step("analysis.dipole_fit", density=Ref("prev"), into="fit"),
    ),
    grid={
        # ~1900 deg² ≈ 4.6% of the sky; source density scaled to a DR1-like depth.
        "sky_fraction": [0.046],
        "nsources": [500000, 1000000, 2000000],
        "injected": [0.0, 0.007, 0.014],
        "seed": [1, 2, 3],
    },
    metric="amplitude",
    metrics=("sigma_amplitude", "shot_noise_amplitude", "significance",
             "direction.l", "direction.b", "n_sources"),
    note="Offline and free — this is the 'measure vs rehearse' decision from "
         "docs/research/euclid-dr1-prep.md, answerable before DR1 lands "
         "(21 Oct 2026). Compare against analysis.kinematic_dipole.",
))


_register(Study(
    id="euclid-dr1-analytic-forecast",
    title="Euclid DR1 analytic σ_D & harmonic leakage gate decision",
    claim="Partial-sky harmonic leakage dominates over Poisson shot noise on the ~1900 deg² footprint.",
    signature="σ_leak substantially exceeds σ_shot, setting the definitive gate decision.",
    refutation="σ_leak negligible relative to shot noise.",
    archives=("Euclid DR1 wide",),
    rows="analytic", leverage="high", status="scoped",
    pipeline=(
        step("analysis.dr1_forecast", area_deg2=1900.0,
             density_arcmin2=P("density"), nside=32,
             d_anom=P("d_anom"), d_kin=0.0047,
             c2_clustering=P("c2"), into="forecast"),
    ),
    grid={
        "density": [0.1, 1.0, 30.0],
        "d_anom": [0.010, 0.012, 0.014],
        "c2": [1e-5, 5e-5, 1e-4],
    },
    metric="snr",
    metrics=("verdict", "sigma_total", "sigma_shot", "sigma_leak", "condition_number"),
    note="Direct gate decision from forecast.py without simulation overhead.",
))



_register(Study(
    # id matches `Observables/Wide binaries as a test of gravity.md` — the two
    # halves of one research object bind by id, they do not duplicate.
    id="wide-binaries-gravity-test",
    title="Wide binaries as a test of gravity",
    claim="Wide binary orbital velocities depart from Newtonian expectation "
          "below the MOND acceleration scale.",
    signature="A systematic upward departure in relative velocity at "
              "separations beyond ~5 kAU that survives contamination cuts.",
    refutation="The departure vanishing once line-of-sight contamination, "
               "hidden tertiaries and Gaia astrometric quality cuts are applied.",
    archives=("Gaia DR3",), rows="~500000", leverage="medium", status="idea",
    note="Claim-only until the contamination model is scoped — the hard part "
         "is not the query, it is the tertiary rejection.",
))


def get(study_id: str) -> Study:
    if study_id not in STUDIES:
        raise KeyError(f"unknown study {study_id!r}; "
                       f"have {', '.join(sorted(STUDIES))}")
    return STUDIES[study_id]


def names() -> list:
    return sorted(STUDIES)


def merge_vault(vault_studies: dict) -> dict:
    """Overlay claim metadata read from the vault onto the code pipelines.

    The vault owns the claim (signature/refutation/status/verdict/leverage);
    code owns the pipeline. Where both exist they are the same object, joined
    by id — so neither side has to duplicate the other.
    """
    from dataclasses import replace
    merged = dict(STUDIES)
    for study_id, fields in (vault_studies or {}).items():
        base = merged.get(study_id)
        keep = {k: v for k, v in fields.items()
                if v not in (None, "", (), []) and hasattr(Study, "__dataclass_fields__")
                and k in Study.__dataclass_fields__}
        keep.pop("pipeline", None)
        keep.pop("grid", None)
        if base is None:
            merged[study_id] = Study(id=study_id, **keep)
        else:
            merged[study_id] = replace(base, **keep)
    return merged
