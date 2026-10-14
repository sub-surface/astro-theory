"""Curated quantitative literature on T CrB (and its symbiotic recurrent-nova siblings).

Every entry carries an ADS bibcode (or arXiv id) and the value *as stated in the abstract*;
`made` is the arXiv submission / ATel posting date in decimal years. Entries whose numbers
were not stated explicitly in an abstract are marked with note="from text" and should be
re-checked against the paper before being quoted elsewhere.

Used by scripts/novae/tcrb_literature.py for the prediction scoreboard, the parameter
forest plot and the accretion-budget figure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Prediction:
    who: str
    ref: str          # bibcode / arXiv id
    made: float       # decimal year the forecast was published
    center: Optional[float]
    lo: float
    hi: float
    method: str
    discrete: Tuple[float, ...] = ()   # orbital-phase style forecasts: list of candidate dates


# Eruption-date forecasts (upper bound "hi" of open-ended forecasts set to the stated horizon).
PREDICTIONS: List[Prediction] = [
    Prediction("Knight & Lindley (1946, via Shears 2024)", "2024RNAAS...8..157S", 1946.2, 2026.5, 2026.0, 2027.0,
               "80-yr recurrence (first ever forecast)"),
    Prediction("Luna+ 2020", "2020ApJ...902L..14L", 2020.73, 2026.0, 2023.0, 2029.0, "recurrence + high state"),
    Prediction("Schaefer 2023a", "2023MNRAS.524.3146S", 2023.18, 2025.5, 2024.2, 2026.8, "1946 light-curve template"),
    Prediction("Marchev+ ATel 16214", "2023ATel16214....1M", 2023.62, None, 2023.62, 2024.12, "'next six months'"),
    Prediction("Schaefer 2023b", "2023JHA....54..436S", 2023.65, 2024.4, 2024.1, 2024.7, "pre-eruption dip timing"),
    Prediction("Cheung+ ATel 16336", "2023ATel16336....1C", 2023.88, 2024.2, 2024.0, 2024.5, "'early 2024' (dip onset)"),
    Prediction("Schneider 2024", "2024RNAAS...8..272S", 2024.78, None, 2025.23, 2026.6,
               "orbital phase of past eruptions", discrete=(2025.235, 2025.860, 2026.482, 2027.106)),
    Prediction("Shara+ 2024", "2024ApJ...977L..48S", 2024.92, None, 2024.92, 2027.0, "'between 2024 and 2026'"),
    Prediction("Starrfield+ 2025", "2025ApJ...982...89S", 2025.12, None, 2025.0, 2027.0, "'2025 or 2026'"),
    Prediction("Hinkle+ 2025", "2025ApJ...983...76H", 2025.16, None, 2025.16, 2026.0, "'expected in 2025'"),
    Prediction("Wallace+ 2025", "2025ApJ...991..200W", 2025.67, None, 2025.67, 2027.67, "'within the next 2 yr'"),
    Prediction("Marchev+ ATel 17784", "2026ATel17784....1M", 2026.35, None, 2026.77, 2026.85,
               "'5-6 months if fading continues'"),
    Prediction("Sello+ 2026", "arXiv:2607.05200", 2026.51, 2027.15, 2027.15, 2027.15, "semi-empirical T_rec model"),
    Prediction("Pei+ 2026 (dip-analogue branch)", "2026MNRAS.550g1361P", 2026.54, 2026.95, 2026.9, 2027.0,
               "conditional: renewed decline = 1945 dip"),
    Prediction("Pei+ 2026 (accretion-deficit branch)", "2026MNRAS.550g1361P", 2026.54, None, 2029.4, 2032.0,
               "conditional lower limit 2029 May"),
    Prediction("Aydi+ 2026", "arXiv:2609.06201", 2026.68, 2026.1, 2014.1, 2038.1, "population T_rec = 80 +/- 12 yr"),
]


@dataclass(frozen=True)
class Measurement:
    who: str
    ref: str
    year: float
    value: float
    err_lo: float = 0.0
    err_hi: float = 0.0
    method: str = ""
    note: str = ""


WD_MASS: List[Measurement] = [
    Measurement("Belczynski & Mikolajewska 1998", "1998MNRAS.296...77B", 1998.4, 1.2, 0.0, 0.0, "ellipsoidal + mass fn", "~"),
    Measurement("Hachisu & Kato 1999", "1999ApJ...517L..47H", 1999.2, 1.35, 0.0, 0.0, "nova light-curve model", "~"),
    Measurement("Kato 2002", "2002ASPC..261..595K", 2002.0, 1.37, 0.0, 0.0, "wind light-curve model"),
    Measurement("Stanishev+ 2004", "2004A&A...415..609S", 2004.0, 1.37, 0.13, 0.13, "H-alpha RV, i~67"),
    Measurement("Tatarnikova 2013", "2013IAUS..281..203T", 2013.0, 1.3, 0.0, 1.7, "IR ellipsoidal (1.3-3)", "lower bound"),
    Measurement("Hinkle+ 2025", "2025ApJ...983...76H", 2025.16, 1.37, 0.01, 0.01, "RV orbit + Gaia distance"),
    Measurement("Munari+ 2025", "2025A&A...701A.176M", 2025.58, 1.35, 0.0, 0.0, "orbit + radiative model"),
    Measurement("Baptista+ 2026", "arXiv:2604.02708", 2026.26, 1.29, 0.0, 0.0, "M_ign(T_rec) disk model"),
]

INCLINATION: List[Measurement] = [
    Measurement("Hachisu & Kato 1999", "1999ApJ...517L..47H", 1999.2, 70, 0, 0, "nova light-curve model", "~"),
    Measurement("Stanishev+ 2004", "2004A&A...415..609S", 2004.0, 67, 0, 0, "H-alpha RV", "~"),
    Measurement("Tatarnikova 2013", "2013IAUS..281..203T", 2013.0, 55, 5, 5, "IR ellipsoidal 50-60 (90%)"),
    Measurement("Munari+ 2025", "2025A&A...701A.176M", 2025.58, 61, 0, 0, "orbit + radiative model"),
    Measurement("Baptista+ 2026", "arXiv:2604.02708", 2026.26, 57.3, 0, 0, "M_ign(T_rec) disk model"),
]

# Accreted / ejected masses and accretion rates (Msun, Msun/yr)
ACCRETION: List[Measurement] = [
    Measurement("Zamanov+ 2023: accreted during SAP 2014-2023", "arXiv:2312.04342", 2023.93, 2e-7, 0, 0, "UBV hot-component L"),
    Measurement("Toala+ 2024: mean Mdot_acc during SAP", "arXiv:2405.08980", 2024.37, 2.6e-8, 0, 0, "X-ray BL temperature"),
    Measurement("Schlindwein+ 2025: Mdot_transfer quiescence", "arXiv:2506.05098", 2025.43, 2.0e-9, 0, 0, "disk model of SAP"),
    Measurement("Schlindwein+ 2025: Mdot_transfer high state", "arXiv:2506.05098", 2025.43, 1.9e-7, 0, 0, "disk model of SAP"),
    Measurement("Schaefer 2025: M_accreted per cycle", "arXiv:2510.01587", 2025.75, 1.38e-6, 0, 0, "accretion luminosity"),
    Measurement("Schaefer 2025: M_ejecta 1946", "arXiv:2510.01587", 2025.75, 7.4e-4, 0.9e-4, 0.9e-4, "orbital period jump"),
    Measurement("Zamanov+ 2023: pre-outburst envelope", "arXiv:2308.04104", 2023.6, 6e-7, 5.5e-7, 0, "5e-8 - 6e-7"),
    Measurement("Jose+ 2025: Mdot_acc needed for 80 yr", "2025A&A...698A.251J", 2025.4, 3e-8, 2e-8, 7e-8, "hydro models 1e-8-1e-7"),
]

# Other fixed facts used across figures
DISTANCE_PC = {"Gaia DR3 (Bailer-Jones)": (887, 29, 18), "Norris+ 2026 (CHARA)": (914, 22, 24)}
GIANT_V_AT_MAX = (10.17, 0.06, "2024AcA....74..265Z")  # red giant V at ellipsoidal maximum (Zamanov+ 2024)
DISK_RADIUS_RSUN = {"Munari+ 2025 hot-spot azimuth": 58.0, "Zamanov+ 2024 H-alpha peaks": 89.0}
ORBITAL_PERIOD_D = {"Fekel+ 2000": 227.57, "Hinkle+ 2025": 227.5494, "Munari+ 2025": 227.5528,
                    "Schaefer 2025 pre-1946": 227.4586, "Schaefer 2025 post-1946": 227.6043}

# The symbiotic recurrent nova family (Jamnik+ 2026: "only four such systems are currently known in the Galaxy")
@dataclass(frozen=True)
class SymbioticRN:
    name: str
    ra: float
    dec: float
    eruptions: Tuple[float, ...]
    p_orb_d: float
    m_wd: float
    donor: str
    dist_kpc: float
    ebv: float
    refs: Tuple[str, ...] = field(default_factory=tuple)


FAMILY: List[SymbioticRN] = [
    SymbioticRN("T CrB", 239.8757, 25.9202, (1217.8, 1787.98, 1866.37, 1946.11), 227.55, 1.35, "M3-4 III",
                0.887, 0.07, ("2025A&A...701A.176M",)),
    SymbioticRN("RS Oph", 267.5549, -6.7079, (1898.4, 1933.6, 1958.6, 1967.3, 1985.1, 2006.12, 2021.61), 453.6,
                1.35, "M0-2 III", 2.68, 0.73, ("2002ASPC..261..595K",)),
    SymbioticRN("V3890 Sgr", 277.6804, -24.0191, (1962.5, 1990.3, 2019.65), 747.6, 1.35, "M6.5 III", 9.0, 0.56,
                ("2021MNRAS.504.2122M", "arXiv:2609.01388")),
    SymbioticRN("V745 Sco", 268.8411, -33.2500, (1937.4, 1989.6, 2014.10), 0.0, 1.38, "M7 III", 7.8, 1.0,
                ("arXiv:2609.01388",)),
]
