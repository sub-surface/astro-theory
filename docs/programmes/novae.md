# P2 · Accreting white dwarfs & recurrent novae

**Status**: active · **Lead objects**: T CrB, plus the symbiotic recurrent-nova family (RS Oph, V3890 Sgr, V745 Sco)
**Code**: [`celestrium/novae/`](../../celestrium/novae/) · [`scripts/novae/`](../../scripts/novae/) · **Data**: `data/tcrb/`, `data/srn/<slug>/`
**Ledger**: EXP-2026-Y in [`../ledger.md`](../ledger.md) · **Results page**: https://claude.ai/artifact/54hXuNBCsx7wbyayrjCBYf
**Dated tests**: `python -m celestrium calendar` (entries in [`celestrium/watchlist.py`](../../celestrium/watchlist.py))

## Question

How do we see a recurrent nova coming? T CrB (eruptions 1866, 1946; next expected 2026–2030) is the test case:
what state is its accretion flow in, and which pre-eruption signatures are real? The longer-range aim is a
calibrated "who's next" list across the Galaxy's symbiotic stars, tested backwards against a century of plates.

## Material results (EXP-2026-Y, real data)

- **September 2026 UV collapse.** Swift UVM2 fell 2.05 mag in 30 days (12.12 → 14.17), back to the pre-2015 quiescent level, while 2–10 keV X-rays rose. Unreported as of 2026-10-14.
- **Not a supply cut.** A viscous disk with its supply switched off fades at most 0.11 mag in 30 days; the fastest 2-mag fade takes 288 days. The inner disk changed state.
- **Inner-edge history** (robust across inversion variants): ~3 R☉ in quiescence → 0.75–1 R☉ through the 2016–22 high state → 2.7 R☉ in the 2023 dip and again now. Absolute Ṁ is **not** identified (brightness–truncation degeneracy; nebular continuum missing).
- **Inclination**: quiescent ellipsoidal half-amplitude 0.182 ± 0.011 mag needs M-giant TiO temperature sensitivity; implies i ≥ 59–76° (lower limit).
- **TESS flicker**: no significant periodicity in S24/25/51/78 (60 red-noise simulations per sector; the 300-sim run was interrupted).
- **Family**: in quiescence RS Oph and V745 Sco are ~2,800× more UV-luminous than T CrB (±0.4–0.8 dex); T CrB's high state reached ~60× its own quiescence.
- **1946 overlay**: the 2023 alignment is falsified; the Sep 2026 alignment implies ~2027.4.

### Literature matrix (from the 2026-10-13 review; curated values with bibcodes in `celestrium/novae/literature.py`)

| Question | Literature position | Key references | This programme |
| :--- | :--- | :--- | :--- |
| White-dwarf mass | 1.29–1.37 M☉ | Stanishev+04; Hinkle+25; Munari+25; Baptista+26 | adopted 1.35 |
| Inclination | 55–70°, method-dependent | Tatarnikova 13; Munari+25; Baptista+26 | i ≥ 59–76° (ellipsoidal, TiO-corrected) |
| High-state driver | enhanced mass transfer through a large disk | Planquart+25; Schlindwein+25; Baptista+26 (vs Iłkiewicz+23) | inner-edge migration; Ṁ not identified |
| UV vs X-ray | anticorrelated | Luna+18; Pei+25; Luna+26 | ρ = −0.49 over 366 same-day pairs (p not autocorrelation-corrected) |
| 2023 dip = 1945 dip? | no | Merc+25 | confirmed (alignment falsified) |
| Dust in the dips | disfavoured | Luna+26; Evans+19 | 30-day collapse too fast for a disk drain → state change |
| ~6000 s periodicity | disputed | Luna+26 | not significant in TESS |
| Eruption date | 15 published windows, 5 lapsed | Schaefer 23a,b; Schneider 24; Pei+26; Sello+26 | recurrence clock median 2028.1 (68%: 2027.1–2030.9) |
| Super-remnant | ~30 pc nebula, optically thin | Shara+24 | eruption-day test in the calendar |

## Rerun / next queue

1. **UV-recovery test** (pre-registered, 5–30 Nov 2026): orbit-locked obscuration vs accretion-state change.
2. **Inversion v4**: add a nebular (Balmer) continuum component and an X-ray-luminosity energy budget; re-test whether Ṁ becomes identifiable (fuel-budget question vs Schaefer 2025's 7.4×10⁻⁴ M☉ ejecta).
3. **Autocorrelation-aware UV/X test** (block bootstrap over visits).
4. **DASCH plates**: real photometry of the 1930s–40s pre-eruption dip.
5. **"Who's next?"**: recurrence-time predictions for known symbiotics (Gaia distances + Swift UV + nova grids), validated retrospectively on DASCH and forward with Gaia DR4 epoch photometry (~Dec 2026). Run an ADS novelty check first.
6. **TESS flicker census** of symbiotics and accreting WDs (extends `tcrb_tess_flicker.py`).
7. Eruption playbook (calendar `tcrb-eruption-playbook`).

## Open questions

Abrupt state changes (thermal switch vs truncation vs in-binary obscuration); why T CrB is the UV-faint family member;
the fuel budget; scoring the published forecasts after the eruption.
