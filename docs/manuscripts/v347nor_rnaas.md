# DRAFT — not submitted. Needs Leon's approval and, ideally, the DASCH original-plate check first.

Target: Research Notes of the AAS (≤ 1000 words, 1 figure). Figure: `docs/figures/rn/v347nor_1943.png`, with
`v347nor_onset.png` as a possible inset. Numbers: `docs/results/v347nor_audit.json` and `v347nor_stamps.json`.
Search design: EXP-2026-Z, `docs/programmes/novae.md`.

---

## An unrecorded 1943 nova eruption of the symbiotic nova Hen 2-147 (V347 Nor) on the Harvard plates

**Authors:** [Leon — full name, affiliation], [acknowledge Claude per journal AI policy]

### Abstract

Hen 2-147 (V347 Nor) is a D-type symbiotic binary with a 373-day Mira inside an expanding ring nebula. Its
spectrum and nebula have led to it being called a symbiotic nova, but no outburst has been reported. In a
pre-registered search of the DASCH light curves of 1,211 symbiotic and cataclysmic binaries, we find that it
erupted in February 1943:
- **Onset:** absent on February 9 and 10 (to B = 12.9 and 11.7), and a strong image on two plate series on
  February 11;
- **Peak:** B = 8.0 on February 26;
- **Decline:** t2 ≈ 60 d and t3 ≈ 150 d;
- **Uniqueness:** over 1,000 plates from 1890–1942 and 1947–1990 that reach well below the outburst brightness
  show nothing comparable.

At the Mira distance the peak was M_B ≈ −6.3. A day-long rise and an amplitude of at least 9 mag mark a nova,
not a Z And-type outburst. The DASCH catalogue missed most of the eruption plates; image cutouts recover them.

### 1. The search

DASCH (Grindlay et al. 2009) digitised the Harvard plates, which cover the whole sky from 1885 to 1990. Most
recurrent-nova eruptions go unobserved (Pagnotta & Schaefer 2014). We pre-registered a search of the DASCH
light curves of 1,211 known symbiotics and cataclysmic variables:
- Thresholds, controls and vetting were fixed in advance, and amendments were logged.
- Each target was compared with 2,101 matched field stars.
- The statistic is a noise-weighted brightening that must persist for at least two nights.
- The vetting threshold is the highest score among the controls.

Ten targets exceeded the threshold, all symbiotics. Nine were known outbursts, slow activity or a plate
artifact. The tenth was Hen 2-147.

### 2. The eruption

GCVS lists V347 Nor as a Mira and records no eruption. In quiescence it has B ≈ 19.8 (SPM4), G = 12.19 and
BP−RP = 5.77 (Gaia DR3).

The DASCH catalogue first measures it on 1943 February 26.06 UT, at B = 7.99 on plate rb12036. That image is
flagged as saturated, and the catalogue records non-detections on most other plates of 1943. We therefore
measured every plate directly from its DASCH image cutout. For each plate we computed S, the density in a 10″
core relative to the scatter in a 100–200″ annulus. The wide annulus is needed because saturated images on these
patrol plates are about 70″ across. Over 725 plates from the quiet years 1938–42 and 1947–52, S does not
exceed 4.7, and its 99th percentile is 1.9.

The cutouts give the following picture (Figure 1):
- **Onset.** The star is absent on February 9.07 (bi00717, limit B = 12.9; S = −0.4) and February 10.07
  (bi00722, limit 11.7). On February 11.10 two independent cameras record a strong image: am22958 (S = 17) and
  rb11996 (S = 22). It is as strong as on February 26, and the star stays strong on every later plate. The rise
  took about a day.
- **Missed plates.** Of the plates in 1943 that the catalogue logs as non-detections, 74 of 79 show the star
  (S > 1.9).
- **Decline.** 70 catalogued detections on 64 nights come from seven plate series. A power-law fit from
  February 26 gives t2 = 58 d (68%: 43–96 d) and t3 = 153 d (132–194 d). The star falls below the plate limits
  by 1947.
- **Before and after.** No plate shows it in 1890–1942 (758 plates reaching below the outburst level) or in
  1947–1990 (315 plates).
- **Position.** The mean offset of 104 detections from the Gaia position is (+0.4 ± 0.6″, −1.2 ± 0.4″). The
  neighbour 6.5″ away is excluded at 12σ.

### 3. What it was

We adopt E(B−V) = 0.47 ± 0.08. At the Mira period-luminosity distance of 3.0 ± 0.4 kpc (Santander-García et al.
2007), B = 8.0 corresponds to M_B = −6.3 (−6.8 to −5.8). At the 1.5 kpc expansion-parallax distance it would be
−4.8. The first unsaturated plate, on day 74, still gives M_B ≈ −4.7 at 3 kpc.

The alternatives do not fit:
- **Z And-type outburst.** These rise over weeks and brighten by 2–4 mag. This event rose in a day and
  brightened by at least 9 mag.
- **Mira maximum.** A Mira does not rise several magnitudes in a day, and fifty years of earlier plates show no
  maximum above the plate limits.
- **Plate defect or asteroid.** The image is fixed in position on 64 nights and seven series.
- **Chance coincidence.** The probability that a catalogued nova lies within 2″ of any of our targets is
  2 × 10⁻⁴.

Hen 2-147 is therefore a symbiotic nova seen in eruption, like RR Tel (Mira P = 387 d, erupted 1944). Its
t3 ≈ 150 d is fast for that class, closer to the fast eruption of the Mira-donor symbiotic V407 Cyg (2010).

### 4. The nebula

The [N II] ring (5.2″ × 2.0″) expands at about 100 km/s (Corradi et al. 1999). From HST images taken in 2001
and 2004, Santander-García et al. (2007) found ballistic ages of 100–230 yr. They noted that the resulting
expansion parallax disagrees with the Mira distance, and suggested that the expanding feature is a shock front
from an outburst about 200 years ago.

A ring launched in 1943 would have to have decelerated as r ∝ t^n with n = 0.25–0.57, as for a shell sweeping up
the Mira wind. Alternatively, the moving feature could be older gas being ionised or shocked by the 1943
outburst, so that its apparent motion tracks a moving front rather than moving gas. The ring therefore cannot
date the eruption. Spectra that separate shock excitation from photoionisation, and a comparison of the
front's apparent speed with the gas velocities, could link the two.

### 5. Caveats

Plate magnitudes of an emission-line source are approximate. Harvard was the only observatory regularly
photographing −57° in 1943. We have asked [pending] DASCH staff to inspect the original plates of February 9–11
and rb12036, and to check the 1943 Harvard Announcement Cards. The light curve, cutout measurements and code are
at [repository URL].

### References

- Corradi, R. L. M., et al. 1999, A&A, 348, 978
- Grindlay, J., et al. 2009, ASPC, 410, 101
- Gromadzki, M., et al. 2009, AcA, 59, 169
- Munari, U., & Patat, F. 1993, A&A, 277, 195
- Pagnotta, A., & Schaefer, B. E. 2014, ApJ, 788, 164
- Santander-García, M., et al. 2007, A&A, 465, 481
- Schlafly, E. F., & Finkbeiner, D. P. 2011, ApJ, 737, 103
- Schlegel, D. J., Finkbeiner, D. P., & Davis, M. 1998, ApJ, 500, 525

**Figure 1.** DASCH light curve of V347 Nor. Black: catalogued detections. Grey: plate limits. Red ticks: plates
whose cutouts show the star although the catalogue records a non-detection. Lower panel: 1942.5–1947.
