# Milestone 3 results — x and α re-measured from Quaia (2026-06-23)

Code: [`../g_xalpha.py`](../g_xalpha.py). Replaces the M1/M2 placeholder
(x=1.7, α=1.0 → D_kin=0.0067) with values measured directly from the Quaia photometry, so
the kinematic expectation D_kin = [2 + x(1+α)]·β is data-grounded, not assumed.

## Method
- **x (number-count slope):** integral counts dN/dΩ(>S) ∝ S⁻ˣ. In magnitude space N(<G) ∝ 10^{sG}
  with x = 2.5·s. We fit s = d log₁₀(dN/dG)/dG from differential counts in a window *below* the
  completeness rollover near the limit (von Hausegger 2024: x must be measured at the flux limit,
  where the dipole effect is dominated). Poisson-weighted linear fit.
- **α (spectral index):** per-source from the G−BP colour with Gaia DR3 Vega zero-points + mean
  wavelengths (Oayda+2024 eq 5): α = (k − m_{G−BP}) / (2.5 log₁₀(ν_BP/ν_G)), k = ZP_G − ZP_BP.
  Taken within 0.5 mag of the flux limit.
- **D_kin:** Monte-Carlo like Oayda — sample α from its empirical near-limit distribution and
  x ~ N(x, σ_x), 50 000×, report mean and [16,84] percentiles.

## Result — reproduces Oayda+2024's kinematic expectation to ~3%

| sample | x | α (median) | **D_kin (this work)** | Oayda+2024 |
|---|---|---|---|---|
| Quaia high (G<20.5) | 0.956 ± 0.006 | +2.44 (σ 0.73) | **0.0065 [0.0057, 0.0073]** | 0.0068 |
| Quaia low (G<20.0) | 1.222 ± 0.008 | +2.62 (σ 0.65) | **0.0079 [0.0071, 0.0088]** | 0.0080 |

- Independent measurement, same numbers → the EB inputs and our D_kin are validated.
- α≈2.4–2.6 looks steep vs a "true" optical SED slope, but it is the *effective* EB index from
  Gaia's broad G/BP bands under the monochromatic approximation — exactly what Oayda use, and it
  reproduces their D_kin, so the convention is self-consistent. (Sign of the ν_BP/ν_G term matters:
  a flipped sign gives the unphysical α≈−2.4 we caught and corrected.)

## Effect on the M2 conclusions (tightened, not changed)
With the measured (not placeholder) D_kin, the principled |b|>40 dipoles become:

| | D (M2) | D_kin (M3) | **D/D_kin** | σ above isotropic floor |
|---|---|---|---|---|
| Quaia low | 0.0109 | 0.0079 | **1.38** | 2.6σ |
| Quaia high | 0.0124 | 0.0065 | **1.91** | 4.1σ |

The significance-vs-noise-floor numbers are unchanged (they don't depend on D_kin). Quaia low sits
~1.4× the kinematic expectation and is statistically consistent with it (M2 injection test);
Quaia high's residual remains ~1.9× but points away from the CMB. Conclusions of M1–M2 stand.

## Next
M4 — run the SAME pipeline on CatWISE2020 (Secrest sample): reproduce D≈0.0155 @ (l,b)≈(238,29),
then apply the principled (ecliptic-latitude) correction and the isotropic-mock null, and place
CatWISE next to Quaia on one figure with confidence intervals vs the literature.
