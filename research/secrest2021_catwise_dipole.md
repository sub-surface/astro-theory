<!-- arXiv:2009.14826  source:ar5iv  https://arxiv.org/abs/2009.14826 -->

# A Test of the Cosmological Principle with Quasars

[Nathan J. Secrest](https://orcid.org/0000-0002-4902-8077) U.S. Naval Observatory, 3450 Massachusetts Ave NW, Washington, DC 20392-5420, USA [Sebastian von Hausegger](https://orcid.org/0000-0002-6274-1424) INRIA, 615 Rue du Jardin-Botanique, 54600 Nancy Grand-Est, France Sorbonne Université, CNRS, Institut d’Astrophysique de Paris, 98bis Bld Arago, Paris 75014, France Rudolf Peierls Centre for Theoretical Physics, University of Oxford, Parks Road, Oxford, OX1 3PU, United Kingdom [Mohamed Rameez](https://orcid.org/0000-0001-5023-5631) Dept. of High Energy Physics, Tata Institute of Fundamental Research, Homi Bhabha Road, Mumbai 400005, India [Roya Mohayaee](https://orcid.org/0000-0002-5944-3995) Sorbonne Université, CNRS, Institut d’Astrophysique de Paris, 98bis Bld Arago, Paris 75014, France [Subir Sarkar](https://orcid.org/0000-0002-3542-858X) Rudolf Peierls Centre for Theoretical Physics, University of Oxford, Parks Road, Oxford, OX1 3PU, United Kingdom [Jacques Colin](https://orcid.org/0000-0003-3300-2507) Sorbonne Université, CNRS, Institut d’Astrophysique de Paris, 98bis Bld Arago, Paris 75014, France

###### Abstract

We study the large-scale anisotropy of the Universe by measuring the dipole in the angular distribution of a flux-limited, all-sky sample of 1.36 million quasars observed by the Wide-field Infrared Survey Explorer (WISE). This sample is derived from the new CatWISE2020 catalog, which contains deep photometric measurements at 3.4 and 4.6 µm from the cryogenic, post-cryogenic, and reactivation phases of the WISE mission. While the direction of the dipole in the quasar sky is similar to that of the cosmic microwave background (CMB), its amplitude is over twice as large as expected, rejecting the canonical, exclusively kinematic interpretation of the CMB dipole with a p-value of $`5 \times 10^{- 7}`$ ($`4.9\hspace{0pt}\sigma`$ for a normal distribution, one-sided), the highest significance achieved to date in such studies. Our results are in conflict with the cosmological principle, a foundational assumption of the concordance $`\Lambda`$CDM model.

cosmology: large-scale structure of universe — cosmology: cosmic background radiation — cosmology: observations — quasars: general — galaxies: active

^(†)^(†)facilities: WISE, Blanco, Sloan^(†)^(†)software: Astropy (Astropy Collaboration et al., [2013](#bib.bib4), [2018](#bib.bib5)), healpy (Zonca et al., [2019](#bib.bib50))

## 1 Introduction

The standard Friedmann-Lemaître-Robertson-Walker (FLRW) cosmology is based on the “cosmological principle”, which posits that the universe is homogeneous and isotropic on large scales. This assumption is supported by the smoothness of the CMB, which has temperature fluctuations of only $`\sim 1`$ part in 100,000 on small angular scales. These higher multipoles of the CMB angular power spectrum are attributed to Gaussian density fluctuations created in the early universe with a nearly scale-invariant spectrum, which have grown through gravitational instability to create the large-scale structure in the present universe. The dipole anisotropy of the CMB is however much larger, being about 1 part in 1000 as observed in the heliocentric rest frame. This is interpreted as due to our motion with respect to the rest frame in which the CMB is isotropic, and is thus called the kinematic dipole. According to the most recent measurements, the inferred velocity is $`369.82 \pm 0.11`$ km s⁻¹ towards $`{{l,b} = {264\hspace{0pt}\overset{\circ}{.}\hspace{0pt}021}},{48\hspace{0pt}\overset{\circ}{.}\hspace{0pt}253}`$ (Planck Collaboration et al., [2020](#bib.bib31)). This motion is usually attributed to the gravitational effect of the inhomogeneous distribution of matter on local scales, originally dubbed the “Great Attractor” (see, e.g., Dressler, [1991](#bib.bib18)).

A consistency check of the above kinematic interpretation of the CMB dipole would be to measure the concomitant effects on higher multipoles in the CMB angular power spectrum (Challinor & van Leeuwen, [2002](#bib.bib11)). However, even the precise measurements of these by Planck allow up to 40% of the observed dipole to be due to effects other than the Solar System’s motion (see discussion in Schwarz et al., [2016](#bib.bib35)). According to galaxy counts in large-scale surveys, the universe is sensibly homogeneous when averaged over scales larger than $`\gtrsim 100`$ Mpc, as is indeed expected from considerations of structure formation in the concordance $`\Lambda`$CDM model. Hence the reference frame of matter at still greater distances should converge to that of the CMB; i.e. the dipole in the distribution of cosmologically distant sources, induced by our motion via special relativistic aberration and Doppler shifting effects, should align both in direction and in amplitude with the CMB dipole. Independent measurements of the distant matter dipole are therefore a crucial test of the cosmological principle, and equivalently of the standard model of cosmology.

Ellis & Baldwin ([1984](#bib.bib20)) proposed that such a test be done using counts of radio sources. These are typically active galactic nuclei (AGNs) at moderate redshift ($`z \sim 1`$), so locally clustered sources ($`z < 0.1`$), which can introduce an additional dipole in the distribution of matter (e.g., Tiwari & Nusser, [2016](#bib.bib43)), are not a significant contaminant. Consider a population of sources with power-law spectra $`S_{\nu} \propto \nu^{- \alpha}`$, and integral source counts per unit solid angle $`{{{{d\hspace{0pt}N}/d}\hspace{0pt}\Omega}\mspace{10mu}{({> S_{\nu}})}} \propto S_{\nu}^{- x}`$, above some limiting flux density $`S_{\nu}`$. If we are moving with velocity $`v \ll c`$ with respect to the frame in which these sources are isotropically distributed, then being “tilted observers” we should see a dipole anisotropy of amplitude (Ellis & Baldwin, [1984](#bib.bib20)):

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\mathcal{D} = {{{\lbrack{2 + {x\hspace{0pt}{({1 + \alpha})}}}\rbrack}\hspace{0pt}v}/c}}.
``` |  | (1) |

The advent of the 1.4 GHz NRAO VLA Sky Survey (NVSS; Condon et al., [1998](#bib.bib15)), which contains $`\sim 1.8`$ million sources, enabled the first estimates of the matter dipole anisotropy (Blake & Wall, [2002](#bib.bib10); Singal, [2011](#bib.bib37); Gibelyou & Huterer, [2012](#bib.bib24); Tiwari et al., [2015](#bib.bib42); Tiwari & Jain, [2015](#bib.bib41); Tiwari & Nusser, [2016](#bib.bib43)). To improve sky coverage, data was added from other radio surveys, e.g. the 325 MHz Westerbork Northern Sky Survey (WENSS; Rengelink et al., [1997](#bib.bib33); Rubart & Schwarz, [2013](#bib.bib34)), the 843 MHz Sydney University Molonglo Sky Survey (SUMMS; Mauch et al., [2003](#bib.bib29); Colin et al., [2017](#bib.bib12); Tiwari & Aluri, [2019](#bib.bib40)) and the 150 MHz TIFR GMRT Sky Survey (TGSS; Bengaly et al., [2018](#bib.bib6); Singal, [2019](#bib.bib38)). However, as was first noted by Singal ([2011](#bib.bib37)), while the direction of the matter dipole is consistent with that of the CMB, its amplitude is several times larger.

In this Letter, we report the first independent measurement of the dipole in the angular distribution of distant quasars using mid-infrared data from the Wide-field Infrared Survey Explorer (WISE; Wright et al., [2010](#bib.bib47)), which surveyed the sky at 3.4 µm, 4.6 µm,  12µm, and 22 µm (W1, W2, W3, and W4). This provides a measurement of the dipole that is independent of the radio survey-based results, as WISE is a space mission with its own unique scanning pattern, not constrained by the same observational systematics that affect ground-based surveys, such as declination limits or atmospheric effects. While WISE, along with 2MASS, has been used before to set useful constraints on the matter dipole (Gibelyou & Huterer, [2012](#bib.bib24); Yoon et al., [2014](#bib.bib49); Alonso et al., [2015](#bib.bib2); Bengaly et al., [2017](#bib.bib8); Rameez et al., [2018](#bib.bib32)), these studies were of relatively nearby galaxies ($`z \sim {0.05 - 0.1}`$) where contamination from local sources can be significant and has to be carefully accounted for. In Section [2](#S2 "2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars"), we detail the quasar sample that we use, and we introduce our methodology in Section [3](#S3 "3 Method ‣ A Test of the Cosmological Principle with Quasars"). Our results are presented in Section [4](#S4 "4 Results ‣ A Test of the Cosmological Principle with Quasars"), and we discuss their significance for cosmology in Section [5](#S5 "5 Discussion ‣ A Test of the Cosmological Principle with Quasars").

## 2 Quasar Sample

Because of the unique power of mid-infrared photometry to pick out AGNs, WISE may be used to create reliable AGN/quasar catalogs based on mid-infrared color alone (e.g., Secrest et al., [2015](#bib.bib36)). We require an AGN sample optimized for cosmological studies, so the objects should preferably be quasars: AGN-dominated and at moderate or high-redshift ($`z \gtrsim 0.1`$; cf., Tiwari & Nusser, [2016](#bib.bib43)). The sample should cover as much of the celestial sphere as is possible to minimize the impact of missing (or masked) regions, and be as deep as possible to contain the largest number of objects and thus have the greatest statistical power.

We created a custom quasar sample from the new CatWISE2020 data release (Eisenhardt et al., [2020](#bib.bib19)), which contains sources from the combined 4-band cryo, 3-band cryo, post-cryo NEOWISE, and reactivation NEOWISE-R data. The CatWISE2020 catalog is 95% complete down to $`\lesssim 17.4`$ mag in W1 and $`\lesssim 17.2`$ mag in W2, respectively 0.3 mag and 1.5 mag deeper than the previous AllWISE catalog. We select all sources in the CatWISE2020 catalog with valid measurements in W1 and W2, which are the most sensitive to AGN emission (e.g., Stern et al., [2012](#bib.bib39)). To avoid any potential directional bias from uncorrected Galactic reddening, we corrected the W1 and W2 magnitudes using the Planck Collaboration et al. ([2014](#bib.bib30)) dust map and the extinction coefficients from Wang & Chen ([2019](#bib.bib45)). To select quasars, we impose the color cut $`{{W1} - {W2}} \geq 0.8`$ (Stern et al., [2012](#bib.bib39)), which indicates AGN-dominated emission following a power-law distribution ($`S_{\nu} \propto \nu^{- \alpha}`$). This yields a raw sample of 141,698,603 objects.

To remove poor-quality photometry near clumpy and resolved nebulae both in our Galaxy (e.g., planetary nebulae) and in nearby galaxies such as the Magellanic Clouds and Andromeda, we produced masks for the nebulae and used masks of size 6 times the 20 mag arcsec⁻² isophotal radii from the 2MASS Large Galaxy Atlas (LGA; Jarrett et al., [2003](#bib.bib27)) for the Magellanic Clouds and Andromeda. In the WISE catalogs, there are often image artifacts near bright stars, caused by density suppression in their vicinity.¹¹1[http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec6_2.html#brt_stars](http://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec6_2.html#brt_stars) We find that circular masks with 2MASS $`K`$ band-dependent radii $`{\log_{10}{({r\hspace{0pt}\deg^{- 1}})}} = {{- {0.134\hspace{0pt}K}} - 0.471}`$ effectively remove these. We removed any remaining areas of poor photometry or artifacts using masks of radius $`\leq {2\hspace{0pt}\deg}`$. In all, we masked 291 sky regions, not including the Galactic plane.

Above a $`W\hspace{0pt}1`$ magnitude of 16.4, uneven source density appears due to overlaps in the ecliptic scanning pattern of WISE, most prevalent at the ecliptic poles. We select a magnitude cut of $`9 > {W1} > 16.4`$ (Vega), where the lower bound guards against potential saturation. While completeness in high source density areas in CatWISE2020 (hereafter CatWISE) is improved over the CatWISE Preliminary Catalog,²²2[https://catwise.github.io/CatWISE2020_2020_07_18.pdf](https://catwise.github.io/CatWISE2020_2020_07_18.pdf) we nonetheless find a drop off in source density below Galactic latitudes of $`{|b|} < {30\hspace{0pt}{^\circ}}`$, and a mild inverse linear trend in source density versus absolute ecliptic latitude. This trend has a slope of $`- 0.051`$ and a zero-latitude intercept of 68.89 deg⁻². We include this fit as a correction and weighting function in our later calculations. For the Galactic plane, we cut out all sources below $`{|b|} < {30\hspace{0pt}{^\circ}}`$. We also found 57 objects with anomalously low mean coverage depth w1cov $`< 80`$ that have high values of $`{W\hspace{0pt}1} - {W\hspace{0pt}2}`$. We remove these, leaving a final sample, after masking, of 1,355,352 quasars, as shown in Figure [1](#S2.F1 "Figure 1 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars").

![Refer to caption](/html/2009.14826/assets/x1.png)

Figure 1: Left: Mollweide density map of our CatWISE quasar sample, in Galactic coordinates. Right: density map smoothed using a moving average on steradian scales, showing a dipole signal. Both maps have been corrected for the residual ecliptic latitude bias (Section [2](#S2 "2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars")).

We calculate spectral indices $`\alpha`$ of our sample in the W1 band by obtaining power-law fits of the form $`S_{\nu} = {k\hspace{0pt}\nu^{- \alpha}}`$, where $`k`$ is the normalization. We produced a lookup table to determine $`\alpha`$ based on $`{W1} - {W2}`$, by calculating synthetic AB magnitudes following Equation 2 of Bessell & Murphy ([2012](#bib.bib9)). The WISE magnitudes are on the Vega magnitude system, so we convert from the AB system using the offsets $`{m_{AB} - m_{Vega}} = 2.673`$, 3.313 for W1 and W2, respectively. These WISE offsets correspond to the constant of $`- 48.60`$ associated with the definition of the synthetic AB magnitude. The normalisation $`k`$ is calculated by inverting the equation for the synthetic magnitude and using the observed W1 AB magnitude. Finally, we calculate the isophotal frequency, at which the flux density $`S_{\nu}`$ equals its mean value within the passband, using Equation A19 in Bessell & Murphy ([2012](#bib.bib9)). As our sample was constructed with the cut $`{{W1} - {W2}} \geq 0.8`$, the distribution peaks at $`\alpha \sim 1`$ and extends to steeper spectral indices, with a mean value of 1.26. Distributions of spectral indices and fluxes for our final sample of quasars are shown in Figure [2](#S2.F2 "Figure 2 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars"). The corresponding mean isophotal frequency is $`8.922 \times 10^{13}`$ Hz, with a dispersion of 0.19%. Our magnitude cut is equivalent to a flux density cut of $`77.77 > S_{\nu} > 0.09`$ mJy.

![Refer to caption](/html/2009.14826/assets/x2.png)

Figure 2: Distribution of flux densities $`S_{\nu}{( \propto \nu^{- \alpha}}`$) and spectral indices $`\alpha`$ (W1 band) in our CatWISE quasar sample, normalized as a probability density function (PDF).

To estimate the distribution of quasar redshifts, we select those within the Sloan Digital Sky Survey (SDSS) Stripe 82, a 275 deg² region of the sky scanned repeatedly by the SDSS, thus achieving an increase of depth of $`\sim 2`$ mag (Annis et al., [2014](#bib.bib3)). In the specObj table for SDSS DR16,³³3[https://www.sdss.org/dr16/spectro/spectro_access](https://www.sdss.org/dr16/spectro/spectro_access) Stripe 82 contains 4.4 times more objects with spectroscopic $`r`$-band magnitudes fainter than 20 (AB) than a comparable sky region in the SDSS main footprint. We use a sub-region of Stripe 82 between $`- 42{^\circ} < R.A. < 45{^\circ}`$, which lies outside the $`{|b|} < {30\hspace{0pt}{^\circ}}`$ Galactic plane mask we employ, and which was observed by the Extended Baryon Oscillation Spectroscopic Survey (eBOSS; Dawson et al., [2016](#bib.bib17)), yielding even deeper spectral coverage. There are 14,402 CatWISE quasars in this region. For photometric information, we cross-match these with the Dark Energy Survey, Data Release 1 (DES1; Abbott et al., [2018](#bib.bib1)), which achieved an $`i`$-band depth of 23.44 mag (AB). By comparing with Gaia DR2 (Gaia Collaboration et al., [2018](#bib.bib23)), we found that, in this part of the Stripe 82 footprint, DES1 has a systematic offset in R.A., Decl. of $`- 86`$ mas and $`+ 124`$ mas. We corrected the DES1 positions for this offset, and use a $`10\hspace{0pt}''`$ match for completeness. To avoid spurious matches, we excluded any associations that had a closer counterpart in the CatWISE catalog not in our quasar sample. This produced counterparts for 14,193 quasars (99%). Matching the DES1 counterpart coordinates onto the specObj table to within $`1\hspace{0pt}''`$ for fiber coverage, we find 8594 matches (61%). The unmatched objects are 0.3 mag fainter in W2 than the matched objects on average, suggesting that they are slightly less luminous or slightly more distant (or both). However, their mean $`r - {W2}`$ value, a measure of AGN obscuration level (e.g., Yan et al., [2013](#bib.bib48)), is 1.9 mag redder than the mean of the matched sample, implying that the unmatched objects are simply too faint at visual wavelengths for the SDSS. Indeed, while 39% of the full DES1-matched sample has $`{r - {W2}} > 6`$ mag (Vega), in line with expectations from the literature for the prevalence of type 2 AGNs (Yan et al., [2013](#bib.bib48)), 79% of the unmatched sample have $`{r - {W2}} > 6`$. This indicates that the objects in our sample without SDSS spectra are predominantly type 2 systems, an effect of the AGN orientation with respect to the line of sight, and so the matched objects may be used to estimate the distribution of redshifts for the full sample. We find a mean redshift of 1.2, with 99% having $`z > 0.1`$, so our sample is almost entirely at moderate to high redshift, as shown in Figure [3](#S2.F3 "Figure 3 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars").

![Refer to caption](/html/2009.14826/assets/x3.png)

Figure 3: Redshift distribution of our CatWISE quasar sample.

## 3 Method

### 3.1 Dipole Estimator

We determine the dipole $`\overset{\rightarrow}{\mathcal{D}}`$ of our sample using a least-squares estimator:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\sum\limits_{p}\left\lbrack {n_{p} - \left( {A_{0} + {\sum\limits_{j = 1}^{3}{A_{1\hspace{0pt}j}\hspace{0pt}d_{j,p}}}} \right)} \right\rbrack^{2}},
``` |  | (2) |

where $`n_{p}`$ denotes the number density of sources in sky pixel $`p`$, $`A_{0}`$ is the mean density (monopole), $`A_{1\hspace{0pt}j}`$ are the amplitudes of the three orthogonal dipole templates $`d_{j,p}`$, and the sum is taken over all unmasked pixels. This expression’s analytical minimum with respect to the monopole and dipole amplitudes $`A_{j}`$ is found by solving a simple linear equation, as implemented in the fit_dipole routine of healpy (Zonca et al., [2019](#bib.bib50)). Using this, the final dipole reads $`\overset{\rightarrow}{\mathcal{D}} = \left( {A_{1,p}/A_{0}},{A_{2,p}/A_{0}},{A_{3,p}/A_{0}} \right)`$. We have verified that this estimator does not suffer from bias in either direction or amplitude for density maps simulated in the manner as described below. Before computing the dipole of the source distribution (Figure [1](#S2.F1 "Figure 1 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars")) the mild inverse linear trend with ecliptic latitude of the source density was taken into account by correcting the latter as described in Section [2](#S2 "2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars").

Similarly to other dipole estimators, e.g. Blake & Wall ([2002](#bib.bib10)); Bengaly et al. ([2019](#bib.bib7)), our estimator explicitly seeks a dipolar pattern. However, it is neither computationally expensive as the minimization is done analytically, nor prone to leakage into higher multipoles, as it does not force a spherical harmonic decomposition on an incomplete sky.⁴⁴4Influence from, e.g., a quadrupole on the estimated dipole was found to be negligible. Estimators that are agnostic with regard to the true underlying signal, such as the linear estimator proposed in, e.g., Fisher et al. ([1987](#bib.bib22)); Crawford ([2009](#bib.bib16)), exhibit biases that, while well understood (Rubart & Schwarz, [2013](#bib.bib34)), make significance estimations difficult. Nevertheless, using other estimators, we find general consistency in the dipole direction and amplitude of our sample after having accounted for their biases, as much as is possible.

### 3.2 Mock data and statistical significance

We generate mock samples of $`N_{init}`$ vectors drawn from a statistically isotropic distribution, whose directions are subsequently modified by special relativistic aberration according to an observer boosted with velocity $`\overset{\rightarrow}{v}`$. Each sample is then masked with the same mask that was applied to the data (Figure [1](#S2.F1 "Figure 1 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars")). In order to respect the exact distribution of flux values and spectral indices in the data, we assign to each simulated source a flux density $`S_{\nu}`$ and a spectral index $`\alpha`$ drawn at random from their empirical distributions (Figure [2](#S2.F2 "Figure 2 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars")). The sampled fluxes are then modulated depending on source position, $`\overset{\rightarrow}{v}`$, and $`\alpha`$. Finally, only sources with $`S_{\nu} > S_{\nu,{cut}}`$ are retained, and the number of remaining sources is reduced to the size of the true sample through random selection.

Under the null hypothesis that the measured dipole $`\overset{\rightarrow}{\mathcal{D}}`$ is a consequence of our motion with respect to a frame shared by both quasars and the CMB, we generate a set of mock skies according to the above recipe. For each simulated sky we compute and record $`{\overset{\rightarrow}{\mathcal{D}}}^{sim}`$. The fraction of mock skies with amplitude $`\mathcal{D}`$ larger than our empirical sample, and with angular distance from the CMB dipole closer than our sample, gives the p-value with which the null hypothesis is rejected. Note that the effect on our results of the distributions of flux and spectral index (Figure [2](#S2.F2 "Figure 2 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars")) is included automatically via the bootstrap approach employed for our simulations.

## 4 Results

![Refer to caption](/html/2009.14826/assets/x4.png)

![Refer to caption](/html/2009.14826/assets/x5.png)

Figure 4: Left panel: Amplitude of the dipole $`\mathcal{D}`$ (solid vertical line) in the CatWISE quasar sample, versus the expectation assuming the kinematic interpretation of the CMB dipole; the distribution of $`\mathcal{D}^{sim}`$ from simulations (Section [3.2](#S3.SS2 "3.2 Mock data and statistical significance ‣ 3 Method ‣ A Test of the Cosmological Principle with Quasars")) is shown along with its median value (dashed vertical line). Right panel: Dipole direction $`\overset{\rightarrow}{\mathcal{D}}`$ in Galactic coordinates (triangle), with the null hypothesis uncertainty region ($`2\hspace{0pt}\sigma`$) in blue Section [4](#S4 "4 Results ‣ A Test of the Cosmological Principle with Quasars"). The probability under the null hypothesis of observing the dipole that we find is $`5 \times 10^{- 7}`$, or $`4.9\hspace{0pt}\sigma`$ for a normal distribution (one-sided).

Our sample of 1,355,352 quasars exhibits a dipole with amplitude $`\mathcal{D} = 0.01554`$, pointing towards $`{(l,b)} = {({238\hspace{0pt}\overset{\circ}{.}\hspace{0pt}2},{28\hspace{0pt}\overset{\circ}{.}\hspace{0pt}8})}`$. This is $`27\hspace{0pt}\overset{\circ}{.}\hspace{0pt}8`$ from the direction of the CMB dipole and over twice as large as the expected amplitude of $`\sim 0.007`$, using Equation [1](#S1.E1 "In 1 Introduction ‣ A Test of the Cosmological Principle with Quasars"). The amplitude and direction are largely unaffected by varying the mask sizes for both estimators. For instance, doubling the size of the point source masks or introducing a $`10\hspace{0pt}{^\circ}`$ mask along the super-galactic plane changes the dipole amplitude by less than $`\lesssim {5\%}`$, and the direction of the found dipole varies by $`\lesssim {5\hspace{0pt}{^\circ}}`$.

When the expected dipole is simulated assuming the kinematic interpretation of the CMB dipole, only five out of $`10^{7}`$ such simulations give $`{\overset{\rightarrow}{\mathcal{D}}}^{sim}`$ with an amplitude larger than the observed value and within the observed angular distance from the CMB dipole (left panel, Figure [4](#S4.F4 "Figure 4 ‣ 4 Results ‣ A Test of the Cosmological Principle with Quasars")). We can therefore reject the null hypothesis with a p-value of $`5 \times 10^{- 7}`$, corresponding to a significance of $`4.9\hspace{0pt}\sigma`$ for a normal distribution (one-sided).

## 5 Discussion

The CatWISE quasar sample exhibits an anomalous dipole, oriented similarly to the CMB dipole but over twice as large. Whereas a “clustering dipole” is also expected from correlations in the spatial distribution of the sources, within the concordance model this can be estimated from the distribution of these sources in redshift (see Appendix [A](#A1 "Appendix A Clustering dipole within the concordance model ‣ A Test of the Cosmological Principle with Quasars")) and the expected matter power spectrum. It is smaller than the dipole we observe in these higher redshift quasars by a factor of $`\sim 65`$.

The unique statistical power of our study has allowed us to confirm the anomalously large matter dipole suggested in previous work, which used objects selected at a different wavelength (radio), using surveys completely independent of WISE, namely NVSS, WENNS, SUMMS, and TGSS. The ecliptic scanning pattern of WISE has no relationship with the CMB dipole, so there is no reason to suspect that the dipole we measure in the CatWISE quasar sample is an artifact of the survey.

After Ellis & Baldwin ([1984](#bib.bib20)) proposed this important observational test of the cosmological principle, agreement was initially claimed between the dipole anisotropy of the CMB and that of radio sources (Blake & Wall, [2002](#bib.bib10)). If the rest frame of distant quasars is indeed that of the CMB, it would support the consensus that there exists a cosmological standard of rest, related to quantities measured in our heliocentric frame via a local special relativistic boost. This underpins modern cosmology: for example, the observed redshifts of Type Ia supernovae are routinely transformed to the “CMB frame”. From this it is deduced that the Hubble expansion rate is accelerating (isotropically), indicating dominance of a cosmological constant, and this has led to today’s concordance $`\Lambda`$CDM model. If the purely kinematic interpretation of the CMB dipole that underpins the above procedure is in fact suspect, then so are the important conclusions that follow from adopting it. In fact, as observed in our heliocentric frame, the inferred acceleration is essentially a dipole aligned approximately with the local bulk flow of galaxies and towards the CMB dipole (Colin et al., [2019](#bib.bib13)), so *cannot* be due to a cosmological constant.

If it is established that the distribution of distant matter in the large-scale universe does not share the same reference frame as the CMB, then it will become imperative to ask whether the differential expansion of space produced by nearby nonlinear structures of voids and walls and filaments can indeed be reduced to just a local boost (Wiltshire et al., [2013](#bib.bib46)). Alternatively, the CMB dipole may need to be interpreted in terms of new physics, e.g. as a remnant of the pre-inflationary universe (Turner, [1991](#bib.bib44)). Gunn ([1988](#bib.bib26)) noted that this issue is closely related to the bulk flow observed in the local universe, which in fact extends out much further than is expected in the concordance $`\Lambda`$CDM model (e.g., Colin et al., [2011](#bib.bib14); Feindt et al., [2013](#bib.bib21)). Further work is needed to clarify these important issues.

As Ellis & Baldwin ([1984](#bib.bib20)) emphasized, a serious disagreement between the standards of rest defined by distant quasars and the CMB may require abandoning the standard FLRW cosmology itself. The importance of the test we have carried out can thus not be overstated.

We thank the anonymous referee for their insightful review that greatly improved this work. We also thank Jean Souchay for discussions that helped motivate this work, Steph LaMassa for helpful suggestions on Stripe 82, and Wilbur Venus for thoughtful comments. N.J.S., M.R. and S.S. gratefully acknowledge the hospitality of the Institut d’Astrophysique de Paris. S.v.H. is supported by the EXPLORAGRAM Inria AeX grant and by the Carlsberg Foundation with grant CF19_0456. The authors made use of dustmaps (Green, [2018](#bib.bib25)) to calculate Galactic reddening. The data and software to reproduce the analysis and plots in this Letter can be found at [https://doi.org/10.5281/zenodo.4431089](https://doi.org/10.5281/zenodo.4431089).

## References

- Abbott et al. (2018) Abbott, T. M. C., Abdalla, F. B., Allam, S., et al. 2018, ApJS, 239, 18, doi: [10.3847/1538-4365/aae9f0](http://doi.org/10.3847/1538-4365/aae9f0)
- Alonso et al. (2015) Alonso, D., Salvador, A. I., Sánchez, F. J., et al. 2015, MNRAS, 449, 670, doi: [10.1093/mnras/stv309](http://doi.org/10.1093/mnras/stv309)
- Annis et al. (2014) Annis, J., Soares-Santos, M., Strauss, M. A., et al. 2014, ApJ, 794, 120, doi: [10.1088/0004-637X/794/2/120](http://doi.org/10.1088/0004-637X/794/2/120)
- Astropy Collaboration et al. (2013) Astropy Collaboration, Robitaille, T. P., Tollerud, E. J., et al. 2013, A&A, 558, A33, doi: [10.1051/0004-6361/201322068](http://doi.org/10.1051/0004-6361/201322068)
- Astropy Collaboration et al. (2018) Astropy Collaboration, Price-Whelan, A. M., Sipőcz, B. M., et al. 2018, AJ, 156, 123, doi: [10.3847/1538-3881/aabc4f](http://doi.org/10.3847/1538-3881/aabc4f)
- Bengaly et al. (2018) Bengaly, C. A. P., Maartens, R., & Santos, M. G. 2018, J. Cosmology Astropart. Phys, 2018, 031, doi: [10.1088/1475-7516/2018/04/031](http://doi.org/10.1088/1475-7516/2018/04/031)
- Bengaly et al. (2019) Bengaly, C. A. P., Siewert, T. M., Schwarz, D. J., & Maartens, R. 2019, MNRAS, 486, 1350, doi: [10.1093/mnras/stz832](http://doi.org/10.1093/mnras/stz832)
- Bengaly et al. (2017) Bengaly, C. A. P., J., Bernui, A., Alcaniz, J. S., Xavier, H. S., & Novaes, C. P. 2017, MNRAS, 464, 768, doi: [10.1093/mnras/stw2268](http://doi.org/10.1093/mnras/stw2268)
- Bessell & Murphy (2012) Bessell, M., & Murphy, S. 2012, PASP, 124, 140, doi: [10.1086/664083](http://doi.org/10.1086/664083)
- Blake & Wall (2002) Blake, C., & Wall, J. 2002, Nature, 416, 150, doi: [10.1038/416150a](http://doi.org/10.1038/416150a)
- Challinor & van Leeuwen (2002) Challinor, A., & van Leeuwen, F. 2002, Phys. Rev. D, 65, 103001, doi: [10.1103/PhysRevD.65.103001](http://doi.org/10.1103/PhysRevD.65.103001)
- Colin et al. (2017) Colin, J., Mohayaee, R., Rameez, M., & Sarkar, S. 2017, MNRAS, 471, 1045, doi: [10.1093/mnras/stx1631](http://doi.org/10.1093/mnras/stx1631)
- Colin et al. (2019) —. 2019, A&A, 631, L13, doi: [10.1051/0004-6361/201936373](http://doi.org/10.1051/0004-6361/201936373)
- Colin et al. (2011) Colin, J., Mohayaee, R., Sarkar, S., & Shafieloo, A. 2011, MNRAS, 414, 264, doi: [10.1111/j.1365-2966.2011.18402.x](http://doi.org/10.1111/j.1365-2966.2011.18402.x)
- Condon et al. (1998) Condon, J. J., Cotton, W. D., Greisen, E. W., et al. 1998, AJ, 115, 1693, doi: [10.1086/300337](http://doi.org/10.1086/300337)
- Crawford (2009) Crawford, F. 2009, Astrophys. J., 692, 887, doi: [10.1088/0004-637X/692/1/887](http://doi.org/10.1088/0004-637X/692/1/887)
- Dawson et al. (2016) Dawson, K. S., Kneib, J.-P., Percival, W. J., et al. 2016, AJ, 151, 44, doi: [10.3847/0004-6256/151/2/44](http://doi.org/10.3847/0004-6256/151/2/44)
- Dressler (1991) Dressler, A. 1991, Nature, 350, 391, doi: [10.1038/350391a0](http://doi.org/10.1038/350391a0)
- Eisenhardt et al. (2020) Eisenhardt, P. R. M., Marocco, F., Fowler, J. W., et al. 2020, ApJS, 247, 69, doi: [10.3847/1538-4365/ab7f2a](http://doi.org/10.3847/1538-4365/ab7f2a)
- Ellis & Baldwin (1984) Ellis, G. F. R., & Baldwin, J. E. 1984, MNRAS, 206, 377, doi: [10.1093/mnras/206.2.377](http://doi.org/10.1093/mnras/206.2.377)
- Feindt et al. (2013) Feindt, U., Kerschhaggl, M., Kowalski, M., et al. 2013, A&A, 560, A90, doi: [10.1051/0004-6361/201321880](http://doi.org/10.1051/0004-6361/201321880)
- Fisher et al. (1987) Fisher, N. I., Lewis, T., & Embleton, B. J. J. 1987, Statistical Analysis of Spherical Data (Cambridge University Press), 29–66
- Gaia Collaboration et al. (2018) Gaia Collaboration, Brown, A. G. A., Vallenari, A., et al. 2018, A&A, 616, A1, doi: [10.1051/0004-6361/201833051](http://doi.org/10.1051/0004-6361/201833051)
- Gibelyou & Huterer (2012) Gibelyou, C., & Huterer, D. 2012, MNRAS, 427, 1994, doi: [10.1111/j.1365-2966.2012.22032.x](http://doi.org/10.1111/j.1365-2966.2012.22032.x)
- Green (2018) Green, G. 2018, The Journal of Open Source Software, 3, 695, doi: [10.21105/joss.00695](http://doi.org/10.21105/joss.00695)
- Gunn (1988) Gunn, J. E. 1988, Astronomical Society of the Pacific Conference Series, Vol. 4, Hubble’s Deviations from Pure Hubble Flow: A Review (San Francisco, CA: ASP), 344
- Jarrett et al. (2003) Jarrett, T. H., Chester, T., Cutri, R., Schneider, S. E., & Huchra, J. P. 2003, AJ, 125, 525, doi: [10.1086/345794](http://doi.org/10.1086/345794)
- Lewis et al. (2000) Lewis, A., Challinor, A., & Lasenby, A. 2000, ApJ, 538, 473, doi: [10.1086/309179](http://doi.org/10.1086/309179)
- Mauch et al. (2003) Mauch, T., Murphy, T., Buttery, H. J., et al. 2003, MNRAS, 342, 1117, doi: [10.1046/j.1365-8711.2003.06605.x](http://doi.org/10.1046/j.1365-8711.2003.06605.x)
- Planck Collaboration et al. (2014) Planck Collaboration, Abergel, A., Ade, P. A. R., et al. 2014, A&A, 571, A11, doi: [10.1051/0004-6361/201323195](http://doi.org/10.1051/0004-6361/201323195)
- Planck Collaboration et al. (2020) Planck Collaboration, Akrami, Y., Arroja, F., et al. 2020, A&A, 641, A1, doi: [10.1051/0004-6361/201833880](http://doi.org/10.1051/0004-6361/201833880)
- Rameez et al. (2018) Rameez, M., Mohayaee, R., Sarkar, S., & Colin, J. 2018, MNRAS, 477, 1772, doi: [10.1093/mnras/sty619](http://doi.org/10.1093/mnras/sty619)
- Rengelink et al. (1997) Rengelink, R. B., Tang, Y., de Bruyn, A. G., et al. 1997, A&AS, 124, 259, doi: [10.1051/aas:1997358](http://doi.org/10.1051/aas:1997358)
- Rubart & Schwarz (2013) Rubart, M., & Schwarz, D. J. 2013, A&A, 555, A117, doi: [10.1051/0004-6361/201321215](http://doi.org/10.1051/0004-6361/201321215)
- Schwarz et al. (2016) Schwarz, D. J., Copi, C. J., Huterer, D., & Starkman, G. D. 2016, Classical and Quantum Gravity, 33, 184001, doi: [10.1088/0264-9381/33/18/184001](http://doi.org/10.1088/0264-9381/33/18/184001)
- Secrest et al. (2015) Secrest, N. J., Dudik, R. P., Dorland, B. N., et al. 2015, ApJS, 221, 12, doi: [10.1088/0067-0049/221/1/12](http://doi.org/10.1088/0067-0049/221/1/12)
- Singal (2011) Singal, A. K. 2011, ApJ, 742, L23, doi: [10.1088/2041-8205/742/2/L23](http://doi.org/10.1088/2041-8205/742/2/L23)
- Singal (2019) —. 2019, Phys. Rev. D, 100, 063501, doi: [10.1103/PhysRevD.100.063501](http://doi.org/10.1103/PhysRevD.100.063501)
- Stern et al. (2012) Stern, D., Assef, R. J., Benford, D. J., et al. 2012, ApJ, 753, 30, doi: [10.1088/0004-637X/753/1/30](http://doi.org/10.1088/0004-637X/753/1/30)
- Tiwari & Aluri (2019) Tiwari, P., & Aluri, P. K. 2019, ApJ, 878, 32, doi: [10.3847/1538-4357/ab1d58](http://doi.org/10.3847/1538-4357/ab1d58)
- Tiwari & Jain (2015) Tiwari, P., & Jain, P. 2015, MNRAS, 447, 2658, doi: [10.1093/mnras/stu2535](http://doi.org/10.1093/mnras/stu2535)
- Tiwari et al. (2015) Tiwari, P., Kothari, R., Naskar, A., Nadkarni-Ghosh, S., & Jain, P. 2015, Astroparticle Physics, 61, 1, doi: [10.1016/j.astropartphys.2014.06.004](http://doi.org/10.1016/j.astropartphys.2014.06.004)
- Tiwari & Nusser (2016) Tiwari, P., & Nusser, A. 2016, J. Cosmology Astropart. Phys, 2016, 062, doi: [10.1088/1475-7516/2016/03/062](http://doi.org/10.1088/1475-7516/2016/03/062)
- Turner (1991) Turner, M. S. 1991, Phys. Rev. D, 44, 3737, doi: [10.1103/PhysRevD.44.3737](http://doi.org/10.1103/PhysRevD.44.3737)
- Wang & Chen (2019) Wang, S., & Chen, X. 2019, ApJ, 877, 116, doi: [10.3847/1538-4357/ab1c61](http://doi.org/10.3847/1538-4357/ab1c61)
- Wiltshire et al. (2013) Wiltshire, D. L., Smale, P. R., Mattsson, T., & Watkins, R. 2013, Phys. Rev. D, 88, 083529, doi: [10.1103/PhysRevD.88.083529](http://doi.org/10.1103/PhysRevD.88.083529)
- Wright et al. (2010) Wright, E. L., Eisenhardt, P. R. M., Mainzer, A. K., et al. 2010, AJ, 140, 1868, doi: [10.1088/0004-6256/140/6/1868](http://doi.org/10.1088/0004-6256/140/6/1868)
- Yan et al. (2013) Yan, L., Donoso, E., Tsai, C.-W., et al. 2013, AJ, 145, 55, doi: [10.1088/0004-6256/145/3/55](http://doi.org/10.1088/0004-6256/145/3/55)
- Yoon et al. (2014) Yoon, M., Huterer, D., Gibelyou, C., Kovács, A., & Szapudi, I. 2014, MNRAS, 445, L60, doi: [10.1093/mnrasl/slu133](http://doi.org/10.1093/mnrasl/slu133)
- Zonca et al. (2019) Zonca, A., Singer, L., Lenz, D., et al. 2019, The Journal of Open Source Software, 4, 1298, doi: [10.21105/joss.01298](http://doi.org/10.21105/joss.01298)

## Appendix A Clustering dipole within the concordance model

The clustering dipole $`\mathcal{D}_{cls}`$ in a sample of objects as seen by a typical observer in the concordance $`\Lambda`$CDM cosmology can be computed given the power spectrum $`P\hspace{0pt}{(k)}`$ of (dark) matter density perturbations (Gibelyou & Huterer, [2012](#bib.bib24)):

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\mathcal{D}_{cls} = \sqrt{\frac{9}{4\hspace{0pt}\pi}\hspace{0pt}C_{1}}},
``` |  | (A1) |

where

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{C_{l} = {b^{2}\hspace{0pt}\frac{2}{\pi}\hspace{0pt}{\int_{0}^{\infty}{f_{l}\hspace{0pt}{(k)}^{2}\hspace{0pt}P\hspace{0pt}{(k)}\hspace{0pt}k^{2}\hspace{0pt}{dk}}}}}.
``` |  | (A2) |

Here $`b`$ is the linear bias of the observed objects with respect to the dark matter and the filter function $`f_{l}\hspace{0pt}{(k)}`$ is

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{f_{l}\hspace{0pt}{(k)}} = {\int_{0}^{\infty}{j_{l}\hspace{0pt}{({k\hspace{0pt}r})}\hspace{0pt}f\hspace{0pt}{(r)}\hspace{0pt}{dr}}}},
``` |  | (A3) |

where $`j_{l}`$ is the spherical Bessel function of order $`l`$ and $`f\hspace{0pt}{(r)}`$ is the probability distribution for the comoving distance $`r`$ to a random object in the survey, given by

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{f\hspace{0pt}{(r)}} = {\frac{H\hspace{0pt}{(z)}}{H_{0}\hspace{0pt}r_{0}}\hspace{0pt}\frac{d\hspace{0pt}N}{d\hspace{0pt}z}}},
``` |  | (A4) |

normalised such that $`{\int_{0}^{\infty}{f\hspace{0pt}{(r)}\hspace{0pt}{dr}}} = 1`$ and $`{{d\hspace{0pt}N}/d}\hspace{0pt}z`$ is the redshift distribution of the observed objects. Employing $`r_{0} = {c/H_{0}} = {3000\hspace{0pt}h^{- 1}}`$ Mpc, Planck 2018 cosmological parameters from Astropy, $`P\hspace{0pt}{(k)}`$ at $`z = 0`$ using camb (Lewis et al., [2000](#bib.bib28)), and a cubic-spline fit to the redshift distributions shown in Figure [3](#S2.F3 "Figure 3 ‣ 2 Quasar Sample ‣ A Test of the Cosmological Principle with Quasars") to determine $`{{d\hspace{0pt}N}/d}\hspace{0pt}z`$, we estimate $`\mathcal{D}_{cls}`$ to be 0.00024 (taking $`b = 1`$) for the CatWISE quasar selection. So, the clustering dipole is quite negligible compared to the observed quasar dipole of $`\mathcal{D} = 0.01554`$ (Section [4](#S4 "4 Results ‣ A Test of the Cosmological Principle with Quasars")).

[◄](/html/2009.14825) [![ar5iv homepage](/assets/ar5iv.png)](/) [Feeling\
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion\
report](/log/2009.14826) [Report\
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2009.14826) [View original\
on arXiv](https://arxiv.org/abs/2009.14826)[►](/html/2009.14827)

[](javascript:toggleColorScheme() "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Sat Mar 2 09:16:40 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)
