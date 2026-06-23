<!-- arXiv:2212.04925  source:ar5iv  https://arxiv.org/abs/2212.04925 -->

# Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole

[Caroline Guandalin](https://orcid.org/0000-0003-1490-9314) School of Physical & Chemical Sciences, Queen Mary University of London, London E1 4NS, UK Jade Piat Aix-Marseille University, Marseille, France School of Physical & Chemical Sciences, Queen Mary University of London, London E1 4NS, UK [Chris Clarkson](https://orcid.org/0000-0001-7363-0722) School of Physical & Chemical Sciences, Queen Mary University of London, London E1 4NS, UK Department of Physics & Astronomy, University of Western Cape, Cape Town 7535, South Africa Department of Mathematics and Applied Mathematics, University of Cape Town 7701, South Africa [Roy Maartens](https://orcid.org/0000-0001-9050-5894) Department of Physics & Astronomy, University of Western Cape, Cape Town 7535, South Africa Institute of Cosmology & Gravitation, University of Portsmouth, Portsmouth PO1 3FX, UK National Institute for Theoretical & Computational Sciences (NITheCS), Cape Town 7535, South Africa

###### Abstract

The Cosmological Principle is part of the foundation that underpins the standard model of the Universe. In the era of precision cosmology, when stress tests of the standard model are uncovering various tensions and possible anomalies, it is critical to check the viability of this principle. A key test is the consistency between the kinematic dipoles of the cosmic microwave background and of the large-scale matter distribution. Results using radio continuum and quasar samples indicate a rough agreement in the directions of the two dipoles, but a larger than expected amplitude of the matter dipole. The resulting tension with the radiation dipole has been estimated at $`\sim {5\hspace{0pt}\sigma}`$ for some cases, suggesting a potential new cosmological tension and a possible violation of the CP. However, the standard formalism for predicting the dipole in the two-dimensional projection of sources overlooks possible evolution effects in the luminosity function. In fact, radial information from the luminosity function is necessary for a correct projection of the three-dimensional source distribution. Using a variety of current models of the quasar luminosity function, we show that neglecting redshift evolution can significantly overestimate the relative velocity amplitude. While the models we investigate are consistent with each other and with current data, the dipole derived from these, which depends on derivatives of the luminosity function, can disagree by more than $`3\hspace{0pt}\sigma`$. This theoretical systematic bias needs to be resolved before robust conclusions can be made about a new cosmic tension.

^(†)^(†)facilities: This work uses data from the extended Baryon Oscillation Spectroscopic Survey of the Sloan Digital Sky Survey (SDSS-IV/eBOSS), as extracted from Table A.1 of [PD2016](#bib.bib34).^(†)^(†)software: We made extensive use of [emcee](https://emcee.readthedocs.io/) (Foreman-Mackey et al., [2013](#bib.bib24)), [CLASS](https://github.com/lesgourg/class_public) (Blas et al., [2011](#bib.bib7)), numpy (Harris et al., [2020](#bib.bib27)), scipy (Virtanen et al., [2020](#bib.bib48)), and matplotlib (Hunter, [2007](#bib.bib28)).

## 1 Introduction

The Cosmological Principle (CP), i.e. that the spatial distribution of matter and radiation in the Universe is statistically homogeneous and isotropic on large enough scales, is perhaps the most fundamental assumption in modern cosmology. It is the basis for modelling the Universe using the Friedmann-Lemaître-Robertson-Walker (FLRW) metric (see, e.g., Ehlers et al., [1968](#bib.bib21); Ellis et al., [1983](#bib.bib23); Stoeger et al., [1995](#bib.bib44); Clarkson & Maartens, [2010](#bib.bib13); Clarkson, [2012](#bib.bib12)). The CP implies that there is a unique cosmic frame defined by observers who measure statistical isotropy and homogeneity of the cosmic microwave background (CMB) and of the matter distribution on sufficiently large scales.

Earth observers are not at rest in the cosmic frame, but are moving with a velocity $`{\mathbf{v}}_{o}^{CMB}`$ relative to the CMB rest frame, i.e., the frame in which the CMB dipole vanishes and the CMB is statistically isotropic (Stewart & Sciama, [1967](#bib.bib43); Peebles & Wilkinson, [1968](#bib.bib35)). This leads to a dipole of $`3362.08 \pm {0.99\hspace{0pt}\mu}`$K, much larger than the $`\ell \geq 2`$ multipoles, which is used to extract a velocity of $`v_{o}^{CMB} = {369.82 \pm {{0.11\hspace{0pt}{km}}/s}}`$ towards $`{(l,b)} = {({264.021 \pm 0.011},{48.253 \pm 0.005})}^{\circ}`$ (Aghanim et al., [2020](#bib.bib2)).

At leading order, the boosted temperature contrast is

|  |  |  |  |
|----|----|----|----|
|  | $`{{{\overset{\sim}{\delta}}_{T}\hspace{0pt}{({\mathbf{n}})}} = {{\delta_{T}\hspace{0pt}{({\mathbf{n}})}} + {{\mathcal{D}_{CMB}\hspace{0pt}{\mathbf{n}}} \cdot {\mathbf{v}}_{o}^{CMB}}}},`$ |  | (1) |

where $`\mathbf{n}`$ is a unit vector in the direction of observation. Using units with $`c = 1`$, the dimensionless CMB dipole factor is $`\mathcal{D}_{CMB} = 1`$.

The CP requires the large-scale distribution of sources on the sky to have the same dipole – i.e., the relative velocity $`{\mathbf{v}}_{o}`$ extracted from these sources should agree with that from the CMB as

|  |  |  |  |
|----|----|----|----|
|  | $`{{{\hat{\mathbf{v}}}_{o} \approx {\hat{\mathbf{v}}}_{o}^{CMB}},{v_{o} \approx v_{o}^{CMB}}}.`$ |  | (2) |

The boosted two-dimensional (2D) number density contrast is

|  |  |  |  |
|----|----|----|----|
|  | $`{{{\overset{\sim}{\delta}}_{2\hspace{0pt}D}\hspace{0pt}{({\mathbf{n}})}} = {{\delta_{2\hspace{0pt}D}\hspace{0pt}{({\mathbf{n}})}} + {{\mathcal{D}\hspace{0pt}{\mathbf{n}}} \cdot {\mathbf{v}}_{o}}}}.`$ |  | (3) |

The dimensionless number count dipole factor $`\mathcal{D}`$ is independent of $`v_{o}`$, but it does depend on the sample.

Ellis & Baldwin ([1984](#bib.bib22)) estimated the dipole factor for the 2D projection of radio sources as

|     |                                                             |     |     |
|-----|-------------------------------------------------------------|-----|-----|
|     |                                                             
       ``` math                                                     
       {\mathcal{D}_{EB} = {2 + {x\hspace{0pt}{({1 + \alpha})}}}},  
       ```                                                          |     | (4) |

in which the power-law indices $`x`$ and $`\alpha`$ are constants that determine the number counts per solid angle ($`\mathcal{N}_{\Omega} \propto S_{c}^{- x}`$, with $`S_{c}`$ being the flux threshold of the survey) and the flux density of sources in their rest frame ($`{S\hspace{0pt}{(\nu_{obs})}} \propto \nu_{obs}^{- \alpha}`$) at fixed observed frequency $`\nu_{obs}`$. This is potentially the most powerful consistency test of the CP, but it faces major observational obstacles.

First, the galaxy sample must cover a wide sky area and a deep redshift range, with a high number density. High redshifts are required to extract the cosmic dipole and the lowest-redshift sources need to be removed to avoid nonlinear contamination (Tiwari & Nusser, [2016](#bib.bib46); Bengaly et al., [2019](#bib.bib5)). Second, measurements of large-scale features face well-known systematic errors (effects of the mask, instrumental/survey systematics, stellar contamination, etc.). In this paper, we focus on a key theoretical systematic error that arises when estimating the dimensionless dipole factor $`\mathcal{D}`$ without using radial information.

Different groups have applied Equation ([4](#S1.E4 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) to extract the relative velocity from the measured dipole in wide-area radio continuum surveys (see, e.g., Blake & Wall, [2002](#bib.bib6); Gibelyou & Huterer, [2012](#bib.bib26); Rubart & Schwarz, [2013](#bib.bib39); Tiwari et al., [2015](#bib.bib45); Ghosh et al., [2016](#bib.bib25); Colin et al., [2017](#bib.bib14); Bengaly et al., [2018](#bib.bib4)). A general conclusion of these analyses is that the dipole direction is typically consistent with that of the CMB, but the relative velocity amplitude is significantly larger:¹¹1Note that Darling ([2022](#bib.bib18)) finds no tension of this form.

|  |  |  |  |
|----|----|----|----|
|  | $`{{{\hat{\mathbf{v}}}_{o}^{EB} \approx {\hat{\mathbf{v}}}_{o}^{CMB}},{v_{o}^{EB} > v_{o}^{CMB}}}.`$ |  | (5) |

Recently, one of the most thorough analyses has used the very large CatWISE2020 sample of quasars to apply the Ellis–Baldwin test (Secrest et al., [2021](#bib.bib42), [2022](#bib.bib41)). Their conclusion reinforces Equation ([5](#S1.E5 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), finding a tension with the CMB velocity amplitude at $`\sim {5\hspace{0pt}\sigma}`$.

Some works have attempted to resolve this tension by invoking superhorizon effects (for example, see Das et al., [2021](#bib.bib19); Domènech et al., [2022](#bib.bib20); Tiwari et al., [2022](#bib.bib47)). In this work, we do not investigate alternative cosmological models. Instead, we consider a possible theoretical systematic in the velocity estimates. We extend the pioneering analysis of Dalang & Bonvin ([2022](#bib.bib17)), who showed that the presence of parameter evolution can lead to significant corrections on the Ellis–Baldwin approximation. We investigate the impact of different quasar luminosity function (QLF) models on the predicted amplitude of the kinematic dipole. In Section [2](#S2 "2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), we review the theoretical predictions for the expected kinematic dipole amplitude in three dimensions, $`\mathcal{D}_{3\hspace{0pt}D}`$, and how it can be projected into the 2D dipole factor $`\mathcal{D}`$ of Equation ([3](#S1.E3 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). In Section [3](#S3 "3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), we introduce the QLF data and the theoretical modelling used for the fits. The Markov Chain Monte Carlo (MCMC) analysis and the goodness-of-fit criterion are described in Section [4](#S4 "4 Method ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"). The results and conclusion are presented in Section [5](#S5 "5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") and Section [6](#S6 "6 Conclusions ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), respectively.

## 2 Correction to the standard dipole

In all of the results characterised by Equation ([5](#S1.E5 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), the dipole is measured and then Equation ([4](#S1.E4 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) is used to extract the relative speed. However, $`\mathcal{D}_{EB}`$ assumes that radial information can be neglected. In reality, $`\mathcal{D}`$ is a radial projection of the three-dimensional redshift-dependent dipole factor (Maartens et al., [2018](#bib.bib30))

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\mathcal{D}_{3\hspace{0pt}D} = {{2 + \frac{\overset{˙}{\mathcal{H}}}{\mathcal{H}^{2}} + \frac{2}{r\hspace{0pt}\mathcal{H}}} - \frac{5\hspace{0pt}s}{r\hspace{0pt}\mathcal{H}} - b_{e}}},
``` |  | (6) |

which can be derived from linear perturbations.²²2There is also an intrinsic dipole sourced by primordial perturbations (Tiwari & Nusser, [2016](#bib.bib46); Nadolny et al., [2021](#bib.bib32)), which is not relevant for our discussion.

The first three terms on the right are purely cosmological ($`r`$ is the comoving line-of-sight distance) and the last two contain the magnification ($`s`$) and evolution ($`b_{e}`$) biases, which depend on the sample’s luminosity function $`\Phi`$ through the background comoving number density (in the source frame)

|  |  |  |  |
|----|----|----|----|
|  | $`{{n\hspace{0pt}{(z,M_{c})}} = {\int_{- \infty}^{M_{c}\hspace{0pt}{(z)}}{{dM}\hspace{0pt}\Phi\hspace{0pt}{(z,M)}}}}.`$ |  | (7) |

The absolute magnitude of a source (corresponding to its intrinsic luminosity) is given by

|  |  |  |  |
|----|----|----|----|
|  | $`{M = {{m\hspace{0pt}{(z)}} - {\mu\hspace{0pt}{(z)}} - {K\hspace{0pt}{(z)}}}},`$ |  | (8) |

where $`m`$ is the apparent magnitude measured at redshift $`z`$ (corresponding to observed flux), $`\mu`$ is the distance modulus (defined by the background luminosity distance) and $`K`$ is the $`K`$-correction. The apparent magnitude cut of the survey $`m_{c}`$ leads to an absolute threshold $`M_{c}\hspace{0pt}{(z)}`$ that depends on redshift via Equation ([8](#S2.E8 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). The magnification and evolution biases become (Challinor & Lewis, [2011](#bib.bib11); Alonso et al., [2015](#bib.bib3); Maartens et al., [2021](#bib.bib31))

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`s\hspace{0pt}{(z,M_{c})}`$ | $`{= \frac{\partial{{\log n}\hspace{0pt}{(z,M_{c})}}}{\partial M_{c}} = {\frac{1}{\ln 10}\hspace{0pt}\frac{\Phi\hspace{0pt}{(z,M_{c})}}{n\hspace{0pt}{(z,M_{c})}}}},`$ |  | (9) |
|  | $`b_{e}\hspace{0pt}{(z,M_{c})}`$ | $`= {- \frac{\partial{{\ln n}\hspace{0pt}{(z,M_{c})}}}{\partial{\ln{({1 + z})}}}}`$ |  |  |
|  |  | $`{= {- \left. {\frac{({1 + z})}{n\hspace{0pt}{(z,M_{c})}}\hspace{0pt}{\int_{- \infty}^{M_{c}\hspace{0pt}{(z)}}{{dM}\hspace{0pt}\frac{\partial{\Phi\hspace{0pt}{(z,M)}}}{\partial z}}}} \right|_{M}}},`$ |  | (10) |

in which $`s`$ determines whether sources will be included or excluded from the sample due to lensing convergence and $`b_{e}`$ describes the deviation of the comoving number density of sources from the conserved case ($`b_{e} = 0`$). Note the partial redshift derivative in $`b_{e}`$ must be taken at fixed magnitude $`M_{c}`$.

The observed source number density $`\mathcal{N}`$ (number per redshift per solid angle), is not the same as the number density $`n`$ (number per comoving volume) measured at the source (Bonvin & Durrer, [2011](#bib.bib8); Challinor & Lewis, [2011](#bib.bib11); Alonso et al., [2015](#bib.bib3); Maartens et al., [2018](#bib.bib30), [2021](#bib.bib31)). At the background level, these quantities are related by

|  |  |  |  |
|----|----|----|----|
|  | $`{\mathcal{N} = \frac{d\hspace{0pt}N}{d\hspace{0pt}z\hspace{0pt}d\hspace{0pt}\Omega} = {\frac{r^{2}}{{({1 + z})}\hspace{0pt}\mathcal{H}}\hspace{0pt}n}},`$ |  | (11) |

where $`N`$ is the number of sources, which is the same in the observer and source frames, and $`{d\hspace{0pt}N} = {\mathcal{N}\hspace{0pt}d\hspace{0pt}z\hspace{0pt}d\hspace{0pt}\Omega} = {n\hspace{0pt}d\hspace{0pt}V}`$, with $`V`$ the comoving volume. The observed number density projected on the sky becomes

|  |  |  |  |
|----|----|----|----|
|  | $`{\mathcal{N}_{\Omega} = {\int_{0}^{\infty}{{dz}\hspace{0pt}\mathcal{N}\hspace{0pt}{(z)}}}}.`$ |  | (12) |

The 2D dipole factor is found by projecting Equation ([6](#S2.E6 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) along the radial direction, weighted by the number counts (Nadolny et al., [2021](#bib.bib32); Dalang & Bonvin, [2022](#bib.bib17))³³3In these papers, their $`\mathcal{D}_{kin}`$ equals our $`\mathcal{D}\hspace{0pt}v_{o}`$.:

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\mathcal{D} = {\int_{0}^{\infty}{{dz}\hspace{0pt}f\hspace{0pt}{(z)}\hspace{0pt}\mathcal{D}_{3\hspace{0pt}D}\hspace{0pt}{(z)}\hspace{0pt}\text{where}\hspace{0pt}f\hspace{0pt}{(z)}}} = \frac{\mathcal{N}\hspace{0pt}{(z)}}{\mathcal{N}_{\Omega}}}.
``` |  | (13) |

The result is

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{\mathcal{D} = {\mathcal{D}_{cosmo} + \mathcal{D}_{mag} + \mathcal{D}_{evol}}},
``` |  | (14) |

where

|  |  |  |  |  |
|----|----|----|----|----|
|  | $`\mathcal{D}_{cosmo}`$ | $`{= {\int_{0}^{\infty}{{dz}\hspace{0pt}f\hspace{0pt}\left\lbrack {2 + \frac{2}{r\hspace{0pt}\mathcal{H}} + \frac{\overset{˙}{\mathcal{H}}}{\mathcal{H}^{2}}} \right\rbrack}}},`$ |  | (15) |
|  | $`\mathcal{D}_{mag}`$ | $`{= {- {2\hspace{0pt}{\int_{0}^{\infty}{{dz}\hspace{0pt}f\hspace{0pt}\frac{x}{r\hspace{0pt}\mathcal{H}}\hspace{0pt}\text{with}\hspace{0pt}x}}}} = {\frac{5}{2}\hspace{0pt}s}},`$ |  | (16) |
|  | $`\mathcal{D}_{evol}`$ | $`{= {- {\int_{0}^{\infty}{{dz}\hspace{0pt}f\hspace{0pt}b_{e}}}}}.`$ |  | (17) |

For a constant magnification bias, $`x`$ corresponds to the Ellis–Baldwin parameter in Equation ([4](#S1.E4 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), but in general it evolves with redshift. The dipole factor $`\mathcal{D}`$, in Equations ([14](#S2.E14 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) – ([17](#S2.E17 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), depends on the evolution bias $`b_{e}`$, which is unique for the sample, but not on the variable source spectral index $`\alpha`$ (Carballo et al., [1999](#bib.bib10); Secrest et al., [2021](#bib.bib42)). This important feature arises from the fact that the absolute magnitude cut, $`M_{c}\hspace{0pt}{(z)}`$, does not depend on $`\alpha`$ at a constant redshift, but at a fixed distance $`r`$ (Dalang & Bonvin, [2022](#bib.bib17)).

The Ellis–Baldwin formula is an approximation to Equation ([13](#S2.E13 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), $`\mathcal{D}_{EB} = {\mathcal{D} + {\Delta\hspace{0pt}\mathcal{D}}}`$, that includes a correction $`\Delta\hspace{0pt}\mathcal{D}`$ which does not affect the dipole direction. The dipole amplitude is extracted directly from data and is equal to the dipole factor times the relative speed

|  |  |  |  |
|----|----|----|----|
|  | $`{{\mathcal{D}_{EB}\hspace{0pt}v_{o}^{EB}} = {\mathcal{D}\hspace{0pt}v_{o}}}.`$ |  | (18) |

When using the exact formula (Equation [13](#S2.E13 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) for the dipole factor $`\mathcal{D}`$, the correct relative speed $`v_{o}`$ is extracted from the measurements. However, using the Ellis–Baldwin approximation $`\mathcal{D}_{EB}`$ we extract an estimate of the relative speed that may differ from $`v_{o}`$: $`v_{o}^{EB} = {v_{o} + {\Delta\hspace{0pt}v_{o}}}`$. For example, if the relative speed is overestimated, i.e., $`v_{o}^{EB} > v_{o}`$, then $`\mathcal{D}_{EB}`$ underestimates $`\mathcal{D}`$, since $`{{\Delta\hspace{0pt}v_{o}}/v_{o}} = {- {{\Delta\hspace{0pt}\mathcal{D}}/\mathcal{D}}}`$ by Equation ([18](#S2.E18 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")).

## 3 Data

The QLF measurements used in this work are from Palanque-Delabrouille et al. ([2016](#bib.bib34)) (PD2016 hereafter) and include 13876 quasars: 7900 were identified through flux variability (Schmidt et al., [2010](#bib.bib40); Palanque-Delabrouille et al., [2011](#bib.bib33)) in the Sloan Digital Sky Survey (SDSS) photometric $`\{ u,g,r,i,z\}`$ bands and 5976 were spectroscopically identified in the Stripe 82 region. The data is part of the extended Baryon Oscillation Spectroscopic Survey (eBOSS) in SDSS-IV and is available in Table A.1 of [PD2016](#bib.bib34). The magnitudes have been corrected for Galactic extinction with a magnitude limit of $`g_{dered} = 22.5`$ in the dereddened $`g`$ band.

### 3.1 Theoretical modelling

The number counts of quasars have two main sources of uncertainty: the bright end has a small number of objects, while the faint end is limited by the survey strategy. The large uncertainties on both ends of the luminosity function leads us to the adoption of a double power-law fit for the comoving space density of quasars (Boyle et al., [2000](#bib.bib9); Croom et al., [2004](#bib.bib15), [2009](#bib.bib16); Ross et al., [2013](#bib.bib38))

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{\Phi\hspace{0pt}{(z,M_{g})}} = \frac{\Phi_{\ast}}{10^{0.4\hspace{0pt}{({a + 1})}\hspace{0pt}{({M_{g} - M_{\ast}})}} + 10^{0.4\hspace{0pt}{({b + 1})}\hspace{0pt}{({M_{g} - M_{\ast}})}}}}.
``` |  | (19) |

Above, $`\Phi_{\ast}`$ is a normalisation factor related to the characteristic number density of quasars, $`M_{\ast}`$ is the break magnitude, associated with their characteristic luminosity, $`a`$ and $`b`$ describe, respectively, the behaviour of the bright and faint ends, and

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{M_{g}\hspace{0pt}{(z)}} = {m_{g} - {\mu\hspace{0pt}{(z)}} - {\lbrack{{K\hspace{0pt}{(z)}} - {K\hspace{0pt}{({z = 2})}}}\rbrack}}
``` |  | (20) |

is the absolute magnitude corresponding to the observed apparent magnitude $`m_{g}`$ in the $`g`$ band after the $`K`$-correction

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{K\hspace{0pt}{(z)}} = {- {2.5\hspace{0pt}{({1 + \alpha_{v}})}\hspace{0pt}{\log_{10}{({1 + z})}}}}},{\alpha_{v} \simeq {- 0.5}}
``` |  | (21) |

has been applied ( [\al@croom2009,palanque2016](#bib.bib16); [\al@croom2009,palanque2016](#bib.bib34), ).

The number density of quasars has a strong redshift dependence and the QLF may evolve in different ways: in the pure-luminosity evolution (PLE) case, the characteristic number density remains constant, but the break magnitude evolves over time; in the pure-density evolution (PDE), the luminosity of individual sources remain constant, while the number density varies with time; or it could be combination of both luminosity and density evolution (LEDE). Redshift dependence may also impact the bright- and faint-end slopes.

Since the QLF is very uncertain, this work aims at investigating different evolution models for $`\Phi_{\ast},M_{\ast},a`$ and $`b`$, and to assess their impact on the kinematic dipole $`\mathcal{D}`$. The models are described in detail in Appendix [A](#A1 "Appendix A Quasar luminosity function models ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"). They are ([\al@croom2004,ross2013,palanque2016](#bib.bib15); [\al@croom2004,ross2013,palanque2016](#bib.bib38); [\al@croom2004,ross2013,palanque2016](#bib.bib34)):

- •
  PLE – the QLF redshift evolution comes from the break magnitude $`M_{\ast}`$, with the pivot scale $`z_{p} = 2.2`$ allowing the bright and faint slopes to vary for low ($`z < z_{p}`$) and high ($`z > z_{p}`$) redshifts.
- •
  LEDE – both luminosity and number density vary with time. We consider a linear (LEDE₇) and quadratic (LEDE₈) redshift dependence on the break magnitude $`M_{\ast}\hspace{0pt}{(z)}`$ and their extension, in which both slopes evolve linearly with redshift (LEDE₇₊₂ and LEDE₈₊₂).
- •
  PLE+LEDE – it combines the PLE functional form for $`z < z_{p}`$, and the LEDE₇ model for $`z > z_{p}`$. We allow the bright slope $`a`$ to evolve with redshift.

## 4 Method

### 4.1 Maximum likelihood

We consider a maximum-likelihood approach to sample the parameter space of the models of Section [3.1](#S3.SS1 "3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"). Because the QLF is estimated from the number counts of quasars inside each magnitude bin, its error can be approximately modelled by Poissonian error bars. Therefore, we model the log-likelihood function as (Pozzetti et al., [2016](#bib.bib36))

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{\ln\mathcal{L}} = {\sum\limits_{i,j}\frac{\Delta_{i,j}^{2}}{\sigma_{i,j}^{2}}}},
``` |  | (22) |

In this equation, $`\Delta_{i,j}^{2} \equiv {{1 - {{{\Phi_{\theta}\hspace{0pt}{(z_{i},M_{j})}}/\Phi_{obs}}\hspace{0pt}{(z_{i},M_{j})}}} + {\ln{\lbrack{{{\Phi_{\theta}\hspace{0pt}{(z_{i},M_{j})}}/\Phi_{obs}}\hspace{0pt}{(z_{i},M_{j})}}\rbrack}}}`$, $`\sigma_{i,j}^{2} = {{1/N_{obs}}\hspace{0pt}{(z_{i},M_{j})}}`$, $`\Phi_{\theta}`$ is given by the model of Equation ([19](#S3.E19 "In 3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). Finally, $`\Phi_{obs}`$ and $`N_{obs}`$ are, respectively, the corresponding QLF measurements and the angle averaged number counts in each redshift $`z_{i}`$ and magnitude $`M_{j}`$ bin. To maximise Equation ([22](#S4.E22 "In 4.1 Maximum likelihood ‣ 4 Method ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), we used the MCMC sampler emcee (Foreman-Mackey et al., [2013](#bib.bib24)) with flat priors for the fitting parameters.

### 4.2 Goodness of fit

To assess the goodness of fit of the models considered in Section [3.1](#S3.SS1 "3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") to the eBOSS data, in Table LABEL:tab:MCMCresults we present the Bayesian information criterion (BIC)

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{BIC} = {{p\hspace{0pt}{\ln{(n)}}} - {2\hspace{0pt}{\ln\mathcal{L}^{\ast}}}}},
``` |  | (23) |

in which $`p`$ is the number of parameters in the model, $`n`$ is the number of data points used, and $`\ln\mathcal{L}^{\ast}`$ is the log-likelihood computed at the best-fitting values (e.g. Liddle, [2004](#bib.bib29)).

## 5 Results

![Refer to caption](/html/2212.04925/assets/figures/derived_quantities.png)

Figure 1: Comoving number density (top), evolution (middle) and magnification (bottom) biases derived from the QLF with different absolute magnitude thresholds, assuming Planck 2015 cosmology (Ade et al., [2016](#bib.bib1)). The pivot redshift, $`z_{p} = 2.2`$, entering the PLE and PLE+LEDE models is indicated by the vertical dashed lines.

|  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|
|  | $`\mathcal{D}_{kin}\hspace{0pt}{\lbrack 10^{- 3}\rbrack}`$ |  |  |  |  |  |
|  | $`w`$CDM | Planck 2015 |  |  | $`+ {\{ b_{e}^{eff},s^{eff}\}}`$ |  |
|  | $`M_{c,g} = {- 25}`$ | $`M_{c,g} = {- 25}`$ | $`M_{c,g} = {- 24.6}`$ | $`M_{c,g} = {M_{c,g}\hspace{0pt}{(z)}}`$ | $`M_{c,g} = {- 25}`$ |  |
|  |  |  |  |  | Best fit | Chains |
| PLE | $`0.72_{- 0.10}^{+ 0.10}`$ | $`0.75_{- 0.10}^{+ 0.10}`$ | $`1.42_{- 0.10}^{+ 0.10}`$ | $`0.39_{- 0.05}^{+ 0.05}`$ | $`0.81`$ | $`0.84_{- 0.12}^{+ 0.12}`$ |
| LEDE₇ | $`0.32_{- 0.03}^{+ 0.03}`$ | $`0.34_{- 0.03}^{+ 0.03}`$ | $`0.90_{- 0.04}^{+ 0.04}`$ | $`0.53_{- 0.06}^{+ 0.06}`$ | $`0.49`$ | $`0.28_{- 0.04}^{+ 0.04}`$ |
| LEDE₇₊₂ | $`0.59_{- 0.06}^{+ 0.06}`$ | $`0.62_{- 0.06}^{+ 0.06}`$ | $`1.14_{- 0.06}^{+ 0.06}`$ | $`0.67_{- 0.05}^{+ 0.05}`$ | $`0.72`$ | $`0.75_{- 0.07}^{+ 0.07}`$ |
| LEDE₈ | $`0.39_{- 0.03}^{+ 0.03}`$ | $`0.42_{- 0.03}^{+ 0.03}`$ | $`1.01_{- 0.04}^{+ 0.04}`$ | $`0.44_{- 0.06}^{+ 0.06}`$ | $`0.54`$ | $`0.38_{- 0.04}^{+ 0.04}`$ |
| LEDE₈₊₂ | $`0.40_{- 0.05}^{+ 0.03}`$ | $`0.42_{- 0.05}^{+ 0.03}`$ | $`0.99_{- 0.04}^{+ 0.04}`$ | $`0.50_{- 0.07}^{+ 0.07}`$ | $`0.53`$ | $`0.42_{- 0.07}^{+ 0.05}`$ |
| PLE+LEDE | $`0.60_{- 0.08}^{+ 0.08}`$ | $`0.62_{- 0.08}^{+ 0.08}`$ | $`1.31_{- 0.09}^{+ 0.08}`$ | $`0.47_{- 0.05}^{+ 0.05}`$ | $`0.72`$ | $`0.77_{- 0.11}^{+ 0.11}`$ |

Table 1: 1$`\sigma`$ constraints for $`\mathcal{D}_{kin} = {{\mathcal{D}\hspace{0pt}v_{o}^{CMB}}/c}`$, in $`10^{- 3}`$ units, obtained from a random subset of 5000 MCMC samples, with Planck 2015 and $`w`$CDM cosmologies. We present the results with different absolute magnitude thresholds of $`- 25`$, $`- 24.6`$ and a varying threshold $`M_{c}\hspace{0pt}{(z)}`$. The effect of neglecting the redshift evolution of the magnification and evolution biases (Equation [24](#S5.E24 "In 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) is shown in the last column for the Planck cosmology and absolute magnitude threshold $`M_{c,g} = {- 25}`$.

![Refer to caption](/html/2212.04925/assets/figures/dipoles.png)

Figure 2: 1$`\sigma`$ constraints for $`\mathcal{D}_{kin} = {\mathcal{D},{v_{o}^{CMB}/c}}`$ obtained from the same random subset of 5000 MCMC samples used to derive the constraints of Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") with the $`\Lambda`$CDM cosmology. The grey dashed line is a reference for the intrinsic dipole (Dalang & Bonvin, [2022](#bib.bib17)). The top panel shows the analysis for a varying magnitude threshold $`M_{c}\hspace{0pt}{(z)}`$, i.e., the corresponding absolute magnitude for $`g = 22.5`$ at each redshift used to compute $`n\hspace{0pt}{(z)}`$. The second and third panels have a fixed threshold given by the last redshift bin (see Figure [3](#A2.F3 "Figure 3 ‣ Appendix B MCMC results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") for illustration), and the last panel neglects the last bin (i.e., $`M_{c}\hspace{0pt}{(z)}`$ corresponds to $`g = 22.5`$ at $`z = 3.25`$). The third panel shows the constraints after neglecting the redshift evolution (Equation [24](#S5.E24 "In 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) with $`s\hspace{0pt}{(z)}`$ and $`b_{e}\hspace{0pt}{(z)}`$ projected in redshift for each MCMC sample (points with error bars), and for projections using the best-fitting functions (dashed lines). Neglecting redshift evolution (third panel) causes significant tensions in some cases, and neglecting data from the last redshift bin shifts the amplitude mainly due to the different redshift range used to obtain $`\mathcal{D}`$ (last panel). Different QLF models can lead to a $`\sim {3\hspace{0pt}\sigma}`$ tension in $`\mathcal{D}`$.

We fitted the QLF models described in Appendix [A](#A1 "Appendix A Quasar luminosity function models ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") to the eBOSS data. In Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), we show the mean of random subsets of 5000 samples, obtained from the complete sets of $`\gtrsim`$ 300000 MCMC chains used to derive the constraints on the QLF models, and the 1$`\sigma`$ constraints obtained for the amplitude of the kinematic dipole from those subsets.⁴⁴4We checked the constraints against different subsamples, and also by varying the number of MCMC chains in the subsets, finding no significant changes. The posteriors for the QLF are presented in Table LABEL:tab:MCMCresults, together with their goodness of fit.

In Figure [1](#S5.F1 "Figure 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), the number density, evolution and magnification biases derived from the best-fitting QLF are shown, after assuming a flat $`\Lambda`$CDM model with parameters $`{h = 0.679},{{n_{s} = 0.9681},{{\sigma_{8} = 0.8154},{\Omega_{m} = 0.3065}}}`$ and $`{\Omega_{b}\hspace{0pt}h^{2}} = 0.02227`$ (Ade et al., [2016](#bib.bib1)) (same cosmology employed in [PD2016](#bib.bib34) for the distance modulus $`\mu`$). To assess the dependence of our results on the fiducial cosmology used to compute Equation ([14](#S2.E14 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), in Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") the constraints obtained with a flat $`w`$CDM cosmology with parameters $`{h = 0.72},{{n_{s} = 0.963},{{\sigma_{8} = 0.852},{{\Omega_{m} = 0.275},{{{\Omega_{b}\hspace{0pt}h^{2}} = 0.02258},{\Omega_{DE} = 0.725}}}}}`$, and $`w = {- 1.2}`$ (Rasera et al., [2022](#bib.bib37)) are also shown. The constraints are robust against this change as seen in Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole").

We considered different absolute magnitude thresholds: $`{M_{c,g}\hspace{0pt}{(z_{8})}} = {- 25}`$, corresponding to the apparent magnitude cut $`g_{dered} = 22.5`$ at the centre $`z_{8} = 3.75`$ of the last redshift bin (e.g. Wang et al., [2020](#bib.bib49)), $`{M_{c,g}\hspace{0pt}{(z_{7})}} = {- 24.6}`$, corresponding to $`g_{dered}`$ at $`z_{7} = 3.25`$, and a varying $`M_{c,g}\hspace{0pt}{(z)}`$ obtained from Equation ([20](#S3.E20 "In 3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) at each redshift used to compute $`{n\hspace{0pt}{(z,M_{c})}},{b_{e}\hspace{0pt}{(z,M_{c})}}`$, and $`s\hspace{0pt}{(z,M_{c})}`$.

In Figure [2](#S5.F2 "Figure 2 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), we show the impact of the magnitude cut $`M_{c}`$ on the results (first, second and fourth panels). By fixing a threshold such as $`M_{c} = {- 25}`$, we discard data at lower redshifts (see Figure [3](#A2.F3 "Figure 3 ‣ Appendix B MCMC results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). Hence, the number density increases by allowing larger values for $`M_{c}`$. By considering $`M_{c} = {- 24.6}`$, we neglect the last redshift bin in the analysis to avoid adding sources too faint to be seen at the highest $`z`$ bin. Thus, we are probing the kinematic dipole with lower-$`z`$ sources and this shifts $`\mathcal{D}`$ to larger values as the integration range changes (fourth panel of Figure [2](#S5.F2 "Figure 2 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). Most of the impact comes from the $`\mathcal{D}_{cosmo}`$ term.

The evolution bias derived from the LEDE₇, LEDE₈ and LEDE₈₊₂ models are very similar at high redshifts. While the evolution bias is rather consistent among the different models, the magnification bias has a larger dependence on the QLF. However, the contributions coming from Equation ([16](#S2.E16 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) exhibit a smaller dispersion due to the $`{1/r}\hspace{0pt}\mathcal{H}`$ suppression in $`\mathcal{D}_{mag}`$. The contributions from the evolution term (Equation [17](#S2.E17 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) are more sensitive to the differences in the QLF. This is manifested in the final 2D amplitude $`\mathcal{D}`$: LEDE₇, LEDE₈ and LEDE₈₊₂ agree within $`\sim {1\hspace{0pt}\sigma}`$, while the LEDE₇₊₂, PLE and PLE+LEDE have $`\sim {1\hspace{0pt}\sigma}`$ consistency among each other.

We also tested the robustness of our results against the minimum redshift considered; the values presented in Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") were obtained by numerically integrating Equations ([15](#S2.E15 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) – ([17](#S2.E17 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) from $`z_{\min} \approx 0`$ to $`z_{\max} = 4`$ (the maximum redshift of the sample). Because the number of sources below $`z = 0.68`$ (minimum redshift) is very small, we find no significant changes in the constraints for the dipole amplitude. Still, it is worth stressing the fact that the intrinsic dipole is considerable for low-$`z`$ sources; therefore, in principle it is better to remove $`z \lesssim 0.5`$ to avoid nonlinear contamination in real data (Tiwari & Nusser, [2016](#bib.bib46); Bengaly et al., [2019](#bib.bib5)).

Finally, considering the Planck 2015 cosmology and $`M_{c,g} = {- 25}`$, we neglect the redshift evolution of the magnification and evolution biases, i.e.,

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{s^{eff} = {\int{{dz}\hspace{0pt}f\hspace{0pt}{(z)}\hspace{0pt}s\hspace{0pt}{(z)}\hspace{0pt}\text{~and~}\hspace{0pt}b_{e}^{eff}}} = {\int{{dz}\hspace{0pt}f\hspace{0pt}{(z)}\hspace{0pt}b_{e}\hspace{0pt}{(z)}}}}.
``` |  | (24) |

This has been done for (a) each one of the 5000 random chains, i.e., calculating $`f\hspace{0pt}{(z)}`$, $`s\hspace{0pt}{(z)}`$ and $`b_{e}\hspace{0pt}{(z)}`$ for each chain and then projecting along $`z`$, and (b) by fixing them to the best-fitting functions. In the first case, the results agree within the 1$`\sigma`$ errors; for the second case, however, the amplitudes are overestimated by more than 3$`\sigma`$ for the LEDE₇ and LEDE₈ models.

Our results summarise the expected dipole in the number counts of quasars from a pure kinematic origin. They are all above the intrinsic dipole expected from the clustering anisotropy (grey dashed line in Figure [2](#S5.F2 "Figure 2 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), which comes from fluctuations in the number density around the FLRW metric (Nadolny et al., [2021](#bib.bib32)) and can be distinguished given the 1$`\sigma`$ errors.

## 6 Conclusions

In this work, we extended the analysis of Dalang & Bonvin ([2022](#bib.bib17)) to include uncertainties on the magnification $`s`$ and evolution $`b_{e}`$ biases (Equations [9](#S2.E9 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") and [2](#S2.Ex1 "2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). We used a maximum-likelihood approach on six QLF models (Equation [19](#S3.E19 "In 3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) to derive the number density $`n\hspace{0pt}{(z)}`$ (Equation [7](#S2.E7 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), $`s\hspace{0pt}{(z)}`$, and $`b_{e}\hspace{0pt}{(z)}`$ (Wang et al., [2020](#bib.bib49)), finding that models of QLF have a non-negligible impact on the biases (Figure [1](#S5.F1 "Figure 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). From MCMC chains, we calculated the kinematic dipole amplitude $`\mathcal{D}`$ (Equation [14](#S2.E14 "In 2 Correction to the standard dipole ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), finding that the QLF introduces a large model dependence on $`\mathcal{D}`$. In some cases, a $`\sim {3\hspace{0pt}\sigma}`$ tension on the dipole amplitude computed with different models (Figure [2](#S5.F2 "Figure 2 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) is found.

The constraints on $`\mathcal{D}`$ are robust to a change of cosmology (see Table [1](#S5.T1 "Table 1 ‣ 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). Neglecting data from the last redshift bin does not degrade the constraints, as the $`1\hspace{0pt}\sigma`$ error bars on $`\mathcal{D}`$ are of the same order, although the amplitude increases with higher magnitude thresholds $`M_{c,g}`$, mainly due to a different redshift range.

We analysed the impact of neglecting the redshift evolution of the magnification and evolution biases by computing their effective values, $`s^{eff}`$ and $`b_{e}^{eff}`$ (Equation [24](#S5.E24 "In 5 Results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")). When these parameters were computed for each MCMC sample, we found a shift in the expected amplitude within the $`1\hspace{0pt}\sigma`$ region of the cases in which the redshift evolution was accounted for; by obtaining them from the best-fitting functions, we found an overestimation of the 2D amplitude $`\mathcal{D}`$ when redshift evolution was neglected in quantities that evolve with $`z`$.

The QLF introduces a model dependence in the dipole amplitude, contrasting significantly with the model-independent null hypothesis of Ellis & Baldwin ([1984](#bib.bib22)) for testing the validity of the CP. As we have shown, with no single description of quasar evolution and multiple models fitting the data equally well, testing the CP with the quasar dipole is complex and not as straightforward as assumed. The difficulty is exacerbated for radio continuum catalogues, which have no redshift information.

Finally, conclusions from our work cannot be directly extrapolated to the results of Secrest et al. ([2022](#bib.bib41)), given the different wavelengths, magnitude thresholds and redshift information for the CatWISE2020 and eBOSS catalogues. In addition, following Dalang & Bonvin ([2022](#bib.bib17)), we used an alternative theoretical approach to Secrest et al. ([2022](#bib.bib41)) (and to previous work on radio continuum samples) – which is based on the evolution bias of the sample, rather than on the spectral indices of individual sources. We find, consistent with Dalang & Bonvin ([2022](#bib.bib17)), that the magnification and evolution biases vary with redshift for all the models considered for the eBOSS quasars, with significant implications for the dipole amplitude. Therefore it is paramount to investigate whether evolution effects are also present in the data considered for the analyses leading to Equation ([5](#S1.E5 "In 1 Introduction ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")), before strong claims about violation of the CP are validated.

## Acknowledgements

We thank Charles Dalang, Carolina Queiroz, José Luis Bernal, Mike Shengbo Wang, and Phil Bull for discussions and comments. C.G. and C.C. are supported by the UK Science & Technology Facilities Council consolidated grant ST/T000341/1. J.P. received support from the French government under the France 2030 investment plan, as part of the Initiative d’Excellence d’Aix-Marseille Université -A\*MIDEX (AMX-19-IET-008). R.M. is supported by the South African Radio Astronomy Observatory and the National Research Foundation (grant No. 75415).

## Appendix A Quasar luminosity function models

In this appendix we describe the QLF models considered for the main analysis.

### A.1 Pure-Luminosity Evolution (PLE)

In this model, the luminosity function evolves solely through a redshift evolution in the break magnitude $`M_{\ast}`$ ([\al@boyle2000, palanque2016](#bib.bib9); [\al@boyle2000, palanque2016](#bib.bib34))

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{M_{\ast}\hspace{0pt}{(z)}} = {{M_{\ast}\hspace{0pt}{(z_{p})}} - {2.5\hspace{0pt}\left\lbrack {{k_{1}\hspace{0pt}{({z - z_{p}})}} + {k_{2}\hspace{0pt}{({z - z_{p}})}^{2}}} \right\rbrack}}}.
``` |  | (A1) |

Early studies showed a preference for this model, but deeper data revealed deviations from the PLE model at redshifts larger than $`z \gtrsim 2`$ (Ross et al., [2013](#bib.bib38)) for $`z_{p} = 0`$. Therefore, the bright- and faint-end slopes are allowed to vary for low ($`z < z_{p}`$) and high ($`z > z_{p}`$) redshifts, $`a_{\{ l,h\}},b_{\{ l,h\}}`$, and so are $`k_{1}`$ and $`k_{2}`$. The pivot redshift is $`z_{p} = 2.2`$. In this model, there are 10 free parameters: $`\{ a_{l},b_{l},a_{h},b_{h},k_{1,l},k_{2,l},k_{1,h},k_{2,h},{M_{\ast}\hspace{0pt}{(z_{p})}},{\log_{10}\Phi_{\ast}}\}`$ ([PD2016](#bib.bib34)).

### A.2 Luminosity and Density Evolution (LEDE)

We also consider the case in which both the luminosity and number density vary with time: the so-called Luminosity Evolution + Density Evolution (LEDE) model ([\al@ross2013, palanque2016](#bib.bib38); [\al@ross2013, palanque2016](#bib.bib34)). We analyse this particular case motivated by the fact that the discontinuity present at the pivot redshift $`z_{p} = 2.2`$ for the derived quantities is removed, thus leading to a smooth behaviour with redshift for the derived quantities.

We will take the density and magnitude evolution, respectively, as

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{{\log_{10}\Phi_{\ast}}\hspace{0pt}{(z)}} = {{{\log_{10}\Phi_{\ast}}\hspace{0pt}{(z_{p})}} + {c_{1}\hspace{0pt}{({z - z_{p}})}} + {c_{2}\hspace{0pt}{({z - z_{p}})}^{2}}}},
``` |  | (A2) |

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{M_{\ast}\hspace{0pt}{(z)}} = {{M_{\ast}\hspace{0pt}{(z_{p})}} + {c_{3}\hspace{0pt}{({z - z_{p}})}}}}.
``` |  | (A3) |

Because it has seven free parameters $`\{ a,b`$, $`c_{1},c_{2},c_{3}`$, $`{\log_{10}\Phi_{\ast}}\hspace{0pt}{(z_{p})}`$, $`M_{\ast}{(z_{p})}\}`$, we will label it LEDE₇ hereafter. Notice that, in this case, the bright and faint ends are fixed for the whole redshift range: $`{a\hspace{0pt}{(z)}} = a`$, $`{b\hspace{0pt}{(z)}} = b`$.

We shall also consider a quadratic redshift dependence for the break magnitude (Boyle et al., [2000](#bib.bib9); Croom et al., [2009](#bib.bib16))

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{M_{\ast}\hspace{0pt}{(z)}} = {{M_{\ast}\hspace{0pt}{(z_{p})}} + {c_{3}\hspace{0pt}{({z - z_{p}})}} + {c_{4}\hspace{0pt}{({z - z_{p}})}^{2}}}}.
``` |  | (A4) |

We dub this case LEDE₈ as it contains eight free parameters: $`\{ a,b`$, $`c_{1},c_{2},c_{3},c_{4}`$, $`{\log_{10}\Phi_{\ast}}\hspace{0pt}{(z_{p})}`$, $`M_{\ast}{(z_{p})}\}`$. Again, $`a`$ and $`b`$ are fixed.

Finally, we consider extensions to LEDE₇ and LEDE₈, in which both slopes are allowed to evolve linearly with redshift

|  |  |  |  |
|----|----|----|----|
|  | 
``` math
{{{a\hspace{0pt}{(z)}} = {{a\hspace{0pt}{(z_{p})}} + {c_{a}\hspace{0pt}{({z - z_{p}})}}}},{{b\hspace{0pt}{(z)}} = {{b\hspace{0pt}{(z_{p})}} + {c_{b}\hspace{0pt}{({z - z_{p}})}}}}}.
``` |  | (A5) |

These will be called, respectively, LEDE₇₊₂ and LEDE₈₊₂, where the $`+ 2`$ indicates the extra $`c_{a}`$ and $`c_{b}`$ parameters introduced for the redshift evolution in the bright and faint ends.

### A.3 Composite Evolution Model (PLE+LEDE)

We also consider the combination of the PLE functional form for $`z < z_{p}`$, and the LEDE₇ model for $`z > z_{p}`$ ([PD2016](#bib.bib34)). Other combinations with LEDE₈ and redshift dependencies of the slopes are possible. However, we focus on this particular combination of PLE + LEDE₇ and allow only the bright slope to evolve with redshift: $`{\alpha\hspace{0pt}{(z)}} = {{\alpha\hspace{0pt}{(z_{p})}} + {c_{a}\hspace{0pt}{({z - z_{p}})}}}`$. In total, we have 10 free parameters: $`\{{\alpha\hspace{0pt}{(z_{p})}},\beta,k_{1},k_{2},c_{1},c_{2},c_{3},c_{a},{{\log_{10}\Phi_{\ast}}\hspace{0pt}{(z_{p})}},{M_{\ast}\hspace{0pt}{(z_{p})}}\}`$.

## Appendix B MCMC results

In Table LABEL:tab:MCMCresults, we present the best-fitting values for the QLF obtained for the six models described in Section [3.1](#S3.SS1 "3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), from the MCMC analysis discussed in Section [4.1](#S4.SS1 "4.1 Maximum likelihood ‣ 4 Method ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole") by considering all eight redshift bins, removing the last bin and neglecting data from the last two bins.

There is a small impact in removing high-redshift information from the MCMC analysis for the LEDE₇ model: the parameters $`c_{1}`$, $`c_{2}`$ and $`c_{3}`$ show a slight degradation in the constraints, while the best-fitting values are in agreement within the $`2\hspace{0pt}\sigma`$ region. The same happens for the PLE analysis, except for the high-redshift parameters, which are severely degraded with the removal of the last two redshift bins (i.e., considering data between $`0.68 < z < 3.5`$), as it is expected since there is no constraining power at the high-$`z`$ end, even though the fits are improved due to the fact that there are more data at low $`z`$ (which can also be seen by the decrease in the BIC value). We can also observe larger errors for the PLE+LEDE $`c_{1},c_{2},c_{3}`$ and $`c_{a}`$ parameters, with a $`2\hspace{0pt}\sigma`$ disagreement for $`c_{1}`$ and $`c_{2}`$ after removing the last two redshift bins (considering only the data between $`0.68 < z < 3`$). On the other hand, LEDE₈, LEDE₈₊₂ and LEDE₇₊₂ are reasonably insensitive to high-redshift information, showcasing similar constraints for the three redshift ranges considered.

In Figure [3](#A2.F3 "Figure 3 ‣ Appendix B MCMC results ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), we show the best-fitting QLF from the corresponding MCMC analysis with all eight redshift bins. Dashed lines mark the limiting magnitude cut $`M_{c} = {- 25}`$ for the last redshift bin. When we consider all redshift bins and a constant magnitude cut for the dipole analysis, we neglect data contained in the shaded region. Discarding higher bins in the dipole analysis is equivalent to moving the dashed line and region of exclusion to fainter magnitudes.

| Model | $`\mathbf{z}`$-range | Parameters |  |  |  |  |  |  | BIC |
|----|----|----|----|----|----|----|----|----|----|
| PLE | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 176.97 |
|  | $`0.68 - 4.0`$ | $`- 25.86_{- 0.12}^{+ 0.12}`$ | $`- 5.63_{- 0.05}^{+ 0.04}`$ |  |  |  |  |  |  |
|  |  | $`a_{l}`$ | $`b_{l}`$ | $`k_{1,l}`$ | $`k_{2,l}`$ |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 3.09_{- 0.10}^{+ 0.09}`$ | $`- 1.31_{- 0.04}^{+ 0.04}`$ | $`- 0.16_{- 0.05}^{+ 0.05}`$ | $`- 0.42_{- 0.04}^{+ 0.04}`$ |  |  |  |  |
|  |  | $`a_{h}`$ | $`b_{h}`$ | $`k_{1,h}`$ | $`k_{2,h}`$ |  |  |  |  |
|  | $`2.2 - 4.0`$ | $`- 2.46_{- 0.05}^{+ 0.05}`$ | $`- 1.10_{- 0.06}^{+ 0.07}`$ | $`- 0.42_{- 0.07}^{+ 0.07}`$ | $`0.01_{- 0.05}^{+ 0.05}`$ |  |  |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 171.88 |
|  | $`0.68 - 3.5`$ | $`- 25.82_{- 0.12}^{+ 0.13}`$ | $`- 5.62_{- 0.05}^{+ 0.05}`$ |  |  |  |  |  |  |
|  |  | $`a_{l}`$ | $`b_{l}`$ | $`k_{1,l}`$ | $`k_{2,l}`$ |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 3.06_{- 0.10}^{+ 0.09}`$ | $`- 1.29_{- 0.04}^{+ 0.04}`$ | $`- 0.16_{- 0.05}^{+ 0.05}`$ | $`- 0.42_{- 0.04}^{+ 0.04}`$ |  |  |  |  |
|  |  | $`a_{h}`$ | $`b_{h}`$ | $`k_{1,h}`$ | $`k_{2,h}`$ |  |  |  |  |
|  | $`2.2 - 3.5`$ | $`- 2.43_{- 0.06}^{+ 0.05}`$ | $`- 1.09_{- 0.07}^{+ 0.07}`$ | $`- 0.47_{- 0.10}^{+ 0.10}`$ | $`0.05_{- 0.08}^{+ 0.08}`$ |  |  |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 151.77 |
|  | $`0.68 - 3.0`$ | $`- 25.63_{- 0.14}^{+ 0.14}`$ | $`- 5.53_{- 0.06}^{+ 0.05}`$ |  |  |  |  |  |  |
|  |  | $`a_{l}`$ | $`b_{l}`$ | $`k_{1,l}`$ | $`k_{2,l}`$ |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 2.93_{- 0.10}^{+ 0.09}`$ | $`- 1.21_{- 0.05}^{+ 0.05}`$ | $`- 0.09_{- 0.06}^{+ 0.06}`$ | $`- 0.37_{- 0.04}^{+ 0.04}`$ |  |  |  |  |
|  |  | $`a_{h}`$ | $`b_{h}`$ | $`k_{1,h}`$ | $`k_{2,h}`$ |  |  |  |  |
|  | $`2.2 - 3.0`$ | $`- 2.37_{- 0.07}^{+ 0.07}`$ | $`- 0.96_{- 0.10}^{+ 0.12}`$ | $`- 1.03_{- 0.26}^{+ 0.24}`$ | $`0.90_{- 0.33}^{+ 0.34}`$ |  |  |  |  |
| PLE+LEDE | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ | $`\alpha\hspace{0pt}{(z_{p})}`$ | $`\beta`$ |  |  |  | 175.50 |
|  | $`0.68 - 4.0`$ | $`- 25.52_{- 0.16}^{+ 0.16}`$ | $`- 5.48_{- 0.06}^{+ 0.06}`$ | $`- 2.80_{- 0.09}^{+ 0.08}`$ | $`- 1.16_{- 0.06}^{+ 0.06}`$ |  |  |  |  |
|  |  | $`k_{1}`$ | $`k_{2}`$ |  |  |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 0.10_{- 0.05}^{+ 0.05}`$ | $`- 0.38_{- 0.04}^{+ 0.04}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{a}`$ |  |  |  |  |
|  | $`2.2 - 4.0`$ | $`- 0.62_{- 0.06}^{+ 0.06}`$ | $`- 0.06_{- 0.05}^{+ 0.05}`$ | $`- 0.67_{- 0.13}^{+ 0.13}`$ | $`- 0.01_{- 0.11}^{+ 0.11}`$ |  |  |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ | $`\alpha\hspace{0pt}{(z_{p})}`$ | $`\beta`$ |  |  |  | 163.30 |
|  | $`0.68 - 3.5`$ | $`- 25.53_{- 0.16}^{+ 0.16}`$ | $`- 5.47_{- 0.06}^{+ 0.06}`$ | $`- 2.80_{- 0.09}^{+ 0.09}`$ | $`- 1.15_{- 0.06}^{+ 0.07}`$ |  |  |  |  |
|  |  | $`k_{1}`$ | $`k_{2}`$ |  |  |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 0.08_{- 0.05}^{+ 0.05}`$ | $`- 0.37_{- 0.04}^{+ 0.04}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{a}`$ |  |  |  |  |
|  | $`2.2 - 3.5`$ | $`- 0.73_{- 0.07}^{+ 0.07}`$ | $`0.07_{- 0.07}^{+ 0.07}`$ | $`- 0.69_{- 0.15}^{+ 0.16}`$ | $`0.09_{- 0.13}^{+ 0.13}`$ |  |  |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ | $`\alpha\hspace{0pt}{(z_{p})}`$ | $`\beta`$ |  |  |  | 140.38 |
|  | $`0.68 - 3.0`$ | $`- 25.58_{- 0.16}^{+ 0.16}`$ | $`- 5.47_{- 0.06}^{+ 0.06}`$ | $`- 2.84_{- 0.10}^{+ 0.09}`$ | $`- 1.16_{- 0.06}^{+ 0.07}`$ |  |  |  |  |
|  |  | $`k_{1}`$ | $`k_{2}`$ |  |  |  |  |  |  |
|  | $`0.68 - 2.2`$ | $`- 0.03_{- 0.06}^{+ 0.06}`$ | $`- 0.34_{- 0.04}^{+ 0.04}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{a}`$ |  |  |  |  |
|  | $`2.2 - 3.0`$ | $`- 1.08_{- 0.12}^{+ 0.12}`$ | $`0.68_{- 0.18}^{+ 0.18}`$ | $`- 0.77_{- 0.26}^{+ 0.28}`$ | $`0.36_{- 0.22}^{+ 0.22}`$ |  |  |  |  |
| LEDE₇ | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 169.05 |
|  | $`0.68 - 4.0`$ | $`- 25.37_{- 0.16}^{+ 0.16}`$ | $`- 5.46_{- 0.05}^{+ 0.05}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |
|  |  | $`- 2.66_{- 0.08}^{+ 0.07}`$ | $`- 1.03_{- 0.07}^{+ 0.08}`$ | $`- 0.43_{- 0.01}^{+ 0.01}`$ | $`- 0.30_{- 0.01}^{+ 0.01}`$ | $`- 0.96_{- 0.04}^{+ 0.04}`$ |  |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 159.27 |
|  | $`0.68 - 3.5`$ | $`- 25.40_{- 0.16}^{+ 0.16}`$ | $`- 5.47_{- 0.06}^{+ 0.05}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |
|  |  | $`- 2.67_{- 0.08}^{+ 0.07}`$ | $`- 1.04_{- 0.07}^{+ 0.08}`$ | $`- 0.44_{- 0.01}^{+ 0.01}`$ | $`- 0.30_{- 0.01}^{+ 0.01}`$ | $`- 0.97_{- 0.04}^{+ 0.04}`$ |  |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 130.82 |
|  | $`0.68 - 3.0`$ | $`- 25.61_{- 0.17}^{+ 0.17}`$ | $`- 5.54_{- 0.06}^{+ 0.06}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |
|  |  | $`- 2.74_{- 0.09}^{+ 0.08}`$ | $`- 1.10_{- 0.07}^{+ 0.07}`$ | $`- 0.49_{- 0.02}^{+ 0.02}`$ | $`- 0.32_{- 0.01}^{+ 0.01}`$ | $`- 1.08_{- 0.04}^{+ 0.04}`$ |  |  |  |
| LEDE₇₊₂ | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 144.39 |
|  | $`0.68 - 4.0`$ | $`- 26.17_{- 0.16}^{+ 0.16}`$ | $`- 5.77_{- 0.07}^{+ 0.06}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |  |  |
|  |  | $`- 0.94_{- 0.06}^{+ 0.06}`$ | $`- 0.4_{- 0.02}^{+ 0.02}`$ | $`- 1.99_{- 0.1}^{+ 0.1}`$ |  |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.84_{- 0.09}^{+ 0.08}`$ | $`- 0.16_{- 0.07}^{+ 0.07}`$ | $`- 1.33_{- 0.05}^{+ 0.05}`$ | $`- 0.5_{- 0.04}^{+ 0.04}`$ |  |  |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 137.44 |
|  | $`0.68 - 3.5`$ | $`- 26.16_{- 0.16}^{+ 0.16}`$ | $`- 5.76_{- 0.07}^{+ 0.07}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |  |  |
|  |  | $`- 0.92_{- 0.07}^{+ 0.06}`$ | $`- 0.39_{- 0.02}^{+ 0.02}`$ | $`- 1.95_{- 0.11}^{+ 0.11}`$ |  |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.84_{- 0.09}^{+ 0.08}`$ | $`- 0.14_{- 0.07}^{+ 0.07}`$ | $`- 1.33_{- 0.05}^{+ 0.05}`$ | $`- 0.47_{- 0.05}^{+ 0.05}`$ |  |  |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 119.68 |
|  | $`0.68 - 3.0`$ | $`- 26.18_{- 0.17}^{+ 0.18}`$ | $`- 5.77_{- 0.07}^{+ 0.07}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ |  |  |  |  |  |
|  |  | $`- 0.87_{- 0.08}^{+ 0.08}`$ | $`- 0.38_{- 0.02}^{+ 0.02}`$ | $`- 1.84_{- 0.13}^{+ 0.14}`$ |  |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.85_{- 0.10}^{+ 0.09}`$ | $`- 0.09_{- 0.06}^{+ 0.06}`$ | $`- 1.33_{- 0.05}^{+ 0.06}`$ | $`- 0.4_{- 0.06}^{+ 0.06}`$ |  |  |  |  |
| LEDE₈ | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 141.64 |
|  | $`0.68 - 4.0`$ | $`- 25.72_{- 0.16}^{+ 0.16}`$ | $`- 5.56_{- 0.06}^{+ 0.06}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |
|  |  | $`- 2.79_{- 0.09}^{+ 0.08}`$ | $`- 1.10_{- 0.06}^{+ 0.07}`$ | $`- 0.37_{- 0.01}^{+ 0.01}`$ | $`- 0.20_{- 0.01}^{+ 0.01}`$ | $`- 0.78_{- 0.04}^{+ 0.04}`$ | $`0.34_{- 0.04}^{+ 0.04}`$ |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 133.70 |
|  | $`0.68 - 3.5`$ | $`- 25.72_{- 0.16}^{+ 0.16}`$ | $`- 5.56_{- 0.06}^{+ 0.06}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |
|  |  | $`- 2.79_{- 0.09}^{+ 0.08}`$ | $`- 1.10_{- 0.06}^{+ 0.07}`$ | $`- 0.37_{- 0.02}^{+ 0.02}`$ | $`- 0.20_{- 0.02}^{+ 0.02}`$ | $`- 0.75_{- 0.05}^{+ 0.04}`$ | $`0.38_{- 0.05}^{+ 0.05}`$ |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 118.76 |
|  | $`0.68 - 3.0`$ | $`- 25.75_{- 0.16}^{+ 0.17}`$ | $`- 5.58_{- 0.06}^{+ 0.06}`$ |  |  |  |  |  |  |
|  |  | $`a`$ | $`b`$ | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |
|  |  | $`- 2.80_{- 0.09}^{+ 0.08}`$ | $`- 1.12_{- 0.06}^{+ 0.07}`$ | $`- 0.40_{- 0.02}^{+ 0.02}`$ | $`- 0.22_{- 0.02}^{+ 0.02}`$ | $`- 0.75_{- 0.07}^{+ 0.07}`$ | $`0.38_{- 0.07}^{+ 0.07}`$ |  |  |
| LEDE₈₊₂ | 8 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 147.19 |
|  | $`0.68 - 4.0`$ | $`- 25.92_{- 0.20}^{+ 0.20}`$ | $`- 5.65_{- 0.09}^{+ 0.08}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |  |  |
|  |  | $`- 0.50_{- 0.19}^{+ 0.09}`$ | $`- 0.21_{- 0.06}^{+ 0.02}`$ | $`- 1.04_{- 0.42}^{+ 0.22}`$ | $`0.32_{- 0.11}^{+ 0.06}`$ |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.83_{- 0.10}^{+ 0.09}`$ | $`- 0.01_{- 0.11}^{+ 0.09}`$ | $`- 1.22_{- 0.08}^{+ 0.08}`$ | $`- 0.17_{- 0.17}^{+ 0.09}`$ |  |  |  |  |
|  | 7 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 138.06 |
|  | $`0.68 - 3.5`$ | $`- 25.90_{- 0.20}^{+ 0.20}`$ | $`- 5.64_{- 0.08}^{+ 0.08}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |  |  |
|  |  | $`- 0.48_{- 0.14}^{+ 0.09}`$ | $`- 0.21_{- 0.04}^{+ 0.02}`$ | $`- 0.99_{- 0.30}^{+ 0.23}`$ | $`0.35_{- 0.08}^{+ 0.06}`$ |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.81_{- 0.10}^{+ 0.09}`$ | $`0.02_{- 0.10}^{+ 0.10}`$ | $`- 1.21_{- 0.08}^{+ 0.08}`$ | $`- 0.17_{- 0.12}^{+ 0.09}`$ |  |  |  |  |
|  | 6 bins | $`M_{\ast}\hspace{0pt}{(z_{p})}`$ | $`{\log_{10}\Phi}\hspace{0pt}{(z_{p})}`$ |  |  |  |  |  | 118.97 |
|  | $`0.68 - 3.0`$ | $`- 25.95_{- 0.21}^{+ 0.23}`$ | $`- 5.67_{- 0.09}^{+ 0.09}`$ |  |  |  |  |  |  |
|  |  | $`c_{1}`$ | $`c_{2}`$ | $`c_{3}`$ | $`c_{4}`$ |  |  |  |  |
|  |  | $`- 0.55_{- 0.12}^{+ 0.11}`$ | $`- 0.24_{- 0.04}^{+ 0.03}`$ | $`- 1.06_{- 0.29}^{+ 0.29}`$ | $`0.33_{- 0.09}^{+ 0.09}`$ |  |  |  |  |
|  |  | $`a\hspace{0pt}{(z_{p})}`$ | $`c_{a}`$ | $`b\hspace{0pt}{(z_{p})}`$ | $`c_{b}`$ |  |  |  |  |
|  |  | $`- 2.80_{- 0.12}^{+ 0.11}`$ | $`0.05_{- 0.12}^{+ 0.11}`$ | $`- 1.24_{- 0.07}^{+ 0.08}`$ | $`- 0.20_{- 0.09}^{+ 0.09}`$ |  |  |  |  |

Table 2: Best-fitting values for the parameters of the models described in Section [3.1](#S3.SS1 "3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"), obtained from the MCMC analysis described in Section [4.1](#S4.SS1 "4.1 Maximum likelihood ‣ 4 Method ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"). The last column gives the Bayesian Information Criterion (BIC, Equation [23](#S4.E23 "In 4.2 Goodness of fit ‣ 4 Method ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole")) for each analysis.

![Refer to caption](/html/2212.04925/assets/figures/allmodels.png)

Figure 3: Best-fitting QLFs for the models described in Section [3.1](#S3.SS1 "3.1 Theoretical modelling ‣ 3 Data ‣ Theoretical systematics in testing the Cosmological Principle with the kinematic quasar dipole"). The corresponding best-fitting parameters are shown in Table LABEL:tab:MCMCresults. The data points and uncertainties are taken from [PD2016](#bib.bib34). The dashed line separates the region below the limiting magnitude cut $`M_{c} = {- 25}`$ for the last redshift bin (Wang et al., [2020](#bib.bib49)). The shaded region is neglected when we consider all eight redshift bins for the dipole analysis.

## References

- Ade et al. (2016) Ade, P. A. R., Aghanim, N., Arnaud, M., et al. 2016, A&A, 594, A7, doi: [10.1051/0004-6361/201525830](http://doi.org/10.1051/0004-6361/201525830)
- Aghanim et al. (2020) Aghanim, N., Akrami, Y., Arroja, F., et al. 2020, Astronomy & Astrophysics, 641, A1, doi: [10.1051/0004-6361/201833880](http://doi.org/10.1051/0004-6361/201833880)
- Alonso et al. (2015) Alonso, D., Bull, P., Ferreira, P. G., Maartens, R., & Santos, M. G. 2015, The Astrophysical Journal, 814, 145, doi: [10.1088/0004-637X/814/2/145](http://doi.org/10.1088/0004-637X/814/2/145)
- Bengaly et al. (2018) Bengaly, C. A., Maartens, R., & Santos, M. G. 2018, Journal of Cosmology and Astroparticle Physics, 2018, 031, doi: [10.1088/1475-7516/2018/04/031](http://doi.org/10.1088/1475-7516/2018/04/031)
- Bengaly et al. (2019) Bengaly, C. A., Siewert, T. M., Schwarz, D. J., & Maartens, R. 2019, Monthly Notices of the Royal Astronomical Society, 486, 1350, doi: [10.1093/mnras/stz832](http://doi.org/10.1093/mnras/stz832)
- Blake & Wall (2002) Blake, C., & Wall, J. 2002, arXiv preprint astro-ph/0203385, doi: [10.1038/416150a](http://doi.org/10.1038/416150a)
- Blas et al. (2011) Blas, D., Lesgourgues, J., & Tram, T. 2011, J. Cosmology Astropart. Phys, 2011, 034–034, doi: [10.1088/1475-7516/2011/07/034](http://doi.org/10.1088/1475-7516/2011/07/034)
- Bonvin & Durrer (2011) Bonvin, C., & Durrer, R. 2011, Phys. Rev. D, 84, 063505, doi: [10.1103/PhysRevD.84.063505](http://doi.org/10.1103/PhysRevD.84.063505)
- Boyle et al. (2000) Boyle, B. J., Shanks, T., Croom, S., et al. 2000, Monthly Notices of the Royal Astronomical Society, 317, 1014, doi: [10.1046/j.1365-8711.2000.03730.x](http://doi.org/10.1046/j.1365-8711.2000.03730.x)
- Carballo et al. (1999) Carballo, R., González-Serrano, J. I., Benn, C. R., Sánchez, S. F., & Vigotti, M. 1999, Monthly Notices of the Royal Astronomical Society, 306, 137, doi: [10.1023/A:1002189323147](http://doi.org/10.1023/A:1002189323147)
- Challinor & Lewis (2011) Challinor, A., & Lewis, A. 2011, Physical Review D, 84, 043516, doi: [10.1103/PhysRevD.84.043516](http://doi.org/10.1103/PhysRevD.84.043516)
- Clarkson (2012) Clarkson, C. 2012, Comptes Rendus Physique, 13, 682, doi: [10.1016/j.crhy.2012.04.005](http://doi.org/10.1016/j.crhy.2012.04.005)
- Clarkson & Maartens (2010) Clarkson, C., & Maartens, R. 2010, Class. Quant. Grav., 27, 124008, doi: [10.1088/0264-9381/27/12/124008](http://doi.org/10.1088/0264-9381/27/12/124008)
- Colin et al. (2017) Colin, J., Mohayaee, R., Rameez, M., & Sarkar, S. 2017, Monthly Notices of the Royal Astronomical Society, 471, 1045, doi: [10.1093/mnras/stx1631](http://doi.org/10.1093/mnras/stx1631)
- Croom et al. (2004) Croom, S. M., Smith, R., Boyle, B., et al. 2004, Monthly Notices of the Royal Astronomical Society, 349, 1397, doi: [10.1111/j.1365-2966.2004.07619.x](http://doi.org/10.1111/j.1365-2966.2004.07619.x)
- Croom et al. (2009) Croom, S. M., Richards, G. T., Shanks, T., et al. 2009, Monthly Notices of the Royal Astronomical Society, 399, 1755, doi: [10.1111/j.1365-2966.2009.15398.x](http://doi.org/10.1111/j.1365-2966.2009.15398.x)
- Dalang & Bonvin (2022) Dalang, C., & Bonvin, C. 2022, Monthly Notices of the Royal Astronomical Society, 512, 3895, doi: [10.1093/mnras/stac726](http://doi.org/10.1093/mnras/stac726)
- Darling (2022) Darling, J. 2022, The Astrophysical Journal Letters, 931, L14, doi: [10.3847/2041-8213/ac6f08](http://doi.org/10.3847/2041-8213/ac6f08)
- Das et al. (2021) Das, K. K., Sankharva, K., & Jain, P. 2022, Journal of Cosmology and Astroparticle Physics, 2021, 035, doi: [10.1088/1475-7516/2021/07/035](http://doi.org/10.1088/1475-7516/2021/07/035)
- Domènech et al. (2022) Domènech, G., Mohayaee, R., Patil, S. P., & Sarkar, S. 2022, Journal of Cosmology and Astroparticle Physics, 2022, 019, doi: [10.1088/1475-7516/2022/10/019](http://doi.org/10.1088/1475-7516/2022/10/019)
- Ehlers et al. (1968) Ehlers, J., Geren, P., & Sachs, R. K. 1968, Journal of Mathematical Physics, 9, 1344, doi: [10.1063/1.1664720](http://doi.org/10.1063/1.1664720)
- Ellis & Baldwin (1984) Ellis, G., & Baldwin, J. 1984, Monthly Notices of the Royal Astronomical Society, 206, 377, doi: [10.1093/mnras/206.2.377](http://doi.org/10.1093/mnras/206.2.377)
- Ellis et al. (1983) Ellis, G., Treciokas, R., & Matravers, D. 1983, Annals of Physics, 150, 487, doi: [10.1016/0003-4916(83)90024-6](http://doi.org/10.1016/0003-4916(83)90024-6)
- Foreman-Mackey et al. (2013) Foreman-Mackey, D., Hogg, D. W., Lang, D., & Goodman, J. 2013, PASP, 125, 306, doi: [10.1086/670067](http://doi.org/10.1086/670067)
- Ghosh et al. (2016) Ghosh, S., Jain, P., Kashyap, G., et al. 2016, Journal of Astrophysics and Astronomy, 37, 1, doi: [10.1007/s12036-016-9395-8](http://doi.org/10.1007/s12036-016-9395-8)
- Gibelyou & Huterer (2012) Gibelyou, C., & Huterer, D. 2012, Monthly Notices of the Royal Astronomical Society, 427, 1994, doi: [10.1111/j.1365-2966.2012.22032.x](http://doi.org/10.1111/j.1365-2966.2012.22032.x)
- Harris et al. (2020) Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 2020, Nature, 585, 357, doi: [10.1038/s41586-020-2649-2](http://doi.org/10.1038/s41586-020-2649-2)
- Hunter (2007) Hunter, J. D. 2007, Computing in Science & Engineering, 9, 90, doi: [10.1109/MCSE.2007.55](http://doi.org/10.1109/MCSE.2007.55)
- Liddle (2004) Liddle, A. R. 2004, Monthly Notices of the Royal Astronomical Society, 351, L49, doi: [10.1111/j.1365-2966.2004.08033.x](http://doi.org/10.1111/j.1365-2966.2004.08033.x)
- Maartens et al. (2018) Maartens, R., Clarkson, C., & Chen, S. 2018, Journal of Cosmology and Astroparticle Physics, 2018, 013, doi: [10.1088/1475-7516/2018/01/013](http://doi.org/10.1088/1475-7516/2018/01/013)
- Maartens et al. (2021) Maartens, R., Fonseca, J., Camera, S., et al. 2021, Journal of Cosmology and Astroparticle Physics, 2021, 009, doi: [10.1088/1475-7516/2021/12/009](http://doi.org/10.1088/1475-7516/2021/12/009)
- Nadolny et al. (2021) Nadolny, T., Durrer, R., Kunz, M., & Padmanabhan, H. 2021, Journal of Cosmology and Astroparticle Physics, 2021, 009, doi: [10.1088/1475-7516/2021/11/009](http://doi.org/10.1088/1475-7516/2021/11/009)
- Palanque-Delabrouille et al. (2011) Palanque-Delabrouille, N., Yèche, C., Myers, A., et al. 2011, Astronomy & Astrophysics, 530, A122, doi: [10.1051/0004-6361/201016254](http://doi.org/10.1051/0004-6361/201016254)
- Palanque-Delabrouille et al. (2016) Palanque-Delabrouille, N., Magneville, C., Yèche, C., et al. 2016, Astronomy & Astrophysics, 587, A41, doi: [10.1051/0004-6361/201527392](http://doi.org/10.1051/0004-6361/201527392)
- Peebles & Wilkinson (1968) Peebles, P., & Wilkinson, D. T. 1968, Physical Review, 174, 2168, doi: [10.1103/PhysRev.174.2168](http://doi.org/10.1103/PhysRev.174.2168)
- Pozzetti et al. (2016) Pozzetti, L., Hirata, C., Geach, J., et al. 2016, Astronomy & Astrophysics, 590, A3, doi: [10.1051/0004-6361/201527081](http://doi.org/10.1051/0004-6361/201527081)
- Rasera et al. (2022) Rasera, Y., Breton, M.-A., Corasaniti, P.-S., et al. 2022, Astronomy & Astrophysics, 661, A90, doi: [10.1051/0004-6361/202141908](http://doi.org/10.1051/0004-6361/202141908)
- Ross et al. (2013) Ross, N. P., McGreer, I. D., White, M., et al. 2013, The Astrophysical Journal, 773, 14, doi: [10.1088/0004-637X/773/1/14](http://doi.org/10.1088/0004-637X/773/1/14)
- Rubart & Schwarz (2013) Rubart, M., & Schwarz, D. J. 2013, Astronomy & Astrophysics, 555, A117, doi: [10.1051/0004-6361/201321215](http://doi.org/10.1051/0004-6361/201321215)
- Schmidt et al. (2010) Schmidt, K. B., Marshall, P. J., Rix, H.-W., et al. 2010, The Astrophysical Journal, 714, 1194, doi: [10.1088/0004-637X/714/2/1194](http://doi.org/10.1088/0004-637X/714/2/1194)
- Secrest et al. (2022) Secrest, N. J., von Hausegger, S., Rameez, M., Mohayaee, R., & Sarkar, S. 2022, The Astrophysical Journal Letters, 937, L31, doi: [10.3847/2041-8213/ac88c0](http://doi.org/10.3847/2041-8213/ac88c0)
- Secrest et al. (2021) Secrest, N. J., von Hausegger, S., Rameez, M., et al. 2021, The Astrophysical journal letters, 908, L51, doi: [10.3847/2041-8213/abdd40](http://doi.org/10.3847/2041-8213/abdd40)
- Stewart & Sciama (1967) Stewart, J., & Sciama, D. 1967, Nature, 216, 748, doi: [10.1038/216748a0](http://doi.org/10.1038/216748a0)
- Stoeger et al. (1995) Stoeger, W. R., Maartens, R., & Ellis, G. 1995, The Astrophysical Journal, 443, 1, doi: [10.1086/175496](http://doi.org/10.1086/175496)
- Tiwari et al. (2015) Tiwari, P., Kothari, R., Naskar, A., Nadkarni-Ghosh, S., & Jain, P. 2015, Astroparticle Physics, 61, 1, doi: [10.1016/j.astropartphys.2014.06.004](http://doi.org/10.1016/j.astropartphys.2014.06.004)
- Tiwari & Nusser (2016) Tiwari, P., & Nusser, A. 2016, Journal of Cosmology and Astroparticle Physics, 2016, 062, doi: [10.1088/1475-7516/2016/03/062](http://doi.org/10.1088/1475-7516/2016/03/062)
- Tiwari et al. (2022) Tiwari, P., Kothari, R., & Jain, P. 2022, The Astrophysical Journal Letters, 2022, L36, doi: [10.3847/2041-8213/ac447a](http://doi.org/10.3847/2041-8213/ac447a)
- Virtanen et al. (2020) Virtanen, P., Gommers, R., Oliphant, T. E., et al. 2020, Nature Methods, 17, 261, doi: [10.1038/s41592-019-0686-2](http://doi.org/10.1038/s41592-019-0686-2)
- Wang et al. (2020) Wang, M. S., Beutler, F., & Bacon, D. 2020, Monthly Notices of the Royal Astronomical Society, 499, 2598, doi: [10.1093/mnras/staa2998](http://doi.org/10.1093/mnras/staa2998)

[◄](/html/2212.04924) [![ar5iv homepage](/assets/ar5iv.png)](/) [Feeling\
lucky?](/feeling_lucky) [](/land_of_honey_and_milk) [Conversion\
report](/log/2212.04925) [Report\
an issue](https://github.com/dginev/ar5iv/issues/new?template=improve-article--arxiv-id-.md&title=Improve+article+2212.04925) [View original\
on arXiv](https://arxiv.org/abs/2212.04925)[►](/html/2212.04926)

[](javascript:toggleColorScheme() "Toggle ar5iv color scheme") [Copyright](https://arxiv.org/help/license) [Privacy Policy](https://arxiv.org/help/policies/privacy_policy)

Generated on Fri Mar 1 11:58:54 2024 by [LaTeXML![Mascot Sammy](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAOCAYAAAD5YeaVAAAAAXNSR0IArs4c6QAAAAZiS0dEAP8A/wD/oL2nkwAAAAlwSFlzAAALEwAACxMBAJqcGAAAAAd0SU1FB9wKExQZLWTEaOUAAAAddEVYdENvbW1lbnQAQ3JlYXRlZCB3aXRoIFRoZSBHSU1Q72QlbgAAAdpJREFUKM9tkL+L2nAARz9fPZNCKFapUn8kyI0e4iRHSR1Kb8ng0lJw6FYHFwv2LwhOpcWxTjeUunYqOmqd6hEoRDhtDWdA8ApRYsSUCDHNt5ul13vz4w0vWCgUnnEc975arX6ORqN3VqtVZbfbTQC4uEHANM3jSqXymFI6yWazP2KxWAXAL9zCUa1Wy2tXVxheKA9YNoR8Pt+aTqe4FVVVvz05O6MBhqUIBGk8Hn8HAOVy+T+XLJfLS4ZhTiRJgqIoVBRFIoric47jPnmeB1mW/9rr9ZpSSn3Lsmir1fJZlqWlUonKsvwWwD8ymc/nXwVBeLjf7xEKhdBut9Hr9WgmkyGEkJwsy5eHG5vN5g0AKIoCAEgkEkin0wQAfN9/cXPdheu6P33fBwB4ngcAcByHJpPJl+fn54mD3Gg0NrquXxeLRQAAwzAYj8cwTZPwPH9/sVg8PXweDAauqqr2cDjEer1GJBLBZDJBs9mE4zjwfZ85lAGg2+06hmGgXq+j3+/DsixYlgVN03a9Xu8jgCNCyIegIAgx13Vfd7vdu+FweG8YRkjXdWy329+dTgeSJD3ieZ7RNO0VAXAPwDEAO5VKndi2fWrb9jWl9Esul6PZbDY9Go1OZ7PZ9z/lyuD3OozU2wAAAABJRU5ErkJggg==)](http://dlmf.nist.gov/LaTeXML/)
