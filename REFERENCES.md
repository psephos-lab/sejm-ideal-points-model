# References

Bibliography for the Sejm ideal-point project (Bayesian spatial voting model,
MCMC). Citation details (volume/pages) should be verified before publication.
**Bold + link** = directly accessible online.

## Ideal point estimation / roll-call analysis (core)
- **Clinton, J., Jackman, S., & Rivers, D. (2004).** *The Statistical Analysis of Roll Call Data.* American Political Science Review 98(2):355–370. — [Princeton PDF](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/ClintonJackmanRivers2004.pdf) — the model, priors, posterior, Gibbs/data-augmentation sampler we follow.
- Jackman, S. (2001). *Multidimensional Analysis of Roll Call Data via Bayesian Simulation.* Political Analysis 9(3):227–241.
- Poole, K. T., & Rosenthal, H. (1997). *Congress: A Political-Economic History of Roll Call Voting.* Oxford UP. (NOMINATE) — **[Voteview.com](https://voteview.com)** (reference/validation site).

## Dynamic ideal points
- **Martin, A. D., & Quinn, K. M. (2002).** *Dynamic Ideal Point Estimation via Markov Chain Monte Carlo for the U.S. Supreme Court, 1953–1999.* Political Analysis 10(2):134–153. — [Princeton PDF](https://www.cs.princeton.edu/courses/archive/fall09/cos597A/papers/MartinQuinn2002.pdf)

## MCMC foundations / our sampler
- Metropolis, N., et al. (1953). *Equation of State Calculations by Fast Computing Machines.* J. Chem. Phys. 21:1087–1092.
- Hastings, W. K. (1970). *Monte Carlo Sampling Methods Using Markov Chains and Their Applications.* Biometrika 57:97–109.
- Geman, S., & Geman, D. (1984). *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images.* IEEE TPAMI 6:721–741.
- **Albert, J. H., & Chib, S. (1993).** *Bayesian Analysis of Binary and Polychotomous Response Data.* JASA 88(422):669–679. — data augmentation for probit (the engine we use).
- Tanner, M. A., & Wong, W. H. (1987). *The Calculation of Posterior Distributions by Data Augmentation.* JASA 82(398):528–540.

## HMC / NUTS (tried first, defeated by geometry)
- Neal, R. M. (2011). *MCMC Using Hamiltonian Dynamics.* In Handbook of Markov Chain Monte Carlo.
- Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler.* JMLR 15:1593–1623.

## Parameter expansion / marginal data augmentation (investigated; inert here)
- Liu, J. S., & Wu, Y. N. (1999). *Parameter Expansion for Data Augmentation.* JASA 94(448):1264–1274.
- van Dyk, D. A., & Meng, X.-L. (2001). *The Art of Data Augmentation.* JCGS 10(1):1–50.
- Imai, K., & van Dyk, D. A. (2005). *A Bayesian Analysis of the Multinomial Probit Model Using Marginal Data Augmentation.* Journal of Econometrics 124(2):311–334.
- Roy, V., & Hobert, J. P. (2007). *Convergence Rates and Asymptotic Standard Errors for MCMC Algorithms for Bayesian Probit Regression.* JRSS-B 69(4):607–623. (Haar PX-DA)

## Priors for variance / separation
- **Gelman, A. (2006).** *Prior Distributions for Variance Parameters in Hierarchical Models.* Bayesian Analysis 1(3):515–534. — why Half-Cauchy over Inverse-Gamma.
- Gelman, A., Jakulin, A., Pittau, M. G., & Su, Y.-S. (2008). *A Weakly Informative Default Prior Distribution for Logistic and Other Regression Models.* Annals of Applied Statistics 2(4):1360–1383. (Cauchy(0, 2.5))

## Convergence diagnostics (R-hat, ESS)
- Gelman, A., & Rubin, D. B. (1992). *Inference from Iterative Simulation Using Multiple Sequences.* Statistical Science 7(4):457–472.
- **Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021).** *Rank-Normalization, Folding, and Localization: An Improved R̂ for Assessing Convergence of MCMC.* Bayesian Analysis 16(2):667–718. — bulk/tail ESS.
- Neal, R. M. (2003). *Slice Sampling.* Annals of Statistics 31(3):705–767. (the "funnel" example)

## Network / social-media ideal points (the other unfilled niche)
- Barberá, P. (2015). *Birds of the Same Feather Tweet Together: Bayesian Ideal Point Estimation Using Twitter Data.* Political Analysis 23(1):76–91.
- *Estimating Ideal Points of British MPs Through Their Social Media Followership.* British Journal of Political Science. — [Cambridge Core](https://www.cambridge.org/core/journals/british-journal-of-political-science/article/estimating-ideal-points-of-british-mps-through-their-social-media-followership/1627B42FE1A547458DB1ED860CE502F1)

## Software / data
- **Sejm API** — https://api.sejm.gov.pl (open, no key).
- Jackman, S. *pscl: Classes and Methods for R* (incl. `ideal()`).
- Martin, A. D., Quinn, K. M., & Park, J. H. *MCMCpack* (R; `MCMCdynamicIRT1d`).
- Phan, D., Pradhan, N., & Jankowiak, M. *Composable Effects for Flexible and Accelerated Probabilistic Programming in NumPyro* (2019).
- Bradbury, J., et al. *JAX: composable transformations of Python+NumPy programs* (2018).
- Kumar, R., Carroll, C., Hartikainen, A., & Martin, O. (2019). *ArviZ: a unified library for exploratory analysis of Bayesian models.* JOSS.

## Note on the σ² = 25 prior
The fixed prior variance 25 (SD = 5) on vote parameters is a **standard diffuse
default**, not a distinctive value from Clinton-Jackman-Rivers (who specify
diffuse normal priors generally). It is in the range of conventional diffuse
defaults (e.g. Jackman's `pscl::ideal`). Cite CJR for the model/prior structure;
attribute the specific value to standard diffuse practice.
