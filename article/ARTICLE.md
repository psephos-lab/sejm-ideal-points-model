# Mapping a Parliament from Its Votes: A Bayesian Ideal-Point Model of the Polish Sejm

**Part 1 — The Main Axis (One Dimension)**

> **Draft scaffold.** Math uses `$…$` / `$$…$$` (renders natively on GitHub,
> Dev.to, Hashnode; on Medium render formulas as images). Figures live in
> `figures/` — export them from `results/` after running the pipeline.
> Language: English (data-science audience). A Polish version is trivial to derive.
> This is **Part 1 of two**: the one-dimensional model and its estimation.
> [`ARTICLE_2D.md`](ARTICLE_2D.md) extends it to a second dimension.

*An educational walk through Bayesian computation: deriving a spatial-voting
model from first principles and estimating it with a data-augmentation Gibbs
sampler — filling a gap in Poland's public data.*

---

## 1. The niche

The United States has [Voteview](https://voteview.com): a public, interactive map
of every legislator's position on the main voting axis, estimated from roll-call votes. Most
other democracies have nothing comparable. Poland has excellent **descriptive**
tools (vote trackers, attendance, party discipline) but **no public estimate of
where individual MPs sit on a latent main axis of division** recovered from their votes.

This project fills that gap for the Sejm (10th term) — and doubles as a hands-on
tour of Bayesian computation: deriving the spatial-voting model from first
principles and estimating it with a data-augmentation Gibbs sampler.

## 2. The model: spatial voting → item response theory

Each MP $i$ has a latent **ideal point** $x_i \in \mathbb{R}$ (position on the
main axis of conflict). Each vote $j$ pits a "yes" outcome against a "no" outcome,
each located in the same space. The two-parameter item-response (2PL) probit form

$$
P(y_{ij} = 1 \mid x_i, \alpha_j, \beta_j) = \Phi(\beta_j x_i - \alpha_j)
$$

is not postulated ad hoc — it falls out of a spatial random-utility model, derived next.

### 2.1 Where the model comes from — the spatial micro-foundation

MP $i$ has a latent ideal point $x_i \in \mathbb{R}$. Each vote $j$ places two
rival outcomes in the same space: a "yes" outcome at $\zeta_j$ and a "no" outcome
at $\psi_j$. An MP's utility from an outcome falls with the **squared distance**
from their ideal point, plus a random shock:

$$
U_i^{\text{yes}} = -(x_i-\zeta_j)^2 + \eta_{ij}, \qquad
U_i^{\text{no}}  = -(x_i-\psi_j)^2 + \nu_{ij}.
$$

The MP votes "yes" when $U_i^{\text{yes}} > U_i^{\text{no}}$. Expanding the
difference of squares, the $x_i^2$ terms **cancel** and a function that is
**linear** in $x_i$ remains:

$$
U_i^{\text{yes}} - U_i^{\text{no}}
= \underbrace{2(\zeta_j-\psi_j)}_{\textstyle \beta_j}\,x_i
\;-\; \underbrace{(\zeta_j^2-\psi_j^2)}_{\textstyle \alpha_j}
\;+\; \varepsilon_{ij},
\qquad \varepsilon_{ij} = \eta_{ij}-\nu_{ij}.
$$

The two vote parameters thus acquire a concrete interpretation:

- $\beta_j = 2(\zeta_j-\psi_j)$ — **discrimination**. Large when the two outcomes
  sit far apart on the axis, i.e. when the vote sharply sorts the chamber along it;
  near zero when the vote carries no positional information (e.g. a procedural matter).
- $\alpha_j = \zeta_j^2-\psi_j^2$ — **threshold / difficulty**, shifting the point
  at which the MP is indifferent between the two outcomes.

The MP votes "yes" exactly when $\beta_j x_i - \alpha_j > \varepsilon_{ij}$.
Taking $\varepsilon_{ij}\sim\mathcal N(0,1)$ — the **probit** model — the symmetry
of the normal gives

$$
\boxed{\,P(y_{ij}=1\mid x_i,\alpha_j,\beta_j) = \Phi(\beta_j x_i - \alpha_j)\,}
$$

where $\Phi$ is the standard normal CDF (the **probit** link).

### 2.2 Likelihood

Assuming conditional independence, the likelihood over observed cells $\mathcal{O}$
of the roll-call matrix is

$$
L(X,\alpha,\beta \mid Y) = \prod_{(i,j)\in\mathcal{O}}
\Phi(\beta_j x_i - \alpha_j)^{y_{ij}}\,\bigl[1-\Phi(\beta_j x_i - \alpha_j)\bigr]^{1-y_{ij}} .
$$

Political scientists call $x_i$ an *ideal point*; psychometricians call the same
object an *ability* in 2PL IRT. They are the same model. (Clinton, Jackman &
Rivers 2004; see `REFERENCES.md`.)

## 3. Priors and identification

Weakly-informative, **fixed proper** priors:

$$
x_i \sim \mathcal{N}(0,1), \qquad
\alpha_j \sim \mathcal{N}(0,\sigma_\alpha^2), \qquad
\beta_j \sim \mathcal{N}(0,\sigma_\beta^2),
\qquad \sigma_\beta = 2,\;\; \sigma_\alpha = 2.5 .
$$

(Why these values, and not a diffuse $\sigma^2=25$ or a Half-Cauchy hyperprior,
is the subject of §6.1 — on a disciplined parliament the choice matters.)

The unit-variance prior on $x$ is doing double duty — it is also an
**identification** device. The linear index $\beta_j x_i - \alpha_j$ has three
symmetries that leave the likelihood unchanged:

1. **Translation** $x_i \mapsto x_i + b$ (absorbed by $\alpha_j$) — fixed by the zero mean.
2. **Scale** $(x_i,\beta_j)\mapsto(cx_i,\beta_j/c)$ — fixed by unit variance.
3. **Reflection** $(x_i,\beta_j)\mapsto(-x_i,-\beta_j)$ — *not* fixed by a symmetric
   prior. We resolve it by **anchoring**: forcing a chosen reference MP (a high-turnout PiS MP) to have $x>0$.

## 4. The posterior, and why we need MCMC

By Bayes' rule the posterior is the likelihood × prior, normalized:

$$
p(X,\alpha,\beta \mid Y) =
\frac{L(Y\mid\cdot)\,\prod_i p(x_i)\,\prod_j p(\alpha_j)\,p(\beta_j)}
     {\displaystyle\int L(Y\mid\cdot)\,\prod_i p(x_i)\,\prod_j p(\alpha_j)\,p(\beta_j)\;dX\,d\alpha\,d\beta} .
$$

The denominator $Z$ is an integral over $n + 2m$ parameters (hundreds of MPs,
thousands of votes) and is **intractable**. We know the posterior only **up to
that constant** — exactly the setting MCMC is built for.

**How MCMC works without $Z$.** $Z$ only *rescales* the density; it changes its
height, not its shape. And every quantity we actually report — a club's mean
position, an MP's credible interval, $P(\text{yes})$ for a vote — is an
**expectation** under the posterior, which we estimate by averaging over draws,
$\mathbb{E}[g(\theta)] \approx \tfrac{1}{S}\sum_s g(\theta^{(s)})$. The constant
$Z$ never enters that average. So instead of computing the posterior density, we
build a Markov chain whose **stationary distribution is the posterior** and read
the answer off the **draws**: the histogram of the $x_i$ samples *is* the
estimated marginal posterior, its mean is the point estimate, and its 5th/95th
percentiles are the credible interval. The Gibbs sampler of the next section goes
one step further — each of its full conditionals is an *already-normalized* known
distribution we can sample directly, so $Z$ plays no role at any step.

## 5. The sampler: Gibbs via Albert–Chib data augmentation

The multiplicative term $\beta_j x_i$ makes the posterior **curved**, and on
near-perfect party-line votes (balanced 50/50 but perfectly aligned with the axis)
the likelihood becomes almost a step function and $\beta_j$ wants to run to
infinity. The ideal-point literature's standard remedy is **data augmentation**.

Introduce a latent utility $y^{\ast}_{ij}$ and observe only its sign:

$$
y^{\ast}_{ij} = \beta_j x_i - \alpha_j + \varepsilon_{ij}, \qquad
\varepsilon_{ij}\sim\mathcal N(0,1), \qquad
y_{ij} = \mathbf{1}\!\left[\,y^{\ast}_{ij} > 0\,\right].
$$

This reproduces the probit model **exactly**, since
$P(y^{\ast}_{ij}>0) = \Phi(\beta_j x_i - \alpha_j)$. Conditional on $y^{\ast}$ the
model is **linear-Gaussian**, and every full conditional is closed-form:

- $y^{\ast}_{ij}\mid\cdot \;\sim\;$ truncated normal (sign set by $y_{ij}$),
- $x_i\mid\cdot \;\sim\;$ Gaussian (1-D regression of $y^{\ast}+\alpha$ on $\beta$),
- $(\beta_j,\alpha_j)\mid\cdot \;\sim\;$ bivariate Gaussian (regression of $y^{\ast}$ on $(x,-1)$).

Data augmentation **linearizes the link** and sidesteps the curved geometry
entirely: each sweep is a handful of closed-form draws — no accept/reject, no
step-size tuning — and it handles perfectly separating votes gracefully.
Vectorized in NumPy, one sweep is a few matrix ops:

```python
# one Gibbs sweep (schematic)
eta   = X @ B.T - alpha[None, :]
ystar = truncated_normal(eta, Y)                 # latent utilities
X     = gaussian_update(ystar, B, alpha)         # ideal points
B, alpha = gaussian_update_votes(ystar, X)       # vote params
standardize(X, B, alpha)                         # parameter expansion (sec 6.2)
```

Synthetic recovery: correlation $\approx 0.99$. ~74 ms/iteration on an M1 CPU.

## 6. Three practical lessons

### 6.1 Separation and the prior

A diffuse hierarchical prior (Half-Cauchy on $\sigma_\beta$) **backfires**: its
heavy tail lets the discrimination scale escape to infinity on perfectly
separating votes. A **fixed, proper** prior ($\sigma_\beta=2$, $\sigma_\alpha=2.5$)
adds prior precision that keeps $\beta$ finite. The conventional diffuse value
$\sigma^2=25$ is *too* loose for a disciplined parliament.

### 6.2 Parameter expansion (scale identification)

With thousands of informative votes, the $\mathcal N(0,1)$ prior is too weak to pin
the scale, which drifts between chains. Standardizing $x$ to mean 0 / SD 1 **each
sweep** — absorbing the scale into $(\beta,\alpha)$ so $\eta$ is unchanged — pins it
hard (Liu–Wu parameter expansion).

### 6.3 Marginal data augmentation is inert here

Could MDA accelerate the (admittedly slow) mixing? **No — and we can show why.**
The single working-scale parameter $g$ is pinned by ~1M observations:

$$
g \approx 1.0036 \pm 0.0007 .
$$

PX-DA's benefit comes from the *variance* of $g$, which here is ~0. The literature's
"10–50× ESS" gains are for probit *regression* (few parameters) or multinomial
probit (a covariance working parameter) — a different regime. The deeper reason:
data augmentation linearizes the *link*, not the bilinear $\beta x$ term, and the
slow mixing comes from the $x\leftrightarrow\beta$ coupling, which no augmentation
fixes. The honest lever is more (parallel) samples.

## 7. Results — one dimension

Final run: 4 chains × 10,000 draws on 499 MPs × 2,570 contested votes.
Convergence: $\hat R_{\max}=1.065$, mean ESS $\approx 608$.

The recovered axis cleanly separates the chamber. Mean position by club:

| Club | mean $x$ |
|---|---|
| KO | −0.99 |
| Lewica | −0.80 |
| Konfederacja | +0.41 |
| PiS | +1.14 |

Face validity, with **no labels given to the model**: at one end of the axis is
Joanna Scheuring-Wielgus (Lewica, −1.12); at the other, PiS MPs Artur Soboń (+1.39)
and Jacek Ozdoba (+1.32). The distribution is strikingly **bimodal** — two blocs with
an almost empty centre, the signature of iron party discipline.

> **Caveat / honesty:** I deliberately do **not** label this axis "left–right".
> Roll-call votes certainly carry ideological signal — but turning that signal into
> an ideological verdict is a judgment I choose not to make. This project supplies
> the **data**, recovered as a **main axis of division**, and leaves the
> interpretation to the reader. In this term that axis most plausibly corresponds to
> the **government–opposition** split (one bloc is the governing coalition, the other
> the opposition) — a term-specific mapping, not an intrinsic ideological scale.

![Ideal points, colored by club](figures/ideal_points.png)
![Distribution by club](figures/club_distributions.png)

## 8. The interactive site & reproducibility

A static site (no backend) presents the estimates: a beeswarm of MPs colored by
club, per-MP profiles (position ± CI, rank, turnout, party loyalty, a chamber
distribution histogram), and per-vote **breakdowns** — for contested votes the
model probability curve $\Phi(\beta x-\alpha)$ with the model cutting point
$x^*=\alpha/\beta$ (à la Voteview); non-contested votes show the vote breakdown
without a cutting line.

Everything is reproducible from public data:

```bash
python fetch_data.py        # roll-call from api.sejm.gov.pl
python run.py               # Gibbs sampler -> ideal points
python make_site_data.py    # export site JSON
```

Repo: **https://github.com/psephos-lab/sejm-ideal-points-model** ·
Data: [api.sejm.gov.pl](https://api.sejm.gov.pl)

## 9. Conclusion

Starting from spatial-voting theory, we derived a 2PL probit ideal-point model and
estimated it with an Albert–Chib data-augmentation Gibbs sampler, whose closed-form
conditionals handle the perfectly party-line votes that make this posterior
awkward. Along the way: a separation-aware prior, parameter expansion for
identification, and a negative result on marginal data augmentation. The output is
the public per-MP ideal-point map that Poland lacked. A companion piece,
[`ARTICLE_2D.md`](ARTICLE_2D.md), extends the model to a second dimension — and a
lesson in how *not* to interpret one.

---

*References: see [`REFERENCES.md`](../REFERENCES.md).*
*Disclaimer: an "ideal point" is a position in vote space, not a judgment of a
politician. Uncertainty grows for MPs who vote rarely.*
