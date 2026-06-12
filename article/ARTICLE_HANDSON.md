# Build It Yourself: A Hands-On Ideal-Point Model of the Polish Sejm

**A code-first walkthrough.** We derive a spatial-voting model from scratch, pull
real roll-call data from the official Sejm API, and estimate every MP's position
with a Gibbs sampler — and at *every* step the math comes paired with the **exact
Python that runs it**, lifted straight from a small public repo you can clone and
execute in five minutes.

> **The repo:** [`sejm-ideal-points-model-demo`](https://github.com/psephos-lab/sejm-ideal-points-model-demo).

By the end you will have run the whole thing on real votes and printed a
left-to-right ordering of the Polish parliament that **no human labelled** — it
falls out of the votes alone.

---

## 1. Why this map exists

The United States has [Voteview](https://voteview.com): a public, interactive map
of every legislator's position on the main voting axis, estimated from roll-call
votes. Most other democracies have nothing comparable. Poland has excellent
**descriptive** tools (vote trackers, attendance, party discipline) but, until
recently, **no public estimate of where individual MPs sit on a latent main axis
of division** recovered from their votes.

This project fills that gap for the Sejm, and the result is live here:
**[mapasejmu.org](https://mapasejmu.org)** — a beeswarm of every MP, six terms,
colored by club. This article is the engine behind that map: we build the
estimator end to end, so the picture stops being a black box.

---

## 2. The model: spatial voting becomes a probit

The model is not postulated ad hoc — it falls out of **spatial voting theory**.
MP $i$ has a latent **ideal point** $x_i \in \mathbb{R}$ (a position on a hidden
axis). Each vote $j$ places two rival outcomes in the same space: a "yes" outcome
at $\zeta_j$ and a "no" outcome at $\psi_j$. An MP's utility from an outcome falls
with the **squared distance** from their ideal point, plus a random shock:

$$
U_i^{\text{yes}} = -(x_i-\zeta_j)^2 + \eta_{ij}, \qquad
U_i^{\text{no}}  = -(x_i-\psi_j)^2 + \nu_{ij}.
$$

The MP votes "yes" when $U_i^{\text{yes}} > U_i^{\text{no}}$. Expand the difference
of squares and the $x_i^2$ terms **cancel**, leaving something **linear** in $x_i$:

$$
U_i^{\text{yes}} - U_i^{\text{no}}
= \underbrace{2(\zeta_j-\psi_j)}_{\textstyle \beta_j}\,x_i
\;-\; \underbrace{(\zeta_j^2-\psi_j^2)}_{\textstyle \alpha_j}
\;+\; \varepsilon_{ij},
\qquad \varepsilon_{ij} = \eta_{ij}-\nu_{ij}.
$$

So the two per-vote parameters acquire a concrete meaning:

- $\beta_j = 2(\zeta_j-\psi_j)$ — **discrimination**: large when the two outcomes
  sit far apart on the axis (the vote sharply sorts the chamber), near zero when
  the vote carries no positional information (a procedural matter).
- $\alpha_j = \zeta_j^2-\psi_j^2$ — **threshold**: shifts the point at which the MP
  is indifferent.

The MP votes "yes" exactly when $\beta_j x_i - \alpha_j > \varepsilon_{ij}$. Take
$\varepsilon_{ij}\sim\mathcal N(0,1)$ — the **probit** model — and the symmetry of
the normal gives the whole model in one line:

$$
\boxed{\,P(y_{ij}=1\mid x_i,\alpha_j,\beta_j) = \Phi(\beta_j x_i - \alpha_j)\,}
$$

where $\Phi$ is the standard normal CDF. Assuming votes are conditionally
independent, the likelihood over the observed cells $\mathcal{O}$ of the roll-call
matrix is

$$
L(X,\alpha,\beta \mid Y) = \prod_{(i,j)\in\mathcal{O}}
\Phi(\beta_j x_i - \alpha_j)^{y_{ij}}\,\bigl[1-\Phi(\beta_j x_i - \alpha_j)\bigr]^{1-y_{ij}} .
$$

(Political scientists call $x_i$ an *ideal point*; psychometricians call the same
object an *ability* in two-parameter IRT. Same model.)

You can watch that model generate its own toy data — this is the self-test at the
bottom of `gibbs.py` (`python gibbs.py` runs it):

```python
# gibbs.py — synthetic data with known structure
true_x = np.sort(rng.normal(size=n))                 # true ideal points
true_beta = rng.normal(0, 1.5, size=m)
true_alpha = rng.normal(0, 1.0, size=m)
eta = true_x[:, None] * true_beta[None, :] - true_alpha[None, :]
Y = (eta + rng.standard_normal((n, m)) > 0).astype(float)   # add N(0,1), threshold at 0
```

That last line *is* the model: form $\beta_j x_i - \alpha_j$, add standard-normal
noise, keep the sign. Our whole job is to invert it — recover `true_x` knowing only
`Y`.

### Priors and the three symmetries

We put **fixed, proper** priors on everything:

$$
x_i \sim \mathcal{N}(0,1), \qquad
\beta_j \sim \mathcal{N}(0,\sigma_\beta^2), \qquad
\alpha_j \sim \mathcal{N}(0,\sigma_\alpha^2),
\qquad \sigma_\beta = 2,\;\; \sigma_\alpha = 2.5 .
$$

The unit-variance prior on $x$ pulls double duty as an **identification** device.
The index $\beta_j x_i - \alpha_j$ has three symmetries that leave the likelihood
unchanged: **translation** (absorbed by $\alpha$, fixed by the zero mean),
**scale** ($x\!\uparrow,\beta\!\downarrow$, fixed by unit variance), and
**reflection** ($x\mapsto-x,\beta\mapsto-\beta$ — *not* fixed by a symmetric prior).
The last one we kill with an **anchor** (§6). The fixed proper prior also matters
numerically: on a perfectly party-line vote a diffuse prior would let $\beta_j$ run
to infinity; $\sigma_\beta=2$ keeps it finite.

---

## 3. Why this needs sampling

By Bayes' rule the posterior is the likelihood × prior, normalized:

$$
p(X,\alpha,\beta \mid Y) =
\frac{L(Y\mid\cdot)\,\prod_i p(x_i)\,\prod_j p(\alpha_j)\,p(\beta_j)}{Z},
\qquad Z = \int L\,p \; dX\,d\alpha\,d\beta .
$$

The normalizing constant $Z$ is an integral over $n + 2m$ parameters — hundreds of
MPs, thousands of votes — and is **intractable**. But we never need it: $Z$ only
rescales the density's *height*, not its *shape*, and everything we report (a
club's mean position, an MP's credible interval) is an **expectation** under the
posterior, estimated by averaging over draws, $\mathbb{E}[g(\theta)]\approx
\frac1S\sum_s g(\theta^{(s)})$ — where $Z$ cancels out. So instead of computing the
density we build a **Markov chain whose stationary distribution is the posterior**
and read the answer off its draws. That is what the next sections construct.

---

## 4. Getting the data: the Sejm API

Everything starts from one public, key-free API: `api.sejm.gov.pl`. The encoding we
target is simple — **1 = yes, 0 = no, missing = abstain/absent**:

```python
# fetch_data.py
API_BASE = "https://api.sejm.gov.pl/sejm"
#   1   = YES
#   0   = NO
#   NaN = ABSENT or ABSTAIN (treated as missing-at-random)
# Only ELECTRONIC votes are included.
```

First the roster — who sits in this term, and in which club:

```python
# fetch_data.py — the MP list
def fetch_mps(term="term10"):
    data = _get(f"{API_BASE}/{term}/MP")
    rows = [{"mp_id": mp["id"],
             "first_name": mp.get("firstLastName", "").split()[0],
             "last_name":  mp.get("lastFirstName", "").split()[0],
             "club":       mp.get("club", "")} for mp in data]
    return pd.DataFrame(rows).set_index("mp_id")
```

Then the votes. The API exposes each *proceeding* as a list of votings
(`votings/{term}/{p}`) with an exact `votingNumber`, and the per-MP detail at
`votings/{term}/{p}/{n}`. We enumerate first, then fetch the details **in
parallel** with a thread pool (the network wait dominates, so threads help):

```python
# fetch_data.py — enumerate every electronic voting, then fetch details in parallel
def _list_proceeding(p):
    lst = _fetch_json(f"{API_BASE}/{term}/votings/{p}") or []
    return [(p, v["votingNumber"]) for v in lst if v.get("kind") == "ELECTRONIC"]

with ThreadPoolExecutor(max_workers=12) as ex:
    candidates = [pn for chunk in ex.map(_list_proceeding, proceedings) for pn in chunk]

def _detail(pn):
    p, n = pn
    return pn, _fetch_json(f"{API_BASE}/{term}/votings/{p}/{n}")

with ThreadPoolExecutor(max_workers=24) as ex:
    details = {pn: d for pn, d in ex.map(_detail, candidates) if d is not None}
```

Each detail carries a `votes` array — one record per MP. We turn it into one column
of the matrix $Y$, mapping YES→1, NO→0, everything else→missing:

```python
# fetch_data.py — assemble one column of Y from a voting's per-MP records
col = np.full(len(mp_ids), np.nan, dtype=np.float32)
for v in detail.get("votes", []):
    idx = mp_index.get(v["MP"])
    if idx is None:
        continue
    if   v.get("vote") == "YES": col[idx] = 1.0
    elif v.get("vote") == "NO":  col[idx] = 0.0
    # ABSTAIN / ABSENT -> stays NaN (missing-at-random in the model)
```

Finally, drop the **near-unanimous** votes: if 95% vote the same way, the vote
tells us nothing about *where* MPs sit, only adds noise. Keep votes whose minority
is at least 5%:

```python
# fetch_data.py — keep only contested votes (minority >= 5%)
keep = []
for j in range(Y.shape[1]):
    observed = Y[:, j][~np.isnan(Y[:, j])]
    if len(observed) and unanimity_threshold > observed.mean() > (1 - unanimity_threshold):
        keep.append(j)
Y_filtered = Y[:, keep]
```

The result is cached to `data/` as a pickle, so you fetch once and iterate offline.
What comes out is a single matrix $Y$ (≈500 MPs × ≈2,500 contested votes) of 1s,
0s, and NaNs. That matrix is the only thing the sampler sees.

---

## 5. The matrix, and breaking the mirror

Inside the sampler, missing entries are handled with a boolean **mask** so they
never contribute to any update:

```python
# gibbs.py — split observed values from the missingness mask
mask = ~np.isnan(Y_raw)
Y = np.where(mask, Y_raw, 0).astype(np.int8)
n, m = Y.shape                       # n MPs, m votes
maskf = mask.astype(np.float64)      # 1.0 where observed, 0.0 where missing
```

Every sum below is a masked dot product, so an MP who skipped a vote simply drops
out of that vote's arithmetic.

Now the reflection symmetry from §2. Flip every sign and $\beta_j x_i$ is
unchanged, so the likelihood can't tell our orientation from its mirror image —
different chains would settle on opposite directions. We pin it by forcing one
high-turnout reference MP to the **positive** side. Picking that MP is the whole of
`anchor.py`:

```python
# anchor.py — the reference MP: most-active member of the anchor club
def find_anchor_idx(Y_raw, mp_ids, mp_info, anchor_club="PiS"):
    n_votes_per_mp = (~np.isnan(Y_raw)).sum(axis=1)
    best_idx, best_count = 0, -1
    for i, mid in enumerate(mp_ids):
        club = mp_info.loc[mid, "club"] if mid in mp_info.index else ""
        if club == anchor_club and n_votes_per_mp[i] > best_count:
            best_idx, best_count = i, n_votes_per_mp[i]
    return best_idx
```

That MP gets special treatment in every sweep (§6.2): drawn from a normal
**truncated to the positive half-line**, so it can never cross zero:

```python
# gibbs.py — single draw from N(mean, sd^2) truncated to (0, +inf)
def _trunc_pos(mean, sd, rng, eps=1e-10):
    phi_lo = ndtr(-mean / sd)
    u = rng.uniform()
    p = np.clip(phi_lo + u * (1.0 - phi_lo), eps, 1.0 - eps)
    return mean + sd * ndtri(p)        # inverse-CDF sampling
```

That sign is the *only* orientation we impose. Everything else is the data's.

---

## 6. The sampler: four closed-form draws

### 6.1 The augmentation trick

Sampling $\Phi(\beta_j x_i-\alpha_j)$ directly is awkward: the product $\beta_j x_i$
makes the posterior curved, and perfectly party-line votes push $\beta_j$ toward
infinity. Albert & Chib's fix is to **invent a latent variable**. For every cell,
imagine a continuous "utility"

$$
y^{\ast}_{ij} = \beta_j x_i - \alpha_j + \varepsilon_{ij}, \qquad
\varepsilon_{ij}\sim\mathcal N(0,1), \qquad
y_{ij} = \mathbf{1}\!\left[\,y^{\ast}_{ij} > 0\,\right].
$$

The observed yes/no is just the **sign** of $y^{\ast}$ — this reproduces the probit
model exactly. But *conditional on* $y^{\ast}$, everything downstream is plain
linear regression with Gaussian noise. So each Gibbs sweep alternates four cheap,
closed-form draws. Here they are, one numbered block at a time (open the
`for it in range(...)` loop in `gibbs.py` to follow along).

### 6.2 Draw the latent utilities $y^{\ast}$

Given the current parameters, each $y^{\ast}_{ij}$ is a normal centred at
$\eta_{ij}=\beta_j x_i-\alpha_j$, **truncated to match the observed vote**: positive
where $y_{ij}=1$, negative where $y_{ij}=0$.

$$
y^{\ast}_{ij}\mid\cdot \;\sim\;
\begin{cases}
\mathcal N(\eta_{ij},1)\restriction(0,\infty) & y_{ij}=1\\
\mathcal N(\eta_{ij},1)\restriction(-\infty,0) & y_{ij}=0
\end{cases}
$$

Vectorized inverse-CDF sampling does all $n\times m$ cells at once:

```python
# gibbs.py — sample N(mean,1) truncated to the side picked by y
def _truncated_normal(mean, y, rng, eps=1e-10):
    u = rng.uniform(size=mean.shape)
    phi_lo = ndtr(-mean)                                   # mass below 0
    p = np.where(y == 1, phi_lo + u * (1.0 - phi_lo),      # y==1 -> (0, +inf)
                         u * phi_lo)                       # y==0 -> (-inf, 0)
    return mean + ndtri(np.clip(p, eps, 1.0 - eps))

# ...inside the sweep:
eta   = x[:, None] * beta[None, :] - alpha[None, :]        # (n, m)
ystar = _truncated_normal(eta, Y, rng)
```

### 6.3 Draw the ideal points $x_i$

With $y^{\ast}$ fixed, recovering each $x_i$ is a one-variable Gaussian regression
of $r_{ij}=y^{\ast}_{ij}+\alpha_j$ on $\beta_j$. The $\mathcal N(0,1)$ prior adds a
$+1$ to the precision:

$$
x_i\mid\cdot \sim \mathcal N\!\left(\frac{\sum_j \beta_j r_{ij}}{1+\sum_j \beta_j^2},\;
\frac{1}{1+\sum_j \beta_j^2}\right)
$$

(sums over that MP's observed votes only). The code reads like the formula:

```python
# gibbs.py — Gaussian full conditional for the ideal points x_i
def update_ideal_points(ystar, beta, alpha, maskf, anchor_idx, rng):
    r = ystar + alpha[None, :]                 # (n, m)
    prec_x = 1.0 + maskf @ (beta ** 2)         # (n,)  = 1 + sum_j beta_j^2  (prior + data)
    rhs_x = (maskf * r) @ beta                 # (n,)  = sum_j beta_j r_ij
    mean_x = rhs_x / prec_x
    sd_x = 1.0 / np.sqrt(prec_x)
    x = mean_x + sd_x * rng.standard_normal(prec_x.shape[0])
    x[anchor_idx] = _trunc_pos(mean_x[anchor_idx], sd_x[anchor_idx], rng)   # keep anchor > 0
    return x
```

That last line is where the anchor from §5 earns its keep.

### 6.4 Draw the vote parameters $(\beta_j,\alpha_j)$

Symmetrically, with $x$ fixed each vote is a regression of $y^{\ast}_{ij}$ on the
predictor $(x_i,-1)$ — a **bivariate** Gaussian for $(\beta_j,\alpha_j)$. The
posterior precision is a $2\times2$ matrix from masked sufficient statistics, with
the priors $p_\beta=1/\sigma_\beta^2$ and $p_\alpha=1/\sigma_\alpha^2$ on the
diagonal:

$$
A_j=\begin{pmatrix}\sum_i x_i^2+p_\beta & -\sum_i x_i\\[2pt] -\sum_i x_i & \sum_i 1+p_\alpha\end{pmatrix},
\qquad
(\beta_j,\alpha_j)\mid\cdot \sim \mathcal N\!\bigl(A_j^{-1}b_j,\;A_j^{-1}\bigr).
$$

Inverting a $2\times2$ in closed form and sampling via its Cholesky factor keeps it
vectorized across all $m$ votes at once:

```python
# gibbs.py — bivariate Gaussian full conditional for (beta_j, alpha_j), all votes at once
def update_vote_params(x, ystar, maskf, pb, pa, rng):
    Sxx = (x ** 2) @ maskf            # sum_i x_i^2   (observed)
    Sx = x @ maskf                    # sum_i x_i
    Sn = maskf.sum(axis=0)            # observed count per vote
    Sxy = x @ (maskf * ystar)         # sum_i x_i y*_ij
    Sy = (maskf * ystar).sum(axis=0)  # sum_i y*_ij

    a11, a12, a22 = Sxx + pb, -Sx, Sn + pa            # precision A = [[a11,a12],[a12,a22]]
    det = a11 * a22 - a12 * a12
    c11, c12, c22 = a22 / det, -a12 / det, a11 / det  # covariance A^{-1}
    mean_b = c11 * Sxy + c12 * (-Sy)
    mean_a = c12 * Sxy + c22 * (-Sy)
    L11 = np.sqrt(c11)                                # 2x2 Cholesky of covariance
    L21 = c12 / L11
    L22 = np.sqrt(np.maximum(c22 - L21 ** 2, 1e-12))
    m = Sn.shape[0]
    z1, z2 = rng.standard_normal(m), rng.standard_normal(m)
    beta = mean_b + L11 * z1
    alpha = mean_a + L21 * z1 + L22 * z2
    return beta, alpha
```

`pb` and `pa` are set once from $\sigma_\beta=2$, $\sigma_\alpha=2.5$:

```python
# gibbs.py — fixed proper priors keep beta finite on perfectly separating votes
pb = 1.0 / sigma_beta ** 2     # sigma_beta = 2.0
pa = 1.0 / sigma_alpha ** 2    # sigma_alpha = 2.5
```

### 6.5 Pin the scale (parameter expansion)

One symmetry is still loose: scale $x$ up and $\beta$ down by the same factor and no
$\eta_{ij}$ changes, so the overall scale drifts between chains. The fix is to
**re-standardize $x$ to mean 0 / SD 1 after every sweep**, pushing the location and
scale into $(\beta,\alpha)$ so that $\eta=\beta x-\alpha$ is *identical* — only the
coordinates change (Liu–Wu parameter expansion):

```python
# gibbs.py — standardize x each sweep; absorb shift/scale into (beta, alpha)
def standardize_scale(x, beta, alpha, anchor_idx):
    b = x.mean()
    x = x - b
    alpha = alpha - beta * b          # center: preserves eta
    s = x.std()
    x = x / s
    beta = beta * s                   # scale: preserves eta (alpha unchanged)
    if x[anchor_idx] < 0:             # keep the anchor strictly positive
        x, beta = -x, -beta
    return x, beta, alpha
```

A cheap step that buys a lot of mixing.

### 6.6 One sweep, then thousands

Stack the four functions and you have the entire engine. This is the real loop in
`gibbs.py` — one sweep calls the four blocks in sequence, then keeps the
post-warmup draws:

```python
# gibbs.py — the whole sampler is this loop (the four functions are §6.2–6.5)
for it in range(num_warmup + num_samples * thin):
    eta = x[:, None] * beta[None, :] - alpha[None, :]
    ystar = _truncated_normal(eta, Y, rng)                               # 1) latent utilities
    x = update_ideal_points(ystar, beta, alpha, maskf, anchor_idx, rng)  # 2) ideal points
    beta, alpha = update_vote_params(x, ystar, maskf, pb, pa, rng)       # 3) vote params
    if standardize:                                                      # 4) parameter expansion
        x, beta, alpha = standardize_scale(x, beta, alpha, anchor_idx)
    if it >= num_warmup and (it - num_warmup) % thin == 0:
        draws_x[(it - num_warmup) // thin] = x                           # keep this draw
```

No accept/reject. No step-size tuning. No gradients. Every line is a closed-form
draw, and one sweep on the full Sejm is ~74 ms on a laptop CPU.

### 6.7 Many chains, in parallel

Within a chain the draws are sequential, but **chains are independent** — the one
part of MCMC that is embarrassingly parallel. The demo runs one OS process per
chain:

```python
# gibbs.py — one process per chain, results stacked into (chains, draws, ...)
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=n_jobs) as ex:
    results = list(ex.map(_chain_worker, tasks))
out = {"x": np.stack([r["x"] for r in results])}
```

Four chains let us measure convergence with $\hat R$ — the
between-vs-within-chain variance ratio (want $< 1.01$).

---

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

> **How to read the axis.** I deliberately do **not** label it "left–right".
> Roll-call votes certainly carry ideological signal — but turning that signal into
> an ideological verdict is a judgment I choose not to make. This project supplies
> the **data**, recovered as a **main axis of division**, and leaves the
> interpretation to you. In this term that axis most plausibly tracks the
> **government–opposition** split — a term-specific mapping.

![Ideal points, colored by club](figures/ideal_points.png)
![Distribution by club](figures/club_distributions.png)

---

## 8. Run it yourself

Now the payoff. Clone the repo and install the handful of dependencies:

```bash
git clone https://github.com/psephos-lab/sejm-ideal-points-model-demo
cd sejm-ideal-points-model-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, pandas, arviz, matplotlib, requests
```

Then run the full pipeline — fetch (cached after the first time), sample, diagnose,
write a per-MP table:

```bash
python run.py --term term10        # X kadencja (since 2023); try term9, term8, …
```

---

## 9. Now make it yours

You have the repo, you've run it, and you've seen every line that matters. Things to
poke at:

- **Tighten or loosen the prior.** `--sigma-beta 1` vs `--sigma-beta 5`: watch what
  happens to discrimination on perfectly party-line votes.
- **Switch terms.** `--term term7`, `--term term8`, … — the same code maps six
  parliaments.
- **Break the anchor.** Comment out the `_trunc_pos` line in §6.3 and watch chains
  disagree on orientation — the reflection symmetry, live.
- **Drop the standardization.** Set `standardize=False` and watch $\hat R$ for the
  scale degrade — §6.5, live.

The model is a few hundred lines of NumPy. Once you've read it once, it stops being
a black box — and that was the whole point.

---

*Repo: [`sejm-ideal-points-model-demo`](https://github.com/psephos-lab/sejm-ideal-points-model-demo)
· Live map: [mapasejmu.org](https://mapasejmu.org)
· Data: [api.sejm.gov.pl](https://api.sejm.gov.pl)*
*Disclaimer: an "ideal point" is a position in vote space, not a judgment of a
politician. Uncertainty grows for MPs who vote rarely.*
