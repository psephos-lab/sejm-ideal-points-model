# Build the Sampler Yourself: A Hands-On Ideal-Point Model of the Polish Sejm

**A code-first companion to [Part 1](ARTICLE.md).** Same model, same math — but
every equation here comes paired with the **exact Python that runs it**, lifted
straight from a small public repo you can clone and execute in five minutes.

> **The repo:** [`sejm-ideal-points-model-demo`](https://github.com/psephos-lab/sejm-ideal-points-model-demo).
> Every snippet below is a real excerpt from its `gibbs.py`, `anchor.py`, or
> `run.py` — not pseudocode. Pure NumPy/SciPy; no jax, no PyMC, no Stan.
> Math uses `$…$`; on Medium, render formulas as images.

```bash
git clone https://github.com/psephos-lab/sejm-ideal-points-model-demo
cd sejm-ideal-points-model-demo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # numpy, scipy, pandas, arviz, matplotlib, requests
```

Keep that terminal open. By the end you will have run the whole thing on real
roll-call data and printed a left-to-right ordering of the Polish parliament that
**no human labelled** — it falls out of the votes alone.

---

## What we're building

Each MP $i$ has a hidden position $x_i$ on a latent axis. Each vote $j$ has a
**discrimination** $\beta_j$ (how sharply it splits the chamber) and a
**threshold** $\alpha_j$. The probability MP $i$ votes "yes" on vote $j$ is

$$
P(y_{ij} = 1 \mid x_i,\alpha_j,\beta_j) = \Phi(\beta_j x_i - \alpha_j),
$$

a two-parameter probit IRT / spatial-voting model ($\Phi$ = standard normal CDF).
[Part 1](ARTICLE.md) derives this from random-utility theory; here we take it as
given and **estimate** all the $x_i,\beta_j,\alpha_j$ at once from the matrix of
yeas and nays.

The estimator is a **Gibbs sampler with Albert–Chib data augmentation**. The whole
trick — and the reason this is a few matrix ops instead of a fight with a
gradient sampler — is one auxiliary variable. We'll get there step by step, and at
each step you'll see the code that does it.

You can even watch the model generate its own toy data. This is the bottom of
`gibbs.py` (`python gibbs.py` runs it as a self-test):

```python
# gibbs.py — synthetic data with known structure
true_x = np.sort(rng.normal(size=n))                 # true ideal points
true_beta = rng.normal(0, 1.5, size=m)
true_alpha = rng.normal(0, 1.0, size=m)
eta = true_x[:, None] * true_beta[None, :] - true_alpha[None, :]
Y = (eta + rng.standard_normal((n, m)) > 0).astype(float)   # probit: add N(0,1), threshold at 0
```

That last line *is* the model: form $\beta_j x_i - \alpha_j$, add standard-normal
noise, keep the sign. Our job is to invert it — recover `true_x` knowing only `Y`.

---

## Step 0 — The data: a matrix of yeas and nays

The input is one matrix $Y$: rows are MPs, columns are votes, entries are 1 (yes),
0 (no), or missing (didn't vote). `run.py` pulls it from the official Sejm API and
drops near-unanimous votes (they carry no positional information):

```python
# run.py — get the roll-call matrix for one term
data = filter_rollcall(fetch_rollcall(term=args.term, verbose=True),
                       unanimity_threshold=args.unanimity_threshold)
Y, mp_ids, mp_info = data["Y"], data["mp_ids"], data["mp_info"]
```

Inside the sampler, missing entries are handled with a boolean **mask** so they
never contribute to any update:

```python
# gibbs.py — split observed values from the missingness mask
mask = ~np.isnan(Y_raw)
Y = np.where(mask, Y_raw, 0).astype(np.int8)
n, m = Y.shape                       # n MPs, m votes
maskf = mask.astype(np.float64)      # 1.0 where observed, 0.0 where missing
```

Every sum in the sampler is written as a masked dot product, so an MP who skipped
a vote simply drops out of that vote's arithmetic.

---

## Step 1 — Break the mirror: the anchor

The model has a symmetry that will wreck it if left alone. Flip **every** sign,

$$
(x_i,\beta_j)\mapsto(-x_i,-\beta_j),
$$

and $\beta_j x_i$ is unchanged — so the likelihood can't tell "our" orientation
from its mirror image, and different chains will settle on opposite directions.
We pin it down by forcing one high-turnout reference MP to the **positive** side.
Picking that MP is pure NumPy (the entire `anchor.py`):

```python
# anchor.py — the reference MP: most-voting member of the anchor club
def find_anchor_idx(Y_raw, mp_ids, mp_info, anchor_club="PiS"):
    n_votes_per_mp = (~np.isnan(Y_raw)).sum(axis=1)
    best_idx, best_count = 0, -1
    for i, mid in enumerate(mp_ids):
        club = mp_info.loc[mid, "club"] if mid in mp_info.index else ""
        if club == anchor_club and n_votes_per_mp[i] > best_count:
            best_idx, best_count = i, n_votes_per_mp[i]
    return best_idx
```

That index gets special treatment inside every sweep (Step 3 below): instead of an
ordinary Gaussian draw, the anchor is drawn from a normal **truncated to the
positive half-line**, so it can never cross zero:

```python
# gibbs.py — single draw from N(mean, sd^2) truncated to (0, +inf)
def _trunc_pos(mean, sd, rng, eps=1e-10):
    phi_lo = ndtr(-mean / sd)
    u = rng.uniform()
    p = np.clip(phi_lo + u * (1.0 - phi_lo), eps, 1.0 - eps)
    return mean + sd * ndtri(p)        # inverse-CDF sampling
```

That choice of sign is the *only* orientation we impose. Everything else is the
data's.

---

## Step 2 — The augmentation trick

Sampling $\Phi(\beta_j x_i-\alpha_j)$ directly is awkward: the product $\beta_j x_i$
makes the posterior curved, and perfectly party-line votes push $\beta_j$ toward
infinity. Albert & Chib's fix is to **invent a latent variable**. For every cell,
imagine a continuous "utility"

$$
y^{\ast}_{ij} = \beta_j x_i - \alpha_j + \varepsilon_{ij}, \qquad
\varepsilon_{ij}\sim\mathcal N(0,1), \qquad
y_{ij} = \mathbf{1}\!\left[\,y^{\ast}_{ij} > 0\,\right].
$$

The observed yes/no is just the **sign** of $y^{\ast}$. This reproduces the probit
model exactly — but *conditional on* $y^{\ast}$, everything downstream is plain
linear regression with Gaussian noise. Each Gibbs sweep now alternates four cheap,
closed-form draws. The rest of this article is those four draws.

---

## Step 3 — The four full conditionals

Open `gibbs.py` and find the `for it in range(total):` loop. Here it is, one
numbered block at a time.

### 3a — Draw the latent utilities $y^{\ast}$

Given the current parameters, each $y^{\ast}_{ij}$ is a normal centred at
$\eta_{ij}=\beta_j x_i-\alpha_j$, **truncated to match the observed vote**:
positive where $y_{ij}=1$, negative where $y_{ij}=0$.

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
    p = np.clip(p, eps, 1.0 - eps)
    return mean + ndtri(p)

# ...inside the sweep:
eta   = x[:, None] * beta[None, :] - alpha[None, :]        # (n, m)
ystar = _truncated_normal(eta, Y, rng)
```

### 3b — Draw the ideal points $x_i$

With $y^{\ast}$ fixed, recovering each $x_i$ is a one-variable Gaussian regression
of $r_{ij}=y^{\ast}_{ij}+\alpha_j$ on $\beta_j$. The $\mathcal N(0,1)$ prior adds a
$+1$ to the precision:

$$
x_i\mid\cdot \sim \mathcal N\!\left(\frac{\sum_j \beta_j r_{ij}}{1+\sum_j \beta_j^2},\;
\frac{1}{1+\sum_j \beta_j^2}\right)
$$

(sums over that MP's observed votes only). The code reads exactly like the formula:

```python
# gibbs.py — Gaussian update for the ideal points
r      = ystar + alpha[None, :]            # (n, m)
beta2  = beta ** 2                         # (m,)
prec_x = 1.0 + maskf @ beta2               # (n,)  = 1 + sum_j beta_j^2   (prior + data)
rhs_x  = (maskf * r) @ beta                # (n,)  = sum_j beta_j r_ij
mean_x = rhs_x / prec_x
sd_x   = 1.0 / np.sqrt(prec_x)
x      = mean_x + sd_x * rng.standard_normal(n)
x[anchor_idx] = _trunc_pos(mean_x[anchor_idx], sd_x[anchor_idx], rng)   # keep anchor > 0
```

That last line is where the anchor from Step 1 earns its keep.

### 3c — Draw the vote parameters $(\beta_j,\alpha_j)$

Symmetrically, with $x$ fixed each vote is a regression of $y^{\ast}_{ij}$ on the
predictor $(x_i,-1)$ — a **bivariate** Gaussian for $(\beta_j,\alpha_j)$. The
posterior precision is a $2\times2$ matrix built from masked sufficient statistics,
with the priors adding $p_\beta=1/\sigma_\beta^2$ and $p_\alpha=1/\sigma_\alpha^2$
on the diagonal:

$$
A_j=\begin{pmatrix}\sum_i x_i^2+p_\beta & -\sum_i x_i\\[2pt] -\sum_i x_i & \sum_i 1+p_\alpha\end{pmatrix},
\qquad
(\beta_j,\alpha_j)\mid\cdot \sim \mathcal N\!\bigl(A_j^{-1}b_j,\;A_j^{-1}\bigr).
$$

Inverting a $2\times2$ in closed form and sampling via its Cholesky factor keeps
the whole thing vectorized across all $m$ votes:

```python
# gibbs.py — bivariate Gaussian update for (beta_j, alpha_j), all votes at once
Sxx = (x ** 2) @ maskf            # sum_i x_i^2   (observed)
Sx  = x @ maskf                   # sum_i x_i
Sn  = maskf.sum(axis=0)           # observed count per vote
Sxy = x @ (maskf * ystar)         # sum_i x_i y*_ij
Sy  = (maskf * ystar).sum(axis=0) # sum_i y*_ij

a11, a12, a22 = Sxx + pb, -Sx, Sn + pa       # precision A = [[a11,a12],[a12,a22]]
det = a11 * a22 - a12 * a12
c11, c12, c22 = a22 / det, -a12 / det, a11 / det   # covariance A^{-1}
mean_b = c11 * Sxy + c12 * (-Sy)
mean_a = c12 * Sxy + c22 * (-Sy)
L11 = np.sqrt(c11); L21 = c12 / L11               # 2x2 Cholesky of covariance
L22 = np.sqrt(np.maximum(c22 - L21 ** 2, 1e-12))
z1, z2 = rng.standard_normal(m), rng.standard_normal(m)
beta  = mean_b + L11 * z1
alpha = mean_a + L21 * z1 + L22 * z2
```

`pb` and `pa` are set once from the priors $\sigma_\beta=2$, $\sigma_\alpha=2.5$:

```python
# gibbs.py — fixed proper priors keep beta finite on perfectly separating votes
pb = 1.0 / sigma_beta ** 2     # sigma_beta = 2.0
pa = 1.0 / sigma_alpha ** 2    # sigma_alpha = 2.5
```

### 3d — Pin the scale (parameter expansion)

One symmetry is still loose: you can scale $x$ up and $\beta$ down by the same
factor without changing any $\eta_{ij}$, so the overall scale drifts between
chains. The fix is to **re-standardize $x$ to mean 0 / SD 1 after every sweep**,
pushing the location and scale into $(\beta,\alpha)$ so that $\eta=\beta x-\alpha$
is *identical* — only the coordinates change (Liu–Wu parameter expansion):

```python
# gibbs.py — standardize x each sweep; absorb shift/scale into (beta, alpha)
if standardize:
    b = x.mean();  x = x - b;  alpha = alpha - beta * b   # center: eta unchanged
    s = x.std();   x = x / s;  beta  = beta * s           # scale:  eta unchanged
    if x[anchor_idx] < 0:                                 # guard the anchor's sign
        x, beta = -x, -beta
```

This single normalization is the difference between chains that agree on a scale
and chains that wander — a cheap line that buys a lot of mixing.

---

## Step 4 — One sweep, then thousands

Stack the four blocks and you have the entire engine. Stripped to its skeleton:

```python
# gibbs.py — the whole sampler is this loop
for it in range(num_warmup + num_samples * thin):
    eta   = x[:, None] * beta[None, :] - alpha[None, :]
    ystar = _truncated_normal(eta, Y, rng)        # 3a  latent utilities
    x     = gaussian_update_x(ystar, beta, alpha) # 3b  ideal points  (+ anchor)
    beta, alpha = gaussian_update_votes(ystar, x) # 3c  vote params
    standardize_(x, beta, alpha)                  # 3d  parameter expansion
    if it >= num_warmup and (it - num_warmup) % thin == 0:
        draws_x[k] = x                            # keep post-warmup draws
```

No accept/reject. No step-size tuning. No gradients. Every line is a closed-form
draw, and one sweep on the full Sejm (≈500 MPs × ≈2,500 votes) is ~74 ms on a
laptop CPU.

---

## Step 5 — Many chains, in parallel

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

Four chains let us measure convergence by comparing them ($\hat R$, the
between-vs-within-chain variance ratio).

---

## Step 6 — Run it, and read the map

Now the payoff. With the venv from the top still active:

```bash
python run.py --term term10        # X kadencja (since 2023); try term9, term8, …
```

`run.py` samples, checks convergence with ArviZ, and writes a per-MP table:

```python
# run.py — convergence, then the deliverable table
idata = az.convert_to_datatree({"x": out["x"]})
rhat  = az.rhat(idata)["x"].values         # want < 1.01
est = pd.DataFrame({
    "mp_id": mp_ids, "club": clubs,
    "x_mean": x_flat.mean(0),
    "x_lo90": np.percentile(x_flat, 5, axis=0),
    "x_hi90": np.percentile(x_flat, 95, axis=0),
    "rhat":  rhat,
}).sort_values("x_mean")
```

On the 10th term (4 chains × 10,000 draws, $\hat R_{\max}=1.065$) the club averages
come out cleanly separated — **with no labels given to the model**:

| Club | mean $x$ |
|---|---|
| KO | −0.99 |
| Lewica | −0.80 |
| Konfederacja | +0.41 |
| PiS | +1.14 |

The chamber is strikingly **bimodal** — two blocs, an almost empty centre: the
signature of tight party discipline. (How to read that axis — and why it's *not*
labelled "left–right" — is discussed in [Part 1](ARTICLE.md).)

---

## Step 7 — Now make it yours

You have the repo, you've run it, and you've seen every line that matters. Things
to poke at:

- **Tighten or loosen the prior.** `--sigma-beta 1` vs `--sigma-beta 5`: watch what
  happens to discrimination on perfectly party-line votes (Part 1, §6.1 explains why
  a diffuse prior backfires here).
- **Switch terms.** `--term term7`, `--term term8`, … — the same code maps six
  parliaments.
- **Break the anchor.** Comment out the `_trunc_pos` line in Step 3b and watch
  chains disagree on orientation — the symmetry from Step 1, live.
- **Drop the standardization.** Set `standardize=False` and watch $\hat R$ for the
  scale degrade — Step 3d, live.
- **Go 2-D.** `run_2d.py` adds a second latent axis; the
  [second article](ARTICLE_2D.md) covers the extra identification work that needs.

The model is a few hundred lines of NumPy. Once you've read it once, it stops being
a black box — and that was the whole point.

---

*Repo: [`sejm-ideal-points-model-demo`](https://github.com/psephos-lab/sejm-ideal-points-model-demo)
· Full method: [`sejm-ideal-points-model`](https://github.com/psephos-lab/sejm-ideal-points-model)
· Data: [api.sejm.gov.pl](https://api.sejm.gov.pl)*
*Disclaimer: an "ideal point" is a position in vote space, not a judgment of a
politician. Uncertainty grows for MPs who vote rarely.*
