"""
Albert-Chib Gibbs sampler for the Bayesian ideal point (2PL probit) model.

This is the canonical approach for ideal point estimation (Clinton-Jackman-Rivers
2004; pscl::ideal). Data augmentation introduces latent utilities y* so that every
full conditional is Gaussian/truncated-Gaussian — sidestepping the curved
multiplicative beta*x geometry that defeats HMC/NUTS on perfectly party-line votes.

Model:
    y*_ij = beta_j * x_i - alpha_j + eps_ij,   eps ~ N(0,1)
    y_ij  = 1[y*_ij > 0]

Priors (fixed, proper — proper prior keeps discrimination finite under separation):
    x_i     ~ N(0, 1)             (also fixes scale/location identification)
    beta_j  ~ N(0, sigma_beta^2)
    alpha_j ~ N(0, sigma_alpha^2)

Identification:
    - scale & location: unit-variance prior on x
    - reflection: anchor MP forced to x > 0 (truncated-normal full conditional)

Full conditionals (vectorized over all MPs / all votes at once):
    y*_ij | .  ~ TruncNormal(beta_j x_i - alpha_j, 1) truncated by sign of y_ij
    x_i   | .  ~ N(.,.)            Gaussian regression of (y* + alpha) on beta
    (beta_j, alpha_j) | . ~ N2(.,.)  bivariate regression of y* on (x, -1)
                                     — exactly the CJR posterior formula
"""

import os
# Limit BLAS threads per process so parallel chains (one process each) don't
# oversubscribe cores. 4 processes x 2 threads ~ 8 logical cores on M1.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.special import ndtr, ndtri  # standard normal CDF and inverse CDF


def _truncated_normal(mean: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                      eps: float = 1e-10) -> np.ndarray:
    """
    Sample from N(mean, 1) truncated to (0, inf) where y==1, (-inf, 0) where y==0.
    Vectorized inverse-CDF method.
    """
    u = rng.uniform(size=mean.shape)
    # Phi(0 - mean) = Phi(-mean): probability mass below the truncation point 0
    phi_lo = ndtr(-mean)
    # y==1 -> sample in (0, inf):   x = mean + Phi^{-1}(phi_lo + u*(1-phi_lo))
    # y==0 -> sample in (-inf, 0):  x = mean + Phi^{-1}(u*phi_lo)
    p = np.where(y == 1, phi_lo + u * (1.0 - phi_lo), u * phi_lo)
    p = np.clip(p, eps, 1.0 - eps)
    return mean + ndtri(p)


def gibbs_ideal_point(
    Y_raw: np.ndarray,
    anchor_idx: int,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    thin: int = 1,
    sigma_beta: float = 2.0,
    sigma_alpha: float = 2.5,
    seed: int = 0,
    standardize: bool = True,
    verbose: bool = True,
    init: dict | None = None,
    store_vote_params: bool = True,
) -> dict:
    """
    Run the Gibbs sampler.

    init: optional {"x","beta","alpha"} to warm-start (continue) a chain from a
          saved state. With num_warmup=0 this seamlessly extends a previous run
          (the latent y* is resampled each sweep, so it need not be saved). Only
          the starting point changes — the target posterior is identical.
    store_vote_params: if False, skip the full (num_samples, n_votes) beta/alpha
          draw arrays (memory-lean for long continuations) and return their
          running posterior means as beta_mean / alpha_mean instead.

    Returns dict of posterior draws:
        x     (num_samples, n_mps)
        beta  (num_samples, n_votes)        [only if store_vote_params]
        alpha (num_samples, n_votes)        [only if store_vote_params]
        beta_mean / alpha_mean (n_votes)    [only if not store_vote_params]
    """
    rng = np.random.default_rng(seed)

    mask = ~np.isnan(Y_raw)
    Y = np.where(mask, Y_raw, 0).astype(np.int8)
    n, m = Y.shape
    maskf = mask.astype(np.float64)

    # Prior precisions
    pb = 1.0 / sigma_beta ** 2
    pa = 1.0 / sigma_alpha ** 2

    # --- Initialization ---
    # Warm start x from standardized per-MP YES rate (cheap, helps mixing)
    yes_rate = np.array([Y[i, mask[i]].mean() if mask[i].any() else 0.5
                         for i in range(n)])
    x = (yes_rate - yes_rate.mean()) / (yes_rate.std() + 1e-9)
    if x[anchor_idx] < 0:           # ensure anchor starts positive
        x = -x
    beta = rng.normal(0, 0.5, size=m)
    alpha = np.zeros(m)
    ystar = np.zeros((n, m))

    # Warm start (continuation): override the random init with a saved sampler
    # state. Only (x, beta, alpha) are needed; y* is resampled first each sweep.
    if init is not None:
        x = np.asarray(init["x"], dtype=np.float64).copy()
        beta = np.asarray(init["beta"], dtype=np.float64).copy()
        alpha = np.asarray(init["alpha"], dtype=np.float64).copy()
        if x[anchor_idx] < 0:                  # sign-flip preserves eta = beta*x - alpha
            x, beta = -x, -beta

    draws_x = np.empty((num_samples, n))
    if store_vote_params:
        draws_beta = np.empty((num_samples, m))
        draws_alpha = np.empty((num_samples, m))
    else:                                      # memory-lean: running means only
        beta_sum = np.zeros(m)
        alpha_sum = np.zeros(m)
        n_kept = 0

    total = num_warmup + num_samples * thin
    for it in range(total):
        # 1) latent utilities y* (only observed cells matter downstream via mask)
        eta = x[:, None] * beta[None, :] - alpha[None, :]
        ystar = _truncated_normal(eta, Y, rng)

        # 2) ideal points x_i  (Gaussian regression of r = y* + alpha on beta)
        r = ystar + alpha[None, :]                 # (n, m)
        beta2 = beta ** 2                          # (m,)
        prec_x = 1.0 + maskf @ beta2               # (n,)  prior precision 1 + sum beta^2
        rhs_x = (maskf * r) @ beta                 # (n,)  sum_j beta_j r_ij
        mean_x = rhs_x / prec_x
        sd_x = 1.0 / np.sqrt(prec_x)
        x = mean_x + sd_x * rng.standard_normal(n)
        # anchor: redraw the reference MP from the positive-truncated conditional
        x[anchor_idx] = _trunc_pos(mean_x[anchor_idx], sd_x[anchor_idx], rng)

        # 3) vote params (beta_j, alpha_j): bivariate Gaussian regression
        # Sufficient stats over observed MPs (masked sums)
        Sxx = (x ** 2) @ maskf                      # (m,)
        Sx = x @ maskf                              # (m,)
        Sn = maskf.sum(axis=0)                      # (m,)
        msy = maskf * ystar
        Sxy = x @ msy                               # (m,) sum_i x_i y*_ij
        Sy = msy.sum(axis=0)                        # (m,)

        # Posterior precision matrix A_j = X*'X* + diag(pb, pa):
        #   [[Sxx+pb, -Sx], [-Sx, Sn+pa]]
        a11 = Sxx + pb
        a12 = -Sx
        a22 = Sn + pa
        det = a11 * a22 - a12 * a12
        # Covariance = A^{-1} = [[a22, -a12], [-a12, a11]] / det
        c11 = a22 / det
        c12 = -a12 / det
        c22 = a11 / det
        # Posterior mean = Cov @ [Sxy, -Sy]
        bx, ay = Sxy, -Sy
        mean_b = c11 * bx + c12 * ay
        mean_a = c12 * bx + c22 * ay
        # Sample bivariate normal via 2x2 Cholesky of covariance
        L11 = np.sqrt(c11)
        L21 = c12 / L11
        L22 = np.sqrt(np.maximum(c22 - L21 ** 2, 1e-12))
        z1 = rng.standard_normal(m)
        z2 = rng.standard_normal(m)
        beta = mean_b + L11 * z1
        alpha = mean_a + L21 * z1 + L22 * z2

        # 4) parameter expansion: pin x to mean 0 / SD 1 each sweep, absorbing the
        # scale+location into (beta, alpha) so eta = beta*x - alpha is unchanged.
        # Kills the weakly-identified scale/location random walk (Liu-Wu PX-DA;
        # Imai-van Dyk for ideal points) -> chains agree on scale, mixing improves.
        if standardize:
            b = x.mean()
            x = x - b
            alpha = alpha - beta * b          # center: preserves eta
            s = x.std()
            x = x / s
            beta = beta * s                   # scale: preserves eta (alpha unchanged)
            # keep the anchor strictly positive (s>0 preserves sign; guard anyway)
            if x[anchor_idx] < 0:
                x = -x
                beta = -beta

        # --- store ---
        if it >= num_warmup and (it - num_warmup) % thin == 0:
            k = (it - num_warmup) // thin
            draws_x[k] = x
            if store_vote_params:
                draws_beta[k] = beta
                draws_alpha[k] = alpha
            else:
                beta_sum += beta
                alpha_sum += alpha
                n_kept += 1

        if verbose and (it + 1) % 200 == 0:
            print(f"  iter {it+1}/{total}", flush=True)

    # always expose the final sampler state, so a lean run is still a continuable
    # checkpoint (warm-start the next extension from x_last/beta_last/alpha_last)
    out = {"x": draws_x, "x_last": x.copy(), "beta_last": beta.copy(), "alpha_last": alpha.copy()}
    if store_vote_params:
        out["beta"] = draws_beta
        out["alpha"] = draws_alpha
    else:
        out["beta_mean"] = beta_sum / max(n_kept, 1)
        out["alpha_mean"] = alpha_sum / max(n_kept, 1)
    return out


def _trunc_pos(mean: float, sd: float, rng: np.random.Generator,
               eps: float = 1e-10) -> float:
    """Single draw from N(mean, sd^2) truncated to (0, inf)."""
    phi_lo = ndtr(-mean / sd)
    u = rng.uniform()
    p = np.clip(phi_lo + u * (1.0 - phi_lo), eps, 1.0 - eps)
    return mean + sd * ndtri(p)


def run_multichain(
    Y_raw: np.ndarray,
    anchor_idx: int,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    thin: int = 1,
    base_seed: int = 0,
    **kwargs,
) -> dict:
    """
    Run several independent chains and stack into (chains, draws, ...) arrays,
    ready for arviz.from_dict.
    """
    xs, betas, alphas = [], [], []
    for c in range(num_chains):
        if kwargs.get("verbose", True):
            print(f"Chain {c+1}/{num_chains}")
        out = gibbs_ideal_point(
            Y_raw, anchor_idx,
            num_warmup=num_warmup, num_samples=num_samples, thin=thin,
            seed=base_seed + c, **kwargs)
        xs.append(out["x"])
        betas.append(out["beta"])
        alphas.append(out["alpha"])
    return {
        "x": np.stack(xs),
        "beta": np.stack(betas),
        "alpha": np.stack(alphas),
    }


def _chain_worker(args: tuple) -> dict:
    """Top-level worker (picklable) for one chain in a separate process."""
    Y_raw, anchor_idx, seed, kw = args
    return gibbs_ideal_point(Y_raw, anchor_idx, seed=seed, verbose=False, **kw)


def run_multichain_parallel(
    Y_raw: np.ndarray,
    anchor_idx: int,
    num_chains: int = 4,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    thin: int = 1,
    base_seed: int = 0,
    n_jobs: int | None = None,
    inits: list | None = None,
    **kwargs,
) -> dict:
    """
    Run independent chains concurrently (one process per chain) and stack into
    (chains, draws, ...) arrays. Exploits chain-level parallelism — the part of
    MCMC that IS embarrassingly parallel (the within-chain sequence is not).

    inits: optional list (length num_chains) of {"x","beta","alpha"} warm-start
           states — pass together with num_warmup=0 to continue a previous run.
    """
    from concurrent.futures import ProcessPoolExecutor

    n_jobs = n_jobs or num_chains
    kw = dict(num_warmup=num_warmup, num_samples=num_samples, thin=thin, **kwargs)
    tasks = []
    for c in range(num_chains):
        kw_c = dict(kw)
        if inits is not None:
            kw_c["init"] = inits[c]
        tasks.append((Y_raw, anchor_idx, base_seed + c, kw_c))

    print(f"Running {num_chains} chains across {n_jobs} processes...")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        results = list(ex.map(_chain_worker, tasks))

    out = {"x": np.stack([r["x"] for r in results])}
    for key in ("x_last", "beta_last", "alpha_last"):      # per-chain final state
        if all(key in r for r in results):
            out[key] = np.stack([r[key] for r in results])
    if all("beta" in r for r in results):
        out["beta"] = np.stack([r["beta"] for r in results])
        out["alpha"] = np.stack([r["alpha"] for r in results])
    if all("beta_mean" in r for r in results):
        out["beta_mean"] = np.stack([r["beta_mean"] for r in results])
        out["alpha_mean"] = np.stack([r["alpha_mean"] for r in results])
    return out


if __name__ == "__main__":
    # Smoke test on synthetic data with known structure
    rng = np.random.default_rng(1)
    n, m = 80, 200
    true_x = np.sort(rng.normal(size=n))
    true_beta = rng.normal(0, 1.5, size=m)
    true_alpha = rng.normal(0, 1.0, size=m)
    eta = true_x[:, None] * true_beta[None, :] - true_alpha[None, :]
    Y = (eta + rng.standard_normal((n, m)) > 0).astype(float)
    # punch in some missingness
    Y[rng.uniform(size=(n, m)) < 0.1] = np.nan

    anchor = int(np.argmax(true_x))  # anchor MP at the positive pole
    out = gibbs_ideal_point(Y, anchor, num_warmup=500, num_samples=500, verbose=True)
    x_hat = out["x"].mean(0)
    # correlation with truth (sign-aligned via anchor)
    r = np.corrcoef(x_hat, true_x)[0, 1]
    print(f"\nRecovery correlation (x_hat vs true_x): {r:.3f}")
