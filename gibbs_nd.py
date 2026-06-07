"""
Multidimensional Albert-Chib Gibbs sampler for ideal points (D >= 1).

Generalizes gibbs.py to D latent dimensions:
    eta_ij = sum_d X_id * B_jd - alpha_j = (X B^T)_ij - alpha_j
    w_ij ~ N(eta_ij, 1),  y_ij = 1[w_ij > 0]

Priors:
    X_i  ~ N(0, I_D)              (per-dim unit scale)
    B_jd ~ N(0, sigma_beta^2)
    alpha_j ~ N(0, sigma_alpha^2)

Identification in D dimensions:
    - scale/location per dim: handled in-sampler by whitening X to mean 0,
      identity covariance each sweep (generalized parameter expansion), absorbing
      the linear map into (B, alpha) so eta is preserved.
    - ROTATION + REFLECTION: NOT fixed during sampling (whitening leaves an
      orthogonal group free). Resolved POST-HOC by Procrustes alignment of every
      draw to a common reference, then a sign convention per dimension.

Full conditionals (batched over all MPs / all votes):
    X_i        ~ N_D(.,.)      D-variate regression of (w + alpha) on B
    (B_j, a_j) ~ N_{D+1}(.,.)  (D+1)-variate regression of w on (X, -1)
"""

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
from scipy.special import ndtr, ndtri


def _truncated_normal(mean, y, rng, eps=1e-10):
    """N(mean,1) truncated to (0,inf) where y==1, (-inf,0) where y==0."""
    u = rng.uniform(size=mean.shape)
    phi_lo = ndtr(-mean)
    p = np.where(y == 1, phi_lo + u * (1.0 - phi_lo), u * phi_lo)
    p = np.clip(p, eps, 1.0 - eps)
    return mean + ndtri(p)


def _batched_mvn(A, rhs, rng):
    """
    Sample from N(A^{-1} rhs, A^{-1}) for a batch of precision matrices.
    A: (K, D, D) SPD; rhs: (K, D). Returns (K, D).
    """
    cov = np.linalg.inv(A)                       # (K,D,D)
    mean = np.einsum("kde,ke->kd", cov, rhs)     # (K,D)
    L = np.linalg.cholesky(cov)                  # (K,D,D)
    z = rng.standard_normal(mean.shape)
    return mean + np.einsum("kde,ke->kd", L, z)


def gibbs_ideal_point_nd(
    Y_raw, D=2, num_warmup=1000, num_samples=2000, thin=1,
    sigma_beta=2.0, sigma_alpha=2.5, seed=0, standardize=True, verbose=True,
):
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(Y_raw)
    Y = np.where(mask, Y_raw, 0).astype(np.int8)
    n, m = Y.shape
    maskf = mask.astype(np.float64)

    pb = 1.0 / sigma_beta ** 2
    pa = 1.0 / sigma_alpha ** 2
    P0 = np.diag([pb] * D + [pa])                # (D+1, D+1) prior precision

    # --- init: X[:,0] from standardized YES-rate, other dims small random ---
    yes_rate = np.array([Y[i, mask[i]].mean() if mask[i].any() else 0.5
                         for i in range(n)])
    X = rng.normal(0, 0.3, size=(n, D))
    X[:, 0] = (yes_rate - yes_rate.mean()) / (yes_rate.std() + 1e-9)
    B = rng.normal(0, 0.5, size=(m, D))
    alpha = np.zeros(m)

    dx = np.empty((num_samples, n, D))
    db = np.empty((num_samples, m, D))
    da = np.empty((num_samples, m))

    total = num_warmup + num_samples * thin
    for it in range(total):
        # 1) latent utilities
        eta = X @ B.T - alpha[None, :]           # (n,m)
        w = _truncated_normal(eta, Y, rng)

        # 2) ideal points X_i  (D-variate regression of r=w+alpha on B)
        r = w + alpha[None, :]
        A_x = np.einsum("ij,jd,je->ide", maskf, B, B) + np.eye(D)[None]   # (n,D,D)
        rhs_x = np.einsum("ij,jd->id", maskf * r, B)                       # (n,D)
        X = _batched_mvn(A_x, rhs_x, rng)

        # 3) vote params (B_j, alpha_j)  ((D+1)-variate regression of w on [X,-1])
        Xa = np.concatenate([X, -np.ones((n, 1))], axis=1)                 # (n,D+1)
        A_b = np.einsum("ij,id,ie->jde", maskf, Xa, Xa) + P0[None]         # (m,D+1,D+1)
        rhs_b = np.einsum("ij,id->jd", maskf * w, Xa)                      # (m,D+1)
        ba = _batched_mvn(A_b, rhs_b, rng)
        B = ba[:, :D]
        alpha = ba[:, D]

        # 4) generalized parameter expansion: whiten X to mean0 / identity cov,
        #    absorb the affine map into (B, alpha) so eta is preserved.
        if standardize:
            b = X.mean(0)                         # (D,)
            X = X - b
            alpha = alpha - B @ b                 # center preserves eta
            S = np.cov(X.T) if D > 1 else np.array([[X[:, 0].var()]])
            vals, vecs = np.linalg.eigh(np.atleast_2d(S))
            vals = np.maximum(vals, 1e-12)
            M = vecs @ np.diag(vals ** -0.5) @ vecs.T     # S^{-1/2}
            X = X @ M
            B = B @ np.linalg.inv(M).T            # preserve eta = X B^T

        if it >= num_warmup and (it - num_warmup) % thin == 0:
            k = (it - num_warmup) // thin
            dx[k] = X; db[k] = B; da[k] = alpha

        if verbose and (it + 1) % 200 == 0:
            print(f"  iter {it+1}/{total}", flush=True)

    return {"x": dx, "beta": db, "alpha": da}


# ---------- Post-hoc identification: Procrustes alignment ----------

def _procrustes_R(Xd, Xref):
    """Orthogonal R (rotation+reflection) minimizing ||Xref - Xd R||."""
    U, _, Vt = np.linalg.svd(Xd.T @ Xref)
    return U @ Vt


def align_draws(draws_x, draws_b, n_iter=3):
    """
    Align all posterior draws of X (and B) to a common orientation via iterated
    orthogonal Procrustes. draws_x: (S, n, D), draws_b: (S, m, D). In place-ish;
    returns aligned copies.
    """
    S, n, D = draws_x.shape
    Xa = draws_x.copy()
    Ba = draws_b.copy()
    ref = Xa[0]
    for _ in range(n_iter):
        for s in range(S):
            R = _procrustes_R(Xa[s], ref)
            Xa[s] = Xa[s] @ R
            Ba[s] = Ba[s] @ R
        ref = Xa.mean(0)
    return Xa, Ba


def target_rotate(draws_x, draws_b, x_target):
    """
    Rotate all draws by a single orthogonal matrix so that dimension 1 aligns
    with x_target (e.g. the 1-D ideal point solution). Makes dim1 comparable to
    the 1-D model and leaves dim2.. as the orthogonal complement. Far more
    interpretable than an arbitrary Procrustes-to-mean basis.
    """
    xm = draws_x.mean(0)                       # (n, D)
    D = xm.shape[1]
    r0 = xm.T @ x_target
    r0 = r0 / np.linalg.norm(r0)
    if D == 2:
        R = np.column_stack([r0, [-r0[1], r0[0]]])
    else:                                       # complete an orthonormal basis
        M = np.column_stack([r0, np.eye(D)[:, 1:]])
        R, _ = np.linalg.qr(M)
        if R[:, 0] @ r0 < 0:
            R[:, 0] *= -1
    return draws_x @ R, draws_b @ R


def fix_signs(draws_x, draws_b, anchor_idx, dim2_ref_idx=None):
    """
    Fix reflection per dimension by convention:
      dim 1: anchor MP (e.g. a PiS hardliner) has positive coordinate.
      dim 2+: the MP with largest |coord| on that dim is positive (stable convention),
              or a supplied reference MP if given.
    """
    Xa = draws_x.copy(); Ba = draws_b.copy()
    mean_x = Xa.mean(0)                            # (n, D)
    D = Xa.shape[2]
    # dim 1: anchor positive
    if mean_x[anchor_idx, 0] < 0:
        Xa[:, :, 0] *= -1; Ba[:, :, 0] *= -1
    # dims 2..: convention = the most extreme MP positive
    for d in range(1, D):
        ref = dim2_ref_idx if (d == 1 and dim2_ref_idx is not None) \
            else int(np.argmax(np.abs(mean_x[:, d])))
        if Xa.mean(0)[ref, d] < 0:
            Xa[:, :, d] *= -1; Ba[:, :, d] *= -1
    return Xa, Ba


def _chain_worker_nd(args):
    Y_raw, D, seed, kw = args
    return gibbs_ideal_point_nd(Y_raw, D=D, seed=seed, verbose=False, **kw)


def run_multichain_nd(Y_raw, D=2, num_chains=4, n_jobs=None, base_seed=0, **kwargs):
    """Run independent N-D chains in parallel (one process each)."""
    from concurrent.futures import ProcessPoolExecutor
    n_jobs = n_jobs or num_chains
    tasks = [(Y_raw, D, base_seed + c, kwargs) for c in range(num_chains)]
    print(f"Running {num_chains} {D}D chains across {n_jobs} processes...")
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        res = list(ex.map(_chain_worker_nd, tasks))
    return {
        "x": np.stack([r["x"] for r in res]),       # (C, draws, n, D)
        "beta": np.stack([r["beta"] for r in res]),
        "alpha": np.stack([r["alpha"] for r in res]),
    }


if __name__ == "__main__":
    # Synthetic 2D recovery test
    rng = np.random.default_rng(2)
    n, m, D = 100, 400, 2
    true_X = rng.normal(size=(n, D))
    true_B = rng.normal(0, 1.3, size=(m, D))
    true_a = rng.normal(0, 1.0, size=m)
    eta = true_X @ true_B.T - true_a[None, :]
    Y = (eta + rng.standard_normal((n, m)) > 0).astype(float)
    Y[rng.uniform(size=(n, m)) < 0.1] = np.nan

    out = gibbs_ideal_point_nd(Y, D=2, num_warmup=500, num_samples=500, verbose=True)
    Xa, Ba = align_draws(out["x"], out["beta"])
    # align posterior mean to truth for scoring (Procrustes to truth)
    xhat = Xa.mean(0)
    R = _procrustes_R(xhat, true_X)
    xhat = xhat @ R
    for d in range(D):
        # allow per-dim sign flip for scoring
        r = np.corrcoef(xhat[:, d], true_X[:, d])[0, 1]
        print(f"dim {d+1}: recovery |corr| = {abs(r):.3f}")
