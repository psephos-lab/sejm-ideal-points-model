"""
Bayesian ideal point estimation for the Polish Sejm.

NOTE: jax-metal 0.1.1 has an unimplemented default_memory_space on Apple Silicon;
we force CPU until Apple releases a compatible plugin update.

Model (logit link — numerically equivalent to probit, more stable in JAX):

  Hyperpriors:
    sigma_beta ~ HalfCauchy(2.5)          — SD of vote discrimination
    sigma_alpha ~ HalfCauchy(2.5)         — SD of vote thresholds
    mu_alpha ~ Normal(0, 1)               — mean threshold (absorbs coalition majority bias)

  Non-centered parameterization (avoids Neal's funnel):
    z_beta[j] ~ Normal(0, 1)
    z_alpha[j] ~ Normal(0, 1)
    beta[j]  = sigma_beta * z_beta[j]
    alpha[j] = mu_alpha + sigma_alpha * z_alpha[j]

  Ideal points (scale + location identification via unit-variance prior):
    x[i] ~ Normal(0, 1)

  Likelihood:
    P(y_ij = 1) = sigmoid(beta[j] * x[i] - alpha[j])

  Identification:
    - Translation and scale fixed by N(0,1) prior on x.
    - Reflection (sign flip of x and beta simultaneously) is NOT fixed by the
      symmetric prior — resolved post-hoc by flipping so that the PiS centroid
      is positive (right-wing = positive axis convention).
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # jax-metal 0.1.1 breaks on Apple Silicon
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="arviz")


def ideal_point_model(Y_obs: jnp.ndarray, mask: jnp.ndarray) -> None:
    """
    Y_obs : (n_mps, n_votes) int32 — 1=YES, 0=NO (NaN positions zeroed, masked out)
    mask  : (n_mps, n_votes) bool  — True where vote was observed
    """
    n_mps, n_votes = Y_obs.shape

    # --- hyperpriors ---
    sigma_beta = numpyro.sample("sigma_beta", dist.HalfCauchy(2.5))
    sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfCauchy(2.5))
    mu_alpha = numpyro.sample("mu_alpha", dist.Normal(0.0, 1.0))

    # --- ideal points ---
    x = numpyro.sample("x", dist.Normal(jnp.zeros(n_mps), jnp.ones(n_mps)))

    # --- vote parameters (non-centered) ---
    z_beta = numpyro.sample("z_beta", dist.Normal(jnp.zeros(n_votes), jnp.ones(n_votes)))
    z_alpha = numpyro.sample("z_alpha", dist.Normal(jnp.zeros(n_votes), jnp.ones(n_votes)))

    beta = numpyro.deterministic("beta", sigma_beta * z_beta)
    alpha = numpyro.deterministic("alpha", mu_alpha + sigma_alpha * z_alpha)

    # --- linear predictor (n_mps, n_votes) ---
    eta = x[:, None] * beta[None, :] - alpha[None, :]

    # --- likelihood with missing-data mask ---
    with numpyro.handlers.mask(mask=mask):
        numpyro.sample("Y", dist.Bernoulli(logits=eta), obs=Y_obs)


def prepare_arrays(Y_raw: np.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert float matrix with NaNs → (Y_int, mask) JAX arrays."""
    mask = ~np.isnan(Y_raw)
    Y_int = np.where(mask, Y_raw, 0).astype(np.int32)
    return jnp.array(Y_int), jnp.array(mask)


def run_nuts(
    Y_raw: np.ndarray,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    seed: int = 42,
) -> object:
    """
    Run NUTS on the ideal point model.

    num_chains chains run in parallel via vmap on the same device.
    Returns an ArviZ InferenceData object.
    """
    Y_obs, mask = prepare_arrays(Y_raw)

    kernel = NUTS(ideal_point_model, target_accept_prob=0.85)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",  # vmap — all chains on GPU simultaneously
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, Y_obs=Y_obs, mask=mask)

    idata = az.from_numpyro(mcmc)
    return idata


def fix_reflection(idata: object, mp_ids: list, mp_info) -> object:
    """
    Resolve the reflection ambiguity: ensure the PiS bloc has positive mean x.
    If not, flip the sign of x and beta across all chains and samples.
    """
    x_samples = idata.posterior["x"].values  # (chains, draws, n_mps)

    pis_mask = np.array([mp_info.loc[mid, "club"] == "PiS" for mid in mp_ids])
    pis_mean = x_samples[:, :, pis_mask].mean()

    if pis_mean < 0:
        idata.posterior["x"] = -idata.posterior["x"]
        idata.posterior["beta"] = -idata.posterior["beta"]
        if "z_beta" in idata.posterior:
            idata.posterior["z_beta"] = -idata.posterior["z_beta"]

    return idata


def diagnostics(idata: object) -> None:
    """Print R-hat and ESS summary for scalar/hyperparameters."""
    summary = az.summary(
        idata,
        var_names=["sigma_beta", "sigma_alpha", "mu_alpha"],
        stat_focus="mean",
    )
    print("\n--- Hyperparameter diagnostics ---")
    print(summary.to_string())

    rhat_x = az.rhat(idata, var_names=["x"])["x"].values
    ess_x = az.ess(idata, var_names=["x"])["x"].values
    print(f"\nx  R-hat: max={rhat_x.max():.3f}  mean={rhat_x.mean():.3f}")
    print(f"x  ESS:   min={ess_x.min():.0f}  mean={ess_x.mean():.0f}")


if __name__ == "__main__":
    # Quick smoke test with synthetic 20x50 data
    rng = np.random.default_rng(0)
    Y_test = rng.choice([0.0, 1.0, np.nan], size=(20, 50), p=[0.35, 0.55, 0.10])
    idata = run_nuts(Y_test, num_warmup=200, num_samples=200, num_chains=2)
    print(az.summary(idata, var_names=["sigma_beta", "sigma_alpha", "mu_alpha"]))
