"""
Dimensionality pilot: does a 3rd ideal-point dimension carry real signal, and what
is the autocorrelation time tau at D = 1, 2, 3 (for sizing a full run)?

Three independent "does dim-d exist" checks + tau:
  (1) model-free scree  — SVD of the (imputed, centered) vote matrix; variance per dim.
  (2) classification gain — % votes correctly predicted by the D-dim fit, and APRE
      vs the per-vote-majority baseline. If D=3 ~ D=2, dim 3 adds nothing.
  (3) discrimination per latent dim — singular values of B (rotation-invariant);
      if sigma_3 << sigma_1, dim 3 carries little discrimination (= noise).
  (tau) ESS on aligned draws -> tau = sweeps / ESS, per dimension.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import arviz as az
from fetch_data import fetch_rollcall, filter_rollcall
from gibbs_nd import run_multichain_nd, align_draws
from model import find_anchor_idx

NW, NS, NC = 700, 1200, 4          # short pilots


def main():
    data = filter_rollcall(fetch_rollcall(verbose=False))
    Y, mp_ids, mp_info = data["Y"], data["mp_ids"], data["mp_info"]
    mask = ~np.isnan(Y)
    Yint = np.where(mask, Y, 0.0)
    n, m = Y.shape
    print(f"data: {n} MPs x {m} contested votes")

    # (1) model-free scree
    col_mean = np.nanmean(Y, axis=0)
    Yimp = np.where(mask, Y, col_mean[None, :])
    Yc = Yimp - Yimp.mean(0)
    sv = np.linalg.svd(Yc, compute_uv=False)
    var = sv ** 2 / (sv ** 2).sum()
    print("\n=== (1) model-free scree (SVD of vote matrix) ===")
    print("singular values 1..6:", np.round(sv[:6], 1))
    print("variance % 1..6:     ", np.round(var[:6] * 100, 2))
    print(f"drop dim2->3: {var[1]/var[2]:.2f}x ; dim3->4: {var[2]/var[3]:.2f}x  (big = sharp cutoff)")

    # baseline: predict each vote's majority for everyone
    maj = (Yint * mask).sum(0) / np.maximum(mask.sum(0), 1) > 0.5
    base_acc = (np.broadcast_to(maj[None, :], Y.shape)[mask] == Yint[mask]).mean()
    print(f"\nbaseline (per-vote majority) accuracy: {base_acc:.4f}")

    anchor = find_anchor_idx(Y, mp_ids, mp_info)
    print(f"\n=== pilots: {NC} chains x {NS} (warmup {NW}, thin 1) ===")
    for D in (1, 2, 3):
        out = run_multichain_nd(Y, D=D, num_chains=NC, n_jobs=NC,
                                num_warmup=NW, num_samples=NS, thin=1)
        C, Dr = out["x"].shape[:2]

        # tau via ESS on aligned draws
        fX, fB = align_draws(out["x"].reshape(C * Dr, n, D), out["beta"].reshape(C * Dr, m, D))
        Xc = fX.reshape(C, Dr, n, D)
        ess_dims = [float(np.mean(az.ess(az.convert_to_datatree({"x": Xc[:, :, :, d]}))["x"].values))
                    for d in range(D)]
        sweeps = C * NS
        tau = sweeps / min(ess_dims)

        # (2) classification accuracy (per-chain mean params; eta is rotation-invariant)
        accs = []
        for c in range(C):
            Xm, Bm, am = out["x"][c].mean(0), out["beta"][c].mean(0), out["alpha"][c].mean(0)
            eta = Xm @ Bm.T - am[None, :]
            accs.append(((eta > 0).astype(float)[mask] == Yint[mask]).mean())
        acc = float(np.mean(accs))
        apre = (acc - base_acc) / (1 - base_acc)

        # (3) discrimination per latent dim: singular values of B
        sB = np.linalg.svd(out["beta"][0].mean(0), compute_uv=False)

        print(f"\nD={D}: ESS/dim={[f'{e:.0f}' for e in ess_dims]}  tau(worst)~{tau:.0f} sweeps")
        print(f"      accuracy={acc:.4f}  APRE={apre:.3f}  B singular values={np.round(sB, 2)}")

    print("\nGuide: dim3 'exists' if (1) var%[2] >> var%[3], (2) APRE jumps D=2->3, "
          "(3) sigma_3(B) not tiny vs sigma_1.")


if __name__ == "__main__":
    main()
