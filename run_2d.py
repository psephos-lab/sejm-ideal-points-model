"""
Two-dimensional ideal point estimation for the Sejm + interpretation of dim 2.

Runs the N-D Gibbs sampler with D=2, aligns draws (Procrustes), fixes signs,
then:
  - reports per-club means on both dimensions
  - lists the votes that load most strongly on dimension 2 (these DEFINE it)
  - saves a 2D scatter of MPs colored by club
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import arviz as az

from fetch_data import fetch_rollcall, filter_rollcall
from gibbs_nd import run_multichain_nd, align_draws, fix_signs, target_rotate
from model import find_anchor_idx
from visualize import CLUB_COLORS, DEFAULT_COLOR

RESULTS = "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--thin", type=int, default=2)
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    data = filter_rollcall(fetch_rollcall(verbose=False))
    Y, mp_ids, mp_info, vote_meta = data["Y"], data["mp_ids"], data["mp_info"], data["vote_meta"]
    n, m = Y.shape
    anchor = find_anchor_idx(Y, mp_ids, mp_info)
    print(f"Matrix {n}x{m}; anchor = {mp_info.loc[mp_ids[anchor],'last_name']} (PiS)")

    t0 = time.time()
    out = run_multichain_nd(Y, D=2, num_chains=args.chains, n_jobs=args.jobs,
                            num_warmup=args.warmup, num_samples=args.samples, thin=args.thin)
    C, Dr, _, D = out["x"].shape
    sweeps = args.warmup + args.samples * args.thin
    print(f"draws: {C} chains x {Dr} = {C*Dr} total ({sweeps} sweeps/chain)")
    print(f"[TIMING] wall = {time.time()-t0:.0f}s")

    # --- align across all draws, target-rotate dim1 to the 1D axis, fix signs ---
    flatX = out["x"].reshape(C * Dr, n, D)
    flatB = out["beta"].reshape(C * Dr, m, D)
    flatX, flatB = align_draws(flatX, flatB, n_iter=4)
    try:
        x1d = np.load(os.path.join(RESULTS, "draws.npz"), allow_pickle=True)["x"]
        x1d = x1d.reshape(-1, n).mean(0)
        flatX, flatB = target_rotate(flatX, flatB, x1d)
        print("Applied target rotation: dim1 aligned to 1D solution")
    except FileNotFoundError:
        print("No 1D solution (results/draws.npz); keeping Procrustes orientation")
    flatX, flatB = fix_signs(flatX, flatB, anchor)

    # --- per-dim convergence (R-hat on aligned draws) ---
    Xc = flatX.reshape(C, Dr, n, D)
    for d in range(D):
        idata = az.convert_to_datatree({"x": Xc[:, :, :, d]})
        rhat = az.rhat(idata)["x"].values
        ess = az.ess(idata)["x"].values
        print(f"dim {d+1}: R-hat max={rhat.max():.3f} mean={rhat.mean():.3f} | ESS min={ess.min():.0f} mean={ess.mean():.0f}")

    x_mean = flatX.mean(0)        # (n, D)
    b_mean = flatB.mean(0)        # (m, D)

    # --- club means on both dims ---
    clubs = np.array([mp_info.loc[mid, "club"] if mid in mp_info.index else "?" for mid in mp_ids])
    print("\n--- Club means (dim1 = govt-opposition, dim2 = ?) ---")
    rows = []
    for c in sorted(set(clubs)):
        msk = clubs == c
        rows.append((c, x_mean[msk, 0].mean(), x_mean[msk, 1].mean(), msk.sum()))
    for c, d1, d2, k in sorted(rows, key=lambda r: r[1]):
        print(f"  {c:18s} dim1={d1:+.2f}  dim2={d2:+.2f}  (n={k})")

    # --- which votes DEFINE dim 2? (high |b2| relative to |b1|) ---
    # angle from dim-1 axis; near 90 deg => pure dim-2 vote
    ang = np.degrees(np.arctan2(np.abs(b_mean[:, 1]), np.abs(b_mean[:, 0])))
    mag = np.hypot(b_mean[:, 0], b_mean[:, 1])
    dim2_score = np.abs(b_mean[:, 1]) * (mag > np.percentile(mag, 50))  # discriminating + dim2-loaded
    top = np.argsort(dim2_score)[::-1][:15]
    print("\n--- Top votes loading on DIMENSION 2 (these define it) ---")
    for j in top:
        t = (vote_meta[j].get("topic") or vote_meta[j].get("title") or "")[:90]
        print(f"  b=({b_mean[j,0]:+.2f},{b_mean[j,1]:+.2f}) ang={ang[j]:.0f}deg | {t}")

    # --- save table + scatter ---
    est = pd.DataFrame({
        "mp_id": mp_ids, "club": clubs,
        "last_name": [mp_info.loc[m_, "last_name"] if m_ in mp_info.index else "" for m_ in mp_ids],
        "dim1": x_mean[:, 0], "dim2": x_mean[:, 1],
    })
    est.to_csv(os.path.join(RESULTS, "ideal_points_2d.csv"), index=False)
    # save x draws (for CI/R-hat) + beta MEAN only (full beta draws would be GBs)
    np.savez_compressed(os.path.join(RESULTS, "draws_2d.npz"),
                        x=Xc, beta_mean=b_mean, mp_ids=np.array(mp_ids))

    fig, ax = plt.subplots(figsize=(11, 9))
    for c in sorted(set(clubs)):
        msk = clubs == c
        ax.scatter(x_mean[msk, 0], x_mean[msk, 1],
                   c=CLUB_COLORS.get(c, DEFAULT_COLOR), s=22, label=c, alpha=0.8,
                   edgecolors="white", linewidths=0.3)
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.axvline(0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax.set_xlabel("Wymiar 1 (główna oś)")
    ax.set_ylabel("Wymiar 2 (wtórny podział)")
    ax.set_title("Punkty idealne 2D — Sejm X kadencji")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "ideal_points_2d.png"), dpi=150)
    print(f"\nSaved results/ideal_points_2d.png and .csv")


if __name__ == "__main__":
    main()
