"""
Main entry point for Sejm ideal point estimation (Albert-Chib Gibbs sampler).

Usage:
  python run.py                          # fetch data + run full model
  python run.py --fetch-only             # only download/cache data
  python run.py --warmup 1000 --samples 2000 --chains 4
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import arviz as az

from fetch_data import fetch_rollcall, filter_rollcall
from gibbs import run_multichain_parallel
from model import find_anchor_idx
from visualize import plot_ideal_points, plot_club_distributions, plot_trace

RESULTS_DIR = "results"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fetch-only", action="store_true")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--jobs", type=int, default=4, help="parallel processes (chains run concurrently)")
    p.add_argument("--thin", type=int, default=1)
    p.add_argument("--sigma-beta", type=float, default=2.0)
    p.add_argument("--sigma-alpha", type=float, default=2.5)
    p.add_argument("--unanimity-threshold", type=float, default=0.95)
    p.add_argument("--term", default="term10", help="Sejm term, e.g. term10 or term9")
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "" if args.term == "term10" else f"_{args.term}"   # term10 keeps default filenames

    # --- Data ---
    print(f"=== Loading data ({args.term}) ===")
    data = filter_rollcall(fetch_rollcall(term=args.term, verbose=True),
                           unanimity_threshold=args.unanimity_threshold)
    Y, mp_ids, mp_info = data["Y"], data["mp_ids"], data["mp_info"]
    obs_rate = (~np.isnan(Y)).mean()
    print(f"\nMatrix: {Y.shape[0]} MPs × {Y.shape[1]} votes")
    print(f"Observation rate: {obs_rate:.1%}  |  Mean YES: {np.nanmean(Y):.1%}")
    print("\nClub sizes:")
    print(mp_info["club"].value_counts().to_string())

    if args.fetch_only:
        return

    # --- Anchor (breaks reflection: a high-turnout PiS MP forced to x>0) ---
    anchor = find_anchor_idx(Y, mp_ids, mp_info)
    a_mid = mp_ids[anchor]
    print(f"\nAnchor (x>0): {mp_info.loc[a_mid,'first_name']} {mp_info.loc[a_mid,'last_name']} (PiS)")

    # --- Sampling ---
    print(f"\n=== Gibbs ({args.warmup} warmup + {args.samples} samples × {args.chains} chains) ===")
    out = run_multichain_parallel(
        Y, anchor,
        num_chains=args.chains,
        n_jobs=args.jobs,
        num_warmup=args.warmup,
        num_samples=args.samples,
        thin=args.thin,
        sigma_beta=args.sigma_beta,
        sigma_alpha=args.sigma_alpha,
    )

    # --- Diagnostics ---
    idata = az.convert_to_datatree({"x": out["x"]})
    rhat = az.rhat(idata)["x"].values
    ess = az.ess(idata)["x"].values
    print("\n--- Convergence (ideal points x) ---")
    print(f"R-hat: max={rhat.max():.3f}  mean={rhat.mean():.3f}  (want < 1.01)")
    print(f"ESS:   min={ess.min():.0f}  mean={ess.mean():.0f}")

    # --- Save raw draws ---
    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"draws{tag}.npz"),
        x=out["x"], beta=out["beta"], alpha=out["alpha"],
        mp_ids=np.array(mp_ids),
    )

    # --- Per-MP estimate table (the deliverable) ---
    x_flat = out["x"].reshape(-1, len(mp_ids))
    est = pd.DataFrame({
        "mp_id": mp_ids,
        "first_name": [mp_info.loc[m, "first_name"] if m in mp_info.index else "" for m in mp_ids],
        "last_name": [mp_info.loc[m, "last_name"] if m in mp_info.index else "" for m in mp_ids],
        "club": [mp_info.loc[m, "club"] if m in mp_info.index else "" for m in mp_ids],
        "x_mean": x_flat.mean(0),
        "x_sd": x_flat.std(0),
        "x_lo90": np.percentile(x_flat, 5, axis=0),
        "x_hi90": np.percentile(x_flat, 95, axis=0),
        "rhat": rhat,
    }).sort_values("x_mean")
    est.to_csv(os.path.join(RESULTS_DIR, f"ideal_points{tag}.csv"), index=False)
    print(f"\nSaved per-MP estimates to {RESULTS_DIR}/ideal_points{tag}.csv")

    # --- Validation: club ordering on the main axis ---
    print("\n--- Mean position by club (along the main axis) ---")
    by_club = est.groupby("club")["x_mean"].agg(["mean", "count"]).sort_values("mean")
    for club, row in by_club.iterrows():
        print(f"  {club:18s} {row['mean']:+.2f}  (n={int(row['count'])})")

    # --- Plots ---
    term_label = {"term10": "X kadencji", "term9": "IX kadencji",
                  "term8": "VIII kadencji", "term7": "VII kadencji"}.get(args.term, args.term)
    plot_ideal_points(x_flat, mp_ids, mp_info, term_label=term_label,
                      save_path=os.path.join(RESULTS_DIR, f"ideal_points{tag}.png"))
    plot_club_distributions(x_flat, mp_ids, mp_info, term_label=term_label,
                            save_path=os.path.join(RESULTS_DIR, f"club_distributions{tag}.png"))
    plot_trace(out["x"], mp_ids, mp_info,
               save_path=os.path.join(RESULTS_DIR, f"trace{tag}.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
