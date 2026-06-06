"""
Main entry point for Sejm ideal point estimation.

Usage:
  python run.py                        # fetch data + run full model
  python run.py --fetch-only           # only download/cache data
  python run.py --warmup 500 --samples 500 --chains 4
"""

import argparse
import os
os.environ["JAX_PLATFORMS"] = "cpu"  # jax-metal 0.1.1 breaks on Apple Silicon
import numpy as np
import arviz as az

from fetch_data import fetch_rollcall, filter_rollcall
from model import run_nuts, fix_reflection, diagnostics
from visualize import plot_ideal_points, plot_club_distributions, plot_trace

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--unanimity-threshold", type=float, default=0.95,
                        help="Drop votes where one option exceeds this fraction of observed votes")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # --- Data ---
    print("=== Loading data ===")
    data = fetch_rollcall(verbose=True)
    data = filter_rollcall(data, unanimity_threshold=args.unanimity_threshold)

    Y = data["Y"]
    mp_ids = data["mp_ids"]
    mp_info = data["mp_info"]

    obs_rate = (~np.isnan(Y)).mean()
    print(f"\nMatrix: {Y.shape[0]} MPs × {Y.shape[1]} votes")
    print(f"Observation rate: {obs_rate:.1%}  |  Mean YES: {np.nanmean(Y):.1%}")
    print("\nClub sizes:")
    print(mp_info["club"].value_counts().to_string())

    if args.fetch_only:
        return

    # --- Sampling ---
    print(f"\n=== Running NUTS ({args.warmup} warmup + {args.samples} samples × {args.chains} chains) ===")
    idata = run_nuts(
        Y,
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
    )

    # --- Post-processing ---
    idata = fix_reflection(idata, mp_ids, mp_info)
    diagnostics(idata)

    # --- Save ---
    idata_path = os.path.join(RESULTS_DIR, "idata.nc")
    idata.to_netcdf(idata_path)
    print(f"\nInference data saved to {idata_path}")

    # --- Plots ---
    plot_ideal_points(idata, mp_ids, mp_info,
                      save_path=os.path.join(RESULTS_DIR, "ideal_points.png"))
    plot_club_distributions(idata, mp_ids, mp_info,
                            save_path=os.path.join(RESULTS_DIR, "club_distributions.png"))
    plot_trace(idata, save_path=os.path.join(RESULTS_DIR, "trace.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
