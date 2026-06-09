"""
Export per-vote model parameters (posterior-mean beta, alpha) for the contested
votes the 1D model was fit on, keyed by "{sitting}_{voting}" so the site can look
them up against the history votes.

Model: P(za) = Phi(beta_j * x_i - alpha_j); model cutting point x* = alpha_j/beta_j.
"""

import argparse
import json
import os
import numpy as np
from fetch_data import fetch_rollcall, filter_rollcall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="term10", help="term10 or term9")
    args = ap.parse_args()
    tag = "" if args.term == "term10" else f"_{args.term}"
    OUT = f"docs/model_params{tag}.json"

    # Prefer the extended continuation checkpoint (posterior means over ALL combined
    # draws); otherwise fall back to the original full-draw npz and average it.
    ext = f"results/draws{tag}_x.npz"
    if os.path.exists(ext):
        d = np.load(ext)
        beta, alpha = d["beta_mean"], d["alpha_mean"]           # (n_votes,)
    else:
        d = np.load(f"results/draws{tag}.npz", allow_pickle=True)
        beta = d["beta"].reshape(-1, d["beta"].shape[-1]).mean(0)
        alpha = d["alpha"].reshape(-1, d["alpha"].shape[-1]).mean(0)

    # filtered vote_meta is in the same column order as beta/alpha (deterministic)
    vm = filter_rollcall(fetch_rollcall(term=args.term, verbose=False))["vote_meta"]
    assert len(vm) == len(beta), f"mismatch {len(vm)} vs {len(beta)}"

    params = {
        f"{v['sitting']}_{v['voting_num']}": [round(float(beta[k]), 4), round(float(alpha[k]), 4)]
        for k, v in enumerate(vm)
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(params, f, separators=(",", ":"))
    print(f"Wrote {OUT}: {len(params)} votes ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
