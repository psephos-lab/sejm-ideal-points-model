"""
Export per-vote model parameters (posterior-mean beta, alpha) for the contested
votes the 1D model was fit on, keyed by "{sitting}_{voting}" so the site can look
them up against the history votes.

Model: P(za) = Phi(beta_j * x_i - alpha_j); model cutting point x* = alpha_j/beta_j.
"""

import json
import numpy as np
from fetch_data import fetch_rollcall, filter_rollcall

OUT = "docs/model_params.json"


def main():
    d = np.load("results/draws.npz", allow_pickle=True)
    beta = d["beta"].reshape(-1, d["beta"].shape[-1]).mean(0)    # (n_votes,)
    alpha = d["alpha"].reshape(-1, d["alpha"].shape[-1]).mean(0)

    # filtered vote_meta is in the same column order as beta/alpha (deterministic)
    vm = filter_rollcall(fetch_rollcall(verbose=False))["vote_meta"]
    assert len(vm) == len(beta), f"mismatch {len(vm)} vs {len(beta)}"

    params = {
        f"{v['sitting']}_{v['voting_num']}": [round(float(beta[k]), 4), round(float(alpha[k]), 4)]
        for k, v in enumerate(vm)
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(params, f, separators=(",", ":"))
    import os
    print(f"Wrote {OUT}: {len(params)} votes ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
