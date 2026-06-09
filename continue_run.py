"""
Continue (warm-start) a saved 1-D Gibbs run: extend each chain from its last
state with NO extra burn-in, concatenate to the previous draws, and recompute
diagnostics. This reuses the prior samples instead of redoing burn-in.

It is a valid Markov-chain continuation: only the starting point changes, the
target posterior is identical — so cross-term comparability is preserved. The
move is also incremental: rerun this again to extend further from the new state.

Memory-lean: the continuation keeps only the (small) x draws plus running
posterior means of beta/alpha (enough for the site's model_params), so a long
extension does not materialise the multi-GB beta/alpha draw arrays.

Usage:
  python continue_run.py --term term8 --more 20000
"""

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import arviz as az

from fetch_data import fetch_rollcall, filter_rollcall
from gibbs import run_multichain_parallel
from model import find_anchor_idx


def diag(arr, label):
    idata = az.convert_to_datatree({"x": arr})
    r = az.rhat(idata)["x"].values
    e = az.ess(idata)["x"].values
    print(f"  {label:22s} R-hat max={r.max():.3f} mean={r.mean():.3f} "
          f"| >1.05:{int((r > 1.05).sum()):3d} >1.1:{int((r > 1.1).sum()):3d} "
          f"| ESS min={e.min():.0f} mean={e.mean():.0f}", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="term8")
    ap.add_argument("--more", type=int, default=20000, help="extra samples per chain")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--input", default=None,
                    help="checkpoint to continue from (default results/draws{tag}.npz); "
                         "pass results/draws{tag}_x.npz to extend a prior continuation")
    args = ap.parse_args()
    tag = "" if args.term == "term10" else f"_{args.term}"
    old_path = args.input or f"results/draws{tag}.npz"

    # data + anchor (identical recipe to run.py, so the anchor matches the old run)
    data = filter_rollcall(fetch_rollcall(term=args.term, verbose=False))
    Y, mp_ids, mp_info = data["Y"], data["mp_ids"], data["mp_info"]
    anchor = find_anchor_idx(Y, mp_ids, mp_info)

    # previous draws -> per-chain final state (warm-start) + old x + old vote-param means.
    # Accepts BOTH formats: run.py's full-draw npz (x/beta/alpha) and a checkpoint
    # from a prior continuation (x + x_last/beta_last/alpha_last + beta_mean/alpha_mean).
    d = np.load(old_path)
    old_x = d["x"]                                   # (C, S, n) — all prior draws
    C, S_old, n = old_x.shape
    if "x_last" in d.files:                          # checkpoint from a prior continuation
        inits = [{"x": d["x_last"][c], "beta": d["beta_last"][c], "alpha": d["alpha_last"][c]}
                 for c in range(C)]
        old_bmean, old_amean = d["beta_mean"], d["alpha_mean"]
    else:                                            # original run.py output (full draws)
        beta_all, alpha_all = d["beta"], d["alpha"]  # (C, S, m) each
        inits = [{"x": old_x[c, -1], "beta": beta_all[c, -1], "alpha": alpha_all[c, -1]}
                 for c in range(C)]
        old_bmean = beta_all.reshape(-1, beta_all.shape[-1]).mean(0)
        old_amean = alpha_all.reshape(-1, alpha_all.shape[-1]).mean(0)
        del beta_all, alpha_all
    old_x = np.asarray(old_x)                         # materialise, then close the npz
    del d

    print(f"[{args.term}] continuing {C} chains from saved final state "
          f"(S_old={S_old}): +{args.more} samples, warmup=0", flush=True)
    out = run_multichain_parallel(
        Y, anchor, num_chains=C, n_jobs=args.jobs,
        num_warmup=0, num_samples=args.more, base_seed=1000,
        inits=inits, store_vote_params=False,
        sigma_beta=2.0, sigma_alpha=2.5,
    )
    new_x = out["x"]                                  # (C, more, n)
    x_full = np.concatenate([old_x, new_x], axis=1)   # (C, S_old+more, n)

    print(f"\n=== diagnostics ({args.term}) ===")
    diag(old_x, f"old ({S_old})")
    diag(new_x, f"new ({args.more})")
    rhat = diag(x_full, f"combined ({S_old + args.more})")

    # combined posterior means of vote params (weighted by sample count) for the site
    nb = out["beta_mean"].mean(0)
    na = out["alpha_mean"].mean(0)
    comb_b = (S_old * old_bmean + args.more * nb) / (S_old + args.more)
    comb_a = (S_old * old_amean + args.more * na) / (S_old + args.more)

    np.savez_compressed(
        f"results/draws{tag}_x.npz", x=x_full, mp_ids=np.array(mp_ids),
        beta_mean=comb_b, alpha_mean=comb_a,
        x_last=out["x_last"], beta_last=out["beta_last"], alpha_last=out["alpha_last"],
    )

    xf = x_full.reshape(-1, n)
    est = pd.DataFrame({
        "mp_id": mp_ids,
        "first_name": [mp_info.loc[m, "first_name"] if m in mp_info.index else "" for m in mp_ids],
        "last_name": [mp_info.loc[m, "last_name"] if m in mp_info.index else "" for m in mp_ids],
        "club": [mp_info.loc[m, "club"] if m in mp_info.index else "" for m in mp_ids],
        "x_mean": xf.mean(0), "x_sd": xf.std(0),
        "x_lo90": np.percentile(xf, 5, axis=0), "x_hi90": np.percentile(xf, 95, axis=0),
        "rhat": rhat,
    }).sort_values("x_mean")
    est.to_csv(f"results/ideal_points{tag}.csv", index=False)

    print("\n--- club ordering on the main axis (combined) ---")
    by_club = est.groupby("club")["x_mean"].agg(["mean", "count"]).sort_values("mean")
    for club, row in by_club.iterrows():
        print(f"  {club:14s} {row['mean']:+.2f}  (n={int(row['count'])})")

    print(f"\nSaved results/draws{tag}_x.npz (x draws + combined beta/alpha means) "
          f"and results/ideal_points{tag}.csv")
    print("Done.")


if __name__ == "__main__":
    main()
