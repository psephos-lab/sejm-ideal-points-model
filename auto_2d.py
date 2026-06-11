"""
Unattended 2-D ideal-point orchestrator for the Sejm model.

For each term: a base 2-D run (`--target` samples/chain), then warm-start
continuation by `--step` (NO re-burn-in) until the worst R-hat over BOTH latent
dimensions is <= `--rhat`, or until `--max` samples/chain is reached (then leave
the term as is and move on to the next).

Why 2-D needs special handling for R-hat: the N-D sampler whitens X each sweep
(generalized parameter expansion), which fixes scale/location but leaves the
rotation+reflection free. So before any R-hat/ESS we Procrustes-align ALL draws
(old + new, pooled across chains) to a common reference. R-hat is then the worst
over the two aligned dimensions.

Memory: ideal-point draws (x) are kept across rounds (small: ~C*S*n*D floats).
Full beta draws are materialised only for the BASE run, to get beta_mean (which
votes define dimension 2). On every continuation the chains are warm-started from
their saved final state with store_vote_params=False, so the multi-GB beta/alpha
draw arrays are never built. At output time beta_mean is rotated by the single
orthogonal map that takes the base reference frame to the final aligned frame, so
it stays consistent with the continued x_mean.

This is a valid Markov-chain continuation (only the start point changes; the target
posterior is identical), so cross-term comparability is preserved.

Outputs per term (tag = "" for term10, else _termN):
  results/ideal_points_2d{tag}.csv   mp_id, club, last_name, dim1/2 (+ sd, CI), rhat
  results/ideal_points_2d{tag}.png   club-coloured 2-D scatter
  results/draws_2d{tag}.npz          x draws (C,S,n,D) + beta_mean + final state

Usage (on the box, from ~/mcmc):
  nohup .venv/bin/python auto_2d.py --terms term9,term8,term7 \
        > results/auto_2d.log 2>&1 &
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az

from fetch_data import fetch_rollcall, filter_rollcall
from gibbs_nd import (run_multichain_nd, align_draws, fix_signs,
                      target_rotate, _procrustes_R)
from model import find_anchor_idx
from visualize import CLUB_COLORS, DEFAULT_COLOR

RESULTS = "results"


def tag_for(term):
    return "" if term == "term10" else f"_{term}"


def load_1d_axis(term, n):
    """Posterior-mean 1-D ideal point for this term (to orient dim1), or None."""
    tag = tag_for(term)
    for p in (f"{RESULTS}/draws{tag}_x.npz", f"{RESULTS}/draws{tag}.npz"):
        if os.path.exists(p):
            try:
                x1 = np.load(p)["x"]            # (C, S, n)
                xm = x1.reshape(-1, x1.shape[-1]).mean(0)
                if xm.shape[0] == n:
                    return xm
            except Exception:
                pass
    return None


def rhat_per_dim(Xc):
    """Xc: (C, S, n, D) ALREADY aligned. Returns max R-hat for each dim."""
    D = Xc.shape[-1]
    out = []
    for d in range(D):
        idata = az.convert_to_datatree({"x": Xc[:, :, :, d]})
        out.append(float(np.nanmax(az.rhat(idata)["x"].values)))
    return out


def align_pool(x_raw):
    """x_raw (C, S, n, D) -> Procrustes-aligned (C, S, n, D) and ref mean (n, D)."""
    C, S, n, D = x_raw.shape
    fx, _ = align_draws(x_raw.reshape(C * S, n, D),
                        np.zeros((C * S, 1, D)), n_iter=4)
    return fx.reshape(C, S, n, D), fx.reshape(C * S, n, D).mean(0)


def run_term(term, a):
    print(f"\n===================== {term} =====================", flush=True)
    data = filter_rollcall(fetch_rollcall(term=term, verbose=False))
    Y, mp_ids, mp_info, vote_meta = (data["Y"], data["mp_ids"],
                                     data["mp_info"], data["vote_meta"])
    n, m = Y.shape
    anchor = find_anchor_idx(Y, mp_ids, mp_info)
    print(f"[{term}] {n} MPs x {m} votes; anchor = "
          f"{mp_info.loc[mp_ids[anchor], 'last_name']}", flush=True)

    # --- base run (keeps beta for dim-2 interpretation) ---
    t0 = time.time()
    out = run_multichain_nd(Y, D=2, num_chains=a.chains, n_jobs=a.jobs,
                            num_warmup=a.warmup, num_samples=a.target, thin=a.thin,
                            store_vote_params=True)
    x_pool = out["x"]                                   # (C, S, n, D), raw
    last = [{"x": out["x_last"][c], "beta": out["beta_last"][c],
             "alpha": out["alpha_last"][c]} for c in range(a.chains)]

    # beta_mean in the base aligned frame (its ref = base_ref)
    C, S, _, D = x_pool.shape
    fX, fB = align_draws(x_pool.reshape(C * S, n, D),
                         out["beta"].reshape(C * S, m, D), n_iter=4)
    base_ref = fX.mean(0)                               # (n, D)
    beta_mean = fB.mean(0)                              # (m, D), base frame
    del fX, fB, out
    n_s = S
    print(f"[{term}] base: {C} chains x {n_s} samples ({time.time()-t0:.0f}s)", flush=True)

    # --- continuation loop, driven by aligned-x R-hat over both dims ---
    while True:
        Xc, _ = align_pool(x_pool)
        rd = rhat_per_dim(Xc)
        r = max(rd)
        print(f"[{term}] samples={n_s}  R-hat dim1={rd[0]:.3f} dim2={rd[1]:.3f} "
              f"max={r:.3f}", flush=True)
        if r <= a.rhat:
            print(f"[{term}] CONVERGED: max R-hat {r:.3f} <= {a.rhat}", flush=True)
            break
        if n_s >= a.max:
            print(f"[{term}] hit cap {a.max}; max R-hat {r:.3f} still > {a.rhat} "
                  f"-> leaving as is", flush=True)
            break
        step = min(a.step, a.max - n_s)
        print(f"[{term}] R-hat {r:.3f} > {a.rhat} -> continue +{step} "
              f"(warm-start)", flush=True)
        tc = time.time()
        cont = run_multichain_nd(Y, D=2, num_chains=a.chains, n_jobs=a.jobs,
                                 num_warmup=0, num_samples=step, thin=a.thin,
                                 inits=last, store_vote_params=False,
                                 base_seed=1000 + n_s)
        x_pool = np.concatenate([x_pool, cont["x"]], axis=1)
        last = [{"x": cont["x_last"][c], "beta": cont["beta_last"][c],
                 "alpha": cont["alpha_last"][c]} for c in range(a.chains)]
        n_s += step
        del cont
        print(f"[{term}]   +{step} done ({time.time()-tc:.0f}s)", flush=True)

    write_outputs(term, x_pool, beta_mean, base_ref, anchor,
                  mp_ids, mp_info, vote_meta, Y)


def write_outputs(term, x_pool, beta_mean, base_ref, anchor,
                  mp_ids, mp_info, vote_meta, Y):
    tag = tag_for(term)
    C, S, n, D = x_pool.shape
    m = beta_mean.shape[0]

    # final alignment of ALL x draws; bring beta_mean into the same frame
    flatX, new_ref = align_pool(x_pool)
    flatX = flatX.reshape(C * S, n, D)
    beta_a = beta_mean @ _procrustes_R(base_ref, new_ref)      # base -> aligned frame

    # orient dim1 to the 1-D solution (rotates x draws + beta identically)
    x1d = load_1d_axis(term, n)
    if x1d is not None:
        flatX, b3 = target_rotate(flatX, beta_a[None], x1d)
        beta_a = b3[0]
        print(f"[{term}] dim1 oriented to 1-D solution", flush=True)
    else:
        print(f"[{term}] no 1-D solution found; keeping Procrustes orientation", flush=True)

    # sign convention per dim (anchor positive on dim1, most-extreme MP on dim2)
    flatX, b3 = fix_signs(flatX, beta_a[None], anchor)
    beta_a = b3[0]

    Xc = flatX.reshape(C, S, n, D)
    rd = rhat_per_dim(Xc)
    x_mean = flatX.mean(0)
    x_sd = flatX.std(0)
    x_lo = np.percentile(flatX, 5, axis=0)
    x_hi = np.percentile(flatX, 95, axis=0)
    # per-MP rhat (worst of the two dims) for the csv
    rh_mp = np.maximum(
        az.rhat(az.convert_to_datatree({"x": Xc[:, :, :, 0]}))["x"].values,
        az.rhat(az.convert_to_datatree({"x": Xc[:, :, :, 1]}))["x"].values,
    )

    clubs = np.array([mp_info.loc[mid, "club"] if mid in mp_info.index else "?"
                      for mid in mp_ids])
    last_name = [mp_info.loc[mid, "last_name"] if mid in mp_info.index else ""
                 for mid in mp_ids]

    est = pd.DataFrame({
        "mp_id": mp_ids, "club": clubs, "last_name": last_name,
        "dim1": x_mean[:, 0], "dim2": x_mean[:, 1],
        "dim1_sd": x_sd[:, 0], "dim2_sd": x_sd[:, 1],
        "dim1_lo90": x_lo[:, 0], "dim1_hi90": x_hi[:, 0],
        "dim2_lo90": x_lo[:, 1], "dim2_hi90": x_hi[:, 1],
        "rhat": rh_mp,
    })
    est.to_csv(f"{RESULTS}/ideal_points_2d{tag}.csv", index=False)
    np.savez_compressed(f"{RESULTS}/draws_2d{tag}.npz",
                        x=Xc, beta_mean=beta_a, mp_ids=np.array(mp_ids))

    # which votes define dim 2 (descriptive)
    ang = np.degrees(np.arctan2(np.abs(beta_a[:, 1]), np.abs(beta_a[:, 0])))
    mag = np.hypot(beta_a[:, 0], beta_a[:, 1])
    score = np.abs(beta_a[:, 1]) * (mag > np.percentile(mag, 50))
    print(f"\n[{term}] R-hat final: dim1={rd[0]:.3f} dim2={rd[1]:.3f} "
          f"(N={C}x{S}={C*S})", flush=True)
    print(f"[{term}] --- top votes loading on DIMENSION 2 ---", flush=True)
    for j in np.argsort(score)[::-1][:10]:
        t = (vote_meta[j].get("topic") or vote_meta[j].get("title") or "")[:80]
        print(f"    b=({beta_a[j,0]:+.2f},{beta_a[j,1]:+.2f}) ang={ang[j]:.0f}deg | {t}",
              flush=True)

    # scatter
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
    ax.set_title(f"Punkty idealne 2D — Sejm {term}")
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/ideal_points_2d{tag}.png", dpi=150)
    plt.close(fig)
    print(f"[{term}] DONE: saved ideal_points_2d{tag}.csv/.png + draws_2d{tag}.npz",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default="term9,term8,term7")
    ap.add_argument("--target", type=int, default=5000, help="base run samples/chain")
    ap.add_argument("--step", type=int, default=5000, help="samples added per continuation")
    ap.add_argument("--max", type=int, default=11000, help="hard cap samples/chain (x chains = total)")
    ap.add_argument("--rhat", type=float, default=1.1, help="continue while max R-hat exceeds this")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    terms = [t if t.startswith("term") else f"term{t}" for t in a.terms.split(",")]
    print(f"AUTO-2D start: terms={terms} target={a.target} step={a.step} "
          f"cap={a.max}/chain rhat<={a.rhat} chains={a.chains} jobs={a.jobs}", flush=True)
    for term in terms:
        try:
            run_term(term, a)
        except Exception:
            print(f"[{term}] ERROR:\n{traceback.format_exc()}", flush=True)
    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
