"""
Interpret dimension 2: for the votes that load most purely on dim 2, show the
full topic AND how each major club actually voted (mean YES rate). The cleavage
in those per-club YES rates reveals what dim 2 *means*.

Run after run_2d.py has produced results/draws_2d.npz.
"""

import warnings; warnings.filterwarnings("ignore")
import numpy as np
from fetch_data import fetch_rollcall, filter_rollcall

MAJOR = ["PiS", "KO", "PSL-TD", "Polska2050", "Lewica", "Razem", "Konfederacja"]


def main():
    data = filter_rollcall(fetch_rollcall(verbose=False))
    Y, mp_ids, mp_info, vote_meta = data["Y"], data["mp_ids"], data["mp_info"], data["vote_meta"]
    clubs = np.array([mp_info.loc[mid, "club"] if mid in mp_info.index else "?" for mid in mp_ids])

    d = np.load("results/draws_2d.npz", allow_pickle=True)
    if "beta_mean" in d:
        b_mean = d["beta_mean"]            # (m, D)
    else:
        beta = d["beta"]
        b_mean = beta.reshape(-1, beta.shape[2], beta.shape[3]).mean(0)

    b1, b2 = b_mean[:, 0], b_mean[:, 1]
    mag = np.hypot(b1, b2)
    angle = np.degrees(np.arctan2(np.abs(b2), np.abs(b1)))
    # "pure dim-2" votes: high overall discrimination AND angle near 90 deg
    pure = (mag > np.percentile(mag, 60)) & (angle > 60)
    order = np.argsort(np.where(pure, np.abs(b2), -1))[::-1][:12]

    def club_yes(j):
        col = Y[:, j]
        out = {}
        for c in MAJOR:
            m = (clubs == c) & ~np.isnan(col)
            out[c] = col[m].mean() if m.sum() else np.nan
        return out

    print("=== Votes loading most purely on DIMENSION 2 ===")
    print("(per-club YES rate; the split across clubs reveals what dim 2 is)\n")
    hdr = "  ".join(f"{c[:5]:>5}" for c in MAJOR)
    print(f"{'b1':>5} {'b2':>6} ang  {hdr}  topic")
    for j in order:
        cy = club_yes(j)
        rates = "  ".join(f"{(cy[c]*100):>4.0f}%" if not np.isnan(cy[c]) else "   - " for c in MAJOR)
        topic = (vote_meta[j].get("topic") or vote_meta[j].get("title") or "")[:70]
        print(f"{b1[j]:+.1f} {b2[j]:+.2f} {angle[j]:.0f}  {rates}  {topic}")

    # Summary: correlation of dim-2 club position with a left-right economic guess
    print("\n--- Club position on dim 2 (sorted) ---")
    x = d["x"]; x_mean = x.reshape(-1, x.shape[2], x.shape[3]).mean(0)
    for c in sorted(set(clubs), key=lambda c: np.nanmean(x_mean[clubs == c, 1])):
        msk = clubs == c
        print(f"  {c:18s} dim2={x_mean[msk,1].mean():+.2f}  dim1={x_mean[msk,0].mean():+.2f}  (n={msk.sum()})")


if __name__ == "__main__":
    main()
