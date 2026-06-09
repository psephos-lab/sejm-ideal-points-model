"""
Regenerate the result figures (ideal_points / club_distributions / trace) for a
term from its saved posterior draws — without re-running the sampler. Prefers the
extended continuation checkpoint results/draws{tag}_x.npz if present, else the
original results/draws{tag}.npz.

Usage:
  python replot.py --term term8
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np

from fetch_data import fetch_rollcall
from visualize import plot_ideal_points, plot_club_distributions, plot_trace

ROMAN = {"term10": "X kadencji", "term9": "IX kadencji",
         "term8": "VIII kadencji", "term7": "VII kadencji"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="term8")
    ap.add_argument("--input", default=None, help="override draws .npz path")
    args = ap.parse_args()
    tag = "" if args.term == "term10" else f"_{args.term}"

    ext = f"results/draws{tag}_x.npz"
    path = args.input or (ext if os.path.exists(ext) else f"results/draws{tag}.npz")
    print(f"[{args.term}] plotting from {path}")

    d = np.load(path)
    x = d["x"]                                   # (chains, draws, n_mps)
    mp_ids = list(d["mp_ids"])
    mp_info = fetch_rollcall(term=args.term, verbose=False)["mp_info"]
    label = ROMAN.get(args.term, args.term)
    x_flat = x.reshape(-1, x.shape[-1])
    print(f"  draws: {x.shape} (chains, samples, MPs)")

    plot_ideal_points(x_flat, mp_ids, mp_info, term_label=label,
                      save_path=f"results/ideal_points{tag}.png")
    plot_club_distributions(x_flat, mp_ids, mp_info, term_label=label,
                            save_path=f"results/club_distributions{tag}.png")
    plot_trace(x, mp_ids, mp_info, save_path=f"results/trace{tag}.png")
    print("Done.")


if __name__ == "__main__":
    main()
