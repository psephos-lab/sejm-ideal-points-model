"""
Unattended multi-term orchestrator for the 1-D Sejm ideal-point model.

For each term: run to `target` samples, then keep continuing by `step`
(warm-start, no re-burn-in) until max R-hat <= `rhat`, or until `max` samples
are reached (then leave as is). Resumes from existing draws if a term already
has some, and regenerates the figures at the end. Designed to run detached
(nohup) on the box with no interaction. Results (CSV + PNG) stay in results/.

Usage (on the box, from ~/mcmc):
  nohup .venv/bin/python auto_terms.py --terms term5,term6,term7 \
        > results/auto.log 2>&1 &
"""
import argparse
import os
import subprocess
import sys
import traceback
import numpy as np
import pandas as pd

PY = sys.executable                      # the venv python running this driver


def sh(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def tag_for(term):
    return "" if term == "term10" else f"_{term}"


def n_samples(term):
    """Per-chain sample count from the best available draws, else 0."""
    tag = tag_for(term)
    for p in (f"results/draws{tag}_x.npz", f"results/draws{tag}.npz"):
        if os.path.exists(p):
            try:
                return int(np.load(p)["x"].shape[1])
            except Exception:
                pass
    return 0


def input_path(term):
    """Continuation checkpoint to warm-start from (prefer the extended _x.npz)."""
    tag = tag_for(term)
    ext = f"results/draws{tag}_x.npz"
    return ext if os.path.exists(ext) else f"results/draws{tag}.npz"


def max_rhat(term):
    csv = f"results/ideal_points{tag_for(term)}.csv"
    if not os.path.exists(csv):
        return None
    return float(pd.read_csv(csv)["rhat"].max())


def process(term, a):
    print(f"\n===================== {term} =====================", flush=True)
    n = n_samples(term)
    if n == 0:
        print(f"[{term}] base run: {a.target} samples (fetches data first if no cache)", flush=True)
        sh([PY, "run.py", "--term", term, "--samples", str(a.target), "--jobs", str(a.jobs)])
        n = n_samples(term)
    else:
        print(f"[{term}] resuming from existing {n} samples", flush=True)

    while True:
        r = max_rhat(term)
        print(f"[{term}] samples={n}  max R-hat={r}", flush=True)
        if r is None:
            print(f"[{term}] no estimates CSV (run failed / no data) -> skipping term", flush=True)
            return
        if r <= a.rhat:
            print(f"[{term}] CONVERGED: max R-hat {r:.3f} <= {a.rhat}", flush=True)
            break
        if n >= a.max:
            print(f"[{term}] hit cap {a.max}; max R-hat {r:.3f} still > {a.rhat} -> leaving as is", flush=True)
            break
        print(f"[{term}] R-hat {r:.3f} > {a.rhat} and {n} < {a.max} -> continue +{a.step}", flush=True)
        sh([PY, "continue_run.py", "--term", term, "--more", str(a.step), "--input", input_path(term)])
        n = n_samples(term)

    sh([PY, "replot.py", "--term", term])
    print(f"[{term}] DONE: final samples={n_samples(term)}, max R-hat={max_rhat(term)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", default="term5,term6,term7")
    ap.add_argument("--target", type=int, default=5000, help="base run size")
    ap.add_argument("--step", type=int, default=5000, help="samples added per continuation")
    ap.add_argument("--max", type=int, default=20000, help="hard cap per term")
    ap.add_argument("--rhat", type=float, default=1.1, help="continue while max R-hat exceeds this")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    terms = [t if t.startswith("term") else f"term{t}" for t in a.terms.split(",")]
    print(f"AUTO start: terms={terms} target={a.target} step={a.step} cap={a.max} rhat<={a.rhat} jobs={a.jobs}",
          flush=True)
    for term in terms:
        try:
            process(term, a)
        except Exception:
            print(f"[{term}] ERROR:\n{traceback.format_exc()}", flush=True)
    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
