"""
Export the 1D ideal point results to a compact JSON for the static GitHub Pages
site (docs/ideal_points.json).
"""

import json
import datetime
import numpy as np
import pandas as pd

CSV = "results/ideal_points.csv"
OUT = "docs/ideal_points.json"
N_VOTES = 2610  # filtered electronic votes used in the 1D model


def main():
    df = pd.read_csv(CSV).sort_values("x_mean").reset_index(drop=True)

    mps = [{
        "id": int(r.mp_id),
        "name": f"{r.first_name} {r.last_name}".strip(),
        "club": r.club,
        "x": round(float(r.x_mean), 4),
        "sd": round(float(r.x_sd), 4),
        "lo": round(float(r.x_lo90), 4),
        "hi": round(float(r.x_hi90), 4),
        "rhat": round(float(r.rhat), 3),
    } for r in df.itertuples()]

    clubs = (df.groupby("club")["x_mean"].agg(["mean", "count"])
               .sort_values("mean").reset_index())
    club_list = [{"club": c.club, "mean": round(float(c.mean), 3), "n": int(c.count)}
                 for c in clubs.itertuples()]

    out = {
        "meta": {
            "title": "Punkty idealne posłów — Sejm X kadencji",
            "term": "X kadencja (od XI 2023)",
            "n_mps": len(mps),
            "n_votes": N_VOTES,
            "generated": datetime.date.today().isoformat(),
            "method": "Bayesowski model przestrzenny (2-parametrowy IRT, probit), "
                      "estymacja MCMC — sampler Gibbsa z augmentacją Alberta-Chiba",
            "source": "api.sejm.gov.pl (głosowania imienne)",
            "note": "Oś to główny wymiar głosowań (rząd–opozycja, układa się jak "
                    "lewica–prawica). Dodatnie = prawa strona osi. Skala: SD=1.",
        },
        "clubs": club_list,
        "mps": mps,
    }

    import os
    os.makedirs("docs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {OUT}: {len(mps)} MPs, {len(club_list)} clubs")


if __name__ == "__main__":
    main()
