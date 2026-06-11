"""
Export the 1D ideal point results to a compact JSON for the static GitHub Pages
site (docs/ideal_points.json).

Enriches each MP with stats computed from the roll-call matrix:
  - votes    : number of contested votes cast (turnout numerator)
  - turnout  : fraction of contested votes cast
  - loyalty  : fraction of cast votes agreeing with the MP's club majority
  - rank     : 1-based position on the main axis (1 = one pole)
  - club_rank: 1-based position within own club
"""

import argparse
import json
import datetime
import os
import numpy as np
import pandas as pd

from fetch_data import fetch_rollcall, filter_rollcall

TERM_INFO = {
    "term10": {"title": "Punkty idealne posłów — Sejm X kadencji",
               "term": "X kadencja (od XI 2023)"},
    "term9":  {"title": "Punkty idealne posłów — Sejm IX kadencji",
               "term": "IX kadencja (2019–2023)"},
    "term8":  {"title": "Punkty idealne posłów — Sejm VIII kadencji",
               "term": "VIII kadencja (2015–2019)"},
    "term7":  {"title": "Punkty idealne posłów — Sejm VII kadencji",
               "term": "VII kadencja (2011–2015)"},
    "term6":  {"title": "Punkty idealne posłów — Sejm VI kadencji",
               "term": "VI kadencja (2007–2011)"},
    "term5":  {"title": "Punkty idealne posłów — Sejm V kadencji",
               "term": "V kadencja (2005–2007)"},
}


def compute_behaviour(df, term="term10"):
    """Returns dicts mp_id -> votes, turnout, loyalty using the filtered matrix."""
    data = filter_rollcall(fetch_rollcall(term=term, verbose=False))
    Y, mp_ids, mp_info = data["Y"], data["mp_ids"], data["mp_info"]
    n, m = Y.shape
    mask = ~np.isnan(Y)
    clubs = np.array([mp_info.loc[i, "club"] if i in mp_info.index else "?" for i in mp_ids])

    votes = mask.sum(axis=1)
    turnout = votes / m

    # club majority per vote, then per-MP agreement on cast votes
    loyalty = np.full(n, np.nan)
    for c in np.unique(clubs):
        members = np.where(clubs == c)[0]
        sub = Y[members]                     # (k, m)
        submask = mask[members]
        with np.errstate(invalid="ignore"):
            club_mean = np.nansum(sub, axis=0) / np.maximum(submask.sum(axis=0), 1)
        majority = (club_mean > 0.5).astype(float)   # (m,)
        for idx in members:
            mj = mask[idx]
            if mj.sum() == 0:
                continue
            agree = (Y[idx][mj] == majority[mj]).mean()
            loyalty[idx] = agree

    return {
        int(mp_ids[i]): {
            "votes": int(votes[i]),
            "turnout": round(float(turnout[i]), 3),
            "loyalty": None if np.isnan(loyalty[i]) else round(float(loyalty[i]), 3),
        } for i in range(n)
    }, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--term", default="term10", help="term10 or term9")
    args = ap.parse_args()
    term = args.term
    tag = "" if term == "term10" else f"_{term}"
    csv_path = f"results/ideal_points{tag}.csv"
    out_path = f"docs/ideal_points{tag}.json"
    info = TERM_INFO.get(term, {"title": f"Punkty idealne — {term}", "term": term})

    df = pd.read_csv(csv_path).sort_values("x_mean").reset_index(drop=True)
    df["club"] = df["club"].fillna("niez.")   # MPs with no club label -> unaffiliated
    behaviour, n_votes = compute_behaviour(df, term)

    df["rank"] = np.arange(1, len(df) + 1)
    df["club_rank"] = df.groupby("club")["x_mean"].rank(method="first").astype(int)
    club_sizes = df["club"].value_counts().to_dict()

    mps = []
    for r in df.itertuples():
        b = behaviour.get(int(r.mp_id), {})
        mps.append({
            "id": int(r.mp_id),
            "name": f"{r.first_name} {r.last_name}".strip(),
            "club": r.club,
            "x": round(float(r.x_mean), 4),
            "sd": round(float(r.x_sd), 4),
            "lo": round(float(r.x_lo90), 4),
            "hi": round(float(r.x_hi90), 4),
            "rhat": round(float(r.rhat), 3),
            "rank": int(r.rank),
            "club_rank": int(r.club_rank),
            "club_size": int(club_sizes[r.club]),
            "votes": b.get("votes"),
            "turnout": b.get("turnout"),
            "loyalty": b.get("loyalty"),
        })

    clubs = (df.groupby("club")["x_mean"].agg(["mean", "count"])
               .sort_values("mean").reset_index())
    club_list = [{"club": c.club, "mean": round(float(c.mean), 3), "n": int(c.count)}
                 for c in clubs.itertuples()]

    out = {
        "meta": {
            "title": info["title"],
            "term": info["term"],
            "n_mps": len(mps),
            "n_votes": n_votes,
            "generated": datetime.date.today().isoformat(),
            "method": "Bayesowski model przestrzenny (2-parametrowy IRT, probit), "
                      "estymacja MCMC — sampler Gibbsa z augmentacją Alberta-Chiba",
            "source": "api.sejm.gov.pl (głosowania imienne)",
            "note": "Główna oś podziału odtworzona z głosowań imiennych (bez etykiet "
                    "ideologicznych). Może odpowiadać podziałowi rząd–opozycja. "
                    "Znak (±) jest umowny (zakotwiczony). Skala: SD=1.",
        },
        "clubs": club_list,
        "mps": mps,
    }

    os.makedirs("docs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {out_path}: {len(mps)} MPs, {len(club_list)} clubs, {n_votes} votes")


if __name__ == "__main__":
    main()
