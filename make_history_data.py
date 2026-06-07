"""
Export per-MP voting history data for the site:
  docs/votes.json     — shared metadata for ALL electronic votes (chronological):
                        date, title, topic, yes/no/abstain counts, contested flag,
                        sitting/voting (for the official PDF link), club majorities.
  docs/mp_votes.json  — {mp_id: "code string"} aligned to votes.json order,
                        code per vote: Y=za, N=przeciw, A=wstrzymał się, .=nieobecny.
                        (Current cache only distinguishes Y/N/.; A appears once the
                        richer 4-state re-fetch is available.)
"""

import json
import os
import numpy as np
from fetch_data import fetch_rollcall

OUT_VOTES = "docs/votes.json"
OUT_MP = "docs/mp_votes.json"


def main():
    data = fetch_rollcall(verbose=False)          # UNFILTERED: full chronological set
    Y, mp_ids, mp_info, vote_meta = data["Y"], data["mp_ids"], data["mp_info"], data["vote_meta"]
    n, m = Y.shape
    mask = ~np.isnan(Y)
    clubs = np.array([mp_info.loc[i, "club"] if i in mp_info.index else "?" for i in mp_ids])
    uclubs = sorted(set(clubs))

    # club majority per vote (Y/N among present members)
    club_rows = {c: np.where(clubs == c)[0] for c in uclubs}

    votes = []
    for j, vm in enumerate(vote_meta):
        col = Y[:, j]; mj = mask[:, j]
        maj = {}                                       # club -> % voting "za" (present members)
        for c, rows in club_rows.items():
            present = rows[mj[rows]]
            if len(present):
                maj[c] = round(100 * float(col[present].mean()))
        yes, no, ab = vm.get("yes", 0), vm.get("no", 0), vm.get("abstain", 0)
        minority = min(yes, no) / max(yes + no, 1)
        votes.append({
            "i": j,
            "d": (vm.get("date", "") or "")[:10],
            "t": vm.get("title", ""),
            "o": vm.get("topic", ""),
            "y": yes, "n": no, "a": ab,
            "s": vm.get("sitting"), "v": vm.get("voting_num"),
            "c": 1 if minority > 0.05 else 0,           # contested
            "m": maj,
        })

    # per-MP code strings aligned to vote order
    mp_votes = {}
    for r, mid in enumerate(mp_ids):
        chars = []
        for j in range(m):
            val = Y[r, j]
            chars.append("Y" if val == 1 else "N" if val == 0 else ".")
        mp_votes[str(mid)] = "".join(chars)

    os.makedirs("docs", exist_ok=True)
    with open(OUT_VOTES, "w", encoding="utf-8") as f:
        json.dump({"n_votes": m, "votes": votes}, f, ensure_ascii=False, separators=(",", ":"))
    with open(OUT_MP, "w", encoding="utf-8") as f:
        json.dump(mp_votes, f, ensure_ascii=False, separators=(",", ":"))

    sz_v = os.path.getsize(OUT_VOTES) / 1024
    sz_m = os.path.getsize(OUT_MP) / 1024
    print(f"Wrote {OUT_VOTES}: {m} votes ({sz_v:.0f} KB)")
    print(f"Wrote {OUT_MP}: {len(mp_votes)} MPs ({sz_m:.0f} KB)")


if __name__ == "__main__":
    main()
