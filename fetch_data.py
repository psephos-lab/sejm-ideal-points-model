"""
Fetch roll-call voting data from the official Sejm API and cache to disk.

Vote encoding in the matrix:
  1   = YES
  0   = NO
  NaN = ABSENT or ABSTAIN (treated as missing-at-random)

Only ELECTRONIC votes are included; ON_LIST (list votes) are skipped.
"""

import os
import pickle
import time
import requests
import numpy as np
import pandas as pd

API_BASE = "https://api.sejm.gov.pl/sejm"
CACHE_DIR = "data"
CACHE_FILE = os.path.join(CACHE_DIR, "term10_rollcall_v2.pkl")  # v2: correct enumeration


def _get(url: str, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _get_or_none(url: str):
    """GET that returns None on 404 (no retry) — for probing voting numbers."""
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        return None
    except Exception:
        # one transient retry
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            return None


def fetch_mps(term: str = "term10") -> pd.DataFrame:
    data = _get(f"{API_BASE}/{term}/MP")
    rows = []
    for mp in data:
        rows.append({
            "mp_id": mp["id"],
            "first_name": mp.get("firstLastName", "").split()[0] if mp.get("firstLastName") else "",
            "last_name": mp.get("lastFirstName", "").split()[0] if mp.get("lastFirstName") else "",
            "club": mp.get("club", ""),
        })
    return pd.DataFrame(rows).set_index("mp_id")


def fetch_rollcall(term: str = "term10", verbose: bool = True) -> dict:
    """
    Returns dict with keys:
      Y       — np.ndarray float32 (n_mps x n_votes), 1/0/NaN
      mp_ids  — list of MP ids (row order)
      vote_ids — list of (sitting, voting_num) tuples (column order)
      mp_info  — DataFrame with club/name per MP
      vote_meta — list of dicts with title/description per vote
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        if verbose:
            print(f"Loading cached data from {CACHE_FILE}")
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)

    if verbose:
        print("Fetching MP list...")
    mp_info = fetch_mps(term)
    mp_ids = sorted(mp_info.index.tolist())
    mp_index = {mid: i for i, mid in enumerate(mp_ids)}

    if verbose:
        print("Fetching sitting list...")
    sittings = _get(f"{API_BASE}/{term}/votings")
    proceedings = sorted({s["proceeding"] for s in sittings})
    if verbose:
        print(f"  {len(proceedings)} proceedings (from {len(sittings)} sitting-days)")

    # Enumerate each proceeding by probing voting numbers until a run of 404s.
    # The per-day `votingsNum` field is unreliable: multi-day proceedings repeat in
    # the list and undercount the per-proceeding voting numbering — which previously
    # caused BOTH duplicated and missing votes. Probing (p, 1..N) is authoritative.
    MAX_GAP, HARD_CAP = 10, 400
    CODE = {"YES": 1, "NO": 0, "ABSTAIN": 2}        # else -> 3 (absent / no record)

    vote_ids, vote_meta, columns, vcolumns = [], [], [], []
    done = 0
    for p in proceedings:
        consec, v_num = 0, 0
        while consec < MAX_GAP and v_num < HARD_CAP:
            v_num += 1
            detail = _get_or_none(f"{API_BASE}/{term}/votings/{p}/{v_num}")
            if detail is None:
                consec += 1
                continue
            consec = 0
            if detail.get("kind") != "ELECTRONIC":
                continue

            col = np.full(len(mp_ids), np.nan, dtype=np.float32)
            vcol = np.full(len(mp_ids), 3, dtype=np.int8)      # 3 = absent / no record
            for v in detail.get("votes", []):
                idx = mp_index.get(v["MP"])
                if idx is None:
                    continue
                vs = v.get("vote", "")
                vcol[idx] = CODE.get(vs, 3)
                if vs == "YES":
                    col[idx] = 1.0
                elif vs == "NO":
                    col[idx] = 0.0
                # ABSTAIN / ABSENT -> NaN in Y (model), but recorded in V (history)

            vote_ids.append((p, v_num))
            vote_meta.append({
                "sitting": p,
                "voting_num": v_num,
                "date": detail.get("date", ""),
                "title": detail.get("title", ""),
                "topic": detail.get("topic", ""),
                "yes": detail.get("yes", 0),
                "no": detail.get("no", 0),
                "abstain": detail.get("abstain", 0),
            })
            columns.append(col)
            vcolumns.append(vcol)

            done += 1
            if verbose and done % 200 == 0:
                print(f"  fetched {done} votes (proceeding {p})", flush=True)

    Y = np.stack(columns, axis=1) if columns else np.empty((len(mp_ids), 0), dtype=np.float32)
    V = np.stack(vcolumns, axis=1) if vcolumns else np.empty((len(mp_ids), 0), dtype=np.int8)

    result = {
        "Y": Y,            # float32: 1=YES, 0=NO, NaN=abstain/absent (model input)
        "V": V,            # int8: 1=YES, 0=NO, 2=ABSTAIN, 3=absent/none (history)
        "mp_ids": mp_ids,
        "vote_ids": vote_ids,
        "mp_info": mp_info,
        "vote_meta": vote_meta,
    }

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(result, f)
    if verbose:
        print(f"Saved to {CACHE_FILE}. Matrix shape: {Y.shape}")

    return result


def filter_rollcall(data: dict, unanimity_threshold: float = 0.95) -> dict:
    """
    Remove near-unanimous votes (less than 5% minority among observed votes).
    Returns a new dict with filtered Y and vote_ids/vote_meta.
    """
    Y = data["Y"]
    vote_meta = data["vote_meta"]
    vote_ids = data["vote_ids"]

    keep = []
    for j in range(Y.shape[1]):
        col = Y[:, j]
        observed = col[~np.isnan(col)]
        if len(observed) == 0:
            continue
        p_yes = observed.mean()
        if unanimity_threshold > p_yes > (1 - unanimity_threshold):
            keep.append(j)

    Y_filtered = Y[:, keep]
    return {
        **data,
        "Y": Y_filtered,
        "vote_ids": [vote_ids[j] for j in keep],
        "vote_meta": [vote_meta[j] for j in keep],
    }


if __name__ == "__main__":
    data = fetch_rollcall(verbose=True)
    filtered = filter_rollcall(data)
    Y = filtered["Y"]
    obs_rate = (~np.isnan(Y)).mean()
    print(f"\nFiltered matrix: {Y.shape[0]} MPs x {Y.shape[1]} votes")
    print(f"Observation rate: {obs_rate:.1%}")
    print(f"Mean YES rate: {np.nanmean(Y):.2%}")

    clubs = filtered["mp_info"]["club"].value_counts()
    print("\nClubs:")
    print(clubs.to_string())
