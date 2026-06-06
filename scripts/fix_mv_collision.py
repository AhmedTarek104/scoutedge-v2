"""
fix_mv_collision.py

Fix name-collision market values using the fbref_tm_mapping.csv direct
FBref-name -> TM player_id mapping.

Root cause: the previous join normalized names (stripping accents) before
building the lookup dict, so "Savio" (Rio Ave, 200k) overwrote "Savinho"
(Man City, 45m) when both reduced to "savio".

Fix:
  1. Use EXACT FBref player name (no normalization) to look up TM player_id
     from fbref_tm_mapping.csv -- this correctly maps accented "Savinho"
     vs plain "Savio" as distinct entries.
  2. When multiple TM IDs share the same exact FBref name, prefer the one
     whose TM club matches the player's team (fuzzy), then fall back to
     highest latest valuation.
  3. Overwrite existing MV values when the ID-based lookup gives a
     meaningfully different value (corrects bad prior matches).

After running, verifies Savinho / Joao Pedro / Vlahovic.
"""

import sys, re, unicodedata
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_FINAL, DATA_RAW

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

def normalize_club(club: str) -> str:
    c = normalize(club)
    for suffix in [" football club", " fc", " sc", " cf", " ac", " fk",
                   " f.c.", " s.c.", " a.c.", " f.a.", " s.a.d."]:
        if c.endswith(suffix):
            c = c[: -len(suffix)].strip()
    return c

# ── Load sources ─────────────────────────────────────────────────────────────

print("Loading sources...")
df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
df["market_value_m"] = pd.to_numeric(df["market_value_m"], errors="coerce")

# TM players snapshot (club names)
tm_players = pd.read_csv(DATA_RAW / "players.csv", low_memory=False,
                         encoding_errors="replace")
tm_club = dict(zip(tm_players["player_id"].astype(float),
                   tm_players["current_club_name"].fillna("").astype(str)))

# Most-recent valuation per player_id
pv = pd.read_csv(DATA_RAW / "player_valuations.csv", low_memory=False)
pv["market_value_m"] = pd.to_numeric(pv["market_value_in_eur"], errors="coerce") / 1e6
pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
pv_latest = (
    pv.sort_values("date", ascending=False)
    .groupby("player_id")
    .first()
    .reset_index()[["player_id", "market_value_m"]]
)
pv_by_id = dict(zip(pv_latest["player_id"].astype(float),
                    pv_latest["market_value_m"]))

# FBref -> TM mapping (EXACT FBref name preserved)
tm_map = pd.read_csv(DATA_RAW / "fbref_tm_mapping.csv", low_memory=False,
                     encoding_errors="replace")
tm_map["player_id"] = (
    tm_map["UrlTmarkt"]
    .str.extract(r"/spieler/(\d+)")
    .squeeze()
    .pipe(pd.to_numeric, errors="coerce")
)
tm_map["latest_mv"] = tm_map["player_id"].map(pv_by_id)
tm_map["tm_club"]   = tm_map["player_id"].map(tm_club)

print(f"  players_final: {len(df)} rows")
print(f"  fbref_tm_mapping: {len(tm_map)} rows")

# ── Build {exact_fbref_name -> list[(player_id, latest_mv, tm_club)]} ────────

from collections import defaultdict
name_to_candidates = defaultdict(list)
for _, row in tm_map.iterrows():
    fbref_name = str(row["PlayerFBref"])
    pid  = row["player_id"]
    mv   = row["latest_mv"]
    club = str(row["tm_club"])
    if pd.notna(pid) and pd.notna(mv) and mv > 0:
        name_to_candidates[fbref_name].append((pid, mv, club))

# ── For each player, resolve the best candidate ───────────────────────────────

def best_candidate(fbref_name: str, fbref_team: str):
    """Return the best (player_id, market_value_m) using team+name disambiguation."""
    candidates = name_to_candidates.get(fbref_name, [])
    if not candidates:
        return None, None

    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1]

    # Multiple candidates: try team matching first
    if fbref_team and HAS_RAPIDFUZZ:
        norm_team = normalize_club(fbref_team)
        scored = []
        for pid, mv, club in candidates:
            sim = fuzz.ratio(norm_team, normalize_club(club))
            scored.append((sim, mv, pid))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        best_sim = scored[0][0]
        if best_sim >= 60:
            return scored[0][2], scored[0][1]

    # Fallback: highest market value wins
    candidates_sorted = sorted(candidates, key=lambda x: -x[1])
    return candidates_sorted[0][0], candidates_sorted[0][1]

# ── Apply corrections ─────────────────────────────────────────────────────────

corrected = 0
for idx, row in df.iterrows():
    fbref_name = row["player"]
    fbref_team = str(row.get("team", ""))
    existing_mv = row["market_value_m"]

    pid, new_mv = best_candidate(fbref_name, fbref_team)
    if pid is None or not pd.notna(new_mv) or new_mv <= 0:
        continue

    # Overwrite if:
    #  - existing value is null, OR
    #  - new value is >3x larger (likely a name collision error)
    if pd.isna(existing_mv) or (existing_mv > 0 and new_mv / existing_mv > 3):
        df.at[idx, "market_value_m"] = new_mv
        corrected += 1

print(f"\nCorrected {corrected} market values")

# ── Verify specific players ───────────────────────────────────────────────────

print("\nVerification:")
checks = {
    "Savio":      ("Manchester City", 40.0),
    "João Pedro": ("Brighton",   30.0),
    "Vlahovic":   ("Juventus",        30.0),
}
for kw, (expected_club, min_mv) in checks.items():
    mask = df["player"].str.contains(kw, case=False, na=False)
    sub  = df[mask][["player", "team", "league_clean", "market_value_m"]]
    for _, r in sub.iterrows():
        p  = r["player"].encode("ascii", "replace").decode()
        mv = r["market_value_m"]
        ok = "OK" if (pd.notna(mv) and mv >= min_mv) else "STILL WRONG"
        print(f"  [{ok}] {p:38s} | {str(r['team']):22s} | MV={mv}")

# ── Save ─────────────────────────────────────────────────────────────────────

out = DATA_FINAL / "players_final.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
print(f"Market value coverage: {df['market_value_m'].notna().sum()}/{len(df)}")
