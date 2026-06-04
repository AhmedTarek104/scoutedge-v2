"""
Re-run market value join for players_final.csv using accent-normalized names.

Targets only players with null market_value_m — does not overwrite existing values.
Uses unicodedata to strip diacritics before fuzzy matching so names like
Vlahović / Álvarez / Díaz match their Transfermarkt equivalents.
"""
import sys
import unicodedata
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_FINAL, DATA_RAW

TODAY = datetime.now()


def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
    print("rapidfuzz available — will use fuzzy matching")
except ImportError:
    HAS_RAPIDFUZZ = False
    print("rapidfuzz not found — falling back to exact normalized match only")


# ── Load Transfermarkt data ───────────────────────────────────────────────────
tm = pd.read_csv(DATA_RAW / "players.csv", low_memory=False, encoding_errors="replace")
tm["market_value_m"] = pd.to_numeric(tm["market_value_in_eur"], errors="coerce") / 1_000_000
tm["contract_expiry"] = tm["contract_expiration_date"]
tm_lookup = (
    tm[["name", "market_value_m", "contract_expiry"]]
    .dropna(subset=["name"])
    .sort_values("market_value_m", ascending=False, na_position="last")
    .drop_duplicates(subset=["name"], keep="first")
)

tm_names = tm_lookup["name"].tolist()
tm_norm  = [normalize(n) for n in tm_names]           # parallel list
tm_norm_to_orig = {normalize(n): n for n in tm_names} # deduplicated lookup

print(f"Transfermarkt: {len(tm_names)} players loaded")


def get_mv(player_name: str):
    key = normalize(player_name)

    # 1. Exact normalized match
    if key in tm_norm_to_orig:
        orig = tm_norm_to_orig[key]
        row = tm_lookup[tm_lookup["name"] == orig].iloc[0]
        return row["market_value_m"], row["contract_expiry"], "exact"

    # 2. Fuzzy on normalized names
    if HAS_RAPIDFUZZ:
        result = process.extractOne(key, tm_norm, scorer=fuzz.WRatio, score_cutoff=82)
        if result:
            orig = tm_names[result[2]]
            row = tm_lookup[tm_lookup["name"] == orig].iloc[0]
            return row["market_value_m"], row["contract_expiry"], f"fuzzy({result[1]:.0f})"

    return np.nan, np.nan, None


# ── Load players_final.csv ────────────────────────────────────────────────────
df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
df["market_value_m"] = pd.to_numeric(df["market_value_m"], errors="coerce")

null_mask = df["market_value_m"].isna()
n_null_before = int(null_mask.sum())
print(f"\nPlayers with null MV before fix: {n_null_before}/{len(df)}")

# ── Re-match only null rows ───────────────────────────────────────────────────
filled = 0
examples = []
cutoff = TODAY + timedelta(days=365)

for idx in df[null_mask].index:
    player_name = df.at[idx, "player"]
    mv, expiry, method = get_mv(player_name)
    if pd.notna(mv):
        df.at[idx, "market_value_m"] = mv
        if pd.isna(df.at[idx, "contract_expiry"]) and pd.notna(expiry):
            df.at[idx, "contract_expiry"] = expiry
            expiry_dt = pd.to_datetime(expiry, errors="coerce")
            if pd.notna(expiry_dt):
                df.at[idx, "contract_expiring"] = bool(expiry_dt <= cutoff)
        filled += 1
        if len(examples) < 10:
            examples.append((player_name, mv, method))

print(f"Newly filled market values: {filled}")
if examples:
    print("\nSample matches:")
    for name, mv, method in examples:
        safe = name.encode("ascii", "replace").decode()
        print(f"  {safe:40s} €{mv:.2f}m  [{method}]")

n_null_after = int(df["market_value_m"].isna().sum())
print(f"\nNull MV after fix: {n_null_after}/{len(df)}")
print(f"Improvement: {n_null_before - n_null_after} players gained a market value")

# ── Save ──────────────────────────────────────────────────────────────────────
out = DATA_FINAL / "players_final.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
