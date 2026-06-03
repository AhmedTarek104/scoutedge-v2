"""
Merge FBref non-Big-5 league data with existing Big 5 players_final.csv.

For non-Big-5 leagues FBref only provides: goals, assists, minutes, shots,
shots_on_target, tackles_won, interceptions, crosses.
Progressive stats, key passes, pass%, dribbles, clearances, blocks
are not available. Scouting scores use whatever features exist and
renormalize weights accordingly.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

ROOT = Path(r"F:\scoutedge_v2")
sys.path.insert(0, str(ROOT / "src"))
from config import (DATA_RAW, DATA_PROC, DATA_FINAL,
                    POSITION_GROUP_MAP, SCOUTING_WEIGHTS,
                    LEAGUE_DIFFICULTY, EXCLUDED_NATIONALITIES,
                    MIN_MINUTES, TARGET_LEAGUES)
from features import compute_efficiency, compute_scouting_scores

TODAY = datetime.now()

# ── Per-90 computation (inline, applied to combined dataset) ──────────────────
PER90_SOURCES = [
    "goals", "assists", "shots", "shots_on_target",
    "tackles_won", "interceptions", "crosses",
    # Big 5 columns (NaN for non-Big-5, will be skipped per-player):
    "xg", "xag", "npxg", "key_passes", "tackles",
    "progressive_carries", "progressive_passes", "progressive_receives",
    "blocks", "clearances", "pressures",
    "dribbles_completed", "dribbles_attempted", "passes_final_third",
]

def compute_per90_combined(df):
    df = df.copy()
    denom = pd.to_numeric(df["minutes"], errors="coerce") / 90.0
    denom = denom.replace(0, np.nan)
    computed = []
    for col in PER90_SOURCES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.var(skipna=True) == 0 or series.isna().all():
            continue
        new_col = f"{col}_p90"
        df[new_col] = (series / denom).round(3)
        computed.append(new_col)
    if "dribbles_completed_p90" in df.columns:
        df["dribbles_p90"] = df["dribbles_completed_p90"]
    print(f"  Per-90 columns computed: {len(computed)}")
    return df


# ── League registry ───────────────────────────────────────────────────────────
LEAGUES = [
    ("TUR", "Super Lig",           "tr"),
    ("NED", "Eredivisie",          "nl"),
    ("POR", "Primeira Liga",       "pt"),
    ("DEN", "Superliga",           "dk"),
    ("BEL", "Belgian Pro League",  "be"),
    ("BRA", "Serie A (Brazil)",    "br"),
    ("ARG", "Primera Division",    "ar"),
    ("SCO", "Scottish Premiership","gb-sct"),
]


def parse_age(s):
    """FBref age '23-095' -> int 23."""
    try:
        return int(str(s).split("-")[0])
    except Exception:
        return np.nan


def build_league_df(code):
    """Load and merge 6 FBref stat files for one league.

    Only extracts columns actually populated for non-Big-5 leagues:
      goals, assists, minutes (standard)
      shots, shots_on_target (shooting)
      tackles_won, interceptions (defense)
      crosses (misc)
    """
    raw = DATA_RAW

    # ── Standard ─────────────────────────────────────────────────────────────
    std = pd.read_csv(raw / f"fbref_{code}_standard.csv",
                      low_memory=False, encoding_errors="replace")
    std = std[std["Player"].astype(str).str.strip() != "Player"].dropna(how="all")
    std = std.rename(columns={
        "Player": "player",
        "Nation": "nationality",
        "Pos":    "position_raw",
        "Squad":  "team",
        "Age":    "age_raw",
        "Playing Time_Min": "minutes",
        "Performance_Gls":  "goals",
        "Performance_Ast":  "assists",
    })
    std["age"]        = std["age_raw"].apply(parse_age)
    std["minutes"]    = pd.to_numeric(std["minutes"],  errors="coerce")
    std["goals"]      = pd.to_numeric(std["goals"],    errors="coerce")
    std["assists"]    = pd.to_numeric(std["assists"],  errors="coerce")
    std["_key"]       = std["player"].astype(str).str.strip().str.lower()
    std = std[["_key", "player", "nationality", "position_raw", "team",
               "age", "minutes", "goals", "assists"]].copy()

    # ── Shooting ─────────────────────────────────────────────────────────────
    sht_path = raw / f"fbref_{code}_shooting.csv"
    if sht_path.exists():
        sht = pd.read_csv(sht_path, low_memory=False, encoding_errors="replace")
        sht = sht[sht["Player"].astype(str).str.strip() != "Player"].dropna(how="all")
        sht["_key"]            = sht["Player"].astype(str).str.strip().str.lower()
        sht["shots"]           = pd.to_numeric(sht.get("Standard_Sh"), errors="coerce")
        sht["shots_on_target"] = pd.to_numeric(sht.get("Standard_SoT"), errors="coerce")
        sht = sht[["_key", "shots", "shots_on_target"]].copy()
        std = std.merge(sht, on="_key", how="left")

    # ── Defense ──────────────────────────────────────────────────────────────
    def_path = raw / f"fbref_{code}_defense.csv"
    if def_path.exists():
        d = pd.read_csv(def_path, low_memory=False, encoding_errors="replace")
        d = d[d["Player"].astype(str).str.strip() != "Player"].dropna(how="all")
        d["_key"]         = d["Player"].astype(str).str.strip().str.lower()
        d["tackles_won"]  = pd.to_numeric(d.get("Tackles_TklW"), errors="coerce")
        d["interceptions"]= pd.to_numeric(d.get("Int"),          errors="coerce")
        d = d[["_key", "tackles_won", "interceptions"]].copy()
        std = std.merge(d, on="_key", how="left")

    # ── Misc ─────────────────────────────────────────────────────────────────
    mis_path = raw / f"fbref_{code}_misc.csv"
    if mis_path.exists():
        m = pd.read_csv(mis_path, low_memory=False, encoding_errors="replace")
        m = m[m["Player"].astype(str).str.strip() != "Player"].dropna(how="all")
        m["_key"]    = m["Player"].astype(str).str.strip().str.lower()
        m["crosses"] = pd.to_numeric(m.get("Performance_Crs"), errors="coerce")
        m = m[["_key", "crosses"]].copy()
        std = std.merge(m, on="_key", how="left")

    std = std.drop(columns=["_key"], errors="ignore")
    std = std.drop_duplicates(subset=["player", "team"])

    n = len(std)
    populated = std.notna().sum()
    print(f"    {n} players — shots={populated.get('shots',0)} "
          f"tkl_won={populated.get('tackles_won',0)} "
          f"int={populated.get('interceptions',0)}", flush=True)
    return std


# ── Step 1: Build all new league dataframes ───────────────────────────────────
print("=" * 60)
print("STEP 1 - Loading and merging new league files")
print("=" * 60)

new_frames = []
for code, league_name, _ in LEAGUES:
    print(f"\n  {code} - {league_name}")
    df = build_league_df(code)
    df["league_clean"]      = league_name
    df["league"]            = league_name
    df["league_difficulty"] = LEAGUE_DIFFICULTY.get(league_name, 0.75)
    new_frames.append(df)

new_all = pd.concat(new_frames, ignore_index=True)
print(f"\nAll new leagues: {len(new_all)} rows")


# ── Step 3: Filters ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 - Filtering")
print("=" * 60)

n0 = len(new_all)
new_all["position_raw"] = new_all["position_raw"].astype(str).str.strip()
new_all = new_all[~new_all["position_raw"].str.upper().str.startswith("GK")]
new_all = new_all[new_all["minutes"].fillna(0) >= MIN_MINUTES]

if "nationality" in new_all.columns:
    excl = [e.lower() for e in EXCLUDED_NATIONALITIES]
    nat  = new_all["nationality"].fillna("").astype(str).str.lower()
    new_all = new_all[~nat.str.contains("|".join(excl), na=False)]

print(f"Before: {n0}  ->  After: {len(new_all)}")


# ── Step 3b: Position group mapping ──────────────────────────────────────────
def map_position(raw):
    raw = str(raw).strip()
    if raw in POSITION_GROUP_MAP:
        return POSITION_GROUP_MAP[raw]
    first = raw.split(",")[0].strip()
    return POSITION_GROUP_MAP.get(first, "CM")

new_all["position_group"] = new_all["position_raw"].apply(map_position)
new_all = new_all[new_all["position_group"] != "GK"]
print("Position groups:")
print(new_all["position_group"].value_counts().to_string())


# ── Step 5: Market values from Transfermarkt ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 - Adding market values")
print("=" * 60)

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("  rapidfuzz not found - using exact match only")

tm = pd.read_csv(DATA_RAW / "players.csv", low_memory=False, encoding_errors="replace")
tm["market_value_m"]  = pd.to_numeric(tm["market_value_in_eur"], errors="coerce") / 1_000_000
tm["contract_expiry"] = tm["contract_expiration_date"]
tm_lookup = (tm[["name", "market_value_m", "contract_expiry"]]
               .dropna(subset=["name"])
               .drop_duplicates(subset=["name"], keep="last"))
tm_names = tm_lookup["name"].tolist()
tm_lower = {n.lower(): n for n in tm_names}


def get_mv(player_name):
    key = str(player_name).lower().strip()
    # Exact lowercase match
    if key in tm_lower:
        r = tm_lookup[tm_lookup["name"] == tm_lower[key]].iloc[0]
        return r["market_value_m"], r["contract_expiry"]
    # Fuzzy
    if HAS_RAPIDFUZZ:
        result = process.extractOne(player_name, tm_names,
                                    scorer=fuzz.WRatio, score_cutoff=82)
        if result:
            r = tm_lookup[tm_lookup["name"] == result[0]].iloc[0]
            return r["market_value_m"], r["contract_expiry"]
    return np.nan, np.nan


print("  Matching market values...", flush=True)
mv_data = new_all["player"].apply(get_mv)
new_all["market_value_m"]  = [x[0] for x in mv_data]
new_all["contract_expiry"] = [x[1] for x in mv_data]

cutoff = TODAY + timedelta(days=365)
expiry_dt = pd.to_datetime(new_all["contract_expiry"], errors="coerce")
new_all["contract_expiring"] = expiry_dt.notna() & (expiry_dt <= cutoff)

n_mv = new_all["market_value_m"].notna().sum()
print(f"  Matched {n_mv}/{len(new_all)} ({100*n_mv/len(new_all):.1f}%)")


# ── Step 6: Combine with existing Big 5 data ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 - Combining with existing Big 5 data")
print("=" * 60)

big5 = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
print(f"  Big 5: {len(big5)} players")
print(f"  New leagues: {len(new_all)} players")

# All columns from both datasets
all_cols = list(dict.fromkeys(list(big5.columns) + list(new_all.columns)))
big5_aligned = big5.reindex(columns=all_cols)
new_aligned  = new_all.reindex(columns=all_cols)

combined = pd.concat([big5_aligned, new_aligned], ignore_index=True)
print(f"  Combined (before dedup): {len(combined)}")

# Prefer Big 5 if same player appears in both
BIG5_LEAGUES = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}
combined["_sort"] = combined["league_clean"].apply(lambda l: 0 if l in BIG5_LEAGUES else 1)
combined = combined.sort_values("_sort")
combined = combined.drop_duplicates(subset=["player", "team"], keep="first")
combined = combined.drop(columns=["_sort"])
print(f"  Combined (after dedup): {len(combined)}")


# ── Step 7: Recompute per-90 and scouting scores ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 - Recomputing per-90 and scouting scores")
print("=" * 60)

# Compute per-90 on the FULL combined dataset so new leagues get all available
# per-90 stats from their raw columns (goals, assists, shots, etc.)
print("  Computing per-90 metrics...")
combined = compute_per90_combined(combined)

# Efficiency metrics
print("  Computing efficiency metrics...")
combined = compute_efficiency(combined)

# Scouting scores — percentile ranks computed across ALL leagues per position
print("  Computing scouting scores (all leagues combined)...")
for col in ["tackle_success_rate", "interceptions_p90", "aerial_duels_won_pct",
            "progressive_passes_p90", "clearances_p90", "progressive_carries_p90",
            "key_passes_p90", "tackles_p90", "assists_p90", "pass_completion_rate",
            "pressures_p90", "xag_p90", "xg_p90", "npxg_p90", "dribbles_p90",
            "goals_p90", "progressive_receives_p90", "shot_accuracy",
            "tackles_won_p90"]:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

combined = compute_scouting_scores(combined)


# ── Step 8: Save ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 - Saving")
print("=" * 60)

out = DATA_FINAL / "players_final.csv"
combined.to_csv(out, index=False)
print(f"  Saved: {out}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

total = len(combined)
print(f"\nTotal players: {total}")

print("\nPlayers per league:")
lc = combined["league_clean"].value_counts()
for league, count in lc.items():
    print(f"  {league:35s} {count:>5}")

n_mv = combined["market_value_m"].notna().sum()
print(f"\nPlayers with market values: {n_mv}/{total} ({100*n_mv/total:.1f}%)")

per90_cols = [c for c in combined.columns if c.endswith("_p90")]
nonzero = [c for c in per90_cols
           if pd.to_numeric(combined[c], errors="coerce").var(skipna=True) > 0]
print(f"\nNon-zero-variance per-90 columns: {len(nonzero)}/{len(per90_cols)}")
for c in nonzero:
    big5_n = combined[combined["league_clean"].isin(BIG5_LEAGUES)][c].notna().sum()
    new_n  = combined[~combined["league_clean"].isin(BIG5_LEAGUES)][c].notna().sum()
    print(f"  {c:40s} Big5={big5_n:4d}  NonBig5={new_n:4d}")

print("\nPosition distribution:")
print(combined[combined["position_group"].notna()]["position_group"]
      .value_counts().to_string())

present = set(combined["league_clean"].unique())
missing = [l for l in TARGET_LEAGUES if l not in present]
print(f"\nAll 13 target leagues: {'YES' if not missing else 'NO - missing: ' + str(missing)}")
