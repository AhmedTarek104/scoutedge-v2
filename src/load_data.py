import sys, re
from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_RAW, DATA_PROC, POSITION_GROUP_MAP, LEAGUE_DIFFICULTY,
                    EXCLUDED_NATIONALITIES, MIN_MINUTES, MIN_AGE, MAX_AGE)

RENAME_MAP = {
    "Player":               "player",
    "Squad":                "team",
    "Comp":                 "league",
    "Pos":                  "position_raw",
    "Age":                  "age",
    "Born":                 "birth_year",
    "Nation":               "nationality",
    "Min":                  "minutes",
    "90s":                  "minutes_90s",
    "Gls":                  "goals",
    "Ast":                  "assists",
    "xG":                   "xg",
    "npxG":                 "npxg",
    "xAG":                  "xag",
    "Sh":                   "shots",
    "SoT":                  "shots_on_target",
    "PrgC":                 "progressive_carries",
    "PrgP":                 "progressive_passes",
    "PrgR":                 "progressive_receives",
    "KP":                   "key_passes",
    "Tkl":                  "tackles",
    "TklW":                 "tackles_won",
    "Int":                  "interceptions",
    "Blocks":               "blocks",
    "Clr":                  "clearances",
    "Crs":                  "crosses",
    "Succ":                 "dribbles_completed",
    "Att_stats_possession": "dribbles_attempted",
    "Won%":                 "aerial_duels_won_pct",
    "Cmp%":                 "pass_completion_rate",
    "1/3":                  "passes_final_third",
}


def load_raw():
    path = DATA_RAW / "players_data-2024_2025.csv"
    df = pd.read_csv(path, low_memory=False)
    df = df[df["Rk"] != "Rk"].reset_index(drop=True)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def rename_columns(df):
    df = df.rename(columns=RENAME_MAP)
    expected = ["player","team","league","position_raw","age","minutes",
                "minutes_90s","goals","assists","xg","xag","npxg","shots",
                "shots_on_target","progressive_carries","progressive_passes",
                "progressive_receives","key_passes","tackles","tackles_won",
                "interceptions","blocks","clearances","crosses",
                "dribbles_completed","dribbles_attempted","aerial_duels_won_pct",
                "pass_completion_rate","passes_final_third","birth_year","nationality"]
    present = [c for c in expected if c in df.columns]
    print(f"\nStandard columns present ({len(present)}): {present}")
    return df


def map_positions(df):
    df["position_group"] = df["position_raw"].map(POSITION_GROUP_MAP)
    mask = df["position_group"].isna()
    if mask.any():
        df.loc[mask, "position_group"] = (
            df.loc[mask, "position_raw"].fillna("").astype(str)
              .str.split(",").str[0].map(POSITION_GROUP_MAP)
        )
    df["position_group"] = df["position_group"].fillna("UNK")
    print("\nPosition group distribution:")
    print(df["position_group"].value_counts())
    return df


def apply_filters(df):
    for col in ["age", "minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    n0 = len(df)
    df = df[df["position_group"] != "GK"]
    print(f"\nAfter removing GK: {len(df)} (removed {n0-len(df)})")

    n0 = len(df)
    df = df[df["minutes"] >= MIN_MINUTES]
    print(f"After minutes >= {MIN_MINUTES}: {len(df)} (removed {n0-len(df)})")

    n0 = len(df)
    df = df[(df["age"] >= MIN_AGE) & (df["age"] <= MAX_AGE)]
    print(f"After age {MIN_AGE}-{MAX_AGE}: {len(df)} (removed {n0-len(df)})")

    if "nationality" in df.columns:
        n0 = len(df)
        excl_lower = [e.lower() for e in EXCLUDED_NATIONALITIES]
        nat_str = df["nationality"].fillna("").astype(str).str.lower()
        excl_mask = nat_str.str.contains("|".join(excl_lower), na=False)
        df = df[~excl_mask]
        print(f"After removing Israeli players: {len(df)} (removed {n0-len(df)})")
    return df


def add_league_difficulty(df):
    def clean_league(raw):
        if pd.isna(raw):
            return raw
        raw = str(raw)
        raw = re.sub(r"^[a-z]{2,3}\s+", "", raw)
        return raw.strip()

    df["league_clean"] = df["league"].apply(clean_league)
    df["league_difficulty"] = df["league_clean"].map(LEAGUE_DIFFICULTY)

    matched   = df["league_difficulty"].notna().sum()
    unmatched = df["league_difficulty"].isna().sum()
    df["league_difficulty"] = df["league_difficulty"].fillna(0.75)

    print(f"\nLeague difficulty: matched={matched}, unmatched(default 0.75)={unmatched}")
    unmatched_leagues = df.loc[df["league_clean"].map(LEAGUE_DIFFICULTY).isna(), "league_clean"].value_counts()
    if len(unmatched_leagues):
        print("  Unmatched leagues:", unmatched_leagues.to_dict())
    matched_leagues = df.loc[df["league_clean"].map(LEAGUE_DIFFICULTY).notna(), "league_clean"].value_counts()
    print("  Matched leagues:", dict(list(matched_leagues.items())[:12]))
    return df


def add_market_values(df):
    players_tm = pd.read_csv(DATA_RAW / "players.csv", low_memory=False,
                              usecols=["player_id", "name", "contract_expiration_date"])

    vals = pd.read_csv(DATA_RAW / "player_valuations.csv", low_memory=False,
                       usecols=["player_id", "date", "market_value_in_eur"])
    vals["date"] = pd.to_datetime(vals["date"], errors="coerce")
    latest_vals = (vals.sort_values("date")
                       .groupby("player_id", as_index=False)
                       .last()[["player_id", "market_value_in_eur"]])
    latest_vals["market_value_m"] = latest_vals["market_value_in_eur"] / 1e6

    tm_data = players_tm.merge(latest_vals, on="player_id", how="left")
    tm_data = tm_data.rename(columns={"contract_expiration_date": "contract_expiry"})

    mapping = pd.read_csv(DATA_RAW / "fbref_tm_mapping.csv", encoding="latin-1", low_memory=False)
    mapping["player_id"] = mapping["UrlTmarkt"].str.extract(r"/spieler/(\d+)")[0].astype("Int64")
    mapping = mapping.dropna(subset=["player_id"])
    mapping["player_id"] = mapping["player_id"].astype(int)

    mapping_with_vals = mapping.merge(tm_data, on="player_id", how="left")
    mapping_with_vals["player_lower"] = mapping_with_vals["PlayerFBref"].str.lower().str.strip()

    df["player_lower"] = df["player"].str.lower().str.strip()

    mv_lookup = (mapping_with_vals[["player_lower", "market_value_m", "contract_expiry"]]
                 .sort_values("market_value_m", ascending=False)
                 .drop_duplicates("player_lower"))

    df_merged = df.merge(mv_lookup, on="player_lower", how="left")

    matched_primary = df_merged["market_value_m"].notna().sum()
    print(f"\nMarket value primary (exact via mapping): {matched_primary}/{len(df_merged)}")

    unmatched_mask = df_merged["market_value_m"].isna()
    n_unmatched = unmatched_mask.sum()

    if n_unmatched > 0:
        fallback_pool = mapping_with_vals.dropna(subset=["market_value_m"]).drop_duplicates("player_lower")
        pool_names = fallback_pool["player_lower"].tolist()
        pool_df    = fallback_pool.set_index("player_lower")[["market_value_m", "contract_expiry"]]

        def fuzzy_lookup(name):
            result = process.extractOne(name, pool_names,
                                        scorer=fuzz.token_sort_ratio, score_cutoff=85)
            if result:
                row = pool_df.loc[result[0]]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row["market_value_m"], row["contract_expiry"]
            return np.nan, np.nan

        unmatched_names = df_merged.loc[unmatched_mask, "player_lower"]
        results = unmatched_names.apply(
            lambda n: pd.Series(fuzzy_lookup(n), index=["mv", "ce"])
        )
        df_merged.loc[unmatched_mask, "market_value_m"] = results["mv"].values
        df_merged.loc[unmatched_mask, "contract_expiry"] = results["ce"].values

        fuzzy_matched = df_merged["market_value_m"].notna().sum() - matched_primary
        print(f"Market value fuzzy matched additional: {fuzzy_matched}")

    total_matched = df_merged["market_value_m"].notna().sum()
    pct = total_matched / len(df_merged) * 100
    print(f"Market value total matched: {total_matched}/{len(df_merged)} ({pct:.1f}%)")

    df_merged = df_merged.drop(columns=["player_lower"], errors="ignore")
    return df_merged


# ── Extra-league loader ────────────────────────────────────────────────────────
# Raw league names chosen so that clean_league() produces the key used in
# LEAGUE_DIFFICULTY (e.g. "tr Super Lig" → strip "tr " → "Super Lig").
# Brazil uses "(Brazil)" suffix to distinguish from Italian "Serie A".
# Scotland uses "sct" 3-char prefix so the regex strips it cleanly.

EXTRA_LEAGUE_CODES = {
    "TUR": "tr Super Lig",
    "NED": "nl Eredivisie",
    "POR": "pt Primeira Liga",
    "DEN": "dk Superliga",
    "BEL": "be Belgian Pro League",
    "BRA": "br Serie A (Brazil)",
    "ARG": "ar Primera Division",
    "SCO": "sct Scottish Premiership",
}

_STD_RENAME = {
    "Player": "player", "Squad": "team", "Pos": "position_raw",
    "Age": "age", "Born": "birth_year", "Nation": "nationality",
    "Playing Time_Min": "minutes", "Playing Time_90s": "minutes_90s",
    "Performance_Gls": "goals", "Performance_Ast": "assists",
}
_SHOOT_RENAME  = {"Standard_Sh": "shots", "Standard_SoT": "shots_on_target"}
_PASS_RENAME   = {"Total_Cmp%": "pass_completion_rate",
                  "KP": "key_passes", "1/3": "passes_final_third"}
_POSS_RENAME   = {"Take-Ons_Succ": "dribbles_completed",
                  "Take-Ons_Att": "dribbles_attempted"}
_DEF_RENAME    = {"Tackles_Tkl": "tackles", "Tackles_TklW": "tackles_won",
                  "Int": "interceptions", "Blocks": "blocks", "Clr": "clearances"}
_MISC_RENAME   = {"Performance_Crs": "crosses"}


def load_extra_leagues_data():
    """Load the 8 non-Big5 FBref league CSV files and return a combined dataframe
    with the same column names as the Big-5 dataset."""
    all_dfs = []

    for code, raw_league in EXTRA_LEAGUE_CODES.items():
        std_path = DATA_RAW / f"fbref_{code}_standard.csv"
        if not std_path.exists():
            print(f"  [SKIP] {code}: {std_path} not found")
            continue

        std = pd.read_csv(std_path, low_memory=False)
        std = std.rename(columns=_STD_RENAME)
        std["league"] = raw_league

        core = [c for c in ["player","team","league","position_raw","age","birth_year",
                             "nationality","minutes","minutes_90s","goals","assists"]
                if c in std.columns]
        df = std[core].copy()

        # FBref extra-league files store age as "YY-DDD" — extract year part
        if "age" in df.columns:
            df["age"] = df["age"].astype(str).str.split("-").str[0]
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

        def _merge(path, rename):
            nonlocal df
            if not path.exists():
                return
            extra = pd.read_csv(path, low_memory=False).rename(columns=rename)
            # normalise key columns
            extra = extra.rename(columns={"Player": "player", "Squad": "team"})
            new_cols = [c for c in rename.values() if c in extra.columns]
            if not new_cols:
                return
            extra_sub = (extra[["player", "team"] + new_cols]
                         .drop_duplicates(["player", "team"]))
            df = df.merge(extra_sub, on=["player", "team"], how="left")

        _merge(DATA_RAW / f"fbref_{code}_shooting.csv",   _SHOOT_RENAME)
        _merge(DATA_RAW / f"fbref_{code}_passing.csv",    _PASS_RENAME)
        _merge(DATA_RAW / f"fbref_{code}_possession.csv", _POSS_RENAME)
        _merge(DATA_RAW / f"fbref_{code}_defense.csv",    _DEF_RENAME)
        _merge(DATA_RAW / f"fbref_{code}_misc.csv",       _MISC_RENAME)

        all_dfs.append(df)
        print(f"  {code}: {len(df)} players  ({raw_league})")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"  Extra-league total: {len(combined)} players across {len(all_dfs)} leagues")
    return combined


def run():
    # ── Big 5 ─────────────────────────────────────────────────────────────────
    df_big5 = load_raw()
    df_big5 = rename_columns(df_big5)

    stat_cols = ["age","minutes","minutes_90s","goals","assists","xg","xag","npxg",
                 "shots","shots_on_target","progressive_carries","progressive_passes",
                 "progressive_receives","key_passes","tackles","tackles_won",
                 "interceptions","blocks","clearances","crosses","dribbles_completed",
                 "dribbles_attempted","aerial_duels_won_pct","pass_completion_rate",
                 "passes_final_third","birth_year"]
    for col in stat_cols:
        if col in df_big5.columns:
            df_big5[col] = pd.to_numeric(df_big5[col], errors="coerce")

    # ── Extra leagues ──────────────────────────────────────────────────────────
    print("\nLoading extra-league data...")
    df_extra = load_extra_leagues_data()

    if not df_extra.empty:
        for col in stat_cols:
            if col in df_extra.columns:
                df_extra[col] = pd.to_numeric(df_extra[col], errors="coerce")
        df = pd.concat([df_big5, df_extra], ignore_index=True)
        print(f"\nCombined total: {len(df)} players ({len(df_big5)} Big5 + {len(df_extra)} extra)")
    else:
        df = df_big5
        print("No extra-league data loaded — using Big 5 only")

    df = map_positions(df)
    df = apply_filters(df)
    df = add_league_difficulty(df)
    df = add_market_values(df)

    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    t1 = df["position_group"].isna().sum() == 0
    print(f"TEST 3.1 position_group no nulls: {'PASS' if t1 else 'FAIL'}")

    if "nationality" in df.columns:
        excl_lower = [e.lower() for e in EXCLUDED_NATIONALITIES]
        nat_str = df["nationality"].fillna("").astype(str).str.lower()
        bad = nat_str.str.contains("|".join(excl_lower), na=False).any()
        t2 = not bad
    else:
        t2 = True
    print(f"TEST 3.2 no Israeli players: {'PASS' if t2 else 'FAIL'}")

    t3 = (df["minutes"] >= MIN_MINUTES).all()
    print(f"TEST 3.3 minutes >= {MIN_MINUTES}: {'PASS' if t3 else 'FAIL'}")

    t4 = "market_value_m" in df.columns
    print(f"TEST 3.4 market_value_m column exists: {'PASS' if t4 else 'FAIL'}")

    print("\nTEST 3.5 - Sample rows:")
    show_cols = [c for c in ["player","team","league_clean","position_group",
                              "age","minutes","goals","market_value_m"] if c in df.columns]
    print(df[show_cols].sample(5, random_state=42).to_string(index=False))

    out = DATA_PROC / "players_clean.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(f"Final: {len(df)} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    run()
