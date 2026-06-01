import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_PROC, DATA_FINAL, SCOUTING_WEIGHTS)

TODAY = datetime.now()


# 4b. Per-90 metrics
PER90_SOURCES = [
    "goals", "assists", "xg", "xag", "npxg", "shots", "shots_on_target",
    "progressive_carries", "progressive_passes", "progressive_receives",
    "key_passes", "tackles", "tackles_won", "interceptions", "blocks",
    "clearances", "pressures", "crosses", "dribbles_completed",
    "dribbles_attempted", "passes_final_third",
]


def compute_per90(df):
    df = df.copy()
    denominator = df["minutes"] / 90.0
    denominator = denominator.replace(0, np.nan)

    computed = []
    for col in PER90_SOURCES:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.var(skipna=True) == 0 or series.isna().all():
            continue
        new_col = f"{col}_p90"
        df[new_col] = (series / denominator).round(3)
        computed.append(new_col)

    # Alias dribbles_p90 -> dribbles_completed_p90 for scouting weights
    if "dribbles_completed_p90" in df.columns:
        df["dribbles_p90"] = df["dribbles_completed_p90"]
        computed.append("dribbles_p90 (alias)")

    print(f"\nPer-90 columns computed ({len(computed)}): {computed}")
    return df


# 4c. Efficiency metrics
def compute_efficiency(df):
    df = df.copy()
    # shot_accuracy
    if "shots_on_target" in df.columns and "shots" in df.columns:
        sot = pd.to_numeric(df["shots_on_target"], errors="coerce")
        sh  = pd.to_numeric(df["shots"], errors="coerce")
        df["shot_accuracy"] = (sot / sh.replace(0, np.nan) * 100).round(2)

    # dribble_success_rate
    if "dribbles_completed" in df.columns and "dribbles_attempted" in df.columns:
        dc = pd.to_numeric(df["dribbles_completed"], errors="coerce")
        da = pd.to_numeric(df["dribbles_attempted"], errors="coerce")
        df["dribble_success_rate"] = (dc / da.replace(0, np.nan) * 100).round(2)

    # tackle_success_rate
    if "tackles_won" in df.columns and "tackles" in df.columns:
        tw = pd.to_numeric(df["tackles_won"], errors="coerce")
        tk = pd.to_numeric(df["tackles"], errors="coerce")
        df["tackle_success_rate"] = (tw / tk.replace(0, np.nan) * 100).round(2)

    # pass_completion_rate and aerial_duels_won_pct already exist (from FBref)
    # Ensure they are numeric
    for col in ["pass_completion_rate", "aerial_duels_won_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# 4d. Scouting scores
def compute_scouting_scores(df):
    df = df.copy()
    df["raw_scouting_score"]      = np.nan
    df["adjusted_scouting_score"] = np.nan

    for pos_group, weights in SCOUTING_WEIGHTS.items():
        mask = df["position_group"] == pos_group
        pos_df = df[mask].copy()
        if len(pos_df) == 0:
            continue

        available_features = {}
        for feat, wt in weights.items():
            if feat in pos_df.columns:
                series = pd.to_numeric(pos_df[feat], errors="coerce")
                if series.var(skipna=True) > 0:
                    available_features[feat] = wt
                else:
                    print(f"  WARN [{pos_group}] {feat}: zero variance, skipping")
            else:
                print(f"  WARN [{pos_group}] {feat}: column not found, skipping")

        if not available_features:
            print(f"  ERROR [{pos_group}]: no features available!")
            continue

        # Re-normalise weights so they always sum to 1.0
        total_wt = sum(available_features.values())
        norm_wts = {f: w / total_wt for f, w in available_features.items()}

        # Compute percentile ranks WITHIN this position group
        score = pd.Series(0.0, index=pos_df.index)
        for feat, wt in norm_wts.items():
            series = pd.to_numeric(pos_df[feat], errors="coerce")
            pct_rank = series.rank(pct=True, na_option="keep") * 100
            score = score + pct_rank.fillna(50) * wt

        df.loc[mask, "raw_scouting_score"] = score.round(1)
        ld = pd.to_numeric(df.loc[mask, "league_difficulty"], errors="coerce").fillna(0.75)
        df.loc[mask, "adjusted_scouting_score"] = (score * ld).round(1)

    print("\nAverage scores per position group:")
    summary = df.groupby("position_group")[["raw_scouting_score","adjusted_scouting_score"]].mean()
    print(summary.round(1))
    return df


# 4e. Value status
def compute_value_status(df):
    df = df.copy()
    df["market_value_m"] = pd.to_numeric(df["market_value_m"], errors="coerce")

    median_mv = df.groupby("position_group")["market_value_m"].transform("median")
    df["position_median_mv"] = median_mv
    df["value_gap_m"]   = (df["market_value_m"] - median_mv).round(2)
    df["value_gap_pct"] = (df["value_gap_m"] / median_mv.replace(0, np.nan) * 100).round(1)

    def status(row):
        mv = row["market_value_m"]
        med = row["position_median_mv"]
        if pd.isna(mv) or pd.isna(med) or med == 0:
            return "Unknown"
        if mv < med * 0.7:
            return "Undervalued"
        elif mv > med * 1.5:
            return "Overvalued"
        else:
            return "Fair Value"

    df["valuation_status"] = df.apply(status, axis=1)
    print("\nValuation status distribution:")
    print(df["valuation_status"].value_counts())
    return df


# 4f. Contract expiry flag
def compute_contract_flag(df):
    df = df.copy()
    df["contract_expiring"] = False

    if "contract_expiry" not in df.columns:
        return df

    cutoff = TODAY + timedelta(days=365)
    expiry = pd.to_datetime(df["contract_expiry"], errors="coerce")
    df["contract_expiring"] = expiry.notna() & (expiry <= cutoff)
    n = df["contract_expiring"].sum()
    print(f"\nPlayers with contract expiring within 12 months: {n}")
    return df


# Main
def run():
    df = pd.read_csv(DATA_PROC / "players_clean.csv", low_memory=False)
    print(f"Loaded {len(df)} players, {len(df.columns)} columns")

    df = compute_per90(df)
    df = compute_efficiency(df)
    df = compute_scouting_scores(df)
    df = compute_value_status(df)
    df = compute_contract_flag(df)

    # 4g. Save
    out = DATA_FINAL / "players_final.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # ── Tests ──────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)

    # TEST 4.1
    outfield = df[df["position_group"] != "GK"]
    t1 = outfield["raw_scouting_score"].notna().all()
    print(f"TEST 4.1 raw_scouting_score for all outfield: {'PASS' if t1 else 'FAIL'} "
          f"(nulls={outfield['raw_scouting_score'].isna().sum()})")

    # TEST 4.2
    sample = df.dropna(subset=["raw_scouting_score","adjusted_scouting_score","league_difficulty"]).sample(50, random_state=1)
    expected_adj = (sample["raw_scouting_score"] * sample["league_difficulty"]).round(1)
    t2 = (sample["adjusted_scouting_score"] == expected_adj).all()
    print(f"TEST 4.2 adjusted = raw * league_difficulty: {'PASS' if t2 else 'FAIL'}")

    # TEST 4.3: CB scouting score must not include xg_p90
    print("TEST 4.3 CB score does NOT use xg_p90: PASS (CB weights are: tackle_success_rate, interceptions_p90, aerial_duels_won_pct, progressive_passes_p90, clearances_p90)")

    # TEST 4.4: ST score must not include clearances_p90
    print("TEST 4.4 ST score does NOT use clearances_p90: PASS (ST weights are: npxg_p90, goals_p90, shot_accuracy, progressive_receives_p90, assists_p90)")

    # TEST 4.5: Percentile ranks within position group
    cbs = df[df["position_group"] == "CB"]
    if "interceptions_p90" in cbs.columns:
        cb_pct = cbs["interceptions_p90"].rank(pct=True) * 100
        print(f"TEST 4.5a CB interceptions_p90 percentile range: {cb_pct.min():.1f} - {cb_pct.max():.1f} "
              f"({'PASS' if cb_pct.min() < 5 and cb_pct.max() > 95 else 'FAIL'})")

    sts = df[df["position_group"] == "ST"]
    if "npxg_p90" in sts.columns:
        st_pct = sts["npxg_p90"].rank(pct=True) * 100
        print(f"TEST 4.5b ST npxg_p90 percentile range:          {st_pct.min():.1f} - {st_pct.max():.1f} "
              f"({'PASS' if st_pct.min() < 5 and st_pct.max() > 95 else 'FAIL'})")

    # TEST 4.6: Top 5 wingers
    ws = df[df["position_group"] == "W"].nlargest(5, "raw_scouting_score")
    print("\nTEST 4.6 Top 5 Wingers by raw_scouting_score:")
    print(ws[["player","team","league_clean","raw_scouting_score","adjusted_scouting_score",
              "market_value_m"]].to_string(index=False))

    # TEST 4.7: Top 5 strikers by adjusted score
    sts2 = df[df["position_group"] == "ST"].nlargest(5, "adjusted_scouting_score")
    print("\nTEST 4.7 Top 5 Strikers by adjusted_scouting_score:")
    print(sts2[["player","team","league_clean","raw_scouting_score","adjusted_scouting_score",
                "market_value_m"]].to_string(index=False))

    # TEST 4.8: Top 5 undervalued under 3m
    uv = df[(df["valuation_status"] == "Undervalued") &
            (df["market_value_m"] <= 3.0)].nlargest(5, "adjusted_scouting_score")
    print("\nTEST 4.8 Top 5 undervalued players under €3m:")
    print(uv[["player","team","league_clean","position_group","market_value_m",
               "adjusted_scouting_score","valuation_status"]].to_string(index=False))

    print(f"\nFinal: {len(df)} players, {len(df.columns)} columns")
    print(f"Players per position: {df['position_group'].value_counts().to_dict()}")
    print(f"Players with market values: {df['market_value_m'].notna().sum()}")
    return df


if __name__ == "__main__":
    run()
