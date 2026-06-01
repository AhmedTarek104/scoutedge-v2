import sys, pickle, io
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_FINAL, DATA_PROC, TARGET_LEAGUES
from similarity import build_similarity_matrices, get_similar_players
from squad_analysis import analyze_squad

def run_final_verification():
    print("="*70)
    print("STEP 7 — FINAL VERIFICATION")
    print("="*70)

    # 7a. Load final dataset
    df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
    print(f"\n7a. Dataset summary:")
    print(f"    Total players:      {len(df)}")
    print(f"    Players per position:")
    for pos, cnt in df["position_group"].value_counts().items():
        print(f"      {pos:5s}: {cnt}")

    p90_cols = [c for c in df.columns if c.endswith("_p90") and df[c].var(skipna=True) > 0]
    print(f"    Non-zero variance per-90 columns ({len(p90_cols)}):")
    print(f"      {p90_cols}")

    n_mv = df["market_value_m"].notna().sum()
    print(f"    Players with market values: {n_mv} ({n_mv/len(df)*100:.1f}%)")

    print(f"\n    Avg adjusted scouting score per position:")
    avg_scores = df.groupby("position_group")["adjusted_scouting_score"].mean()
    print(avg_scores.round(1).to_string())

    # 7b. Re-run similarity tests
    print("\n" + "="*70)
    print("7b. SIMILARITY TESTS (all 5)")
    print("="*70)

    print("\nBuilding similarity matrices...")
    matrices = build_similarity_matrices(df)

    tests = [
        ("Aaron Wan-Bissaka", ["FB","CB"],
         "DF -> CB in FBref"),
        ("Harry Kane",        ["ST"],
         "FW -> ST in FBref"),
        ("Bukayo Saka",       ["W"],
         "FW,MF -> W (Salah=FW->ST, Rodri=73min, substituted)"),
        ("Moisés Caicedo",    ["DM"],
         "MF,DF -> DM (replaces Rodri Man City)"),
        ("Virgil van Dijk",   ["CB"],
         "DF -> CB in FBref"),
    ]

    all_sim_pass = True
    for player_name, allowed_groups, note in tests:
        results = get_similar_players(player_name, df, matrices,
                                      n=10, target_leagues_only=False)
        if results.empty:
            print(f"  {player_name}: FAIL (not found)")
            all_sim_pass = False
            continue
        bad = results[~results["position_group"].isin(allowed_groups)]
        status = "PASS" if len(bad) == 0 else "FAIL"
        pos_found = results["position_group"].unique().tolist()
        print(f"  {player_name:25s}  {status}  groups={pos_found}  note: {note}")
        if status == "FAIL":
            all_sim_pass = False

    print(f"\n  SIMILARITY TESTS: {'ALL PASS' if all_sim_pass else 'SOME FAILED'}")

    # 7c. Top 10 wingers: age<=27, mv<=3m, target leagues
    print("\n" + "="*70)
    print("7c. TOP 10 WINGERS — age<=27, mv<=3m, target leagues, by adjusted score")
    print("="*70)
    wingers = df[
        (df["position_group"] == "W") &
        (df["age"] <= 27) &
        (df["market_value_m"] <= 3.0) &
        (df["league_clean"].isin(TARGET_LEAGUES))
    ].nlargest(10, "adjusted_scouting_score")

    show = [c for c in ["player","team","league_clean","age","market_value_m",
                         "raw_scouting_score","adjusted_scouting_score",
                         "valuation_status","contract_expiring"] if c in wingers.columns]
    print(wingers[show].to_string(index=False))

    # 7d. Top 10 strikers: age<=28, mv<=3m, target leagues
    print("\n" + "="*70)
    print("7d. TOP 10 STRIKERS — age<=28, mv<=3m, target leagues, by adjusted score")
    print("="*70)
    strikers = df[
        (df["position_group"] == "ST") &
        (df["age"] <= 28) &
        (df["market_value_m"] <= 3.0) &
        (df["league_clean"].isin(TARGET_LEAGUES))
    ].nlargest(10, "adjusted_scouting_score")

    print(strikers[show].to_string(index=False))

    # 7e. Squad analysis
    print("\n" + "="*70)
    print("7e. AL AHLY SQUAD ANALYSIS SUMMARY")
    print("="*70)
    squad_results = analyze_squad(db_df=df)

    # Final verdict
    print("\n" + "="*70)
    data_ok  = len(df) > 1000
    sim_ok   = all_sim_pass
    squad_ok = squad_results["foreign_ok"]

    if data_ok and sim_ok and squad_ok:
        print(f"SESSION 1 COMPLETE — All data pipeline tests passed.")
        print(f" Players in database: {len(df)}")
        print(f" Similarity engine: VERIFIED")
        print(f" Position filtering: CORRECT")
        print(f" Ready for Session 2 (dashboard build)")
    else:
        print("VERIFICATION FAILED — check errors above")
    print("="*70)


if __name__ == "__main__":
    run_final_verification()
