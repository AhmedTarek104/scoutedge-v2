"""Recompute scouting scores on the existing players_final.csv using unified weights."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r"F:\scoutedge_v2")
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_FINAL
from features import compute_scouting_scores

print("Loading players_final.csv...")
df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
print(f"  {len(df)} players, {len(df.columns)} columns")

print("\nRecomputing scores with unified per-player weight redistribution...")
df = compute_scouting_scores(df)

out = DATA_FINAL / "players_final.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")

# ── Verification ──────────────────────────────────────────────────────────────
BIG5 = {"Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

print("\n=== Top 10 Wingers by adjusted_scouting_score ===")
ws = (df[df["position_group"] == "W"]
        .nlargest(10, "adjusted_scouting_score")
        [["player", "team", "league_clean", "goals_p90", "assists_p90",
          "xg_p90", "crosses_p90", "raw_scouting_score", "adjusted_scouting_score"]])
print(ws.to_string(index=False))

print("\n=== Score distribution by league group ===")
df["_grp"] = df["league_clean"].apply(lambda l: "Big 5" if l in BIG5 else "Non-Big-5")
grp = df.groupby("_grp")[["raw_scouting_score", "adjusted_scouting_score"]].agg(
    ["mean", "median", "std"]).round(1)
print(grp.to_string())
df.drop(columns=["_grp"], inplace=True)

print("\n=== Score distribution per league ===")
league_stats = (df.groupby("league_clean")["adjusted_scouting_score"]
                  .agg(["count", "mean", "median"])
                  .round(1)
                  .sort_values("mean", ascending=False))
print(league_stats.to_string())

print("\n=== Top 5 per position — Non-Big-5 only ===")
non_b5 = df[~df["league_clean"].isin(BIG5)]
for pos in ["CB", "FB", "DM", "CM", "AM", "W", "ST"]:
    top = (non_b5[non_b5["position_group"] == pos]
             .nlargest(3, "adjusted_scouting_score")
             [["player", "league_clean", "adjusted_scouting_score"]])
    print(f"\n  {pos}:")
    for _, r in top.iterrows():
        print(f"    {r['player']:25s} {r['league_clean']:25s} {r['adjusted_scouting_score']:.1f}")
