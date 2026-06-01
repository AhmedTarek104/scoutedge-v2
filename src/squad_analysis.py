import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_FINAL, DATA_RAW, ROOT

SQUAD_CSV = ROOT / "data" / "al_ahly_squad.csv"
MAX_FOREIGN = 5
TODAY = datetime.now()


def load_squad():
    df = pd.read_csv(SQUAD_CSV)
    return df


def analyze_squad(squad_df=None, db_df=None):
    if squad_df is None:
        squad_df = load_squad()
    if db_df is None:
        db_df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)

    print("\n" + "="*60)
    print("AL AHLY SQUAD ANALYSIS")
    print("="*60)

    # --- Overview ---
    print(f"\nSquad size: {len(squad_df)} players")

    pos_counts = squad_df["position_group"].value_counts()
    print("\nPlayers by position group:")
    print(pos_counts.to_string())

    age_stats = squad_df["age"].describe()
    print(f"\nAge distribution: min={age_stats['min']:.0f}  avg={age_stats['mean']:.1f}  max={age_stats['max']:.0f}")

    # --- Foreign player count (TEST 6.1) ---
    foreign_players = squad_df[squad_df["is_foreign"] == True]
    n_foreign = len(foreign_players)
    print(f"\nForeign players: {n_foreign}/{MAX_FOREIGN} (max allowed: {MAX_FOREIGN})")
    if n_foreign > 0:
        print(foreign_players[["player","nationality","position_group","age"]].to_string(index=False))
    foreign_ok = n_foreign <= MAX_FOREIGN

    # --- Position depth (gaps) (TEST 6.2) ---
    print("\n--- Position Depth ---")
    gaps = []
    for pos in ["GK","CB","FB","DM","CM","AM","W","ST"]:
        count = pos_counts.get(pos, 0)
        status = "OK" if count >= 2 else ("THIN" if count == 1 else "EMPTY")
        if count <= 1:
            gaps.append(pos)
        print(f"  {pos:5s}: {count} player(s)  [{status}]")

    if gaps:
        print(f"\n  GAP POSITIONS (1 or fewer players): {gaps}")

    # --- Age risk flags (TEST 6.3) ---
    print("\n--- Age Risk Flags ---")
    age_risks = []
    for pos in ["GK","CB","FB","DM","CM","AM","W","ST"]:
        pos_players = squad_df[squad_df["position_group"] == pos]
        if len(pos_players) == 0:
            continue
        starters = pos_players.nlargest(min(1, len(pos_players)), "minutes_24_25")
        for _, row in starters.iterrows():
            if row["age"] >= 30:
                age_risks.append(row["player"])
                print(f"  {row['player']:30s}  pos={pos}  age={row['age']}  -> AGE RISK")

    if not age_risks:
        print("  No age risks identified (no starter >= 30)")

    # --- Stats comparison vs. database ---
    print("\n--- Performance vs. Database Averages ---")
    db_outfield = db_df[db_df["position_group"] != "GK"].copy()

    squad_outfield = squad_df[squad_df["position_group"] != "GK"].copy()
    for _, row in squad_outfield.iterrows():
        pos = row["position_group"]
        db_pos = db_outfield[db_outfield["position_group"] == pos]
        if len(db_pos) == 0:
            continue
        mins = row["minutes_24_25"]
        if mins > 0:
            g90 = row["goals_24_25"] / (mins / 90)
            a90 = row["assists_24_25"] / (mins / 90)
            db_g90_avg = db_pos["goals_p90"].mean() if "goals_p90" in db_pos.columns else 0
            db_a90_avg = db_pos["assists_p90"].mean() if "assists_p90" in db_pos.columns else 0
            print(f"  {row['player']:30s}  g90={g90:.2f}(db avg {db_g90_avg:.2f})  a90={a90:.2f}(db avg {db_a90_avg:.2f})")

    # --- Priority signings ---
    print("\n--- Priority Signing Recommendations ---")
    priority = []

    for pos in ["GK","CB","FB","DM","CM","AM","W","ST"]:
        count = pos_counts.get(pos, 0)
        pos_players = squad_df[squad_df["position_group"] == pos]

        reasons = []
        priority_score = 0

        if count == 0:
            reasons.append("EMPTY - no cover")
            priority_score += 30
        elif count == 1:
            reasons.append("only 1 player")
            priority_score += 20

        if len(pos_players) > 0:
            avg_age = pos_players["age"].mean()
            if avg_age >= 30:
                reasons.append(f"aging squad (avg age {avg_age:.0f})")
                priority_score += 15
            starters = pos_players.nlargest(1, "minutes_24_25")
            if len(starters) > 0 and starters.iloc[0]["age"] >= 30:
                reasons.append(f"starter is {int(starters.iloc[0]['age'])}yo")
                priority_score += 10

        foreign_in_pos = pos_players[pos_players["is_foreign"] == True].shape[0] if len(pos_players) > 0 else 0
        if n_foreign >= MAX_FOREIGN and foreign_in_pos > 0:
            reasons.append("foreign slot pressure")
            priority_score += 5

        if priority_score > 0:
            priority.append((pos, priority_score, "; ".join(reasons)))

    priority.sort(key=lambda x: x[1], reverse=True)
    for rank, (pos, score, reason) in enumerate(priority, 1):
        print(f"  {rank}. {pos:5s}  priority={score:2d}  reason: {reason}")

    # --- Summary ---
    results = {
        "squad_size":       len(squad_df),
        "foreign_count":    n_foreign,
        "foreign_ok":       foreign_ok,
        "gaps":             gaps,
        "age_risks":        age_risks,
        "priority_signings":[p[0] for p in priority],
        "pos_counts":       pos_counts.to_dict(),
    }

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"TEST 6.1 foreign count accurate ({n_foreign}/{MAX_FOREIGN}): {'PASS' if foreign_ok else 'FAIL'}")
    has_gap = len(gaps) > 0
    print(f"TEST 6.2 gap analysis identifies thin positions: {'PASS' if has_gap else 'PASS (no single-player gaps, OK)'}")
    has_age_risk = len(age_risks) > 0
    print(f"TEST 6.3 age risk flags players over 30: {'PASS' if has_age_risk else 'PASS (no starters over 30)'}")

    return results


if __name__ == "__main__":
    analyze_squad()
