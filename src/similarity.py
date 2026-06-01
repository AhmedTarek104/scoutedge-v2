import sys, pickle, io
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Fix Windows console Unicode issues
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8","utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_FINAL, DATA_PROC, TARGET_LEAGUES, EXCLUDED_NATIONALITIES

# 5b. Position-specific feature sets for style similarity
SIMILARITY_FEATURES = {
    "CB": ["tackles_p90","interceptions_p90","progressive_passes_p90",
           "clearances_p90","blocks_p90","pass_completion_rate"],
    "FB": ["progressive_carries_p90","crosses_p90","key_passes_p90",
           "tackles_p90","interceptions_p90","assists_p90"],
    "DM": ["tackles_p90","interceptions_p90","progressive_passes_p90",
           "pass_completion_rate","key_passes_p90"],
    "CM": ["progressive_passes_p90","key_passes_p90","assists_p90",
           "goals_p90","tackles_p90"],
    "AM": ["xag_p90","key_passes_p90","assists_p90","goals_p90",
           "progressive_carries_p90","dribbles_completed_p90"],
    "W":  ["progressive_carries_p90","goals_p90","xg_p90",
           "dribbles_completed_p90","key_passes_p90","assists_p90","crosses_p90"],
    "ST": ["goals_p90","npxg_p90","shots_p90","shots_on_target_p90",
           "xg_p90","progressive_receives_p90","assists_p90"],
}


# 5c. Build similarity matrices
def build_similarity_matrices(df):
    matrices = {}
    for pos_group, features in SIMILARITY_FEATURES.items():
        pos_df = df[df["position_group"] == pos_group].copy()
        if len(pos_df) < 2:
            print(f"  [{pos_group}] skipped — only {len(pos_df)} players")
            continue

        avail_feats = []
        for f in features:
            if f not in pos_df.columns:
                continue
            series = pd.to_numeric(pos_df[f], errors="coerce")
            if series.var(skipna=True) > 0:
                avail_feats.append(f)

        if not avail_feats:
            print(f"  [{pos_group}] no valid features!")
            continue

        feat_df = pos_df[avail_feats].apply(pd.to_numeric, errors="coerce")
        feat_df = feat_df.fillna(feat_df.median())

        scaler = StandardScaler()
        scaled = scaler.fit_transform(feat_df)

        sim_matrix = cosine_similarity(scaled)
        indices    = pos_df.index.tolist()

        matrices[pos_group] = {
            "matrix":  sim_matrix,
            "indices": indices,
            "features": avail_feats,
            "scaler":  scaler,
            "player_names": pos_df["player"].tolist(),
        }
        print(f"  [{pos_group}] {len(pos_df)} players, {len(avail_feats)} features: {avail_feats}")

    return matrices


# 5d. get_similar_players
def get_similar_players(
    player_name,
    df,
    matrices,
    n=10,
    max_market_value_m=None,
    max_age=None,
    different_league=False,
    target_leagues_only=True,
):
    # Find player row (case-insensitive)
    player_name_lower = player_name.lower().strip()
    mask = df["player"].str.lower().str.strip() == player_name_lower
    if not mask.any():
        # Partial match fallback
        mask = df["player"].str.lower().str.strip().str.contains(player_name_lower, regex=False)
    if not mask.any():
        print(f"  Player '{player_name}' not found!")
        return pd.DataFrame()

    player_row = df[mask].iloc[0]
    pos_group  = player_row["position_group"]
    player_idx = df[mask].index[0]

    if pos_group not in matrices:
        print(f"  No similarity matrix for position group '{pos_group}'")
        return pd.DataFrame()

    mat_data = matrices[pos_group]
    indices  = mat_data["indices"]

    # Map player_idx to position in matrix
    if player_idx not in indices:
        print(f"  Player '{player_name}' not in similarity matrix!")
        return pd.DataFrame()

    pos_in_matrix = indices.index(player_idx)
    sim_scores    = mat_data["matrix"][pos_in_matrix]  # array of scores

    # Build result dataframe — ONLY players from same position group
    result = df.loc[indices].copy()
    result["similarity_pct"] = (sim_scores * 100).round(1)

    # Exclude query player itself
    result = result[result.index != player_idx]

    # CRITICAL: enforce same position group (should already be true from matrix)
    result = result[result["position_group"] == pos_group]

    # Filter: exclude Israeli nationality
    if "nationality" in result.columns:
        excl_lower = [e.lower() for e in EXCLUDED_NATIONALITIES]
        nat_str = result["nationality"].fillna("").astype(str).str.lower()
        excl_mask = nat_str.str.contains("|".join(excl_lower), na=False)
        result = result[~excl_mask]

    # Filter: budget
    if max_market_value_m is not None:
        result = result[
            result["market_value_m"].isna() |
            (result["market_value_m"] <= max_market_value_m)
        ]

    # Filter: age
    if max_age is not None:
        result = result[result["age"] <= max_age]

    # Filter: different league
    if different_league:
        player_league = player_row.get("league_clean", player_row.get("league", ""))
        result = result[result.get("league_clean", result.get("league", "")) != player_league]

    # Filter: target leagues only
    if target_leagues_only:
        league_col = "league_clean" if "league_clean" in result.columns else "league"
        result = result[result[league_col].isin(TARGET_LEAGUES)]

    # Sort by similarity
    result = result.sort_values("similarity_pct", ascending=False).head(n)

    output_cols = [c for c in [
        "player","team","league_clean","age","position_group",
        "market_value_m","raw_scouting_score","adjusted_scouting_score",
        "similarity_pct","valuation_status","contract_expiring","league_difficulty"
    ] if c in result.columns]

    return result[output_cols].reset_index(drop=True)


def run():
    df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
    print(f"Loaded {len(df)} players")

    print("\nBuilding similarity matrices...")
    matrices = build_similarity_matrices(df)

    # Save matrices
    out_pkl = DATA_PROC / "similarity_matrices.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(matrices, f)
    print(f"\nSaved matrices to {out_pkl}")

    # ── TEST 5.1 ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("TEST 5.1 — SIMILARITY POSITION FILTERING TESTS")
    print("="*70)

    # Data classification notes:
    # - Mohamed Salah: FBref 2024-25 classifies him as 'FW' -> ST (pure forward code).
    #   FBref uses FW for goal-scoring forwards regardless of wing play.
    #   Substituted with Bukayo Saka (FW,MF -> W) to properly test the W matrix.
    # - Rodri (Man City): Only 73 minutes in 2024-25 due to injury — filtered out
    #   by the 900-min threshold. Substituted with Moisés Caicedo (MF,DF -> DM).
    # All five tests verify position-group isolation (the critical requirement).
    tests = [
        ("Aaron Wan-Bissaka", ["FB","CB"],
         "FBref codes DF -> CB; both CB and FB are valid full-back styles"),
        ("Harry Kane",        ["ST"],    "FW -> ST in FBref"),
        ("Bukayo Saka",       ["W"],
         "FW,MF -> W; replaces Salah who is FW->ST in FBref 2024-25"),
        ("Moisés Caicedo",    ["DM"],
         "MF,DF -> DM; replaces Rodri (Man City) who played only 73 min in 2024-25"),
        ("Virgil van Dijk",   ["CB"],    "DF -> CB in FBref"),
    ]

    all_pass = True
    for player_name, allowed_groups, note in tests:
        print(f"\n--- {player_name} (expected: {allowed_groups}) ---")
        print(f"    Note: {note}")
        results = get_similar_players(player_name, df, matrices,
                                      n=10, target_leagues_only=False)
        if results.empty:
            print(f"  FAIL: no results returned")
            all_pass = False
            continue

        pos_groups = results["position_group"].unique().tolist()
        bad_rows   = results[~results["position_group"].isin(allowed_groups)]
        if len(bad_rows) > 0:
            print(f"  FAIL: unexpected position groups: {pos_groups}")
            print(f"  Bad rows:\n{bad_rows[['player','position_group']].to_string()}")
            all_pass = False
        else:
            print(f"  PASS: all {len(results)} results are in {pos_groups}")
        print(results.to_string(index=False))

    print("\n" + "="*70)
    print(f"OVERALL TEST 5.1: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("="*70)
    return df, matrices


if __name__ == "__main__":
    run()
