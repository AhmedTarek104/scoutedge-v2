from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"
DATA_FINAL = ROOT / "data" / "final"

SEASON = "2024-2025"

POSITION_GROUP_MAP = {
    "GK":    "GK",
    "DF":    "CB",
    "DF,MF": "FB",
    "DF,FW": "FB",
    "MF,DF": "DM",
    "MF":    "CM",
    "MF,FW": "AM",
    "FW,MF": "W",
    "FW":    "ST",
    "FW,DF": "W",
}

SCOUTING_WEIGHTS = {
    "CB": {
        "tackle_success_rate":     0.25,
        "interceptions_p90":       0.25,
        "tackles_won_p90":         0.20,
        "progressive_passes_p90":  0.15,
        "clearances_p90":          0.15,
    },
    "FB": {
        "progressive_carries_p90": 0.20,
        "key_passes_p90":          0.20,
        "tackles_won_p90":         0.20,
        "tackle_success_rate":     0.15,
        "interceptions_p90":       0.15,
        "assists_p90":             0.10,
    },
    "DM": {
        "tackle_success_rate":     0.25,
        "tackles_won_p90":         0.25,
        "interceptions_p90":       0.20,
        "pass_completion_rate":    0.15,
        "progressive_passes_p90":  0.15,
    },
    "CM": {
        "progressive_passes_p90":  0.20,
        "key_passes_p90":          0.20,
        "assists_p90":             0.20,
        "tackle_success_rate":     0.15,
        "goals_p90":               0.15,
        "pass_completion_rate":    0.10,
    },
    "AM": {
        "xag_p90":                 0.25,
        "key_passes_p90":          0.25,
        "assists_p90":             0.20,
        "goals_p90":               0.15,
        "dribble_success_rate":    0.15,
    },
    "W": {
        "progressive_carries_p90": 0.20,
        "goals_p90":               0.20,
        "xg_p90":                  0.15,
        "dribble_success_rate":    0.20,
        "assists_p90":             0.10,
        "key_passes_p90":          0.15,
    },
    "ST": {
        "goals_p90":               0.25,
        "npxg_p90":                0.25,
        "goals_per_shot":          0.20,
        "shots_on_target_p90":     0.15,
        "assists_p90":             0.15,
    },
}

LEAGUE_TIERS = {
    # Tier 1 — Big 5
    "Premier League":       1,
    "La Liga":              1,
    "Bundesliga":           1,
    "Serie A":              1,
    "Ligue 1":              1,
    # Tier 2 — Strong
    "Eredivisie":           2,
    "Primeira Liga":        2,
    "Super Lig":            2,
    "Belgian Pro League":   2,
    # Tier 3 — Developing
    "Superliga":            3,
    "Scottish Premiership": 3,
    "Serie A (Brazil)":     3,
    "Primera Division":     3,
}

LEAGUE_DIFFICULTY = {
    "Premier League":          1.00,
    "La Liga":                 1.00,
    "Bundesliga":              1.00,
    "Serie A":                 1.00,
    "Ligue 1":                 1.00,
    "Eredivisie":              0.88,
    "Primeira Liga":           0.88,
    "Super Lig":               0.80,
    "Belgian Pro League":      0.80,
    "Superliga":               0.72,
    "Scottish Premiership":    0.72,
    "Serie A (Brazil)":        0.75,
    "Primera Division":        0.75,
    "MLS":                     0.68,
    "Saudi Pro League":        0.68,
    "Egyptian Premier League": 0.65,
}

TARGET_LEAGUES = [
    "Super Lig",
    "Superliga",
    "Eredivisie",
    "Primeira Liga",
    "Belgian Pro League",
    "Serie A (Brazil)",
    "Primera Division",
    "Scottish Premiership",
    "Ligue 1",
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
]

EXCLUDED_NATIONALITIES = ["Israel", "Israeli"]

DEFAULT_MAX_BUDGET_M = 3.0
MIN_MINUTES = 900
MIN_AGE = 16
MAX_AGE = 35

COLOR_PRIMARY = "#CC0000"
COLOR_SUCCESS = "#00C853"
COLOR_WARNING = "#FFB300"
COLOR_DARK    = "#0D0D0D"
COLOR_CARD    = "#1A1A1A"
COLOR_TEXT    = "#F5F5F5"
COLOR_MUTED   = "#888888"
COLOR_BORDER  = "#2A2A2A"
