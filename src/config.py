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
    # Weights use only metrics available across ALL 13 leagues.
    # Big-5-only metrics (xg_p90, key_passes_p90) are included where relevant;
    # compute_scouting_scores redistributes their weight per-player when absent.
    "CB": {
        "tackles_won_p90":   0.40,
        "interceptions_p90": 0.40,
        "goals_p90":         0.10,
        "crosses_p90":       0.10,
    },
    "FB": {
        "crosses_p90":       0.35,
        "tackles_won_p90":   0.30,
        "interceptions_p90": 0.20,
        "assists_p90":       0.15,
    },
    "DM": {
        "tackles_won_p90":   0.45,
        "interceptions_p90": 0.40,
        "assists_p90":       0.15,
    },
    "CM": {
        "goals_p90":         0.30,
        "assists_p90":       0.30,
        "tackles_won_p90":   0.25,
        "key_passes_p90":    0.15,
    },
    "AM": {
        "assists_p90":       0.35,
        "goals_p90":         0.30,
        "key_passes_p90":    0.25,
        "xg_p90":            0.10,
    },
    "W": {
        "goals_p90":         0.30,
        "assists_p90":       0.25,
        "xg_p90":            0.25,
        "crosses_p90":       0.20,
    },
    "ST": {
        "goals_p90":            0.40,
        "xg_p90":               0.35,
        "shots_on_target_p90":  0.15,
        "assists_p90":          0.10,
    },
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
