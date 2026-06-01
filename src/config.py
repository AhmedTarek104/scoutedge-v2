from pathlib import Path

ROOT       = Path(r"F:\scoutedge_v2")
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
        "tackle_success_rate":    0.25,
        "interceptions_p90":      0.25,
        "aerial_duels_won_pct":   0.20,
        "progressive_passes_p90": 0.20,
        "clearances_p90":         0.10,
    },
    "FB": {
        "progressive_carries_p90": 0.25,
        "key_passes_p90":          0.20,
        "tackles_p90":             0.20,
        "interceptions_p90":       0.15,
        "assists_p90":             0.20,
    },
    "DM": {
        "tackles_p90":             0.30,
        "interceptions_p90":       0.25,
        "pass_completion_rate":    0.20,
        "progressive_passes_p90":  0.15,
        "pressures_p90":           0.10,
    },
    "CM": {
        "progressive_passes_p90":  0.25,
        "key_passes_p90":          0.20,
        "assists_p90":             0.20,
        "tackles_p90":             0.20,
        "goals_p90":               0.15,
    },
    "AM": {
        "xag_p90":                 0.30,
        "key_passes_p90":          0.25,
        "assists_p90":             0.20,
        "progressive_carries_p90": 0.15,
        "goals_p90":               0.10,
    },
    "W": {
        "progressive_carries_p90": 0.25,
        "goals_p90":               0.25,
        "xg_p90":                  0.20,
        "dribbles_p90":            0.20,
        "assists_p90":             0.10,
    },
    "ST": {
        "npxg_p90":                0.30,
        "goals_p90":               0.25,
        "shot_accuracy":           0.20,
        "progressive_receives_p90":0.15,
        "assists_p90":             0.10,
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
