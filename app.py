import sys, pickle, re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (DATA_FINAL, DATA_PROC, ROOT, TARGET_LEAGUES, SCOUTING_WEIGHTS,
                    EXCLUDED_NATIONALITIES)
from similarity import build_similarity_matrices, get_similar_players

# ── Data Loading ──────────────────────────────────────────────────────────────

print("Loading players_final.csv...")
DF = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
DF["market_value_m"]       = pd.to_numeric(DF["market_value_m"],       errors="coerce")
DF["age"]                  = pd.to_numeric(DF["age"],                  errors="coerce")
DF["minutes"]              = pd.to_numeric(DF["minutes"],              errors="coerce")
DF["adjusted_scouting_score"] = pd.to_numeric(DF["adjusted_scouting_score"], errors="coerce")
DF["raw_scouting_score"]   = pd.to_numeric(DF["raw_scouting_score"],   errors="coerce")
DF["contract_expiring"]    = DF["contract_expiring"].astype(str).str.lower().isin(["true","1","yes"])

# Deduplicate: where the same player appears at multiple clubs (mid-season transfer),
# keep the record with the highest adjusted_scouting_score.
_pre_dedup = len(DF)
DF = (DF.sort_values("adjusted_scouting_score", ascending=False, na_position="last")
       .drop_duplicates(subset="player", keep="first")
       .reset_index(drop=True))
if len(DF) < _pre_dedup:
    print(f"  Removed {_pre_dedup - len(DF)} duplicate player records (mid-season transfers)")

# Pre-compute estimated MV for players with null market_value_m
# Median of valued peers grouped by (position, league, age_bucket)
def _age_bucket(age):
    if pd.isna(age): return "unk"
    a = int(age)
    if a <= 21: return "u22"
    if a <= 24: return "22-24"
    if a <= 27: return "25-27"
    if a <= 29: return "28-29"
    return "30p"

DF["_age_bucket"] = DF["age"].apply(_age_bucket)
_valued = DF[DF["market_value_m"].notna()]
_mv_med_full = _valued.groupby(["position_group","league_clean","_age_bucket"])["market_value_m"].median()
_mv_med_pos  = _valued.groupby(["position_group","_age_bucket"])["market_value_m"].median()

def _est_mv(r):
    if pd.notna(r["market_value_m"]):
        return np.nan
    k3 = (r["position_group"], r["league_clean"], r["_age_bucket"])
    if k3 in _mv_med_full.index:
        return round(float(_mv_med_full[k3]), 1)
    k2 = (r["position_group"], r["_age_bucket"])
    if k2 in _mv_med_pos.index:
        return round(float(_mv_med_pos[k2]), 1)
    return np.nan

DF["estimated_mv_m"] = DF.apply(_est_mv, axis=1)
print(f"  {len(DF)} players loaded")

SQUAD_CSV = ROOT / "data" / "al_ahly_squad.csv"
SQUAD = pd.read_csv(SQUAD_CSV)
SQUAD["market_value_m"] = pd.to_numeric(SQUAD["market_value_m"], errors="coerce")
SQUAD["age"]            = pd.to_numeric(SQUAD["age"],            errors="coerce")
SQUAD["is_foreign"]     = SQUAD["is_foreign"].astype(str).str.lower().isin(["true","1","yes"])
_TODAY = datetime.now()
SQUAD["contract_expiry_dt"] = pd.to_datetime(SQUAD["contract_expiry"], errors="coerce")
SQUAD["contract_expiring"]  = SQUAD["contract_expiry_dt"].apply(
    lambda d: pd.notna(d) and (d - _TODAY).days <= 365
)
print(f"  {len(SQUAD)} squad players loaded")

pkl_path = DATA_PROC / "similarity_matrices.pkl"
MATRICES = None
if pkl_path.exists():
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress sklearn version mismatch noise
            with open(pkl_path, "rb") as fh:
                MATRICES = pickle.load(fh)
        print("  Cached similarity matrices loaded")
    except Exception as e:
        print(f"  Cache load failed ({e}), rebuilding...")
        MATRICES = None
if MATRICES is None:
    print("  Building similarity matrices...")
    MATRICES = build_similarity_matrices(DF)
    try:
        DATA_PROC.mkdir(parents=True, exist_ok=True)
        with open(pkl_path, "wb") as fh:
            pickle.dump(MATRICES, fh)
        print("  Matrices saved")
    except Exception as e:
        print(f"  Could not save matrices: {e}")

LEAGUES_IN_DATA = sorted(DF["league_clean"].dropna().unique().tolist())

# ── Design System ─────────────────────────────────────────────────────────────

BG_DARK    = "#0D0D0D"
BG_CARD    = "#1A1A1A"
BG_CARD2   = "#222222"
BORDER     = "#2A2A2A"
RED        = "#CC0000"
GREEN      = "#00C853"
AMBER      = "#FFB300"
TEXT       = "#F5F5F5"
TEXT_MUTED = "#888888"
BLUE       = "#1565C0"

CHART_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=BG_CARD,
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT, family="Inter, system-ui"),
    margin=dict(l=40, r=20, t=50, b=40),
)

CARD = {
    "background": BG_CARD,
    "border":        f"1px solid {BORDER}",
    "borderRadius":  "10px",
    "padding":       "16px",
}

H = {"color": TEXT, "fontWeight": "600", "fontFamily": "Inter, system-ui"}

FONT = "Inter, system-ui, sans-serif"

TBL_HEADER = {
    "background": BG_CARD2, "color": TEXT,
    "fontWeight": "700", "border": f"1px solid {BORDER}",
    "fontFamily": FONT, "fontSize": "12px",
}
TBL_CELL = {
    "background": BG_CARD, "color": TEXT,
    "border": f"1px solid {BORDER}",
    "fontFamily": FONT, "fontSize": "12px", "padding": "8px 10px",
}

# ── Player filter definitions ─────────────────────────────────────────────────
# 3 simple boolean filters: each is a hard filter (removes non-qualifying players)
# when toggled ON. Missing data = player passes (benefit of the doubt).
FILTER_DEF_POSITIONS  = {"CB", "FB", "DM", "CM"}   # Defensively Strong
FILTER_GOAL_POSITIONS = {"ST", "W", "AM", "CM"}     # Goal Threat

# rgba equivalents for Plotly fillcolor (Plotly 6+ rejects 8-char hex)
RGBA_RED   = "rgba(204,0,0,0.2)"
RGBA_BLUE  = "rgba(21,101,192,0.2)"
RGBA_GREEN = "rgba(0,200,83,0.2)"
RGBA_AMBER = "rgba(255,179,0,0.2)"
RGBA_COLORS = [RGBA_RED, RGBA_BLUE, RGBA_GREEN, RGBA_AMBER]

RADAR_METRICS = {
    "CB": [("Tackling",     "tackle_success_rate"),
           ("Interceptions","interceptions_p90"),
           ("Prog Pass",    "progressive_passes_p90"),
           ("Clearances",   "clearances_p90"),
           ("Aerial",       "aerial_duels_won_pct")],
    "FB": [("Prog Carries", "progressive_carries_p90"),
           ("Key Passes",   "key_passes_p90"),
           ("Crosses",      "crosses_p90"),
           ("Tackling",     "tackles_p90"),
           ("Assists",      "assists_p90")],
    "DM": [("Tackling",     "tackles_p90"),
           ("Interceptions","interceptions_p90"),
           ("Prog Passes",  "progressive_passes_p90"),
           ("Pass Acc",     "pass_completion_rate"),
           ("Key Passes",   "key_passes_p90")],
    "CM": [("Prog Passes",  "progressive_passes_p90"),
           ("Key Passes",   "key_passes_p90"),
           ("Assists",      "assists_p90"),
           ("Tackling",     "tackles_p90"),
           ("Goals",        "goals_p90")],
    "AM": [("xAG",          "xag_p90"),
           ("Key Passes",   "key_passes_p90"),
           ("Assists",      "assists_p90"),
           ("Prog Carries", "progressive_carries_p90"),
           ("Goals",        "goals_p90")],
    "W":  [("Prog Carries", "progressive_carries_p90"),
           ("Goals",        "goals_p90"),
           ("xG",           "xg_p90"),
           ("Dribbles",     "dribbles_completed_p90"),
           ("Assists",      "assists_p90")],
    "ST": [("npxG",         "npxg_p90"),
           ("Goals",        "goals_p90"),
           ("Shot Acc",     "shot_accuracy"),
           ("Prog Rcvs",    "progressive_receives_p90"),
           ("Assists",      "assists_p90")],
    "GK": [("Save%",        "Save%")],
}

# ── Helper Functions ──────────────────────────────────────────────────────────

def pct_rank(value, series):
    """Percentile rank of value within series (0-100)."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty or pd.isna(value):
        return 50
    return round(float((clean < value).mean() * 100), 1)

def stat_coverage(r, pos):
    """Return (available, total) non-null stat counts for a player vs their position template."""
    metrics = RADAR_METRICS.get(pos, [])
    total   = len(metrics)
    avail   = sum(
        1 for _, col in metrics
        if col in DF.columns and pd.notna(pd.to_numeric(r.get(col), errors="coerce"))
    )
    return avail, total

def coverage_badge(avail, total):
    """Return a styled badge html element for data coverage."""
    label = f"({avail}/{total} stats)"
    if avail >= 4:
        return badge(f"Full Profile {label}", GREEN, GREEN + "22")
    if avail == 3:
        return badge(f"Good Profile {label}", AMBER, AMBER + "22")
    return badge(f"Limited Data {label}", RED, RED + "22")

def coverage_dots(avail, total):
    """Return filled/empty dot string e.g. '⬤⬤⬤○○'."""
    return "⬤" * avail + "○" * (total - avail)

def age_color(age):
    if age >= 30:
        return RED
    if age >= 27:
        return AMBER
    return GREEN

def val_badge_color(status):
    if status == "Undervalued":
        return GREEN
    if status == "Overvalued":
        return RED
    return TEXT_MUTED

def kpi_card(title, value, subtitle="", color=TEXT):
    return html.Div([
        html.Div(title, style={"color": TEXT_MUTED, "fontSize": "11px",
                               "fontWeight": "600", "textTransform": "uppercase",
                               "letterSpacing": "0.5px", "marginBottom": "4px",
                               "fontFamily": FONT}),
        html.Div(str(value), style={"color": color, "fontSize": "28px",
                                    "fontWeight": "700", "fontFamily": FONT,
                                    "lineHeight": "1.1"}),
        html.Div(subtitle, style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "marginTop": "2px", "fontFamily": FONT}),
    ], style={**CARD, "textAlign": "center"})

def badge(text, color=GREEN, bg=None):
    bg = bg or (color + "22")
    return html.Span(text, style={
        "background": bg, "color": color,
        "border": f"1px solid {color}",
        "borderRadius": "4px", "padding": "2px 8px",
        "fontSize": "11px", "fontWeight": "700",
        "fontFamily": FONT, "whiteSpace": "nowrap",
    })

_AGE_EMOJI = {
    "High Potential": "🌱",
    "Developing":     "📈",
    "Prime":          "⭐",
    "Experienced":    "✓",
    "Declining":      "⚠",
}

def age_stage_emoji(ctx):
    return _AGE_EMOJI.get(str(ctx), "")

# ── Formation Pitch ───────────────────────────────────────────────────────────

FORMATION_XI = [
    ("ST",  "Mohamed Sherif",      50,  8),
    ("LW",  "Trezeguet",           12, 18),
    ("RW",  "Zizo",                88, 18),
    ("CM",  "Emam Ashour",         32, 38),
    ("DM",  "Aliou Dieng",         50, 38),
    ("CM",  "Marwan Ateya",        68, 38),
    ("LB",  "Youssef Belammari",    8, 62),
    ("CB",  "Hady Reyad",          30, 62),
    ("CB",  "Beckham",             70, 62),
    ("RB",  "Mohamed Hany",        92, 62),
    ("GK",  "Mohamed El Shenawy", 50, 85),
]

def player_node(role, name, x_pct, y_pct, age, is_foreign, expiring):
    dot_color = age_color(age)
    short = name.split()[-1] if " " in name else name
    icons = ""
    if expiring:
        icons += " ⚡"
    if is_foreign:
        icons += " 🌍"
    initials = "".join(p[0].upper() for p in name.split()[:2]) if " " in name else name[:2].upper()
    return html.Div([
        html.Div(initials, style={
            "width": "36px", "height": "36px", "borderRadius": "50%",
            "background": dot_color, "border": "2px solid rgba(255,255,255,0.8)",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "margin": "0 auto", "fontSize": "10px", "fontWeight": "700",
            "color": "white", "fontFamily": FONT,
        }),
        html.Div(short + icons, style={
            "fontSize": "9px", "color": "white", "marginTop": "2px",
            "textShadow": "1px 1px 2px #000", "whiteSpace": "nowrap",
            "fontFamily": FONT, "fontWeight": "500",
        }),
        html.Div(str(int(age)), style={
            "fontSize": "9px", "color": "rgba(255,255,255,0.75)",
            "textShadow": "1px 1px 2px #000", "fontFamily": FONT,
        }),
    ], style={
        "position": "absolute",
        "left":      f"{x_pct}%",
        "top":       f"{y_pct}%",
        "transform": "translate(-50%, -50%)",
        "textAlign": "center",
        "zIndex":    "10",
        "minWidth":  "46px",
    })

def build_formation_pitch():
    squad_idx = {r["player"]: r for r in SQUAD.to_dict("records")}
    nodes = []
    for role, name, xp, yp in FORMATION_XI:
        p = squad_idx.get(name, {})
        age      = float(p.get("age", 28))
        foreign  = bool(p.get("is_foreign", False))
        expiring = bool(p.get("contract_expiring", False))
        nodes.append(player_node(role, name, xp, yp, age, foreign, expiring))

    pitch_lines = [
        # Halfway line
        html.Div(style={"position":"absolute","top":"50%","left":"0","width":"100%",
                        "height":"2px","background":"rgba(255,255,255,0.35)"}),
        # Center circle
        html.Div(style={"position":"absolute","top":"50%","left":"50%",
                        "transform":"translate(-50%,-50%)","width":"70px","height":"70px",
                        "borderRadius":"50%","border":"2px solid rgba(255,255,255,0.35)"}),
        # Center dot
        html.Div(style={"position":"absolute","top":"50%","left":"50%",
                        "transform":"translate(-50%,-50%)","width":"5px","height":"5px",
                        "borderRadius":"50%","background":"rgba(255,255,255,0.6)"}),
        # Penalty area top (attacking end)
        html.Div(style={"position":"absolute","top":"0","left":"22%","width":"56%","height":"16%",
                        "border":"2px solid rgba(255,255,255,0.35)","borderTop":"none",
                        "boxSizing":"border-box"}),
        # Penalty area bottom (defensive end)
        html.Div(style={"position":"absolute","bottom":"0","left":"22%","width":"56%","height":"16%",
                        "border":"2px solid rgba(255,255,255,0.35)","borderBottom":"none",
                        "boxSizing":"border-box"}),
        # Goal top
        html.Div(style={"position":"absolute","top":"0","left":"38%","width":"24%","height":"4%",
                        "border":"2px solid rgba(255,255,255,0.5)","borderTop":"none",
                        "background":"rgba(255,255,255,0.05)"}),
        # Goal bottom
        html.Div(style={"position":"absolute","bottom":"0","left":"38%","width":"24%","height":"4%",
                        "border":"2px solid rgba(255,255,255,0.5)","borderBottom":"none",
                        "background":"rgba(255,255,255,0.05)"}),
    ]

    legend = html.Div([
        html.Span("⬤ ", style={"color": GREEN, "fontSize": "10px"}),
        html.Span("<27  ", style={"color": TEXT_MUTED, "fontSize": "10px", "marginRight": "8px"}),
        html.Span("⬤ ", style={"color": AMBER, "fontSize": "10px"}),
        html.Span("27-29  ", style={"color": TEXT_MUTED, "fontSize": "10px", "marginRight": "8px"}),
        html.Span("⬤ ", style={"color": RED, "fontSize": "10px"}),
        html.Span("30+  ", style={"color": TEXT_MUTED, "fontSize": "10px", "marginRight": "8px"}),
        html.Span("⚡ ", style={"fontSize": "10px"}),
        html.Span("Expiring  ", style={"color": TEXT_MUTED, "fontSize": "10px", "marginRight": "8px"}),
        html.Span("🌍 ", style={"fontSize": "10px"}),
        html.Span("Foreign", style={"color": TEXT_MUTED, "fontSize": "10px"}),
    ], style={"marginTop": "8px", "textAlign": "center"})

    return html.Div([
        html.Div(pitch_lines + nodes, style={
            "position":     "relative",
            "background":   "#1a5c1a",
            "borderRadius": "6px",
            "height":       "480px",
            "width":        "100%",
            "overflow":     "hidden",
            "border":       "2px solid #2d7d2d",
        }),
        legend,
    ])

# ── Tab 1: Squad Intelligence ─────────────────────────────────────────────────

def build_gap_data():
    pos_counts = SQUAD["position_group"].value_counts().to_dict()
    rows = []
    for pos in ["GK", "CB", "FB", "DM", "CM", "W", "ST"]:
        cnt = pos_counts.get(pos, 0)
        grp = SQUAD[SQUAD["position_group"] == pos]
        avg_age = grp["age"].mean() if len(grp) > 0 else 0
        if cnt == 0:
            status = "🔴 EMPTY"
            action = f"Must sign {pos} immediately"
        elif cnt == 1:
            status = "🔴 CRITICAL"
            action = f"Sign {pos} backup — one injury = crisis"
        elif avg_age >= 29:
            status = "🟡 MONITOR"
            action = "Plan younger option before peak ages out"
        else:
            status = "🟢 OK"
            action = "Adequate depth and age profile"
        rows.append({"Pos": pos, "Count": cnt,
                     "Avg Age": f"{avg_age:.0f}" if cnt > 0 else "–",
                     "Status": status, "Recommendation": action})
    return rows

def build_priority_cards():
    pos_counts = SQUAD["position_group"].value_counts().to_dict()
    priority = []
    for pos in ["GK", "CB", "FB", "DM", "CM", "W", "ST"]:
        grp = SQUAD[SQUAD["position_group"] == pos]
        cnt = pos_counts.get(pos, 0)
        avg_age = grp["age"].mean() if len(grp) > 0 else 0
        score = 0
        reasons = []
        if cnt == 0:
            score += 30; reasons.append("no players")
        elif cnt == 1:
            score += 20; reasons.append("only 1 player")
        if avg_age >= 30:
            score += 15; reasons.append(f"avg age {avg_age:.0f}")
        elif avg_age >= 28:
            score += 8;  reasons.append(f"avg age {avg_age:.0f}")
        if score > 0:
            priority.append((pos, score, " · ".join(reasons)))
    priority.sort(key=lambda x: x[1], reverse=True)

    cards = []
    for pos, score, reason in priority[:3]:
        grp = SQUAD[SQUAD["position_group"] == pos]
        age_txt = " / ".join(
            f"{r['player'].split()[-1]} ({int(r['age'])})"
            for _, r in grp.iterrows()
        ) if len(grp) > 0 else "None"
        cards.append(html.Div([
            html.Div([
                html.Span(pos, style={
                    "background": RED, "color": "white",
                    "padding": "2px 10px", "borderRadius": "4px",
                    "fontWeight": "800", "fontSize": "12px",
                    "marginRight": "10px", "fontFamily": FONT,
                }),
                html.Span(reason, style={"color": TEXT_MUTED, "fontSize": "12px", "fontFamily": FONT}),
            ], style={"marginBottom": "4px"}),
            html.Div(f"Current: {age_txt}",
                     style={"fontSize": "11px", "color": TEXT_MUTED, "marginBottom": "8px",
                            "fontFamily": FONT}),
            html.Button(f"Find {pos} Targets →",
                id={"type": "find-targets-btn", "pos": pos},
                n_clicks=0,
                style={
                    "background": RED, "color": "white", "border": "none",
                    "borderRadius": "6px", "padding": "6px 14px",
                    "fontSize": "12px", "fontWeight": "600", "cursor": "pointer",
                    "fontFamily": FONT,
                }),
        ], style={**CARD, "background": BG_CARD2, "marginBottom": "8px"}))
    return cards

def build_tab1():
    sq = SQUAD
    n_foreign  = int(sq["is_foreign"].sum())
    n_expiring = int(sq["contract_expiring"].sum())
    n_age_risk = int(sq[sq["position_group"].isin(["CB","FB","DM","CM","W","ST"])]["age"]
                     .apply(lambda a: a >= 30).sum())
    total_val  = sq["market_value_m"].sum()

    if n_foreign == 5:
        fc_color = RED
    elif n_foreign == 4:
        fc_color = AMBER
    else:
        fc_color = GREEN

    return html.Div([
        # KPI Strip
        dbc.Row([
            dbc.Col(kpi_card("Squad Size", len(sq), "registered players"), md=True),
            dbc.Col(kpi_card("Foreign Players", f"{n_foreign} / 5 slots",
                             "CAF foreign player limit", color=fc_color), md=True),
            dbc.Col(kpi_card("Expiring Contracts", n_expiring,
                             "expire within 12 months",
                             color=AMBER if n_expiring > 0 else GREEN), md=True),
            dbc.Col(kpi_card("Age Risk", n_age_risk,
                             "outfield starters ≥ 30",
                             color=AMBER if n_age_risk > 3 else GREEN), md=True),
            dbc.Col(kpi_card("Squad Value", f"€{total_val:.1f}m",
                             "total market value"), md=True),
        ], className="g-3", style={"marginBottom": "16px"}),

        # Formation + Depth
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("⚽ Starting XI — 4-3-3", style={**H, "marginBottom": "12px", "fontSize": "15px"}),
                    build_formation_pitch(),
                ], style=CARD),
            ], md=7),
            dbc.Col([
                html.Div([
                    html.H5("📊 Position Depth", style={**H, "marginBottom": "12px", "fontSize": "15px"}),
                    dash_table.DataTable(
                        id="t1-gap-table",
                        data=build_gap_data(),
                        columns=[
                            {"name": c, "id": c}
                            for c in ["Pos", "Count", "Avg Age", "Status", "Recommendation"]
                        ],
                        style_table={"overflowX": "auto"},
                        style_header=TBL_HEADER,
                        style_cell={**TBL_CELL, "textAlign": "left"},
                        style_cell_conditional=[
                            {"if": {"column_id": "Pos"},    "width": "40px", "textAlign": "center", "fontWeight": "700"},
                            {"if": {"column_id": "Count"},  "width": "50px", "textAlign": "center"},
                            {"if": {"column_id": "Avg Age"},"width": "60px", "textAlign": "center"},
                        ],
                        style_data_conditional=[
                            {"if": {"filter_query": '{Status} contains "EMPTY" || {Status} contains "CRITICAL"'},
                             "color": RED},
                            {"if": {"filter_query": '{Status} contains "MONITOR"'}, "color": AMBER},
                            {"if": {"filter_query": '{Status} contains "OK"'},      "color": GREEN},
                        ],
                    ),
                ], style=CARD),
            ], md=5),
        ], className="g-3", style={"marginBottom": "16px"}),

        # Priority signings
        dbc.Row([dbc.Col([
            html.Div([
                html.H5("🎯 Priority Signings", style={**H, "marginBottom": "12px", "fontSize": "15px"}),
                html.P("Top 3 urgent positions. Click to launch targeted search.",
                       style={"color": TEXT_MUTED, "fontSize": "12px", "marginBottom": "12px",
                              "fontFamily": FONT}),
                *build_priority_cards(),
            ], style=CARD),
        ], md=12)]),
    ], style={"padding": "16px"})

# ── Tab 2: Player Discovery ───────────────────────────────────────────────────

TARGET_LEAGUE_ORDER = [
    "Super Lig",
    "Primeira Liga",
    "Eredivisie",
    "Superliga",
    "Belgian Pro League",
    "Serie A (Brazil)",
    "Primera Division",
    "Scottish Premiership",
]

BIG5_LEAGUE_ORDER = [
    "Premier League",
    "La Liga",
    "Bundesliga",
    "Serie A",
    "Ligue 1",
]

# Player counts per league — computed once at startup
_LEAGUE_COUNTS = DF["league_clean"].value_counts().to_dict()

SORT_OPTIONS = [
    {"label": "Adjusted Score (default)", "value": "adjusted_scouting_score"},
    {"label": "Raw Score",                "value": "raw_scouting_score"},
    {"label": "Market Value ↑",           "value": "mv_asc"},
    {"label": "Market Value ↓",           "value": "market_value_m"},
    {"label": "Age ↑",                    "value": "age"},
]

def build_tab2(filter_state=None):
    fs = filter_state or {}
    return html.Div([
        dbc.Row([
            # Sidebar
            dbc.Col([html.Div([
                html.Div("FILTERS", style={
                    "color": RED, "fontSize": "11px", "fontWeight": "700",
                    "letterSpacing": "1px", "marginBottom": "12px", "fontFamily": FONT,
                }),

                html.Label(["Position ", html.Span("*", style={"color": RED})],
                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "fontWeight": "700", "fontFamily": FONT,
                                  "marginBottom": "6px", "display": "block"}),
                dcc.RadioItems(
                    id="t2-pos",
                    options=[{"label": p, "value": p}
                             for p in ["CB","FB","DM","CM","AM","W","ST"]],
                    value=fs.get("pos"),
                    inputStyle={"display": "none"},
                    className="pos-pills",
                    style={"marginBottom": "14px"},
                ),

                # ── League selector — two groups ─────────────────────────
                html.Div([
                    html.Button("✓ All Leagues", id="t2-all-leagues-btn",
                                n_clicks=0,
                                style={"fontSize":"10px","padding":"2px 7px",
                                       "marginRight":"4px","cursor":"pointer",
                                       "background":"#2A2A2A","color":"#F5F5F5",
                                       "border":"1px solid #444","borderRadius":"3px"}),
                    html.Button("Target Only", id="t2-target-only-btn",
                                n_clicks=0,
                                style={"fontSize":"10px","padding":"2px 7px",
                                       "cursor":"pointer",
                                       "background":"#2A2A2A","color":"#FFB300",
                                       "border":"1px solid #444","borderRadius":"3px"}),
                ], style={"marginBottom":"6px"}),
                html.Div("Al Ahly Target Leagues",
                         style={"color": AMBER, "fontSize": "10px", "fontWeight": "700",
                                "letterSpacing": "0.5px", "textTransform": "uppercase",
                                "fontFamily": FONT, "marginBottom": "4px"}),
                dcc.Checklist(
                    id="t2-leagues-target",
                    options=[
                        {"label": f" {l} ({_LEAGUE_COUNTS.get(l, 0)})", "value": l}
                        for l in TARGET_LEAGUE_ORDER
                    ],
                    value=fs.get("leagues_target", list(TARGET_LEAGUE_ORDER)),
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"display": "block", "marginBottom": "3px",
                                "fontSize": "11px", "color": TEXT, "fontFamily": FONT},
                ),
                html.Div("Big 5 Leagues",
                         style={"color": TEXT_MUTED, "fontSize": "10px", "fontWeight": "700",
                                "letterSpacing": "0.5px", "textTransform": "uppercase",
                                "fontFamily": FONT, "marginTop": "8px", "marginBottom": "4px"}),
                dcc.Checklist(
                    id="t2-leagues-big5",
                    options=[
                        {"label": f" {l} ({_LEAGUE_COUNTS.get(l, 0)})", "value": l}
                        for l in BIG5_LEAGUE_ORDER
                    ],
                    value=fs.get("leagues_big5", []),
                    inputStyle={"marginRight": "5px"},
                    labelStyle={"display": "block", "marginBottom": "3px",
                                "fontSize": "11px", "color": TEXT_MUTED, "fontFamily": FONT},
                ),
                # Hidden combined store read by the filter callback
                dcc.Store(id="t2-leagues", data=list(TARGET_LEAGUE_ORDER)),

                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Label("Player Criteria",
                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "fontWeight": "700", "fontFamily": FONT,
                                  "marginBottom": "6px", "display": "block"}),

                html.Div(id="t2-age-lbl",
                         style={"color": AMBER, "fontSize": "11px", "fontFamily": FONT}),
                dcc.Slider(id="t2-age", min=16, max=35, value=fs.get("age", 28), step=1,
                           marks={16:"16", 20:"20", 25:"25", 28:"28", 35:"35"},
                           tooltip={"placement":"bottom"}),

                html.Div(id="t2-mv-lbl",
                         style={"color": AMBER, "fontSize": "11px",
                                "fontFamily": FONT, "marginTop": "4px"}),
                dcc.Slider(id="t2-mv", min=0, max=15, value=fs.get("mv", 3), step=0.5,
                           marks={0:"€0", 5:"€5m", 10:"€10m", 15:"€15m"},
                           tooltip={"placement":"bottom"}),

                html.Div(id="t2-mins-lbl",
                         style={"color": AMBER, "fontSize": "11px",
                                "fontFamily": FONT, "marginTop": "4px"}),
                dcc.Slider(id="t2-mins", min=900, max=3000, value=fs.get("mins", 900), step=100,
                           marks={900:"900", 1800:"1800", 3000:"3000"},
                           tooltip={"placement":"bottom"}),

                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Label("Contract Filter",
                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "fontWeight": "700", "fontFamily": FONT,
                                  "display": "block", "marginBottom": "4px"}),
                dcc.RadioItems(id="t2-contract",
                               options=[{"label": " All players",        "value": "all"},
                                        {"label": " Expiring ⚡ only",   "value": "expiring"}],
                               value=fs.get("contract", "all"),
                               inputStyle={"marginRight": "5px"},
                               labelStyle={"display": "block", "marginBottom": "3px",
                                           "fontSize": "12px", "color": TEXT, "fontFamily": FONT}),

                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Label("MV Data",
                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "fontWeight": "700", "fontFamily": FONT,
                                  "display": "block", "marginBottom": "4px"}),
                dcc.RadioItems(id="t2-mv-source",
                               options=[
                                   {"label": " All players",      "value": "all"},
                                   {"label": " Valued only",      "value": "valued"},
                                   {"label": " Estimated only",   "value": "estimated"},
                               ],
                               value=fs.get("mv_source", "all"),
                               inputStyle={"marginRight": "5px"},
                               labelStyle={"display": "block", "marginBottom": "3px",
                                           "fontSize": "12px", "color": TEXT, "fontFamily": FONT}),
                html.Div("★ = actual Transfermarkt value  ·  ~ est. = peer-group median estimate",
                         style={"color": TEXT_MUTED, "fontSize": "9px", "fontFamily": FONT,
                                "marginTop": "3px", "fontStyle": "italic"}),

                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Label("Sort By",
                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                  "fontWeight": "700", "fontFamily": FONT}),
                dcc.Dropdown(id="t2-sort", options=SORT_OPTIONS,
                             value=fs.get("sort", "adjusted_scouting_score"), clearable=False,
                             style={"marginBottom": "12px", "fontFamily": FONT, "fontSize": "12px"}),

                html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),
                html.Div([
                    html.Span("🎯 Player Filters",
                              style={"color": AMBER, "fontSize": "11px",
                                     "fontWeight": "700", "fontFamily": FONT}),
                    html.Div("Removes non-qualifying players · missing data = passes",
                             style={"color": TEXT_MUTED, "fontSize": "10px",
                                    "fontFamily": FONT, "marginBottom": "8px"}),
                    # Defensively Strong — CB, FB, DM, CM only
                    html.Div(id="t2-filter-def-wrap", style={"display": "none"}, children=[
                        dbc.Switch(id="t2-filter-def", label="⚔ Defensively Strong",
                                   value=fs.get("filter_def", False),
                                   style={"fontSize": "12px", "color": TEXT,
                                          "fontFamily": FONT, "marginBottom": "6px"}),
                        html.Div("tackle_success_rate ≥55th AND interceptions ≥50th",
                                 style={"color": TEXT_MUTED, "fontSize": "10px",
                                        "fontFamily": FONT, "marginBottom": "8px",
                                        "marginLeft": "12px"}),
                    ]),
                    # Goal Threat — ST, W, AM, CM only
                    html.Div(id="t2-filter-goal-wrap", style={"display": "none"}, children=[
                        dbc.Switch(id="t2-filter-goal", label="🎯 Goal Threat",
                                   value=fs.get("filter_goal", False),
                                   style={"fontSize": "12px", "color": TEXT,
                                          "fontFamily": FONT, "marginBottom": "6px"}),
                        html.Div("goals/90 ≥55th OR xG/90 ≥55th",
                                 style={"color": TEXT_MUTED, "fontSize": "10px",
                                        "fontFamily": FONT, "marginBottom": "8px",
                                        "marginLeft": "12px"}),
                    ]),
                    # Progressive — all positions
                    dbc.Switch(id="t2-filter-prog", label="🔄 Progressive / Creative",
                               value=fs.get("filter_prog", False),
                               style={"fontSize": "12px", "color": TEXT,
                                      "fontFamily": FONT, "marginBottom": "6px"}),
                    html.Div("prog carries ≥55th OR key passes ≥55th OR prog passes ≥55th",
                             style={"color": TEXT_MUTED, "fontSize": "10px",
                                    "fontFamily": FONT, "marginBottom": "4px",
                                    "marginLeft": "12px"}),
                ]),

                html.Div(style={"height": "12px"}),
                html.Button("Apply Filters", id="t2-apply", n_clicks=0,
                            style={
                                "width": "100%", "background": RED, "color": "white",
                                "border": "none", "borderRadius": "6px",
                                "padding": "9px 0", "fontWeight": "700",
                                "cursor": "pointer", "fontFamily": FONT,
                                "marginBottom": "6px", "fontSize": "13px",
                            }),
                html.Button("Reset", id="t2-reset", n_clicks=0,
                            style={
                                "width": "100%", "background": BG_CARD2, "color": TEXT_MUTED,
                                "border": f"1px solid {BORDER}", "borderRadius": "6px",
                                "padding": "7px 0", "cursor": "pointer",
                                "fontFamily": FONT, "fontSize": "12px",
                            }),
            ], style={**CARD, "overflowY": "auto", "maxHeight": "90vh",
                      "position": "sticky", "top": "0"})], md=3),

            # Main panel
            dbc.Col([html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id="t2-hdr",
                                     children=html.P(
                                         "← Select a position to begin scouting",
                                         style={"color": TEXT_MUTED, "margin": "0",
                                                "fontFamily": FONT}
                                     ))),
                    dbc.Col([
                        html.Button("↓ Export CSV", id="t2-export-btn", n_clicks=0,
                                    style={
                                        "background": BG_CARD2, "color": TEXT_MUTED,
                                        "border": f"1px solid {BORDER}",
                                        "borderRadius": "6px", "padding": "6px 12px",
                                        "cursor": "pointer", "fontFamily": FONT,
                                        "fontSize": "11px", "marginRight": "6px",
                                    }),
                        dcc.Download(id="t2-dl"),
                    ], width="auto"),
                ], align="center", className="mb-3"),

                html.Div(id="t2-shortlist-msg",
                         style={"color": GREEN, "fontSize": "12px",
                                "marginBottom": "8px", "fontFamily": FONT}),

                html.Div(id="t2-tbl-wrap",
                         children=html.P(
                             "Select a position to load results.",
                             style={"color": TEXT_MUTED, "textAlign": "center",
                                    "padding": "60px 0", "fontFamily": FONT}
                         )),
            ], style=CARD)], md=9),
        ]),
    ], style={"padding": "16px"})

# ── Tab 3: Player Profile ─────────────────────────────────────────────────────

def build_tab3():
    return html.Div([
        html.Div(id="t3-body", children=[
            html.P("Click a player row in Player Discovery to view their full profile.",
                   style={"color": TEXT_MUTED, "textAlign": "center",
                          "padding": "80px 0", "fontSize": "15px",
                          "fontFamily": FONT}),
        ]),
    ], style={"padding": "16px"})

# ── Tab 4: Shortlist ──────────────────────────────────────────────────────────

def build_tab4():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Row([
                        dbc.Col(html.H5("⭐ My Shortlist",
                                        style={**H, "fontSize": "15px", "margin": "0"})),
                        dbc.Col([
                            html.Span(id="t4-count",
                                      style={"color": TEXT_MUTED, "fontSize": "13px",
                                             "fontFamily": FONT, "marginRight": "12px"}),
                            html.Button("Clear All", id="t4-clear", n_clicks=0,
                                        style={
                                            "background": "transparent", "color": RED,
                                            "border": f"1px solid {RED}",
                                            "borderRadius": "6px", "padding": "4px 12px",
                                            "cursor": "pointer", "fontFamily": FONT,
                                            "fontSize": "12px",
                                        }),
                        ], width="auto"),
                    ], align="center", className="mb-3"),
                    html.Div(id="t4-table-wrap"),
                ], style=CARD),
            ], md=12),
        ], className="g-3", style={"marginBottom": "16px"}),

        dbc.Row([dbc.Col([
            html.Div(id="t4-comparison-panel", style=CARD),
        ], md=12)]),
    ], style={"padding": "16px"})

# ── Tab 5: Replacement Finder ─────────────────────────────────────────────────

def build_tab5():
    all_opts = [
        {"label": f"{r['player']}  ({r['team']}, {r['position_group']})", "value": r["player"]}
        for _, r in DF.sort_values("adjusted_scouting_score", ascending=False).iterrows()
    ]
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5("🔄 Reference Player",
                            style={**H, "marginBottom": "12px", "fontSize": "15px"}),
                    html.P("Find a cheaper, younger version of any player",
                           style={"color": TEXT_MUTED, "fontSize": "12px",
                                  "fontFamily": FONT, "marginBottom": "10px"}),
                    dcc.Dropdown(id="t5-player", options=all_opts,
                                 placeholder="Search any player…",
                                 optionHeight=45, style={"fontFamily": FONT}),
                    html.P(
                        f"Searches the 13-league scouting database ({len(DF):,} players). "
                        "Al Ahly squad not included — use Squad Intelligence to scout current squad players.",
                        style={"color": TEXT_MUTED, "fontSize": "10px", "fontFamily": FONT,
                               "marginTop": "5px", "marginBottom": "0", "fontStyle": "italic"},
                    ),
                    html.Div(id="t5-mini", style={"marginTop": "10px"}),
                ], style={**CARD, "marginBottom": "12px"}),

                html.Div([
                    html.H5("🎯 Criteria",
                            style={**H, "marginBottom": "12px", "fontSize": "15px"}),
                    html.Label("Target Budget",
                               style={"color": TEXT_MUTED, "fontSize": "11px",
                                      "fontWeight": "700", "fontFamily": FONT}),
                    html.Div(id="t5-bgt-lbl",
                             style={"color": AMBER, "fontSize": "11px", "fontFamily": FONT}),
                    dcc.Slider(id="t5-budget", min=0, max=15, value=3, step=0.5,
                               marks={0:"€0", 5:"€5m", 10:"€10m", 15:"€15m"},
                               tooltip={"placement":"bottom"}),
                    html.Label("Max Age",
                               style={"color": TEXT_MUTED, "fontSize": "11px",
                                      "fontWeight": "700", "fontFamily": FONT,
                                      "marginTop": "10px", "display": "block"}),
                    dcc.Slider(id="t5-age", min=16, max=35, value=28, step=1,
                               marks={16:"16", 20:"20", 25:"25", 28:"28", 35:"35"},
                               tooltip={"placement":"bottom"}),
                    dbc.Row([
                        dbc.Col(html.Label("Different League",
                                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                                  "fontWeight": "700", "fontFamily": FONT}),
                                width=8),
                        dbc.Col(dbc.Switch(id="t5-diff-league", value=True,
                                           label="", style={"marginTop": "4px"}), width=4),
                    ], align="center", style={"marginTop": "10px"}),
                    dbc.Row([
                        dbc.Col(html.Label("Target Leagues Only",
                                           style={"color": TEXT_MUTED, "fontSize": "11px",
                                                  "fontWeight": "700", "fontFamily": FONT}),
                                width=8),
                        dbc.Col(dbc.Switch(id="t5-tgt-leagues", value=True,
                                           label="", style={"marginTop": "4px"}), width=4),
                    ], align="center", style={"marginTop": "6px"}),
                    html.Div(style={"height": "12px"}),
                    html.Button("Find Replacements", id="t5-find", n_clicks=0,
                                style={
                                    "width": "100%", "background": RED, "color": "white",
                                    "border": "none", "borderRadius": "6px",
                                    "padding": "9px 0", "fontWeight": "700",
                                    "cursor": "pointer", "fontFamily": FONT,
                                    "fontSize": "13px",
                                }),
                ], style=CARD),
            ], md=4),

            dbc.Col([
                html.Div(id="t5-results", children=[
                    html.P("Select a reference player and click Find Replacements",
                           style={"color": TEXT_MUTED, "textAlign": "center",
                                  "padding": "60px 0", "fontFamily": FONT}),
                ], style=CARD),
            ], md=8),
        ]),
    ], style={"padding": "16px"})

# ── Tab 6: Market Intelligence ────────────────────────────────────────────────

def build_scatter_fig():
    """Build the Al Ahly Target Zone scatter chart. Called at layout time so the
    graph has a real figure from the moment it enters the DOM — avoids the race
    condition where the callback fires before the component exists."""
    df = DF[(DF["market_value_m"] <= 5.0) & (DF["age"].between(18, 35))].copy()
    df = df.dropna(subset=["age", "adjusted_scouting_score"])

    DARK = "#1A1A1A"
    GRID = "#2A2A2A"
    pos_color_map = {
        "CB": BLUE,       "FB": "#00BCD4", "DM": "#9C27B0",
        "CM": AMBER,      "AM": "#FF5722", "W":  GREEN,    "ST": RED,
    }

    fig = go.Figure()

    if not df.empty:
        # Target zone highlight
        fig.add_shape(
            type="rect", x0=22, x1=28, y0=60, y1=105,
            fillcolor="rgba(0,200,83,0.10)",
            line=dict(color="rgba(0,200,83,0.40)", width=1),
            layer="below",
        )
        fig.add_annotation(
            x=25, y=102, text="Al Ahly Target Zone",
            font=dict(color=GREEN, size=10, family=FONT),
            showarrow=False,
            bgcolor="rgba(34,34,34,0.80)",
        )

        for pos, color in pos_color_map.items():
            sub = df[df["position_group"] == pos]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["age"],
                y=sub["adjusted_scouting_score"],
                mode="markers",
                marker=dict(
                    color=color,
                    size=(sub["market_value_m"].fillna(0.5) * 4).clip(4, 18),
                    opacity=0.80,
                    line=dict(width=0),
                ),
                name=pos,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]}<br>"
                    "Age: %{x:.0f}  ·  Score: %{y:.1f}<br>"
                    "MV: €%{customdata[2]:.2f}m<extra></extra>"
                ),
                customdata=list(zip(
                    sub["player"].fillna(""),
                    sub["team"].fillna(""),
                    sub["market_value_m"].fillna(0),
                )),
            ))

    fig.update_layout(
        paper_bgcolor=DARK,
        plot_bgcolor=DARK,
        font=dict(color=TEXT, family=FONT),
        margin=dict(l=44, r=16, t=16, b=60),
        xaxis=dict(
            title="Age", gridcolor=GRID, zeroline=False,
            range=[18, 36], color=TEXT, tickfont=dict(color=TEXT_MUTED),
        ),
        yaxis=dict(
            title="Adjusted Score", gridcolor=GRID, zeroline=False,
            color=TEXT, tickfont=dict(color=TEXT_MUTED),
        ),
        legend=dict(
            orientation="h", y=-0.22,
            font=dict(size=10, color=TEXT),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=330,
    )
    return fig


def build_tab6():
    return html.Div([
        dbc.Row([
            dbc.Col([html.Div([
                html.H5("💎 Best Value Under €3m",
                        style={**H, "marginBottom": "8px", "fontSize": "14px"}),
                dcc.Graph(id="t6-best-value",
                          config={"displayModeBar": False},
                          style={"height": "340px"}),
            ], style=CARD)], md=6),
            dbc.Col([html.Div([
                html.H5("📊 Top Scoring Players by League",
                        style={**H, "marginBottom": "8px", "fontSize": "14px"}),
                dcc.Graph(id="t6-undervalued",
                          config={"displayModeBar": False},
                          style={"height": "340px"}),
            ], style=CARD)], md=6),
        ], className="g-3", style={"marginBottom": "16px"}),

        dbc.Row([
            dbc.Col([html.Div([
                html.Div([
                    html.H5("⚡ Contract Expiring Targets",
                            style={**H, "fontSize": "14px", "margin": "0",
                                   "display": "inline-block"}),
                    dcc.Dropdown(id="t6-pos-filter",
                                 options=[{"label": "All Positions", "value": "ALL"}] +
                                         [{"label": p, "value": p}
                                          for p in ["CB","FB","DM","CM","AM","W","ST"]],
                                 value="ALL", clearable=False,
                                 style={"width": "160px", "display": "inline-block",
                                        "marginLeft": "16px", "verticalAlign": "middle",
                                        "fontFamily": FONT, "fontSize": "12px"}),
                ], style={"marginBottom": "10px"}),
                html.Div(id="t6-expiry-table"),
            ], style=CARD)], md=6),
            dbc.Col([html.Div([
                html.H5("🎯 Al Ahly Target Zone",
                        style={**H, "marginBottom": "8px", "fontSize": "14px"}),
                html.P("Age 22-28 · Score >60 · Budget ≤€5m",
                       style={"color": TEXT_MUTED, "fontSize": "11px",
                              "marginBottom": "4px", "fontFamily": FONT}),
                dcc.Graph(id="t6-scatter",
                          figure=build_scatter_fig(),
                          config={"displayModeBar": False},
                          style={"height": "340px"}),
            ], style=CARD)], md=6),
        ], className="g-3"),
    ], style={"padding": "16px"})

# ── App Init + Layout ─────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="ScoutEdge v2",
)
server = app.server

TAB_STYLE = {
    "background": BG_DARK, "color": TEXT_MUTED,
    "borderTop": "none", "borderLeft": "none", "borderRight": "none",
    "borderBottom": f"1px solid {BORDER}",
    "padding": "10px 20px", "fontFamily": FONT,
    "fontSize": "13px", "fontWeight": "500",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": TEXT, "borderBottom": f"3px solid {RED}",
    "background": BG_DARK,
}

app.layout = html.Div([
    dcc.Store(id="shortlist-store",   storage_type="session", data=[]),
    dcc.Store(id="selected-player",   data=None),
    dcc.Store(id="nav-pos-store",     data=None),
    dcc.Store(id="t2-filter-store",   storage_type="session", data=None),

    # Header
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Span("⚽ ", style={"color": RED, "fontSize": "22px"}),
                html.Span("ScoutEdge v2",
                          style={"color": RED, "fontWeight": "700",
                                 "fontSize": "18px", "fontFamily": FONT}),
                html.Span(" · Al Ahly SC · Recruitment Intelligence",
                          style={"color": TEXT_MUTED, "fontSize": "13px",
                                 "fontFamily": FONT, "marginLeft": "6px"}),
            ]),
            dbc.Col(
                html.Div(f"{len(DF):,} players · {len(LEAGUES_IN_DATA)} leagues · {int(DF['market_value_m'].notna().sum()):,} valued",
                         style={"color": TEXT_MUTED, "fontSize": "12px",
                                "fontFamily": FONT, "textAlign": "right"}),
                width="auto",
            ),
        ], align="center"),
    ], style={
        "background": BG_DARK,
        "borderBottom": f"3px solid {RED}",
        "padding": "12px 24px",
    }),

    # Tabs
    dcc.Tabs(
        id="main-tabs", value="tab-squad",
        style={"background": BG_DARK},
        children=[
            dcc.Tab(label="Squad Intelligence",  value="tab-squad",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Player Discovery",    value="tab-discovery",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Player Profile",      value="tab-profile",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(id="tab-shortlist-label",
                    label="Shortlist (0)",       value="tab-shortlist",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Replacement Finder",  value="tab-replacement",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
            dcc.Tab(label="Market Intelligence", value="tab-market",
                    style=TAB_STYLE, selected_style=TAB_SELECTED),
        ],
    ),

    # Tab content
    html.Div(id="tab-content",
             style={"background": BG_DARK, "minHeight": "calc(100vh - 100px)"}),

], style={"fontFamily": FONT, "background": BG_DARK, "minHeight": "100vh"})

# ── Callbacks ─────────────────────────────────────────────────────────────────

# ── Profile content helper (shared by render_tab and render_profile) ──────────

def _profile_content(player_name, shortlist):
    """Build the full player profile card content.  Called from both render_tab
    (when the tab first opens) and render_profile (when selected-player changes
    while already on the profile tab) so there is no race condition between the
    two callbacks fighting over t3-body."""
    if not player_name:
        return html.P("Click a player row in Player Discovery to view their full profile.",
                      style={"color": TEXT_MUTED, "textAlign": "center",
                             "padding": "80px 0", "fontSize": "15px",
                             "fontFamily": FONT})

    pr = DF[DF["player"] == player_name]
    if pr.empty:
        return html.P(f"Player '{player_name}' not found in database.",
                      style={"color": TEXT_MUTED, "textAlign": "center",
                             "padding": "80px 0", "fontFamily": FONT})
    r   = pr.iloc[0]
    pos = str(r.get("position_group", ""))
    mv  = r.get("market_value_m")
    exp = bool(r.get("contract_expiring", False))

    shortlist_names = [p["player"] for p in (shortlist or [])]
    star_label = "★ In Shortlist" if player_name in shortlist_names else "☆ Add to Shortlist"

    avail, total = stat_coverage(r, pos)

    mv_est = r.get("estimated_mv_m")
    if pd.notna(mv):
        mv_badge = badge(f"€{mv:.2f}m", BLUE, BLUE + "22")
    elif pd.notna(mv_est):
        mv_badge = badge(f"~€{mv_est:.1f}m est.", TEXT_MUTED, TEXT_MUTED + "22")
    else:
        mv_badge = badge("–", TEXT_MUTED, TEXT_MUTED + "22")
    exp_badge  = badge("⚡ Expiring", AMBER, AMBER + "22") if exp else None
    cov_badge  = coverage_badge(avail, total)
    age_ctx    = str(r.get("age_score_context", ""))
    age_emoji  = age_stage_emoji(age_ctx)
    age_ctx_badge = (badge(f"{age_emoji} {age_ctx}", TEXT_MUTED, TEXT_MUTED + "22")
                     if age_ctx else None)
    badges     = [mv_badge, html.Span(" "), cov_badge]
    if age_ctx_badge:
        badges += [html.Span(" "), age_ctx_badge]
    if exp_badge:
        badges += [html.Span(" "), exp_badge]

    header = html.Div([
        dbc.Row([
            dbc.Col([
                html.H3(player_name, style={"color": TEXT, "fontWeight": "700",
                                            "fontFamily": FONT, "marginBottom": "4px"}),
                html.Div([
                    html.Span(f"{r.get('team', '–')}  ·  ", style={"color": TEXT_MUTED, "fontFamily": FONT}),
                    html.Span(f"{r.get('league_clean', '–')}  ·  ", style={"color": TEXT_MUTED, "fontFamily": FONT}),
                    html.Span(f"Age {int(r['age']) if pd.notna(r['age']) else '–'}  ·  ", style={"color": TEXT_MUTED, "fontFamily": FONT}),
                    html.Span(re.sub(r"^[a-z]{2,3} ", "",
                                     str(r.get("nationality") or "–") if pd.notna(r.get("nationality")) else "–"),
                              style={"color": TEXT_MUTED, "fontFamily": FONT}),
                ], style={"marginBottom": "8px"}),
                html.Div(badges),
            ]),
            dbc.Col([
                html.Button(star_label, id="t3-shortlist-btn",
                            **{"data-player": player_name},
                            n_clicks=0,
                            style={
                                "background": BG_CARD2, "color": AMBER,
                                "border": f"1px solid {AMBER}",
                                "borderRadius": "6px", "padding": "8px 16px",
                                "cursor": "pointer", "fontFamily": FONT,
                                "fontWeight": "600", "fontSize": "13px",
                            }),
            ], width="auto", style={"display": "flex", "alignItems": "center"}),
        ], align="center"),
    ], style={**CARD, "marginBottom": "16px"})

    metrics      = RADAR_METRICS.get(pos, [])
    pos_df       = DF[DF["position_group"] == pos]
    n_leagues    = DF["league_clean"].nunique()
    player_league = str(r.get("league_clean", ""))

    stat_bars = []
    for label, col in metrics:
        # Always render every stat in the position template — show "no data" when missing
        has_data = (col in DF.columns and col in r.index and
                    pd.notna(pd.to_numeric(r.get(col), errors="coerce")))

        if has_data:
            val_num   = pd.to_numeric(r[col], errors="coerce")
            pct       = pct_rank(val_num, pos_df[col])
            bar_color = (RED + "88" if pct <= 33 else AMBER + "88" if pct <= 66 else GREEN + "88")
            stat_bars.append(html.Div([
                dbc.Row([
                    dbc.Col(html.Div(label, style={"color": TEXT, "fontSize": "12px",
                                                   "fontFamily": FONT, "fontWeight": "500"}), width=4),
                    dbc.Col(html.Div(f"{val_num:.2f}", style={"color": TEXT_MUTED, "fontSize": "11px",
                                                               "fontFamily": FONT, "textAlign": "right"}), width=2),
                    dbc.Col(html.Div([html.Div(style={"width": f"{pct}%", "height": "8px",
                                                       "background": bar_color, "borderRadius": "4px",
                                                       "transition": "width 0.3s"})],
                                     style={"background": BORDER, "borderRadius": "4px", "overflow": "hidden"}), width=4),
                    dbc.Col(html.Div(f"Top {100-int(pct)}%", style={"color": TEXT_MUTED, "fontSize": "10px",
                                                                      "fontFamily": FONT, "textAlign": "right"}), width=2),
                ], align="center", className="mb-1"),
            ]))
        else:
            # No data for this stat in this player's league — show placeholder row
            stat_bars.append(html.Div([
                dbc.Row([
                    dbc.Col(html.Div(label, style={"color": TEXT_MUTED, "fontSize": "12px",
                                                   "fontFamily": FONT, "fontWeight": "500"}), width=4),
                    dbc.Col(html.Div("–", style={"color": TEXT_MUTED, "fontSize": "11px",
                                                  "fontFamily": FONT, "textAlign": "right"}), width=2),
                    dbc.Col(html.Div([],  # empty grey track
                                     style={"background": BORDER, "borderRadius": "4px",
                                            "height": "8px", "overflow": "hidden"}), width=4),
                    dbc.Col(html.Div("no data", style={"color": TEXT_MUTED, "fontSize": "10px",
                                                        "fontFamily": FONT, "textAlign": "right",
                                                        "fontStyle": "italic"}), width=2),
                ], align="center", className="mb-1"),
            ]))

    peer_subtitle = f"vs all {pos} peers  ·  {n_leagues} leagues"
    stat_panel = html.Div([
        html.H6("Performance Profile", style={"color": TEXT, "fontWeight": "600",
                                               "fontFamily": FONT, "marginBottom": "2px"}),
        html.Div(peer_subtitle,
                 style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT, "marginBottom": "12px"}),
        *stat_bars,
    ], style=CARD)

    radar_metrics = RADAR_METRICS.get(pos, [])
    # Only plot axes where this player actually has data — avoids misleading
    # "Top 50%" fallback for stats not measured in their league.
    r_labels, r_values = [], []
    for lbl, col in radar_metrics:
        if col not in DF.columns:
            continue
        val_num = pd.to_numeric(r.get(col), errors="coerce")
        if pd.isna(val_num):
            continue
        r_labels.append(lbl)
        r_values.append(pct_rank(val_num, pos_df[col]))

    if len(r_labels) >= 3:
        fig_radar = go.Figure()
        closed_r     = r_values + [r_values[0]]
        closed_theta = r_labels + [r_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=closed_r,
            theta=closed_theta,
            fill="toself",
            fillcolor="rgba(204,0,0,0.2)",
            line=dict(color=RED, width=2),
            name=player_name,
            hovertemplate="<b>%{theta}</b><br>Percentile: Top %{r:.0f}%<extra></extra>",
        ))
        fig_radar.update_layout(
            **CHART_DEFAULTS,
            polar=dict(bgcolor=BG_CARD2,
                       radialaxis=dict(visible=True, range=[0, 100],
                                       tickfont=dict(size=8, color=TEXT_MUTED), gridcolor=BORDER),
                       angularaxis=dict(tickfont=dict(size=10, color=TEXT))),
            showlegend=True,
            legend=dict(x=0, y=1.1, font=dict(size=10, color=TEXT)),
            height=320,
            hoverlabel=dict(bgcolor="#1A1A1A", font_size=13, font_color="#F5F5F5"),
        )
        radar_panel = html.Div([
            html.H6("Radar Chart", style={"color": TEXT, "fontWeight": "600",
                                           "fontFamily": FONT, "marginBottom": "2px"}),
            html.Div(f"Percentile vs all {pos} peers  ·  available stats only",
                     style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT, "marginBottom": "8px"}),
            dcc.Graph(figure=fig_radar, config={"displayModeBar": False}),
        ], style=CARD)
    else:
        available_names = " · ".join(r_labels) if r_labels else "none"
        radar_panel = html.Div([
            html.H6("Radar Chart", style={"color": TEXT, "fontWeight": "600",
                                           "fontFamily": FONT, "marginBottom": "10px"}),
            html.Div([
                html.Div("⚠ Limited data available for this league",
                         style={"color": AMBER, "fontWeight": "700", "fontSize": "13px",
                                "fontFamily": FONT, "marginBottom": "8px"}),
                html.Div(f"Available: {available_names}",
                         style={"color": TEXT_MUTED, "fontSize": "12px",
                                "fontFamily": FONT, "marginBottom": "6px"}),
                html.Div("Full radar requires 3+ stats · only Big 5 leagues report all metrics via FBref",
                         style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT}),
            ], style={"background": BG_CARD2, "borderRadius": "8px", "padding": "20px",
                      "border": f"1px solid {AMBER}33", "marginTop": "8px"}),
            html.P(
                f"Advanced stats for {player_league} require Opta or StatsBomb data. "
                f"Contact us to discuss a data partnership.",
                style={"color": TEXT_MUTED, "fontSize": "11px", "fontStyle": "italic",
                       "fontFamily": FONT, "marginTop": "10px", "marginBottom": "0"},
            ),
        ], style=CARD)

    ahly_at_pos = SQUAD[SQUAD["position_group"] == pos]
    if len(ahly_at_pos) > 0:
        ahly_p    = ahly_at_pos.iloc[0]
        ahly_name = ahly_p["player"]
        ahly_mv   = ahly_p.get("market_value_m")
        target_score = float(r["adjusted_scouting_score"]) if pd.notna(r["adjusted_scouting_score"]) else 0
        verdict_txt, verdict_color = (
            ("POTENTIAL UPGRADE", GREEN) if target_score >= 70 else
            ("SIMILAR LEVEL",    AMBER)  if target_score >= 50 else
            ("BELOW THRESHOLD",  RED)
        )
        # Build stats string for a player: "X goals, Y assists (24/25)"
        def _stats_str(g, a):
            if pd.notna(g) and pd.notna(a):
                return f"{int(g)}g, {int(a)}a (24/25)"
            return "–"

        comp_rows = []
        for lbl, col in [("Stats (24/25)", "adjusted_scouting_score"), ("Age", "age"), ("Value", "market_value_m")]:
            tgt_val  = r.get(col)
            ahly_val = ahly_p.get(col)
            if col == "market_value_m":
                tgt_est = r.get("estimated_mv_m")
                if pd.notna(tgt_val):
                    tgt_str = f"€{tgt_val:.2f}m"
                elif pd.notna(tgt_est):
                    tgt_str = f"~€{tgt_est:.1f}m est."
                else:
                    tgt_str = "–"
                ahly_str = f"€{ahly_val:.2f}m" if pd.notna(ahly_val) else "–"
            elif col == "adjusted_scouting_score":
                # Target: show their season goals/assists from DF
                tgt_g = pd.to_numeric(r.get("goals"),   errors="coerce")
                tgt_a = pd.to_numeric(r.get("assists"), errors="coerce")
                tgt_str = _stats_str(tgt_g, tgt_a)
                # Al Ahly: show their 24/25 stats from squad CSV
                ah_g = pd.to_numeric(ahly_p.get("goals_2425"),   errors="coerce")
                ah_a = pd.to_numeric(ahly_p.get("assists_2425"), errors="coerce")
                ahly_str = _stats_str(ah_g, ah_a)
            else:
                tgt_str  = str(int(tgt_val))  if pd.notna(tgt_val)  else "–"
                ahly_str = str(int(ahly_val)) if pd.notna(ahly_val) else "–"
            comp_rows.append({"Metric": lbl, "Target": tgt_str, "Al Ahly": ahly_str})
        comp_tbl = dash_table.DataTable(
            data=comp_rows,
            columns=[{"name": c, "id": c} for c in ["Metric", "Target", "Al Ahly"]],
            style_header=TBL_HEADER,
            style_cell={**TBL_CELL, "fontSize": "12px", "textAlign": "center"},
        )
        vs_panel = html.Div([
            html.H6(f"{player_name} vs {ahly_name}",
                    style={"color": TEXT, "fontWeight": "600", "fontFamily": FONT, "marginBottom": "8px"}),
            comp_tbl,
            html.Div(style={"height": "10px"}),
            html.Div([badge(verdict_txt, verdict_color, verdict_color + "22")],
                     style={"textAlign": "center"}),
        ], style=CARD)
    else:
        vs_panel = html.Div([
            html.H6("vs Al Ahly", style={"color": TEXT, "fontWeight": "600",
                                         "fontFamily": FONT, "marginBottom": "8px"}),
            html.P(f"No current Al Ahly player at {pos}.",
                   style={"color": TEXT_MUTED, "fontSize": "12px", "fontFamily": FONT}),
            html.P("This would be a new addition to the squad.",
                   style={"color": TEXT_MUTED, "fontSize": "12px", "fontFamily": FONT}),
        ], style=CARD)

    row1 = dbc.Row([
        dbc.Col(stat_panel, md=4), dbc.Col(radar_panel, md=4), dbc.Col(vs_panel, md=4),
    ], className="g-3", style={"marginBottom": "16px"})

    try:
        sim_df = get_similar_players(player_name, DF, MATRICES, n=8, target_leagues_only=True)
    except Exception:
        sim_df = pd.DataFrame()

    if sim_df.empty:
        sim_section = html.P("No similar players found.", style={"color": TEXT_MUTED, "fontFamily": FONT})
    else:
        sim_data = []
        for _, sr in sim_df.iterrows():
            smv = sr.get("market_value_m")
            _full = DF[DF["player"] == sr["player"]]
            r_avail, r_total = (stat_coverage(_full.iloc[0], str(sr.get("position_group", pos)))
                                if not _full.empty else (0, len(RADAR_METRICS.get(pos, []))))
            sim_data.append({
                "★": "☆", "Player": sr["player"], "Club": sr.get("team", ""),
                "League": sr.get("league_clean", ""),
                "Age":    int(sr["age"]) if pd.notna(sr["age"]) else "–",
                "MV (€m)": f"€{smv:.2f}m" if pd.notna(smv) else "–",
                "Sim%":   f"{sr['similarity_pct']:.0f}% ({r_avail}/{r_total})",
                "Score":  round(float(sr["adjusted_scouting_score"]), 1) if pd.notna(sr.get("adjusted_scouting_score")) else "–",
                "_avail": r_avail,
            })
        sim_section = dash_table.DataTable(
            data=sim_data,
            columns=[{"name": c, "id": c} for c in ["★", "Player", "Club", "League", "Age", "MV (€m)", "Sim%", "Score", "_avail"]],
            hidden_columns=["_avail"],
            style_header=TBL_HEADER,
            style_cell={**TBL_CELL, "fontSize": "12px", "textAlign": "left"},
            style_cell_conditional=[
                {"if": {"column_id": "★"},    "width": "28px", "textAlign": "center",
                 "color": AMBER, "cursor": "pointer"},
                {"if": {"column_id": "Sim%"}, "textAlign": "center", "fontWeight": "700"},
            ],
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "background": BG_CARD2},
                {"if": {"filter_query": "{_avail} < 4", "column_id": "Sim%"},
                 "color": TEXT_MUTED, "fontStyle": "italic", "fontWeight": "400"},
            ],
        )

    sim_note = html.P(
        f"Similarity based on {avail} available stat{'s' if avail != 1 else ''} · "
        f"Sim% shows (stats used / position total) · grey = fewer than 4 stats, treat with caution.",
        style={"color": TEXT_MUTED, "fontSize": "11px", "fontStyle": "italic",
               "fontFamily": FONT, "marginTop": "8px", "marginBottom": "0"},
    )

    row2 = html.Div([
        dbc.Row([
            dbc.Col(html.H6("Most Similar Players",
                            style={"color": TEXT, "fontWeight": "600", "fontFamily": FONT,
                                   "marginBottom": "0"}), width="auto"),
            dbc.Col(
                dbc.Checklist(
                    id="sim-target-leagues-toggle",
                    options=[{"label": "Target leagues only", "value": "target"}],
                    value=["target"],
                    inline=True,
                    style={"fontSize": "12px", "color": TEXT_MUTED, "fontFamily": FONT},
                    inputStyle={"marginRight": "4px"},
                    labelStyle={"cursor": "pointer"},
                ),
                width="auto", className="ms-auto",
                style={"display": "flex", "alignItems": "center"},
            ),
        ], align="center", className="mb-2"),
        html.Div(f"Same position ({pos}) · click ★ to shortlist",
                 style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT, "marginBottom": "10px"}),
        html.Div(id="sim-players-content", children=[sim_section, sim_note]),
    ], style=CARD)

    return html.Div([header, row1, row2])


@app.callback(
    Output("sim-players-content", "children"),
    Input("sim-target-leagues-toggle", "value"),
    State("selected-player", "data"),
    prevent_initial_call=True,
)
def refresh_sim_players(toggle_val, player_name):
    if not player_name:
        return []
    target_only = "target" in (toggle_val or [])
    try:
        sim_df = get_similar_players(player_name, DF, MATRICES, n=8, target_leagues_only=target_only)
    except Exception:
        sim_df = pd.DataFrame()

    player_row = DF[DF["player"].str.lower().str.strip() == player_name.lower().strip()]
    pos        = player_row.iloc[0]["position_group"] if not player_row.empty else ""

    if sim_df.empty:
        sim_section = html.P("No similar players found.", style={"color": TEXT_MUTED, "fontFamily": FONT})
    else:
        est_mv_map2 = DF.set_index("player")["estimated_mv_m"].to_dict()
        sim_data = []
        for _, sr in sim_df.iterrows():
            smv     = sr.get("market_value_m")
            smv_est = est_mv_map2.get(sr["player"])
            if pd.notna(smv):
                mv_col = f"€{smv:.2f}m"
            elif pd.notna(smv_est):
                mv_col = f"~€{smv_est:.1f}m est."
            else:
                mv_col = "–"
            _full = DF[DF["player"] == sr["player"]]
            r_avail, r_total = (stat_coverage(_full.iloc[0], str(sr.get("position_group", pos)))
                                if not _full.empty else (0, len(RADAR_METRICS.get(pos, []))))
            sim_data.append({
                "★": "☆", "Player": sr["player"], "Club": sr.get("team", ""),
                "League": sr.get("league_clean", ""),
                "Age":    int(sr["age"]) if pd.notna(sr["age"]) else "–",
                "MV (€m)": mv_col,
                "Sim%":   f"{sr['similarity_pct']:.0f}% ({r_avail}/{r_total})",
                "Score":  round(float(sr["adjusted_scouting_score"]), 1) if pd.notna(sr.get("adjusted_scouting_score")) else "–",
                "_avail": r_avail,
            })
        sim_section = dash_table.DataTable(
            data=sim_data,
            columns=[{"name": c, "id": c} for c in ["★", "Player", "Club", "League", "Age", "MV (€m)", "Sim%", "Score", "_avail"]],
            hidden_columns=["_avail"],
            style_header=TBL_HEADER,
            style_cell={**TBL_CELL, "fontSize": "12px", "textAlign": "left"},
            style_cell_conditional=[
                {"if": {"column_id": "★"},    "width": "28px", "textAlign": "center",
                 "color": AMBER, "cursor": "pointer"},
                {"if": {"column_id": "Sim%"}, "textAlign": "center", "fontWeight": "700"},
            ],
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "background": BG_CARD2},
                {"if": {"filter_query": '{MV (€m)} contains "~"', "column_id": "MV (€m)"},
                 "color": AMBER, "fontStyle": "italic"},
                {"if": {"filter_query": "{_avail} < 4", "column_id": "Sim%"},
                 "color": TEXT_MUTED, "fontStyle": "italic", "fontWeight": "400"},
            ],
            tooltip_data=[
                {"MV (€m)": {"value": "⚠ Estimated value (peer-group median) — no Transfermarkt data. "
                                       "Real transfer cost may differ.", "type": "markdown"}}
                if "~" in (row.get("MV (€m)", "") or "") else {}
                for row in sim_data
            ],
            tooltip_delay=0,
            tooltip_duration=None,
        )

    avail, total = stat_coverage(player_row.iloc[0] if not player_row.empty else pd.Series(), pos)
    sim_note = html.P(
        f"Similarity based on {avail}/{total} available stats · "
        f"Sim% shows (stats used / position total) · grey = fewer than 4 stats, treat with caution.",
        style={"color": TEXT_MUTED, "fontSize": "11px", "fontStyle": "italic",
               "fontFamily": FONT, "marginTop": "8px", "marginBottom": "0"},
    )
    return [sim_section, sim_note]


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    [State("selected-player",  "data"),
     State("shortlist-store",  "data"),
     State("t2-filter-store",  "data")],
)
def render_tab(tab, player_name, shortlist, t2_filter_state):
    if tab == "tab-squad":
        return build_tab1()
    if tab == "tab-discovery":
        return build_tab2(t2_filter_state)
    if tab == "tab-profile":
        # Render profile inline so t3-body content arrives with the tab switch,
        # eliminating the race condition vs the render_profile callback.
        return html.Div([
            html.Div(id="t3-body", children=_profile_content(player_name, shortlist)),
        ], style={"padding": "16px"})
    if tab == "tab-shortlist":
        return build_tab4()
    if tab == "tab-replacement":
        return build_tab5()
    if tab == "tab-market":
        return build_tab6()
    return html.Div()

# Tab 1 — navigate to discovery with pre-set position
@app.callback(
    [Output("main-tabs", "value"),
     Output("nav-pos-store", "data")],
    Input({"type": "find-targets-btn", "pos": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def navigate_to_discovery(n_clicks_list):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    trigger_id = ctx.triggered[0]["prop_id"]
    import json
    try:
        id_dict = json.loads(trigger_id.split(".")[0])
        pos = id_dict.get("pos")
    except Exception:
        return no_update, no_update
    if all(n == 0 for n in (n_clicks_list or [])):
        return no_update, no_update
    return "tab-discovery", pos

# Tab 2 — utility buttons: "All Leagues" and "Target Only"
@app.callback(
    [Output("t2-leagues-target", "value"),
     Output("t2-leagues-big5",   "value")],
    [Input("t2-all-leagues-btn",  "n_clicks"),
     Input("t2-target-only-btn",  "n_clicks")],
    prevent_initial_call=True,
)
def handle_league_buttons(all_clicks, target_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "t2-all-leagues-btn":
        return list(TARGET_LEAGUE_ORDER), list(BIG5_LEAGUE_ORDER)
    # "Target Only" — reset to default
    return list(TARGET_LEAGUE_ORDER), []


# Tab 2 — merge two league checklists into combined store
@app.callback(
    Output("t2-leagues", "data"),
    [Input("t2-leagues-target", "value"),
     Input("t2-leagues-big5",   "value")],
)
def merge_league_selections(target_vals, big5_vals):
    return (target_vals or []) + (big5_vals or [])

# Tab 2 — slider labels
@app.callback(
    [Output("t2-age-lbl",  "children"),
     Output("t2-mv-lbl",   "children"),
     Output("t2-mins-lbl", "children")],
    [Input("t2-age", "value"),
     Input("t2-mv",  "value"),
     Input("t2-mins","value")],
)
def update_slider_labels(age, mv, mins):
    return (f"Max Age: {age}",
            f"Budget cap: €{mv}m",
            f"Min Minutes: {mins}")

# Tab 2 — show/hide position-specific filter toggles
@app.callback(
    [Output("t2-filter-def-wrap",  "style"),
     Output("t2-filter-goal-wrap", "style")],
    Input("t2-pos", "value"),
)
def update_filter_visibility(pos):
    show = {"display": "block"}
    hide = {"display": "none"}
    return (
        show if pos in FILTER_DEF_POSITIONS  else hide,
        show if pos in FILTER_GOAL_POSITIONS else hide,
    )

# Tab 2 — pre-populate position from navigation store
@app.callback(
    Output("t2-pos", "value"),
    Input("nav-pos-store", "data"),
    prevent_initial_call=True,
)
def set_nav_position(pos):
    return pos or no_update

# Tab 2 — persist filter state so it survives tab switches
@app.callback(
    Output("t2-filter-store", "data"),
    Input("t2-apply", "n_clicks"),
    [State("t2-pos",            "value"),
     State("t2-leagues-target", "value"),
     State("t2-leagues-big5",   "value"),
     State("t2-age",            "value"),
     State("t2-mv",             "value"),
     State("t2-mins",           "value"),
     State("t2-contract",       "value"),
     State("t2-mv-source",      "value"),
     State("t2-sort",           "value"),
     State("t2-filter-def",     "value"),
     State("t2-filter-goal",    "value"),
     State("t2-filter-prog",    "value")],
    prevent_initial_call=True,
)
def save_t2_filter_state(n, pos, leagues_target, leagues_big5, age, mv, mins,
                          contract, mv_source, sort, filter_def, filter_goal, filter_prog):
    if not n:
        return no_update
    return {
        "pos": pos, "leagues_target": leagues_target or [], "leagues_big5": leagues_big5 or [],
        "age": age, "mv": mv, "mins": mins, "contract": contract, "mv_source": mv_source,
        "sort": sort, "filter_def": bool(filter_def), "filter_goal": bool(filter_goal),
        "filter_prog": bool(filter_prog),
    }

# Tab 2 — apply filters + update results table
@app.callback(
    [Output("t2-tbl-wrap", "children"),
     Output("t2-hdr",      "children")],
    [Input("t2-apply", "n_clicks"),
     Input("t2-reset", "n_clicks")],
    [State("t2-pos",          "value"),
     State("t2-leagues",      "data"),
     State("t2-age",          "value"),
     State("t2-mv",           "value"),
     State("t2-mins",         "value"),
     State("t2-contract",     "value"),
     State("t2-sort",         "value"),
     State("t2-filter-def",   "value"),
     State("t2-filter-goal",  "value"),
     State("t2-filter-prog",  "value"),
     State("t2-mv-source",    "value"),
     State("shortlist-store", "data")],
    prevent_initial_call=True,
)
def update_discovery(apply_n, reset_n, pos, leagues, max_age, max_mv, min_mins,
                     contract, sort_by, filter_def, filter_goal, filter_prog, mv_source, shortlist):
    ctx = callback_context
    if not ctx.triggered or not pos:
        return (html.P("Select a position to load results.",
                       style={"color": TEXT_MUTED, "textAlign": "center",
                              "padding": "60px 0", "fontFamily": FONT}),
                html.P("← Select a position to begin scouting",
                       style={"color": TEXT_MUTED, "margin": "0", "fontFamily": FONT}))

    triggered = ctx.triggered[0]["prop_id"]
    if "reset" in triggered:
        return (html.P("Filters reset. Adjust criteria and click Apply Filters.",
                       style={"color": TEXT_MUTED, "textAlign": "center",
                              "padding": "60px 0", "fontFamily": FONT}),
                html.P("Filters reset", style={"color": TEXT_MUTED,
                                               "margin": "0", "fontFamily": FONT}))

    df = DF.copy()
    df = df[df["position_group"] == pos]
    if leagues:
        df = df[df["league_clean"].isin(leagues)]
    if max_age:
        df = df[df["age"] <= max_age]
    if max_mv is not None:
        df = df[df["market_value_m"].isna() | (df["market_value_m"] <= max_mv)]
    if min_mins:
        df = df[df["minutes"].fillna(0) >= min_mins]
    if contract == "expiring":
        df = df[df["contract_expiring"] == True]
    if mv_source == "valued":
        df = df[df["market_value_m"].notna()]
    elif mv_source == "estimated":
        df = df[df["market_value_m"].isna() & df["estimated_mv_m"].notna()]

    # ── 3 Boolean Filters ────────────────────────────────────────────────────
    # Thresholds computed on the FULL position group (not just the filtered subset)
    # so they are stable regardless of league/age/budget filters applied above.
    pos_all = DF[DF["position_group"] == pos]

    if filter_def and pos in FILTER_DEF_POSITIONS:
        tsr  = pd.to_numeric(df["tackle_success_rate"], errors="coerce")
        intc = pd.to_numeric(df["interceptions_p90"],   errors="coerce")
        tsr_thresh  = pd.to_numeric(pos_all["tackle_success_rate"], errors="coerce").quantile(0.55)
        intc_thresh = pd.to_numeric(pos_all["interceptions_p90"],   errors="coerce").quantile(0.50)
        # Fails only when BOTH stats present AND either falls below threshold
        fails = tsr.notna() & intc.notna() & ((tsr < tsr_thresh) | (intc < intc_thresh))
        df = df[~fails]

    if filter_goal and pos in FILTER_GOAL_POSITIONS:
        gp90 = pd.to_numeric(df["goals_p90"], errors="coerce")
        g_thresh = pd.to_numeric(pos_all["goals_p90"], errors="coerce").quantile(0.55)
        g_ok = gp90.notna() & (gp90 >= g_thresh)

        if "xg_p90" in df.columns:
            xg90 = pd.to_numeric(df["xg_p90"], errors="coerce")
            xg_thresh = pd.to_numeric(pos_all["xg_p90"], errors="coerce").quantile(0.55)
            xg_ok = xg90.notna() & (xg90 >= xg_thresh)
        else:
            xg_ok = pd.Series(False, index=df.index)
            xg90  = pd.Series(np.nan, index=df.index)

        all_missing = gp90.isna() & (xg90.isna() if "xg_p90" in df.columns else pd.Series(True, index=df.index))
        df = df[all_missing | g_ok | xg_ok]

    if filter_prog:
        pc90 = pd.to_numeric(df["progressive_carries_p90"], errors="coerce")
        pc_thresh = pd.to_numeric(pos_all["progressive_carries_p90"], errors="coerce").quantile(0.55)
        pc_ok = pc90.notna() & (pc90 >= pc_thresh)

        kp90, kp_ok = pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)
        if "key_passes_p90" in df.columns:
            kp90 = pd.to_numeric(df["key_passes_p90"], errors="coerce")
            kp_thresh = pd.to_numeric(pos_all["key_passes_p90"], errors="coerce").quantile(0.55)
            kp_ok = kp90.notna() & (kp90 >= kp_thresh)

        pp90, pp_ok = pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)
        if "progressive_passes_p90" in df.columns:
            pp90 = pd.to_numeric(df["progressive_passes_p90"], errors="coerce")
            pp_thresh = pd.to_numeric(pos_all["progressive_passes_p90"], errors="coerce").quantile(0.55)
            pp_ok = pp90.notna() & (pp90 >= pp_thresh)

        all_missing = pc90.isna() & kp90.isna() & pp90.isna()
        df = df[all_missing | pc_ok | kp_ok | pp_ok]

    # Sort
    if sort_by == "mv_asc":
        df = df.sort_values("market_value_m", ascending=True)
    elif sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=(sort_by == "age"))
    else:
        df = df.sort_values("adjusted_scouting_score", ascending=False)

    n = len(df)
    if n == 0:
        return (html.P("No players match your criteria.",
                       style={"color": TEXT_MUTED, "textAlign": "center",
                              "padding": "60px 0", "fontFamily": FONT}),
                html.P(f"No results for {pos}", style={"color": TEXT_MUTED,
                                                        "margin": "0", "fontFamily": FONT}))

    shortlist_names = [p["player"] for p in (shortlist or [])]
    _total_stats    = len(RADAR_METRICS.get(pos, []))

    table_data = []
    for _, r in df.iterrows():
        star = "★" if r["player"] in shortlist_names else "☆"
        mv_real = r.get("market_value_m")
        mv_est  = r.get("estimated_mv_m")
        if pd.notna(mv_real):
            mv_str = f"€{mv_real:.2f}m"
        elif pd.notna(mv_est):
            mv_str = f"~€{mv_est:.1f}m est."
        else:
            mv_str = "–"
        contract_ico = "⚡" if r.get("contract_expiring") else ""
        r_avail, _ = stat_coverage(r, pos)
        age_ctx = age_stage_emoji(r.get("age_score_context", ""))
        table_data.append({
            "★":       star,
            "Player":  r["player"],
            "Club":    r.get("team", ""),
            "League":  r.get("league_clean", ""),
            "Age":     int(r["age"]) if pd.notna(r["age"]) else "–",
            "~":       age_ctx,
            "Pos":     r.get("position_group", ""),
            "MV (€m)": mv_str,
            "Raw":     round(float(r["raw_scouting_score"]), 1) if pd.notna(r["raw_scouting_score"]) else "–",
            "Score":   round(float(r["adjusted_scouting_score"]), 1) if pd.notna(r["adjusted_scouting_score"]) else "–",
            "Data":    "Full" if r_avail >= _total_stats else "Basic",
            "Exp":     contract_ico,
        })

    _col_names = ["★", "Player", "Club", "League", "Age", "~", "Pos", "MV (€m)", "Raw", "Score", "Data", "Exp"]

    # Build active-filter note for header
    active_filters = []
    if filter_def  and pos in FILTER_DEF_POSITIONS:  active_filters.append("⚔ Def Strong")
    if filter_goal and pos in FILTER_GOAL_POSITIONS: active_filters.append("🎯 Goal Threat")
    if filter_prog: active_filters.append("🔄 Progressive")
    filter_note = f"  ·  {' + '.join(active_filters)}" if active_filters else ""

    tbl = dash_table.DataTable(
        id="t2-table",
        data=table_data,
        columns=[{"name": c, "id": c} for c in _col_names],
        page_size=25,
        sort_action="native",
        filter_action="native",
        row_selectable=False,
        style_table={"overflowX": "auto"},
        style_header=TBL_HEADER,
        style_cell={**TBL_CELL, "textAlign": "left", "maxWidth": "160px",
                    "overflow": "hidden", "textOverflow": "ellipsis"},
        style_cell_conditional=[
            {"if": {"column_id": "★"},    "width": "30px", "textAlign": "center",
             "cursor": "pointer", "color": AMBER, "fontSize": "14px"},
            {"if": {"column_id": "Age"},  "width": "40px", "textAlign": "center"},
            {"if": {"column_id": "~"},    "width": "24px", "textAlign": "center",
             "fontSize": "12px", "padding": "0"},
            {"if": {"column_id": "Pos"},  "width": "40px", "textAlign": "center"},
            {"if": {"column_id": "Raw"},  "width": "55px", "textAlign": "center"},
            {"if": {"column_id": "Score"},"width": "55px", "textAlign": "center"},
            {"if": {"column_id": "Data"}, "width": "55px", "textAlign": "center",
             "fontWeight": "600", "fontSize": "11px"},
            {"if": {"column_id": "Exp"},  "width": "30px", "textAlign": "center",
             "color": AMBER},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": BG_CARD2},
            {"if": {"filter_query": '{★} = "★"'}, "color": AMBER},
            {"if": {"filter_query": '{MV (€m)} contains "~"', "column_id": "MV (€m)"},
             "color": AMBER, "fontStyle": "italic"},
            {"if": {"filter_query": '{Data} = "Full"',  "column_id": "Data"}, "color": GREEN},
            {"if": {"filter_query": '{Data} = "Basic"', "column_id": "Data"}, "color": TEXT_MUTED},
        ],
        tooltip_data=[
            {"MV (€m)": {"value": "⚠ Estimated value (peer-group median) — no Transfermarkt data. "
                                   "Real transfer cost may differ.", "type": "markdown"}}
            if "~" in (row.get("MV (€m)", "") or "") else {}
            for row in table_data
        ],
        active_cell=None,
        tooltip_delay=0,
        tooltip_duration=None,
    )

    hdr = html.Div([
        html.Span(f"{n} player{'s' if n != 1 else ''} match",
                  style={"color": TEXT, "fontWeight": "700",
                         "fontSize": "14px", "fontFamily": FONT}),
        html.Span(f"  ·  {pos}  ·  click ☆ to shortlist · click row to profile",
                  style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT}),
        html.Span(filter_note, style={"color": AMBER, "fontSize": "11px", "fontFamily": FONT}),
    ])
    return tbl, hdr

# Tab 2 — row click → profile or shortlist
@app.callback(
    [Output("selected-player",              "data"),
     Output("main-tabs",                    "value", allow_duplicate=True),
     Output("shortlist-store",              "data",  allow_duplicate=True),
     Output("t2-shortlist-msg",             "children")],
    Input("t2-table", "active_cell"),
    [State("t2-table",        "data"),
     State("shortlist-store", "data")],
    prevent_initial_call=True,
)
def handle_table_click(active_cell, table_data, shortlist):
    if not active_cell or not table_data:
        return no_update, no_update, no_update, no_update
    row   = table_data[active_cell["row"]]
    pname = row.get("Player", "")
    col   = active_cell.get("column_id", "")

    if col == "★":
        shortlist = list(shortlist or [])
        names = [p["player"] for p in shortlist]
        if pname in names:
            shortlist = [p for p in shortlist if p["player"] != pname]
            msg = f"✓ Removed {pname} from shortlist"
            return no_update, no_update, shortlist, msg
        if len(shortlist) >= 10:
            return no_update, no_update, no_update, "⚠ Shortlist full (max 10 players)"
        pr = DF[DF["player"] == pname]
        if not pr.empty:
            r = pr.iloc[0]
            shortlist.append({
                "player":               pname,
                "team":                 str(r.get("team", "")),
                "league":               str(r.get("league_clean", "")),
                "age":                  float(r["age"]) if pd.notna(r["age"]) else 0,
                "position_group":       str(r.get("position_group", "")),
                "market_value_m":       float(r["market_value_m"]) if pd.notna(r["market_value_m"]) else 0,
                "adjusted_scouting_score": float(r["adjusted_scouting_score"]) if pd.notna(r["adjusted_scouting_score"]) else 0,
                "contract_expiring":    bool(r.get("contract_expiring", False)),
            })
        msg = f"⭐ Added {pname} to shortlist ({len(shortlist)}/10)"
        return no_update, no_update, shortlist, msg
    else:
        return pname, "tab-profile", no_update, no_update

# Tab 2 — export CSV
@app.callback(
    Output("t2-dl", "data"),
    Input("t2-export-btn", "n_clicks"),
    State("t2-table", "data"),
    prevent_initial_call=True,
)
def export_csv(n, table_data):
    if not n or not table_data:
        return no_update
    df_exp = pd.DataFrame(table_data)
    return dcc.send_data_frame(df_exp.to_csv, "scoutedge_results.csv", index=False)

# Tab 3 — render player profile
# Uses _profile_content() so the same rendering logic is shared with render_tab.
# Only depends on selected-player (not main-tabs) so it handles the case where
# selected-player changes while the profile tab is already open, without racing
# against render_tab (which renders the profile on the initial tab switch).
@app.callback(
    Output("t3-body", "children"),
    Input("selected-player", "data"),
    [State("main-tabs",       "value"),
     State("shortlist-store", "data")],
    prevent_initial_call=True,
)
def render_profile(player_name, tab, shortlist):
    if tab != "tab-profile":
        return no_update
    return _profile_content(player_name, shortlist)

# Tab 3 — add/remove from shortlist via profile button
@app.callback(
    Output("shortlist-store", "data", allow_duplicate=True),
    Input("t3-shortlist-btn", "n_clicks"),
    [State("selected-player",   "data"),
     State("shortlist-store",   "data")],
    prevent_initial_call=True,
)
def profile_shortlist_toggle(n, player_name, shortlist):
    if not n or not player_name:
        return no_update
    shortlist = list(shortlist or [])
    names = [p["player"] for p in shortlist]
    if player_name in names:
        return [p for p in shortlist if p["player"] != player_name]
    if len(shortlist) >= 10:
        return no_update
    pr = DF[DF["player"] == player_name]
    if pr.empty:
        return no_update
    r = pr.iloc[0]
    shortlist.append({
        "player":               player_name,
        "team":                 str(r.get("team", "")),
        "league":               str(r.get("league_clean", "")),
        "age":                  float(r["age"]) if pd.notna(r["age"]) else 0,
        "position_group":       str(r.get("position_group", "")),
        "market_value_m":       float(r["market_value_m"]) if pd.notna(r["market_value_m"]) else 0,
        "adjusted_scouting_score": float(r["adjusted_scouting_score"]) if pd.notna(r["adjusted_scouting_score"]) else 0,
        "contract_expiring":    bool(r.get("contract_expiring", False)),
    })
    return shortlist

# Tab 4 — render shortlist
@app.callback(
    [Output("t4-table-wrap",  "children"),
     Output("t4-count",       "children"),
     Output("tab-shortlist-label", "label")],
    Input("shortlist-store",  "data"),
)
def render_shortlist(shortlist):
    shortlist = shortlist or []
    n = len(shortlist)
    count_txt = f"{n} player{'s' if n != 1 else ''} · max 10"
    tab_label = f"Shortlist ({n})"

    if n == 0:
        empty = html.Div([
            html.Div("⭐", style={"fontSize": "40px", "textAlign": "center",
                                   "marginBottom": "12px"}),
            html.P("No players added yet.",
                   style={"color": TEXT_MUTED, "textAlign": "center",
                          "fontFamily": FONT, "fontWeight": "600"}),
            html.P("Click ☆ on any player in Player Discovery to add them here.",
                   style={"color": TEXT_MUTED, "textAlign": "center",
                          "fontSize": "13px", "fontFamily": FONT}),
        ], style={"padding": "40px 0"})
        return empty, count_txt, tab_label

    rows = []
    for p in shortlist:
        mv = p.get("market_value_m")
        rows.append({
            "Remove":   "✕",
            "Player":   p["player"],
            "Club":     p.get("team", ""),
            "League":   p.get("league", ""),
            "Age":      int(p["age"]) if p.get("age") else "–",
            "Pos":      p.get("position_group", ""),
            "MV (€m)":  f"€{mv:.2f}m" if mv and mv > 0 else "–",
            "Score":    round(p.get("adjusted_scouting_score", 0), 1),
            "Contract": "⚡" if p.get("contract_expiring") else "",
        })

    tbl = dash_table.DataTable(
        id="t4-main-table",
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["Remove", "Player", "Club", "League", "Age", "Pos",
                  "MV (€m)", "Score", "Contract"]],
        row_selectable="multi",
        selected_rows=[],
        style_header=TBL_HEADER,
        style_cell={**TBL_CELL, "textAlign": "left"},
        style_cell_conditional=[
            {"if": {"column_id": "Remove"},   "width": "40px", "textAlign": "center",
             "color": RED, "cursor": "pointer", "fontWeight": "700"},
            {"if": {"column_id": "Contract"}, "width": "40px", "textAlign": "center",
             "color": AMBER},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": BG_CARD2},
        ],
        active_cell=None,
    )
    return tbl, count_txt, tab_label

# Tab 4 — remove from shortlist
@app.callback(
    Output("shortlist-store", "data", allow_duplicate=True),
    Input("t4-main-table", "active_cell"),
    [State("t4-main-table", "data"),
     State("shortlist-store", "data")],
    prevent_initial_call=True,
)
def remove_from_shortlist(active_cell, table_data, shortlist):
    if not active_cell or not table_data:
        return no_update
    col = active_cell.get("column_id", "")
    if col != "Remove":
        return no_update
    row  = table_data[active_cell["row"]]
    name = row.get("Player", "")
    return [p for p in (shortlist or []) if p["player"] != name]

# Tab 4 — clear all shortlist
@app.callback(
    Output("shortlist-store", "data", allow_duplicate=True),
    Input("t4-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_shortlist(n):
    if not n:
        return no_update
    return []

# Tab 4 — comparison radar when 2+ rows selected
@app.callback(
    Output("t4-comparison-panel", "children"),
    Input("t4-main-table", "selected_rows"),
    State("t4-main-table", "data"),
    prevent_initial_call=True,
)
def update_comparison(selected_rows, table_data):
    if not selected_rows or len(selected_rows) < 2:
        return html.P("Select 2–4 players above (checkboxes) to compare them.",
                      style={"color": TEXT_MUTED, "textAlign": "center",
                             "padding": "20px 0", "fontFamily": FONT})
    if len(selected_rows) > 4:
        selected_rows = selected_rows[:4]

    players = [table_data[i]["Player"] for i in selected_rows]
    pos_groups = [table_data[i].get("Pos", "") for i in selected_rows]

    # Use first player's pos for radar metrics
    pos = pos_groups[0] if pos_groups else "W"
    metrics = RADAR_METRICS.get(pos, [])

    COLORS = [RED, BLUE, GREEN, AMBER]
    fig = go.Figure()
    comp_data = {lbl: [] for lbl, _ in metrics}

    for i, pname in enumerate(players):
        pr = DF[DF["player"] == pname]
        if pr.empty:
            continue
        r = pr.iloc[0]
        pos_df = DF[DF["position_group"] == r.get("position_group", pos)]
        vals = []
        for lbl, col in metrics:
            if col not in DF.columns:
                vals.append(0)
                comp_data[lbl].append("–")
                continue
            v = pd.to_numeric(r[col], errors="coerce")
            pct = pct_rank(v, pos_df[col])
            vals.append(pct)
            comp_data[lbl].append(f"{pct:.0f}%")

        if vals:
            lbls = [lbl for lbl, _ in metrics]
            fig.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=lbls + [lbls[0]],
                fill="toself",
                fillcolor=RGBA_COLORS[i % len(RGBA_COLORS)],
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                name=pname,
            ))

    fig.update_layout(
        **CHART_DEFAULTS,
        polar=dict(
            bgcolor=BG_CARD2,
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=8, color=TEXT_MUTED),
                            gridcolor=BORDER),
            angularaxis=dict(tickfont=dict(size=10, color=TEXT)),
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10, color=TEXT)),
        height=350,
    )

    comp_rows = []
    for lbl, _ in metrics:
        row_d = {"Metric": lbl}
        for j, pname in enumerate(players):
            row_d[pname[:20]] = comp_data[lbl][j] if j < len(comp_data[lbl]) else "–"
        comp_rows.append(row_d)

    col_names = ["Metric"] + [p[:20] for p in players]
    comp_tbl = dash_table.DataTable(
        data=comp_rows,
        columns=[{"name": c, "id": c} for c in col_names],
        style_header=TBL_HEADER,
        style_cell={**TBL_CELL, "textAlign": "center", "fontSize": "11px"},
    )

    return html.Div([
        html.H6("Player Comparison",
                style={**H, "fontSize": "14px", "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig, config={"displayModeBar": False}), md=6),
            dbc.Col(comp_tbl, md=6),
        ]),
    ])

# Tab 5 — slider labels
@app.callback(
    Output("t5-bgt-lbl", "children"),
    Input("t5-budget", "value"),
)
def update_t5_budget_lbl(v):
    return f"Target budget: €{v}m"

# Tab 5 — reference player mini card
@app.callback(
    Output("t5-mini", "children"),
    Input("t5-player", "value"),
)
def update_t5_mini(player_name):
    if not player_name:
        return html.Div()
    pr = DF[DF["player"] == player_name]
    if pr.empty:
        return html.Div()
    r  = pr.iloc[0]
    mv = r.get("market_value_m")
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div([
                html.Div(player_name, style={"color": TEXT, "fontWeight": "700",
                                              "fontSize": "14px", "fontFamily": FONT}),
                html.Div(f"{r.get('team','')}  ·  {r.get('position_group','')}  ·  Age {int(r['age']) if pd.notna(r['age']) else '–'}",
                         style={"color": TEXT_MUTED, "fontSize": "12px", "fontFamily": FONT}),
            ])),
            dbc.Col([
                html.Div(f"€{mv:.2f}m" if pd.notna(mv) else "–",
                         style={"color": BLUE, "fontWeight": "700",
                                "fontSize": "15px", "fontFamily": FONT}),
                html.Div(f"Score: {r.get('adjusted_scouting_score',0):.1f}",
                         style={"color": TEXT_MUTED, "fontSize": "11px", "fontFamily": FONT}),
            ], width="auto"),
        ], align="center"),
    ], style={**CARD, "background": BG_CARD2})

# Tab 5 — find replacements
@app.callback(
    Output("t5-results", "children"),
    Input("t5-find", "n_clicks"),
    [State("t5-player",      "value"),
     State("t5-budget",      "value"),
     State("t5-age",         "value"),
     State("t5-diff-league", "value"),
     State("t5-tgt-leagues", "value")],
    prevent_initial_call=True,
)
def find_replacements(n, player_name, budget, max_age, diff_league, tgt_only):
    if not n or not player_name:
        return html.P("Select a reference player and click Find Replacements.",
                      style={"color": TEXT_MUTED, "textAlign": "center",
                             "padding": "40px 0", "fontFamily": FONT})

    pr = DF[DF["player"] == player_name]
    if pr.empty:
        return html.P("Player not found.",
                      style={"color": TEXT_MUTED, "fontFamily": FONT})
    ref = pr.iloc[0]
    ref_mv    = ref.get("market_value_m")
    ref_pos   = str(ref.get("position_group", ""))
    ref_score = float(ref.get("adjusted_scouting_score", 0))

    try:
        res_df = get_similar_players(
            player_name, DF, MATRICES, n=20,
            max_market_value_m=budget,
            max_age=max_age,
            different_league=bool(diff_league),
            target_leagues_only=bool(tgt_only),
        )
    except Exception as e:
        return html.P(f"Error: {e}", style={"color": RED, "fontFamily": FONT})

    # Enforce same position
    if not res_df.empty and "position_group" in res_df.columns:
        res_df = res_df[res_df["position_group"] == ref_pos]

    if res_df.empty:
        return html.Div([
            html.H6(f"Replacements for {player_name}",
                    style={**H, "marginBottom": "8px"}),
            html.P("No replacements found with current criteria.",
                   style={"color": TEXT_MUTED, "fontFamily": FONT}),
        ])

    est_mv_map3 = DF.set_index("player")["estimated_mv_m"].to_dict()
    rows = []
    for _, sr in res_df.iterrows():
        smv     = sr.get("market_value_m")
        smv_est = est_mv_map3.get(sr["player"])
        if pd.notna(smv):
            mv_col = f"€{smv:.2f}m"
        elif pd.notna(smv_est):
            mv_col = f"~€{smv_est:.1f}m est."
        else:
            mv_col = "–"
        sim_pct = float(sr.get("similarity_pct", 0))
        _full = DF[DF["player"] == sr["player"]]
        r_avail, r_total = (stat_coverage(_full.iloc[0], ref_pos)
                            if not _full.empty else (0, len(RADAR_METRICS.get(ref_pos, []))))
        rows.append({
            "★":       "☆",
            "Player":  sr["player"],
            "Club":    sr.get("team", ""),
            "League":  sr.get("league_clean", ""),
            "Age":     int(sr["age"]) if pd.notna(sr["age"]) else "–",
            "MV (€m)": mv_col,
            "Sim%":    f"{sim_pct:.0f}% ({r_avail}/{r_total})",
            "Score":   round(float(sr.get("adjusted_scouting_score", 0)), 1),
            "Exp":     "⚡" if sr.get("contract_expiring") else "",
            "_avail":  r_avail,
        })

    tbl = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["★", "Player", "Club", "League", "Age", "MV (€m)", "Sim%", "Score", "Exp", "_avail"]],
        hidden_columns=["_avail"],
        style_header=TBL_HEADER,
        style_cell={**TBL_CELL, "textAlign": "left", "fontSize": "12px"},
        style_cell_conditional=[
            {"if": {"column_id": "★"},   "width": "28px", "textAlign": "center",
             "color": AMBER},
            {"if": {"column_id": "Exp"}, "width": "28px", "textAlign": "center",
             "color": AMBER},
            {"if": {"column_id": "Sim%"}, "textAlign": "center", "fontWeight": "700"},
        ],
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": BG_CARD2},
            {"if": {"filter_query": '{MV (€m)} contains "~"', "column_id": "MV (€m)"},
             "color": AMBER, "fontStyle": "italic"},
            {"if": {"filter_query": "{_avail} < 4", "column_id": "Sim%"},
             "color": TEXT_MUTED, "fontStyle": "italic", "fontWeight": "400"},
        ],
        tooltip_data=[
            {"MV (€m)": {"value": "⚠ Estimated value (peer-group median) — no Transfermarkt data. "
                                   "Real transfer cost may differ.", "type": "markdown"}}
            if "~" in (row.get("MV (€m)", "") or "") else {}
            for row in rows
        ],
        tooltip_delay=0,
        tooltip_duration=None,
    )

    ref_avail, ref_total = stat_coverage(ref, ref_pos)
    return html.Div([
        html.H6(f"Replacements for {player_name}",
                style={**H, "fontSize": "14px", "marginBottom": "4px"}),
        html.Div(f"Same position ({ref_pos})  ·  Budget ≤€{budget}m  ·  Age ≤{max_age}  ·  {len(rows)} found",
                 style={"color": TEXT_MUTED, "fontSize": "12px",
                        "marginBottom": "8px", "fontFamily": FONT}),
        html.P(f"Sim% shows (stats used / position total) · grey = fewer than 4 stats, treat with caution.",
               style={"color": TEXT_MUTED, "fontSize": "10px", "fontFamily": FONT,
                      "fontStyle": "italic", "marginBottom": "10px"}),
        tbl,
    ])

# Tab 6 — best value chart
@app.callback(
    Output("t6-best-value", "figure"),
    Input("main-tabs", "value"),
)
def update_best_value(tab):
    df = DF[DF["market_value_m"] <= 3.0].copy()
    df = df.sort_values("adjusted_scouting_score", ascending=False).head(15)
    if df.empty:
        return go.Figure()

    pos_colors = {"CB": BLUE, "FB": "#00BCD4", "DM": "#9C27B0",
                  "CM": AMBER, "AM": "#FF5722", "W": GREEN, "ST": RED}
    colors_list = [pos_colors.get(p, TEXT_MUTED) for p in df["position_group"]]

    fig = go.Figure(go.Bar(
        x=df["adjusted_scouting_score"],
        y=df["player"],
        orientation="h",
        marker=dict(color=colors_list),
        hovertemplate="%{y}<br>Score: %{x:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_DEFAULTS,
        title=dict(text="Top 15 · Sorted by Score", font=dict(size=12)),
        xaxis=dict(title="Adjusted Score", gridcolor=BORDER),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        height=330,
    )
    return fig

# Tab 6 — top scoring players by league
@app.callback(
    Output("t6-undervalued", "figure"),
    Input("main-tabs", "value"),
)
def update_top_scoring_by_league(tab):
    df = DF.dropna(subset=["adjusted_scouting_score", "league_clean"]).copy()
    avg_scores = (df.groupby("league_clean")["adjusted_scouting_score"]
                    .mean()
                    .reset_index(name="avg_score")
                    .sort_values("avg_score", ascending=False))
    if avg_scores.empty:
        return go.Figure()

    league_colors = [RED, BLUE, GREEN, AMBER, "#9C27B0", "#00BCD4", "#FF5722",
                     "#795548", "#607D8B", "#E91E63"]
    colors_list = [league_colors[i % len(league_colors)]
                   for i in range(len(avg_scores))]

    fig = go.Figure(go.Bar(
        x=avg_scores["league_clean"],
        y=avg_scores["avg_score"],
        marker=dict(color=colors_list, opacity=0.85),
        hovertemplate="%{x}<br>Avg Score: %{y:.1f}<extra></extra>",
    ))
    fig.update_layout(
        **CHART_DEFAULTS,
        title=dict(text="Average Adjusted Scouting Score", font=dict(size=12)),
        xaxis=dict(title="", tickfont=dict(size=10), tickangle=-30),
        yaxis=dict(title="Avg Score", gridcolor=BORDER),
        height=330,
    )
    return fig

# Tab 6 — expiry table
@app.callback(
    Output("t6-expiry-table", "children"),
    [Input("main-tabs", "value"),
     Input("t6-pos-filter", "value")],
)
def update_expiry_table(tab, pos_filter):
    df = DF[DF["contract_expiring"] == True].copy()
    if pos_filter and pos_filter != "ALL":
        df = df[df["position_group"] == pos_filter]
    df = df.sort_values("adjusted_scouting_score", ascending=False)

    if df.empty:
        return html.P("No expiring contracts found.",
                      style={"color": TEXT_MUTED, "fontFamily": FONT})

    rows = []
    for _, r in df.head(25).iterrows():
        mv = r.get("market_value_m")
        rows.append({
            "Player":  r["player"],
            "Club":    r.get("team", ""),
            "League":  r.get("league_clean", ""),
            "Age":     int(r["age"]) if pd.notna(r["age"]) else "–",
            "Pos":     r.get("position_group", ""),
            "MV":      f"€{mv:.2f}m" if pd.notna(mv) else "–",
            "Score":   round(float(r["adjusted_scouting_score"]), 1),
            "Status":  "⚡ EXPIRING",
        })

    return dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in
                 ["Player", "Club", "League", "Age", "Pos", "MV", "Score", "Status"]],
        style_header=TBL_HEADER,
        style_cell={**TBL_CELL, "textAlign": "left", "fontSize": "11px",
                    "padding": "6px 8px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": BG_CARD2},
            {"if": {"column_id": "Status"}, "color": AMBER, "fontWeight": "700"},
        ],
        page_size=10,
    )

# Tab 6 — scatter chart (callback kept for completeness; initial figure is built
# inline in build_tab6() to avoid the dynamic-layout race condition)
@app.callback(
    Output("t6-scatter", "figure"),
    Input("main-tabs", "value"),
)
def update_scatter(tab):
    if tab != "tab-market":
        return no_update
    return build_scatter_fig()

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=8050)
