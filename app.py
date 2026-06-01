import sys, pickle
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import (DATA_FINAL, DATA_PROC, ROOT, TARGET_LEAGUES, SCOUTING_WEIGHTS,
                    EXCLUDED_NATIONALITIES,
                    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_DARK,
                    COLOR_CARD, COLOR_TEXT, COLOR_MUTED, COLOR_BORDER)
from similarity import build_similarity_matrices, get_similar_players
from squad_analysis import load_squad

# ── Startup ───────────────────────────────────────────────────────────────────
print("Loading players_final.csv...")
DF = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
DF["market_value_m"] = pd.to_numeric(DF["market_value_m"], errors="coerce")
DF["age"] = pd.to_numeric(DF["age"], errors="coerce")
DF["minutes"] = pd.to_numeric(DF["minutes"], errors="coerce")
DF["contract_expiring"] = DF["contract_expiring"].astype(str).str.lower().isin(["true", "1", "yes"])
print(f"Loaded {len(DF)} players")

SQUAD = load_squad()
SQUAD["is_foreign"] = SQUAD["is_foreign"].astype(str).str.lower().isin(["true", "1", "yes"])
_TODAY = datetime.now()
SQUAD["contract_expiring"] = pd.to_datetime(SQUAD["contract_expiry"], errors="coerce").apply(
    lambda d: (d - _TODAY).days <= 365 if pd.notna(d) else False
)

pkl_path = DATA_PROC / "similarity_matrices.pkl"
if pkl_path.exists():
    with open(pkl_path, "rb") as f:
        MATRICES = pickle.load(f)
    print("Loaded cached similarity matrices")
else:
    MATRICES = build_similarity_matrices(DF)
    with open(pkl_path, "wb") as f:
        pickle.dump(MATRICES, f)

LEAGUES_IN_DATA = sorted(DF["league_clean"].dropna().unique().tolist())
POS_OPTIONS = [{"label": p, "value": p} for p in ["CB", "FB", "DM", "CM", "AM", "W", "ST"]]
SORT_OPTIONS = [
    {"label": "Adjusted Score", "value": "adjusted_scouting_score"},
    {"label": "Raw Score",      "value": "raw_scouting_score"},
    {"label": "Market Value",   "value": "market_value_m"},
    {"label": "Age",            "value": "age"},
    {"label": "Value Gap %",    "value": "value_gap_pct"},
]

CARD_STYLE = {
    "background": COLOR_CARD,
    "border": f"1px solid {COLOR_BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
}
HDR = {"color": COLOR_TEXT, "fontWeight": "700"}

RADAR_METRICS = {
    "CB": [("Tackling",    "tackle_success_rate"),
           ("Interceptions","interceptions_p90"),
           ("Prog Pass",   "progressive_passes_p90"),
           ("Clearances",  "clearances_p90"),
           ("Aerial",      "aerial_duels_won_pct")],
    "FB": [("Prog Carries","progressive_carries_p90"),
           ("Key Passes",  "key_passes_p90"),
           ("Crosses",     "crosses_p90"),
           ("Tackling",    "tackles_p90"),
           ("Assists",     "assists_p90")],
    "DM": [("Tackling",   "tackles_p90"),
           ("Interceptions","interceptions_p90"),
           ("Prog Passes", "progressive_passes_p90"),
           ("Pass Acc",    "pass_completion_rate"),
           ("Key Passes",  "key_passes_p90")],
    "CM": [("Prog Passes", "progressive_passes_p90"),
           ("Key Passes",  "key_passes_p90"),
           ("Assists",     "assists_p90"),
           ("Tackling",    "tackles_p90"),
           ("Goals",       "goals_p90")],
    "AM": [("xAG",         "xag_p90"),
           ("Key Passes",  "key_passes_p90"),
           ("Assists",     "assists_p90"),
           ("Prog Carries","progressive_carries_p90"),
           ("Goals",       "goals_p90")],
    "W":  [("Prog Carries","progressive_carries_p90"),
           ("Goals",       "goals_p90"),
           ("xG",          "xg_p90"),
           ("Dribbles",    "dribbles_completed_p90"),
           ("Assists",     "assists_p90")],
    "ST": [("npxG",        "npxg_p90"),
           ("Goals",       "goals_p90"),
           ("Shot Acc",    "shot_accuracy"),
           ("Prog Rcvs",   "progressive_receives_p90"),
           ("Assists",     "assists_p90")],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt_mv(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"€{v:.1f}m"

def age_dot(age):
    if age is None:
        return "\U0001f7e1"
    age = int(age)
    if age < 28:
        return "\U0001f7e2"
    if age <= 30:
        return "\U0001f7e1"
    return "\U0001f534"

def get_pct(pos_df, val, col):
    if col not in pos_df.columns:
        return 50.0
    series = pd.to_numeric(pos_df[col], errors="coerce").dropna()
    if len(series) == 0 or pd.isna(val):
        return 50.0
    return float((series < float(val)).sum() / len(series) * 100)

def kpi_card(title, value, subtitle="", color=COLOR_TEXT):
    return html.Div([
        html.P(title, style={"color": COLOR_MUTED, "fontSize": "11px",
                              "textTransform": "uppercase", "letterSpacing": "1px", "margin": "0 0 4px"}),
        html.H2(str(value), style={"color": color, "margin": "0", "fontSize": "2rem", "fontWeight": "800"}),
        html.P(subtitle, style={"color": COLOR_MUTED, "fontSize": "12px", "margin": "4px 0 0"}),
    ], style={**CARD_STYLE, "textAlign": "center"})

def status_style(status):
    if status == "Undervalued":
        return {"background": "#00C85322", "color": COLOR_SUCCESS, "padding": "2px 8px",
                "borderRadius": "12px", "fontSize": "11px", "fontWeight": "600"}
    if status == "Overvalued":
        return {"background": "#CC000022", "color": COLOR_PRIMARY, "padding": "2px 8px",
                "borderRadius": "12px", "fontSize": "11px", "fontWeight": "600"}
    return {"background": "#88888822", "color": COLOR_MUTED, "padding": "2px 8px",
            "borderRadius": "12px", "fontSize": "11px"}

# ── Tab 1: Squad Intelligence ─────────────────────────────────────────────────
def _squad_dict():
    return {r["player"]: r.to_dict() for _, r in SQUAD.iterrows()}

def _pitch_card(label, key, sq):
    row = sq.get(key, {})
    age = row.get("age") if row else None
    foreign = row.get("is_foreign", False)
    exp = row.get("contract_expiring", False)
    icons = ("⚡" if exp else "") + ("\U0001f30d" if foreign else "")
    return html.Div([
        html.Div(age_dot(age), style={"fontSize": "10px"}),
        html.Div(label, style={"color": COLOR_TEXT, "fontSize": "11px", "fontWeight": "700"}),
        html.Div(f"{age}y {icons}" if age else icons, style={"color": COLOR_MUTED, "fontSize": "10px"}),
    ], style={"background": "rgba(26,26,26,0.92)", "border": f"1px solid {COLOR_BORDER}",
              "borderRadius": "6px", "padding": "6px 8px", "textAlign": "center", "minWidth": "72px"})

def build_pitch():
    sq = _squad_dict()
    rs = {"display": "flex", "justifyContent": "center", "gap": "10px", "marginBottom": "10px"}
    return html.Div([
        html.Div([
            html.Div("4-3-3 Formation",
                     style={"color": COLOR_MUTED, "fontSize": "11px", "textAlign": "center",
                            "marginBottom": "8px", "fontWeight": "600"}),
            html.Div([_pitch_card("Bencharki", "Achraf Bencharki", sq),
                      _pitch_card("M. Sherif", "Mohamed Sherif", sq),
                      _pitch_card("Trezeguet", "Trezeguet", sq)], style=rs),
            html.Div([_pitch_card("El Shahat", "Hussein El Shahat", sq),
                      _pitch_card("Ashour",    "Emam Ashour", sq),
                      _pitch_card("Otaka",     "Marwan Otaka", sq)], style=rs),
            html.Div([_pitch_card("A. Eid",    "Ahmed Eid", sq),
                      _pitch_card("Y. Ibrahim","Yasser Ibrahim", sq),
                      _pitch_card("M. Hany",   "Mohamed Hany", sq),
                      _pitch_card("M. Ateya",  "Marwan Ateya", sq)], style=rs),
            html.Div([_pitch_card("El Shenawy","Mohamed El Shenawy", sq)], style=rs),
        ], style={"background": "linear-gradient(180deg,#1a4a1a 0%,#0d2e0d 100%)",
                  "border": "2px solid #2d6a2d", "borderRadius": "8px", "padding": "20px 16px"}),
        html.Div([
            html.Span("\U0001f7e2 <28  ", style={"color": COLOR_MUTED, "fontSize": "11px"}),
            html.Span("\U0001f7e1 28-30  ", style={"color": COLOR_MUTED, "fontSize": "11px"}),
            html.Span("\U0001f534 >30  ", style={"color": COLOR_MUTED, "fontSize": "11px"}),
            html.Span("⚡ Expiring  ", style={"color": COLOR_WARNING, "fontSize": "11px"}),
            html.Span("\U0001f30d Foreign", style={"color": COLOR_MUTED, "fontSize": "11px"}),
        ], style={"textAlign": "center", "marginTop": "6px"}),
    ])

def build_gap_data():
    pc = SQUAD["position_group"].value_counts().to_dict()
    rows = []
    for pos in ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]:
        cnt = pc.get(pos, 0)
        pp = SQUAD[SQUAD["position_group"] == pos]
        avg_age = pp["age"].mean() if len(pp) > 0 else 0
        if cnt == 0:
            status, action = "\U0001f534 EMPTY", f"Urgently sign {pos}"
        elif cnt == 1:
            status, action = "\U0001f534 CRITICAL", f"Sign {pos} backup"
        elif avg_age >= 29:
            status, action = "\U0001f7e1 MONITOR", "Plan younger option"
        else:
            status, action = "\U0001f7e2 OK", "No action needed"
        rows.append({"Position": pos, "Players": cnt, "Status": status, "Action": action})
    return rows

def build_priority_cards():
    items = [
        ("GK", "Aging starter — El Shenawy is 36yo"),
        ("DM", "Only Aliou Dieng — no backup at DM"),
        ("AM", "Only Ben Romdhane — no backup at AM"),
        ("W",  "Average age 31 — squad aging at winger"),
    ]
    out = []
    for pos, reason in items:
        out.append(html.Div([
            html.Div([
                html.Span(pos, style={"background": COLOR_PRIMARY, "color": "white",
                                      "padding": "3px 10px", "borderRadius": "4px",
                                      "fontWeight": "800", "fontSize": "13px", "marginRight": "10px"}),
                html.Span(reason, style={"color": COLOR_MUTED, "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            dbc.Button(f"Find {pos} Targets →",
                       id=f"btn-t1-{pos.lower()}", color="danger", size="sm", n_clicks=0),
        ], style={**CARD_STYLE, "marginBottom": "8px"}))
    return out

tab1_layout = html.Div([
    dbc.Row([
        dbc.Col(kpi_card("Squad Size", len(SQUAD), "registered players"), md=3),
        dbc.Col(kpi_card("Foreign Players", "5 / 5", "at maximum (CAF rules)", color=COLOR_PRIMARY), md=3),
        dbc.Col(kpi_card("Age Risks", "1", "GK starter aged 36", color=COLOR_WARNING), md=3),
        dbc.Col(kpi_card("Priority Gaps", "2", "DM and AM need cover", color="#FF5252"), md=3),
    ], className="g-3", style={"marginBottom": "16px"}),

    dbc.Row([
        dbc.Col([html.Div([
            html.H5("\U000026bd Starting XI", style={**HDR, "marginBottom": "12px"}),
            build_pitch(),
        ], style=CARD_STYLE)], md=6),

        dbc.Col([html.Div([
            html.H5("\U0001f4ca Position Depth", style={**HDR, "marginBottom": "12px"}),
            dash_table.DataTable(
                id="gap-table", data=build_gap_data(),
                columns=[{"name": c, "id": c} for c in ["Position", "Players", "Status", "Action"]],
                style_table={"overflowX": "auto"},
                style_header={"background": COLOR_BORDER, "color": COLOR_TEXT,
                               "fontWeight": "700", "border": "none"},
                style_cell={"background": COLOR_CARD, "color": COLOR_TEXT,
                             "border": f"1px solid {COLOR_BORDER}", "padding": "8px 12px", "fontSize": "13px"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Status} contains "CRITICAL" || {Status} contains "EMPTY"'},
                     "color": COLOR_PRIMARY},
                    {"if": {"filter_query": '{Status} contains "MONITOR"'}, "color": COLOR_WARNING},
                    {"if": {"filter_query": '{Status} contains "OK"'}, "color": COLOR_SUCCESS},
                ],
            ),
        ], style=CARD_STYLE)], md=6),
    ]),

    dbc.Row([dbc.Col([html.Div([
        html.H5("\U0001f3af Priority Signings", style={**HDR, "marginBottom": "12px"}),
        *build_priority_cards(),
    ], style=CARD_STYLE)], md=12)]),
], style={"padding": "16px"})

# ── Tab 2: Player Discovery ───────────────────────────────────────────────────
tab2_layout = html.Div([
    dbc.Row([
        dbc.Col([html.Div([
            html.H5("\U0001f50d Scout Search", style={**HDR, "marginBottom": "14px"}),
            html.Label("Position Group *",
                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            dcc.Dropdown(id="t2-pos", options=POS_OPTIONS, placeholder="Select position...",
                         clearable=False, style={"marginBottom": "12px"}),
            html.Label("Leagues", style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            dcc.Checklist(id="t2-leagues",
                          options=[{"label": l, "value": l} for l in LEAGUES_IN_DATA],
                          value=LEAGUES_IN_DATA,
                          style={"color": COLOR_TEXT, "fontSize": "12px"},
                          inputStyle={"marginRight": "6px"},
                          labelStyle={"display": "block", "marginBottom": "3px"}),
            html.Div(style={"height": "10px"}),
            html.Label("Max Age", style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            html.Div(id="t2-age-lbl", style={"color": COLOR_WARNING, "fontSize": "11px"}),
            dcc.Slider(id="t2-age", min=16, max=35, value=30, step=1,
                       marks={16: "16", 20: "20", 25: "25", 30: "30", 35: "35"}),
            html.Label("Max Market Value",
                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600", "marginTop": "10px"}),
            html.Div(id="t2-mv-lbl", style={"color": COLOR_WARNING, "fontSize": "11px"}),
            dcc.Slider(id="t2-mv", min=0, max=15, value=3, step=0.5,
                       marks={0: "€0", 5: "€5m", 10: "€10m", 15: "€15m"}),
            html.Label("Min Minutes",
                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600", "marginTop": "10px"}),
            dcc.Slider(id="t2-mins", min=900, max=3000, value=900, step=100,
                       marks={900: "900", 1800: "1800", 3000: "3000"}),
            html.Div(style={"height": "8px"}),
            html.Label("Contract Status",
                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            dcc.RadioItems(id="t2-contract",
                           options=[{"label": "All players", "value": "all"},
                                    {"label": "Expiring only (⚡)", "value": "expiring"}],
                           value="all", inputStyle={"marginRight": "6px"},
                           labelStyle={"display": "block", "marginBottom": "3px"},
                           style={"color": COLOR_TEXT, "fontSize": "12px"}),
            html.Div(style={"height": "8px"}),
            html.Label("Valuation", style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            dcc.Checklist(id="t2-val",
                          options=["Undervalued", "Fair Value", "Overvalued", "Unknown"],
                          value=["Undervalued", "Fair Value", "Overvalued", "Unknown"],
                          inputStyle={"marginRight": "6px"},
                          labelStyle={"display": "block", "marginBottom": "3px"},
                          style={"color": COLOR_TEXT, "fontSize": "12px"}),
            html.Div(style={"height": "8px"}),
            html.Label("Sort By", style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
            dcc.Dropdown(id="t2-sort", options=SORT_OPTIONS,
                         value="adjusted_scouting_score", clearable=False),
            html.Div(style={"height": "12px"}),
            dbc.Button("Apply Filters", id="t2-apply", color="danger", className="w-100 mb-2"),
            dbc.Button("Reset", id="t2-reset", color="secondary", className="w-100", size="sm"),
        ], style={**CARD_STYLE, "height": "100%", "overflowY": "auto", "maxHeight": "90vh"})], md=3),

        dbc.Col([html.Div([
            dbc.Row([
                dbc.Col(html.Div(id="t2-hdr",
                                 children=html.P("Select a position to begin scouting.",
                                                 style={"color": COLOR_MUTED, "margin": "0"}))),
                dbc.Col([dbc.Button("↓ Export CSV", id="t2-export-btn",
                                    color="secondary", size="sm"),
                         dcc.Download(id="t2-dl")], width="auto"),
            ], align="center", className="mb-3"),
            dbc.Input(id="t2-search", placeholder="Search player or club...", type="text",
                      style={"background": COLOR_DARK, "color": COLOR_TEXT,
                             "border": f"1px solid {COLOR_BORDER}", "marginBottom": "8px"}),
            html.Div(id="t2-tbl-wrap",
                     children=html.P("No position selected.",
                                     style={"color": COLOR_MUTED, "textAlign": "center", "padding": "40px 0"})),
        ], style=CARD_STYLE)], md=9),
    ]),
], style={"padding": "16px"})

# ── Tab 3: Player Profile ─────────────────────────────────────────────────────
tab3_layout = html.Div([
    html.Div(id="t3-body", children=[
        html.P("Click a player row in Player Discovery to view their profile.",
               style={"color": COLOR_MUTED, "textAlign": "center",
                      "padding": "80px 0", "fontSize": "16px"}),
    ]),
], style={"padding": "16px"})

# ── Tab 4: Replacement Finder ─────────────────────────────────────────────────
_all_opts = [
    {"label": f"{r.player} ({r.team}, {r.position_group})", "value": r.player}
    for _, r in DF.sort_values("adjusted_scouting_score", ascending=False).iterrows()
]

tab4_layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("\U0001f504 Reference Player", style={**HDR, "marginBottom": "12px"}),
                dcc.Dropdown(id="t4-player", options=_all_opts,
                             placeholder="Search any player...",
                             style={"marginBottom": "8px"}, optionHeight=45),
                html.Div(id="t4-mini"),
            ], style=CARD_STYLE),
            html.Div([
                html.H5("\U0001f3af Criteria", style={**HDR, "marginBottom": "12px"}),
                html.Label("Max Budget",
                            style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"}),
                html.Div(id="t4-bgt-lbl", style={"color": COLOR_WARNING, "fontSize": "11px"}),
                dcc.Slider(id="t4-budget", min=0, max=15, value=3, step=0.5,
                           marks={0: "€0", 5: "€5m", 10: "€10m", 15: "€15m"}),
                html.Label("Max Age",
                            style={"color": COLOR_MUTED, "fontSize": "12px",
                                   "fontWeight": "600", "marginTop": "12px"}),
                dcc.Slider(id="t4-age", min=16, max=35, value=28, step=1,
                           marks={16: "16", 20: "20", 25: "25", 28: "28", 35: "35"}),
                dbc.Row([
                    dbc.Col(html.Label("Different League",
                                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"})),
                    dbc.Col(dcc.RadioItems(id="t4-diff",
                                          options=[{"label": "Yes", "value": True},
                                                   {"label": "No", "value": False}],
                                          value=False, inline=True,
                                          inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                                          style={"color": COLOR_TEXT, "fontSize": "12px"})),
                ], align="center", style={"marginTop": "12px"}),
                dbc.Row([
                    dbc.Col(html.Label("Target Leagues Only",
                                       style={"color": COLOR_MUTED, "fontSize": "12px", "fontWeight": "600"})),
                    dbc.Col(dcc.RadioItems(id="t4-tgt",
                                          options=[{"label": "Yes", "value": True},
                                                   {"label": "No", "value": False}],
                                          value=True, inline=True,
                                          inputStyle={"marginRight": "4px", "marginLeft": "8px"},
                                          style={"color": COLOR_TEXT, "fontSize": "12px"})),
                ], align="center", style={"marginTop": "8px"}),
                dbc.Button("Find Replacements", id="t4-find",
                           color="danger", className="w-100 mt-3"),
            ], style=CARD_STYLE),
        ], md=4),

        dbc.Col([
            html.Div(id="t4-res", style=CARD_STYLE,
                     children=html.P("Select a player and click Find Replacements.",
                                     style={"color": COLOR_MUTED, "textAlign": "center", "padding": "60px 0"})),
        ], md=8),
    ]),
], style={"padding": "16px"})

# ── Tab 5: Market Intelligence ─────────────────────────────────────────────────
tab5_layout = html.Div([
    dbc.Row([
        dbc.Col([html.Div([
            html.H5("\U0001f4b0 Best Value Players (MV ≤ €3m)",
                    style={**HDR, "marginBottom": "8px"}),
            dcc.Graph(id="t5-bv", config={"displayModeBar": False}),
        ], style=CARD_STYLE)], md=6),
        dbc.Col([html.Div([
            html.H5("\U0001f4c9 Value Gap by League", style={**HDR, "marginBottom": "8px"}),
            dcc.Graph(id="t5-vg", config={"displayModeBar": False}),
        ], style=CARD_STYLE)], md=6),
    ]),
    dbc.Row([
        dbc.Col([html.Div([
            dbc.Row([
                dbc.Col(html.H5("⚡ Contract Expiring Targets", style=HDR)),
                dbc.Col(dcc.Dropdown(id="t5-pos-flt",
                                     options=[{"label": "All Positions", "value": "ALL"}] + POS_OPTIONS,
                                     value="ALL", clearable=False,
                                     style={"width": "160px"}), width="auto"),
            ], align="center", className="mb-2"),
            html.Div(id="t5-exp-tbl"),
        ], style=CARD_STYLE)], md=6),
        dbc.Col([html.Div([
            html.H5("\U0001f3af Age-Value Sweet Spot", style={**HDR, "marginBottom": "8px"}),
            dcc.Graph(id="t5-sc", config={"displayModeBar": False}),
        ], style=CARD_STYLE)], md=6),
    ]),
], style={"padding": "16px"})

# ── App layout ────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="ScoutEdge v2",
)
server = app.server

app.layout = html.Div([
    dcc.Store(id="store-player", data=""),
    dcc.Store(id="store-t2pos", data=""),
    dcc.Store(id="store-t2data", data=[]),

    html.Div(style={"height": "3px", "background": COLOR_PRIMARY}),
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Span("⚽ ", style={"fontSize": "22px"}),
                html.Span("ScoutEdge",
                          style={"color": COLOR_PRIMARY, "fontWeight": "800", "fontSize": "20px"}),
                html.Span(" v2", style={"color": COLOR_MUTED, "fontSize": "13px", "marginLeft": "4px"}),
            ], width="auto"),
            dbc.Col(html.P("Al Ahly SC • Recruitment Intelligence • 2024-25",
                           style={"color": COLOR_MUTED, "margin": "0", "fontSize": "13px"})),
            dbc.Col(html.P(f"{len(DF):,} players • {DF['market_value_m'].notna().sum():,} valued",
                           style={"color": COLOR_MUTED, "margin": "0",
                                  "fontSize": "12px", "textAlign": "right"})),
        ], align="center", style={"padding": "10px 0"}),
    ], fluid=True,
       style={"background": COLOR_CARD, "borderBottom": f"1px solid {COLOR_BORDER}"}),

    dcc.Tabs(id="tabs", value="t1",
             children=[
                 dcc.Tab(label="\U0001f3df  Squad Intelligence",   value="t1"),
                 dcc.Tab(label="\U0001f50d  Player Discovery",     value="t2"),
                 dcc.Tab(label="\U0001f464  Player Profile",       value="t3"),
                 dcc.Tab(label="\U0001f504  Replacement Finder",   value="t4"),
                 dcc.Tab(label="\U0001f4ca  Market Intelligence",  value="t5"),
             ],
             colors={"border": COLOR_BORDER, "primary": COLOR_PRIMARY, "background": COLOR_DARK}),

    html.Div(id="tab-body",
             style={"background": COLOR_DARK, "minHeight": "calc(100vh - 100px)"}),
], style={"background": COLOR_DARK, "fontFamily": "Inter,system-ui,sans-serif", "minHeight": "100vh"})


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tab(t):
    if t == "t1": return tab1_layout
    if t == "t2": return tab2_layout
    if t == "t3": return tab3_layout
    if t == "t4": return tab4_layout
    if t == "t5": return tab5_layout
    return tab1_layout


# Tab1: "Find Targets" buttons navigate to Tab2 with position pre-filled
@app.callback(
    Output("tabs", "value", allow_duplicate=True),
    Output("store-t2pos", "data", allow_duplicate=True),
    [Input(f"btn-t1-{p}", "n_clicks") for p in ["gk", "cb", "fb", "dm", "cm", "am", "w", "st"]],
    prevent_initial_call=True,
)
def find_targets(*args):
    tid = dash.callback_context.triggered_id
    if not tid:
        return no_update, no_update
    pos = tid.replace("btn-t1-", "").upper()
    return "t2", pos


@app.callback(Output("t2-pos", "value", allow_duplicate=True),
              Input("store-t2pos", "data"), prevent_initial_call=True)
def prefill_pos(pos):
    return pos if pos else no_update


# Tab2: slider labels
@app.callback(Output("t2-age-lbl", "children"), Input("t2-age", "value"))
def t2_age_lbl(v): return f"Max age: {v}"

@app.callback(Output("t2-mv-lbl", "children"), Input("t2-mv", "value"))
def t2_mv_lbl(v): return f"Max: €{v:.1f}m"

@app.callback(Output("t4-bgt-lbl", "children"), Input("t4-budget", "value"))
def t4_bgt_lbl(v): return f"Budget: €{v:.1f}m"


# Tab2: reset filters
@app.callback(
    Output("t2-pos", "value"),
    Output("t2-age", "value"),
    Output("t2-mv", "value"),
    Output("t2-mins", "value"),
    Output("t2-contract", "value"),
    Output("t2-sort", "value"),
    Input("t2-reset", "n_clicks"),
    prevent_initial_call=True,
)
def t2_reset(_):
    return None, 30, 3, 900, "all", "adjusted_scouting_score"


# Tab2: main filter callback
@app.callback(
    Output("t2-tbl-wrap", "children"),
    Output("t2-hdr", "children"),
    Output("store-t2data", "data"),
    Input("t2-apply", "n_clicks"),
    Input("t2-pos", "value"),
    State("t2-leagues", "value"),
    State("t2-age", "value"),
    State("t2-mv", "value"),
    State("t2-mins", "value"),
    State("t2-contract", "value"),
    State("t2-val", "value"),
    State("t2-sort", "value"),
    State("t2-search", "value"),
    prevent_initial_call=False,
)
def t2_filter(n, pos, leagues, max_age, max_mv, min_mins, contract, valuation, sort_col, search):
    no_pos = html.P("Select a position to begin scouting.",
                    style={"color": COLOR_MUTED, "textAlign": "center", "padding": "40px 0"})
    if not pos:
        return no_pos, html.P("Select a position.", style={"color": COLOR_MUTED}), []

    filt = DF[DF["position_group"] == pos].copy()
    if leagues:
        filt = filt[filt["league_clean"].isin(leagues)]
    if max_age:
        filt = filt[filt["age"] <= max_age]
    if max_mv is not None:
        filt = filt[filt["market_value_m"].isna() | (filt["market_value_m"] <= max_mv)]
    if min_mins:
        filt = filt[filt["minutes"] >= min_mins]
    if contract == "expiring":
        filt = filt[filt["contract_expiring"] == True]
    if valuation:
        filt = filt[filt["valuation_status"].isin(valuation)]
    if search:
        s = search.lower()
        filt = filt[filt["player"].str.lower().str.contains(s, na=False) |
                    filt["team"].str.lower().str.contains(s, na=False)]

    asc = sort_col in ["age", "market_value_m"]
    if sort_col and sort_col in filt.columns:
        filt = filt.sort_values(sort_col, ascending=asc, na_position="last")

    if len(filt) == 0:
        return (html.P("No players match these filters.",
                       style={"color": COLOR_MUTED, "textAlign": "center", "padding": "40px 0"}),
                html.P("0 players found.", style={"color": COLOR_MUTED}), [])

    tdata = []
    for _, r in filt.iterrows():
        tdata.append({
            "Player": r["player"],
            "Club":   r.get("team", ""),
            "League": r.get("league_clean", ""),
            "Age":    int(r["age"]) if pd.notna(r.get("age")) else "",
            "Nat":    str(r.get("nationality", ""))[:3] if pd.notna(r.get("nationality")) else "",
            "Pos":    r.get("position_group", ""),
            "MV":     fmt_mv(r.get("market_value_m")),
            "Raw":    f"{r['raw_scouting_score']:.0f}" if pd.notna(r.get("raw_scouting_score")) else "-",
            "Adj":    f"{r['adjusted_scouting_score']:.0f}" if pd.notna(r.get("adjusted_scouting_score")) else "-",
            "Status": r.get("valuation_status", "Unknown"),
            "Ctrt":   "⚡" if r.get("contract_expiring") else "",
        })

    tbl = dash_table.DataTable(
        id="t2-tbl", data=tdata,
        columns=[{"name": c, "id": c}
                 for c in ["Player", "Club", "League", "Age", "Nat", "Pos",
                            "MV", "Raw", "Adj", "Status", "Ctrt"]],
        page_size=25, page_action="native", sort_action="native",
        row_selectable="single", selected_rows=[],
        style_table={"overflowX": "auto"},
        style_header={"background": COLOR_BORDER, "color": COLOR_TEXT,
                       "fontWeight": "700", "border": "none", "fontSize": "12px"},
        style_cell={"background": COLOR_CARD, "color": COLOR_TEXT,
                    "border": f"1px solid {COLOR_BORDER}",
                    "padding": "8px 10px", "fontSize": "12px",
                    "maxWidth": "160px", "whiteSpace": "normal"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": "#111111"},
            {"if": {"column_id": "Status", "filter_query": '{Status} = "Undervalued"'},
             "color": COLOR_SUCCESS},
            {"if": {"column_id": "Status", "filter_query": '{Status} = "Overvalued"'},
             "color": COLOR_PRIMARY},
            {"if": {"column_id": "Ctrt"}, "color": COLOR_WARNING, "fontSize": "14px"},
            {"if": {"state": "selected"},
             "background": "#2A1A1A", "border": f"1px solid {COLOR_PRIMARY}"},
        ],
        style_filter={"background": COLOR_DARK, "color": COLOR_TEXT,
                      "border": f"1px solid {COLOR_BORDER}"},
    )

    hdr = html.P(f"{len(filt)} {pos} players found",
                 style={"color": COLOR_TEXT, "fontWeight": "600", "margin": "0"})
    return tbl, hdr, tdata


# Tab2: row click -> Tab3
@app.callback(
    Output("store-player", "data"),
    Output("tabs", "value", allow_duplicate=True),
    Input("t2-tbl", "selected_rows"),
    State("store-t2data", "data"),
    prevent_initial_call=True,
)
def t2_row_click(rows, tdata):
    if not rows or not tdata:
        return no_update, no_update
    return tdata[rows[0]]["Player"], "t3"


# Tab2: export CSV
@app.callback(Output("t2-dl", "data"),
              Input("t2-export-btn", "n_clicks"),
              State("store-t2data", "data"),
              prevent_initial_call=True)
def t2_export(n, tdata):
    if not n or not tdata:
        return no_update
    return dcc.send_data_frame(pd.DataFrame(tdata).to_csv,
                               "scoutedge_discovery.csv", index=False)


# Tab3: player profile
@app.callback(Output("t3-body", "children"), Input("store-player", "data"))
def t3_profile(player_name):
    if not player_name:
        return html.P("Click a player row in Player Discovery to view their profile.",
                      style={"color": COLOR_MUTED, "textAlign": "center",
                             "padding": "80px 0", "fontSize": "16px"})

    mask = DF["player"].str.lower().str.strip() == player_name.lower().strip()
    if not mask.any():
        mask = DF["player"].str.lower().str.contains(
            player_name.lower().strip(), na=False, regex=False)
    if not mask.any():
        return html.P(f"'{player_name}' not found.",
                      style={"color": COLOR_PRIMARY, "textAlign": "center", "padding": "40px 0"})

    r = DF[mask].iloc[0]
    pos = r.get("position_group", "CM")
    pos_df = DF[DF["position_group"] == pos]

    # Header
    exp_badge = (html.Span(" ⚡ Expiring",
                            style={"background": f"{COLOR_WARNING}22", "color": COLOR_WARNING,
                                   "padding": "2px 8px", "borderRadius": "12px",
                                   "fontSize": "11px", "marginLeft": "8px"})
                 if r.get("contract_expiring") else html.Span())

    hdr = html.Div([
        dbc.Row([
            dbc.Col([
                html.H3(r["player"], style={"color": COLOR_TEXT, "margin": "0", "fontWeight": "800"}),
                html.P(f"{r.get('team', '')} • {r.get('league_clean', '')} • "
                       f"Age {int(r['age']) if pd.notna(r.get('age')) else '?'} • "
                       f"{r.get('nationality', '')}",
                       style={"color": COLOR_MUTED, "margin": "4px 0 0", "fontSize": "13px"}),
            ]),
            dbc.Col([
                html.Div([
                    html.Span(fmt_mv(r.get("market_value_m")),
                              style={"background": "#1565C022", "color": "#5C9FE8",
                                     "padding": "4px 12px", "borderRadius": "6px",
                                     "fontWeight": "700", "fontSize": "15px", "marginRight": "8px"}),
                    exp_badge,
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                html.P(f"Adj Score: {r['adjusted_scouting_score']:.0f}  •  "
                       f"Raw: {r['raw_scouting_score']:.0f}  •  {r.get('valuation_status', '')}",
                       style={"color": COLOR_MUTED, "fontSize": "12px",
                              "margin": "6px 0 0", "textAlign": "right"}),
            ]),
        ]),
    ], style={**CARD_STYLE, "marginBottom": "12px"})

    # Stat bars
    weights = SCOUTING_WEIGHTS.get(pos, {})
    bars = []
    for feat in weights:
        val = r.get(feat)
        pct = get_pct(pos_df, val, feat)
        bar_c = (COLOR_SUCCESS if pct >= 67
                 else COLOR_WARNING if pct >= 34
                 else "#FF5252")
        display = f"{float(val):.2f}" if pd.notna(val) else "N/A"
        label = feat.replace("_p90", "").replace("_", " ").title()
        bars.append(html.Div([
            dbc.Row([
                dbc.Col(html.Span(label, style={"color": COLOR_TEXT, "fontSize": "11px"}), md=5),
                dbc.Col(html.Span(display, style={"color": COLOR_MUTED,
                                                   "fontSize": "10px", "textAlign": "right"}), md=2),
                dbc.Col([html.Div(
                    style={"background": COLOR_BORDER, "borderRadius": "3px",
                           "height": "7px", "overflow": "hidden"},
                    children=[html.Div(style={"width": f"{pct:.0f}%",
                                              "background": bar_c, "height": "100%"})])], md=3),
                dbc.Col(html.Span(f"T{100-int(pct)}%",
                                  style={"color": COLOR_MUTED, "fontSize": "9px",
                                         "textAlign": "right"}), md=2),
            ], align="center", style={"marginBottom": "7px"}),
        ]))

    # Radar chart
    metrics = RADAR_METRICS.get(pos, RADAR_METRICS["CM"])
    rvals = [round(get_pct(pos_df, r.get(col), col), 1) for _, col in metrics]
    rlbls = [lbl for lbl, _ in metrics]
    radar_fig = go.Figure(go.Scatterpolar(
        r=rvals + [rvals[0]], theta=rlbls + [rlbls[0]],
        fill="toself",
        line=dict(color=COLOR_PRIMARY, width=2),
        fillcolor="rgba(204,0,0,0.15)",
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color=COLOR_MUTED,
                            gridcolor=COLOR_BORDER, tickfont=dict(size=9, color=COLOR_MUTED)),
            angularaxis=dict(color=COLOR_TEXT, gridcolor=COLOR_BORDER),
            bgcolor=COLOR_CARD,
        ),
        paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, family="Inter,system-ui"),
        showlegend=False, margin=dict(l=50, r=50, t=20, b=20), height=290,
    )

    # vs Al Ahly comparison
    ah_pos = SQUAD[SQUAD["position_group"] == pos]
    if len(ah_pos) == 0:
        comp = html.P(f"No Al Ahly {pos} to compare — this would be a new position.",
                      style={"color": COLOR_MUTED, "fontSize": "12px", "fontStyle": "italic"})
    else:
        ah = ah_pos.nlargest(1, "minutes_24_25").iloc[0]
        comp_rows = []
        for feat in list(weights.keys())[:5]:
            if feat == "goals_p90":
                ah_v = (ah["goals_24_25"] / (ah["minutes_24_25"] / 90)
                        if ah["minutes_24_25"] > 0 else 0)
            elif feat == "assists_p90":
                ah_v = (ah["assists_24_25"] / (ah["minutes_24_25"] / 90)
                        if ah["minutes_24_25"] > 0 else 0)
            else:
                ah_v = None
            target_v = r.get(feat)
            lbl = feat.replace("_p90", "").replace("_", " ").title()
            ah_s = f"{ah_v:.2f}" if ah_v is not None else "-"
            tgt_s = f"{float(target_v):.2f}" if pd.notna(target_v) else "-"
            if ah_v is not None and pd.notna(target_v):
                tc = COLOR_SUCCESS if float(target_v) > ah_v else "#FF5252"
            else:
                tc = COLOR_MUTED
            comp_rows.append(html.Tr([
                html.Td(lbl, style={"padding": "5px 8px", "color": COLOR_TEXT, "fontSize": "11px"}),
                html.Td(ah_s, style={"padding": "5px 8px", "color": COLOR_MUTED,
                                      "fontSize": "11px", "textAlign": "center"}),
                html.Td(tgt_s, style={"padding": "5px 8px", "color": tc,
                                       "fontSize": "11px", "textAlign": "center",
                                       "fontWeight": "600"}),
            ]))
        ah_db = DF[DF["player"].str.lower() == ah["player"].lower()]
        ah_sc = float(ah_db["adjusted_scouting_score"].iloc[0]) if len(ah_db) > 0 else None
        tsc = float(r.get("adjusted_scouting_score", 0))
        if ah_sc is not None:
            diff = tsc - ah_sc
            verdict = ("UPGRADE" if diff > 5
                       else "DOWNGRADE" if diff < -5
                       else "SIMILAR")
            vc = (COLOR_SUCCESS if verdict == "UPGRADE"
                  else COLOR_PRIMARY if verdict == "DOWNGRADE"
                  else COLOR_WARNING)
            comp_rows.append(html.Tr([
                html.Td("Adj Score", style={"padding": "5px 8px", "color": COLOR_TEXT,
                                             "fontSize": "11px", "fontWeight": "700"}),
                html.Td(f"{ah_sc:.0f}", style={"padding": "5px 8px", "color": COLOR_TEXT,
                                                "textAlign": "center", "fontWeight": "700"}),
                html.Td(f"{tsc:.0f}", style={"padding": "5px 8px", "color": vc,
                                              "textAlign": "center", "fontWeight": "700"}),
            ]))
            vbadge = html.Div(verdict,
                              style={"background": f"{vc}22", "color": vc,
                                     "padding": "4px 16px", "borderRadius": "4px",
                                     "fontWeight": "800", "textAlign": "center",
                                     "marginTop": "8px", "fontSize": "13px"})
        else:
            vbadge = html.Span()
        comp = html.Div([
            html.P(f"vs {ah['player']}", style={"color": COLOR_MUTED, "fontSize": "11px",
                                                  "marginBottom": "6px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Stat", style={"padding": "5px 8px", "color": COLOR_MUTED,
                                           "fontSize": "10px", "background": COLOR_BORDER}),
                    html.Th(ah["player"].split()[0],
                            style={"padding": "5px 8px", "color": COLOR_MUTED,
                                   "fontSize": "10px", "background": COLOR_BORDER,
                                   "textAlign": "center"}),
                    html.Th("Target", style={"padding": "5px 8px", "color": COLOR_MUTED,
                                              "fontSize": "10px", "background": COLOR_BORDER,
                                              "textAlign": "center"}),
                ])),
                html.Tbody(comp_rows),
            ], style={"width": "100%", "borderCollapse": "collapse"}),
            vbadge,
        ])

    row1 = dbc.Row([
        dbc.Col([html.Div([
            html.H6("Scouting Stats",
                    style={**HDR, "fontSize": "12px", "marginBottom": "10px"}),
            *bars,
        ], style=CARD_STYLE)], md=4),
        dbc.Col([html.Div([
            html.H6(f"Radar — {pos}",
                    style={**HDR, "fontSize": "12px", "marginBottom": "6px"}),
            dcc.Graph(figure=radar_fig, config={"displayModeBar": False}),
        ], style=CARD_STYLE)], md=4),
        dbc.Col([html.Div([
            html.H6("vs Al Ahly",
                    style={**HDR, "fontSize": "12px", "marginBottom": "6px"}),
            comp,
        ], style=CARD_STYLE)], md=4),
    ])

    # Similar players
    sim = get_similar_players(player_name, DF, MATRICES, n=8, target_leagues_only=True)
    if len(sim) == 0:
        sim_content = html.P("No similar players found in target leagues.",
                             style={"color": COLOR_MUTED})
    else:
        sdata = []
        for _, sr in sim.iterrows():
            sdata.append({
                "Player": sr["player"],
                "Club":   sr.get("team", ""),
                "League": sr.get("league_clean", ""),
                "Age":    int(sr["age"]) if pd.notna(sr.get("age")) else "",
                "MV":     fmt_mv(sr.get("market_value_m")),
                "Sim%":   f"{sr['similarity_pct']:.0f}%",
                "Adj":    f"{sr['adjusted_scouting_score']:.0f}",
                "Status": sr.get("valuation_status", ""),
            })
        sim_content = dash_table.DataTable(
            data=sdata,
            columns=[{"name": c, "id": c}
                     for c in ["Player", "Club", "League", "Age", "MV", "Sim%", "Adj", "Status"]],
            style_table={"overflowX": "auto"},
            style_header={"background": COLOR_BORDER, "color": COLOR_TEXT,
                           "fontWeight": "700", "border": "none", "fontSize": "11px"},
            style_cell={"background": COLOR_CARD, "color": COLOR_TEXT,
                        "border": f"1px solid {COLOR_BORDER}",
                        "padding": "7px 10px", "fontSize": "11px"},
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "background": "#111111"},
                {"if": {"column_id": "Status", "filter_query": '{Status} = "Undervalued"'},
                 "color": COLOR_SUCCESS},
            ],
        )

    row2 = html.Div([
        html.H5(f"Most Similar {pos} Players (Target Leagues)",
                style={**HDR, "marginBottom": "10px"}),
        sim_content,
    ], style=CARD_STYLE)

    return html.Div([hdr, row1, row2])


# Tab4: mini profile
@app.callback(Output("t4-mini", "children"), Input("t4-player", "value"))
def t4_mini(name):
    if not name:
        return html.Span()
    mask = DF["player"] == name
    if not mask.any():
        return html.Span()
    r = DF[mask].iloc[0]
    return html.Div([
        dbc.Row([
            dbc.Col(html.H6(r["player"], style={"color": COLOR_TEXT, "margin": "0", "fontWeight": "700"})),
            dbc.Col(html.Span(fmt_mv(r.get("market_value_m")),
                              style={"color": "#5C9FE8", "fontWeight": "600", "float": "right"})),
        ]),
        html.P(f"{r.get('position_group', '')} • {r.get('team', '')} • "
               f"Age {int(r['age']) if pd.notna(r.get('age')) else '?'} • "
               f"Adj: {r.get('adjusted_scouting_score', 0):.0f}",
               style={"color": COLOR_MUTED, "fontSize": "12px", "margin": "4px 0 0"}),
    ], style={"background": COLOR_DARK, "padding": "10px", "borderRadius": "6px",
              "border": f"1px solid {COLOR_BORDER}", "marginTop": "6px"})


# Tab4: find replacements
@app.callback(
    Output("t4-res", "children"),
    Input("t4-find", "n_clicks"),
    State("t4-player", "value"),
    State("t4-budget", "value"),
    State("t4-age", "value"),
    State("t4-diff", "value"),
    State("t4-tgt", "value"),
    prevent_initial_call=True,
)
def t4_find(n, name, budget, max_age, diff, tgt):
    if not n or not name:
        return html.P("Select a player first.",
                      style={"color": COLOR_MUTED, "textAlign": "center", "padding": "40px 0"})

    results = get_similar_players(
        name, DF, MATRICES, n=15,
        max_market_value_m=budget,
        max_age=max_age,
        different_league=bool(diff),
        target_leagues_only=bool(tgt),
    )

    mask = DF["player"] == name
    ref = DF[mask].iloc[0] if mask.any() else None
    ref_mv = ref.get("market_value_m", np.nan) if ref is not None else np.nan
    ref_pos = ref.get("position_group", "") if ref is not None else ""

    if len(results) == 0:
        return html.P(f"No replacements found for {name} within budget.",
                      style={"color": COLOR_MUTED, "textAlign": "center", "padding": "40px 0"})

    tdata = []
    for _, r in results.iterrows():
        cmv = r.get("market_value_m", np.nan)
        if pd.notna(ref_mv) and pd.notna(cmv):
            cheaper = ref_mv - cmv
            cs = (f"€{cheaper:.1f}m cheaper" if cheaper > 0
                  else f"€{abs(cheaper):.1f}m more")
        else:
            cs = "-"
        tdata.append({
            "Player":     r["player"],
            "Club":       r.get("team", ""),
            "League":     r.get("league_clean", ""),
            "Age":        int(r["age"]) if pd.notna(r.get("age")) else "",
            "MV":         fmt_mv(cmv),
            "Sim%":       f"{r['similarity_pct']:.0f}%",
            "Adj":        f"{r['adjusted_scouting_score']:.0f}",
            "Cheaper By": cs,
            "Status":     r.get("valuation_status", ""),
            "Ctrt":       "⚡" if r.get("contract_expiring") else "",
        })

    tbl = dash_table.DataTable(
        data=tdata,
        columns=[{"name": c, "id": c}
                 for c in ["Player", "Club", "League", "Age", "MV",
                            "Sim%", "Adj", "Cheaper By", "Status", "Ctrt"]],
        style_table={"overflowX": "auto"},
        style_header={"background": COLOR_BORDER, "color": COLOR_TEXT,
                       "fontWeight": "700", "border": "none", "fontSize": "11px"},
        style_cell={"background": COLOR_CARD, "color": COLOR_TEXT,
                    "border": f"1px solid {COLOR_BORDER}",
                    "padding": "7px 10px", "fontSize": "11px"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "background": "#111111"},
            {"if": {"column_id": "Status", "filter_query": '{Status} = "Undervalued"'},
             "color": COLOR_SUCCESS},
            {"if": {"column_id": "Ctrt"}, "color": COLOR_WARNING},
        ],
    )
    return html.Div([
        html.H5(f"Replacements for {name}", style={**HDR, "marginBottom": "4px"}),
        html.P(f"Same position ({ref_pos}) • Budget ≤ €{budget:.1f}m • "
               f"Age ≤ {max_age} • {len(results)} found",
               style={"color": COLOR_MUTED, "fontSize": "12px", "marginBottom": "10px"}),
        tbl,
    ])


# Tab5: charts
@app.callback(
    Output("t5-bv", "figure"),
    Output("t5-vg", "figure"),
    Output("t5-sc", "figure"),
    Input("tabs", "value"),
)
def t5_charts(tab):
    empty = go.Figure()
    empty.update_layout(paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_CARD,
                        font=dict(color=COLOR_TEXT))
    if tab != "t5":
        return empty, empty, empty

    pc = {"CB": "#5C9FE8", "FB": "#00C853", "DM": "#FFB300",
          "CM": "#FF7043", "AM": "#AB47BC", "W": COLOR_PRIMARY, "ST": "#26C6DA"}

    # Chart 1: best value under 3m
    bv = DF[DF["market_value_m"] <= 3.0].nlargest(15, "adjusted_scouting_score").copy()
    fig1 = go.Figure()
    for p in bv["position_group"].unique():
        sub = bv[bv["position_group"] == p]
        fig1.add_trace(go.Bar(
            x=sub["adjusted_scouting_score"], y=sub["player"],
            orientation="h", name=p, marker_color=pc.get(p, COLOR_MUTED),
            text=[f"{r2.team} | {fmt_mv(r2.market_value_m)}" for _, r2 in sub.iterrows()],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Score: %{x}<extra></extra>",
        ))
    fig1.update_layout(
        paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, size=11),
        xaxis=dict(color=COLOR_MUTED, gridcolor=COLOR_BORDER, range=[0, 108]),
        yaxis=dict(color=COLOR_TEXT, autorange="reversed"),
        barmode="overlay", showlegend=True,
        legend=dict(bgcolor=COLOR_CARD, bordercolor=COLOR_BORDER, font=dict(size=10)),
        margin=dict(l=10, r=100, t=10, b=30), height=400,
    )

    # Chart 2: value gap by league
    vg = DF[DF["value_gap_pct"].notna() & DF["league_clean"].notna()].copy()
    fig2 = go.Figure()
    for lg in sorted(vg["league_clean"].unique()):
        sub = vg[vg["league_clean"] == lg]["value_gap_pct"]
        col = COLOR_SUCCESS if sub.median() < 0 else COLOR_PRIMARY
        fig2.add_trace(go.Box(
            y=sub, name=lg[:14], marker_color=col,
            line_color=col, fillcolor=f"{col}22",
        ))
    fig2.update_layout(
        paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, size=11),
        yaxis=dict(color=COLOR_MUTED, gridcolor=COLOR_BORDER, title="Value Gap %"),
        xaxis=dict(color=COLOR_TEXT),
        showlegend=False, margin=dict(l=50, r=10, t=10, b=30), height=400,
    )

    # Chart 3: age-value scatter
    sc = DF[(DF["market_value_m"] <= 5) & DF["market_value_m"].notna()].copy()
    cmap = {"Undervalued": COLOR_SUCCESS, "Fair Value": COLOR_MUTED,
            "Overvalued": COLOR_PRIMARY, "Unknown": COLOR_BORDER}
    fig3 = go.Figure()
    fig3.add_shape(type="rect", x0=22, x1=27, y0=60, y1=100,
                   fillcolor="rgba(0,200,83,0.08)",
                   line=dict(color=COLOR_SUCCESS, width=1, dash="dot"))
    fig3.add_annotation(x=24.5, y=97, text="Al Ahly Target Zone",
                        font=dict(color=COLOR_SUCCESS, size=10), showarrow=False)
    for st in ["Undervalued", "Fair Value", "Overvalued"]:
        sub = sc[sc["valuation_status"] == st]
        if len(sub) == 0:
            continue
        fig3.add_trace(go.Scatter(
            x=sub["age"], y=sub["adjusted_scouting_score"],
            mode="markers", name=st,
            marker=dict(size=sub["market_value_m"].clip(0.5, 5) * 3,
                        color=cmap[st], opacity=0.7,
                        line=dict(width=0.5, color=COLOR_BORDER)),
            text=sub["player"] + "<br>" + sub["team"] + "<br>" + sub["market_value_m"].apply(fmt_mv),
            hovertemplate="<b>%{text}</b><br>Age: %{x}<br>Score: %{y}<extra></extra>",
        ))
    fig3.update_layout(
        paper_bgcolor=COLOR_CARD, plot_bgcolor=COLOR_CARD,
        font=dict(color=COLOR_TEXT, size=11),
        xaxis=dict(color=COLOR_MUTED, gridcolor=COLOR_BORDER, title="Age", range=[15, 36]),
        yaxis=dict(color=COLOR_MUTED, gridcolor=COLOR_BORDER, title="Adj Score", range=[0, 105]),
        legend=dict(bgcolor=COLOR_CARD, bordercolor=COLOR_BORDER, font=dict(size=10)),
        margin=dict(l=50, r=10, t=10, b=30), height=400,
    )
    return fig1, fig2, fig3


# Tab5: expiring contracts table
@app.callback(
    Output("t5-exp-tbl", "children"),
    Input("t5-pos-flt", "value"),
    Input("tabs", "value"),
)
def t5_exp(pos_f, tab):
    if tab != "t5":
        return html.Span()
    exp = DF[DF["contract_expiring"] == True].copy()
    if pos_f and pos_f != "ALL":
        exp = exp[exp["position_group"] == pos_f]
    exp = exp.sort_values("adjusted_scouting_score", ascending=False).head(20)
    if len(exp) == 0:
        return html.P("No expiring contracts found.", style={"color": COLOR_MUTED})
    tdata = [{
        "Player": r["player"], "Club": r.get("team", ""),
        "League": r.get("league_clean", ""),
        "Age":    int(r["age"]) if pd.notna(r.get("age")) else "",
        "Pos":    r["position_group"],
        "MV":     fmt_mv(r.get("market_value_m")),
        "Adj":    f"{r['adjusted_scouting_score']:.0f}",
        "Expiry": str(r.get("contract_expiry", ""))[:10],
    } for _, r in exp.iterrows()]
    return dash_table.DataTable(
        data=tdata,
        columns=[{"name": c, "id": c}
                 for c in ["Player", "Club", "League", "Age", "Pos", "MV", "Adj", "Expiry"]],
        style_table={"overflowX": "auto"},
        style_header={"background": COLOR_BORDER, "color": COLOR_TEXT,
                       "fontWeight": "700", "border": "none", "fontSize": "11px"},
        style_cell={"background": COLOR_CARD, "color": COLOR_TEXT,
                    "border": f"1px solid {COLOR_BORDER}",
                    "padding": "6px 8px", "fontSize": "11px"},
        style_data_conditional=[{"if": {"row_index": "odd"}, "background": "#111111"}],
        page_size=10,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)
