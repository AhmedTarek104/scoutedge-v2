# ScoutEdge v2 — Al Ahly Recruitment Intelligence Platform

A data-driven football recruitment platform built for Al Ahly SC, covering **3,016 players across 13 leagues** with market value intelligence, similarity search, and squad gap analysis.

**Live app:** [https://practical-hope-production-87ab.up.railway.app/](https://practical-hope-production-87ab.up.railway.app/)

---

## What it does

Al Ahly targets players from feeder leagues (Super Lig, Eredivisie, Primeira Liga, Superliga, Belgian Pro League, Serie A Brazil, Primera Division, Scottish Premiership) and the Big 5. This platform helps their scouting team:

- Discover undervalued players by position, age, league, and budget
- Profile any player with radar charts, percentile stats, and market value context
- Find the 8 most statistically similar players for any target
- Identify Al Ahly squad gaps and get position-specific recommendations
- Manage a 10-player shortlist with comparison tables
- Find affordable replacements for existing squad members

---

## Data coverage

| Metric | Value |
|---|---|
| Players | 3,016 (deduplicated from 3,090 raw records) |
| Leagues | 13 |
| Seasons | 2024–25 |
| Market value coverage | 90.3% (real + estimated) |
| Real Transfermarkt MVs | 63.5% (ID-matched, no name fuzzy matching) |
| Best MV prediction model | Gradient Boosting R² 0.82 |

**Leagues covered:**
- Target: Super Lig · Eredivisie · Primeira Liga · Superliga · Belgian Pro League · Serie A (Brazil) · Primera Division · Scottish Premiership
- Big 5: Premier League · La Liga · Bundesliga · Serie A · Ligue 1

---

## App tabs

| Tab | What it shows |
|---|---|
| Squad Intelligence | Al Ahly squad overview, age distribution, contract alerts, position gaps |
| Player Discovery | Filterable table — position, league, age, budget, contract, MV data type |
| Player Profile | Radar chart, percentile stats, similar players, comparison vs Al Ahly equivalent |
| Shortlist | Side-by-side comparison of up to 10 saved players |
| Replacement Finder | Budget-constrained replacement search for any player |
| Market Intelligence | Value vs performance scatter, undervalued player rankings |

---

## Market value approach

Players without Transfermarkt data show an estimated MV (`~€Xm est.`) derived from the median value of peers in the same position group, league tier, and age bucket. This keeps the table useful without misrepresenting unknown values.

ID-based matching pipeline: `FBref player name → fbref_id (8-char hex) → tm_player_id → market_value_m`. No fuzzy name matching.

---

## Tech stack

- **Backend / data:** Python, pandas, scikit-learn, numpy
- **Similarity engine:** cosine similarity on per-position feature sets, position-filtered
- **UI:** Plotly Dash + Dash Bootstrap Components
- **Data sources:** FBref (stats), Transfermarkt via Kaggle (market values + contracts)
- **Deployment:** Railway, gunicorn

---

## Local setup

```bash
git clone https://github.com/AhmedTarek104/scoutedge-v2.git
cd scoutedge-v2
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:8050
```

The app auto-builds the similarity matrices on first run if the cache doesn't exist (~30 seconds).

---

## Repository structure

```
app.py                  # Dash application (single file)
Procfile                # Railway / gunicorn entrypoint
requirements.txt
src/
  config.py             # League tiers, position maps, scouting weights
  load_data.py          # Data pipeline: FBref + Transfermarkt ID join
  features.py           # Scouting score computation, age-adjusted scoring
  similarity.py         # Cosine similarity engine, position filtering
  squad_analysis.py     # Al Ahly squad gap detection
data/
  al_ahly_squad.csv     # Current Al Ahly squad (24/25)
  final/
    players_final.csv   # 3,090 raw records → 3,016 after dedup, 69 features
```
