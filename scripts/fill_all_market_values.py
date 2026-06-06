"""
fill_all_market_values.py

Fill missing market_value_m in players_final.csv using all available sources.

SOURCE 1 -- players.csv (TM snapshot)       exact + fuzzy-75
SOURCE 2 -- fbref_tm_mapping + valuations   ID-based, no fuzzy needed
SOURCE 3 -- appearances.csv + valuations    club-assisted ID lookup
SOURCE 4 -- Transfermarkt web scraping      league market-value pages
SOURCE 5 -- SofaScore public API            player search + stats
"""

import sys, time, unicodedata, re, json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from config import DATA_FINAL, DATA_RAW

try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    print("WARNING: rapidfuzz not installed -- fuzzy matching disabled")

# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

def safe_print(msg: str):
    print(msg.encode("ascii", "replace").decode())

STATS = {}   # source -> count filled

def fill_from_dict(df: pd.DataFrame, lookup: dict, source: str,
                   threshold: int = 0) -> pd.DataFrame:
    """Fill null market_value_m rows from a {normalized_name: mv} dict."""
    filled = 0
    null_idx = df[df["market_value_m"].isna()].index
    for idx in null_idx:
        key = normalize(df.at[idx, "player"])
        if key in lookup and pd.notna(lookup[key]) and lookup[key] > 0:
            df.at[idx, "market_value_m"] = lookup[key]
            filled += 1
    STATS[source] = filled
    print(f"  {source}: filled {filled} of {len(null_idx)} remaining nulls")
    return df

def fuzzy_fill(df: pd.DataFrame, tm_names: list, tm_mv: dict,
               source: str, threshold: int = 75) -> pd.DataFrame:
    """Fill null rows with fuzzy match at given threshold."""
    if not HAS_RAPIDFUZZ:
        STATS[source] = 0
        return df
    tm_norm = [normalize(n) for n in tm_names]
    filled = 0
    null_idx = df[df["market_value_m"].isna()].index
    for idx in null_idx:
        key = normalize(df.at[idx, "player"])
        result = process.extractOne(key, tm_norm, scorer=fuzz.WRatio,
                                    score_cutoff=threshold)
        if result:
            orig = tm_names[result[2]]
            mv = tm_mv.get(orig)
            if pd.notna(mv) and mv > 0:
                df.at[idx, "market_value_m"] = mv
                filled += 1
    STATS[source] = filled
    print(f"  {source}: filled {filled} of {len(null_idx)} remaining nulls")
    return df

# ── Load players_final ────────────────────────────────────────────────────────

print("=" * 60)
print("Loading players_final.csv")
print("=" * 60)
df = pd.read_csv(DATA_FINAL / "players_final.csv", low_memory=False)
df["market_value_m"] = pd.to_numeric(df["market_value_m"], errors="coerce")
n_null_start = int(df["market_value_m"].isna().sum())
print(f"Players: {len(df)}  |  With MV: {(~df['market_value_m'].isna()).sum()}  |  Null: {n_null_start}")
print("\nNull by league:")
print(df[df["market_value_m"].isna()]["league_clean"].value_counts().to_string())

# =============================================================================
# SOURCE 1 -- players.csv (Transfermarkt snapshot)
# =============================================================================
print("\n" + "=" * 60)
print("SOURCE 1 -- players.csv  (exact-normalized + fuzzy-75)")
print("=" * 60)

tm = pd.read_csv(DATA_RAW / "players.csv", low_memory=False,
                 encoding_errors="replace")
print(f"  Columns: {list(tm.columns)}")
print(f"  market_value_in_eur exists: {'market_value_in_eur' in tm.columns}")

tm["market_value_m"] = pd.to_numeric(tm["market_value_in_eur"], errors="coerce") / 1e6
tm_lookup = (
    tm[["name", "market_value_m"]]
    .dropna(subset=["name"])
    .sort_values("market_value_m", ascending=False, na_position="last")
    .drop_duplicates(subset=["name"], keep="first")
)
tm_names = tm_lookup["name"].tolist()
tm_norm_map = {normalize(n): n for n in tm_names}
tm_mv_by_name = dict(zip(tm_names, tm_lookup["market_value_m"]))

# 1a -- exact normalized
lookup_exact = {normalize(n): mv for n, mv in tm_mv_by_name.items() if pd.notna(mv)}
df = fill_from_dict(df, lookup_exact, "S1-exact-normalized")

# 1b -- fuzzy at threshold 75 (catches romanized/partially-accented variants)
df = fuzzy_fill(df, tm_names, tm_mv_by_name, "S1-fuzzy-75", threshold=75)

# =============================================================================
# SOURCE 2 -- fbref_tm_mapping.csv -> player_valuations.csv (ID-based)
# =============================================================================
print("\n" + "=" * 60)
print("SOURCE 2 -- fbref_tm_mapping.csv + player_valuations.csv")
print("=" * 60)

# Build most-recent valuation per player_id
pv = pd.read_csv(DATA_RAW / "player_valuations.csv", low_memory=False)
pv["market_value_m"] = pd.to_numeric(pv["market_value_in_eur"], errors="coerce") / 1e6
pv["date"] = pd.to_datetime(pv["date"], errors="coerce")
pv_latest = (
    pv.sort_values("date", ascending=False)
    .groupby("player_id")
    .first()
    .reset_index()[["player_id", "market_value_m"]]
)
pv_by_id = dict(zip(
    pv_latest["player_id"].astype(float),
    pv_latest["market_value_m"]
))
print(f"  player_valuations: {len(pv_by_id)} unique player_ids with data")

# Load FBref->TM mapping
tm_map = pd.read_csv(DATA_RAW / "fbref_tm_mapping.csv", low_memory=False,
                     encoding_errors="replace")
tm_map["player_id"] = (
    tm_map["UrlTmarkt"]
    .str.extract(r"/spieler/(\d+)")
    .squeeze()
    .pipe(pd.to_numeric, errors="coerce")
)
tm_map["mv_from_valuations"] = tm_map["player_id"].map(pv_by_id)

# Build FBref-name -> mv lookup (exact normalized)
s2_lookup = {}
for _, row in tm_map.iterrows():
    mv = row["mv_from_valuations"]
    if pd.notna(mv) and mv > 0:
        s2_lookup[normalize(str(row["PlayerFBref"]))] = mv

df = fill_from_dict(df, s2_lookup, "S2-fbref-tm-mapping")

# =============================================================================
# SOURCE 3 -- appearances.csv + player_valuations.csv (club-assisted ID lookup)
# =============================================================================
print("\n" + "=" * 60)
print("SOURCE 3 -- appearances.csv + player_valuations.csv")
print("=" * 60)

# Competition IDs for our target leagues
COMP_MAP = {
    "Belgian Pro League":   "BE1",
    "Superliga":            "DK1",
    "Scottish Premiership": "SC1",
    "Serie A (Brazil)":     "BRA1",
    "Primera Division":     "AR1N",
    "Eredivisie":           "NL1",
    "Super Lig":            "TR1",
    "Primeira Liga":        "PO1",
    # Big 5
    "Premier League": "GB1",
    "La Liga":        "ES1",
    "Bundesliga":     "L1",
    "Serie A":        "IT1",
    "Ligue 1":        "FR1",
}

# Only load appearances for our relevant competitions
target_comps = set(COMP_MAP.values())
try:
    app_df = pd.read_csv(DATA_RAW / "appearances.csv", low_memory=False,
                          encoding_errors="replace",
                          usecols=["player_id", "player_name", "competition_id"])
    app_df = app_df[app_df["competition_id"].isin(target_comps)].copy()
    print(f"  Appearances in target leagues: {len(app_df)}")

    # Normalize appearance player names
    app_df["_norm"] = app_df["player_name"].apply(normalize)

    # Build (norm_name, competition_id) -> player_id lookup
    # Keep most-frequent assignment for ambiguous names
    name_comp_to_id = (
        app_df.groupby(["_norm", "competition_id"])["player_id"]
        .agg(lambda x: x.mode()[0])   # most common player_id
        .reset_index()
    )
    name_comp_lookup = {
        (row["_norm"], row["competition_id"]): row["player_id"]
        for _, row in name_comp_to_id.iterrows()
    }

    # For null-MV rows, try (norm_player, competition_id) -> player_id -> valuation
    s3_filled = 0
    null_idx = df[df["market_value_m"].isna()].index
    for idx in null_idx:
        league = df.at[idx, "league_clean"]
        comp   = COMP_MAP.get(league)
        if not comp:
            continue
        key = (normalize(df.at[idx, "player"]), comp)
        pid = name_comp_lookup.get(key)
        if pid is None:
            continue
        mv = pv_by_id.get(float(pid))
        if pd.notna(mv) and mv > 0:
            df.at[idx, "market_value_m"] = mv
            s3_filled += 1

    STATS["S3-appearances"] = s3_filled
    print(f"  S3-appearances: filled {s3_filled}")
except Exception as e:
    STATS["S3-appearances"] = 0
    print(f"  S3-appearances: ERROR -- {e}")

# =============================================================================
# SOURCE 4 -- Transfermarkt website scraping
# =============================================================================
print("\n" + "=" * 60)
print("SOURCE 4 -- Transfermarkt web scraping")
print("=" * 60)

TM_LEAGUES = [
    ("Belgian Pro League",   "jupiler-pro-league",              "BE1"),
    ("Superliga",            "superliga",                       "DK1"),
    ("Scottish Premiership", "scottish-premiership",            "SC1"),
    ("Serie A (Brazil)",     "campeonato-brasileiro-serie-a",   "BRA1"),
    ("Primera Division",     "primera-division",                "AR1N"),
    ("Eredivisie",           "eredivisie",                      "NL1"),
    ("Super Lig",            "super-lig",                       "TR1"),
    ("Primeira Liga",        "liga-nos",                        "PO1"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def parse_tm_value(text: str) -> float | None:
    """Parse '€25.00m', '€500k', etc. to float millions."""
    text = text.strip().replace("\xa0", "").replace(" ", "")
    if not text or text in ("-", "-"):
        return None
    text = text.lstrip("€$£")
    try:
        if text.endswith("bn"):
            return float(text[:-2]) * 1000
        if text.endswith("m"):
            return float(text[:-1])
        if text.endswith("k"):
            return float(text[:-1]) / 1000
        return float(text) / 1_000_000  # raw EUR
    except ValueError:
        return None

def _parse_page(soup) -> dict:
    """Parse one TM market-value page, return {norm_name: mv}."""
    result = {}
    table = soup.find("table", {"class": "items"})
    if not table:
        return result
    for row in table.find_all("tr", {"class": ["odd", "even"]}):
        cols = row.find_all("td")
        if len(cols) < 7:
            continue
        # td[3] has the clean player name; td[-1] has the market value
        name = cols[3].get_text(strip=True) if len(cols) > 3 else ""
        # fallback: look for an anchor in td[1]
        if not name and len(cols) > 1:
            a = cols[1].find("a")
            name = a.get_text(strip=True) if a else ""
        mv = parse_tm_value(cols[-1].get_text(strip=True))
        if name and mv is not None:
            result[normalize(name)] = mv
    return result

def _get_last_page(soup) -> int:
    """Detect the last page number from TM pagination HTML."""
    pager = soup.find("ul", {"class": "tm-pagination"})
    if not pager:
        return 1
    last = 1
    for a in pager.find_all("a", href=True):
        m = re.search(r"/page/(\d+)", a["href"])
        if m:
            last = max(last, int(m.group(1)))
    return last

def scrape_tm_league(slug: str, comp_id: str, max_pages: int = 20) -> dict:
    """Scrape all pages of the TM market-value page for one league.
    Returns {normalized_player_name: market_value_m}.
    """
    base_url = f"https://www.transfermarkt.com/{slug}/marktwerte/wettbewerb/{comp_id}"
    result = {}
    try:
        # Page 1 — also determines total page count
        resp = requests.get(base_url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code} for {slug}")
            return result
        soup = BeautifulSoup(resp.text, "html.parser")
        result.update(_parse_page(soup))
        total_pages = min(_get_last_page(soup), max_pages)
        print(f"    Page 1/{total_pages} — {len(result)} players so far")

        for page in range(2, total_pages + 1):
            time.sleep(2)
            url = f"{base_url}/page/{page}"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} on page {page} — stopping")
                break
            page_vals = _parse_page(BeautifulSoup(resp.text, "html.parser"))
            if not page_vals:
                print(f"    Empty page {page} — stopping")
                break
            result.update(page_vals)
            print(f"    Page {page}/{total_pages} — {len(result)} players so far")
    except Exception as e:
        print(f"    Error scraping {slug}: {e}")
    return result

tm_scrape_total = 0
for league_name, slug, comp_id in TM_LEAGUES:
    null_in_league = int((df["market_value_m"].isna() & (df["league_clean"] == league_name)).sum())
    if null_in_league == 0:
        print(f"  {league_name}: 0 nulls -- skip")
        continue
    print(f"  Scraping {league_name} ({null_in_league} nulls)...")
    league_vals = scrape_tm_league(slug, comp_id)
    if league_vals:
        league_mask = df["market_value_m"].isna() & (df["league_clean"] == league_name)
        filled_in_league = 0
        for idx in df[league_mask].index:
            key = normalize(df.at[idx, "player"])
            if key in league_vals and pd.notna(league_vals[key]):
                df.at[idx, "market_value_m"] = league_vals[key]
                filled_in_league += 1
                tm_scrape_total += 1
        print(f"    => Filled {filled_in_league}/{null_in_league} for {league_name}")
    else:
        print(f"    0 values parsed (page blocked or empty)")
    time.sleep(3)

STATS["S4-tm-scrape"] = tm_scrape_total

# =============================================================================
# SOURCE 5 -- SofaScore public API
# =============================================================================
print("\n" + "=" * 60)
print("SOURCE 5 -- SofaScore API")
print("=" * 60)

SS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

def sofascore_search_mv(player_name: str) -> float | None:
    """Search SofaScore for a player and return their market value if available."""
    url = f"https://api.sofascore.com/api/v1/search/players?q={requests.utils.quote(player_name)}"
    try:
        resp = requests.get(url, headers=SS_HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        players = data.get("players", [])
        if not players:
            return None
        # Take the first result (best match)
        p = players[0].get("player", players[0])
        mv_raw = p.get("marketValueCurrency") or p.get("marketValue")
        if mv_raw and isinstance(mv_raw, (int, float)) and mv_raw > 0:
            return float(mv_raw) / 1_000_000
        # Some endpoints return proposedMarketValue
        pmv = p.get("proposedMarketValue")
        if pmv and isinstance(pmv, (int, float)) and pmv > 0:
            return float(pmv) / 1_000_000
    except Exception:
        pass
    return None

null_still = df[df["market_value_m"].isna()]
print(f"  {len(null_still)} players still null -- trying SofaScore for sample")

ss_filled = 0
ss_tried = 0
# Only try players where we have a reasonable chance (European leagues)
euro_leagues = {"Belgian Pro League", "Primeira Liga", "Scottish Premiership",
                "Eredivisie", "Super Lig", "Superliga",
                "Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"}

null_euro = null_still[null_still["league_clean"].isin(euro_leagues)]
print(f"  Trying {len(null_euro)} European players via SofaScore...")

for idx, row in null_euro.iterrows():
    player_name = row["player"]
    mv = sofascore_search_mv(player_name)
    if mv is not None and mv > 0:
        df.at[idx, "market_value_m"] = mv
        ss_filled += 1
    ss_tried += 1
    if ss_tried % 20 == 0:
        print(f"    Progress: {ss_tried}/{len(null_euro)} tried, {ss_filled} filled")
    time.sleep(0.5)

STATS["S5-sofascore"] = ss_filled
print(f"  S5-sofascore: filled {ss_filled} of {len(null_euro)} tried")

# =============================================================================
# Final summary + save
# =============================================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
n_null_end = int(df["market_value_m"].isna().sum())
print(f"\nNull MV before:  {n_null_start}")
for src, cnt in STATS.items():
    print(f"  {src}: {cnt}")
print(f"Total filled:    {n_null_start - n_null_end}")
print(f"Null MV after:   {n_null_end}")
pct = 100 * (len(df) - n_null_end) / len(df)
print(f"Coverage:        {len(df) - n_null_end}/{len(df)} ({pct:.1f}%)")

print("\nRemaining nulls by league:")
print(df[df["market_value_m"].isna()]["league_clean"].value_counts().to_string())

out = DATA_FINAL / "players_final.csv"
df.to_csv(out, index=False)
print(f"\nSaved: {out}")
