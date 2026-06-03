"""
Download FBref player stats for non-Big-5 leagues using undetected Chrome.
Non-headless mode required to pass Cloudflare's JS challenge.
"""
import sys, time, pandas as pd
from io import StringIO
from pathlib import Path

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

OUT_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LEAGUES = [
    ("TUR", 26,  "Turkish Süper Lig"),
    ("NED", 23,  "Dutch Eredivisie"),
    ("POR", 32,  "Portuguese Primeira Liga"),
    ("DEN", 50,  "Danish Superliga"),
    ("BEL", 37,  "Belgian Pro League"),
    ("BRA", 24,  "Brazilian Série A"),
    ("ARG", 21,  "Argentine Primera División"),
    ("SCO", 40,  "Scottish Premiership"),
]

# (url_segment, save_name)
STAT_TYPES = [
    ("stats",      "standard"),
    ("shooting",   "shooting"),
    ("passing",    "passing"),
    ("possession", "possession"),
    ("defense",    "defense"),
    ("misc",       "misc"),
]

CF_WAIT     = 25   # seconds to let Cloudflare clear
PAGE_SLEEP  = 10   # seconds between page loads


def flatten_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for top, bot in df.columns:
            top, bot = str(top).strip(), str(bot).strip()
            if bot in ("", "nan") or bot == top:
                cols.append(top)
            elif top.startswith("Unnamed"):
                cols.append(bot)
            else:
                cols.append(f"{top}_{bot}")
        df.columns = cols
    return df


def clean_table(df):
    """Remove FBref's repeated mid-table header rows and all-NaN rows."""
    first = df.columns[0]
    df = df[df[first].astype(str).str.strip() != str(first).strip()]
    df = df[df[first].astype(str).str.strip() != "Player"]
    df = df.dropna(how="all")
    return df.reset_index(drop=True)


def wait_for_page(driver, timeout=30):
    """Wait until Cloudflare challenge clears (title no longer 'Just a moment...')."""
    for _ in range(timeout):
        time.sleep(1)
        title = driver.title
        if "moment" not in title.lower():
            return True
    return False


def extract_table(driver):
    """Pull the main stats table from current FBref page source."""
    src = driver.page_source
    try:
        tables = pd.read_html(StringIO(src), header=[0, 1])
    except ValueError:
        return None
    if not tables:
        return None
    tables = [flatten_cols(t) for t in tables]
    # Pick the table with the most rows that has a 'Player' column
    candidates = [t for t in tables if "Player" in t.columns]
    if not candidates:
        candidates = tables
    main = max(candidates, key=len)
    return clean_table(main)


# ── Launch browser once ───────────────────────────────────────────────────────
print("Launching Chrome (visible window — required for Cloudflare)…", flush=True)
opts = uc.ChromeOptions()
opts.add_argument("--window-size=1400,900")
opts.add_argument("--disable-blink-features=AutomationControlled")
driver = uc.Chrome(options=opts, version_main=148)
driver.set_page_load_timeout(45)

results = []
total   = len(LEAGUES) * len(STAT_TYPES)
done    = 0

try:
    for code, league_id, league_name in LEAGUES:
        print(f"\n{'='*62}", flush=True)
        print(f"{league_name}  (ID={league_id})", flush=True)
        print(f"{'='*62}", flush=True)

        for url_seg, save_name in STAT_TYPES:
            done += 1
            out  = OUT_DIR / f"fbref_{code}_{save_name}.csv"
            url  = f"https://fbref.com/en/comps/{league_id}/{url_seg}/players/"
            tag  = f"[{done:2d}/{total}] {save_name:12s}"

            if out.exists() and out.stat().st_size > 2000:
                existing = pd.read_csv(out, low_memory=False)
                print(f"  {tag} SKIP (cached {len(existing):,} rows)", flush=True)
                results.append((code, save_name, len(existing), out.name, "cached"))
                continue

            print(f"  {tag} loading… ", end="", flush=True)
            try:
                driver.get(url)
                cleared = wait_for_page(driver, timeout=CF_WAIT)
                if not cleared:
                    print("FAIL (Cloudflare timeout)", flush=True)
                    results.append((code, save_name, 0, out.name, "FAILED"))
                    continue

                df = extract_table(driver)
                if df is None or df.empty:
                    print("FAIL (no table)", flush=True)
                    results.append((code, save_name, 0, out.name, "FAILED"))
                else:
                    df.to_csv(out, index=False)
                    print(f"OK  {len(df):,} rows", flush=True)
                    results.append((code, save_name, len(df), out.name, "OK"))

            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                results.append((code, save_name, 0, out.name, f"ERROR"))

            time.sleep(PAGE_SLEEP)

finally:
    driver.quit()

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("DOWNLOAD SUMMARY")
print(f"{'='*62}")

ok     = [r for r in results if r[4] in ("OK", "cached")]
failed = [r for r in results if r[4] not in ("OK", "cached")]

print(f"\nSuccessful: {len(ok)}/{total}")
for code, stat, rows, fname, status in ok:
    flag = " (cached)" if status == "cached" else ""
    print(f"  {fname:46s} {rows:>6,} rows{flag}")

if failed:
    print(f"\nFailed ({len(failed)}):")
    for code, stat, rows, fname, status in failed:
        print(f"  {fname:46s} {status}")

# Sample columns
sample = OUT_DIR / "fbref_TUR_standard.csv"
if sample.exists():
    s = pd.read_csv(sample, low_memory=False)
    print(f"\nSample columns — fbref_TUR_standard.csv ({len(s):,} rows):")
    print(" ", list(s.columns))
