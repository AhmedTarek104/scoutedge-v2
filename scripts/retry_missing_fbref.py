"""Retry the 7 missing FBref files using the already-downloaded chromedriver exe."""
import time, pandas as pd
from io import StringIO
from pathlib import Path

import undetected_chromedriver as uc

CD_EXE = r"C:\Users\LEGION\appdata\roaming\undetected_chromedriver\undetected\chromedriver-win32\chromedriver.exe"
OUT    = Path(__file__).parent.parent / "data" / "raw"

MISSING = [
    (21, "ARG", "passing",    "passing"),
    (21, "ARG", "possession", "possession"),
    (21, "ARG", "defense",    "defense"),
    (21, "ARG", "misc",       "misc"),
    (40, "SCO", "stats",      "standard"),
    (40, "SCO", "shooting",   "shooting"),
    (40, "SCO", "passing",    "passing"),
]


def flatten(df):
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for t, b in df.columns:
            t, b = str(t).strip(), str(b).strip()
            if b in ("", "nan") or b == t:
                cols.append(t)
            elif t.startswith("Unnamed"):
                cols.append(b)
            else:
                cols.append(f"{t}_{b}")
        df.columns = cols
    return df


opts = uc.ChromeOptions()
opts.add_argument("--window-size=1400,900")

print("Launching Chrome with existing driver…", flush=True)
driver = uc.Chrome(options=opts, driver_executable_path=CD_EXE)

try:
    for lid, code, seg, save in MISSING:
        out = OUT / f"fbref_{code}_{save}.csv"
        if out.exists() and out.stat().st_size > 2000:
            print(f"  SKIP  {out.name}")
            continue

        url = f"https://fbref.com/en/comps/{lid}/{seg}/players/"
        print(f"  GET   {out.name}  … ", end="", flush=True)
        driver.get(url)

        cleared = False
        for _ in range(30):
            time.sleep(1)
            if "moment" not in driver.title.lower():
                cleared = True
                break

        if not cleared:
            print("TIMEOUT", flush=True)
            time.sleep(15)
            continue

        try:
            tables = pd.read_html(StringIO(driver.page_source), header=[0, 1])
        except ValueError:
            print("NO TABLE", flush=True)
            continue

        tables = [flatten(t) for t in tables]
        candidates = [t for t in tables if "Player" in t.columns]
        main = max(candidates if candidates else tables, key=len)
        main = main[main["Player"].astype(str).str.strip() != "Player"]
        main = main.dropna(how="all")
        main.to_csv(out, index=False)
        print(f"OK  {len(main):,} rows → {out.name}", flush=True)
        time.sleep(15)

finally:
    driver.quit()

print("\nFinal file check:")
for _, code, _, save in MISSING:
    p = OUT / f"fbref_{code}_{save}.csv"
    if p.exists():
        n = len(pd.read_csv(p, low_memory=False))
        print(f"  {p.name:46s} {n:>5,} rows")
    else:
        print(f"  {p.name:46s} MISSING")
