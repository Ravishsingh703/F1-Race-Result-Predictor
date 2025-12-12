
from __future__ import annotations
import pandas as pd
import numpy as np
from fastf1.ergast import Ergast
from pathlib import Path
import sys, importlib

# ============================================================
# CONFIG RELOAD PATCH
# ============================================================

try:
    # If running inside the global runner
    if "Code.config" in sys.modules:
        config = sys.modules["Code.config"]
        importlib.reload(config)
    else:
        # Standalone fallback
        import Code.config as config
        sys.modules["Code.config"] = config

except Exception as e:
    print(f"[WARN] Could not reload Code.config dynamically: {e}")
    import Code.config as config


# ============================================================
# CONFIG VARIABLES
# ============================================================

YEAR = config.YEAR
OUT_DIR = Path("/Users/ravish/Documents/F1-Project-Folder/Data_Processed/quali_feat/Champ_Rating")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / f"champ_standing_live_{YEAR}.csv"
CHAMP_FILE = OUT_DIR / f"champ_standing_live_{YEAR}.csv"

print(f"✅ Rolling Form Config Loaded: YEAR={YEAR}")





# === LIVE STANDINGS (with reliable Abbreviation) =========================



def _make_abbr(row: pd.Series) -> str:
    """
    Best-effort 3-letter code:
    1) driverCode if available (ergast 'driver_code')
    2) first 3 of familyName (LastName)
    3) first letter of first name + first 2 of last name
    Always uppercased.
    """
    for key in ("driver_code", "driverCode"):
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            return str(row[key]).upper().strip()

    fam = None
    for key in ("driver_familyName", "familyName", "LastName"):
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            fam = str(row[key]).strip()
            break

    giv = None
    for key in ("driver_givenName", "givenName", "FirstName"):
        if key in row and pd.notna(row[key]) and str(row[key]).strip():
            giv = str(row[key]).strip()
            break

    if fam and len(fam) >= 3:
        return fam[:3].upper()
    if giv and fam:
        return (giv[:1] + fam[:2]).upper()
    return "UNK"

def rebuild_live_standings_with_abbr(year: int = YEAR) -> pd.DataFrame:
    erg = Ergast(result_type="pandas", auto_cast=True)
    res = erg.get_driver_standings(season=year, round="last")
    if not res or not res.content or res.content[0] is None or res.content[0].empty:
        raise RuntimeError(f"No standings from Ergast for {year}.")

    raw = res.content[0].copy()

    # Flexible renames
    rename_map = {
        "points": "TotalPoints",
        "wins": "Wins",
        "position": "Position",
        "driver_givenName": "FirstName",
        "driver_familyName": "LastName",
        "constructor_name": "TeamName",
        "constructors_name": "TeamName",
    }
    for old, new in rename_map.items():
        if old in raw.columns:
            raw.rename(columns={old: new}, inplace=True)

    # Build Abbreviation robustly
    raw["Abbreviation"] = raw.apply(_make_abbr, axis=1).astype(str)

    # Keep useful columns only (whatever exists)
    keep = [c for c in [
        "Position", "Abbreviation", "FirstName", "LastName", "TeamName",
        "TotalPoints", "Wins"
    ] if c in raw.columns]
    df = raw[keep].copy()

    # Coerce numerics
    for c in ("Position", "TotalPoints", "Wins"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Abbreviation"] = df["Abbreviation"].astype(str).str.upper().str.strip()
    df = df.sort_values("Position", na_position="last").reset_index(drop=True)

    df.to_csv(OUT_FILE, index=False)
    print(f"✅ Saved live standings with Abbreviation -> {OUT_FILE}")
    return df

# Run A)
stand_df = rebuild_live_standings_with_abbr(YEAR)




# ============================================================
# 🏁 Championship Rating (Points → Rating 1–100)
# ============================================================




def compute_champ_rating(points_series: pd.Series) -> pd.Series:
    """
    Compute Championship Rating (1–100) relative to leader.
    Leader always gets 100, 0-point drivers get 1.
    """
    max_pts = points_series.max()
    if pd.isna(max_pts) or max_pts == 0:
        return pd.Series([1.0] * len(points_series), index=points_series.index)
    rating = 1 + 99 * (points_series / max_pts)
    return np.round(rating, 2)


def update_champ_rating(year: int = YEAR):
    print(f"\n🏎️ Updating Championship Rating for {year}...")

    champ_path = CHAMP_FILE
    if not champ_path.exists():
        print(f"[WARN] Championship file not found: {champ_path}")
        return

    df = pd.read_csv(champ_path)

    # detect points column (case-insensitive)
    pts_col = next((c for c in df.columns if "points" in c.lower()), None)
    if not pts_col:
        print("[ERROR] No points column found.")
        return

    # compute champ rating
    df["Champ_Rating"] = compute_champ_rating(df[pts_col])

    # ensure correct order (place Champ_Rating after points)
    cols = df.columns.tolist()
    if "Champ_Rating" in cols:
        cols.remove("Champ_Rating")
    insert_at = cols.index(pts_col) + 1
    cols.insert(insert_at, "Champ_Rating")
    df = df[cols]

    # overwrite file safely
    df.to_csv(champ_path, index=False)
    print(f"✅ Championship Rating updated successfully for {year}")
    print(df[[pts_col, "Champ_Rating"]].head(10).to_string(index=False))


# Run Championship Rating update
update_champ_rating(YEAR)






# === FIXED ROLLING RESULT RATING (decay & safe merge) ======================



FASTF1_DIR = Path("/Users/ravish/Documents/F1-Project-Folder/Data_Processed/fastf1") / str(YEAR)
CHAMP_FILE = OUT_FILE  # same file from (A)

def _base_race_score(pos: float, pts: float) -> float:
    """Your requested scale: P1=100; P10≈20; P20=10. Small 2% boost if scored points."""
    if pd.isna(pos) or pos <= 0:
        return np.nan
    if pos == 1:
        score = 100.0
    elif pos <= 10:
        score = 100.0 - ((pos - 1) * 8.8889)
    elif pos <= 20:
        score = 20.0 - ((pos - 10) * 1.0)
    else:
        score = 5.0
    if pd.notna(pts) and pts > 0:
        score *= 1.02
    return max(score, 1.0)

def _weighted_avg_descending(scores: list[float], decay: float) -> float:
    """Most recent race has weight 1; previous *= decay; etc."""
    if not scores:
        return np.nan
    weights = [decay ** i for i in range(len(scores))]
    return float(np.average(scores, weights=weights))

def compute_rolling_result_rating(year: int = YEAR, decay: float = 0.75):
    race_files = sorted(FASTF1_DIR.glob(f"{year}_R*_features.csv"))
    if not race_files:
        print(f"[WARN] No race files in {FASTF1_DIR}")
        return

    tall = []
    for f in race_files:
        try:
            df = pd.read_csv(f)
            if not {"FinishPosition", "Abbreviation", "Points"}.issubset(df.columns):
                if "Position" in df.columns and "Abbreviation" in df.columns:
                    df = df.rename(columns={"Position": "FinishPosition"})
                else:
                    continue
            df["Abbreviation"] = df["Abbreviation"].astype(str).str.upper().str.strip()
            df["FinishPosition"] = pd.to_numeric(df["FinishPosition"], errors="coerce")
            df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

            df["RaceScore"] = df.apply(lambda r: _base_race_score(r["FinishPosition"], r["Points"]), axis=1)
            df["Round"] = int(f.stem.split("_R")[1].split("_")[0])
            tall.append(df[["Abbreviation", "Round", "RaceScore"]])
        except Exception as e:
            print(f"[WARN] Skipped {f.name}: {e}")

    if not tall:
        print("[WARN] No usable race data found for rolling result rating.")
        return

    allres = pd.concat(tall, ignore_index=True)
    allres = allres.sort_values(["Abbreviation", "Round"], ascending=[True, False])

    # ✅ FIX: explicitly build DataFrame, no MultiIndex nonsense
    results = []
    for abbr, group in allres.groupby("Abbreviation"):
        raw_score = _weighted_avg_descending(group["RaceScore"].tolist(), decay)
        results.append({"Abbreviation": abbr, "Rolling_Result_Raw": raw_score})
    agg = pd.DataFrame(results)

    # Normalize to 1–100 safely
    if agg["Rolling_Result_Raw"].nunique() == 1:
        agg["Rolling_Result_Rating"] = 50.0
    else:
        maxv, minv = agg["Rolling_Result_Raw"].max(), agg["Rolling_Result_Raw"].min()
        agg["Rolling_Result_Rating"] = (
            1 + 99 * (agg["Rolling_Result_Raw"] - minv) / (maxv - minv)
        ).round(2)

    champ = pd.read_csv(CHAMP_FILE)
    if "Abbreviation" not in champ.columns:
        champ = rebuild_live_standings_with_abbr(year)

    champ["Abbreviation"] = champ["Abbreviation"].astype(str).str.upper().str.strip()
    agg["Abbreviation"] = agg["Abbreviation"].astype(str).str.upper().str.strip()

    merged = champ.merge(agg[["Abbreviation", "Rolling_Result_Rating"]], on="Abbreviation", how="left")
    merged.to_csv(CHAMP_FILE, index=False)
    print(f"✅ Rolling_Result_Rating updated successfully in {CHAMP_FILE.name}")

    print("\n=== Top Drivers by Rolling Result Rating ===")
    print(merged[["Position","Abbreviation","TotalPoints","Rolling_Result_Rating"]]
          .sort_values("Position").head(10).to_string(index=False))

# Run B)
compute_rolling_result_rating(YEAR, decay=0.75)





# ============================================================
# 🏆 Race Dominance Score (2025, decay=0.75)
# ============================================================



def _gap_to_num(value):
    """Convert '+12.345' → 12.345, handle 'Lap'/'DNF'."""
    if pd.isna(value): return np.nan
    val = str(value).strip()
    if val.startswith("+"):
        try: return float(val.replace("+",""))
        except: return np.nan
    return np.nan

def _dominance_from_gap(row, k=0.03):
    """Compute dominance score based on gap & position."""
    pos = row.get("FinishPosition", np.nan)
    gap = row.get("GapToLeader_s", np.nan)
    ftime = str(row.get("FinishTime", "")).strip()

    # Winner or zero gap
    if ftime == "+0.000" or pos == 1:
        return 100.0

    # Lapped cars
    if "Lap" in ftime:
        return max(0.2 * (100 - 3*(pos-1)), 1)

    # DNF / Retired
    if "DNF" in ftime or "Ret" in ftime:
        return max(0.1 * (100 - 3*(pos-1)), 1)

    # Finished normally
    if not pd.isna(gap):
        base = 100 * np.exp(-k * gap)
        adj = base / (1 + 0.05*(pos-1))
        return adj

    return 1.0  # fallback low score

def compute_race_dominance_score(year: int = YEAR, decay: float = 0.75):
    print(f"\n🏎️ Computing Race Dominance Score for {year}...")

    race_files = sorted(FASTF1_DIR.glob(f"{year}_R*_features.csv"))
    if not race_files:
        print(f"[WARN] No race data found for {year}")
        return

    tall = []
    for f in race_files:
        try:
            df = pd.read_csv(f)
            if not {"FinishTime", "Abbreviation", "FinishPosition"}.issubset(df.columns):
                continue

            df["Abbreviation"] = df["Abbreviation"].astype(str).str.upper().str.strip()
            df["GapToLeader_s"] = df["FinishTime"].apply(_gap_to_num)
            df["Dominance_Score"] = df.apply(_dominance_from_gap, axis=1)

            round_str = f.stem.split("_R")[1].split("_")[0]
            df["Round"] = int(round_str)
            tall.append(df[["Abbreviation", "Round", "Dominance_Score"]])
        except Exception as e:
            print(f"[WARN] Skipped {f.name}: {e}")

    if not tall:
        print("[WARN] No usable race data found for dominance.")
        return

    allres = pd.concat(tall, ignore_index=True)
    allres = allres.sort_values(["Abbreviation", "Round"], ascending=[True, False])

    # Apply decay-weighted average
    results = []
    for abbr, group in allres.groupby("Abbreviation"):
        weights = [decay ** i for i in range(len(group))]
        score = np.average(group["Dominance_Score"].tolist(), weights=weights)
        results.append({"Abbreviation": abbr, "Race_Dominance_Score": score})
    agg = pd.DataFrame(results)

    # Normalize 1–100
    maxv, minv = agg["Race_Dominance_Score"].max(), agg["Race_Dominance_Score"].min()
    agg["Race_Dominance_Score"] = (
        1 + 99 * (agg["Race_Dominance_Score"] - minv) / (maxv - minv)
    ).round(2)

    # Merge into champ file
    if CHAMP_FILE.exists():
        champ = pd.read_csv(CHAMP_FILE)
        champ["Abbreviation"] = champ["Abbreviation"].astype(str).str.upper().str.strip()
        merged = champ.merge(agg, on="Abbreviation", how="left")
        merged.to_csv(CHAMP_FILE, index=False)
        print(f"✅ Race_Dominance_Score added to {CHAMP_FILE.name}")
    else:
        agg.to_csv(CHAMP_FILE, index=False)
        print(f"✅ Created {CHAMP_FILE.name} with Race_Dominance_Score")

    print("\n=== Top Drivers by Race Dominance Score ===")
    print(agg.sort_values("Race_Dominance_Score", ascending=False).head(10).to_string(index=False))

# Run
compute_race_dominance_score(YEAR)



# ============================================================
# Rolling Form Computation 
# ============================================================

def compute_rolling_form(year: int = YEAR):
    champ_path = OUT_FILE  # same file for all updates
    if not champ_path.exists():
        print(f"[WARN] Championship file not found: {champ_path}")
        return

    df = pd.read_csv(champ_path)

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    required = {
        "Champ_Rating": None,
        "Rolling_Result_Rating": None,
        "Race_Dominance_Score": None
    }

    # Match columns flexibly
    for col in df.columns:
        if "champ" in col.lower() and "rating" in col.lower():
            required["Champ_Rating"] = col
        elif "rolling" in col.lower() and "result" in col.lower():
            required["Rolling_Result_Rating"] = col
        elif "dominance" in col.lower():
            required["Race_Dominance_Score"] = col

    # Validate all exist
    if any(v is None for v in required.values()):
        missing = [k for k, v in required.items() if v is None]
        print(f"[ERROR] Missing columns for Rolling Form: {missing}")
        return

    # Compute rolling form safely
    df["Rolling_Form"] = (
        0.2 * df[required["Champ_Rating"]] +
        0.4 * df[required["Rolling_Result_Rating"]] +
        0.4 * df[required["Race_Dominance_Score"]]
    ).round(2)

    # Normalize again 1–100 range (optional for balance)
    minv, maxv = df["Rolling_Form"].min(), df["Rolling_Form"].max()
    if maxv > minv:
        df["Rolling_Form"] = 1 + 99 * (df["Rolling_Form"] - minv) / (maxv - minv)
        df["Rolling_Form"] = df["Rolling_Form"].round(2)

    # Reorder columns — put at the end neatly
    cols = df.columns.tolist()
    if "Rolling_Form" in cols:
        cols.remove("Rolling_Form")
        cols.append("Rolling_Form")
    df = df[cols]

    df.to_csv(champ_path, index=False)
    print(f"✅ Rolling Form computed and saved to {champ_path.name}\n")

    print("=== Top 10 Drivers by Rolling Form ===")
    preview_cols = ["Abbreviation", "TotalPoints", "Champ_Rating",
                    "Rolling_Result_Rating", "Race_Dominance_Score", "Rolling_Form"]
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df.sort_values("Rolling_Form", ascending=False)[preview_cols].head(10).to_string(index=False))


# Run Rolling Form Calculation
compute_rolling_form(YEAR)





# ============================================================
#  Remove Old Driver (DOO)
# ============================================================

def remove_inactive_driver(abbr: str = "DOO", year: int = YEAR):
    champ_path = OUT_FILE
    if not champ_path.exists():
        print(f"[WARN] Championship file not found: {champ_path}")
        return

    df = pd.read_csv(champ_path)
    if "Abbreviation" not in df.columns:
        print("[WARN] No Abbreviation column found — skipping cleanup.")
        return

    before = len(df)
    df = df[df["Abbreviation"].astype(str).str.upper() != abbr.upper()]
    after = len(df)

    if before == after:
        print(f"[INFO] No driver with Abbreviation '{abbr}' found.")
    else:
        df.to_csv(champ_path, index=False)
        print(f"✅ Removed {before - after} row(s) with Abbreviation '{abbr}'. File updated.")

# Run cleanup
remove_inactive_driver("DOO", YEAR)





