# Code/quali_feat.py
from __future__ import annotations
import os
import pandas as pd
from pathlib import Path
import fastf1
from fastf1.events import get_event_schedule
from datetime import datetime, timezone, timedelta
# ============================================================
# CONFIG RELOAD PATCH
# ============================================================

import sys
import importlib

try:
    # If config already exists (from global runner)
    if "Code.config" in sys.modules:
        config = sys.modules["Code.config"]
        importlib.reload(config)
    else:
        # Normal import if not already loaded
        import Code.config as config
        sys.modules["Code.config"] = config

except Exception as e:
    print(f"[WARN] Could not reload Code.config dynamically: {e}")
    import Code.config as config

# Pull latest config values
FASTF1 = config.FASTF1
QUALI_FEAT = config.QUALI_FEAT
YEAR = config.YEAR
ROUND = config.ROUND


# ============================================================
# CACHE SETUP
# ============================================================

def enable_cache(path: str = "Data/fastf1_cache"):
    os.makedirs(path, exist_ok=True)
    fastf1.Cache.enable_cache(path)
    print(f"[INFO] Cache enabled at: {path}")


# ============================================================
# SESSION LOADER
# ============================================================

def load_session(year: int, grand_prix: str, kind: str = "Q"):
    """
    Load a Qualifying session for the specified year and event.
    """
    sess = fastf1.get_session(year, grand_prix, kind)
    sess.load()
    print(f"[INFO] Loaded {kind} session for {grand_prix} {year}")
    return sess


# ============================================================
# QUALIFYING FEATURE BUILDER
# ============================================================

def extract_quali_features(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Build qualifying data with computed BestQuali and GapToPole_s.

    Rules:
    - BestQuali = minimum of Q1/Q2/Q3 (in seconds)
    - GapToPole_s = 0.000 for Position == 1.0
                    else (BestQuali of driver) - (BestQuali of Position==1.0)
    - If GapToPole_s missing, copy from Position-1, else Position-2, else 0.000
    """
    res = session.results.copy()

    keep_cols = ["Abbreviation", "DriverNumber", "TeamName", "Q1", "Q2", "Q3", "Position"]
    df = res[[c for c in keep_cols if c in res.columns]].copy()

    # Convert Q1/Q2/Q3 timedelta → seconds
    for c in ["Q1", "Q2", "Q3"]:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: x.total_seconds() if pd.notna(x) else None)

    # Compute best qualifying time
    q_cols = [c for c in ["Q1", "Q2", "Q3"] if c in df.columns]
    df["BestQuali"] = df[q_cols].min(axis=1, skipna=True)

    # Pole sitter time
    
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.sort_values("Position").reset_index(drop=True)

    pole_best = None
    try:
        pole_best = float(df.loc[df["Position"] == 1.0, "BestQuali"].iloc[0])
    except Exception:
        pole_best = None

    # Compute GapToPole_s (driver_time - pole_time)
    df["GapToPole_s"] = df.apply(
        lambda row: 0.000
        if row["Position"] == 1.0
        else (row["BestQuali"] - pole_best)
        if pd.notna(row["BestQuali"]) and pole_best is not None
        else None,
        axis=1
    )

    # Fill missing GapToPole_s using previous drivers
    df["GapToPole_s"] = pd.to_numeric(df["GapToPole_s"], errors="coerce")
    for i in range(len(df)):
        if pd.isna(df.loc[i, "GapToPole_s"]):
            if i > 0 and pd.notna(df.loc[i - 1, "GapToPole_s"]):
                df.loc[i, "GapToPole_s"] = df.loc[i - 1, "GapToPole_s"]
            elif i > 1 and pd.notna(df.loc[i - 2, "GapToPole_s"]):
                df.loc[i, "GapToPole_s"] = df.loc[i - 2, "GapToPole_s"]
            else:
                df.loc[i, "GapToPole_s"] = 0.000

    df["GapToPole_s"] = df["GapToPole_s"].round(3)

    # Add metadata
    df["EventName"] = session.event.EventName
    df["Year"] = session.event.year
    df["Round"] = session.event.RoundNumber

    # Reorder neatly
    ordered = [
        "EventName", "Year", "Round", "Abbreviation", "DriverNumber",
        "TeamName", "Position", "Q1", "Q2", "Q3", "BestQuali", "GapToPole_s"
    ]
    df = df[[c for c in ordered if c in df.columns]]
    print(f"[INFO] ✅ Extracted quali features for {session.event.EventName}")
    return df


# ============================================================
# RUNNER FUNCTION
# ============================================================

def run_quali_feature_extraction(year: int, round_number: int, force_rebuild: bool = False):
    """
    Extract qualifying data and save to:
    Data_Processed/quali_feat/<year>/<round>_quali_feat_<year>.csv
    Stops early if session has no valid data.
    """
    enable_cache("Data/fastf1_cache")

    out_dir = Path("Data_Processed") / "quali_feat" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{round_number:02d}_quali_feat_{year}.csv"

    # Skip logic
    if out_path.exists() and not force_rebuild:
        print(f"➤ {out_path.name} exists, skipping")
        return

    # Load session metadata
    schedule = get_event_schedule(year)
    gp_name = schedule.loc[schedule["RoundNumber"] == round_number, "EventName"].values[0]
    print(f"[INFO] Attempting qualifying extraction for {gp_name} (Round {round_number})")

    try:
        session = load_session(year, gp_name, "Q")
    except Exception as e:
        print(f"🟡 Skipping {gp_name}: No qualifying data available ({e})")
        raise RuntimeError("no data")

    # Extract data
    df = extract_quali_features(session)

    # ✅ Sanity check — stop if no usable data
    if df.empty or "BestQuali" not in df.columns or df["BestQuali"].dropna().empty:
        print(f"🟡 No valid timing data for {gp_name}. Stopping before file creation.")
        raise RuntimeError("no valid data")

    # Save only if valid
    df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")


# ============================================================
# MAIN RUNNER (multi-round, stops after missing data twice)
# ============================================================



if __name__ == "__main__":
    print("\n===============================")
    print("🚀 Starting Smart Quali Feature Extraction")
    print(f"Year: {YEAR}, Starting from Round: {ROUND}")
    print("===============================\n")

    try:
        enable_cache("Data/fastf1_cache")

        schedule = get_event_schedule(YEAR)
        now_utc = datetime.now(timezone.utc)
        processed_any = False

        # Loop from current round onwards
        for rnd in range(ROUND, len(schedule) + 1):
            try:
                event = schedule.loc[schedule["RoundNumber"] == rnd].iloc[0]
                gp_name = event["EventName"]
            except IndexError:
                print(f"🚫 No round {rnd} found for {YEAR}. Season complete.")
                break

            out_dir = Path("Data_Processed") / "quali_feat" / str(YEAR)
            out_file = out_dir / f"{rnd:02d}_quali_feat_{YEAR}.csv"

            # --- Skip if file already exists ---
            if out_file.exists():
                print(f"✅ R{rnd:02d} - {gp_name}: already exists, skipping.")
                continue

            # --- Fetch Q3 session start time ---
            try:
                sess = fastf1.get_session(YEAR, rnd, "Q")
                q3_start = sess.event.get("Session4Date", None)
            except Exception:
                q3_start = None

            if q3_start is None:
                print(f"⚠️ No Q3 start time found for R{rnd:02d} ({gp_name}). Skipping.")
                continue

            q3_end_est = q3_start + timedelta(hours=1)

            # --- Timing logic ---
            if now_utc < q3_start:
                until_start = q3_start - now_utc
                print(f"🕓 R{rnd:02d} - {gp_name} Q3 has not started yet. Starts in {until_start}.")
                break
            elif q3_start <= now_utc <= q3_end_est:
                print(f"🏁 R{rnd:02d} - {gp_name} Q3 currently ongoing. Try again later.")
                break
            else:
                print(f"✅ R{rnd:02d} - {gp_name} Q3 finished. Building quali data...")
                run_quali_feature_extraction(year=YEAR, round_number=rnd, force_rebuild=True)
                processed_any = True
                print(f"✅ Completed Quali build for R{rnd:02d} - {gp_name}.\n")

        if not processed_any:
            print("\n⚙️ No new qualifying sessions required update.")
        else:
            print("\n✅ Incremental Quali updates finished successfully!")

    except Exception as e:
        print(f"\n❌ Quali Feature Extraction failed: {e}")
        
        
        
        
        
        
        