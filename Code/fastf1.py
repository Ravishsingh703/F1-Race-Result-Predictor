# Code/fastf1_features.py

from __future__ import annotations
import os
from typing import Iterable, Optional, Tuple, Dict, List
import sys
import traceback
from datetime import datetime, timedelta, timezone
import re
import numpy as np
import pandas as pd
import fastf1
from fastf1 import utils
from fastf1.events import get_event_schedule
from pathlib import Path
import importlib
import Code.config as config  
import sys

# ============================================================
# CONFIG RELOAD PATCH
# ============================================================

# Force reload each time script runs, to pick up new YEAR/ROUND/etc.
try:
    # Check if global runner already loaded config
    if "Code.config" in sys.modules:
        config = sys.modules["Code.config"]
        importlib.reload(config)
    else:
        # Standard import for standalone runs
        import Code.config as config
        sys.modules["Code.config"] = config

except Exception as e:
    print(f"[WARN] Could not reload Code.config dynamically: {e}")
    import Code.config as config

# Pull latest config values dynamically
FASTF1 = config.FASTF1
YEAR = config.YEAR
ROUND = config.ROUND



# -----------------------------
# Core session helpers
# -----------------------------

def enable_cache(path: str = "Data/fastf1_cache"):
    os.makedirs(path, exist_ok=True)
    fastf1.Cache.enable_cache(path)


def load_session(year: int, grand_prix: str, kind: str = "R"):
    """
    kind: 'R'=Race, 'Q'=Qualifying, 'FP1'/'FP2'/'FP3', 'S'=Sprint
    """
    sess = fastf1.get_session(year, grand_prix, kind)
    sess.load()
    return sess


def get_race_results(session) -> pd.DataFrame:
    """
    Minimal race results table.
    """
    res = session.results
    keep = ["Abbreviation", "DriverNumber", "TeamName", "GridPosition",
            "Position", "Status", "Points"]
    return res[keep].copy()


def get_quali_results(session) -> pd.DataFrame:
    """
    Minimal qualifying results (best laps per driver).
    """
    if session.name != "Qualifying":
        raise ValueError("Pass a Qualifying session")
    res = session.results
    # Best lap time is in 'Q1', 'Q2', 'Q3' cols where available
    keep = ["Abbreviation", "DriverNumber", "TeamName",
            "Q1", "Q2", "Q3", "Position"]
    return res[keep].copy()


def get_driver_laps(session, abbr: str) -> pd.DataFrame:
    return session.laps.pick_drivers([abbr]).copy()



# -----------------------------
# Race-by-race feature extractors
# -----------------------------

def q3_time(session, abbr: str) -> Optional[pd.Timedelta]:
    """
    Driver's Q3 lap time (if present). Return None if no Q3.
    """
    if session.name != "Qualifying":
        raise ValueError("q3_time expects a Qualifying session")
    res = get_quali_results(session)
    row = res.loc[res["Abbreviation"] == abbr]
    if row.empty:
        return None
    t = row["Q3"].iloc[0]
    return t if pd.notna(t) else None


def pole_q3_time(session) -> Optional[pd.Timedelta]:
    """
    Pole sitter's Q3 time for this qualifying session.
    """
    if session.name != "Qualifying":
        raise ValueError("pole_q3_time expects a Qualifying session")
    res = get_quali_results(session)
    # Position == 1 row's Q3 if available, otherwise use best of Q2/Q1
    pole = res.sort_values("Position").head(1)
    if pole.empty:
        return None
    for col in ["Q3", "Q2", "Q1"]:
        tt = pole[col].iloc[0]
        if pd.notna(tt):
            return tt
    return None



    
    
    
    


def average_pole_q3_for_event(event_name: str, years: Iterable[int]) -> Optional[pd.Timedelta]:
    """
    Compute average pole Q3 time for the given event_name across given years.
    Only considers years where Q3 exists; otherwise uses best available quali segment.
    """
    times = []
    for y in years:
        try:
            q = load_session(y, event_name, "Q")
            t = pole_q3_time(q)
            if t is not None:
                times.append(t)
        except Exception:
            continue
    if not times:
        return None
    return sum(times, pd.Timedelta(0)) / len(times)



def quali_gap_to_pole(session) -> pd.DataFrame:
    """
    For qualifying session: gap of each driver to pole time (seconds).
    """
    if session.name != "Qualifying":
        raise ValueError("quali_gap_to_pole expects a Qualifying session")
    res = get_quali_results(session)
    ptime = pole_q3_time(session)
    # Fallback to best of Q2/Q1 if no Q3
    if ptime is None:
        # find pole best across Q1/Q2
        pole_best = res.sort_values("Position").head(1)[["Q1", "Q2", "Q3"]].iloc[0]
        ptime = min([x for x in pole_best if pd.notna(x)], default=None)
    if ptime is None:
        res["GapToPole_s"] = np.nan
        return res

    # Driver best quali time across segments
    best = res[["Q1", "Q2", "Q3"]].apply(lambda r: min([x for x in r if pd.notna(x)], default=pd.NaT), axis=1)
    gaps = (best - ptime).dt.total_seconds()
    out = res.copy()
    out["BestQuali"] = best
    out["GapToPole_s"] = gaps
    return out


def positions_gained(session) -> pd.DataFrame:
    """
    GridPosition - Position for each finisher (NaN for DNFs).
    """
    res = get_race_results(session)
    out = res.copy()
    # If Position is NaN (DNF), leave gained as NaN
    out["PositionsGained"] = out["GridPosition"] - out["Position"]
    return out


def rolling_form_points(results_history: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    results_history: rows across races for the **same season** with at least columns:
      ['RoundNumber', 'Abbreviation', 'Points']
    Returns per driver the rolling mean of points for last N races.
    """
    df = results_history.sort_values(["Abbreviation", "RoundNumber"]).copy()
    df["RollingForm_Points"] = df.groupby("Abbreviation")["Points"].rolling(lookback, min_periods=1).mean().reset_index(0, drop=True)
    return df


def driver_track_score(history_results: pd.DataFrame, event_name: str, min_years: int = 2) -> pd.DataFrame:
    """
    Average points per appearance for each driver at the given event_name.
    history_results must include rows across multiple years with at least:
      ['Year', 'EventName', 'Abbreviation', 'Points']
    """
    f = history_results.query("EventName == @event_name").copy()
    grp = f.groupby("Abbreviation").agg(
        TrackStarts=("Points", "count"),
        TrackPoints=("Points", "sum")
    )
    grp["TrackScore"] = grp["TrackPoints"] / grp["TrackStarts"]
    grp.loc[grp["TrackStarts"] < min_years, "TrackScore"] = np.nan  # too little data
    return grp.reset_index()





    





def weather_summary(session) -> Dict[str, Optional[float]]:
    """
    Aggregate simple weather signals: avg air temp, track temp, wind, precipitation.
    """
    try:
        w = session.weather_data.copy()
    except Exception:
        return dict(air_temp=None, track_temp=None, wind_speed=None, rain=None)
    if w is None or w.empty:
        return dict(air_temp=None, track_temp=None, wind_speed=None, rain=None)
    return dict(
        air_temp=float(w["AirTemp"].mean()) if "AirTemp" in w else None,
        track_temp=float(w["TrackTemp"].mean()) if "TrackTemp" in w else None,
        wind_speed=float(w["WindSpeed"].mean()) if "WindSpeed" in w else None,
        rain=float(w["Rainfall"].mean()) if "Rainfall" in w else None,
    )


def safety_car_stats(session) -> dict:
    """
    Count SC and VSC deployments by tracking state transitions in Race Control Messages.
    """
    rcm = getattr(session, "race_control_messages", None)
    if rcm is None or rcm.empty or "Message" not in rcm.columns:
        return {"safety_car": 0, "vsc": 0}

    msgs = rcm["Message"].astype(str).str.upper().tolist()

    sc_active = False
    vsc_active = False
    sc_count = 0
    vsc_count = 0

    for m in msgs:
        if "VIRTUAL SAFETY CAR" in m:
            if any(k in m for k in ("ENDING", "END", "FINISH")):
                vsc_active = False
            elif not vsc_active:
                vsc_active = True
                vsc_count += 1

        elif "SAFETY CAR" in m:
            if any(k in m for k in ("ENDING", "END", "IN THIS LAP", "FINISH")):
                sc_active = False
            elif not sc_active:
                sc_active = True
                sc_count += 1

        if "RESUME" in m or "GREEN" in m:
            sc_active = False
            vsc_active = False

    return {"safety_car": sc_count, "vsc": vsc_count}





from typing import Optional

def overtaking_difficulty(event_name: str) -> Optional[int]:
    """
    Return overtaking difficulty on a 1..100 scale
    (1 = easiest to overtake, 100 = hardest).
    Returns None if we don't have a mapping.

    This uses a canonical mapping + a few aliases to cope with
    different event naming across seasons.
    """
    # Canonical difficulty values (1 easy … 100 hardest)
    DIFF = {
        "Bahrain Grand Prix": 22,
        "Argentine Grand Prix": 23,
        "Azerbaijan Grand Prix": 26,
        "Chinese Grand Prix": 29,
        "Brazilian Grand Prix": 31,
        "Turkish Grand Prix": 33,
        "Mexican Grand Prix": 37,
        "German Grand Prix": 39,
        "Belgian Grand Prix": 41,
        "Malaysian Grand Prix": 42,
        "European Grand Prix": 44,
        "Austrian Grand Prix": 48,
        "United States Grand Prix": 50,
        "Korean Grand Prix": 51,
        "Italian Grand Prix": 52,
        "British Grand Prix": 53,
        "Japanese Grand Prix": 55,
        "Canadian Grand Prix": 56,
        "Portuguese Grand Prix": 58,
        "Abu Dhabi Grand Prix": 59,
        "French Grand Prix": 61,
        "Australian Grand Prix": 63,
        "Russian Grand Prix": 65,
        "Singapore Grand Prix": 66,
        "Spanish Grand Prix": 67,
        "Hungarian Grand Prix": 69,
        "San Marino Grand Prix": 71,
        "Monaco Grand Prix": 78,
    }

    # Common aliases -> canonical keys above
    ALIAS = {
        # Modern naming vs historic canonical
        "Mexico City Grand Prix": "Mexican Grand Prix",
        "São Paulo Grand Prix": "Brazilian Grand Prix",
        "Sao Paulo Grand Prix": "Brazilian Grand Prix",
        "Emilia Romagna Grand Prix": "San Marino Grand Prix",
        "Imola Grand Prix": "San Marino Grand Prix",

        # Regional/legacy labels seen in some datasets
        "USA Grand Prix": "United States Grand Prix",
        "United States GP": "United States Grand Prix",
        "Great Britain Grand Prix": "British Grand Prix",
        "UK Grand Prix": "British Grand Prix",
        "Italia Grand Prix": "Italian Grand Prix",
        "Italia GP": "Italian Grand Prix",
        "Belgium Grand Prix": "Belgian Grand Prix",
        "España Grand Prix": "Spanish Grand Prix",
        "Espana Grand Prix": "Spanish Grand Prix",
        "Hungary Grand Prix": "Hungarian Grand Prix",
    }

    name = (event_name or "").strip()

    # 1) Exact match
    if name in DIFF:
        return DIFF[name]

    # 2) Alias map
    alias = ALIAS.get(name)
    if alias and alias in DIFF:
        return DIFF[alias]

    # 3) Light normalization: remove leading "Formula 1" and trailing "Grand Prix"
    base = name.replace("FORMULA 1", "").replace("Formula 1", "").strip()
    if base in DIFF:
        return DIFF[base]
    alias = ALIAS.get(base)
    if alias and alias in DIFF:
        return DIFF[alias]

    short = base.replace("Grand Prix", "").strip()

    # 4) Last-resort fuzzy-ish fallback: look for a canonical key that contains the short token
    #    (e.g., "Monaco" -> "Monaco Grand Prix")
    if short:
        lowered = short.lower()
        for k in DIFF:
            if lowered in k.lower():
                return DIFF[k]

    # Not found
    return None


def compute_finish_times(session_race: fastf1.core.Session) -> pd.DataFrame:
    """
    Compute relative finish times (gap to winner) for all drivers in a race.
    - Winner: "+0.000"
    - Finished drivers: "+X.XXX" (gap to winner)
    - Lapped drivers: "Lap"
    - DNF / Retired drivers: "DNF"
    
    Returns:
        DataFrame with columns ['Abbreviation', 'FinishTime']
    """

    try:
        # Get race results and ensure basic structure
        res = session_race.results.copy()
        if res is None or res.empty:
            print("[WARN] No race results found.")
            return pd.DataFrame(columns=["Abbreviation", "FinishTime"])

        res["Abbreviation"] = res["Abbreviation"].astype(str)
        res["TimeSec"] = res["Time"].apply(
            lambda x: x.total_seconds() if pd.notna(x) else np.nan
        )

        # Compute finish times relative to winner
        finish_times = []
        for _, r in res.iterrows():
            abbr = r["Abbreviation"]
            status = str(r.get("Status", "")).lower()
            pos = r["Position"]
            time_val = r["TimeSec"]

            # Handle DNF / lapped
            if "dnf" in status or "ret" in status:
                ft = "DNF"
            elif "lap" in status or pd.isna(time_val):
                ft = "Lap"
            else:
                winner_time = res["TimeSec"].min()
                gap = time_val - winner_time
                ft = f"+{gap:.3f}"

            # Force race winner to +0.000
            if pos == 1:
                ft = "+0.000"

            finish_times.append((abbr, ft))

        # Return structured output
        finish_df = pd.DataFrame(finish_times, columns=["Abbreviation", "FinishTime"])
        return finish_df

    except Exception as e:
        print(f"[ERROR] compute_finish_times() failed: {e}")
        return pd.DataFrame(columns=["Abbreviation", "FinishTime"])







def live_champ_standings(results_to_date: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute running driver and constructor standings from results rows to date.
    rows need: ['RoundNumber','Abbreviation','TeamName','Points']
    """
    dr = results_to_date.groupby("Abbreviation", as_index=False)["Points"].sum().sort_values("Points", ascending=False)
    ct = results_to_date.groupby("TeamName", as_index=False)["Points"].sum().sort_values("Points", ascending=False)
    return dr.reset_index(drop=True), ct.reset_index(drop=True)




def compute_pit_count(session_race: fastf1.core.Session) -> dict[str, int]:
    """
    Count pit stops using laps metadata for each driver.
    We count a stop if either PitInTime or PitOutTime is present on a lap.
    Returns {Abbreviation: count}.
    """
    pit_counts: dict[str, int] = {}
    try:
        # list of 3-letter abbreviations in official results order
        drivers = session_race.results["Abbreviation"].astype(str).tolist()
        laps = session_race.laps  # full laps dataframe

        for abbr in drivers:
            dlaps = laps.pick_drivers([abbr])
            if dlaps is None or dlaps.empty:
                pit_counts[abbr] = np.nan
                continue
            has_pit = (
                dlaps.get("PitInTime", pd.Series(index=dlaps.index, dtype="timedelta64[ns]")).notna()
                | dlaps.get("PitOutTime", pd.Series(index=dlaps.index, dtype="timedelta64[ns]")).notna()
            )
            pit_counts[abbr] = int(has_pit.sum())
    except Exception:
        # If anything goes wrong, leave as NaN to avoid crashing the pipeline
        pass
    return pit_counts




# -----------------------------
# High-level assembly for one race (per-driver row)
# -----------------------------


def assemble_race_features(
    session_race: fastf1.core.Session,
    session_quali: Optional[fastf1.core.Session] = None,
    history_results: Optional[pd.DataFrame] = None,
    avg_pole_years: Optional[Iterable[int]] = None,
    season_results_to_date: Optional[pd.DataFrame] = None,
    round_number: Optional[int] = None
) -> pd.DataFrame:
    """
    Builds standardized race features for a given F1 round (2018–present).
    Column order and naming exactly match historical schema.
    """

    event_name = session_race.event.EventName
    print(f"[INFO] Building race features for {event_name}")

    # --- Load data from sessions ---
    race_results = get_race_results(session_race)
    quali_gap = quali_gap_to_pole(session_quali) if session_quali is not None else None
    sc_stats = safety_car_stats(session_race)
    weather = weather_summary(session_race)
    finish_df = compute_finish_times(session_race)

    # --- Construct rows ---
    rows = []
    for _, r in race_results.iterrows():
        abbr = r["Abbreviation"]
        team = r["TeamName"]
        grid_pos = r["GridPosition"]
        finish_pos = r["Position"]
        status = r["Status"]
        points = r["Points"]

        positions_gained = (
            int(grid_pos) - int(finish_pos)
            if pd.notna(grid_pos) and pd.notna(finish_pos)
            else np.nan
        )

        # Quali gap to pole
        quali_gap_val = np.nan
        if quali_gap is not None and abbr in quali_gap["Abbreviation"].values:
            q_row = quali_gap.loc[quali_gap["Abbreviation"] == abbr]
            if not q_row.empty:
                quali_gap_val = float(q_row["GapToPole_s"].iloc[0])

        # Finish time
        ftime_row = finish_df.loc[finish_df["Abbreviation"] == abbr, "FinishTime"]
        finish_time = ftime_row.iloc[0] if not ftime_row.empty else np.nan

        # --- Assemble final row ---
        row = {
            "EventName": event_name,
            "Abbreviation": abbr,
            "TeamName": team,
            "GridPosition": grid_pos,
            "FinishPosition": finish_pos,
            "Status": status,
            "Points": points,
            "PositionsGained": positions_gained,
            "QualiGapToPole_s": quali_gap_val,
            "PitCount": getattr(r, "PitCount", np.nan),  # use if available in race data
            "SC_Count": sc_stats.get("safety_car", np.nan),
            "VSC_Count": sc_stats.get("vsc", np.nan),
            "AirTemp": weather.get("air_temp", np.nan),
            "TrackTemp": weather.get("track_temp", np.nan),
            "WindSpeed": weather.get("wind_speed", np.nan),
            "OvertakingDifficulty": overtaking_difficulty(event_name),
            "FinishTime": finish_time,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # --- Fill and reorder columns ---
    df["QualiGapToPole_s"] = df["QualiGapToPole_s"].fillna(method="ffill")

    ordered_cols = [
        "EventName", "Abbreviation", "TeamName", "GridPosition", "FinishPosition",
        "Status", "Points", "PositionsGained", "QualiGapToPole_s", "PitCount",
        "SC_Count", "VSC_Count", "AirTemp", "TrackTemp", "WindSpeed",
        "OvertakingDifficulty", "FinishTime"
    ]

    df = df.reindex(columns=ordered_cols, fill_value=np.nan)

    print(f"[INFO] ✅ Race features ready for {event_name} ({len(df)} drivers)")
    return df           
       
        

      





# -----------------------------
# Tiny helper
# -----------------------------

def save_csv(df, path: str):
    """
    Always save under Data_Processed folder (project root).
    """
    import os
    from pathlib import Path

    # Force save inside Data_Processed
    base_dir = Path("Data_Processed")
    full_path = base_dir / path

    full_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_path, index=False)

    print(f"✅ Saved: {full_path}")
    return full_path  # <-- return the Path object






def _to_secs(x):
    """Timedelta/None -> float seconds (or pd.NA)."""
    if x is None or pd.isna(x):
        return pd.NA
    try:
        return x.total_seconds()
    except Exception:
        return pd.NA


def quali_features(session_q: "fastf1.core.Session") -> pd.DataFrame:
    """
    Extract qualifying metrics per driver:
      - Q1/Q2/Q3 times (s)
      - q_best (best of Q1-3) (s)
      - pole_time (s) and gap_to_pole (s)
      - keeps Abbreviation, DriverNumber, Position as well
    """
    res = session_q.results.copy()

    # Some sessions lack Q1/Q2/Q3 for some drivers; keep what exists
    keep_base = ["Abbreviation", "DriverNumber", "Position"]
    keep_q = [c for c in ["Q1", "Q2", "Q3"] if c in res.columns]
    keep = keep_base + keep_q
    res = res[keep].copy()

    # Convert Q1/Q2/Q3 to seconds
    for c in ["Q1", "Q2", "Q3"]:
        if c in res.columns:
            res[c] = res[c].map(_to_secs)

    # Best quali time available for each driver
    q_cols = [c for c in ["Q1", "Q2", "Q3"] if c in res.columns]
    if q_cols:
        res["q_best"] = res[q_cols].min(axis=1, skipna=True)
    else:
        res["q_best"] = pd.NA

    # Pole time (min of all bests)
    pole_time = res["q_best"].min(skipna=True)
    res["pole_time"] = pole_time

    # Gap to pole (positive seconds)
    res["gap_to_pole"] = res["q_best"] - pole_time

    return res



    

    
# ============================================================
# Runner Block
# ============================================================


if __name__ == "__main__":
    print("\n===============================")
    print("🚀 Starting FastF1 Smart Incremental Runner")
    print(f"Year: {YEAR}, Starting from Round: {ROUND}")
    print("===============================\n")

    try:
        enable_cache("Data/fastf1_cache")

        out_dir = Path("Data_Processed") / "fastf1" / str(YEAR)
        out_dir.mkdir(parents=True, exist_ok=True)

        schedule = fastf1.get_event_schedule(YEAR)
        now_utc = datetime.now(timezone.utc)
        processed_any = False

        for rnd in range(ROUND, len(schedule) + 1):
            try:
                event = schedule.loc[schedule["RoundNumber"] == rnd].iloc[0]
                gp_name = event["EventName"]
            except IndexError:
                print(f"🚫 No round {rnd} found for {YEAR}. Season complete.")
                break

            out_file = out_dir / f"{YEAR}_R{rnd:02d}_features.csv"

            # Skip if already exists
            if out_file.exists():
                print(f"✅ R{rnd:02d} - {gp_name}: already exists, skipping.")
                continue

            # Get race timing data
            try:
                sess_race = fastf1.get_session(YEAR, rnd, "R")
                race_start = sess_race.event.get("Session5Date", None)
            except Exception:
                race_start = None

            if race_start is None:
                print(f"⚠️ No race start time found for R{rnd:02d} ({gp_name}). Skipping.")
                continue

            race_end_est = race_start + timedelta(hours=3)

            # Timing logic
            if now_utc < race_start:
                until_start = race_start - now_utc
                print(f"🕓 R{rnd:02d} - {gp_name} has not commenced yet. Starts in {until_start}.")
                break
            elif race_start <= now_utc <= race_end_est:
                print(f"🏁 R{rnd:02d} - {gp_name} currently ongoing. Try again later.")
                break
            else:
                print(f"✅ R{rnd:02d} - {gp_name} likely finished. Building FastF1 data...")

                # Load race session
                sess_race.load()

                # Try to load qualifying session
                try:
                    sess_quali = fastf1.get_session(YEAR, rnd, "Q")
                    sess_quali.load()
                except Exception:
                    sess_quali = None

                # Assemble and save features
                df = assemble_race_features(sess_race, sess_quali, round_number=rnd)
                save_csv(df, f"fastf1/{YEAR}/{YEAR}_R{rnd:02d}_features.csv")

                processed_any = True
                print(f"✅ Completed FastF1 build for R{rnd:02d} - {gp_name}.\n")

        if not processed_any:
            print("\n⚙️ No new rounds required update.")
        else:
            print("\n✅ Incremental FastF1 updates finished successfully!")

    except Exception as e:
        print(f"\n❌ FastF1 update failed: {e}")