# Code/fastf1_features.py

from __future__ import annotations
import os
from typing import Iterable, Optional, Tuple, Dict, List
from typing import Optional
import sys
import traceback
import re
import numpy as np
import pandas as pd
import fastf1
from fastf1 import utils
from fastf1.events import get_event_schedule


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


def dnf_rate_on_track(history_results: pd.DataFrame, event_name: str) -> pd.DataFrame:
    """
    DNF rate per driver at this event_name.
    history_results requires: ['EventName','Abbreviation','Status']
    """
    f = history_results.query("EventName == @event_name").copy()
    f["IsDNF"] = ~f["Status"].fillna("").str.contains("Finished", case=False)
    grp = f.groupby("Abbreviation").agg(Starts=("IsDNF", "count"), DNFs=("IsDNF", "sum"))
    grp["DNFRate"] = grp["DNFs"] / grp["Starts"].replace(0, np.nan)
    return grp.reset_index()








_PENALTY_POSITIVE = (
    "time penalty", "drive through", "drive-through", "stop and go", "stop/go",
    "stop & go", "grid penalty", "5 second", "10 second", "5s", "10s"
)
_PENALTY_NEGATIVE = (  # phrases we should NOT count
    "no further action", "investigation", "investigating", "noted",
    "deleted", "cancelled", "warning", "black and white", "b&w flag",
)

# light heuristic to score severity
def _penalty_severity(msg: str) -> int:
    m = msg.lower()
    if "stop" in m and "go" in m:
        return 4
    if "drive" in m and "through" in m:
        return 4
    if "10" in m and "second" in m or "10s" in m:
        return 3
    if "5" in m and "second" in m or "5s" in m:
        return 2
    if "grid penalty" in m:
        return 2
    if "time penalty" in m:
        return 2
    return 1

def penalties_count_from_rcm(session, abbr: str) -> dict:
    """
    Count confirmed penalties for a driver using race control messages.

    Returns:
        {"count": int, "severity": int, "hits": [str, ...]}
    """
    try:
        r = session.results
    except Exception:
        session.load()
        r = session.results

    if r is None or r.empty:
        return {"count": 0, "severity": 0, "hits": []}

    # map ABBR -> DriverNumber (string to match RCM)
    try:
        drvnum = str(int(r.loc[r["Abbreviation"] == abbr, "DriverNumber"].iloc[0]))
    except Exception:
        return {"count": 0, "severity": 0, "hits": []}

    # get race-control messages
    rcm = getattr(session, "race_control_messages", None)
    if rcm is None or rcm.empty:
        return {"count": 0, "severity": 0, "hits": []}

    # normalize columns
    cols = {c.lower(): c for c in rcm.columns}
    msg_col = cols.get("message") or cols.get("text")
    dnum_col = cols.get("drivernumber")

    if msg_col is None:
        return {"count": 0, "severity": 0, "hits": []}

    df = rcm.copy()
    df[msg_col] = df[msg_col].astype(str)

    # If there is a proper driver number column, use it; otherwise regex the text.
    if dnum_col:
        mask_driver = df[dnum_col].astype(str) == drvnum
    else:
        # look for “Car 44”, “#44”, “No. 44”, “car number 44”, etc.
        pat = re.compile(rf"(?:^|\b)(?:car|#|no\.?|number)?\s*{re.escape(drvnum)}\b", re.I)
        mask_driver = df[msg_col].str.contains(pat)

    df = df[mask_driver]

    if df.empty:
        return {"count": 0, "severity": 0, "hits": []}

    m = df[msg_col].str.lower()

    # must include any positive cue…
    pos = np.zeros(len(df), dtype=bool)
    for k in _PENALTY_POSITIVE:
        pos |= m.str.contains(k)

    # …and NOT contain any negative cue
    neg = np.zeros(len(df), dtype=bool)
    for k in _PENALTY_NEGATIVE:
        neg |= m.str.contains(k)

    confirmed = df[pos & ~neg]
    if confirmed.empty:
        return {"count": 0, "severity": 0, "hits": []}

    hits = confirmed[msg_col].tolist()
    severity = int(sum(_penalty_severity(x) for x in hits))
    return {"count": int(len(hits)), "severity": severity, "hits": hits}







def get_pit_stop_times(session, abbr: str) -> pd.DataFrame:
    """
    Compute pit stop durations for a driver using PitInTime (lap N)
    and PitOutTime (lap N+1). Returns a DataFrame with seconds.ms.
    Filters out obviously bad values.
    """
    laps = session.laps.pick_drivers([abbr]).copy()

    # Make sure the columns we need exist
    needed = ["LapNumber", "PitInTime", "PitOutTime"]
    missing = [c for c in needed if c not in laps.columns]
    if missing:
        print(f"{abbr}: missing columns {missing} in laps table.")
        return pd.DataFrame(columns=["LapIn", "PitDuration_s"])

    # Keep only what we need and align in/out on successive rows
    x = laps[["LapNumber", "PitInTime", "PitOutTime"]].reset_index(drop=True)

    # Find laps where driver entered the pits
    pit_in_idx = x.index[~x["PitInTime"].isna()].to_numpy()
    if len(pit_in_idx) == 0:
        return pd.DataFrame(columns=["LapIn", "PitDuration_s"])

    rows = []
    for i in pit_in_idx:
        # The corresponding pit OUT is typically on the NEXT lap
        if i + 1 < len(x) and pd.notna(x.at[i + 1, "PitOutTime"]):
            tin = x.at[i, "PitInTime"]
            tout = x.at[i + 1, "PitOutTime"]
            dur = (tout - tin).total_seconds()

            # keep only sane pit-lane durations (in seconds)
            if 0.5 <= dur <= 60:
                rows.append({
                    "LapIn": int(x.at[i, "LapNumber"]),
                    "PitDuration_s": float(dur)
                })

    return pd.DataFrame(rows)



def pit_stop_efficiency(session, abbr: str, avg_reference: float = 24.0) -> dict:
    """
    Use pit-lane durations from get_pit_stop_times and compare to 24s avg.
    Returns: pit_count, pit_loss_avg_s (mean pit-lane s), pit_efficiency_delta (mean - 24).
    """
    stops = get_pit_stop_times(session, abbr)  # must return ["LapIn","PitDuration_s"]
    if stops is None or stops.empty:
        return {"pit_count": 0, "pit_loss_avg_s": np.nan, "pit_efficiency_delta": np.nan}

    pit_durations = stops["PitDuration_s"].to_numpy(dtype=float)
    # discard absurd values
    pit_durations = pit_durations[(pit_durations >= 10.0) & (pit_durations <= 60.0)]
    if pit_durations.size == 0:
        return {"pit_count": 0, "pit_loss_avg_s": np.nan, "pit_efficiency_delta": np.nan}

    avg_pit = float(np.mean(pit_durations))
    return {
        "pit_count": int(pit_durations.size),
        "pit_loss_avg_s": avg_pit,
        "pit_efficiency_delta": avg_pit - float(avg_reference)
    }



def normalize_pitloss_to_eff(file_path: str) -> None:
    
    df = pd.read_csv(file_path)

    if "PitLossAvg_s" not in df.columns:
        print(f"⚠️ Skipping {file_path}: no PitLossAvg_s column")
        return

    # Compute race-wide mean pit loss
    gp_mean = df["PitLossAvg_s"].mean(skipna=True)

    # New column values
    df["Pitstop_eff"] = df["PitLossAvg_s"] - gp_mean

    # Move Pitstop_eff right after PitLossAvg_s
    cols = df.columns.tolist()
    cols.insert(cols.index("PitLossAvg_s") + 1, cols.pop(cols.index("Pitstop_eff")))
    df = df[cols]

    # Save back
    df.to_csv(file_path, index=False)
    print(f"✅ Updated {file_path} with Pitstop_eff")
    
    
    
    
    
    


def team_points_trend(season_results: pd.DataFrame, team_name: str, round_number: int, lookback: int = 3) -> Optional[float]:
    """
    Rolling average of team points over last N rounds prior to current round.
    season_results needs: ['RoundNumber','TeamName','Points']
    """
    hist = season_results.query("TeamName == @team_name & RoundNumber < @round_number").copy()
    if hist.empty:
        return None
    hist = hist.sort_values("RoundNumber")
    roll = hist["Points"].rolling(lookback, min_periods=1).mean().iloc[-1]
    return float(roll) if pd.notna(roll) else None


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










def live_champ_standings(results_to_date: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute running driver and constructor standings from results rows to date.
    rows need: ['RoundNumber','Abbreviation','TeamName','Points']
    """
    dr = results_to_date.groupby("Abbreviation", as_index=False)["Points"].sum().sort_values("Points", ascending=False)
    ct = results_to_date.groupby("TeamName", as_index=False)["Points"].sum().sort_values("Points", ascending=False)
    return dr.reset_index(drop=True), ct.reset_index(drop=True)


# -----------------------------
# High-level assembly for one race (per-driver row)
# -----------------------------


def assemble_race_features(
    session_race,
    session_quali=None,
    history_results: Optional[pd.DataFrame] = None,
    avg_pole_years: Optional[Iterable[int]] = None,
    season_results_to_date: Optional[pd.DataFrame] = None,
    round_number: Optional[int] = None
) -> pd.DataFrame:
    event_name = session_race.event.EventName
    quali_gap = quali_gap_to_pole(session_quali) if session_quali is not None else None
    race_res = get_race_results(session_race)
    out_rows = []

    # precompute history pieces (unchanged)
    avg_pole = None
    if session_quali is not None and avg_pole_years:
        try:
            avg_pole = average_pole_q3_for_event(event_name, avg_pole_years)
        except Exception:
            avg_pole = None

    track_score_df = None
    dnf_track_df = None
    if history_results is not None:
        try:
            track_score_df = driver_track_score(history_results, event_name)
        except Exception:
            track_score_df = None
        try:
            dnf_track_df = dnf_rate_on_track(history_results, event_name)
        except Exception:
            dnf_track_df = None

    # >>> compute once
    sc_stats = safety_car_stats(session_race)
    weather = weather_summary(session_race)

    for _, r in race_res.iterrows():
        abbr = r["Abbreviation"]
        team = r["TeamName"]

        # base row
        row = {
            "EventName": event_name,
            "Abbreviation": abbr,
            "TeamName": team,
            "GridPosition": r["GridPosition"],
            "FinishPosition": r["Position"],
            "Status": r["Status"],
            "Points": r["Points"],
        }

        # positions gained
        if pd.notna(r["GridPosition"]) and pd.notna(r["Position"]):
            row["PositionsGained"] = int(r["GridPosition"]) - int(r["Position"])
        else:
            row["PositionsGained"] = np.nan

        # quali gap only (you removed Q3 ratio)
        if session_quali is not None and quali_gap is not None:
            qrow = quali_gap.loc[quali_gap["Abbreviation"] == abbr]
            row["QualiGapToPole_s"] = (
                float(qrow["GapToPole_s"].iloc[0]) if not qrow.empty and pd.notna(qrow["GapToPole_s"].iloc[0])
                else np.nan
            )
        else:
            row["QualiGapToPole_s"] = np.nan
            
       
        

        # pit stop metrics (your updated function)
        pit = pit_stop_efficiency(session_race, abbr)
        row["PitCount"]     = pit["pit_count"]
        row["PitLossAvg_s"] = pit["pit_loss_avg_s"]

        # penalties
        row["PenaltiesCount"] = penalties_count_from_rcm(session_race, abbr)

        # team trend if you’re passing season results (else NaN)
        if season_results_to_date is not None and round_number is not None:
            row["TeamTrend_Points3"] = team_points_trend(season_results_to_date, team, round_number, 3)
        else:
            row["TeamTrend_Points3"] = np.nan

        # track history
        if track_score_df is not None:
            trow = track_score_df.loc[track_score_df["Abbreviation"] == abbr]
            row["TrackScore"] = float(trow["TrackScore"].iloc[0]) if not trow.empty else np.nan
        else:
            row["TrackScore"] = np.nan

        if dnf_track_df is not None:
            drow = dnf_track_df.loc[dnf_track_df["Abbreviation"] == abbr]
            row["DNFRateOnTrack"] = float(drow["DNFRate"].iloc[0]) if not drow.empty else np.nan
        else:
            row["DNFRateOnTrack"] = np.nan

        # >>> assign the precomputed controls/weather to EACH row
        row["SC_Count"]  = sc_stats.get("safety_car", 0)
        row["VSC_Count"] = sc_stats.get("vsc", 0)

        row["AirTemp"]   = weather["air_temp"]
        row["TrackTemp"] = weather["track_temp"]
        row["WindSpeed"] = weather["wind_speed"]
        row["Rain"]      = weather["rain"]

        # static difficulty
        row["OvertakingDifficulty"] = overtaking_difficulty(event_name)

        out_rows.append(row)

    return pd.DataFrame(out_rows)






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





import pandas as pd

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




#-----------------
#Runner block
#-----------------


def run_incremental_updates(
    year: int = 2025,
    start_round: int = 17,
    max_rounds_to_try: int = 30,
    miss_limit_consecutive: int = 1,
) -> None:
    """
    Process race features from `start_round` forward.
    - Skips files that already exist.
    - Stops after `miss_limit_consecutive` **consecutive** failures (e.g., round not available yet).
    - Saves to Data_Processed/fastf1/<year>/<year>_R## _features.csv
    """
    # Ensure cache is enabled (safe to call if already done)
    try:
        enable_cache("Data/fastf1_cache")
    except Exception:
        pass

    out_dir = Path("Data_Processed") / "fastf1" / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    consec_misses = 0
    saved, skipped, failed = [], [], []

    print(f"\n=== Incremental update: {year} from R{start_round:02d} ===")
    for rnd in range(start_round, start_round + max_rounds_to_try):
        out_path = out_dir / f"{year}_R{rnd:02d}_features.csv"
        if out_path.exists():
            print(f"➤ {out_path.name} exists, skipping")
            skipped.append(rnd)
            consec_misses = 0  # existing file is a "non-miss" event
            continue

        try:
            # Load sessions (Race; Quali is optional)
            sess_r = load_session(year, rnd, "R")
            try:
                sess_q = load_session(year, rnd, "Q")
            except Exception:
                sess_q = None  # proceed without quali if not available

            # Build features
            df = assemble_race_features(
                session_race=sess_r,
                session_quali=sess_q,
                history_results=None,
                avg_pole_years=None,
                season_results_to_date=None,
                round_number=rnd,
            )
            if df is None or df.empty:
                raise RuntimeError("assemble_race_features returned empty DataFrame")

            # Save under year subfolder
            rel_path = f"fastf1/{year}/{year}_R{rnd:02d}_features.csv"
            save_csv(df, rel_path)  # save_csv always roots at Data_Processed/
            print(f"✅ Saved {year} R{rnd:02d} -> {rel_path} (rows={len(df)})")
            saved.append(rnd)
            consec_misses = 0  # reset on success

        except Exception as e:
            consec_misses += 1
            failed.append((rnd, str(e)))
            print(f"❌ Error {year} R{rnd:02d}: {e} (consec miss {consec_misses})")
            # Optional (comment out if you don't want stack traces)
            # traceback.print_exc(limit=1)

            if consec_misses >= miss_limit_consecutive:
                print("▲ Stopping: consecutive-miss threshold reached.")
                break

    # Summary
    print("\n=== Summary ===")
    if saved:
        print("Saved rounds: ", saved)
    if skipped:
        print("Skipped (already existed): ", skipped)
    if failed:
        print("Failed: ", failed)
    print("All updated.")

# Example: run only when you execute fastf1.py directly
if __name__ == "__main__":
    run_incremental_updates(year=2025, start_round=17, miss_limit_consecutive=1)
    
    
    
    
    


  
    
    
    
    
    
    
    
    
    
    
    
    
    
    











