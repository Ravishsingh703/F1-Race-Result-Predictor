# update_current_driver_stats.py
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

# ========= CONFIG =========
YEAR = 2025
ROOT = Path.home() / "Documents" / "F1-Project-Folder"
PROC = ROOT / "Data_Processed" / "fastf1" / str(YEAR)
OUT  = ROOT / "Data_Processed" / "driver_stats_over_years" / f"current_driver_stats_{YEAR}.csv"

# Rolling-form settings
DECAY = 0.75            # per-race exponential recency (older *= 0.75)
POINTS_TABLE = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}
SCALE = 100.0 / 25.0    # avg points (0..25) -> 0..100

# ========= Helpers =========
def feature_files() -> list[Path]:
    return sorted(PROC.glob(f"{YEAR}_R*_features.csv"), key=_round_of)

def _round_of(p: Path) -> int:
    m = re.search(rf"{YEAR}_R(\d+)_", p.name)
    return int(m.group(1)) if m else 0

def _points_series(df: pd.DataFrame) -> pd.Series:
    """Return race points per row (float). Fallback: compute from Position."""
    if "Points" in df.columns:
        return pd.to_numeric(df["Points"], errors="coerce").fillna(0.0).astype(float)
    pos_col = next((c for c in ["FinishPosition", "Position", "FinishPos"] if c in df.columns), None)
    if pos_col is None:
        return pd.Series(0.0, index=df.index)
    pos = pd.to_numeric(df[pos_col], errors="coerce")
    pts = pos.map(lambda p: POINTS_TABLE.get(int(p), 0) if pd.notna(p) else 0)
    return pts.astype(float)

def _finished_mask(df: pd.DataFrame) -> pd.Series:
    if "Status" in df.columns:
        s = df["Status"].astype(str).str.lower()
        return s.eq("finished") | s.eq("fin") | s.str.contains("finished")
    pos_col = next((c for c in ["FinishPosition", "Position"] if c in df.columns), None)
    if pos_col is not None:
        return pd.to_numeric(df[pos_col], errors="coerce").notna()
    return pd.Series(False, index=df.index)

def _dnf_mask(df: pd.DataFrame) -> pd.Series:
    return ~_finished_mask(df)

def _penalized_mask(df: pd.DataFrame) -> pd.Series:
    col = next((c for c in ["PenaltiesCount", "Penalties", "PenaltyCount"] if c in df.columns), None)
    if col is None:
        return pd.Series(False, index=df.index)
    v = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return v > 0

def _pit_eff_series(df: pd.DataFrame) -> pd.Series:
    """
    Get pit efficiency series from a race file.
    Prefer explicit efficiency columns; otherwise fall back to pit loss avg and mean-center it.
    """
    # Feature files commonly have 'Pitstop_eff'
    for c in ["Pitstop_eff", "PitStopEff", "PitstopEff"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")

    # Fallback: mean-center pit loss to produce an "efficiency-like" value
    for c in ["PitLossAvg_s", "PitLossAvg", "pit_loss_avg_s"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            return x - x.mean(skipna=True)

    return pd.Series(np.nan, index=df.index)

def _rolling_form(all_race_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Weighted avg points across races with exponential recency (newest weight=1, then *= DECAY).
    """
    if not all_race_frames:
        return pd.DataFrame(columns=["Abbreviation", "RollingForm"])

    n = len(all_race_frames)
    weights = [DECAY ** (n - 1 - i) for i in range(n)]
    wsum = float(sum(weights))
    weights = [w / wsum for w in weights]

    chunks = []
    for w, df in zip(weights, all_race_frames):
        sub = pd.DataFrame({
            "Abbreviation": df["Abbreviation"],
            "w_points": _points_series(df) * w
        })
        chunks.append(sub)

    cat = pd.concat(chunks, ignore_index=True)
    avgp = cat.groupby("Abbreviation", as_index=False)["w_points"].sum()
    avgp["RollingForm"] = (avgp["w_points"] * SCALE).clip(0, 100)
    return avgp[["Abbreviation", "RollingForm"]]

# ========= Build stats =========
def build_current_stats() -> tuple[pd.DataFrame, dict]:
    files = feature_files()
    if not files:
        raise FileNotFoundError(f"No processed races in {PROC}")

    race_frames = [pd.read_csv(p) for p in files]

    mini = []
    for df in race_frames:
        keep = [c for c in [
            "Abbreviation", "TeamName", "Position", "FinishPosition", "Status",
            "Points", "PenaltiesCount", "Pitstop_eff", "PitStopEff", "PitLossAvg_s"
        ] if c in df.columns]
        if "Abbreviation" not in keep:
            continue
        mini.append(df[keep].copy())

    rows = []
    for abbr, grp in pd.concat(mini, ignore_index=True).groupby("Abbreviation"):
        team = grp["TeamName"].dropna().iloc[0] if "TeamName" in grp.columns and grp["TeamName"].notna().any() else None
        races = len(grp)

        pts_total = float(_points_series(grp).sum())
        finished = int(_finished_mask(grp).sum())
        dnf = int(_dnf_mask(grp).sum())
        penal = int(_penalized_mask(grp).sum())

        pit = _pit_eff_series(grp)
        avg_pit_eff = float(pit.mean(skipna=True)) if pit.notna().any() else np.nan

        rows.append({
            "Abbreviation": abbr,
            "TeamName": team,
            "TotalRaces": races,
            "PointsTotal": pts_total,
            "FinishRate": finished / races if races > 0 else 0.0,
            "DNFRate":    dnf / races if races > 0 else 0.0,
            "PenaltyRate": penal / races if races > 0 else 0.0,
            # IMPORTANT: write using the canonical name used in current stats:
            "AvgPitEff": avg_pit_eff,
        })

    stats = pd.DataFrame(rows)

    # Rolling form
    rf = _rolling_form(race_frames)
    stats = stats.merge(rf, on="Abbreviation", how="left")

    # tidy types
    for c in ["PointsTotal", "FinishRate", "DNFRate", "PenaltyRate", "AvgPitEff", "RollingForm"]:
        if c in stats.columns:
            stats[c] = pd.to_numeric(stats[c], errors="coerce")

    # order by points desc then abbr
    stats = stats.sort_values(["PointsTotal", "Abbreviation"], ascending=[False, True]).reset_index(drop=True)

    meta = {
        "rounds": [_round_of(p) for p in files],
        "n_rounds": len(files),
        "last_file": files[-1].name
    }
    return stats, meta








def main():
    new_stats, meta = build_current_stats()

    if OUT.exists():
        old = pd.read_csv(OUT)
        # Compare on shared metrics
        cmp_cols = ["PointsTotal", "FinishRate", "DNFRate", "PenaltyRate", "AvgPitEff", "RollingForm"]
        common = ["Abbreviation"] + [c for c in cmp_cols if c in new_stats.columns and c in old.columns]
        merged = (old[common]
                  .merge(new_stats[common], on="Abbreviation", how="outer", suffixes=("_old", "_new")))
        # count any differences (rounded to avoid tiny float noise)
        changed = 0
        for c in common:
            if c == "Abbreviation": 
                continue
            a = pd.to_numeric(merged[f"{c}_old"], errors="coerce").round(6)
            b = pd.to_numeric(merged[f"{c}_new"], errors="coerce").round(6)
            changed += int((a != b).sum())
        if changed == 0:
            print(f"✅ No update needed — {OUT.name} already reflects {meta['n_rounds']} rounds")
            return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_stats.to_csv(OUT, index=False)
    print(f"✅ Updated {OUT.name}: {len(new_stats)} drivers, rounds={meta['n_rounds']} (last={meta['last_file']}).")

if __name__ == "__main__":
    main()
    

    
    
# --- Cleanup Block: Remove DOO if present ---


try:
    df = pd.read_csv(OUT)
    if "DOO" in df["Abbreviation"].values:
        df = df[df["Abbreviation"] != "DOO"]
        df.to_csv(OUT, index=False)
        print("🧹 Removed DOO from rolling form stats file.")
except Exception as e:
    print("⚠️ Cleanup step failed:", e)
    
    
    
    
    
    
    
    
    
    
"""DNF Chances"""


ALPHA = 1.0   # Laplace prior for Beta smoothing
BETA  = 1.0

# blend weights (driver-heavy by default)
L_DRIVER = 0.5
L_TEAM   = 0.2
L_TRACK  = 0.2
L_SEASON = 0.1

# Markov-style adjustment gain for driver component
GAMMA = 0.5

# project root
ROOT = Path.home() / "Documents" / "F1-Project-Folder"
PROC = ROOT / "Data_Processed" / "fastf1"
OUTP = ROOT / "Data_Processed" / "prediction_data"
OUTP.mkdir(parents=True, exist_ok=True)


def _is_finished(status: str) -> bool:
    """Heuristic: 'Finished' or '+N Lap(s)' => finished. Everything else counts as DNF."""
    if not isinstance(status, str):
        return False
    s = status.strip().lower()
    return ("finished" in s) or ("lap" in s)  # e.g., '+1 Lap', '+2 Laps'

def _beta_mean_var(F: int, N: int, alpha: float = ALPHA, beta: float = BETA) -> tuple[float, float]:
    """Beta posterior mean/variance for a Binomial proportion."""
    a = F + alpha
    b = (N - F) + beta
    mean = a / (a + b)
    var  = (a * b) / (((a + b)**2) * (a + b + 1.0))
    return float(mean), float(var)

def _safe_group_counts(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Return F (dnf count) and N (starts) grouped by keys in `by`."""
    g = df.groupby(by, dropna=False, as_index=False).agg(F=("DNF", "sum"), N=("DNF", "size"))
    return g

def _load_history(years: range | list[int]) -> pd.DataFrame:
    """Load all processed race feature CSVs for the given years into one DataFrame."""
    years = list(years)
    frames = []
    for y in years:
        ydir = PROC / str(y)
        if not ydir.exists():
            continue
        for p in sorted(ydir.glob(f"{y}_R*_features.csv")):
            try:
                df = pd.read_csv(p)
                df["Year"] = y
                frames.append(df)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _prepare_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create a boolean DNF flag from Status, keep only columns we need."""
    use = df.copy()
    # columns we try to use later
    for c in ["Abbreviation", "TeamName", "Country", "EventName", "Status", "Round"]:
        if c not in use.columns:
            use[c] = np.nan
    # DNF flag
    use["DNF"] = ~use["Status"].apply(_is_finished)
    return use

def _posterior_tables(hist: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build Beta-smoothed posteriors (mean,var) for:
      - driver (Abbreviation)
      - team   (TeamName)
      - track  (Country if present else EventName)
      - season (Year)
    """
    h = _prepare_flags(hist)

    # pick track key consistent with your earlier choice
    track_key = "Country" if "Country" in h.columns and h["Country"].notna().any() else "EventName"

    # driver
    dr = _safe_group_counts(h, ["Abbreviation"])
    dr[["p_mean", "p_var"]] = dr.apply(
        lambda r: pd.Series(_beta_mean_var(int(r.F), int(r.N))), axis=1
    )

    # team (use historical team label as-is)
    tm = _safe_group_counts(h, ["TeamName"])
    tm[["p_mean", "p_var"]] = tm.apply(
        lambda r: pd.Series(_beta_mean_var(int(r.F), int(r.N))), axis=1
    )

    # track
    tr = _safe_group_counts(h, [track_key])
    tr.rename(columns={track_key: "TrackKey"}, inplace=True)
    tr[["p_mean", "p_var"]] = tr.apply(
        lambda r: pd.Series(_beta_mean_var(int(r.F), int(r.N))), axis=1
    )

    # season (current-year-to-date component: you can also recompute with partial season later)
    sz = _safe_group_counts(h, ["Year"])
    sz[["p_mean", "p_var"]] = sz.apply(
        lambda r: pd.Series(_beta_mean_var(int(r.F), int(r.N))), axis=1
    )

    return {"driver": dr, "team": tm, "track": tr, "season": sz}




def _driver_markov_adjust(hist: pd.DataFrame, year: int, driver_abbr: str,
                          p_driver_base: float, gamma: float = GAMMA) -> float:
    """
    Adjust driver component based on current-season observed vs expected DNFs.
    """
    h = _prepare_flags(hist)
    this_year = h[(h["Year"] == year) & (h["Abbreviation"] == driver_abbr)]
    F_obs = int(this_year["DNF"].sum())
    N_obs = int(len(this_year))
    E = p_driver_base * N_obs
    # exponential nudge; denominator +1 avoids div-by-zero early season
    return float(p_driver_base * np.exp(gamma * ((F_obs - E) / (E + 1.0))))








def compute_dnf_chance_for_event(
    target_features_csv: Path,
    history_years: range | list[int],
    season_year: int,
    *,
    save: bool = True
) -> pd.DataFrame:
    """
    Build a DNF chance table for each driver in the target event.
    Writes: Data_Processed/prediction_data/dnf_chance_<basename>.csv
    Columns:
      Abbreviation, TeamName, TrackKey, p_driver, p_team, p_track, p_season,
      p_driver_adj, p_final, SE_final
    """
    target_features_csv = Path(target_features_csv)
    event_df = pd.read_csv(target_features_csv).copy()

    # Identify per-driver identity for the event
    cols_ok = [c for c in ["Abbreviation", "TeamName", "Country", "EventName"] if c in event_df.columns]
    if not cols_ok:
        raise ValueError("Target features file missing Abbreviation/Team/Track columns.")
    # choose track key
    track_key = "Country" if "Country" in event_df.columns and event_df["Country"].notna().any() else "EventName"

    # Load history and build posteriors
    hist = _load_history(history_years)
    if hist.empty:
        raise ValueError("No history found under Data_Processed/fastf1 for the requested years.")
    posts = _posterior_tables(hist)

    # index for fast lookups
    dr = posts["driver"].set_index("Abbreviation")
    tm = posts["team"].set_index("TeamName")
    tr = posts["track"].set_index("TrackKey")
    sz = posts["season"].set_index("Year")

    # season posterior for the requested season (to-date)
    if season_year in sz.index:
        p_season, v_season = float(sz.loc[season_year, "p_mean"]), float(sz.loc[season_year, "p_var"])
    else:
        # fallback to global
        p_season = float((posts["season"]["F"].sum() + ALPHA) / (posts["season"]["N"].sum() + ALPHA + BETA))
        # conservative variance: average of season variances
        v_season = float(posts["season"]["p_var"].mean())

    rows = []
    for _, r in event_df.iterrows():
        abbr = r.get("Abbreviation")
        team = r.get("TeamName")
        tkey = r.get(track_key)

        # pull components with safe fallbacks
        if pd.notna(abbr) and abbr in dr.index:
            p_d, v_d = float(dr.loc[abbr, "p_mean"]), float(dr.loc[abbr, "p_var"])
        else:
            # global fallback
            totalF, totalN = int(posts["driver"]["F"].sum()), int(posts["driver"]["N"].sum())
            p_d, v_d = _beta_mean_var(totalF, totalN)

        if pd.notna(team) and team in tm.index:
            p_t, v_t = float(tm.loc[team, "p_mean"]), float(tm.loc[team, "p_var"])
        else:
            totalF, totalN = int(posts["team"]["F"].sum()), int(posts["team"]["N"].sum())
            p_t, v_t = _beta_mean_var(totalF, totalN)

        if pd.notna(tkey) and tkey in tr.index:
            p_c, v_c = float(tr.loc[tkey, "p_mean"]), float(tr.loc[tkey, "p_var"])
        else:
            totalF, totalN = int(posts["track"]["F"].sum()), int(posts["track"]["N"].sum())
            p_c, v_c = _beta_mean_var(totalF, totalN)

        # Markov adjustment for current season on driver component
        p_d_adj = _driver_markov_adjust(hist, season_year, str(abbr), p_d, GAMMA)

        # blend on probability scale
        p_final = (
            L_DRIVER * p_d_adj +
            L_TEAM   * p_t +
            L_TRACK  * p_c +
            L_SEASON * p_season
        )

        # approximate SE of the blend (assuming independence of sources)
        v_final = (
            (L_DRIVER**2) * v_d +
            (L_TEAM**2)   * v_t +
            (L_TRACK**2)  * v_c +
            (L_SEASON**2) * v_season
        )
        se_final = float(np.sqrt(max(v_final, 0.0)))

        rows.append({
            "Abbreviation": abbr,
            "TeamName": team,
            "TrackKey": tkey,
            "p_driver": p_d,
            "p_team": p_t,
            "p_track": p_c,
            "p_season": p_season,
            "p_driver_adj": p_d_adj,
            "p_final": p_final,
            "SE_final": se_final
        })

    out = pd.DataFrame(rows)

    if save:
        out_path = OUTP / f"dnf_chance_{target_features_csv.stem}.csv"
        out.to_csv(out_path, index=False)
        print(f"Saved -> {out_path} (rows={len(out)})")

    return out




#-------------------------
#runner block
#-------------------------




if __name__ == "__main__":
    from pathlib import Path
    import re

    YEAR = 2025
    ROOT = Path.home() / "Documents" / "F1-Project-Folder"
    PROC = ROOT / "Data_Processed" / "fastf1" / str(YEAR)
    OUTDIR = ROOT / "Data_Processed" / "prediction_data"

    # Helper to get round number from filename
    def _round_of(p: Path) -> int:
        m = re.search(rf"{YEAR}_R(\d+)_", p.name)
        return int(m.group(1)) if m else 0

    files = sorted(PROC.glob(f"{YEAR}_R*_features.csv"), key=_round_of)
    if not files:
        raise FileNotFoundError(f"No processed race files found in {PROC}")

    latest = files[-1]
    rnum = _round_of(latest)
    out_path = OUTDIR / f"dnf_chance_{YEAR}_R{rnum:02d}_features.csv"

    if out_path.exists():
        print(f"✅ No update needed — {out_path.name} already exists for round {rnum}.")
    else:
        # Recompute DNF chance using all races up to this round
        dfs = [pd.read_csv(p) for p in files]
        df_all = pd.concat(dfs, ignore_index=True)

        # === your existing DNF computation function ===
        dnf_chances = compute_dnf_chance(df_all)

        OUTDIR.mkdir(parents=True, exist_ok=True)
        dnf_chances.to_csv(out_path, index=False)
        print(f"✅ Updated DNF chances saved -> {out_path.name} (round={rnum})")
        
        







"""Pace Score Calculator"""



from pathlib import Path
import pandas as pd
import numpy as np
import re

# Defaults you chose
PERCENTILE_CAP = 0.90   # p for g_cap
S_CAP          = 20.0   # score at the cap
S_MIN          = 12.0   # floor score so nobody hits 0
EPS_MAD        = 0.02   # small epsilon to avoid tiny-MAD blowups (not strictly needed here)

def _pick_gap_column(df: pd.DataFrame) -> str:
    """
    Pick the best available qualifying 'gap to pole' column from your features.
    Preference order matches your pipeline.
    """
    candidates = [
        "QualiGapToPole_s",   # your assemble_race_features output
        "GapToPole_s",        # generic name used earlier
        "Q3_GapToPole_s",     # if you stored Q3 explicitly
        "BestQualiGapToPole_s"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No qualifying gap-to-pole column found. Looked for: {candidates}")

def compute_event_pace(df: pd.DataFrame,
                       p_cap: float = PERCENTILE_CAP,
                       s_cap: float = S_CAP,
                       s_min: float = S_MIN) -> pd.DataFrame:
    """
    Compute Pace100 for a SINGLE EVENT from a features DataFrame that contains quali gap-to-pole.

    Returns a new DataFrame with at least ['Abbreviation','Pace100'] and
    keeps useful context columns if present (EventName, Round, TeamName).
    """
    df = df.copy()
    gap_col = _pick_gap_column(df)

    # keep minimal id/context columns if present
    keep_cols = [c for c in ["Abbreviation", "Driver", "FullName", "TeamName",
                             "EventName", "Round", "Country", "CircuitShortName"]
                 if c in df.columns]

    out = df[keep_cols].copy() if keep_cols else pd.DataFrame()
    out["Abbreviation"] = out.get("Abbreviation", df.get("Abbreviation"))

    # valid gaps
    gaps = pd.to_numeric(df[gap_col], errors="coerce")
    valid_mask = gaps.notna() & (gaps >= 0)

    if valid_mask.sum() == 0:
        raise ValueError("No valid quali gaps found in this features file.")

    # percentile cap and decay constant
    g_cap = float(np.quantile(gaps[valid_mask], p_cap))
    # guard: if g_cap is extremely small, push it up a bit so exp() has range
    g_cap = max(g_cap, 0.10)

    tau = g_cap / np.log(100.0 / s_cap)

    # compute raw scores for valid runners
    g_clamped = np.minimum(gaps, g_cap)
    s_raw = 100.0 * np.exp(-g_clamped / tau)

    # floor
    pace100 = np.maximum(s_raw, s_min)

    # fill missing/no-time with event mean of valid pace
    event_mean = float(pace100[valid_mask].mean())
    pace100_final = pace100.copy()
    pace100_final[~valid_mask] = event_mean

    out["Pace100"] = pace100_final

    # Round to 2 decimals for readability
    out["Pace100"] = out["Pace100"].round(2)

    return out

def save_event_pace(features_csv: Path,
                    out_dir: Path,
                    p_cap: float = PERCENTILE_CAP,
                    s_cap: float = S_CAP,
                    s_min: float = S_MIN) -> Path:
    """
    Load a single-event features CSV, compute Pace100, and save
    to Data_Processed/prediction_data as pace_scores_<basename>.csv
    """
    features_csv = Path(features_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(features_csv)
    pace_df = compute_event_pace(df, p_cap=p_cap, s_cap=s_cap, s_min=s_min)

    out_path = out_dir / f"pace_scores_{features_csv.stem}.csv"
    pace_df.to_csv(out_path, index=False)
    return out_path








#------------------------
#Runner Block
#------------------------


if __name__ == "__main__":
    from pathlib import Path
    import re
    import pandas as pd

    YEAR = 2025
    ROOT = Path.home() / "Documents" / "F1-Project-Folder"
    QUALI_DIR = ROOT / "Data_Processed" / "quali_feat" / str(YEAR)
    OUT_DIR   = ROOT / "Data_Processed" / "prediction_data"

    QUALI_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find files like: 17_quali_feat_2025.csv
    files = sorted(QUALI_DIR.glob(f"*_quali_feat_{YEAR}.csv"))
    if not files:
        raise FileNotFoundError(f"No quali features found in {QUALI_DIR}")

    def _round_from_name(p: Path) -> int:
        m = re.match(rf"(\d+)_quali_feat_{YEAR}\.csv$", p.name)
        return int(m.group(1)) if m else -1

    latest = max(files, key=_round_from_name)
    rnd = _round_from_name(latest)
    if rnd < 0:
        raise RuntimeError(f"Could not parse round number from {latest.name}")

    print(f"🔎 Latest quali file detected: R{rnd:02d} -> {latest.name}")

    # Compute and save pace for latest only (overwrite to be safe)
    df = pd.read_csv(latest)
    pace_df = compute_event_pace(df, p_cap=PERCENTILE_CAP, s_cap=S_CAP, s_min=S_MIN)

    out_path = OUT_DIR / f"pace_scores_{YEAR}_R{rnd}_features.csv"
    pace_df.to_csv(out_path, index=False)
    print(f"✅ Saved latest pace scores to: {out_path}")


















