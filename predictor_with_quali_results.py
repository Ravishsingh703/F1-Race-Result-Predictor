
# predictor_quali.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import re


# ===== Weights =====
w_grid = 55
w_form = 15
w_pen  = 5
w_pace = 5
beta   = 0.50   # grid decay factor

# ===== Paths =====
ROOT = Path.home() / "Documents" / "F1-Project-Folder"
FEAT = ROOT / "Data_Processed" / "fastf1" / "2025" / "2025_R17_features.csv"
PACE = ROOT / "Data_Processed" / "prediction_data" / "pace_scores_2025_R17_features.csv"
DNFF = ROOT / "Data_Processed" / "prediction_data" / "dnf_chance_2025_R17_features.csv"
CURR = ROOT / "Data_Processed" / "driver_stats_over_years" / "current_driver_stats_2025.csv"
OUTD = ROOT / "Data_Processed" / "prediction_data"
OUTD.mkdir(parents=True, exist_ok=True)

# ===== Load =====
feat = pd.read_csv(FEAT)[["Abbreviation", "GridPosition",]]
pace = pd.read_csv(PACE)[["Abbreviation", "Pace100"]]
dnf  = pd.read_csv(DNFF)[["Abbreviation", "p_final"]].rename(columns={"p_final": "DNFChance"})
curr = pd.read_csv(CURR)[["Abbreviation", "RollingForm", "PenaltyRate"]]

# ===== Merge =====
df = (
    feat
    .merge(curr, on="Abbreviation", how="left")
    .merge(pace, on="Abbreviation", how="left")
    .merge(dnf,  on="Abbreviation", how="left")
    .sort_values("GridPosition", kind="mergesort")
    .reset_index(drop=True)
)

# Coerce numerics & clip
for c in ["GridPosition", "RollingForm", "PenaltyRate", "Pace100", "DNFChance"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["RollingForm"] = df["RollingForm"].fillna(0.0).clip(0, 100)
df["PenaltyRate"] = df["PenaltyRate"].fillna(0.0).clip(0, 1)
df["Pace100"]     = df["Pace100"].fillna(0.0).clip(0, 100)
df["DNFChance"]   = df["DNFChance"].fillna(0.0).clip(0, 1)

# ===== Features scaled 0–1 =====
df["Form01"]     = df["RollingForm"] / 100.0
df["LowPenalty"] = 1.0 - df["PenaltyRate"]
df["Pace01"]     = df["Pace100"] / 100.0
df["Safe01"]     = 1.0 - df["DNFChance"]

# ===== GridAdv (backfill NaN gaps with previous driver’s value) =====
grid_pos = df["GridPosition"].fillna(df["GridPosition"].max())
grid_adv = np.exp(-beta * (grid_pos - 1).clip(lower=0))
df["GridAdv"] = pd.Series(grid_adv).fillna(method="ffill")

# ===== Score =====
df["Score"] = (
    w_grid * df["GridAdv"]
  + w_form * df["Form01"]
  + w_pen  * df["LowPenalty"]
  + w_pace * df["Pace01"]
) * df["Safe01"]

# ===== Softmax -> WinProb =====

gamma = 1.0     # 1.0 = linear; 0.7 = flatter; 1.3 = peakier
eps   = 1e-9    # small smoothing to avoid division by zero

s = df["Score"].astype(float).to_numpy()

# Shift so the smallest becomes ~0 (keeps relative gaps)
shift = min(s.min(), 0.0)
s_pos = s - shift

# Optional power shaping (set gamma=1.0 if you want pure linear)
weights = np.power(s_pos, gamma)

# Add tiny epsilon so backmarkers don't become exactly zero
weights = weights + eps

# Normalize to probabilities
df["WinProb"] = weights / weights.sum()











#-----------------------
#Monte-Carlo skeleton
#-----------------------


def monte_carlo_finish_table(df: pd.DataFrame,
                             n_iter: int = 150000,
                             tau: float = 20.0,
                             dnf_scale: float = 1.0,
                             jitter_sigma: float = 0.05,
                             rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = np.random.default_rng() if rng is None else rng
    drivers = df["Abbreviation"].tolist()
    n = len(drivers)

    # Base strengths from your Score
    base_strength = np.exp(df["Score"].to_numpy() / float(tau))  # shape (n,)
    safe01 = (1.0 - df["DNFRate"].to_numpy()).clip(0, 1)         # if not present, compute

    # tallies
    pos_counts = np.zeros((n, n), dtype=np.int64)  # [driver_index, finish_position-1]
    win_counts = np.zeros(n, dtype=np.int64)
    podium_counts = np.zeros(n, dtype=np.int64)
    top10_counts = np.zeros(n, dtype=np.int64)

    for _ in range(n_iter):
        # 1) DNFs
        dnf_draw = rng.random(n) > (safe01 ** dnf_scale)
        alive_idx = np.where(~dnf_draw)[0]

        if alive_idx.size == 0:
            # corner case: everybody DNF -> skip this iteration
            continue

        # 2) strength jitter this iteration
        eps = rng.normal(0.0, jitter_sigma, size=n)
        s_iter = base_strength * np.exp(eps)

        # Remove DNFs from contention
        avail = alive_idx.tolist()
        order = []

        # 3) Plackett–Luce cascade sampling
        while avail:
            weights = s_iter[avail]
            # numeric safety
            wsum = weights.sum()
            if wsum <= 0:
                # fallback: uniform among remainers
                probs = np.ones_like(weights) / len(weights)
            else:
                probs = weights / wsum
            pick = rng.choice(len(avail), p=probs)
            chosen = avail.pop(pick)
            order.append(chosen)

        # 4) Assign race positions to alive (DNFs implicitly at the back, if you want to track them)
        # Here we count only finishers in 1..len(alive), DNFs can be ignored or tallied as pos=n+something
        for p, idx in enumerate(order, start=1):
            pos_counts[idx, p-1] += 1
            if p == 1: win_counts[idx] += 1
            if p <= 3: podium_counts[idx] += 1
            if p <= 10: top10_counts[idx] += 1

    # Convert counts to probabilities
    denom = max(n_iter, 1)
    pos_prob = pos_counts / denom
    out = pd.DataFrame({
        "Abbreviation": drivers,
        "WinProb": win_counts / denom,
        "PodiumProb": podium_counts / denom,
        "Top10Prob": top10_counts / denom,
        "ExpFinish": (pos_prob * np.arange(1, n+1)).sum(axis=1)
    })

    # Attach full position distributions (P1..Pn) if you want
    for p in range(1, n+1):
        out[f"P{p}"] = pos_prob[:, p-1]

    # Sort by WinProb (or ExpFinish ascending)
    out = out.sort_values("WinProb", ascending=False).reset_index(drop=True)
    return out











# ==== MONTE CARLO FINISH ORDER RUNNER ===============================



ROOT = Path.home() / "Documents" / "F1-Project-Folder"
PRED_DIR = ROOT / "Data_Processed" / "prediction_data"
FINAL_DIR = PRED_DIR / "final_predictions"
SIMS = 1_000_000
SEED = 123



def _latest_win_probs() -> Path:
    cand = sorted(PRED_DIR.glob("win_probs_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not cand:
        raise FileNotFoundError(f"No win_probs_*.csv in {PRED_DIR}")
    return cand[0]


def _ensure_df(src: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    p = Path(src)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def _weights_from_df(df: pd.DataFrame, *, temp: float = 1.0) -> np.ndarray:
    """
    Build a positive weight vector for Plackett–Luce sampling.
    - Prefer WinProb if present (already 0..1).
    - Otherwise derive from Score via a softmax (temperature controls sharpness).
    """
    if "WinProb" in df.columns:
        w = pd.to_numeric(df["WinProb"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        # guard against exact zeros
        return np.clip(w, 1e-12, None)

    if "Score" in df.columns:
        s = pd.to_numeric(df["Score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        # Stabilize then softmax so better scores => larger weights
        s = (s - np.nanmean(s)) / (np.nanstd(s) + 1e-9)      # normalize
        x = s / max(temp, 1e-6)                               # temperature
        x = x - np.nanmax(x)                                  # numeric stability
        w = np.exp(x)
        return np.clip(w, 1e-12, None)

    raise ValueError("Prediction table needs a 'WinProb' or 'Score' column.")
    




def _round_number_guess(df: pd.DataFrame, event_name: str) -> int:
    # 1) if df already has Round or RoundNumber
    for c in ("RoundNumber", "Round"):
        if c in df.columns:
            try:
                val = int(pd.to_numeric(df[c], errors="coerce").dropna().iloc[0])
                return val
            except Exception:
                pass

    # 2) try to parse from any pace_scores_*_*R##*_features.csv present
    pace_files = sorted(PRED_DIR.glob("pace_scores_*R*_features.csv"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for p in pace_files:
        m = re.search(r"R(\d+)", p.stem)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass

    # 3) last resort: use next index based on existing final_predictions files
    existing = sorted(FINAL_DIR.glob("*_position_probability.csv"))
    if existing:
        nums = []
        for f in existing:
            m = re.match(r"(\d+)_position_probability\.csv$", f.name)
            if m:
                try:
                    nums.append(int(m.group(1)))
                except Exception:
                    pass
        if nums:
            return max(nums) + 1
    return 0


def _plackett_luce_counts(drivers: np.ndarray, weights: np.ndarray,
                          sims: int, seed: int) -> np.ndarray:
    """
    Return counts[pos, driver] after sampling full orders sims times
    using Gumbel-Max keys ~ log(weights) + Gumbel(0,1).
    """
    rng = np.random.default_rng(seed)
    n = len(drivers)
    counts = np.zeros((n, n), dtype=np.int64)
    wlog = np.log(weights)
    for _ in range(sims):
        keys = wlog + rng.gumbel(size=n)
        order = np.argsort(-keys)  # winner first
        counts[np.arange(n), order] += 1
    return counts


def build_position_probs(win_probs_src: str | Path | pd.DataFrame,
                         sims: int = SIMS, seed: int = SEED) -> tuple[pd.DataFrame, str, int]:
    
    # load
    src_path = (win_probs_src if isinstance(win_probs_src, (str, Path))
                else _latest_win_probs())
    if not isinstance(win_probs_src, pd.DataFrame):
        df = _ensure_df(src_path)
        src_path = Path(src_path)
    else:
        df = win_probs_src.copy()
        src_path = Path("win_probs_from_dataframe.csv")

    # order by win-prob for stability (optional)
    if "WinProb" in df.columns:
        df = df.sort_values("WinProb", ascending=False, kind="mergesort").reset_index(drop=True)

    drivers = df["Abbreviation"].astype(str).to_numpy()
    weights = _weights_from_df(df)
    n = len(drivers)

    counts = _plackett_luce_counts(drivers, weights, sims=sims, seed=seed)
    probs = counts / float(sims)  # [pos, driver]

    # --- Patch: zero-out tail probs for strong favorites ---
    for j in range(probs.shape[1]):  # loop over drivers
        row = probs[:, j]  # distribution over positions
        csum = row.cumsum()
        # If by position k cumulative prob > 0.98, then set all remaining positions to 0
        cutoff = np.argmax(csum > 0.98)
        if csum[-1] > 0.98 and cutoff < len(row) - 1:
            row[cutoff+1:] = 0.0
            # renormalize the distribution so it sums to 1
            row /= row.sum()
            probs[:, j] = row


    # assemble output
    cols = ["EventName", "Abbreviation"] + [f"P{i}" for i in range(1, n + 1)]
    out = pd.DataFrame(columns=cols)
    out["EventName"] = _event_name_from(df, src_path)
    out["Abbreviation"] = drivers
    for i in range(n):
        out[f"P{i+1}"] = probs[i, :]  # note: probs[pos, driver]

    # match the 20-column shape if you always want P1..P20:
    for j in range(n + 1, 21):
        out[f"P{j}"] = 0.0

    # Optional: mean finishing position
    positions = np.arange(1, n + 1)
    out["MeanPos"] = (probs[:n, :].T @ positions)

    # metadata for naming
    ev = str(out["EventName"].iloc[0])
    rnd = _round_number_guess(df, ev)

    return out, ev, rnd



if __name__ == "__main__":
    try:
        # Locate the latest win_probs file
        win_probs_csv = _latest_win_probs()

        # Build position probability table
        pos_df, _, round_no = build_position_probs(win_probs_csv, sims=SIMS, seed=SEED)

        # Drop EventName column if present
        if "EventName" in pos_df.columns:
            pos_df = pos_df.drop(columns=["EventName"])

        # Ensure output directory exists
        FINAL_DIR.mkdir(parents=True, exist_ok=True)

        # Save with round number in filename
        out_path = FINAL_DIR / f"{round_no:02d}_position_probability.csv"
        pos_df.to_csv(out_path, index=False)

        print(f"✅ Position probability file saved: {out_path}")
        print(f"   Round: {round_no}")
        print(pos_df.head(10).to_string(index=False))

    except Exception as e:
        print("❌ Runner failed:", e)
        
        

        
        
        
        
        

# final_prediction_runner.py


# ---- Config ----
ROOT = Path.home() / "Documents" / "F1-Project-Folder"
PRED_DIR = ROOT / "Data_Processed" / "prediction_data" / "final_predictions"
N_SIMS = 10_000_000
SEED = 123

def latest_position_prob_file() -> Path:
    files = sorted(PRED_DIR.glob("*_position_probability.csv"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No *_position_probability.csv found in {PRED_DIR}")
    return files[0]

def round_from_filename(p: Path) -> int:
    m = re.match(r"(\d+)_position_probability\.csv$", p.name)
    return int(m.group(1)) if m else 0

def simulate_final_order(prob_csv: Path, n_sims: int = N_SIMS, seed: int = SEED) -> pd.DataFrame:
    df = pd.read_csv(prob_csv)
    # Identify position columns P1..Pn
    pos_cols = [c for c in df.columns if c.startswith("P")]
    pos_cols = sorted(pos_cols, key=lambda s: int(s[1:]))
    if not pos_cols:
        raise ValueError("No P1..Pn columns found in probability file.")

    drivers = df["Abbreviation"].astype(str).to_numpy()
    n_drivers = len(drivers)

    probs = df[pos_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, None)
    row_sums = probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums
    cdf = np.cumsum(probs, axis=1)

    rng = np.random.default_rng(seed)
    U = rng.random((n_sims, n_drivers))
    sampled_pos = np.empty((n_sims, n_drivers), dtype=np.int16)
    for j in range(n_drivers):
        sampled_pos[:, j] = np.searchsorted(cdf[j], U[:, j], side="right") + 1

    mean_pos = sampled_pos.mean(axis=0)
    order = np.argsort(mean_pos)
    predicted_position = np.empty_like(order)
    predicted_position[order] = np.arange(1, n_drivers + 1)

    out = pd.DataFrame({
        "Abbreviation": drivers,
        "MeanPos": mean_pos,
        "PredictedPosition": predicted_position
    }).sort_values("PredictedPosition").reset_index(drop=True)

    return out

if __name__ == "__main__":
    prob_file = latest_position_prob_file()
    rnd = round_from_filename(prob_file)
    print(f"🔎 Using position-prob table: {prob_file.name} (round {rnd})")

    final_df = simulate_final_order(prob_file, n_sims=N_SIMS, seed=SEED)

    out_path = prob_file.parent / f"{rnd}_final_prediction.csv"
    final_df[["Abbreviation", "PredictedPosition"]].to_csv(out_path, index=False)
    print(f"✅ Saved final standings -> {out_path}")
    print(final_df.head(10).to_string(index=False))
        






