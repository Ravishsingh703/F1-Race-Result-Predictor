# ============================================================
# 🏎️ PREDICTOR — Unified Quali-Based Prediction Engine
# ============================================================

from __future__ import annotations

# ===== Core Libraries =====
import os
import re
import math
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# ===== FastF1 / External Data =====
import fastf1
from fastf1.ergast import Ergast

# ===== Config & Reload =====
import sys, importlib

try:
    # If running inside the global runner
    if "Code.config" in sys.modules:
        config = sys.modules["Code.config"]
        importlib.reload(config)
    else:
        # Fallback for standalone runs
        import Code.config as config
        sys.modules["Code.config"] = config

except Exception as e:
    print(f"[WARN] Could not reload Code.config dynamically: {e}")
    import Code.config as config


# ============================================================
# 🧭 Setup Globals
# ============================================================

YEAR  = config.YEAR
ROUND = config.ROUND
ROOT  = Path.home() / "Documents" / "F1-Project-Folder"
DATA  = ROOT / "Data_Processed"

print(f"✅ Predictor Config Loaded: YEAR={YEAR}, ROUND={ROUND}")



# ============================================================
#  Pace Score + Grid Position Extractor (Quali → Champ_Rating)
# ============================================================

print(f"\n🏎️ Computing Pace Scores for {YEAR} – Round {ROUND}")

# ---- Paths ----
QUALI_FILE = DATA / "quali_feat" / str(YEAR) / f"{ROUND:02d}_quali_feat_{YEAR}.csv"
CHAMP_FILE = DATA / "quali_feat" / "Champ_Rating" / f"champ_standing_live_{YEAR}.csv"
OUT_FILE   = DATA / "quali_feat" / "Champ_Rating" / f"all_scores_after_quali_r{ROUND}_{YEAR}.csv"

# ---- Configurable scaling constant ----
TAU = 0.7   # exponential drop-off constant

# ============================================================
# Load Qualifying Data
# ============================================================

if not QUALI_FILE.exists():
    raise FileNotFoundError(f"❌ Qualifying file not found: {QUALI_FILE}")

qdf = pd.read_csv(QUALI_FILE)
qdf.columns = [c.strip() for c in qdf.columns]  # clean column names

# Flexible grid column detection
grid_col = None
for c in ["GridPosition", "Position", "Pos", "StartingGrid"]:
    if c in qdf.columns:
        grid_col = c
        break

if grid_col is None:
    raise KeyError("❌ Could not find grid position column in quali file.")

# Make sure key fields exist
if "GapToPole_s" not in qdf.columns or "Abbreviation" not in qdf.columns:
    raise KeyError("❌ Quali file must contain 'Abbreviation' and 'GapToPole_s'.")

qdf["Abbreviation"] = qdf["Abbreviation"].astype(str).str.upper().str.strip()
qdf["GapToPole_s"]  = pd.to_numeric(qdf["GapToPole_s"], errors="coerce").fillna(0.0)
qdf["GridPosition"] = pd.to_numeric(qdf[grid_col], errors="coerce").astype("Int64")

# ============================================================
# Compute Pace Score
# ============================================================

# Exponential decay: small gaps stay close to 100, large gaps fall faster
qdf["PaceScore"] = 100 * np.exp(-qdf["GapToPole_s"] / TAU)
qdf["PaceScore"] = qdf["PaceScore"].clip(1, 100).round(2)

print("\n📊 Sample pace scaling:")
print(qdf[["Abbreviation", "GapToPole_s", "PaceScore", "GridPosition"]].head(10))

# ============================================================
# Merge with Championship File
# ============================================================

if not CHAMP_FILE.exists():
    raise FileNotFoundError(f"❌ Championship file not found: {CHAMP_FILE}")

cdf = pd.read_csv(CHAMP_FILE)
cdf["Abbreviation"] = cdf["Abbreviation"].astype(str).str.upper().str.strip()

# Keep only relevant columns
keep_cols = [c for c in cdf.columns if c.lower() in
             {"abbreviation", "totalpoints", "champ_rating", "rolling_form"}]
cdf = cdf[keep_cols].copy()

# Merge pace + grid position
merged = cdf.merge(
    qdf[["Abbreviation", "PaceScore", "GridPosition"]],
    on="Abbreviation",
    how="left"
)

# ============================================================
# Save Output
# ============================================================

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_FILE, index=False)

print(f"\n✅ Saved combined pace + grid file → {OUT_FILE.name}")
print(merged.head(10))










# ============================================================
#  Win Probability Generator (Weighted by Champ/Form/Pace)
# ============================================================

print(f"\n🏁 Computing Win Probabilities for {YEAR} – Round {ROUND}")

# ---- Input & Output Paths ----
IN_FILE  = DATA / "quali_feat" / "Champ_Rating" / f"all_scores_after_quali_r{ROUND}_{YEAR}.csv"
OUT_DIR  = DATA / "prediction_data" / "2025_Predictions" / f"R{ROUND}_Prediction"
OUT_FILE = OUT_DIR / f"R{ROUND}_win_probability.csv"

# ---- Weights (adjustable later) ----
W_CHAMP = 10
W_FORM  = 25
W_PACE  = 65

# ============================================================
# Load Data
# ============================================================

if not IN_FILE.exists():
    raise FileNotFoundError(f"❌ Input file not found: {IN_FILE}")

df = pd.read_csv(IN_FILE)
df.columns = [c.strip() for c in df.columns]

required_cols = {"Abbreviation", "Champ_Rating", "Rolling_Form", "PaceScore"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"❌ Missing columns in input file: {missing}")

df["Abbreviation"] = df["Abbreviation"].astype(str).str.upper().str.strip()

# Ensure numeric fields are valid
for c in ["Champ_Rating", "Rolling_Form", "PaceScore"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# ============================================================
# Compute Weighted Win Score
# ============================================================

df["RawScore"] = (
    W_CHAMP * df["Champ_Rating"]
  + W_FORM  * df["Rolling_Form"]
  + W_PACE  * df["PaceScore"]
)

# Normalize to probabilities summing to 100
df["WinProb"] = (df["RawScore"] / df["RawScore"].sum()) * 100
df["WinProb"] = df["WinProb"].round(7)

# ============================================================
#  Handle Grid Position + Sorting
# ============================================================

grid_col = None
for c in ["GridPosition", "Position", "StartingGrid"]:
    if c in df.columns:
        grid_col = c
        break

if grid_col is None:
    df["GridPosition"] = np.nan
else:
    df.rename(columns={grid_col: "GridPosition"}, inplace=True)

# Sort by grid position (ascending)
df = df.sort_values("GridPosition", ascending=True, na_position="last").reset_index(drop=True)

# Final output selection
out = df[["Abbreviation", "GridPosition", "WinProb"]]

# ============================================================
# Save Output
# ===========================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print(f"\n✅ Win probability file created → {OUT_FILE}")
print(out.head(10))





# ============================================================
# Monte Carlo Simulator – Stage 1 (Clean Version)
# 10M simulations estimating per-position probabilities
# ============================================================

print(f"\n🎯 Running 10M Monte Carlo Simulations for Round {ROUND} ({YEAR})")

# ---- Input & Output Paths ----
WIN_FILE = DATA / "prediction_data" / "2025_Predictions" / f"R{ROUND}_Prediction" / f"R{ROUND}_win_probability.csv"
OUT_FILE = DATA / "prediction_data" / "2025_Predictions" / f"R{ROUND}_Prediction" / f"R{ROUND}_finish_probability_matrix.csv"

N_ITER = 10_000_000  # total simulations
CHUNK  = 250_000     # per batch for memory safety
MIN_PROB = 1e-6
RNG = np.random.default_rng(42)

# ============================================================
# Load Win Probabilities
# ============================================================

if not WIN_FILE.exists():
    raise FileNotFoundError(f"❌ Win probability file not found: {WIN_FILE}")

df = pd.read_csv(WIN_FILE)
if not {"Abbreviation", "WinProb"}.issubset(df.columns):
    raise KeyError("❌ File must contain 'Abbreviation' and 'WinProb' columns.")

df["Abbreviation"] = df["Abbreviation"].astype(str).str.upper().str.strip()
df["WinProb"] = pd.to_numeric(df["WinProb"], errors="coerce").fillna(0.0)
df = df[df["WinProb"] > 0].reset_index(drop=True)

drivers = df["Abbreviation"].tolist()
n = len(drivers)

# Normalize weights
w = np.asarray(df["WinProb"], dtype=float)
w = np.clip(w, MIN_PROB, None)
w = w / w.sum()
logw = np.log(w)

print(f"[INFO] Loaded {n} drivers with normalized probabilities.")

# ============================================================
# Initialize Position Tallies
# ============================================================

pos_counts = np.zeros((n, n), dtype=np.int64)  # [driver, pos-1]

# ============================================================
# Run Monte Carlo Simulation
# ============================================================

remaining = N_ITER
while remaining > 0:
    m = min(CHUNK, remaining)

    # Efficient permutation sampling using Gumbel–Max trick
    u = RNG.random((m, n))
    g = -np.log(-np.log(u))
    scores = logw + g

    order = np.argsort(-scores, axis=1)  # descending (highest score = P1)

    # Tally counts for each finishing position
    for k in range(n):
        idx_at_k = order[:, k]
        bc = np.bincount(idx_at_k, minlength=n)
        pos_counts[:, k] += bc

    remaining -= m
    print(f"[Progress] {N_ITER - remaining:,}/{N_ITER:,} simulations complete...")

print("\n✅ Monte Carlo simulation finished.")

# ============================================================
# Convert to Probabilities
# ============================================================

pos_prob = pos_counts / float(N_ITER)
pos_prob[pos_prob < MIN_PROB] = 0.0

# ============================================================
# Build and Save Output
# ============================================================

out = pd.DataFrame({"Abbreviation": drivers})
for k in range(n):
    out[f"P{k+1}"] = np.round(pos_prob[:, k] * 100, 5)  # convert to % probability

# Keep only P1–Pn probabilities
out = out.sort_values("P1", ascending=False).reset_index(drop=True)

# Save
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_FILE, index=False)

print(f"\n✅ Finish probability matrix saved → {OUT_FILE}")
print(out.head(10))










# ============================================================
#  Monte Carlo – Stage 2
#  Generates final predicted finishing order (10M sims)
#  Output: Position, Abbreviation
# ============================================================

print(f"\n Running Stage 2 Monte Carlo Final Prediction for Round {ROUND} ({YEAR})")

# ---- Input & Output Paths ----
MATRIX_FILE = DATA / "prediction_data" / "2025_Predictions" / f"R{ROUND}_Prediction" / f"R{ROUND}_finish_probability_matrix.csv"
OUT_FILE    = DATA / "prediction_data" / "2025_Predictions" / f"R{ROUND}_Prediction" / f"R{ROUND}_Final_Prediction.csv"

N_ITER = 10_000_000
RNG = np.random.default_rng(123)

# ============================================================
# Load Finish Probability Matrix
# ============================================================

if not MATRIX_FILE.exists():
    raise FileNotFoundError(f"❌ Probability matrix not found: {MATRIX_FILE}")

df = pd.read_csv(MATRIX_FILE)
if "Abbreviation" not in df.columns:
    raise KeyError("❌ 'Abbreviation' column missing in finish probability matrix.")

# Keep only P# columns
p_cols = [c for c in df.columns if c.startswith("P")]
if not p_cols:
    raise ValueError("❌ No position probability columns found (expected P1..Pn).")

# Convert to numeric probabilities (0–1)
for c in p_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) / 100.0

drivers = df["Abbreviation"].tolist()
n = len(drivers)

print(f"[INFO] Loaded {n} drivers and {len(p_cols)} position columns.")

# ============================================================
# Sequential Monte Carlo Simulation
# ============================================================

remaining = df.copy()
final_order = []

for pos_idx in range(1, n + 1):
    col = f"P{pos_idx}"
    if col not in remaining.columns:
        print(f"[WARN] Column {col} missing, stopping early.")
        break

    # Extract probabilities for current position among remaining drivers
    probs = remaining[col].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-9, None)
    probs = probs / probs.sum()  # renormalize to 1

    # Monte Carlo draw (vectorized multinomial)
    counts = RNG.multinomial(N_ITER, probs)
    winner_idx = np.argmax(counts)
    winner = remaining.iloc[winner_idx]["Abbreviation"]

    print(f" P{pos_idx:<2}: {winner}")

    final_order.append({"Position": pos_idx, "Abbreviation": winner})

    # Remove chosen driver before next position
    remaining = remaining[remaining["Abbreviation"] != winner].reset_index(drop=True)

# ============================================================
# Save Final Order
# ============================================================

final_df = pd.DataFrame(final_order)
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
final_df.to_csv(OUT_FILE, index=False)

print(f"\n✅ Final predicted order saved → {OUT_FILE}")
print(final_df.head(10))