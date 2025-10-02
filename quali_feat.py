# Code/get_quali_data.py

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# your helpers
from Code.fastf1 import enable_cache, load_session, quali_features, save_csv

# ---------------- config ----------------
YEAR = 2025
START_RND = 17             # change after each race
END_RND: Optional[int] = 35
MISS_LIMIT_CONSEC = 1      # stop as soon as a round isn't available
CACHE_DIR = "Data/fastf1_cache"
OUT_ROOT = Path("Data_Processed") / "quali_feat"

def export_quali_minimal(year:int, start_rnd:int, end_rnd:Optional[int]) -> None:
    enable_cache(CACHE_DIR)
    out_dir = OUT_ROOT / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)

    last = end_rnd if end_rnd is not None else start_rnd + 50
    consec_misses = 0
    print(f"\n=== Quali minimal export {year} R{start_rnd:02d}..R{last:02d} ===")

    for rnd in range(start_rnd, last+1):
        rel_path = f"quali_feat/{year}/{rnd:02d}_quali_feat_{year}.csv"
        out_path = Path("Data_Processed") / rel_path
        if out_path.exists():
            print(f"➤ {out_path.name} exists, skipping")
            consec_misses = 0
            continue

        try:
            # 1) Load QUALI session
            q = load_session(year, rnd, "Q")

            # 2) Build raw quali features
            raw = quali_features(q)
            if raw is None or raw.empty:
                raise RuntimeError("empty quali_features")

            df = raw.copy()

            # 3) Robustly derive needed columns
            # Grid position
            grid_col = "GridPosition" if "GridPosition" in df.columns else (
                "Position" if "Position" in df.columns else None
            )
            if grid_col is None:
                raise KeyError("No 'GridPosition'/'Position' column in quali features")

            # Gap to pole in seconds
            if "gap_to_pole" in df.columns:
                gap_s = pd.to_numeric(df["gap_to_pole"], errors="coerce")
            else:
                # derive from best quali time
                q_best = pd.to_numeric(df.get("q_best"), errors="coerce")
                if not np.isfinite(q_best).any():
                    raise RuntimeError("Cannot derive gap_to_pole: no 'gap_to_pole' or 'q_best'")
                pole_time = np.nanmin(q_best)
                gap_s = q_best - pole_time

            # 4) Minimal table
            out = pd.DataFrame({
                "Abbreviation": df["Abbreviation"],
                "GridPosition": pd.to_numeric(df[grid_col], errors="coerce"),
                "QualiGapToPole_s": pd.to_numeric(gap_s, errors="coerce"),
            })

            # 5) Clean up
            out = out.sort_values("GridPosition", kind="mergesort").reset_index(drop=True)

            # Ensure pole is exactly 0.000
            if len(out) and pd.notna(out.loc[0, "GridPosition"]) and int(out.loc[0, "GridPosition"]) == 1:
                out.loc[0, "QualiGapToPole_s"] = 0.0

            # Forward-fill any missing gaps (DNF/NoTime -> inherit previous)
            out["QualiGapToPole_s"] = (
                out["QualiGapToPole_s"]
                .ffill()      # use previous runner’s gap
                .bfill()      # in case the first was NaN for some reason
                .clip(lower=0.0)
                .round(3)
            )

            # Final types
            out["GridPosition"] = out["GridPosition"].astype("Int64")

            # 6) Save (only 3 columns)
            # use your save_csv wrapper to keep everything under Data_Processed/
            save_csv(out, rel_path)
            print(f"✅ Saved {rel_path} (rows={len(out)})")
            consec_misses = 0

        except Exception as e:
            consec_misses += 1
            print(f"❌ {year} R{rnd:02d} failed: {e} (consec miss {consec_misses})")
            if consec_misses >= MISS_LIMIT_CONSEC:
                print("▲ Stopping: consecutive-miss threshold reached.")
                break

    print("Done.")

if __name__ == "__main__":
    export_quali_minimal(YEAR, START_RND, END_RND)