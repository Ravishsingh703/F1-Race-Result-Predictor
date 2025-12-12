from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import runpy, sys

# ============================================================
# ⚙️ CONFIG IMPORT (works both standalone or from global runner)
# ============================================================

CONFIG_PATH = Path("/Users/ravish/Documents/F1-Project-Folder/Code/config.py")

spec = importlib.util.spec_from_file_location("Code.config", CONFIG_PATH)
config = importlib.util.module_from_spec(spec)
sys.modules["Code.config"] = config
spec.loader.exec_module(config)

# pull live values directly
YEAR = config.YEAR
ROUND = config.ROUND
GP_NAME = config.GP_NAME
ROOT = Path(config.ROOT)

# ============================================================
# 📂 PATHS
# ============================================================
DATA_FILE = ROOT / "Data_Processed" / "wdc_prediction" / "final_predictions" / f"{GP_NAME} Grand Prix_final_order.csv"
IMG_DIR = ROOT / "image_asset" / "drivers" / "celebrations"
VIS_DIR = ROOT / "visualisations" / f"R{ROUND}_{GP_NAME.replace(' ', '_')}_Grand_Prix"
VIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = VIS_DIR / "pre_quali_predictions.png"

# ============================================================
# 🎨 TEAM COLORS 
# ============================================================
TEAM_COLOURS = {
    "Red Bull Racing": "#1E41FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "Kick Sauber": "#52E252",
    "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD",
}

DRIVER_TO_TEAM = {
    "NOR": "McLaren", "VER": "Red Bull Racing", "RUS": "Mercedes", "ANT": "Mercedes",
    "ALB": "Williams", "STR": "Aston Martin", "HUL": "Kick Sauber", "LEC": "Ferrari",
    "PIA": "McLaren", "HAM": "Ferrari", "GAS": "Alpine", "TSU": "Racing Bulls",
    "OCO": "Haas F1 Team", "BEA": "Haas F1 Team", "LAW": "Red Bull Racing",
    "BOR": "Kick Sauber", "ALO": "Aston Martin", "SAI": "Williams",
    "DOO": "Alpine", "HAD": "Racing Bulls"
}

# ============================================================
# 📊 LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
df["Abbreviation"] = df["Abbreviation"].str.strip().str.upper()
df["PosNum"] = df["Position"].str.extract(r"P(\d+)").astype(int)
df = df.sort_values("PosNum")
df["TeamName"] = df["Abbreviation"].map(DRIVER_TO_TEAM)
df["Color"] = df["TeamName"].map(TEAM_COLOURS).fillna("#888888")

# ============================================================
# 🖼️ VISUAL CREATION
# ============================================================
plt.style.use("default")
fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("#D9D9D9")  # grey background
ax.set_facecolor("#D9D9D9")

x_positions = np.arange(len(df))
bar_heights = (21 - df["PosNum"])
bars = ax.bar(x_positions, bar_heights, color=df["Color"], edgecolor="black", linewidth=1.1)

# chart limits and style
ax.set_ylim(0, max(bar_heights) + 6)
ax.set_xlim(-0.5, len(df) - 0.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(df["Abbreviation"], color="black", fontsize=10, fontname="Times New Roman", fontweight="bold")
ax.tick_params(axis="y", left=False, labelleft=False)

# title
ax.set_title(
    f"{GP_NAME} Grand Prix — Pre-Quali Predictions ({YEAR})",
    fontsize=18, color="black", fontname="Times New Roman", fontweight="bold", pad=20
)

# ============================================================
# 🏎️ ADD DRIVER IMAGES
# ============================================================
for x, row in zip(x_positions, df.itertuples()):
    abbrev = row.Abbreviation
    img_path = IMG_DIR / f"{abbrev}.png"

    if not img_path.exists():
        print(f"⚠️ Missing image for {abbrev}")
        continue

    try:
        img = Image.open(img_path).convert("RGBA")
        w, h = img.size
        scale = 2
        new_h = int(500 * scale)
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_arr = np.array(img)

        y_top = (21 - row.PosNum) + 1
        ax.imshow(
            img_arr,
            extent=(x - 1, x + 0.7, y_top, y_top + 2.2),
            aspect="auto",
            zorder=5,
        )

        ax.text(x, y_top + 2, f"P{row.PosNum}",
                ha="center", va="bottom",
                color="black", fontsize=10, weight="bold", fontname="Times New Roman")

    except Exception as e:
        print(f"⚠️ Error with {abbrev}: {e}")

# ============================================================
# 💾 SAVE
# ============================================================
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="#D9D9D9")
plt.close(fig)
print(f"✅ Visual saved → {OUT_FILE}")





# ============================================================
# 📂 PATHS
# ============================================================
DATA_FILE = (
    ROOT / "Data_Processed" / "prediction_data" /
    f"{YEAR}_Predictions" / f"R{ROUND}_Prediction" /
    f"R{ROUND}_Final_Prediction.csv"
)
IMG_DIR = ROOT / "image_asset" / "drivers" / "celebrations"
VIS_DIR = ROOT / "visualisations" / f"R{ROUND}_{GP_NAME.replace(' ', '_')}_Grand_Prix"
VIS_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = VIS_DIR / "post_quali_predictions.png"

# ============================================================
# 🎨 TEAM COLORS
# ============================================================
TEAM_COLOURS = {
    "Red Bull Racing": "#1E41FF",
    "Ferrari": "#DC0000",
    "Mercedes": "#00D2BE",
    "McLaren": "#FF8700",
    "Aston Martin": "#006F62",
    "Alpine": "#0090FF",
    "Williams": "#005AFF",
    "Kick Sauber": "#52E252",
    "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD",
}

DRIVER_TO_TEAM = {
    "NOR": "McLaren", "VER": "Red Bull Racing", "RUS": "Mercedes", "ANT": "Mercedes",
    "ALB": "Williams", "STR": "Aston Martin", "HUL": "Kick Sauber", "LEC": "Ferrari",
    "PIA": "McLaren", "HAM": "Ferrari", "GAS": "Alpine", "TSU": "Racing Bulls",
    "OCO": "Haas F1 Team", "BEA": "Haas F1 Team", "LAW": "Red Bull Racing",
    "BOR": "Kick Sauber", "ALO": "Aston Martin", "SAI": "Williams",
    "DOO": "Alpine", "HAD": "Racing Bulls"
}

# ============================================================
# 📊 LOAD DATA (auto-detects column name)
# ============================================================
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

pos_col = None
for candidate in ["Predicted_Position", "Position", "Pos"]:
    if candidate in df.columns:
        pos_col = candidate
        break

if pos_col is None or "Abbreviation" not in df.columns:
    raise ValueError(
        f"CSV must contain 'Abbreviation' and one of ['Predicted_Position', 'Position', 'Pos'] columns.\n"
        f"Available columns: {list(df.columns)}"
    )

df["Abbreviation"] = df["Abbreviation"].str.strip().str.upper()
df["PosNum"] = df[pos_col].astype(int)
df = df.sort_values("PosNum")
df["TeamName"] = df["Abbreviation"].map(DRIVER_TO_TEAM)
df["Color"] = df["TeamName"].map(TEAM_COLOURS).fillna("#888888")

# ============================================================
# 🖼️ VISUAL CREATION
# ============================================================
plt.style.use("default")
fig, ax = plt.subplots(figsize=(15, 7))
fig.patch.set_facecolor("#D9D9D9")
ax.set_facecolor("#D9D9D9")

x_positions = np.arange(len(df))
bar_heights = (21 - df["PosNum"])
bars = ax.bar(x_positions, bar_heights, color=df["Color"], edgecolor="black", linewidth=1.1)

ax.set_ylim(0, max(bar_heights) + 6)
ax.set_xlim(-0.5, len(df) - 0.5)
ax.set_xticks(x_positions)
ax.set_xticklabels(df["Abbreviation"], color="black", fontsize=10, fontname="Times New Roman", fontweight="bold")
ax.tick_params(axis="y", left=False, labelleft=False)

ax.set_title(
    f"{GP_NAME} Grand Prix — Post-Quali Predictions ({YEAR})",
    fontsize=18, color="black", fontname="Times New Roman", fontweight="bold", pad=20
)

# ============================================================
# 🏎️ ADD DRIVER IMAGES
# ============================================================
for x, row in zip(x_positions, df.itertuples()):
    abbrev = row.Abbreviation
    img_path = IMG_DIR / f"{abbrev}.png"

    if not img_path.exists():
        print(f"⚠️ Missing image for {abbrev}")
        continue

    try:
        img = Image.open(img_path).convert("RGBA")
        w, h = img.size
        scale = 1
        new_h = int(100 * scale)
        new_w = int(w * (new_h / h))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        img_arr = np.array(img)

        y_top = (21 - row.PosNum) + 1
        ax.imshow(
            img_arr,
            extent=(x - 1, x + 0.7, y_top, y_top + 2.5),
            aspect="auto",
            zorder=5,
        )

        ax.text(x, y_top + 2.2, f"P{row.PosNum}",
                ha="center", va="bottom",
                color="black", fontsize=10, weight="bold", fontname="Times New Roman")

    except Exception as e:
        print(f"⚠️ Error with {abbrev}: {e}")

# ============================================================
# 💾 SAVE
# ============================================================
plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", facecolor="#D9D9D9")
plt.close(fig)
print(f"✅ Visual saved → {OUT_FILE}")




# ============================================================
# 📂 PATHS
# ============================================================
DATA_FILE = (
    ROOT / "Data_Processed" / "prediction_data" /
    f"{YEAR}_Predictions" / f"R{ROUND}_Prediction" /
    f"R{ROUND}_Final_Prediction.csv"
)
IMG_DIR = ROOT / "image_asset" / "drivers" / "celebrations"
PODIUM_BG = ROOT / "image_asset" / "podium" / "podium.png"
OUT_DIR = ROOT / "visualisations" / f"R{ROUND}_{GP_NAME.replace(' ', '_')}_Grand_Prix"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "podium_visual.png"

# ============================================================
# 🏁 LOAD DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

pos_col = None
for c in ["Predicted_Position", "Position", "Pos"]:
    if c in df.columns:
        pos_col = c
        break

if pos_col is None:
    raise ValueError("CSV must contain a position column (Predicted_Position, Position, or Pos).")

df["Abbreviation"] = df["Abbreviation"].str.strip().str.upper()
df["PosNum"] = df[pos_col].astype(int)
df = df.sort_values("PosNum")
top3 = df.head(3).reset_index(drop=True)

# ============================================================
# 🧩 LOAD PODIUM BACKGROUND
# ============================================================
bg = Image.open(PODIUM_BG).convert("RGBA")
canvas = bg.copy()
bg_w, bg_h = canvas.size  # 1024 x 1024

# ============================================================
# 🏆 DRIVER POSITIONS (adjusted lower + double size)
# ============================================================
PODIUM_POSITIONS = {
    1: (bg_w * 0.50, bg_h * 0.55, 0.95),  # Center
    2: (bg_w * 0.28, bg_h * 0.60, 0.90),  # Left
    3: (bg_w * 0.72, bg_h * 0.62, 0.90),  # Right
}

# ============================================================
# 🏎️ PLACE DRIVERS
# ============================================================
for _, row in top3.iterrows():
    pos = int(row["PosNum"])
    abbrev = row["Abbreviation"]
    img_path = IMG_DIR / f"{abbrev}.png"

    if not img_path.exists():
        print(f"⚠️ Missing image for {abbrev}")
        continue

    driver_img = Image.open(img_path).convert("RGBA")
    w, h = driver_img.size

    # Double previous driver size
    base_h = 350
    scale = PODIUM_POSITIONS.get(pos, (0, 0, 0.9))[2]
    new_h = int(base_h * scale)
    new_w = int(w * (new_h / h))
    driver_img = driver_img.resize((new_w, new_h), Image.LANCZOS)

    # Lower driver slightly
    x, y, _ = PODIUM_POSITIONS[pos]
    x_offset = int(x - new_w / 2)
    y_offset = int(y - new_h + 100)

    canvas.alpha_composite(driver_img, dest=(x_offset, y_offset))

# ============================================================
# 💾 SAVE OUTPUT
# ============================================================
canvas.save(OUT_FILE)
print(f"✅ Podium visual created → {OUT_FILE}")









