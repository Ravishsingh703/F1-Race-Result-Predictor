
from pathlib import Path

# ===== Root Paths =====
ROOT = Path("/Users/ravish/Documents/F1-Project-Folder")
DATA = ROOT / "Data_Processed"
CODE = ROOT / "Code"

# ===== Subfolders =====
FASTF1 = DATA / "fastf1"
DRIVER_STATS = DATA / "driver_stats_over_years"
TRACK_SCORES = DATA / "track_scores"
QUALI_FEAT = DATA / "quali_feat" / "2025"
PREDICTION = DATA / "prediction_data"
WDC = DATA / "wdc_prediction"

# ===== General Settings =====
YEAR = 2025
ROUND = 24                # Update this before each GP
GP_NAME = "Abu Dhabi"
ROOT = "/Users/ravish/Documents/F1-Project-Folder"

# ===== File Helpers =====
def round_file(folder: Path, prefix: str, suffix: str = "features.csv") -> Path:
    """Builds round-specific file paths safely."""
    return folder / f"{prefix}_{YEAR}_R{ROUND:02d}_{suffix}"

def quali_file() -> Path:
    """Get qualifying feature file for the round."""
    return QUALI_FEAT / f"{ROUND:02d}_quali_feat_{YEAR}.csv"

def ensure_folder(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    return folder
