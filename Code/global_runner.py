import runpy
import sys
import types
import traceback
from pathlib import Path

# ============================================================
# CONFIG LOADING
# ============================================================

CONFIG_PATH = Path("/Users/ravish/Documents/F1-Project-Folder/Code/config.py")

try:
    print("🔧 Loading configuration...")
    config_data = runpy.run_path(str(CONFIG_PATH))
    YEAR = config_data.get("YEAR")
    ROUND = config_data.get("ROUND")
    print(f"✅ Config loaded: YEAR={YEAR}, ROUND={ROUND}")

    # Register Code.config globally so submodules can import it
    sys.modules["Code"] = types.ModuleType("Code")
    sys.modules["Code.config"] = types.ModuleType("Code.config")

    for key, val in config_data.items():
        setattr(sys.modules["Code.config"], key, val)

    print("🌍 Config registered globally as Code.config")

except Exception as e:
    print(f"❌ Failed to load config: {e}")
    sys.exit(1)


# ============================================================
# PIPELINE SETUP
# ============================================================

ROOT = Path("/Users/ravish/Documents/F1-Project-Folder/Code")

PIPELINE = [
    "quali_feat.py",
    "fastf1.py",
    "rolling_form.py",
    "predictor_with_quali_results.py",
    "remaining_race_simulator.py",
]

print("\n===============================")
print("🏎️  F1 GLOBAL RUNNER — Full Pipeline (Quali → Final Simulator)")
print("===============================\n")


# ============================================================
# SCRIPT EXECUTION LOOP
# ============================================================

for script in PIPELINE:
    script_path = ROOT / script

    if not script_path.exists():
        print(f"⚠️  Missing script: {script}")
        continue

    print(f"\n🚀 Running {script} ...")

    try:
        # Re-register config fresh for each stage
        sys.modules["Code"] = types.ModuleType("Code")
        sys.modules["Code.config"] = types.ModuleType("Code.config")
        for key, val in config_data.items():
            setattr(sys.modules["Code.config"], key, val)

        # === Special case: inject fallback logic for remaining_race_simulator dynamically ===
        if script == "remaining_race_simulator.py":
            sim_path = script_path
            sim_code = sim_path.read_text()

            # Add fallback logic for IN_FILE resolution
            if "all_scores_after_quali_r" in sim_code:
                sim_code = sim_code.replace(
                    'IN_FILE  = DATA / "quali_feat" / "Champ_Rating" / f"all_scores_after_quali_r{ROUND}_{YEAR}.csv"',
                    (
                        "IN_FILE  = DATA / 'quali_feat' / 'Champ_Rating' / "
                        "f'all_scores_after_quali_r{ROUND}_{YEAR}.csv'\n"
                        "ALT_FILE = DATA / 'quali_feat' / 'Champ_Rating' / f'champ_standing_live_{YEAR}.csv'\n"
                        "from pathlib import Path\n"
                        "if not IN_FILE.exists():\n"
                        "    print(f\"[WARN] {IN_FILE.name} not found. Falling back to {ALT_FILE.name}\")\n"
                        "    IN_FILE = ALT_FILE"
                    ),
                )
                # Run patched version
                exec(sim_code, {"__name__": "__main__"})
                print(f"✅ Finished {script}\n")
                continue  # move to next script

        # === Normal execution for all other scripts ===
        runpy.run_path(str(script_path), run_name="__main__")
        print(f"✅ Finished {script}\n")

    except ImportError as e:
        if "run_path" in str(e) or "not in sys.modules" in str(e):
            print(f"[WARN] Could not reload Code.config dynamically: {e}")
        else:
            print(f"❌ Import error in {script}: {e}")
        traceback.print_exc()
        print("🛑 Stopping pipeline to avoid inconsistent data.\n")
        break

    except Exception as e:
        print(f"❌ Error in {script}: {e}")
        traceback.print_exc()
        print("🛑 Stopping pipeline to avoid inconsistent data.\n")
        break


# ============================================================
# WRAP-UP
# ============================================================

print("===============================")
print("🏁 GLOBAL RUNNER FINISHED — All 5 Stages Completed Successfully!")
print("===============================")