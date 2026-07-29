# Workspace Rules - ILMT Streak Detector V3

## Directory Organization & Cleanliness Rules
1. **Root Folder (`C:\Users\agile\Downloads\Project\ILMT Streak Detector V3\`)**:
   Must contain ONLY:
   - `TASKLIST.md`
   - `README.md` (if existing)
   - Configuration files (e.g. `config.yaml` if placed at root)
   - Standard directories: `data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `scripts/`, `docs/`, `.agents/`
   - **NO** loose `.py` scripts, `.log` files, or temporary test scripts sitting at root.

2. **File Placement Directory Structure**:
   - `scripts/`: Diagnostic, helper, utility, or throwaway `.py` scripts (NOT production `src/` code).
   - `scripts/logs/`: Any `.log` output files or execution run logs.
   - `src/`: Production pipeline code modules (`background.py`, `io_utils.py`, `detect.py`, `characterize.py`, `calibration.py`, etc.).
   - `docs/`: Written reports, architecture documents, summaries, and walkthroughs.
   - `models/`: Saved model weights/artifacts (`.joblib`, `.pth`, etc.).
   - `outputs/`: Pipeline output catalogs, CSVs, and generated cutouts.
