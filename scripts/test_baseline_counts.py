import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

with open(base_dir / "src/config.yaml") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg["detection"]
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
bypass_cfg = det_cfg.get("long_thin_bypass", None)

frames = [
    "20221025-0028.fits",
    "20230519-0009.fits",
    "20230317-0008.fits",
    "20221026-0057.fits",
    "20221101-0030.fits",
    "20230602-0022.fits",
    "20231115-0003.fits",
    "20240114-0028.fits",
    "20240405-0004.fits"
]

prev_counts = {
    "20221025-0028.fits": 19,
    "20230519-0009.fits": 5,
    "20230317-0008.fits": 2,
    "20221026-0057.fits": 18,
    "20221101-0030.fits": 12,
    "20230602-0022.fits": 6,
    "20231115-0003.fits": 15,
    "20240114-0028.fits": 8,
    "20240405-0004.fits": 10
}

# Labeled CSVs
df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(base_dir / "data/labeled/batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)
streak_keys = set(zip(df_all[df_all["label"] == "streak"]["frame_name"], df_all[df_all["label"] == "streak"]["candidate_id"]))
print(f"Total labeled streak rows: {len(streak_keys)}")

for fn in frames:
    d, _ = load_fits(frame_dir / fn)
    bg, std = estimate_background(d, method="background2d")
    cands, count_before, count_after = find_blob_candidates(
        d, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass=bypass_cfg
    )
    stitched = stitch_collinear_fragments(cands, d, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score)
    print(f"{fn}: selected {len(selected)} (expected prev: {prev_counts[fn]})")
    # Check if 1714 or 18665 are in selected
    cids = [c["label"] for c in selected]
    if 1714 in cids:
        print(f"   -> 1714 IS IN SELECTED for {fn}!")
    if 18665 in cids:
        print(f"   -> 18665 IS IN SELECTED for {fn}!")
