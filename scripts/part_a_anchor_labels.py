import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
with open(base_dir / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Save snapshot of config
snapshot_path = base_dir / "data/labeled/detection_config_snapshot.yaml"
with open(snapshot_path, "w") as f:
    yaml.safe_dump(cfg, f, default_flow_style=False)
print(f"Saved snapshot to {snapshot_path}")

det_cfg = cfg.get("detection", {})
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)

b1_path = base_dir / "data/labeled/batch_1_v2_labels.csv"
b2_path = base_dir / "data/labeled/batch_2_v2_labels.csv"

df1 = pd.read_csv(b1_path)
df2 = pd.read_csv(b2_path)

all_frames = list(dict.fromkeys(df1["frame_name"].tolist() + df2["frame_name"].tolist()))

# Cache extracted candidate properties: (frame_name, candidate_id) -> props
cand_props = {}

for fname in all_frames:
    fpath = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files") / fname
    data, header = load_fits(fpath)
    bg, std = estimate_background(data, method="background2d")
    cands, _, _ = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px
    )
    stitched = stitch_collinear_fragments(cands, data, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score)

    for c in selected:
        cid = c["label"]
        cy, cx = c["centroid"]
        r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
        dy = r1 - r0
        dx = c1 - c0
        ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
        length = float(c.get("axis_major_length", np.hypot(dy, dx)))
        cand_props[(fname, cid)] = {
            "centroid_row": round(float(cy), 2),
            "centroid_col": round(float(cx), 2),
            "bbox_r0": int(r0),
            "bbox_c0": int(c0),
            "bbox_r1": int(r1),
            "bbox_c1": int(c1),
            "length_px": round(length, 2),
            "angle_deg": round(ang, 2)
        }

def update_df(df, csv_path):
    cols_to_add = ["centroid_row", "centroid_col", "bbox_r0", "bbox_c0", "bbox_r1", "bbox_c1", "length_px", "angle_deg"]
    for col in cols_to_add:
        if col not in df.columns:
            df[col] = np.nan

    mismatches = 0
    for idx, row in df.iterrows():
        key = (row["frame_name"], int(row["candidate_id"]))
        if key in cand_props:
            p = cand_props[key]
            for col in cols_to_add:
                df.at[idx, col] = p[col]
        else:
            print(f"MISSING: {key}")
            mismatches += 1

    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path.name}: {len(df)} rows, {mismatches} mismatches")
    return mismatches

m1 = update_df(df1, b1_path)
m2 = update_df(df2, b2_path)
print(f"Total mismatches: {m1 + m2} (expected 0)")
