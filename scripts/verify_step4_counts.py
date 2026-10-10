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
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

with open(base_dir / "src/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

det_cfg = cfg.get("detection", {})
edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
border_margin_px = det_cfg.get("border_margin_px", 15)
border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
bypass_cfg = det_cfg.get("long_thin_bypass", {})

df1 = pd.read_csv(base_dir / "data/labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(base_dir / "data/labeled/batch_2_v2_labels.csv")
df_all = pd.concat([df1, df2]).reset_index(drop=True)
frames = list(dict.fromkeys(df_all['frame_name'].tolist()))

results = []
fresh_props = {}

for fname in frames:
    data, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(data, method="background2d")
    
    # 1. Before bypass
    c_bef, _, _ = find_blob_candidates(
        data, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass_enabled=False
    )
    s_bef = stitch_collinear_fragments(c_bef, data, bg, std)
    sel_bef, _ = rank_and_select_candidates(s_bef, min_abs_score=min_abs_score)
    
    # 2. After bypass
    c_aft, _, _ = find_blob_candidates(
        data, bg, std,
        edge_mask_max_x=edge_mask_max_x,
        border_margin_px=border_margin_px,
        border_max_length_px=border_max_length_px,
        left_edge_max_col=left_edge_max_col,
        left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
        long_thin_bypass=bypass_cfg
    )
    s_aft = stitch_collinear_fragments(c_aft, data, bg, std)
    sel_aft, _ = rank_and_select_candidates(s_aft, min_abs_score=min_abs_score)
    
    results.append({
        "frame": fname,
        "before": len(sel_bef),
        "after": len(sel_aft)
    })
    
    for c in sel_aft:
        cid = c["label"]
        cy, cx = c["centroid"]
        r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
        fresh_props[(fname, cid)] = {
            "centroid_row": round(float(cy), 2),
            "centroid_col": round(float(cx), 2),
            "bbox_r0": int(r0),
            "bbox_c0": int(c0),
            "bbox_r1": int(r1),
            "bbox_c1": int(c1)
        }
    print(f"Processed {fname}: before={len(sel_bef)}, after={len(sel_aft)}", flush=True)

# Verify all 91 labeled rows
mismatches = 0
for idx, row in df_all.iterrows():
    key = (row["frame_name"], int(row["candidate_id"]))
    if key not in fresh_props:
        print(f"MISSING CANDIDATE: {key}")
        mismatches += 1
    else:
        p = fresh_props[key]
        if abs(p["centroid_row"] - row["centroid_row"]) > 0.05 or abs(p["centroid_col"] - row["centroid_col"]) > 0.05:
            print(f"CENTROID MISMATCH {key}: stored=({row['centroid_row']}, {row['centroid_col']}) fresh=({p['centroid_row']}, {p['centroid_col']})")
            mismatches += 1
        if p["bbox_r0"] != row["bbox_r0"] or p["bbox_c0"] != row["bbox_c0"] or p["bbox_r1"] != row["bbox_r1"] or p["bbox_c1"] != row["bbox_c1"]:
            print(f"BBOX MISMATCH {key}: stored=({row['bbox_r0']}, {row['bbox_c0']}, {row['bbox_r1']}, {row['bbox_c1']}) fresh=({p['bbox_r0']}, {p['bbox_c0']}, {p['bbox_r1']}, {p['bbox_c1']})")
            mismatches += 1

print("\n================ SELECTED COUNTS PER FRAME ================")
print(f"{'Frame':<22} | {'Before':<8} | {'After':<8} | {'Diff'}")
print("-" * 50)
for r in results:
    diff = r['after'] - r['before']
    diff_str = f"+{diff}" if diff > 0 else f"{diff}"
    print(f"{r['frame']:<22} | {r['before']:<8} | {r['after']:<8} | {diff_str}")

print(f"\nTotal labeled row mismatches: {mismatches} (expected 0)", flush=True)
