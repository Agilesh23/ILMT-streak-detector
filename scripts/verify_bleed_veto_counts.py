import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

base_dir = Path(r"c:\Users\agile\Downloads\Project\ILMT Streak Detector v3")
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
data_dir = base_dir / "data/labeled"

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
bleed_veto_cfg = det_cfg.get("bleed_veto", None)

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
df1 = pd.read_csv(data_dir / "batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)

streak_rows = df_all[df_all["label"] == "streak"]
print(f"Total labeled streak rows: {len(streak_rows)} (expected 15)")

fresh_cands_by_frame = {}
vetoed_by_frame = {}
selected_counts = {}

print("\n--- Running 9 frames ---")
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
        long_thin_bypass=bypass_cfg,
        bleed_veto=bleed_veto_cfg
    )
    fresh_cands_by_frame[fn] = {c["label"]: c for c in cands}
    
    stitched = stitch_collinear_fragments(cands, d, bg, std)
    
    # Run rank_and_select with drop_bleed_trails=False and True to compare
    sel_no_veto, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score, drop_bleed_trails=False)
    
    # Now with veto (drop_bleed_trails=True)
    print(f"\n[{fn}] Running selection with bleed veto:")
    sel_with_veto, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score, drop_bleed_trails=True)
    
    vetoed_ids = [c["label"] for c in stitched if c.get("is_bleed_trail", False)]
    vetoed_by_frame[fn] = vetoed_ids
    selected_counts[fn] = len(sel_with_veto)
    
    print(f"[{fn}] count_no_veto={len(sel_no_veto)} | count_with_veto={len(sel_with_veto)} | vetoed_ids={vetoed_ids}")

print("\n================ COUNTS SUMMARY ================")
print(f"{'Frame':<20} | {'Prev':<6} | {'New':<6} | {'Vetoed IDs'}")
print("-" * 55)
for fn in frames:
    print(f"{fn:<20} | {prev_counts[fn]:<6} | {selected_counts[fn]:<6} | {vetoed_by_frame[fn]}")

# Confirm none of the 15 labeled streak rows is vetoed
all_vetoed_set = set()
for fn, vlist in vetoed_by_frame.items():
    for vid in vlist:
        all_vetoed_set.add((fn, vid))

vetoed_streaks = []
for idx, r in streak_rows.iterrows():
    key = (r["frame_name"], int(r["candidate_id"]))
    if key in all_vetoed_set:
        vetoed_streaks.append(key)

print(f"\nVetoed labeled streaks: {len(vetoed_streaks)} (expected 0): {vetoed_streaks}")

# Confirm all labeled rows still reproduce with 0 mismatches
mismatches = 0
for idx, row in df_all.iterrows():
    fn = row["frame_name"]
    cid = int(row["candidate_id"])
    key = (fn, cid)
    c = fresh_cands_by_frame[fn].get(cid)
    if c is None:
        print(f"MISSING CANDIDATE: {key}")
        mismatches += 1
        continue
    
    cy, cx = c["centroid"]
    r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
    
    # Check centroid within 0.05 px
    if abs(cy - row["centroid_row"]) > 0.05 or abs(cx - row["centroid_col"]) > 0.05:
        print(f"CENTROID MISMATCH {key}: stored=({row['centroid_row']}, {row['centroid_col']}) fresh=({cy:.2f}, {cx:.2f})")
        mismatches += 1
    # Check bbox
    if int(r0) != int(row["bbox_r0"]) or int(c0) != int(row["bbox_c0"]) or int(r1) != int(row["bbox_r1"]) or int(c1) != int(row["bbox_c1"]):
        print(f"BBOX MISMATCH {key}: stored=[{row['bbox_r0']}:{row['bbox_r1']}, {row['bbox_c0']}:{row['bbox_c1']}] fresh=[{r0}:{r1}, {c0}:{c1}]")
        mismatches += 1

print(f"\nTotal labeled row mismatches: {mismatches} (expected 0)")
