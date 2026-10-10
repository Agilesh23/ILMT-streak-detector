import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from collections import Counter

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

# 99 labeled rows
df1 = pd.read_csv(data_dir / "batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)
print(f"Total labeled rows: {len(df_all)}")

unique_frames = df_all["frame_name"].unique()

# Precompute pipeline outputs per frame
frame_pipeline = {}
for fn in unique_frames:
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
    stitched = stitch_collinear_fragments(cands, d, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score)
    
    frame_pipeline[fn] = {
        "raw_cands": {c["label"]: c for c in cands},
        "stitched": {c["label"]: c for c in stitched},
        "selected": {c["label"]: c for c in selected},
        "all_stitched_list": stitched
    }

# Check each labeled row
row_levels = []
patterns = []

for idx, row in df_all.iterrows():
    fn = row["frame_name"]
    cid = int(row["candidate_id"])
    lbl = row["label"]
    grp = row.get("group_id", "")
    
    fp = frame_pipeline[fn]
    in_a = cid in fp["raw_cands"]
    in_b = cid in fp["stitched"]
    in_c = cid in fp["selected"]
    
    stitched_from = fp["stitched"][cid].get("stitched_from", []) if in_b else []
    
    # Also check if cid is a member of any stitched candidate
    member_of = []
    for sc in fp["all_stitched_list"]:
        if cid in sc.get("stitched_from", []):
            member_of.append(sc["label"])
            
    row_info = {
        "frame_name": fn,
        "candidate_id": cid,
        "label": lbl,
        "group_id": grp,
        "in_a": in_a,
        "in_b": in_b,
        "in_c": in_c,
        "stitched_from": stitched_from,
        "member_of": member_of
    }
    row_levels.append(row_info)
    patterns.append((in_a, in_b, in_c))

counts = Counter(patterns)

print("\n=== PATTERN COUNTS (a=raw, b=post-stitch, c=selected) ===")
for p, cnt in sorted(counts.items()):
    print(f"Pattern (a={p[0]}, b={p[1]}, c={p[2]}): {cnt} rows")

print("\n=== ROWS EXISTING AT ONE LEVEL BUT NOT ANOTHER ===")
diff_rows = [r for r in row_levels if not (r["in_a"] == r["in_b"] == r["in_c"])]
for r in diff_rows:
    print(f"({r['frame_name']}, {r['candidate_id']}) label={r['label']} | a={r['in_a']}, b={r['in_b']}, c={r['in_c']} | stitched_from={r['stitched_from']} | member_of={r['member_of']}")

print("\n=== POST-STITCH CANDIDATES WITH >1 STITCHED_FROM MEMBER ===")
# Look across all frames' stitched candidates
for fn in unique_frames:
    fp = frame_pipeline[fn]
    df_f = df_all[df_all["frame_name"] == fn]
    for sc in fp["all_stitched_list"]:
        members = sc.get("stitched_from", [])
        if len(members) > 1:
            member_details = []
            for mid in members:
                m_row = df_f[df_f["candidate_id"] == mid]
                if len(m_row) > 0:
                    mlbl = m_row.iloc[0]["label"]
                    mgrp = m_row.iloc[0]["group_id"]
                else:
                    mlbl = "NOT_IN_CSV"
                    mgrp = ""
                member_details.append(f"id={mid} (label={mlbl}, group={mgrp})")
            in_sel = sc["label"] in fp["selected"]
            print(f"[{fn}] Stitched candidate {sc['label']} (selected={in_sel}) stitched_from {len(members)} members:")
            for md in member_details:
                print(f"   - {md}")
