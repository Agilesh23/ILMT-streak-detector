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

target_rows = [
    ("20221025-0028.fits", 6527, "streak", "whole_streak", "g_20221025", (1038, 0, 2957, 4096)),
    ("20230519-0009.fits", 5768, "streak", "part_of_3673", "g_20230519", (1271, 784, 3443, 3779)),
    ("20221026-0057.fits", 1183, "streak", "same_streak_as_1175", "g_20221026", (347, 1110, 427, 3328)),
    ("20221026-0057.fits", 1714, "fp", "bleed_trail", "", (607, 540, 2938, 599)),
    ("20230602-0022.fits", 11716, "streak", "whole_streak", "g_20230602", (2224, 2042, 4032, 4096)),
    ("20231115-0003.fits", 888, "streak", "collinear_frag", "g_20231115", (153, 0, 2329, 2463)),
    ("20231115-0003.fits", 12982, "streak", "collinear_frag", "g_20231115", (2334, 2473, 3769, 4096)),
    ("20240114-0028.fits", 5446, "streak", "whole_streak", "g_20240114", (966, 0, 2690, 4096)),
]

frames = list(dict.fromkeys([r[0] for r in target_rows]))

for fname in frames:
    d, _ = load_fits(frame_dir / fname)
    bg, std = estimate_background(d, method="background2d")
    cands, _, _ = find_blob_candidates(
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
    
    # Check where each candidate is found
    expected_for_frame = [r for r in target_rows if r[0] == fname]
    for row in expected_for_frame:
        cid = row[1]
        exp_bbox = row[5] # (r0, c0, r1, c1)
        
        in_sel = [c for c in selected if c["label"] == cid]
        in_stitched = [c for c in stitched if c["label"] == cid]
        in_cands = [c for c in cands if c["label"] == cid]
        
        c = in_sel[0] if in_sel else (in_stitched[0] if in_stitched else (in_cands[0] if in_cands else None))
        if c is not None:
            r0, c0, r1, c1 = c.get("bbox_full", c.get("bbox_local"))
            diffs = [abs(r0 - exp_bbox[0]), abs(c0 - exp_bbox[1]), abs(r1 - exp_bbox[2]), abs(c1 - exp_bbox[3])]
            max_diff = max(diffs)
            cy, cx = c["centroid"]
            dy = r1 - r0
            dx = c1 - c0
            ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
            length = float(c.get("axis_major_length", np.hypot(dy, dx)))
            score = c.get("composite_score", c.get("score", 0.0))
            print(f"[{fname}] cid={cid}: in_sel={len(in_sel)}, in_stitch={len(in_stitched)}, in_cands={len(in_cands)}")
            print(f"   Found bbox: [{r0}:{r1}, {c0}:{c1}], Expected: [{exp_bbox[0]}:{exp_bbox[2]}, {exp_bbox[1]}:{exp_bbox[3]}], max_diff={max_diff}")
            print(f"   score={score:.2f}, area={c['area']}, aspect={c['aspect_ratio']:.2f}, max_int={c['max_intensity']:.2f}")
            print(f"   centroid=({cy:.2f}, {cx:.2f}), length={length:.2f}, angle={ang:.2f}")
        else:
            print(f"[{fname}] cid={cid}: NOT FOUND!")
