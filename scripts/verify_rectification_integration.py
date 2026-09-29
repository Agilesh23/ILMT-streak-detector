"""
Verification Script for Integrated Pipeline Long-Streak Rectification (V3 Cutout).

Imports and executes extract_candidate_cutout from src/characterize.py to verify
the integrated weighted TLS line refitting, endpoint gradient trim margin, and
exact 0-margin in-bounds boundary ray length clipping against config.yaml parameters.
"""

from pathlib import Path
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup base_dir relative to this script file
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import extract_candidate_cutout


def main():
    # 1. Load parameters from config.yaml
    config_path = src_dir / "config.yaml"
    if not config_path.exists():
        config_path = base_dir / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    det_cfg = config.get("detection", {})
    n_sigma = det_cfg.get("n_sigma", 3.0)
    min_aspect_ratio = det_cfg.get("min_aspect_ratio", 3.0)
    min_eccentricity = det_cfg.get("min_eccentricity", 0.9)
    max_area = det_cfg.get("max_area", 15000.0)

    bg_cfg = config.get("background", {})
    bg_method = bg_cfg.get("default_method", "background2d")

    rank_cfg = config.get("ranking", {})
    top_n = rank_cfg.get("top_n", 3)

    frame_path = r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230519-0009.fits"
    print(f"Loading frame for integrated verification: {frame_path}")
    print(f"Using config parameters: n_sigma={n_sigma}, min_aspect_ratio={min_aspect_ratio}, min_eccentricity={min_eccentricity}, max_area={max_area}, bg_method='{bg_method}'")

    # 2. Pipeline processing
    data, header = load_fits(frame_path)
    bg, std = estimate_background(data, method=bg_method)

    unstitched_candidates, count_before, count_shape = find_blob_candidates(
        data, bg, std,
        n_sigma=n_sigma,
        min_aspect_ratio=min_aspect_ratio,
        min_eccentricity=min_eccentricity,
        max_area=max_area
    )

    unstitched_lookup = {c["label"]: c for c in unstitched_candidates}

    stitched_candidates = stitch_collinear_fragments(unstitched_candidates, data, bg, std)
    selected_candidates, cutoff_score = rank_and_select_candidates(stitched_candidates)

    # Target long stitched candidate 3673
    target_cand = None
    for c in selected_candidates:
        if c.get("label") == 3673 or c.get("n_fragments", 1) > 1:
            target_cand = c
            break

    if target_cand is None:
        target_cand = max(selected_candidates, key=lambda c: c.get("axis_major_length", 0.0))

    cand_id = target_cand.get("label")
    merged_labels = target_cand.get("stitched_from", [cand_id])
    detected_length = float(target_cand.get("axis_major_length", 0.0))

    frame_diagonal = float(np.hypot(data.shape[0], data.shape[1]))
    unclipped_target_length = 2.0 * detected_length + 50.0  # 2.0x + 50px (after 15px per end trim)

    # 3. Extract cutout using NOW-INTEGRATED pipeline code path (extract_candidate_cutout with 0 margin)
    patch_v3, start_pt, end_pt, refit_angle_deg, final_clipped_length = extract_candidate_cutout(
        data=data,
        candidate=target_cand,
        box_width=100,
        unstitched_lookup=unstitched_lookup,
        trim_margin_px=15.0,
        max_boundary_margin_px=0.0
    )

    clipping_applied_px = unclipped_target_length - final_clipped_length

    # 4. Save as 20230519-0009_cand_3673_cutout_v3.png
    output_dir = base_dir / "outputs" / "labeling_batch_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"20230519-0009_cand_{cand_id}_cutout_v3.png"

    p_vmin, p_vmax = float(np.percentile(patch_v3, 1)), float(np.percentile(patch_v3, 99))
    if p_vmax <= p_vmin:
        p_vmin, p_vmax = float(np.min(patch_v3)), float(np.max(patch_v3))
    if p_vmax <= p_vmin:
        p_vmin, p_vmax = 0.0, 1.0

    plt.imsave(str(out_path), patch_v3, cmap="gray", origin="lower", vmin=p_vmin, vmax=p_vmax)

    print("\n=== INTEGRATED PIPELINE RECTIFICATION VERIFICATION (V3 - ZERO MARGIN) ===")
    print(f"Target Candidate ID:                   {cand_id}")
    print(f"Constituent Fragment Count:            {len(merged_labels)} (Labels: {merged_labels})")
    print(f"Frame Dimensions & Diagonal:           {data.shape[1]}x{data.shape[0]} px | Diagonal: {frame_diagonal:.2f} px")
    print(f"Refit Weighted TLS Line Angle:         {refit_angle_deg:.4f} degrees")
    print(f"Unclipped Target Cutout Length:        {unclipped_target_length:.2f} px")
    print(f"Final Boundary-Clipped Cutout Length:  {final_clipped_length:.2f} px")
    print(f"Boundary Clipping Applied:             {clipping_applied_px:.2f} px ({clipping_applied_px / unclipped_target_length * 100.0:.1f}% reduction)")
    print(f"Extracted Segment Start Point:         ({start_pt[0]:.2f}, {start_pt[1]:.2f})")
    print(f"Extracted Segment End Point:           ({end_pt[0]:.2f}, {end_pt[1]:.2f})")
    print(f"Cutout Patch Array Shape:              {patch_v3.shape}")
    print(f"File Save Path:                        {out_path}")


if __name__ == "__main__":
    main()
