import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup base_dir relative to this script file
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"

sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import extract_streak_patch

output_dir = base_dir / "outputs" / "labeling_batch_1"
output_dir.mkdir(parents=True, exist_ok=True)

csv_dir = base_dir / "data" / "labeled"
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / "batch_1_labels.csv"

frames = [
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221025-0028.fits",
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230519-0009.fits",
    r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits",
]

rows = []
cutout_count = 0
context_count = 0

for frame_path_str in frames:
    frame_path = Path(frame_path_str)
    frame_name = frame_path.name
    frame_stem = frame_path.stem
    print(f"Processing frame: {frame_name}...")
    
    # 1. Load & Process Pipeline
    data, header = load_fits(frame_path)
    bg, std = estimate_background(data, method='background2d')
    candidates, count_before, count_shape = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0
    )
    stitched = stitch_collinear_fragments(candidates, data, bg, std)
    selected, cutoff_score = rank_and_select_candidates(stitched, top_n=3)
    
    # Take top 3 per frame
    top3 = selected[:3]
    h_img, w_img = data.shape
    f_vmin, f_vmax = float(np.percentile(data, 1)), float(np.percentile(data, 99))
    
    for cand in top3:
        cand_id = cand["label"]
        cy, cx = cand["centroid"]
        min_row, min_col, max_row, max_col = cand.get("bbox_full", cand.get("bbox_local"))
        
        # Calculate angle directly from candidate bounding box / orientation
        dy = max_row - min_row
        dx = max_col - min_col
        angle_rad = float(np.arctan2(dy, dx))
        
        detected_length = cand.get("axis_major_length", float(np.hypot(dy, dx)))
        target_length = 2.0 * detected_length + 80.0
        new_half_len = target_length / 2.0
        
        start_pt = (cx - new_half_len * np.cos(angle_rad), cy - new_half_len * np.sin(angle_rad))
        end_pt = (cx + new_half_len * np.cos(angle_rad), cy + new_half_len * np.sin(angle_rad))
        
        try:
            patch, _ = extract_streak_patch(data, start_pt, end_pt, box_width=100, output_length=int(target_length))
        except Exception:
            r_s = max(0, int(min_row - 5))
            r_e = min(h_img, int(max_row + 5))
            c_s = max(0, int(min_col - 5))
            c_e = min(w_img, int(max_col + 5))
            patch = data[r_s:r_e, c_s:c_e]
            
        # a) Save Cutout Image (raw image only, NO text/titles)
        cutout_filename = f"{frame_stem}_cand_{cand_id}_cutout.png"
        cutout_filepath = output_dir / cutout_filename
        
        p_vmin, p_vmax = float(np.percentile(patch, 1)), float(np.percentile(patch, 99))
        if p_vmax <= p_vmin:
            p_vmin, p_vmax = float(np.min(patch)), float(np.max(patch))
        if p_vmax <= p_vmin:
            p_vmin, p_vmax = 0.0, 1.0
            
        plt.imsave(str(cutout_filepath), patch, cmap='gray', origin='lower', vmin=p_vmin, vmax=p_vmax)
        cutout_count += 1
        
        # b) Save Context Image (whole-frame thumbnail with red bounding box)
        context_filename = f"{frame_stem}_cand_{cand_id}_context.png"
        context_filepath = output_dir / context_filename
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data, cmap='gray', origin='lower', vmin=f_vmin, vmax=f_vmax)
        rect_w = max(1, max_col - min_col)
        rect_h = max(1, max_row - min_row)
        rect = patches.Rectangle(
            (min_col, min_row), rect_w, rect_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        ax.plot(cx, cy, 'cyan', marker='x', markersize=6)
        ax.set_title(f"Whole-Frame Context: {frame_name} | Candidate {cand_id}\nBBox: (x={min_col}..{max_col}, y={min_row}..{max_row})", fontsize=10)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        plt.tight_layout()
        fig.savefig(str(context_filepath), dpi=120, bbox_inches='tight')
        plt.close(fig)
        context_count += 1
        
        # Add to CSV row list
        rows.append({
            "candidate_id": cand_id,
            "frame_name": frame_name,
            "cutout_path": str(cutout_filepath.relative_to(base_dir)),
            "context_path": str(context_filepath.relative_to(base_dir)),
            "score": round(float(cand.get("composite_score", 0.0)), 2),
            "area": int(cand.get("area", 0)),
            "aspect_ratio": round(float(cand.get("aspect_ratio", 0.0)), 2),
            "max_intensity": round(float(cand.get("max_intensity", 0.0)), 2),
            "label": "",
            "notes": ""
        })

# Save CSV
df = pd.DataFrame(rows)
df.to_csv(str(csv_path), index=False)

print(f"\n=== BATCH 1 LABELING WORKFLOW GENERATED ===")
print(f"Cutout Images Generated: {cutout_count}")
print(f"Context Images Generated: {context_count}")
print(f"Total Images Generated:   {cutout_count + context_count}")
print(f"CSV Rows Generated:       {len(df)}")
print(f"Images Output Directory:  {output_dir}")
print(f"CSV File Path:           {csv_path}")
