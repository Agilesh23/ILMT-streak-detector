"""
Batch 2 Manual Labeling Dataset Generator for ILMT Streak Detector V3.

Executes full pipeline across 6 new test frames, generating rectified cutouts (with integrated C3 TLS refit and 0-margin boundary clipping),
whole-frame context images, batch_2_labels.csv, and execution log in scripts/logs/.
"""

import sys
from pathlib import Path
import datetime
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup paths relative to script location
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import extract_candidate_cutout

# Setup output & log directories
output_dir = base_dir / "outputs" / "labeling_batch_2"
output_dir.mkdir(parents=True, exist_ok=True)

csv_dir = base_dir / "data" / "labeled"
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / "batch_2_labels.csv"

log_dir = base_dir / "scripts" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "generate_labeling_batch2.log"


def log_message(msg: str, f_log=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    if f_log:
        f_log.write(line + "\n")
        f_log.flush()


def main():
    with open(log_path, "w", encoding="utf-8") as f_log:
        log_message("=== STARTING BATCH 2 LABELING DATASET GENERATION ===", f_log)

        # 1. Load config.yaml
        config_path = src_dir / "config.yaml"
        if not config_path.exists():
            config_path = base_dir / "config.yaml"

        log_message(f"Loading configuration from: {config_path}", f_log)
        with open(config_path, "r", encoding="utf-8") as f_cfg:
            config = yaml.safe_load(f_cfg)

        det_cfg = config.get("detection", {})
        n_sigma = det_cfg.get("n_sigma", 3.0)
        min_aspect_ratio = det_cfg.get("min_aspect_ratio", 3.0)
        min_eccentricity = det_cfg.get("min_eccentricity", 0.9)
        max_area = det_cfg.get("max_area", 15000.0)

        bg_cfg = config.get("background", {})
        bg_method = bg_cfg.get("default_method", "background2d")

        rank_cfg = config.get("ranking", {})
        top_n = rank_cfg.get("top_n", 3)

        log_message(f"Config loaded: n_sigma={n_sigma}, min_aspect_ratio={min_aspect_ratio}, min_eccentricity={min_eccentricity}, max_area={max_area}, bg_method='{bg_method}', top_n={top_n}", f_log)

        # 2. Define 6 NEW test frames (excluding batch 1's 20221025-0028, 20230519-0009, 20230317-0008)
        new_frames = [
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221026-0057.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221101-0030.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230602-0022.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20231115-0003.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240114-0028.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240405-0004.fits",
        ]

        rows = []
        cutout_count = 0
        context_count = 0

        for frame_path_str in new_frames:
            frame_path = Path(frame_path_str)
            frame_name = frame_path.name
            frame_stem = frame_path.stem

            if not frame_path.exists():
                log_message(f"WARNING: File not found {frame_path}, skipping.", f_log)
                continue

            log_message(f"Processing frame: {frame_name}...", f_log)

            # 3. Pipeline execution
            data, header = load_fits(frame_path)
            bg, std = estimate_background(data, method=bg_method)

            unstitched_cands, count_before, count_shape = find_blob_candidates(
                data, bg, std,
                n_sigma=n_sigma,
                min_aspect_ratio=min_aspect_ratio,
                min_eccentricity=min_eccentricity,
                max_area=max_area
            )

            unstitched_lookup = {c["label"]: c for c in unstitched_cands}

            stitched_cands = stitch_collinear_fragments(unstitched_cands, data, bg, std)
            selected_cands, cutoff_score = rank_and_select_candidates(stitched_cands, top_n=top_n)

            top_selected = selected_cands[:top_n]
            log_message(f"  [{frame_name}] Blobs: {count_before} -> ShapeFilter: {count_shape} -> Stitched: {len(stitched_cands)} -> Selected: {len(top_selected)} (Cutoff: {cutoff_score:.2f})", f_log)

            h_img, w_img = data.shape
            f_vmin, f_vmax = float(np.percentile(data, 1)), float(np.percentile(data, 99))

            for cand in top_selected:
                cand_id = cand["label"]
                cy, cx = cand["centroid"]
                min_row, min_col, max_row, max_col = cand.get("bbox_full", cand.get("bbox_local"))

                # 4. Extract Cutout using integrated C3 rectification (weighted TLS refit + 0-margin boundary clipping)
                patch, start_pt, end_pt, refit_angle_deg, final_length = extract_candidate_cutout(
                    data=data,
                    candidate=cand,
                    box_width=100,
                    unstitched_lookup=unstitched_lookup,
                    trim_margin_px=15.0,
                    max_boundary_margin_px=0.0
                )

                # Save Cutout Image (raw image only, NO text/titles)
                cutout_filename = f"{frame_stem}_cand_{cand_id}_cutout.png"
                cutout_filepath = output_dir / cutout_filename

                p_vmin, p_vmax = float(np.percentile(patch, 1)), float(np.percentile(patch, 99))
                if p_vmax <= p_vmin:
                    p_vmin, p_vmax = float(np.min(patch)), float(np.max(patch))
                if p_vmax <= p_vmin:
                    p_vmin, p_vmax = 0.0, 1.0

                plt.imsave(str(cutout_filepath), patch, cmap="gray", origin="lower", vmin=p_vmin, vmax=p_vmax)
                cutout_count += 1

                # Save Context Image (whole-frame thumbnail with red bounding box)
                context_filename = f"{frame_stem}_cand_{cand_id}_context.png"
                context_filepath = output_dir / context_filename

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(data, cmap="gray", origin="lower", vmin=f_vmin, vmax=f_vmax)
                rect_w = max(1, max_col - min_col)
                rect_h = max(1, max_row - min_row)
                rect = patches.Rectangle(
                    (min_col, min_row), rect_w, rect_h,
                    linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)
                ax.plot(cx, cy, "cyan", marker="x", markersize=6)
                ax.set_title(f"Whole-Frame Context: {frame_name} | Candidate {cand_id}\nBBox: (x={min_col}..{max_col}, y={min_row}..{max_row})", fontsize=10)
                ax.set_xlabel("X (px)")
                ax.set_ylabel("Y (px)")
                plt.tight_layout()
                fig.savefig(str(context_filepath), dpi=120, bbox_inches="tight")
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
                log_message(f"    Saved images for candidate {cand_id} (Length: {final_length:.1f}px, Score: {cand.get('composite_score', 0.0):.2f})", f_log)

        # 5. Save batch_2_labels.csv
        df = pd.DataFrame(rows)
        df.to_csv(str(csv_path), index=False)

        log_message("\n=== BATCH 2 LABELING WORKFLOW GENERATION COMPLETE ===", f_log)
        log_message(f"Cutout Images Generated: {cutout_count}", f_log)
        log_message(f"Context Images Generated: {context_count}", f_log)
        log_message(f"Total Images Generated:   {cutout_count + context_count}", f_log)
        log_message(f"CSV Rows Generated:       {len(df)}", f_log)
        log_message(f"Images Output Directory:  {output_dir}", f_log)
        log_message(f"CSV File Path:           {csv_path}", f_log)
        log_message(f"Log File Path:           {log_path}", f_log)


if __name__ == "__main__":
    main()
