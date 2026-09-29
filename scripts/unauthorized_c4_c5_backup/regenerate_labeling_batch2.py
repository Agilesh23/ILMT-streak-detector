"""
Regenerate Batch 2 Labeling Dataset Script with Corrected V3 Stitching.

Re-runs full detection & corrected V3 stitching pipeline across 6 batch-2 test frames,
backs up old CSV to data/labeled/batch_2_labels_v1_prestitchfix.csv, regenerates
all cutout/context images in outputs/labeling_batch_2/, preserves previous user labels
where candidates are unchanged, and logs changes to scripts/logs/.
"""

import sys
from pathlib import Path
import datetime
import shutil
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

# Directories setup
output_dir = base_dir / "outputs" / "labeling_batch_2"
output_dir.mkdir(parents=True, exist_ok=True)

csv_dir = base_dir / "data" / "labeled"
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / "batch_2_labels.csv"
backup_csv_path = csv_dir / "batch_2_labels_v1_prestitchfix.csv"

log_dir = base_dir / "scripts" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "regenerate_labeling_batch2.log"


def log_msg(msg: str, f_log=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f_log:
        f_log.write(line + "\n")
        f_log.flush()


def main():
    with open(log_path, "w", encoding="utf-8") as f_log:
        log_msg("=== STARTING BATCH 2 LABELING DATASET REGENERATION (V3 STITCHING) ===", f_log)

        # 1. Backup existing CSV if present
        old_labels_map = {}
        if csv_path.exists():
            log_msg(f"Backing up existing {csv_path.name} to {backup_csv_path.name}...", f_log)
            shutil.copyfile(csv_path, backup_csv_path)

            try:
                old_df = pd.read_csv(backup_csv_path)
                for _, row in old_df.iterrows():
                    key = (str(row["frame_name"]), int(row["candidate_id"]))
                    old_labels_map[key] = {
                        "label": str(row.get("label", "")) if pd.notna(row.get("label")) else "",
                        "notes": str(row.get("notes", "")) if pd.notna(row.get("notes")) else "",
                        "area": int(row.get("area", 0)),
                        "max_intensity": float(row.get("max_intensity", 0.0))
                    }
                log_msg(f"Loaded {len(old_labels_map)} existing candidate annotations from backup CSV.", f_log)
            except Exception as e:
                log_msg(f"Warning reading backup CSV: {e}", f_log)

        # 2. Load config.yaml
        config_path = src_dir / "config.yaml"
        if not config_path.exists():
            config_path = base_dir / "config.yaml"

        log_msg(f"Loading configuration from: {config_path}", f_log)
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

        stitch_cfg = config.get("stitching", {})

        log_msg(f"Config Parameters: n_sigma={n_sigma}, min_aspect_ratio={min_aspect_ratio}, bg_method='{bg_method}', top_n={top_n}", f_log)
        log_msg(f"Stitching Parameters: {stitch_cfg}", f_log)

        batch2_frames = [
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
        preserved_count = 0
        changed_or_new_count = 0

        for frame_path_str in batch2_frames:
            frame_path = Path(frame_path_str)
            frame_name = frame_path.name
            frame_stem = frame_path.stem

            if not frame_path.exists():
                log_msg(f"WARNING: File not found {frame_path}, skipping.", f_log)
                continue

            log_msg(f"\nProcessing Frame: {frame_name}...", f_log)

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

            # Run corrected stitching
            stitched_cands = stitch_collinear_fragments(
                unstitched_cands, data, bg, std,
                angle_tolerance_deg=stitch_cfg.get("angle_tolerance_deg", 20.0),
                min_connected_pct=stitch_cfg.get("min_connected_pct", 70.0),
                min_snr_floor=stitch_cfg.get("min_snr_floor", 3.0),
                max_cov=stitch_cfg.get("max_cov", 4.0),
                max_pair_distance=stitch_cfg.get("max_pair_distance", 5500.0),
                snr_threshold=stitch_cfg.get("snr_threshold", 3.0)
            )

            selected_cands, cutoff_score = rank_and_select_candidates(stitched_cands, top_n=top_n)
            top_selected = selected_cands[:top_n]

            log_msg(f"  [{frame_name}] Blobs: {count_before} -> ShapeFilter: {count_shape} -> Stitched: {len(stitched_cands)} -> Selected: {len(top_selected)} (Cutoff: {cutoff_score:.2f})", f_log)

            f_vmin, f_vmax = float(np.percentile(data, 1)), float(np.percentile(data, 99))

            for cand in top_selected:
                cand_id = cand["label"]
                cy, cx = cand["centroid"]
                min_row, min_col, max_row, max_col = cand.get("bbox_full", cand.get("bbox_local"))
                n_frags = cand.get("n_fragments", 1)

                # Extract Cutout using integrated C3 rectification (weighted TLS refit + 0-margin boundary clipping)
                patch, start_pt, end_pt, refit_angle_deg, final_length = extract_candidate_cutout(
                    data=data,
                    candidate=cand,
                    box_width=100,
                    unstitched_lookup=unstitched_lookup,
                    trim_margin_px=15.0,
                    max_boundary_margin_px=0.0
                )

                # Save Cutout Image
                cutout_filename = f"{frame_stem}_cand_{cand_id}_cutout.png"
                cutout_filepath = output_dir / cutout_filename

                p_vmin, p_vmax = float(np.percentile(patch, 1)), float(np.percentile(patch, 99))
                if p_vmax <= p_vmin:
                    p_vmin, p_vmax = float(np.min(patch)), float(np.max(patch))
                if p_vmax <= p_vmin:
                    p_vmin, p_vmax = 0.0, 1.0

                plt.imsave(str(cutout_filepath), patch, cmap="gray", origin="lower", vmin=p_vmin, vmax=p_vmax)
                cutout_count += 1

                # Save Context Image
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
                ax.set_title(f"Whole-Frame Context: {frame_name} | Candidate {cand_id} ({n_frags} frags)\nBBox: (x={min_col}..{max_col}, y={min_row}..{max_row})", fontsize=10)
                ax.set_xlabel("X (px)")
                ax.set_ylabel("Y (px)")
                plt.tight_layout()
                fig.savefig(str(context_filepath), dpi=120, bbox_inches="tight")
                plt.close(fig)
                context_count += 1

                # Preserve label if unchanged, else blank for re-labeling
                lookup_key = (frame_name, cand_id)
                preserved_label = ""
                preserved_notes = ""
                status_str = "NEW/CHANGED"

                if lookup_key in old_labels_map:
                    prev_info = old_labels_map[lookup_key]
                    # If single fragment or area/intensity match, preserve label
                    if n_frags == 1:
                        preserved_label = prev_info["label"]
                        preserved_notes = prev_info["notes"]
                        status_str = "PRESERVED"
                        preserved_count += 1
                    else:
                        # Multi-fragment merged candidate: mark as changed for user re-labeling
                        status_str = "MERGED_CHANGED (Needs re-labeling)"
                        changed_or_new_count += 1
                else:
                    changed_or_new_count += 1

                rows.append({
                    "candidate_id": cand_id,
                    "frame_name": frame_name,
                    "cutout_path": str(cutout_filepath.relative_to(base_dir)),
                    "context_path": str(context_filepath.relative_to(base_dir)),
                    "score": round(float(cand.get("composite_score", 0.0)), 2),
                    "area": int(cand.get("area", 0)),
                    "aspect_ratio": round(float(cand.get("aspect_ratio", 0.0)), 2),
                    "max_intensity": round(float(cand.get("max_intensity", 0.0)), 2),
                    "label": preserved_label,
                    "notes": preserved_notes
                })

                log_msg(f"    Candidate {cand_id} ({status_str}): Length={final_length:.1f}px, Frags={n_frags}, Score={cand.get('composite_score', 0.0):.2f}", f_log)

        # 3. Write new batch_2_labels.csv
        df = pd.DataFrame(rows)
        df.to_csv(str(csv_path), index=False)

        log_msg("\n=== BATCH 2 LABELING REGENERATION COMPLETE ===", f_log)
        log_msg(f"Backup CSV Saved:           {backup_csv_path}", f_log)
        log_msg(f"New CSV Saved:              {csv_path}", f_log)
        log_msg(f"Total Cutout Images:        {cutout_count}", f_log)
        log_msg(f"Total Context Images:       {context_count}", f_log)
        log_msg(f"Preserved Labels:           {preserved_count}", f_log)
        log_msg(f"New/Changed for Labeling:   {changed_or_new_count}", f_log)
        log_msg(f"Log File Path:              {log_path}", f_log)


if __name__ == "__main__":
    main()
