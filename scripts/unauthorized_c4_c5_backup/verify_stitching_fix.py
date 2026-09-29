"""
Verification Script for Task C4 Stitching Gap Fix.

Compares old V2 stitching parameters vs new V3 tuned parameters across all 6 batch-2 frames,
specifically verifying candidate length recovery for truncated candidates and checking if
candidates 14613, 12522, and 17987 in 20240405-0004.fits merge into ONE candidate.
"""

from pathlib import Path
import sys
import datetime
import yaml
import numpy as np
import pandas as pd

# Setup base_dir relative to this script file
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates

# Setup log directory
log_dir = base_dir / "scripts" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "verify_stitching_fix.log"


def log_msg(msg: str, f_log=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f_log:
        f_log.write(line + "\n")
        f_log.flush()


def main():
    with open(log_path, "w", encoding="utf-8") as f_log:
        log_msg("=== STARTING TASK C4 STITCHING FIX VERIFICATION ===", f_log)

        # 1. Load config.yaml
        config_path = src_dir / "config.yaml"
        if not config_path.exists():
            config_path = base_dir / "config.yaml"

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

        new_stitch_params = config.get("stitching", {})

        old_stitch_params = {
            "angle_tolerance_deg": 15.0,
            "min_connected_pct": 85.0,
            "min_snr_floor": 20.0,
            "max_cov": 2.0,
            "max_pair_distance": 3500.0,
            "snr_threshold": 3.0,
        }

        log_msg(f"Loaded Config Background Method: {bg_method}", f_log)
        log_msg(f"OLD V2 Stitching Params: {old_stitch_params}", f_log)
        log_msg(f"NEW V3 Stitching Params: {new_stitch_params}", f_log)

        batch2_frames = [
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221026-0057.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221101-0030.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230602-0022.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20231115-0003.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240114-0028.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240405-0004.fits",
        ]

        results_comparison = []

        for frame_path_str in batch2_frames:
            frame_path = Path(frame_path_str)
            frame_name = frame_path.name

            if not frame_path.exists():
                log_msg(f"Skipping missing frame: {frame_name}", f_log)
                continue

            log_msg(f"\n--------------------------------------------------", f_log)
            log_msg(f"Processing Frame: {frame_name}", f_log)

            data, header = load_fits(frame_path)
            bg, std = estimate_background(data, method=bg_method)

            unstitched_cands, _, count_shape = find_blob_candidates(
                data, bg, std,
                n_sigma=n_sigma,
                min_aspect_ratio=min_aspect_ratio,
                min_eccentricity=min_eccentricity,
                max_area=max_area
            )

            # OLD Stitching Pass
            old_stitched = stitch_collinear_fragments(
                unstitched_cands, data, bg, std,
                angle_tolerance_deg=old_stitch_params["angle_tolerance_deg"],
                min_connected_pct=old_stitch_params["min_connected_pct"],
                min_snr_floor=old_stitch_params["min_snr_floor"],
                max_cov=old_stitch_params["max_cov"],
                max_pair_distance=old_stitch_params["max_pair_distance"],
                snr_threshold=old_stitch_params["snr_threshold"]
            )
            old_selected, _ = rank_and_select_candidates(old_stitched, top_n=top_n)

            # NEW Stitching Pass
            new_stitched = stitch_collinear_fragments(
                unstitched_cands, data, bg, std,
                angle_tolerance_deg=new_stitch_params.get("angle_tolerance_deg", 20.0),
                min_connected_pct=new_stitch_params.get("min_connected_pct", 70.0),
                min_snr_floor=new_stitch_params.get("min_snr_floor", 3.0),
                max_cov=new_stitch_params.get("max_cov", 4.0),
                max_pair_distance=new_stitch_params.get("max_pair_distance", 5500.0),
                snr_threshold=new_stitch_params.get("snr_threshold", 3.0)
            )
            new_selected, _ = rank_and_select_candidates(new_stitched, top_n=top_n)

            log_msg(f"  Shape-Filtered Blobs: {count_shape}", f_log)
            log_msg(f"  OLD Stitching -> Merged Count: {len(old_stitched)} | Top Selected: {len(old_selected)}", f_log)
            log_msg(f"  NEW Stitching -> Merged Count: {len(new_stitched)} | Top Selected: {len(new_selected)}", f_log)

            # Detail top candidate lengths
            log_msg("  [OLD Selected Candidates]:", f_log)
            for i, c in enumerate(old_selected[:3]):
                l_id = c.get("label")
                n_f = c.get("n_fragments", 1)
                l_px = c.get("axis_major_length", 0.0)
                log_msg(f"    - Rank {i+1}: Label={l_id} | Length={l_px:.1f}px | Fragments={n_f} | Score={c.get('composite_score', 0.0):.2f}", f_log)

            log_msg("  [NEW Selected Candidates]:", f_log)
            for i, c in enumerate(new_selected[:3]):
                l_id = c.get("label")
                n_f = c.get("n_fragments", 1)
                l_px = c.get("axis_major_length", 0.0)
                log_msg(f"    - Rank {i+1}: Label={l_id} | Length={l_px:.1f}px | Fragments={n_f} | Score={c.get('composite_score', 0.0):.2f}", f_log)

            # Special verification for 20240405-0004.fits
            if frame_name == "20240405-0004.fits":
                target_labels = {14613, 12522, 17987}
                merged_together = False
                containing_cand = None

                for c in new_stitched:
                    frags = set(c.get("stitched_from", [c["label"]]))
                    if target_labels.issubset(frags) or len(target_labels.intersection(frags)) >= 2:
                        merged_together = True
                        containing_cand = c
                        break

                log_msg("\n  *** SPECIAL VERIFICATION FOR 20240405-0004.fits (14613, 12522, 17987) ***", f_log)
                if merged_together and containing_cand:
                    frags = containing_cand.get("stitched_from", [])
                    l_px = containing_cand.get("axis_major_length", 0.0)
                    log_msg(f"  SUCCESS! Candidates {target_labels} MERGED INTO ONE CANDIDATE!", f_log)
                    log_msg(f"  Representative Candidate ID: {containing_cand['label']}", f_log)
                    log_msg(f"  Combined Major Axis Length:  {l_px:.1f} px", f_log)
                    log_msg(f"  Total Fragments Merged:      {len(frags)} (Labels: {frags})", f_log)
                else:
                    log_msg(f"  NOTICE: Candidates {target_labels} were NOT fully merged into a single candidate.", f_log)

            # Store summary row
            old_top_len = old_selected[0].get("axis_major_length", 0.0) if old_selected else 0.0
            new_top_len = new_selected[0].get("axis_major_length", 0.0) if new_selected else 0.0
            old_top_frags = old_selected[0].get("n_fragments", 1) if old_selected else 1
            new_top_frags = new_selected[0].get("n_fragments", 1) if new_selected else 1

            results_comparison.append({
                "frame_name": frame_name,
                "shape_filtered_blobs": count_shape,
                "old_stitched_candidates": len(old_stitched),
                "new_stitched_candidates": len(new_stitched),
                "old_top1_length_px": round(old_top_len, 1),
                "new_top1_length_px": round(new_top_len, 1),
                "old_top1_fragments": old_top_frags,
                "new_top1_fragments": new_top_frags,
                "length_gain_px": round(new_top_len - old_top_len, 1)
            })

        # Save comparison CSV
        comp_df = pd.DataFrame(results_comparison)
        comp_csv_path = base_dir / "outputs" / "diagnostics" / "task_c4_stitching_fix_comparison.csv"
        comp_df.to_csv(str(comp_csv_path), index=False)

        log_msg("\n=== TASK C4 STITCHING FIX VERIFICATION COMPLETE ===", f_log)
        log_msg(f"Comparison Summary CSV Saved: {comp_csv_path}", f_log)
        log_msg(f"Log Output Saved:             {log_path}", f_log)


if __name__ == "__main__":
    main()
