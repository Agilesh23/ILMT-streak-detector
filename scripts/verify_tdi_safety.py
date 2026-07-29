"""
TDI Artifact Rejection Safety Verification Script for Frame 20230317-0008.fits.

Verifies whether relaxed V3 stitching parameters cause TDI artifact candidates (e.g. 66 / 3984)
to be erroneously merged into false streaks or rank into top candidate selection.
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
log_path = log_dir / "verify_tdi_safety.log"


def log_msg(msg: str, f_log=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f_log:
        f_log.write(line + "\n")
        f_log.flush()


def main():
    with open(log_path, "w", encoding="utf-8") as f_log:
        log_msg("=== STARTING TDI ARTIFACT SAFETY CHECK (20230317-0008.fits) ===", f_log)

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

        frame_path = r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits"
        log_msg(f"Target Frame: {frame_path}", f_log)

        if not Path(frame_path).exists():
            log_msg(f"ERROR: Frame not found: {frame_path}", f_log)
            return

        # 2. Pipeline processing
        data, header = load_fits(frame_path)
        bg, std = estimate_background(data, method=bg_method)

        unstitched_cands, count_before, count_shape = find_blob_candidates(
            data, bg, std,
            n_sigma=n_sigma,
            min_aspect_ratio=min_aspect_ratio,
            min_eccentricity=min_eccentricity,
            max_area=max_area
        )

        cand_by_label = {c["label"]: c for c in unstitched_cands}
        log_msg(f"Shape-Filtered Blobs: {count_shape}", f_log)

        # Target candidate IDs 66 and 3984 (or nearest labels)
        target_ids = [66, 3984]
        found_target_cands = {}
        for t_id in target_ids:
            if t_id in cand_by_label:
                found_target_cands[t_id] = cand_by_label[t_id]
            else:
                # Find candidate closest to label t_id if exact label shifted
                found_target_cands[t_id] = min(unstitched_cands, key=lambda c: abs(c["label"] - t_id))

        log_msg("Target TDI Blobs in Unstitched Set:", f_log)
        for t_id, c in found_target_cands.items():
            log_msg(f"  - Target {t_id} (Matched Label {c['label']}): Centroid=({c['centroid'][1]:.1f}, {c['centroid'][0]:.1f}), Area={c['area']}px, Aspect={c['aspect_ratio']:.1f}, MaxInt={c['max_intensity']:.2f}", f_log)

        # 3. OLD Stitching Pass
        old_stitched = stitch_collinear_fragments(
            unstitched_cands, data, bg, std,
            angle_tolerance_deg=old_stitch_params["angle_tolerance_deg"],
            min_connected_pct=old_stitch_params["min_connected_pct"],
            min_snr_floor=old_stitch_params["min_snr_floor"],
            max_cov=old_stitch_params["max_cov"],
            max_pair_distance=old_stitch_params["max_pair_distance"],
            snr_threshold=old_stitch_params["snr_threshold"]
        )
        old_selected, old_cutoff = rank_and_select_candidates(old_stitched, top_n=top_n)

        # 4. NEW Stitching Pass (Relaxed V3)
        new_stitched = stitch_collinear_fragments(
            unstitched_cands, data, bg, std,
            angle_tolerance_deg=new_stitch_params.get("angle_tolerance_deg", 20.0),
            min_connected_pct=new_stitch_params.get("min_connected_pct", 70.0),
            min_snr_floor=new_stitch_params.get("min_snr_floor", 3.0),
            max_cov=new_stitch_params.get("max_cov", 4.0),
            max_pair_distance=new_stitch_params.get("max_pair_distance", 5500.0),
            snr_threshold=new_stitch_params.get("snr_threshold", 3.0)
        )
        new_selected, new_cutoff = rank_and_select_candidates(new_stitched, top_n=top_n)

        log_msg(f"\nOLD Stitching Result: {len(old_stitched)} candidates | Top Selected: {len(old_selected)}", f_log)
        log_msg(f"NEW Stitching Result: {len(new_stitched)} candidates | Top Selected: {len(new_selected)} (Cutoff Score: {new_cutoff:.2f})", f_log)

        # Track candidate 66 and 3984 under OLD vs NEW
        def _find_merged_container(cand_list, target_label):
            for c in cand_list:
                frags = c.get("stitched_from", [c["label"]])
                if target_label in frags:
                    return c
            return None

        log_msg("\n--- TDI ARTIFACT CANDIDATES TRACKING ---", f_log)
        merged_together_new = False
        container_66_new = _find_merged_container(new_stitched, 66)
        container_3984_new = _find_merged_container(new_stitched, 3984)

        if container_66_new and container_3984_new:
            if container_66_new["label"] == container_3984_new["label"]:
                merged_together_new = True

        for t_id in [66, 3984]:
            old_c = _find_merged_container(old_stitched, t_id)
            new_c = _find_merged_container(new_stitched, t_id)

            old_frags = old_c.get("stitched_from", [old_c["label"]]) if old_c else []
            new_frags = new_c.get("stitched_from", [new_c["label"]]) if new_c else []
            old_len = old_c.get("axis_major_length", 0.0) if old_c else 0.0
            new_len = new_c.get("axis_major_length", 0.0) if new_c else 0.0

            log_msg(f"Target Candidate {t_id}:", f_log)
            log_msg(f"  OLD Params -> Representative Label: {old_c.get('label') if old_c else 'None'} | Length: {old_len:.1f}px | Fragments: {len(old_frags)} {old_frags}", f_log)
            log_msg(f"  NEW Params -> Representative Label: {new_c.get('label') if new_c else 'None'} | Length: {new_len:.1f}px | Fragments: {len(new_frags)} {new_frags}", f_log)

        log_msg(f"\nDid Candidates 66 and 3984 merge together under NEW params?: {merged_together_new}", f_log)

        # 5. Report Top-N Selected Candidates under NEW params
        log_msg("\n--- TOP RANKED SELECTED CANDIDATES UNDER NEW PARAMS ---", f_log)
        tdi_in_top_selected = False
        for i, c in enumerate(new_selected):
            c_id = c.get("label")
            frags = c.get("stitched_from", [c_id])
            l_px = c.get("axis_major_length", 0.0)
            score = c.get("composite_score", 0.0)
            contains_tdi = bool(set(target_ids).intersection(set(frags)))
            if contains_tdi:
                tdi_in_top_selected = True
            log_msg(f"  Rank {i+1}: Label={c_id} | Length={l_px:.1f}px | Fragments={len(frags)} {frags} | Score={score:.2f} | ContainsTDI={contains_tdi}", f_log)

        log_msg(f"\nSAFETY CONCLUSION:", f_log)
        if merged_together_new:
            log_msg("  WARNING: TDI candidates 66 and 3984 MERGED together under new parameters!", f_log)
        else:
            log_msg("  SAFE: TDI candidates 66 and 3984 did NOT merge together under new parameters.", f_log)

        if tdi_in_top_selected:
            log_msg("  WARNING: TDI artifact candidates ranked into the top selection set!", f_log)
        else:
            log_msg("  SAFE: TDI artifact candidates did NOT rank into the top selection set.", f_log)


if __name__ == "__main__":
    main()
