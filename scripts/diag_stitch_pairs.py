"""
Diagnostic script to evaluate collinear fragment stitching criteria on labeled streak fragments.
Frames analyzed: 20240209-0036, 20240211-0013, 20240216-0030.
"""

from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, uniform_filter1d
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates
from src.io_utils import load_fits


def _bbox_angle(c: dict) -> float:
    bbox = c.get("bbox_full", c.get("bbox_local", (0, 0, 1, 1)))
    return float(np.degrees(np.arctan2(bbox[2] - bbox[0], bbox[3] - bbox[1])) % 180.0)


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def evaluate_line_connection(
    sub_img: np.ndarray,
    std_img: np.ndarray,
    cand1: dict,
    cand2: dict,
    snr_threshold: float = 3.0,
    kernel_length: int = 61,
    kernel_width: int = 5,
) -> dict:
    y1, x1 = cand1["centroid"]
    y2, x2 = cand2["centroid"]
    dy, dx = y2 - y1, x2 - x1
    dist = float(np.hypot(dy, dx))

    if dist < 10.0:
        return {
            "dist": dist,
            "pct_above": 0.0,
            "min_snr": 0.0,
            "median_snr": 0.0,
            "cov": 999.0,
        }

    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    n_pts = int(np.ceil(dist))
    t_steps = np.linspace(0, dist, n_pts)
    w_offsets = np.linspace(-kernel_width / 2.0, kernel_width / 2.0, kernel_width)

    sample_y = y1 + t_steps[:, None] * uy + w_offsets[None, :] * ny
    sample_x = x1 + t_steps[:, None] * ux + w_offsets[None, :] * nx

    coords = np.array([sample_y.ravel(), sample_x.ravel()])
    sampled_sub = map_coordinates(sub_img, coords, order=1, mode="constant", cval=0.0).reshape((n_pts, kernel_width))
    sampled_std = map_coordinates(std_img, coords, order=1, mode="nearest").reshape((n_pts, kernel_width))

    signal_1d = np.mean(sampled_sub, axis=1)
    std_1d = np.mean(sampled_std, axis=1)

    response_1d = uniform_filter1d(signal_1d, size=kernel_length, mode="nearest")
    gain_factor = np.sqrt(kernel_length * kernel_width)
    snr_profile = response_1d * gain_factor / np.maximum(std_1d, 1e-6)

    crop_start = int(0.10 * n_pts)
    crop_end = int(0.90 * n_pts)
    gap_snr = snr_profile[crop_start:crop_end] if crop_end > crop_start else snr_profile

    median_snr = float(np.median(gap_snr))
    min_snr = float(np.min(gap_snr))
    pct_above = float(np.mean(gap_snr >= snr_threshold) * 100.0)
    cov = float(np.std(gap_snr) / median_snr) if median_snr > 0 else 999.0

    return {
        "dist": dist,
        "pct_above": pct_above,
        "min_snr": min_snr,
        "median_snr": median_snr,
        "cov": cov,
    }


def main():
    config_path = base_dir / "src/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})
    edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
    border_margin_px = det_cfg.get("border_margin_px", 15)
    border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
    left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
    left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
    bypass_cfg = det_cfg.get("long_thin_bypass", None)
    bleed_veto_cfg = det_cfg.get("bleed_veto", None)

    # Threshold constants matching detect.py / config.yaml
    thresh_min_dist = 10.0
    thresh_max_dist = 4500.0
    thresh_angle_tol = 15.0
    thresh_min_connected_pct = 85.0
    thresh_min_snr_floor = 20.0
    thresh_max_cov = 2.0
    snr_sample_thresh = 3.0

    target_stems = ["20240209-0036", "20240211-0013", "20240216-0030"]

    # Locate FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        stem = p.stem.lower()
        for ts in target_stems:
            if stem == ts.lower():
                fits_by_stem[ts] = p

    # Load verdicts
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    if not verdicts_path.exists():
        print(f"Error: {verdicts_path} not found.")
        sys.exit(1)

    df_v = pd.read_csv(verdicts_path)
    col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
    col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
    col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]
    col_note = [c for c in df_v.columns if "note" in c.lower()][0]

    df_v["stem"] = df_v[col_frame].astype(str).apply(lambda s: Path(s).stem)
    is_streak = df_v[col_verdict].astype(str).str.strip().str.lower() == "streak"
    is_part = df_v[col_note].astype(str).str.strip().str.lower().str.contains("part of a streak", regex=False)
    filtered_v = df_v[is_streak & is_part]

    all_pair_rows = []

    for stem in target_stems:
        fits_file = fits_by_stem.get(stem)
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS file for {stem} not found.")
            continue

        frame_v = filtered_v[filtered_v["stem"] == stem]
        target_cids = set(frame_v[col_cid].astype(int).tolist())

        if not target_cids:
            print(f"Frame {stem}: No candidates with verdict 'streak' and note 'part of a streak'.")
            continue

        # Rebuild candidates exactly as g1_run.py does before stitching
        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)

        raw_candidates, _, _ = find_blob_candidates(
            data,
            bg,
            std,
            edge_mask_max_x=edge_mask_max_x,
            border_margin_px=border_margin_px,
            border_max_length_px=border_max_length_px,
            left_edge_max_col=left_edge_max_col,
            left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
            long_thin_bypass=bypass_cfg,
            bleed_veto=bleed_veto_cfg,
        )

        matched_cands = [c for c in raw_candidates if int(c["label"]) in target_cids]
        matched_cands = sorted(matched_cands, key=lambda c: int(c["label"]))
        n_matched = len(matched_cands)

        frame_pairs = []
        fail_counter = Counter()
        first_fail_counter = Counter()

        for i in range(n_matched):
            for j in range(i + 1, n_matched):
                c1 = matched_cands[i]
                c2 = matched_cands[j]
                cid1 = int(c1["label"])
                cid2 = int(c2["label"])

                y1, x1 = c1["centroid"]
                y2, x2 = c2["centroid"]
                dy, dx = y2 - y1, x2 - x1
                dist = float(np.hypot(dy, dx))

                pass_dist_min = dist >= thresh_min_dist
                pass_dist_max = dist <= thresh_max_dist

                conn_angle = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
                conn_angle_mirrored = float((180.0 - conn_angle) % 180.0)
                b_ang1 = _bbox_angle(c1)
                b_ang2 = _bbox_angle(c2)
                diff_ang1 = _angle_diff(conn_angle, b_ang1)
                diff_ang2 = _angle_diff(conn_angle, b_ang2)
                diff_ang_mirrored_1 = _angle_diff(conn_angle_mirrored, b_ang1)
                diff_ang_mirrored_2 = _angle_diff(conn_angle_mirrored, b_ang2)

                pass_ang1 = diff_ang1 <= thresh_angle_tol
                pass_ang2 = diff_ang2 <= thresh_angle_tol
                pass_angle = pass_ang1 and pass_ang2
                pass_angle_mirrored_both = (diff_ang_mirrored_1 <= thresh_angle_tol) and (diff_ang_mirrored_2 <= thresh_angle_tol)

                # Line profile sampling
                line_res = evaluate_line_connection(
                    sub,
                    std_map,
                    c1,
                    c2,
                    snr_threshold=snr_sample_thresh,
                    kernel_length=61,
                    kernel_width=5,
                )

                pct_above = line_res["pct_above"]
                min_snr = line_res["min_snr"]
                median_snr = line_res["median_snr"]
                cov = line_res["cov"]

                pass_conn_pct = pct_above >= thresh_min_connected_pct
                pass_snr_floor = min_snr >= thresh_min_snr_floor
                pass_cov = cov <= thresh_max_cov

                overall_pass = (
                    pass_dist_min
                    and pass_dist_max
                    and pass_angle
                    and pass_conn_pct
                    and pass_snr_floor
                    and pass_cov
                )

                stitch_pass_if_angle_fixed = (
                    pass_dist_min
                    and pass_dist_max
                    and pass_angle_mirrored_both
                    and pass_conn_pct
                    and pass_snr_floor
                    and pass_cov
                )

                # Determine first failing test according to stitcher execution order
                first_fail = "none"
                if not pass_dist_min:
                    first_fail = "pair_dist_min"
                elif not pass_dist_max:
                    first_fail = "pair_dist_max"
                elif not pass_angle:
                    first_fail = "angle_diff"
                elif not pass_conn_pct:
                    first_fail = "connected_pct"
                elif not pass_snr_floor:
                    first_fail = "snr_floor"
                elif not pass_cov:
                    first_fail = "cov"

                if not pass_dist_min:
                    fail_counter["pair_dist_min"] += 1
                if not pass_dist_max:
                    fail_counter["pair_dist_max"] += 1
                if not pass_angle:
                    fail_counter["angle_diff"] += 1
                if not pass_conn_pct:
                    fail_counter["connected_pct"] += 1
                if not pass_snr_floor:
                    fail_counter["snr_floor"] += 1
                if not pass_cov:
                    fail_counter["cov"] += 1

                if first_fail != "none":
                    first_fail_counter[first_fail] += 1

                row_dict = {
                    "frame": stem,
                    "cand_id_1": cid1,
                    "cand_id_2": cid2,
                    "dist_px": round(dist, 2),
                    "dist_min_thresh": thresh_min_dist,
                    "pass_dist_min": pass_dist_min,
                    "dist_max_thresh": thresh_max_dist,
                    "pass_dist_max": pass_dist_max,
                    "conn_angle_deg": round(conn_angle, 2),
                    "conn_angle_mirrored_deg": round(conn_angle_mirrored, 2),
                    "bbox_angle_1_deg": round(b_ang1, 2),
                    "bbox_angle_2_deg": round(b_ang2, 2),
                    "angle_diff_1_deg": round(diff_ang1, 2),
                    "angle_diff_2_deg": round(diff_ang2, 2),
                    "angle_diff_mirrored_1_deg": round(diff_ang_mirrored_1, 2),
                    "angle_diff_mirrored_2_deg": round(diff_ang_mirrored_2, 2),
                    "angle_tol_deg": thresh_angle_tol,
                    "pass_angle_1": pass_ang1,
                    "pass_angle_2": pass_ang2,
                    "pass_angle_both": pass_angle,
                    "pass_angle_mirrored_both": pass_angle_mirrored_both,
                    "connected_pct": round(pct_above, 2),
                    "connected_pct_thresh": thresh_min_connected_pct,
                    "pass_connected_pct": pass_conn_pct,
                    "min_snr": round(min_snr, 2),
                    "min_snr_floor_thresh": thresh_min_snr_floor,
                    "pass_snr_floor": pass_snr_floor,
                    "median_snr": round(median_snr, 2),
                    "cov": round(cov, 2),
                    "cov_thresh": thresh_max_cov,
                    "pass_cov": pass_cov,
                    "stitch_pass": overall_pass,
                    "stitch_pass_if_angle_fixed": stitch_pass_if_angle_fixed,
                    "first_failing_test": first_fail,
                }
                frame_pairs.append(row_dict)
                all_pair_rows.append(row_dict)

        # Print per frame summary
        n_pairs = len(frame_pairs)
        most_common_first = first_fail_counter.most_common(1)
        first_fail_str = f"{most_common_first[0][0]} ({most_common_first[0][1]} pairs)" if most_common_first else "none"

        mirrored_passed_pairs = [p for p in frame_pairs if p["pass_angle_mirrored_both"]]
        pairs_pass_if_angle_fixed = sum(1 for p in frame_pairs if p["stitch_pass_if_angle_fixed"])
        fail_among_mirrored = Counter()
        for p in mirrored_passed_pairs:
            if not p["pass_dist_min"]:
                fail_among_mirrored["pair_dist_min"] += 1
            if not p["pass_dist_max"]:
                fail_among_mirrored["pair_dist_max"] += 1
            if not p["pass_connected_pct"]:
                fail_among_mirrored["connected_pct"] += 1
            if not p["pass_snr_floor"]:
                fail_among_mirrored["snr_floor"] += 1
            if not p["pass_cov"]:
                fail_among_mirrored["cov"] += 1

        print(f"Frame {stem}:")
        print(f"  Matched streak fragments: {n_matched}")
        print(f"  Total candidate pairs: {n_pairs}")
        print("  Failures per test:")
        print(f"    pair_dist_min (< {thresh_min_dist}px): {fail_counter['pair_dist_min']}")
        print(f"    pair_dist_max (> {thresh_max_dist}px): {fail_counter['pair_dist_max']}")
        print(f"    angle_diff (> {thresh_angle_tol}deg): {fail_counter['angle_diff']}")
        print(f"    connected_pct (< {thresh_min_connected_pct}%): {fail_counter['connected_pct']}")
        print(f"    snr_floor (< {thresh_min_snr_floor}): {fail_counter['snr_floor']}")
        print(f"    cov (> {thresh_max_cov}): {fail_counter['cov']}")
        print(f"  First test failing most often: {first_fail_str}")
        print(f"  Pairs passing mirrored angle test: {len(mirrored_passed_pairs)}")
        print(f"  Pairs passing everything if angle fixed: {pairs_pass_if_angle_fixed}")
        print("  Remaining test failures among pairs passing mirrored angle test:")
        print(f"    pair_dist_min (< {thresh_min_dist}px): {fail_among_mirrored['pair_dist_min']}")
        print(f"    pair_dist_max (> {thresh_max_dist}px): {fail_among_mirrored['pair_dist_max']}")
        print(f"    connected_pct (< {thresh_min_connected_pct}%): {fail_among_mirrored['connected_pct']}")
        print(f"    snr_floor (< {thresh_min_snr_floor}): {fail_among_mirrored['snr_floor']}")
        print(f"    cov (> {thresh_max_cov}): {fail_among_mirrored['cov']}")
        print()

    # Save output CSV
    out_dir = base_dir / "data/diag"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stitch_pairs_v2.csv"
    df_out = pd.DataFrame(all_pair_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved {len(df_out)} pair evaluations to {out_csv}")


if __name__ == "__main__":
    main()
