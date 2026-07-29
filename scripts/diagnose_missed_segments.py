"""
Diagnostic Script for Missed/Truncated Streak Segments (Batch 2 Candidates).

Extrapolates candidate line profiles across the full frame to determine if missed
streak segments are BELOW n_sigma=3.0 (detection sensitivity problem) or ABOVE
n_sigma=3.0 (stitching gap/distance/angle tolerance problem).
"""

from pathlib import Path
import sys
import datetime
import yaml
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates, uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup base_dir relative to this script
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import fit_weighted_tls_line, compute_max_inbounds_ray_length

# Directories setup
csv_path = base_dir / "data" / "labeled" / "batch_2_labels.csv"
diag_dir = base_dir / "outputs" / "diagnostics"
diag_dir.mkdir(parents=True, exist_ok=True)

log_dir = base_dir / "scripts" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / "diagnose_missed_segments.log"


def log_msg(msg: str, f_log=None):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if f_log:
        f_log.write(line + "\n")
        f_log.flush()


def main():
    with open(log_path, "w", encoding="utf-8") as f_log:
        log_msg("=== STARTING MISSED SEGMENT DIAGNOSIS FOR BATCH 2 CANDIDATES ===", f_log)

        # Load config
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

        stitch_cfg = config.get("stitching", {})
        max_pair_dist = stitch_cfg.get("max_pair_distance", 3500.0)
        angle_tol = stitch_cfg.get("angle_tolerance_deg", 15.0)

        batch2_frames = [
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221026-0057.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221101-0030.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230602-0022.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20231115-0003.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240114-0028.fits",
            r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20240405-0004.fits",
        ]

        summary_rows = []

        for frame_path_str in batch2_frames:
            frame_path = Path(frame_path_str)
            frame_name = frame_path.name
            frame_stem = frame_path.stem

            if not frame_path.exists():
                log_msg(f"Skipping missing frame: {frame_name}", f_log)
                continue

            log_msg(f"\nAnalyzing Frame: {frame_name}", f_log)

            # Load & detect
            data, header = load_fits(frame_path)
            bg, std = estimate_background(data, method=bg_method)

            unstitched_cands, _, _ = find_blob_candidates(
                data, bg, std,
                n_sigma=n_sigma,
                min_aspect_ratio=min_aspect_ratio,
                min_eccentricity=min_eccentricity,
                max_area=max_area
            )

            cand_by_label = {c["label"]: c for c in unstitched_cands}
            stitched_cands = stitch_collinear_fragments(unstitched_cands, data, bg, std)
            selected_cands, _ = rank_and_select_candidates(stitched_cands, top_n=top_n)

            top_cands = selected_cands[:top_n]
            h_img, w_img = data.shape
            sub_img = data - bg
            std_map = std if isinstance(std, np.ndarray) else np.full_like(sub_img, std)

            for cand in top_cands:
                cand_id = cand["label"]
                cy, cx = cand["centroid"]
                stitched_labels = set(cand.get("stitched_from", [cand_id]))
                min_r, min_c, max_r, max_c = cand.get("bbox_full", cand.get("bbox_local"))

                # Line orientation via TLS if multi-fragment, else bbox orientation
                fragment_centroids = []
                fragment_weights = []
                for lbl in stitched_labels:
                    if lbl in cand_by_label:
                        frag = cand_by_label[lbl]
                        f_cy, f_cx = frag["centroid"]
                        fragment_centroids.append((f_cx, f_cy))
                        fragment_weights.append(frag.get("area", 1.0))

                if len(fragment_centroids) > 1:
                    center, (vx, vy), line_angle_deg = fit_weighted_tls_line(fragment_centroids, fragment_weights)
                    center_x, center_y = center
                else:
                    dy = max_r - min_r
                    dx = max_c - min_c
                    line_angle_deg = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
                    rad = np.radians(line_angle_deg)
                    vx, vy = float(np.cos(rad)), float(np.sin(rad))
                    center_x, center_y = cx, cy

                # Full frame edge-to-edge ray length
                desired_len = float(np.hypot(w_img, h_img))
                back_len, fwd_len = compute_max_inbounds_ray_length(
                    (h_img, w_img), (center_x, center_y), (vx, vy), desired_len, margin_px=0.0
                )

                total_full_len = back_len + fwd_len
                n_pts = int(np.ceil(total_full_len))

                # 1D line sampling along full ray
                t_steps = np.linspace(-back_len, fwd_len, n_pts)
                sample_x = center_x + t_steps * vx
                sample_y = center_y + t_steps * vy

                sample_coords = np.array([sample_y, sample_x])
                sampled_sub = map_coordinates(sub_img, sample_coords, order=1, mode='constant', cval=0.0)
                sampled_std = map_coordinates(std_map, sample_coords, order=1, mode='nearest')

                snr_1d = sampled_sub / np.maximum(sampled_std, 1e-6)
                smooth_snr_1d = uniform_filter1d(snr_1d, size=15, mode='nearest')

                # Identify all pre-stitching candidates along extended line (within 30px perp distance)
                collinear_unstitched = []
                unstitched_above_thresh = []
                unstitched_below_thresh = []

                nx, ny = -vy, vx
                for u_lbl, u_cand in cand_by_label.items():
                    uc_y, uc_x = u_cand["centroid"]
                    u_dx = uc_x - center_x
                    u_dy = uc_y - center_y
                    perp_dist = abs(u_dx * nx + u_dy * ny)
                    proj_dist = u_dx * vx + u_dy * vy

                    if perp_dist <= 30.0 and (-back_len - 50 <= proj_dist <= fwd_len + 50):
                        u_bbox = u_cand.get("bbox_full", u_cand.get("bbox_local", (0, 0, 1, 1)))
                        u_angle = float(np.degrees(np.arctan2(u_bbox[2] - u_bbox[0], u_bbox[3] - u_bbox[1])) % 180.0)
                        angle_diff = min(abs(u_angle - line_angle_deg) % 180.0, 180.0 - (abs(u_angle - line_angle_deg) % 180.0))

                        cand_info = {
                            "label": u_lbl,
                            "centroid": (uc_x, uc_y),
                            "proj_dist": proj_dist,
                            "perp_dist": perp_dist,
                            "max_intensity": u_cand.get("max_intensity", 0.0),
                            "area": u_cand.get("area", 0),
                            "aspect": u_cand.get("aspect_ratio", 0.0),
                            "angle_diff": angle_diff,
                            "in_current_stitch": u_lbl in stitched_labels
                        }
                        collinear_unstitched.append(cand_info)

                        if u_lbl not in stitched_labels:
                            if u_cand.get("max_intensity", 0.0) >= n_sigma:
                                unstitched_above_thresh.append(cand_info)
                            else:
                                unstitched_below_thresh.append(cand_info)

                # Determine if candidate is TRUNCATED
                # Check for significant extended signal (>3.0 SNR for >=50px) outside current candidate projection span
                cand_projs = [c["proj_dist"] for c in collinear_unstitched if c["in_current_stitch"]]
                min_cand_p = min(cand_projs) if cand_projs else -100.0
                max_cand_p = max(cand_projs) if cand_projs else 100.0

                outside_mask = (t_steps < (min_cand_p - 100.0)) | (t_steps > (max_cand_p + 100.0))
                extended_high_snr_px = int(np.sum((smooth_snr_1d >= n_sigma) & outside_mask))
                n_missed_blobs_above = len(unstitched_above_thresh)

                is_truncated = (n_missed_blobs_above > 0) or (extended_high_snr_px > 100)
                status_flag = "TRUNCATED" if is_truncated else "COMPLETE"

                # Root cause determination
                if not is_truncated:
                    root_cause = "N/A (Complete)"
                elif n_missed_blobs_above > 0:
                    root_cause = f"STITCHING_GAP (Found {n_missed_blobs_above} unstitched blobs >3.0-sigma)"
                else:
                    root_cause = f"DETECTION_THRESHOLD (Continuum signal present <3.0-sigma threshold, {extended_high_snr_px}px)"

                log_msg(f"  Candidate {cand_id}: Status={status_flag} | StitchedFrags={len(stitched_labels)} | UnstitchedAboveThresh={n_missed_blobs_above} | RootCause={root_cause}", f_log)

                summary_rows.append({
                    "candidate_id": cand_id,
                    "frame_name": frame_name,
                    "status": status_flag,
                    "stitched_fragment_count": len(stitched_labels),
                    "unstitched_blobs_above_thresh": n_missed_blobs_above,
                    "extended_signal_px": extended_high_snr_px,
                    "root_cause": root_cause,
                    "line_angle_deg": round(line_angle_deg, 2),
                    "current_length_px": round(cand.get("axis_major_length", 0.0), 1),
                    "full_extent_length_px": round(total_full_len, 1),
                })

                # Diagnostic plot for TRUNCATED candidates
                if is_truncated:
                    diag_fig_path = diag_dir / f"{frame_stem}_cand_{cand_id}_line_profile.png"
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [1, 1]})

                    # Panel 1: Full-Frame Strip View
                    vmin, vmax = float(np.percentile(sub_img, 1)), float(np.percentile(sub_img, 99))
                    ax1.imshow(sub_img, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
                    ax1.plot([sample_x[0], sample_x[-1]], [sample_y[0], sample_y[-1]], 'yellow', linestyle='--', linewidth=1.5, label='Extrapolated Line')
                    ax1.plot(cx, cy, 'cyan', marker='x', markersize=8, label='Candidate Centroid')

                    # Overlay bounding box
                    rect = plt.Rectangle((min_c, min_r), max_c - min_c, max_r - min_r, fill=False, edgecolor='red', linewidth=2, label='Current BBox')
                    ax1.add_patch(rect)

                    # Mark missed blobs
                    for m_cand in unstitched_above_thresh:
                        mc_x, mc_y = m_cand["centroid"]
                        ax1.plot(mc_x, mc_y, 'magenta', marker='o', markersize=7, label='Unstitched Fragment (>3-sigma)')

                    ax1.set_title(f"Diagnostic Map: {frame_name} | Candidate {cand_id} ({status_flag})\n{root_cause}", fontsize=11, fontweight='bold')
                    ax1.set_xlabel("X (px)")
                    ax1.set_ylabel("Y (px)")
                    ax1.legend(loc='upper right', fontsize=8)

                    # Panel 2: 1D SNR Profile along Extrapolated Ray
                    ax2.plot(t_steps, smooth_snr_1d, 'blue', linewidth=1.5, label='Smoothed 1D SNR Profile')
                    ax2.axhline(n_sigma, color='red', linestyle='--', linewidth=1.5, label=f'Threshold (n_sigma={n_sigma})')
                    ax2.axvspan(min_cand_p, max_cand_p, color='cyan', alpha=0.25, label='Current Stitched Candidate Extent')

                    ax2.set_title("1D SNR Profile along Full Extrapolated Streak Path", fontsize=10)
                    ax2.set_xlabel("Ray Path Distance from Center (px)")
                    ax2.set_ylabel("Signal-to-Noise Ratio (SNR)")
                    ax2.grid(True, linestyle=':', alpha=0.6)
                    ax2.legend(loc='upper right', fontsize=8)

                    plt.tight_layout()
                    fig.savefig(str(diag_fig_path), dpi=130, bbox_inches='tight')
                    plt.close(fig)
                    log_msg(f"    Saved diagnostic plot: {diag_fig_path.name}", f_log)

        # Save Summary CSV
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = diag_dir / "batch_2_missed_segments_summary.csv"
        summary_df.to_csv(str(summary_csv_path), index=False)

        log_msg("\n=== MISSED SEGMENTS DIAGNOSIS COMPLETE ===", f_log)
        log_msg(f"Summary Table Saved: {summary_csv_path}", f_log)
        log_msg(f"Total Candidates Analyzed: {len(summary_df)}", f_log)
        if len(summary_df) > 0:
            n_trunc = len(summary_df[summary_df["status"] == "TRUNCATED"])
            log_msg(f"Truncated Candidates: {n_trunc} / {len(summary_df)}", f_log)


if __name__ == "__main__":
    main()
