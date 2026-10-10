# TASK D0b-fix1: extent floor and raw-centroid check
"""
Line grouping and streak line verification module.
Implements:
1. Sequential RANSAC line candidate generation on raw blob centroids.
2. Segment peak-offset line refinement (least-squares shift and rotation).
3. Whole-line and segment brightness scoring (z_peak, n_segments_z_gt_3, slope).
4. Threshold acceptance gating (z_peak >= 6.0 and n_segments >= 4).
5. Continuous along-line extent determination via cluster merging of significant bins.
6. Candidate grouping within the line extent.
"""

from collections import defaultdict
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import map_coordinates

# Configuration constants
PERP_TOL_PX = 12
MIN_INLIERS = 5
MIN_SPAN_PX = 800
MAX_LINES = 3
RANSAC_ITERS = 2000
SEED = 42

PEAK_SEARCH_PX = 20
SIGNAL_HALF_WIDTH = 3
POS_BG_OFFSETS = list(range(30, 94, 7))  # [30, 37, 44, 51, 58, 65, 72, 79, 86, 93]
NEG_BG_OFFSETS = [-w for w in POS_BG_OFFSETS]
BG_OFFSETS = sorted(NEG_BG_OFFSETS + POS_BG_OFFSETS)

N_SEGMENTS = 8
BIN_PX = 50
ACCEPT_Z_PEAK = 6.0
ACCEPT_MIN_SEGMENTS = 4
SEGMENT_Z = 3.0

EXTENT_BIN_Z = 2.0
EXTENT_MAX_GAP_BINS = 4
EXTENT_PAD_PX = 100


def get_frame_boundary_intersections(mean_y: float, mean_x: float, uy: float, ux: float, h: int, w: int):
    """
    Compute intersection points of the infinite line (mean_y + t*uy, mean_x + t*ux)
    with the 4 outer image borders [0, h] x [0, w].
    Returns (t_b1, t_b2, (y_b1, x_b1), (y_b2, x_b2)) where t_b1 < t_b2.
    """
    ts = []
    # y = 0
    if abs(uy) > 1e-7:
        t = (0.0 - mean_y) / uy
        x = mean_x + t * ux
        if 0.0 <= x <= w:
            ts.append((t, 0.0, x, "bottom"))
    # y = h
    if abs(uy) > 1e-7:
        t = (float(h) - mean_y) / uy
        x = mean_x + t * ux
        if 0.0 <= x <= w:
            ts.append((t, float(h), x, "top"))
    # x = 0
    if abs(ux) > 1e-7:
        t = (0.0 - mean_x) / ux
        y = mean_y + t * uy
        if 0.0 <= y <= h:
            ts.append((t, y, 0.0, "left"))
    # x = w
    if abs(ux) > 1e-7:
        t = (float(w) - mean_x) / ux
        y = mean_y + t * uy
        if 0.0 <= y <= h:
            ts.append((t, y, float(w), "right"))

    unique_ts = []
    for item in sorted(ts, key=lambda x: x[0]):
        if not unique_ts or abs(item[0] - unique_ts[-1][0]) > 1.0:
            unique_ts.append(item)

    if len(unique_ts) >= 2:
        return unique_ts[0], unique_ts[-1]
    return None, None


def find_lines_ransac(candidates: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Sequential RANSAC line finder on candidate centroids followed by PCA refit.
    """
    if len(candidates) < MIN_INLIERS:
        return []

    coords = np.array([c["centroid"] for c in candidates], dtype=np.float64)
    remaining_indices = list(range(len(candidates)))
    lines = []

    while len(lines) < MAX_LINES and len(remaining_indices) >= MIN_INLIERS:
        rem_arr = np.array(remaining_indices)
        pts = coords[rem_arr]
        M = len(pts)

        if M < MIN_INLIERS:
            break

        best_inliers_mask = None
        best_count = 0
        best_span = 0.0

        for _ in range(RANSAC_ITERS):
            idx1, idx2 = rng.choice(M, size=2, replace=False)
            p1 = pts[idx1]
            p2 = pts[idx2]

            dy = p2[0] - p1[0]
            dx = p2[1] - p1[1]
            dist = np.hypot(dy, dx)
            if dist < 1e-4:
                continue

            uy = dy / dist
            ux = dx / dist

            diff_y = pts[:, 0] - p1[0]
            diff_x = pts[:, 1] - p1[1]
            perp_dist = np.abs(diff_y * ux - diff_x * uy)

            inlier_mask = perp_dist <= PERP_TOL_PX
            n_inl = int(np.sum(inlier_mask))
            if n_inl < MIN_INLIERS:
                continue

            projs = diff_y[inlier_mask] * uy + diff_x[inlier_mask] * ux
            span = float(np.max(projs) - np.min(projs))

            if span < MIN_SPAN_PX:
                continue

            if n_inl > best_count or (n_inl == best_count and span > best_span):
                best_count = n_inl
                best_span = span
                best_inliers_mask = inlier_mask

        if best_inliers_mask is None:
            break

        inlier_global_indices = rem_arr[best_inliers_mask]
        inlier_pts = coords[inlier_global_indices]

        # PCA Refit
        mean_pt = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - mean_pt

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx_sort]

        vec = eigvecs[:, 0]
        uy, ux = float(vec[0]), float(vec[1])
        if ux < 0 or (abs(ux) < 1e-7 and uy < 0):
            uy, ux = -uy, -ux

        diff_y = inlier_pts[:, 0] - mean_pt[0]
        diff_x = inlier_pts[:, 1] - mean_pt[1]
        perp_dists = np.abs(diff_y * ux - diff_x * uy)
        rms_residual = float(np.sqrt(np.mean(perp_dists ** 2)))

        projs = diff_y * uy + diff_x * ux
        span_px = float(np.max(projs) - np.min(projs))
        t_min = float(np.min(projs))
        t_max = float(np.max(projs))

        angle_deg = float(np.degrees(np.arctan2(uy, ux)) % 180.0)
        inlier_cands = [candidates[i] for i in inlier_global_indices]

        lines.append({
            "line_id": len(lines) + 1,
            "inlier_global_indices": inlier_global_indices,
            "inliers": inlier_cands,
            "n_inliers": len(inlier_cands),
            "span_px": round(span_px, 2),
            "rms_residual_px": round(rms_residual, 2),
            "angle_deg": round(angle_deg, 2),
            "mean_y": mean_pt[0],
            "mean_x": mean_pt[1],
            "uy": uy,
            "ux": ux,
            "t_min": t_min,
            "t_max": t_max,
        })

        inlier_set = set(inlier_global_indices)
        remaining_indices = [i for i in remaining_indices if i not in inlier_set]

    return lines


def sample_full_line_strip(sub: np.ndarray, mean_y: float, mean_x: float, uy: float, ux: float, h: int, w: int, half_w: int = 100):
    """
    Sample a straightened strip along the full line from frame edge to edge,
    with perpendicular offsets w in [-half_w, +half_w] px.
    """
    pt_b1, pt_b2 = get_frame_boundary_intersections(mean_y, mean_x, uy, ux, h, w)
    if pt_b1 is not None and pt_b2 is not None:
        t_start = pt_b1[0]
        t_end = pt_b2[0]
    else:
        t_start = -2000.0
        t_end = 2000.0

    w_vals = np.arange(-float(half_w), float(half_w) + 1.0, 1.0)
    t_vals = np.arange(t_start, t_end + 1.0, 1.0)
    T_grid, W_grid = np.meshgrid(t_vals, w_vals)

    Y_strip = mean_y + T_grid * uy - W_grid * ux
    X_strip = mean_x + T_grid * ux + W_grid * uy

    strip = map_coordinates(
        sub,
        [Y_strip.ravel(), X_strip.ravel()],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(Y_strip.shape)

    return strip, t_vals, w_vals, t_start, t_end


def compute_S(strip: np.ndarray, w_vals: np.ndarray, w0: float, bin_len: int = 50) -> float:
    """
    Compute S(w0) = median over along-line bins of mean over rows with |w - w0| <= 3.
    """
    row_mask = np.abs(w_vals - w0) <= SIGNAL_HALF_WIDTH
    if not np.any(row_mask):
        return 0.0

    col_means = np.mean(strip[row_mask, :], axis=0)
    n_cols = len(col_means)
    if n_cols == 0:
        return 0.0

    n_bins = max(1, n_cols // bin_len)
    bin_means = []
    for b in range(n_bins):
        c0 = b * bin_len
        c1 = min((b + 1) * bin_len, n_cols) if b == n_bins - 1 else (b + 1) * bin_len
        if c1 > c0:
            bin_means.append(float(np.mean(col_means[c0:c1])))

    return float(np.median(bin_means)) if bin_means else 0.0


def score_strip_line(strip: np.ndarray, w_vals: np.ndarray, t_vals: np.ndarray):
    """
    Compute peak search, background distribution, z_peak, and segment peak stats.
    """
    # Whole-line peak search over w0 in [-20, +20]
    search_S = {w0: compute_S(strip, w_vals, float(w0), bin_len=BIN_PX) for w0 in range(-PEAK_SEARCH_PX, PEAK_SEARCH_PX + 1)}
    peak_offset = max(search_S.keys(), key=lambda w0: search_S[w0])
    peak_val = search_S[peak_offset]

    bg_vals = [compute_S(strip, w_vals, float(w0), bin_len=BIN_PX) for w0 in BG_OFFSETS]
    mu_bg = float(np.mean(bg_vals))
    sigma_bg = float(np.std(bg_vals))
    z_peak = float((peak_val - mu_bg) / sigma_bg) if sigma_bg > 1e-6 else 0.0

    # 8 equal segments along the crossing
    n_cols = strip.shape[1]
    seg_edges = np.linspace(0, n_cols, N_SEGMENTS + 1, dtype=int)
    z_seg_peaks = []
    peak_offsets_seg = []
    t_mid_segs = []

    for s in range(N_SEGMENTS):
        c0 = seg_edges[s]
        c1 = seg_edges[s + 1]
        seg_strip = strip[:, c0:c1]

        seg_search_S = {w0: compute_S(seg_strip, w_vals, float(w0), bin_len=BIN_PX) for w0 in range(-PEAK_SEARCH_PX, PEAK_SEARCH_PX + 1)}
        p_off_seg = max(seg_search_S.keys(), key=lambda w0: seg_search_S[w0])
        p_val_seg = seg_search_S[p_off_seg]

        bg_seg_vals = [compute_S(seg_strip, w_vals, float(w0), bin_len=BIN_PX) for w0 in BG_OFFSETS]
        mu_bg_s = float(np.mean(bg_seg_vals))
        sigma_bg_s = float(np.std(bg_seg_vals))
        z_s = float((p_val_seg - mu_bg_s) / sigma_bg_s) if sigma_bg_s > 1e-6 else 0.0

        z_seg_peaks.append(round(z_s, 2))
        peak_offsets_seg.append(int(p_off_seg))
        t_mid_segs.append(float(np.mean(t_vals[c0:c1])))

    n_gt_3 = sum(1 for z in z_seg_peaks if z > SEGMENT_Z)

    # Slope of peak_offset_seg vs 1-based segment index
    seg_idx_valid = [s + 1 for s in range(N_SEGMENTS) if z_seg_peaks[s] > SEGMENT_Z]
    offsets_valid = [peak_offsets_seg[s] for s in range(N_SEGMENTS) if z_seg_peaks[s] > SEGMENT_Z]

    if len(seg_idx_valid) >= 3:
        poly = np.polyfit(seg_idx_valid, offsets_valid, 1)
        slope_val = round(float(poly[0]), 3)
    else:
        slope_val = np.nan

    return {
        "peak_offset": int(peak_offset),
        "peak_val": float(peak_val),
        "z_peak": round(z_peak, 2),
        "z_seg_peaks": z_seg_peaks,
        "peak_offsets_seg": peak_offsets_seg,
        "t_mid_segs": t_mid_segs,
        "n_segments_z_gt_3": n_gt_3,
        "slope_px_per_segment": slope_val,
    }


def determine_line_extent(strip: np.ndarray, w_vals: np.ndarray, t_start: float, t_end: float, mean_y: float, mean_x: float, uy: float, ux: float):
    """
    Determine the along-line streak extent by identifying continuous clusters
    of 50 px bins with z_bin >= EXTENT_BIN_Z.
    """
    n_cols = strip.shape[1]
    n_bins = max(1, n_cols // BIN_PX)
    z_bins = []
    signal_bins = []

    row_mask_signal = np.abs(w_vals - 0.0) <= SIGNAL_HALF_WIDTH
    row_masks_bg = [np.abs(w_vals - float(w0)) <= SIGNAL_HALF_WIDTH for w0 in BG_OFFSETS]

    for b in range(n_bins):
        c0 = b * BIN_PX
        c1 = min((b + 1) * BIN_PX, n_cols)
        bin_patch = strip[:, c0:c1]

        sig_val = float(np.mean(bin_patch[row_mask_signal, :]))
        signal_bins.append(sig_val)

        bg_vals = [float(np.mean(bin_patch[rm, :])) for rm in row_masks_bg]
        mu_b = float(np.mean(bg_vals))
        sigma_b = float(np.std(bg_vals))
        zb = (sig_val - mu_b) / max(sigma_b, 1e-6)
        z_bins.append(zb)

    # Find clusters of marked bins (z_bin >= EXTENT_BIN_Z) with gaps <= EXTENT_MAX_GAP_BINS
    marked_indices = [b for b, zb in enumerate(z_bins) if zb >= EXTENT_BIN_Z]

    if not marked_indices:
        t_s = t_start
        t_e = t_end
    else:
        clusters = []
        curr_clust = [marked_indices[0]]

        for idx in marked_indices[1:]:
            if idx - curr_clust[-1] <= EXTENT_MAX_GAP_BINS + 1:
                curr_clust.append(idx)
            else:
                clusters.append(curr_clust)
                curr_clust = [idx]
        clusters.append(curr_clust)

        # Select cluster with largest total signal
        best_clust = max(clusters, key=lambda cl: sum(signal_bins[b] for b in range(cl[0], cl[-1] + 1)))
        b_first = best_clust[0]
        b_last = best_clust[-1]

        c_start = b_first * BIN_PX
        c_end = min((b_last + 1) * BIN_PX, n_cols)

        t_clust_start = t_start + c_start
        t_clust_end = t_start + c_end

        t_s = max(t_start, t_clust_start - EXTENT_PAD_PX)
        t_e = min(t_end, t_clust_end + EXTENT_PAD_PX)

    # Endpoints in (row, col)
    p0 = (round(mean_y + t_s * uy, 1), round(mean_x + t_s * ux, 1))
    p1 = (round(mean_y + t_e * uy, 1), round(mean_x + t_e * ux, 1))

    return round(t_s, 1), round(t_e, 1), (p0, p1)


def find_streak_lines(candidates: list[dict], data: np.ndarray, bg: float | np.ndarray) -> list[dict]:
    """
    Main line grouping and verification pipeline.
    Returns list of line dicts.
    """
    h_img, w_img = data.shape
    sub = data - bg

    rng = np.random.default_rng(SEED)
    initial_lines = find_lines_ransac(candidates, rng=rng)
    final_lines = []

    for line in initial_lines:
        mean_y = line["mean_y"]
        mean_x = line["mean_x"]
        uy = line["uy"]
        ux = line["ux"]
        old_angle = line["angle_deg"]

        # Step 2: Segment peak-offset evaluation for refinement
        strip_init, t_vals_init, w_vals_init, t_s_init, t_e_init = sample_full_line_strip(
            sub, mean_y, mean_x, uy, ux, h_img, w_img, half_w=100
        )
        init_stats = score_strip_line(strip_init, w_vals_init, t_vals_init)

        # Fit offset(t) = a + b*t for segments with z_seg_peak > SEGMENT_Z
        valid_t = []
        valid_w = []
        for s in range(N_SEGMENTS):
            if init_stats["z_seg_peaks"][s] > SEGMENT_Z:
                valid_t.append(init_stats["t_mid_segs"][s])
                valid_w.append(init_stats["peak_offsets_seg"][s])

        refined = False
        if len(valid_t) >= 3:
            poly = np.polyfit(valid_t, valid_w, 1)
            b_slope = float(poly[0])
            a_intercept = float(poly[1])

            # Shift centroid by a along normal (-ux, uy)
            mean_y = mean_y - a_intercept * ux
            mean_x = mean_x + a_intercept * uy

            # Rotate line by delta_theta = atan(b)
            delta_theta = math.atan(b_slope)
            theta_old_rad = math.atan2(uy, ux)
            theta_new_rad = theta_old_rad + delta_theta

            uy = math.sin(theta_new_rad)
            ux = math.cos(theta_new_rad)
            if ux < 0 or (abs(ux) < 1e-7 and uy < 0):
                uy, ux = -uy, -ux
            refined = True

        # Step 3: Score refined line
        strip_ref, t_vals_ref, w_vals_ref, t_s_ref, t_e_ref = sample_full_line_strip(
            sub, mean_y, mean_x, uy, ux, h_img, w_img, half_w=100
        )
        ref_stats = score_strip_line(strip_ref, w_vals_ref, t_vals_ref)

        z_peak = ref_stats["z_peak"]
        n_gt_3 = ref_stats["n_segments_z_gt_3"]

        # Step 4: Acceptance check
        accepted = bool(z_peak >= ACCEPT_Z_PEAK and n_gt_3 >= ACCEPT_MIN_SEGMENTS)

        # Step 5: Extent determination
        if accepted:
            extent_start, extent_end, _ = determine_line_extent(
                strip_ref, w_vals_ref, t_s_ref, t_e_ref, mean_y, mean_x, uy, ux
            )

            # Compute inlier_t_min and inlier_t_max after refinement (d_perp <= PERP_TOL_PX)
            inlier_t_vals = []
            for c in line["inliers"]:
                cy, cx = c["centroid"]
                t_c = (cy - mean_y) * uy + (cx - mean_x) * ux
                d_perp = abs((cy - mean_y) * ux - (cx - mean_x) * uy)
                if d_perp <= PERP_TOL_PX:
                    inlier_t_vals.append(t_c)

            if inlier_t_vals:
                inlier_t_min = float(min(inlier_t_vals))
                inlier_t_max = float(max(inlier_t_vals))
            else:
                inlier_t_min = float(extent_start)
                inlier_t_max = float(extent_end)

            inlier_floor_start = inlier_t_min - EXTENT_PAD_PX
            inlier_floor_end = inlier_t_max + EXTENT_PAD_PX

            if inlier_floor_start < extent_start:
                extent_source_start = "inlier_floor"
                cand_t_s = inlier_floor_start
            else:
                extent_source_start = "brightness"
                cand_t_s = extent_start

            if inlier_floor_end > extent_end:
                extent_source_end = "inlier_floor"
                cand_t_e = inlier_floor_end
            else:
                extent_source_end = "brightness"
                cand_t_e = extent_end

            # Clipped to the frame
            final_t_s = max(t_s_ref, cand_t_s)
            final_t_e = min(t_e_ref, cand_t_e)

            t_s_ext = round(final_t_s, 1)
            t_e_ext = round(final_t_e, 1)

            p0 = (round(mean_y + final_t_s * uy, 1), round(mean_x + final_t_s * ux, 1))
            p1 = (round(mean_y + final_t_e * uy, 1), round(mean_x + final_t_e * ux, 1))
            endpoints = (p0, p1)
        else:
            t_s_ext, t_e_ext, endpoints = None, None, None
            extent_source_start, extent_source_end = None, None

        # Step 6: Group members
        inlier_labels = [int(c["label"]) for c in line["inliers"]]
        if accepted:
            members = []
            for c in candidates:
                cy, cx = c["centroid"]
                t_c = (cy - mean_y) * uy + (cx - mean_x) * ux
                d_perp = abs((cy - mean_y) * ux - (cx - mean_x) * uy)
                if d_perp <= PERP_TOL_PX and (t_s_ext <= t_c <= t_e_ext):
                    members.append(int(c["label"]))
            fragment_ids = members
            n_fragments = len(members)
            members_if_accepted = members
        else:
            fragment_ids = []
            n_fragments = 0
            members_if_accepted = inlier_labels

        cur_angle_deg = float(np.degrees(np.arctan2(uy, ux)) % 180.0)

        final_lines.append({
            "line_id": line["line_id"],
            "centroid": (round(float(mean_y), 2), round(float(mean_x), 2)),
            "unit_vector": (round(float(uy), 5), round(float(ux), 5)),
            "angle_deg": round(cur_angle_deg, 2),
            "n_inliers": line["n_inliers"],
            "span_px": line["span_px"],
            "rms_residual_px": line["rms_residual_px"],
            "refined": refined,
            "peak_offset": ref_stats["peak_offset"],
            "z_peak": ref_stats["z_peak"],
            "n_segments_z_gt_3": n_gt_3,
            "slope_px_per_segment": ref_stats["slope_px_per_segment"],
            "accepted": accepted,
            "t_start": t_s_ext,
            "t_end": t_e_ext,
            "extent_source_start": extent_source_start,
            "extent_source_end": extent_source_end,
            "endpoints": endpoints,
            "fragment_ids": fragment_ids,
            "n_fragments": n_fragments,
            "members_if_accepted": members_if_accepted,
        })

    return final_lines
