# TASK D1 + D1a + D1b + D1c: feature extraction for candidates_v2 (refined axis scoring, wide-band baseline, wing SNR)
"""
Feature extraction for every candidate in data/d0b/candidates_v2.csv.

TASK D1a enhancements:
  - Axis refinement for singles: grid search over angle offset [-3, +3] deg (step 0.1)
    and perpendicular shift [-20, +20] px (step 2) around box centre.
  - Scoring: median over along-axis segments of peak of smoothed transverse profile (sigma).
  - Saturation: uses FITS header saturation keyword if present, otherwise fixed 60000 ADU.
  - Wide-band baseline: measured from |w| in [45, 60] px over window [-60, +60] px.
  - Saves angle_offset_deg, shift_px, refined_r0, refined_c0, refined_r1, refined_c1.
  - Prints star FWHM fallback count, wide-band lines diagnostics, and angle offset distribution.
"""

from collections import defaultdict
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter1d, map_coordinates

# ── Paths & Setup ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from src.background import estimate_background
from src.io_utils import load_fits

CANDIDATES_CSV      = BASE_DIR / "data" / "d0b" / "candidates_v2.csv"
LABEL_OVERRIDES_CSV = BASE_DIR / "data" / "d0b" / "label_overrides.csv"
ALL_LINES_CSV       = BASE_DIR / "data" / "d0b" / "all_lines_v2.csv"
OUTPUT_DIR          = BASE_DIR / "data" / "d1"
OUTPUT_CSV          = OUTPUT_DIR / "features_v2.csv"

FITS_SEARCH_ROOT = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")


# ── FITS Fast Lookup ──────────────────────────────────────────────────────────
def build_fits_index():
    index = {}
    search_dirs = [
        FITS_SEARCH_ROOT / "test_files",
        FITS_SEARCH_ROOT / "sample_notebook" / "data",
        FITS_SEARCH_ROOT,
        BASE_DIR / "data",
    ]
    for s_dir in search_dirs:
        if not s_dir.exists():
            continue
        for p in s_dir.rglob("*.fits"):
            stem = p.stem.lower()
            if stem not in index:
                index[stem] = p
    return index


def ensure_fits(name: str) -> str:
    s = str(name).strip()
    return s if s.lower().endswith(".fits") else s + ".fits"


# ── Saturation Level Determination ────────────────────────────────────────────
def get_frame_saturation_level(header: dict) -> tuple[float, str]:
    """
    Look for FITS header saturation keyword; otherwise fallback to fixed 60000 ADU.
    """
    for k in ["SATURATE", "SATURAT", "SATURATION", "SATLEVEL"]:
        for hk in header:
            if hk.upper() == k:
                try:
                    val = float(header[hk])
                    if val > 1000.0:
                        return val, f"header '{hk}'"
                except (ValueError, TypeError):
                    pass
    return 60000.0, "fixed 60000 ADU"


# ── Star FWHM Estimator ───────────────────────────────────────────────────────
def estimate_frame_star_fwhm(data: np.ndarray, sub_sigma: np.ndarray, sat_thresh: float, default_fwhm: float = 6.79) -> tuple[float, bool]:
    """
    Empirically measure median FWHM of isolated, unsaturated stars in the frame.
    Returns (fwhm_val, is_fallback).
    """
    h, w = data.shape
    peak_thresh = 0.95 * sat_thresh

    step = 128
    peak_coords = []
    for r in range(step // 2, h - step // 2, step):
        for c in range(step // 2, min(w - step // 2, 3900), step):
            block_sig = sub_sigma[r - 32:r + 32, c - 32:c + 32]
            block_raw = data[r - 32:r + 32, c - 32:c + 32]
            max_idx = np.unravel_index(np.argmax(block_sig), block_sig.shape)
            pk_sig = block_sig[max_idx]
            pk_raw = block_raw[max_idx]

            if 10.0 <= pk_sig <= 150.0 and pk_raw < peak_thresh:
                pr = r - 32 + max_idx[0]
                pc = c - 32 + max_idx[1]
                if 20 <= pr < h - 20 and 20 <= pc < w - 20:
                    peak_coords.append((pr, pc))

    fwhm_list = []
    half_box = 10
    box_size = 2 * half_box + 1

    for pr, pc in peak_coords[:80]:
        stamp = data[pr - half_box:pr + half_box + 1, pc - half_box:pc + half_box + 1].astype(np.float64)
        if stamp.shape != (box_size, box_size):
            continue

        perim = np.concatenate([stamp[0, :], stamp[-1, :], stamp[:, 0], stamp[:, -1]])
        bg_local = float(np.median(perim))
        net = np.maximum(0.0, stamp - bg_local)
        peak_val = float(np.max(net))

        if peak_val < 10.0:
            continue

        above_half = net >= (0.5 * peak_val)
        area_half = float(np.sum(above_half))

        if 2.0 <= area_half <= 150.0:
            fwhm_est = 2.0 * math.sqrt(area_half / math.pi)
            if 3.0 <= fwhm_est <= 15.0:
                fwhm_list.append(fwhm_est)

    if len(fwhm_list) >= 5:
        return float(np.median(fwhm_list)), False
    return float(default_fwhm), True


# ── Geometry & Refinement Helpers ─────────────────────────────────────────────
def parse_endpoints(ep_str):
    """Parse '(r0, c0); (r1, c1)' to ((r0, c0), (r1, c1)) or None."""
    if not ep_str or (isinstance(ep_str, float) and np.isnan(ep_str)):
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ep_str))
    if len(nums) >= 4:
        return (float(nums[0]), float(nums[1])), (float(nums[2]), float(nums[3]))
    return None


def dist_point_to_segment(pts_r, pts_c, r0, c0, r1, c1):
    """Distance from array of (r, c) points to line segment (r0, c0)-(r1, c1)."""
    dr = r1 - r0
    dc = c1 - c0
    l2 = dr * dr + dc * dc
    if l2 < 1e-8:
        return np.hypot(pts_r - r0, pts_c - c0)
    t = np.clip(((pts_r - r0) * dr + (pts_c - c0) * dc) / l2, 0.0, 1.0)
    proj_r = r0 + t * dr
    proj_c = c0 + t * dc
    return np.hypot(pts_r - proj_r, pts_c - proj_c)


def refine_single_axis(
    sub_sigma: np.ndarray,
    c_row: float,
    c_col: float,
    p0: tuple[float, float],
    p1: tuple[float, float],
    length_px: float,
) -> tuple[float, float, tuple[float, float], tuple[float, float], bool, float, float]:
    """
    Search perpendicular shift [-40, +40] px and angle offset [-3, +3] deg around box centre.
    Coarse-to-fine search:
      - Coarse: angle step 0.5 deg, shift step 4 px.
      - Fine around best: angle step 0.1 deg, shift step 1 px.
      - Subsample axis every 4 px along axis.
    Score: peak of baseline-subtracted median transverse profile over whole axis
           (w in [-60, +60], baseline from |w| in [45, 60]), divided by std in baseline region.
    Acceptance: keep raw diagonal unless refined score beats raw score by > 20%.
    Returns (best_d_theta, best_shift, ref_p0, ref_p1, refined_accepted, score_raw, score_refined).
    """
    dr = p1[0] - p0[0]
    dc = p1[1] - p0[1]
    base_theta = math.atan2(dr, dc)

    # Subsample axis every 4 px along axis
    t_axis = np.arange(-length_px / 2.0, length_px / 2.0 + 0.1, 4.0)
    if len(t_axis) < 2:
        t_axis = np.array([-length_px / 2.0, length_px / 2.0], dtype=float)

    # Total transverse coverage: shifts [-40, 40] + profile window [-60, 60] -> [-100, 100] (201 pts)
    w_tot = np.arange(-100.0, 101.0, 1.0)
    T_mesh, W_mesh = np.meshgrid(t_axis, w_tot, indexing="ij")

    # Profile baseline mask for w in [-60, +60] px
    w_range = np.arange(-60.0, 61.0, 1.0)
    base_mask = (np.abs(w_range) >= 45.0) & (np.abs(w_range) <= 60.0)

    # Cache sampled median profile strip per angle
    cache_strips = {}

    def get_med_strip(d_theta_val: float) -> np.ndarray:
        key = round(float(d_theta_val), 3)
        if key in cache_strips:
            return cache_strips[key]
        ang_rad = base_theta + math.radians(key)
        u_r = math.sin(ang_rad)
        u_c = math.cos(ang_rad)
        v_r = -u_c
        v_c = u_r

        r_grid = c_row + T_mesh * u_r + W_mesh * v_r
        c_grid = c_col + T_mesh * u_c + W_mesh * v_c

        sampled = map_coordinates(
            sub_sigma, [r_grid.ravel(), c_grid.ravel()],
            order=1, mode="constant", cval=0.0
        ).reshape(r_grid.shape)

        strip_med = np.median(sampled, axis=0)  # shape (201,)
        cache_strips[key] = strip_med
        return strip_med

    def eval_score(strip: np.ndarray, shift_val: float) -> float:
        idx0 = int(round(shift_val + 40.0))
        prof = strip[idx0 : idx0 + 121]
        bg = float(np.median(prof[base_mask]))
        net = prof - bg
        pk = float(np.max(net))
        std = float(np.std(prof[base_mask]))
        if std > 1e-9 and not np.isnan(std):
            return float(pk / std)
        return 0.0

    # 1. Raw score at (angle_offset_deg = 0.0, shift_px = 0.0)
    strip_0 = get_med_strip(0.0)
    score_raw = eval_score(strip_0, 0.0)

    # 2. Coarse Search: angle step 0.5 deg, shift step 4 px
    coarse_angles = np.arange(-3.0, 3.01, 0.5)
    coarse_shifts = np.arange(-40.0, 40.1, 4.0)

    best_coarse_score = -1e9
    best_coarse_theta = 0.0
    best_coarse_shift = 0.0

    for d_th in coarse_angles:
        strip = get_med_strip(d_th)
        for s_px in coarse_shifts:
            sc = eval_score(strip, s_px)
            if sc > best_coarse_score:
                best_coarse_score = sc
                best_coarse_theta = float(d_th)
                best_coarse_shift = float(s_px)

    # 3. Fine Search: angle step 0.1 deg, shift step 1 px around best coarse
    th_min = max(-3.0, round(best_coarse_theta - 0.5, 2))
    th_max = min(3.0, round(best_coarse_theta + 0.5, 2))
    fine_angles = np.round(np.arange(th_min, th_max + 0.01, 0.1), 1)

    s_min = max(-40.0, round(best_coarse_shift - 4.0, 1))
    s_max = min(40.0, round(best_coarse_shift + 4.0, 1))
    fine_shifts = np.round(np.arange(s_min, s_max + 0.1, 1.0), 0)

    best_score = best_coarse_score
    best_theta = best_coarse_theta
    best_shift = best_coarse_shift

    for d_th in fine_angles:
        strip = get_med_strip(d_th)
        for s_px in fine_shifts:
            sc = eval_score(strip, s_px)
            if sc > best_score:
                best_score = sc
                best_theta = float(d_th)
                best_shift = float(s_px)

    score_refined = float(best_score)

    # 4. Acceptance: keep raw better diagonal unless refined beats raw by > 20%
    if score_raw > 0.0:
        refined_accepted = bool(score_refined > 1.20 * score_raw)
    else:
        refined_accepted = bool(score_refined > 0.0)

    if refined_accepted:
        best_d_theta = float(best_theta)
        best_shift_out = float(best_shift)
        best_ang_rad = base_theta + math.radians(best_d_theta)
        u_r_best = math.sin(best_ang_rad)
        u_c_best = math.cos(best_ang_rad)
        v_r_best = -u_c_best
        v_c_best = u_r_best

        c_row_ref = c_row + best_shift_out * v_r_best
        c_col_ref = c_col + best_shift_out * v_c_best

        ref_p0 = (c_row_ref - (length_px / 2.0) * u_r_best, c_col_ref - (length_px / 2.0) * u_c_best)
        ref_p1 = (c_row_ref + (length_px / 2.0) * u_r_best, c_col_ref + (length_px / 2.0) * u_c_best)
    else:
        best_d_theta = 0.0
        best_shift_out = 0.0
        ref_p0 = p0
        ref_p1 = p1

    return best_d_theta, best_shift_out, ref_p0, ref_p1, refined_accepted, score_raw, score_refined


# ── Feature Extraction Core ───────────────────────────────────────────────────
def extract_candidate_features(
    row: pd.Series,
    data: np.ndarray,
    sub: np.ndarray,
    sub_sigma: np.ndarray,
    dist_sat: np.ndarray,
    star_fwhm_frame: float,
    sat_thresh: float,
    all_lines_map: dict,
) -> dict:
    h, w = data.shape
    row_type = str(row.get("type", "single")).lower()

    # 1. Axis Determination & Diagonal Signals
    if row_type == "line":
        eps = parse_endpoints(row.get("endpoints", None))
        if eps is None:
            raise ValueError(f"Line candidate missing valid endpoints: {row.get('endpoints')}")
        p0, p1 = eps
        axis_choice = "endpoints"
        sig_rise = np.nan
        sig_fall = np.nan
        d_theta_best = 0.0
        shift_best = 0.0
        p0_ref = p0
        p1_ref = p1
        refined_accepted = False
        score_raw = np.nan
        score_refined = np.nan
        length_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    else:
        r0 = float(row["bbox_r0"])
        c0 = float(row["bbox_c0"])
        r1 = float(row["bbox_r1"])
        c1 = float(row["bbox_c1"])
        c_row = (r0 + r1) / 2.0
        c_col = (c0 + c1) / 2.0

        diag_len = max(5, int(round(np.hypot(r1 - r0, c1 - c0))))
        t_samples = np.linspace(0.0, 1.0, diag_len)

        # Rising diagonal: (r0, c0) -> (r1, c1)
        r_rise = r0 + t_samples * (r1 - r0)
        c_rise = c0 + t_samples * (c1 - c0)
        vals_rise = map_coordinates(sub_sigma, [r_rise, c_rise], order=1, mode="constant", cval=0.0)
        sig_rise = float(np.mean(vals_rise))

        # Falling diagonal: (r1, c0) -> (r0, c1)
        r_fall = r1 + t_samples * (r0 - r1)
        c_fall = c0 + t_samples * (c1 - c0)
        vals_fall = map_coordinates(sub_sigma, [r_fall, c_fall], order=1, mode="constant", cval=0.0)
        sig_fall = float(np.mean(vals_fall))

        if sig_rise >= sig_fall:
            axis_choice = "rising"
            p0_init = (r0, c0)
            p1_init = (r1, c1)
        else:
            axis_choice = "falling"
            p0_init = (r1, c0)
            p1_init = (r0, c1)

        length_px = float(np.hypot(p1_init[0] - p0_init[0], p1_init[1] - p0_init[1]))
        if length_px < 1.0:
            length_px = 1.0

        # Axis Refinement for singles
        (
            d_theta_best,
            shift_best,
            p0_ref,
            p1_ref,
            refined_accepted,
            score_raw,
            score_refined,
        ) = refine_single_axis(
            sub_sigma, c_row, c_col, p0_init, p1_init, length_px
        )

    r0_ax, c0_ax = p0_ref
    r1_ax, c1_ax = p1_ref
    dr = r1_ax - r0_ax
    dc = c1_ax - c0_ax
    ax_len = float(np.hypot(dr, dc))
    if ax_len < 1.0:
        ax_len = 1.0
        dr = 1.0
        dc = 0.0

    ur = dr / ax_len
    uc = dc / ax_len
    vr = -uc
    vc = ur

    # 2. Angle from Vertical
    ang_h_deg = math.degrees(math.atan2(ur, uc))
    angle_from_vert = abs(90.0 - abs(ang_h_deg))
    if angle_from_vert > 90.0:
        angle_from_vert = 180.0 - angle_from_vert

    # 3. Min Edge Distance & CCD Edge Mask Region
    edge_d0 = min(r0_ax, h - 1 - r0_ax, c0_ax, w - 1 - c0_ax)
    edge_d1 = min(r1_ax, h - 1 - r1_ax, c1_ax, w - 1 - c1_ax)
    min_edge_dist = float(max(0.0, min(edge_d0, edge_d1)))
    x_over_3950 = bool(max(c0_ax, c1_ax) >= 3950.0)

    # 4. Saturated Fraction within 10 px of Refined Axis
    pad_b = 15
    b_r0 = max(0, int(math.floor(min(r0_ax, r1_ax) - pad_b)))
    b_r1 = min(h, int(math.ceil(max(r0_ax, r1_ax) + pad_b + 1)))
    b_c0 = max(0, int(math.floor(min(c0_ax, c1_ax) - pad_b)))
    b_c1 = min(w, int(math.ceil(max(c0_ax, c1_ax) + pad_b + 1)))

    grid_r, grid_c = np.indices((b_r1 - b_r0, b_c1 - b_c0))
    grid_r = grid_r + b_r0
    grid_c = grid_c + b_c0

    d_to_ax = dist_point_to_segment(grid_r, grid_c, r0_ax, c0_ax, r1_ax, c1_ax)
    within_10 = d_to_ax <= 10.0
    if np.any(within_10):
        sat_pix = data[grid_r[within_10], grid_c[within_10]] >= sat_thresh
        saturated_frac = float(np.mean(sat_pix))
    else:
        saturated_frac = 0.0

    # 5. Distance to Nearest Saturated Pixel
    n_sample_axis = max(10, int(round(ax_len)))
    ax_t = np.linspace(0.0, ax_len, n_sample_axis)
    ax_r = np.clip(np.round(r0_ax + ax_t * ur).astype(int), 0, h - 1)
    ax_c = np.clip(np.round(c0_ax + ax_t * uc).astype(int), 0, w - 1)
    dist_near_sat = float(np.min(dist_sat[ax_r, ax_c]))

    # 6. Transverse Profile & FWHM & Peak SNR
    # Extended to w in [-60, +60] px with baseline taken from |w| in [45, 60] px
    w_range = np.arange(-60.0, 61.0, 1.0)
    T_grid, W_grid = np.meshgrid(ax_t, w_range)
    R_grid = r0_ax + T_grid * ur + W_grid * vr
    C_grid = c0_ax + T_grid * uc + W_grid * vc

    trans_sig = map_coordinates(sub_sigma, [R_grid.ravel(), C_grid.ravel()],
                                order=1, mode="constant", cval=0.0).reshape(R_grid.shape)

    med_prof = np.median(trans_sig, axis=1)
    base_mask = (np.abs(w_range) >= 45.0) & (np.abs(w_range) <= 60.0)
    bg_wings = float(np.median(med_prof[base_mask]))
    net_prof = med_prof - bg_wings
    peak_snr = float(np.max(net_prof))

    wing_std = float(np.std(med_prof[base_mask]))
    if wing_std > 1e-9 and not np.isnan(wing_std):
        profile_snr_wing = float(peak_snr / wing_std)
    else:
        profile_snr_wing = np.nan

    # FWHM: NaN only when half-maximum crossing is not found on both sides
    pk_idx = int(np.argmax(net_prof))
    if peak_snr > 0.0:
        half_max = 0.5 * peak_snr
        left_below = np.where(net_prof[:pk_idx] <= half_max)[0]
        right_below = np.where(net_prof[pk_idx + 1:] <= half_max)[0]

        if len(left_below) > 0 and len(right_below) > 0:
            i_left = left_below[-1]
            y0_l, y1_l = net_prof[i_left], net_prof[i_left + 1]
            w0_l, w1_l = w_range[i_left], w_range[i_left + 1]
            t_l = (half_max - y0_l) / (y1_l - y0_l) if abs(y1_l - y0_l) > 1e-9 else 0.5
            w_left = w0_l + t_l * (w1_l - w0_l)

            i_r_off = right_below[0]
            i_right = pk_idx + 1 + i_r_off
            y0_r, y1_r = net_prof[i_right - 1], net_prof[i_right]
            w0_r, w1_r = w_range[i_right - 1], w_range[i_right]
            t_r = (half_max - y0_r) / (y1_r - y0_r) if abs(y1_r - y0_r) > 1e-9 else 0.5
            w_right = w0_r + t_r * (w1_r - w0_r)

            width_fwhm = float(w_right - w_left)
        else:
            width_fwhm = np.nan
    else:
        width_fwhm = np.nan

    width_vs_star = float(width_fwhm / star_fwhm_frame) if not np.isnan(width_fwhm) else np.nan

    # 7. Straightness along Refined Axis
    seg_step = 25.0
    n_segs = max(2, int(round(ax_len / seg_step)))
    seg_centers_t = (np.arange(n_segs) + 0.5) * (ax_len / n_segs)
    seg_offsets = []

    w_calc = np.arange(-20, 21, 1.0)
    for t_c in seg_centers_t:
        t_sub = np.linspace(max(0.0, t_c - seg_step / 2.0), min(ax_len, t_c + seg_step / 2.0), 5)
        T_s, W_s = np.meshgrid(t_sub, w_calc)
        R_s = r0_ax + T_s * ur + W_s * vr
        C_s = c0_ax + T_s * uc + W_s * vc
        vals_s = map_coordinates(sub, [R_s.ravel(), C_s.ravel()],
                                 order=1, mode="constant", cval=0.0).reshape(R_s.shape)
        p_1d = np.mean(vals_s, axis=1)
        base_s = np.percentile(p_1d, 20)
        pos_s = np.maximum(0.0, p_1d - base_s)
        sum_p = np.sum(pos_s)
        if sum_p > 1e-4:
            cen_w = float(np.sum(w_calc * pos_s) / sum_p)
            seg_offsets.append(cen_w)
        else:
            seg_offsets.append(0.0)

    seg_arr = np.array(seg_offsets)
    t_arr = seg_centers_t
    if len(seg_arr) >= 2 and np.std(t_arr) > 1e-4:
        fit_coeffs = np.polyfit(t_arr, seg_arr, deg=1)
        fit_vals = np.polyval(fit_coeffs, t_arr)
        straightness = float(np.sqrt(np.mean((seg_arr - fit_vals) ** 2)))
    else:
        straightness = 0.0

    # 8. Profile Uniformity along Refined Axis
    near_axis_mask = np.abs(w_range) <= 3.0
    seg_len_pts = max(1, len(ax_t) // n_segs)
    above_3sig_count = 0
    for s_i in range(n_segs):
        i0 = s_i * seg_len_pts
        i1 = min(len(ax_t), (s_i + 1) * seg_len_pts) if s_i < n_segs - 1 else len(ax_t)
        if i1 > i0:
            block = trans_sig[near_axis_mask, i0:i1]
            if np.max(block) >= 3.0:
                above_3sig_count += 1
    profile_uniformity = float(above_3sig_count / n_segs)

    # 9. Line-specific metadata: group_extent_px & n_segments
    if row_type == "line":
        f_stem = Path(str(row["frame_name"])).stem.lower()
        l_id_val = row.get("line_id", 1)
        try:
            l_id_int = int(float(l_id_val))
        except (ValueError, TypeError):
            l_id_int = 1

        all_l_info = all_lines_map.get((f_stem, l_id_int), {})
        group_extent_px = float(all_l_info.get("span_px", ax_len))

        n_seg_val = row.get("n_segments_z_gt_3", all_l_info.get("n_segments_z_gt_3", np.nan))
        try:
            n_segments = int(float(n_seg_val))
        except (ValueError, TypeError):
            n_segments = np.nan
    else:
        group_extent_px = np.nan
        n_segments = np.nan

    return {
        "axis_choice": axis_choice,
        "signal_rising": round(sig_rise, 4) if not np.isnan(sig_rise) else np.nan,
        "signal_falling": round(sig_fall, 4) if not np.isnan(sig_fall) else np.nan,
        "refined_accepted": bool(refined_accepted),
        "score_raw": round(score_raw, 2) if not np.isnan(score_raw) else np.nan,
        "score_refined": round(score_refined, 2) if not np.isnan(score_refined) else np.nan,
        "angle_offset_deg": round(d_theta_best, 2),
        "shift_px": round(shift_best, 1),
        "refined_r0": round(r0_ax, 2),
        "refined_c0": round(c0_ax, 2),
        "refined_r1": round(r1_ax, 2),
        "refined_c1": round(c1_ax, 2),
        "straightness": round(straightness, 3),
        "width_fwhm_px": round(width_fwhm, 2) if not np.isnan(width_fwhm) else np.nan,
        "width_vs_star": round(width_vs_star, 3) if not np.isnan(width_vs_star) else np.nan,
        "profile_uniformity": round(profile_uniformity, 3),
        "peak_snr": round(peak_snr, 2),
        "profile_snr_wing": round(profile_snr_wing, 2) if not np.isnan(profile_snr_wing) else np.nan,
        "pca_angle_from_vertical": round(angle_from_vert, 2),
        "min_edge_dist_px": round(min_edge_dist, 1),
        "x_over_3950": bool(x_over_3950),
        "saturated_frac": round(saturated_frac, 4),
        "dist_nearest_saturated_px": round(dist_near_sat, 1),
        "group_extent_px": round(group_extent_px, 1) if not np.isnan(group_extent_px) else np.nan,
        "n_segments": n_segments if not np.isnan(n_segments) else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("TASK D1 + D1a + D1b + D1c: Feature Extraction (Refined Axis Scoring & Wide-Band Baseline)")
    print("=" * 75)

    if not CANDIDATES_CSV.exists():
        print(f"Error: {CANDIDATES_CSV} not found.")
        sys.exit(1)

    df_cands = pd.read_csv(CANDIDATES_CSV)
    df_cands["frame_name"] = df_cands["frame_name"].apply(ensure_fits)
    print(f"Loaded {len(df_cands)} candidates across {df_cands['frame_name'].nunique()} frames.")

    # 1. Load label overrides
    overrides_map = {}
    if LABEL_OVERRIDES_CSV.exists():
        df_ov = pd.read_csv(LABEL_OVERRIDES_CSV)
        for _, ov_row in df_ov.iterrows():
            fn = ensure_fits(str(ov_row["frame_name"])).lower()
            cid = str(ov_row["candidate_id"]).strip()
            try:
                cid = str(int(float(cid)))
            except (ValueError, TypeError):
                pass
            overrides_map[(fn, cid)] = str(ov_row["new_label"]).strip().lower()
        print(f"Loaded {len(overrides_map)} label overrides from {LABEL_OVERRIDES_CSV.name}.")

    # 2. Load all_lines_v2.csv metadata
    all_lines_map = {}
    if ALL_LINES_CSV.exists():
        df_al = pd.read_csv(ALL_LINES_CSV)
        for _, al_row in df_al.iterrows():
            f_stem = str(al_row["frame"]).strip().lower()
            l_id = int(al_row["line_id"])
            all_lines_map[(f_stem, l_id)] = al_row.to_dict()

    # 3. Build FITS index
    print("Indexing FITS files...")
    fits_index = build_fits_index()
    print(f"Found {len(fits_index)} FITS files in search directories.")

    # 4. Process frame by frame
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_grouped = df_cands.groupby("frame_name", sort=False)

    feature_records = []
    total_frames = len(frames_grouped)
    star_fwhm_fallback_count = 0

    for f_idx, (frame_name, group) in enumerate(frames_grouped, 1):
        f_stem = Path(frame_name).stem.lower()
        fits_path = fits_index.get(f_stem)

        print(f"[{f_idx:02d}/{total_frames:02d}] {frame_name} ({len(group):>2} cands)... ", end="", flush=True)

        if fits_path is None or not fits_path.exists():
            print(f"FAILED (FITS not found)")
            for _, row in group.iterrows():
                row_type = str(row.get("type", "single")).lower()
                cand_id_raw = row["line_id"] if row_type == "line" else row["candidate_id"]
                cand_id = f"L{int(float(cand_id_raw))}" if row_type == "line" else str(int(float(cand_id_raw)))
                initial_label = str(row.get("verdict", "unlabeled")).strip().lower()
                final_label = overrides_map.get((frame_name.lower(), cand_id), initial_label)
                feature_records.append({
                    "cand_v2_id": row.get("cand_v2_id", f"{f_stem}_{cand_id}"),
                    "frame_name": frame_name,
                    "candidate_id": cand_id,
                    "kind": row_type,
                    "label": final_label,
                    "axis_choice": np.nan, "signal_rising": np.nan, "signal_falling": np.nan,
                    "refined_accepted": False, "score_raw": np.nan, "score_refined": np.nan,
                    "angle_offset_deg": np.nan, "shift_px": np.nan,
                    "refined_r0": np.nan, "refined_c0": np.nan, "refined_r1": np.nan, "refined_c1": np.nan,
                    "straightness": np.nan, "width_fwhm_px": np.nan, "width_vs_star": np.nan,
                    "profile_uniformity": np.nan, "peak_snr": np.nan, "profile_snr_wing": np.nan,
                    "pca_angle_from_vertical": np.nan, "min_edge_dist_px": np.nan,
                    "x_over_3950": np.nan, "saturated_frac": np.nan,
                    "dist_nearest_saturated_px": np.nan, "group_extent_px": np.nan,
                    "n_segments": np.nan,
                })
            continue

        try:
            data, header = load_fits(fits_path)
            bg, std = estimate_background(data, method="background2d")
            sub = data - bg
            std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, float(std))
            sub_sigma = sub / np.clip(std_map, 1e-6, None)

            # Saturation level determination
            sat_thresh, sat_source = get_frame_saturation_level(header)
            sat_mask = data >= sat_thresh
            n_sat_pixels = int(np.sum(sat_mask))

            if n_sat_pixels > 0:
                dist_sat = distance_transform_edt(~sat_mask)
            else:
                dist_sat = np.full_like(data, 10000.0, dtype=np.float32)

            # Star FWHM estimation & fallback tracking
            star_fwhm, is_fallback = estimate_frame_star_fwhm(data, sub_sigma, sat_thresh)
            if is_fallback:
                star_fwhm_fallback_count += 1

        except Exception as exc:
            print(f"FAILED (preproc: {exc})")
            continue

        c_ok = 0
        c_fail = 0
        for _, row in group.iterrows():
            row_type = str(row.get("type", "single")).lower()
            cand_id_raw = row["line_id"] if row_type == "line" else row["candidate_id"]
            try:
                cand_id = f"L{int(float(cand_id_raw))}" if row_type == "line" else str(int(float(cand_id_raw)))
            except (ValueError, TypeError):
                cand_id = str(cand_id_raw)

            initial_label = str(row.get("verdict", "unlabeled")).strip().lower()
            final_label = overrides_map.get((frame_name.lower(), cand_id), initial_label)

            try:
                feats = extract_candidate_features(
                    row, data, sub, sub_sigma, dist_sat, star_fwhm, sat_thresh, all_lines_map
                )
                c_ok += 1
            except Exception as exc:
                c_fail += 1
                feats = {
                    "axis_choice": np.nan, "signal_rising": np.nan, "signal_falling": np.nan,
                    "refined_accepted": False, "score_raw": np.nan, "score_refined": np.nan,
                    "angle_offset_deg": np.nan, "shift_px": np.nan,
                    "refined_r0": np.nan, "refined_c0": np.nan, "refined_r1": np.nan, "refined_c1": np.nan,
                    "straightness": np.nan, "width_fwhm_px": np.nan, "width_vs_star": np.nan,
                    "profile_uniformity": np.nan, "peak_snr": np.nan, "profile_snr_wing": np.nan,
                    "pca_angle_from_vertical": np.nan, "min_edge_dist_px": np.nan,
                    "x_over_3950": np.nan, "saturated_frac": np.nan,
                    "dist_nearest_saturated_px": np.nan, "group_extent_px": np.nan,
                    "n_segments": np.nan,
                }

            rec = {
                "cand_v2_id": row.get("cand_v2_id", f"{f_stem}_{cand_id}"),
                "frame_name": frame_name,
                "candidate_id": cand_id,
                "kind": row_type,
                "label": final_label,
            }
            rec.update(feats)
            feature_records.append(rec)

        fb_str = " (fallback)" if is_fallback else ""
        print(f"Sat: {sat_thresh:.0f} ({sat_source}, {n_sat_pixels} sat px) | Star: {star_fwhm:.2f}px{fb_str} | OK ({c_ok} ok, {c_fail} fail)")

    # 5. Output CSV
    df_out = pd.DataFrame(feature_records)
    export_cols = [
        "cand_v2_id", "frame_name", "candidate_id", "kind", "label",
        "axis_choice", "signal_rising", "signal_falling",
        "refined_accepted", "score_raw", "score_refined",
        "angle_offset_deg", "shift_px", "refined_r0", "refined_c0", "refined_r1", "refined_c1",
        "straightness", "width_fwhm_px", "width_vs_star", "profile_uniformity",
        "peak_snr", "profile_snr_wing", "pca_angle_from_vertical", "min_edge_dist_px", "x_over_3950",
        "saturated_frac", "dist_nearest_saturated_px", "group_extent_px", "n_segments",
    ]
    df_out[export_cols].to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 75)
    print(f"OUTPUT: Written {len(df_out)} rows to {OUTPUT_CSV}")
    print("=" * 75)

    # 6. Diagnostics & Reports
    # a) Star FWHM Fallback
    pct_fb = 100.0 * star_fwhm_fallback_count / max(1, total_frames)
    print(f"\nStar FWHM Fallback (6.79 px) used in: {star_fwhm_fallback_count} / {total_frames} frames ({pct_fb:.1f}%).")

    # b) Wide-Band Lines Diagnostics
    print("\nWide-Band Lines Verification (width_fwhm_px, profile_snr_wing, peak_snr):")
    print("-" * 80)
    for target_id in ["20230615-0006_L1", "20240502-0026_L1"]:
        match_row = df_out[df_out["cand_v2_id"] == target_id]
        if not match_row.empty:
            w_val = match_row.iloc[0]["width_fwhm_px"]
            snr_wing = match_row.iloc[0]["profile_snr_wing"]
            snr_val = match_row.iloc[0]["peak_snr"]
            w_star = match_row.iloc[0]["width_vs_star"]
            print(f"  {target_id:<18}: width_fwhm_px = {w_val:.2f} px, profile_snr_wing = {snr_wing:.2f}, peak_snr = {snr_val:.2f} sigma (width_vs_star = {w_star:.2f})")
        else:
            print(f"  {target_id:<18}: NOT FOUND")

    # c) Target 6 Confirmed Streak Singles Diagnostics
    target_6_ids = ["7", "8", "13", "6141", "11875", "12494"]
    match_6 = df_out[(df_out["kind"] == "single") & (df_out["candidate_id"].astype(str).isin(target_6_ids)) & (df_out["label"] == "streak")]
    print("\nTarget 6 Confirmed Streak Singles:")
    print("-" * 96)
    print(f"{'frame_name':<20} {'id':<8} {'angle_offset_deg':<18} {'shift_px':<10} {'accepted':<10} {'peak_snr':<10} {'snr_wing':<10} {'width_fwhm':<10}")
    print("-" * 96)
    for _, r in match_6.iterrows():
        w_str = f"{r['width_fwhm_px']:.2f}" if not pd.isna(r['width_fwhm_px']) else "NaN"
        pw_str = f"{r['profile_snr_wing']:.2f}" if not pd.isna(r['profile_snr_wing']) else "NaN"
        print(f"{r['frame_name']:<20} {str(r['candidate_id']):<8} {r['angle_offset_deg']:<18.2f} {r['shift_px']:<10.1f} {str(r['refined_accepted']):<10} {r['peak_snr']:<10.2f} {pw_str:<10} {w_str:<10}")

    # d) Confirmed Streak Singles Peak SNR < 1 Count
    streak_singles = df_out[(df_out["kind"] == "single") & (df_out["label"] == "streak")]
    low_snr_streak_singles = streak_singles[streak_singles["peak_snr"] < 1.0]
    print(f"\nConfirmed streak singles with peak_snr < 1: {len(low_snr_streak_singles)} / {len(streak_singles)}")

    # e) Singles Refinement Acceptance Count
    singles_df = df_out[df_out["kind"] == "single"]
    n_singles_acc = int(singles_df["refined_accepted"].sum())
    print(f"\nSingles with refined_accepted True: {n_singles_acc} / {len(singles_df)} ({100.0 * n_singles_acc / max(1, len(singles_df)):.1f}%)")

    # f) Singles with |shift_px| >= 39 px (Boundary Shifts)
    n_boundary_shift = int((singles_df["shift_px"].abs() >= 39.0).sum())
    print(f"\nSingles with |shift_px| >= 39 px: {n_boundary_shift} / {len(singles_df)}")

    # g) Singles Movement Diagnostics
    if not singles_df.empty:
        offsets = singles_df["angle_offset_deg"].dropna()
        moved_gt_05 = int(np.sum(np.abs(offsets) > 0.5))
        pct_moved = 100.0 * moved_gt_05 / len(offsets)
        print(f"\nSingles Axis Movement (|angle_offset_deg| > 0.5 deg): {moved_gt_05} / {len(offsets)} ({pct_moved:.1f}%)")

        print("\nDistribution of angle_offset_deg for singles:")
        print("-" * 50)
        print(f"  Min   : {offsets.min():+.2f} deg")
        print(f"  25%   : {offsets.quantile(0.25):+.2f} deg")
        print(f"  Median: {offsets.median():+.2f} deg")
        print(f"  75%   : {offsets.quantile(0.75):+.2f} deg")
        print(f"  Max   : {offsets.max():+.2f} deg")
        print(f"  Mean  : {offsets.mean():+.2f} deg ± {offsets.std():.2f} deg")

        bins = [-3.05, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.05]
        bin_labels = ["[-3.0, -2.0)", "[-2.0, -1.0)", "[-1.0, -0.5)", "[-0.5, +0.5]", "(+0.5, +1.0]", "(+1.0, +2.0]", "(+2.0, +3.0]"]
        counts, _ = np.histogram(offsets, bins=bins)
        print("\n  Binned Angle Offset Counts:")
        for b_label, c in zip(bin_labels, counts):
            print(f"    {b_label:<16}: {c:>4} ({100.0 * c / len(offsets):>5.1f}%)")

        # h) Singles NaN Width Count
        nan_singles_width = int(singles_df["width_fwhm_px"].isna().sum())
        pct_nan_w = 100.0 * nan_singles_width / max(1, len(singles_df))
        print(f"\nSingles with NaN width_fwhm_px: {nan_singles_width} / {len(singles_df)} ({pct_nan_w:.1f}%)")

    # 7. NaN Count per Feature
    feature_cols = [
        "straightness", "width_fwhm_px", "width_vs_star", "profile_uniformity",
        "peak_snr", "profile_snr_wing", "pca_angle_from_vertical", "min_edge_dist_px", "x_over_3950",
        "saturated_frac", "dist_nearest_saturated_px", "group_extent_px", "n_segments",
    ]

    print("\nNaN Count Per Feature:")
    print("-" * 45)
    for col in feature_cols:
        nan_count = df_out[col].isna().sum()
        pct = (nan_count / len(df_out)) * 100.0
        print(f"  {col:<26}: {nan_count:>4} ({pct:>5.1f}%)")

    # 8. Comparison Table: Mean and Std for streak vs fp
    print("\nFeature Summary: Mean ± Std (streak vs fp)")
    print("-" * 75)
    print(f"{'Feature':<26} | {'Streak (n=' + str((df_out['label']=='streak').sum()) + ')':<22} | {'FP (n=' + str((df_out['label']=='fp').sum()) + ')':<22}")
    print("-" * 75)

    df_streak = df_out[df_out["label"] == "streak"]
    df_fp = df_out[df_out["label"] == "fp"]

    for col in feature_cols:
        val_s = pd.to_numeric(df_streak[col], errors="coerce").dropna()
        val_f = pd.to_numeric(df_fp[col], errors="coerce").dropna()

        s_str = f"{val_s.mean():.2f} ± {val_s.std():.2f}" if len(val_s) > 1 else (f"{val_s.mean():.2f}" if len(val_s) == 1 else "N/A")
        f_str = f"{val_f.mean():.2f} ± {val_f.std():.2f}" if len(val_f) > 1 else (f"{val_f.mean():.2f}" if len(val_f) == 1 else "N/A")

        print(f"{col:<26} | {s_str:<22} | {f_str:<22}")
    print("-" * 75)


if __name__ == "__main__":
    main()
