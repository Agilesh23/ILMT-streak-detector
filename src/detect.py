"""
Candidate Detection and Radon Feature Extraction Module for ILMT Streak Detector.

Implements thresholding, binary dilation, regionprops shape filtering,
localized Radon sinogram feature extraction, and seeded fragment stitching.
"""

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, rotate as ndimage_rotate, map_coordinates, uniform_filter1d
from scipy.signal import fftconvolve
from skimage.measure import label, regionprops
from skimage.transform import radon


def find_blob_candidates(
    data: np.ndarray,
    bg: float | np.ndarray,
    std: float | np.ndarray,
    n_sigma: float = 3.0,
    min_aspect_ratio: float = 3.0,
    min_eccentricity: float = 0.9,
    max_area: float = 15000.0,
    dilation_radius: int = 2,
    y_offset: int = 0,
    edge_mask_max_x: float | None = None,
    border_margin_px: int | None = None,
    border_max_length_px: float | None = None,
    left_edge_max_col: float | None = None,
    left_edge_max_angle_from_vertical_deg: float | None = None,
    long_thin_bypass: dict | None = None,
    long_thin_bypass_enabled: bool = True,
    long_thin_bypass_min_major_px: float = 1000.0,
    long_thin_bypass_min_aspect: float = 30.0
) -> tuple[list[dict], int, int]:
    """
    Detect candidate elongated blobs in background-subtracted image data.

    Parameters
    ----------
    edge_mask_max_x : float or None
        If set, candidates with centroid X >= this value are excluded
        (masks vertical CCD boundary artifact).
    border_margin_px : int or None
        If set (along with border_max_length_px), blobs whose bbox falls
        within this many pixels of any frame edge AND whose major axis
        length < border_max_length_px are dropped.
    border_max_length_px : float or None
        Length threshold for border mask (see border_margin_px).
    left_edge_max_col : float or None
        If set, candidates whose bbox lies entirely within columns 0..left_edge_max_col
        AND whose orientation angle is within left_edge_max_angle_from_vertical_deg
        of vertical are dropped.
    left_edge_max_angle_from_vertical_deg : float or None
        Angular tolerance from vertical (90 deg) for left-edge glow filter.
    """
    h_img, w_img = data.shape
    sub = data - bg
    thresh_map = n_sigma * std
    binary = sub > thresh_map

    if dilation_radius > 0:
        struct = generate_binary_structure(2, 1)
        binary = binary_dilation(binary, structure=struct, iterations=dilation_radius)

    labeled = label(binary)
    props = regionprops(labeled, intensity_image=sub)

    count_before = len(props)
    candidates = []

    if long_thin_bypass is not None:
        bypass_enabled = long_thin_bypass.get("enabled", True)
        bypass_min_major_px = long_thin_bypass.get("min_major_px", 1000.0)
        bypass_min_aspect = long_thin_bypass.get("min_aspect", 30.0)
    else:
        bypass_enabled = long_thin_bypass_enabled
        bypass_min_major_px = long_thin_bypass_min_major_px
        bypass_min_aspect = long_thin_bypass_min_aspect

    for prop in props:
        major = prop.axis_major_length
        minor = prop.axis_minor_length
        ecc = prop.eccentricity
        aspect_ratio = major / minor if minor > 0 else 999.0

        area_ok = prop.area <= max_area or (
            bypass_enabled
            and major >= bypass_min_major_px
            and aspect_ratio >= bypass_min_aspect
        )

        if aspect_ratio >= min_aspect_ratio and ecc >= min_eccentricity and area_ok:
            cy_local, cx_local = prop.centroid
            min_row, min_col, max_row, max_col = prop.bbox

            # CCD edge artifact mask
            if edge_mask_max_x is not None and cx_local >= edge_mask_max_x:
                continue

            # Short-blob border mask: drop if bbox touches any frame edge
            # AND the blob is shorter than the length threshold.
            if border_margin_px is not None and border_max_length_px is not None:
                near_edge = (
                    min_row < border_margin_px or
                    min_col < border_margin_px or
                    max_row > h_img - border_margin_px or
                    max_col > w_img - border_margin_px
                )
                if near_edge and major < border_max_length_px:
                    continue

            # Left-edge glow artifact mask: drop candidate whose bbox lies entirely
            # within columns 0-left_edge_max_col and whose angle is within
            # left_edge_max_angle_from_vertical_deg of vertical.
            if left_edge_max_col is not None and left_edge_max_angle_from_vertical_deg is not None:
                if max_col <= left_edge_max_col:
                    dy = max_row - min_row
                    dx = max_col - min_col
                    ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
                    dev_from_vert = min(abs(ang - 90.0), 180.0 - abs(ang - 90.0))
                    if dev_from_vert <= left_edge_max_angle_from_vertical_deg:
                        continue

            candidates.append({
                "label": prop.label,
                "centroid": (float(cy_local + y_offset), float(cx_local)),
                "centroid_local": (float(cy_local), float(cx_local)),
                "bbox_full": (int(min_row + y_offset), int(min_col), int(max_row + y_offset), int(max_col)),
                "bbox_local": (int(min_row), int(min_col), int(max_row), int(max_col)),
                "area": prop.area,
                "eccentricity": float(ecc),
                "axis_major_length": float(major),
                "axis_minor_length": float(minor),
                "aspect_ratio": float(aspect_ratio),
                "mean_intensity": float(prop.intensity_mean),
                "max_intensity": float(prop.intensity_max),
            })

    count_after = len(candidates)
    return candidates, count_before, count_after


def compute_composite_score(cand: dict) -> float:
    """
    Compute a pre-Radon composite shape/intensity score S in [0, 100] for candidate ranking.
    """
    aspect = cand["aspect_ratio"]
    ecc = cand["eccentricity"]
    max_int = cand.get("max_intensity", 0.0)

    s_aspect = min(aspect / 10.0, 1.0)
    s_ecc = max(0.0, min((ecc - 0.9) / 0.1, 1.0))
    s_intensity = min(max_int / 100.0, 1.0) if max_int > 0 else 0.0

    score = (0.50 * s_aspect + 0.35 * s_ecc + 0.15 * s_intensity) * 100.0
    return float(score)


def rank_and_select_candidates(
    candidates: list[dict],
    min_abs_score: float = 47.0
) -> tuple[list[dict], float]:
    """
    Rank blob candidates using a pre-Radon composite shape/intensity score
    and select those meeting an absolute score floor.
    """
    if not candidates:
        return [], 0.0

    for c in candidates:
        c["composite_score"] = compute_composite_score(c)

    sorted_candidates = sorted(candidates, key=lambda c: c["composite_score"], reverse=True)

    selected = [c for c in sorted_candidates if c["composite_score"] >= min_abs_score]
    return selected, min_abs_score


def radon_features(cutout: np.ndarray, angle_offset_deg: float = 0.0) -> dict:
    """
    Compute localized Radon transform sinogram features on a rectified cutout patch.
    """
    if cutout.size == 0 or np.all(cutout == 0):
        return {
            "radon_peak": 0.0,
            "raw_angle_deg": 0.0,
            "peak_angle_deg": 0.0,
            "radon_snr": 0.0,
            "fwhm_rho": 0.0,
            "peak_rho_idx": 0,
        }

    theta = np.linspace(0.0, 180.0, 180, endpoint=False)
    sinogram = radon(cutout, theta=theta, circle=False)

    rho_idx, angle_idx = np.unravel_index(np.argmax(sinogram), sinogram.shape)
    peak_val = float(sinogram[rho_idx, angle_idx])
    raw_angle_deg = float(theta[angle_idx])

    flat_sino = sinogram.flatten()
    cutoff = np.percentile(flat_sino, 95)
    bg_vals = flat_sino[flat_sino < cutoff]
    sino_bg_mean = float(np.mean(bg_vals)) if len(bg_vals) > 0 else 0.0
    sino_bg_std = float(np.std(bg_vals)) if len(bg_vals) > 0 else 1.0
    radon_snr = (peak_val - sino_bg_mean) / sino_bg_std if sino_bg_std > 0 else 0.0

    rho_profile = sinogram[:, angle_idx]
    half_max = (peak_val + sino_bg_mean) / 2.0
    above_half = np.where(rho_profile >= half_max)[0]
    fwhm_rho = float(len(above_half)) if len(above_half) > 0 else 0.0

    calibrated_angle_deg = (raw_angle_deg + angle_offset_deg) % 180.0

    return {
        "radon_peak": peak_val,
        "raw_angle_deg": raw_angle_deg,
        "peak_angle_deg": calibrated_angle_deg,
        "radon_snr": float(radon_snr),
        "fwhm_rho": fwhm_rho,
        "peak_rho_idx": int(rho_idx),
    }


# ---------------------------------------------------------------------------
# Fragment Stitching (added post Phase-8 experiment validation)
# ---------------------------------------------------------------------------

def _check_pair_connected(
    data: np.ndarray,
    bg: np.ndarray,
    std: np.ndarray,
    cand1: dict,
    cand2: dict,
    snr_threshold: float = 3.0,
    min_connected_pct: float = 85.0,
    min_snr_floor: float = 20.0,
    max_cov: float = 2.0,
    kernel_length: int = 61,
    kernel_width: int = 5,
) -> bool:
    """
    Check if two candidates are bridged by a continuous linear streak signal.

    Uses fast 1D line sampling along the centroid-to-centroid path with 1D box
    filtering of size kernel_length across cross-section width kernel_width.
    """
    sub_img = data - bg
    std_img = std

    y1, x1 = cand1["centroid"]
    y2, x2 = cand2["centroid"]
    dy, dx = y2 - y1, x2 - x1
    dist = float(np.hypot(dy, dx))

    if dist < 10.0:
        return False

    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    n_pts = int(np.ceil(dist))
    t_steps = np.linspace(0, dist, n_pts)
    w_offsets = np.linspace(-kernel_width / 2.0, kernel_width / 2.0, kernel_width)

    sample_y = y1 + t_steps[:, None] * uy + w_offsets[None, :] * ny
    sample_x = x1 + t_steps[:, None] * ux + w_offsets[None, :] * nx

    coords = np.array([sample_y.ravel(), sample_x.ravel()])
    sampled_sub = map_coordinates(sub_img, coords, order=1, mode='constant', cval=0.0).reshape((n_pts, kernel_width))
    sampled_std = map_coordinates(std_img, coords, order=1, mode='nearest').reshape((n_pts, kernel_width))

    signal_1d = np.mean(sampled_sub, axis=1)
    std_1d = np.mean(sampled_std, axis=1)

    response_1d = uniform_filter1d(signal_1d, size=kernel_length, mode='nearest')
    gain_factor = np.sqrt(kernel_length * kernel_width)
    snr_profile = response_1d * gain_factor / np.maximum(std_1d, 1e-6)

    crop_start = int(0.10 * n_pts)
    crop_end = int(0.90 * n_pts)
    gap_snr = snr_profile[crop_start:crop_end] if crop_end > crop_start else snr_profile

    median_snr = float(np.median(gap_snr))
    min_snr = float(np.min(gap_snr))
    pct_above = float(np.mean(gap_snr >= snr_threshold) * 100.0)
    cov = float(np.std(gap_snr) / median_snr) if median_snr > 0 else 999.0

    return (
        pct_above >= min_connected_pct and
        min_snr >= min_snr_floor and
        cov <= max_cov
    )


def stitch_collinear_fragments(
    candidates: list[dict],
    data: np.ndarray,
    bg: np.ndarray,
    std: np.ndarray,
    angle_tolerance_deg: float = 15.0,
    min_connected_pct: float = 85.0,
    min_snr_floor: float = 20.0,
    max_cov: float = 2.0,
    max_pair_distance: float = 4500.0,
    snr_threshold: float = 3.0,
) -> list[dict]:
    """
    Merge candidate fragments that belong to the same physical streak.

    Algorithm:
    1. Build a connection graph: edge (i, j) if:
       (a) centroid distance <= max_pair_distance.
       (b) connecting-vector angle aligns with BOTH candidate i and j bbox orientations
           within angle_tolerance_deg (angular pre-filter).
       (c) _check_pair_connected() confirms continuous signal.
    2. Find connected components (union-find).
    3. Merge each multi-member component: expand bbox, update centroid,
       keep highest-scoring member as representative, tag 'stitched_from'.

    Parameters
    ----------
    candidates : list[dict]
        Output of find_blob_candidates or rank_and_select_candidates.
    data, bg, std : 2D float32 arrays
        Full-frame image, background, and noise maps.
    angle_tolerance_deg : float, default 15.0 (config: stitching.angle_tolerance_deg)
        Max angular deviation between connecting vector and candidate bbox angle.
    min_connected_pct : float, default 85.0 (config: stitching.min_connected_pct)
        Minimum % of path samples above snr_threshold.
    min_snr_floor : float, default 20.0 (config: stitching.min_snr_floor)
        Minimum path-minimum SNR. Primary FP discriminator.
    max_cov : float, default 2.0 (config: stitching.max_cov)
        Maximum coefficient of variation of path SNR profile.
    max_pair_distance : float, default 4500.0 (config: stitching.max_pair_distance)
        Maximum centroid distance (px) to evaluate a pair.
    snr_threshold : float, default 3.0
        Per-sample threshold for coverage counting.

    Returns
    -------
    list[dict]
        Candidate list with stitched fragments merged. Singletons unchanged.
    """
    if len(candidates) < 2:
        return candidates

    for c in candidates:
        if "composite_score" not in c:
            c["composite_score"] = compute_composite_score(c)

    n = len(candidates)

    def _bbox_angle(c: dict) -> float:
        bbox = c.get("bbox_full", c.get("bbox_local", (0, 0, 1, 1)))
        return float(np.degrees(np.arctan2(bbox[2] - bbox[0], bbox[3] - bbox[1])) % 180.0)

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = candidates[i], candidates[j]
            y1, x1 = ci["centroid"]
            y2, x2 = cj["centroid"]
            dist = float(np.hypot(y2 - y1, x2 - x1))
            if dist < 10.0 or dist > max_pair_distance:
                continue

            conn_angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0)
            
            # Pre-filter: BOTH candidates must align with connecting angle within tolerance
            if (_angle_diff(conn_angle, _bbox_angle(ci)) > angle_tolerance_deg or
                    _angle_diff(conn_angle, _bbox_angle(cj)) > angle_tolerance_deg):
                continue

            if _check_pair_connected(
                data, bg, std, ci, cj,
                snr_threshold=snr_threshold,
                min_connected_pct=min_connected_pct,
                min_snr_floor=min_snr_floor,
                max_cov=max_cov,
            ):
                union(i, j)

    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    merged: list[dict] = []
    for root, idxs in groups.items():
        if len(idxs) == 1:
            merged.append(candidates[idxs[0]])
            continue

        group_cands = [candidates[i] for i in idxs]
        rep = max(group_cands, key=lambda c: c["composite_score"])

        all_bboxes = [c.get("bbox_full", c.get("bbox_local")) for c in group_cands]
        my0 = min(b[0] for b in all_bboxes)
        mx0 = min(b[1] for b in all_bboxes)
        my1 = max(b[2] for b in all_bboxes)
        mx1 = max(b[3] for b in all_bboxes)

        merged_cand = dict(rep)
        merged_cand["centroid"] = ((my0 + my1) / 2.0, (mx0 + mx1) / 2.0)
        merged_cand["bbox_full"] = (my0, mx0, my1, mx1)
        merged_cand["stitched_from"] = [c["label"] for c in group_cands]
        merged_cand["n_fragments"] = len(group_cands)
        merged_cand["axis_major_length"] = float(np.hypot(my1 - my0, mx1 - mx0))

        merged.append(merged_cand)

    return merged


def grow_streak(
    candidates: list[dict],
    data: np.ndarray,
    bg: float | np.ndarray,
    std: float | np.ndarray,
    step_px: float = 5.0,
    half_width_px: float = 4.0,
    min_snr: float = 1.5,
    max_gap_px: float = 50.0,
    edge_mask_max_x: float = 3950.0,
    corridor_angle_tol_deg: float = 15.0,
    corridor_max_offset_px: float = 25.0
) -> list[dict]:
    """
    Task C10: Grow candidates along their fitted lines and perform corridor merging.

    For each seed candidate:
      - Fits a line and marches bidirectionally in step_px increments.
      - At each step, evaluates median signal across streak width (ignoring star spikes).
      - Halts when SNR stays below min_snr for > max_gap_px, or hits image boundaries.
      - Halts at edge_mask_max_x and flags reaches_edge_strip = True.

    Corridor merge:
      - Merges collinear candidates lying within corridor_angle_tol_deg and corridor_max_offset_px.
      - Records total_length, gap_length, gap_mean_snr, mean_full_snr, and reaches_edge_strip.
    """
    if not candidates:
        return []

    sub_sigma = (data - bg) / np.maximum(std, 1e-4)
    h, w = data.shape
    transverse_offsets = np.arange(-half_width_px, half_width_px + 1.0, 1.0)

    # 1. Grow each candidate along its fitted line
    grown_cands = []
    for cand in candidates:
        cy, cx = cand["centroid"]
        min_r, min_c, max_r, max_c = cand.get("bbox_full", cand.get("bbox_local"))
        dy, dx = max_r - min_r, max_c - min_c

        orientation = cand.get("orientation", None)
        if orientation is not None:
            angle_rad = orientation
        else:
            angle_rad = np.arctan2(dy, dx)

        ux = np.cos(angle_rad)
        uy = np.sin(angle_rad)
        perp_ux, perp_uy = -uy, ux

        p_pts = np.array([[min_c, min_r], [max_c, min_r], [min_c, max_r], [max_c, max_r]])
        t_projs = [(p[0] - cx) * ux + (p[1] - cy) * uy for p in p_pts]
        seed_t_min, seed_t_max = min(t_projs), max(t_projs)

        reaches_edge_strip = False

        def march(t_start: float, direction: float) -> float:
            nonlocal reaches_edge_strip
            curr_t = t_start
            consec_low = 0.0
            best_t = curr_t

            while True:
                curr_t += direction * step_px
                xc = cx + curr_t * ux
                yc = cy + curr_t * uy

                if xc < 0 or xc >= w or yc < 0 or yc >= h:
                    break
                if xc >= edge_mask_max_x:
                    reaches_edge_strip = True
                    t_cap = (edge_mask_max_x - cx) / ux if abs(ux) > 1e-6 else curr_t
                    best_t = t_cap
                    break

                xs = xc + transverse_offsets * perp_ux
                ys = yc + transverse_offsets * perp_uy
                vals = map_coordinates(sub_sigma, np.vstack([ys, xs]), order=1, mode='constant', cval=0.0)
                step_med = float(np.median(vals))

                if step_med >= min_snr:
                    consec_low = 0.0
                    best_t = curr_t
                else:
                    consec_low += step_px
                    if consec_low > max_gap_px:
                        break
            return best_t

        final_t_max = march(seed_t_max, +1.0)
        final_t_min = march(seed_t_min, -1.0)

        p0 = (cx + final_t_min * ux, cy + final_t_min * uy)
        p1 = (cx + final_t_max * ux, cy + final_t_max * uy)
        g_len = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))

        grown_cands.append({
            "cand_id": cand["label"],
            "seed_cand": cand,
            "cx": cx,
            "cy": cy,
            "ux": ux,
            "uy": uy,
            "angle_rad": angle_rad,
            "t_min": final_t_min,
            "t_max": final_t_max,
            "p0": p0,
            "p1": p1,
            "grown_length": g_len,
            "reaches_edge_strip": reaches_edge_strip
        })

    # 2. Corridor merge among candidates
    n = len(grown_cands)
    parent = list(range(n))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = grown_cands[i], grown_cands[j]
            diff = abs(c1["angle_rad"] - c2["angle_rad"]) % np.pi
            diff_deg = min(diff, np.pi - diff) * 180.0 / np.pi
            if diff_deg > corridor_angle_tol_deg:
                continue

            dx = c2["cx"] - c1["cx"]
            dy = c2["cy"] - c1["cy"]
            dist_pair = np.hypot(dx, dy)

            p_ang = np.arctan2(dy, dx)
            p_diff1 = min(abs(c1["angle_rad"] - p_ang) % np.pi, np.pi - (abs(c1["angle_rad"] - p_ang) % np.pi)) * 180.0 / np.pi
            p_diff2 = min(abs(c2["angle_rad"] - p_ang) % np.pi, np.pi - (abs(c2["angle_rad"] - p_ang) % np.pi)) * 180.0 / np.pi

            perp_dist = abs(dx * (-c1["uy"]) + dy * c1["ux"])

            is_collinear = (p_diff1 <= corridor_angle_tol_deg and p_diff2 <= corridor_angle_tol_deg) or (perp_dist <= corridor_max_offset_px and dist_pair < 300.0)
            if is_collinear:
                union(i, j)

    from collections import defaultdict
    clusters: dict[int, list[dict]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(grown_cands[i])

    merged_streaks = []
    for r, group in clusters.items():
        if len(group) == 1:
            g = group[0]
            cx, cy = g["cx"], g["cy"]
            ux, uy = g["ux"], g["uy"]
            perp_ux, perp_uy = -uy, ux
            steps = np.arange(g["t_min"], g["t_max"], step_px)
            snrs = []
            for t in steps:
                xc = cx + t * ux
                yc = cy + t * uy
                vals = map_coordinates(sub_sigma, np.vstack([yc + transverse_offsets * perp_uy, xc + transverse_offsets * perp_ux]), order=1, mode='constant', cval=0.0)
                snrs.append(float(np.median(vals)))
            mean_snr = float(np.mean(snrs)) if snrs else 0.0

            merged_cand = dict(g["seed_cand"])
            merged_cand["total_length"] = round(g["grown_length"], 1)
            merged_cand["gap_length"] = 0.0
            merged_cand["gap_mean_snr"] = 0.0
            merged_cand["mean_full_snr"] = round(mean_snr, 2)
            merged_cand["reaches_edge_strip"] = g["reaches_edge_strip"]
            merged_cand["p_start"] = g["p0"]
            merged_cand["p_end"] = g["p1"]
            merged_cand["stitched_from"] = [g["cand_id"]]
            merged_streaks.append(merged_cand)
            continue

        rep = max(group, key=lambda c: c["grown_length"])
        cx_rep, cy_rep = rep["cx"], rep["cy"]

        group_centers = np.array([[g["cx"], g["cy"]] for g in group])
        if len(group_centers) >= 2:
            c_mean = np.mean(group_centers, axis=0)
            _, _, vh = np.linalg.svd(group_centers - c_mean)
            ux_rep, uy_rep = vh[0]
            if ux_rep < 0:
                ux_rep, uy_rep = -ux_rep, -uy_rep
        else:
            ux_rep, uy_rep = rep["ux"], rep["uy"]

        perp_ux, perp_uy = -uy_rep, ux_rep

        all_t_starts = []
        all_t_ends = []
        seg_intervals = []
        for g in group:
            p0 = g["p0"]
            p1 = g["p1"]
            t0 = (p0[0] - cx_rep) * ux_rep + (p0[1] - cy_rep) * uy_rep
            t1 = (p1[0] - cx_rep) * ux_rep + (p1[1] - cy_rep) * uy_rep
            s_min, s_max = min(t0, t1), max(t0, t1)
            all_t_starts.append(s_min)
            all_t_ends.append(s_max)
            seg_intervals.append((s_min, s_max))

        merged_t_min = min(all_t_starts)
        merged_t_max = max(all_t_ends)

        p_start = (cx_rep + merged_t_min * ux_rep, cy_rep + merged_t_min * uy_rep)
        p_end = (cx_rep + merged_t_max * ux_rep, cy_rep + merged_t_max * uy_rep)

        reaches_strip = any(g["reaches_edge_strip"] for g in group)
        if p_end[0] >= edge_mask_max_x:
            reaches_strip = True
            t_cap = (edge_mask_max_x - cx_rep) / ux_rep if abs(ux_rep) > 1e-6 else merged_t_max
            p_end = (edge_mask_max_x, cy_rep + t_cap * uy_rep)

        total_len = float(np.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))

        seg_intervals = sorted(seg_intervals, key=lambda x: x[0])
        merged_intervals: list[list[float]] = []
        for s0, s1 in seg_intervals:
            if not merged_intervals:
                merged_intervals.append([s0, s1])
            else:
                if s0 <= merged_intervals[-1][1]:
                    merged_intervals[-1][1] = max(merged_intervals[-1][1], s1)
                else:
                    merged_intervals.append([s0, s1])

        gap_intervals = []
        for k in range(len(merged_intervals) - 1):
            gap_intervals.append((merged_intervals[k][1], merged_intervals[k+1][0]))

        gap_len = sum(ge - gs for gs, ge in gap_intervals)

        gap_snrs = []
        for gs, ge in gap_intervals:
            gap_steps = np.arange(gs, ge, step_px)
            for gt in gap_steps:
                g_xc = cx_rep + gt * ux_rep
                g_yc = cy_rep + gt * uy_rep
                vals = map_coordinates(sub_sigma, np.vstack([g_yc + transverse_offsets * perp_uy, g_xc + transverse_offsets * perp_ux]), order=1, mode='constant', cval=0.0)
                gap_snrs.append(float(np.median(vals)))
        mean_gap_snr = float(np.mean(gap_snrs)) if gap_snrs else 0.0

        full_steps = np.arange(merged_t_min, merged_t_max, step_px)
        full_snrs = []
        for ft in full_steps:
            f_xc = cx_rep + ft * ux_rep
            f_yc = cy_rep + ft * uy_rep
            vals = map_coordinates(sub_sigma, np.vstack([f_yc + transverse_offsets * perp_uy, f_xc + transverse_offsets * perp_ux]), order=1, mode='constant', cval=0.0)
            full_snrs.append(float(np.median(vals)))
        mean_full_snr = float(np.mean(full_snrs)) if full_snrs else 0.0

        merged_cand = dict(rep["seed_cand"])
        merged_cand["total_length"] = round(total_len, 1)
        merged_cand["gap_length"] = round(gap_len, 1)
        merged_cand["gap_mean_snr"] = round(mean_gap_snr, 2)
        merged_cand["mean_full_snr"] = round(mean_full_snr, 2)
        merged_cand["reaches_edge_strip"] = reaches_strip
        merged_cand["p_start"] = p_start
        merged_cand["p_end"] = p_end
        merged_cand["stitched_from"] = [g["cand_id"] for g in group]
        merged_streaks.append(merged_cand)

    return merged_streaks

