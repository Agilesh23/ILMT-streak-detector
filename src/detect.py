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
    edge_mask_max_x: float | None = None
) -> tuple[list[dict], int, int]:
    """
    Detect candidate elongated blobs in background-subtracted image data.

    Parameters
    ----------
    edge_mask_max_x : float or None
        If set, candidates with centroid X >= this value are excluded
        (masks vertical CCD boundary artifact).
    """
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

    for prop in props:
        major = prop.axis_major_length
        minor = prop.axis_minor_length
        ecc = prop.eccentricity
        aspect_ratio = major / minor if minor > 0 else 999.0

        if aspect_ratio >= min_aspect_ratio and ecc >= min_eccentricity and prop.area <= max_area:
            cy_local, cx_local = prop.centroid
            min_row, min_col, max_row, max_col = prop.bbox

            # CCD edge artifact mask
            if edge_mask_max_x is not None and cx_local >= edge_mask_max_x:
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
    top_n: int = 3,
    score_margin: float = 0.15
) -> tuple[list[dict], float]:
    """
    Rank blob candidates using a pre-Radon shape/intensity composite score and apply a soft cutoff.
    """
    if not candidates:
        return [], 0.0

    for c in candidates:
        c["composite_score"] = compute_composite_score(c)

    sorted_candidates = sorted(candidates, key=lambda c: c["composite_score"], reverse=True)

    if len(sorted_candidates) <= top_n:
        return sorted_candidates, 0.0

    nth_score = sorted_candidates[top_n - 1]["composite_score"]
    cutoff_score = float(nth_score * (1.0 - score_margin))

    selected = [c for c in sorted_candidates if c["composite_score"] >= cutoff_score]
    return selected, cutoff_score


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
