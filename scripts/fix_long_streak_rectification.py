"""
Fix Cutout Drift for Long Stitched Candidates via Weighted PCA/TLS Line Fitting.

Refits the streak line through all constituent fragment centroids rather than
relying on a simple 2-point bounding box endpoint vector.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup base_dir relative to this script file
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import extract_streak_patch


def fit_weighted_tls_line(centroids: list[tuple[float, float]], weights: list[float]) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Fit a Total Least Squares (TLS) / Weighted Principal Component line to points.

    JUSTIFICATION FOR WEIGHTED PCA / TLS OVER POLYNOMIAL OLS:
    Standard ordinary least squares (e.g. np.polyfit y = m*x + c) fails on vertical or
    near-vertical streaks because the slope m = dy/dx approaches infinity (singularity).
    Weighted PCA / TLS fits a line by finding the principal eigenvector of the 2D weighted
    covariance matrix of (x, y) coordinates. This is rotationally invariant, handles
    arbitrary streak orientations seamlessly, and minimizes orthogonal distance from
    centroids to the fitted line. Weighting by fragment area ensures larger, well-defined
    fragments dominate the line orientation over tiny noise blobs.

    Parameters
    ----------
    centroids : list[tuple[float, float]]
        List of (x, y) centroid coordinates.
    weights : list[float]
        List of non-negative scalar weights (e.g. fragment area or major axis length).

    Returns
    -------
    center, direction, angle_deg : tuple
        - center: (center_x, center_y) weighted centroid
        - direction: (vx, vy) unit direction vector along streak
        - angle_deg: angle of line in degrees in [0, 180)
    """
    pts = np.array(centroids, dtype=np.float64)  # shape (N, 2): column 0 is X, column 1 is Y
    w = np.array(weights, dtype=np.float64)
    w_sum = np.sum(w)

    if w_sum <= 0:
        w = np.ones(len(pts), dtype=np.float64)
        w_sum = np.sum(w)

    # Weighted mean centroid
    center_x = float(np.sum(w * pts[:, 0]) / w_sum)
    center_y = float(np.sum(w * pts[:, 1]) / w_sum)
    center = (center_x, center_y)

    # Center coordinates
    dx = pts[:, 0] - center_x
    dy = pts[:, 1] - center_y

    # Weighted 2x2 covariance matrix
    cov_xx = np.sum(w * dx * dx) / w_sum
    cov_xy = np.sum(w * dx * dy) / w_sum
    cov_yy = np.sum(w * dy * dy) / w_sum

    cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

    # Principal Component Analysis (eigen-decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # Principal direction is eigenvector for largest eigenvalue
    primary_vector = eigenvectors[:, np.argmax(eigenvalues)]

    vx, vy = float(primary_vector[0]), float(primary_vector[1])

    # Standardize direction so angle is in [0, 180)
    angle_rad = np.arctan2(vy, vx)
    angle_deg = float(np.degrees(angle_rad) % 180.0)

    # Unit direction vector aligned with angle_deg
    rad_norm = np.radians(angle_deg)
    direction = (float(np.cos(rad_norm)), float(np.sin(rad_norm)))

    return center, direction, angle_deg


def main():
    frame_path = r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230519-0009.fits"
    print(f"Loading frame: {frame_path}")

    # Pipeline processing
    data, header = load_fits(frame_path)
    bg, std = estimate_background(data, method="background2d")

    unstitched_candidates, count_before, count_shape = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0
    )

    # Map unstitched candidates by label for quick lookup
    cand_by_label = {c["label"]: c for c in unstitched_candidates}

    stitched_candidates = stitch_collinear_fragments(unstitched_candidates, data, bg, std)
    selected_candidates, cutoff_score = rank_and_select_candidates(stitched_candidates)

    # Target long stitched candidate 3673 (or largest candidate in selected set)
    target_cand = None
    for c in selected_candidates:
        if c.get("label") == 3673 or c.get("n_fragments", 1) > 1:
            target_cand = c
            break

    if target_cand is None:
        target_cand = max(selected_candidates, key=lambda c: c.get("axis_major_length", 0.0))

    cand_id = target_cand.get("label")
    merged_labels = target_cand.get("stitched_from", [cand_id])
    n_fragments = len(merged_labels)

    cy, cx = target_cand["centroid"]
    min_row, min_col, max_row, max_col = target_cand.get("bbox_full", target_cand.get("bbox_local"))

    # Old 2-point angle calculation
    old_dy = max_row - min_row
    old_dx = max_col - min_col
    old_angle_deg = float(np.degrees(np.arctan2(old_dy, old_dx)) % 180.0)

    # Gather all fragment centroids and areas for refit
    fragment_centroids = []
    fragment_weights = []

    for lbl in merged_labels:
        if lbl in cand_by_label:
            frag = cand_by_label[lbl]
            f_cy, f_cx = frag["centroid"]
            f_area = frag.get("area", 1.0)
            fragment_centroids.append((f_cx, f_cy))
            fragment_weights.append(f_area)
        else:
            fragment_centroids.append((cx, cy))
            fragment_weights.append(target_cand.get("area", 1.0))

    # Refit line using Weighted TLS/PCA if stitched (>1 fragments)
    if n_fragments > 1 and len(fragment_centroids) > 1:
        center, (vx, vy), new_angle_deg = fit_weighted_tls_line(fragment_centroids, fragment_weights)
        refit_center_x, refit_center_y = center
    else:
        refit_center_x, refit_center_y = cx, cy
        new_angle_deg = old_angle_deg
        rad_old = np.radians(old_angle_deg)
        vx, vy = np.cos(rad_old), np.sin(rad_old)

    detected_length = target_cand.get("axis_major_length", float(np.hypot(old_dy, old_dx)))
    target_length = 2.0 * detected_length + 80.0
    half_len = target_length / 2.0

    # Compute new extrapolated endpoints along refit direction
    start_pt = (refit_center_x - half_len * vx, refit_center_y - half_len * vy)
    end_pt = (refit_center_x + half_len * vx, refit_center_y + half_len * vy)

    # Extract corrected cutout patch
    patch_v2, _ = extract_streak_patch(data, start_pt, end_pt, box_width=100, output_length=int(target_length))

    # Save v2 cutout image
    output_dir = base_dir / "outputs" / "labeling_batch_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"20230519-0009_cand_{cand_id}_cutout_v2.png"

    p_vmin, p_vmax = float(np.percentile(patch_v2, 1)), float(np.percentile(patch_v2, 99))
    if p_vmax <= p_vmin:
        p_vmin, p_vmax = float(np.min(patch_v2)), float(np.max(patch_v2))
    if p_vmax <= p_vmin:
        p_vmin, p_vmax = 0.0, 1.0

    plt.imsave(str(out_path), patch_v2, cmap="gray", origin="lower", vmin=p_vmin, vmax=p_vmax)

    print("\n=== RECTIFICATION FIX COMPARISON ===")
    print(f"Target Candidate ID:                   {cand_id}")
    print(f"Number of Merged Fragments Used:       {len(fragment_centroids)} (Labels: {merged_labels})")
    print(f"Original 2-Point Line Angle:           {old_angle_deg:.4f} degrees")
    print(f"Refit Weighted TLS Line Angle:         {new_angle_deg:.4f} degrees")
    print(f"Angular Difference:                    {abs(new_angle_deg - old_angle_deg):.4f} degrees")
    print(f"Saved Refit Cutout V2 Image:           {out_path}")


if __name__ == "__main__":
    main()
