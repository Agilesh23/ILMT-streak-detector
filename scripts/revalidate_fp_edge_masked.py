"""
Diagnostic Verification Script for Task C4f:
Re-validating 20230317-0008.fits as a False-Positive Baseline Frame with CCD Edge Artifact Masked (X >= 3950 Excluded).

Evaluates candidate pair SNR metrics for min_snr_floor=10.0 and min_snr_floor=12.0 when the vertical CCD edge
artifact (X >= 3950) is excluded from analysis.

Instructions for User:
Execute from PowerShell:
    python scripts/revalidate_fp_edge_masked.py
"""

from pathlib import Path
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d, map_coordinates

# Setup project path
script_dir = Path(__file__).resolve().parent
base_dir = script_dir.parent
src_dir = base_dir / "src"
sys.path.insert(0, str(src_dir))

from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates


def analyze_pair_masked(c1, c2, data, bg, std, edge_mask_x=3950.0, snr_threshold=3.0, max_pair_distance=3500.0, angle_tolerance_deg=15.0):
    y1, x1 = c1["centroid"]
    y2, x2 = c2["centroid"]

    # Filter out candidates sitting in the CCD edge artifact column (X >= 3950)
    if x1 >= edge_mask_x or x2 >= edge_mask_x:
        return None

    dy, dx = y2 - y1, x2 - x1
    dist = float(np.hypot(dy, dx))

    bbox1 = c1.get("bbox_full", c1.get("bbox_local", (0, 0, 1, 1)))
    bbox2 = c2.get("bbox_full", c2.get("bbox_local", (0, 0, 1, 1)))
    ang1 = float(np.degrees(np.arctan2(bbox1[2] - bbox1[0], bbox1[3] - bbox1[1])) % 180.0)
    ang2 = float(np.degrees(np.arctan2(bbox2[2] - bbox2[0], bbox2[3] - bbox2[1])) % 180.0)
    
    vec_ang = float(np.degrees(np.arctan2(dy, dx)) % 180.0)
    
    d1 = abs(vec_ang - ang1) % 180.0
    diff1 = min(d1, 180.0 - d1)
    d2 = abs(vec_ang - ang2) % 180.0
    diff2 = min(d2, 180.0 - d2)

    angle_pass = (diff1 <= angle_tolerance_deg) and (diff2 <= angle_tolerance_deg)
    dist_pass = (dist <= max_pair_distance)

    if dist < 10.0 or not (dist_pass and angle_pass):
        return None

    ux, uy = dx / dist, dy / dist
    nx, ny = -uy, ux

    kernel_length = 61
    kernel_width = 5

    n_pts = int(np.ceil(dist))
    t_steps = np.linspace(0, dist, n_pts)
    w_offsets = np.linspace(-kernel_width / 2.0, kernel_width / 2.0, kernel_width)

    sample_y = y1 + t_steps[:, None] * uy + w_offsets[None, :] * ny
    sample_x = x1 + t_steps[:, None] * ux + w_offsets[None, :] * nx

    # Exclude pairs whose sampling ray path passes through the edge artifact region (X >= 3950)
    if np.any(sample_x >= edge_mask_x):
        return None

    sub_img = data - bg
    std_img = std

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

    return {
        "c1_label": c1.get("label"), "c2_label": c2.get("label"),
        "dist": dist, "diff1": diff1, "diff2": diff2,
        "pct_above": pct_above, "min_snr": min_snr, "cov": cov, "median_snr": median_snr
    }


def evaluate_frame(f_path: Path, edge_mask_x=3950.0):
    print("--------------------------------------------------")
    print(f"Evaluating FP Frame: {f_path.name}")
    data, header = load_fits(f_path)
    bg, std = estimate_background(data, method="background2d")

    cands, _, _ = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0
    )

    unmasked_count = len(cands)
    masked_cands = [c for c in cands if c["centroid"][1] < edge_mask_x]
    masked_count = len(masked_cands)

    print(f"  Total Raw Blob Candidates:                {unmasked_count}")
    print(f"  Candidates After Masking Edge (X < 3950):  {masked_count} (Excluded {unmasked_count - masked_count} edge blobs)")

    pair_results = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            res = analyze_pair_masked(cands[i], cands[j], data, bg, std, edge_mask_x=edge_mask_x)
            if res is not None:
                pair_results.append(res)

    print(f"  Valid Inner Collinear Pairs (dist <= 3500px, angle <= 15°): {len(pair_results)}")

    if pair_results:
        min_snrs = [r['min_snr'] for r in pair_results]
        max_fp_snr = float(np.max(min_snrs))
        p95_fp_snr = float(np.percentile(min_snrs, 95))
        p90_fp_snr = float(np.percentile(min_snrs, 90))

        print(f"\n  --- Masked Metric Statistics for {f_path.name} ---")
        print(f"  Max FP Min-SNR Floor:          {max_fp_snr:.2f}")
        print(f"  95th Percentile Min-SNR:       {p95_fp_snr:.2f}")
        print(f"  90th Percentile Min-SNR:       {p90_fp_snr:.2f}")

        print("\n  --- Threshold Verification (Edge Artifact Masked) ---")
        for thresh in [10.0, 12.0]:
            violations = [r for r in pair_results if r['min_snr'] >= thresh]
            margin = thresh - max_fp_snr
            status = "PASSED (100% FP Rejection)" if len(violations) == 0 else f"FAILED ({len(violations)} FP accepted)"
            print(f"    min_snr_floor={thresh:<4.1f} | Max FP SNR={max_fp_snr:<5.2f} | Margin={f'+{margin:.2f}s':<7} | Accepted FP Pairs={len(violations)}/{len(pair_results)} | Status: {status}")
    else:
        print("  No collinear candidate pairs met geometric pre-filters in inner frame.")


def main():
    print("=== TASK C4f RE-VALIDATION WITH CCD EDGE ARTIFACT MASKED (X < 3950) ===\n")

    test_frames = [
        r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits",
        r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221025-0028.fits",
        r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20221026-0057.fits"
    ]

    for f_str in test_frames:
        f_path = Path(f_str)
        if f_path.exists():
            evaluate_frame(f_path, edge_mask_x=3950.0)


if __name__ == "__main__":
    main()
