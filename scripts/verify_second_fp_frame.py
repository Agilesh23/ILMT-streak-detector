"""
Verification Script for Task C4c: Testing Proposed Thresholds (min_snr_floor=10.0 and 12.0)
Against Second Confirmed False-Positive Frame (20230317-0008.fits).

PROVENANCE OF FP FRAME 20230317-0008.fits:
Confirmed in data/labeled/batch_1_labels.csv. All candidates detected in this frame
(IDs 66, 3984, 10204) were manually audited and labeled as label=0 (False Positives:
"not a streak a random patch", "gradient along with noise").

Instructions for User:
Execute from PowerShell:
    python scripts/verify_second_fp_frame.py
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


def analyze_pair(c1, c2, data, bg, std, snr_threshold=3.0, max_pair_distance=3500.0, angle_tolerance_deg=15.0):
    sub_img = data - bg
    std_img = std

    y1, x1 = c1["centroid"]
    y2, x2 = c2["centroid"]
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
        "dist": dist, "diff1": diff1, "diff2": diff2,
        "pct_above": pct_above, "min_snr": min_snr, "cov": cov, "median_snr": median_snr
    }


def main():
    fp_frame_path = base_dir.parent / "ILMT_Contest_Materials" / "test_files" / "20230317-0008.fits"
    if not fp_frame_path.exists():
        fp_frame_path = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files\20230317-0008.fits")

    print(f"=== TASK C4c SECOND FP FRAME THRESHOLD VERIFICATION ===")
    print(f"Frame Path: {fp_frame_path}")
    print(f"Provenance: Confirmed FP Frame (batch_1_labels.csv: candidates 66, 3984, 10204 all label=0)\n")

    data, header = load_fits(fp_frame_path)
    bg, std = estimate_background(data, method="background2d")

    cands, _, _ = find_blob_candidates(
        data, bg, std,
        n_sigma=3.0,
        min_aspect_ratio=3.0,
        min_eccentricity=0.9,
        max_area=15000.0
    )

    print(f"Shape-Filtered Candidates Detected: {len(cands)}")

    pairs_tested = 0
    pair_metrics = []

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            res = analyze_pair(cands[i], cands[j], data, bg, std)
            if res is not None:
                pairs_tested += 1
                pair_metrics.append(res)

    print(f"Collinear Candidate Pairs Tested (dist <= 3500px, angle <= 15°): {pairs_tested}\n")

    candidate_thresholds = [10.0, 12.0]

    if pair_metrics:
        min_snrs = [r['min_snr'] for r in pair_metrics]
        max_fp_snr = float(np.max(min_snrs))
        p95_fp_snr = float(np.percentile(min_snrs, 95))
        p90_fp_snr = float(np.percentile(min_snrs, 90))

        print(f"--- Measured SNR Distribution for {fp_frame_path.name} ---")
        print(f"Max Measured Min-SNR Floor:          {max_fp_snr:.2f}")
        print(f"95th Percentile Min-SNR Floor:       {p95_fp_snr:.2f}")
        print(f"90th Percentile Min-SNR Floor:       {p90_fp_snr:.2f}\n")

        print("--- Candidate Threshold Evaluation Summary ---")
        print(f"{'Threshold (min_snr_floor)':<25} | {'Max FP SNR':<12} | {'Safety Margin':<15} | {'FP Pairs Accepted':<20} | {'Status'}")
        print("-" * 95)

        for thresh in candidate_thresholds:
            violations = [r for r in pair_metrics if r['min_snr'] >= thresh]
            margin = thresh - max_fp_snr
            status = "PASSED (100% FP Rejection)" if len(violations) == 0 else f"FAILED ({len(violations)} FP accepted)"
            print(f"{thresh:<25.1f} | {max_fp_snr:<12.2f} | {f'+{margin:.2f} sigma':<15} | {f'{len(violations)} / {pairs_tested}':<20} | {status}")

    else:
        print("No collinear candidate pairs met initial distance and angle pre-filters in this frame.")


if __name__ == "__main__":
    main()
