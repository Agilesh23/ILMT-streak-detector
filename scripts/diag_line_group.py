# TASK D0b-diag: line-based grouping test
"""
Diagnostic script for line-based candidate grouping via sequential RANSAC on raw candidate centroids.
Evaluates candidate grouping on Set A (known multi-fragment streak frames) and Set B (negative control frames).
"""

import os
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Top-level algorithm constants
PERP_TOL_PX = 12
MIN_INLIERS = 5
MIN_SPAN_PX = 800
MAX_LINES = 3
RANSAC_ITERS = 2000

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates
from src.io_utils import load_fits


def find_lines_ransac(candidates: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Find straight lines through candidate centroids using sequential RANSAC,
    followed by PCA orthogonal distance regression refitting to inliers.
    """
    if len(candidates) < MIN_INLIERS:
        return []

    coords = np.array([c["centroid"] for c in candidates], dtype=np.float64)  # (N, 2) [row=y, col=x]
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

            # Perpendicular distance to line through p1: |(y - y1)*ux - (x - x1)*uy|
            diff_y = pts[:, 0] - p1[0]
            diff_x = pts[:, 1] - p1[1]
            perp_dist = np.abs(diff_y * ux - diff_x * uy)

            inlier_mask = perp_dist <= PERP_TOL_PX
            n_inl = int(np.sum(inlier_mask))
            if n_inl < MIN_INLIERS:
                continue

            # Span along sampled line direction
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

        # Inliers found: map to global indices
        inlier_global_indices = rem_arr[best_inliers_mask]
        inlier_pts = coords[inlier_global_indices]

        # Refit each line to its inliers by PCA
        mean_pt = np.mean(inlier_pts, axis=0)  # (mean_y, mean_x)
        centered = inlier_pts - mean_pt

        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx_sort]

        # Principal eigenvector (line direction)
        vec = eigvecs[:, 0]
        uy, ux = float(vec[0]), float(vec[1])
        if ux < 0 or (abs(ux) < 1e-7 and uy < 0):
            uy, ux = -uy, -ux

        # Orthogonal residuals of inliers to the PCA line
        diff_y = inlier_pts[:, 0] - mean_pt[0]
        diff_x = inlier_pts[:, 1] - mean_pt[1]
        perp_dists = np.abs(diff_y * ux - diff_x * uy)
        rms_residual = float(np.sqrt(np.mean(perp_dists ** 2)))

        # Projection along PCA direction
        projs = diff_y * uy + diff_x * ux
        span_px = float(np.max(projs) - np.min(projs))
        t_min = float(np.min(projs))
        t_max = float(np.max(projs))

        # Angle convention: atan2(dy, dx) mod 180 degrees
        angle_deg = float(np.degrees(np.arctan2(uy, ux)) % 180.0)

        inlier_cands = [candidates[i] for i in inlier_global_indices]

        line_info = {
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
        }
        lines.append(line_info)

        # Remove inliers and search again
        inlier_set = set(inlier_global_indices)
        remaining_indices = [i for i in remaining_indices if i not in inlier_set]

    return lines


def read_text_file(path: Path) -> str:
    """Read text file safely handling potential UTF-16 or UTF-8 encodings."""
    try:
        with open(path, "r", encoding="utf-16") as f:
            return f.read()
    except (UnicodeDecodeError, UnicodeError):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()


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

    # Set A frames
    frames_a = ["20240209-0036", "20240211-0013", "20240216-0030"]

    # Discover all FITS frames
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        stem = p.stem.lower()
        fits_by_stem[stem] = p

    # Load verdicts.csv
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    verdict_by_frame_cand = {}
    streaks_per_frame = {}
    all_fp_frames = set()

    if verdicts_path.exists():
        df_v = pd.read_csv(verdicts_path)
        col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
        col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
        col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]

        df_v["stem"] = df_v[col_frame].astype(str).apply(lambda s: Path(s).stem.lower())
        df_v["verdict_clean"] = df_v[col_verdict].astype(str).str.strip().str.lower()

        for stem, group in df_v.groupby("stem"):
            n_strk = int(np.sum(group["verdict_clean"] == "streak"))
            n_fp = int(np.sum(group["verdict_clean"] == "fp"))
            streaks_per_frame[stem] = n_strk
            if n_strk == 0 and n_fp > 0:
                all_fp_frames.add(stem)

            for _, row in group.iterrows():
                cid = int(row[col_cid])
                verdict_by_frame_cand[(stem, cid)] = row["verdict_clean"]

    # Identify Set B frames
    zero_call_frames = set()
    run_log_path = base_dir / "data/g1/run_log.txt"
    if run_log_path.exists():
        log_content = read_text_file(run_log_path)
        # Matches e.g. "20240209-0036.fits: selected=12, p>=0.5=0"
        matches = re.findall(r"(\d{8}-\d{4})(?:\.fits)?:\s*selected=\d+,\s*p>=0\.5=0\b", log_content)
        zero_call_frames = set(m.lower() for m in matches)

    frames_b = sorted(list((zero_call_frames | all_fp_frames) - set(f.lower() for f in frames_a)))

    out_csv_dir = base_dir / "data/d0b"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = base_dir / "outputs/d0b"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    csv_rows = []

    print("TASK D0b-diag: Line-Based Grouping via Sequential RANSAC")
    print("=" * 70)
    print("SET A (Known Multi-Fragment Streaks):")

    for stem in frames_a:
        fits_file = fits_by_stem.get(stem.lower())
        if not fits_file or not fits_file.exists():
            print(f"  Warning: FITS for {stem} not found.")
            continue

        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = data.shape

        # Rebuild raw candidates BEFORE stitching, ranking, or selection
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

        n_raw = len(raw_candidates)
        print(f"\nFrame: {stem} | Raw candidate blobs: {n_raw}")

        lines = find_lines_ransac(raw_candidates, rng=rng)
        total_streaks_in_frame = streaks_per_frame.get(stem.lower(), 0)

        # Fraction captured by best line
        best_line_streaks = 0
        line_evals = []
        for line in lines:
            n_strk = 0
            n_fp = 0
            n_unlabeled = 0
            for cand in line["inliers"]:
                cid = int(cand["label"])
                v = verdict_by_frame_cand.get((stem.lower(), cid))
                if v == "streak":
                    n_strk += 1
                elif v == "fp":
                    n_fp += 1
                else:
                    n_unlabeled += 1

            best_line_streaks = max(best_line_streaks, n_strk)
            line_evals.append({
                "line": line,
                "n_streak": n_strk,
                "n_fp": n_fp,
                "n_unlabeled": n_unlabeled,
            })

        frac_best = round(best_line_streaks / total_streaks_in_frame, 4) if total_streaks_in_frame > 0 else 0.0

        if not lines:
            print("  No straight lines met inlier (>=5) and span (>=800px) thresholds.")
        else:
            print(f"  Lines found: {len(lines)} | Best line streak capture: {best_line_streaks}/{total_streaks_in_frame} ({frac_best*100:.1f}%)")
            for le in line_evals:
                l = le["line"]
                print(
                    f"    Line {l['line_id']}: n_inliers={l['n_inliers']:2d}, span={l['span_px']:6.1f}px, "
                    f"rms={l['rms_residual_px']:4.2f}px, angle={l['angle_deg']:5.1f} deg | "
                    f"streak={le['n_streak']:2d}, fp={le['n_fp']:2d}, unlabeled={le['n_unlabeled']:2d}"
                )

                csv_rows.append({
                    "frame": stem,
                    "line_id": l["line_id"],
                    "n_inliers": l["n_inliers"],
                    "span_px": l["span_px"],
                    "rms_residual_px": l["rms_residual_px"],
                    "angle_deg": l["angle_deg"],
                    "n_streak": le["n_streak"],
                    "n_fp": le["n_fp"],
                    "n_unlabeled": le["n_unlabeled"],
                    "fraction_frame_streaks_best_line": frac_best,
                })

        # Save plot for every frame in Set A
        save_line_plot(
            data, bg, std, stem, raw_candidates, line_evals, verdict_by_frame_cand,
            out_img_dir / f"{stem}_lines.png"
        )

    # Process Set B (negatives)
    print("\n" + "=" * 70)
    print(f"SET B (Negative Control Frames, N={len(frames_b)}):")

    scanned_b = 0
    frames_with_lines_b = 0
    total_lines_b = 0
    frames_b_found_lines = []

    for stem in frames_b:
        fits_file = fits_by_stem.get(stem.lower())
        if not fits_file or not fits_file.exists():
            continue

        scanned_b += 1
        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")

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

        lines = find_lines_ransac(raw_candidates, rng=rng)
        total_streaks_in_frame = streaks_per_frame.get(stem.lower(), 0)

        if lines:
            frames_with_lines_b += 1
            total_lines_b += len(lines)
            frames_b_found_lines.append((stem, len(lines)))

            line_evals = []
            best_line_streaks = 0
            for line in lines:
                n_strk = 0
                n_fp = 0
                n_unlabeled = 0
                for cand in line["inliers"]:
                    cid = int(cand["label"])
                    v = verdict_by_frame_cand.get((stem.lower(), cid))
                    if v == "streak":
                        n_strk += 1
                    elif v == "fp":
                        n_fp += 1
                    else:
                        n_unlabeled += 1

                best_line_streaks = max(best_line_streaks, n_strk)
                line_evals.append({
                    "line": line,
                    "n_streak": n_strk,
                    "n_fp": n_fp,
                    "n_unlabeled": n_unlabeled,
                })

            frac_best = round(best_line_streaks / total_streaks_in_frame, 4) if total_streaks_in_frame > 0 else 0.0

            for le in line_evals:
                l = le["line"]
                csv_rows.append({
                    "frame": stem,
                    "line_id": l["line_id"],
                    "n_inliers": l["n_inliers"],
                    "span_px": l["span_px"],
                    "rms_residual_px": l["rms_residual_px"],
                    "angle_deg": l["angle_deg"],
                    "n_streak": le["n_streak"],
                    "n_fp": le["n_fp"],
                    "n_unlabeled": le["n_unlabeled"],
                    "fraction_frame_streaks_best_line": frac_best,
                })

            # Save plot for Set B frames that have at least one line
            save_line_plot(
                data, bg, std, stem, raw_candidates, line_evals, verdict_by_frame_cand,
                out_img_dir / f"{stem}_lines.png"
            )

    print(f"  Frames scanned in Set B: {scanned_b}")
    print(f"  Frames with at least one line: {frames_with_lines_b}")
    print(f"  Total suspect false lines found: {total_lines_b}")
    if frames_b_found_lines:
        print(f"  Frames with lines: {frames_b_found_lines}")
    print()

    # Save CSV
    out_csv = out_csv_dir / "lines.csv"
    df_out = pd.DataFrame(csv_rows)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved lines summary to {out_csv}")


def save_line_plot(
    data: np.ndarray,
    bg: np.ndarray,
    std: np.ndarray,
    stem: str,
    raw_candidates: list[dict],
    line_evals: list[dict],
    verdict_by_frame_cand: dict,
    out_path: Path,
):
    """Save visualization with drawn lines, inlier boxes, and S/F verdict labels."""
    h_img, w_img = data.shape
    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
    sub_sigma = sub / np.clip(std_map, 1e-6, None)

    scale_ds = max(1, int(np.ceil(max(h_img, w_img) / 1000.0)))
    h_ds = h_img // scale_ds
    w_ds = w_img // scale_ds
    frame_ds = sub_sigma[:h_ds * scale_ds, :w_ds * scale_ds].reshape(
        h_ds, scale_ds, w_ds, scale_ds
    ).mean(axis=(1, 3))

    vmin, vmax = np.percentile(frame_ds, [5, 99.5])

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(
        frame_ds,
        cmap="gray",
        origin="lower",
        extent=[0, w_img, 0, h_img],
        vmin=vmin,
        vmax=vmax,
    )

    colors = ["cyan", "magenta", "yellow", "orange", "lime"]

    for idx, le in enumerate(line_evals):
        l = le["line"]
        color = colors[idx % len(colors)]

        # Draw line segment spanning across inliers
        t0 = l["t_min"] - 30.0
        t1 = l["t_max"] + 30.0
        y0 = l["mean_y"] + t0 * l["uy"]
        x0 = l["mean_x"] + t0 * l["ux"]
        y1 = l["mean_y"] + t1 * l["uy"]
        x1 = l["mean_x"] + t1 * l["ux"]

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=2.0, linestyle="-",
                label=f"Line {l['line_id']}: n={l['n_inliers']}, span={l['span_px']:.0f}px")

        # Draw inlier boxes and S/F labels
        for cand in l["inliers"]:
            cid = int(cand["label"])
            r0, c0, r1, c1 = cand.get("bbox_full", cand.get("bbox_local"))
            w = c1 - c0
            h = r1 - r0
            rect = patches.Rectangle(
                (c0, r0), w, h,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)

            v = verdict_by_frame_cand.get((stem.lower(), cid))
            if v == "streak":
                ax.text(c1 + 8, r1, "S", color="lime", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))
            elif v == "fp":
                ax.text(c1 + 8, r1, "F", color="red", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6))

    ax.set_title(f"{stem} - Line Grouping (Sequential RANSAC)\nLines found: {len(line_evals)}", fontsize=11)
    ax.set_xlim(0, w_img)
    ax.set_ylim(0, h_img)
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    if line_evals:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
