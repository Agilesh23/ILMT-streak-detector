# TASK D0b-fix1: extent floor and raw-centroid check
"""
Batch line grouping runner across 86 public frames from data/g1/frame_20*.csv.
Runs find_streak_lines from src/line_group, joins labels from verdicts.csv,
outputs data/d0b/all_lines.csv, and generates contact sheets in outputs/d0b/all/.
"""

import math
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates
from src.io_utils import load_fits
from src.line_group import (
    find_streak_lines,
    get_frame_boundary_intersections,
    sample_full_line_strip,
    PERP_TOL_PX,
)

KNOWN_REAL = [
    ("20240209-0036", 1),
    ("20240211-0013", 1),
    ("20240216-0030", 1),
    ("20230615-0006", 1),
    ("20240502-0026", 1),
]

KNOWN_FALSE = [
    ("20230513-0034", 1),
    ("20231121-0025", 1),
    ("20241205-0034", 1),
    ("20241205-0037", 1),
    ("20250123-0003", 1),
    ("20250123-0003", 2),
]


def make_strip_tile(sub: np.ndarray, line: dict, h_img: int, w_img: int):
    """
    Generate straightened strip (+-60 px, binned to <= 800 px, sigma 1.5) for contact sheet tile.
    """
    mean_y, mean_x = line["centroid"]
    uy, ux = line["unit_vector"]

    pt_b1, pt_b2 = get_frame_boundary_intersections(mean_y, mean_x, uy, ux, h_img, w_img)
    if pt_b1 is not None and pt_b2 is not None:
        t_b1 = pt_b1[0]
        t_b2 = pt_b2[0]
    else:
        t_b1 = -2000.0
        t_b2 = 2000.0

    t_vals = np.arange(t_b1, t_b2 + 1.0, 1.0)
    w_vals = np.arange(-60.0, 61.0, 1.0)  # 121 rows
    T_grid, W_grid = np.meshgrid(t_vals, w_vals)

    Y_strip = mean_y + T_grid * uy - W_grid * ux
    X_strip = mean_x + T_grid * ux + W_grid * uy

    strip_raw = map_coordinates(sub, [Y_strip.ravel(), X_strip.ravel()], order=1, mode="constant", cval=0.0).reshape(Y_strip.shape)
    strip_blur = gaussian_filter(strip_raw, sigma=1.5)

    n_cols = len(t_vals)
    bin_f = max(1, int(math.ceil(n_cols / 800.0)))
    n_trim = (n_cols // bin_f) * bin_f
    if bin_f > 1:
        strip_disp = strip_blur[:, :n_trim].reshape(121, n_trim // bin_f, bin_f).mean(axis=2)
    else:
        strip_disp = strip_blur

    vmin = float(np.percentile(strip_blur, 5.0))
    vmax = float(np.percentile(strip_blur, 99.5))

    # Extent ticks on top border
    if line["t_start"] is not None and line["t_end"] is not None:
        tick_t0 = line["t_start"]
        tick_t1 = line["t_end"]
    else:
        # For rejected lines, use inlier span
        t_mid = 0.0
        tick_t0 = -line["span_px"] / 2.0
        tick_t1 = line["span_px"] / 2.0

    col_t0 = (tick_t0 - t_b1) / float(bin_f)
    col_t1 = (tick_t1 - t_b1) / float(bin_f)

    return strip_disp, vmin, vmax, col_t0, col_t1


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

    # Find 86 frame CSVs
    g1_frame_csvs = sorted(list((base_dir / "data/g1").glob("frame_20*.csv")))
    stems_to_run = [p.stem.replace("frame_", "") for p in g1_frame_csvs]

    # Search for FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        fits_by_stem[p.stem.lower()] = p

    # Load verdicts.csv to map candidate labels to verdicts
    verdict_map = {}
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    if verdicts_path.exists():
        df_v = pd.read_csv(verdicts_path)
        col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
        col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
        col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]

        for _, row in df_v.iterrows():
            st = Path(str(row[col_frame])).stem.lower()
            cid = int(row[col_cid])
            v = str(row[col_verdict]).strip().lower()
            verdict_map[(st, cid)] = v

    out_csv_dir = base_dir / "data/d0b"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_sheets_dir = base_dir / "outputs/d0b/all2"
    out_sheets_dir.mkdir(parents=True, exist_ok=True)

    all_lines_records = []
    lines_for_sheets = []
    scanned_frames = 0
    total_lines = 0
    accepted_lines = 0
    rejected_lines = 0
    eval_status_map = {}

    print(f"TASK D0b-build: Line Grouping Module on {len(stems_to_run)} Frames")
    print("=" * 80)

    for idx, stem in enumerate(stems_to_run, 1):
        fits_file = fits_by_stem.get(stem.lower())
        if not fits_file or not fits_file.exists():
            print(f"[{idx:2d}/{len(stems_to_run)}] {stem}: FITS not found, skipping.")
            continue

        scanned_frames += 1
        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        h_img, w_img = data.shape

        # Rebuild candidates identically
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

        lines = find_streak_lines(raw_candidates, data, bg)
        n_lines_found = len(lines)
        total_lines += n_lines_found

        acc_in_frame = sum(1 for l in lines if l["accepted"])
        accepted_lines += acc_in_frame
        rejected_lines += (n_lines_found - acc_in_frame)

        print(f"[{idx:2d}/{len(stems_to_run)}] {stem}: raw_cands={len(raw_candidates):3d} | lines_found={n_lines_found} | accepted={acc_in_frame}")

        for line in lines:
            lid = line["line_id"]
            is_acc = line["accepted"]
            eval_status_map[(stem, lid)] = is_acc

            # Member evaluation for verdicts
            eval_members = line["fragment_ids"] if is_acc else line["members_if_accepted"]
            n_strk = 0
            n_fp = 0
            n_unlab = 0

            for cid in eval_members:
                v = verdict_map.get((stem.lower(), cid))
                if v == "streak":
                    n_strk += 1
                elif v == "fp":
                    n_fp += 1
                else:
                    n_unlab += 1

            cy, cx = line["centroid"]
            uy, ux = line["unit_vector"]
            endpoints = line["endpoints"]
            if endpoints is not None:
                ep0_r, ep0_c = endpoints[0]
                ep1_r, ep1_c = endpoints[1]
            else:
                ep0_r, ep0_c, ep1_r, ep1_c = None, None, None, None

            record = {
                "frame": stem,
                "line_id": lid,
                "accepted": is_acc,
                "refined": line["refined"],
                "z_peak": line["z_peak"],
                "n_segments_z_gt_3": line["n_segments_z_gt_3"],
                "slope_px_per_segment": line["slope_px_per_segment"],
                "peak_offset": line["peak_offset"],
                "n_inliers": line["n_inliers"],
                "span_px": line["span_px"],
                "rms_residual_px": line["rms_residual_px"],
                "angle_deg": line["angle_deg"],
                "centroid_y": cy,
                "centroid_x": cx,
                "uy": uy,
                "ux": ux,
                "t_start": line["t_start"],
                "t_end": line["t_end"],
                "endpoint_0_r": ep0_r,
                "endpoint_0_c": ep0_c,
                "endpoint_1_r": ep1_r,
                "endpoint_1_c": ep1_c,
                "n_fragments": line["n_fragments"],
                "fragment_ids": ";".join(str(m) for m in line["fragment_ids"]),
                "n_streak": n_strk,
                "n_fp": n_fp,
                "n_unlabeled": n_unlab,
            }
            all_lines_records.append(record)

            # Pre-render strip tile for contact sheet
            tile_img, vmin_t, vmax_t, ct0, ct1 = make_strip_tile(sub, line, h_img, w_img)
            lines_for_sheets.append({
                "record": record,
                "tile_img": tile_img,
                "vmin": vmin_t,
                "vmax": vmax_t,
                "col_t0": ct0,
                "col_t1": ct1,
            })

    # Save CSV
    out_csv_path = out_csv_dir / "all_lines_v2.csv"
    df_all_lines = pd.DataFrame(all_lines_records)
    df_all_lines.to_csv(out_csv_path, index=False)
    print(f"\nSaved all lines CSV to: {out_csv_path}")

    # Generate Contact Sheets: 6 lines per sheet
    # Sorting: accepted lines first, sorted by z_peak desc; then rejected lines sorted by z_peak desc
    accepted_items = [it for it in lines_for_sheets if it["record"]["accepted"]]
    rejected_items = [it for it in lines_for_sheets if not it["record"]["accepted"]]

    accepted_items.sort(key=lambda it: it["record"]["z_peak"], reverse=True)
    rejected_items.sort(key=lambda it: it["record"]["z_peak"], reverse=True)

    ordered_sheet_items = accepted_items + rejected_items
    n_total_tiles = len(ordered_sheet_items)
    n_sheets = int(math.ceil(n_total_tiles / 6.0))

    for s_idx in range(n_sheets):
        batch = ordered_sheet_items[s_idx * 6 : (s_idx + 1) * 6]
        fig, axes = plt.subplots(len(batch), 1, figsize=(14, 2.5 * len(batch)), dpi=120, squeeze=False)

        for row_i, item in enumerate(batch):
            ax = axes[row_i, 0]
            rec = item["record"]
            disp = item["tile_img"]

            ax.imshow(
                disp,
                cmap="gray",
                origin="lower",
                extent=[0, disp.shape[1], -60, 60],
                vmin=item["vmin"],
                vmax=item["vmax"],
                aspect="auto",
            )

            status_str = "ACCEPTED" if rec["accepted"] else "REJECTED"
            status_color = "green" if rec["accepted"] else "red"
            title_text = (
                f"{rec['frame']} | Line {rec['line_id']} | z_peak={rec['z_peak']:.2f} | "
                f"z>3 segs={rec['n_segments_z_gt_3']}/8 | {status_str}"
            )
            ax.set_title(title_text, fontsize=9.5, fontweight="bold", color=status_color, loc="left")
            ax.set_ylabel("Offset (px)", fontsize=7.5)
            ax.set_xlabel("Along Line (binned px)", fontsize=7.5)

            # Ticks on top border
            t0_x = item["col_t0"]
            t1_x = item["col_t1"]
            ax.plot([t0_x, t0_x], [45, 60], color="red", linewidth=2.0)
            ax.plot([t1_x, t1_x], [45, 60], color="red", linewidth=2.0)

        fig.tight_layout()
        sheet_path = out_sheets_dir / f"sheet_{s_idx + 1:02d}.png"
        fig.savefig(sheet_path)
        plt.close(fig)

    print(f"Saved {n_sheets} contact sheets to: {out_sheets_dir}")

    # Summary report
    print("\n" + "=" * 80)
    print("RUN SUMMARY")
    print("=" * 80)
    print(f"Frames scanned: {scanned_frames}")
    print(f"Total lines found: {total_lines}")
    print(f"Accepted lines: {accepted_lines}")
    print(f"Rejected lines: {rejected_lines}")
    print("\nBENCHMARK LINES STATUS:")
    print("-" * 50)
    print("KNOWN REAL LINES:")
    for f, lid in KNOWN_REAL:
        st = eval_status_map.get((f, lid))
        st_label = "ACCEPTED" if st is True else ("REJECTED" if st is False else "NOT FOUND")
        print(f"  {f} Line {lid}: {st_label}")

    print("\nKNOWN FALSE LINES:")
    for f, lid in KNOWN_FALSE:
        st = eval_status_map.get((f, lid))
        st_label = "ACCEPTED" if st is True else ("REJECTED" if st is False else "NOT FOUND")
        print(f"  {f} Line {lid}: {st_label}")
    print("=" * 80)


if __name__ == "__main__":
    main()
