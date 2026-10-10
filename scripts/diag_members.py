# TASK D0b-fix1: extent floor and raw-centroid check
"""
Diagnostic script to investigate why labeled streak candidates are missing from line members.
For each frame with an accepted line in data/d0b/all_lines_v2.csv (or all_lines.csv):
- Rebuilds the line using find_streak_lines from src/line_group (seed 42).
- For every candidate labeled 'streak' in data/g1/verdicts.csv for that frame, checks:
  exists_in_raw, in_g1_csv, in_members, raw centroid, d_perp, t, inside_extent, reason_missing,
  nearest_raw_label, nearest_raw_dist_px.
Outputs data/d0b/member_check.csv.
"""

from collections import Counter
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates
from src.io_utils import load_fits
from src.line_group import find_streak_lines, PERP_TOL_PX


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

    # 1. Read data/d0b/all_lines_v2.csv (fallback all_lines.csv) and filter accepted lines
    all_lines_path = base_dir / "data/d0b/all_lines_v2.csv"
    if not all_lines_path.exists():
        all_lines_path = base_dir / "data/d0b/all_lines.csv"
    if not all_lines_path.exists():
        print(f"Error: Neither all_lines_v2.csv nor all_lines.csv found.")
        sys.exit(1)

    df_all_lines = pd.read_csv(all_lines_path)
    df_acc = df_all_lines[df_all_lines["accepted"] == True]

    if df_acc.empty:
        print("No accepted lines found in all_lines.csv.")
        sys.exit(0)

    # 2. Read verdicts.csv
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    if not verdicts_path.exists():
        print(f"Error: {verdicts_path} not found.")
        sys.exit(1)

    df_v = pd.read_csv(verdicts_path)
    col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
    col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
    col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]

    df_v["stem"] = df_v[col_frame].astype(str).apply(lambda s: Path(s).stem.lower())
    df_v["verdict_clean"] = df_v[col_verdict].astype(str).str.strip().str.lower()
    df_streaks = df_v[df_v["verdict_clean"] == "streak"]

    # 3. Locate FITS files
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        fits_by_stem[p.stem.lower()] = p

    # Process each frame with accepted lines
    unique_accepted_frames = sorted(df_acc["frame"].unique())

    out_records = []
    total_streak_cands = 0
    total_in_members = 0
    total_reasons = Counter()

    print("TASK D0b-members: Diagnosing Missing Labeled Streak Members")
    print("=" * 80)

    for stem in unique_accepted_frames:
        fits_file = fits_by_stem.get(str(stem).lower())
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS for {stem} not found.")
            continue

        # Load G1 CSV for this frame
        g1_csv_path = base_dir / f"data/g1/frame_{stem}.csv"
        df_g1 = pd.read_csv(g1_csv_path) if g1_csv_path.exists() else pd.DataFrame()

        # Rebuild candidates and lines exactly as run_line_group.py
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

        lines = find_streak_lines(raw_candidates, data, bg)
        acc_lines = [l for l in lines if l["accepted"]]

        if not acc_lines:
            continue

        # Raw candidates map and spatial array
        raw_by_id = {int(c["label"]): c for c in raw_candidates}
        raw_coords = np.array([c["centroid"] for c in raw_candidates]) if raw_candidates else np.empty((0, 2))
        raw_labels_list = [int(c["label"]) for c in raw_candidates]

        # Get all labeled streak candidates for this frame from verdicts.csv
        frame_streaks = df_streaks[df_streaks["stem"] == str(stem).lower()]
        streak_cids = frame_streaks[col_cid].astype(int).tolist()

        frame_acc_lines = df_acc[df_acc["frame"] == stem]
        for _, acc_row in frame_acc_lines.iterrows():
            target_line_id = int(acc_row["line_id"])
            matching = [l for l in acc_lines if l["line_id"] == target_line_id]
            if not matching:
                line = acc_lines[0]
            else:
                line = matching[0]

            line_id = line["line_id"]
            mean_y, mean_x = line["centroid"]
            uy, ux = line["unit_vector"]
            t_start = line["t_start"]
            t_end = line["t_end"]
            fragment_ids = set(line["fragment_ids"])

            frame_reasons = Counter()
            frame_in_members = 0

            for cid in streak_cids:
                total_streak_cands += 1
                exists_in_raw = (cid in raw_by_id)

                # Check if in G1 CSV
                in_g1_csv = False
                cy, cx = np.nan, np.nan
                if not df_g1.empty and (cid in df_g1["candidate_id"].values):
                    in_g1_csv = True
                    row_g1 = df_g1[df_g1["candidate_id"] == cid].iloc[0]
                    cy = float(row_g1["centroid_row"])
                    cx = float(row_g1["centroid_col"])
                elif exists_in_raw:
                    cy, cx = raw_by_id[cid]["centroid"]

                # If not in raw, find nearest raw candidate
                nearest_raw_label = None
                nearest_raw_dist_px = np.nan
                if (not exists_in_raw) and len(raw_coords) > 0 and not np.isnan(cy) and not np.isnan(cx):
                    dists = np.hypot(raw_coords[:, 0] - cy, raw_coords[:, 1] - cx)
                    min_idx = int(np.argmin(dists))
                    nearest_raw_dist_px = round(float(dists[min_idx]), 2)
                    nearest_raw_label = int(raw_labels_list[min_idx])

                in_members = (cid in fragment_ids)
                if in_members:
                    frame_in_members += 1
                    total_in_members += 1

                # Raw blob lookup
                raw_cand = raw_by_id.get(cid)
                if raw_cand is not None:
                    raw_cy, raw_cx = raw_cand["centroid"]
                    raw_t = (raw_cy - mean_y) * uy + (raw_cx - mean_x) * ux
                    raw_d_perp = abs((raw_cy - mean_y) * ux - (raw_cx - mean_x) * uy)
                    raw_in_extent = bool(t_start <= raw_t <= t_end)
                    raw_d_perp_ok = bool(raw_d_perp <= PERP_TOL_PX)
                else:
                    raw_cy, raw_cx = np.nan, np.nan
                    raw_t = np.nan
                    raw_d_perp = np.nan
                    raw_in_extent = False
                    raw_d_perp_ok = False

                if (not np.isnan(raw_cy)) and (not np.isnan(cy)):
                    raw_vs_g1_dist_px = round(float(np.hypot(raw_cy - cy, raw_cx - cx)), 2)
                else:
                    raw_vs_g1_dist_px = np.nan

                # Compute d_perp and t for G1 centroid
                if not np.isnan(cy) and not np.isnan(cx):
                    t_cand = (cy - mean_y) * uy + (cx - mean_x) * ux
                    d_perp = abs((cy - mean_y) * ux - (cx - mean_x) * uy)
                    inside_extent = bool(t_start <= t_cand <= t_end)
                    d_perp_ok = bool(d_perp <= PERP_TOL_PX)
                else:
                    t_cand = np.nan
                    d_perp = np.nan
                    inside_extent = False
                    d_perp_ok = False

                # Determine reason_missing
                if in_members:
                    reason_missing = "in_members"
                elif not exists_in_raw:
                    reason_missing = "not_in_raw"
                elif (not raw_d_perp_ok) or (not raw_in_extent):
                    reason_missing = "raw_blob_off_line"
                else:
                    reason_missing = "other"

                frame_reasons[reason_missing] += 1
                total_reasons[reason_missing] += 1

                out_records.append({
                    "frame": stem,
                    "line_id": line_id,
                    "candidate_id": cid,
                    "exists_in_raw": exists_in_raw,
                    "in_g1_csv": in_g1_csv,
                    "in_members": in_members,
                    "centroid_row": round(cy, 2) if not np.isnan(cy) else np.nan,
                    "centroid_col": round(cx, 2) if not np.isnan(cx) else np.nan,
                    "raw_centroid_row": round(raw_cy, 2) if not np.isnan(raw_cy) else np.nan,
                    "raw_centroid_col": round(raw_cx, 2) if not np.isnan(raw_cx) else np.nan,
                    "raw_vs_g1_dist_px": raw_vs_g1_dist_px,
                    "d_perp": round(d_perp, 2) if not np.isnan(d_perp) else np.nan,
                    "t": round(t_cand, 2) if not np.isnan(t_cand) else np.nan,
                    "inside_extent": inside_extent,
                    "d_perp_ok": d_perp_ok,
                    "raw_d_perp": round(raw_d_perp, 2) if not np.isnan(raw_d_perp) else np.nan,
                    "raw_t": round(raw_t, 2) if not np.isnan(raw_t) else np.nan,
                    "raw_in_extent": raw_in_extent,
                    "reason_missing": reason_missing,
                    "nearest_raw_label": nearest_raw_label,
                    "nearest_raw_dist_px": nearest_raw_dist_px,
                })

            print(f"Frame {stem} (Line {line_id}):")
            print(f"  Labeled streak candidates: {len(streak_cids)}")
            print(f"  In line members:           {frame_in_members}/{len(streak_cids)}")
            print("  Breakdown:")
            for r_key in ["in_members", "not_in_raw", "raw_blob_off_line", "other"]:
                if frame_reasons[r_key] > 0:
                    print(f"    {r_key:<18}: {frame_reasons[r_key]}")
            print()

    # Save output CSV
    out_csv = base_dir / "data/d0b/member_check.csv"
    df_out = pd.DataFrame(out_records)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print overall totals
    print("\n" + "=" * 80)
    print("TOTALS ACROSS ALL ACCEPTED FRAMES")
    print("=" * 80)
    print(f"Total labeled streak candidates: {total_streak_cands}")
    print(f"Total in line members:           {total_in_members} ({total_in_members / max(1, total_streak_cands) * 100:.1f}%)")
    print("Breakdown by status:")
    for r_key in ["in_members", "not_in_raw", "raw_blob_off_line", "other"]:
        print(f"  {r_key:<18}: {total_reasons[r_key]:3d}")
    print("=" * 80)


if __name__ == "__main__":
    main()
