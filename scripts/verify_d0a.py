# TASK D0a: fix the stitcher's mirrored angle test
"""
Verification script for TASK D0a: fix the stitcher's mirrored angle test.
Evaluates collinear fragment stitching on frames 20240209-0036, 20240211-0013, and 20240216-0030
comparing angle_mode='legacy' vs angle_mode='signed'.
"""

import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

base_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from src.background import estimate_background
from src.detect import find_blob_candidates, stitch_collinear_fragments
from src.io_utils import load_fits


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

    target_stems = ["20240209-0036", "20240211-0013", "20240216-0030"]

    # Search for FITS frames
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    fits_by_stem = {}
    for p in search_root.rglob("*.fits"):
        stem = p.stem.lower()
        for ts in target_stems:
            if stem == ts.lower():
                fits_by_stem[ts] = p

    # Load verdicts to identify labeled streak fragments
    verdicts_path = base_dir / "data/g1/verdicts.csv"
    if not verdicts_path.exists():
        print(f"Error: {verdicts_path} not found.")
        sys.exit(1)

    df_v = pd.read_csv(verdicts_path)
    col_frame = [c for c in df_v.columns if "frame" in c.lower()][0]
    col_cid = [c for c in df_v.columns if "cand" in c.lower() or "id" in c.lower()][0]
    col_verdict = [c for c in df_v.columns if "verdict" in c.lower()][0]

    df_v["stem"] = df_v[col_frame].astype(str).apply(lambda s: Path(s).stem)
    is_streak = df_v[col_verdict].astype(str).str.strip().str.lower() == "streak"
    streak_verdicts = df_v[is_streak]

    out_csv_dir = base_dir / "data/d0a"
    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = base_dir / "outputs/d0a"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    print("TASK D0a: Verifying stitcher mirrored angle fix")
    print("=" * 60)

    for stem in target_stems:
        fits_file = fits_by_stem.get(stem)
        if not fits_file or not fits_file.exists():
            print(f"Warning: FITS file for {stem} not found.")
            continue

        frame_v = streak_verdicts[streak_verdicts["stem"] == stem]
        target_streak_ids = set(frame_v[col_cid].astype(int).tolist())

        # Load data and rebuild candidates exactly as g1_run.py does
        data, _ = load_fits(fits_file)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
        h_img, w_img = data.shape

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

        # Run stitching with legacy angle mode
        cands_legacy = [dict(c) for c in raw_candidates]
        stitched_legacy = stitch_collinear_fragments(
            cands_legacy, data, bg, std, angle_mode="legacy"
        )
        n_legacy = len(stitched_legacy)

        # Run stitching with signed angle mode
        cands_signed = [dict(c) for c in raw_candidates]
        stitched_signed = stitch_collinear_fragments(
            cands_signed, data, bg, std, angle_mode="signed"
        )
        n_signed = len(stitched_signed)

        # Count separate stitched candidates containing labeled streak fragments
        streak_groups_legacy = [
            sc for sc in stitched_legacy
            if any(m in target_streak_ids for m in sc.get("stitched_from", [sc["label"]]))
        ]
        streak_groups_signed = [
            sc for sc in stitched_signed
            if any(m in target_streak_ids for m in sc.get("stitched_from", [sc["label"]]))
        ]

        n_sg_legacy = len(streak_groups_legacy)
        n_sg_signed = len(streak_groups_signed)
        n_target_frags = len(target_streak_ids)

        print(f"Frame: {stem}")
        print(f"  Raw candidate blobs: {n_raw}")
        print(f"  Total candidates: legacy={n_legacy}, signed={n_signed}")
        print(f"  Labeled streak fragments in verdicts.csv: {n_target_frags}")
        print(f"  Separate stitched candidates containing streak fragments:")
        print(f"    legacy: {n_sg_legacy} groups")
        print(f"    signed: {n_sg_signed} groups")
        print()

        summary_rows.append({
            "frame": stem,
            "n_raw": n_raw,
            "n_legacy": n_legacy,
            "n_signed": n_signed,
            "n_streak_fragments_labeled": n_target_frags,
            "n_streak_groups_legacy": n_sg_legacy,
            "n_streak_groups_signed": n_sg_signed,
        })

        # Save output visualization for signed mode: outputs/d0a/<frame>_signed.png
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

        cmap_colors = plt.cm.tab20.colors
        color_idx = 0

        # Draw stitched groups
        for cand in stitched_signed:
            members = cand.get("stitched_from", [cand["label"]])
            n_frag = len(members)
            bbox = cand.get("bbox_full", cand.get("bbox_local"))
            r0, c0, r1, c1 = bbox
            width = c1 - c0
            height = r1 - r0

            is_streak_group = any(m in target_streak_ids for m in members)

            if n_frag > 1:
                color = cmap_colors[color_idx % len(cmap_colors)]
                color_idx += 1
                lw = 2.0
                rect = patches.Rectangle(
                    (c0, r0),
                    width,
                    height,
                    linewidth=lw,
                    edgecolor=color,
                    facecolor="none",
                )
                ax.add_patch(rect)
                label_txt = f"n={n_frag}"
                if is_streak_group:
                    label_txt += " (streak)"
                ax.text(
                    c1 + 10,
                    r1,
                    label_txt,
                    color=color,
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6),
                )
            elif is_streak_group:
                # Unstitched streak fragment
                rect = patches.Rectangle(
                    (c0, r0),
                    width,
                    height,
                    linewidth=1.2,
                    edgecolor="yellow",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(
                    c1 + 10,
                    r1,
                    "n=1 (streak)",
                    color="yellow",
                    fontsize=7,
                    bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.6),
                )

        ax.set_title(
            f"{stem} (angle_mode='signed') - Stitched Candidates\n"
            f"Total: {n_signed} | Streak Groups: {n_sg_signed}",
            fontsize=11,
        )
        ax.set_xlim(0, w_img)
        ax.set_ylim(0, h_img)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

        out_img_path = out_img_dir / f"{stem}_signed.png"
        fig.tight_layout()
        fig.savefig(out_img_path)
        plt.close(fig)
        print(f"  Saved figure: {out_img_path}")

    # Write summary CSV
    out_csv_path = out_csv_dir / "stitch_counts.csv"
    df_counts = pd.DataFrame(summary_rows)
    df_counts.to_csv(out_csv_path, index=False)
    print(f"\nSaved stitch summary counts to {out_csv_path}")


if __name__ == "__main__":
    main()
