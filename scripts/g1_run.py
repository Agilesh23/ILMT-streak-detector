import argparse
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation, gaussian_filter
from skimage.measure import label as sk_label
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
g1_dir = data_dir / "g1"
g1_dir.mkdir(parents=True, exist_ok=True)
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)
outputs_g1_dir = base_dir / "outputs/g1"
outputs_g1_dir.mkdir(parents=True, exist_ok=True)
calls_dir = outputs_g1_dir / "calls"
calls_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import width_estimation, extract_streak_patch

feature_cols = [
    "width_ratio",
    "snr_mean",
    "sat_frac",
    "straightness",
    "angle_from_vertical",
    "aspect",
    "edge_dist",
    "length_px",
]

csv_cols = [
    "frame_name",
    "candidate_id",
    "centroid_row",
    "centroid_col",
    "bbox_r0",
    "bbox_c0",
    "bbox_r1",
    "bbox_c1",
    "width_ratio",
    "snr_mean",
    "sat_frac",
    "straightness",
    "angle_from_vertical",
    "aspect",
    "edge_dist",
    "length_px",
    "p_streak",
]


def parse_args():
    parser = argparse.ArgumentParser(description="G1 Pipeline Runner")
    parser.add_argument("--stride", type=int, default=3, help="Stride for frame sampling (default 3)")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N frames")
    parser.add_argument("--redo", action="store_true", default=False, help="Ignore existing frame CSVs and re-run detection")
    return parser.parse_args()


def get_crop(img, cy, cx, size):
    half = size // 2
    r0 = int(round(cy)) - half
    r1 = r0 + size
    c0 = int(round(cx)) - half
    c1 = c0 + size

    h, w = img.shape
    r0_cl = max(0, r0)
    r1_cl = min(h, r1)
    c0_cl = max(0, c0)
    c1_cl = min(w, c1)

    sub_crop = img[r0_cl:r1_cl, c0_cl:c1_cl]
    pad_top = max(0, -r0)
    pad_bottom = max(0, r1 - h)
    pad_left = max(0, -c0)
    pad_right = max(0, c1 - w)

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = np.pad(sub_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
    else:
        crop = sub_crop
    return crop, (c0, c1, r0, r1)


def save_call_png(cand_info, sub_sig, out_path):
    h_f, w_f = sub_sig.shape
    fig_call, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # (a) Full frame downscaled to <= 600 px with candidate bbox in red
    scale_ds = max(1, int(np.ceil(max(h_f, w_f) / 600.0)))
    h_ds, w_ds = h_f // scale_ds, w_f // scale_ds
    frame_ds = sub_sig[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
    vmin_a, vmax_a = np.percentile(frame_ds, [5, 99.5])

    ax_a.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_f, 0, h_f], vmin=vmin_a, vmax=vmax_a)
    rect_a = patches.Rectangle(
        (cand_info["bbox_c0"], cand_info["bbox_r0"]),
        cand_info["bbox_c1"] - cand_info["bbox_c0"],
        cand_info["bbox_r1"] - cand_info["bbox_r0"],
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax_a.add_patch(rect_a)
    ax_a.set_xlim(0, w_f)
    ax_a.set_ylim(0, h_f)

    # (b) Crop in sigma units (-1 to +5)
    if cand_info["length_px"] < 100.0:
        crop, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub_sig, cand_info["centroid_row"], cand_info["centroid_col"], 300)
    else:
        crop_800, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub_sig, cand_info["centroid_row"], cand_info["centroid_col"], 800)
        h_8, w_8 = crop_800.shape
        crop = crop_800[:h_8 // 2 * 2, :w_8 // 2 * 2].reshape(h_8 // 2, 2, w_8 // 2, 2).mean(axis=(1, 3))

    ax_b.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=-1.0, vmax=5.0)
    rect_b = patches.Rectangle(
        (cand_info["bbox_c0"], cand_info["bbox_r0"]),
        cand_info["bbox_c1"] - cand_info["bbox_c0"],
        cand_info["bbox_r1"] - cand_info["bbox_r0"],
        linewidth=1.5,
        edgecolor="red",
        facecolor="none",
    )
    ax_b.add_patch(rect_b)
    ax_b.set_xlim(c0_cr, c1_cr)
    ax_b.set_ylim(r0_cr, r1_cr)

    title_text = (
        f"{cand_info['frame_name']} | ID {cand_info['candidate_id']} | "
        f"p={cand_info['p_streak']:.3f} | L={cand_info['length_px']:.1f}px"
    )
    ax_a.set_title(title_text, fontsize=10, fontweight="bold")
    ax_b.set_title("Crop (-1 to +5 sigma)", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=100)
    plt.close(fig_call)


def main():
    args = parse_args()

    # Load configuration
    with open(base_dir / "src/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    det_cfg = cfg.get("detection", {})
    edge_mask_max_x = det_cfg.get("edge_mask_max_x", 3950.0)
    border_margin_px = det_cfg.get("border_margin_px", 15)
    border_max_length_px = det_cfg.get("border_max_length_px", 100.0)
    left_edge_max_col = det_cfg.get("left_edge_max_col", 60.0)
    left_edge_max_angle_from_vertical_deg = det_cfg.get("left_edge_max_angle_from_vertical_deg", 20.0)
    min_abs_score = cfg.get("ranking", {}).get("min_abs_score", 47.0)
    bypass_cfg = det_cfg.get("long_thin_bypass", None)
    bleed_veto_cfg = det_cfg.get("bleed_veto", None)
    psf_fwhm_px = cfg.get("characterization", {}).get("psf_fwhm_px", 6.79)

    # 1. Discover frames
    search_root = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials")
    excluded_stems = {
        "img",
        "20221025-0028",
        "20221026-0057",
        "20221101-0030",
        "20230317-0008",
        "20230519-0009",
        "20230602-0022",
        "20231115-0003",
        "20240114-0028",
        "20240405-0004",
    }

    seen_names = {}
    path_by_stem = {}
    path_by_name = {}
    for p in search_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".fits":
            name_lower = p.name.lower()
            stem_lower = p.stem.lower()
            path_by_stem[stem_lower] = p
            path_by_name[name_lower] = p
            if name_lower not in seen_names:
                seen_names[name_lower] = p

    valid_frames = []
    for name_lower, p in seen_names.items():
        stem = p.stem.lower()
        if stem in excluded_stems:
            continue
        valid_frames.append(p)

    valid_frames = sorted(valid_frames, key=lambda x: x.name)
    sampled_frames = valid_frames[::args.stride]
    if args.limit is not None:
        sampled_frames = sampled_frames[:args.limit]

    with open(g1_dir / "sample_frames.txt", "w") as f:
        for p in sampled_frames:
            f.write(f"{p.name}\n")

    # 2. Train or load Model M1
    model_path = models_dir / "m1_v1.joblib"
    if model_path.exists():
        clf = joblib.load(model_path)
    else:
        df_feat = pd.read_csv(base_dir / "data/features/features_v1.csv")
        X_train = df_feat[feature_cols].values
        y_train = (df_feat["label"] == "streak").astype(int).values
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42)
        clf.fit(X_train, y_train)
        joblib.dump(clf, model_path)

    # 3. Per-frame loop
    frame_cache = {}

    for p in sampled_frames:
        fname = p.name
        stem = p.stem
        csv_path = g1_dir / f"frame_{stem}.csv"

        if csv_path.exists() and not args.redo:
            df_frame = pd.read_csv(csv_path)
            n_sel = len(df_frame)
            p5_frame = df_frame[df_frame["p_streak"] >= 0.5]
            n_p5 = len(p5_frame)
            print(f"{fname}: selected={n_sel}, p>=0.5={n_p5}")

            # Save individual calls PNGs if not existing
            for _, r in p5_frame.iterrows():
                cid = int(r["candidate_id"])
                call_path = calls_dir / f"{stem}_{cid}.png"
                if not call_path.exists():
                    if stem not in frame_cache:
                        d_raw, _ = load_fits(p)
                        bg_raw, std_raw = estimate_background(d_raw, method="background2d")
                        sub_raw = d_raw - bg_raw
                        std_map_raw = std_raw if isinstance(std_raw, np.ndarray) else np.full_like(sub_raw, std_raw)
                        sub_sig = sub_raw / np.clip(std_map_raw, 1e-6, None)
                        frame_cache[stem] = sub_sig
                    save_call_png(r.to_dict(), frame_cache[stem], call_path)
            continue

        # Pipeline execution on new frame
        data, header = load_fits(p)
        bg, std = estimate_background(data, method="background2d")
        sub = data - bg
        std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
        sub_sigma = sub / np.clip(std_map, 1e-6, None)
        frame_cache[stem] = sub_sigma
        frame_max = float(np.max(data))
        h_img, w_img = data.shape

        thresh_map = 3.0 * std
        binary = sub > thresh_map
        struct = generate_binary_structure(2, 1)
        binary = binary_dilation(binary, structure=struct, iterations=2)
        labeled_map = sk_label(binary)

        candidates, _, _ = find_blob_candidates(
            data, bg, std,
            edge_mask_max_x=edge_mask_max_x,
            border_margin_px=border_margin_px,
            border_max_length_px=border_max_length_px,
            left_edge_max_col=left_edge_max_col,
            left_edge_max_angle_from_vertical_deg=left_edge_max_angle_from_vertical_deg,
            long_thin_bypass=bypass_cfg,
            bleed_veto=bleed_veto_cfg,
        )
        stitched = stitch_collinear_fragments(candidates, data, bg, std)
        selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score, drop_bleed_trails=True)

        rows = []
        for cand in selected:
            cid = int(cand["label"])
            members = cand.get("stitched_from", [cid])
            cand_mask = np.isin(labeled_map, members)
            coords = np.argwhere(cand_mask)

            if len(coords) < 2:
                cy, cx = cand["centroid"]
                r0, c0, r1, c1 = cand.get("bbox_full", cand.get("bbox_local"))
                coords = np.array([[cy, cx], [cy + 1, cx + 1]], dtype=np.float64)
            else:
                r0 = int(np.min(coords[:, 0]))
                r1 = int(np.max(coords[:, 0])) + 1
                c0 = int(np.min(coords[:, 1]))
                c1 = int(np.max(coords[:, 1])) + 1

            pts_y = coords[:, 0].astype(np.float64)
            pts_x = coords[:, 1].astype(np.float64)
            cy = float(np.mean(pts_y))
            cx = float(np.mean(pts_x))
            dx = pts_x - cx
            dy = pts_y - cy

            cov = np.cov(np.vstack([dx, dy]))
            eigvals, eigvecs = np.linalg.eigh(cov)
            idx_sort = np.argsort(eigvals)[::-1]
            eigvals = eigvals[idx_sort]
            eigvecs = eigvecs[:, idx_sort]

            vec = eigvecs[:, 0]
            u_x, u_y = float(vec[0]), float(vec[1])
            if u_x < 0 or (abs(u_x) < 1e-7 and u_y < 0):
                u_x, u_y = -u_x, -u_y
            perp_x, perp_y = -u_y, u_x

            d_perp = dx * perp_x + dy * perp_y
            straightness = float(np.sqrt(np.mean(d_perp ** 2)))

            lam0 = max(0.0, float(eigvals[0]))
            lam1 = max(0.0, float(eigvals[1]))
            aspect = float(np.sqrt(lam0 / max(lam1, 1e-6)))

            ang_h = float(np.degrees(np.arctan2(u_y, u_x)))
            if ang_h > 90.0:
                ang_h -= 180.0
            elif ang_h < -90.0:
                ang_h += 180.0
            angle_from_vertical = float(abs(90.0 - abs(ang_h)))

            t_proj = dx * u_x + dy * u_y
            t_min = float(np.min(t_proj))
            t_max = float(np.max(t_proj))
            length_px = float(t_max - t_min)
            p0 = (cx + t_min * u_x, cy + t_min * u_y)
            p1 = (cx + t_max * u_x, cy + t_max * u_y)

            edge_dist = float(min(r0, c0, h_img - r1, w_img - c1))
            pix_raw = data[coords[:, 0], coords[:, 1]]
            sat_frac = float(np.mean(pix_raw >= 0.9 * frame_max))

            box_w = 40
            patch_len = max(10, int(np.ceil(length_px)))
            try:
                patch, _ = extract_streak_patch(sub, p0, p1, box_width=box_w, output_length=patch_len)
            except Exception:
                patch = np.zeros((box_w, patch_len), dtype=np.float32)

            w_fit = width_estimation(patch, n_sigma=2.0, psf_fwhm=psf_fwhm_px)
            if w_fit is not None and w_fit.get("width_vs_psf_ratio") is not None and not np.isnan(w_fit["width_vs_psf_ratio"]):
                width_ratio = float(w_fit["width_vs_psf_ratio"])
            else:
                cross_prof = np.median(patch, axis=1)
                base = np.percentile(cross_prof, 20)
                pk = np.max(cross_prof)
                half = base + 0.5 * (pk - base)
                above = cross_prof >= half
                fwhm_est = float(np.sum(above)) if np.any(above) else 2.0
                width_ratio = float(fwhm_est / psf_fwhm_px)

            transverse_offsets = np.arange(-4, 5)
            t_steps = np.linspace(t_min, t_max, max(10, int(length_px / 2.0)))
            snrs = []
            for t in t_steps:
                xc_t = cx + t * u_x
                yc_t = cy + t * u_y
                sample_xs = xc_t + transverse_offsets * perp_x
                sample_ys = yc_t + transverse_offsets * perp_y
                vals = map_coordinates(sub_sigma, np.vstack([sample_ys, sample_xs]), order=1, mode="constant", cval=0.0)
                snrs.append(float(np.median(vals)))
            snr_mean = float(np.mean(snrs)) if snrs else 0.0

            feats = [width_ratio, snr_mean, sat_frac, straightness, angle_from_vertical, aspect, edge_dist, length_px]
            p_streak = float(clf.predict_proba([feats])[0, 1])

            row_data = {
                "frame_name": fname,
                "candidate_id": cid,
                "centroid_row": round(cy, 2),
                "centroid_col": round(cx, 2),
                "bbox_r0": r0,
                "bbox_c0": c0,
                "bbox_r1": r1,
                "bbox_c1": c1,
                "width_ratio": round(width_ratio, 3),
                "snr_mean": round(snr_mean, 3),
                "sat_frac": round(sat_frac, 4),
                "straightness": round(straightness, 3),
                "angle_from_vertical": round(angle_from_vertical, 2),
                "aspect": round(aspect, 2),
                "edge_dist": round(edge_dist, 1),
                "length_px": round(length_px, 1),
                "p_streak": round(p_streak, 4),
            }
            rows.append(row_data)

            if p_streak >= 0.5:
                call_path = calls_dir / f"{stem}_{cid}.png"
                save_call_png(row_data, sub_sigma, call_path)

        df_out = pd.DataFrame(rows, columns=csv_cols)
        df_out.to_csv(csv_path, index=False)

        n_sel = len(df_out)
        n_p5 = int((df_out["p_streak"] >= 0.5).sum()) if n_sel > 0 else 0
        print(f"{fname}: selected={n_sel}, p>=0.5={n_p5}")

        # Figure 2: outputs/g1/faint_<name>.png
        h_down = h_img // 4
        w_down = w_img // 4
        sub_4x4 = sub[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).mean(axis=(1, 3))
        smoothed = gaussian_filter(sub_4x4, sigma=1.5)
        vmin2, vmax2 = np.percentile(smoothed, [20, 99.8])

        fig_w = min(11.0, w_down / 100.0)
        fig_h = fig_w * (h_down / w_down)
        fig_faint, ax_faint = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
        ax_faint.imshow(smoothed, cmap="gray", origin="lower", extent=[0, w_img, 0, h_img], vmin=vmin2, vmax=vmax2)

        for r_entry in rows:
            if r_entry["p_streak"] >= 0.5:
                bx = patches.Rectangle(
                    (r_entry["bbox_c0"], r_entry["bbox_r0"]),
                    r_entry["bbox_c1"] - r_entry["bbox_c0"],
                    r_entry["bbox_r1"] - r_entry["bbox_r0"],
                    linewidth=1.2,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_faint.add_patch(bx)
                ax_faint.text(
                    r_entry["bbox_c1"] + 10,
                    r_entry["bbox_r1"],
                    str(r_entry["candidate_id"]),
                    color="red",
                    fontsize=8,
                    fontweight="bold",
                )

        ax_faint.set_xlim(0, w_img)
        ax_faint.set_ylim(0, h_img)
        ax_faint.set_title(f"Faint view: {fname} (p>=0.5 calls: {n_p5})", fontsize=10)
        plt.tight_layout()
        plt.savefig(outputs_g1_dir / f"faint_{stem}.png", bbox_inches="tight", dpi=100)
        plt.close(fig_faint)

    # 4. Regenerate review_NN.png sheets from ALL data/g1/frame_*.csv
    all_csv_paths = sorted(g1_dir.glob("frame_20*.csv"))
    all_review_candidates = []
    total_selected_all = 0
    total_p5_all = 0
    zero_call_frames_all = []

    for c_path in all_csv_paths:
        df_c = pd.read_csv(c_path)
        total_selected_all += len(df_c)
        p5_rows = df_c[df_c["p_streak"] >= 0.5]
        n_p5 = len(p5_rows)
        total_p5_all += n_p5
        fn = str(df_c["frame_name"].iloc[0]) if len(df_c) > 0 else c_path.stem.replace("frame_", "") + ".fits"
        if n_p5 == 0:
            zero_call_frames_all.append(fn)

        for _, r in p5_rows.iterrows():
            item_dict = r.to_dict()
            s_stem = Path(str(item_dict["frame_name"])).stem.lower()
            fits_path = path_by_stem.get(s_stem, search_root / item_dict["frame_name"])
            item_dict["fpath"] = fits_path
            item_dict["stem"] = Path(str(item_dict["frame_name"])).stem
            all_review_candidates.append(item_dict)

    all_review_candidates = sorted(all_review_candidates, key=lambda x: (str(x["frame_name"]), int(x["candidate_id"])))

    # Clear old review sheets
    for old_rev in outputs_g1_dir.glob("review_*.png"):
        try:
            old_rev.unlink()
        except Exception:
            pass

    n_rev = len(all_review_candidates)
    if n_rev > 0:
        cands_per_file = 6
        n_files = int(np.ceil(n_rev / cands_per_file))

        for file_idx in range(n_files):
            batch = all_review_candidates[file_idx * cands_per_file : (file_idx + 1) * cands_per_file]
            fig_rev, axes_rev = plt.subplots(len(batch), 2, figsize=(14, 4 * len(batch)), dpi=100)
            if len(batch) == 1:
                axes_rev = np.array([axes_rev])

            for row_i, cand_info in enumerate(batch):
                fn_stem = cand_info["stem"]
                if fn_stem not in frame_cache:
                    d_raw, _ = load_fits(cand_info["fpath"])
                    bg_raw, std_raw = estimate_background(d_raw, method="background2d")
                    sub_raw = d_raw - bg_raw
                    std_map_raw = std_raw if isinstance(std_raw, np.ndarray) else np.full_like(sub_raw, std_raw)
                    sub_sig = sub_raw / np.clip(std_map_raw, 1e-6, None)
                    frame_cache[fn_stem] = sub_sig
                    if len(frame_cache) > 4:
                        frame_cache.pop(next(iter(frame_cache)))

                sub_sig = frame_cache[fn_stem]
                h_f, w_f = sub_sig.shape

                # Ensure call png exists as well
                call_path = calls_dir / f"{fn_stem}_{int(cand_info['candidate_id'])}.png"
                if not call_path.exists():
                    save_call_png(cand_info, sub_sig, call_path)

                # (a) Full frame downscaled to <= 600 px with candidate bbox in red
                scale_ds = max(1, int(np.ceil(max(h_f, w_f) / 600.0)))
                h_ds, w_ds = h_f // scale_ds, w_f // scale_ds
                frame_ds = sub_sig[:h_ds * scale_ds, :w_ds * scale_ds].reshape(h_ds, scale_ds, w_ds, scale_ds).mean(axis=(1, 3))
                vmin_a, vmax_a = np.percentile(frame_ds, [5, 99.5])

                ax_a = axes_rev[row_i, 0]
                ax_a.imshow(frame_ds, cmap="gray", origin="lower", extent=[0, w_f, 0, h_f], vmin=vmin_a, vmax=vmax_a)
                rect_a = patches.Rectangle(
                    (cand_info["bbox_c0"], cand_info["bbox_r0"]),
                    cand_info["bbox_c1"] - cand_info["bbox_c0"],
                    cand_info["bbox_r1"] - cand_info["bbox_r0"],
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_a.add_patch(rect_a)
                ax_a.set_xlim(0, w_f)
                ax_a.set_ylim(0, h_f)

                # (b) Crop in sigma units (-1 to +5)
                if cand_info["length_px"] < 100.0:
                    crop, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub_sig, cand_info["centroid_row"], cand_info["centroid_col"], 300)
                else:
                    crop_800, (c0_cr, c1_cr, r0_cr, r1_cr) = get_crop(sub_sig, cand_info["centroid_row"], cand_info["centroid_col"], 800)
                    h_8, w_8 = crop_800.shape
                    crop = crop_800[:h_8 // 2 * 2, :w_8 // 2 * 2].reshape(h_8 // 2, 2, w_8 // 2, 2).mean(axis=(1, 3))

                ax_b = axes_rev[row_i, 1]
                ax_b.imshow(crop, cmap="gray", origin="lower", extent=[c0_cr, c1_cr, r0_cr, r1_cr], vmin=-1.0, vmax=5.0)
                rect_b = patches.Rectangle(
                    (cand_info["bbox_c0"], cand_info["bbox_r0"]),
                    cand_info["bbox_c1"] - cand_info["bbox_c0"],
                    cand_info["bbox_r1"] - cand_info["bbox_r0"],
                    linewidth=1.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax_b.add_patch(rect_b)
                ax_b.set_xlim(c0_cr, c1_cr)
                ax_b.set_ylim(r0_cr, r1_cr)

                title_text = (
                    f"{cand_info['frame_name']} | ID {cand_info['candidate_id']} | "
                    f"p={cand_info['p_streak']:.3f} | L={cand_info['length_px']:.1f}px | SNR={cand_info['snr_mean']:.2f}"
                )
                ax_a.set_title(title_text, fontsize=9.5, fontweight="bold")
                ax_b.set_title("Crop (-1 to +5 sigma)", fontsize=9)

            plt.tight_layout()
            plt.savefig(outputs_g1_dir / f"review_{file_idx + 1:02d}.png", bbox_inches="tight", dpi=100)
            plt.close(fig_rev)

    # 5. Final summary from ALL data/g1/frame_*.csv
    n_proc_all = len(all_csv_paths)
    mean_calls_all = (total_p5_all / n_proc_all) if n_proc_all > 0 else 0.0
    print("\n=== SUMMARY ===")
    print(f"Frames processed: {n_proc_all}")
    print(f"Total selected: {total_selected_all}")
    print(f"Total with p >= 0.5: {total_p5_all}")
    print(f"Mean per frame: {mean_calls_all:.2f}")
    print(f"Frames with zero calls ({len(zero_call_frames_all)}): {zero_call_frames_all}")


if __name__ == "__main__":
    main()
