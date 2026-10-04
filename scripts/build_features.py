import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation
from skimage.measure import label as sk_label, regionprops
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

base_dir = Path(__file__).resolve().parent.parent
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")
data_dir = base_dir / "data"
features_dir = data_dir / "features"
features_dir.mkdir(parents=True, exist_ok=True)
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background
from detect import find_blob_candidates, stitch_collinear_fragments, rank_and_select_candidates
from characterize import width_estimation, extract_streak_patch

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

# Load all 99 labeled rows
df1 = pd.read_csv(data_dir / "labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "labeled/batch_3_v2_labels.csv")
df_all = pd.concat([df1, df2, df3]).reset_index(drop=True)
df_all["group_id"] = df_all["group_id"].fillna("")

frames = [
    "20221025-0028.fits",
    "20230519-0009.fits",
    "20230317-0008.fits",
    "20221026-0057.fits",
    "20221101-0030.fits",
    "20230602-0022.fits",
    "20231115-0003.fits",
    "20240114-0028.fits",
    "20240405-0004.fits"
]

all_features = []
selected_keys = set()
frame_data_cache = {}

for fn in frames:
    fpath = frame_dir / fn
    data, header = load_fits(fpath)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
    sub_sigma = sub / np.clip(std_map, 1e-6, None)
    frame_max = float(np.max(data))
    h_img, w_img = data.shape

    # Reconstruct labeled binary mask of blobs
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
        bleed_veto=bleed_veto_cfg
    )
    stitched = stitch_collinear_fragments(candidates, data, bg, std)
    selected, _ = rank_and_select_candidates(stitched, min_abs_score=min_abs_score, drop_bleed_trails=True)

    frame_data_cache[fn] = {
        "data": data,
        "sub": sub,
        "sub_sigma": sub_sigma,
        "labeled_map": labeled_map,
        "selected": selected,
        "frame_max": frame_max
    }

    df_frame = df_all[df_all["frame_name"] == fn]

    for cand in selected:
        cid = int(cand["label"])
        selected_keys.add((fn, cid))

        match_row = df_frame[df_frame["candidate_id"] == cid]
        if len(match_row) > 0:
            row_item = match_row.iloc[0]
            label_val = row_item["label"]
            group_val = row_item["group_id"]
        else:
            label_val = "unlabeled"
            group_val = ""

        members = cand.get("stitched_from", [cid])
        n_members = len(members)

        # Merged pixel set for stitched candidate
        cand_mask = np.isin(labeled_map, members)
        coords = np.argwhere(cand_mask)  # (row, col)

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

        # 1. Straightness: RMS perpendicular distance of mask pixels to PCA line
        d_perp = dx * perp_x + dy * perp_y
        straightness = float(np.sqrt(np.mean(d_perp ** 2)))

        # 2. Aspect: PCA major / minor
        lam0 = max(0.0, float(eigvals[0]))
        lam1 = max(0.0, float(eigvals[1]))
        aspect = float(np.sqrt(lam0 / max(lam1, 1e-6)))

        # 3. Angle from vertical (PCA)
        ang_h = float(np.degrees(np.arctan2(u_y, u_x)))
        if ang_h > 90.0: ang_h -= 180.0
        elif ang_h < -90.0: ang_h += 180.0
        angle_from_vertical = float(abs(90.0 - abs(ang_h)))

        # 4. Length and projection along line
        t_proj = dx * u_x + dy * u_y
        t_min = float(np.min(t_proj))
        t_max = float(np.max(t_proj))
        length_px = float(t_max - t_min)

        p0 = (cx + t_min * u_x, cy + t_min * u_y)
        p1 = (cx + t_max * u_x, cy + t_max * u_y)

        # 5. Edge distance
        edge_dist = float(min(r0, c0, h_img - r1, w_img - c1))

        # 6. Saturated fraction
        pix_raw = data[coords[:, 0], coords[:, 1]]
        sat_frac = float(np.mean(pix_raw >= 0.9 * frame_max))

        # 7. Rectified patch for width_ratio and along-line profiles
        box_w = 40
        patch_len = max(10, int(np.ceil(length_px)))
        try:
            patch, _ = extract_streak_patch(sub, p0, p1, box_width=box_w, output_length=patch_len)
        except Exception:
            patch = np.zeros((box_w, patch_len), dtype=np.float32)

        # Width ratio
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

        # 8. Profile CV (50 px steps, star pixels masked)
        # Star mask: mask pixels > 95th percentile per column/row
        masked_patch = patch.copy()
        p95 = np.percentile(patch, 95)
        masked_patch[masked_patch > p95] = p95

        step_len = 50
        n_steps = max(1, int(np.ceil(patch_len / step_len)))
        step_means = []
        for s in range(n_steps):
            col_s = s * step_len
            col_e = min(patch_len, (s + 1) * step_len)
            block = masked_patch[:, col_s:col_e]
            if block.size > 0:
                step_means.append(float(np.mean(block)))

        if len(step_means) >= 2 and np.mean(step_means) > 0:
            profile_cv = float(np.std(step_means) / np.mean(step_means))
        else:
            profile_cv = 0.0

        # 9. Mean along-line SNR (median across the transverse width)
        transverse_offsets = np.arange(-4, 5)
        t_steps = np.linspace(t_min, t_max, max(10, int(length_px / 2.0)))
        snrs = []
        for t in t_steps:
            xc = cx + t * u_x
            yc = cy + t * u_y
            sample_xs = xc + transverse_offsets * perp_x
            sample_ys = yc + transverse_offsets * perp_y
            vals = map_coordinates(sub_sigma, np.vstack([sample_ys, sample_xs]), order=1, mode='constant', cval=0.0)
            snrs.append(float(np.median(vals)))
        snr_mean = float(np.mean(snrs)) if snrs else 0.0

        feat_row = {
            "frame_name": fn,
            "candidate_id": cid,
            "label": label_val,
            "group_id": group_val,
            "n_members": n_members,
            "straightness": round(straightness, 3),
            "aspect": round(aspect, 2),
            "width_ratio": round(width_ratio, 3),
            "profile_cv": round(profile_cv, 4),
            "snr_mean": round(snr_mean, 3),
            "angle_from_vertical": round(angle_from_vertical, 2),
            "edge_dist": round(edge_dist, 1),
            "sat_frac": round(sat_frac, 4),
            "length_px": round(length_px, 1),
            "_p0": p0,
            "_p1": p1,
            "_cx": cx,
            "_cy": cy,
            "_r0": r0, "_r1": r1, "_c0": c0, "_c1": c1,
            "_step_means": step_means,
            "_snrs": snrs
        }
        all_features.append(feat_row)

df_features = pd.DataFrame(all_features)
export_cols = [
    "frame_name", "candidate_id", "label", "group_id", "n_members",
    "straightness", "aspect", "width_ratio", "profile_cv", "snr_mean",
    "angle_from_vertical", "edge_dist", "sat_frac", "length_px"
]
csv_out = features_dir / "features_v1.csv"
df_features[export_cols].to_csv(csv_out, index=False)

# Identify skipped labeled rows
skipped_rows = []
for idx, r in df_all.iterrows():
    key = (r["frame_name"], int(r["candidate_id"]))
    if key not in selected_keys:
        skipped_rows.append(key)

print(f"Skipped labeled rows not in final selected list ({len(skipped_rows)}): {sorted(skipped_rows)}")

# Build 12-panel review figure: 6 streak, 6 fp
panels_to_plot = [
    ("20230519-0009.fits", 3673),
    ("20231115-0003.fits", 888),
    ("20230317-0008.fits", 3618),
    ("20221026-0057.fits", 1175),
    ("20221025-0028.fits", 6527),
    ("20230602-0022.fits", 11716),
    ("20221025-0028.fits", 18693),
    ("20221026-0057.fits", 11455),
    ("20221025-0028.fits", 10277),
    ("20230519-0009.fits", 10540),
    ("20221101-0030.fits", 9983),
    ("20240405-0004.fits", 6509),
]

fig = plt.figure(figsize=(24, 18), dpi=100)
outer_grid = gridspec.GridSpec(3, 4, wspace=0.28, hspace=0.35)

table_rows = []

for idx, (fn, cid) in enumerate(panels_to_plot):
    match_f = [f for f in all_features if f["frame_name"] == fn and f["candidate_id"] == cid]
    if not match_f:
        continue
    f = match_f[0]
    table_rows.append({
        "Frame": fn,
        "ID": cid,
        "Label": f["label"],
        "Straight": f["straightness"],
        "Aspect": f["aspect"],
        "W_ratio": f["width_ratio"],
        "Prof_CV": f["profile_cv"],
        "SNR_mean": f["snr_mean"],
        "Ang_vert": f["angle_from_vertical"],
        "Edge_dist": f["edge_dist"],
        "Sat_frac": f["sat_frac"],
        "Length": f["length_px"]
    })

    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[idx], height_ratios=[2, 1], hspace=0.25)
    ax_top = fig.add_subplot(inner_grid[0])
    ax_bot = fig.add_subplot(inner_grid[1])

    fd = frame_data_cache[fn]
    sub = fd["sub"]

    margin = 40
    r0 = max(0, f["_r0"] - margin)
    r1 = min(sub.shape[0], f["_r1"] + margin)
    c0 = max(0, f["_c0"] - margin)
    c1 = min(sub.shape[1], f["_c1"] + margin)

    crop = sub[r0:r1, c0:c1]
    vmin, vmax = np.percentile(crop, [5, 99.5])
    ax_top.imshow(np.clip(crop, vmin, vmax), cmap="gray", origin="lower", extent=[c0, c1, r0, r1])

    p0 = f["_p0"]
    p1 = f["_p1"]
    line_col = "#00FF00" if f["label"] == "streak" else "#FF7F00"
    ax_top.plot([p0[0], p1[0]], [p0[1], p1[1]], color=line_col, lw=1.5, ls="--")
    ax_top.plot([p0[0], p1[0]], [p0[1], p1[1]], "o", color=line_col, markersize=4)

    title_str = (
        f"[{f['label'].upper()}] ID {cid} ({fn.split('.')[0]})\n"
        f"str={f['straightness']:.2f}px | asp={f['aspect']:.1f} | w_rat={f['width_ratio']:.2f}\n"
        f"cv={f['profile_cv']:.3f} | snr={f['snr_mean']:.2f} | ang_v={f['angle_from_vertical']:.1f}° | sat={f['sat_frac']:.3f}"
    )
    ax_top.set_title(title_str, fontsize=8.5, fontweight="bold")
    ax_top.tick_params(labelsize=7)

    # Bottom: along-line profile
    snrs = f["_snrs"]
    xs = np.linspace(0, f["length_px"], len(snrs))
    ax_bot.plot(xs, snrs, color="steelblue", lw=1.2)
    ax_bot.axhline(0, color="gray", lw=0.8, ls=":")
    ax_bot.set_ylabel("SNR", fontsize=7.5)
    ax_bot.set_xlabel("Along-line pos (px)", fontsize=7.5)
    ax_bot.tick_params(labelsize=7)

review_png = review_dir / "features_check.png"
plt.savefig(review_png, bbox_inches="tight", dpi=100)
plt.close()

# Print compact feature table for the 12 rows
df_12 = pd.DataFrame(table_rows)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)
print("\n=== FEATURE TABLE (12 PANELS) ===")
print(df_12.to_string(index=False))
