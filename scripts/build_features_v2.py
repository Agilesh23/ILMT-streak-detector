import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import map_coordinates, generate_binary_structure, binary_dilation, gaussian_filter, distance_transform_edt
from skimage.measure import label as sk_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_FEATURES = [
    "width_ratio",
    "ridge_rms",
    "profile_cv",
    "snr_mean",
    "angle_from_vertical",
    "sat_frac",
    "line_z",
]
print(f"MODEL_FEATURES = {MODEL_FEATURES}")

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
    "20240405-0004.fits",
]


def run_line_search(search_img, cx, cy, H_img, W_img, H_down, W_down):
    thetas = np.arange(180, dtype=np.float64)
    theta_rad = np.radians(thetas)[:, None]
    s = np.arange(-500, 501, 4, dtype=np.float64)[None, :]

    xs_full = cx + s * np.cos(theta_rad)
    ys_full = cy + s * np.sin(theta_rad)

    in_frame = (xs_full >= 0) & (xs_full < W_img) & (ys_full >= 0) & (ys_full < H_img)
    angle_in_frame = np.all(in_frame, axis=1)

    xs_down = np.clip((xs_full / 4.0).astype(int), 0, W_down - 1)
    ys_down = np.clip((ys_full / 4.0).astype(int), 0, H_down - 1)

    samples = search_img[ys_down, xs_down]
    valid = ~np.isnan(samples)
    valid_frac = np.mean(valid, axis=1)

    best_val = -1e9
    best_angle = None

    for i in range(180):
        if not angle_in_frame[i]:
            continue
        if valid_frac[i] < 0.6:
            continue
        val = float(np.mean(samples[i, valid[i]]))
        if val > best_val:
            best_val = val
            best_angle = i

    if best_angle is None:
        return None, 0
    return float(best_val), int(best_angle)


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

    # Build search image (once per frame)
    h_down = h_img // 4
    w_down = w_img // 4
    sub_4x4 = sub[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).mean(axis=(1, 3))
    smoothed = gaussian_filter(sub_4x4, sigma=1.5)
    med = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - med)))
    search_img = (smoothed - med) / (1.4826 * max(mad, 1e-6))

    # Mask saturated cores
    sat_full = (data >= 0.9 * frame_max)
    if np.any(sat_full):
        dist_sat_full = distance_transform_edt(~sat_full)
    else:
        dist_sat_full = np.full_like(data, 1e5, dtype=np.float32)

    dist_sat_down = dist_sat_full[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).min(axis=(1, 3))
    is_near_sat = (dist_sat_down <= 40.0)
    nan_mask = (search_img > 5.0) & is_near_sat
    search_img[nan_mask] = np.nan

    # Controls: 40 random centres per frame (seeded)
    control_bests = []
    rng = np.random.default_rng(seed=42)
    attempts = 0
    while len(control_bests) < 40 and attempts < 10000:
        attempts += 1
        cr = rng.integers(100, h_img - 100)
        cc = rng.integers(100, w_img - 100)
        if dist_sat_full[cr, cc] < 150.0:
            continue
        c_best, _ = run_line_search(search_img, float(cc), float(cr), h_img, w_img, h_down, w_down)
        if c_best is not None:
            control_bests.append(c_best)

    ctrl_arr = np.array(control_bests)
    med_ctrl = float(np.median(ctrl_arr))
    mad_ctrl = float(np.median(np.abs(ctrl_arr - med_ctrl)))
    norm_mad = float(1.4826 * max(mad_ctrl, 1e-6))

    # Binary mask and candidate detection
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

    frame_data_cache[fn] = {
        "data": data,
        "sub": sub,
        "sub_sigma": sub_sigma,
        "labeled_map": labeled_map,
        "selected": selected,
        "frame_max": frame_max,
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

        # Line search for candidate
        cand_best, cand_line_angle = run_line_search(search_img, cx, cy, h_img, w_img, h_down, w_down)
        if cand_best is not None:
            line_z = float((cand_best - med_ctrl) / norm_mad)
            line_angle = float(cand_line_angle)
        else:
            line_z = 0.0
            line_angle = 0.0

        # PCA orientation and length
        cov = np.cov(np.vstack([dx, dy]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        vec = eigvecs[:, 0]
        u_xp, u_yp = float(vec[0]), float(vec[1])
        if u_xp < 0 or (abs(u_xp) < 1e-7 and u_yp < 0):
            u_xp, u_yp = -u_xp, -u_yp
        pca_angle = float(np.degrees(np.arctan2(u_yp, u_xp))) % 180.0

        t_proj_pca = dx * u_xp + dy * u_yp
        length_px_pca = float(np.max(t_proj_pca) - np.min(t_proj_pca))

        # 2. axis_angle selection
        if length_px_pca >= 100.0:
            axis_angle = pca_angle
        else:
            axis_angle = line_angle

        rad = np.radians(axis_angle)
        u_x = float(np.cos(rad))
        u_y = float(np.sin(rad))
        perp_x = -u_y
        perp_y = u_x

        t_proj = dx * u_x + dy * u_y
        t_min = float(np.min(t_proj))
        t_max = float(np.max(t_proj))
        length_px = float(t_max - t_min)

        p0 = (cx + t_min * u_x, cy + t_min * u_y)
        p1 = (cx + t_max * u_x, cy + t_max * u_y)

        # Aspect
        lam0 = max(0.0, float(eigvals[0]))
        lam1 = max(0.0, float(eigvals[1]))
        aspect = float(np.sqrt(lam0 / max(lam1, 1e-6)))

        # angle_from_vertical using axis_angle
        ang_h = axis_angle if axis_angle <= 90.0 else axis_angle - 180.0
        angle_from_vertical = float(abs(90.0 - abs(ang_h)))

        # Edge distance and saturation fraction
        edge_dist = float(min(r0, c0, h_img - r1, w_img - c1))
        pix_raw = data[coords[:, 0], coords[:, 1]]
        sat_frac = float(np.mean(pix_raw >= 0.9 * frame_max))

        # 3. ridge_rms
        step_r = max(5.0, length_px / 20.0)
        t_steps_r = np.arange(t_min, t_max + 0.01, step_r)
        w_cross = np.arange(-15, 16)
        offsets = []
        for t in t_steps_r:
            xc = cx + t * u_x
            yc = cy + t * u_y
            sample_xs = xc + w_cross * perp_x
            sample_ys = yc + w_cross * perp_y
            prof = map_coordinates(sub, np.vstack([sample_ys, sample_xs]), order=1, mode="constant", cval=0.0)
            p95 = np.percentile(prof, 95)
            prof_clean = np.minimum(prof, p95)
            base = np.percentile(prof_clean, 20)
            weights = np.maximum(0.0, prof_clean - base)
            tot = np.sum(weights)
            if tot > 0:
                offset = float(np.sum(w_cross * weights) / tot)
                offsets.append(offset)

        if len(offsets) >= 4:
            ridge_rms = float(np.sqrt(np.mean(np.array(offsets) ** 2)))
        else:
            ridge_rms = np.nan

        # Rectified patch for width_ratio and profile_cv
        box_w = 40
        patch_len = max(10, int(np.ceil(length_px)))
        try:
            patch, _ = extract_streak_patch(sub, p0, p1, box_width=box_w, output_length=patch_len)
        except Exception:
            patch = np.zeros((box_w, patch_len), dtype=np.float32)

        # width_ratio
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

        # profile_cv
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

        if len(step_means) >= 4 and np.mean(step_means) > 0:
            profile_cv = float(np.std(step_means) / np.mean(step_means))
        else:
            profile_cv = np.nan

        # snr_mean along-line
        transverse_offsets = np.arange(-4, 5)
        t_steps_snr = np.linspace(t_min, t_max, max(10, int(length_px / 2.0)))
        snrs = []
        for t in t_steps_snr:
            xc = cx + t * u_x
            yc = cy + t * u_y
            sample_xs = xc + transverse_offsets * perp_x
            sample_ys = yc + transverse_offsets * perp_y
            vals = map_coordinates(sub_sigma, np.vstack([sample_ys, sample_xs]), order=1, mode="constant", cval=0.0)
            snrs.append(float(np.median(vals)))
        snr_mean = float(np.mean(snrs)) if snrs else 0.0

        feat_row = {
            "frame_name": fn,
            "candidate_id": cid,
            "label": label_val,
            "group_id": group_val,
            "n_members": n_members,
            "ridge_rms": round(ridge_rms, 3) if not np.isnan(ridge_rms) else np.nan,
            "aspect": round(aspect, 2),
            "width_ratio": round(width_ratio, 3) if not np.isnan(width_ratio) else np.nan,
            "profile_cv": round(profile_cv, 4) if not np.isnan(profile_cv) else np.nan,
            "snr_mean": round(snr_mean, 3),
            "angle_from_vertical": round(angle_from_vertical, 2),
            "edge_dist": round(edge_dist, 1),
            "sat_frac": round(sat_frac, 4),
            "length_px": round(length_px, 1),
            "line_z": round(line_z, 3),
            "line_angle": round(line_angle, 1),
            "axis_angle": round(axis_angle, 1),
            "_p0": p0,
            "_p1": p1,
            "_cx": cx,
            "_cy": cy,
            "_cand_mask": cand_mask,
        }
        all_features.append(feat_row)

df_features = pd.DataFrame(all_features)
export_cols = [
    "frame_name", "candidate_id", "label", "group_id", "n_members",
    "ridge_rms", "aspect", "width_ratio", "profile_cv", "snr_mean",
    "angle_from_vertical", "edge_dist", "sat_frac", "length_px",
    "line_z", "line_angle", "axis_angle",
]
csv_out = features_dir / "features_v2.csv"
df_features[export_cols].to_csv(csv_out, index=False)

# 5. Build review figure: features_check_v2.png (same 12 candidates as v1)
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

fig, axes = plt.subplots(len(panels_to_plot), 3, figsize=(18, 48), dpi=100)
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
        "width_ratio": f["width_ratio"],
        "ridge_rms": f["ridge_rms"],
        "profile_cv": f["profile_cv"],
        "snr_mean": f["snr_mean"],
        "angle_from_vertical": f["angle_from_vertical"],
        "sat_frac": f["sat_frac"],
        "line_z": f["line_z"],
        "line_angle": f["line_angle"],
        "axis_angle": f["axis_angle"],
    })

    fd = frame_data_cache[fn]
    sub_sigma = fd["sub_sigma"]
    cx, cy = f["_cx"], f["_cy"]
    cand_mask = f["_cand_mask"]
    p0, p1 = f["_p0"], f["_p1"]

    title_str = f"ID: {cid} ({f['label']}) | z={f['line_z']:.2f} | line_ang={f['line_angle']:.1f}° | axis_ang={f['axis_angle']:.1f}°"

    # Column (a): 120x120 crop, sigma units, fixed stretch -1 to +5, no overlay
    crop120, (c0_120, c1_120, r0_120, r1_120) = get_crop(sub_sigma, cy, cx, 120)
    ax_a = axes[idx, 0]
    ax_a.imshow(crop120, cmap="gray", origin="lower", extent=[c0_120, c1_120, r0_120, r1_120], vmin=-1.0, vmax=5.0)
    ax_a.set_title(title_str, fontsize=9.5, fontweight="bold")
    ax_a.set_xlim(c0_120, c1_120)
    ax_a.set_ylim(r0_120, r1_120)

    # Column (b): same crop smoothed gaussian sigma 2, stretch -0.5 to +2, mask contour (cyan), axis_angle line (orange)
    crop120_smooth = gaussian_filter(crop120, sigma=2.0)
    ax_b = axes[idx, 1]
    ax_b.imshow(crop120_smooth, cmap="gray", origin="lower", extent=[c0_120, c1_120, r0_120, r1_120], vmin=-0.5, vmax=2.0)
    mask_crop, _ = get_crop(cand_mask.astype(np.float32), cy, cx, 120)
    if np.any(mask_crop > 0.5):
        xs = np.arange(c0_120, c1_120) + 0.5
        ys = np.arange(r0_120, r1_120) + 0.5
        X, Y = np.meshgrid(xs, ys)
        ax_b.contour(X, Y, mask_crop, levels=[0.5], colors=["cyan"], linewidths=1.2)
    ax_b.plot([p0[0], p1[0]], [p0[1], p1[1]], color="orange", lw=1.8)
    ax_b.set_title("Smoothed (σ=2) + Mask & Axis", fontsize=8.5)
    ax_b.set_xlim(c0_120, c1_120)
    ax_b.set_ylim(r0_120, r1_120)

    # Column (c): 1000x1000 crop with search segment at line_angle in yellow
    crop1000, (c0_1000, c1_1000, r0_1000, r1_1000) = get_crop(sub_sigma, cy, cx, 1000)
    vmin3, vmax3 = np.percentile(crop1000, [5, 99.5])
    ax_c = axes[idx, 2]
    ax_c.imshow(crop1000, cmap="gray", origin="lower", extent=[c0_1000, c1_1000, r0_1000, r1_1000], vmin=vmin3, vmax=vmax3)
    l_rad = np.radians(f["line_angle"])
    lu_x = np.cos(l_rad)
    lu_y = np.sin(l_rad)
    p_start = (cx - 500 * lu_x, cy - 500 * lu_y)
    p_end = (cx + 500 * lu_x, cy + 500 * lu_y)
    ax_c.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="yellow", lw=2.0)
    ax_c.set_title("1000x1000 + ±500px Line Angle", fontsize=8.5)
    ax_c.set_xlim(c0_1000, c1_1000)
    ax_c.set_ylim(r0_1000, r1_1000)

plt.tight_layout()
plt.savefig(review_dir / "features_check_v2.png", bbox_inches="tight", dpi=100)
plt.close()

# 6. Print required outputs
df_12 = pd.DataFrame(table_rows)
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 1000)
print("\n=== 12-ROW FEATURE TABLE ===")
print(df_12.to_string(index=False))

df_streaks = df_features[df_features["label"] == "streak"][["frame_name", "candidate_id", "line_z", "line_angle", "axis_angle"]].copy()
df_streaks["frame"] = df_streaks["frame_name"].str.replace(".fits", "", regex=False)
cols_streak = ["frame", "candidate_id", "line_z", "line_angle", "axis_angle"]
print(f"\n=== LINE_Z FOR ALL {len(df_streaks)} STREAK ROWS ===")
print(df_streaks[cols_streak].to_string(index=False))

df_fp = df_features[df_features["label"] == "fp"][["frame_name", "candidate_id", "line_z", "line_angle"]].copy()
df_fp["frame"] = df_fp["frame_name"].str.replace(".fits", "", regex=False)
df_fp_top10 = df_fp.sort_values(by="line_z", ascending=False).head(10)
cols_fp = ["frame", "candidate_id", "line_z", "line_angle"]
print("\n=== TOP 10 HIGHEST LINE_Z AMONG FP ROWS ===")
print(df_fp_top10[cols_fp].to_string(index=False))
