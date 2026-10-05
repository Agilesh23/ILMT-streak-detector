import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.measure import label as sk_label, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# Load inputs
df1 = pd.read_csv(data_dir / "labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "labeled/batch_3_v2_labels.csv")
df_labels = pd.concat([df1, df2, df3]).reset_index(drop=True)
df_labels["frame_clean"] = df_labels["frame_name"].astype(str).str.replace(".fits", "", regex=False)

df_v2 = pd.read_csv(features_dir / "features_v2.csv")
df_v2["frame_clean"] = df_v2["frame_name"].astype(str).str.replace(".fits", "", regex=False)

# Grid for +/-500 px segment sampled every 4 px in 21 blocks of 48 px (12 samples each)
block_centers = np.arange(-10, 11) * 48.0  # (21,)
sample_offsets_in_block = np.arange(-22, 23, 4.0)  # (12,)
s_grid = block_centers[:, None] + sample_offsets_in_block[None, :]  # (21, 12)


def eval_segment(search_img, cx, cy, theta_deg, h_img, w_img, h_down, w_down):
    rad = np.radians(theta_deg)
    cos_t = np.cos(rad)
    sin_t = np.sin(rad)

    xs = cx + s_grid * cos_t
    ys = cy + s_grid * sin_t

    in_frame = (xs >= 0) & (xs < w_img) & (ys >= 0) & (ys < h_img)

    xs_down = np.clip((xs / 4.0).astype(int), 0, w_down - 1)
    ys_down = np.clip((ys / 4.0).astype(int), 0, h_down - 1)

    vals = search_img[ys_down, xs_down]
    valid = in_frame & (~np.isnan(vals))

    n_valid = np.sum(valid, axis=1)
    block_valid = (n_valid >= 6)  # >= 50% of 12 samples

    if np.sum(block_valid) < 10:
        return None, None, int(np.sum(block_valid))

    block_means = np.full(21, np.nan)
    for b in range(21):
        if block_valid[b]:
            block_means[b] = float(np.mean(vals[b, valid[b]]))

    stat = float(np.median(block_means[block_valid]))
    return stat, block_means, int(np.sum(block_valid))


def get_crop(img, cy_down, cx_down, size_down, pad_val=np.nan):
    half = size_down // 2
    r0 = int(round(cy_down)) - half
    r1 = r0 + size_down
    c0 = int(round(cx_down)) - half
    c1 = c0 + size_down

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
        crop = np.pad(sub_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=pad_val)
    else:
        crop = sub_crop
    return crop


plot_targets = [
    ("20230317-0008.fits", 3618),
    ("20230602-0022.fits", 11716),
    ("20221026-0057.fits", 1175),
    ("20230519-0009.fits", 3673),
    ("20230519-0009.fits", 10540),
    ("20221025-0028.fits", 18693),
]
plot_data = {}

perp_offsets = [-500, -400, -300, -200, -160, -120, -80, -40, 40, 80, 120, 160, 200, 300, 400, 500]

unique_frames = df_v2["frame_name"].unique()
results = []

for fn in unique_frames:
    fpath = frame_dir / fn
    data, header = load_fits(fpath)
    bg, std = estimate_background(data, method="background2d")
    sub = data - bg
    std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
    frame_max = float(np.max(data))
    h_img, w_img = data.shape
    h_down = h_img // 4
    w_down = w_img // 4

    # Star mask from UNBINNED sub
    binary_stars = (sub > 5.0 * std_map)
    labeled_stars = sk_label(binary_stars)
    regions = regionprops(labeled_stars, intensity_image=data)
    star_mask_unbinned = np.zeros((h_img, w_img), dtype=bool)

    for r in regions:
        major = r.major_axis_length
        minor = r.minor_axis_length
        ratio = (minor / major) if major > 0 else 1.0

        if ratio < 0.33:
            continue

        is_round = (ratio > 0.5)
        has_sat = (r.max_intensity >= 0.9 * frame_max)

        if is_round or has_sat:
            cy_s, cx_s = r.centroid
            radius = max(25.0, 3.0 * np.sqrt(r.area / np.pi))
            r_min = max(0, int(np.floor(cy_s - radius)))
            r_max = min(h_img, int(np.ceil(cy_s + radius)) + 1)
            c_min = max(0, int(np.floor(cx_s - radius)))
            c_max = min(w_img, int(np.ceil(cx_s + radius)) + 1)
            ys, xs = np.ogrid[r_min:r_max, c_min:c_max]
            star_mask_unbinned[r_min:r_max, c_min:c_max] |= ((ys - cy_s) ** 2 + (xs - cx_s) ** 2 <= radius ** 2)

    # Search image
    sub_4x4 = sub[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).mean(axis=(1, 3))
    smoothed = gaussian_filter(sub_4x4, sigma=1.5)
    med = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - med)))
    search_img = (smoothed - med) / (1.4826 * max(mad, 1e-6))

    star_mask_4x4 = star_mask_unbinned[:h_down * 4, :w_down * 4].reshape(h_down, 4, w_down, 4).any(axis=(1, 3))
    search_img[star_mask_4x4] = np.nan

    # Controls for length_px < 100: 40 random centres per frame (seeded)
    control_maxima = []
    rng = np.random.default_rng(seed=42)
    attempts = 0
    while len(control_maxima) < 40 and attempts < 10000:
        attempts += 1
        cr = rng.integers(100, h_img - 100)
        cc = rng.integers(100, w_img - 100)
        if star_mask_unbinned[cr, cc]:
            continue

        best_ctrl_stat = -1e9
        has_valid_angle = False
        for theta in range(180):
            c_stat, _, _ = eval_segment(search_img, float(cc), float(cr), theta, h_img, w_img, h_down, w_down)
            if c_stat is not None and c_stat > best_ctrl_stat:
                best_ctrl_stat = c_stat
                has_valid_angle = True

        if has_valid_angle:
            control_maxima.append(best_ctrl_stat)

    ctrl_arr = np.array(control_maxima) if len(control_maxima) > 0 else np.array([0.0])
    med_ctrl_short = float(np.median(ctrl_arr))
    mad_ctrl_short = float(np.median(np.abs(ctrl_arr - med_ctrl_short)))
    scale_short = float(max(1.4826 * mad_ctrl_short, 0.05))

    frame_cands = df_v2[df_v2["frame_name"] == fn]

    for _, row in frame_cands.iterrows():
        cid = int(row["candidate_id"])
        length_px = float(row["length_px"])
        axis_angle = float(row["axis_angle"])
        label_val = str(row["label"])

        m = df_labels[(df_labels["frame_name"] == fn) & (df_labels["candidate_id"] == cid)]
        if len(m) > 0:
            stored_cy = float(m.iloc[0]["centroid_row"])
            stored_cx = float(m.iloc[0]["centroid_col"])
        else:
            stored_cy = float(h_img / 2.0)
            stored_cx = float(w_img / 2.0)

        if length_px >= 100.0:
            angle = axis_angle
            stat, block_means, n_valid = eval_segment(search_img, stored_cx, stored_cy, angle, h_img, w_img, h_down, w_down)

            rad = np.radians(angle)
            perp_x = -np.sin(rad)
            perp_y = np.cos(rad)

            ctrl_stats = []
            for d in perp_offsets:
                cx_c = stored_cx + d * perp_x
                cy_c = stored_cy + d * perp_y
                c_stat, _, _ = eval_segment(search_img, cx_c, cy_c, angle, h_img, w_img, h_down, w_down)
                if c_stat is not None:
                    ctrl_stats.append(c_stat)

            if len(ctrl_stats) > 0:
                c_arr = np.array(ctrl_stats)
                ctrl_med = float(np.median(c_arr))
                ctrl_mad = float(np.median(np.abs(c_arr - ctrl_med)))
                scale = float(max(1.4826 * ctrl_mad, 0.05))
            else:
                ctrl_med = 0.0
                scale = 0.05

            if stat is not None:
                line_z = float((stat - ctrl_med) / scale)
            else:
                line_z = 0.0
                stat = 0.0
                block_means = np.full(21, np.nan)
                n_valid = 0

            line_angle = angle
        else:
            best_stat = -1e9
            best_angle = 0
            best_bmeans = None
            best_n_valid = 0

            for theta in range(180):
                s_val, bmeans, n_val = eval_segment(search_img, stored_cx, stored_cy, theta, h_img, w_img, h_down, w_down)
                if s_val is not None and s_val > best_stat:
                    best_stat = s_val
                    best_angle = theta
                    best_bmeans = bmeans
                    best_n_valid = n_val

            ctrl_med = med_ctrl_short
            scale = scale_short

            if best_bmeans is not None:
                stat = best_stat
                block_means = best_bmeans
                n_valid = best_n_valid
                line_z = float((stat - ctrl_med) / scale)
                line_angle = float(best_angle)
            else:
                stat = 0.0
                block_means = np.full(21, np.nan)
                n_valid = 0
                line_z = 0.0
                line_angle = 0.0

        res_entry = {
            "frame_name": fn,
            "candidate_id": cid,
            "label": label_val,
            "line_z": round(line_z, 3),
            "line_angle": round(line_angle, 1),
        }
        results.append(res_entry)

        if (fn, cid) in plot_targets:
            plot_data[(fn, cid)] = {
                "cx": stored_cx,
                "cy": stored_cy,
                "line_z": line_z,
                "line_angle": line_angle,
                "stat": stat,
                "ctrl_med": ctrl_med,
                "scale": scale,
                "block_means": block_means,
                "n_valid": n_valid,
                "label": label_val,
                "search_img": search_img,
                "star_mask_4x4": star_mask_4x4,
            }

df_out = pd.DataFrame(results)
export_cols = ["frame_name", "candidate_id", "line_z", "line_angle"]
df_out[export_cols].to_csv(features_dir / "linez_v3.csv", index=False)

# Save outputs/review/linez_v3_check.png
fig, axes = plt.subplots(len(plot_targets), 2, figsize=(14, 24), dpi=100)

for idx, target_key in enumerate(plot_targets):
    p = plot_data.get(target_key)
    if p is None:
        continue

    fn, cid = target_key
    cx, cy = p["cx"], p["cy"]
    line_angle = p["line_angle"]
    stat = p["stat"]
    ctrl_med = p["ctrl_med"]
    scale = p["scale"]
    block_means = p["block_means"]
    n_valid = p["n_valid"]
    label_val = p["label"]
    search_img = p["search_img"]
    star_mask_4x4 = p["star_mask_4x4"]

    c0_full = cx - 500.0
    c1_full = cx + 500.0
    r0_full = cy - 500.0
    r1_full = cy + 500.0

    # Col 1: 1000x1000 crop (250x250 in search_img)
    search_crop = get_crop(search_img, cy / 4.0, cx / 4.0, 250, pad_val=np.nan)
    mask_crop = get_crop(star_mask_4x4.astype(float), cy / 4.0, cx / 4.0, 250, pad_val=0.0)

    ax1 = axes[idx, 0]
    ax1.imshow(search_crop, cmap="gray", origin="lower", extent=[c0_full, c1_full, r0_full, r1_full], vmin=-0.5, vmax=3.0)

    mask_rgba = np.zeros((search_crop.shape[0], search_crop.shape[1], 4), dtype=np.float32)
    mask_rgba[mask_crop > 0.5, 0] = 1.0  # Red
    mask_rgba[mask_crop > 0.5, 3] = 0.35  # Translucent
    ax1.imshow(mask_rgba, origin="lower", extent=[c0_full, c1_full, r0_full, r1_full])

    rad = np.radians(line_angle)
    p0 = (cx - 500.0 * np.cos(rad), cy - 500.0 * np.sin(rad))
    p1 = (cx + 500.0 * np.cos(rad), cy + 500.0 * np.sin(rad))
    ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], color="yellow", lw=2.0)
    ax1.set_xlim(c0_full, c1_full)
    ax1.set_ylim(r0_full, r1_full)

    title_str = f"ID: {cid} | {label_val} | z={p['line_z']:.2f} | angle={line_angle:.1f}° | valid blocks={n_valid}/21"
    ax1.set_title(title_str, fontsize=10, fontweight="bold")

    # Col 2: bar plot of block means
    ax2 = axes[idx, 1]
    for b in range(21):
        if not np.isnan(block_means[b]):
            ax2.bar(block_centers[b], block_means[b], width=40.0, color="steelblue", edgecolor="navy", align="center")

    ax2.axhline(stat, color="crimson", lw=2.0, ls="--", label=f"Stat: {stat:.2f}")
    ax2.axhspan(ctrl_med - 3.0 * scale, ctrl_med + 3.0 * scale, color="gray", alpha=0.25, label=f"Ctrl med ± 3σ ({scale:.2f})")
    ax2.axhline(ctrl_med, color="black", lw=1.0, ls=":", label=f"Ctrl med: {ctrl_med:.2f}")

    ax2.set_xlim(-520, 520)
    ax2.set_xlabel("Offset along segment (px)", fontsize=8.5)
    ax2.set_ylabel("Block Mean (σ)", fontsize=8.5)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.set_title(f"Block Means (stat={stat:.2f})", fontsize=10)

plt.tight_layout()
plt.savefig(review_dir / "linez_v3_check.png", bbox_inches="tight", dpi=100)
plt.close()

# Print required outputs
row_3618 = df_out[(df_out["candidate_id"] == 3618) & (df_out["frame_name"].str.contains("20230317-0008"))]
z_3618 = float(row_3618.iloc[0]["line_z"]) if len(row_3618) > 0 else -999.0

df_streaks = df_out[df_out["label"] == "streak"]
df_fps = df_out[df_out["label"] == "fp"]

max_fp_z = float(df_fps["line_z"].max())
failing_streaks = df_streaks[df_streaks["line_z"] < 2.0]
is_3618_above_all_fp = (z_3618 > max_fp_z)
no_streak_below_2 = (len(failing_streaks) == 0)

if is_3618_above_all_fp and no_streak_below_2:
    status_str = f"PASS: line_z of 3618 ({z_3618:.3f}) > max fp ({max_fp_z:.3f}) and all 14 streaks >= 2.0"
else:
    reasons = []
    if not is_3618_above_all_fp:
        reasons.append(f"3618 line_z ({z_3618:.3f}) <= max fp ({max_fp_z:.3f})")
    if not no_streak_below_2:
        fail_list = [f"{r['frame_name'].split('.')[0]}:{r['candidate_id']}(z={r['line_z']:.2f})" for _, r in failing_streaks.iterrows()]
        reasons.append(f"failing streaks (<2.0): {fail_list}")
    status_str = f"FAIL: {'; '.join(reasons)}"

print(status_str)

print("\n=== ALL 14 STREAK ROWS ===")
print(df_streaks[["frame_name", "candidate_id", "line_z", "line_angle"]].to_string(index=False))

print("\n=== TOP 10 HIGHEST FP ROWS ===")
df_top10_fp = df_fps.sort_values(by="line_z", ascending=False).head(10)
print(df_top10_fp[["frame_name", "candidate_id", "line_z", "line_angle"]].to_string(index=False))
