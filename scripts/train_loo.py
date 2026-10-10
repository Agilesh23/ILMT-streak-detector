import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

base_dir = Path(__file__).resolve().parent.parent
data_dir = base_dir / "data"
features_dir = data_dir / "features"
review_dir = base_dir / "outputs/review"
review_dir.mkdir(parents=True, exist_ok=True)
frame_dir = Path(r"C:\Users\agile\Downloads\Project\ILMT_Contest_Materials\test_files")

sys.path.insert(0, str(base_dir / "src"))
from io_utils import load_fits
from background import estimate_background

# Load data
df = pd.read_csv(features_dir / "features_v1.csv")
y_true = (df["label"] == "streak").astype(int).values
frames = df["frame_name"].unique()

m1_features = [
    "width_ratio",
    "snr_mean",
    "sat_frac",
    "straightness",
    "angle_from_vertical",
    "aspect",
    "edge_dist",
    "length_px",
]
m2_features = [
    "width_ratio",
    "snr_mean",
    "sat_frac",
    "straightness",
    "angle_from_vertical",
]

# Load labels for centroids and bboxes
df1 = pd.read_csv(data_dir / "labeled/batch_1_v2_labels.csv")
df2 = pd.read_csv(data_dir / "labeled/batch_2_v2_labels.csv")
df3 = pd.read_csv(data_dir / "labeled/batch_3_v2_labels.csv")
df_labels = pd.concat([df1, df2, df3]).reset_index(drop=True)

# Grid for Model R
L_grid = [50, 100, 200, 400, 800]
A_grid = [5, 10, 20, 40]

oof_pred_R = np.zeros(len(df), dtype=int)
oof_pred_M1 = np.zeros(len(df), dtype=int)
oof_prob_M1 = np.zeros(len(df), dtype=float)
oof_pred_M2 = np.zeros(len(df), dtype=int)
oof_prob_M2 = np.zeros(len(df), dtype=float)


def calc_metrics(y_t, y_p):
    tp = int(np.sum((y_p == 1) & (y_t == 1)))
    fp = int(np.sum((y_p == 1) & (y_t == 0)))
    fn = int(np.sum((y_p == 0) & (y_t == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1


# Leave-one-frame-out cross-validation
for held_out_frame in frames:
    val_mask = (df["frame_name"] == held_out_frame).values
    train_mask = ~val_mask

    y_train = y_true[train_mask]
    y_val = y_true[val_mask]

    # Model R: grid search on training fold
    best_f1 = -1.0
    best_L, best_A = 100, 10
    train_len = df.loc[train_mask, "length_px"].values
    train_asp = df.loc[train_mask, "aspect"].values

    for L in L_grid:
        for A in A_grid:
            pred_r = ((train_len >= L) & (train_asp >= A)).astype(int)
            _, _, _, _, _, f1_r = calc_metrics(y_train, pred_r)
            if f1_r > best_f1:
                best_f1 = f1_r
                best_L = L
                best_A = A

    val_len = df.loc[val_mask, "length_px"].values
    val_asp = df.loc[val_mask, "aspect"].values
    oof_pred_R[val_mask] = ((val_len >= best_L) & (val_asp >= best_A)).astype(int)

    # Model M1
    X_train_m1 = df.loc[train_mask, m1_features].values
    X_val_m1 = df.loc[val_mask, m1_features].values
    clf_m1 = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42)
    clf_m1.fit(X_train_m1, y_train)
    p_m1 = clf_m1.predict_proba(X_val_m1)[:, 1]
    oof_prob_M1[val_mask] = p_m1
    oof_pred_M1[val_mask] = (p_m1 >= 0.5).astype(int)

    # Model M2
    X_train_m2 = df.loc[train_mask, m2_features].values
    X_val_m2 = df.loc[val_mask, m2_features].values
    clf_m2 = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42)
    clf_m2.fit(X_train_m2, y_train)
    p_m2 = clf_m2.predict_proba(X_val_m2)[:, 1]
    oof_prob_M2[val_mask] = p_m2
    oof_pred_M2[val_mask] = (p_m2 >= 0.5).astype(int)

# Train M1 on all data for feature importances
clf_full_m1 = RandomForestClassifier(n_estimators=300, class_weight="balanced", min_samples_leaf=2, random_state=42)
clf_full_m1.fit(df[m1_features].values, y_true)

# 1. Print pooled performance
mask_long = (df["length_px"] >= 100.0).values

print("=== 1. MODEL PERFORMANCE (LEAVE-ONE-FRAME-OUT) ===")
for name, preds in [("R", oof_pred_R), ("M1", oof_pred_M1), ("M2", oof_pred_M2)]:
    tp, fp, fn, p, r, f1 = calc_metrics(y_true, preds)
    tp_l, fp_l, fn_l, p_l, r_l, f1_l = calc_metrics(y_true[mask_long], preds[mask_long])
    tp_s, fp_s, fn_s, p_s, r_s, f1_s = calc_metrics(y_true[~mask_long], preds[~mask_long])
    print(
        f"Model {name:2s} | All: TP={tp:2d} FP={fp:2d} FN={fn:2d} Prec={p:.3f} Rec={r:.3f} F1={f1:.3f} | "
        f">=100px: TP={tp_l:2d} FP={fp_l:2d} FN={fn_l:2d} Prec={p_l:.3f} Rec={r_l:.3f} F1={f1_l:.3f} | "
        f"<100px: TP={tp_s:2d} FP={fp_s:2d} FN={fn_s:2d} Prec={p_s:.3f} Rec={r_s:.3f} F1={f1_s:.3f}"
    )

# 2. Per-frame breakdown for M1
print("\n=== 2. M1 PER-FRAME BREAKDOWN ===")
for fn in frames:
    idx = (df["frame_name"] == fn).values
    tp, fp, fn_cnt, _, _, _ = calc_metrics(y_true[idx], oof_pred_M1[idx])
    fn_clean = fn.split(".")[0]
    print(f"{fn_clean:14s}: TP={tp:2d}, FP={fp:2d}, FN={fn_cnt:2d}")

# 3. Misclassifications for M1
print("\n=== 3. M1 MISCLASSIFICATIONS ===")
wrong_mask = (oof_pred_M1 != y_true)
df_wrong = df[wrong_mask].copy()
df_wrong["p_streak"] = oof_prob_M1[wrong_mask]
df_wrong["pred"] = oof_pred_M1[wrong_mask]

for _, row in df_wrong.iterrows():
    fn_clean = row["frame_name"].split(".")[0]
    print(
        f"Frame: {fn_clean}, ID: {int(row['candidate_id'])}, Label: {row['label']}, "
        f"p(streak): {row['p_streak']:.3f}, length_px: {row['length_px']:.1f}"
    )

# 4. Feature importances for M1 (top 5)
print("\n=== 4. M1 TOP-5 FEATURE IMPORTANCES ===")
importances = clf_full_m1.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(5, len(m1_features))):
    idx = sorted_idx[i]
    print(f"{i + 1}. {m1_features[idx]:20s}: {importances[idx]:.4f} ({importances[idx] * 100:.1f}%)")

# Save outputs/review/loo_errors.png
n_errors = min(16, len(df_wrong))
if n_errors > 0:
    ncols = min(4, n_errors)
    nrows = int(np.ceil(n_errors / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=100)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    frame_cache = {}
    for i, (_, row) in enumerate(df_wrong.head(16).iterrows()):
        fn = row["frame_name"]
        cid = int(row["candidate_id"])
        true_label = row["label"]
        p = row["p_streak"]
        pred_label = "streak" if p >= 0.5 else "fp"

        if fn not in frame_cache:
            data, _ = load_fits(frame_dir / fn)
            bg, std = estimate_background(data, method="background2d")
            sub = data - bg
            std_map = std if isinstance(std, np.ndarray) else np.full_like(sub, std)
            sub_sigma = sub / np.clip(std_map, 1e-6, None)
            frame_cache[fn] = sub_sigma

        sub_sigma = frame_cache[fn]
        h_img, w_img = sub_sigma.shape

        # Lookup centroid and bbox
        m_label = df_labels[(df_labels["frame_name"] == fn) & (df_labels["candidate_id"] == cid)]
        if len(m_label) > 0:
            cy = float(m_label.iloc[0]["centroid_row"])
            cx = float(m_label.iloc[0]["centroid_col"])
            r0 = int(m_label.iloc[0]["bbox_r0"])
            c0 = int(m_label.iloc[0]["bbox_c0"])
            r1 = int(m_label.iloc[0]["bbox_r1"])
            c1 = int(m_label.iloc[0]["bbox_c1"])
        else:
            cy, cx = h_img / 2.0, w_img / 2.0
            r0, c0, r1, c1 = 0, 0, 0, 0

        # 300x300 crop centred on centroid
        half = 150
        r_start = int(round(cy)) - half
        r_end = r_start + 300
        c_start = int(round(cx)) - half
        c_end = c_start + 300

        r0_cl = max(0, r_start)
        r1_cl = min(h_img, r_end)
        c0_cl = max(0, c_start)
        c1_cl = min(w_img, c_end)

        sub_crop = sub_sigma[r0_cl:r1_cl, c0_cl:c1_cl]
        pad_top = max(0, -r_start)
        pad_bottom = max(0, r_end - h_img)
        pad_left = max(0, -c_start)
        pad_right = max(0, c_end - w_img)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            crop = np.pad(sub_crop, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)
        else:
            crop = sub_crop

        r_ax, c_ax = divmod(i, ncols)
        ax = axes[r_ax, c_ax]
        ax.imshow(crop, cmap="gray", origin="lower", extent=[c_start, c_end, r_start, r_end], vmin=-1.0, vmax=5.0)

        # Draw bbox
        rect = patches.Rectangle(
            (c0, r0),
            c1 - c0,
            r1 - r0,
            linewidth=1.5,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.set_xlim(c_start, c_end)
        ax.set_ylim(r_start, r_end)

        fn_short = fn.split(".")[0]
        ax.set_title(f"{fn_short} {cid} {true_label}->{pred_label} p={p:.2f}", fontsize=8.5, fontweight="bold")

    # Turn off unused subplots
    for j in range(n_errors, nrows * ncols):
        r_ax, c_ax = divmod(j, ncols)
        axes[r_ax, c_ax].axis("off")

    plt.tight_layout()
    plt.savefig(review_dir / "loo_errors.png", bbox_inches="tight", dpi=100)
    plt.close()
