# TASK D2: Random Forest with leave-one-frame-out
"""
Random Forest classification and leave-one-frame-out validation.

1. Reads data/d1/features_v2.csv.
2. Trains and validates on labeled rows (label in ['streak', 'fp']).
3. Feature Set A:
     - width_fwhm_px
     - width_vs_star
     - profile_uniformity
     - straightness
     - log1p(peak_snr)
     - log1p(profile_snr_wing)
     - pca_angle_from_vertical
     - dist_nearest_saturated_px
     - width_missing (flag)
4. Feature Set B: Set A + min_edge_dist_px, x_over_3950 (length-proxy effect).
5. Validation: Leave-One-Frame-Out (LOFO) by frame_name.
   Imputes NaNs using training-fold medians only.
6. Scores all 695 rows:
   - Out-of-fold p for labeled rows.
   - Predictions from model trained on all labeled rows for unlabeled rows.
7. Saves data/d2/scores_v2.csv.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / "data" / "d1" / "features_v2.csv"
OUTPUT_DIR = BASE_DIR / "data" / "d2"
SCORES_CSV = OUTPUT_DIR / "scores_v2.csv"


# ── Feature Preparation ───────────────────────────────────────────────────────
def prepare_feature_data(df: pd.DataFrame):
    df_feat = df.copy()

    # Numerical transformations
    df_feat["width_fwhm_px"] = pd.to_numeric(df_feat["width_fwhm_px"], errors="coerce")
    df_feat["width_vs_star"] = pd.to_numeric(df_feat["width_vs_star"], errors="coerce")
    df_feat["profile_uniformity"] = pd.to_numeric(df_feat["profile_uniformity"], errors="coerce")
    df_feat["straightness"] = pd.to_numeric(df_feat["straightness"], errors="coerce")

    raw_peak = pd.to_numeric(df_feat["peak_snr"], errors="coerce")
    df_feat["log_peak_snr"] = np.where(raw_peak.isna(), np.nan, np.log1p(np.maximum(0.0, raw_peak)))

    raw_wing = pd.to_numeric(df_feat["profile_snr_wing"], errors="coerce")
    df_feat["log_profile_snr_wing"] = np.where(raw_wing.isna(), np.nan, np.log1p(np.maximum(0.0, raw_wing)))

    df_feat["pca_angle_from_vertical"] = pd.to_numeric(df_feat["pca_angle_from_vertical"], errors="coerce")
    df_feat["dist_nearest_saturated_px"] = pd.to_numeric(df_feat["dist_nearest_saturated_px"], errors="coerce")
    df_feat["width_missing"] = df_feat["width_fwhm_px"].isna().astype(float)

    # Set B extra features
    df_feat["min_edge_dist_px"] = pd.to_numeric(df_feat["min_edge_dist_px"], errors="coerce")
    df_feat["x_over_3950"] = df_feat["x_over_3950"].astype(str).str.lower().isin(["true", "1"]).astype(float)

    return df_feat


# ── Leave-One-Frame-Out Cross-Validation ──────────────────────────────────────
def run_lofo_cv(df_labeled: pd.DataFrame, feature_cols: list[str], target_col: str = "is_streak"):
    frames = df_labeled["frame_name"].unique()
    oof_preds = np.full(len(df_labeled), np.nan, dtype=float)

    for test_frame in frames:
        train_idx = df_labeled.index[df_labeled["frame_name"] != test_frame].tolist()
        test_idx = df_labeled.index[df_labeled["frame_name"] == test_frame].tolist()

        train_sub = df_labeled.loc[train_idx]
        test_sub = df_labeled.loc[test_idx]

        # Training fold medians for imputation
        medians = train_sub[feature_cols].median()
        # Fallback to 0 if all values in train fold are NaN
        medians = medians.fillna(0.0)

        X_train = train_sub[feature_cols].fillna(medians).values
        y_train = train_sub[target_col].values

        X_test = test_sub[feature_cols].fillna(medians).values

        clf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)

        # Predict probability for streak class (1)
        if 1 in clf.classes_:
            class_1_idx = int(np.where(clf.classes_ == 1)[0][0])
            p_test = clf.predict_proba(X_test)[:, class_1_idx]
        else:
            p_test = np.zeros(len(test_sub))

        # Assign back into oof array
        loc_positions = df_labeled.index.get_indexer(test_idx)
        oof_preds[loc_positions] = p_test

    return oof_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("TASK D2: Random Forest Classification with Leave-One-Frame-Out Validation")
    print("=" * 80)

    if not FEATURES_CSV.exists():
        print(f"Error: {FEATURES_CSV} not found.")
        sys.exit(1)

    df_raw = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df_raw)} total candidates across {df_raw['frame_name'].nunique()} frames.")

    df_full = prepare_feature_data(df_raw)

    # Filter labeled rows
    df_full["label_clean"] = df_full["label"].astype(str).str.lower().str.strip()
    labeled_mask = df_full["label_clean"].isin(["streak", "fp"])
    df_labeled = df_full[labeled_mask].copy().reset_index(drop=True)
    df_labeled["is_streak"] = (df_labeled["label_clean"] == "streak").astype(int)

    n_streak = int((df_labeled["is_streak"] == 1).sum())
    n_fp = int((df_labeled["is_streak"] == 0).sum())
    n_frames_labeled = df_labeled["frame_name"].nunique()
    print(f"Labeled subset: {len(df_labeled)} candidates ({n_streak} streak, {n_fp} fp) across {n_frames_labeled} frames.\n")

    # Feature sets
    cols_A = [
        "width_fwhm_px",
        "width_vs_star",
        "profile_uniformity",
        "straightness",
        "log_peak_snr",
        "log_profile_snr_wing",
        "pca_angle_from_vertical",
        "dist_nearest_saturated_px",
        "width_missing",
    ]

    cols_B = cols_A + ["min_edge_dist_px", "x_over_3950"]

    # ── 1. LOFO Validation: Feature Set A ─────────────────────────────────────
    print("Running Leave-One-Frame-Out Validation for Feature Set A...")
    oof_A = run_lofo_cv(df_labeled, cols_A)
    auc_A = roc_auc_score(df_labeled["is_streak"], oof_A)
    preds_A = (oof_A >= 0.5).astype(int)
    cm_A = confusion_matrix(df_labeled["is_streak"], preds_A, labels=[0, 1])
    prec_A = precision_score(df_labeled["is_streak"], preds_A, zero_division=0)
    rec_A = recall_score(df_labeled["is_streak"], preds_A, zero_division=0)

    # ── 2. LOFO Validation: Feature Set B (length proxy effect) ───────────────
    print("Running Leave-One-Frame-Out Validation for Feature Set B...")
    oof_B = run_lofo_cv(df_labeled, cols_B)
    auc_B = roc_auc_score(df_labeled["is_streak"], oof_B)
    preds_B = (oof_B >= 0.5).astype(int)
    cm_B = confusion_matrix(df_labeled["is_streak"], preds_B, labels=[0, 1])
    prec_B = precision_score(df_labeled["is_streak"], preds_B, zero_division=0)
    rec_B = recall_score(df_labeled["is_streak"], preds_B, zero_division=0)

    # ── 3. Performance Comparison Table ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS (Leave-One-Frame-Out, threshold p = 0.5)")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Feature Set A':<22} | {'Feature Set B (with length proxies)':<30}")
    print("-" * 80)
    print(f"{'Pooled ROC AUC':<25} | {auc_A:<22.4f} | {auc_B:<30.4f}")
    print(f"{'Precision (p >= 0.5)':<25} | {prec_A:<22.4f} | {prec_B:<30.4f}")
    print(f"{'Recall (p >= 0.5)':<25} | {rec_A:<22.4f} | {rec_B:<30.4f}")
    tn_a, fp_a, fn_a, tp_a = cm_A.ravel()
    tn_b, fp_b, fn_b, tp_b = cm_B.ravel()
    print(f"{'Confusion Matrix':<25} | TP={tp_a:<3} FP={fp_a:<3} FN={fn_a:<3} TN={tn_a:<3} | TP={tp_b:<3} FP={fp_b:<3} FN={fn_b:<3} TN={tn_b:<3}")
    print("-" * 80)

    # ── 4. Detailed Misclassification Audit for Set A ────────────────────────
    df_labeled["oof_p_A"] = oof_A

    missed_streaks = df_labeled[(df_labeled["is_streak"] == 1) & (df_labeled["oof_p_A"] < 0.5)]
    false_positives = df_labeled[(df_labeled["is_streak"] == 0) & (df_labeled["oof_p_A"] >= 0.5)]

    print(f"\nLabeled Streaks with Out-Of-Fold p < 0.5 ({len(missed_streaks)} candidates):")
    print("-" * 75)
    if not missed_streaks.empty:
        for _, r in missed_streaks.iterrows():
            print(f"  Frame: {r['frame_name']:<20} | ID: {str(r['candidate_id']):<8} | Kind: {r['kind']:<7} | OOF p: {r['oof_p_A']:.4f}")
    else:
        print("  None (100% streak recall at p = 0.5)!")

    print(f"\nFalse Positives with Out-Of-Fold p >= 0.5 ({len(false_positives)} candidates):")
    print("-" * 75)
    if not false_positives.empty:
        for _, r in false_positives.iterrows():
            print(f"  Frame: {r['frame_name']:<20} | ID: {str(r['candidate_id']):<8} | Kind: {r['kind']:<7} | OOF p: {r['oof_p_A']:.4f}")
    else:
        print("  None (0 false alarms at p = 0.5)!")

    # ── 5. Feature Importances for Model A Fitted on All Labeled Rows ─────────
    med_all_labeled = df_labeled[cols_A].median().fillna(0.0)
    X_all_labeled = df_labeled[cols_A].fillna(med_all_labeled).values
    y_all_labeled = df_labeled["is_streak"].values

    clf_A_full = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=0,
        n_jobs=-1,
    )
    clf_A_full.fit(X_all_labeled, y_all_labeled)

    importances = clf_A_full.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCES (Model A fitted on all labeled rows)")
    print("=" * 80)
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:2d}. {cols_A[idx]:<28}: {importances[idx]:.4f}")
    print("-" * 80)

    # ── 6. Score All 695 Rows with Set A ──────────────────────────────────────
    print("\nScoring all 695 candidates with Feature Set A...")

    # Assign out-of-fold p to labeled rows; for unlabeled rows, predict via full model
    final_p = np.full(len(df_full), np.nan, dtype=float)

    # Map out-of-fold predictions back to full dataframe
    labeled_indices_full = df_full.index[labeled_mask].tolist()
    final_p[labeled_indices_full] = oof_A

    # Unlabeled rows scored with clf_A_full
    unlabeled_mask = ~labeled_mask
    if np.any(unlabeled_mask):
        unlabeled_sub = df_full[unlabeled_mask]
        X_unlabeled = unlabeled_sub[cols_A].fillna(med_all_labeled).values
        class_1_idx = int(np.where(clf_A_full.classes_ == 1)[0][0])
        p_unlabeled = clf_A_full.predict_proba(X_unlabeled)[:, class_1_idx]
        unlabeled_indices_full = df_full.index[unlabeled_mask].tolist()
        final_p[unlabeled_indices_full] = p_unlabeled

    df_full["p_streak"] = final_p

    # Compute rank_in_frame (1 = highest p_streak in frame)
    df_full["rank_in_frame"] = df_full.groupby("frame_name")["p_streak"].rank(
        ascending=False, method="min"
    ).astype(int)

    # Save to data/d2/scores_v2.csv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cols = ["frame_name", "candidate_id", "kind", "label", "p_streak", "rank_in_frame"]
    df_full[out_cols].to_csv(SCORES_CSV, index=False)
    print(f"Saved {len(df_full)} scored rows to: {SCORES_CSV}")

    # ── 7. Top 30 Unlabeled Candidates with Highest p_streak ──────────────────
    df_unlabeled = df_full[unlabeled_mask].copy()
    top_30_unlabeled = df_unlabeled.sort_values(by="p_streak", ascending=False).head(30)

    print("\n" + "=" * 80)
    print("TOP 30 UNLABELED CANDIDATES (Highest p_streak)")
    print("=" * 80)
    for rank, (_, r) in enumerate(top_30_unlabeled.iterrows(), 1):
        print(f"  {rank:2d}. Frame: {r['frame_name']:<20} | ID: {str(r['candidate_id']):<8} | Kind: {r['kind']:<7} | p_streak: {r['p_streak']:.4f} (Rank in frame: {r['rank_in_frame']})")
    print("-" * 80)

    # ── 8. Frame-Level Rank-1 Diagnostics across 86 Frames ────────────────────
    total_frames = df_full["frame_name"].nunique()

    # Find rank-1 candidate for each frame (or candidates sharing rank 1)
    df_rank1 = df_full[df_full["rank_in_frame"] == 1]
    rank1_by_frame = df_rank1.groupby("frame_name").first()

    n_rank1_streak = int((rank1_by_frame["label_clean"] == "streak").sum())
    n_rank1_unlabeled_ge_05 = int(
        ((rank1_by_frame["label_clean"] == "unlabeled") & (rank1_by_frame["p_streak"] >= 0.5)).sum()
    )

    print("\nFRAME-LEVEL RANK-1 SUMMARY:")
    print("-" * 60)
    print(f"  Total frames analyzed                        : {total_frames}")
    print(f"  Frames with rank-1 candidate labeled 'streak': {n_rank1_streak} / {total_frames} ({100.0 * n_rank1_streak / total_frames:.1f}%)")
    print(f"  Frames with rank-1 candidate unlabeled & p>=0.5: {n_rank1_unlabeled_ge_05} / {total_frames} ({100.0 * n_rank1_unlabeled_ge_05 / total_frames:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
