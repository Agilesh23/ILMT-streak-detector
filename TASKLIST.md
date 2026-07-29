# ILMT Streak Detector V3 - Task List

- [x] A1. Set up V3 project folder, carry over proven modules from V2
- [x] A2. Sanity-check carried-over code runs on 2-3 real frames
- [x] B1. Confirm detection+stitching handles both large and faint streaks on a quick sample
- [x] C1. Build/refine labeling workflow (cutout images + CSV) for manual labeling
- [x] C2-prep. Generate labeling batch 2 script (scripts/generate_labeling_batch2.py) for 6 new frames
- [x] C2-DIAG. Create missed-segments diagnostic script (scripts/diagnose_missed_segments.py) to analyze candidate truncation root cause
- [ ] C2. (User) Manually label a batch of candidates (batch 1 & batch 2)
- [x] C3. Integrate long-streak rectification fix into pipeline (+ trim residual gradient) - Folded weighted TLS line refit, 15px trim margin, & exact zero-margin boundary ray clipping into src/characterize.py. Aligned sky-patch extraction.
- [x] C4. Fix stitching gap tolerance for long faint streaks - Updated src/config.yaml (min_snr_floor=3.0, min_connected_pct=70.0, max_pair_distance=5500.0, angle_tolerance_deg=20.0); verified stitching fix & TDI safety.
- [/] C5. Regenerate batch 2 labeling images/CSV with corrected stitching - Created scripts/regenerate_labeling_batch2.py (backs up old CSV to batch_2_labels_v1_prestitchfix.csv).
- [ ] D1. Extract minimal effective feature set for labeled candidates
- [ ] D2. Train RandomForest classifier, report metrics
- [ ] D3. Validate classifier against known tricky cases (TDI artifact, faint streak)
- [ ] E1. Run trained model on remaining unknown frames, output detections catalog
- [ ] E2. (User) Spot-check results
