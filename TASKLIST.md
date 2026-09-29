# ILMT Streak Detector V3 - Task List

- [x] A1. Set up V3 project folder, carry over proven modules from V2
- [x] A2. Sanity-check carried-over code runs on 2-3 real frames
- [x] B1. Confirm detection+stitching handles both large and faint streaks on a quick sample
- [x] C1. Build/refine labeling workflow (cutout images + CSV) for manual labeling
- [ ] C2. (User) Manually label a batch of candidates (batch 1 & batch 2)
- [x] C3. Integrate long-streak rectification fix into pipeline (+ trim residual gradient) - Folded weighted TLS line refit, 15px trim margin, & exact zero-margin boundary ray clipping (margin_px=0.0) into src/characterize.py. Aligned sky-patch extraction.
- [x] C4. CCD edge artifact investigation - Edge mask X<3950 adopted in detect.py; min_snr_floor tested (3/5/10/20), no effect on FP count with mask active.
- [ ] D1. Extract minimal effective feature set for labeled candidates
- [ ] D2. Train RandomForest classifier, report metrics
- [ ] D3. Validate classifier against known tricky cases (TDI artifact, faint streak)
- [ ] E1. Run trained model on remaining unknown frames, output detections catalog
- [ ] E2. (User) Spot-check results
