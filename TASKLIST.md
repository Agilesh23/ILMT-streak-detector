<!-- TASK G10: .gitignore, checks, push to main -->
# ILMT Streak Detector V3 - Task List

## DONE
- [x] A1-A3 Project setup, V2 modules carried over
- [x] B1 Labeling batch generation workflow
- [x] B2 Long-streak TLS refit, 15px end-trim, boundary clipping
- [x] C4 Stitching investigation closed: min_snr_floor has no effect (stay at 20)
- [x] C5 CCD edge mask (X<3950) adopted and wired into the pipeline
- [x] C6 Short-blob border mask (margin 15px, length <100px)
- [x] C7 Absolute selection floor (min_abs_score=47)
- [x] C8 Confirm known true streaks survived (streak 3673, 16881+14253, and the diagonal in 20230317-0008)
- [x] C9 Regenerate labeling batches 1 and 2 into *_v2 folders (keep old ones)
- [x] C10 Corridor merge accepted, growth stays off (tested on 5 known streaks and 10 FPs)
- [x] C2 Label all cutouts as Streak / False Positive (include junk and bleed trails) - 10 streak / 81 fp of 91 labeled across batches 1 & 2
- [x] Anchor labels with location in CSVs and configuration snapshot (commit fe13c8f)
- [x] Left-edge glow rule in detect.py and config.yaml (commit 94a7ffa)
- [x] Long-thin bypass in detect.py and config.yaml
- [x] C11 Add group_id column to label CSVs (commit 09085fd)
- [x] Batch 3 labeling (commit 09085fd)
- [x] Bleed veto in detect.py and config.yaml (commit 7663be8)
- [x] D0a Angle fix: bbox angle cannot tell lean direction; known limit
- [x] D0b-diag RANSAC centroid line fit captures 100% of labeled fragments on Set A, finds 8 suspect lines on negatives
- [x] D0b-ref Reference strips confirm real streaks show continuous core, false lines show disjoint stars
- [x] D0b-score Whole-line z-score separates real from false lines across full frame crossing
- [x] D0b-score2 Peak-offset search (+-20px) improves z-score to 11.2-46.0 on real streaks vs 0.7-3.9 on false lines
- [x] D0b Line grouping by straight-line fit of centroids, validated along the whole line (provisional thresholds ACCEPT_Z_PEAK=6, ACCEPT_MIN_SEGMENTS=4, tested on 86 frames, in-sample)
- [x] D0b-fix1 Extent floor from member positions; raw-centroid diagnostic (49/49 labeled members on lines, 19 lines found, 11 accepted)
- [x] D0c-fix1 Fix G1 file glob in build_candidates_v2.py (glob matched frame_review.csv, which has no candidate_id)
- [x] D0c Build candidates_v2: 695 candidates (11 lines, 684 singles), 44 streak / 23 fp labeled
- [x] G9a 1 duplicate pair found in candidates_v2 (270/10323)
- [x] G9b Both new lines visually confirmed as streaks (wide soft bands, 35-40 px)
- [x] G9c Clean reference images saved
- [x] G9d Audit script could not detect mislabels (no lines in frames with streak singles); only 270 re-checked by eye and confirmed streak
- [x] G9 Dedupe: moot, only pair was 270/10323; 10323 is an unrelated object (relabeled fp via label_overrides.csv)
- [x] D1 Feature extraction for candidates_v2 -> data/d1/features_v2.csv
- [x] D1a Refine single-candidate axis, fix saturation threshold, fix wide-band baseline

## NOW (data and labeling)
> Note: 8 physical streaks in 8 of 9 frames (20221101-0030 has none), 15 streak rows.
- [ ] G9e Reusable visualizer scripts/viz_check.py (IN PROGRESS)
- [ ] G9g Interactive review tool scripts/label_check.py (IN PROGRESS)
- [ ] G9h Fix n key in label_check.py (input() fails inside Qt key callback) (IN PROGRESS)
- [ ] G9i Inventory existing visualizer/labeller; decide which to use for all review (IN PROGRESS)
- [ ] G9j Review batch for label_tool.py from candidates_v2.csv (IN PROGRESS)
- [ ] G9k Blind review batch: no p/score/label on images (IN PROGRESS)
- [ ] D1b Add profile_snr_wing feature (IN PROGRESS)
- [ ] D2 Random Forest, leave-one-frame-out (IN PROGRESS)
- [ ] D1c Refinement fails on faint/wide streaks (6 of 33 confirmed singles have peak_snr < 0.35) (IN PROGRESS)
- [ ] G10 Push to GitHub main (IN PROGRESS)

## FEATURES AND MODEL
- [ ] D1 Feature extraction: straightness, width vs local star FWHM, profile uniformity, PCA angle from vertical, distance to frame edges, saturated-pixel fraction, group extent; no raw length. Key = (frame_name, candidate_id)
- [ ] G3 Saturated-star features: distance to nearest saturated star, angle vs scan axis (to reject bleed trails)
- [ ] G3b Streak width compared with local star width (real streaks match star FWHM)
- [ ] G4 Injection-recovery: paste fake streaks (a star profile trailed along a line, SNR 3-100, varied length and angle) into real frames; measure how often we recover them at each brightness; add them as extra training positives
- [ ] G7 Test light smoothing for detection only (features measured on unsmoothed image); keep only if recall at low SNR improves.
- [ ] D2 Train Random Forest
- [ ] D3 Validate by leave-one-group-out, not random splits

## VALIDATION AT SCALE
- [ ] G1 Run on more of the 166 public frames: streaks found and false positives per frame
- [ ] G2 Final precision/recall on labeled real frames
- [ ] G6 Compare our counts and precision with the earlier study's 78-83 confirmed streaks on the same 166 frames (counts and precision only)

## LATER
- [ ] E1 Extended length by long-window integration checked against parallel control lines
- [ ] Calibration output (astrometry and photometry) per streak
- [ ] Final report and documentation in docs/

## NOTES
- Suggested order: D0, G9, D1 with G3 and G3b, G4, D2, D3, then G1, G2, G6
- Earlier stitching FP evaluation assumed 20230317-0008 and 20221026-0057 had no streaks; both do.
- Single-candidate boxes: a streak's box diagonal is its axis; D1 must test both diagonals (D0a limit: lean direction unknown).
- Wide soft-band streaks exist (20230615-0006_L1, 20240502-0026_L1): width is a feature, never a hard reject.
- Do not train on min_edge_dist_px or x_over_3950 (length proxies), saturated_frac (all zero), group_extent_px or n_segments (lines only).
