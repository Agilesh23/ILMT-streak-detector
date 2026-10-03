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

## NOW (data and labeling)
> Note: 8 physical streaks in 8 of 9 frames (20221101-0030 has none), 15 streak rows.
- [ ] D0 Fix candidate definition (raw blob vs post-stitch) so training and inference use the same objects; see the level report
- [ ] G9 Dedupe candidates that cover the same streak (5768/3673, 888/12982) before scoring

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
