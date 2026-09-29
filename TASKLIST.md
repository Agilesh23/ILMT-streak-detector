# ILMT Streak Detector V3 - Task List

## DONE
- [x] A1-A3 Project setup, V2 modules carried over
- [x] B1 Labeling batch generation workflow
- [x] B2 Long-streak TLS refit, 15px end-trim, boundary clipping
- [x] C4 Stitching investigation closed: min_snr_floor has no effect (stay at 20)
- [x] C5 CCD edge mask (X<3950) adopted and wired into the pipeline
- [x] C6 Short-blob border mask (margin 15px, length <100px)
- [x] C7 Absolute selection floor (min_abs_score=47)

## NOW (data and labeling)
- [ ] C8 Candidate 18693 in 20221025-0028 is a galaxy (curved, bright core), not a streak. Label it fp with note "galaxy" once the label tool exists.
- [x] C9 Regenerate labeling batches 1 and 2 into *_v2 folders (keep old ones)
- [ ] C2 Label all cutouts as Streak / False Positive (include junk and bleed trails)

## FEATURES AND MODEL
- [ ] D1 Feature extraction: length, SNR profile, linearity, flux ratio, straightness (line-fit residual), brightness profile along the length (uniform vs core-and-fade), and width vs local star FWHM, to reject galaxies
- [ ] G3 Saturated-star features: distance to nearest saturated star, angle vs scan axis (to reject bleed trails)
- [ ] G3b Streak width compared with local star width (real streaks match star FWHM)
- [ ] G4 Injection-recovery: paste fake streaks (a star profile trailed along a line, SNR 3-100, varied length and angle) into real frames; measure how often we recover them at each brightness; add them as extra training positives
- [ ] G7 Test light smoothing for detection only (features measured on unsmoothed image); keep only if recall at low SNR improves.
- [ ] D2 Train Random Forest
- [ ] D3 Validate on a held-out set: precision and recall, split by frame (not by candidate)

## VALIDATION AT SCALE
- [ ] G1 Run on more of the 166 public frames: streaks found and false positives per frame
- [ ] G2 Final precision/recall on labeled real frames
- [ ] G6 Compare our counts and precision with the earlier study's 78-83 confirmed streaks on the same 166 frames (counts and precision only)

## LATER
- [ ] Calibration output (astrometry and photometry) per streak
- [ ] Final report and documentation in docs/

## NOTES
- Suggested order: C8, C9, C2, D1 with G3 and G3b, G4, D2, D3, then G1, G2, G6
