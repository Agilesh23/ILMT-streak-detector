# Status Report Protocol

Whenever I ask "where are we now", "where should we proceed", "status", or similar, write the report in plain, simple language. Explain what each item means and why it matters, and don't assume I remember earlier context. Sections:

1. Big picture (2-3 sentences): what the project is, and where we are in the plan.
2. Progress: X of Y tasks done, and which phase we are in.
3. Verified state: run git status, git log -n 5 --oneline, and scripts/sanity_check_v3.py. Report the selected candidate counts per frame and any errors. Do not modify anything.
4. Task table: every task in TASKLIST.md as DONE / IN PROGRESS / NOT STARTED / BLOCKED, each with a one-line plain explanation of what it is. For DONE items, give evidence (commit hash or file).
5. Drift check: uncommitted changes, scripts not tied to any task, tasks marked done that the repo does not support. Flag out-of-scope work.
6. Open risks: anything unresolved, in plain words (for example, "we have too few labeled examples to train reliably").
7. Next step: one recommended task. Explain what it is, why it comes next, the exact steps, and the files involved. Then name the two tasks after it.
8. Ask for approval before starting. Do not start work on your own.

Keep it structured and concise. Do not re-read files that are not needed for the report.

## Background
- The ILMT is a zenith-pointing telescope. Satellites and debris show up as thin bright streaks in its images.
- Our pipeline finds candidate streaks with classical image processing, then a Random Forest classifier (trained on hand-labeled candidates) decides Streak vs False Positive.
- Main false-positive sources: CCD edge artifacts (X>=3950), noise blobs at frame borders, chance alignments of stars, and bleed trails from saturated bright stars (a vertical line through a very bright star is not a satellite).
- The 166 public frames (9 nights, Oct 24-Nov 1 2022) are a shared dataset. An earlier published study ran a deep-learning model on them and reported 78-83 visually confirmed streaks. It did not report recall on real data, so we can compare counts and precision only.
- Saturated stars have an elongated feature along the scan direction, and they caused most false detections in that earlier study.
- A streak from a distant object is unresolved, so its width should match the width of nearby stars (FWHM about 5-9 px).
