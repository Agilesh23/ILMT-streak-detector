# Workspace Rules - ILMT Streak Detector V3

## Directory Organization & Cleanliness Rules
1. **Root Folder (`C:\Users\agile\Downloads\Project\ILMT Streak Detector V3\`)**:
   Must contain ONLY:
   - `TASKLIST.md`
   - `README.md` (if existing)
   - Configuration files (e.g. `config.yaml` if placed at root)
   - Standard directories: `data/`, `notebooks/`, `src/`, `models/`, `outputs/`, `scripts/`, `docs/`, `.agents/`
   - **NO** loose `.py` scripts, `.log` files, or temporary test scripts sitting at root.

2. **File Placement Directory Structure**:
   - `scripts/`: Diagnostic, helper, utility, or throwaway `.py` scripts (NOT production `src/` code).
   - `scripts/logs/`: Any `.log` output files or execution run logs.
   - `src/`: Production pipeline code modules (`background.py`, `io_utils.py`, `detect.py`, `characterize.py`, `calibration.py`, etc.).
   - `docs/`: Written reports, architecture documents, summaries, and walkthroughs.
   - `models/`: Saved model weights/artifacts (`.joblib`, `.pth`, etc.).
   - `outputs/`: Pipeline output catalogs, CSVs, and generated cutouts.

## Working rules
- Write code only. Never run scripts; the user runs them in PowerShell. Run git add and git commit only when the user tells you to, with the exact message given. End every task with the file names and the exact run command.
- Read only the files the task names. No directory listings or searches unless told.
- Keep replies under 40 words. No tables, no summaries of what the code does.
- Never tick a checkbox in TASKLIST.md. The user ticks it after a visual check.
- Every task that produces results must save a figure to outputs/review/ for the user to inspect, and print only compact numbers.
- Use (frame_name, candidate_id) as the key. Do not change labels or thresholds unless the task says so.

## Status Report Protocol
Whenever the user asks "where are we now", "where should we proceed", "status", or similar, write the report in plain, simple language based only on reading `TASKLIST.md` and the output of `git log -n 5 --oneline` (no scripts, no sanity check). Explain what each item means and why it matters, and don't assume earlier context is remembered. Sections:

1. Big picture (2-3 sentences): what the project is, and where we are in the plan.
2. Progress: X of Y tasks done, and which phase we are in.
3. Verified state: report recent commits from `git log -n 5 --oneline` (do not run scripts or sanity checks).
4. Task table: every task in `TASKLIST.md` as DONE / IN PROGRESS / NOT STARTED / BLOCKED, each with a one-line plain explanation of what it is. For DONE items, give evidence from the log.
5. Drift check: uncommitted changes, scripts not tied to any task, tasks marked done that the repo does not support. Flag out-of-scope work.
6. Open risks: anything unresolved, in plain words (for example, "we have too few labeled examples to train reliably").
7. Next step: one recommended task. Explain what it is, why it comes next, the exact steps, and the files involved. Then name the two tasks after it.
8. Ask for approval before starting. Do not start work on your own.

Keep it structured and concise. Do not re-read files that are not needed for the report.

## Background
For project background and known false-positive sources, read docs/STATUS_PROTOCOL.md (Background section) before writing a status report.

## Scope Rules
- Do only the task requested. Do not start other TASKLIST.md items or create extra work on your own.
- Never edit src/config.yaml, TASKLIST.md, or existing scripts unless the task requires it.
