"""ILMT Candidate Labeling Tool

Interactive tool to label candidate cutouts and context images side by side.
Saves progress to CSV on each keypress and automatically resumes from the first
unlabeled candidate.

Controls:
  s : Mark candidate as 'streak' and advance
  f : Mark candidate as 'fp' (false positive) and advance
  u : Undo - return to previous candidate
  n : Add or edit note for current candidate
  q : Quit and save session
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# Clear default matplotlib keymaps that clash with our keys (s, f, u, n, q)
plt.rcParams['keymap.save'] = []
plt.rcParams['keymap.quit'] = []
plt.rcParams['keymap.fullscreen'] = []
plt.rcParams['keymap.yscale'] = []
plt.rcParams['keymap.xscale'] = []

_clash_keys = {'s', 'f', 'u', 'n', 'q'}
for _k in list(plt.rcParams.keys()):
    if _k.startswith('keymap.'):
        plt.rcParams[_k] = [
            v for v in plt.rcParams[_k]
            if v.lower() not in _clash_keys and not any(v.lower() == f"ctrl+{c}" for c in _clash_keys)
        ]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_default_csv(batch_num: int = 1) -> Path:
    return PROJECT_ROOT / "data" / "labeled" / f"batch_{batch_num}_v2_labels.csv"


class LabelTool:
    def __init__(self, csv_path: Path, dry_run: bool = False):
        self.csv_path = Path(csv_path).resolve()
        self.dry_run = dry_run

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)
        if "label" not in self.df.columns:
            self.df["label"] = ""
        if "notes" not in self.df.columns:
            self.df["notes"] = ""

        self.df["label"] = self.df["label"].astype(object)
        self.df["notes"] = self.df["notes"].astype(object)

        # Resume at first unlabeled candidate
        unlabeled_mask = self.df["label"].isna() | (self.df["label"].astype(str).str.strip() == "")
        unlabeled_indices = self.df[unlabeled_mask].index.tolist()

        if unlabeled_indices:
            self.current_idx = unlabeled_indices[0]
            print(f"Resuming at candidate {self.current_idx + 1} of {len(self.df)} (unlabeled: {len(unlabeled_indices)}).")
        else:
            print(f"All {len(self.df)} candidates are already labeled! Starting at candidate 1 in review mode.")
            self.current_idx = 0

        self.fig = None
        self.ax_cutout = None
        self.ax_context = None

    def save_csv(self):
        if self.dry_run:
            return
        self.df.to_csv(self.csv_path, index=False, na_rep="")

    def get_candidate_row(self, idx: int):
        return self.df.iloc[idx]

    def resolve_path(self, rel_path: str) -> Path:
        p = Path(rel_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / p

    def update_view(self):
        row = self.get_candidate_row(self.current_idx)
        cand_id = row["candidate_id"]
        frame_name = row["frame_name"]
        score = row.get("score", 0.0)
        area = row.get("area", 0)
        aspect = row.get("aspect_ratio", 0.0)

        cutout_path = self.resolve_path(row["cutout_path"])
        context_path = self.resolve_path(row["context_path"])

        self.ax_cutout.clear()
        self.ax_context.clear()

        if cutout_path.exists():
            img_cutout = Image.open(cutout_path)
            self.ax_cutout.imshow(img_cutout)
            self.ax_cutout.set_title(f"Cutout (Cand {cand_id})", fontsize=11, fontweight="bold")
        else:
            self.ax_cutout.text(0.5, 0.5, f"Missing:\n{cutout_path.name}", ha="center", va="center")
            self.ax_cutout.set_title(f"Cutout Missing", fontsize=11, color="red")
        self.ax_cutout.axis("off")

        if context_path.exists():
            img_context = Image.open(context_path)
            self.ax_context.imshow(img_context)
            self.ax_context.set_title(f"Context ({frame_name})", fontsize=11, fontweight="bold")
        else:
            self.ax_context.text(0.5, 0.5, f"Missing:\n{context_path.name}", ha="center", va="center")
            self.ax_context.set_title(f"Context Missing", fontsize=11, color="red")
        self.ax_context.axis("off")

        current_label = row["label"]
        if pd.isna(current_label) or str(current_label).strip() == "":
            label_display = "UNLABELED"
        else:
            label_display = str(current_label).upper()

        current_note = row["notes"]
        note_display = f" | Note: {current_note}" if (pd.notna(current_note) and str(current_note).strip() != "") else ""

        title_text = (
            f"Candidate {self.current_idx + 1} / {len(self.df)} (ID: {cand_id})\n"
            f"Score: {score:.1f} | Area: {area} | Aspect: {aspect:.2f} | Current: [{label_display}]{note_display}\n"
            f"[s] Streak   [f] False Positive   [u] Undo   [n] Note   [q] Quit & Save"
        )
        self.fig.suptitle(title_text, fontsize=12, fontweight="medium")
        self.fig.canvas.draw()

    def set_label_and_advance(self, label: str):
        row = self.get_candidate_row(self.current_idx)
        cand_id = row["candidate_id"]
        self.df.at[self.current_idx, "label"] = label
        self.save_csv()
        print(f"[{self.current_idx + 1}/{len(self.df)}] Cand {cand_id} -> {label}")

        if self.current_idx < len(self.df) - 1:
            self.current_idx += 1
            self.update_view()
        else:
            print("Reached the last candidate in this batch!")
            self.update_view()

    def undo(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            row = self.get_candidate_row(self.current_idx)
            print(f"Undo -> back to candidate {self.current_idx + 1}/{len(self.df)} (Cand {row['candidate_id']}, current label: '{row['label']}')")
            self.update_view()
        else:
            print("Already at the first candidate.")

    def prompt_note(self):
        row = self.get_candidate_row(self.current_idx)
        cand_id = row["candidate_id"]
        current_note = "" if pd.isna(row["notes"]) else str(row["notes"])

        note = None
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            note = simpledialog.askstring("Add Note", f"Enter note for Cand {cand_id}:", initialvalue=current_note)
            root.destroy()
        except Exception:
            note = None

        if note is None:
            print(f"\nEnter note for Cand {cand_id} (current: '{current_note}'): ", end="", flush=True)
            try:
                line = input().strip()
                if line:
                    note = line
            except EOFError:
                note = None

        if note is not None:
            self.df.at[self.current_idx, "notes"] = note
            self.save_csv()
            print(f"Saved note for Cand {cand_id}: '{note}'")
            self.update_view()

    def on_key(self, event):
        if not event.key:
            return
        k = event.key.lower()
        if k == "s":
            self.set_label_and_advance("streak")
        elif k == "f":
            self.set_label_and_advance("fp")
        elif k == "u":
            self.undo()
        elif k == "n":
            self.prompt_note()
        elif k == "q":
            print("Saving and quitting...")
            self.save_csv()
            import matplotlib.pyplot as plt
            plt.close(self.fig)

    def run(self):
        import matplotlib.pyplot as plt
        self.fig, (self.ax_cutout, self.ax_context) = plt.subplots(1, 2, figsize=(12, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_view()
        plt.tight_layout()
        plt.show()


def run_test(csv_path: Path, n_candidates: int = 3):
    """Test the labeling workflow on N candidates in dry-run mode without modifying the CSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"--- Running Label Tool Test ({n_candidates} candidates, NO SAVING) ---")
    file_stat_before = csv_path.stat().st_mtime
    content_before = csv_path.read_bytes()

    tool = LabelTool(csv_path=csv_path, dry_run=True)
    tool.fig, (tool.ax_cutout, tool.ax_context) = plt.subplots(1, 2, figsize=(10, 5))

    n_test = min(n_candidates, len(tool.df))
    for i in range(n_test):
        row = tool.get_candidate_row(i)
        cutout = tool.resolve_path(row["cutout_path"])
        context = tool.resolve_path(row["context_path"])

        assert cutout.exists(), f"Cutout missing: {cutout}"
        assert context.exists(), f"Context missing: {context}"

        tool.current_idx = i
        tool.update_view()
        print(f"Candidate {i + 1}/{n_test}: ID {row['candidate_id']} | Cutout & Context verified ok.")

    from matplotlib.backend_bases import KeyEvent, key_press_handler

    # Verify clashing keymaps are empty
    assert len(plt.rcParams["keymap.save"]) == 0, "keymap.save is not empty"
    assert len(plt.rcParams["keymap.quit"]) == 0, "keymap.quit is not empty"
    assert len(plt.rcParams["keymap.fullscreen"]) == 0, "keymap.fullscreen is not empty"
    assert len(plt.rcParams["keymap.yscale"]) == 0, "keymap.yscale is not empty"
    assert len(plt.rcParams["keymap.xscale"]) == 0, "keymap.xscale is not empty"

    # Test keypress 's': default handler does nothing (no save dialog), on_key labels streak
    tool.current_idx = 0
    event_s = KeyEvent("key_press_event", tool.fig.canvas, "s")
    key_press_handler(event_s, tool.fig.canvas)  # verify default handler is no-op
    tool.on_key(event_s)  # tool keypress handler
    assert tool.df.at[0, "label"] == "streak", "Failed to label 'streak' on 's' keypress"
    assert tool.current_idx == 1, "Failed to advance on 's' keypress"

    # Test 'f' and 'u' key events
    event_f = KeyEvent("key_press_event", tool.fig.canvas, "f")
    tool.on_key(event_f)
    assert tool.df.at[1, "label"] == "fp", "Failed to label 'fp' on 'f' keypress"
    assert tool.current_idx == 2, "Failed to advance on 'f' keypress"

    event_u = KeyEvent("key_press_event", tool.fig.canvas, "u")
    tool.on_key(event_u)
    assert tool.current_idx == 1, "Failed to undo back to candidate index 1"

    plt.close(tool.fig)

    # Verify CSV was not touched
    file_stat_after = csv_path.stat().st_mtime
    content_after = csv_path.read_bytes()
    assert content_before == content_after, "ERROR: CSV content was modified during test!"
    assert file_stat_before == file_stat_after, "ERROR: CSV file modified time changed!"

    print(f"Test PASSED: {n_test} candidates verified. CSV was NOT modified (dry-run clean).")


def main():
    parser = argparse.ArgumentParser(description="ILMT Candidate Labeling Tool")
    parser.add_argument("csv_path", nargs="?", default=None, help="Path to batch labels CSV")
    parser.add_argument("--batch", type=int, default=None, help="Batch number (e.g. 1 or 2)")
    parser.add_argument("--dry-run", action="store_true", help="Run without writing changes to CSV")
    parser.add_argument("--test", type=int, default=None, help="Test on N candidates without saving and exit")

    args = parser.parse_args()

    if args.csv_path:
        csv_path = Path(args.csv_path)
    elif args.batch is not None:
        csv_path = get_default_csv(args.batch)
    else:
        csv_path = get_default_csv(1)

    if args.test is not None:
        run_test(csv_path, n_candidates=args.test)
        return

    tool = LabelTool(csv_path=csv_path, dry_run=args.dry_run)
    tool.run()


if __name__ == "__main__":
    main()
