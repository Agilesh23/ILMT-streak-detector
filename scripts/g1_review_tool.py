import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

# Disable matplotlib default key bindings that clash with our keys
plt.rcParams["keymap.save"] = []
plt.rcParams["keymap.quit"] = []
plt.rcParams["keymap.fullscreen"] = []
plt.rcParams["keymap.yscale"] = []
plt.rcParams["keymap.xscale"] = []

_clash_keys = {"s", "f", "u", "n", "q", "0", "1", "2"}
for _k in list(plt.rcParams.keys()):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = [
            v for v in plt.rcParams[_k]
            if v.lower() not in _clash_keys and not any(v.lower() == f"ctrl+{c}" for c in _clash_keys)
        ]

base_dir = Path(__file__).resolve().parent.parent
g1_dir = base_dir / "data/g1"
outputs_g1_dir = base_dir / "outputs/g1"
calls_dir = outputs_g1_dir / "calls"


class CallsReviewTool:
    def __init__(self):
        self.csv_path = g1_dir / "verdicts.csv"

        # Load existing verdicts if present
        existing = {}
        if self.csv_path.exists():
            df_old = pd.read_csv(self.csv_path)
            for _, r in df_old.iterrows():
                key = (str(r["frame_name"]), int(r["candidate_id"]))
                v_val = "" if pd.isna(r.get("verdict")) else str(r["verdict"]).strip()
                n_val = "" if pd.isna(r.get("notes")) else str(r["notes"]).strip()
                existing[key] = (v_val, n_val)

        # Collect all calls with p_streak >= 0.5 from data/g1/frame_*.csv
        all_calls = []
        for c_file in sorted(g1_dir.glob("frame_20*.csv")):
            df_frame = pd.read_csv(c_file)
            p5_rows = df_frame[df_frame["p_streak"] >= 0.5]
            for _, r in p5_rows.iterrows():
                fn = str(r["frame_name"])
                cid = int(r["candidate_id"])
                p = float(r["p_streak"])
                key = (fn, cid)
                v_stored, n_stored = existing.get(key, ("", ""))
                all_calls.append({
                    "frame_name": fn,
                    "candidate_id": cid,
                    "p_streak": p,
                    "verdict": v_stored,
                    "notes": n_stored,
                })

        # Sort by frame_name, then candidate_id
        self.items = sorted(all_calls, key=lambda x: (x["frame_name"], x["candidate_id"]))
        if not self.items:
            print("No calls with p >= 0.5 found in data/g1/frame_*.csv.")
            return

        # Resume at first item without verdict
        unlabeled = [i for i, it in enumerate(self.items) if not it["verdict"]]
        if unlabeled:
            self.current_idx = unlabeled[0]
            print(f"Resuming at call {self.current_idx + 1} of {len(self.items)} (unlabeled: {len(unlabeled)}).")
        else:
            print(f"All {len(self.items)} calls are already reviewed! Review mode starting at call 1.")
            self.current_idx = 0

        self.fig = None
        self.ax = None

    def save_csv(self):
        # Update existing records, never overwrite a stored verdict with empty
        existing_df = pd.DataFrame(self.items, columns=["frame_name", "candidate_id", "p_streak", "verdict", "notes"])
        existing_df.to_csv(self.csv_path, index=False)

    def prompt_note(self):
        it = self.items[self.current_idx]
        current_note = it["notes"]
        note = None
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            note = simpledialog.askstring("Add Note", f"Note for {it['frame_name']} ID {it['candidate_id']}:", initialvalue=current_note)
            root.destroy()
        except Exception:
            note = None

        if note is None:
            print(f"\nEnter note for {it['frame_name']} ID {it['candidate_id']} (current: '{current_note}'): ", end="", flush=True)
            try:
                line = input().strip()
                if line:
                    note = line
            except EOFError:
                note = None

        if note is not None:
            it["notes"] = note
            self.save_csv()
            print(f"[{self.current_idx + 1}/{len(self.items)}] {it['frame_name']} Cand {it['candidate_id']} -> note: '{note}'")
            self.update_view()

    def update_view(self):
        it = self.items[self.current_idx]
        stem = Path(it["frame_name"]).stem
        call_png = calls_dir / f"{stem}_{it['candidate_id']}.png"

        self.ax.clear()
        if call_png.exists():
            img = Image.open(call_png)
            self.ax.imshow(img)
        else:
            self.ax.text(0.5, 0.5, f"Missing:\n{call_png.name}", ha="center", va="center", color="red", fontsize=12)
        self.ax.axis("off")

        v_display = it["verdict"].upper() if it["verdict"] else "UNREVIEWED"
        note_str = f" | Note: {it['notes']}" if it["notes"] else ""
        title = (
            f"Call {self.current_idx + 1}/{len(self.items)}: {it['frame_name']} ID {it['candidate_id']} (p={it['p_streak']:.3f})\n"
            f"Verdict: [{v_display}]{note_str}\n"
            f"[s] Streak   [f] False Alarm   [u] Undo   [n] Note   [q] Quit & Save"
        )
        self.fig.suptitle(title, fontsize=12, fontweight="medium")
        self.fig.canvas.draw()

    def on_key(self, event):
        if not event.key:
            return
        k = event.key.lower()
        if k in ("s", "f"):
            v = "streak" if k == "s" else "fp"
            it = self.items[self.current_idx]
            it["verdict"] = v
            self.save_csv()
            print(f"[{self.current_idx + 1}/{len(self.items)}] {it['frame_name']} Cand {it['candidate_id']} -> {v}")
            if self.current_idx < len(self.items) - 1:
                self.current_idx += 1
                self.update_view()
            else:
                print("Reached the last call!")
                self.update_view()
        elif k == "u":
            if self.current_idx > 0:
                self.current_idx -= 1
                it = self.items[self.current_idx]
                print(f"Undo -> back to call {self.current_idx + 1}/{len(self.items)} ({it['frame_name']} Cand {it['candidate_id']}, verdict: '{it['verdict']}')")
                self.update_view()
            else:
                print("Already at the first call.")
        elif k == "n":
            self.prompt_note()
        elif k == "q":
            print("Saving and quitting...")
            self.save_csv()
            plt.close(self.fig)

    def run(self):
        if not self.items:
            return
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_view()
        plt.show()


class FramesReviewTool:
    def __init__(self, n_sample: int = 20, seed: int = 7):
        self.csv_path = g1_dir / "frame_review.csv"

        # Load existing reviews
        existing = {}
        if self.csv_path.exists():
            df_old = pd.read_csv(self.csv_path)
            for _, r in df_old.iterrows():
                fn = str(r["frame_name"])
                nv = "" if pd.isna(r.get("n_visible")) else str(r["n_visible"]).strip()
                nt = "" if pd.isna(r.get("notes")) else str(r["notes"]).strip()
                existing[fn] = (nv, nt)

        # Collect all processed frames
        all_frames = []
        for c_file in sorted(g1_dir.glob("frame_20*.csv")):
            df_frame = pd.read_csv(c_file)
            fn = str(df_frame["frame_name"].iloc[0]) if len(df_frame) > 0 else c_file.stem.replace("frame_", "") + ".fits"
            all_frames.append(fn)

        all_frames = sorted(list(set(all_frames)))
        if not all_frames:
            print("No processed frames found in data/g1/frame_*.csv.")
            self.items = []
            return

        # Seeded random sample
        rng = np.random.default_rng(seed=seed)
        if len(all_frames) <= n_sample:
            sampled = all_frames
        else:
            indices = sorted(rng.choice(len(all_frames), size=n_sample, replace=False))
            sampled = [all_frames[i] for i in indices]

        self.items = []
        for fn in sampled:
            nv_stored, nt_stored = existing.get(fn, ("", ""))
            self.items.append({
                "frame_name": fn,
                "n_visible": nv_stored,
                "notes": nt_stored,
            })

        # Resume at first unreviewed frame
        unreviewed = [i for i, it in enumerate(self.items) if not it["n_visible"]]
        if unreviewed:
            self.current_idx = unreviewed[0]
            print(f"Resuming at frame {self.current_idx + 1} of {len(self.items)} (unreviewed: {len(unreviewed)}).")
        else:
            print(f"All {len(self.items)} sampled frames are already reviewed! Review mode starting at frame 1.")
            self.current_idx = 0

        self.fig = None
        self.ax = None

    def save_csv(self):
        # Update existing records, never overwrite a stored review with empty
        existing_all = {}
        if self.csv_path.exists():
            df_old = pd.read_csv(self.csv_path)
            for _, r in df_old.iterrows():
                fn = str(r["frame_name"])
                existing_all[fn] = {
                    "frame_name": fn,
                    "n_visible": r.get("n_visible", ""),
                    "notes": r.get("notes", ""),
                }

        for it in self.items:
            fn = it["frame_name"]
            existing_all[fn] = {
                "frame_name": fn,
                "n_visible": it["n_visible"],
                "notes": it["notes"],
            }

        df_save = pd.DataFrame(list(existing_all.values()), columns=["frame_name", "n_visible", "notes"])
        df_save.to_csv(self.csv_path, index=False)

    def prompt_note(self):
        it = self.items[self.current_idx]
        current_note = it["notes"]
        note = None
        try:
            import tkinter as tk
            from tkinter import simpledialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            note = simpledialog.askstring("Add Note", f"Note for {it['frame_name']}:", initialvalue=current_note)
            root.destroy()
        except Exception:
            note = None

        if note is None:
            print(f"\nEnter note for {it['frame_name']} (current: '{current_note}'): ", end="", flush=True)
            try:
                line = input().strip()
                if line:
                    note = line
            except EOFError:
                note = None

        if note is not None:
            it["notes"] = note
            self.save_csv()
            print(f"[{self.current_idx + 1}/{len(self.items)}] {it['frame_name']} -> note: '{note}'")
            self.update_view()

    def update_view(self):
        it = self.items[self.current_idx]
        stem = Path(it["frame_name"]).stem
        faint_png = outputs_g1_dir / f"faint_{stem}.png"

        self.ax.clear()
        if faint_png.exists():
            img = Image.open(faint_png)
            self.ax.imshow(img)
        else:
            self.ax.text(0.5, 0.5, f"Missing:\n{faint_png.name}", ha="center", va="center", color="red", fontsize=12)
        self.ax.axis("off")

        v_display = it["n_visible"] if it["n_visible"] != "" else "UNREVIEWED"
        note_str = f" | Note: {it['notes']}" if it["notes"] else ""
        title = (
            f"Frame {self.current_idx + 1}/{len(self.items)}: {it['frame_name']}\n"
            f"Visible streaks: [{v_display}]{note_str}\n"
            f"[0] 0 streaks   [1] 1 streak   [2] 2 streaks   [u] Undo   [n] Note   [q] Quit & Save"
        )
        self.fig.suptitle(title, fontsize=12, fontweight="medium")
        self.fig.canvas.draw()

    def on_key(self, event):
        if not event.key:
            return
        k = event.key.lower()
        if k in ("0", "1", "2"):
            it = self.items[self.current_idx]
            it["n_visible"] = int(k)
            self.save_csv()
            print(f"[{self.current_idx + 1}/{len(self.items)}] {it['frame_name']} -> n_visible={k}")
            if self.current_idx < len(self.items) - 1:
                self.current_idx += 1
                self.update_view()
            else:
                print("Reached the last sampled frame!")
                self.update_view()
        elif k == "u":
            if self.current_idx > 0:
                self.current_idx -= 1
                it = self.items[self.current_idx]
                print(f"Undo -> back to frame {self.current_idx + 1}/{len(self.items)} ({it['frame_name']}, n_visible: '{it['n_visible']}')")
                self.update_view()
            else:
                print("Already at the first frame.")
        elif k == "n":
            self.prompt_note()
        elif k == "q":
            print("Saving and quitting...")
            self.save_csv()
            plt.close(self.fig)

    def run(self):
        if not self.items:
            return
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.update_view()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="G1 Visual Review Tool")
    parser.add_argument("--mode", choices=["calls", "frames"], required=True, help="Review mode: 'calls' or 'frames'")
    parser.add_argument("--n", type=int, default=20, help="Number of frames to sample in 'frames' mode (default 20)")
    args = parser.parse_args()

    if args.mode == "calls":
        tool = CallsReviewTool()
        tool.run()
    elif args.mode == "frames":
        tool = FramesReviewTool(n_sample=args.n, seed=7)
        tool.run()


if __name__ == "__main__":
    main()
