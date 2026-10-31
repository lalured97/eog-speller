"""Calibration screen (Pair C, step C6).

Shows a big arrow with a beep; the user looks that way and back. The filtered
signal is recorded, and at the end ml.calibration_math turns it into this
user's numbers (scales for the ML model, thresholds for the baseline),
saved in data/calibration.json. Started from main.py with --calibrate.
"""
import tkinter as tk

from ml.calibration_math import compute_calibration, save

ARROWS = {"LEFT": "←", "RIGHT": "→", "UP": "↑", "DOWN": "↓"}


class CalibrationWindow:
    def __init__(self, root, repeats=5, gap_s=3.0, on_done=None, fs=250, inject=None):
        self.root, self.fs, self.gap_s = root, fs, gap_s
        self.plan = ["LEFT", "RIGHT", "UP", "DOWN"] * repeats
        self.h, self.v, self.prompts = [], [], []
        self.i = -1
        self.on_done, self.inject = on_done, inject
        root.winfo_toplevel().title("Calibration")
        root.configure(bg="black")
        self.var = tk.StringVar(value="Get ready... look at the centre")
        tk.Label(root, textvariable=self.var, font=("Arial", 40, "bold"),
                 bg="black", fg="white", width=24, height=6).pack()
        root.after(3000, self.next_prompt)

    def add_samples(self, h, v):
        """main.py calls this with every filtered sample."""
        self.h.append(h); self.v.append(v)

    def next_prompt(self):
        self.i += 1
        if self.i >= len(self.plan):
            return self.finish()
        cmd = self.plan[self.i]
        self.prompts.append((len(self.h) / self.fs, cmd))
        self.var.set(f"{ARROWS[cmd]}  Look {cmd.lower()}, then back")
        self.root.bell()
        if self.inject:                       # synthetic mode: fake the user
            self.root.after(350, lambda: self.inject(cmd))
        self.root.after(int(self.gap_s * 1000), self.next_prompt)

    def finish(self):
        th = compute_calibration(self.h, self.v, self.prompts, self.fs)
        save(th)
        if th["n_windows"] < 0.7 * len(self.plan):
            self.var.set("Some movements were not seen.\nCheck electrodes and run again")
        else:
            self.var.set("Done! Calibration saved")
        print("Saved data/calibration.json:", {k: round(x, 1) for k, x in th.items()})
        if self.on_done:
            self.root.after(1500, lambda: self.on_done(th))
