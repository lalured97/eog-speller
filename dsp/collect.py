"""Cued data collection for the labelled dataset (Pair B, step B5).

A session shows a cue (big arrow or word) with a beep every 3 s, in random
order: 10 x LEFT, RIGHT, UP, DOWN, BLINK, DOUBLE_BLINK, REST (70 cues,
about 3.5 minutes). The subject does the movement and looks back to the
centre. Raw samples and cue times are saved with dsp.dataset.save_session.

Real board:
  python -m dsp.collect --subject lalith --session 1 --source serial --port COM5
Test with fake signal (cues are acted out automatically):
  python -m dsp.collect --subject test --session 1
Make a whole fake dataset (6 people x 2 sessions) without any window:
  python -m dsp.collect --simulate
"""
import argparse
import random

import numpy as np

from config import FS
from dsp.dataset import CLASSES, save_session

CUE_TEXT = {"LEFT": "←  LEFT", "RIGHT": "RIGHT  →", "UP": "↑  UP",
            "DOWN": "↓  DOWN", "BLINK": "BLINK once", "DOUBLE_BLINK": "BLINK BLINK",
            "REST": "+  stay still"}


def make_plan(repeats=10, seed=None):
    plan = CLASSES * repeats
    random.Random(seed).shuffle(plan)
    return plan


def simulate(n_subjects=6, n_sessions=2, repeats=10, gap=3.0):
    """Write a fake dataset to data/sessions (for Pair C to start with)."""
    from dsp.generator import make_recording, subject_profile
    for sub in range(1, n_subjects + 1):
        prof = subject_profile(sub)
        for ses in range(1, n_sessions + 1):
            plan = make_plan(repeats, seed=sub * 10 + ses)
            t, h, v, labels = make_recording(plan, gap=gap, seed=sub * 100 + ses, **prof)
            name = f"fake{sub}_s{ses:02d}"
            save_session(name, t, h, v, labels, notes="synthetic")
            print(f"saved {name}: {len(labels)} cues, {t[-1]:.0f} s")


def run_window(args):
    import tkinter as tk
    if args.source == "serial":
        from dsp.sources import SerialSource
        src = SerialSource(args.port)
    else:
        from dsp.sources import SyntheticSource
        src = SyntheticSource()
    plan = make_plan(args.repeats)
    samples, labels = [], []
    root = tk.Tk()
    root.title("Data collection")
    root.configure(bg="black")
    var = tk.StringVar(value="Electrodes on? Look at the centre.\nStarting in 5 s")
    tk.Label(root, textvariable=var, font=("Arial", 44, "bold"), bg="black",
             fg="white", width=22, height=6).pack()
    info = tk.StringVar()
    tk.Label(root, textvariable=info, bg="black", fg="#aaaaaa").pack()
    state = {"i": -1}

    def pump():
        samples.extend(src.read())
        root.after(20, pump)

    def next_cue():
        state["i"] += 1
        i = state["i"]
        if i >= len(plan):
            return finish()
        cmd = plan[i]
        labels.append((len(samples) / FS, cmd))
        var.set(CUE_TEXT[cmd])
        info.set(f"cue {i + 1} / {len(plan)}")
        root.bell()                                   # the audio cue
        if hasattr(src, "inject") and cmd != "REST":  # fake subject reacts
            root.after(350, lambda: src.inject(cmd))
        root.after(int(args.gap * 1000 * 0.6), lambda: var.set("+"))
        root.after(int(args.gap * 1000), next_cue)

    def finish():
        arr = np.array(samples)
        t = np.arange(len(arr)) / FS
        name = f"{args.subject}_s{args.session:02d}"
        save_session(name, t, arr[:, 0], arr[:, 1], labels, notes=args.notes)
        var.set(f"Saved {name}\n{len(labels)} cues")
        print(f"Saved data/sessions/{name}.csv and _labels.csv")
        root.after(2000, root.destroy)

    root.after(20, pump)
    root.after(5000, next_cue)
    root.mainloop()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--subject", default="test")
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--source", choices=["synthetic", "serial"], default="synthetic")
    p.add_argument("--port", default="COM5")
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--gap", type=float, default=3.0)
    p.add_argument("--notes", default="")
    p.add_argument("--simulate", action="store_true")
    a = p.parse_args()
    if a.simulate:
        simulate()
    else:
        run_window(a)
