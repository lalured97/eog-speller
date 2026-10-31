"""Labelled dataset files (Pair B owns the format; Pair C reads it).

One session = two files in data/sessions/:
  <subject>_s<nn>.csv          t_s,h_uV,v_uV   raw samples (not filtered)
  <subject>_s<nn>_labels.csv   t_s,label       cue time and what was asked
and one line in data/sessions/sessions_log.csv (who, when, notes).

Pair C uses only these two functions:
  list_sessions()                 -> ["lalith_s01", ...]
  load_session(name)              -> t, h, v, labels   (h, v already filtered)
"""
import csv
import glob
import os
import time

import numpy as np

from dsp.filters import filter_array

FOLDER = "data/sessions"
CLASSES = ["LEFT", "RIGHT", "UP", "DOWN", "BLINK", "DOUBLE_BLINK", "REST"]


def save_session(name, t, h, v, labels, notes="", folder=FOLDER):
    os.makedirs(folder, exist_ok=True)
    base = os.path.join(folder, name)
    np.savetxt(base + ".csv", np.column_stack([t, h, v]), delimiter=",",
               header="t_s,h_uV,v_uV", comments="", fmt="%.4f")
    with open(base + "_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "label"])
        for t0, lab in labels:
            w.writerow([f"{t0:.4f}", lab])
    log = os.path.join(folder, "sessions_log.csv")
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["session", "date", "seconds", "n_cues", "notes"])
        w.writerow([name, time.strftime("%Y-%m-%d %H:%M"), f"{t[-1]:.0f}",
                    len(labels), notes])


def list_sessions(folder=FOLDER):
    files = glob.glob(os.path.join(folder, "*_labels.csv"))
    return sorted(os.path.basename(f)[:-len("_labels.csv")] for f in files)


def subject_of(name):
    """'lalith_s02' -> 'lalith'"""
    return name.rsplit("_s", 1)[0]


def load_session(name, folder=FOLDER, filtered=True):
    base = os.path.join(folder, name)
    d = np.loadtxt(base + ".csv", delimiter=",", skiprows=1)
    t, h, v = d[:, 0], d[:, 1], d[:, 2]
    if filtered:
        h, v = filter_array(h), filter_array(v)
    labels = []
    with open(base + "_labels.csv") as f:
        for row in csv.DictReader(f):
            labels.append((float(row["t_s"]), row["label"]))
    return t, h, v, labels
