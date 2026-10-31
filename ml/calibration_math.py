"""Per-user calibration numbers (Pair C, step C6). No screen code here.

The calibration screen prompts LEFT, RIGHT, UP, DOWN (5 times each) and
records the filtered signal. From that recording we compute, for this user:
  trigger           onset threshold for the live classifier
  h_scale, v_scale  typical movement size -> features are divided by these
  h_slope, h_amp,   thresholds for the threshold detector (baseline)
  v_slope, v_amp
saved in data/calibration.json.

Run:  python -m ml.calibration_math     (shows why calibration matters)
"""
import json

import numpy as np

from config import FS
from ml.windows import cut_windows, scales_from, slope, trigger_from

PATH = "data/calibration.json"


def compute_calibration(h, v, prompts, fs=FS, slope_frac=0.4, amp_frac=0.3):
    h, v = np.asarray(h, float), np.asarray(v, float)
    trig = trigger_from(h, v, prompts, fs)
    X, y = cut_windows(h, v, prompts, trig, fs)
    h_scale, v_scale = scales_from(X, y)
    cal = {"trigger": trig, "h_scale": h_scale, "v_scale": v_scale}
    for ch, idx, moves in (("h", 0, ("LEFT", "RIGHT")), ("v", 1, ("UP", "DOWN"))):
        ws = [w[idx] for w, lab in zip(X, y) if lab in moves]
        if ws:
            cal[f"{ch}_slope"] = slope_frac * float(np.median([np.abs(slope(w)).max() for w in ws]))
            cal[f"{ch}_amp"] = amp_frac * float(np.median([np.ptp(w) for w in ws]))
    cal["n_windows"] = int(len(y))
    return cal


def save(cal, path=PATH):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def load(path=PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    from dsp.filters import filter_array
    from dsp.generator import make_recording
    from ml.evaluate import score
    from ml.threshold_detector import detect_array

    weak = dict(amp=200, noise=15, mains=100)
    t, h, v, prompts = make_recording(["LEFT", "RIGHT", "UP", "DOWN"] * 5, gap=3.0,
                                      seed=99, **weak)
    cal = compute_calibration(filter_array(h), filter_array(v), prompts)
    print("Calibration:", {k: round(x, 1) for k, x in cal.items()})
    for name, use in [("default thresholds", None), ("calibrated", cal)]:
        c = n = f = 0
        for s in range(5):
            t, h, v, lab = make_recording(seed=s, **weak)
            ev = detect_array(filter_array(h), filter_array(v), use)
            cc, _, _, fa = score(lab, ev)
            c += cc; n += len(lab); f += len(fa)
        print(f"threshold detector, {name:20s}: {100 * c / n:.0f}% correct, {f} false events")
