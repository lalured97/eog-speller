"""Windowing (Pair C, step C2): cut one fixed-length window per movement.

A window starts PRE s before the movement onset and ends POST s after it,
so it contains the whole "look there and come back" (or both blinks).
The SAME onset rule is used for training (here) and live (ml/live.py).

Onset = first sample where the 20 ms change on H or V exceeds `trigger`.
trigger = 35% of the typical movement speed of that person/session.
"""
import numpy as np

from config import FS

PRE, POST = 0.2, 1.2                 # seconds before / after onset
L = int((PRE + POST) * FS)           # samples per window (350)
K = int(0.02 * FS)                   # 20 ms lag for the slope
SEARCH = 1.5                         # look for onset up to 1.5 s after a cue
MOVES = ("LEFT", "RIGHT", "UP", "DOWN")


def slope(x):
    """20 ms change, same length as x (first K values are 0)."""
    d = np.zeros_like(x)
    d[K:] = x[K:] - x[:-K]
    return d


def trigger_from(h, v, labels, fs=FS, frac=0.35):
    """Typical movement speed after the cues -> onset threshold."""
    dh, dv = np.abs(slope(h)), np.abs(slope(v))
    peaks = []
    for t0, lab in labels:
        if lab == "REST":
            continue
        a, b = int(t0 * fs), int((t0 + SEARCH) * fs)
        peaks.append(max(dh[a:b].max(initial=0), dv[a:b].max(initial=0)))
    return frac * float(np.median(peaks)) if peaks else 40.0


def find_onset(dmax, a, b, trigger):
    idx = np.nonzero(dmax[a:b] > trigger)[0]
    return a + int(idx[0]) if idx.size else None


def cut(h, v, start):
    """Window of both channels from sample `start`, baseline removed."""
    w = np.stack([h[start:start + L], v[start:start + L]]).astype(float)
    n_pre = int(PRE * FS)
    return w - w[:, :n_pre].mean(axis=1, keepdims=True)


def cut_windows(h, v, labels, trigger=None, fs=FS):
    """-> X (n, 2, L), y (n,). Cues with no clear movement are skipped;
    REST cues where the person moved anyway (e.g. a natural blink) are skipped."""
    if trigger is None:
        trigger = trigger_from(h, v, labels, fs)
    dmax = np.maximum(np.abs(slope(h)), np.abs(slope(v)))
    X, y = [], []
    pre = int(PRE * fs)
    for t0, lab in labels:
        a, b = int(t0 * fs), int((t0 + SEARCH) * fs)
        on = find_onset(dmax, a, b, trigger)
        if lab == "REST":
            if on is not None:
                continue
            start = a + int(0.3 * fs)
        else:
            if on is None:
                continue
            start = on - pre
        if start < 0 or start + L > len(h):
            continue
        X.append(cut(h, v, start)); y.append(lab)
    return np.array(X), np.array(y)


def scales_from(X, y):
    """Person's typical movement size per channel (used to normalise)."""
    hs = [np.ptp(w[0]) for w, lab in zip(X, y) if lab in ("LEFT", "RIGHT")]
    vs = [np.ptp(w[1]) for w, lab in zip(X, y) if lab in ("UP", "DOWN")]
    return (float(np.median(hs)) if hs else 400.0,
            float(np.median(vs)) if vs else 400.0)
