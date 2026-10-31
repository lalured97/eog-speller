"""Feature extraction (Pair C, step C3).

Two kinds of input for the classifiers:
  summary(w, scales) -> 22 hand-made numbers (for Random Forest / kNN)
  raw(w, scales)     -> the window shrunk to 35 points per channel
                        (Usakli et al. 2010 style template matching)
Windows are divided by the person's scale (from calibration), so a model
trained on some people works on a new person with a different signal size.
"""
import numpy as np

from ml.windows import K

NAMES = []
for ch in ("h", "v"):
    NAMES += [f"{ch}_{n}" for n in ("max", "min", "ptp", "t_max", "t_min", "slope_max",
                                     "slope_min", "t_slope_max", "t_slope_min", "energy")]
NAMES += ["log_ratio_hv", "n_fast_v"]


def summary(w, scales):
    out = []
    n = w.shape[1]
    for c in range(2):
        x = w[c] / scales[c]
        d = x[K:] - x[:-K]
        out += [x.max(), x.min(), np.ptp(x), x.argmax() / n, x.argmin() / n,
                d.max(), d.min(), d.argmax() / n, d.argmin() / n, np.mean(x ** 2)]
    hp = np.ptp(w[0] / scales[0]); vp = np.ptp(w[1] / scales[1])
    dv = np.diff(w[1] / scales[1], K)
    fast = np.abs(dv) > 0.25
    n_fast = np.count_nonzero(np.diff(fast.astype(int)) == 1)   # bursts of fast V change
    out += [np.log((hp + 1e-3) / (vp + 1e-3)), n_fast]
    return np.array(out)


def raw(w, scales, points=35):
    x = w / np.array(scales)[:, None]
    n = (x.shape[1] // points) * points
    return x[:, :n].reshape(2, points, -1).mean(axis=2).ravel()


def featurise(X, scales, kind="summary"):
    f = summary if kind == "summary" else raw
    return np.array([f(w, scales) for w in X])
