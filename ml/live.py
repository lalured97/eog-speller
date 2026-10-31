"""Real-time inference (Pair C, step C5).

Same interface as the threshold detector:  .step(h_uV, v_uV) -> list of events.
It waits for a movement onset (same rule as training), collects the window,
classifies it, and reports the class (REST is not reported).
Each event arrives about 1.2 s after the movement starts (the window length).
"""
from collections import deque

import joblib
import numpy as np

from config import FS
from ml.features import featurise
from ml.windows import K, L, POST, PRE, cut

OPPOSITE = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}


class LiveClassifier:
    def __init__(self, calibration, model_path="data/models/model.joblib",
                 return_s=2.0, fs=FS):
        saved = joblib.load(model_path)
        self.model, self.kind = saved["model"], saved["kind"]
        cal = calibration or {}
        self.scales = (cal.get("h_scale", 400.0), cal.get("v_scale", 400.0))
        self.trigger = cal.get("trigger", 60.0)
        self.buf_h = deque(maxlen=2 * L)
        self.buf_v = deque(maxlen=2 * L)
        self.n = 0
        self.onset = None                   # sample number of a pending onset
        self.ready_after = 0                # no new onset before this sample
        self.return_s, self.fs = return_s, fs
        self.last = (None, -1e9)            # (last event, time)

    def step(self, h, v):
        self.buf_h.append(h); self.buf_v.append(v)
        self.n += 1
        out = []
        if self.onset is None and self.n >= self.ready_after and len(self.buf_h) > K:
            d = max(abs(h - self.buf_h[-1 - K]), abs(v - self.buf_v[-1 - K]))
            if d > self.trigger:
                self.onset = self.n
        if self.onset is not None and self.n - self.onset >= int(POST * self.fs):
            hh = np.array(self.buf_h)[-L:]
            vv = np.array(self.buf_v)[-L:]
            w = cut(hh, vv, 0)
            label = self.model.predict(featurise(w[None], self.scales, self.kind))[0]
            old_onset, self.onset = self.onset, None
            self.ready_after = self.n + int(0.1 * self.fs)
            if label in ("BLINK", "REST"):
                # A blink can start a window just before a real movement.
                # Look again for a new onset after the blink, inside this window.
                self.onset = self._rescan(old_onset + int(0.35 * self.fs))
            t = self.n / self.fs
            prev, t_prev = self.last
            is_return = (OPPOSITE.get(prev) == label and t - t_prev < self.return_s)
            if label != "REST" and not is_return:
                out.append(label)
                self.last = (label, t)
        return out

    def _rescan(self, from_n):
        """Sample number of the first fast change at or after from_n, or None."""
        h = np.array(self.buf_h); v = np.array(self.buf_v)
        first_n = self.n - len(h) + 1           # sample number of h[0]
        i0 = max(K, from_n - first_n)
        if i0 >= len(h):
            return None
        d = np.maximum(np.abs(h[i0:] - h[i0 - K:len(h) - K]),
                       np.abs(v[i0:] - v[i0 - K:len(v) - K]))
        idx = np.nonzero(d > self.trigger)[0]
        return first_n + i0 + int(idx[0]) if idx.size else None
