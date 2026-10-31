"""Threshold detector: the BASELINE recogniser (Pair C, step C1).

No training needed, so it works from day one. The ML classifier (ml/live.py)
is compared against it, and it is the fallback if the model is not ready.

Input: filtered H and V samples (uV), one pair at a time.
Output: event names from config.EVENTS.

How it works
- A saccade is a fast change: the 20 ms difference d = x[n] - x[n-5]
  crosses `slope_thr`. The event lasts while |d| stays high.
  Its size (x after - x before) must exceed `amp_thr`.
- Looking somewhere and back gives TWO saccades (out, then back).
  The first one is the command; the return within `return_s` is ignored.
- A blink on V looks like UP then DOWN very quickly (< `blink_max_s`).
  So UP is only reported once no fast DOWN has followed.
- Two blinks within `double_s` -> DOUBLE_BLINK.

Thresholds are per user: calibration (ml/calibration_math.py) fills them in.
"""
from collections import deque

from config import FS

DEFAULT_THRESHOLDS = {
    "h_slope": 60.0, "h_amp": 150.0,   # uV per 20 ms, uV
    "v_slope": 60.0, "v_amp": 150.0,
}


class _Channel:
    """Finds saccades on one channel. Returns (sign, amplitude) when one ends."""

    def __init__(self, slope_thr, amp_thr, fs=FS):
        self.k = max(1, int(0.02 * fs))           # 20 ms lag
        self.hist = deque(maxlen=self.k + 1)
        self.slope_thr, self.amp_thr = slope_thr, amp_thr
        self.fs = fs
        self.active = False
        self.start_val = 0.0
        self.n = 0

    def step(self, x):
        self.hist.append(x)
        if len(self.hist) <= self.k:
            return None
        d = x - self.hist[0]
        if not self.active and abs(d) > self.slope_thr:
            self.active, self.n = True, 0
            self.start_val = self.hist[0]
            self.sign = 1 if d > 0 else -1
        elif self.active:
            self.n += 1
            ended = abs(d) < 0.5 * self.slope_thr or (d > 0) != (self.sign > 0)
            if ended or self.n > 0.4 * self.fs:
                self.active = False
                amp = x - self.start_val
                if abs(amp) > self.amp_thr and self.n >= 2:
                    return (1 if amp > 0 else -1), amp
        return None


class Detector:
    def __init__(self, thresholds=None, fs=FS, return_s=1.5,
                 blink_max_s=0.45, double_s=0.9, refractory_s=0.3):
        th = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
        self.h = _Channel(th["h_slope"], th["h_amp"], fs)
        self.v = _Channel(th["v_slope"], th["v_amp"], fs)
        self.fs, self.t = fs, 0.0
        self.return_s, self.blink_max_s = return_s, blink_max_s
        self.double_s, self.refractory_s = double_s, refractory_s
        self.ignore = {"h": (0, 0.0), "v": (0, 0.0)}  # (sign to ignore, until)
        self.pending_up = None      # time of an UP that might be a blink
        self.last_blink = -10.0
        self.last_event = -10.0

    def _emit(self, name, out):
        out.append(name)
        self.last_event = self.t

    def step(self, h, v):
        """Feed one filtered sample pair. Returns a list of events (often empty)."""
        self.t += 1 / self.fs
        out = []

        # An UP that was not followed by a quick DOWN is a real UP.
        if self.pending_up is not None and self.t - self.pending_up > self.blink_max_s:
            self.pending_up = None
            self._emit("UP", out)
            self.ignore["v"] = (-1, self.t + self.return_s)

        sh = self.h.step(h)
        if sh:
            sign = sh[0]
            ign_sign, until = self.ignore["h"]
            if sign == ign_sign and self.t < until:
                self.ignore["h"] = (0, 0.0)               # the return saccade
            elif self.t - self.last_event > self.refractory_s:
                self._emit("RIGHT" if sign > 0 else "LEFT", out)
                self.ignore["h"] = (-sign, self.t + self.return_s)

        sv = self.v.step(v)
        if sv:
            sign = sv[0]
            ign_sign, until = self.ignore["v"]
            if sign < 0 and self.pending_up is not None:
                self.pending_up = None                    # UP + fast DOWN = blink
                if self.t - self.last_blink < self.double_s:
                    self._emit("DOUBLE_BLINK", out)
                    self.last_blink = -10.0
                else:
                    self._emit("BLINK", out)
                    self.last_blink = self.t
            elif sign == ign_sign and self.t < until:
                self.ignore["v"] = (0, 0.0)
            elif sign > 0:
                self.pending_up = self.t
            elif self.t - self.last_event > self.refractory_s:
                self._emit("DOWN", out)
                self.ignore["v"] = (1, self.t + self.return_s)
        return out


def detect_array(h, v, thresholds=None, fs=FS):
    """Run the detector over whole arrays. Returns [(time_s, event), ...]."""
    d = Detector(thresholds, fs)
    events = []
    for a, b in zip(h, v):
        for e in d.step(a, b):
            events.append((d.t, e))
    return events
