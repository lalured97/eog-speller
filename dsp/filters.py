"""Digital filter pipeline (Pair B, step B3).

50 Hz notch -> median filter (keeps saccade edges sharp) -> 30 Hz low-pass.
StreamFilter works one sample at a time, so the same code runs on live data.

Run:  python -m dsp.filters      (plots before/after on synthetic data)
"""
from collections import deque

import numpy as np
from scipy import signal

from config import FS


class StreamFilter:
    """Causal filter for ONE channel. Call .step(x) for every new sample."""

    def __init__(self, fs=FS, notch_hz=50.0, median_ms=60, lowpass_hz=30.0):
        self.bn, self.an = signal.iirnotch(notch_hz, 30, fs)
        self.zn = signal.lfilter_zi(self.bn, self.an) * 0
        self.bl, self.al = signal.butter(2, lowpass_hz, "lowpass", fs=fs)
        self.zl = signal.lfilter_zi(self.bl, self.al) * 0
        n = max(3, int(fs * median_ms / 1000) | 1)     # odd length
        self.buf = deque(maxlen=n)
        self.started = False

    def step(self, x):
        if not self.started:           # start filters at the first value
            self.zn = signal.lfilter_zi(self.bn, self.an) * x
            self.zl = signal.lfilter_zi(self.bl, self.al) * x
            self.started = True
        y, self.zn = signal.lfilter(self.bn, self.an, [x], zi=self.zn)
        self.buf.append(y[0])
        m = float(np.median(self.buf))
        y, self.zl = signal.lfilter(self.bl, self.al, [m], zi=self.zl)
        return float(y[0])


def filter_array(x, **kw):
    """Run StreamFilter over a whole array (same result as live)."""
    f = StreamFilter(**kw)
    return np.array([f.step(s) for s in x])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dsp.generator import make_recording
    t, h, v, _ = make_recording()
    vf = filter_array(v)
    for name, sig in [("raw", v), ("filtered", vf)]:
        f, p = signal.welch(sig, FS, nperseg=1024)
        print(f"{name:9s} power at 50 Hz: {p[np.argmin(abs(f - 50))]:.3f}")
    fig, ax = plt.subplots(2, 1, figsize=(12, 6))
    ax[0].plot(t, v, label="raw"); ax[0].plot(t, vf, label="filtered")
    ax[0].set_xlim(5, 15); ax[0].legend(); ax[0].set_ylabel("V (uV)")
    for name, sig in [("raw", v), ("filtered", vf)]:
        f, p = signal.welch(sig, FS, nperseg=1024)
        ax[1].semilogy(f, p, label=name)
    ax[1].set_xlabel("frequency (Hz)"); ax[1].legend()
    plt.tight_layout(); plt.show()
