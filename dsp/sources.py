"""Signal sources (Pair B, step B4).

Every source has .read() which returns a list of (h_uV, v_uV) samples
that have arrived since the last call. The rest of the program does not
care where the samples come from, so switching from fake data to the
real board is one command-line option.

  SyntheticSource  live fake EOG; .inject("LEFT") adds an eye movement
  CsvSource        plays back a recorded file (t_s,h_uV,v_uV)
  SerialSource     the real board: lines "h,v" in ADC counts from the Arduino
"""
import time

import numpy as np

from config import BAUD, FS, counts_to_uV


class SyntheticSource:
    def __init__(self, amp=400.0, noise=5.0, mains=30.0, fs=FS):
        self.fs, self.noise, self.mains, self.amp = fs, noise, mains, amp
        self.pending_h = np.zeros(0)
        self.pending_v = np.zeros(0)
        self.n = 0
        self.t_last = time.time()
        self.rng = np.random.default_rng()

    def inject(self, cmd):
        from dsp.generator import movement
        h, v = movement(cmd, self.amp, fs=self.fs)
        L = max(len(h), len(self.pending_h))
        ph = np.zeros(L); ph[:len(self.pending_h)] = self.pending_h; ph[:len(h)] += h
        pv = np.zeros(L); pv[:len(self.pending_v)] = self.pending_v; pv[:len(v)] += v
        self.pending_h, self.pending_v = ph, pv

    def read(self):
        now = time.time()
        k = int((now - self.t_last) * self.fs)
        if k <= 0:
            return []
        self.t_last += k / self.fs
        h = np.zeros(k); v = np.zeros(k)
        m = min(k, len(self.pending_h))
        h[:m] = self.pending_h[:m]; v[:m] = self.pending_v[:m]
        self.pending_h = self.pending_h[m:]; self.pending_v = self.pending_v[m:]
        t = (self.n + np.arange(k)) / self.fs
        self.n += k
        hum = self.mains * np.sin(2 * np.pi * 50 * t)
        h += hum + self.rng.normal(0, self.noise, k)
        v += hum + self.rng.normal(0, self.noise, k)
        return list(zip(h, v))


class CsvSource:
    def __init__(self, path, fs=FS):
        d = np.loadtxt(path, delimiter=",", skiprows=1)
        self.h, self.v = d[:, 1], d[:, 2]
        self.fs, self.i, self.t0 = fs, 0, time.time()

    def read(self):
        j = min(len(self.h), int((time.time() - self.t0) * self.fs))
        out = list(zip(self.h[self.i:j], self.v[self.i:j]))
        self.i = j
        return out


class SerialSource:
    def __init__(self, port, baud=BAUD):
        import serial                      # pip install pyserial
        self.ser = serial.Serial(port, baud, timeout=0)
        self.buf = b""
        self.bad_lines = 0

    def read(self):
        self.buf += self.ser.read(self.ser.in_waiting or 1)
        *lines, self.buf = self.buf.split(b"\n")
        out = []
        for line in lines:
            try:
                h, v = line.strip().split(b",")
                out.append((counts_to_uV(int(h)), counts_to_uV(int(v))))
            except ValueError:
                self.bad_lines += 1        # half-line at start-up, text, etc.
        return out
