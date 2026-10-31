"""Check the serial stream from the UNO Q (Pair B, step B4).

Run:  python -m dsp.check_stream COM5
Prints samples per second and the signal range every second.
Target: about 250 samples/s and no bad lines.
"""
import sys
import time

from dsp.sources import SerialSource

port = sys.argv[1] if len(sys.argv) > 1 else "COM5"
src = SerialSource(port)
t0, n, lo, hi = time.time(), 0, 1e9, -1e9
while True:
    for h, v in src.read():
        n += 1
        lo, hi = min(lo, h, v), max(hi, h, v)
    if time.time() - t0 >= 1:
        print(f"{n} samples/s   range {lo:.0f} .. {hi:.0f} uV   bad lines {src.bad_lines}")
        t0, n, lo, hi = time.time(), 0, 1e9, -1e9
    time.sleep(0.01)
