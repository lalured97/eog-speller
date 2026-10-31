"""Check the quality of one recorded session (Pair B, step B8).

Run:  python -m dsp.inspect_session lalith_s01
Prints how many cues of each class produced a usable window (using the same
windowing as Pair C) and plots the session with the cues marked.
A good session has at least 8 of 10 usable cues for every class.
"""
import sys
from collections import Counter

import matplotlib.pyplot as plt

from dsp.dataset import CLASSES, load_session
from ml.windows import cut_windows, trigger_from

name = sys.argv[1] if len(sys.argv) > 1 else "fake1_s01"
t, h, v, labels = load_session(name)
trig = trigger_from(h, v, labels)
X, y = cut_windows(h, v, labels, trig)
asked = Counter(lab for _, lab in labels)
got = Counter(y)
print(f"{name}: {t[-1]:.0f} s, onset threshold {trig:.0f} uV per 20 ms")
for c in CLASSES:
    flag = "" if got[c] >= 0.8 * asked[c] else "   <-- check electrodes / instructions"
    print(f"  {c:13s} usable {got[c]:2d} / {asked[c]:2d}{flag}")

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(13, 6))
ax[0].plot(t, h, lw=0.7); ax[0].set_ylabel("H (uV)")
ax[1].plot(t, v, lw=0.7, color="tab:orange"); ax[1].set_ylabel("V (uV)")
ax[1].set_xlabel("time (s)")
for t0, lab in labels:
    for a in ax:
        a.axvline(t0, color="gray", lw=0.4)
    ax[0].text(t0, ax[0].get_ylim()[1] * 0.85, lab, fontsize=6, rotation=90)
ax[0].set_title(name)
plt.tight_layout(); plt.show()
