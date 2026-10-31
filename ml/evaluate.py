"""Compare the two recognisers on new people (Pair C, step C7).

Run:  python -m ml.evaluate
For each fake person: train the ML model on the OTHER people's sessions,
calibrate on a short calibration recording of this person, then run both
the threshold detector and the live ML classifier on a test recording
that also contains natural (un-commanded) blinks.

An event is correct if it matches the cue and arrives within 2.5 s.
"Harmful false events" = wrong events other than BLINK (the speller
ignores single blinks, so those cannot type anything).
"""
from collections import Counter

import numpy as np

from dsp.filters import filter_array
from dsp.generator import make_recording, subject_profile


def score(labels, events, window=2.5):
    used, correct, per_cmd = set(), 0, Counter()
    totals = Counter(c for _, c in labels)
    for t0, cmd in labels:
        for i, (te, ev) in enumerate(events):
            if i not in used and ev == cmd and 0 <= te - t0 <= window:
                used.add(i); correct += 1; per_cmd[cmd] += 1
                if cmd == "DOUBLE_BLINK":      # threshold detector also reports its first BLINK
                    for j, (tb, eb) in enumerate(events):
                        if eb == "BLINK" and 0 <= tb - t0 <= window:
                            used.add(j)
                break
    false_events = [e for i, e in enumerate(events) if i not in used]
    return correct, per_cmd, totals, false_events


def run(recogniser, h, v):
    events = []
    for a, b in zip(h, v):
        for e in recogniser.step(a, b):
            events.append((recogniser_time(recogniser), e))
    return events


def recogniser_time(r):
    return r.t if hasattr(r, "t") else r.n / r.fs


if __name__ == "__main__":
    import joblib
    from dsp.dataset import list_sessions, subject_of
    from ml.calibration_math import compute_calibration
    from ml.live import LiveClassifier
    from ml.threshold_detector import Detector
    from ml.train import build, make_model

    names = [n for n in list_sessions() if n.startswith("fake")]
    if not names:
        raise SystemExit("Run first: python -m dsp.collect --simulate")
    F, y, groups = build(names)
    print(f"{'person':8s} {'threshold':>22s} {'ML (Random Forest)':>24s}")
    tot = {"thr": [0, 0, 0], "ml": [0, 0, 0]}
    for g in sorted(set(groups)):
        sid = int(g.replace("fake", ""))
        prof = subject_profile(sid)
        model = make_model("rf").fit(F["summary"][groups != g], y[groups != g])
        joblib.dump({"model": model, "kind": "summary", "name": "rf"}, "data/models/_eval.joblib")
        t, h, v, prompts = make_recording(["LEFT", "RIGHT", "UP", "DOWN"] * 5, gap=3.0,
                                          seed=500 + sid, **prof)
        cal = compute_calibration(filter_array(h), filter_array(v), prompts)
        t, h, v, labels = make_recording(gap=3.0, seed=900 + sid, **prof)
        hf, vf = filter_array(h), filter_array(v)
        row = f"{g:8s}"
        for key, rec in (("thr", Detector(cal)),
                         ("ml", LiveClassifier(cal, "data/models/_eval.joblib"))):
            c, per, totals, false = score(labels, run(rec, hf, vf))
            harmful = [e for _, e in false if e != "BLINK"]
            tot[key][0] += c; tot[key][1] += len(labels); tot[key][2] += len(harmful)
            row += f"   {100 * c / len(labels):5.1f}% ok, {len(harmful):2d} harmful"
        print(row)
    print("-" * 58)
    print(f"{'overall':8s}" + "".join(
        f"   {100 * a / b:5.1f}% ok, {f:2d} harmful" for a, b, f in tot.values()))
