"""Synthetic two-channel EOG generator (Pair B, step B2).

Run:  python -m dsp.generator
Makes fake recordings with known labels so every part of the pipeline can be
tested before the hardware exists. Units: microvolts (uV) at the electrodes.

Each command is acted out the way a user does it:
  LEFT/RIGHT/UP/DOWN -> look there, hold, look back to centre
  BLINK              -> one blink
  DOUBLE_BLINK       -> two blinks about 0.35 s apart
  REST               -> nothing (keep looking at the centre)
"""
import numpy as np
from scipy import signal

from config import FS


def _step(t, t0, amp, rise=0.04):
    """Smooth step (a saccade) of size amp starting at t0."""
    z = np.clip((t - t0 - rise / 2) / (rise / 8), -50, 50)
    return amp / (1 + np.exp(-z))


def _bump(t, t0, amp, dur=0.2):
    """Raised-cosine bump (a blink) of height amp lasting dur s."""
    x = np.zeros_like(t)
    m = (t >= t0) & (t < t0 + dur)
    x[m] = amp * 0.5 * (1 - np.cos(2 * np.pi * (t[m] - t0) / dur))
    return x


def _act(t, cmd, t0, a, v_ratio, hold, gap2=0.35):
    """Add one command to zero arrays. Returns (h, v)."""
    h = np.zeros_like(t)
    v = np.zeros_like(t)
    if cmd in ("LEFT", "RIGHT"):
        s = -1 if cmd == "LEFT" else 1
        h = _step(t, t0, s * a) - _step(t, t0 + hold, s * a)
    elif cmd in ("UP", "DOWN"):
        s = 1 if cmd == "UP" else -1
        av = a * v_ratio
        v = _step(t, t0, s * av) - _step(t, t0 + hold, s * av)
    elif cmd == "BLINK":
        v = _bump(t, t0, 1.2 * a)
    elif cmd == "DOUBLE_BLINK":
        v = _bump(t, t0, 1.2 * a) + _bump(t, t0 + gap2, 1.2 * a)
    return h, v


def make_recording(commands=None, gap=2.5, seed=0, amp=400.0, v_ratio=1.0,
                   hold=0.8, hold_jitter=0.0, reaction=0.0, mains=30.0,
                   noise=5.0, drift=True, natural_blinks=0.0):
    """Return t, h, v (uV) and labels [(cue_time_s, command), ...].

    reaction       : the movement starts this many seconds after the cue
                     (a random 0.6x-1.4x of it each time)
    hold_jitter    : hold time varies by +/- this many seconds
    natural_blinks : un-commanded blinks per minute (real people blink!)
    """
    rng = np.random.default_rng(seed)
    if commands is None:
        commands = ["LEFT", "RIGHT", "UP", "DOWN", "BLINK", "DOUBLE_BLINK"] * 4
        rng.shuffle(commands)

    dur = gap * (len(commands) + 1)
    t = np.arange(0, dur, 1 / FS)
    h = np.zeros_like(t)
    v = np.zeros_like(t)
    labels = []
    busy = []
    for i, cmd in enumerate(commands):
        cue = gap * (i + 1) + rng.uniform(-0.2, 0.2)
        labels.append((cue, cmd))
        t0 = cue + reaction * rng.uniform(0.6, 1.4)
        a = amp * rng.uniform(0.8, 1.2)
        hd = hold + rng.uniform(-hold_jitter, hold_jitter)
        dh, dv = _act(t, cmd, t0, a, v_ratio, hd, gap2=rng.uniform(0.3, 0.42))
        h += dh; v += dv
        if cmd != "REST":
            busy.append((t0 - 0.3, t0 + hd + 0.5))

    # un-commanded natural blinks, placed away from the commands
    n_nat = int(natural_blinks * dur / 60)
    for _ in range(n_nat):
        for _try in range(20):
            tb = rng.uniform(1, dur - 1)
            if all(not (a0 < tb < b0) for a0, b0 in busy):
                v += _bump(t, tb, 1.1 * amp * rng.uniform(0.8, 1.2))
                busy.append((tb - 0.3, tb + 0.5))
                break

    # Real front end has a 0.16 Hz high-pass: held gaze slowly decays.
    b, a_ = signal.butter(1, 0.16, "highpass", fs=FS)
    h, v = signal.lfilter(b, a_, h), signal.lfilter(b, a_, v)

    if drift:
        h += np.cumsum(rng.normal(0, 0.3, t.size))
        v += np.cumsum(rng.normal(0, 0.3, t.size))
    hum = mains * np.sin(2 * np.pi * 50 * t)
    h += hum + rng.normal(0, noise, t.size)
    v += hum + rng.normal(0, noise, t.size)
    return t, h, v, labels


def subject_profile(subject_id):
    """Every fake 'person' has a different signal strength and noise."""
    rng = np.random.default_rng(1000 + subject_id)
    return dict(amp=rng.uniform(220, 500), v_ratio=rng.uniform(0.6, 1.0),
                noise=rng.uniform(4, 12), mains=rng.uniform(10, 60),
                hold=rng.uniform(0.6, 1.0), hold_jitter=0.2, reaction=0.35,
                natural_blinks=5.0)


def movement(cmd, amp=400.0, hold=0.8, fs=FS):
    """Clean 2-second H and V waveforms for one command (no noise).
    Used by the live synthetic source when you press a key."""
    t = np.arange(0, 2.0, 1 / fs)
    return _act(t, cmd, 0.05, amp, 1.0, hold)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t, h, v, labels = make_recording(natural_blinks=6)
    np.savetxt("data/synthetic.csv", np.column_stack([t, h, v]), delimiter=",",
               header="t_s,h_uV,v_uV", comments="", fmt="%.5f")
    print("Saved data/synthetic.csv with", len(labels), "commands")
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    ax[0].plot(t, h); ax[0].set_ylabel("H (uV)")
    ax[1].plot(t, v); ax[1].set_ylabel("V (uV)"); ax[1].set_xlabel("time (s)")
    for t0, cmd in labels:
        ax[0].axvline(t0, color="gray", lw=0.5)
        ax[0].text(t0, ax[0].get_ylim()[1] * 0.8, cmd, fontsize=7, rotation=90)
    plt.tight_layout(); plt.show()
