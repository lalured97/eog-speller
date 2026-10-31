"""Main app: source -> filters -> recogniser -> speller (Pair C, step C9).

From the repo folder, with the venv on:

  python main.py                         fake signal; arrow keys / SPACE inject eye movements
  python main.py --calibrate             calibration first, then the speller
  python main.py --recogniser threshold  use the baseline instead of the ML model
  python main.py --source csv --file data/synthetic.csv
  python main.py --source serial --port COM5 --calibrate     real board
  python main.py --target WATER --user NAME   timed test (result in data/results.csv)
  python main.py ... --record data/rec_name_01.csv           also save the raw signal

Recogniser: "ml" (data/models/model.joblib from ml.train) is used when the
model exists, otherwise the threshold detector. Every event is logged to
data/session_log.csv with the recogniser name.
"""
import argparse
import csv
import os
import time
import tkinter as tk

from config import FS
from dsp.filters import StreamFilter
from ml import calibration_math
from ml.threshold_detector import Detector
from ui.calibration import CalibrationWindow
from ui.speller import SpellerWindow, bind_keyboard
from ui.speller_model import SpellerModel

PLOT_SECONDS, PLOT_W, PLOT_H = 5, 700, 160


def make_source(args):
    if args.source == "serial":
        from dsp.sources import SerialSource
        return SerialSource(args.port)
    if args.source == "csv":
        from dsp.sources import CsvSource
        return CsvSource(args.file)
    from dsp.sources import SyntheticSource
    return SyntheticSource()


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        os.makedirs("data", exist_ok=True)
        self.src = make_source(args)
        self.fh, self.fv = StreamFilter(), StreamFilter()
        self.hist_h, self.hist_v = [], []
        self.log = open("data/session_log.csv", "a", newline="")
        self.logw = csv.writer(self.log)
        self.inject = getattr(self.src, "inject", None)
        self.rec = open(args.record, "w") if args.record else None
        self.n_rec = 0
        if self.rec:
            self.rec.write("t_s,h_uV,v_uV\n")

        self.main = tk.Frame(root); self.main.pack()
        self.plot = tk.Canvas(root, width=PLOT_W, height=PLOT_H, bg="black")
        self.plot.pack(pady=5)
        self.speller = None
        self.cal = None
        if args.calibrate:
            self.cal = CalibrationWindow(self.main, on_done=self.start_speller,
                                         inject=self.inject)
        else:
            self.start_speller(calibration_math.load())
        root.after(20, self.tick)

    def start_speller(self, cal):
        print("Calibration:", cal or "none (defaults) - run with --calibrate")
        self.det = self.make_recogniser(cal)
        for w in self.main.winfo_children():
            w.destroy()
        self.cal = None
        self.speller = SpellerWindow(self.main, SpellerModel(target=self.args.target))
        # In synthetic mode arrow keys fake an eye movement in the SIGNAL,
        # so the whole filter + detector chain is exercised.
        bind_keyboard(self.root, self.inject or self.speller.on_event)

    def make_recogniser(self, cal):
        model = "data/models/model.joblib"
        want = self.args.recogniser
        if want == "ml" and not os.path.exists(model):
            print("No ML model yet (run: python -m ml.train). Using the threshold detector.")
            want = "threshold"
        self.rec_name = want
        if want == "ml":
            from ml.live import LiveClassifier
            return LiveClassifier(cal, model)
        return Detector(cal)

    def tick(self):
        for h, v in self.src.read():
            if self.rec:
                self.rec.write(f"{self.n_rec / FS:.4f},{h:.2f},{v:.2f}\n")
                self.n_rec += 1
            h, v = self.fh.step(h), self.fv.step(v)
            self.hist_h.append(h); self.hist_v.append(v)
            if self.cal:
                self.cal.add_samples(h, v)
            elif self.speller:
                for ev in self.det.step(h, v):
                    self.logw.writerow([f"{time.time():.3f}", self.rec_name, ev])
                    self.speller.on_event(ev)
        n = PLOT_SECONDS * FS
        self.hist_h, self.hist_v = self.hist_h[-n:], self.hist_v[-n:]
        self.draw_plot()
        if self.speller and self.args.target:
            res = self.speller.m.result()
            if res and not getattr(self, "saved", False):
                self.save_result(res)
        self.root.after(20, self.tick)

    def draw_plot(self):
        self.plot.delete("all")
        for data, color, y0 in [(self.hist_h, "#56ccf2", PLOT_H * 0.3),
                                (self.hist_v, "#f2994a", PLOT_H * 0.75)]:
            if len(data) < 2:
                continue
            step = max(1, len(data) // PLOT_W)
            pts = []
            for i in range(0, len(data), step):
                x = i * PLOT_W / (PLOT_SECONDS * FS)
                y = y0 - data[i] * 0.06          # 0.06 px per uV
                pts += [x, max(0, min(PLOT_H, y))]
            self.plot.create_line(*pts, fill=color)
        self.plot.create_text(30, 10, text="H", fill="#56ccf2")
        self.plot.create_text(30, PLOT_H * 0.55, text="V", fill="#f2994a")

    def save_result(self, res):
        self.saved = True
        new = not os.path.exists("data/results.csv")
        with open("data/results.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user", "source", "recogniser", *res.keys()])
            if new:
                w.writeheader()
            w.writerow({"user": self.args.user, "source": self.args.source,
                        "recogniser": self.rec_name, **res})
        print("Result saved:", res)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthetic", "csv", "serial"], default="synthetic")
    p.add_argument("--port", default="COM5")
    p.add_argument("--file", default="data/synthetic.csv")
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--target", default=None)
    p.add_argument("--user", default="test")
    p.add_argument("--record", default=None, help="save raw samples to this CSV")
    p.add_argument("--recogniser", choices=["ml", "threshold"], default="ml")
    args = p.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()
