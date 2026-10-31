"""Speller logic with no drawing code (Pair C, step C8).

Selection = DWELL + BLINK CONFIRM (as in the project proposal):
  1. LEFT/RIGHT/UP/DOWN move the cursor.
  2. Keep still on a key for DWELL_S seconds -> the key is "armed".
  3. A deliberate DOUBLE_BLINK while armed types the key.
Single blinks are ignored (people blink 12-19 times a minute), and dwell on
its own never types anything (the "Midas touch" problem).
Backspace = the DEL key.

Kept separate from the window so it can be tested without a screen:
  python -m ui.speller_model
"""
import time

LETTERS = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ_.?") + ["DEL"],
    ["NEEDS", "CLEAR", "SPEAK", "", "", ""],
]
NEEDS = [
    ["WATER", "FOOD", "TOILET"],
    ["HELP", "PAIN", "YES"],
    ["NO", "THANK YOU", "BACK"],
]


DWELL_S = 1.0


class SpellerModel:
    def __init__(self, target=None, dwell_s=DWELL_S):
        self.page = "letters"
        self.r = self.c = 0
        self.text = ""
        self.message = ""            # last full message from the needs page
        self.target = target         # e.g. "WATER" for the timed test
        self.t_start = None
        self.t_done = None
        self.selections = 0
        self.deletes = 0
        self.ignored_blinks = 0          # double blinks before the key was armed
        self.dwell_s = dwell_s
        self.last_move = time.time()

    @property
    def grid(self):
        return LETTERS if self.page == "letters" else NEEDS

    def current(self):
        return self.grid[self.r][self.c]

    def dwell_progress(self, now=None):
        """0..1: how far the current key is towards being armed."""
        now = time.time() if now is None else now
        return min(1.0, (now - self.last_move) / self.dwell_s)

    def armed(self, now=None):
        return self.dwell_progress(now) >= 1.0

    def handle(self, event, now=None):
        """Apply one recogniser event. Returns True if the screen should redraw."""
        now = time.time() if now is None else now
        if self.t_start is None and event in ("LEFT", "RIGHT", "UP", "DOWN", "DOUBLE_BLINK"):
            self.t_start = now
        rows, cols = len(self.grid), len(self.grid[0])
        if event in ("LEFT", "RIGHT", "UP", "DOWN"):
            if event == "LEFT":
                self.c = (self.c - 1) % cols
            elif event == "RIGHT":
                self.c = (self.c + 1) % cols
            elif event == "UP":
                self.r = (self.r - 1) % rows
            else:
                self.r = (self.r + 1) % rows
            self.last_move = now
            return True
        if event == "DOUBLE_BLINK":
            if self.armed(now):
                self.select(now)
                self.last_move = now        # must dwell again before the next letter
            else:
                self.ignored_blinks += 1
            return True
        return False                        # single BLINK: ignored

    def select(self, now=None):
        key = self.current()
        self.selections += 1
        if key == "":
            return
        if self.page == "needs":
            if key == "BACK":
                self.page, self.r, self.c = "letters", 0, 0
            else:
                self.message = key
            return
        if key == "DEL":
            self.text = self.text[:-1]; self.deletes += 1
        elif key == "CLEAR":
            self.text = ""
        elif key == "NEEDS":
            self.page, self.r, self.c = "needs", 0, 0
        elif key == "SPEAK":
            self.message = self.text
        elif key == "_":
            self.text += " "
        else:
            self.text += key
        if self.target and self.text == self.target and self.t_done is None:
            self.t_done = time.time() if now is None else now

    def result(self):
        """Numbers for the evaluation table (C5)."""
        if self.t_start is None or self.t_done is None:
            return None
        secs = self.t_done - self.t_start
        return {"target": self.target, "seconds": round(secs, 1),
                "letters_per_min": round(60 * len(self.target) / secs, 2),
                "selections": self.selections, "deletes": self.deletes,
                "ignored_blinks": self.ignored_blinks}


if __name__ == "__main__":
    m = SpellerModel(target="WATER")
    t = 0.0
    for ev in ["DOWN"] * 3 + ["RIGHT"] * 4:     # W is row 3, column 4
        t += 0.5
        m.handle(ev, now=t)
    m.handle("DOUBLE_BLINK", now=t + 0.3)     # too soon: not armed yet
    assert m.text == "" and m.ignored_blinks == 1
    m.handle("BLINK", now=t + 1.2)            # natural blink: ignored
    m.handle("DOUBLE_BLINK", now=t + 1.5)     # after dwell: types W
    print("typed:", m.text, "| cursor on:", m.current())
    assert m.text == "W"
    print("speller model OK")
