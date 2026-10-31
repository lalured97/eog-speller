"""Speller window (Pair C, step C8). Uses Tkinter (built into Python).

Run on its own to test with the keyboard:
    python -m ui.speller
Arrow keys = eye movements, SPACE = double blink (confirm), B = single blink.
The highlighted key turns green when it is armed (after the dwell time);
only then does a double blink type it.
"""
import tkinter as tk

from ui.speller_model import SpellerModel

BG, CELL, HI, ARMED, TXT = "#1e1e1e", "#2d2d30", "#f2c94c", "#6fcf97", "#ffffff"


class SpellerWindow:
    def __init__(self, root, model=None):
        self.root, self.m = root, model or SpellerModel()
        root.winfo_toplevel().title("EOG Speller")
        root.configure(bg=BG)
        self.textvar = tk.StringVar()
        tk.Label(root, textvariable=self.textvar, font=("Consolas", 28),
                 bg="black", fg=TXT, anchor="w", width=30).pack(fill="x", padx=10, pady=10)
        self.msgvar = tk.StringVar()
        tk.Label(root, textvariable=self.msgvar, font=("Arial", 18, "bold"),
                 bg=BG, fg="#6fcf97").pack()
        self.gridframe = tk.Frame(root, bg=BG)
        self.gridframe.pack(padx=10, pady=10)
        self.bar = tk.Canvas(root, width=300, height=12, bg="#333333", highlightthickness=0)
        self.bar.pack()
        self.statusvar = tk.StringVar(value="ready")
        tk.Label(root, textvariable=self.statusvar, bg=BG, fg="#aaaaaa").pack(pady=4)
        self.cells = []
        self.drawn_page = None
        self.redraw()
        self.tick()

    def tick(self):
        """Every 50 ms: update the dwell bar and the armed colour."""
        p = self.m.dwell_progress()
        self.bar.delete("all")
        self.bar.create_rectangle(0, 0, 300 * p, 12, fill=ARMED if p >= 1 else HI, width=0)
        if self.cells:
            lab = self.cells[self.m.r][self.m.c]
            lab.configure(bg=ARMED if p >= 1 else HI)
        self.root.after(50, self.tick)

    def build_grid(self):
        for w in self.gridframe.winfo_children():
            w.destroy()
        self.cells = []
        for r, row in enumerate(self.m.grid):
            line = []
            for c, key in enumerate(row):
                lab = tk.Label(self.gridframe, text=key, width=8 if self.m.page == "needs" else 5,
                               height=2, font=("Arial", 20, "bold"), bg=CELL, fg=TXT)
                lab.grid(row=r, column=c, padx=3, pady=3)
                line.append(lab)
            self.cells.append(line)
        self.drawn_page = self.m.page

    def redraw(self):
        if self.drawn_page != self.m.page:
            self.build_grid()
        for r, line in enumerate(self.cells):
            for c, lab in enumerate(line):
                on = (r, c) == (self.m.r, self.m.c)
                lab.configure(bg=HI if on else CELL, fg="black" if on else TXT)
        self.textvar.set(self.m.text + "|")
        self.msgvar.set(self.m.message)

    def flash(self):
        lab = self.cells[self.m.r][self.m.c]
        lab.configure(bg="#ffffff")
        self.root.after(200, self.redraw)

    def on_event(self, event):
        """Called by main.py for every recogniser event."""
        before = self.m.selections
        if self.m.handle(event):
            self.redraw()
            if self.m.selections > before:
                self.flash()
                self.statusvar.set(f"typed with {event}")
            elif event == "DOUBLE_BLINK":
                self.statusvar.set("hold still until the key turns green, then blink twice")
            else:
                self.statusvar.set(f"last event: {event}")
            res = self.m.result()
            if res:
                self.statusvar.set(f"TEST DONE: {res}")


def bind_keyboard(root, callback):
    """Arrow keys and SPACE produce the same events the detector sends."""
    keys = {"Left": "LEFT", "Right": "RIGHT", "Up": "UP", "Down": "DOWN",
            "space": "DOUBLE_BLINK", "b": "BLINK"}
    for k, ev in keys.items():
        root.bind(f"<{k}>" if len(k) > 1 else k, lambda e, ev=ev: callback(ev))


if __name__ == "__main__":
    root = tk.Tk()
    win = SpellerWindow(root, SpellerModel(target="WATER"))
    bind_keyboard(root, win.on_event)
    root.mainloop()
