import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 250                      # sampling rate (Hz), same as your EOG plan
t = np.arange(0, 2, 1/fs)     # 2 seconds of time

clean = np.where(t > 1, 300e-6, 0)            # a 300 µV step, like a saccade
noisy = clean + 50e-6*np.sin(2*np.pi*50*t)    # add 50 Hz mains hum

b, a = signal.iirnotch(50, 30, fs)            # 50 Hz notch filter
filtered = signal.filtfilt(b, a, noisy)

plt.plot(t, noisy*1e6, label="noisy")
plt.plot(t, filtered*1e6, label="filtered")
plt.xlabel("time (s)"); plt.ylabel("µV"); plt.legend()
plt.title("Setup test: fake saccade + 50 Hz noise")
plt.show()