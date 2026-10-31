# EOG Speller

Eye-controlled typing using electrooculography (EOG). IIIT Bangalore E-Lab project.
Electrodes -> analog front end -> UNO Q + ADS1115 -> filters -> ML classifier
(threshold detector as baseline) -> speller with dwell + blink confirm.

## Quick start (no hardware needed)
```
venv\Scripts\activate
python -m dsp.collect --simulate      # fake labelled dataset: 6 people x 2 sessions
python -m ml.train                    # train + leave-one-person-out test, saves the model
python main.py --calibrate            # arrow keys = eye movements, SPACE = double blink
python -m ml.evaluate                 # ML vs threshold baseline on new fake people
```

## Pairs and folders
| Pair | Owns |
|---|---|
| A - Hardware | `sim/`, `hardware/` |
| B - Acquisition & Data | `firmware/`, `dsp/`, `data/sessions/` (the labelled dataset) |
| C - ML & Interface | `ml/`, `ui/`, `main.py`, `data/models/`, results |

`config.py`, this README and `.gitignore` are shared: change them only after telling all pairs.

## Interfaces (agreed by all pairs)
| From -> To | Agreement |
|---|---|
| A -> B | H on ADS1115 AIN0, V on AIN1. Centred at 1.65 V, about +/-0.5 V for a +/-1 mV eye signal. Look RIGHT -> H rises, look UP -> V rises. Total gain in `config.ANALOG_GAIN`. Common ground. |
| UNO Q -> PC | `Monitor` over USB, one line per sample `h,v`, raw ADC counts, 250 samples/s |
| Sources | every source has `.read()` -> list of (h_uV, v_uV) |
| B -> C (dataset) | `data/sessions/<person>_sNN.csv` (t_s,h_uV,v_uV raw) + `_labels.csv` (t_s,label). Read with `dsp.dataset.load_session(name)` (returns filtered signals). Labels: LEFT RIGHT UP DOWN BLINK DOUBLE_BLINK REST |
| Recogniser -> speller | `.step(h_uV, v_uV)` returns a list of events (same names, REST never sent) |
| Calibration | `data/calibration.json`: trigger, h_scale, v_scale, h_slope, h_amp, v_slope, v_amp |
