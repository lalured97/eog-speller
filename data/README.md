# data
- `sessions/` labelled dataset from Pair B (`<person>_sNN.csv` + `_labels.csv`, `sessions_log.csv`).
  Fake sessions (`fake*`) are not committed: make them with `python -m dsp.collect --simulate`.
- `models/` trained models (not committed: run `python -m ml.train`).
- `calibration.json` the current user's calibration. `session_log.csv` events. `results.csv` test results.
Do not upload downloaded datasets or papers (the repo is public).
