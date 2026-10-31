"""Train and evaluate classifiers (Pair C, step C4).

Run:  python -m ml.train
Uses every session in data/sessions (from Pair B). Tests three models with
leave-one-person-out: train on everyone except one person, test on that
person, repeat for each person (the honest test for a new user).
Saves the chosen model to data/models/model.joblib.

  rf        Random Forest on summary features
  knn       k-nearest neighbours on summary features
  knn_raw   nearest neighbour on the shrunk raw window (like Usakli 2010)
"""
import argparse
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsp.dataset import CLASSES, list_sessions, load_session, subject_of
from ml.features import featurise
from ml.windows import cut_windows, scales_from, trigger_from

MODEL_PATH = "data/models/model.joblib"


def make_model(name):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=0)
    if name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    return KNeighborsClassifier(n_neighbors=1)


def build(names):
    """-> dict kind -> X, plus y and groups (person of each window)."""
    feats = {"summary": [], "raw": []}
    ys, groups = [], []
    for name in names:
        t, h, v, labels = load_session(name)
        trig = trigger_from(h, v, labels)
        X, y = cut_windows(h, v, labels, trig)
        sc = scales_from(X, y)
        for kind in feats:
            feats[kind].append(featurise(X, sc, kind))
        ys.append(y); groups += [subject_of(name)] * len(y)
    return ({k: np.vstack(v) for k, v in feats.items()},
            np.concatenate(ys), np.array(groups))


def loso(model_name, F, y, groups):
    kind = "raw" if model_name == "knn_raw" else "summary"
    X = F[kind]
    pred = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        m = make_model(model_name).fit(X[~test], y[~test])
        pred[test] = m.predict(X[test])
    return accuracy_score(y, pred), pred


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="rf", choices=["rf", "knn", "knn_raw"])
    a = p.parse_args()
    names = list_sessions()
    if not names:
        raise SystemExit("No sessions in data/sessions. Ask Pair B, or run: "
                         "python -m dsp.collect --simulate")
    F, y, groups = build(names)
    print(f"{len(names)} sessions, {len(set(groups))} people, {len(y)} windows")
    print("windows per class:", {c: int((y == c).sum()) for c in CLASSES})
    results = {}
    for name in ["rf", "knn", "knn_raw"]:
        acc, pred = loso(name, F, y, groups)
        results[name] = pred
        print(f"leave-one-person-out accuracy  {name:8s}: {100 * acc:.1f}%")
    print(f"\nConfusion matrix for {a.save} (rows = true, columns = predicted):")
    cm = confusion_matrix(y, results[a.save], labels=CLASSES)
    print(" " * 13 + " ".join(f"{c[:6]:>6}" for c in CLASSES))
    for c, row in zip(CLASSES, cm):
        print(f"{c:>12} " + " ".join(f"{n:6d}" for n in row))
    kind = "raw" if a.save == "knn_raw" else "summary"
    model = make_model(a.save).fit(F[kind], y)
    os.makedirs("data/models", exist_ok=True)
    joblib.dump({"model": model, "kind": kind, "name": a.save}, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH} ({a.save}, trained on all {len(y)} windows)")
    if a.save == "rf":
        from ml.features import NAMES
        top = sorted(zip(model.feature_importances_, NAMES), reverse=True)[:8]
        print("Most useful features:", ", ".join(f"{n} {i:.2f}" for i, n in top))
