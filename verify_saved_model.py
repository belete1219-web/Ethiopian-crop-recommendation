"""Verify that the saved model reproduces the reported results (run: python verify_saved_model.py).

Rebuilds the stratified 70/15/15 split (random_state=42) from the CSV, loads the saved model, and checks that
  1. the model and its metadata agree on features and classes,
  2. test accuracy / Macro-F1 / top-3 accuracy on the 90 held-out rows equal outputs/test_metrics.csv,
  3. the model was NOT fitted on the test rows (a model that had seen them would score ~100%, not ~82%),
  4. it was fitted on the train + validation rows, whose feature ranges match model_metadata.json.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


df = pd.read_csv(BASE / "data" / "Crop_recommendation_Ethio.csv")
X, y = df[FEATURES], df["label"].str.strip().str.lower()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

model = joblib.load(BASE / "crop_recommendation_model.joblib")
meta = json.loads((BASE / "model_metadata.json").read_text())
saved = pd.read_csv(BASE / "outputs" / "test_metrics.csv").iloc[0]

print(f"Rows: train={len(X_tr)} validation={len(X_val)} test={len(X_te)} | model: {type(model).__name__}, "
      f"{model.n_estimators} trees | metadata says: {meta['selected_model']}")
check("model features match metadata", list(model.feature_names_in_) == meta["features"] == FEATURES)
check("model classes match metadata", [str(c) for c in model.classes_] == meta["classes"],
      f"{len(model.classes_)} classes")

proba = model.predict_proba(X_te)
pred = model.classes_[proba.argmax(1)]
acc = accuracy_score(y_te, pred)
mf1 = f1_score(y_te, pred, average="macro")
top3 = top_k_accuracy_score(y_te, proba, k=3, labels=model.classes_)
check("test accuracy reproduces saved value", np.isclose(acc, saved["accuracy"]),
      f"{acc:.4f} vs {saved['accuracy']:.4f}")
check("test Macro-F1 reproduces saved value", np.isclose(mf1, saved["macro_f1"]),
      f"{mf1:.4f} vs {saved['macro_f1']:.4f}")
check("test top-3 accuracy reproduces saved value", np.isclose(top3, saved["top3_accuracy"]),
      f"{top3:.4f} vs {saved['top3_accuracy']:.4f}")
check("metadata test metrics agree",
      np.isclose(meta["test_metrics"]["accuracy"], acc) and np.isclose(meta["test_metrics"]["top3_accuracy"], top3))

fit_rows = pd.concat([X_tr, X_val])
y_fit = pd.concat([y_tr, y_val])
fit_acc = accuracy_score(y_fit, model.predict(fit_rows))
check("model is not fitted on test rows (held-out accuracy well below fitted-rows accuracy)",
      acc < 0.95 and fit_acc > acc, f"fitted rows {fit_acc:.3f} vs test {acc:.3f}")
ref = meta["input_reference"]["feature_ranges"]
check("metadata feature ranges equal the train+validation rows",
      all(np.isclose(ref[f]["min"], fit_rows[f].min()) and np.isclose(ref[f]["max"], fit_rows[f].max())
          for f in FEATURES))
check("test rows and fitted rows do not overlap", set(X_te.index).isdisjoint(fit_rows.index))

sample = pd.DataFrame([[60, 30, 35, 22, 60, 6.4, 700]], columns=FEATURES)
p = model.predict_proba(sample)[0]
top = p.argsort()[::-1][:3]
print("Example top-3:", [(str(model.classes_[i]), round(float(p[i]), 3)) for i in top])
check("example prediction matches outputs/example_top3_prediction.csv",
      [str(model.classes_[i]) for i in top]
      == pd.read_csv(BASE / "outputs" / "example_top3_prediction.csv")["recommended_crop"].tolist())

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
