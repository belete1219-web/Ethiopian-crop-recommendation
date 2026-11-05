"""Experiment tracking with MLflow (run: python track_experiments.py ; view: mlflow ui --backend-store-uri sqlite:///mlflow.db).

Re-runs the four Module 5 candidate models with the SAME split (stratified 70/15/15, random_state=42) and
hyper-parameters as the notebook, logs parameters and validation metrics for each, then logs a final run that
evaluates the SAVED model on the held-out test rows. Everything is also exported to outputs/mlflow_runs.csv so a
reviewer can read the results without starting the MLflow UI.
"""
import json
import warnings
from pathlib import Path

import joblib
import os

# MLflow tries to record the Git branch and URL with each run. If Git is not installed (common on Windows)
# it prints long warnings; they are harmless, so silence them. Must be set before importing mlflow.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SEED = 42

mlflow.set_tracking_uri(f"sqlite:///{BASE / 'mlflow.db'}")
mlflow.set_experiment("crop-recommendation-module6")

df = pd.read_csv(BASE / "data" / "Crop_recommendation_Ethio.csv")
X, y = df[FEATURES], df["label"].str.strip().str.lower()
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)

DATA_NOTE = "600-row CSV of undocumented origin; statistically appears synthetic; not verified Ethiopian field data"
candidates = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}

for name, model in candidates.items():
    with mlflow.start_run(run_name=f"candidate: {name}"):
        mlflow.set_tags({"stage": "model_selection", "data_note": DATA_NOTE})
        mlflow.log_params({"model": name, "seed": SEED, "split": "70/15/15 stratified",
                           "n_train": len(X_tr), "n_validation": len(X_val), **model.get_params()})
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        mlflow.log_metrics({"val_accuracy": accuracy_score(y_val, pred),
                            "val_macro_f1": f1_score(y_val, pred, average="macro", zero_division=0)})

# Final run: the model that was actually saved (fitted on train + validation), scored on the untouched test rows.
saved = joblib.load(BASE / "crop_recommendation_model.joblib")
meta = json.loads((BASE / "model_metadata.json").read_text())
proba = saved.predict_proba(X_te)
pred = saved.classes_[proba.argmax(1)]
with mlflow.start_run(run_name="final: saved Random Forest on held-out test"):
    mlflow.set_tags({"stage": "final_evaluation", "data_note": DATA_NOTE,
                     "claim_limit": "software-validation result, not agricultural validation"})
    mlflow.log_params({"model": meta["selected_model"], "seed": SEED, "fitted_on": meta["fitted_on"],
                       "n_test": len(X_te), "sklearn": meta["software"]["scikit-learn"],
                       "selection_metric": "validation Macro-F1"})
    mlflow.log_metrics({"test_accuracy": accuracy_score(y_te, pred),
                        "test_macro_f1": f1_score(y_te, pred, average="macro"),
                        "test_top3_accuracy": top_k_accuracy_score(y_te, proba, k=3, labels=saved.classes_)})
    for art in ("model_metadata.json", "outputs/confusion_matrix.csv", "outputs/per_class_metrics.csv",
                "outputs/five_fold_cv.csv"):
        mlflow.log_artifact(str(BASE / art))
    mlflow.log_artifact(str(BASE / "crop_recommendation_model.joblib"), artifact_path="model")

runs = mlflow.search_runs(experiment_names=["crop-recommendation-module6"], order_by=["start_time ASC"])
keep = ["tags.mlflow.runName", "params.model", "metrics.val_accuracy", "metrics.val_macro_f1",
        "metrics.test_accuracy", "metrics.test_macro_f1", "metrics.test_top3_accuracy", "run_id"]
out = runs[keep].rename(columns=lambda c: c.split(".", 1)[-1])
out.to_csv(BASE / "outputs" / "mlflow_runs.csv", index=False)
print(out.drop(columns="run_id").round(4).to_string(index=False))
