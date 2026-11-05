"""Input-familiarity checks shared by the Streamlit app, the notebook and the tests.

A number can be a *valid* input (finite, pH between 0 and 14, humidity between 0 and 100)
and still be an *unreliable* input, because the model only learned from rows that fall
inside a limited region of feature space.  Two kinds of check are made:

1. Per-feature range check  - is each value inside the range seen when the model was fitted?
2. Combination check        - even if every value is individually in range, is the combination
                              close to at least one fitted row?  (nearest-neighbour distance in
                              standardised feature space, calibrated on the fitted rows themselves)

The thresholds are stored in model_metadata.json by the notebook, so they always describe the
same rows that the saved model was fitted on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Severity order: higher = worse.
LEVELS = {"familiar": 0, "edge": 1, "unreliable": 2}


@dataclass(frozen=True)
class Finding:
    level: str            # "edge" or "unreliable"
    kind: str             # "outside_range", "edge_of_range" or "unfamiliar_combination"
    feature: str | None   # None for the combination check
    message: str


def build_reference(X_fit: pd.DataFrame, features: Iterable[str] = FEATURES) -> dict:
    """Summarise the rows the model was fitted on (ranges + nearest-neighbour calibration)."""
    features = list(features)
    X = X_fit[features].astype(float)
    mean, std = X.mean(), X.std(ddof=0).replace(0, 1.0)
    Z = ((X - mean) / std).to_numpy()
    d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)                 # leave-one-out: a row is not its own neighbour
    nn = d.min(axis=1)
    return {
        "fitted_rows": int(len(X)),
        "feature_ranges": {
            f: {"min": float(X[f].min()), "max": float(X[f].max()),
                "p01": float(X[f].quantile(0.01)), "p99": float(X[f].quantile(0.99))}
            for f in features
        },
        "scaler": {"mean": {f: float(mean[f]) for f in features},
                   "std": {f: float(std[f]) for f in features}},
        "novelty_check": {
            "method": "distance to nearest fitted row in standardised feature space (leave-one-out calibration)",
            "threshold_p99": float(np.percentile(nn, 99)),
            "loo_distance_percentiles": {str(p): float(np.percentile(nn, p)) for p in (50, 90, 95, 99, 100)},
        },
    }


def nearest_distance(values: Mapping[str, float], reference: Mapping, fitted_features: pd.DataFrame) -> float:
    """Distance from one input to its nearest fitted row (standardised units)."""
    feats = list(reference["feature_ranges"])
    mean, std = reference["scaler"]["mean"], reference["scaler"]["std"]
    z = np.array([(float(values[f]) - mean[f]) / std[f] for f in feats])
    Z = ((fitted_features[feats].astype(float) - pd.Series(mean)) / pd.Series(std)).to_numpy()
    return float(np.linalg.norm(Z - z, axis=1).min())


def assess_inputs(values: Mapping[str, float], reference: Mapping,
                  fitted_features: pd.DataFrame | None = None,
                  labels: Mapping[str, str] | None = None) -> list[Finding]:
    """Return a list of findings (empty list = every value looks familiar to the model)."""
    labels = labels or {}
    findings: list[Finding] = []
    outside_any = False
    for f, r in reference["feature_ranges"].items():
        v, name = float(values[f]), labels.get(f, f)
        lo, hi = r["min"], r["max"]
        if v < lo or v > hi:
            outside_any = True
            findings.append(Finding(
                "unreliable", "outside_range", f,
                f"{name} = {v:g} is outside the range the model was fitted on ({lo:.4g} to {hi:.4g}). "
                "The model has never seen a value like this, so its ranking is an extrapolation."))
        elif v < r["p01"] or v > r["p99"]:
            findings.append(Finding(
                "edge", "edge_of_range", f,
                f"{name} = {v:g} is at the extreme edge of the fitted range ({lo:.4g} to {hi:.4g}); "
                "fewer than 1 in 100 fitted rows are this extreme."))
    if fitted_features is not None and not outside_any:
        dist = nearest_distance(values, reference, fitted_features)
        thr = reference["novelty_check"]["threshold_p99"]
        if dist > thr:
            findings.append(Finding(
                "unreliable", "unfamiliar_combination", None,
                "Each value is individually within the fitted range, but this combination of values is unlike "
                f"any row the model was fitted on (distance {dist:.2f} vs. {thr:.2f} for the most unusual 1% of "
                "fitted rows)."))
    return findings


def overall_level(findings: list[Finding]) -> str:
    """'familiar', 'edge' or 'unreliable' - the worst level among the findings."""
    worst = "familiar"
    for f in findings:
        if LEVELS[f.level] > LEVELS[worst]:
            worst = f.level
    return worst
