"""Monitoring report for logged API traffic (run: python monitor_model.py logs/simulated_week1.jsonl logs/simulated_week2.jsonl).

For each log file it reports, against the data the model was fitted on:
  * share of requests flagged "unreliable" (outside fitted range / unfamiliar combination) and "edge"
  * median and low-confidence share of the top-1 score
  * Population Stability Index (PSI) per input feature (10 quantile bins of the fitted rows)
  * top-1 crop mix
and raises ALERT / WATCH flags using the starting thresholds below. The thresholds are conventional starting
points (PSI > 0.2 is a common "significant shift" rule of thumb) and must be tuned with real traffic.

Limits: this detects INPUT drift and unfamiliar inputs. It cannot measure accuracy, because real crop outcomes
(the ground truth) are not collected by the prototype; see the monitoring plan for how outcomes would be gathered.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
THRESHOLDS = {"unreliable_share": (0.05, 0.10), "psi": (0.10, 0.20), "low_conf_share": (0.30, 0.50)}  # (watch, alert)
fit = pd.read_csv(BASE / "outputs" / "model_training_features.csv")


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def flag(value, key):
    watch, alert = THRESHOLDS[key]
    return "ALERT" if value >= alert else "WATCH" if value >= watch else "ok"


def summarise(path: Path) -> dict:
    recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(recs)
    inputs = pd.DataFrame(list(df["inputs"]))
    out = {"file": path.name, "source": ",".join(sorted(df["source"].unique())), "requests": len(df),
           "unreliable_share": float((df["level"] == "unreliable").mean()),
           "edge_share": float((df["level"] == "edge").mean()),
           "median_top1_score": float(df["top1_score"].median()),
           "low_conf_share": float((df["top1_score"] < 0.5).mean()),
           "psi": {f: psi(fit[f], inputs[f]) for f in FEATURES},
           "top1_mix": df["top1"].value_counts(normalize=True).round(3).to_dict()}
    return out


def render(s: dict) -> str:
    lines = [f"## {s['file']}  (source tag: {s['source']}; {s['requests']} requests)", ""]
    lines += ["| Check | Value | Status |", "|---|---|---|",
              f"| Share flagged unreliable | {s['unreliable_share']:.1%} | {flag(s['unreliable_share'], 'unreliable_share')} |",
              f"| Share flagged edge-of-range | {s['edge_share']:.1%} | - |",
              f"| Median top-1 score | {s['median_top1_score']:.2f} | - |",
              f"| Share with top-1 score < 0.5 | {s['low_conf_share']:.1%} | {flag(s['low_conf_share'], 'low_conf_share')} |"]
    for f, v in s["psi"].items():
        lines.append(f"| PSI: {f} | {v:.3f} | {flag(v, 'psi')} |")
    mix = ", ".join(f"{k} {v:.0%}" for k, v in list(s["top1_mix"].items())[:5])
    lines += ["", f"Top-1 crop mix (top 5): {mix}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]] or [BASE / "logs" / "predictions.jsonl"]
    report = ["# Monitoring report", "",
              "Compared with the 510 rows the model was fitted on. PSI thresholds: watch >= 0.10, alert >= 0.20.", ""]
    for p in paths:
        report.append(render(summarise(p)))
    text = "\n".join(report)
    (BASE / "outputs").mkdir(exist_ok=True)
    (BASE / "outputs" / "monitoring_report.md").write_text(text)
    print(text)
