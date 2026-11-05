"""Generate SIMULATED API traffic to demonstrate monitoring (run: python simulate_requests.py).

All traffic is synthetic and every log line is tagged "source": "SIMULATED". It demonstrates that the monitoring
pipeline works; it is NOT evidence of how the model performs for real farmers.

  week1: 200 requests resembling the fitted data (small random noise added to resampled fitted rows)
  week2: 200 requests where a hypothetical new region shifts temperature (+4) and rainfall (-250) and a few
         users type values in the wrong units (rainfall entered in cm, i.e. divided by 10)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
LOGS = BASE / "logs"
LOGS.mkdir(exist_ok=True)
os.environ["LOG_SOURCE"] = "SIMULATED"
rng = np.random.default_rng(7)

fit = pd.read_csv(BASE / "outputs" / "model_training_features.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def make(n, shift=False):
    rows = fit[FEATURES].sample(n, replace=True, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    rows = rows + rng.normal(0, 0.15, rows.shape) * fit[FEATURES].std().to_numpy()     # small measurement noise
    if shift:
        rows["temperature"] += 4.0
        rows["rainfall"] -= 250.0
        wrong_units = rng.random(n) < 0.08
        rows.loc[wrong_units, "rainfall"] /= 10.0
    rows["humidity"] = rows["humidity"].clip(0, 100)
    rows["ph"] = rows["ph"].clip(0, 14)
    for c in ["N", "P", "K", "rainfall"]:
        rows[c] = rows[c].clip(lower=0)
    return rows


from fastapi.testclient import TestClient  # noqa: E402

for week, shift in (("week1", False), ("week2", True)):
    log = LOGS / f"simulated_{week}.jsonl"
    log.unlink(missing_ok=True)
    os.environ["PREDICTION_LOG"] = str(log)
    import importlib
    import api
    importlib.reload(api)
    with TestClient(api.app) as client:
        ok = 0
        for _, r in make(200, shift).iterrows():
            ok += client.post("/predict", json={k: float(v) for k, v in r.items()}).status_code == 200
    print(f"{week}: {ok}/200 simulated requests answered -> {log.relative_to(BASE)}")
