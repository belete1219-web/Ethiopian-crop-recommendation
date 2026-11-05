"""FastAPI prediction service for the Crop Recommendation prototype (Module 6).

Run:   uvicorn api:app --host 0.0.0.0 --port 8000      (interactive docs at /docs)

It serves the same saved model and the same input-familiarity checks (input_checks.py) as the Streamlit app.
Every prediction request is appended to a JSON-lines log (inputs, reliability level, top-1 crop and score only: no
names, locations or other personal data) so that monitor_model.py can watch for drift and unreliable inputs.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from input_checks import FEATURES, assess_inputs, overall_level

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "crop_recommendation_model.joblib"
META_PATH = BASE / "model_metadata.json"
FITTED_ROWS_PATH = BASE / "outputs" / "model_training_features.csv"
PREDICTION_LOG = Path(os.environ.get("PREDICTION_LOG", BASE / "logs" / "predictions.jsonl"))
LOG_PREDICTIONS = os.environ.get("LOG_PREDICTIONS", "1") == "1"
LOG_SOURCE = os.environ.get("LOG_SOURCE", "live")      # "SIMULATED" when replaying synthetic traffic
API_VERSION = "1.0.0"

DISCLAIMER = ("Prototype trained on a 600-row file of undocumented origin that appears to be synthetic. Scores are "
              "model outputs, not calibrated probabilities, yields or fertilizer advice. Not validated on Ethiopian "
              "field data.")

logger = logging.getLogger("crop_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text())
        state["meta"] = meta
        state["reference"] = meta["input_reference"]
        state["fitted_rows"] = pd.read_csv(FITTED_ROWS_PATH) if FITTED_ROWS_PATH.exists() else None
        logger.info("Model loaded: %s (%d classes)", meta["selected_model"], len(state["model"].classes_))
    except Exception:                                    # service starts, but /health reports not ready
        logger.exception("Could not load model artifacts")
        state.clear()
    yield


app = FastAPI(title="Crop Recommendation API (prototype)", version=API_VERSION, lifespan=lifespan,
              description=DISCLAIMER)


class CropInput(BaseModel):
    """Physical limits only. Whether a value is *familiar to the model* is reported separately as a warning."""
    model_config = ConfigDict(extra="forbid")
    N: float = Field(ge=0, le=2000, allow_inf_nan=False, description="Nitrogen, same units as training data")
    P: float = Field(ge=0, le=2000, allow_inf_nan=False, description="Phosphorus, same units as training data")
    K: float = Field(ge=0, le=2000, allow_inf_nan=False, description="Potassium, same units as training data")
    temperature: float = Field(ge=-60, le=70, allow_inf_nan=False, description="Temperature, training-data units")
    humidity: float = Field(ge=0, le=100, allow_inf_nan=False, description="Relative humidity, 0-100 %")
    ph: float = Field(ge=0, le=14, allow_inf_nan=False, description="Soil pH, 0-14")
    rainfall: float = Field(ge=0, le=20000, allow_inf_nan=False, description="Rainfall, training-data units")


def _log_prediction(values: dict, level: str, top: list[dict]) -> None:
    if not LOG_PREDICTIONS:
        return
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": LOG_SOURCE,
               "inputs": values, "level": level, "top1": top[0]["crop"], "top1_score": top[0]["score"],
               "api_version": API_VERSION}
        with PREDICTION_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:                                    # logging must never break a prediction
        logger.exception("Could not write prediction log")


@app.get("/health")
def health():
    ready = "model" in state
    return JSONResponse({"status": "ok" if ready else "model_not_loaded", "api_version": API_VERSION},
                        status_code=200 if ready else 503)


@app.get("/model-info")
def model_info():
    if "model" not in state:
        return JSONResponse({"detail": "Model not loaded."}, status_code=503)
    m = state["meta"]
    return {"selected_model": m["selected_model"], "features": m["features"], "classes": m["classes"],
            "test_metrics": m["test_metrics"], "fitted_rows": state["reference"]["fitted_rows"],
            "feature_ranges": state["reference"]["feature_ranges"], "software": m.get("software"),
            "dataset_provenance": m["dataset_provenance"], "disclaimer": DISCLAIMER}


@app.post("/predict")
def predict(item: CropInput):
    if "model" not in state:
        return JSONResponse({"detail": "Model not loaded."}, status_code=503)
    values = {f: float(getattr(item, f)) for f in FEATURES}
    findings = assess_inputs(values, state["reference"], state["fitted_rows"])
    level = overall_level(findings)
    proba = state["model"].predict_proba(pd.DataFrame([values], columns=FEATURES))[0]
    order = np.argsort(proba)[::-1][:3]
    top = [{"rank": i + 1, "crop": str(state["model"].classes_[j]), "score": round(float(proba[j]), 4)}
           for i, j in enumerate(order)]
    _log_prediction(values, level, top)
    return {"top3": top, "reliability": level, "warnings": [f.message for f in findings],
            "model": state["meta"]["selected_model"], "api_version": API_VERSION, "disclaimer": DISCLAIMER}


@app.exception_handler(RequestValidationError)
async def invalid_input(request: Request, exc: RequestValidationError):
    """422 with a readable message per field. The offending value is not echoed back (it may be NaN/inf, which
    cannot be serialised to JSON, and echoing user input is unnecessary)."""
    problems = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "problem": e["msg"]}
                for e in exc.errors()]
    return JSONResponse({"detail": "Invalid input.", "errors": problems}, status_code=422)


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Never leak stack traces to the caller; give an id the support team can search for in the server log."""
    error_id = uuid.uuid4().hex[:8]
    logger.error("Unhandled error id=%s path=%s", error_id, request.url.path, exc_info=exc)
    return JSONResponse({"detail": "Internal error. Quote this id to support.", "error_id": error_id}, status_code=500)
