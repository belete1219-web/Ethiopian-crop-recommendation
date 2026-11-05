"""Tests of the FastAPI service (run: python test_api.py  or  pytest test_api.py).

Covers: a valid request, every invalid-input path (missing, wrong type, NaN, physically impossible, unknown field),
the out-of-range / unfamiliar-combination warnings, the health and model-info endpoints, and that logging
failures cannot break a prediction.
"""
import os
import tempfile
from pathlib import Path

os.environ["PREDICTION_LOG"] = str(Path(tempfile.mkdtemp()) / "predictions.jsonl")

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

TYPICAL = dict(N=60, P=30, K=35, temperature=22, humidity=60, ph=6.4, rainfall=700)
CLIENT_CTX = TestClient(api.app)


def client():
    return CLIENT_CTX


def test_health_and_model_info():
    with TestClient(api.app) as c:
        assert c.get("/health").json()["status"] == "ok"
        info = c.get("/model-info").json()
        assert info["selected_model"] == "Random Forest" and len(info["classes"]) == 10


def test_valid_request_returns_top3_sorted_and_familiar():
    with TestClient(api.app) as c:
        r = c.post("/predict", json=TYPICAL)
        assert r.status_code == 200
        body = r.json()
        scores = [t["score"] for t in body["top3"]]
        assert len(scores) == 3 and scores == sorted(scores, reverse=True)
        assert body["reliability"] == "familiar" and body["warnings"] == []
        assert "disclaimer" in body


def test_missing_field_rejected():
    with TestClient(api.app) as c:
        bad = {k: v for k, v in TYPICAL.items() if k != "ph"}
        assert c.post("/predict", json=bad).status_code == 422


def test_wrong_type_rejected():
    with TestClient(api.app) as c:
        assert c.post("/predict", json={**TYPICAL, "N": "lots"}).status_code == 422


def test_unknown_field_rejected():
    with TestClient(api.app) as c:
        assert c.post("/predict", json={**TYPICAL, "region": "Amhara"}).status_code == 422


def test_non_finite_rejected():
    with TestClient(api.app) as c:
        r = c.post("/predict", content='{"N": NaN, "P": 30, "K": 35, "temperature": 22, "humidity": 60, '
                                       '"ph": 6.4, "rainfall": 700}', headers={"content-type": "application/json"})
        assert r.status_code == 422


def test_physically_impossible_values_rejected():
    with TestClient(api.app) as c:
        for field, value in dict(ph=15, humidity=101, N=-1, rainfall=-5, temperature=-300).items():
            assert c.post("/predict", json={**TYPICAL, field: value}).status_code == 422, (field, value)


def test_out_of_range_value_is_answered_but_flagged_unreliable():
    with TestClient(api.app) as c:
        r = c.post("/predict", json={**TYPICAL, "temperature": -5, "rainfall": 20})
        assert r.status_code == 200
        body = r.json()
        assert body["reliability"] == "unreliable" and len(body["warnings"]) >= 2


def test_edge_of_range_is_caution():
    with TestClient(api.app) as c:
        r = c.post("/predict", json={**TYPICAL, "ph": 7.5})
        assert r.status_code == 200 and r.json()["reliability"] in ("edge", "unreliable")


def test_prediction_is_logged_without_personal_fields():
    with TestClient(api.app) as c:
        c.post("/predict", json=TYPICAL)
    line = Path(os.environ["PREDICTION_LOG"]).read_text().splitlines()[-1]
    assert '"inputs"' in line and '"top1"' in line and "name" not in line.replace("top1", "")


def test_logging_failure_does_not_break_prediction():
    original = api.PREDICTION_LOG
    api.PREDICTION_LOG = Path("/proc/forbidden/predictions.jsonl")     # cannot be created
    try:
        with TestClient(api.app) as c:
            assert c.post("/predict", json=TYPICAL).status_code == 200
    finally:
        api.PREDICTION_LOG = original


def test_unexpected_error_hides_details_and_gives_error_id():
    with TestClient(api.app, raise_server_exceptions=False) as c:
        original = api.state["model"]

        class Broken:
            classes_ = original.classes_
            def predict_proba(self, X):
                raise RuntimeError("secret internal detail")

        api.state["model"] = Broken()
        try:
            r = c.post("/predict", json=TYPICAL)
        finally:
            api.state["model"] = original
        assert r.status_code == 500 and "secret" not in r.text and "error_id" in r.json()


def test_health_reports_503_when_model_missing():
    with TestClient(api.app) as c:
        saved = api.state.pop("model")
        try:
            assert c.get("/health").status_code == 503
            assert c.post("/predict", json=TYPICAL).status_code == 503
        finally:
            api.state["model"] = saved


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS", name)
        except Exception as e:                          # noqa: BLE001
            failed += 1
            print("FAIL", name, repr(e))
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)
