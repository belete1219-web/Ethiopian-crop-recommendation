"""Tests of the input-familiarity logic that do not need Streamlit (run: python test_input_checks.py)."""
import json
from pathlib import Path

import pandas as pd

from input_checks import assess_inputs, overall_level

BASE = Path(__file__).resolve().parent
REF = json.loads((BASE / "model_metadata.json").read_text())["input_reference"]
FIT = pd.read_csv(BASE / "outputs" / "model_training_features.csv")
TYPICAL = dict(N=60, P=30, K=35, temperature=22, humidity=60, ph=6.4, rainfall=700)


def check(**over):
    findings = assess_inputs({**TYPICAL, **over}, REF, FIT)
    return overall_level(findings), {(f.kind, f.feature) for f in findings}


def test_typical_values_are_familiar():
    assert check() == ("familiar", set())


def test_every_feature_is_range_checked():
    # Valid numbers (finite, pH 0-14, humidity 0-100) that the model never saw.
    for feature, value in dict(N=500, P=400, K=300, temperature=-5, humidity=100, ph=2.0, rainfall=20).items():
        level, kinds = check(**{feature: value})
        assert level == "unreliable" and ("outside_range", feature) in kinds, (feature, value, kinds)


def test_edge_of_range_is_caution_only():
    level, kinds = check(ph=7.5)               # fitted max 7.78, 99th percentile 7.35
    assert level == "edge" and kinds == {("edge_of_range", "ph")}


def test_unusual_combination_of_ordinary_values():
    level, kinds = check(N=85, P=20, K=50, ph=5.6, temperature=28, humidity=80, rainfall=300)
    assert level == "unreliable" and kinds == {("unfamiliar_combination", None)}


def test_real_rows_are_rarely_flagged():
    flagged = sum(overall_level(assess_inputs(r.to_dict(), REF, FIT)) == "unreliable" for _, r in FIT.iterrows())
    assert flagged / len(FIT) < 0.05            # fitted rows themselves must almost never be called unreliable


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
