"""Automated checks of the Streamlit app's input warnings (run: python test_app_validation.py)."""
from streamlit.testing.v1 import AppTest

DEFAULTS = {"Nitrogen (N)": 60.0, "Phosphorus (P)": 30.0, "Potassium (K)": 35.0, "Soil pH": 6.4,
            "Temperature": 22.0, "Humidity (%)": 60.0, "Rainfall": 700.0}


def widget(at, label):
    return next(w for w in at.number_input if w.label == label)


def run(**overrides):
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    for label, value in {**DEFAULTS, **overrides}.items():
        widget(at, label).set_value(value)
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def texts(elements):
    return " | ".join(e.value for e in elements)


def test_typical_values_have_no_warning():
    at = run()
    assert not at.error and not at.warning, texts(at.error) + texts(at.warning)
    assert at.success and "within the range" in at.success[0].value
    assert len(at.dataframe[0].value) == 3


def test_out_of_range_values_are_flagged():
    at = run(Temperature=-5.0, Rainfall=20.0)
    assert at.error and "Low reliability" in at.error[0].value
    msgs = texts(at.warning)
    assert "Temperature = -5" in msgs and "Rainfall = 20" in msgs and "outside the range" in msgs
    assert len(at.dataframe[0].value) == 3          # still returns a ranking, but labelled low reliability


def test_each_value_ordinary_but_combination_unusual_is_flagged():
    at = run(**{"Nitrogen (N)": 85.0, "Phosphorus (P)": 20.0, "Potassium (K)": 50.0, "Soil pH": 5.6,
                "Temperature": 28.0, "Humidity (%)": 80.0, "Rainfall": 300.0})
    assert at.error and "Low reliability" in at.error[0].value
    assert "combination" in texts(at.warning)


def test_edge_of_range_value_gets_caution_not_error():
    at = run(**{"Soil pH": 7.5})                   # fitted max is 7.78, 99th percentile is 7.35
    assert not at.error
    assert at.warning and "Use with caution" in at.warning[0].value


def test_physical_limits_still_enforced_by_widgets():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    ph = widget(at, "Soil pH")
    assert (ph.min, ph.max) == (0.0, 14.0)
    hum = widget(at, "Humidity (%)")
    assert (hum.min, hum.max) == (0.0, 100.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
