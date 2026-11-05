import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from input_checks import FEATURES, assess_inputs, overall_level

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "crop_recommendation_model.joblib"
META_PATH = BASE / "model_metadata.json"
FITTED_ROWS_PATH = BASE / "outputs" / "model_training_features.csv"

LABELS = {"N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)", "temperature": "Temperature",
          "humidity": "Humidity (%)", "ph": "Soil pH", "rainfall": "Rainfall"}

st.set_page_config(
    page_title="Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Page dimensions, colors, and responsive styling
st.markdown(
    """
    <style>
        /* Whole application background */
        .stApp {
            background: linear-gradient(135deg, #F4F7F5 0%, #E9F0EB 100%);
            color: #263B2A;
        }

        /* Main page width and top/bottom/left/right spacing */
        .block-container {
            width: 100%;
            max-width: 1200px;
            padding-top: 35px !important;
            padding-bottom: 50px !important;
            padding-left: 45px !important;
            padding-right: 45px !important;
            margin-left: auto;
            margin-right: auto;
            box-sizing: border-box;
        }

        /* Page header */
        .crop-header {
            background: linear-gradient(135deg, #1E5631 0%, #4C9A2A 100%);
            padding: 30px 35px;
            border-radius: 16px;
            border-left: 6px solid #A3E12E;
            box-shadow: 0 10px 25px rgba(30, 86, 49, 0.14);
            margin-bottom: 24px;
        }
        .crop-header h1 {
            color: #FFFFFF !important;
            font-size: 28px !important;
            font-weight: 700 !important;
            line-height: 1.3;
            margin: 0 0 10px 0 !important;
        }
        .crop-header p {
            color: #E2F0D9 !important;
            font-size: 14px;
            line-height: 1.6;
            margin: 0;
        }

        /* Input form */
        div[data-testid="stForm"] {
            background-color: #FFFFFF;
            padding: 28px !important;
            border: 1px solid #D0EBD5 !important;
            border-radius: 16px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
            margin-bottom: 24px;
        }

        /* Input fields */
        .stNumberInput div[data-baseweb="input"] {
            background-color: #FAFDFB;
            border-radius: 8px;
            border-color: #D0EBD5;
        }
        .stNumberInput input {
            color: #263B2A;
        }
        label, label p {
            color: #2D4635 !important;
            font-weight: 600 !important;
        }

        /* Primary form submit button */
        .stFormSubmitButton > button {
            background-color: #228B22;
            color: #FFFFFF;
            border: 0;
            border-radius: 8px;
            min-height: 44px;
            font-weight: 600;
        }
        .stFormSubmitButton > button:hover {
            background-color: #176B17;
            color: #FFFFFF;
            border: 0;
        }

        /* Results tables and expanders */
        div[data-testid="stExpander"] {
            border: 1px solid #D0EBD5;
            border-radius: 12px;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 10px;
            overflow: hidden;
        }

        /* Comfortable spacing on mobile devices */
        @media (max-width: 768px) {
            .block-container {
                padding-top: 20px !important;
                padding-bottom: 25px !important;
                padding-left: 15px !important;
                padding-right: 15px !important;
            }
            .crop-header {
                padding: 22px 20px;
            }
            .crop-header h1 {
                font-size: 23px !important;
            }
            div[data-testid="stForm"] {
                padding: 18px !important;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="crop-header">
        <h1>🌱 Crop Recommendation Based on Soil and Climate Conditions</h1>
        <p>
            Enter soil nutrient and climate measurements to see the three crop classes
            with the highest model scores. These are model outputs, not guaranteed yields.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reference():
    """Fitted-range summary (from model_metadata.json) and the rows the model was fitted on."""
    if not META_PATH.exists():
        return None, None
    ref = json.loads(META_PATH.read_text()).get("input_reference")
    rows = pd.read_csv(FITTED_ROWS_PATH) if FITTED_ROWS_PATH.exists() else None
    return ref, rows


if not MODEL_PATH.exists():
    st.error("Saved model not found. Ensure crop_recommendation_model.joblib is next to app.py.")
    st.stop()

model = load_model()
reference, fitted_rows = load_reference()


def hint(feature):
    """Help text shown next to each input: the range the model has actually seen."""
    if not reference:
        return None
    r = reference["feature_ranges"][feature]
    return (f"The model was fitted on values from {r['min']:.4g} to {r['max']:.4g}. "
            "Values outside this range are warned about because the model has not seen them.")


if reference is not None and fitted_rows is None:
    st.warning("outputs/model_training_features.csv was not found, so the check for unusual combinations of values "
               "is switched off. Per-feature range checks still apply.")

if reference is None:
    st.warning("Fitted-range information (model_metadata.json) was not found, so unfamiliar-value "
               "checks are switched off. Only basic physical limits are applied.")

with st.form("recommendation_form"):
    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input(LABELS["N"], min_value=0.0, value=60.0, step=1.0, help=hint("N"))
        p = st.number_input(LABELS["P"], min_value=0.0, value=30.0, step=1.0, help=hint("P"))
        k = st.number_input(LABELS["K"], min_value=0.0, value=35.0, step=1.0, help=hint("K"))
        ph = st.number_input(LABELS["ph"], min_value=0.0, max_value=14.0, value=6.4, step=0.1, help=hint("ph"))
    with col2:
        temperature = st.number_input(LABELS["temperature"], value=22.0, step=0.5, help=hint("temperature"))
        humidity = st.number_input(LABELS["humidity"], min_value=0.0, max_value=100.0, value=60.0, step=1.0,
                                   help=hint("humidity"))
        rainfall = st.number_input(LABELS["rainfall"], min_value=0.0, value=700.0, step=10.0, help=hint("rainfall"))
    submitted = st.form_submit_button("🌱 Recommend top 3 crops", type="primary")

if submitted:
    values = dict(zip(FEATURES, [n, p, k, temperature, humidity, ph, rainfall]))
    if not np.all(np.isfinite(list(values.values()))):
        st.error("Please enter finite numeric values.")
    else:
        findings = assess_inputs(values, reference, fitted_rows, LABELS) if reference else []
        level = overall_level(findings)

        # Warnings come first, so they are read before the recommendation.
        if level == "unreliable":
            st.error("**Low reliability: unfamiliar input.** A valid number is not necessarily a reliable input. "
                     "The model can still output a confident-looking score for values it has never seen. "
                     "Do not rely on the ranking below.")
        elif level == "edge":
            st.warning("**Use with caution:** at least one value is at the extreme edge of what the model was fitted on.")
        elif reference:
            st.success("All values are within the range and combinations the model was fitted on.")
        for f in findings:
            (st.warning if f.level == "unreliable" else st.info)(f.message)

        sample = pd.DataFrame([list(values.values())], columns=FEATURES)
        probabilities = model.predict_proba(sample)[0]
        order = np.argsort(probabilities)[::-1][:min(3, len(model.classes_))]
        results = pd.DataFrame({
            "Rank": np.arange(1, len(order) + 1),
            "Recommended crop": model.classes_[order],
            "Model score": probabilities[order]
        })
        results["Model score"] = results["Model score"].map(lambda x: f"{x:.1%}")
        st.subheader("Top 3 recommendations" + (" (low reliability)" if level == "unreliable" else ""))
        st.dataframe(results, hide_index=True)
        st.caption(
            "Scores reflect the fitted model's class probabilities and have not been validated "
            "as real-world probabilities or crop-success guarantees. A high score does not mean the "
            "input is reliable: check the warnings above."
        )

if reference:
    with st.expander("Range of values the model was fitted on"):
        rng = pd.DataFrame(reference["feature_ranges"]).T[["min", "max"]]
        rng.index = [LABELS[i] for i in rng.index]
        st.dataframe(rng.round(2))
        st.caption(f"Based on the {reference['fitted_rows']} rows used to fit the saved model. Units are not "
                   "documented in the source file; use the same units as the training data.")

with st.expander("About this prototype"):
    st.write(
        "Inputs: N, P, K, temperature, humidity, pH, rainfall. "
        "The training file was supplied without documentation, and its statistical structure indicates the rows were "
        "synthetically generated; they are not verified Ethiopian field observations, and crop labels are not recorded "
        "field outcomes. Results are software-validation results only. "
        "Validate with documented local field data and agronomic experts before operational use."
    )
