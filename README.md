# Crop Recommendation Based on Soil and Climate Conditions

## Project overview
A multiclass machine-learning prototype that ranks three crop labels from seven input features:
`N`, `P`, `K`, `temperature`, `humidity`, `ph`, and `rainfall`.

## Dataset

**Short version:** the 600-row file is treated as **synthetic** data of **undocumented origin**. It is **not** verified
Ethiopian field data, and the results must not be used to make real farm recommendations.

The supplied dataset has 600 rows, 7 numeric input features, and a `label` target with 10 crop classes (60 rows per
class). The included CSV is `data/Crop_recommendation_Ethio.csv`.

**Where it came from.** The file was supplied to the group as a CSV with no accompanying documentation: no author,
collection site, dates, measurement units, or labelling protocol. Who produced it, and with what code or method, could
not be verified. The file name and crop names suggest an Ethiopian focus, but nothing shows that any row is an
Ethiopian field observation.

**Were the records synthetically generated?** Almost certainly yes, but this is an inference from the data, not a
documented fact. The notebook (Task 3) computes the evidence and saves it to `outputs/data_provenance_checks.csv`:
- exactly 60 rows for every crop;
- every one of the 70 crop-by-feature distributions is consistent with a normal distribution (field data, especially
  rainfall, is usually skewed);
- features are uncorrelated within each crop (mean absolute correlation 0.100, against about 0.104 expected from
  independent random draws);
- rainfall has a near-constant spread across crops (SD 131 to 153) even though crop means range from 441 to 1,415
  (coefficient of variation 10% to 31%), whereas real variability usually grows with the mean;
- humidity is capped at exactly 100 and rainfall floored at exactly 100 in one row each, and all values carry about 15
  decimal places, which suggests generated values clipped at a bound rather than rounded instrument readings.

**How were the crop labels assigned?** The structure indicates each row was generated *for* a crop, around a
crop-specific profile of nutrient, climate and pH values. The label is therefore the category the row was generated for.
It is **not** a recorded outcome such as the crop a farmer grew, or the crop that gave the best yield on that plot.
The profiles themselves are undocumented.

**What "no synthetic data was added" means.** Our preparation step only validates the file (numeric conversion, dropping
missing or non-finite rows, trimming and lower-casing labels). We did not generate, augment, oversample or edit rows. This
is a statement about our pipeline only; it says nothing about how the supplied rows were produced.

Because the model learns to separate ten generated, overlapping profiles, the reported scores are
**software-validation results, not agricultural validation**. Units (for example whether temperature is in °C and
rainfall in mm) are also undocumented.

## Files
- `README.md` — basic instructions for installing, running, and using the project.
- `Group_Contributions.md` — List of Group Two members, role areas and specific contributions.
- `Ethiopian_Crop_Recommendation_Implementation.ipynb` — notebook organized into the eight project tasks; runs locally or in Colab, and is saved with its outputs.
- `data/Crop_recommendation_Ethio.csv` — input dataset.
- `crop_recommendation_model.joblib` — fitted model (Random Forest), selected by validation Macro-F1 and fitted on train + validation rows.
- `model_metadata.json` — feature list, labels, model name, provenance note, test metrics, and the fitted-range summary used by the app's input checks.
- `input_checks.py` — warns when an input is outside, or unlike, what the model was fitted on (used by the app and notebook).
- `outputs/` — every table and chart the report cites: validation comparison, test metrics, classification report, per-class metrics, confusion matrix, top-k accuracy, provenance checks, EDA charts, feature importance, and `model_training_features.csv` (the rows the model was fitted on).
- `requirements.txt` — Python package dependencies.
- `Project Implementation Report.docx` — report of method, measured results, limitations, and conclusion. It embeds the EDA charts, model-comparison chart, confusion matrix, class-level results, a test-accuracy vs top-3 explanation, and the app's input-validation screenshots.
- `build_report.py` — rebuilds the report from `outputs/`, `evidence/` and `model_metadata.json` (`python build_report.py`), so figures and numbers always match the saved model.
- `app.py` — Streamlit interface for entering values and viewing top-three results with reliability warnings.
- `test_model.py` — minimal inference example.
- `verify_saved_model.py` — checks that the saved model reproduces the reported metrics on the held-out rows.
- `test_app_validation.py` — automated tests of the app's input warnings (needs Streamlit).
- `test_input_checks.py` — Streamlit-free tests of the range, edge and combination checks.
- `evidence/` — screenshots of the running app and the log of the verification run.

## Run locally
Use Python 3.11+ (tested on 3.12) in a virtual environment:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Then, from the project folder:
```bash
pip install notebook                  # only if Jupyter is not already installed
jupyter notebook Ethiopian_Crop_Recommendation_Implementation.ipynb   # run all cells; regenerates outputs/, the model and metadata
python verify_saved_model.py          # confirms the saved model reproduces the reported test metrics
python test_input_checks.py           # tests the input-check logic (no Streamlit needed)
python test_app_validation.py         # tests the app's input warnings
python build_report.py                # rebuilds Project Implementation Report.docx
streamlit run app.py                  # starts the web interface (http://localhost:8501)
```
The notebook reads `data/Crop_recommendation_Ethio.csv` automatically. The app expects
`crop_recommendation_model.joblib`, `model_metadata.json`, `input_checks.py` and `outputs/model_training_features.csv`
in the project folder.

## Run in Google Colab
1. Upload `Ethiopian_Crop_Recommendation_Implementation.ipynb` to Google Colab.
2. Run cells from top to bottom.
3. When prompted, upload `Crop_recommendation_Ethio.csv`.
4. Review the data checks, EDA, validation comparison, test metrics, confusion matrix, and top-k results.
5. The last task tests the recommendation function and downloads a ZIP of the model, metadata and evaluation outputs.

## Input validation in the app
A valid number is not necessarily a reliable input for this model. Besides the physical limits (pH 0 to 14, humidity
0 to 100%, no negative nutrients or rainfall), the app compares every entry with the rows the model was fitted on:
- **Outside the fitted range** (below the minimum or above the maximum seen in training): a red "Low reliability"
  banner and a per-feature warning that names the range.
- **Edge of the fitted range** (more extreme than 99% of fitted rows): a caution message.
- **Unfamiliar combination**: even if each value is individually in range, the combination is flagged when it is farther
  from every fitted row than the most unusual 1% of fitted rows are from their nearest neighbour.

The reason: the model gives a confident-looking score even for impossible inputs (temperature −5 with rainfall 20 still
returns barley at 55%). On the 90 held-out test rows the check flags 4 as outside range and none as an unfamiliar
combination, but it flags 79% of random combinations whose values all lie inside their individual ranges
(`outputs/input_check_validation.csv`). Units are undocumented, so enter values in the same units as the training data;
the app shows the fitted range beside every input.

## Run the model without the app
```python
import joblib
import pandas as pd

model = joblib.load("crop_recommendation_model.joblib")

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

sample = pd.DataFrame(
    [[60, 30, 35, 22, 60, 6.4, 700]],
    columns=features
)

probabilities = model.predict_proba(sample)[0]
order = probabilities.argsort()[::-1][:3]

for rank, i in enumerate(order, start=1):
    print(
        f"{rank}. {model.classes_[i]} "
        f"— model score: {probabilities[i]:.2%}"
    )
```

## Results at a glance (held-out test set, 90 rows)
- **Test accuracy 0.833**: the single highest-scored crop is the true label for 75 of 90 rows.
- **Top-3 accuracy 0.967**: the true label is among the three highest-scored crops for 87 of 90 rows. This is an easier
  test (random guessing gives 10% for top-1 and 30% for top-3) and it does not say which of the three is right.
- Macro-F1 0.831. Five-fold cross-validated Macro-F1 0.802 ± 0.032 (`outputs/five_fold_cv.csv`).

## Interpretation and responsible use
- The model recommends labels learned from this dataset; it does not calculate yield, profitability, fertilizer prescriptions, or guaranteed crop suitability.
- Inputs should be measured in units consistent with the training dataset.
- Field validation with trusted Ethiopian soil, climate, and crop outcome data is needed before operational agricultural use.
- Do not enter values the app warns about, and do not treat high scores as probabilities calibrated for real-world outcomes; calibration has not been established.

## Module 6: deployment, monitoring and maintenance

**Group members and responsibilities:** see `Group_Contributions.md` (Module 5 and Module 6 tables).
**Data statement:** the training data are treated as synthetic and of undocumented origin; see `docs/Simulated_Data_Statement.docx`. The
monitoring demonstration additionally uses **simulated API traffic** (`simulate_requests.py`), tagged `SIMULATED` in every log line.
**External links:** Not publicly deployed. No GitHub repository or hosted Streamlit URL exists for this submission; reviewers should use the local run instructions in this README and the screenshots and logs in `evidence/`.

| File | Purpose |
|---|---|
| `api.py` | FastAPI service: `GET /health`, `GET /model-info`, `POST /predict` (top-3, reliability level, warnings). Pydantic validates every input (missing, wrong type, NaN/inf, physically impossible, unknown field -> HTTP 422), unexpected errors return an `error_id` instead of a stack trace, and each prediction is appended to `logs/predictions.jsonl` (inputs and top-1 only, no personal data). |
| `test_api.py` | 13 automated API tests (valid, invalid, warnings, logging failure, hidden errors, health when the model is missing). |
| `track_experiments.py` | MLflow tracking: logs the four candidate models (parameters, validation metrics) and a final run for the saved model on the held-out test rows. Results also exported to `outputs/mlflow_runs.csv`. |
| `monitor_model.py` | Monitoring report from the prediction log: unreliable-input share, low-confidence share, PSI drift per feature, crop mix, with WATCH/ALERT flags. |
| `simulate_requests.py` | Generates simulated traffic (a baseline week and a drifted week) to demonstrate the monitor. |
| `Dockerfile`, `requirements-deploy.txt`, `.dockerignore` | Container recipe for the API (written but **not built or run** by the authors of this package; see limitations). |
| `check_docker_context.py` | Static check of the Docker build context that works without Docker: COPY sources exist and are not excluded, pins match `requirements.txt`, runtime packages are listed. It does **not** build the image. |
| `.github/` | `workflows/ci.yml` (tests, then Docker build and API smoke test), `dependabot.yml`, issue form with S1 to S3 severity, pull-request release checklist. Written for this repository; see "GitHub configuration" below. |
| `.streamlit/config.toml`, `.env.example` | Headless Streamlit settings with usage statistics off; documented environment variable (`PREDICTION_LOG`). |
| `docs/` | Monitoring/support/maintenance plan, one-page retrospective, simulated-data statement, and the capstone video script (`Capstone_Video_Script.docx`). The recorded MP4 is submitted separately (10 minutes or less, 250 MB or less). |
| `evidence/module6_verification_log.txt`, `evidence/api_live_demo.txt` | Test results and a live API session (HTTP status and JSON for valid and invalid requests). |

### Run the API
```bash
pip install -r requirements.txt
uvicorn api:app --port 8000          # interactive documentation at http://localhost:8000/docs
curl -X POST localhost:8000/predict -H "content-type: application/json" \
  -d '{"N":60,"P":30,"K":35,"temperature":22,"humidity":60,"ph":6.4,"rainfall":700}'
python test_api.py                   # API tests
```
### Track experiments and monitor
```bash
python track_experiments.py                          # then: mlflow ui --backend-store-uri sqlite:///mlflow.db
python simulate_requests.py                          # SIMULATED traffic -> logs/simulated_week1.jsonl, simulated_week2.jsonl
python monitor_model.py logs/simulated_week1.jsonl logs/simulated_week2.jsonl   # -> outputs/monitoring_report.md
```
### Deploy
- **Streamlit Community Cloud:** push this folder to GitHub, create an app pointing at `app.py`. `requirements.txt` pins the scikit-learn/pandas/numpy versions the saved model needs.
- **Docker (API):** `docker build -t crop-api:1.0 .` then `docker run --rm -p 8000:8000 crop-api:1.0`. The authors have not built this image; `python check_docker_context.py` checks the context statically, and the `docker` job in `.github/workflows/ci.yml` builds it and smoke-tests `/health`, `/predict` and an invalid request (HTTP 422) on every push.

### What is implemented, simulated, and future work
- **Implemented and tested:** model, Streamlit app with input-familiarity warnings, FastAPI service, MLflow logging, monitoring script, automated tests.
- **Simulated:** the training data (inferred) and the monitoring traffic (explicitly generated).
- **Written but not yet run:** the GitHub Actions workflow (`.github/workflows/ci.yml`) and the Docker image. Neither has been executed by the authors of this package; the first push to GitHub will run them.
- **Future:** real documented Ethiopian data (EthioSIS, NASA POWER), outcome feedback for accuracy monitoring, calibration, vulnerability-scanned Docker image, authentication and rate limiting for a public API.
- **Known issue found while preparing Module 6:** the Module 5 README quoted test accuracy 0.822 / Macro-F1 0.821, but the shipped model and `outputs/test_metrics.csv` give 0.833 / 0.831 (verified by `verify_saved_model.py`). The README and report now state 0.833 / 0.831. A second stale figure (barley at 63% for the impossible input temperature -5, rainfall 20) was corrected to 55%, the value the shipped model returns (see `evidence/app_2_out_of_range.png` and `outputs/input_check_examples.csv`). `evidence/verification_log.txt` is the earlier Module 5 run and is kept only as a historical record.


## GitHub configuration
Nothing here is published yet; these steps create the repository the maintenance plan refers to.
```bash
git init -b main
git add .
git commit -m "Crop recommendation prototype: Module 6 submission"
# create an empty repository on github.com (no README or licence), then:
git remote add origin https://github.com/<owner>/<repository>.git
git push -u origin main
#cd C:\Users\belet\desktop\edit\ECRBSCConditionsproject
 #  git config --global user.name "Your Name"
   #git config --global user.email "your-github-email@example.com"
``
On GitHub:
1. **Actions:** the `CI` workflow runs on every push and pull request to `main`. It installs the pinned `requirements.txt`, runs `verify_saved_model.py`, `test_input_checks.py`, `test_api.py`, `test_app_validation.py`, `track_experiments.py` and `monitor_model.py`, then builds the Docker image, starts it and checks `/health`, `/predict` and an invalid request. Add the repository's CI badge to this README only after the first run succeeds.
2. **Labels:** create `bug`, plus severity labels S1, S2, S3 to match the issue form (S1 service down or wrong output shown as reliable; S2 wrong or confusing warnings; S3 cosmetic or documentation).
3. **Branch protection (Settings, Branches):** require the `tests` and `docker` checks and one review before merging into `main`.
4. **Releases:** tag each release (for example `v1.0.0`) after the pull-request checklist is complete; keep the previous model file so a release can be rolled back.
5. **Dependabot:** `.github/dependabot.yml` opens monthly update pull requests. Treat any change to scikit-learn, numpy or pandas as a model-compatibility change: it must pass `verify_saved_model.py` and may require retraining.
6. **Hosted app (optional):** on Streamlit Community Cloud choose the repository, branch `main`, main file `app.py`, Python 3.12. It installs `requirements.txt`, which includes MLflow and FastAPI, so the first build is slow but works. Only after the app is live, replace "Not publicly deployed" above with its URL.
7. **Secrets:** none are needed. Do not commit `.env`, keys or real prediction logs (`.gitignore` excludes them).

## Submission contents
- Project folder (this ZIP): code, model, data, notebook, report, `outputs/`, `evidence/`, `docs/`, `logs/`.
- Capstone video: submitted separately as an MP4 (10 minutes or less, 250 MB or less); the script is `docs/Capstone_Video_Script.docx`.
- No credentials, API keys or confidential data are included; the training data are treated as synthetic (see `docs/Simulated_Data_Statement.docx`).
