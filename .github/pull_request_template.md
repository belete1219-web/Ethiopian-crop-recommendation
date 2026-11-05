## What changed and why

## Release checklist (from docs/Monitoring_Support_Maintenance_Plan.docx)
- [ ] `python verify_saved_model.py` passes
- [ ] `python test_input_checks.py`, `python test_api.py` and `python test_app_validation.py` pass
- [ ] `python track_experiments.py` run; new model logged next to the current one
- [ ] README and report metrics match `outputs/test_metrics.csv` (rebuild with `python build_report.py`)
- [ ] `model_metadata.json` records dataset version, scikit-learn version, seed and metrics
- [ ] A failing test was added first for any defect fixed
- [ ] No credentials, keys or personal data added
