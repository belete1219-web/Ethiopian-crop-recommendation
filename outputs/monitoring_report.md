# Monitoring report

Compared with the 510 rows the model was fitted on. PSI thresholds: watch >= 0.10, alert >= 0.20.

## simulated_week1.jsonl  (source tag: SIMULATED; 200 requests)

| Check | Value | Status |
|---|---|---|
| Share flagged unreliable | 0.5% | ok |
| Share flagged edge-of-range | 15.5% | - |
| Median top-1 score | 0.74 | - |
| Share with top-1 score < 0.5 | 13.5% | ok |
| PSI: N | 0.034 | ok |
| PSI: P | 0.062 | ok |
| PSI: K | 0.067 | ok |
| PSI: temperature | 0.046 | ok |
| PSI: humidity | 0.085 | ok |
| PSI: ph | 0.039 | ok |
| PSI: rainfall | 0.072 | ok |

Top-1 crop mix (top 5): wheat 12%, sesame 12%, haricot_bean 12%, chickpea 12%, enset 10%

## simulated_week2.jsonl  (source tag: SIMULATED; 200 requests)

| Check | Value | Status |
|---|---|---|
| Share flagged unreliable | 29.0% | ALERT |
| Share flagged edge-of-range | 17.0% | - |
| Median top-1 score | 0.59 | - |
| Share with top-1 score < 0.5 | 36.5% | WATCH |
| PSI: N | 0.049 | ok |
| PSI: P | 0.038 | ok |
| PSI: K | 0.066 | ok |
| PSI: temperature | 0.913 | ALERT |
| PSI: humidity | 0.008 | ok |
| PSI: ph | 0.049 | ok |
| PSI: rainfall | 1.297 | ALERT |

Top-1 crop mix (top 5): sesame 18%, sorghum 16%, maize 15%, barley 10%, enset 10%
