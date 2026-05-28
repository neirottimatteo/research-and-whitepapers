# Walk-Forward Validation — Command Reference
many commands are old:
new importatn ones are:
python.exe walk_forward.py --dataset E --ensemble --weights 1 0 --feature-set plus --confirmed --threshold 50  
python.exe data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --min-fold 4 --feature-set plus --confirmed --uniform-color

All commands assume you're in the `recession/` directory.
Use `.venv\Scripts\python.exe` on Windows (or `.venv/bin/python` on Linux/macOS).

---

## `walk_forward.py` — Walk-Forward Validation & Threshold Calibration

Two stages run in sequence for each (algorithm, dataset) pair:
1. **Threshold calibration** — train on all data, find the minimum threshold ≥ 30% where all non-recession probability peaks are suppressed to zero.
2. **Walk-forward validation** — for each recession episode k, train on data through recession k + 12 months, then test on all subsequent history.

```powershell
# Dataset E — individual NaN-native models
.venv\Scripts\python.exe walk_forward.py --dataset E --model xgboost
.venv\Scripts\python.exe walk_forward.py --dataset E --model hist_gradient_boosting
.venv\Scripts\python.exe walk_forward.py --dataset E --model all

# Dataset E — ensemble (XGBoost + HistGB blended)
.venv\Scripts\python.exe walk_forward.py --dataset E --ensemble
.venv\Scripts\python.exe walk_forward.py --dataset E --ensemble --weights 0.6 0.4
.venv\Scripts\python.exe walk_forward.py --dataset E --ensemble --weights 0.4 0.6

# Dataset E — individual + ensemble in one run
.venv\Scripts\python.exe walk_forward.py --dataset E --model all --ensemble

# With confirmed filter (suppresses FP spikes shorter than 4 consecutive months)
.venv\Scripts\python.exe walk_forward.py --dataset E --model xgboost --confirmed
.venv\Scripts\python.exe walk_forward.py --dataset E --model all --confirmed
.venv\Scripts\python.exe walk_forward.py --dataset E --ensemble --confirmed
.venv\Scripts\python.exe walk_forward.py --dataset E --ensemble --weights 0.6 0.4 --confirmed
.venv\Scripts\python.exe walk_forward.py --dataset E --model all --ensemble --confirmed

# Other datasets (A/B/C/D support all 10 algorithms)
.venv\Scripts\python.exe walk_forward.py --dataset A --model all
.venv\Scripts\python.exe walk_forward.py --dataset B --model xgboost
.venv\Scripts\python.exe walk_forward.py --dataset C --model all

# Run all algorithms × all datasets A/B/C/D (does not include E)
.venv\Scripts\python.exe walk_forward.py
```

> `--dataset all` covers A/B/C/D only. Dataset E must be requested explicitly.
> Only `xgboost` and `hist_gradient_boosting` are valid for Dataset E — they are the only NaN-native models.
> `--ensemble` always uses Dataset E regardless of `--dataset`.

### Flags summary

| Flag | Values | Default | Notes |
|---|---|---|---|
| `--dataset` | `A` `B` `C` `D` `E` `all` | `all` | `all` = A/B/C/D only; E must be explicit |
| `--model` | `xgboost` `logistic` `random_forest` `gradient_boosting` `hist_gradient_boosting` `ada_boost` `extra_trees` `svc` `mlp` `naive_bayes` `all` | `all` | |
| `--confirmed` | flag | off | Apply 4-consecutive-month minimum duration filter; suppresses short FP spikes before reporting lead times and FP counts. Does not affect AUC. |
| `--ensemble` | flag | off | Blend XGBoost + HistGB on Dataset E |
| `--weights` | `W_XGB W_HGB` | `0.5 0.5` | Two floats summing to 1.0; only used with `--ensemble` |

### Output — what to read

Each model prints a per-fold table:

```
Fold  TrainThru   N     1953  1957  ...  2008   2020⚠    FP/yr    AUC
   1  1954-05     1      --   18m   ...   17m  [14m⚠]      1.6  0.548
   ...
  10  2009-06    10      --    --   ...    --  [15m⚠]      0.8  0.681
```

- **Lead time columns** (e.g. `18m`): months before recession onset that the model first crossed the threshold
- **`MISS`**: model did not alarm before onset in that fold
- **`2020⚠`**: COVID recession, shown for reference but never counted in FP/yr
- **`FP/yr`**: false positive alarms per year in the out-of-sample test period (~1.3 out-of-sample is normal)
- **`AUC`**: ROC-AUC on the test period

Calibrated thresholds are saved to `output/raw/best_thresholds.json`.

---

## History chart (requires walk_forward.py to be run first)

The history chart loads the full-history model saved by `walk_forward.py` — run `walk_forward.py --dataset E --model all` (or `--ensemble`) first.

```powershell
# Raw probabilities
.venv\Scripts\python.exe data_visualization.py --plot history --dataset E

# Confirmed-filtered (4-month minimum duration, separate output file)
.venv\Scripts\python.exe data_visualization.py --plot history --dataset E --confirmed

# Save only, no browser popup
.venv\Scripts\python.exe data_visualization.py --plot history --dataset E --no-show
.venv\Scripts\python.exe data_visualization.py --plot history --dataset E --confirmed --no-show
```

Outputs:
- `output/html/viz_history_E.html` — raw probabilities (interactive)
- `output/html/viz_history_E_confirmed.html` — confirmed-filtered probabilities

---

## SHAP (requires walk_forward.py to be run first)

SHAP decomposes each month's score into per-feature contributions. Loads the full-history model saved by `walk_forward.py` — run `walk_forward.py --dataset E --model xgboost` (or `--model all`) first.

```powershell
# Summary + timeline in one command (recommended starting point)
.venv\Scripts\python.exe data_visualization.py --plot shap --dataset E --model-type xgboost
.venv\Scripts\python.exe data_visualization.py --plot shap --dataset E --model-type hist_gradient_boosting

# Save only, no browser popup
.venv\Scripts\python.exe data_visualization.py --plot shap --dataset E --model-type xgboost --no-show
```

### Summary (beeswarm) — global feature importance + direction
```powershell
.venv\Scripts\python.exe data_visualization.py --plot shap-summary --dataset E --model-type xgboost
.venv\Scripts\python.exe data_visualization.py --plot shap-summary --dataset E --model-type hist_gradient_boosting
```
Output: `output/images/viz_shap_summary_e_xgboost.png`

### Waterfall — why did the model score a specific month?
```powershell
# A specific alarm month (e.g. Sept 2007, 15 months before recession)
.venv\Scripts\python.exe data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --date 2007-09

# Current reading
.venv\Scripts\python.exe data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --date 2026-04
```
Output: `output/images/viz_shap_waterfall_e_xgboost_{date}.png`

### Timeline — how each signal built up over time (interactive)
```powershell
.venv\Scripts\python.exe data_visualization.py --plot shap-timeline --dataset E --model-type xgboost
.venv\Scripts\python.exe data_visualization.py --plot shap-timeline --dataset E --model-type hist_gradient_boosting

# Show top 15 features (default: top 10)
.venv\Scripts\python.exe data_visualization.py --plot shap-timeline --dataset E --model-type xgboost --top-n 15
```
Output: `output/html/viz_shap_timeline_e_xgboost.html` (interactive) + PNG snapshot

---

## Training cutoff — T-24 months

All training is frozen at today minus 24 months. Recent months have uncertain NBER labels (NBER revises retroactively) and the 12-month forward target is structurally unlabeled for the most recent year. The cutoff advances automatically:

```python
_TRAIN_CUTOFF = pd.Timestamp.today() - pd.DateOffset(months=24)
```

Walk-forward fold `train_end` dates are historical (last: 2010-06), so they are unaffected in practice.

---

## Algorithm keys

| Key | Algorithm | NaN-native |
|---|---|---|
| `xgboost` | XGBoost GBDT (GPU-accelerated) | Yes |
| `hist_gradient_boosting` | Sklearn Hist Gradient Boosting | Yes |
| `gradient_boosting` | Sklearn Gradient Boosting | No |
| `logistic` | Logistic Regression | No |
| `random_forest` | Random Forest | No |
| `ada_boost` | AdaBoost | No |
| `extra_trees` | Extra Trees | No |
| `svc` | Support Vector Classifier (RBF kernel) | No |
| `mlp` | Multi-Layer Perceptron | No |
| `naive_bayes` | Gaussian Naive Bayes | No |

Only NaN-native algorithms can run on Dataset E.

---

## Dataset keys

| Key | Window | Features | What it covers |
|---|---|---|---|
| `A` | 1960–present | 9 | Labor (SAHM, UNRATE) + industrial (INDPRO) + housing (HOUST) |
| `B` | 1977–present | 19 | A + yield curve (T10Y2Y), jobless claims (ICSA), permits (PERMIT), sentiment (UMCSENT) |
| `C` | 1950–present | 9 | Deep history: INDPRO, UNRATE, PAYEMS, CPI — no SAHM/yield curve |
| `D` | 1991–present | 12 | Financial stress: VIX, credit spread (BAA10Y), PCE, CCSA, manufacturing hours |
| `E` | 1950–present | 53 | **Combined** — union of A+B+C+D + deep-history expansion. NaN where series not yet existed. Best overall. |

**Dataset E is the recommended choice.**

---

## Output locations

```
output/
├── images/   — all PNG charts
├── html/     — all interactive Plotly HTML files (open in any browser)
└── raw/      — CSV exports, best_thresholds.json, saved models (.joblib)
```
