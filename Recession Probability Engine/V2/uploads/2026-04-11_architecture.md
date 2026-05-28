# Recession Probability Engine — Architecture & Implementation

**Last updated:** 2026-04-11 - this is all old

---

## The architecture in one sentence

There are **three datasets**, each feeding **one sub-model**. The three sub-models are blended into a single ensemble probability. `main.py` is the production pipeline. Three separate CLI scripts (`model_shootout.py`, `tuner.py`, `data_visualization.py`) are research/exploration tools that are completely independent from it.

---

## The three datasets

| | Dataset A | Dataset B | Dataset C |
|---|---|---|---|
| **Starts** | 1960 | 1977 | 1950 |
| **Features** | 9 | 14 (9 + 5) | 9 (different 9) |
| **What it adds** | Core macro: INDPRO, UNRATE, SAHM, HOUST | Yield curve + jobless claims | Deep history: PAYEMS, CPI |
| **Why it exists** | Long view — recessions back to 1960 | Modern view — yield curve is the best single predictor since 1977 | Pre-1960 recessions (1950s) that A misses |

Dataset B's 14 features are literally `FEATURE_COLS_A + 5 more` — it is a strict superset of A. Dataset C shares the INDPRO and UNRATE features but replaces SAHM/HOUST/yield curve/claims with PAYEMS (payrolls) and CPI, which are available back to the late 1940s.

The **target label** is the same in all three: `target_12m = 1` if a recession occurs in any of the next 12 months, 0 otherwise.

---

## `ai_model.py` — the core logic

### `ModelType` enum + `_DEFAULT_PARAMS`

Ten algorithm types are defined. Each has a dict of default hyperparameters in `_DEFAULT_PARAMS`. The `MODEL_TYPE` constant in `config.py` (currently `"xgboost"`) controls which one the production pipeline uses.

| Key | Algorithm | Notes |
|---|---|---|
| `xgboost` | XGBoost GBDT | GPU-accelerated via `device="cuda"` |
| `logistic` | Logistic Regression | Requires `StandardScaler` |
| `random_forest` | Random Forest | `class_weight="balanced"` |
| `gradient_boosting` | Sklearn GBT | `sample_weight` at fit time |
| `hist_gradient_boosting` | Sklearn HistGBT | `sample_weight`; permutation importance |
| `ada_boost` | AdaBoost | `algorithm="SAMME"` (sklearn ≥ 1.4) |
| `extra_trees` | Extremely Randomised Trees | `class_weight="balanced"` |
| `svc` | SVC (RBF kernel) | `probability=True`; `StandardScaler` |
| `mlp` | Multi-Layer Perceptron | `StandardScaler`; no `sample_weight` |
| `naive_bayes` | Gaussian Naive Bayes | `sample_weight` at fit time |

### `RecessionSubModel` — the universal wrapper

This is the fundamental unit. It wraps any of the ten algorithms behind a consistent `train()` / `predict_proba()` / `feature_importances()` interface.

**Parameter loading priority** (most to least specific):
1. Explicit `params=` argument passed to constructor
2. `best_params.json` keyed by `"<model_type>_<name>"` — only if `USE_TUNED_PARAMS=True` in config
3. `_DEFAULT_PARAMS[model_type]` — always the fallback

**How `train()` handles class imbalance** — recessions are ~15–25% of months depending on dataset:

| Algorithm | Imbalance strategy |
|---|---|
| XGBoost | `scale_pos_weight = count(0) / count(1)` |
| Logistic Regression | `class_weight="balanced"` + `StandardScaler` |
| Random Forest | `class_weight="balanced"` |
| Gradient Boosting | `sample_weight` vector passed to `.fit()` |
| Hist Gradient Boosting | `sample_weight` vector passed to `.fit()` |
| AdaBoost | `sample_weight` vector passed to `.fit()` |
| Extra Trees | `class_weight="balanced"` |
| SVC | `class_weight="balanced"` + `StandardScaler` |
| MLP | No weighting (MLPClassifier ignores `sample_weight`); early stopping helps |
| Naive Bayes | `sample_weight` vector passed to `.fit()` |

The `StandardScaler` for Logistic/SVC/MLP is stored on the instance (`self._scaler`) and reused in `predict_proba()` to apply the same transformation at inference time.

**`feature_importances()` dispatch:**

| Algorithm group | Method |
|---|---|
| XGBoost, Random Forest, Gradient Boosting, AdaBoost, Extra Trees | `.feature_importances_` (Gini / mean decrease impurity) |
| Logistic Regression | `abs(coef_[0])` — magnitude of standardised coefficients |
| Hist Gradient Boosting, SVC, MLP | Permutation importance on stored training data (10 repeats, `scoring="roc_auc"`) |
| Naive Bayes | `abs(theta_[1] - theta_[0])` — difference in per-class feature means |

All importances are normalised to sum 1 before returning. `self._X_train` and `self._y_train` are stored during `train()` for HistGB/SVC/MLP permutation importance.

### `RecessionEnsemble` — the blending layer

Takes model A, model B, and optionally model C. The weights must sum to 1:

```
weight_a = 1.0 - weight_b - weight_c
```

With config defaults `WEIGHT_B=0.45`, `WEIGHT_C=0.15`:

- `weight_a = 0.40` — Long View
- `weight_b = 0.45` — Modern View (highest because yield curve is most predictive)
- `weight_c = 0.15` — Deep History (adds pre-1960 signal)

The `predict()` method enforces that all three input DataFrames share the same `DatetimeIndex` before blending. It raises a `ValueError` with the actual date ranges if they don't match.

### `HoldOutValidator` — chronological integrity

This is how the model is validated honestly. The key principle: **the model never sees the test period during training**.

| Test | Train window | Test window | Goal |
|---|---|---|---|
| 2008 stress test | 1960–2006 | 2007–2010 | Spike before Lehman collapse |

The 2020 COVID recession is deliberately excluded from validation. It was triggered by an exogenous one-time event (pandemic lockdowns), not by endogenous macro deterioration that the indicators can detect in advance. A model that fails to predict a pandemic is not a bad model.

**The 3-dataset index intersection**: the shared test window is `A.index ∩ B.index ∩ C.index`. Since C starts 1950 and the 2008 test window starts after 1977, C never reduces the window — but the intersection is always enforced correctly.

Three fresh `RecessionSubModel` instances are created and trained inside each `_run_test()` call. They share nothing with the production model.

### `ProductionTrainer` — the real forecast

Trains on 100% of the data — no holdout. This is what produces the current recession probability. After calling `.train()`:

- `.predict_history()` — monthly probabilities back to 1977 (where A and B first overlap)
- `.predict_current()` — single dict for the most recent month: `prob_a`, `prob_b`, `prob_c`, `prob_ensemble`, `sahm_current`, `spread_current`

---

## `main.py` — the production pipeline

Steps run sequentially, each printing visible output:

| Step | What it does | Output |
|---|---|---|
| 0 | GPU check — XGBoost on CPU vs CUDA | Console |
| 1 | Load raw FRED series, print date ranges | Console (optional, commented out by default) |
| 2 | Build all 3 datasets, print shapes + recession rates | Console |
| 3 | Train Sub-Model A on full data, print feature importances | Console |
| 4 | 2008 hold-out validation | `validation_2008.png`, `holdout_2008.csv` |
| 6 | Production model — 100% data, final probability | `recession_probabilities.png` |
| 7 | Threshold calibration — minimum pre-recession probability across all historical recessions | `threshold_analysis.png` |
| 8 | All-10-algorithm comparison on 2008 holdout (Dataset A) | `algorithm_comparison_2008.png` |

**Step 8 — algorithm comparison**: loops over all 10 `ModelType` values, runs each through `SingleModelValidator` on Dataset A with the 2008 holdout boundary, prints a ranked table (ROC-AUC, Brier, peak prob, peak date), and calls `charts.plot_algorithm_comparison()` to produce a side-by-side ROC curves + AUC bar chart.

---

## `model_shootout.py` — the research tool

**Completely separate from the production ensemble.** Each algorithm is tested alone — results are never blended.

`SingleModelValidator` is the key class. For a given dataset + algorithm it:
1. Cuts the data at the hold-out boundary
2. Trains on the training slice
3. Predicts on the test slice
4. Returns ROC-AUC, Brier score, peak probability, and the full probability series

Example output:

```
================================================================
  MODEL SHOOTOUT -- Dataset A | 2008 holdout
================================================================

  Algorithm              ROC-AUC    Brier   Peak prob   Peak date
  --------------------------------------------------------------
  XGBoost                  0.962    0.137      95.3%    Dec 2007
  Random Forest            0.891    0.155      95.1%    Nov 2007
  ...

  Best ROC-AUC : XGBoost (0.962)
  Best Brier   : XGBoost (0.137)
```

`use_gpu` is automatically forced to `False` for all non-XGBoost models because sklearn models are CPU-only.

**Usage:**

```powershell
python model_shootout.py --dataset A --test 2008
python model_shootout.py --dataset A --test 2008 --model xgboost
python model_shootout.py --all
```

`MODEL_LABELS` (a plain dict in `model_shootout.py`) must contain an entry for every `ModelType` value — it is used in the comparison table and by `main.py` Step 8.

---

## `tuner.py` — hyperparameter search

Uses `GridSearchCV` with `TimeSeriesSplit(n_splits=5)`.

`TimeSeriesSplit` is mandatory — standard k-fold would randomly mix time periods together, letting the model train on 2010 data to predict 2005 (data leakage). `TimeSeriesSplit` always trains on the past and tests on the future within each fold.

**The logistic regression pipeline**: because `StandardScaler` must be fit on the training fold only (fitting it on the full dataset before CV would leak the test distribution into the scaler), `tuner.py` wraps it in a `sklearn.pipeline.Pipeline`:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(class_weight="balanced")),
])
```

GridSearchCV then handles the scaler correctly inside each fold. Param keys are prefixed `clf__C`, `clf__penalty`, etc., and after the search the prefix is stripped before saving to JSON.

**Output format** — `best_params.json`:

```json
{
  "xgboost_A": {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05},
  "random_forest_B": {"n_estimators": 300, "max_depth": 5}
}
```

To apply tuned params, set `USE_TUNED_PARAMS=True` in `config.py`. `RecessionSubModel` will load them automatically on the next run.

**Usage:**

```powershell
python tuner.py --model xgboost --dataset A
python tuner.py --model random_forest --dataset B
python tuner.py --all
python tuner.py --show
```

---

## `data_visualization.py` — interactive Plotly plots + matplotlib exploratory plots

All are standalone. None requires the production model to have been run already (except `--plot probability`, which trains it internally).

### Plotly interactive plots (open in browser)

| Mode | What it produces |
|---|---|
| `importance` | Horizontal bar charts of feature importance for all 10 algorithms × 3 sub-models. 10 buttons at top switch algorithm; Sub-Models A, B, C always shown side-by-side. SVC/MLP/HistGB use permutation importance; tree models use Gini; logistic uses coefficient magnitude. |
| `roc` | ROC curves for all 10 algorithms on the 2008 holdout on the same axes. Legend is interactive: single-click toggles a curve, double-click isolates it, double-click again restores all. |

### Matplotlib exploratory plots

| Mode | What it produces |
|---|---|
| `series` | 3×N grid of raw FRED series with NBER recession shading. Raw numbers as fetched — no feature engineering. |
| `features` | Grid of all engineered features for a chosen dataset. Red shading shows when `target_12m=1`. Respects `--start`/`--end`. |
| `probability` | Trains the full production model internally, plots the probability history back to 1977. |
| `correlation` | Seaborn heatmap of pairwise Pearson correlations between features. |
| `labels` | Two panels: timeline of `target_12m`, and bar chart of class balance. Shows the imbalance before training. |
| `pairplot` | Seaborn pairplot of all features, coloured by `target_12m`. |

**Usage:**

```powershell
python data_visualization.py --plot importance
python data_visualization.py --plot roc
python data_visualization.py --plot importance --model-type random_forest
python data_visualization.py --plot series
python data_visualization.py --plot features --start 2005-01 --end 2012-12
python data_visualization.py --plot probability --start 2018-01
python data_visualization.py --plot correlation --dataset A
python data_visualization.py --plot labels --dataset C
python data_visualization.py --plot all
python data_visualization.py --plot features --no-show  # save PNG, no popup
```

Plotly charts save to `output/html/` (interactive) and `output/images/` (static PNG via kaleido). Matplotlib charts save to `output/images/`.

---

## File map

```
recession/
├── config.py               — all constants (API key, weights, model type, paths)
├── data_finder.py          — FRED fetch, cache, alignment, feature engineering, get_datasets()
├── ai_model.py             — ModelType (10 types), RecessionSubModel, RecessionEnsemble,
│                             HoldOutValidator (2008 only), ProductionTrainer,
│                             compute_threshold_table
├── charts.py               — all matplotlib chart functions:
│                             plot_validation_single, plot_all (2-panel),
│                             plot_algorithm_comparison, plot_threshold_analysis
├── main.py                 — production pipeline (Steps 0, 2–4, 6–8)
├── gpu_test.py             — standalone CUDA benchmark
│
├── model_shootout.py       — CLI: compare all 10 algorithms individually on a holdout
├── tuner.py                — CLI: GridSearchCV hyperparameter search
├── data_visualization.py   — CLI: interactive Plotly + matplotlib exploratory plots
│
├── best_params.json        — written by tuner.py, read by RecessionSubModel
├── tech_stack.md           — project reference card
├── documentation/          — architecture notes
└── data/
    └── raw/                — cached FRED CSVs (auto-created on first run)
        ├── INDPRO.csv
        ├── UNRATE.csv
        ├── SAHMCURRENT.csv
        ├── HOUST.csv
        ├── T10Y2Y.csv
        ├── ICSA.csv
        ├── PAYEMS.csv
        ├── CPIAUCSL.csv
        └── USREC.csv
```
