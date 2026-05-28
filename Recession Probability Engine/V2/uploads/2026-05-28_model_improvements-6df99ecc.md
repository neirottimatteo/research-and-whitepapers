# Model Improvements — 2026-05-28

## Summary

Three changes, ordered by impact on AUC:

1. **EMA signal filter** replaces confirmed as primary production filter — primary driver of AUC 0.835 → 0.959
2. **Data alignment fix** (commit `4d24b77`) — smaller contribution to AUC (0.827 → 0.835 with confirmed)
3. **Raw export** added (`output/export_raw/`)

> **Correction from earlier analysis:** the AUC jump is primarily from EMA, not the alignment fix.
> AUC is computed on EMA-smoothed probabilities in walk_forward.py — EMA smoothing improves the
> model's ranking of pre-recession months by reducing noise, which directly raises AUC.
> The alignment fix alone (confirmed filter, weights 1.0/0.0) only improved AUC from 0.827 → 0.835.

---

## 1. EMA Signal Filter (primary AUC driver)

### Motivation

The confirmed filter requires N consecutive months above threshold before activating. It is path-blind: a gradual 40→50% build-up over 4 months is treated identically to a sudden spike from 0% to 70%. Both require the same N-month wait.

The EMA filter accumulates history continuously:

```
EMA_t = α × raw_t + (1−α) × EMA_{t−1}
```

A gradual build-up crosses 50% EMA sooner than an isolated spike of equal peak magnitude.

### Why EMA affects AUC

In `walk_forward.py`, AUC is computed on the **filtered** probabilities (line ~790):
```python
prob_filtered = _compute_ema(prob_test) if ema else _filter_short_streaks(...)
eval_r = _evaluate_fold(test_dates, prob_filtered, ...)  # AUC computed here
```
EMA smoothing reduces month-to-month noise and produces a more stable probability series, which improves the model's ranking of pre-recession months vs. normal months — directly raising AUC.

### Configuration

```python
# config.py
EMA_ALPHA: float = 0.35   # ~3 sustained months of ~80% raw needed to cross 50% from cold start
```

### Performance comparison (all runs: after alignment fix, weights 1.0/0.0)

| Filter | Avg AUC | Avg lead time | 2001 fold 8 | Fold 11 FP/yr |
|--------|---------|---------------|-------------|---------------|
| Confirmed | 0.835 | 11.1 months | MISS | 0.0 |
| EMA (α=0.35) | **0.959** | **13.2 months** | **4m lead** | **0.0** |

EMA improves AUC by **+0.124** and lead time by **+2.1 months** vs confirmed. It also detects the 2001 recession in fold 8 where confirmed missed it — the 2001 build-up had month-to-month variability that kept resetting the confirmed filter's consecutive-month counter; EMA accumulates it smoothly.

### Dangerous FP streaks (fold ≥ 8, excl. 2020) with EMA α=0.35

| Fold | Period | Duration | Max EMA |
|------|--------|----------|---------|
| 8 | 2006-09 | 1 month | 53.7% |
| 8 | 2023-01 to 02 | 2 months | 55.1% |
| 10 | 2023-02 | 1 month | 58.5% |

All marginal (1–2 months, below 59%) and concentrated in the 2006 housing stress and 2022–2023 rate-hike periods.

### Implementation

- `_compute_ema(prob: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray` in `walk_forward.py`
- `--ema` flag in `walk_forward.py` and `data_visualization.py`
- `--confirmed` still available for reference

### Pipeline change

`main.py` now runs:
1. **EMA** (primary) → `output/export/recession_series.json`
2. **Raw** (reference) → `output/export_raw/recession_series_raw.json`

Confirmed removed from production pipeline.

---

## 2. Data Alignment Fix (commit `4d24b77`)

### What was wrong

`_align_to_monthly()` in `data_finder.py` applied `MonthEnd(0)` to all series, snapping data to the end of its reference period. FRED series are published with a lag:

- **Monthly series**: March data released in April → was appearing at March 31, should be April 30
- **Quarterly series**: Q4 data released ~4 months after quarter-end → was appearing at Dec 31, should be Jan 31 of the following year

This was lookahead leakage: the model saw data 1–3 months earlier than it would have in real time.

### The fix

```python
elif freq == "quarterly":
    s.index = s.index + pd.offsets.MonthEnd(4)   # Q4 Oct 1 → Jan 31
else:
    s.index = s.index + pd.offsets.MonthEnd(2)   # Mar 1 → Apr 30
```

### Impact on AUC

Confirmed filter, weights 1.0/0.0: **0.827 → 0.835** (+0.008). Meaningful correction of data integrity, but small AUC contribution compared to EMA.

---

## 3. Raw Export Added

`output/export_raw/recession_series_raw.json` — unfiltered series, same format as main export.

---

## 4. Prediction Date Fix (`latest_date` in main.py)

`latest_date` was resolving to a future month when FRED quarterly data extended the feature index past today. Fixed by filtering strictly before current month start:

```python
current_month_start = pd.Timestamp(today.year, today.month, 1)
last = complete[complete.index < current_month_start].index[-1]
```

Dynamic: running in May → April, June → May, etc.

---

## Updated Commands

### Production run
```powershell
python main.py --no-show --cache
python main.py --no-show           # fresh FRED download
```

### Walk-forward validation
```powershell
# EMA filter (primary)
python walk_forward.py --dataset E --ensemble --weights 1 0 --feature-set plus --threshold 50 --ema

# Raw (no filter)
python walk_forward.py --dataset E --ensemble --weights 1 0 --feature-set plus --threshold 50

# Confirmed filter (reference)
python walk_forward.py --dataset E --ensemble --weights 1 0 --feature-set plus --threshold 50 --confirmed
```

### Visualization
```powershell
# EMA filter — history-folds (primary chart)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --min-fold 4 --uniform-color --ema

# EMA filter — full history
python data_visualization.py --plot history --dataset E --ensemble --weights 1 0 --feature-set plus --ema

# Raw
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --min-fold 4 --uniform-color

# SHAP (filter-independent)
python data_visualization.py --plot shap-summary --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --no-show
python data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --date 2026-04 --no-show
```

---

## Output Structure (updated)

```
output/
├── export/                         ← primary (EMA-filtered)
│   ├── recession_series.json
│   └── waterfall_latest.png
├── export_raw/                     ← raw unfiltered (new)
│   └── recession_series_raw.json
├── html/
│   ├── viz_history_folds_E_ens_ema.html    ← primary
│   ├── viz_history_E_ens_ema.html          ← primary
│   ├── viz_history_folds_E_ens.html        ← raw reference
│   └── viz_history_E_ens.html             ← raw reference
└── images/
    ├── viz_shap_summary_e_xgboost_ens.png
    └── viz_shap_waterfall_e_xgboost_ens_YYYY-MM.png
```
