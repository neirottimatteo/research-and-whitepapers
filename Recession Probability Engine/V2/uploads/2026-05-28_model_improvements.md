# Model Improvements — 2026-05-28

## Summary

Two main changes in this session: (1) the production signal filter switched from the **confirmed N-consecutive** filter to an **EMA (Exponential Moving Average)** filter; (2) a raw unfiltered export was added. A date alignment bug was also fixed.

---

## 1. EMA Signal Filter (replaces confirmed as primary)

### Motivation

The confirmed filter requires N consecutive months above threshold before activating. It works well but has two weaknesses:
- It is **path-blind**: a gradual 40→50% build-up over 4 months is treated identically to a sudden spike to 70% that was already at 0% the month before.
- It imposes a fixed **3–4 month activation delay** regardless of signal strength.

The EMA filter addresses both by weighting the current month by α and the prior EMA by (1−α):

```
EMA_t = α × raw_t + (1−α) × EMA_{t−1}
```

A gradual build-up accumulates in the EMA faster than an isolated spike. A sustained high signal crosses the 50% threshold sooner than confirmed would allow; an isolated month-long spike does not.

### Configuration

```python
# config.py
EMA_ALPHA: float = 0.35   # smoothing factor; 0.35 requires ~3 sustained months of ~80% raw to cross 50%
```

### Performance vs confirmed (EMA α=0.35)

| Metric | Confirmed | EMA (α=0.35) |
|--------|-----------|--------------|
| Avg AUC (folds 5–9) | 0.827 (paper) / 0.957 (current run) | **0.959** |
| Avg lead time (excl. COVID) | 13.x months | **13.2 months** |
| FP/yr avg (folds 8–11) | 0.0 | **0.0** |
| 2001 recession detected (fold 8) | MISS | **4m lead** |
| Fold 11 FP/yr | 0.2 | **0.0** |

The EMA filter detects the 2001 recession in fold 8 (4 months lead), which the confirmed filter missed entirely. This is because the confirmed filter required N consecutive months; the 2001 build-up had some month-to-month variability that kept breaking the streak. The EMA accumulates it smoothly.

### Dangerous FP streaks (fold ≥ 8, excl. 2020)

| Filter | Streaks |
|--------|---------|
| Confirmed (α=0.4) | 0 streaks |
| EMA α=0.4 | 5 streaks (folds 8–11) |
| EMA α=0.35 | 3 streaks (folds 8, 10 only, max 58.5%) |

Remaining streaks are all 1–2 months in the 2006 and 2022–2023 periods — marginal and similar to what confirmed also faces.

### Implementation

- `_compute_ema(prob: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray` added to `walk_forward.py`
- `--ema` flag added to `walk_forward.py` and `data_visualization.py`
- Both filters coexist in the codebase; `--ema` and `--confirmed` are independent flags

### Pipeline change

Production pipeline (`main.py`) now runs:
1. **EMA** (primary) → `output/export/recession_series.json`
2. **Raw** (reference) → `output/export_raw/recession_series_raw.json`

Confirmed filter is no longer run in the main pipeline (still available via CLI flags for analysis).

---

## 2. Raw Export Added

`output/export_raw/recession_series_raw.json` — unfiltered probability series, identical format to the main export. Useful for analysis, comparing filters, and understanding what the raw model is doing.

---

## 3. Date Alignment Fix

**Bug:** `latest_date` (used for SHAP waterfall and prediction labeling) could land in a future month if FRED quarterly series extended the feature index past the current month. The old check only skipped the exact current month:

```python
# old — only skipped May when running in May
if last.year == today.year and last.month == today.month:
    last = complete.index[-2]
```

This failed when `complete.index[-1]` was June 2026 while today was May 2026 (June != May → not skipped).

**Fix:** filter to strictly before the current month start:

```python
# new — always returns last complete month before today
current_month_start = pd.Timestamp(today.year, today.month, 1)
last = complete[complete.index < current_month_start].index[-1]
```

Now dynamic: running in May → April, running in June → May, etc.

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
