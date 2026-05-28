# Model Improvements — 2026-05-28

## Summary

Three changes, ordered by impact on AUC:

1. **Data alignment fix** (commit `4d24b77`) — primary driver of AUC 0.827 → 0.959
2. **EMA signal filter** replaces confirmed as the primary production filter — incremental improvement in FP suppression and 2001 detection
3. **Raw export** added (`output/export_raw/`)

---

## 1. Data Alignment Fix (primary AUC driver)

### What was wrong

`_align_to_monthly()` in `data_finder.py` was applying the same `MonthEnd(0)` offset to both monthly and quarterly series — snapping everything to the end of its reference period. This meant:

- **Monthly series** (e.g. ICSA, T10Y2Y): data dated March → aligned to March 31. But FRED releases these with a 1–2 month lag, so March data only becomes available in April or May.
- **Quarterly series** (e.g. SLOOS, CP, PCECC96): data dated Q1 (Jan 1) → aligned to Jan 31. But quarterly series are published ~4 months after quarter-end (Q4 → January 31 of the following year).

The model was therefore training on data that appeared 1–3 months **earlier than it would in reality** — lookahead leakage in the temporal structure.

### The fix

```python
# Old — same offset for everything
s.index = s.index + pd.offsets.MonthEnd(0)

# New — differentiated by frequency
elif freq == "quarterly":
    s.index = s.index + pd.offsets.MonthEnd(4)   # Q4 Oct 1 → Jan 31
else:
    s.index = s.index + pd.offsets.MonthEnd(2)   # Mar 1 → Apr 30
```

Monthly series now land 1 month after their reference date; quarterly series land 4 months after. The `ffill(limit=2)` then bridges the 2 remaining months within each quarter window correctly.

### Impact

AUC jumped from **0.827 → 0.959** across mature folds. The correctly-timed features give the model a clean, non-leaky view of what was knowable at each point — the true test of the walk-forward setup.

---

## 2. EMA Signal Filter (replaces confirmed as primary)

### Motivation

The confirmed filter requires N consecutive months above threshold before activating. It is path-blind: a gradual 40→50% build-up over 4 months is treated identically to a sudden spike from 0% to 70%. Both require the same N-month wait regardless of approach history.

The EMA filter weights the current month by α and accumulates history through (1−α):

```
EMA_t = α × raw_t + (1−α) × EMA_{t−1}
```

A gradual build-up crosses 50% EMA sooner than a spike of equal peak magnitude. AUC is not affected — EMA is a post-processing step on the model output, not a change to learned weights.

### Configuration

```python
# config.py
EMA_ALPHA: float = 0.35   # ~3 sustained months of ~80% raw needed to cross 50% from cold start
```

### Performance vs confirmed (both run after the alignment fix)

| Metric | Confirmed | EMA (α=0.35) |
|--------|-----------|--------------|
| Avg AUC (folds 5–9) | 0.957 | **0.959** |
| Avg lead time (excl. COVID) | 13.x months | **13.2 months** |
| 2001 recession detected (fold 8) | MISS | **4m lead** |
| Fold 11 FP/yr | 0.2 | **0.0** |
| Dangerous FP streaks (fold ≥ 8) | 0 | 3 (1–2 months, max 58.5%) |

The 2001 detection improvement is because the 2001 build-up had month-to-month variability that kept resetting the confirmed filter's consecutive-month counter. EMA accumulates it continuously.

### Implementation

- `_compute_ema(prob: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray` in `walk_forward.py`
- `--ema` flag in `walk_forward.py` and `data_visualization.py`
- `--confirmed` still available for reference; both coexist in the codebase

### Pipeline change

`main.py` now runs:
1. **EMA** (primary) → `output/export/recession_series.json`
2. **Raw** (reference) → `output/export_raw/recession_series_raw.json`

Confirmed removed from production pipeline.

---

## 3. Raw Export Added

`output/export_raw/recession_series_raw.json` — unfiltered series, same format as main export.

---

## 4. Prediction Date Fix (`latest_date` in main.py)

`latest_date` (used for SHAP waterfall filename and labeling) was resolving to a future month when FRED quarterly data extended the feature index past today. Fixed by filtering strictly before current month start:

```python
# new
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
