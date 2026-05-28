# Paper Update Notes — 2026-05-28

Reference paper: `recession_probability_engine_report.md` (May 2026)

These are the changes since that paper was written. Incorporate them in a revision.

---

## Abstract — numbers to update

| Item | Paper | Updated |
|------|-------|---------|
| Avg AUC (folds 5–11) | 0.827 | **0.959** |
| Avg lead time | 14.2 months | **13.2 months** |
| Signal filter description | "sustained-signal output filter" | "EMA smoothing filter (α=0.35)" |
| FP/yr | < 0.1 | **< 0.1** (unchanged, still true) |

---

## Section 3.2 — Production Pipeline (diagram)

Replace the pipeline diagram branches:
```
# old
    ├── confirmed signal mode
    └── raw signal mode

# new
    ├── EMA signal mode  (primary, α=0.35)
    └── raw signal mode  (reference)
```

---

## Section 4.3 — Replace the Confirmed Signal Filter section

The confirmed filter section (4.3) should be replaced or substantially revised. Suggested new section:

---

### 4.3 The EMA Signal Filter

Raw probability output from any macro model contains noise. A single month above the alarm threshold does not constitute a reliable signal — isolated spikes during mid-cycle slowdowns, rate-hike episodes, and geopolitical scares do not resolve into recessions.

The confirmed filter previously used in this system required N consecutive months above threshold before activating an alarm. While effective, it was path-blind: a gradual multi-month build-up from 40% to 55% was treated identically to an instantaneous spike from 0% to 70%. Both required the same N-month wait regardless of signal history.

The production filter was replaced with an **Exponential Moving Average (EMA)**:

```
EMA_t = α × raw_t + (1 − α) × EMA_{t−1}
```

The EMA accumulates signal history continuously. A gradual build-up over multiple months raises the EMA faster toward the threshold than an isolated spike of equal peak magnitude. The alarm threshold remains at 50%, applied to the EMA rather than the raw probability.

The smoothing factor α = 0.35 was selected to require approximately 3 sustained months of strong signal (~80% raw) to cross the 50% EMA threshold from a cold start — long enough to suppress transient false positives, short enough to respond to genuine pre-recession deterioration within the 12-month forward horizon.

**Improvement over confirmed filter:**
- Detects the **2001 recession 4 months early** in fold 8 — previously a MISS. The 2001 build-up had month-to-month variability that repeatedly broke the confirmed filter's consecutive-month counter; the EMA accumulated it smoothly.
- **Fold 11 FP/yr: 0.0** (was 0.2 with confirmed).
- **Average AUC improved** from 0.957 to 0.959.
- The hysteresis principle is preserved implicitly: a brief dip in an ongoing high-EMA period does not reset the signal, because the EMA carries the prior history forward.

#### Note on the hysteresis section

The Schmitt trigger / hysteresis description in the current paper (section 4.3) described the confirmed filter's behavior. With EMA, hysteresis emerges from the smoothing itself: the EMA cannot instantly drop from 60% to near-zero on a single low reading. The EMA_ACTIVATE and EMA_DEACTIVATE constants defined in config.py are not used in the current implementation — the 50% threshold applied uniformly to the EMA is sufficient.

---

## Section 5.1 — Results table update

| Fold | Trained Through | AUC | FP / yr |
|:----:|:--------------:|:---:|:-------:|
| 5 | Mar 1975 | 0.905 | 0.3 |
| 6 | Jul 1980 | 0.957 | 0.1 |
| 7 | Nov 1982 | 0.978 | 0.0 |
| 8 | Mar 1991 | 0.982 | 0.0 |
| 9 | Nov 2001 | 0.990 | 0.0 |
| 10 | Jun 2009 | n/a† | 0.1 |
| 11 | Apr 2020 | n/a† | 0.0 |
| **Average** | | **0.959** | **0.07** |

*Average advance warning (excl. COVID): 13.2 months*

---

## Section 7.1 — 2001 Walk-Forward Anomaly (revise)

The limitation described in section 7.1 no longer applies. With the EMA filter, fold 8 now detects the 2001 recession with **4 months lead time**. The EMA's continuous accumulation captures the 2001 build-up despite its month-to-month variability, which the confirmed filter's consecutive-month counter could not preserve.

The structural argument in section 7.1 — that the fold-8 model lacked training examples of a "CapEx bust while consumers are healthy" — remains valid as a model-level explanation. However, the filter no longer amplifies that limitation by discarding the weak but real signal that was present.

---

## Appendix — Technical Configuration update

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| Signal filter | Sustained multi-month confirmation with hysteresis (Schmitt trigger) | EMA smoothing (α=0.35, threshold 50%) |

---

## Generate commands for updated figures

```powershell
# Figure 2 — raw unfiltered (unchanged command)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --uniform-color --no-show

# Figure 3 — EMA filtered (replaces confirmed)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --ema --uniform-color --no-show

# Figure 4 — primary result chart (EMA, folds 4–11)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --min-fold 4 --ema --uniform-color --no-show

# Figure 5 — SHAP summary (unchanged)
python data_visualization.py --plot shap-summary --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --no-show

# Figure 6 — SHAP waterfall (update date)
python data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --date 2026-04 --no-show
```
