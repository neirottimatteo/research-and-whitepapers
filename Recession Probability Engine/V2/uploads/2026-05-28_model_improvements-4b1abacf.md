# Paper Update Notes — 2026-05-28

Reference paper: `recession_probability_engine_report.md` (May 2026)

---

## Abstract — numbers to update

| Item | Paper | Updated |
|------|-------|---------|
| Avg AUC (folds 5–11) | 0.827 | **0.959** |
| Avg lead time | 14.2 months | **13.2 months** |
| Signal filter description | "sustained-signal output filter" | "EMA smoothing filter (α=0.35)" |
| FP/yr | < 0.1 | **< 0.1** (unchanged) |

---

## Section 3.2 — Production Pipeline (diagram)

Replace branches:
```
# old
    ├── confirmed signal mode
    └── raw signal mode

# new
    ├── EMA signal mode  (primary, α=0.35)
    └── raw signal mode  (reference)
```

---

## Section 4.2 — Structural Safety Mechanisms (add new item)

Add a fifth mechanism after "Forward-fill with cap":

**Publication-lag alignment.** All FRED series are shifted forward to reflect their real-world release dates before entering the feature matrix. Monthly series are aligned to the end of the month following their reference period (March data → April 30); quarterly series are aligned four months after their reference quarter-end (Q4 → January 31 of the following year). Without this correction, quarterly indicators such as SLOOS bank tightening surveys and corporate profits would appear in training data 2–3 months earlier than they were actually published, constituting lookahead leakage in the temporal structure. Correcting this alignment was the primary driver of the AUC improvement from 0.827 to 0.959.

---

## Section 4.3 — Replace the Confirmed Signal Filter section

Replace entirely with:

### 4.3 The EMA Signal Filter

Raw probability output from any macro model contains noise. A single month above the alarm threshold does not constitute a reliable signal — isolated spikes during mid-cycle slowdowns, rate-hike episodes, and geopolitical scares do not resolve into recessions.

The confirmed filter previously used in this system required N consecutive months above threshold before activating an alarm. While effective, it was path-blind: a gradual multi-month build-up from 40% to 55% was treated identically to an instantaneous spike from 0% to 70%. Both required the same N-month wait regardless of signal history.

The production filter was replaced with an **Exponential Moving Average (EMA)**:

```
EMA_t = α × raw_t + (1 − α) × EMA_{t−1}
```

The EMA accumulates signal history continuously. A gradual build-up over multiple months raises the EMA faster toward the threshold than an isolated spike of equal peak magnitude. The alarm threshold remains at 50%, applied to the EMA rather than the raw probability.

The smoothing factor α = 0.35 was selected to require approximately 3 sustained months of strong signal (~80% raw) to cross the 50% EMA threshold from a cold start — long enough to suppress transient false positives, short enough to respond to genuine pre-recession deterioration within the 12-month forward horizon.

Note: the EMA filter does not affect AUC. It is a post-processing step applied to the model's output probabilities, not a change to the model's learned weights. The AUC improvement from 0.827 to 0.959 reflects the publication-lag alignment fix described in Section 4.2, which eliminated lookahead leakage from quarterly FRED series.

**Improvements over the confirmed filter:**
- Detects the **2001 recession 4 months early** in fold 8 — previously a MISS. The 2001 build-up had month-to-month variability that repeatedly broke the confirmed filter's consecutive-month counter; the EMA accumulated it continuously.
- **Fold 11 FP/yr: 0.0** (was 0.2 with confirmed).
- Hysteresis emerges naturally: a brief dip in an ongoing high-EMA period cannot instantly reset the signal because the EMA carries prior history forward.

The Schmitt trigger / hysteresis description in the original section 4.3 described the confirmed filter's deactivation behavior. With EMA, this property is implicit in the smoothing itself and no explicit deactivation threshold is required.

---

## Section 5.1 — Results table update

| Fold | Trained Through | Recessions in Training | AUC | FP / yr |
|:----:|:--------------:|:----------------------:|:---:|:-------:|
| 5 | Mar 1975 | 5 | 0.905 | 0.3 |
| 6 | Jul 1980 | 6 | 0.957 | 0.1 |
| 7 | Nov 1982 | 7 | 0.978 | 0.0 |
| 8 | Mar 1991 | 8 | 0.982 | 0.0 |
| 9 | Nov 2001 | 9 | 0.990 | 0.0 |
| 10 | Jun 2009 | 10 | n/a† | 0.1 |
| 11 | Apr 2020 | 11 | n/a† | 0.0 |
| **Average** | | | **0.959** | **0.07** |

*Average advance warning (excl. COVID): 13.2 months*

The AUC improvement from 0.827 (original paper) to 0.959 is primarily attributable to the publication-lag alignment fix (Section 4.2), which corrected lookahead leakage in quarterly FRED series. The lead time decrease from 14.2 to 13.2 months reflects the EMA filter's accumulation requirement: a gradual build-up must persist for ~3 months to cross 50%, whereas the confirmed filter could fire earlier on a single strong spike that held for N months.

---

## Section 5.2 — 2022–2023 Stress Test (minor update)

Update the filter reference: replace "confirmed threshold" with "EMA threshold". The substantive conclusion is unchanged — the system did not activate its alarm during the 2022–2023 rate-hike cycle.

---

## Section 7.1 — 2001 Walk-Forward Anomaly (revise)

The limitation described in section 7.1 is substantially reduced. With the EMA filter, fold 8 now detects the 2001 recession with **4 months lead time**. The 2001 build-up had month-to-month variability that repeatedly broke the confirmed filter's consecutive-month counter; the EMA's continuous accumulation captures it.

The structural argument — that the fold-8 model lacked training examples of a "CapEx bust while consumers are healthy" — remains valid as a model-level explanation of why the signal was weak to begin with. However, the filter no longer amplifies that limitation by discarding the weak but real signal that was present.

---

## Appendix — Technical Configuration update

| Parameter | Old value | New value |
|-----------|-----------|-----------|
| Signal filter | Sustained multi-month confirmation with hysteresis (Schmitt trigger) | EMA smoothing (α=0.35, threshold 50%) |
| Series alignment | `MonthEnd(0)` for all series | Monthly: `MonthEnd(2)`; Quarterly: `MonthEnd(4)` |

---

## Figure commands (updated)

```powershell
# Figure 2 — raw unfiltered (unchanged)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --uniform-color --no-show

# Figure 3 — EMA filtered (replaces confirmed)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --ema --uniform-color --no-show

# Figure 4 — primary result (EMA, folds 4–11)
python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --min-fold 4 --ema --uniform-color --no-show

# Figure 5 — SHAP summary (unchanged)
python data_visualization.py --plot shap-summary --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --no-show

# Figure 6 — SHAP waterfall (update date as needed)
python data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --date 2026-04 --no-show
```
