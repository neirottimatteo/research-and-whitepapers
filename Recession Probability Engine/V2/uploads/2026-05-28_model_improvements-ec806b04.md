# Paper Update Notes — 2026-05-28

Reference paper: `recession_probability_engine_report.md` (May 2026)

> **Key correction:** the original paper's AUC of 0.827 used the confirmed filter. The new AUC of 0.959
> is primarily driven by the **EMA filter**, not the data alignment fix. Confirmed after alignment fix
> gives AUC 0.835. EMA after alignment gives 0.959. The documentation should reflect EMA as the
> primary methodological contribution to performance.
> Additionally: AUC IS affected by EMA because walk_forward.py computes AUC on the filtered
> (EMA-smoothed) probabilities — not the raw model output.

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

## Section 4.2 — Structural Safety Mechanisms (add item)

Add a fifth mechanism after "Forward-fill with cap":

**Publication-lag alignment.** All FRED series are shifted forward to reflect their real-world release dates before entering the feature matrix. Monthly series are aligned to the end of the month following their reference period (March data → April 30); quarterly series are aligned four months after their reference quarter-end (Q4 → January 31 of the following year). Without this correction, quarterly indicators such as SLOOS bank tightening surveys and corporate profits would appear in training data 1–3 months earlier than they were actually published.

---

## Section 4.3 — Replace the Confirmed Signal Filter section

Replace entirely with:

### 4.3 The EMA Signal Filter

Raw probability output from any macro model contains noise. A single month above the alarm threshold does not constitute a reliable signal — isolated spikes during mid-cycle slowdowns, rate-hike episodes, and geopolitical scares do not resolve into recessions.

The confirmed filter previously used required N consecutive months above threshold before activating. It was path-blind: a gradual multi-month build-up from 40% to 55% was treated identically to an instantaneous spike from 0% to 70%. The confirmed filter also introduces a fixed 3–4 month activation delay and discards the cumulative signal history at every brief interruption.

The production filter was replaced with an **Exponential Moving Average (EMA)**:

```
EMA_t = α × raw_t + (1 − α) × EMA_{t−1}
```

The EMA accumulates signal history continuously. A gradual build-up raises the EMA faster toward threshold than an isolated spike of equal peak magnitude. The alarm fires when the EMA crosses 50%, applied to the smoothed series rather than the raw probability.

The smoothing factor α = 0.35 requires approximately 3 sustained months of strong signal (~80% raw) to cross the 50% EMA threshold from a cold start.

**Note on AUC:** Unlike a purely threshold-based post-filter, EMA directly affects the AUC reported in walk-forward validation. The AUC is computed on the EMA-smoothed probability series, not the raw model output. EMA smoothing reduces month-to-month noise and produces a more stable ranking of pre-recession months vs. normal months, which raises AUC. This is the intended behavior — the EMA represents the operational signal the system produces, and AUC should reflect the quality of that signal.

**Measured improvement over confirmed filter** (both evaluated after the publication-lag alignment fix, pure XGBoost ensemble):

| Filter | Avg AUC | Avg lead time | 2001 (fold 8) |
|--------|---------|---------------|---------------|
| Confirmed | 0.835 | 11.1 months | MISS |
| EMA (α=0.35) | **0.959** | **13.2 months** | **4m lead** |

The EMA improves AUC by +0.124 and average lead time by +2.1 months. It also detects the 2001 recession in fold 8, which the confirmed filter missed: the 2001 pre-recession build-up had month-to-month variability that repeatedly broke the confirmed filter's consecutive-month counter. The EMA accumulated it continuously.

Hysteresis emerges implicitly from the smoothing: a brief dip during a sustained high-EMA period cannot instantly reset the signal, because the EMA carries prior history forward. No explicit Schmitt trigger deactivation threshold is required.

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

The AUC improvement from 0.827 (original paper, confirmed filter) to 0.959 is primarily driven by the EMA filter. Running the confirmed filter under the same conditions (post alignment-fix, pure XGBoost) gives AUC 0.835, demonstrating that the alignment fix alone contributed only +0.008. The EMA filter accounts for the remaining +0.124 improvement.

---

## Section 5.2 — 2022–2023 Stress Test (minor update)

Replace "confirmed threshold" with "EMA threshold". Substantive conclusion unchanged.

---

## Section 7.1 — 2001 Walk-Forward Anomaly (revise substantially)

The limitation described in section 7.1 no longer applies. With the EMA filter, fold 8 now detects the 2001 recession with **4 months lead time**. The confirmed filter missed it because the 2001 probability build-up had month-to-month variability; each brief dip below threshold reset the consecutive-month counter before the alarm could activate. The EMA's continuous accumulation captures the same underlying signal without requiring uninterrupted threshold exceedance.

The structural argument — that the fold-8 model lacked training examples of a "CapEx bust while consumers are healthy" — remains valid as a model-level explanation of why the signal was weaker than for other recessions. However, the filter no longer discards the weak but genuine signal that was present.

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
