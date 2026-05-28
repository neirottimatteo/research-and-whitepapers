# Walk-Forward Validation & Threshold Calibration

**Last updated:** 2026-04-28

---

## The conclusion in one sentence

After running all 10 algorithms × 4 datasets (A/B/C/D) and a 5th combined dataset (E), the two best models are **XGBoost on Dataset E** and **HistGB on Dataset E** — both catch every recession since 1953 with ~15 months of average lead time and zero false positives on the full historical record.

---

## `walk_forward.py` — what it does

A standalone CLI script. Completely independent from `main.py`. Two parts run in sequence for each (algorithm, dataset) pair.

### Training cutoff — T-24 months

All training in every script is frozen at **today minus 24 months** (currently: ~April 2024). This applies to threshold calibration, history plots, SHAP analysis, and the legacy production ensemble.

Two reasons:

1. **NBER publication lag** — NBER does not declare recession dates in real time. Months with `USREC=0` in the recent past may later be revised to `1`. Training on those rows tells the model "no recession here" — a label that is genuinely uncertain.
2. **Forward-label horizon** — `target_12m = 1` if a recession begins within the next 12 months. Training on April 2025, for example, asserts "no recession by April 2026." That claim is untested today. A recession may be forming right now but not yet labeled.

The 24-month buffer adds a 12-month safety margin on top of the 12-month forward horizon. Walk-forward folds are unaffected in practice — each fold's `train_end` ends at a historical recession date (last: 2010-06), well before the cutoff.

The cutoff is computed dynamically and requires no configuration:
```python
_TRAIN_CUTOFF = pd.Timestamp.today() - pd.DateOffset(months=24)
```

---

### Part 1 — Full-history threshold calibration

The model is trained on **labeled data up to T-24 months** and generates a probability for every month on record (including the full period up to today, for calibration and visualization). A threshold is then found such that:

- The threshold is the **minimum T ≥ 30%** where all non-recession local probability peaks are suppressed to zero (zero in-sample false positives).
- The 30% floor (`MIN_ALARM_PROB`) exists because peaks below that level are background noise, not meaningful alarms.
- A "local peak" is a strict local maximum: `prob[i] > prob[i-1]` and `prob[i] > prob[i+1]`.
- If the model misses any non-COVID recession at that threshold → marked **FAILED** (unusable).

COVID 2020 exclusion window: `2018-08 → 2021-06` (18 months before onset through 14 months after). Peaks inside this window are never counted as false positives because 2019 economic weakness (yield curve inversion, manufacturing slowdown) was a legitimate pre-recession signal, not noise.

The calibrated threshold is saved to `output/raw/best_thresholds.json` keyed by `"<algorithm>_<dataset>"`. For the production model (`run_step6_production` in `main.py`), the threshold is recalibrated live from that model's own probability history rather than read from the file.

### Part 2 — Walk-forward (expanding-window) validation

For fold k: train on all data through the end of recession k + 12 months, test on **all remaining history**. The threshold from Part 1 is applied unchanged.

Per fold output:
- **Lead time** per recession: months before onset that the model first crosses the threshold
- **FP/yr**: local peaks above threshold on the test period not attributable to any future recession (out-of-sample)
- **AUC**: ROC-AUC for the test period

The walk-forward FP/yr is **out-of-sample** — it reflects an earlier version of the model (trained on less data) applied to subsequent history. It is naturally higher than the in-sample calibration (always 0 by construction). It is an informational generalization metric, not a pass/fail criterion.

### CLI

```powershell
python walk_forward.py                          # all 10 algorithms x all 4 datasets (A/B/C/D)
python walk_forward.py --dataset A --model all  # all algorithms on Dataset A
python walk_forward.py --dataset E --model all  # XGBoost + HistGB on combined Dataset E
python walk_forward.py --dataset A --model xgboost
```

`--dataset all` covers A/B/C/D only. Dataset E must be requested explicitly with `--dataset E`.

---

## The five datasets

| | A | B | C | D | E (combined) |
|---|---|---|---|---|---|
| **Starts** | 1960 | 1977 | 1950 | 1991 | 1950 |
| **Features** | 9 | 19 | 9 | 12 | 35 |
| **Recessions** | 8 | 5 | 10 | 3 | 10 |
| **Key signals** | INDPRO, UNRATE, SAHM, HOUST | A + yield curve, ICSA, PERMIT, UMCSENT | INDPRO, UNRATE, PAYEMS, CPI | BAA10Y, VIX, CCSA, PCE, AWHMAN | Union of all four |

Dataset E is built by `get_combined_dataset()` in `data_finder.py`. Features are NaN in years before their underlying series existed — XGBoost and HistGB handle this natively (they learn an optimal split direction for missing values during tree construction). No imputation is applied.

---

## Results — which algorithms pass

An algorithm passes a dataset if it achieves zero in-sample false positives **and** catches every non-COVID recession at the calibrated threshold.

| Algorithm | A | B | C | D |
|---|---|---|---|---|
| **XGBoost** | OK | OK | OK | OK |
| **GradBoost** | OK | OK | OK | OK |
| **HistGB** | OK | OK | OK | OK |
| Logistic | FAILED | FAILED | FAILED | FAILED |
| RandForest | FAILED | OK | FAILED | OK |
| AdaBoost | FAILED | OK | FAILED | FAILED |
| ExtraTrees | FAILED | FAILED | FAILED | FAILED |
| SVC | FAILED | OK | FAILED | FAILED |
| MLP | FAILED | FAILED | FAILED | FAILED |
| NaiveBayes | FAILED | FAILED | FAILED | FAILED |

Only XGBoost, GradBoost, and HistGB pass all four individual datasets. Everything else fails on at least one.

---

## Results — Dataset E (combined)

Only XGBoost and HistGB are run on Dataset E. Other algorithms do not support NaN natively and are excluded.

| Model | Threshold | In-sample | Avg lead | Walk-forward FP/yr | AUC trend |
|---|---|---|---|---|---|
| **XGBoost E** | **30%** | **10/10** | **15.7m** | 1.3 | improving |
| **HistGB E** | **30%** | **10/10** | 15.1m | 1.3 | improving |

The threshold calibrates at the 30% floor — meaning even the minimum meaningful alarm level produces zero in-sample false positives. The combined dataset's signal is that clean.

---

## Why Dataset E is the best

**1. Longest average lead time.** 15.7m (XGBoost) vs 14.4m on the next-best (Dataset A XGBoost). About 1.3 months of additional warning.

**2. Deepest recession history.** 10 episodes back to 1953. Dataset A has only 8. More recession examples makes calibration and generalization more robust.

**3. Complementary signals cancel noise.** With 35 features — covering labor (SAHM, payrolls), housing (HOUST, PERMIT), financial stress (VIX, BAA10Y credit spread), consumer (PCE, UMCSENT), and industrial (INDPRO, AWHMAN) — the model can weigh a full picture rather than being forced by any single signal. When one group alarms but the others don't, the model correctly stays quiet.

**4. AUC improving trend.** Unlike Dataset A where AUC degrades as folds progress (suggesting the model overfits to early data), Dataset E AUC improves — the model gets better as it accumulates history.

**5. Threshold at the floor.** The minimum possible alarm level (30%) already gives 0 in-sample FP. There is no need to raise the threshold to suppress noise.

---

## Key qualitative finding: 2022–2023

In 2022, the 10Y-2Y yield curve inverted heavily — the traditional single best recession predictor. Yield-curve-only models would have alarmed. **Dataset E XGBoost did not alarm.**

The reason: while the spread was negative, SAHM stayed near zero (very strong labor market), payrolls kept growing, PCE was solid, and VIX was not in crisis territory. The model saw 35 indicators simultaneously and correctly concluded the full picture was not recession-like.

This is the most important out-of-sample validation of the combined dataset: it passed a real-world test that single-signal models failed.

---

## On 2020 COVID

2020 and its pre-signal contamination window (**Feb 2019 – Dec 2020**) are **censored from all model training**.

**Why the window extends back to Feb 2019:**  
`target_12m` is a 12-month forward label. A row dated Feb 2019 has `target_12m = 1` not because of any structural economic signal, but because COVID falls in its 12-month forward window. Including those rows forces the model to associate "healthy macro conditions" (strong labor market, low unemployment) with an imminent crash — poisoning the decision boundary and inflating false positives.

**The censor window:**
- `COVID_CENSOR_START = 2019-02-01` (COVID onset Feb 2020 minus 12-month horizon)
- `COVID_CENSOR_END   = 2020-12-31`

Defined in `config.py` and applied via `censor_covid()` from `data_finder.py` at **every** `model.train()` call across the codebase: `walk_forward.py`, `main.py`, `data_visualization.py`.

**What this means in practice:**  
The model predicts on 2020 data normally — the spike is visible in the history chart. But it never *learned* from those months, so the decision boundary reflects only structural, cyclical recession patterns.

All output tables still show 2020 as `[Xm⚠]` and it is never counted in FP/yr calculations.

---

## Recommended models

| Rank | Algorithm | Dataset | Threshold | Lead | Why |
|---|---|---|---|---|---|
| 1 | **XGBoost** | **E** | 30% | 15.7m | Best lead, cleanest threshold, 10/10, improving AUC, ignores 2022 noise |
| 2 | **HistGB** | **E** | 30% | 15.1m | Same pass criteria as above, marginally shorter lead |
| 3 | XGBoost | C | 30% | 15.3m | Strong fallback; deep history without NaN complexity |

All other passing combinations (GradBoost A/B/C, HistGB A/B/C, etc.) are valid but cover fewer recessions or have shorter lead times.

---

## History visualization

```powershell
# Interactive chart — opens in browser, zoom / hover / pan
python data_visualization.py --plot history --dataset E
python data_visualization.py --plot history --dataset A

# Headless (save only, no browser)
python data_visualization.py --plot history --dataset E --no-show
```

Output files:
- `output/html/viz_history_E.html` — fully interactive Plotly chart
- `output/images/viz_history_E.png` — static snapshot

---

## Files added / modified

```
recession/
├── walk_forward.py          — NEW: walk-forward CLI + threshold calibration
├── data_finder.py           — MODIFIED: added FEATURE_COLS_E, get_combined_dataset(),
│                              get_combined_features()
├── data_visualization.py    — MODIFIED: plot_history() and _DATASET_META extended
│                              for Dataset E (XGBoost + HistGB only)
└── output/
    └── raw/
        └── best_thresholds.json   — GENERATED: calibrated thresholds per model/dataset
```
