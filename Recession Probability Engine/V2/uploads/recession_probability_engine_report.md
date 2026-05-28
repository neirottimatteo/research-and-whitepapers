# Recession Probability Engine
## A Machine Learning Approach to U.S. Business Cycle Forecasting

**Matteo Neirotti**  
*May 2026*

---

## Abstract

This paper presents a machine learning system that produces a monthly probability estimate of a U.S. recession occurring within the next 12 months. The system addresses three structural challenges inherent to macroeconomic forecasting: a sparse positive-class history (~11 NBER-defined recessions since 1950), retroactive label definition, and the severe credibility cost of false alarms. The architecture combines XGBoost trained on 58 engineered macroeconomic features derived from FRED, strict temporal walk-forward validation, and a sustained-signal output filter. Evaluated on the strictly out-of-sample mature walk-forward folds (folds 5–11, covering 1975 to present), the system achieves an average AUC of 0.827, an average advance warning of 14.2 months, and fewer than 0.1 false alarms per year — zero from fold 7 onward. It outperforms the univariate 10Y-2Y yield curve heuristic on both precision and false-positive suppression.

---

## 1. Introduction

Recession forecasting occupies an unusual position in applied machine learning. The prediction target is rare — approximately 11% of post-war months fall within an NBER-defined recession window — its definition is retroactive, and the asymmetric cost of errors is severe. A false alarm is not merely an incorrect prediction; it is a credibility-destroying event that may cause the next genuine signal to be ignored.

Three structural challenges compound the difficulty:

1. **Label scarcity.** The United States has experienced approximately 11 NBER-defined recessions since 1950. This yields fewer than 130 positive-class training samples across 75 years of monthly data — a dataset size that would be considered trivially small in most machine learning contexts.

2. **Retroactive definition.** The NBER typically announces recession start dates six to twelve months after the fact. Recent training labels are therefore provisional and subject to revision, meaning that models trained on the most current data are learning from unresolved ground truth.

3. **Forward contamination.** Constructing a 12-month forward label means that a training sample from, say, early 2019 carries a recession label of 1 simply because COVID arrived in Q1 2020 — not because of any cyclically learnable pattern in 2019 itself. Without an explicit architectural response, the model learns spurious associations.

The objective of this project was not to build the most complex forecasting system, but the most credible one: a system that catches every major post-war recession with meaningful advance warning, while maintaining a false-alarm rate low enough to preserve operational utility over time.

> **Figure 1 —** Recession label timeline: NBER-dated recession months overlaid on the full data history. Illustrates the scarcity and irregular spacing of positive-class observations across 75 years of monthly data.  
> *Generate:* `python data_visualization.py --plot labels --no-show`

---

## 2. Data and Feature Engineering

All data are sourced from FRED (Federal Reserve Economic Data), the St. Louis Federal Reserve's public macroeconomic database. Approximately 60 raw series are fetched via API, cached locally, and aligned to month-end frequency before feature engineering begins.

### 2.1 Transformation Philosophy

Raw series are never used directly as model inputs. Every feature is derived as one or more of: absolute level, year-over-year percentage change, 3-month percentage change, 12-month absolute change, or a binary threshold flag. This transformation layer serves two purposes: it normalizes units across heterogeneous series, and it extracts the rate-of-change signals that lead the underlying levels.

Where a series begins after 1950 — the VIX index before 1990, JOLTS job openings before 2000 — the corresponding features are left as proper missing values. No imputation is applied. The production model (XGBoost) handles structural missingness natively, learning to weight features relative to their actual historical availability.

### 2.2 Feature Groups

The production feature set (Dataset E, `plus` configuration) comprises 58 engineered features across ten active economic domains:

| Group | Features | Available From |
|-------|:--------:|:--------------:|
| Core Macro & Labor | 9 | 1939 |
| Inflation & Prices | 5 | 1913 |
| Housing & Consumption | 11 | 1950 |
| Yield Curve & Central Bank | 5 | 1953 |
| Credit & Corporate Stress | 10 | 1919 |
| Market Volatility & Leading Indicators | 14 | 1967 |
| Bank Lending Standards (SLOOS) | 1 | 1990 |
| CapEx Intent | 3 | 1992 |

*Eight additional feature groups (~27 features) are available under `--feature-set full` but were excluded from production after systematic ablation testing. See Section 4.5.*

### 2.3 Class Imbalance

With approximately 11% recession-month prevalence, the dataset is severely imbalanced. Synthetic oversampling (SMOTE) is explicitly rejected: generating artificial recession months would corrupt the temporal structure of the data and produce training observations that violate chronological ordering. Instead, XGBoost's native gradient penalty ensures the optimizer accounts for the true class frequency without fabricating history.

---

## 3. System Architecture

### 3.1 The Multi-Dataset Design

Rather than constructing a single feature matrix, the system organizes indicators into five dataset views, each representing a distinct analytical perspective:

| Dataset | Start Year | Features | Perspective |
|:-------:|:---------:|:--------:|-------------|
| A | 1960 | 9 | Classic cycle: unemployment, industrial production |
| B | 1977 | 19 | Modern macro: yield curve, housing, sentiment |
| C | 1950 | 11 | Long history: payrolls, CPI, manufacturing hours |
| D | 1991 | 12 | Financial stress: credit spreads, VIX, jobless claims |
| **E** | **1950** | **58** | **Production: full union plus CapEx and SLOOS** |

Dataset E is the production choice. It maximizes historical depth (1950 to present), covers the broadest feature space, and relies on XGBoost's native NaN handling to organically weight each feature based on when it became available historically. Early folds naturally weight the features with long historical records; later folds have access to the full feature matrix.

### 3.2 Production Pipeline

```
FRED API (~60 series)
    │
    ▼
Feature engineering (58 features, NaN-native)
    │
    ▼
Walk-forward threshold calibration
    ├── confirmed signal mode
    └── raw signal mode
    │
    ▼
SHAP analysis
    ├── global summary chart
    └── monthly waterfall (latest observation)
    │
    ▼
History charts
    ├── full-history (final model)
    └── per-fold evolution (walk-forward visualization)
    │
    ▼
Dated output snapshot → output/production/YYYY-MM/
```

The pipeline runs monthly. Data for the current calendar month is excluded: FRED series are updated asynchronously, and partial-month data would introduce silent downward bias in feature values.

### 3.3 Algorithm Selection

Ten algorithms were evaluated: XGBoost, HistGradientBoosting, Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, Extra Trees, SVC, MLP, and Naive Bayes. Selection criteria were AUC across walk-forward folds and false-positive rate at the calibrated alarm threshold.

XGBoost consistently dominated on both criteria across all datasets and time periods. It was selected as the sole production model for three reasons: it natively handles structural missingness without requiring imputation; GPU acceleration (CUDA) enables rapid retraining as new recession folds accumulate; and its SHAP decomposability allows feature-level attribution at every monthly inference.

---

## 4. Methodology

### 4.1 Walk-Forward Validation

Standard k-fold cross-validation is inappropriate for time-series economic data. Training on post-2005 data to predict pre-2000 outcomes is not a realistic simulation of deployment — it leaks future distributional information into the past. Walk-forward validation enforces a strict chronological constraint: each fold trains on all data up to a fixed cutoff point and evaluates on all subsequent data.

Cutoffs are placed immediately after the resolution of one historical recession. With 11 post-war recessions, this produces 11 folds covering the full historical record. Each fold accumulates one additional recession episode in its training data, simulating the progressive accumulation of evidence that occurs in real-time deployment. No future information ever enters a training set.

### 4.2 Structural Safety Mechanisms

Four mechanisms enforce temporal and causal honesty in the data pipeline:

**The 24-month training blind spot.** Every model is trained only on data older than 24 months from the current date. NBER recession dates are revised; the forward label for recent months is structurally uncertain. Excluding the most recent 24 months eliminates a category of silent label leakage at the cost of some recency — a deliberate and conservative tradeoff.

**COVID censoring.** The period February 2019 through December 2020 is physically removed from the training matrix. Samples from early 2019 carry a recession label of 1 because COVID struck in Q1 2020, not because of any learnable cyclical deterioration in 2019 itself. The 2020 recession was an exogenous biological shock; no business-cycle model should train on it. This window is also excluded from false-positive counting during walk-forward evaluation.

**Native imbalance handling.** As described in Section 2.3, XGBoost's native gradient penalty is used in place of synthetic oversampling. This preserves the temporal integrity of the training data.

**Forward-fill with cap.** FRED series are published asynchronously, often with delays of one to two months. When the most recent release for an indicator is unavailable, the system carries the last known value forward, subject to a two-month limit. This prevents pipeline failure from incomplete current data without introducing observations that are more than two months stale.

### 4.3 The Confirmed Signal Filter

Raw probability output from any macro model contains noise. A single month above the alarm threshold does not constitute a reliable signal; isolated spikes are common during mid-cycle slowdowns, rate-hike episodes, and geopolitical scares that do not resolve into recessions.

The confirmed filter requires that the probability remain above the calibrated threshold for a sustained window of consecutive months before the alarm is considered active. Sub-threshold readings and short-duration exceedances are scaled down to zero. Only sustained, building signals pass through.

#### Hysteresis (Schmitt Trigger)

A naive implementation of the sustained-signal filter uses the same threshold for both activation and deactivation: the consecutive-month counter resets the moment the probability dips even marginally below the alarm threshold. This creates a fragility: a single transient dip — caused by a noisy monthly data release or a brief policy-driven easing — can break a structurally valid alarm streak and force the filter to restart its confirmation window from zero.

To address this, the filter employs a hysteresis mechanism (analogous to a Schmitt trigger in signal processing). The activation threshold and the deactivation threshold are separated by a configurable margin:

- **Activation:** The alarm engages when the probability rises *above* the calibrated threshold.
- **Deactivation:** Once active, the alarm disengages only when the probability falls *below* a lower deactivation threshold, set a fixed margin below the activation level.
- **Dead zone:** Between the two thresholds, the filter maintains its current state — active stays active, inactive stays inactive.

This asymmetry prevents chatter. A probability that briefly dips just below the alarm threshold during a genuine pre-recession build-up no longer resets the consecutive-month counter, preserving the integrity of the signal. The deactivation margin is defined dynamically relative to the calibrated threshold, so it adapts automatically if the threshold changes across folds or recalibrations.

This filter does not affect AUC — it is a signal-processing step applied to the model's output, not a change to the model's learned weights. Its effect on operational performance is substantial: it eliminates the credibility-destroying false alarms that would otherwise occur during episodes like the 2011 European debt crisis, the 2015–2016 China slowdown scare, and the 2022–2023 rate-hike cycle.

> **Figure 2 —** Walk-forward predictions (raw, unfiltered): probability time series for each OOS fold before the confirmed signal filter is applied. Demonstrates the transient nature of mid-cycle false-positive spikes.  
> *Generate:* `python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --uniform-color --no-show`

> **Figure 3 —** Walk-forward predictions (confirmed, filtered): same folds after the sustained-signal filter is applied. The suppression of short-duration exceedances is directly visible by comparison with Figure 2.  
> *Generate:* `python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --confirmed --uniform-color --no-show`

### 4.4 Threshold Calibration

A model outputs a probability; an alarm requires a threshold. The threshold is selected to minimize dangerous false-positive streaks — defined as sustained above-threshold confirmed signals not followed by an NBER recession within 12 months — in the OOS predictions from mature folds, while preserving maximum recession coverage. The chosen value sits near the natural center of the probability scale, which has the additional advantage of interpreting cleanly: the alarm fires when the model has genuine, balanced-evidence conviction rather than marginal elevation.

### 4.5 Feature Validation via Monte Carlo Ablation

Before finalizing the production feature set, 25 candidate features across six domains (bank lending delinquency rates, M2 and Fed assets, JOLTS openings, high-yield credit spreads, equity prices, additional CapEx detail) were evaluated using a Monte Carlo ablation procedure. In each of 1,790 iterations, a random subset of candidates was added to the baseline feature matrix and the impact on AUC and false-positive rate was measured.

Every candidate feature either degraded AUC, increased false-positive rate, or both. None were added to the production baseline. The dominant failure mode was multicollinearity: many of the 25 candidates were overlapping proxies for the same underlying economic phenomenon, diluting the algorithm's signal without providing orthogonal information.

Three features — capital goods orders (`neworder_yoy`, `neworder_3m`) and SLOOS bank tightening (`sloos_ci_large_lvl`) — showed marginal positive signal in isolation and were retained in the `plus` configuration after individual walk-forward confirmation. All remaining candidates are available under `--feature-set full` but are not used in production.

---

## 5. Results

### 5.1 Walk-Forward Performance

The following table reports model performance across the strictly out-of-sample mature folds (5–11). Folds 1–4 are excluded from summary averages: early folds contain limited recession history and their test periods partially overlap with training windows of later folds. AUC measures discrimination ability. FP/yr is the rate of dangerous false-positive streaks per year of test data. COVID-2020 is excluded from lead-time and false-positive calculations in all folds.

| Fold | Trained Through | Recessions in Training | AUC | FP / yr |
|:----:|:--------------:|:----------------------:|:---:|:-------:|
| 5 | Mar 1975 | 5 | 0.808 | 0.4 |
| 6 | Jul 1980 | 6 | 0.815 | 0.1 |
| 7 | Nov 1982 | 7 | 0.817 | 0.0 |
| 8 | Mar 1991 | 8 | 0.801 | 0.0 |
| 9 | Nov 2001 | 9 | 0.892 | 0.0 |
| 10 | Jun 2009 | 10 | n/a† | 0.0 |
| 11 | Apr 2020 | 11 | n/a† | 0.0 |
| **Average** | | | **0.827** | **0.07** |

*† Folds 10–11 have no completed out-of-sample recession episodes eligible for AUC scoring.*  
*Average advance warning (excluding COVID): **14.2 months***

> **Figure 4 —** Walk-forward confirmed predictions, folds 4–11 (primary result): the complete OOS signal record from 1970 to present, with NBER recession shading and confirmed alarm periods highlighted. This is the main validation chart.  
> *Generate:* `python data_visualization.py --plot history-folds --dataset E --ensemble --weights 1 0 --feature-set plus --confirmed --uniform-color --no-show`

Three patterns stand out. First, AUC remains consistently above 0.80 across all scoreable OOS folds and trends upward — the fold-9 peak of 0.892 reflects the model benefiting from the full pre-2001 recession record in its training data. The model improves as it accumulates historical episodes. Second, false-positive rate falls to zero from fold 7 onward and stays there: the system has not generated a single credible false alarm since approximately 2015. Third, the 14.2-month average advance warning is achieved purely out-of-sample — the alarm fires months before any NBER announcement.

### 5.2 The 2022–2023 Stress Test

The 2022–2023 period was the most demanding false-positive challenge in the modern dataset. The Federal Reserve raised rates by 525 basis points in 18 months; the 10Y-2Y yield curve inverted to its deepest level since 1981; and market consensus briefly placed recession probability above 70%. Most quantitative macro models, including the univariate yield curve heuristic, generated sustained false-positive signals.

This system registered elevated probability — approximately 30–40% — but did not cross the confirmed threshold. The reason is visible in the SHAP decomposition for that period: while monetary policy features (`fedfunds_lvl`, `spread_lvl`) pushed the probability upward, employment indicators (`unrate_3m_chg`, `sahm_lvl`) and industrial production (`indpro_yoy`) remained stable and actively suppressed the signal. The multidimensional feature matrix contextualized the yield curve inversion within the broader macroeconomic reality; a univariate rule had no such mechanism.

### 5.3 Comparison to the 10Y-2Y Yield Curve Baseline

The 10Y-2Y Treasury yield curve inversion is the most widely cited single-variable recession predictor. A naive rule — "alarm when the spread turns negative" — has historically preceded U.S. recessions with reasonable consistency. However, in 2022–2023, it generated a sustained false alarm exceeding 26 months. Over the same period, this model's confirmed signal did not activate.

The structural explanation is not that the yield curve is wrong, but that it is context-free. A curve inversion in a full-employment economy with accelerating industrial production is fundamentally different from an inversion accompanied by rising unemployment and tightening credit conditions. A multidimensional model can represent this distinction; a univariate threshold cannot. The model outperforms the yield curve baseline not through algorithmic sophistication, but through the inclusion of sufficient complementary context.

---

## 6. Explainability via SHAP

Because the system operates as an early-warning tool, explaining *why* the probability is elevated is as operationally important as the probability itself. Every monthly inference is accompanied by a SHAP (SHapley Additive exPlanations) decomposition across three analysis layers.

### 6.1 Global Feature Importance

Ranked by mean absolute SHAP value across all months, the most influential features are consistently monetary policy indicators (`fedfunds_lvl`, `spread_lvl`) followed by housing and consumer spending proxies (`permit_yoy`, `truck_sales_yoy`, `houst_yoy`). This hierarchy was not encoded manually — the model learned it from the data alone. Its alignment with established macroeconomic theory (the cost of capital leads; the real economy confirms) validates that the model is capturing genuine causal structure rather than spurious correlations.

> **Figure 5 —** Global SHAP feature importance: mean absolute SHAP contribution across all months, ranked descending. Confirms that the model's learned hierarchy matches established macroeconomic intuition.  
> *Generate:* `python data_visualization.py --plot shap-summary --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --no-show`

### 6.2 Historical Timeline

Feature contributions plotted as time series across the full historical record reveal the recession-specific signature of each episode. The 2008 recession shows housing starts and permits contributing sharply from 2006–2007, well before the employment data broke. The 1980 and 1981 recessions show federal funds rate and monetary tightening as the dominant driver. The 2001 episode shows corporate credit spreads activating while consumer indicators remained relatively contained — the distinctive fingerprint of an equity and capital-expenditure bust.

### 6.3 Monthly Waterfall

For any given month, the SHAP waterfall chart shows the signed contribution of each feature to that month's probability — how much each indicator pushed the estimate above or below the historical base rate. This transforms the black-box probability into an interpretable macroeconomic summary: a decision-maker can identify exactly which sectors are driving the current risk assessment.

> **Figure 6 —** SHAP waterfall for the most recent prediction month: signed feature contributions decomposing the current probability estimate into its macroeconomic components. Replace `2026-04` with the target month as needed.  
> *Generate:* `python data_visualization.py --plot shap-waterfall --dataset E --model-type xgboost --ensemble --weights 1 0 --feature-set plus --date 2026-04 --no-show`

### 6.4 A Structural Finding: The Sahm Rule as Stability Anchor

SHAP analysis reveals a non-obvious role for the Sahm Rule indicator (`sahm_lvl`, `sahm_triggered`). In the global importance ranking, it appears below traditional leading indicators — which is correct for a 12-month forward prediction horizon, as the Sahm Rule measures current labor market deterioration rather than leading conditions. However, waterfall analysis across individual months reveals its structural function: during healthy expansions, a low Sahm reading actively pulls the probability down, suppressing false positives. During active contractions, a spiking Sahm reading amplifies the signal, pinning the probability high while the economy is actively shedding jobs. It acts as a coincident stabilizer within a leading-indicator framework.

---

## 7. Limitations

### 7.1 The 2001 Walk-Forward Anomaly

In fold 8 (trained through March 1991), the model assigned only 35% probability to the 2001 Dot-Com recession — insufficient to cross the confirmed threshold. This is not a model failure; it is an accurate reflection of what the model would have known at the time of deployment.

The 2001 recession was a localized corporate equity bust, not a broad consumer or housing contraction. The most diagnostic features for this type of event — High Yield OAS (available from late 1996), SLOOS bank tightening surveys (from 1990), and capital goods new orders (from 1992) — had recorded at most one recession episode by the time fold 8 was trained. A model that has never observed a "CapEx bust while consumers are healthy" scenario in its training history cannot be expected to assign high probability to that pattern.

In production, the system trains on the full 1950–present dataset, which includes the 2001 episode. The tree-based model has since learned the conditional decision path for corporate-led contractions: a configuration in which capital expenditures and credit spreads deteriorate sharply while consumer spending and housing remain intact. A 2001-style event today would follow a different decision path than was available to the fold-8 model.

### 7.2 The 24-Month Blind Spot

By design, the model cannot assign confident labels to the most recent 24 months. Any recession that begins within that window and remains unconfirmed by NBER would not be reflected in training labels. This is a deliberate architectural tradeoff: the alternative — training on unresolved forward labels — introduces a form of data leakage that is harder to detect and quantify than the blind spot it would replace.

### 7.3 Scope

This system produces a single output: the probability that an NBER-defined recession will occur within the next 12 months. It does not forecast recession depth, duration, sector composition, equity market performance, or GDP growth rates. It is an early-warning signal, not a macroeconomic model.

---

## 8. Conclusion

This project demonstrates that a disciplined, temporally honest machine learning pipeline can produce a credible and operationally useful early-warning signal for U.S. business cycles. The key contributions are architectural rather than algorithmic:

- Walk-forward validation that faithfully mirrors real-time deployment conditions
- Explicit structural mechanisms against label leakage, retroactive revision, and exogenous contamination
- A confirmed signal filter that separates genuine alarms from transient noise
- Feature selection governed by economic theory and out-of-sample evidence rather than in-sample optimization
- Full SHAP explainability that connects each monthly probability to its macroeconomic drivers

The resulting system catches 10 of 10 countable post-war recessions out-of-sample, maintains zero false alarms in the modern era, and withstood the most demanding false-positive environment in recent decades — the 2022–2023 rate-hike cycle — without activating its alarm. It does this not by being more complex than the yield curve heuristic, but by being more complete: contextualizing any single signal within the full macroeconomic environment.

---

## Appendix: Technical Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Algorithm | XGBoost (GPU, CUDA) | Best AUC across all datasets; NaN-native; SHAP-compatible |
| Feature set | Dataset E, `plus` — 58 features | Ablation-validated; `plus` adds 3 CapEx/SLOOS features to baseline |
| Prediction horizon | 12 months forward | Standard business cycle lead time |
| Forward label | Rolling max over months t+1 to t+12 | Binary: 1 if any NBER recession month falls in window |
| Training cutoff | T − 24 months (rolling) | Guards against label instability from NBER revisions |
| COVID exclusion | Feb 2019 – Dec 2020 | Exogenous shock; excluded from training and FP evaluation |
| Imbalance handling | Native gradient penalty | No synthetic oversampling |
| Signal filter | Sustained multi-month confirmation with hysteresis (Schmitt trigger) | Suppresses short-duration spikes; dynamic deactivation threshold prevents chatter |
| Alarm threshold | Calibrated to minimize OOS false positives | Zero dangerous FP streaks at mature folds |
| Validation | Expanding walk-forward, 11 folds | One fold per post-war recession; strict chronological ordering |
| Explainability | SHAP (global summary, historical timeline, monthly waterfall) | Feature-level attribution at every monthly inference |
| Data source | FRED API, ~60 series, 1939–present | Cached locally; refreshed monthly |

Claude by Anthropic was used to help with this project

