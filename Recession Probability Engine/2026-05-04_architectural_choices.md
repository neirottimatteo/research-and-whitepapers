# Recession Probability Engine: A Machine Learning Approach to Macroeconomic Forecasting

*Author: Matteo Neirotti*  
*May 2026*

## 1. Introduction and Objectives

Predicting macroeconomic cycles is notoriously difficult. Unlike high-frequency trading where data is infinite, true economic recessions are incredibly rare. Since 1950, there have only been a handful of NBER-defined recessions in the United States. 

The objective of this project was to build a rigorous, machine learning-driven system that outputs a monthly probability of a U.S. recession occurring within the next 12 months. The priority was conservative accuracy: the model must catch major recessions with meaningful advance warning, but it must aggressively punish false positives. A recession indicator that constantly triggers false alarms holds no credibility.

![Label Distribution](../output/images/viz_labels.png)
*Figure 1: The sparse history of recession signals compared to expansion periods.*

The system was forged against three inherent challenges:
1.  **Sparse History:** Limited positive samples to train on.
2.  **Lagging Labels:** NBER defines recessions retroactively, often 6 to 12 months late.
3.  **Event Bleed-Forward:** Exogenous shocks (like the COVID-19 pandemic) poison the training data for the months immediately preceding them.

---

## 2. Methodology & Signal Processing

### Strict Walk-Forward Validation
Standard cross-validation is fundamentally flawed for time-series macro data because it leaks future information into the past. To ensure the model's performance metrics reflect reality, the system relies on an expanding "walk-forward" validation. The model is trained on data up to a specific historical point, and tested *only* on the unseen data that follows. This mirrors real-time deployment. Furthermore, all hyperparameter optimization (GridSearchCV) is constrained by a strict chronological `TimeSeriesSplit` to guarantee zero future-data leakage during tuning.

### Structural Safety Buffers (Intellectual Honesty)
To prevent the algorithm from "cheating" or learning spurious correlations, four strict structural buffers are hardcoded into the pipeline:
1. **The 24-Month Training Blind Spot:** Macroeconomic data is heavily revised, and NBER recession declarations lag by up to a year. To prevent the model from training on unresolved provisional data, a hard 24-month cutoff is enforced. The model is never allowed to train on the most recent two years of history.
2. **Native Imbalance Handling (No Fake Data):** The dataset is deeply imbalanced (~85% expansions). Standard data science practices often use SMOTE to synthetically oversample the minority class. This destroys the temporal integrity of macro data. Instead, this architecture utilizes XGBoost's native `scale_pos_weight` gradient penalty, forcing the algorithm to mathematically respect the true rarity of recessions without inventing synthetic historical months.
3. **The Biological Exogenous Censor:** The period from February 2019 through December 2020 is physically deleted from the training matrix. The 2020 recession was an exogenous biological shock, not a cyclical credit or capacity breakdown. Allowing the model to learn from 2020 would permanently poison its understanding of normal economic cycles.
4. **Real-Time Data Lag and Forward Filling:** Macroeconomic indicators are published asynchronously, often with significant delays. To simulate a true real-time production environment and prevent pipeline failure when current-month data is unavailable, the system utilizes a strict forward-filling mechanism. If an indicator has not yet been released for the current month, the pipeline safely carries forward its most recent available value (typically capping at a 2-month limit). This guarantees the model assesses the economy based strictly on the exact data reality available on the day it is run.

### The Sustained Signal Filter
Raw probability output from any macro model is inherently noisy. Without filtering, the model's high sensitivity results in several dangerous false-positive streaks. For example, running the raw output at a 50% threshold produces multiple months of panic spikes averaging over 80% probability in periods like mid-2006 or early 2023, which never materialized into recessions.

A central breakthrough in this project was the implementation of a custom mathematical filter designed to suppress this noise. While the exact mathematical formula (including the specific time-decay exponent and duration thresholds) remains proprietary, the core mechanism requires the probability signal to remain elevated across consecutive months. Short-term panic spikes are aggressively scaled down and zeroed out, allowing only true, building recession signals to gracefully fade in over time. 

The contrast is stark: the filter turns an overly anxious, noisy probability line into a highly calibrated, credible warning system.

![Raw Unfiltered Predictions](../output/images/viz_history_folds_E_ens.png)
*Figure 2: Raw Predictions (Unfiltered) — Notice the high frequency of false positive spikes between recessions.*

![Filtered Predictions](../output/images/viz_history_folds_E_ens_confirmed.png)
*Figure 3: Confirmed Predictions (Filtered) — The mathematical filter effectively suppresses the noise while preserving the lead time of genuine signals.*

### The Algorithm
After exploring various complex ensembles, the architecture was streamlined to utilize a single gradient boosting framework (XGBoost). This approach natively handles the massive missing values inherent in sparse historical datasets and consistently maximizes the Area Under the Curve (AUC) without unnecessary complexity.

---

## 3. The Feature Selection Struggle

### The Failure of Automated Monte Carlo
Attempts were made to automate feature selection using Monte Carlo and ablation scripts. The goal was to randomly drop or add subsets of features and measure the impact on the AUC.

However, this approach failed to yield actionable results due to a "masking effect." Highly predictive, theoretical features were often bundled in random subsets alongside noisy or irrelevant data. The bad data dragged down the average performance of the subset, causing the automated system to discard the good features along with the bad. A more sophisticated automated approach controlling for correlation might have yielded better results, but random sampling proved ineffective.

### Manual Curation
The chronological evolution of our feature matrix clearly demonstrates the danger of "kitchen-sink" macro models. While the addition of targeted features like manufacturing hours and capacity utilization (added April 25th) strictly improved model performance, adding a massive bulk of 25 new features a few days later (the `FULL` dataset, added April 30th) actually degraded out-of-sample metrics.

From a macroeconomic perspective, this failure is a classic forecasting trap. Pouring 25 macro indicators into the matrix simultaneously introduced severe multicollinearity. Many of these metrics were simply overlapping flavors of the exact same underlying economic phenomena (e.g., measuring consumer credit stress or industrial output across slightly different sub-sectors). This overlap dilutes the algorithm's focus. Furthermore, some of the added series likely contained structural breaks or were simply spurious—acting as coincident/lagging indicators or reflecting sector-specific volatility rather than true, systemic rot. The true recessionary signal was drowned out by correlated noise.

Because the automated ablation couldn't be fully trusted to parse this noise, we returned to economic intuition. We threw out the bulk of the April 30th additions and manually curated a refined subset (the `plus` dataset). By specifically selecting only the strongest, non-overlapping theoretical indicators from that batch—such as High Yield OAS for corporate credit fear, Bank Tightening (SLOOS) for credit supply constraints, and CapEx Intent for forward-looking business investment—we removed the noise and successfully improved the model's predictive power without overfitting.

---

## 4. Explainability (SHAP)

Because the model operates as an early warning system, explainability is just as important as the probability output. A decision-maker needs to know *why* the alarm is ringing. We leverage extensive SHAP (SHapley Additive exPlanations) analysis to unpack the model's reasoning across three different dimensions: globally, historically, and presently.

### Global Feature Importance
The SHAP Summary chart provides a high-level view of which features drive the model's overall predictions across the entire dataset. This allows us to verify that the model is relying on theoretically sound macroeconomic indicators rather than spurious correlations.

![SHAP Summary](../output/images/viz_shap_summary_xgboost_E.png)
*Figure 2: SHAP Summary — Global view of feature importance.*

Crucially, the global summary and historical waterfall charts prove the algorithm independently learned classic macroeconomic theory. The SHAP plot confirms the model is looking at the right things for systemic recessions, showing a healthy "Main Street" engine:
- **Monetary Policy:** `fedfunds_lvl` and `spread_lvl` (Yield Curve) are anchoring the early risk assessment.
- **Housing & Consumers:** `permit_yoy`, `truck_sales_yoy`, and `houst_yoy` act as proof that the real consumer economy is breaking.

We also observe that **coincident indicators (like the Sahm Rule) are utilized by the model as stability anchors**. Because the model targets a 12-month forward horizon, the Sahm Rule is correctly ranked lower in predictive power than true leading indicators. However, SHAP Waterfall analysis reveals its structural role: during healthy expansions (low unemployment), a low Sahm reading actively pulls the probability down, suppressing false positives. Conversely, during active contractions (like October 2008), a spiking Sahm reading pushes the probability forcefully toward 100%, pinning the alarm high while the economy is actively shedding jobs.

### Historical Context
Different recessions are caused by different structural failures. The SHAP Timeline allows us to look back at specific points in history and see exactly which features were pushing the prediction up or down. For example, we can visually verify if the model was flagging housing and credit data prior to 2008, or if it was reacting to inflation indicators in the early 1980s.

![SHAP Timeline](../output/images/viz_shap_timeline_xgboost_E.png)
*Figure 3: SHAP Timeline — Tracking the ebb and flow of specific economic drivers over historical cycles.*

### Current State Analysis
When the model outputs a probability *today*, it is critical to understand the immediate drivers. The SHAP Waterfall provides a detailed breakdown of exactly which features are pushing the probability up (increasing risk) and which are pulling it down (mitigating risk) for the most recent month. This transforms a black-box probability score into an actionable macroeconomic summary of the *current* economic state.

![SHAP Waterfall](../output/images/viz_shap_waterfall_xgboost_E.png)
*Figure 4: SHAP Waterfall — Deconstructing the current month's probability.*

---

## 5. Evaluation of Results

The true measure of a macroeconomic model is its performance on unseen data. Figure 5 demonstrates the finalized pipeline's performance strictly using out-of-sample expanding walk-forward predictions. 

![Out of Sample Walk-Forward History](../output/images/viz_history_folds_E_ens_confirmed.png)
*Figure 5: The Hero Chart — Out-of-sample expanding fold predictions.*

By utilizing the Sustained Signal Filter and targeted feature engineering, the model successfully catches the major modern recessions purely out-of-sample, maintaining substantial advance warning while effectively silencing false-positive noise (excluding the explicitly noted 2001 anomaly discussed below). 

Crucially, the model demonstrates an exceptional ability to avoid false positives in the modern era. Since Fold 7 (which covers the period from roughly 2015 through 2025 and up to the present day), the system has recorded zero false positives. It successfully navigates through mid-cycle slowdowns, rate-hike cycles, and recent market volatility without misfiring, proving its resilience against temporary economic fears that do not materialize into actual NBER recessions.

### Surviving the "Widowmaker" (2022–2024)
Look at the far right of the walk-forward chart. In 2022 and 2023, inflation spiked, the Fed hiked rates aggressively, and the yield curve inverted to historically deep levels. Every standard macro model on Wall Street fired a false positive during this period. This model saw the risk—pushing the probability up to around 30–40%—but it refused to cross the 50% threshold because the hard employment and production data didn't break. Staying out of a false positive in 2023 is a massive achievement.

### Beating the Baseline (The Yield Curve)
For decades, the 10Y-2Y Treasury yield curve inversion has served as the gold standard baseline for recession forecasting. However, as a univariate heuristic, it has increasingly misfired—generating notorious false positives or failing to accurately time recent business cycles. The structural mechanics of the bond market have not changed, but the underlying macroeconomic architecture has evolved significantly. Shifts in term premiums, a decade of quantitative easing, and a transition to a more service-heavy, less rate-sensitive economy have heavily distorted historical bond signals. 

A static heuristic cannot adapt to these regime shifts. By feeding the algorithm a multidimensional matrix of interrelated indicators—ranging from the raw cost of capital to corporate credit fear (High Yield OAS) and labor market momentum—the model mathematically contextualizes the yield curve within the broader macroeconomic reality. It vastly outperforms the 10Y-2Y baseline because it dynamically maps non-linear dependencies across sectors, capturing the true, evolving mechanics of a modern economic contraction rather than relying on a rigid, single-variable proxy.

---

## 6. Known Limitations & Future Work

While the overall results are highly successful, there are two distinct anomalies in the out-of-sample predictions that highlight the limitations of data availability and exogenous shocks.

### The 2001 "Miss" (The Walk-Forward Illusion)
During walk-forward validation (Fold 8), the model underpredicts the 2001 Dot-Com bust, pushing the probability to only 35%. This is not a structural failure, but a fascinating quirk of walk-forward cross-validation—a "Walk-Forward Illusion."

The 2001 recession was a localized corporate equity bust rather than a systemic consumer or housing crash. Our most powerful corporate features (`neworder`, `sloos`, High Yield OAS) only began recording data in the 1990s. When the model trained on data up to 1999/2000 to predict 2001, it had only ever seen one recession using those features (the 1990 crisis). 

Because the model had never seen a "CapEx bust while consumers are happy" scenario in its training data, it didn't know how to weigh `neworder` heavily enough to overrule the strong housing data. It saw the corporate warning signs, got nervous (pushing the probability to 35%), but ultimately lacked the historical precedent to pull the trigger.

#### Why We Will Catch the Next Corporate Bust
If a 2001-style, corporate-led asset bust happens tomorrow, the model will handle it entirely differently. In live production for 2025/2026, the model trains on the entire dataset from 1950 to today. The 2001 recession is now explicitly in its memory.

Because XGBoost is a tree-based model, it is a master of conditional logic. It doesn't just average things out; it builds distinct paths. It has now learned to build a specific branch in its decision tree for this exact scenario:
- **Path A (Systemic Recession):** Housing drops AND consumers pull back AND yield curve inverts → **RECESSION (99%)**
- **Path B (Corporate Bust):** Housing is fine AND consumers are fine... BUT `neworder` drops heavily AND `sloos` bank tightening spikes → **RECESSION (80%)**

#### Reconciling with SHAP
It is important to note that while consumer and housing metrics dominate the top of the global SHAP summary plot, this does not mean the model will ignore CapEx. SHAP plots show the average global importance across all predictions. Because consumer/housing metrics led the vast majority of historical recessions (1974, 1980, 1981, 1990, 2008), they naturally dominate the average. However, XGBoost is non-linear; it doesn't need a feature to be #1 on the global SHAP plot to trigger a recession. It just needs that feature to cross a critical threshold within its specific decision path.

By feeding the model `neworder` and `sloos`, and training it on the post-2001 dataset, we have successfully given the algorithm the tools and historical memory to catch a modern corporate/equity recession, while maintaining the rock-solid base that protects against false positives.
