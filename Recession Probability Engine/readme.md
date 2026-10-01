# Recession Probability Engine

An ML system that estimates, every month, the probability of a U.S. recession within the next 12 months. It is built for temporal honesty and a low false-alarm rate.

### Latest: Version 4.0 (October 2026)
👉 [**Technical white paper, V4 (PDF)**](./V4/Recession%20Probability%20Engine%20v4.pdf) · [HTML](./V4/Recession%20Probability%20Engine%20v4.html)

### System overview
- **Model:** a single XGBoost classifier on 57 engineered features from 47 FRED series, 1950 – present.
- **Validation:** 11-fold walk-forward, one fold per post-war recession. Every label is used only once its 12-month outcome is known, and each fold also learns from the first year of its recovery as those labels arrive.
- **Signal:** EMA-filtered probability, with the alarm at 50%.
- **Explainability:** a SHAP decomposition for every monthly reading.

### Out-of-sample record (published series, Dec 1971 – Oct 2025)
- **Recessions caught:** 6 of 6, with an average of 11.3 months of advance warning.
- **False alarms:** zero months above the alarm line without a recession ahead since 1985, and 10 over the whole record.
- **Discrimination:** ROC-AUC 0.948 · PR-AUC 0.813.
- **Benchmark:** the classic yield-curve probit (Wright, 2006), run through the identical walk-forward, scores ROC-AUC 0.840 with 28 false-alarm months, although it warns earlier before 2001 and 2008.

### Previous versions
- [V3](./V3/): production trained through the end of the last recession
- [V2.1](./V2.1/): 12-month fold embargo
- [V2](./V2/): EMA signal filter and publication-lag alignment
- [V1](./Recession_Probability_Engine.pdf): the original white paper
