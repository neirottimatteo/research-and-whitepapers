# Recession Probability Engine

An ML system that estimates, every month, the probability of a U.S. recession within the next 12 months. It is built for temporal honesty and a low false-alarm rate.

### Latest: Version 4.0, public edition (October 2026)
👉 [**Technical white paper, V4.0 (PDF)**](./V4.0/Recession%20Probability%20Engine%20v4.0.pdf) · [HTML](./V4.0/Recession%20Probability%20Engine%20v4.0.html)

V4.0 is the public edition of version 4: its main results, and what each part of the method guarantees. The exact rules and settings are in the full technical paper, available from the author on request.

### System overview
- **Model:** a single XGBoost classifier on public FRED data, 1950 – present.
- **Validation:** walk-forward, one test period per post-war recession. No label is used before its 12-month outcome is known.
- **Signal:** a smoothed monthly probability, with the alarm at 50%.
- **Explainability:** a SHAP decomposition for every monthly reading.

### Out-of-sample record (published series, 1971 – 2025)
- **Recessions caught:** 6 of 6, with an average of 11.3 months of advance warning.
- **False alarms:** zero months above the alarm line without a recession ahead since 1985, and 10 over the whole record.
- **Discrimination:** ROC-AUC 0.948 · PR-AUC 0.813.
- **Benchmark:** the classic yield-curve probit (Wright, 2006), run through the identical walk-forward, scores ROC-AUC 0.840 with 28 false-alarm months, although it warns earlier before 2001 and 2008.
