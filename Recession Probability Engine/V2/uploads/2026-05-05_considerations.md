# May 5, 2026 — Feature Additions: CapEx Detail, Temp Labor, Equity Wealth

Seven new features were added on May 5 to sharpen the model's ability to detect 2001-style corporate-led recessions. These features target the gap identified in the 2001 recession study: when a CapEx bust unfolds while consumers remain healthy, the current production features (which are weighted toward housing and broad labor market data) provide weaker signal.

All seven features are currently available under `--feature-set full`. None have been promoted to the production `plus` baseline pending walk-forward validation.

---

## Features Added

### Business Equipment Production
* **`ipbuseq_yoy`** — Industrial Production: Business Equipment (Year-over-Year % change)  
* **`ipbuseq_3m_chg`** — Industrial Production: Business Equipment (3-month % change)

**Rationale:** Business equipment production is a sub-index of INDPRO that leads the broader industrial production index during capital-expenditure-driven slowdowns. In a classic CapEx bust (e.g., 2001), business equipment collapses before total IP breaks. INDPRO is already in the baseline; this adds a more sensitive leading component.

### Temporary Help Employment
* **`temphelps_yoy`** — All Employees: Temporary Help Services (Year-over-Year % change)  
* **`temphelps_3m`** — All Employees: Temporary Help Services (3-month % change)

**Rationale:** Temporary help is the marginal hiring and firing lever in the labor market. Firms cut temp workers before reducing permanent headcount. Temp employment has historically turned negative 3–6 months before the broader nonfarm payroll series, making it a higher-frequency leading indicator within the labor domain.

### Equity Wealth Proxy
* **`sp500_yoy`** — S&P 500 Stock Price Index (Year-over-Year % change)

**Rationale:** Equity market performance links to consumer confidence and the wealth effect, and reflects forward-looking corporate earnings expectations. During corporate busts (2001, 2008), equity prices decline well in advance of broad economic contraction. Note: equity prices are partially coincident/forward-looking and carry significant noise; they were kept out of production for this reason.

### BAA10Y Explicit Alias (Ablation Control)
* **`baa10y_lvl`** — Moody's Baa minus 10Y Treasury Spread (Absolute Level) — *explicit alias of `credit_spread_lvl`*  
* **`baa10y_3m_chg`** — Moody's Baa minus 10Y Treasury Spread (3-month change) — *explicit alias of `credit_spread_3m_chg`*

**Rationale:** These are not new data — they are the same FRED series `BAA10Y` already present in the baseline as `credit_spread_lvl` and `credit_spread_3m_chg`. Retained as separately named columns so that ablation runs can independently test whether the model benefits from seeing this spread under two different feature names. This is a methodological control, not an informational addition.

---

## Status

| Feature | Status | Notes |
|---------|--------|-------|
| `ipbuseq_yoy` | `--feature-set full` | Pending walk-forward validation |
| `ipbuseq_3m_chg` | `--feature-set full` | Pending walk-forward validation |
| `temphelps_yoy` | `--feature-set full` | Pending walk-forward validation |
| `temphelps_3m` | `--feature-set full` | Pending walk-forward validation |
| `sp500_yoy` | `--feature-set full` | High noise; not expected to pass FP criterion |
| `baa10y_lvl` | `--feature-set full` | Ablation alias only |
| `baa10y_3m_chg` | `--feature-set full` | Ablation alias only |
