# Dataset E: Master Feature Matrix Log

This document tracks the 85 engineered macroeconomic features that comprise Dataset E. The matrix utilizes NaN-native algorithms (XGBoost, HistGradientBoosting) to organically process historical indicators based on their chronological availability. 

Features are grouped by their economic domain, with the date of integration logged for each.

---

## 1. Core Macro & Labor
These indicators track the structural health of output and employment.

* **`indpro_yoy`**: Industrial Production Index (Year-over-Year % change) — *Initial*
* **`indpro_3m`**: Industrial Production Index (3-month % change) — *Initial*
* **`unrate_lvl`**: Unemployment Rate (Absolute Level) — *Initial*
* **`unrate_3m_chg`**: Unemployment Rate (3-month absolute change) — *Initial*
* **`unrate_12m_chg`**: Unemployment Rate (12-month absolute change) — *Initial*
* **`payems_yoy`**: Nonfarm Payrolls (Year-over-Year % change) — *Initial*
* **`payems_3m`**: Nonfarm Payrolls (3-month % change) — *Initial*
* **`mfg_hours_lvl`**: Average Weekly Manufacturing Hours (Absolute Level) — *Added: 2026-04-25*
* **`mfg_hours_3m_chg`**: Average Weekly Manufacturing Hours (3-month absolute change) — *Added: 2026-04-25*

## 2. Inflation & Prices
These indicators track producer and consumer cost pressures, which heavily influence central bank policy.

* **`cpi_yoy`**: Consumer Price Index (Year-over-Year % change) — *Initial*
* **`cpi_3m_chg`**: Consumer Price Index (3-month acceleration/2nd derivative) — *Initial*
* **`ppi_yoy`**: Producer Price Index for All Commodities (Year-over-Year % change) — *Added: 2026-04-25*
* **`ppi_3m`**: Producer Price Index for All Commodities (3-month % change) — *Added: 2026-04-25*
* **`oil_shock`**: WTI Crude Oil (% deviation from 24-month rolling mean) — *Added: 2026-04-25*

## 3. Housing & Consumption
These indicators track consumer behavior, large-ticket purchases, and the highly cyclical housing market.

* **`houst_yoy`**: Housing Starts (Year-over-Year % change) — *Initial*
* **`houst_3m`**: Housing Starts (3-month % change) — *Initial*
* **`pce_yoy`**: Real Personal Consumption Expenditures (Year-over-Year % change) — *Added: 2026-04-14*
* **`pce_3m`**: Real Personal Consumption Expenditures (3-month % change) — *Added: 2026-04-14*
* **`truck_sales_yoy`**: Heavy Truck Sales (Year-over-Year % change) — *Added: 2026-04-14*
* **`mfg_sales_yoy`**: Real Manufacturing and Trade Industries Sales (Year-over-Year % change) — *Added: 2026-04-25*
* **`permit_yoy`**: Building Permits (Year-over-Year % change) — *Added: 2026-04-25*
* **`permit_3m`**: Building Permits (3-month % change) — *Added: 2026-04-25*
* **`sentiment_lvl`**: U of Michigan Consumer Sentiment (Absolute Level) — *Added: 2026-04-25*
* **`sentiment_yoy`**: U of Michigan Consumer Sentiment (Year-over-Year % change) — *Added: 2026-04-25*
* **`sentiment_3m_chg`**: U of Michigan Consumer Sentiment (3-month absolute change) — *Added: 2026-04-25*

## 4. Yield Curve & Central Bank
These indicators capture the cost of capital and the bond market's expectations for future growth versus current Federal Reserve policy.

* **`spread_lvl`**: 10Y-2Y Treasury Yield Spread (Absolute Level) — *Initial*
* **`spread_inverted`**: 10Y-2Y Treasury Yield Spread (Binary inversion flag) — *Initial*
* **`spread_3m_chg`**: 10Y-2Y Treasury Yield Spread (3-month absolute change) — *Initial*
* **`fedfunds_lvl`**: Federal Funds Effective Rate (Absolute Level) — *Initial*
* **`fedfunds_12m_chg`**: Federal Funds Effective Rate (12-month absolute change) — *Initial*

## 5. Credit & Corporate Stress
These indicators track systemic liquidity, the willingness of banks to lend, and the risk premium demanded for corporate debt.

* **`baa_aaa_spread`**: Moody's Baa minus Aaa Corporate Bond Yield (Absolute Level) — *Added: 2026-04-12*
* **`baa_aaa_spread_3m_chg`**: Moody's Baa minus Aaa Corporate Bond Yield (3-month absolute change) — *Added: 2026-04-12*
* **`credit_spread_lvl`**: Moody's Baa minus 10Y Treasury Spread (Absolute Level) — *Added: 2026-04-12*
* **`credit_spread_3m_chg`**: Moody's Baa minus 10Y Treasury Spread (3-month absolute change) — *Added: 2026-04-12*
* **`credit_spread_high`**: Moody's Baa minus 10Y Treasury Spread (Binary high-stress flag >3.5%) — *Added: 2026-04-12*
* **`busloans_3m`**: Commercial and Industrial Loans (3-month % change) — *Added: 2026-04-14*
* **`busloans_yoy`**: Commercial and Industrial Loans (Year-over-Year % change) — *Added: 2026-04-14*
* **`bank_credit_accel`**: Total Bank Credit (3-month acceleration/2nd derivative) — *Added: 2026-04-14*
* **`consumer_credit_accel`**: Total Consumer Credit (Year-over-Year acceleration/2nd derivative) — *Added: 2026-04-14*
* **`corp_profits_yoy`**: Corporate Profits After Tax (Year-over-Year % change) — *Added: 2026-04-25*

## 6. Market Volatility & Leading Indicators
These indicators act as the "tip of the spear," capturing real-time shifts in financial stress, industrial capacity limits, and sudden labor market cracks.

* **`icsa_yoy`**: Initial Jobless Claims (Year-over-Year % change) — *Added: 2026-04-18*
* **`icsa_3m`**: Initial Jobless Claims (3-month % change) — *Added: 2026-04-18*
* **`ccsa_yoy`**: Continued Jobless Claims (Year-over-Year % change) — *Added: 2026-04-18*
* **`ccsa_3m`**: Continued Jobless Claims (3-month % change) — *Added: 2026-04-18*
* **`parttime_3m`**: Part-Time Employment for Economic Reasons (3-month % change) — *Added: 2026-04-18*
* **`parttime_6m`**: Part-Time Employment for Economic Reasons (6-month % change) — *Added: 2026-04-18*
* **`sahm_lvl`**: Sahm Rule Indicator (Absolute Level) — *Added: 2026-04-18*
* **`sahm_triggered`**: Sahm Rule Indicator (Binary trigger flag >=0.5) — *Added: 2026-04-18*
* **`vix_lvl`**: CBOE Volatility Index (Absolute Level) — *Added: 2026-04-18*
* **`vix_spike`**: CBOE Volatility Index (Binary spike flag >30) — *Added: 2026-04-18*
* **`vix_3m_chg`**: CBOE Volatility Index (3-month absolute change) — *Added: 2026-04-18*
* **`tcu_lvl`**: Capacity Utilization (Absolute Level) — *Added: 2026-04-25*
* **`tcu_yoy`**: Capacity Utilization (Year-over-Year % change) — *Added: 2026-04-25*
* **`tcu_stalling`**: Capacity Utilization (Binary stalling flag) — *Added: 2026-04-25*

## 7. Credit Supply — Bank Lending Standards (SLOOS)
The Senior Loan Officer Opinion Survey captures *willingness* to lend, which leads actual loan volumes by 12–18 months. Unlike loan-volume indicators, these reflect decisions before they appear in the data. A value above 0 means net tightening; above 25 is historically recession-predictive.

* **`sloos_ci_large_lvl`**: Net % of Banks Tightening C&I Standards for Large/Mid Firms (Absolute Level) — *Added: 2026-04-30* **[PLUS]**
* **`sloos_ci_small_lvl`**: Net % of Banks Tightening C&I Standards for Small Firms (Absolute Level) — *Added: 2026-04-30*
* **`sloos_tightening`**: SLOOS (Binary flag: any net tightening, >0) — *Added: 2026-04-30*
* **`sloos_stress`**: SLOOS (Binary flag: historically significant stress, >25) — *Added: 2026-04-30*

## 8. Consumer & Business Delinquency Rates
Delinquencies are the middle ground between consumer sentiment and unemployment — they show financial stress while borrowers are still technically employed. Quarterly, forward-filled to monthly.

* **`delq_credit_card_lvl`**: Delinquency Rate on Credit Card Loans, All Commercial Banks (Absolute Level) — *Added: 2026-04-30*
* **`delq_credit_card_yoy`**: Delinquency Rate on Credit Card Loans (Year-over-Year % change) — *Added: 2026-04-30*
* **`delq_ci_lvl`**: Delinquency Rate on C&I Business Loans, All Commercial Banks (Absolute Level) — *Added: 2026-04-30*
* **`delq_mortgage_lvl`**: Delinquency Rate on Single-Family Residential Mortgages (Absolute Level) — *Added: 2026-04-30*
* **`delq_mortgage_yoy`**: Delinquency Rate on Single-Family Residential Mortgages (Year-over-Year % change) — *Added: 2026-04-30*

## 9. Structural Liquidity & Monetary Momentum
These capture the volume and flow of money in the system, complementing the cost-of-money indicators already present (FEDFUNDS, yield curve).

* **`m2real_yoy`**: Real M2 Money Stock (Year-over-Year % change) — *Added: 2026-04-30*
* **`m2real_3m`**: Real M2 Money Stock (3-month % change) — *Added: 2026-04-30*
* **`fed_assets_yoy`**: Federal Reserve Total Assets / QE-QT proxy (Year-over-Year % change) — *Added: 2026-04-30*
* **`reallnresn_yoy`**: Real Estate Residential Loans, All Commercial Banks (Year-over-Year % change) — *Added: 2026-04-30*

## 10. Forward-Looking Business Investment (CapEx Intent)
These indicators measure what businesses are *committing to* for the next 12–18 months, rather than current output. Nondefense capital goods orders are the gold standard for private investment intent.

* **`neworder_yoy`**: Manufacturers' New Orders: Nondefense Capital Goods Excl. Aircraft (Year-over-Year % change) — *Added: 2026-04-30* **[PLUS]**
* **`neworder_3m`**: Manufacturers' New Orders: Nondefense Capital Goods Excl. Aircraft (3-month % change) — *Added: 2026-04-30* **[PLUS]**
* **`jolts_yoy`**: JOLTS Job Openings (Year-over-Year % change) — *Added: 2026-04-30*
* **`overtime_lvl`**: Average Weekly Overtime Hours of Production Workers (Absolute Level) — *Added: 2026-04-30*
* **`overtime_3m_chg`**: Average Weekly Overtime Hours of Production Workers (3-month absolute change) — *Added: 2026-04-30*

## 11. High-Stress Financial Spreads
These catch shadow-banking and cross-asset stress that the Baa/Aaa corporate spread often misses. The high-yield OAS is the primary "fear gauge" for corporate credit. The 10Y-3M spread is preferred by many economists over 10Y-2Y for recession prediction.

* **`hy_spread_lvl`**: ICE BofA US High Yield Index Option-Adjusted Spread (Absolute Level) — *Added: 2026-04-30* 
* **`hy_spread_3m_chg`**: ICE BofA US High Yield Index OAS (3-month absolute change) — *Added: 2026-04-30*
* **`hy_spread_high`**: ICE BofA US High Yield Index OAS (Binary high-stress flag >500 bps) — *Added: 2026-04-30*
* **`nfci_lvl`**: Chicago Fed National Financial Conditions Index (Absolute Level, aggregates 105 indicators) — *Added: 2026-04-30*
* **`nfci_3m_chg`**: Chicago Fed National Financial Conditions Index (3-month absolute change) — *Added: 2026-04-30*
* **`t10y3m_lvl`**: 10Y Treasury minus 3-Month T-Bill Yield Spread (Absolute Level) — *Added: 2026-04-30*
* **`t10y3m_inverted`**: 10Y Treasury minus 3-Month T-Bill Spread (Binary inversion flag <0) — *Added: 2026-04-30*

## 12. CapEx Detail, Temp Labor, Equity Wealth & BAA10Y Explicit
These features sharpen the 2001-style bust signal: business equipment production collapses before the broader IP index, temp labor is the marginal hire/fire lever that turns before payrolls, equity wealth links to consumption and confidence, and BAA10Y gives a cleaner corporate spread name for explicit ablation. Note: `baa10y_lvl` / `baa10y_3m_chg` are aliases of `credit_spread_lvl` / `credit_spread_3m_chg` (same FRED series `BAA10Y`) — retained as separate named columns for ablation control.

* **`ipbuseq_yoy`**: Industrial Production: Business Equipment (Year-over-Year % change) — *Added: 2026-05-05* 
* **`ipbuseq_3m_chg`**: Industrial Production: Business Equipment (3-month % change) — *Added: 2026-05-05* 
* **`temphelps_yoy`**: All Employees: Temporary Help Services (Year-over-Year % change) — *Added: 2026-05-05* 
* **`temphelps_3m`**: All Employees: Temporary Help Services (3-month % change) — *Added: 2026-05-05* 
* **`baa10y_lvl`**: Moody's Baa minus 10Y Treasury Spread — explicit alias of `credit_spread_lvl` (Absolute Level) — *Added: 2026-05-05* 
* **`baa10y_3m_chg`**: Moody's Baa minus 10Y Treasury Spread — explicit alias of `credit_spread_3m_chg` (3-month absolute change) — *Added: 2026-05-05* 
* **`sp500_yoy`**: S&P 500 Stock Price Index (Year-over-Year % change) — *Added: 2026-05-05* 