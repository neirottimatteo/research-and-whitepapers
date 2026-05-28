### 01/05/2026
## drop


==================================================================================================================
  FINAL RESULTS — DROP MODE
==================================================================================================================

  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  DROP SCORES after 1000 iteration(s)   [score = avg(ΔAUC when PRESENT) − avg(ΔAUC when ABSENT)]
  Positive score = feature helps AUC when present -> KEEP
  Positive FP score = feature causes DangerFP when present -> REMOVE
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Feature                               n_in  n_out   XGB_AUC   HGB_AUC    AvgAUC   XGB_FP   HGB_FP    AvgFP  Verdict
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  pce_3m                                 813    187   +0.0070   +0.0097   +0.0083   -1.742   -1.472   -1.607  NEUTRAL
  sahm_lvl                               824    176   +0.0085   +0.0004   +0.0044   -0.695   -1.016   -0.855  NEUTRAL
  houst_3m                               832    168   +0.0016   +0.0030   +0.0023   +3.928   +1.296   +2.612  REMOVE
  sentiment_yoy                          800    200   +0.0006   +0.0036   +0.0021   -1.482   -1.091   -1.287  NEUTRAL
  sentiment_lvl                          801    199   +0.0041   -0.0019   +0.0011   +1.706   +0.471   +1.088  REMOVE
  pce_yoy                                807    193   -0.0001   +0.0019   +0.0009   +2.764   +1.728   +2.246  REMOVE
  permit_3m                              828    172   -0.0003   +0.0008   +0.0002   -2.011   -0.236   -1.123  NEUTRAL
  sahm_triggered                         792    208   -0.0001   -0.0006   -0.0003   -0.030   -0.012   -0.021  NEUTRAL
  sentiment_3m_chg                       814    186   +0.0003   -0.0010   -0.0004   -0.738   +0.534   -0.102  NEUTRAL
  mfg_sales_yoy                          827    173   -0.0014   -0.0002   -0.0008   -0.234   +0.753   +0.259  REMOVE
  truck_sales_yoy                        807    193   -0.0017   -0.0009   -0.0013   -0.523   +0.701   +0.089  REMOVE
  houst_yoy                              829    171   -0.0010   -0.0029   -0.0019   +1.204   +0.910   +1.057  REMOVE
  permit_yoy                             832    168   -0.0081   -0.0087   -0.0084   -2.253   -7.018   -4.636  NEUTRAL
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────

  VERDICT SUMMARY (drop mode)
  REMOVE  : 6  (presence causes DangerFP, not worth keeping)
    houst_3m                             AUC=+0.0023  FP=+2.612  (n_in=832)
    pce_yoy                              AUC=+0.0009  FP=+2.246  (n_in=807)
    sentiment_lvl                        AUC=+0.0011  FP=+1.088  (n_in=801)
    houst_yoy                            AUC=-0.0019  FP=+1.057  (n_in=829)
    mfg_sales_yoy                        AUC=-0.0008  FP=+0.259  (n_in=827)
    truck_sales_yoy                      AUC=-0.0013  FP=+0.089  (n_in=807)
  KEEP    : 0  (load-bearing for AUC, no FP cost)
  TRIM    : 0  (hurts AUC when present, no FP benefit — safe to remove)
  NEUTRAL : 7
  INSUFFICIENT: 0

  Candidates for removal from FEATURE_COLS_E: ['houst_yoy', 'houst_3m', 'pce_yoy', 'truck_sales_yoy', 'mfg_sales_yoy', 'sentiment_lvl']
  Verify by running walk_forward.py after removing them from data_finder.py.




  ## add

    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ADD SCORES after 1790 iteration(s)   [score = avg(ΔAUC when added) − avg(ΔAUC when not added)]
  Strict FP rule: EXCLUDE if avg ΔDangerFP > 0  (unless AUC gain ≥ 0.10)
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  Feature                               n_in  n_out   XGB_AUC   HGB_AUC    AvgAUC   XGB_FP   HGB_FP    AvgFP  Verdict
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  nfci_lvl                               329   1461   -0.0001   -0.0003   -0.0002   +0.924   +1.388   +1.156  EXCLUDE
  delq_mortgage_yoy                      296   1494   -0.0007   -0.0018   -0.0013   +1.433   +3.577   +2.505  EXCLUDE
  neworder_yoy                           309   1481   -0.0011   -0.0020   -0.0015   +0.984   +1.810   +1.397  EXCLUDE
  hy_spread_lvl                          315   1475   -0.0011   -0.0020   -0.0016   +0.946   +1.563   +1.254  EXCLUDE
  hy_spread_high                         309   1481   -0.0011   -0.0024   -0.0018   +0.897   +1.704   +1.301  EXCLUDE
  delq_mortgage_lvl                      285   1505   -0.0015   -0.0022   -0.0019   +3.952   +4.362   +4.157  EXCLUDE
  delq_ci_lvl                            293   1497   -0.0013   -0.0027   -0.0020   +1.210   +1.434   +1.322  EXCLUDE
  sloos_ci_small_lvl                     305   1485   -0.0014   -0.0027   -0.0021   +2.161   +4.731   +3.446  EXCLUDE
  overtime_lvl                           268   1522   -0.0017   -0.0026   -0.0021   +0.553   +1.214   +0.883  EXCLUDE
  fed_assets_yoy                         285   1505   -0.0014   -0.0029   -0.0021   +1.624   +2.175   +1.899  EXCLUDE
  sloos_stress                           292   1498   -0.0015   -0.0028   -0.0022   +1.274   +1.531   +1.403  EXCLUDE
  sloos_tightening                       306   1484   -0.0016   -0.0029   -0.0022   +1.304   +1.964   +1.634  EXCLUDE
  nfci_3m_chg                            301   1489   -0.0019   -0.0028   -0.0023   +1.042   +1.536   +1.289  EXCLUDE
  t10y3m_lvl                             300   1490   -0.0019   -0.0028   -0.0024   +1.225   +2.012   +1.618  EXCLUDE
  neworder_3m                            314   1476   -0.0018   -0.0029   -0.0024   +1.315   +1.690   +1.502  EXCLUDE
  delq_credit_card_lvl                   298   1492   -0.0018   -0.0033   -0.0025   +4.861   +6.406   +5.633  EXCLUDE
  delq_credit_card_yoy                   285   1505   -0.0017   -0.0035   -0.0026   +0.793   +2.067   +1.430  EXCLUDE
  reallnresn_yoy                         293   1497   -0.0017   -0.0036   -0.0027   +1.153   +1.879   +1.516  EXCLUDE
  overtime_3m_chg                        286   1504   -0.0019   -0.0035   -0.0027   +0.779   +1.137   +0.958  EXCLUDE
  t10y3m_inverted                        277   1513   -0.0020   -0.0036   -0.0028   +1.113   +2.046   +1.579  EXCLUDE
  jolts_yoy                              307   1483   -0.0023   -0.0040   -0.0031   +0.637   +1.622   +1.129  EXCLUDE
  hy_spread_3m_chg                       305   1485   -0.0023   -0.0042   -0.0032   +1.129   +1.834   +1.482  EXCLUDE
  sloos_ci_large_lvl                     274   1516   -0.0025   -0.0043   -0.0034   +1.797   +4.376   +3.087  EXCLUDE
  m2real_3m                              301   1489   -0.0041   -0.0074   -0.0058   +0.939   +1.364   +1.151  HARMFUL
  m2real_yoy                             326   1464   -0.0187   -0.0294   -0.0240   +1.847   +2.716   +2.282  HARMFUL
  ────────────────────────────────────────────────────────────────────────────────────────────────────────────────