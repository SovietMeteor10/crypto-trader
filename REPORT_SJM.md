# SJM Regime Filter Experiment Report

Date: 2026-03-21 20:15

## Summary

| Variant | Train Sharpe | Val Sharpe | WF % Positive | Holdout Sharpe | Conclusion |
|---------|-------------|-----------|---------------|---------------|------------|
| SJM-V1: BTC Feature Set A, 3 regimes | 1.8775 | 1.073 | 57.1 | not run | FAIL: WF too low |
| SJM-V2: BTC Feature Set B (funding), 3 regimes | 2.3008 | 1.0444 | 57.1 | not run | FAIL: overfit |
| SJM-V3: SOL Feature Set A, 3 regimes | 1.7705 | 2.0036 | 60.7 | 0.9065 | FAIL: flags ['long_drawdown'] |
| SJM-V4: BTC Feature Set A, 2 regimes (bull/bear) | 2.8717 | 1.3891 | 60.7 | not run | FAIL: overfit |
| SJM-V5: Joint optimisation, BTC Feature Set A, 3 regimes | 2.0031 | 0.9406 | 50.0 | not run | FAIL: overfit |

## Variant Details

### SJM-V1: BTC Feature Set A, 3 regimes

- Feature set: A
- N regimes: 3
- Best params: {'sjm_lambda': 1.9255250932419297, 'sjm_window': 329, 'trade_in_neutral': True}
- Train Sharpe: 1.8775
- Val Sharpe: 1.073
- Overfit check: PASS
- WF % positive: 57.1
- WF mean Sharpe: 0.0504
- Holdout Sharpe: not run
- Conclusion: FAIL: WF too low

### SJM-V2: BTC Feature Set B (funding), 3 regimes

- Feature set: B
- N regimes: 3
- Best params: {'sjm_lambda': 3.6061385036954263, 'sjm_window': 299, 'trade_in_neutral': True}
- Train Sharpe: 2.3008
- Val Sharpe: 1.0444
- Overfit check: FAIL
- WF % positive: 57.1
- WF mean Sharpe: 0.6999
- Holdout Sharpe: not run
- Conclusion: FAIL: overfit

### SJM-V3: SOL Feature Set A, 3 regimes

- Feature set: A
- N regimes: 3
- Best params: {'sjm_lambda': 1.6573239546018446, 'sjm_window': 378, 'trade_in_neutral': True}
- Train Sharpe: 1.7705
- Val Sharpe: 2.0036
- Overfit check: PASS
- WF % positive: 60.7
- WF mean Sharpe: 1.4391
- Holdout Sharpe: 0.9065
- Holdout flags: {'overfit': np.False_, 'insufficient_trades': False, 'high_btc_corr': False, 'negative_skew': False, 'long_drawdown': True, 'consecutive_losses': False}
- Conclusion: FAIL: flags ['long_drawdown']

### SJM-V4: BTC Feature Set A, 2 regimes (bull/bear)

- Feature set: A
- N regimes: 2
- Best params: {'sjm_lambda': 5.428544151001011, 'sjm_window': 340, 'trade_in_neutral': False}
- Train Sharpe: 2.8717
- Val Sharpe: 1.3891
- Overfit check: FAIL
- WF % positive: 60.7
- WF mean Sharpe: 0.5789
- Holdout Sharpe: not run
- Conclusion: FAIL: overfit

### SJM-V5: Joint optimisation, BTC Feature Set A, 3 regimes

- Feature set: A
- N regimes: 3
- Best params: {'fast_period': 38, 'slow_period': 147, 'adx_period': 17, 'adx_threshold': 23, 'sjm_lambda': 4.9893012157580205, 'sjm_window': 675, 'trade_in_neutral': True}
- Train Sharpe: 2.0031
- Val Sharpe: 0.9406
- Overfit check: FAIL
- WF % positive: 50.0
- WF mean Sharpe: 0.7986
- Holdout Sharpe: not run
- Conclusion: FAIL: overfit

## Assessment

No variant passed all checks. The SJM regime filter does not solve the drawdown problem in its current form. Consider:

- Different feature engineering (macro data, on-chain metrics)
- Different signal (not trend-following)
- Different asset
- Accepting the drawdown duration flag as inherent to trend-following in crypto

## Steelman: Why Will This Fail in Live Trading?

1. Regime labels are assigned retrospectively — the SJM sees the full window before labelling
2. Greedy prediction (nearest centroid) may disagree with what a full re-fit would produce
3. Feature standardisation uses in-sample mean/std which shifts as new data arrives
4. The jump penalty lambda was optimised on historical data — future regime dynamics may differ
5. Crypto regime durations are non-stationary — 2021-2023 patterns may not repeat
