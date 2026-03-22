# V3 Improvement Experiments Report

Date: 2026-03-22 02:18

## Results Table

| Configuration | WF% | Val Sharpe | Holdout Sharpe | Max DD | All flags |
|--------------|-----|------------|----------------|--------|-----------|
| V3 baseline | 60.7 | 2.00 | 0.9065 | 3.22% | CLEAR |
| exp1_pullback | N/A | -0.5645 | N/A | N/A% | FAIL: overfit |
| exp2_trailing | 53.6 | 1.5431 | N/A | N/A% | FAIL: WF too low (53.6%) |
| exp3_portfolio | N/A | N/A | N/A | N/A% | FAIL: BTC SJM FAIL: WF too low (42.9%) |

## Best Configuration

**V3 baseline remains the best configuration.**
No experiment improved holdout Sharpe above V3 baseline (0.91).

## Realistic Monthly Return Expectations

At $1,000 capital (holdout Sharpe ~0.91):
- Expected monthly return: ~0.2% ($2/month)
- After fees and slippage: ~$1-2/month net
- This is NOT viable at $1,000 capital

At $25,000 funded account (same Sharpe):
- Expected monthly return: ~0.2% ($50/month)
- With proper sizing and 3x leverage: ~$100-150/month
- Viable but modest; requires consistent execution

## Steelman: Why Will This Fail in Live Trading?

1. SJM regime labels are retrospective — greedy prediction at live edge may disagree
2. The strategy's edge comes from 2021-2022 bull+bear cycle; future cycles may differ
3. Crypto market structure is changing (institutional adoption, ETFs, regulation)
4. Transaction costs may increase; funding rate dynamics may shift
5. 0.91 Sharpe is marginal — a few bad months could destroy confidence and cause early exit

## Deployment Decision

**CONDITIONAL — V3 is statistically valid but Sharpe is marginal.**
Recommend 3-month paper trade before any real capital.
Only viable at funded account scale ($25k+), not retail ($1k).
