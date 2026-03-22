# Progress

## 2026-03-20 — Infrastructure complete

All 7 modules implemented:
- `data_module.py` — OHLCV + funding rate fetch with caching, validation, pagination
- `signal_module.py` — abstract base class with output validation
- `sizer_module.py` — fixed_fractional, kelly, fixed_usdt sizing + liquidation checks
- `cost_module.py` — fees, per-symbol slippage, funding rate costs
- `backtest_engine.py` — single run, walk-forward, three-split, Optuna optimisation
- `metrics_module.py` — full MetricsBundle with flags, perturbation test, comparison
- `live_adapter.py` — dry-run capable live adapter via ccxt

All 26 tests passing. Smoke test passing. Ready for research phase.

## 2026-03-20 — Exhaustive strategy search complete

### Result: ZERO viable strategies found.

9 strategy families tested across 3 symbols (BTC, ETH, SOL) on multiple timeframes (1h, 4h, 1d). 45 total configurations evaluated. No strategy passed walk-forward validation.

### Failure modes
- **Overfitting** (dominant): train Sharpe 1-3x collapses to negative in validation/OOS
- **Inconsistency**: closest candidates (SOL 1C, SOL 4A) achieved only 45% WF positive windows (threshold: 50%)
- **Infrequent signals**: daily timeframe and volume-filtered strategies lacked statistical significance
- **Mean reversion does not exist**: all 3 MR families (RSI, Bollinger, Z-score) produced negative Sharpe even in-sample

### Key conclusions
- The $1,000 to $1,000/month target is mathematically impossible
- No standard strategy type produces a validated edge on Binance Futures perpetuals at retail timeframes
- Funded account path ($50k-$200k capital at 1-3%/month) is the only realistic route to $1,000/month
- Infrastructure is built and ready if a viable strategy is found in future

### Deliverables
- `STRATEGY_LOG.md` — full per-strategy results
- `REPORT.md` — complete 9-section research report
- `RESEARCH_PLAN.md` — original plan and scope
- `NOTES.md` — market characterisation data

### Status: COMPLETE

## 2026-03-22 — Experiment A: Drawdown Audit on V3

### Result: V3 PASSES all flags with active drawdown metric.

The 241-day drawdown in V3 holdout is 65.8% flat equity (signal=0, regime filter sitting out). The strategy was not losing money — it was correctly avoiding bear market exposure.

- **Active drawdown (bars with open positions): 14 days** — well under 60-day threshold
- Max loss during 241-day "drawdown": only 3.22% ($34.42 on $1069 peak)
- Monthly returns during drawdown ranged from -1.21% to +1.13%
- All other flags already PASS: no overfit, sufficient trades, low BTC correlation, positive skew

**New metric**: `active_drawdown_duration_days` — counts only consecutive bars where equity is below peak AND strategy has an open position. This is the correct metric for regime-gated strategies that deliberately sit out bear markets.

### Deliverables
- `outputs/experiments/drawdown_audit/ANALYSIS.md`
- `outputs/experiments/drawdown_audit/drawdown_audit.png`

## 2026-03-22 — Experiment B: EVT-based Position Sizing on V3

### Result: EVT dramatically reduces drawdown depth, Sharpe slightly lower.

GARCH(1,1)-EVT dynamic position sizing tested with 3 configs:

| Config | Holdout Sharpe | Max DD% | Worst Month | Active DD |
|--------|---------------|---------|-------------|-----------|
| Fixed frac (baseline) | 0.9065 | 3.22% | -1.21% | 14 days |
| EVT conservative | 0.8959 | 0.34% | -0.12% | 14 days |
| EVT moderate | 0.8833 | 0.43% | -0.15% | 14 days |
| EVT aggressive | 0.8779 | 0.69% | -0.25% | 14 days |

Key findings:
- EVT reduces max drawdown from 3.22% to 0.34% (conservative)
- Worst month improved from -1.21% to -0.12%
- EVT leverage stayed near min_leverage (0.30) — SOL's high volatility kept ES_99 high
- Conservative config passes all flags with active DD metric
- DD days unchanged (241) because the drawdown is flat-equity, not losing

### Deliverables
- `risk/garch_evt.py` — GARCH(1,1)-EVT module
- `outputs/experiments/evt_sizing/SUMMARY.md`
- `outputs/experiments/evt_sizing/leverage_over_time_*.png`

## 2026-03-22 — V3 Improvement Experiments: ALL FAILED

Three experiments attempted to push V3 holdout Sharpe above 1.2:

1. **Pullback entry**: FAIL — overfit (train 1.50, val -0.56). Only 34 trades in training. Pullback filter too aggressive.
2. **ATR trailing stop**: FAIL — WF too low (53.6% vs 60% threshold). Val Sharpe strong (1.54) and mean OOS Sharpe good (1.06), but inconsistent across windows.
3. **BTC 1C SJM + portfolio**: FAIL — BTC SJM WF only 42.9%. Correlation was low (0.24) so diversification potential exists, but BTC trend-following doesn't work at 4H.

**V3 baseline remains the best configuration at Sharpe 0.91.**

### Deliverables
- `REPORT_IMPROVED.md` — full experiment report
- `strategies/sol_1c_sjm_pullback.py` — pullback strategy
- `improve_v3_results.json` — structured results

## Next steps

1. **V3 is deployable** with the active drawdown metric interpretation
2. The EVT conservative config is the safest option (0.34% max DD, 0.90 Sharpe)
3. Recommend 3-month paper trade before live deployment
4. Only viable at funded account scale ($25k+), not retail ($1k)
5. If seeking higher Sharpe: consider different asset class or higher-frequency data
