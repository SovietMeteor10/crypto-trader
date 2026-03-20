# Research Report: Exhaustive Crypto Strategy Search

**Date:** 2026-03-20
**Objective:** Find a systematic futures strategy on Binance perpetuals that can generate $1,000/month from $1,000 capital.
**Universe:** BTC/USDT, ETH/USDT, SOL/USDT perpetual futures
**Data:** 2021-01 to 2025-03 (OHLCV + funding rates)
**Result:** No viable strategy found. Zero candidates passed walk-forward validation.

---

## 1. Constraint Check

The target of $1,000/month from $1,000 capital requires 100% monthly returns. This is mathematically implausible for any systematic strategy with controlled risk.

- At 3x leverage: need 33% monthly on underlying = 396% annualised
- At 5x leverage: need 20% monthly = 240% annualised
- At 10x leverage: need 10% monthly = 120% annualised (but ruin probability approaches certainty)
- Best documented OOS systematic crypto Sharpe: 1.5-2.5
- At Sharpe 2.0 with 60% vol (BTC): expected return ~120% annually = ~6.6%/month
- At 3x leverage: ~20%/month -- still 5x short of target with extreme drawdown risk

**Conclusion:** The $1,000 to $1,000/month target is mathematically impossible at any responsible leverage level. This was known before testing began but the full search was conducted to be thorough.

---

## 2. Market Characterisation Summary

### Hurst Exponent
All assets show mildly trending behaviour (H = 0.57-0.62). SOL daily is strongest at 0.619. The mild trending character supports trend-following strategies but the edge is not extreme. Hurst is stable across years (2021-2025) with no structural break.

### Autocorrelation
- 1h: negligible across all assets
- 4h: weakly negative (-0.032 to -0.037)
- 1d: moderately negative (BTC -0.040, ETH -0.057, SOL -0.047)

The negative autocorrelation at 4h/1d suggests mild mean-reversion exists in theory, but it is too weak to overcome trading costs.

### Volatility
- BTC: 62% annualised (declining from 81% in 2021 to 54% in 2025)
- ETH: 80% annualised, highest cross-year variance
- SOL: 125% annualised, extremely volatile (declining from 163% in 2021)

Volatility compression over time reduces the opportunity set for all strategies.

### Funding Rates
- Overwhelmingly positive: 75-88% of periods across all assets
- Mean: ~0.01%/8h (roughly 10 bps per 8-hour period)
- Funding-return correlation: weak (0.04 to -0.11)

Funding carry shorts have a structural edge of ~10 bps/8h, but directional risk dwarfs the carry income. Without delta-neutral hedging (requiring spot capital), funding carry is just a short-biased strategy.

### Regime History
- 2021: bull (BTC +119%, ETH +811%, SOL +33,753%)
- 2022: bear (BTC -56%, ETH -52%, SOL -86%)
- 2023: recovery (BTC +182%, ETH +113%, SOL +1,530%)
- 2024: bull (BTC +154%, ETH +80%, SOL +160%)
- 2025 YTD: bearish (BTC -19%, ETH -86%, SOL -59%)

Any strategy must survive both bull and bear years. Pure long bias produces flattering backtests but fails in bear markets.

---

## 3. Strategy Search Log Summary

9 strategy families were tested across 3 symbols and multiple timeframes, producing 45 test configurations.

| Family | Type | Best Result | Failure Mode |
|--------|------|-------------|-------------|
| 1A Dual MA Cross | Trend | SOL 4h: WF Sharpe -7.92 | INCONSISTENT |
| 1B Breakout+Volume | Trend | All <0.5 screen Sharpe | INFREQUENT |
| 1C Trend+Regime | Trend | SOL 4h: 45% WF positive | INCONSISTENT |
| 2A RSI Reversion | Mean Rev | All negative OOS | OVERFIT |
| 2B Bollinger Reversion | Mean Rev | All negative OOS | OVERFIT |
| 2C Z-Score Reversion | Mean Rev | All negative OOS | OVERFIT |
| 3A Funding Carry | Carry | All negative validation | OVERFIT |
| 4A Vol Breakout | Volatility | SOL 1h: 45% WF positive | INCONSISTENT |
| 5A Cross Momentum | Momentum | WF Sharpe -1.98 | OVERFIT/INCONSISTENT |

**Zero strategies passed all validation checks.**

Full details in STRATEGY_LOG.md.

---

## 4. Closest Candidate Analysis

Two strategies came closest to passing: SOL trend+regime (1C) on 4h and SOL vol breakout (4A) on 1h. Both achieved 45% walk-forward positive windows against the 50% threshold.

### SOL 1C: Trend Following with Regime Filter (4h)
- Screen Sharpe: 1.92
- Train Sharpe: 1.60
- Validation Sharpe: 1.24
- Walk-forward OOS Sharpe: -1.47
- Walk-forward positive windows: 45%

This was the most promising candidate. The regime filter (ADX-based) correctly identified trending periods in-sample. However, regime transitions in OOS data did not match historical patterns. The filter that worked in 2021-2023 produced false signals in 2024-2025.

### SOL 4A: Volatility Breakout (1h)
- Screen Sharpe: 1.22
- Train Sharpe: 0.60
- Validation Sharpe: 2.22
- Walk-forward OOS Sharpe: -0.79
- Walk-forward positive windows: 45%

The suspiciously high validation Sharpe (2.22 vs train 0.60) suggests the validation period captured one or two large vol expansion events that the strategy rode. This is not a robust edge; it is a lucky draw in the validation window.

### Why 45% is not close enough
A strategy with 45% positive walk-forward windows is expected to lose money more often than it makes money in any given period. Over a year of live trading, this means prolonged drawdowns that will trigger either ruin or abandonment. The 50% threshold is already generous; serious systematic traders require 60%+.

---

## 5. Why This Will Fail (Steelman)

Even if we relaxed our criteria and deployed the closest candidates, here is why they would fail:

1. **The edge is too small.** Hurst exponents of 0.58-0.62 indicate mild trending, not strong trending. After costs (0.04% taker fees per side, funding, slippage), the edge is consumed. A round-trip costs roughly 12-15 bps; the signal needs to overcome this on every trade.

2. **Overfitting is the dominant failure mode.** Train Sharpe ratios of 1-3 collapsing to negative OOS is the signature of overfitting. With 4-6 parameters per strategy and 4 years of data, the optimiser finds patterns that do not persist. This is not a fixable problem with "better parameters" -- it is a structural limitation of the data/parameter ratio.

3. **Crypto markets are efficient at retail timeframes.** The 1h/4h/1d timeframes are well-arbitraged by market makers and HFT firms. Any signal visible at these frequencies has been priced in by faster participants. The mild trending character captured by Hurst reflects macro flows (institutional allocation, leverage cycles), not exploitable microstructure.

4. **Regime changes destroy everything.** The 2021 bull, 2022 bear, 2023 recovery, 2024 bull, 2025 bear sequence means any single-regime strategy backtest looks brilliant but fails in the next regime. Regime filters attempt to solve this but they themselves overfit to known regime boundaries.

5. **SOL-specific risk.** All three closest candidates were on SOL. SOL has the highest volatility (125% annualised) which inflates backtest returns. It also has thinner liquidity than BTC/ETH, higher slippage in practice, and idiosyncratic risk (FTX collapse in 2022 destroyed SOL's market structure temporarily). A strategy that only works on one asset on one timeframe is fragile.

6. **Mean reversion does not exist.** All three mean reversion families (RSI, Bollinger, Z-score) produced negative Sharpe ratios even in-sample on crypto. This is consistent with the positive Hurst exponents. Crypto assets trend; they do not revert at retail timeframes. Anyone selling a "crypto mean reversion" strategy is selling noise.

7. **Funding carry is a trap.** Positive funding 75-88% of the time looks like free money. But the 12-25% of the time funding is negative corresponds to violent rallies where short positions are destroyed. The carry income (~4.5% annually at 0.01%/8h) is dwarfed by a single 20% adverse move.

---

## 6. Realistic Expectations

For a systematic crypto strategy on Binance Futures perpetuals with responsible risk management:

| Capital | Leverage | Monthly Return (realistic) | Monthly PnL | Drawdown Risk |
|---------|----------|---------------------------|-------------|---------------|
| $1,000 | 3x | 3-8% | $30-80 | 30-50% |
| $10,000 | 3x | 3-8% | $300-800 | 30-50% |
| $50,000 | 2-3x | 2-5% | $1,000-2,500 | 20-40% |
| $100,000 | 2x | 1-3% | $1,000-3,000 | 15-30% |

These figures assume a strategy with OOS Sharpe of 1.0-1.5, which this research did not find. The numbers above represent what would be achievable if a valid strategy existed.

**Key insight:** $1,000/month requires $50,000-$100,000 in capital at realistic return rates. There is no shortcut. The $1,000 to $1,000/month target is not a matter of finding the right strategy; it is a matter of insufficient capital.

---

## 7. Implementation Plan

**None.** No strategy passed walk-forward validation. There is nothing to implement.

Deploying the closest candidates (SOL 1C or SOL 4A at 45% WF positive) would be expected to lose money. The infrastructure built during this project (data pipeline, backtest engine, walk-forward validation, live adapter) is ready if a viable strategy is found in future, but deploying without a validated edge is gambling, not trading.

---

## 8. Funded Account Recommendation

The only realistic path to $1,000/month is through a funded trading account (prop firm or capital allocation programme). This changes the economics entirely:

### Why funded accounts work
- Capital problem solved: $50,000-$200,000 provided by the firm
- At 1-3% monthly return on $100,000 = $1,000-$3,000/month
- 1-3% monthly is achievable with a validated systematic strategy
- Firm takes 10-20% of profits; trader keeps 80-90%
- Risk of ruin is limited to the evaluation fee, not the full capital

### Crypto prop firms to research
- Funded trading programmes that accept algorithmic/systematic traders
- Evaluation criteria typically: profit target with drawdown limits
- Some accept API-based execution (compatible with our live_adapter.py)

### What you need to qualify
1. A validated strategy (which this research did not produce on crypto -- consider other markets)
2. Track record of disciplined execution
3. Understanding of the firm's drawdown rules and position limits
4. Evaluation fee ($100-$500 typically)

### Alternative: trade a different market
The infrastructure built here is adaptable. Forex and equity index futures have deeper liquidity, longer histories, and documented systematic edges. Consider:
- Equity index futures (ES, NQ) with trend-following
- Forex majors with carry + momentum
- Commodity futures with seasonal patterns

These markets have decades of academic evidence for systematic edges. Crypto does not.

---

## 9. What to Do If Everything Failed

Everything did fail. Here is what to do next.

### Accept the result
This was an honest, exhaustive search. 9 strategy families, 3 symbols, multiple timeframes, proper walk-forward validation with realistic costs. The result is clear: there is no systematic edge available on Binance Futures perpetuals at retail timeframes with standard strategy types. This is not a failure of the research; it is a truthful answer to the question.

### Do not
- **Do not deploy an unvalidated strategy.** The temptation to "just try it live with small size" is the path to slow losses. A strategy that fails walk-forward validation will lose money in live trading. The backtest is not being pessimistic; it is being realistic.
- **Do not add more indicators.** Adding RSI to a failing MA crossover does not create an edge; it creates a more complex way to overfit. Complexity is not the solution.
- **Do not switch to shorter timeframes (1m, 5m).** Sub-hourly crypto is dominated by market makers with co-located infrastructure. Retail latency makes this a guaranteed loss.
- **Do not increase leverage.** Higher leverage does not create returns; it creates larger losses faster.

### Do instead
1. **Research funded account programmes.** This is the immediate next step. Find crypto or multi-asset prop firms that accept systematic traders. The capital problem is the binding constraint, not the strategy problem.
2. **Consider other asset classes.** Forex and equity index futures have documented systematic edges with decades of data. The infrastructure built here (backtest engine, walk-forward validation, live adapter) can be adapted.
3. **Build a track record on paper.** If you find a candidate strategy in another market, paper trade it for 3-6 months to build evidence before committing capital.
4. **Study market microstructure.** The edges that exist in crypto are at the microsecond level (market making, latency arbitrage). These require infrastructure investment ($10,000+ for co-location and market data) but represent real, documented edges.
5. **Revisit in 12 months.** Crypto market structure is evolving. Institutional participation is increasing. New instruments (options, structured products) may create edges that do not exist today. The infrastructure is built and ready.

### The honest summary
The $1,000 to $1,000/month target was never achievable with $1,000 capital using systematic trading on Binance Futures. The research confirms this with evidence. The funded account path ($50,000-$200,000 capital at 1-3%/month) is the only realistic route to $1,000/month. Everything else is wishful thinking.

---

## Appendices

### A. Infrastructure Built
All modules are implemented, tested, and ready for future use:
- `data_module.py` -- OHLCV + funding rate fetch with caching and validation
- `signal_module.py` -- abstract base class with output validation
- `sizer_module.py` -- fixed fractional, Kelly, fixed USDT sizing with liquidation checks
- `cost_module.py` -- fees, per-symbol slippage, funding rate costs
- `backtest_engine.py` -- single run, walk-forward, three-split, Optuna optimisation
- `metrics_module.py` -- full MetricsBundle with flags, perturbation test, comparison
- `live_adapter.py` -- dry-run capable live adapter via ccxt

### B. Files
- `STRATEGY_LOG.md` -- detailed per-strategy results
- `RESEARCH_PLAN.md` -- original research plan and scope
- `NOTES.md` -- market characterisation notes
- `PROGRESS.md` -- project progress log
