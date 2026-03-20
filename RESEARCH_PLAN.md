# Research Plan — Exhaustive Crypto Strategy Search

**Constraint check conclusion:** $1,000 to $1,000/month (100% monthly return) is mathematically implausible for any low-variance systematic strategy. At realistic Sharpe 1.5 and 60% annualised vol with 3x leverage, expect ~15% monthly with extreme variance. A funded account with $100k capital needs only 1% monthly to hit $1,000/month — that is the viable path.

## Research question

Which systematic futures strategy (if any) from the standard taxonomy produces a positive out-of-sample Sharpe ratio above 1.0 on Binance perpetuals (BTC, ETH, SOL) across walk-forward validation, with realistic costs and leverage constraints?

## Scope decision

**Deep** — requires full experimental pipeline with walk-forward validation across 10+ strategy variants. Estimated 3-5 hours of compute and analysis.

## Constraint check (detailed)

- Starting capital: $1,000
- Target: $1,000/month = 100% monthly
- At 3x leverage: need 33% monthly on underlying = 396% annually
- At 5x leverage: need 20% monthly = 240% annually
- At 10x leverage: need 10% monthly = 120% annually
- Best documented OOS systematic crypto Sharpe: ~1.5-2.5
- At Sharpe 2.0 with 60% vol (BTC annualised): return ~120% annually, ~6.6%/month
- At 3x leverage: ~20%/month — still 5x short of target
- **Revised realistic target**: 5-15% monthly at 3x leverage, high variance
- **Funded account path**: $100k funded account at 1%/month = $1,000/month — achievable

## Source list

- Binance Futures OHLCV + funding rates (already cached, 2021-01 to 2025-03)
- Market characterisation from experiments/market_props/
- Strategy implementations via crypto_infra framework

## Hypothesis

Based on market characterisation:
- Hurst 0.58-0.62 across all assets/timeframes: mildly trending, supports trend-following
- Lag-1 autocorrelation is negative at 4h/1d: short-term mean reversion exists but weak
- Funding rates overwhelmingly positive (75-88%): funding carry shorts have structural edge
- SOL has highest vol (120%+) and strongest Hurst (0.62 on daily): best candidate for trend-following
- Structure is consistent across years (Hurst stable 0.57-0.66): no regime break

**Prior**: Trend-following on daily timeframe with regime filter (Family 1C) is most likely to pass. Mean reversion at 1h/4h may produce a second candidate. Funding carry is structurally attractive but directional risk is the killer.

## Success criteria

- All strategy families tested and logged in STRATEGY_LOG.md
- At least one strategy passes walk-forward validation OR a clear negative result documented
- Full REPORT.md with realistic expectations and funded account analysis

## Token budget note

Single session. Market characterisation done. Strategy testing is compute-bound, not context-bound. All results written to disk incrementally.
