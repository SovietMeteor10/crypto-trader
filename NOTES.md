# Notes — Exhaustive Crypto Strategy Search

## Market properties

### Hurst exponent (all assets, all timeframes: 0.57-0.62)
- All assets show mild trending behaviour (H > 0.5)
- SOL daily is strongest: H=0.619
- Stable across years (2021-2025): no structural break
- Implication: trend-following is supported, but the edge is not extreme

### Autocorrelation
- 1h: negligible (BTC -0.003, ETH +0.007, SOL -0.020)
- 4h: weakly negative (-0.032 to -0.037) — mild mean reversion
- 1d: moderately negative (BTC -0.040, ETH -0.057, SOL -0.047)
- Implication: short-term mean reversion exists at 4h/1d but is weak. Not enough for a standalone MR strategy without other edges.

### Volatility
- BTC: 62% annualised, declining from 81% (2021) to 54% (2025)
- ETH: 80%, highest variance across years
- SOL: 125%, extremely volatile, declining from 163% (2021)
- Implication: SOL offers most opportunity per unit capital but also most risk. Position sizing critical.

### Funding rates
- BTC: mean 0.0115%, positive 87.8% of periods, max 541 consecutive positive
- ETH: mean 0.0125%, positive 86.6%, max 609 consecutive positive
- SOL: mean 0.0011%, positive 74.8%, max 496 consecutive positive
- Funding-return correlation: weak (BTC 0.04, ETH 0.10, SOL -0.11)
- Implication: funding carry shorts have ~10 bps/8h structural edge, but directional risk dwarfs it. Not viable standalone without delta hedging. Funding as a signal additive has marginal value.

### Yearly regimes
- 2021: bull (BTC +119%, ETH +811%, SOL +33753%), high vol
- 2022: bear (BTC -56%, ETH -52%, SOL -86%), moderate vol
- 2023: recovery (BTC +182%, ETH +113%, SOL +1530%), lower vol
- 2024: bull (BTC +154%, ETH +80%, SOL +160%), moderate vol
- 2025 YTD: bearish (BTC -19%, ETH -86%, SOL -59%)
- Implication: any strategy must survive bull and bear years. Pure long bias will look great in backtest but fail in bear markets.

## Key findings so far
- Data supports trend-following (Hurst > 0.5) more than mean reversion
- Negative lag-1 autocorrelation at 4h/1d is a mild contrarian signal
- Funding carry is not viable standalone (directional risk >> carry income)
- SOL is best trend-following candidate (highest Hurst, highest vol)
- Vol compression post-2021 may reduce future opportunity

## What this means for Anthony
- Trend-following with regime filter is the primary candidate
- Mean reversion at shorter timeframes is worth testing but expectations should be low
- Funded account path at realistic returns (5-15%/month with 3x leverage) is viable
- $1,000 to $1,000/month directly is not — funded accounts are the path
