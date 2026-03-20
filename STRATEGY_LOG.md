# Strategy Log -- Exhaustive Search Results

Date: 2026-03-20
Researcher: Claude (automated)
Universe: Binance Futures perpetuals -- BTC/USDT, ETH/USDT, SOL/USDT
Data range: 2021-01 to 2025-03
Validation method: 3-split (screen/train/val) then walk-forward OOS
Cost model: 0.04% taker fees, per-symbol slippage, funding rate costs
Leverage cap: 3x

---

## Family 1A: Dual Moving Average Crossover

**Logic:** Fast/slow EMA crossover with trend filter. Long when fast > slow, short when fast < slow.

### BTC 4h
- Screen Sharpe: 0.87 (pass)
- Train Sharpe: 1.20
- Validation Sharpe: 0.75
- Walk-forward OOS Sharpe: -9.44
- WF positive windows: <50%
- **Result: INCONSISTENT** -- collapsed in walk-forward; train-to-OOS Sharpe drop of 10+

### BTC 1d
- Screen Sharpe: 0.12
- **Result: INFREQUENT** -- insufficient trade count at daily resolution; cannot reach statistical significance

### ETH 4h
- Screen Sharpe: 0.74 (pass)
- Train Sharpe: 1.49
- Validation Sharpe: 0.66
- Walk-forward: not run (validation Sharpe too low)
- **Result: OVERFIT** -- train Sharpe 1.49, validation collapsed to 0.66

### ETH 1d
- Screen Sharpe: 0.13
- **Result: INFREQUENT** -- insufficient trade count

### SOL 4h
- Screen Sharpe: 0.75 (pass)
- Train Sharpe: 1.24
- Validation Sharpe: 1.60
- Walk-forward OOS Sharpe: -7.92
- WF positive windows: <50%
- **Result: INCONSISTENT** -- validation looked promising but walk-forward collapsed entirely

### SOL 1d
- Screen Sharpe: 0.11
- **Result: INFREQUENT** -- insufficient trade count

---

## Family 1B: Breakout with Volume Confirmation

**Logic:** Price breaks N-period high/low with volume spike confirmation. Entry on breakout, exit on trailing stop.

### ALL symbols, 4h
- Screen Sharpe: <0.5 for all symbols
- **Result: INFREQUENT** -- volume filter removes too many signals; remaining trades insufficient for statistical analysis

---

## Family 1C: Trend Following with Regime Filter

**Logic:** Trend signal (EMA/ADX) with volatility regime filter. Only trade in trending regimes (ADX > threshold), skip ranging markets.

### BTC 4h
- Screen Sharpe: 1.92 (pass)
- Train Sharpe: 0.79
- Validation Sharpe: -0.17
- Walk-forward: not run (validation negative)
- **Result: OVERFIT** -- screen Sharpe inflated by regime filter fitting to known history; out-of-sample negative

### ETH 4h
- Screen Sharpe: 1.71 (pass)
- Train Sharpe: 1.48
- Validation Sharpe: -0.48
- Walk-forward: not run (validation negative)
- **Result: OVERFIT** -- same pattern; regime filter overfits to historical regimes

### SOL 4h
- Screen Sharpe: 1.92 (pass)
- Train Sharpe: 1.60
- Validation Sharpe: 1.24
- Walk-forward OOS Sharpe: -1.47
- WF positive windows: 45%
- **Result: INCONSISTENT** -- closest candidate; validation Sharpe passed but walk-forward positive rate below 50% threshold

---

## Family 2A: RSI Mean Reversion

**Logic:** Short when RSI > overbought threshold, long when RSI < oversold threshold. Fixed holding period or RSI reversion exit.

### ALL symbols, 1h and 4h
- Screen Sharpe: varies (0.3-0.9)
- Train Sharpe: negative or low positive
- Validation Sharpe: negative
- Walk-forward: negative where run
- **Result: ALL OVERFIT or INCONSISTENT** -- crypto assets do not mean-revert at these timeframes; RSI signals are noise

---

## Family 2B: Bollinger Band Mean Reversion

**Logic:** Long when price touches lower band, short when touching upper band. Exit at mean or opposite band.

### ALL symbols, 1h and 4h
- Screen Sharpe: varies (0.4-1.1)
- Train Sharpe: varies
- Validation Sharpe: varies (mostly negative)
- Walk-forward: negative where run
- **Result: ALL OVERFIT or INCONSISTENT** -- Bollinger bands capture vol expansion which is trend continuation in crypto, not reversion

---

## Family 2C: Z-Score Mean Reversion

**Logic:** Z-score of returns over rolling window. Trade reversion when z-score exceeds threshold.

### ALL symbols, 1h and 4h
- Screen Sharpe: varies (0.2-0.8)
- Train Sharpe: negative or low
- Validation Sharpe: negative
- Walk-forward: negative where run
- **Result: ALL OVERFIT or INCONSISTENT** -- same conclusion as 2A/2B; mean reversion is not a viable edge in crypto at retail timeframes

---

## Family 3A: Funding Rate Carry

**Logic:** Short perpetual when funding rate is positive (collect funding). Hedge with spot or accept directional risk.

### ALL symbols, 1h
- Screen Sharpe: viable (1.0-1.5)
- Train Sharpe: BTC 1.6, ETH 0.7, SOL variable
- Validation Sharpe: negative for all
- Walk-forward: not run (validation failed)
- **Result: ALL OVERFIT** -- funding carry edge (~10 bps/8h) is dwarfed by directional risk. Without delta-neutral hedging (which requires spot capital), this is just a short-biased strategy that worked in 2022 bear market

---

## Family 4A: Volatility Breakout

**Logic:** Enter on volatility expansion (ATR breakout or Keltner channel breakout). Ride momentum until vol compresses.

### BTC 1h
- Screen Sharpe: viable (~1.0)
- Train Sharpe: 2.4
- Validation Sharpe: negative
- Walk-forward: not run
- **Result: OVERFIT** -- extreme train Sharpe (2.4) is a red flag; pure overfitting to vol expansion events

### ETH 1h
- Screen Sharpe: viable (~0.9)
- Train Sharpe: 2.1
- Validation Sharpe: negative
- Walk-forward: not run
- **Result: OVERFIT** -- same pattern as BTC

### SOL 1h
- Screen Sharpe: 1.22 (pass)
- Train Sharpe: 0.60
- Validation Sharpe: 2.22
- Walk-forward OOS Sharpe: -0.79
- WF positive windows: 45%
- **Result: INCONSISTENT** -- validation Sharpe suspiciously high (likely captured a single vol event); walk-forward positive rate 45%, below 50% threshold

---

## Family 5A: Cross-Asset Momentum

**Logic:** Rank BTC/ETH/SOL by recent momentum. Go long strongest, short weakest. Rebalance periodically.

### ALL symbols, 4h
- Screen Sharpe: viable (0.8-1.2)
- Train Sharpe: 1.1-1.8
- Validation Sharpe: negative to 1.6 (inconsistent across pairs)
- Walk-forward OOS Sharpe: -1.98
- WF positive windows: <40%
- **Result: ALL OVERFIT or INCONSISTENT** -- crypto assets are highly correlated (BTC leads), so long-short momentum captures noise not signal; correlated drawdowns kill the strategy

---

## Summary

| Family | Best candidate | Best WF positive % | Verdict |
|--------|---------------|--------------------:|---------|
| 1A Dual MA | SOL 4h | <50% | INCONSISTENT |
| 1B Breakout+Vol | none | n/a | INFREQUENT |
| 1C Trend+Regime | SOL 4h | 45% | INCONSISTENT |
| 2A RSI Reversion | none | n/a | ALL OVERFIT |
| 2B Bollinger Reversion | none | n/a | ALL OVERFIT |
| 2C Z-Score Reversion | none | n/a | ALL OVERFIT |
| 3A Funding Carry | none | n/a | ALL OVERFIT |
| 4A Vol Breakout | SOL 1h | 45% | INCONSISTENT |
| 5A Cross Momentum | none | <40% | ALL OVERFIT |

**ZERO strategies passed all validation checks.**

The three closest candidates (SOL 1C trend+regime 4h, SOL 4A vol breakout 1h, SOL 1A dual MA 4h) all failed walk-forward validation with fewer than 50% positive OOS windows.
