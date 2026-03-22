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

---

## V3 Improvement Experiments (2026-03-22)

Baseline: SOL 1C SJM V3 — Holdout Sharpe 0.91, WF 60.7%, all flags CLEAR (active DD metric).

### Exp1-Pullback: SOL 1C SJM + Pullback Entry
Pullback atr mult: 1.386
Require reversal bar: False
Signal density vs base: 11.2% of base signals used (very aggressive filter)
Train Sharpe: 1.4977 (34 trades)
Val Sharpe: -0.5645 (19 trades)
WF % positive: not run
Mean OOS Sharpe: not run
Holdout Sharpe: not run
All flags clear: N/A
Conclusion: FAIL: overfit — train positive, validation negative
Delta vs V3: N/A

Notes: Pullback filter reduced signals dramatically (11.2% of base). Optuna
pushed pullback_atr_mult to 1.39 (near max range), effectively minimising the
filter. Even so, only 34 train trades — insufficient for reliable statistics.
The pullback concept needs a different base signal with more frequent entries.

### Exp2-Trailing: SOL 1C SJM + ATR Trailing Stop
ATR stop mult: 2.571
ATR stop period: 8
Exit breakdown: trailing stop reduces avg trade duration from 39.4h to 27.2h
Avg trade duration: 27.2h (V3 base: 39.4h)
Train Sharpe: 1.7201 (168 trades)
Val Sharpe: 1.5431 (78 trades)
WF % positive: 53.6% (threshold: 60%)
Mean OOS Sharpe: 1.0633
Holdout Sharpe: not run
All flags clear: N/A
Conclusion: FAIL: WF too low (53.6%) — 7 points below threshold
Delta vs V3: N/A

Notes: Trailing stop showed promise — val Sharpe strong (1.54) and WF mean
OOS Sharpe actually higher than V3 (1.06 vs 0.91). But WF consistency dropped
from 60.7% to 53.6%. The stop cuts trades short too often in choppy markets,
hurting consistency even though average OOS Sharpe improved.

### Exp3-Portfolio: SOL V3 + BTC 1C SJM Two-Asset Portfolio
SOL V3 vs BTC 1C return correlation: 0.237
BTC 1C SJM train Sharpe: 1.325
BTC 1C SJM val Sharpe: 0.869
BTC 1C SJM WF % positive: 42.9%
BTC 1C SJM WF mean Sharpe: -0.584
Portfolio holdout Sharpe: not run (BTC SJM failed WF)
Conclusion: FAIL: BTC SJM failed walk-forward (42.9%)
Delta vs SOL V3 alone: N/A

Notes: Correlation was low (0.24), confirming diversification potential exists.
BTC SJM passed overfit check but failed WF comprehensively (42.9%, negative mean).
BTC is harder to trade with trend-following than SOL at 4H — lower volatility
and more mean-reverting behaviour at this timeframe.

### Combination: Pullback + Trailing Stop
Not run — neither individual experiment reached holdout, so combination
was not triggered per the stopping rules.

### Summary

| Configuration | WF% | Val Sharpe | Holdout Sharpe | Conclusion |
|--------------|------|------------|----------------|------------|
| V3 baseline | 60.7 | 2.00 | 0.91 | PASS (best) |
| V3 + Pullback | N/A | -0.56 | N/A | FAIL: overfit |
| V3 + Trailing | 53.6 | 1.54 | N/A | FAIL: WF low |
| BTC 1C SJM | 42.9 | 0.87 | N/A | FAIL: WF low |

**V3 baseline remains the best configuration.** None of the three improvements
managed to pass all validation gates. The trailing stop (Exp 2) came closest
with strong val Sharpe and OOS mean, but lacked consistency across WF windows.

---

## OFI-Filtered V3 (Order Flow Phase 2)

### OFI-Filtered V3

OFI neutral threshold: -0.0688
Neutral regime entries surviving filter: 218.2% (filter creates entry churn, not reduces it)
Train Sharpe: 1.7705
Val Sharpe: 1.4273
WF % positive: 53.6
Holdout Sharpe: not run
All flags clear: NO
Delta vs V3 baseline (0.91): N/A (holdout not run)
Conclusion: FAIL: WF too low (53.6%)

**Analysis:** The OFI filter in neutral regime degrades V3 rather than improving it.
Val Sharpe drops from 2.00 (baseline) to 1.43 (OFI-filtered). The filter causes
entry/exit churn in neutral regime — when OFI flips sign, positions get closed and
re-entered, adding transaction costs and whipsawing. WF consistency drops from 60.7%
to 53.6%, below the 60% threshold. The order flow signal (t=2.60 in characterisation)
is too weak and noisy at 15-min resolution to usefully filter 4H trend entries.

**The order flow data does not improve V3.** V3 baseline (Sharpe 0.91) remains best.


### Phase2-B1: V3 + smart money gate

smart_div_threshold: -0.0508
Entries surviving filter: 109.1%
Train Sharpe: 1.9345
Val Sharpe: 2.6294
WF % positive: 42.9
Holdout Sharpe: not run
All flags clear: NO
Delta vs V3 (0.91): N/A
Conclusion: FAIL: WF too low (42.9%)


### Phase2-B2: V3 + BTC structure SJM

Best params: {'fast_period': 22, 'slow_period': 196, 'adx_period': 12, 'adx_threshold': 37, 'sjm_lambda': 0.33304350681789774, 'sjm_window': 314, 'trade_in_neutral': False}
Train Sharpe: 0.8011
Val Sharpe: 0.0198
WF % positive: N/A
Holdout Sharpe: N/A
All flags clear: NO
Delta vs V3 (0.91): N/A
Conclusion: FAIL: overfit


### Phase2-A: Standalone contrarian BTC 1H

Best params: {'crowd_quantile_high': 0.8881917525898656, 'crowd_quantile_low': 0.2043510426107178, 'rolling_window': 336, 'max_hold_bars': 12, 'require_smart_confirm': True, 'smart_div_threshold': -0.16714549919598412}
Train Sharpe: 0.9874
Val Sharpe: -1.0411
WF % positive: N/A
Holdout Sharpe: N/A
All flags clear: NO
Delta vs V3 (0.91): N/A
Conclusion: FAIL: overfit


### Phase2-C: BTC→SOL cross-asset

Best params: {'smart_div_threshold': 0.09398121564997641, 'hold_bars': 7}
Train Sharpe: 1.5427
Val Sharpe: 0.8764
WF % positive: 22.7
Holdout Sharpe: not run
All flags clear: NO
Delta vs V3 (0.91): N/A
Conclusion: FAIL: WF too low (22.7%)


### LGBM: LightGBM market structure model

Features: 1H MS aggregated + 4H OHLCV + regime context + time (79 total)
Best config: conf=0.52, depth=4, leaves=15, lr=0.05
Disguised momentum: NO
Top features: ['rvol_1w', 'dow_sin', 'ret_4h', 'ret_1d', 'trend_regime']
WF % positive: 100.0%
WF mean OOS Sharpe: 1.11
Holdout Sharpe: 0.05
Dir accuracy: 43.0%
Coverage: 28.2%
Delta vs V3: -0.86
Conclusion: FAIL


### LGBM-V2: LightGBM fixed data+features+target

Fixes: extended data to 2020, reduced 79→14 features, binary classification
Features: ['ls_ratio_last', 'ls_ratio_mean', 'smart_dumb_div_last', 'smart_dumb_div_mean', 'taker_ratio_last', 'crowd_long_last', 'oi_chg_1h_mean', 'ret_1d', 'ret_1w', 'rvol_1w', 'adx_14', 'ma_cross_12_26', 'btc_macro_trend', 'vol_regime']
Best config: conf=N/A, depth=N/A, leaves=N/A, lr=N/A
Disguised momentum: NO
WF % positive: 0.0%
WF mean OOS Sharpe: -999.00
Holdout Sharpe: N/A
Dir accuracy: 0.0%
Coverage: 0.0%
Delta vs V3: N/A
Conclusion: FAIL: WF 0.0%


### LGBM-V2: LightGBM fixed data+features+target

Fixes: extended data to 2020, reduced 79→14 features, binary classification
Features: ['ls_ratio_last', 'ls_ratio_mean', 'smart_dumb_div_last', 'smart_dumb_div_mean', 'taker_ratio_last', 'crowd_long_last', 'oi_chg_1h_mean', 'ret_1d', 'ret_1w', 'rvol_1w', 'adx_14', 'ma_cross_12_26', 'btc_macro_trend', 'vol_regime']
Best config: conf=0.52, depth=3, leaves=8, lr=0.05
Disguised momentum: NO
WF % positive: 100.0%
WF mean OOS Sharpe: 2.49
Holdout Sharpe: 0.18
Dir accuracy: 50.5%
Coverage: 69.8%
Delta vs V3: -0.73
Conclusion: FAIL: holdout 0.18


### MTF-A: V3 + Daily Filter + LS Gate

Daily MA period: 20
Daily buffer: 0.9867720377153604%
LS quantile high: 0.8428758381700371
Use LS filter: False
Signal survival rate vs V3: 103.0%
Train Sharpe: 2.5923
Val Sharpe: 3.1381
WF % positive: 57.1
Holdout Sharpe: not run
Delta vs V3 (0.91): N/A
All flags clear: NO
Conclusion: FAIL: WF 57.1%

### MTF-B: Supertrend SOL 4H

ATR period: 19
Multiplier: 3.5104482600764686
Use daily filter: True
Train Sharpe: 4.0535
Val Sharpe: 3.4002
WF % positive: 92.9
Holdout Sharpe: 3.3687
Correlation with V3 holdout: 0.5277
Conclusion: holdout 3.37, corr 0.53


### Daily MA SOL: Pure daily MA trend signal

Best params: ma_period=10, buffer_pct=1.5515768816280384
Unoptimised (MA=26, buf=0) holdout Sharpe: 6.0805
Train Sharpe: 5.6522
Val Sharpe: 6.1294
WF % positive: 100.0
Holdout Sharpe: 9.7375
Delta vs V3 (0.91): +8.83
All flags clear: YES
Conclusion: PASS
