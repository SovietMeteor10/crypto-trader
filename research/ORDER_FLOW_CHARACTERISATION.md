# Order Flow Characterisation Report

**Date:** 2026-03-22
**Data:** BTCUSDT, SOLUSDT aggTrades from Binance Futures
**Period:** 2023-01-01 to 2024-12-31
**Total trades analysed:** 936,394,835 BTC + 515,758,120 SOL = 1.45 billion
**Bar frequency:** 15-minute (70,080 bars per asset)

---

## Executive Summary

Order flow data on Binance perpetual futures contains statistically significant but extremely weak predictive information beyond OHLCV. The strongest signals come from **microstructure features** (Kyle's Lambda, Roll spread, arrival rate) rather than directional flow (OFI). The best single result is **Roll spread predicting BTC returns at 1-hour horizon** (t=4.19, R²=0.025%). All R² values are below 0.03%, meaning order flow explains less than 3 basis points of return variance at any horizon. The predictive power is **regime-dependent**, strongest in correction/ranging periods and weakest during strong bull moves. Cross-asset flow (BTC OFI → SOL returns) shows no significant signal. VPIN does **not** predict larger price moves — contrary to the literature.

**Bottom line:** Order flow features are statistically detectable but economically marginal. They may add value as secondary features in an existing strategy framework (e.g., timing entries in V3), but are insufficient to build a standalone strategy.

---

## Data Statistics

| Metric | BTCUSDT | SOLUSDT |
|--------|---------|---------|
| Total trades | 936,394,835 | 515,758,120 |
| Parquet size | 5.90 GB | 3.09 GB |
| 15-min bars | 70,080 | 70,080 |
| Avg trades/bar | 13,362 | 7,360 |
| Daily avg trades | 1.28M | 0.71M |

Data quality: Complete for all 731 days. No gaps. All 24 months downloaded and parsed successfully.

---

## Order Flow Imbalance Findings

### OFI Distribution

| Statistic | BTCUSDT | SOLUSDT |
|-----------|---------|---------|
| Mean | -0.0071 | -0.0110 |
| Std | 0.1620 | 0.1155 |
| Skewness | +0.024 | -0.011 |
| Kurtosis | +0.070 | +0.449 |
| \|OFI\| > 0.7 | 0.01% | 0.00% |

Both distributions are near-Gaussian with very slight leptokurtosis (SOL slightly more fat-tailed). OFI is symmetric around zero with a slight negative bias (marginally more selling pressure). Strongly directional bars (|OFI| > 0.7) are essentially nonexistent — the market is overwhelmingly two-sided at 15-min resolution.

### OFI Serial Correlation

| | BTCUSDT | SOLUSDT |
|--|---------|---------|
| Lag-1 ACF | +0.016 | +0.029 |
| Decay to insignificance | Lag 8 (2 hours) | Lag 4 (1 hour) |

OFI has very weak positive autocorrelation — order flow momentum exists but is negligible. SOL's OFI decorrelates faster than BTC's, consistent with SOL's higher volatility and faster mean reversion in flow.

### OFI vs Next-Bar Return

| | BTCUSDT | SOLUSDT |
|--|---------|---------|
| All bars | -0.0040 | -0.0060 |
| High volume bars | +0.0010 | -0.0068 |
| Low volume bars | -0.0130 | -0.0059 |

OFI has near-zero correlation with next-bar returns. The slightly negative sign suggests **very mild mean reversion** in order flow impact — buying pressure at time T is followed by marginally negative returns at T+1. This is consistent with market maker inventory management but is far too weak to trade.

---

## Predictive Power Table

Features tested at 1-bar (15min), 4-bar (1h), 16-bar (4h), and 64-bar (16h) horizons.
Standardised betas reported (per 1 std move in feature).

### BTCUSDT

| Feature | 15min (t) | 1h (t) | 4h (t) | 16h (t) |
|---------|-----------|--------|--------|---------|
| OFI | -1.04 | -0.06 | 0.59 | 0.64 |
| OFI MA 1h | -1.70 | -0.39 | 0.20 | 0.13 |
| OFI MA 4h | 0.89 | 1.75 | -1.51 | 1.25 |
| **Kyle's Lambda** | **3.85** | 1.39 | 0.07 | 0.23 |
| **Roll Spread** | **2.27** | **4.19** | 1.93 | 0.71 |
| **Arrival Rate** | **3.18** | **2.07** | 0.73 | 1.01 |
| Amihud | 0.17 | 0.74 | -0.38 | 0.54 |

### SOLUSDT

| Feature | 15min (t) | 1h (t) | 4h (t) | 16h (t) |
|---------|-----------|--------|--------|---------|
| OFI | -1.57 | 0.11 | **2.21** | 0.75 |
| OFI MA 1h | **-2.24** | -1.16 | **2.60** | 0.70 |
| OFI MA 4h | -0.24 | 1.71 | 1.66 | 0.02 |
| **Kyle's Lambda** | **2.62** | **2.88** | -0.60 | 1.19 |
| **Roll Spread** | **2.56** | **2.87** | 0.86 | 0.86 |
| **Arrival Rate** | **3.28** | 0.82 | 0.20 | 1.18 |
| Amihud | 0.94 | 0.82 | -0.07 | -0.15 |

**Bold** = |t| > 2.0 (significant at 5%).

### Key observations:

1. **Microstructure features dominate**: Kyle's Lambda, Roll spread, and arrival rate have the most significant t-stats. These measure market quality (liquidity, spread, activity) rather than directional flow.

2. **Positive betas on microstructure**: Higher lambda, wider spreads, and more trades all predict **positive** returns at short horizons. This is a volatility/activity premium — active markets tend to drift up slightly.

3. **OFI is weak for BTC but moderately useful for SOL at 4h**: SOL OFI and OFI MA 1h are significant at the 4-hour horizon with positive betas. This means buying pressure in SOL today predicts positive SOL returns 4 hours later — a momentum effect.

4. **All R² values are below 0.03%**: Even the best signal (BTC Roll Spread → 1h, t=4.19) has R²=0.025%. This is statistically significant but economically tiny.

5. **OFI reversal at very short horizon (SOL)**: OFI MA 1h predicts negative 15-min returns (t=-2.24) but positive 4h returns (t=2.60). Short-term mean reversion, longer-term momentum.

---

## Regime-Conditional Predictive Power

Best feature per asset tested across 4 regimes:
- BTC: Roll Spread → 1h (4 bars)
- SOL: Arrival Rate → 15min (1 bar)

### BTCUSDT (Roll Spread → 1h return)

| Period | Description | t-stat | R² | Significant? |
|--------|-------------|--------|-----|-------------|
| P1: 2023 H1 | Recovery/ranging | 1.79 | 0.019% | No |
| P2: 2023 H2 | Bull market begin | **2.99** | 0.051% | **Yes** |
| P3: 2024 H1 | ETF approval/bull | 1.19 | 0.008% | No |
| P4: 2024 H2 | Correction/ranging | **2.52** | 0.036% | **Yes** |

### SOLUSDT (Arrival Rate → 15min return)

| Period | Description | t-stat | R² | Significant? |
|--------|-------------|--------|-----|-------------|
| P1: 2023 H1 | Recovery | 1.95 | 0.022% | No |
| P2: 2023 H2 | Bull begin | **2.53** | 0.036% | **Yes** |
| P3: 2024 H1 | ETF bull peak | -0.25 | 0.000% | No |
| P4: 2024 H2 | Correction | **4.49** | 0.115% | **Yes** |

### Interpretation

The predictive power is **strongly regime-dependent**:
- **Strongest in P2 and P4** (transition/ranging periods)
- **Weakest in P3** (strong directional bull, ETF-driven)
- This makes intuitive sense: during strong one-way moves, microstructure signals are overwhelmed by macro flows; during ranging/transitional markets, microstructure edges surface

**Implication for Phase 2:** A strategy using order flow features MUST include a regime filter. The signal disappears during strong macro-driven moves.

---

## Informed Trading Events (VPIN)

### BTCUSDT

| Metric | Value |
|--------|-------|
| VPIN mean | 0.129 |
| VPIN P90 | 0.150 |
| VPIN P95 threshold | 0.151 |
| High-VPIN bars | 141 |
| Avg \|4H move\| after high VPIN | 0.333% |
| Avg \|4H move\| baseline | 0.350% |
| T-test (high VPIN vs baseline) | t=-0.66, p=0.51 |

### SOLUSDT

| Metric | Value |
|--------|-------|
| VPIN mean | 0.075 |
| VPIN P90 | 0.090 |
| VPIN P95 threshold | 0.096 |
| High-VPIN bars | 139 |
| Avg \|4H move\| after high VPIN | 0.656% |
| Avg \|4H move\| baseline | 0.758% |
| T-test (high VPIN vs baseline) | t=-2.61, p=0.01 |

### Interpretation

**VPIN does NOT predict larger price moves** — in fact, for SOL, high-VPIN periods are followed by *smaller* absolute moves (significant at p=0.01). This contradicts the theoretical expectation and the Kitvanitphasu et al. finding.

Possible explanations:
1. **VPIN measured on 1-month sample** — may not be representative
2. **Crypto VPIN levels (0.07-0.13)** are much lower than the documented 0.45-0.47 in earlier literature — suggests the market microstructure has changed, possibly due to increased HFT/market maker activity dampening imbalances
3. High VPIN may coincide with high volatility periods where the market *has already moved*, not predictive of future moves

**Conclusion:** VPIN is not a useful predictive feature for this data. Do not use in Phase 2.

---

## Cross-Asset Flow Analysis

| Pair | h=1 (t) | h=4 (t) | h=16 (t) |
|------|---------|---------|----------|
| BTC OFI → BTC ret (own) | -1.13 | -0.05 | 0.57 |
| SOL OFI → SOL ret (own) | -1.57 | 0.13 | **2.20** |
| BTC OFI → SOL ret (cross) | -1.10 | -0.46 | 1.17 |
| SOL OFI → BTC ret (cross) | -0.09 | -0.31 | 1.83 |

### Interpretation

- **SOL OFI has marginal own-asset predictive power at 16-bar horizon** (t=2.20, p~0.028)
- **No significant cross-asset signal** — BTC flow does not predict SOL returns and vice versa
- SOL→BTC cross signal (t=1.83) is suggestive but not significant
- The 16-bar (4h) horizon is where the strongest OFI signal lives, consistent with 3B findings

**Conclusion:** Cross-asset order flow is not a useful feature. Each asset's own order flow is marginally useful at the 4h horizon only.

---

## Key Findings for Phase 2 Strategy Design

1. **Best feature: Roll Spread (BTC) / Arrival Rate (SOL)** — microstructure quality features, not directional OFI
2. **Best horizon: 1h (BTC), 15min (SOL)** — short-term signals only
3. **Signal consistency: REGIME-DEPENDENT** — works in P2 and P4 (ranging/transitional), fails in P3 (strong bull)
4. **Cross-asset signal: DOES NOT EXIST** — each asset's own features only
5. **VPIN: NOT USEFUL** — does not predict price moves, contradicts literature
6. **OFI directional signal: WEAK but present for SOL at 4h** — mean reversion at 15min, momentum at 4h
7. **All R² < 0.03%** — order flow alone is insufficient for a standalone strategy

### Recommended approach for Phase 2:

**Do NOT build a standalone order flow strategy.** The signals are too weak (R² < 0.03%).

Instead, use order flow features as **entry timing filters** within the existing V3 SOL trend-following framework:
- When V3 generates a signal, check if Roll spread or arrival rate conditions favour entry
- Use OFI MA at 4h as a confirmation filter (enter longs only when OFI is positive = buying pressure momentum)
- Apply regime filter to disable order flow features during strong macro moves (P3-like conditions)

Expected improvement: marginal (maybe 0.05-0.10 Sharpe uplift), not transformative. The V3 strategy's edge comes from regime detection, not microstructure.

---

## What This Data Does NOT Show

1. **OFI does not predict BTC returns at any horizon** — all t-stats below 2.0
2. **VPIN is not informative** — contradicts theoretical expectations
3. **Amihud illiquidity has zero predictive power** — not useful
4. **OFI MA 4h (longer smoothing) does not improve over shorter windows**
5. **Cross-asset flow contains no tradeable information**
6. **Kyle's Lambda signal decays completely after 1 hour** — pure microstructure noise

---

## Data Quality Notes

- Monthly ZIP files from Binance public data vault downloaded successfully for all 24 months
- Monthly files include headers (columns: agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker)
- VPIN computed on 1-month samples (2023-05 for SOL, 2023-09 for BTC) due to memory constraints — 12GB RAM insufficient for multi-month trade-level processing
- No missing days detected in either asset
- Roll measure uses 10-bar rolling window; negative serial covariance clipped to zero
- Kyle's Lambda uses 4-bar rolling OLS estimate
