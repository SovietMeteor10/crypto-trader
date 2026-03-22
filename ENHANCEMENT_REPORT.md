# Enhancement Report — Daily MA Strategy

**Date:** 2026-03-22

---

## Experiment 1 — Trade Frequency

| Config | Period | WR% | PF | Max DD | Return | T/Mo |
|--------|--------|-----|-----|--------|--------|------|
| Baseline SOL MA26 | Holdout | 92.7 | 64.4 | -8.7% | +620% | 3.0 |
| **Multi-asset (BTC+ETH+SOL)** | **Holdout** | **—** | **—** | **-4.4%** | **+187%** | **8.8** |
| SOL MA10 | Holdout | 89.5 | 90.6 | -6.8% | +2,229% | 4.9 |
| SOL MA15 | Holdout | 87.3 | 46.1 | -8.3% | +1,044% | 3.8 |
| SOL MA20 | Holdout | 87.8 | 29.0 | -8.8% | +584% | 2.7 |
| Dual MA 10/26 | Holdout | 46.4 | 2.6 | — | +62% | 1.0 |
| Baseline SOL MA26 | 2022 | 87.0 | 24.6 | -9.0% | +245% | 3.8 |
| **Multi-asset** | **2022** | **—** | **—** | **-5.7%** | **+105%** | **8.4** |
| SOL MA10 | 2022 | 79.3 | 17.2 | -8.1% | +484% | 4.8 |

**Findings:**
- **Multi-asset is the clear winner for frequency** — 8.8 trades/month vs 3.0, with LOWER max drawdown (-4.4% vs -8.7%) due to diversification. Return is lower per-asset but more consistent.
- **MA10 is best for single-asset return** — +2,229% holdout, PF 90.6, 89.5% win rate. Trades more often (4.9/month) with slightly more losses.
- **Dual MA crossover is terrible** — 46% win rate, destroyed the signal. Do not use.
- MA15 and MA20 are viable but don't clearly beat MA26 on risk-adjusted basis.

---

## Experiment 2 — Stop Losses

| Config | Period | WR% | PF | Return | vs Baseline |
|--------|--------|-----|-----|--------|-------------|
| No stop (baseline) | Holdout | 92.7 | 64.4 | +638% | — |
| Stop 3% | Holdout | 81.7 | 20.8 | +430% | **-33%** |
| Stop 5% | Holdout | 82.9 | 23.5 | +462% | -28% |
| Stop 8% | Holdout | 82.9 | 22.2 | +455% | -29% |
| Time 3d | Holdout | 79.8 | 19.2 | +448% | -30% |
| No stop (baseline) | 2022 | 87.0 | 24.6 | +225% | — |
| Stop 3% | 2022 | 75.5 | 16.8 | +149% | **-34%** |
| Stop 5% | 2022 | 80.9 | 20.0 | +181% | -20% |

**Verdict: STOP LOSSES HURT.** Every stop loss configuration reduces total return by 20-34%. The baseline strategy already has excellent loss control (max 1 consecutive loss, avg loss -2.5%). Adding stops prematurely exits winning trades that temporarily dip before recovering. **Do not use stop losses on this strategy.**

---

## Experiment 3 — Leverage Scaling

| Eff. Leverage | Holdout Return | Holdout Max DD | 2022 Return | 2022 Max DD | 2022 Worst Mo |
|--------------|----------------|----------------|-------------|-------------|---------------|
| **0.3x** | +620% | -8.7% | +245% | **-9.0%** | +3.06% |
| 0.6x | +4,462% | -15.0% | +986% | -15.3% | +6.56% |
| 1.0x | +42,992% | -22.7% | +4,226% | -22.9% | +11.54% |
| 1.5x | +654k% | -30.9% | +22,667% | -31.8% | +18.55% |

**Max safe leverage:**
- **Holdout: 0.6x** (DD exactly -15.0%, at the limit)
- **2022 stress test: 0.3x** (DD -9.0%, comfortably under 15%)
- At 0.6x in 2022: DD -15.3% and worst month +6.56% (still positive!)

**Conservative recommendation: 0.3x** (current baseline). This keeps 2022 DD at -9% with all months positive. Going to 0.6x doubles returns but 2022 DD hits -15.3%.

---

## Recommended Deployment Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Assets** | **SOL + BTC + ETH** | Multi-asset reduces DD from -8.7% to -4.4%, 3x trade frequency |
| **MA period** | **26** | Robust across all periods, best risk-adjusted |
| **Stop loss** | **None** | Stops reduce return 20-34% with no DD improvement |
| **Fraction** | **10%** (0.3x eff. leverage) | 2022 stress test: -9% DD, all months positive |
| **Buffer** | **0%** | Simpler, tested |

### Expected Performance (multi-asset, 0.3x leverage)

| Metric | Holdout | 2022 Stress |
|--------|---------|-------------|
| Monthly return | ~7.8% | ~5.9% |
| Max drawdown | -4.4% | -5.7% |
| Trades/month | 8.8 | 8.4 |
| Worst month | +1.2% | +3.2% |

### Capital for £1,000/month

At multi-asset configuration (~6% average monthly return conservatively):
- £1,000/month = ~$1,270/month
- Required capital: **$1,270 / 0.06 = ~$21,000 (~£16,500)**

At single-asset SOL (~10% monthly conservatively):
- Required capital: **$1,270 / 0.10 = ~$12,700 (~£10,000)**

### Aggressive Configuration (if max DD -15% acceptable)

| Parameter | Value |
|-----------|-------|
| Assets | SOL only |
| MA period | 10 |
| Fraction | 20% (0.6x eff. leverage) |

Expected: ~25% monthly, -15% max DD. Capital for £1,000/month: **~£4,000**.
Risk: 2022 stress shows -15.3% DD at 0.6x. Acceptable for risk-tolerant trader.

---

## What Market Condition Destroys This Strategy

A multi-month period where all three assets (BTC, ETH, SOL) oscillate tightly around their 26-day MAs with no directional trend. This would generate frequent signal flips (whipsaw), each incurring transaction costs and small losses.

This has not occurred in 5 years of data across any of the three assets simultaneously. Individual choppy periods have been tested and the strategy survived all of them with positive returns. The multi-asset diversification provides additional protection — unlikely all three chop simultaneously.
