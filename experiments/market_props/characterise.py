"""Market characterisation: autocorrelation, Hurst, volatility, funding analysis."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
from crypto_infra import DataModule
from statsmodels.tsa.stattools import acf

dm = DataModule()

def hurst_rs(ts, max_lag=100):
    """Hurst exponent via rescaled range."""
    ts = ts.dropna().values
    if len(ts) < max_lag * 2:
        max_lag = len(ts) // 4
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
        rs_vals = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean_adj = chunk - chunk.mean()
            cumdev = np.cumsum(mean_adj)
            R = cumdev.max() - cumdev.min()
            S = chunk.std()
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            tau.append(np.mean(rs_vals))
        else:
            tau.append(1.0)
    if len(tau) < 3:
        return 0.5
    poly = np.polyfit(np.log(list(lags)[:len(tau)]), np.log(tau), 1)
    return poly[0]

results = []

for sym in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
    for tf in ['1h', '4h', '1d']:
        df = dm.get_ohlcv(sym, tf, '2021-01-01', '2025-03-20')
        returns = df['close'].pct_change().dropna()
        
        # Autocorrelation
        ac = acf(returns, nlags=48, fft=True)
        lag1_ac = ac[1]
        # Find first zero crossing
        zero_cross = None
        for i in range(1, len(ac)):
            if ac[i] <= 0:
                zero_cross = i
                break
        
        # Hurst exponent
        H = hurst_rs(returns)
        
        # Volatility by year
        tf_map = {'1h': 8760, '4h': 2190, '1d': 365}
        ann_factor = np.sqrt(tf_map[tf])
        
        vol_stats = {}
        for year in [2021, 2022, 2023, 2024, 2025]:
            yr_ret = returns[returns.index.year == year]
            if len(yr_ret) > 10:
                vol_stats[year] = round(yr_ret.std() * ann_factor * 100, 1)
        
        overall_vol = round(returns.std() * ann_factor * 100, 1)
        
        regime = "trending" if H > 0.55 else ("mean-reverting" if H < 0.45 else "random walk")
        ac_regime = "positive (trend)" if lag1_ac > 0.01 else ("negative (MR)" if lag1_ac < -0.01 else "negligible")
        
        r = {
            'symbol': sym.split('/')[0],
            'timeframe': tf,
            'lag1_autocorr': round(lag1_ac, 4),
            'ac_zero_cross_lag': zero_cross,
            'ac_regime': ac_regime,
            'hurst': round(H, 3),
            'hurst_regime': regime,
            'overall_vol_ann_pct': overall_vol,
            'vol_by_year': vol_stats,
        }
        results.append(r)
        print(f"{r['symbol']} {tf}: H={r['hurst']:.3f} ({regime}), lag1_AC={lag1_ac:.4f} ({ac_regime}), vol={overall_vol}%")

# Funding rate analysis
print("\n--- FUNDING RATE ANALYSIS ---")
for sym in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
    fr = dm.get_funding_rates(sym, '2021-01-01', '2025-03-20')
    ohlcv_1h = dm.get_ohlcv(sym, '1h', '2021-01-01', '2025-03-20')
    
    # Max consecutive positive periods
    pos = (fr > 0).astype(int)
    max_consec_pos = 0
    current = 0
    for v in pos:
        if v: current += 1; max_consec_pos = max(max_consec_pos, current)
        else: current = 0
    
    # Does funding predict next 8h return?
    # Align funding to 8h returns
    ret_8h = ohlcv_1h['close'].pct_change(8).dropna()
    common = fr.index.intersection(ret_8h.index)
    if len(common) > 100:
        corr = fr.loc[common].corr(ret_8h.loc[common])
    else:
        corr = float('nan')
    
    base = sym.split('/')[0]
    print(f"{base}: mean_rate={fr.mean():.6f}, pct_pos={( fr > 0).mean():.1%}, "
          f"max_consec_pos={max_consec_pos}, funding_return_corr={corr:.4f}")

# Yearly regime analysis
print("\n--- YEARLY REGIME ANALYSIS ---")
for sym in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
    df = dm.get_ohlcv(sym, '1d', '2021-01-01', '2025-03-20')
    returns = df['close'].pct_change().dropna()
    base = sym.split('/')[0]
    for year in [2021, 2022, 2023, 2024, 2025]:
        yr = returns[returns.index.year == year]
        if len(yr) < 30:
            continue
        H = hurst_rs(yr, max_lag=min(50, len(yr)//3))
        ac1 = acf(yr, nlags=5, fft=True)[1]
        ann_ret = ((1 + yr.mean())**365 - 1) * 100
        vol = yr.std() * np.sqrt(365) * 100
        print(f"  {base} {year}: H={H:.3f}, AC1={ac1:.4f}, ann_ret={ann_ret:.0f}%, vol={vol:.0f}%")

print("\nDONE")
