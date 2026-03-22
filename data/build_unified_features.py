"""
Build unified market structure feature DataFrame combining:
- Futures OHLCV (already cached)
- Spot OHLCV (already downloaded)
- Binance daily metrics (OI, LS ratios, taker volume)
- Funding rates (already in DataModule)
"""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
from pathlib import Path

CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/market_structure")
START = '2020-09-01'
END = '2024-12-31'


def build_unified(symbol_futures, symbol_spot, symbol_raw):
    """Build unified feature set."""
    safe = symbol_futures.replace('/', '_').replace(':', '_')
    out_path = CACHE_DIR / f"{safe}_unified_1h.parquet"

    print(f"\n{'='*50}")
    print(f"Building unified features: {symbol_futures}")
    print('='*50)

    # Futures OHLCV
    from crypto_infra import DataModule
    dm = DataModule()
    futures = dm.get_ohlcv(symbol_futures, '1h', START, END)
    idx = futures.index
    print(f"  Futures: {len(futures)} bars")

    # Spot OHLCV
    spot_path = CACHE_DIR / f"{safe}_spot_1h.parquet"
    if spot_path.exists():
        spot = pd.read_parquet(spot_path)
        spot.index = pd.DatetimeIndex(spot.index)
        print(f"  Spot: {len(spot)} bars")
    else:
        spot = pd.DataFrame()
        print("  Spot: not available")

    # Metrics (OI, LS, taker)
    metrics_path = CACHE_DIR / f"{symbol_raw}_metrics_1h.parquet"
    if metrics_path.exists():
        metrics = pd.read_parquet(metrics_path)
        metrics.index = pd.DatetimeIndex(metrics.index)
        print(f"  Metrics: {len(metrics)} bars, columns: {list(metrics.columns)}")
    else:
        metrics = pd.DataFrame()
        print("  Metrics: not available")

    # Funding rates
    try:
        funding = dm.get_funding_rates(symbol_futures, START, END)
        print(f"  Funding: {len(funding)} rates")
    except Exception as e:
        print(f"  Funding error: {e}")
        funding = pd.Series(dtype=float)

    def align(series_or_df, col=None):
        if isinstance(series_or_df, pd.Series):
            return series_or_df.reindex(idx, method='ffill')
        if series_or_df is None or len(series_or_df) == 0:
            return pd.Series(np.nan, index=idx)
        if col and col in series_or_df.columns:
            return series_or_df[col].reindex(idx, method='ffill')
        return series_or_df.reindex(idx, method='ffill')

    f = pd.DataFrame(index=idx)

    # ── Price and returns ──
    f['close'] = futures['close']
    f['ret_1h'] = futures['close'].pct_change()
    f['ret_4h'] = futures['close'].pct_change(4)
    f['ret_24h'] = futures['close'].pct_change(24)
    f['volume'] = futures['volume']

    # ── Basis ──
    if len(spot) > 0 and 'close' in spot.columns:
        spot_close = align(spot, 'close')
        f['basis'] = (f['close'] - spot_close) / spot_close * 100
        f['basis_ma_24h'] = f['basis'].rolling(24).mean()
        f['basis_vs_ma'] = f['basis'] - f['basis_ma_24h']

    # ── Open Interest ──
    if len(metrics) > 0 and 'oi' in metrics.columns:
        oi = align(metrics, 'oi')
        f['oi'] = oi
        f['oi_chg_1h'] = oi.pct_change()
        f['oi_chg_4h'] = oi.pct_change(4)
        f['oi_chg_24h'] = oi.pct_change(24)
        f['oi_ma_24h'] = oi.rolling(24).mean()
        f['oi_vs_ma'] = (oi - f['oi_ma_24h']) / f['oi_ma_24h']
        # Price-OI regime
        f['price_up_oi_up'] = ((f['ret_4h'] > 0) & (f['oi_chg_4h'] > 0)).astype(int)
        f['price_up_oi_down'] = ((f['ret_4h'] > 0) & (f['oi_chg_4h'] < 0)).astype(int)
        f['price_dn_oi_up'] = ((f['ret_4h'] < 0) & (f['oi_chg_4h'] > 0)).astype(int)
        f['price_dn_oi_down'] = ((f['ret_4h'] < 0) & (f['oi_chg_4h'] < 0)).astype(int)

    # ── Global L/S ratio ──
    if len(metrics) > 0 and 'ls_global' in metrics.columns:
        ls = align(metrics, 'ls_global')
        f['ls_ratio'] = ls
        f['ls_ma_24h'] = ls.rolling(24).mean()
        f['ls_vs_ma'] = ls - f['ls_ma_24h']
        q85 = ls.rolling(168).quantile(0.85)
        q15 = ls.rolling(168).quantile(0.15)
        f['crowd_long'] = (ls > q85).astype(int)
        f['crowd_short'] = (ls < q15).astype(int)

    # ── Top trader ratio ──
    if len(metrics) > 0 and 'ls_top' in metrics.columns:
        ls_t = align(metrics, 'ls_top')
        f['ls_top'] = ls_t
        if 'ls_ratio' in f.columns:
            f['smart_dumb_div'] = ls_t - f['ls_ratio']

    # ── Taker volume ratio ──
    if len(metrics) > 0 and 'taker_ratio' in metrics.columns:
        tv = align(metrics, 'taker_ratio')
        f['taker_ratio'] = tv
        f['taker_ma_4h'] = tv.rolling(4).mean()
        f['taker_ma_24h'] = tv.rolling(24).mean()
        f['taker_momentum'] = tv - f['taker_ma_24h']
        f['taker_extreme_buy'] = (tv > 1.5).astype(int)
        f['taker_extreme_sell'] = (tv < 0.67).astype(int)

    # ── Funding rate ──
    if len(funding) > 0:
        fund = align(funding)
        f['funding'] = fund
        f['funding_ma_24h'] = fund.rolling(24).mean()
        f['funding_trend'] = fund - f['funding_ma_24h']
        f['funding_ext_long'] = (fund > 0.001).astype(int)
        f['funding_ext_short'] = (fund < -0.0002).astype(int)
        f['cum_funding_7d'] = fund.rolling(168).sum()

    # ── Liquidation proxy ──
    if 'oi_chg_1h' in f.columns and 'taker_extreme_sell' in f.columns:
        f['liq_proxy_long'] = (
            (f['oi_chg_1h'] < -0.005) &
            (f['ret_1h'] < -0.005) &
            (f['taker_extreme_sell'] == 1)
        ).astype(int)
        f['liq_proxy_short'] = (
            (f['oi_chg_1h'] < -0.005) &
            (f['ret_1h'] > 0.005) &
            (f['taker_extreme_buy'] == 1)
        ).astype(int)

    # ── Composite squeeze setup ──
    squeeze_cols = ['oi_chg_4h', 'crowd_long', 'funding_ext_long']
    if all(c in f.columns for c in squeeze_cols):
        f['squeeze_setup'] = (
            (f['oi_chg_4h'] > 0.01) &
            (f['crowd_long'] == 1) &
            (f['funding_ext_long'] == 1)
        ).astype(int)

    f = f.replace([np.inf, -np.inf], np.nan)

    f.to_parquet(out_path, compression='snappy')
    print(f"\nSaved: {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {f.shape}")

    # Data quality summary
    non_nan = (1 - f.isna().mean()).sort_values(ascending=False)
    print(f"\nData coverage (% non-NaN):")
    for col, pct in non_nan.items():
        marker = " ***" if pct < 0.9 else ""
        print(f"  {col:30s}: {pct*100:.1f}%{marker}")

    return f


if __name__ == "__main__":
    pairs = [
        ('BTC/USDT:USDT', 'BTC/USDT', 'BTCUSDT'),
        ('SOL/USDT:USDT', 'SOL/USDT', 'SOLUSDT'),
    ]
    for fut, spot, raw in pairs:
        build_unified(fut, spot, raw)

    import subprocess
    subprocess.run(['du', '-sh',
                    '/home/ubuntu/projects/crypto-trader/data_cache/market_structure/'])
