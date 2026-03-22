"""
Download and cache Binance market structure data.
All data is pre-aggregated by Binance — lightweight and fast.
Resolution: 1H only to conserve disk space.
Period: 2022-2024.
"""

import requests
import ccxt
import pandas as pd
import numpy as np
import time
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/market_structure")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START = '2022-01-01'
END = '2024-12-31'


def check_disk_space(min_free_gb: float = 5.0):
    free_gb = shutil.disk_usage('/home/ubuntu').free / 1e9
    print(f"Disk space free: {free_gb:.1f} GB")
    if free_gb < min_free_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free — stopping to protect system")


def binance_futures_request(endpoint: str, params: dict) -> list:
    """Generic paginated request to Binance Futures data endpoints."""
    base = "https://fapi.binance.com"
    all_data = []
    current_ts = params.get('startTime')
    end_ts = int(pd.Timestamp(END).timestamp() * 1000)

    while current_ts < end_ts:
        params['startTime'] = current_ts
        try:
            resp = requests.get(f"{base}{endpoint}", params=params, timeout=15)
            if resp.status_code == 429:
                print(f"  Rate limited, waiting 10s...")
                time.sleep(10)
                continue
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            all_data.extend(data)
            # Use the last timestamp to advance
            last_ts = data[-1].get('timestamp', data[-1].get('time', 0))
            if isinstance(last_ts, str):
                last_ts = int(pd.Timestamp(last_ts).timestamp() * 1000)
            current_ts = int(last_ts) + 3600000  # +1H
            time.sleep(0.15)
        except Exception as e:
            print(f"  Request error: {e}")
            time.sleep(3)
            current_ts += 500 * 3600000  # skip ahead
            continue

    return all_data


def to_df(data: list, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if timestamp_col not in df.columns:
        # Try common alternatives
        for alt in ['time', 'openTime']:
            if alt in df.columns:
                timestamp_col = alt
                break
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms', utc=True)
    df = df.set_index(timestamp_col).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def fetch_open_interest(symbol: str) -> pd.DataFrame:
    print(f"  Fetching Open Interest: {symbol}")
    raw = symbol.replace('/USDT:USDT', 'USDT')
    data = binance_futures_request(
        '/futures/data/openInterestHist',
        {'symbol': raw, 'period': '1h', 'limit': 500,
         'startTime': int(pd.Timestamp(START).timestamp() * 1000)}
    )
    df = to_df(data)
    if 'sumOpenInterest' in df.columns:
        return df[['sumOpenInterest', 'sumOpenInterestValue']]
    return df


def fetch_ls_global(symbol: str) -> pd.DataFrame:
    print(f"  Fetching Global L/S Ratio: {symbol}")
    raw = symbol.replace('/USDT:USDT', 'USDT')
    data = binance_futures_request(
        '/futures/data/globalLongShortAccountRatio',
        {'symbol': raw, 'period': '1h', 'limit': 500,
         'startTime': int(pd.Timestamp(START).timestamp() * 1000)}
    )
    return to_df(data)


def fetch_ls_top(symbol: str) -> pd.DataFrame:
    print(f"  Fetching Top Trader L/S Ratio: {symbol}")
    raw = symbol.replace('/USDT:USDT', 'USDT')
    data = binance_futures_request(
        '/futures/data/topLongShortAccountRatio',
        {'symbol': raw, 'period': '1h', 'limit': 500,
         'startTime': int(pd.Timestamp(START).timestamp() * 1000)}
    )
    return to_df(data)


def fetch_taker_volume(symbol: str) -> pd.DataFrame:
    print(f"  Fetching Taker Volume Ratio: {symbol}")
    raw = symbol.replace('/USDT:USDT', 'USDT')
    data = binance_futures_request(
        '/futures/data/takerlongshortRatio',
        {'symbol': raw, 'period': '1h', 'limit': 500,
         'startTime': int(pd.Timestamp(START).timestamp() * 1000)}
    )
    return to_df(data)


def fetch_spot_ohlcv(symbol_spot: str) -> pd.DataFrame:
    """Fetch spot OHLCV for basis calculation."""
    print(f"  Fetching Spot OHLCV: {symbol_spot}")
    exchange = ccxt.binance({'enableRateLimit': True})
    since = int(pd.Timestamp(START).timestamp() * 1000)
    end_ts = int(pd.Timestamp(END).timestamp() * 1000)
    all_candles = []
    current = since
    while current < end_ts:
        try:
            candles = exchange.fetch_ohlcv(symbol_spot, '1h', since=current, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            current = candles[-1][0] + 3600000
            time.sleep(0.05)
        except Exception as e:
            print(f"  Spot OHLCV error: {e}")
            time.sleep(3)
            current += 1000 * 3600000
    df = pd.DataFrame(all_candles,
                      columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    df = df[~df.index.duplicated(keep='first')]
    return df


def cached(path: Path, fetch_fn):
    if path.exists():
        print(f"  Cache hit: {path.name}")
        return pd.read_parquet(path)
    df = fetch_fn()
    if df is not None and len(df) > 0:
        df.to_parquet(path, compression='snappy')
        print(f"  Saved: {path.name} ({path.stat().st_size / 1e6:.2f} MB, {len(df)} rows)")
    else:
        print(f"  WARNING: No data for {path.name}")
    return df


def build_features(symbol_futures: str, symbol_spot: str) -> pd.DataFrame:
    """
    Build unified market structure feature DataFrame at 1H resolution.
    All sources aligned to futures OHLCV index.
    """
    check_disk_space()
    safe = symbol_futures.replace('/', '_').replace(':', '_')
    print(f"\n{'=' * 50}\nBuilding features for {symbol_futures}\n{'=' * 50}")

    # Load futures OHLCV (already cached)
    from crypto_infra import DataModule
    dm = DataModule()
    futures = dm.get_ohlcv(symbol_futures, '1h', START, END)
    idx = futures.index
    print(f"  Futures OHLCV: {len(futures)} bars")

    # Download all sources
    p = lambda name: CACHE_DIR / f"{safe}_{name}.parquet"

    spot = cached(p('spot_1h'), lambda: fetch_spot_ohlcv(symbol_spot))
    oi = cached(p('oi_1h'), lambda: fetch_open_interest(symbol_futures))
    ls_global = cached(p('ls_global'), lambda: fetch_ls_global(symbol_futures))
    ls_top = cached(p('ls_top'), lambda: fetch_ls_top(symbol_futures))
    taker = cached(p('taker_1h'), lambda: fetch_taker_volume(symbol_futures))

    try:
        funding = dm.get_funding_rates(symbol_futures, START, END)
    except Exception as e:
        print(f"  Funding rates error: {e}")
        funding = pd.Series(dtype=float)

    # Align everything to futures index
    def align(df_or_series, col=None):
        if isinstance(df_or_series, pd.Series):
            return df_or_series.reindex(idx, method='ffill')
        if df_or_series is None or len(df_or_series) == 0:
            return pd.Series(np.nan, index=idx)
        if col and col in df_or_series.columns:
            return df_or_series[col].reindex(idx, method='ffill')
        return df_or_series.reindex(idx, method='ffill')

    f = pd.DataFrame(index=idx)

    # Price and returns
    f['close'] = futures['close']
    f['ret_1h'] = futures['close'].pct_change()
    f['ret_4h'] = futures['close'].pct_change(4)
    f['ret_24h'] = futures['close'].pct_change(24)
    f['volume'] = futures['volume']

    # Basis (futures premium over spot)
    if spot is not None and len(spot) > 0:
        spot_close = align(spot, 'close')
        f['basis'] = (f['close'] - spot_close) / spot_close * 100
        f['basis_ma_24h'] = f['basis'].rolling(24).mean()
        f['basis_vs_ma'] = f['basis'] - f['basis_ma_24h']

    # Open Interest
    if oi is not None and len(oi) > 0 and 'sumOpenInterest' in oi.columns:
        oi_s = align(oi, 'sumOpenInterest')
        f['oi'] = oi_s
        f['oi_chg_1h'] = oi_s.pct_change()
        f['oi_chg_4h'] = oi_s.pct_change(4)
        f['oi_chg_24h'] = oi_s.pct_change(24)
        f['oi_ma_24h'] = oi_s.rolling(24).mean()
        f['oi_vs_ma'] = (oi_s - f['oi_ma_24h']) / f['oi_ma_24h']
        # Price-OI regime
        f['price_up_oi_up'] = ((f['ret_4h'] > 0) & (f['oi_chg_4h'] > 0)).astype(int)
        f['price_up_oi_down'] = ((f['ret_4h'] > 0) & (f['oi_chg_4h'] < 0)).astype(int)
        f['price_dn_oi_up'] = ((f['ret_4h'] < 0) & (f['oi_chg_4h'] > 0)).astype(int)
        f['price_dn_oi_down'] = ((f['ret_4h'] < 0) & (f['oi_chg_4h'] < 0)).astype(int)

    # Global long/short ratio
    if ls_global is not None and len(ls_global) > 0 and 'longShortRatio' in ls_global.columns:
        ls = align(ls_global, 'longShortRatio')
        f['ls_ratio'] = ls
        f['ls_ma_24h'] = ls.rolling(24).mean()
        f['ls_vs_ma'] = ls - f['ls_ma_24h']
        q85 = ls.rolling(168).quantile(0.85)
        q15 = ls.rolling(168).quantile(0.15)
        f['crowd_long'] = (ls > q85).astype(int)
        f['crowd_short'] = (ls < q15).astype(int)

    # Top trader ratio
    if ls_top is not None and len(ls_top) > 0 and 'longShortRatio' in ls_top.columns:
        ls_t = align(ls_top, 'longShortRatio')
        f['ls_top'] = ls_t
        if 'ls_ratio' in f.columns:
            f['smart_dumb_div'] = ls_t - f['ls_ratio']

    # Taker volume ratio
    if taker is not None and len(taker) > 0 and 'buySellRatio' in taker.columns:
        tv = align(taker, 'buySellRatio')
        f['taker_ratio'] = tv
        f['taker_ma_4h'] = tv.rolling(4).mean()
        f['taker_ma_24h'] = tv.rolling(24).mean()
        f['taker_momentum'] = tv - f['taker_ma_24h']
        f['taker_extreme_buy'] = (tv > 1.5).astype(int)
        f['taker_extreme_sell'] = (tv < 0.67).astype(int)

    # Funding rate
    if len(funding) > 0:
        fund = align(funding)
        f['funding'] = fund
        f['funding_ma_24h'] = fund.rolling(24).mean()
        f['funding_trend'] = fund - f['funding_ma_24h']
        f['funding_ext_long'] = (fund > 0.001).astype(int)
        f['funding_ext_short'] = (fund < -0.0002).astype(int)
        f['cum_funding_7d'] = fund.rolling(168).sum()

    # Liquidation proxy
    if 'oi_chg_1h' in f.columns and 'taker_ratio' in f.columns:
        f['liq_proxy_long'] = (
            (f['oi_chg_1h'] < -0.005) &
            (f['ret_1h'] < -0.005) &
            (f.get('taker_extreme_sell', pd.Series(0, index=idx)) == 1)
        ).astype(int)
        f['liq_proxy_short'] = (
            (f['oi_chg_1h'] < -0.005) &
            (f['ret_1h'] > 0.005) &
            (f.get('taker_extreme_buy', pd.Series(0, index=idx)) == 1)
        ).astype(int)

    # Composite squeeze setup
    squeeze_cols = ['oi_chg_4h', 'crowd_long', 'funding_ext_long']
    if all(c in f.columns for c in squeeze_cols):
        f['squeeze_setup'] = (
            (f['oi_chg_4h'] > 0.01) &
            (f['crowd_long'] == 1) &
            (f['funding_ext_long'] == 1)
        ).astype(int)

    f = f.replace([np.inf, -np.inf], np.nan)

    # Save
    out = CACHE_DIR / f"{safe}_unified_1h.parquet"
    f.to_parquet(out, compression='snappy')
    print(f"\nSaved: {out.name} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {f.shape} | Columns: {len(f.columns)}")
    nan_pct = f.isna().mean().sort_values(ascending=False)
    print(f"Top NaN columns:\n{nan_pct.head(10)}")

    return f


if __name__ == "__main__":
    check_disk_space(min_free_gb=5.0)
    pairs = [('BTC/USDT:USDT', 'BTC/USDT'), ('SOL/USDT:USDT', 'SOL/USDT')]
    for fut, spot in pairs:
        features = build_features(fut, spot)
        print(f"\n{fut}: {len(features):,} bars, {features.shape[1]} features")

    print("\nDisk usage after download:")
    import subprocess
    subprocess.run(['du', '-sh',
                    '/home/ubuntu/projects/crypto-trader/data_cache/market_structure/'])
    subprocess.run(['du', '-sh',
                    '/home/ubuntu/projects/crypto-trader/data_cache/'])
