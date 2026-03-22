"""
Download Binance daily metrics from public data vault.
Contains: OI, top trader LS, global LS, taker volume — at 5min resolution.
Resample to 1H and save as parquet.
"""
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import shutil
import time
import gc
from pathlib import Path
from datetime import date, timedelta

BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/market_structure")


def check_disk_space(min_free_gb=5.0):
    free_gb = shutil.disk_usage('/home/ubuntu').free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(f"Only {free_gb:.1f} GB free — stopping")


def download_metrics(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download daily metrics and concatenate."""
    out_path = CACHE_DIR / f"{symbol}_metrics_1h.parquet"
    if out_path.exists():
        print(f"Cache hit: {out_path.name}")
        return pd.read_parquet(out_path)

    check_disk_space()
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start
    all_dfs = []
    failed = []

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                zf = zipfile.ZipFile(io.BytesIO(resp.content))
                with zf.open(zf.namelist()[0]) as f:
                    df = pd.read_csv(f)
                    df['create_time'] = pd.to_datetime(df['create_time'], utc=True)
                    all_dfs.append(df)
            elif resp.status_code == 404:
                failed.append(date_str)
            else:
                failed.append(date_str)
        except Exception as e:
            failed.append(date_str)

        current += timedelta(days=1)
        if current.day == 1:
            print(f"  Downloaded through {date_str} ({len(all_dfs)} days OK, {len(failed)} failed)")
        time.sleep(0.05)

    if not all_dfs:
        raise RuntimeError(f"No metrics data for {symbol}")

    raw = pd.concat(all_dfs, ignore_index=True)
    raw = raw.set_index('create_time').sort_index()
    raw = raw[~raw.index.duplicated(keep='first')]

    # Convert to numeric
    for col in raw.columns:
        if col != 'symbol':
            raw[col] = pd.to_numeric(raw[col], errors='coerce')

    # Resample 5min → 1H
    hourly = raw.drop(columns=['symbol'], errors='ignore').resample('1h').agg({
        'sum_open_interest': 'last',
        'sum_open_interest_value': 'last',
        'count_toptrader_long_short_ratio': 'last',
        'sum_toptrader_long_short_ratio': 'last',
        'count_long_short_ratio': 'last',
        'sum_taker_long_short_vol_ratio': 'mean',  # average over the hour
    }).dropna(how='all')

    # Rename for clarity
    hourly = hourly.rename(columns={
        'sum_open_interest': 'oi',
        'sum_open_interest_value': 'oi_value',
        'count_toptrader_long_short_ratio': 'ls_top',
        'sum_toptrader_long_short_ratio': 'ls_top_position',
        'count_long_short_ratio': 'ls_global',
        'sum_taker_long_short_vol_ratio': 'taker_ratio',
    })

    hourly.to_parquet(out_path, compression='snappy')
    print(f"\nSaved: {out_path.name} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Shape: {hourly.shape}, range: {hourly.index.min()} to {hourly.index.max()}")
    if failed:
        print(f"Failed dates: {len(failed)} (first few: {failed[:5]})")
    return hourly


if __name__ == "__main__":
    for symbol in ['BTCUSDT', 'SOLUSDT']:
        print(f"\n{'='*50}")
        print(f"Downloading {symbol} metrics")
        print('='*50)
        df = download_metrics(symbol, '2022-01-01', '2024-12-31')
        print(f"{symbol}: {len(df)} hourly bars")
