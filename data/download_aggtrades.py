"""
Download Binance aggTrades from public data vault.
Uses monthly files when available (fewer requests).
Parses each month to individual parquet, then merges.
Memory-efficient: processes one month at a time.
"""
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date, timedelta
import time
import gc

BASE_URL_MONTHLY = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
BASE_URL_DAILY = "https://data.binance.vision/data/futures/um/daily/aggTrades"
CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/aggtrades")

# Output columns we actually need (drop ID columns to save memory)
KEEP_COLS = ['transact_time', 'price', 'qty', 'is_buyer_maker']


def _download_file(url: str, dest: Path) -> bool:
    """Download a file with retry."""
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=120)
            if response.status_code == 200:
                dest.write_bytes(response.content)
                return True
            elif response.status_code == 404:
                return False
            print(f"  HTTP {response.status_code}, retry {attempt+1}")
        except Exception as e:
            print(f"  Error: {e}, retry {attempt+1}")
        time.sleep(2)
    return False


def _parse_zip_to_parquet(zip_path: Path, parquet_path: Path, start_date=None, end_date=None) -> int:
    """
    Parse a single aggTrades zip directly to parquet.
    Handles both header and headerless CSV formats.
    Returns number of trades.
    """
    if parquet_path.exists():
        # Already parsed
        pf = pd.read_parquet(parquet_path, columns=['transact_time'])
        n = len(pf)
        del pf
        return n

    with zipfile.ZipFile(zip_path) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            # Peek at first line to detect header
            first_line = f.readline().decode().strip()
            f.seek(0)

            has_header = 'agg_trade_id' in first_line

            if has_header:
                df = pd.read_csv(
                    f,
                    usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                    dtype={'price': 'float64', 'quantity': 'float64',
                           'transact_time': 'int64', 'is_buyer_maker': 'bool'},
                )
                df = df.rename(columns={'quantity': 'qty'})
            else:
                df = pd.read_csv(
                    f, header=None,
                    names=['agg_trade_id', 'price', 'qty',
                           'first_trade_id', 'last_trade_id',
                           'transact_time', 'is_buyer_maker'],
                    usecols=['price', 'qty', 'transact_time', 'is_buyer_maker'],
                    dtype={'price': 'float64', 'qty': 'float64',
                           'transact_time': 'int64', 'is_buyer_maker': 'bool'},
                )

    df['transact_time'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True)

    # Filter to date range if specified
    if start_date:
        df = df[df['transact_time'].dt.date >= start_date]
    if end_date:
        df = df[df['transact_time'].dt.date <= end_date]

    df = df.sort_values('transact_time').reset_index(drop=True)
    df.to_parquet(parquet_path, index=False)
    n = len(df)
    del df
    gc.collect()
    return n


def download_aggtrades(
    symbol: str,
    start_date: str,
    end_date: str,
    force_redownload: bool = False,
) -> str:
    """
    Download and cache aggTrades for symbol between dates.
    Returns path to final merged parquet (does NOT load into memory).
    """
    raw_dir = CACHE_DIR / "raw" / symbol
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir = CACHE_DIR / "parsed" / symbol
    parsed_dir.mkdir(parents=True, exist_ok=True)
    final_parquet = CACHE_DIR / f"{symbol}_{start_date}_{end_date}.parquet"

    if final_parquet.exists() and not force_redownload:
        print(f"Already have {final_parquet}")
        return str(final_parquet)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Generate months
    months = []
    current = start.replace(day=1)
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = current.replace(year=current.year+1, month=1)
        else:
            current = current.replace(month=current.month+1)

    total_trades = 0
    month_parquets = []

    for month_str in months:
        zip_name = f"{symbol}-aggTrades-{month_str}.zip"
        zip_path = raw_dir / zip_name
        month_parquet = parsed_dir / f"{symbol}_{month_str}.parquet"

        # Download if needed
        if not zip_path.exists():
            url = f"{BASE_URL_MONTHLY}/{symbol}/{zip_name}"
            print(f"Downloading {month_str}...", end=" ", flush=True)
            if _download_file(url, zip_path):
                size_mb = zip_path.stat().st_size / 1e6
                print(f"OK ({size_mb:.0f} MB)")
            else:
                print(f"FAILED — trying daily fallback")
                n = _download_daily_month(symbol, month_str, raw_dir, parsed_dir, start, end)
                if n > 0:
                    month_parquets.append(str(parsed_dir / f"{symbol}_{month_str}.parquet"))
                    total_trades += n
                continue
            time.sleep(0.2)
        else:
            size_mb = zip_path.stat().st_size / 1e6
            print(f"Cached {month_str} ({size_mb:.0f} MB)", end=" ")

        # Parse to parquet
        print(f"→ parsing...", end=" ", flush=True)
        try:
            n = _parse_zip_to_parquet(zip_path, month_parquet, start, end)
            print(f"{n:,} trades")
            total_trades += n
            month_parquets.append(str(month_parquet))
        except Exception as e:
            print(f"PARSE ERROR: {e}")

        gc.collect()

    # Merge using pyarrow to avoid loading everything into pandas
    print(f"\nMerging {len(month_parquets)} months into final parquet...", flush=True)
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Write merged parquet using pyarrow writer (streams, no full load)
    writer = None
    total_rows = 0
    for mp in month_parquets:
        table = pq.read_table(mp)
        if writer is None:
            writer = pq.ParquetWriter(str(final_parquet), table.schema)
        writer.write_table(table)
        total_rows += len(table)
        del table
        gc.collect()

    if writer:
        writer.close()

    print(f"\nDone: {total_rows:,} trades for {symbol}")
    print(f"Parquet size: {final_parquet.stat().st_size / 1e9:.2f} GB")

    return str(final_parquet)


def _download_daily_month(symbol, month_str, raw_dir, parsed_dir, global_start, global_end):
    """Download daily files for a single month as fallback."""
    year, month = int(month_str[:4]), int(month_str[5:7])
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    end = min(end, global_end)
    start = max(start, global_start)

    daily_dfs = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        zip_path = raw_dir / f"{symbol}-aggTrades-{date_str}.zip"

        if not zip_path.exists():
            url = f"{BASE_URL_DAILY}/{symbol}/{symbol}-aggTrades-{date_str}.zip"
            if not _download_file(url, zip_path):
                current += timedelta(days=1)
                continue
            time.sleep(0.1)

        try:
            with zipfile.ZipFile(zip_path) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    first_line = f.readline().decode().strip()
                    f.seek(0)
                    has_header = 'agg_trade_id' in first_line
                    if has_header:
                        df = pd.read_csv(f,
                            usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'],
                            dtype={'price': 'float64', 'quantity': 'float64',
                                   'transact_time': 'int64', 'is_buyer_maker': 'bool'})
                        df = df.rename(columns={'quantity': 'qty'})
                    else:
                        df = pd.read_csv(f, header=None,
                            names=['agg_trade_id', 'price', 'qty',
                                   'first_trade_id', 'last_trade_id',
                                   'transact_time', 'is_buyer_maker'],
                            usecols=['price', 'qty', 'transact_time', 'is_buyer_maker'],
                            dtype={'price': 'float64', 'qty': 'float64',
                                   'transact_time': 'int64', 'is_buyer_maker': 'bool'})
            df['transact_time'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True)
            daily_dfs.append(df)
        except Exception as e:
            print(f"  daily parse error {date_str}: {e}")
        current += timedelta(days=1)

    if daily_dfs:
        merged = pd.concat(daily_dfs, ignore_index=True)
        merged = merged.sort_values('transact_time').reset_index(drop=True)
        out = parsed_dir / f"{symbol}_{month_str}.parquet"
        merged.to_parquet(out, index=False)
        n = len(merged)
        del merged, daily_dfs
        gc.collect()
        return n
    return 0


if __name__ == "__main__":
    for symbol in ['BTCUSDT', 'SOLUSDT']:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol}")
        print('='*60)
        path = download_aggtrades(symbol, '2023-01-01', '2024-12-31')
        print(f"Saved to: {path}")
