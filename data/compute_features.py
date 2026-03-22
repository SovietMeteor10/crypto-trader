"""
Compute order flow bar features from per-month parquet files.
Memory-efficient: processes one month at a time.
"""
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from data.order_flow import compute_bar_features, compute_vpin

CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/aggtrades")
PARSED_DIR = CACHE_DIR / "parsed"


def _process_large_month(parquet_path: Path, freq: str) -> pd.DataFrame:
    """Process a large month parquet in day-sized chunks to avoid OOM."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    n_groups = pf.metadata.num_row_groups

    # Read in row groups, aggregate trades by day, compute bars
    chunk_bars = []
    for i in range(n_groups):
        table = pf.read_row_group(i)
        trades = table.to_pandas()
        del table

        if len(trades) > 0:
            bars = compute_bar_features(trades, freq=freq)
            chunk_bars.append(bars)

        del trades
        gc.collect()

    if not chunk_bars:
        return pd.DataFrame()

    # If only one row group, just read in halves by row count
    if n_groups <= 1:
        df = pd.read_parquet(parquet_path)
        mid = len(df) // 2
        bars1 = compute_bar_features(df.iloc[:mid], freq=freq)
        del df
        gc.collect()
        df = pd.read_parquet(parquet_path)
        bars2 = compute_bar_features(df.iloc[mid:], freq=freq)
        del df
        gc.collect()
        result = pd.concat([bars1, bars2])
        result = result[~result.index.duplicated(keep='first')]
        return result.sort_index()

    result = pd.concat(chunk_bars)
    result = result[~result.index.duplicated(keep='first')]
    return result.sort_index()


def compute_bars_from_months(symbol: str, freq: str = '15min') -> pd.DataFrame:
    """Compute bar features from per-month parquet files."""
    bars_cache = CACHE_DIR / f"{symbol}_bars_{freq.replace('min','m')}.parquet"

    if bars_cache.exists():
        print(f"Loading cached bars from {bars_cache}")
        bars = pd.read_parquet(bars_cache)
        bars.index = pd.DatetimeIndex(bars.index)
        return bars

    month_dir = PARSED_DIR / symbol
    month_files = sorted(month_dir.glob(f"{symbol}_*.parquet"))
    print(f"Computing {freq} bars for {symbol} from {len(month_files)} months")

    all_bars = []
    for mf in month_files:
        month_name = mf.stem.split('_')[-1]
        print(f"  {month_name}...", end=" ", flush=True)

        file_size = mf.stat().st_size
        # If file > 100MB on disk (~20M+ trades), process via row groups
        if file_size > 100_000_000:
            bars = _process_large_month(mf, freq)
        else:
            trades = pd.read_parquet(mf)
            if len(trades) == 0:
                print("empty")
                continue
            bars = compute_bar_features(trades, freq=freq)
            del trades

        print(f"{len(bars)} bars")
        all_bars.append(bars)
        del bars
        gc.collect()

    result = pd.concat(all_bars)
    del all_bars
    gc.collect()

    # Remove duplicates at month boundaries
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()

    # Recompute rolling features across full series
    result['ofi_ma_1h'] = result['ofi'].rolling(4).mean()
    result['ofi_ma_4h'] = result['ofi'].rolling(16).mean()

    net_flow = result['buy_volume'] - result['sell_volume']
    net_flow_norm = net_flow / result['volume'].replace(0, np.nan).apply(np.sqrt)
    result['kyle_lambda'] = (
        result['log_ret'].rolling(4).cov(net_flow_norm) /
        net_flow_norm.rolling(4).var().replace(0, np.nan)
    )

    ret_cov = result['log_ret'].rolling(10).cov(result['log_ret'].shift(1))
    result['roll_spread'] = 2 * np.sqrt((-ret_cov).clip(lower=0))

    result = result.replace([np.inf, -np.inf], np.nan)

    result.to_parquet(bars_cache)
    print(f"Saved {len(result)} bars to {bars_cache}")
    return result


def compute_vpin_from_months(
    symbol: str,
    sample_months: list = None,
    n_buckets_window: int = 50,
) -> pd.Series:
    """Compute VPIN from per-month parquet files."""
    vpin_cache = CACHE_DIR / f"{symbol}_vpin.parquet"

    if vpin_cache.exists():
        print(f"Loading cached VPIN from {vpin_cache}")
        vpin_df = pd.read_parquet(vpin_cache)
        return vpin_df.iloc[:, 0]

    month_dir = PARSED_DIR / symbol

    if sample_months is None:
        sample_months = ['2023-03', '2023-07', '2023-11', '2024-01', '2024-05', '2024-09']

    # Load sampled months
    dfs = []
    for m in sample_months:
        mf = month_dir / f"{symbol}_{m}.parquet"
        if mf.exists():
            df = pd.read_parquet(mf)
            dfs.append(df)
            print(f"  Loaded {m}: {len(df):,} trades")

    if not dfs:
        raise RuntimeError("No data for VPIN")

    trades = pd.concat(dfs, ignore_index=True).sort_values('transact_time')
    del dfs
    gc.collect()

    print(f"Computing VPIN on {len(trades):,} trades...")
    vpin = compute_vpin(trades, n_buckets_window=n_buckets_window)

    vpin.to_frame('vpin').to_parquet(vpin_cache)
    print(f"Saved VPIN to {vpin_cache}")

    del trades
    gc.collect()
    return vpin


if __name__ == "__main__":
    for symbol in ['SOLUSDT', 'BTCUSDT']:
        print(f"\n{'='*60}")
        print(f"{symbol}")
        print('='*60)
        bars = compute_bars_from_months(symbol, freq='15min')
        print(f"Total bars: {len(bars):,}")
        print(f"Range: {bars.index.min()} to {bars.index.max()}")
        print(f"Columns: {list(bars.columns)}")

        # Also compute VPIN (sampled months only)
        vpin = compute_vpin_from_months(symbol)
        print(f"VPIN points: {len(vpin):,}")
