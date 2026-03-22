"""
Order flow feature engineering from Binance aggTrades data.

Computes:
- Order Flow Imbalance (OFI) at multiple frequencies
- Kyle's Lambda (price impact per unit flow)
- Roll Measure (spread proxy from serial covariance)
- Trade arrival rate and size distribution
- VPIN (volume-synchronised informed trading probability)
- Amihud illiquidity ratio
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_bar_features(
    trades: pd.DataFrame,
    freq: str = '15min',
) -> pd.DataFrame:
    """
    Aggregate raw trades into bar-level order flow features.

    Parameters
    ----------
    trades : pd.DataFrame
        Raw aggTrades with columns: transact_time, price, qty, is_buyer_maker
    freq : str
        Bar frequency: '1min', '15min', '1h', '4h'

    Returns
    -------
    pd.DataFrame with index=bar_timestamp and columns:
        open, high, low, close, volume,
        buy_volume, sell_volume, n_trades,
        ofi, ofi_ma, kyle_lambda, roll_spread,
        arrival_rate, large_trade_pct, amihud
    """
    trades = trades.set_index('transact_time').sort_index()

    # Classify trades
    # is_buyer_maker=True: seller aggressed (sell trade)
    # is_buyer_maker=False: buyer aggressed (buy trade)
    trades['buy_vol'] = trades['qty'].where(~trades['is_buyer_maker'], 0)
    trades['sell_vol'] = trades['qty'].where(trades['is_buyer_maker'], 0)
    trades['dollar_vol'] = trades['qty'] * trades['price']

    # Resample to bars
    bars = trades.resample(freq).agg(
        open=('price', 'first'),
        high=('price', 'max'),
        low=('price', 'min'),
        close=('price', 'last'),
        volume=('qty', 'sum'),
        dollar_volume=('dollar_vol', 'sum'),
        buy_volume=('buy_vol', 'sum'),
        sell_volume=('sell_vol', 'sum'),
        n_trades=('qty', 'count'),
    ).dropna(subset=['open'])

    # Order Flow Imbalance
    total_vol = bars['volume'].replace(0, np.nan)
    bars['ofi'] = (bars['buy_volume'] - bars['sell_volume']) / total_vol

    # Rolling OFI (smoothed signal)
    bars['ofi_ma_1h'] = bars['ofi'].rolling(4).mean()   # 4 bars at 15min = 1h
    bars['ofi_ma_4h'] = bars['ofi'].rolling(16).mean()  # 16 bars = 4h

    # Kyle's Lambda: price_change / net_order_flow
    # Use log returns for price change
    bars['log_ret'] = np.log(bars['close'] / bars['close'].shift(1))
    net_flow = (bars['buy_volume'] - bars['sell_volume'])
    # Normalise by sqrt(volume) to account for varying bar sizes
    net_flow_norm = net_flow / bars['volume'].replace(0, np.nan).apply(np.sqrt)
    # Rolling 4-bar estimate
    bars['kyle_lambda'] = (
        bars['log_ret'].rolling(4).cov(net_flow_norm) /
        net_flow_norm.rolling(4).var().replace(0, np.nan)
    )

    # Roll Measure: 2 * sqrt(-Cov(r_t, r_{t-1}))
    # Negative serial covariance of returns = bid-ask bounce
    ret_cov = bars['log_ret'].rolling(10).cov(bars['log_ret'].shift(1))
    bars['roll_spread'] = 2 * np.sqrt((-ret_cov).clip(lower=0))

    # Trade arrival rate (trades per minute)
    freq_minutes = pd.Timedelta(freq).total_seconds() / 60
    bars['arrival_rate'] = bars['n_trades'] / freq_minutes

    # Average trade size
    bars['avg_trade_size'] = bars['volume'] / bars['n_trades'].replace(0, np.nan)

    # Amihud Illiquidity: |return| / dollar_volume
    bars['amihud'] = bars['log_ret'].abs() / bars['dollar_volume'].replace(0, np.nan)

    # Clean infinite and extreme values
    bars = bars.replace([np.inf, -np.inf], np.nan)

    return bars


def compute_bar_features_chunked(
    parquet_path: str,
    freq: str = '15min',
    chunk_days: int = 7,
) -> pd.DataFrame:
    """
    Compute bar features from a large parquet file in chunks.
    Processes chunk_days at a time to manage memory.
    """
    import pyarrow.parquet as pq

    print(f"Loading metadata from {parquet_path}...")
    df = pd.read_parquet(parquet_path, columns=['transact_time'])
    min_date = df['transact_time'].min()
    max_date = df['transact_time'].max()
    del df

    print(f"Date range: {min_date} to {max_date}")

    all_bars = []
    current = min_date.normalize()
    end = max_date.normalize() + pd.Timedelta(days=1)

    while current < end:
        chunk_end = current + pd.Timedelta(days=chunk_days)
        print(f"Processing {current.date()} to {chunk_end.date()}...", end=" ", flush=True)

        # Read chunk with filter
        chunk = pd.read_parquet(
            parquet_path,
            filters=[
                ('transact_time', '>=', current),
                ('transact_time', '<', chunk_end),
            ]
        )

        if len(chunk) > 0:
            bars = compute_bar_features(chunk, freq=freq)
            all_bars.append(bars)
            print(f"{len(chunk):,} trades -> {len(bars)} bars")
        else:
            print("no data")

        del chunk
        current = chunk_end

    result = pd.concat(all_bars)
    # Remove duplicate indices from chunk boundaries
    result = result[~result.index.duplicated(keep='first')]
    result = result.sort_index()

    # Recompute rolling features across full series (chunk boundaries broke them)
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

    print(f"\nTotal: {len(result)} bars from {result.index.min()} to {result.index.max()}")
    return result


def compute_vpin(
    trades: pd.DataFrame,
    volume_bucket_size: Optional[float] = None,
    n_buckets_window: int = 50,
) -> pd.Series:
    """
    Compute VPIN (Volume-Synchronised Probability of Informed Trading).

    Uses vectorised cumulative volume approach for performance.

    VPIN = mean |V_buy - V_sell| / V_bucket over rolling n_buckets_window
    """
    trades = trades.sort_values('transact_time').copy()
    trades['buy_vol'] = trades['qty'].where(~trades['is_buyer_maker'], 0)
    trades['sell_vol'] = trades['qty'].where(trades['is_buyer_maker'], 0)

    if volume_bucket_size is None:
        trades['date'] = trades['transact_time'].dt.date
        daily_vol = trades.groupby('date')['qty'].sum().mean()
        volume_bucket_size = daily_vol / 50
        print(f"VPIN bucket size: {volume_bucket_size:.2f} (1/50 daily volume)")

    # Vectorised bucket assignment using cumulative volume
    cum_vol = trades['qty'].cumsum()
    trades['bucket_id'] = (cum_vol / volume_bucket_size).astype(int)

    # Aggregate by bucket
    bucket_agg = trades.groupby('bucket_id').agg(
        end_time=('transact_time', 'last'),
        buy_vol=('buy_vol', 'sum'),
        sell_vol=('sell_vol', 'sum'),
        total_vol=('qty', 'sum'),
    )

    bucket_agg['imbalance'] = (bucket_agg['buy_vol'] - bucket_agg['sell_vol']).abs()
    bucket_agg['vpin'] = (
        bucket_agg['imbalance'].rolling(n_buckets_window).mean() /
        volume_bucket_size
    )

    result = bucket_agg.set_index('end_time')['vpin'].dropna()
    return result


def compute_vpin_chunked(
    parquet_path: str,
    volume_bucket_size: Optional[float] = None,
    n_buckets_window: int = 50,
    sample_months: Optional[list] = None,
) -> pd.Series:
    """
    Compute VPIN from large parquet in chunks.
    If sample_months provided, only use those months (e.g. ['2023-07', '2024-01']).
    """
    print(f"Computing VPIN from {parquet_path}...")

    if sample_months:
        # Load only specific months for speed
        df = pd.read_parquet(parquet_path)
        month_str = df['transact_time'].dt.to_period('M').astype(str)
        mask = month_str.isin(sample_months)
        trades = df[mask].copy()
        del df
        print(f"Sampled {len(trades):,} trades from months: {sample_months}")
    else:
        trades = pd.read_parquet(parquet_path)

    return compute_vpin(trades, volume_bucket_size, n_buckets_window)


def compute_kyle_lambda_rolling(
    bars: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Estimate Kyle's lambda via rolling OLS of returns on signed volume.

    Lambda_t = Cov(r, signed_vol) / Var(signed_vol) over window bars

    Higher lambda = more price-sensitive market (thinner liquidity)
    """
    signed_vol = bars['buy_volume'] - bars['sell_volume']
    returns = bars['close'].pct_change()

    lambda_series = pd.Series(np.nan, index=bars.index)

    for i in range(window, len(bars)):
        r = returns.iloc[i-window:i].values
        sv = signed_vol.iloc[i-window:i].values

        mask = ~(np.isnan(r) | np.isnan(sv))
        r, sv = r[mask], sv[mask]

        if len(r) < window // 2:
            continue

        cov_rv = np.cov(r, sv)
        if cov_rv[1, 1] > 0:
            lambda_series.iloc[i] = cov_rv[0, 1] / cov_rv[1, 1]

    return lambda_series
