"""
Feature engineering for LightGBM market structure model.
Decision frequency: 4H bars.
Features combine 1H market structure (aggregated) + 4H OHLCV + regime.

Critical: all features are computed at bar T.
The model predicts the NEXT bar (T+1).
The 1-bar lag is applied in build_feature_matrix(), not here.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')


def load_market_structure(symbol: str) -> pd.DataFrame:
    safe = symbol.replace('/', '_').replace(':', '_')
    path = Path(f"/home/ubuntu/projects/crypto-trader/data_cache/market_structure/{safe}_unified_1h.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df


def aggregate_1h_to_4h(ms_1h: pd.DataFrame, ohlcv_4h: pd.DataFrame) -> pd.DataFrame:
    idx_4h = ohlcv_4h.index
    result = pd.DataFrame(index=idx_4h)

    cols = ['ls_ratio', 'smart_dumb_div', 'taker_ratio', 'oi_chg_1h',
            'basis_vs_ma', 'ls_vs_ma', 'taker_momentum', 'funding',
            'crowd_long', 'crowd_short', 'oi_vs_ma']
    available = [c for c in cols if c in ms_1h.columns]

    for col in available:
        series = ms_1h[col]
        agg_last = series.resample('4h').last()
        agg_mean = series.resample('4h').mean()
        agg_change = series.resample('4h').last() - series.resample('4h').first()
        agg_std = series.resample('4h').std()

        result[f'{col}_last'] = agg_last.reindex(idx_4h, method='ffill')
        result[f'{col}_mean'] = agg_mean.reindex(idx_4h, method='ffill')
        result[f'{col}_change'] = agg_change.reindex(idx_4h, method='ffill')
        result[f'{col}_std'] = agg_std.reindex(idx_4h, method='ffill')

    return result


def compute_ohlcv_features(ohlcv_4h: pd.DataFrame) -> pd.DataFrame:
    close = ohlcv_4h['close']
    high = ohlcv_4h['high']
    low = ohlcv_4h['low']
    vol = ohlcv_4h['volume']
    ret = close.pct_change()

    f = pd.DataFrame(index=ohlcv_4h.index)

    for n, label in [(1, '4h'), (3, '12h'), (6, '1d'), (18, '3d'),
                     (42, '1w'), (90, '2w'), (180, '1m')]:
        f[f'ret_{label}'] = close.pct_change(n)

    for n, label in [(6, '1d'), (42, '1w'), (180, '1m')]:
        f[f'rvol_{label}'] = ret.rolling(n).std() * np.sqrt(252 * 6)

    f['rvol_ratio'] = f['rvol_1d'] / f['rvol_1w'].replace(0, np.nan)

    # ADX
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                     (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    dip = dm_plus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
    dim = dm_minus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
    dx = ((dip - dim).abs() / (dip + dim).replace(0, np.nan)) * 100
    f['adx_14'] = dx.ewm(span=14, adjust=False).mean()
    f['di_diff'] = dip - dim

    for fast, slow in [(12, 26), (50, 200)]:
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        f[f'ma_cross_{fast}_{slow}'] = (ema_f - ema_s) / ema_s * 100

    f['vol_ratio_1w'] = vol / vol.rolling(42).mean().replace(0, np.nan)
    f['vol_trend'] = vol.rolling(6).mean() / vol.rolling(42).mean().replace(0, np.nan)

    roll_high = high.rolling(42).max()
    roll_low = low.rolling(42).min()
    f['price_position'] = (close - roll_low) / (roll_high - roll_low).replace(0, np.nan)

    f['body_ratio'] = (close - ohlcv_4h['open']).abs() / tr.replace(0, np.nan)
    f['body_direction'] = np.sign(close - ohlcv_4h['open'])

    return f


def compute_regime_features(ohlcv_4h: pd.DataFrame, btc_ms_1h: pd.DataFrame = None) -> pd.DataFrame:
    close = ohlcv_4h['close']
    f = pd.DataFrame(index=ohlcv_4h.index)

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    f['btc_macro_trend'] = (ma50 > ma200).astype(int)
    f['btc_macro_strength'] = (ma50 - ma200) / ma200 * 100

    rvol_short = close.pct_change().rolling(6).std()
    rvol_long = close.pct_change().rolling(180).std()
    f['vol_regime'] = (rvol_short > rvol_long).astype(int)
    f['vol_percentile'] = rvol_short.rolling(180).rank(pct=True)

    ret = close.pct_change()
    f['trend_regime'] = ret.rolling(42).mean().abs() / ret.rolling(42).std().replace(0, np.nan)

    roll_max = close.rolling(180).max()
    f['drawdown_pct'] = (close - roll_max) / roll_max * 100

    if btc_ms_1h is not None and 'smart_dumb_div' in btc_ms_1h.columns:
        sd_4h = btc_ms_1h['smart_dumb_div'].resample('4h').last()
        sd_4h = sd_4h.reindex(ohlcv_4h.index, method='ffill')
        f['btc_smart_div_regime'] = (sd_4h > 0).astype(int)
        f['btc_smart_div_level'] = sd_4h

        ls_4h = btc_ms_1h['ls_ratio'].resample('4h').last()
        ls_4h = ls_4h.reindex(ohlcv_4h.index, method='ffill')
        f['btc_crowd_level'] = ls_4h
        f['btc_crowd_extreme_long'] = (ls_4h > ls_4h.rolling(168).quantile(0.85)).astype(int)
        f['btc_crowd_extreme_short'] = (ls_4h < ls_4h.rolling(168).quantile(0.15)).astype(int)

    return f


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df['hour_sin'] = np.sin(2 * np.pi * idx.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * idx.hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return df


def build_feature_matrix(
    symbol_futures='SOL/USDT:USDT',
    symbol_btc='BTC/USDT:USDT',
    start='2022-01-01',
    end='2024-12-31',
    target_horizon=1,
    target_threshold=0.003,
):
    from crypto_infra import DataModule
    dm = DataModule()

    sol_4h = dm.get_ohlcv(symbol_futures, '4h', start, end)
    btc_4h = dm.get_ohlcv(symbol_btc, '4h', start, end)
    sol_ms = load_market_structure(symbol_futures)
    btc_ms = load_market_structure(symbol_btc)

    ms_feats = aggregate_1h_to_4h(sol_ms, sol_4h)
    ohlcv_feats = compute_ohlcv_features(sol_4h)
    regime_feats = compute_regime_features(btc_4h, btc_ms)

    X = pd.concat([ms_feats, ohlcv_feats, regime_feats], axis=1)
    X = add_time_features(X)

    fwd_ret = sol_4h['close'].pct_change().shift(-target_horizon)
    y = pd.Series(0, index=X.index)
    y[fwd_ret > target_threshold] = 1
    y[fwd_ret < -target_threshold] = -1

    df = X.copy()
    df['target'] = y
    df['fwd_ret'] = fwd_ret

    feature_cols = list(X.columns)
    df[feature_cols] = df[feature_cols].shift(1)
    df = df.dropna()

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    print(f"Feature matrix: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"Features: {len(feature_cols)}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df, feature_cols
