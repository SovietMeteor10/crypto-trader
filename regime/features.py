"""
Feature engineering for regime detection.
All features are standardised (zero mean, unit variance) before SJM fitting.
"""

import numpy as np
import pandas as pd


def compute_feature_set_A(btc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal set — BTC-only momentum and volatility features.
    Based on Cortese et al. finding that momentum and market activity
    are the most important features for crypto regime detection.
    """
    df = btc_df.copy()
    close = df['close']
    volume = df['volume']

    features = pd.DataFrame(index=df.index)

    # Momentum at multiple horizons (bars, at 4H bars: 1d=6, 1w=42, 1m=182)
    features['mom_1d'] = close.pct_change(6)
    features['mom_1w'] = close.pct_change(42)
    features['mom_1m'] = close.pct_change(182)

    # Realised volatility (annualised)
    ret = close.pct_change()
    features['rvol_1w'] = ret.rolling(42).std() * np.sqrt(252 * 6)
    features['rvol_1m'] = ret.rolling(182).std() * np.sqrt(252 * 6)

    # Volume activity (relative to rolling mean)
    features['vol_ratio'] = volume / volume.rolling(42).mean()

    # Trend strength (ADX proxy: ratio of directional movement to range)
    high, low = df['high'], df['low']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    features['adx_proxy'] = (
        dm_plus.rolling(14).mean() - dm_minus.rolling(14).mean()
    ) / tr.rolling(14).mean().replace(0, np.nan)

    # Drawdown from recent high (captures bear regime)
    rolling_max = close.rolling(182).max()
    features['drawdown'] = (close - rolling_max) / rolling_max

    return features.dropna()


def compute_feature_set_B(btc_df: pd.DataFrame, funding_rates: pd.Series) -> pd.DataFrame:
    """
    Extended set — adds funding rate signal.
    Per Koki et al., funding rate predicts regime transitions in crypto.
    """
    features = compute_feature_set_A(btc_df)

    # Align funding rates to bar index
    funding_aligned = funding_rates.reindex(btc_df.index, method='ffill')
    funding_aligned = funding_aligned.loc[features.index]

    # Funding rate level and trend
    features['funding_level'] = funding_aligned
    features['funding_ma'] = funding_aligned.rolling(42).mean()
    features['funding_trend'] = funding_aligned - funding_aligned.rolling(42).mean()

    return features.dropna()


def standardise(features: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardise all features. Required before SJM fitting."""
    return (features - features.mean()) / features.std().replace(0, 1)
