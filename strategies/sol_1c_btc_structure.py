"""
SOL 1C with BTC market structure SJM regime filter.

Replaces SOL price-derived SJM features with BTC market structure:
  smart_dumb_div, ls_ratio, ls_vs_ma, taker_momentum, oi_vs_ma

The SJM fits on these BTC structural features instead of price features,
labelling regimes by mean smart_dumb_div per cluster.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.signal_module import SignalModule
from regime.sjm import StatisticalJumpModel
from regime.features import standardise
from pathlib import Path

BTC_STRUCTURE_PATH = Path(
    "/home/ubuntu/projects/crypto-trader/data_cache/market_structure/"
    "BTC_USDT_USDT_unified_1h.parquet"
)

# BTC structure features to use for SJM
STRUCTURE_COLS = ['smart_dumb_div', 'ls_ratio', 'ls_vs_ma', 'taker_momentum', 'oi_vs_ma']


def load_btc_structure_4h() -> pd.DataFrame:
    """Load BTC structure features resampled to 4H."""
    df = pd.read_parquet(BTC_STRUCTURE_PATH)
    df.index = pd.DatetimeIndex(df.index)
    # Resample 1H → 4H: take last value in each 4H window
    avail_cols = [c for c in STRUCTURE_COLS if c in df.columns]
    return df[avail_cols].resample('4h').last().dropna(how='all')


class SOL1C_BTCStructure(SignalModule):
    """
    SOL Trend + ADX with BTC market structure SJM regime filter.

    Uses BTC structural positioning features instead of price features
    for the SJM regime classifier.
    """

    def __init__(self, btc_data: pd.DataFrame):
        self.btc_data = btc_data
        self._btc_structure = load_btc_structure_4h()

    @property
    def name(self) -> str:
        return "sol_1c_btc_structure"

    @property
    def parameter_space(self) -> dict:
        return {
            "fast_period":      ("int",   5,  50),
            "slow_period":      ("int",  50, 200),
            "adx_period":       ("int",   7,  30),
            "adx_threshold":    ("int",  15, 40),
            "sjm_lambda":       ("float", 0.01, 5.0),
            "sjm_window":       ("int",   60, 500),
            "trade_in_neutral": ("categorical", [True, False]),
        }

    @staticmethod
    def _compute_adx(data: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = data['high'], data['low'], data['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm <= minus_dm)] = 0
        minus_dm[(minus_dm <= plus_dm)] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.fillna(0)

    def _get_regime_series(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """Compute regime labels using SJM on BTC market structure features."""
        sjm_window = params['sjm_window']
        sjm_lambda = params['sjm_lambda']

        # Align BTC structure to SOL 4H index
        features_raw = self._btc_structure.reindex(data.index, method='ffill')
        common_idx = data.index.intersection(features_raw.dropna(how='all').index)

        if len(common_idx) < 60:
            return pd.Series('unknown', index=data.index)

        features_clean = features_raw.loc[common_idx].fillna(0)
        features_std = standardise(features_clean)

        regime_labels = pd.Series('unknown', index=data.index)
        min_fit_bars = 60
        refit_interval = max(sjm_window // 3, 30)
        n = len(common_idx)

        refit_points = list(range(min_fit_bars, n, refit_interval))
        if not refit_points:
            return regime_labels

        for rp_idx, rp in enumerate(refit_points):
            start_idx = max(0, rp - sjm_window)
            window_features = features_std.iloc[start_idx:rp].values

            if len(window_features) < min_fit_bars:
                continue

            sjm = StatisticalJumpModel(n_regimes=3, jump_penalty=sjm_lambda)
            sjm.fit(window_features)

            # Label regimes by mean smart_dumb_div in each cluster
            # (positive = top traders long = bull)
            if 'smart_dumb_div' in features_clean.columns:
                sd_values = features_clean['smart_dumb_div'].iloc[start_idx:rp].values
            else:
                sd_values = features_clean.iloc[:, 0].iloc[start_idx:rp].values

            regimes_in_window = sjm.result_.regimes
            regime_means = {}
            for rid in np.unique(regimes_in_window):
                mask = regimes_in_window == rid
                regime_means[rid] = float(np.mean(sd_values[mask]))

            sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
            label_map = {}
            if len(sorted_regimes) >= 3:
                label_map[sorted_regimes[0][0]] = 'bear'
                label_map[sorted_regimes[1][0]] = 'neutral'
                label_map[sorted_regimes[2][0]] = 'bull'
            elif len(sorted_regimes) == 2:
                label_map[sorted_regimes[0][0]] = 'bear'
                label_map[sorted_regimes[1][0]] = 'bull'
            else:
                label_map[sorted_regimes[0][0]] = 'neutral'

            next_rp = refit_points[rp_idx + 1] if rp_idx + 1 < len(refit_points) else n
            predict_slice = features_std.iloc[rp:next_rp].values

            if len(predict_slice) == 0:
                continue

            regime_ids = sjm.predict(predict_slice)
            for j, rid in enumerate(regime_ids):
                bar_idx = rp + j
                label = label_map.get(rid, 'unknown')
                loc = data.index.get_loc(common_idx[bar_idx])
                regime_labels.iloc[loc] = label

        return regime_labels

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """Generate SOL 1C signal gated by BTC structure SJM regime."""
        fast_p = params['fast_period']
        slow_p = params['slow_period']
        if fast_p >= slow_p:
            fast_p, slow_p = min(fast_p, slow_p), max(fast_p, slow_p)

        close = data['close']
        fast_ma = close.ewm(span=fast_p, adjust=False).mean()
        slow_ma = close.ewm(span=slow_p, adjust=False).mean()
        adx = self._compute_adx(data, params['adx_period'])

        trend_signal = pd.Series(0, index=data.index, dtype=int)
        trending = adx > params['adx_threshold']
        trend_signal[(fast_ma > slow_ma) & trending] = 1
        trend_signal[(fast_ma < slow_ma) & trending] = -1

        regimes = self._get_regime_series(data, params)

        gated = trend_signal.copy()
        bear_mask = regimes == 'bear'
        unknown_mask = regimes == 'unknown'

        if not params.get('trade_in_neutral', True):
            neutral_mask = regimes == 'neutral'
            gated[bear_mask | neutral_mask] = 0
        else:
            gated[bear_mask] = 0

        gated[unknown_mask] = 0
        return gated.fillna(0).astype(int)
