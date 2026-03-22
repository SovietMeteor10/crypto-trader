"""
SOL 1C with Statistical Jump Model regime filter.

Strategy logic:
- Use existing SOL 1C (Trend + ADX) signal
- Gate with SJM regime: only enter positions in bull or neutral regime
- In bear regime: close any open position, emit signal = 0

The SJM is fitted on a rolling window of BTC features (train period only).
For walk-forward: re-fit SJM at each window boundary using training data.
At inference time: assign current bar to nearest centroid (greedy).
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.signal_module import SignalModule
from regime.sjm import StatisticalJumpModel
from regime.features import compute_feature_set_A, compute_feature_set_B, standardise


class SOL1C_SJM(SignalModule):
    """
    SOL Trend + ADX with SJM regime filter.

    Parameters (passed via params dict):
    - fast_period: fast EMA period (bars)
    - slow_period: slow EMA period (bars)
    - adx_period: ADX calculation period (bars)
    - adx_threshold: minimum ADX to enter trend trade
    - sjm_lambda: jump penalty for SJM
    - sjm_window: rolling window length (bars) for SJM fitting
    - trade_in_neutral: whether to trade in neutral regime

    Requires btc_data to be passed at construction time.
    """

    def __init__(self, btc_data: pd.DataFrame, feature_set: str = 'A',
                 funding_rates: pd.Series = None, use_sol_features: bool = False,
                 n_regimes: int = 3,
                 fixed_sol_params: dict = None):
        self.btc_data = btc_data
        self.feature_set = feature_set
        self.funding_rates = funding_rates
        self.use_sol_features = use_sol_features
        self.n_regimes = n_regimes
        self.fixed_sol_params = fixed_sol_params  # when set, only SJM params are optimized

    @property
    def name(self) -> str:
        return "sol_1c_sjm_regime"

    @property
    def parameter_space(self) -> dict:
        if self.fixed_sol_params:
            # Only expose SJM parameters for optimization
            return {
                "sjm_lambda":       ("float", 0.01, 5.0),
                "sjm_window":       ("int",   182, 720),
                "trade_in_neutral": ("categorical", [True, False]),
            }
        return {
            # SOL 1C parameters
            "fast_period":    ("int",   5,  50),
            "slow_period":    ("int",  50, 200),  # min=50 to prevent fast > slow
            "adx_period":     ("int",   7,  30),
            "adx_threshold":  ("int",  15, 40),
            # SJM parameters
            "sjm_lambda":     ("float", 0.01, 5.0),
            "sjm_window":     ("int",   182, 720),
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

    def _compute_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """Compute features based on config."""
        if self.feature_set == 'B' and self.funding_rates is not None:
            return compute_feature_set_B(ohlcv_data, self.funding_rates)
        return compute_feature_set_A(ohlcv_data)

    def _get_regime_series(
        self,
        data: pd.DataFrame,
        params: dict,
    ) -> pd.Series:
        """
        Compute regime labels using rolling SJM.
        Fits SJM on windows and batch-predicts forward bars.
        Returns: Series of 'bull', 'neutral', 'bear', or 'unknown'.
        """
        sjm_window = params['sjm_window']
        sjm_lambda = params['sjm_lambda']

        # Use SOL data for features if configured, otherwise BTC
        if self.use_sol_features:
            feature_source = data
        else:
            feature_source = self.btc_data.reindex(data.index, method='ffill').dropna()

        features_raw = self._compute_features(feature_source)

        common_idx = data.index.intersection(features_raw.index)
        if len(common_idx) == 0:
            return pd.Series('unknown', index=data.index)

        features_std = standardise(features_raw.loc[common_idx])

        regime_labels = pd.Series('unknown', index=data.index)
        min_fit_bars = 90

        # Fit SJM on windows, predict batch of bars until next refit
        refit_interval = max(sjm_window // 3, 60)
        n = len(common_idx)

        # Build list of refit points
        refit_points = list(range(min_fit_bars, n, refit_interval))
        if not refit_points:
            return regime_labels

        for rp_idx, rp in enumerate(refit_points):
            start_idx = max(0, rp - sjm_window)
            window_features = features_std.iloc[start_idx:rp].values

            if len(window_features) < min_fit_bars:
                continue

            sjm = StatisticalJumpModel(
                n_regimes=self.n_regimes,
                jump_penalty=sjm_lambda,
            )
            sjm.fit(window_features)

            window_returns = features_raw['mom_1d'].iloc[start_idx:rp].values
            label_map = sjm.label_regimes(sjm.result_.regimes, window_returns)

            # Predict from rp to next refit point (or end)
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
        """Generate SOL 1C signal gated by SJM regime."""
        # Merge fixed SOL params if set
        if self.fixed_sol_params:
            params = {**self.fixed_sol_params, **params}

        # Enforce fast < slow
        fast_p = params['fast_period']
        slow_p = params['slow_period']
        if fast_p >= slow_p:
            fast_p, slow_p = min(fast_p, slow_p), max(fast_p, slow_p)

        # --- SOL 1C signal ---
        close = data['close']
        fast_ma = close.ewm(span=fast_p, adjust=False).mean()
        slow_ma = close.ewm(span=slow_p, adjust=False).mean()
        adx = self._compute_adx(data, params['adx_period'])

        trend_signal = pd.Series(0, index=data.index, dtype=int)
        trending = adx > params['adx_threshold']
        trend_signal[(fast_ma > slow_ma) & trending] = 1
        trend_signal[(fast_ma < slow_ma) & trending] = -1

        # --- SJM regime gate ---
        regimes = self._get_regime_series(data, params)

        # Gate: zero out signal in bear regime
        gated_signal = trend_signal.copy()
        bear_mask = regimes == 'bear'

        if not params.get('trade_in_neutral', True):
            neutral_mask = regimes == 'neutral'
            gated_signal[bear_mask | neutral_mask] = 0
        else:
            gated_signal[bear_mask] = 0

        # Zero out where regime is unknown (insufficient data)
        unknown_mask = regimes == 'unknown'
        gated_signal[unknown_mask] = 0

        return gated_signal.fillna(0).astype(int)
