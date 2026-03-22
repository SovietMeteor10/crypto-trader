"""
Multi-timeframe V3: Daily filter + 4H V3 signal + 1H ls_ratio gate.
All V3 SJM parameters frozen. Only new MTF parameters optimised.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from strategies.sol_1c_sjm import SOL1C_SJM


class SOL1C_SJM_MTF(SOL1C_SJM):

    def __init__(self, btc_data, daily_ohlcv, ms_1h,
                 feature_set='A', funding_rates=None,
                 use_sol_features=False, n_regimes=3, fixed_sol_params=None):
        super().__init__(btc_data, feature_set, funding_rates,
                         use_sol_features, n_regimes, fixed_sol_params)
        self.daily_ohlcv = daily_ohlcv
        self.ms_1h = ms_1h

    @property
    def name(self):
        return "sol_1c_sjm_mtf"

    @property
    def parameter_space(self):
        base = super().parameter_space
        base.update({
            "daily_ma_period": ("int", 20, 100),
            "daily_buffer_pct": ("float", 0.0, 1.0),
            "ls_quantile_high": ("float", 0.70, 0.90),
            "ls_quantile_low": ("float", 0.10, 0.30),
            "ls_rolling_window": ("int", 48, 336),
            "use_ls_filter": ("categorical", [True, False]),
        })
        return base

    def generate(self, data, params):
        # Merge fixed SOL params
        if self.fixed_sol_params:
            params = {**self.fixed_sol_params, **params}

        # Get base V3 signal
        base_signal = super().generate(data, params)

        # Layer 1: Daily macro direction
        daily = self.daily_ohlcv['close']
        period = params['daily_ma_period']
        buffer = params['daily_buffer_pct'] / 100.0
        daily_ma = daily.rolling(period).mean()

        daily_dir = pd.Series(0, index=daily.index)
        daily_dir[daily > daily_ma * (1 + buffer)] = 1
        daily_dir[daily < daily_ma * (1 - buffer)] = -1
        daily_dir_4h = daily_dir.reindex(data.index, method='ffill').fillna(0)

        # Layer 3: LS ratio gate
        use_ls = params.get('use_ls_filter', True)
        if use_ls and 'ls_ratio' in self.ms_1h.columns:
            ls = self.ms_1h['ls_ratio']
            window = params['ls_rolling_window']
            q_high = ls.rolling(window).quantile(params['ls_quantile_high'])
            q_low = ls.rolling(window).quantile(params['ls_quantile_low'])
            ls_4h = ls.reindex(data.index, method='ffill')
            q_high_4h = q_high.reindex(data.index, method='ffill')
            q_low_4h = q_low.reindex(data.index, method='ffill')
        else:
            use_ls = False

        # Apply filters
        filtered = base_signal.copy()

        # Daily filter: suppress signals conflicting with macro direction
        filtered[(daily_dir_4h == 1) & (base_signal == -1)] = 0
        filtered[(daily_dir_4h == -1) & (base_signal == 1)] = 0
        filtered[daily_dir_4h == 0] = 0

        # LS filter: suppress entries where crowd is extreme in signal direction
        if use_ls:
            filtered[(filtered == 1) & (ls_4h > q_high_4h)] = 0
            filtered[(filtered == -1) & (ls_4h < q_low_4h)] = 0

        return filtered.fillna(0).astype(int)
