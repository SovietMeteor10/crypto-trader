"""
Supertrend strategy on SOL 4H. ATR-based trend following.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.signal_module import SignalModule


class SupertrendSOL(SignalModule):

    def __init__(self, btc_data=None, daily_ohlcv=None):
        self.btc_data = btc_data
        self.daily_ohlcv = daily_ohlcv

    @property
    def name(self):
        return "supertrend_sol"

    @property
    def parameter_space(self):
        return {
            "atr_period": ("int", 7, 21),
            "multiplier": ("float", 1.5, 4.0),
            "use_daily_filter": ("categorical", [True, False]),
            "daily_ma_period": ("int", 20, 100),
            "daily_buffer_pct": ("float", 0.0, 1.0),
        }

    def generate(self, data, params):
        high = data['high']
        low = data['low']
        close = data['close']
        atr_period = params['atr_period']
        multiplier = params['multiplier']

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        hl2 = (high + low) / 2
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        n = len(close)
        final_upper = basic_upper.values.copy()
        final_lower = basic_lower.values.copy()
        supertrend = np.full(n, np.nan)
        signal = np.zeros(n, dtype=int)

        for i in range(1, n):
            # Final upper band
            if basic_upper.iloc[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
                final_upper[i] = basic_upper.iloc[i]
            else:
                final_upper[i] = final_upper[i-1]

            # Final lower band
            if basic_lower.iloc[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
                final_lower[i] = basic_lower.iloc[i]
            else:
                final_lower[i] = final_lower[i-1]

            # Supertrend
            if np.isnan(supertrend[i-1]):
                supertrend[i] = final_upper[i]
                signal[i] = -1
            elif supertrend[i-1] == final_upper[i-1]:
                if close.iloc[i] > final_upper[i]:
                    supertrend[i] = final_lower[i]
                    signal[i] = 1
                else:
                    supertrend[i] = final_upper[i]
                    signal[i] = -1
            else:
                if close.iloc[i] < final_lower[i]:
                    supertrend[i] = final_upper[i]
                    signal[i] = -1
                else:
                    supertrend[i] = final_lower[i]
                    signal[i] = 1

        result = pd.Series(signal, index=data.index)

        # Daily filter (optional)
        if params.get('use_daily_filter', False) and self.daily_ohlcv is not None:
            daily = self.daily_ohlcv['close']
            period = params['daily_ma_period']
            buffer = params.get('daily_buffer_pct', 0.5) / 100.0
            daily_ma = daily.rolling(period).mean()
            daily_dir = pd.Series(0, index=daily.index)
            daily_dir[daily > daily_ma * (1 + buffer)] = 1
            daily_dir[daily < daily_ma * (1 - buffer)] = -1
            dd4h = daily_dir.reindex(data.index, method='ffill').fillna(0)

            result[(dd4h == 1) & (result == -1)] = 0
            result[(dd4h == -1) & (result == 1)] = 0
            result[dd4h == 0] = 0

        return result.fillna(0).astype(int)
