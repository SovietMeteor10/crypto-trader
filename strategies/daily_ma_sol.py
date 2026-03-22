"""
Simplest possible daily MA strategy on SOL 4H.
Long when daily close > MA, short when below. Two parameters only.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.signal_module import SignalModule


class DailyMASOL(SignalModule):

    def __init__(self, daily_ohlcv: pd.DataFrame):
        self.daily_ohlcv = daily_ohlcv

    @property
    def name(self):
        return "daily_ma_sol"

    @property
    def parameter_space(self):
        return {
            "ma_period": ("int", 10, 60),
            "buffer_pct": ("float", 0.0, 2.0),
        }

    def generate(self, data, params):
        daily = self.daily_ohlcv['close']
        period = params['ma_period']
        buffer = params['buffer_pct'] / 100.0

        daily_ma = daily.rolling(period).mean()

        daily_signal = pd.Series(0, index=daily.index)
        daily_signal[daily > daily_ma * (1 + buffer)] = 1
        daily_signal[daily < daily_ma * (1 - buffer)] = -1

        # Forward fill to 4H index
        signal_4h = daily_signal.reindex(data.index, method='ffill').fillna(0)
        return signal_4h.astype(int)
