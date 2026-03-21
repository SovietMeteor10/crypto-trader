"""Experiment 3: BTC Lead-Lag at 5-15 Minute Bars.
Based on Kurihara & Matsumoto (2026): BTC Granger-causes altcoin returns."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class BTCLeadLagSignal(SignalModule):
    def __init__(self, btc_data: pd.DataFrame):
        self.btc_data = btc_data

    @property
    def name(self): return "btc_lead_lag_5m"

    @property
    def parameter_space(self):
        return {
            "btc_lookback": ("int", 1, 6),
            "btc_threshold": ("float", 0.001, 0.008),
            "hold_bars": ("int", 1, 12),
            "min_btc_vol_mult": ("float", 1.0, 3.0),
        }

    def generate(self, data, params):
        btc = self.btc_data.reindex(data.index, method='ffill')
        btc_return = btc['close'].pct_change(params['btc_lookback'])

        btc_vol = btc['close'].pct_change().rolling(24).std()
        btc_vol_median = btc_vol.rolling(288).median()
        vol_elevated = btc_vol > btc_vol_median * params['min_btc_vol_mult']

        signal = pd.Series(0, index=data.index, dtype=int)
        position_direction = 0
        bars_held = 0

        start_idx = max(params['btc_lookback'] + 1, 289)
        for i in range(start_idx, len(data)):
            if position_direction != 0:
                bars_held += 1
                if bars_held >= params['hold_bars']:
                    signal.iloc[i] = 0
                    position_direction = 0
                    bars_held = 0
                    continue

            btc_ret = btc_return.iloc[i]
            if pd.isna(btc_ret) or not vol_elevated.iloc[i]:
                signal.iloc[i] = position_direction
                continue

            if btc_ret > params['btc_threshold']:
                if position_direction != 1:
                    position_direction = 1
                    bars_held = 0
                signal.iloc[i] = 1
            elif btc_ret < -params['btc_threshold']:
                if position_direction != -1:
                    position_direction = -1
                    bars_held = 0
                signal.iloc[i] = -1
            else:
                signal.iloc[i] = position_direction

        return signal.fillna(0).astype(int)
