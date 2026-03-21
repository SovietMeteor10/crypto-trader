"""Experiment 1: Large-Move Mean Reversion at 2H timeframe.
Based on Ferretti & Puca (2021): negative autocorrelation peaks at 2H,
larger moves mean-revert more strongly."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class LargeMoveReversionSignal(SignalModule):
    @property
    def name(self): return "large_move_reversion_2h"

    @property
    def parameter_space(self):
        return {
            "atr_period": ("int", 8, 48),
            "entry_threshold": ("float", 1.0, 3.5),
            "max_hold_bars": ("int", 1, 8),
        }

    def generate(self, data, params):
        atr_period = params["atr_period"]
        entry_threshold = params["entry_threshold"]
        max_hold_bars = params["max_hold_bars"]

        hl = data['high'] - data['low']
        hc = (data['high'] - data['close'].shift(1)).abs()
        lc = (data['low'] - data['close'].shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        bar_move = (data['close'] - data['open']) / atr

        signal = pd.Series(0, index=data.index, dtype=int)
        position_direction = 0
        bars_in_position = 0
        entry_midpoint = None

        for i in range(atr_period, len(data)):
            if position_direction != 0:
                bars_in_position += 1
                current_price = data['close'].iloc[i]

                exit_by_time = bars_in_position >= max_hold_bars
                exit_by_reversion = (
                    (position_direction == 1 and current_price >= entry_midpoint) or
                    (position_direction == -1 and current_price <= entry_midpoint)
                )

                if exit_by_time or exit_by_reversion:
                    signal.iloc[i] = 0
                    position_direction = 0
                    bars_in_position = 0
                    entry_midpoint = None
                else:
                    signal.iloc[i] = position_direction

            if position_direction == 0:
                move = bar_move.iloc[i]
                if pd.isna(move):
                    continue
                if move > entry_threshold:
                    signal.iloc[i] = -1
                    position_direction = -1
                    bars_in_position = 0
                    entry_midpoint = (data['open'].iloc[i] + data['close'].iloc[i]) / 2
                elif move < -entry_threshold:
                    signal.iloc[i] = 1
                    position_direction = 1
                    bars_in_position = 0
                    entry_midpoint = (data['open'].iloc[i] + data['close'].iloc[i]) / 2

        return signal.fillna(0).astype(int)
