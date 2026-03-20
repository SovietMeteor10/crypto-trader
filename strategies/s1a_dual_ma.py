"""Strategy 1A: Dual Moving Average Crossover with ATR stops."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class DualMACrossSignal(SignalModule):
    @property
    def name(self) -> str:
        return "1A_dual_ma_cross"

    @property
    def parameter_space(self) -> dict:
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 200),
            "atr_mult": ("float", 1.0, 4.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        fast = data["close"].ewm(span=params["fast_period"], adjust=False).mean()
        slow = data["close"].ewm(span=params["slow_period"], adjust=False).mean()

        signal = pd.Series(0, index=data.index, dtype=int)
        signal[fast > slow] = 1
        signal[fast < slow] = -1

        # ATR trailing stop — flatten if price moves against by atr_mult * ATR
        atr_period = 14
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_period).mean()

        position = 0
        entry_price = 0.0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(len(data)):
            sig = signal.iloc[i]
            price = data["close"].iloc[i]
            current_atr = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0

            if position != 0 and current_atr > 0:
                stop_dist = params["atr_mult"] * current_atr
                if position == 1 and price < entry_price - stop_dist:
                    position = 0
                elif position == -1 and price > entry_price + stop_dist:
                    position = 0

            if position == 0 and sig != 0:
                position = sig
                entry_price = price
            elif position != 0 and sig != 0 and sig != position:
                position = sig
                entry_price = price

            result.iloc[i] = position

        return result
