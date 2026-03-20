"""Strategy 1B: Breakout with Volume Confirmation."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class BreakoutVolumeSignal(SignalModule):
    @property
    def name(self) -> str:
        return "1B_breakout_volume"

    @property
    def parameter_space(self) -> dict:
        return {
            "lookback": ("int", 10, 60),
            "vol_mult": ("float", 1.2, 3.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        lb = params["lookback"]
        vm = params["vol_mult"]

        high_roll = data["high"].rolling(lb).max()
        low_roll = data["low"].rolling(lb).min()
        avg_vol = data["volume"].rolling(lb).mean()

        position = 0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(lb, len(data)):
            price = data["close"].iloc[i]
            vol = data["volume"].iloc[i]
            prev_high = high_roll.iloc[i - 1] if i > 0 else high_roll.iloc[i]
            prev_low = low_roll.iloc[i - 1] if i > 0 else low_roll.iloc[i]
            avg_v = avg_vol.iloc[i]

            vol_confirm = vol > vm * avg_v if avg_v > 0 else False

            if price > prev_high and vol_confirm:
                position = 1
            elif price < prev_low and vol_confirm:
                position = -1
            # Stay in position until opposite breakout

            result.iloc[i] = position

        return result
