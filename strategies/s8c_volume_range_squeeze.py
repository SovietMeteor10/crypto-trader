"""Strategy 8C: Volume Surge + Range Contraction Squeeze."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class VolumeRangeSqueezeSignal(SignalModule):
    @property
    def name(self) -> str:
        return "8C_volume_range_squeeze"

    @property
    def parameter_space(self) -> dict:
        return {
            "vol_surge_mult": ("float", 1.5, 3.0),
            "range_contraction_pct": ("float", 0.3, 0.7),
            "lookback": ("int", 8, 24),
            "hold_bars": ("int", 4, 16),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        vol_surge_mult = params["vol_surge_mult"]
        range_contraction_pct = params["range_contraction_pct"]
        lookback = params["lookback"]
        hold_bars = params["hold_bars"]

        volume = data["volume"]
        bar_range = data["high"] - data["low"]

        vol_avg = volume.rolling(lookback, min_periods=1).mean()
        range_avg = bar_range.rolling(lookback, min_periods=1).mean()

        result = pd.Series(0, index=data.index, dtype=int)

        position = 0
        hold_countdown = 0
        squeeze_high = np.nan
        squeeze_low = np.nan
        squeeze_detected = False

        for i in range(len(data)):
            if hold_countdown > 0:
                hold_countdown -= 1
                if hold_countdown == 0:
                    position = 0

            vol_i = volume.iloc[i]
            range_i = bar_range.iloc[i]
            vol_avg_i = vol_avg.iloc[i]
            range_avg_i = range_avg.iloc[i]
            close_i = data["close"].iloc[i]

            if np.isnan(vol_avg_i) or np.isnan(range_avg_i) or range_avg_i == 0:
                result.iloc[i] = position
                continue

            # Detect squeeze: volume surge AND range contraction
            vol_surging = vol_i > vol_surge_mult * vol_avg_i
            range_contracted = range_i < range_contraction_pct * range_avg_i

            if vol_surging and range_contracted and position == 0 and hold_countdown == 0:
                squeeze_high = data["high"].iloc[i]
                squeeze_low = data["low"].iloc[i]
                squeeze_detected = True
            elif squeeze_detected and position == 0:
                # Trade breakout of squeeze bar range
                if close_i > squeeze_high:
                    position = 1
                    hold_countdown = hold_bars
                    squeeze_detected = False
                elif close_i < squeeze_low:
                    position = -1
                    hold_countdown = hold_bars
                    squeeze_detected = False

            result.iloc[i] = position

        return result
