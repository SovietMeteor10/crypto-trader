"""Strategy 8A: Volume-Weighted Momentum."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class VolumeWeightedMomentumSignal(SignalModule):
    @property
    def name(self) -> str:
        return "8A_volume_weighted_momentum"

    @property
    def parameter_space(self) -> dict:
        return {
            "volume_lookback": ("int", 24, 168),
            "volume_z_threshold": ("float", 1.0, 3.0),
            "price_move_pct": ("float", 0.3, 2.0),
            "hold_bars": ("int", 2, 12),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        vol_lookback = params["volume_lookback"]
        vol_z_thresh = params["volume_z_threshold"]
        price_move_pct = params["price_move_pct"]
        hold_bars = params["hold_bars"]

        volume = data["volume"]
        vol_mean = volume.rolling(vol_lookback, min_periods=1).mean()
        vol_std = volume.rolling(vol_lookback, min_periods=1).std()
        vol_z = (volume - vol_mean) / vol_std.replace(0, np.nan)
        vol_z = vol_z.fillna(0.0)

        # Price move over last 4 bars
        price_change_4 = data["close"].pct_change(4) * 100.0

        result = pd.Series(0, index=data.index, dtype=int)
        position = 0
        hold_countdown = 0

        for i in range(len(data)):
            if hold_countdown > 0:
                hold_countdown -= 1
                if hold_countdown == 0:
                    position = 0

            if position == 0:
                z = vol_z.iloc[i]
                pm = price_change_4.iloc[i]
                if not np.isnan(z) and not np.isnan(pm):
                    if z > vol_z_thresh and abs(pm) > price_move_pct:
                        position = 1 if pm > 0 else -1
                        hold_countdown = hold_bars

            result.iloc[i] = position

        return result
