"""Strategy 4A: Volatility Breakout."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class VolBreakoutSignal(SignalModule):
    @property
    def name(self) -> str:
        return "4A_vol_breakout"

    @property
    def parameter_space(self) -> dict:
        return {
            "vol_lookback": ("int", 24, 168),
            "spike_mult": ("float", 1.5, 3.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        returns = data["close"].pct_change()
        realised_vol = returns.rolling(params["vol_lookback"]).std()
        vol_mean = realised_vol.rolling(params["vol_lookback"]).mean()
        vol_std = realised_vol.rolling(params["vol_lookback"]).std()

        position = 0
        result = pd.Series(0, index=data.index, dtype=int)
        cooldown = 0

        for i in range(params["vol_lookback"] * 2, len(data)):
            rv = realised_vol.iloc[i]
            vm = vol_mean.iloc[i]
            vs = vol_std.iloc[i]

            if np.isnan(rv) or np.isnan(vm) or np.isnan(vs) or vs == 0:
                result.iloc[i] = position
                continue

            if cooldown > 0:
                cooldown -= 1
                result.iloc[i] = position
                continue

            vol_zscore = (rv - vm) / vs

            if vol_zscore > params["spike_mult"] and position == 0:
                # Direction from recent price action
                ret_short = data["close"].iloc[i] / data["close"].iloc[i - 4] - 1
                if ret_short > 0:
                    position = 1
                else:
                    position = -1
                cooldown = params["vol_lookback"] // 4
            elif position != 0:
                # Exit when vol normalises
                if vol_zscore < 0.5:
                    position = 0

            result.iloc[i] = position

        return result
