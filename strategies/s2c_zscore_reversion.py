"""Strategy 2C: Statistical Mean Reversion with Z-Score."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class ZScoreReversionSignal(SignalModule):
    @property
    def name(self) -> str:
        return "2C_zscore_reversion"

    @property
    def parameter_space(self) -> dict:
        return {
            "lookback": ("int", 24, 168),
            "entry_z": ("float", 1.5, 3.0),
            "exit_z": ("float", 0.0, 0.5),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        returns = data["close"].pct_change()
        roll_mean = returns.rolling(params["lookback"]).mean()
        roll_std = returns.rolling(params["lookback"]).std()
        zscore = (returns - roll_mean) / roll_std.replace(0, 1e-10)

        position = 0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(params["lookback"], len(data)):
            z = zscore.iloc[i]
            if np.isnan(z):
                result.iloc[i] = position
                continue

            if position == 0:
                if z < -params["entry_z"]:
                    position = 1  # long on extreme negative z
                elif z > params["entry_z"]:
                    position = -1  # short on extreme positive z
            elif position == 1:
                if z >= -params["exit_z"]:
                    position = 0
            elif position == -1:
                if z <= params["exit_z"]:
                    position = 0

            result.iloc[i] = position

        return result
