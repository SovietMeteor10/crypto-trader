"""Strategy 2B: Bollinger Band Reversion."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class BollingerReversionSignal(SignalModule):
    @property
    def name(self) -> str:
        return "2B_bollinger_reversion"

    @property
    def parameter_space(self) -> dict:
        return {
            "period": ("int", 10, 50),
            "std_dev": ("float", 1.5, 3.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        mid = data["close"].rolling(params["period"]).mean()
        std = data["close"].rolling(params["period"]).std()
        upper = mid + params["std_dev"] * std
        lower = mid - params["std_dev"] * std

        position = 0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(params["period"], len(data)):
            price = data["close"].iloc[i]
            m = mid.iloc[i]
            u = upper.iloc[i]
            l = lower.iloc[i]

            if np.isnan(m):
                result.iloc[i] = position
                continue

            if position == 0:
                if price < l:
                    position = 1  # long at lower band
                elif price > u:
                    position = -1  # short at upper band
            elif position == 1:
                if price >= m:
                    position = 0  # exit at mid
            elif position == -1:
                if price <= m:
                    position = 0  # exit at mid

            result.iloc[i] = position

        return result
