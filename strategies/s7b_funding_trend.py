"""Strategy 7B: Funding Rate Momentum via Price Momentum."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class FundingTrendSignal(SignalModule):
    @property
    def name(self) -> str:
        return "7B_funding_trend"

    @property
    def parameter_space(self) -> dict:
        return {
            "lookback_periods": ("int", 3, 12),
            "trend_threshold": ("float", 0.00005, 0.0003),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        lookback = params["lookback_periods"]
        threshold = params["trend_threshold"]

        # 8h momentum readings
        momentum_8h = data["close"].pct_change(8)

        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(len(data)):
            if i < 8 + lookback:
                continue

            # Check if the last `lookback` momentum readings are monotonically
            # increasing or decreasing (each differs from previous by > threshold)
            readings = []
            for k in range(lookback + 1):
                idx = i - lookback + k
                val = momentum_8h.iloc[idx]
                if np.isnan(val):
                    break
                readings.append(val)

            if len(readings) < lookback + 1:
                continue

            # Check consecutive increases
            increasing = True
            decreasing = True
            for j in range(1, len(readings)):
                if readings[j] - readings[j - 1] < threshold:
                    increasing = False
                if readings[j - 1] - readings[j] < threshold:
                    decreasing = False

            if increasing:
                # Crowded longs — go short
                result.iloc[i] = -1
            elif decreasing:
                # Crowded shorts — go long
                result.iloc[i] = 1

        return result
