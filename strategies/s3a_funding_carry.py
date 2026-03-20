"""Strategy 3A: Funding Rate Carry."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class FundingCarrySignal(SignalModule):
    @property
    def name(self) -> str:
        return "3A_funding_carry"

    @property
    def parameter_space(self) -> dict:
        return {
            "funding_threshold": ("float", 0.0001, 0.001),
            "hold_periods": ("int", 1, 6),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """
        Uses a simple proxy: estimate funding from price momentum.
        When price is trending up strongly, funding is likely positive -> go short.
        When price is trending down strongly, funding is likely negative -> go long.
        Real implementation would use actual funding rate data.
        """
        # Approximate funding signal from 8h momentum
        mom_8 = data["close"].pct_change(8)
        threshold = params["funding_threshold"] * 100  # scale to return pct

        position = 0
        bars_held = 0
        hold_limit = params["hold_periods"] * 8  # convert 8h periods to 1h bars
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(8, len(data)):
            m = mom_8.iloc[i]
            if np.isnan(m):
                result.iloc[i] = position
                continue

            if position != 0:
                bars_held += 1
                if bars_held >= hold_limit:
                    position = 0
                    bars_held = 0

            if position == 0:
                if m > threshold:
                    position = -1  # short when funding likely positive
                    bars_held = 0
                elif m < -threshold:
                    position = 1  # long when funding likely negative
                    bars_held = 0

            result.iloc[i] = position

        return result
