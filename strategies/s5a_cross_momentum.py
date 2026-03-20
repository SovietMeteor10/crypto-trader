"""Strategy 5A: Cross-Asset Momentum (single-asset implementation — ranks by momentum)."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class CrossMomentumSignal(SignalModule):
    """
    Single-asset version: goes long if asset's momentum is positive,
    short if negative. Scaled by momentum strength.
    For true cross-asset, would need multi-symbol backtest support.
    """
    @property
    def name(self) -> str:
        return "5A_cross_momentum"

    @property
    def parameter_space(self) -> dict:
        return {
            "momentum_period": ("int", 24, 168),
            "rebalance_bars": ("int", 4, 24),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        mom = data["close"].pct_change(params["momentum_period"])

        position = 0
        bars_since_rebal = 0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(params["momentum_period"], len(data)):
            bars_since_rebal += 1

            if bars_since_rebal >= params["rebalance_bars"]:
                m = mom.iloc[i]
                if np.isnan(m):
                    result.iloc[i] = position
                    continue
                if m > 0:
                    position = 1
                elif m < 0:
                    position = -1
                else:
                    position = 0
                bars_since_rebal = 0

            result.iloc[i] = position

        return result
