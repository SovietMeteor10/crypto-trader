"""Strategy 2A: RSI Mean Reversion."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class RSIReversionSignal(SignalModule):
    @property
    def name(self) -> str:
        return "2A_rsi_reversion"

    @property
    def parameter_space(self) -> dict:
        return {
            "rsi_period": ("int", 7, 21),
            "oversold": ("int", 20, 40),
            "overbought": ("int", 60, 80),
            "max_hold_bars": ("int", 4, 24),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        rsi = self._compute_rsi(data["close"], params["rsi_period"])

        position = 0
        bars_held = 0
        result = pd.Series(0, index=data.index, dtype=int)

        for i in range(len(data)):
            r = rsi.iloc[i]
            if np.isnan(r):
                result.iloc[i] = 0
                continue

            if position != 0:
                bars_held += 1
                # Exit conditions
                if bars_held >= params["max_hold_bars"]:
                    position = 0
                    bars_held = 0
                elif position == 1 and r >= 50:
                    position = 0
                    bars_held = 0
                elif position == -1 and r <= 50:
                    position = 0
                    bars_held = 0

            if position == 0:
                if r < params["oversold"]:
                    position = 1
                    bars_held = 0
                elif r > params["overbought"]:
                    position = -1
                    bars_held = 0

            result.iloc[i] = position

        return result

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
