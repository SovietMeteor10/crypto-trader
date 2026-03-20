"""Strategy 1C: Trend Following with ADX Regime Filter."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule

class TrendRegimeSignal(SignalModule):
    @property
    def name(self) -> str:
        return "1C_trend_regime"

    @property
    def parameter_space(self) -> dict:
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 200),
            "adx_period": ("int", 10, 30),
            "adx_threshold": ("int", 20, 40),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        fast = data["close"].ewm(span=params["fast_period"], adjust=False).mean()
        slow = data["close"].ewm(span=params["slow_period"], adjust=False).mean()

        # ADX calculation
        adx = self._compute_adx(data, params["adx_period"])

        signal = pd.Series(0, index=data.index, dtype=int)
        trending = adx > params["adx_threshold"]

        signal[(fast > slow) & trending] = 1
        signal[(fast < slow) & trending] = -1

        return signal

    @staticmethod
    def _compute_adx(data: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = data["high"], data["low"], data["close"]
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        plus_dm[(plus_dm <= minus_dm)] = 0
        minus_dm[(minus_dm <= plus_dm)] = 0

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.fillna(0)
