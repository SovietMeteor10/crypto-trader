"""Test helpers — trivial signal modules for testing infrastructure."""

import pandas as pd
from crypto_infra.signal_module import SignalModule


class AlwaysLongSignal(SignalModule):
    """Always returns +1 (long)."""

    @property
    def name(self) -> str:
        return "always_long"

    @property
    def parameter_space(self) -> dict:
        return {"dummy": ("int", 1, 10)}

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(1, index=data.index, dtype=int)


class AlwaysFlatSignal(SignalModule):
    """Always returns 0 (flat)."""

    @property
    def name(self) -> str:
        return "always_flat"

    @property
    def parameter_space(self) -> dict:
        return {"dummy": ("int", 1, 10)}

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        return pd.Series(0, index=data.index, dtype=int)


class SimpleMACrossSignal(SignalModule):
    """Simple MA crossover for testing optimisation."""

    @property
    def name(self) -> str:
        return "simple_ma_cross"

    @property
    def parameter_space(self) -> dict:
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 100),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        fast = data["close"].rolling(params["fast_period"]).mean()
        slow = data["close"].rolling(params["slow_period"]).mean()
        signal = pd.Series(0, index=data.index, dtype=int)
        signal[fast > slow] = 1
        signal[fast < slow] = -1
        # Fill NaN from rolling with 0
        signal = signal.fillna(0).astype(int)
        return signal


class FragileSignal(SignalModule):
    """Signal that is extremely sensitive to parameter changes."""

    @property
    def name(self) -> str:
        return "fragile"

    @property
    def parameter_space(self) -> dict:
        return {"magic_period": ("int", 10, 50)}

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        period = params["magic_period"]
        signal = pd.Series(0, index=data.index, dtype=int)
        # Only go long on exact period match, random-ish behaviour
        for i in range(len(data)):
            if i % period == 0:
                signal.iloc[i] = 1
            elif i % (period + 1) == 0:
                signal.iloc[i] = -1
        return signal
