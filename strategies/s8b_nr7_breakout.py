"""Strategy 8B: Narrow Range 7 Breakout."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class NR7BreakoutSignal(SignalModule):
    @property
    def name(self) -> str:
        return "8B_nr7_breakout"

    @property
    def parameter_space(self) -> dict:
        return {
            "lookback": ("int", 5, 10),
            "breakout_pct": ("float", 0.1, 0.5),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        lookback = params["lookback"]
        breakout_pct = params["breakout_pct"]

        bar_range = data["high"] - data["low"]

        result = pd.Series(0, index=data.index, dtype=int)

        position = 0
        nr_high = np.nan
        nr_low = np.nan
        watching_breakout = False

        for i in range(len(data)):
            if i < lookback:
                continue

            current_range = bar_range.iloc[i]
            close_i = data["close"].iloc[i]

            # Check if current bar is NR (narrowest range in last N bars)
            past_ranges = bar_range.iloc[i - lookback + 1:i + 1]
            is_nr = current_range <= past_ranges.min()

            if is_nr:
                # NR bar detected: set up breakout levels, reset position
                nr_high = data["high"].iloc[i]
                nr_low = data["low"].iloc[i]
                watching_breakout = True
                position = 0
            elif watching_breakout:
                # After NR bar, watch for breakout
                nr_range = nr_high - nr_low
                confirm_dist = nr_range * breakout_pct / 100.0 if nr_range > 0 else 0

                if position == 0:
                    if close_i > nr_high + confirm_dist:
                        position = 1
                    elif close_i < nr_low - confirm_dist:
                        position = -1

            result.iloc[i] = position

        return result
