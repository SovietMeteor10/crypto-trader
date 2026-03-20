"""Strategy 6B: Funding Payment Momentum — trade against crowded funding."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class FundingMomentumSignal(SignalModule):
    @property
    def name(self) -> str:
        return "6B_funding_momentum"

    @property
    def parameter_space(self) -> dict:
        return {
            "funding_threshold": ("float", 0.0002, 0.001),
            "entry_delay_bars": ("int", 1, 3),
            "hold_bars": ("int", 2, 8),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        threshold = params["funding_threshold"]
        entry_delay = params["entry_delay_bars"]
        hold_bars = params["hold_bars"]

        # Approximate funding pressure from 8-hour return momentum
        momentum_8h = data["close"].pct_change(8)

        result = pd.Series(0, index=data.index, dtype=int)

        pending_signal = 0
        pending_countdown = 0
        hold_countdown = 0
        position = 0

        for i in range(len(data)):
            mom = momentum_8h.iloc[i]
            if np.isnan(mom):
                result.iloc[i] = 0
                continue

            # If currently holding, count down
            if hold_countdown > 0:
                hold_countdown -= 1
                if hold_countdown == 0:
                    position = 0

            # If waiting for entry delay
            if pending_countdown > 0:
                pending_countdown -= 1
                if pending_countdown == 0 and position == 0:
                    position = pending_signal
                    hold_countdown = hold_bars
                    pending_signal = 0

            # Generate new signal only when flat and not pending
            if position == 0 and pending_countdown == 0:
                if mom > threshold:
                    # High funding likely — go short
                    pending_signal = -1
                    pending_countdown = entry_delay
                elif mom < -threshold:
                    # Negative funding likely — go long
                    pending_signal = 1
                    pending_countdown = entry_delay

            result.iloc[i] = position

        return result
