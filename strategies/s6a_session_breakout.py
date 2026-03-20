"""Strategy 6A: Session Breakout — trade breakouts of prior session ranges."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class SessionBreakoutSignal(SignalModule):
    @property
    def name(self) -> str:
        return "6A_session_breakout"

    @property
    def parameter_space(self) -> dict:
        return {
            "session": ("categorical", ["asia", "london", "ny"]),
            "confirmation_pct": ("float", 0.1, 0.5),
            "stop_atr_mult": ("float", 1.0, 3.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        session = params["session"]
        confirmation_pct = params["confirmation_pct"]
        stop_atr_mult = params["stop_atr_mult"]

        # Session hour boundaries (UTC)
        session_hours = {
            "asia": (0, 8),
            "london": (8, 16),
            "ny": (13, 21),
        }
        start_h, end_h = session_hours[session]

        # ATR for stops
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - data["close"].shift(1)).abs(),
            (data["low"] - data["close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        hours = data.index.hour if hasattr(data.index, 'hour') else pd.Series(0, index=data.index)

        result = pd.Series(0, index=data.index, dtype=int)

        prev_session_high = np.nan
        prev_session_low = np.nan
        current_session_high = np.nan
        current_session_low = np.nan
        in_session_prev = False

        position = 0
        entry_price = 0.0

        for i in range(len(data)):
            h = hours[i] if hasattr(hours, '__getitem__') else 0
            high_i = data["high"].iloc[i]
            low_i = data["low"].iloc[i]
            close_i = data["close"].iloc[i]
            atr_i = atr.iloc[i] if not np.isnan(atr.iloc[i]) else 0.0

            # Determine if we are inside the session
            if start_h < end_h:
                in_session = start_h <= h < end_h
            else:
                in_session = h >= start_h or h < end_h

            # Track session range
            if in_session:
                if not in_session_prev:
                    # New session started — save previous session range, reset
                    if not np.isnan(current_session_high):
                        prev_session_high = current_session_high
                        prev_session_low = current_session_low
                    current_session_high = high_i
                    current_session_low = low_i
                else:
                    current_session_high = max(current_session_high, high_i)
                    current_session_low = min(current_session_low, low_i)
            else:
                if in_session_prev:
                    # Session just ended — save range
                    if not np.isnan(current_session_high):
                        prev_session_high = current_session_high
                        prev_session_low = current_session_low
                    current_session_high = np.nan
                    current_session_low = np.nan
                    # Reset position at end of session
                    position = 0

            in_session_prev = in_session

            # Check ATR stop
            if position != 0 and atr_i > 0:
                stop_dist = stop_atr_mult * atr_i
                if position == 1 and close_i < entry_price - stop_dist:
                    position = 0
                elif position == -1 and close_i > entry_price + stop_dist:
                    position = 0

            # Breakout logic — only when we have a previous session range
            if not np.isnan(prev_session_high) and not np.isnan(prev_session_low):
                range_size = prev_session_high - prev_session_low
                if range_size > 0:
                    confirm_dist = range_size * confirmation_pct / 100.0
                    if position == 0:
                        if close_i > prev_session_high + confirm_dist:
                            position = 1
                            entry_price = close_i
                        elif close_i < prev_session_low - confirm_dist:
                            position = -1
                            entry_price = close_i

            result.iloc[i] = position

        return result
