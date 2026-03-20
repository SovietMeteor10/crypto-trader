"""Strategy 7A: Improved Funding Carry with Stops."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class BasisCarrySignal(SignalModule):
    @property
    def name(self) -> str:
        return "7A_basis_carry"

    @property
    def parameter_space(self) -> dict:
        return {
            "entry_funding_threshold": ("float", 0.0002, 0.001),
            "exit_funding_threshold": ("float", -0.0001, 0.0002),
            "price_stop_pct": ("float", 0.5, 3.0),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        entry_thresh = params["entry_funding_threshold"]
        exit_thresh = params["exit_funding_threshold"]
        stop_pct = params["price_stop_pct"]

        # 8h return as funding proxy, then 24h rolling mean (3 x 8h periods)
        ret_8h = data["close"].pct_change(8)
        # 24h ~ 24 bars (assuming 1h bars) rolling mean
        funding_proxy = ret_8h.rolling(24, min_periods=1).mean()

        result = pd.Series(0, index=data.index, dtype=int)
        position = 0
        entry_price = 0.0

        for i in range(len(data)):
            fp = funding_proxy.iloc[i]
            close_i = data["close"].iloc[i]

            if np.isnan(fp):
                result.iloc[i] = 0
                continue

            # Check stop loss
            if position != 0 and entry_price > 0:
                pct_move = (close_i - entry_price) / entry_price * 100.0
                if position == 1 and pct_move < -stop_pct:
                    position = 0
                elif position == -1 and pct_move > stop_pct:
                    position = 0

            # Check exit on funding proxy crossing exit threshold
            if position == -1 and fp < exit_thresh:
                position = 0
            elif position == 1 and fp > -exit_thresh:
                position = 0

            # Entry signals
            if position == 0:
                if fp > entry_thresh:
                    position = -1
                    entry_price = close_i
                elif fp < -entry_thresh:
                    position = 1
                    entry_price = close_i

            result.iloc[i] = position

        return result
