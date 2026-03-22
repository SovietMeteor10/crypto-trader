"""
SOL 1C SJM with pullback entry logic.

Entry conditions (all must be true):
1. SJM regime is bull or neutral (unchanged from V3)
2. Fast MA > Slow MA (uptrend) or Fast MA < Slow MA (downtrend)
3. ADX > adx_threshold (trend is strong)
4. Price has pulled back to within pullback_atr_mult * ATR of the fast MA
5. The pullback bar's close is moving back in the trend direction
   (i.e. a bullish close on a pullback in an uptrend)

Exit conditions (unchanged from V3):
- MA crossover in opposite direction
- ADX drops below adx_threshold
- OR: regime turns bear (SJM gate)
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from strategies.sol_1c_sjm import SOL1C_SJM


class SOL1C_SJM_Pullback(SOL1C_SJM):
    """V3 strategy with pullback entry filter."""

    @property
    def name(self) -> str:
        return "sol_1c_sjm_pullback"

    @property
    def parameter_space(self) -> dict:
        base = super().parameter_space
        base.update({
            "pullback_atr_mult": ("float", 0.1, 1.5),
            "require_reversal_bar": ("categorical", [True, False]),
        })
        return base

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        # Get base V3 signal (includes SJM regime gating)
        base_signal = super().generate(data, params)

        # Merge fixed SOL params if set
        if self.fixed_sol_params:
            merged = {**self.fixed_sol_params, **params}
        else:
            merged = params

        close = data['close']
        high = data['high']
        low = data['low']

        fast_ma = close.ewm(span=merged['fast_period'], adjust=False).mean()

        # ATR for pullback distance measurement
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(merged['adx_period']).mean()

        # Pullback condition: price within pullback_atr_mult * ATR of fast MA
        dist_from_ma = (close - fast_ma).abs()
        pullback_atr = params.get('pullback_atr_mult', 0.5)
        near_ma = dist_from_ma <= (atr * pullback_atr)

        # Reversal bar confirmation
        bullish_bar = close > data['open']
        bearish_bar = close < data['open']

        require_reversal = params.get('require_reversal_bar', True)

        # The pullback filter modifies ENTRY only.
        # We need to track position state to distinguish entry bars from hold bars.
        # When base_signal transitions from 0 to non-zero, that's an entry.
        # When base_signal stays the same non-zero value, that's a hold.
        # We only apply the pullback filter on entry bars.
        pullback_signal = pd.Series(0, index=data.index, dtype=int)
        in_position = 0  # current position direction

        for i in range(1, len(data)):
            base = int(base_signal.iloc[i])

            if base == 0:
                # Base says flat — always flat
                in_position = 0
                pullback_signal.iloc[i] = 0
                continue

            if in_position == base:
                # Already in a position in same direction — this is a HOLD bar
                # Keep the position (no pullback filter on holds)
                pullback_signal.iloc[i] = base
                continue

            if in_position == -base:
                # Direction flip — treat as exit then new entry attempt
                in_position = 0

            # This is an ENTRY bar (in_position == 0 and base != 0)
            # Apply pullback filter
            if pd.isna(atr.iloc[i]) or atr.iloc[i] == 0:
                pullback_signal.iloc[i] = 0
                continue

            if not near_ma.iloc[i]:
                # Price too far from MA — skip entry, stay flat
                pullback_signal.iloc[i] = 0
                continue

            if require_reversal:
                if base == 1 and not bullish_bar.iloc[i]:
                    pullback_signal.iloc[i] = 0
                    continue
                if base == -1 and not bearish_bar.iloc[i]:
                    pullback_signal.iloc[i] = 0
                    continue

            # Passed all filters — enter
            in_position = base
            pullback_signal.iloc[i] = base

        return pullback_signal
