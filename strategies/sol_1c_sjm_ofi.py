"""
SOL 1C with SJM regime filter + OFI neutral-regime entry gate.

Identical to SOL1C_SJM (strategies/sol_1c_sjm.py) except:
- In SJM neutral regime: only enter long when OFI_MA_1H > threshold,
  only enter short when OFI_MA_1H < -threshold
- In SJM bull regime: entry logic unchanged (OFI filter OFF)
- In SJM bear regime: no entry (unchanged)

OFI_MA_1H is loaded from pre-computed 15-min order flow bars,
forward-filled to align with the 4H bar index.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from strategies.sol_1c_sjm import SOL1C_SJM
from pathlib import Path

OFI_BARS_PATH = Path("/home/ubuntu/projects/crypto-trader/data_cache/aggtrades/SOLUSDT_bars_15m.parquet")


def load_ofi_ma_1h() -> pd.Series:
    """Load OFI MA 1H from pre-computed 15-min bars."""
    bars = pd.read_parquet(OFI_BARS_PATH, columns=['ofi_ma_1h'])
    bars.index = pd.DatetimeIndex(bars.index)
    return bars['ofi_ma_1h']


class SOL1C_SJM_OFI(SOL1C_SJM):
    """SOL 1C + SJM + OFI neutral-regime filter."""

    def __init__(self, btc_data, feature_set='A', funding_rates=None,
                 use_sol_features=False, n_regimes=3, fixed_sol_params=None):
        super().__init__(btc_data, feature_set, funding_rates,
                         use_sol_features, n_regimes, fixed_sol_params)
        self._ofi_ma_1h = load_ofi_ma_1h()

    @property
    def name(self) -> str:
        return "sol_1c_sjm_ofi"

    @property
    def parameter_space(self) -> dict:
        base = super().parameter_space
        base['ofi_neutral_threshold'] = ("float", -0.1, 0.1)
        return base

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """Generate SOL 1C signal gated by SJM regime + OFI in neutral."""
        # Merge fixed SOL params if set
        if self.fixed_sol_params:
            merged = {**self.fixed_sol_params, **params}
        else:
            merged = params.copy()

        # Enforce fast < slow
        fast_p = merged['fast_period']
        slow_p = merged['slow_period']
        if fast_p >= slow_p:
            fast_p, slow_p = min(fast_p, slow_p), max(fast_p, slow_p)

        # --- SOL 1C signal ---
        close = data['close']
        fast_ma = close.ewm(span=fast_p, adjust=False).mean()
        slow_ma = close.ewm(span=slow_p, adjust=False).mean()
        adx = self._compute_adx(data, merged['adx_period'])

        trend_signal = pd.Series(0, index=data.index, dtype=int)
        trending = adx > merged['adx_threshold']
        trend_signal[(fast_ma > slow_ma) & trending] = 1
        trend_signal[(fast_ma < slow_ma) & trending] = -1

        # --- SJM regime gate ---
        regimes = self._get_regime_series(data, merged)

        # --- OFI alignment ---
        # Forward-fill 15-min OFI MA to the 4H bar index
        ofi_threshold = params.get('ofi_neutral_threshold', 0.0)
        ofi_aligned = self._ofi_ma_1h.reindex(data.index, method='ffill')

        # --- Apply gates ---
        gated_signal = trend_signal.copy()

        # Bear: always zero
        bear_mask = regimes == 'bear'
        gated_signal[bear_mask] = 0

        # Unknown: always zero
        unknown_mask = regimes == 'unknown'
        gated_signal[unknown_mask] = 0

        # Neutral: apply OFI filter
        neutral_mask = regimes == 'neutral'
        if merged.get('trade_in_neutral', True):
            # OFI filter in neutral regime:
            # Long only when OFI > threshold, short only when OFI < -threshold
            ofi_blocks_long = neutral_mask & (ofi_aligned <= ofi_threshold) & (trend_signal > 0)
            ofi_blocks_short = neutral_mask & (ofi_aligned >= -ofi_threshold) & (trend_signal < 0)
            # Also block when OFI is NaN (no OFI data for this bar, e.g. training period)
            ofi_nan = neutral_mask & ofi_aligned.isna()
            # NaN OFI: let trade through (no filter available, e.g. training period)
            gated_signal[ofi_blocks_long & ~ofi_nan] = 0
            gated_signal[ofi_blocks_short & ~ofi_nan] = 0
        else:
            # trade_in_neutral=False: zero out neutral entirely
            gated_signal[neutral_mask] = 0

        # Bull: no OFI filter (unchanged)

        return gated_signal.fillna(0).astype(int)
