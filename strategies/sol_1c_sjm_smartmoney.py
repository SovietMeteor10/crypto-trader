"""
SOL 1C + SJM + BTC smart money divergence entry gate.

Identical to SOL1C_SJM except:
- Long entries only allowed when BTC smart_dumb_div > threshold
- Short entries only allowed when BTC smart_dumb_div < -threshold
- All V3 SJM parameters frozen from sjm_results.json
- One new parameter: smart_div_threshold
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from strategies.sol_1c_sjm import SOL1C_SJM
from pathlib import Path

BTC_STRUCTURE_PATH = Path(
    "/home/ubuntu/projects/crypto-trader/data_cache/market_structure/"
    "BTC_USDT_USDT_unified_1h.parquet"
)


def load_btc_smart_div() -> pd.Series:
    """Load BTC smart_dumb_div from market structure features."""
    df = pd.read_parquet(BTC_STRUCTURE_PATH, columns=['smart_dumb_div'])
    df.index = pd.DatetimeIndex(df.index)
    return df['smart_dumb_div']


class SOL1C_SJM_SmartMoney(SOL1C_SJM):
    """SOL 1C + SJM + BTC smart money divergence gate."""

    def __init__(self, btc_data, feature_set='A', funding_rates=None,
                 use_sol_features=False, n_regimes=3, fixed_sol_params=None):
        super().__init__(btc_data, feature_set, funding_rates,
                         use_sol_features, n_regimes, fixed_sol_params)
        self._btc_smart_div = load_btc_smart_div()

    @property
    def name(self) -> str:
        return "sol_1c_sjm_smartmoney"

    @property
    def parameter_space(self) -> dict:
        base = super().parameter_space
        base['smart_div_threshold'] = ("float", -0.3, 0.3)
        return base

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """Generate V3 signal with smart money divergence gate."""
        # Get base V3 signal (includes SJM regime gating)
        base_signal = super().generate(data, params)

        # Align BTC smart_dumb_div to SOL 4H index via forward fill
        smart_div = self._btc_smart_div.reindex(data.index, method='ffill')
        threshold = params.get('smart_div_threshold', 0.0)

        # Apply smart money gate
        gated = base_signal.copy()

        # Long entries: only when smart_div > threshold
        long_blocked = (base_signal > 0) & (
            smart_div.isna() | (smart_div <= threshold)
        )
        # Short entries: only when smart_div < -threshold
        short_blocked = (base_signal < 0) & (
            smart_div.isna() | (smart_div >= -threshold)
        )

        # Don't block entries where smart_div data is unavailable
        # (training period 2021-2022 has no market structure data)
        has_data = ~smart_div.isna()
        gated[long_blocked & has_data] = 0
        gated[short_blocked & has_data] = 0

        return gated.fillna(0).astype(int)
