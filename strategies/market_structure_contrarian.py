"""
Standalone contrarian strategy using L/S ratio and smart money divergence.
Runs on BTC at 1H frequency.

Signal logic:
- When ls_ratio > rolling 85th percentile (crowd maximally long): short
- When ls_ratio < rolling 15th percentile (crowd maximally short): long
- Optional smart money confirmation
- Exit after max_hold_bars or when crowd normalises
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.signal_module import SignalModule
from pathlib import Path

BTC_STRUCTURE_PATH = Path(
    "/home/ubuntu/projects/crypto-trader/data_cache/market_structure/"
    "BTC_USDT_USDT_unified_1h.parquet"
)


def load_btc_structure() -> pd.DataFrame:
    """Load BTC market structure features at 1H."""
    df = pd.read_parquet(BTC_STRUCTURE_PATH)
    df.index = pd.DatetimeIndex(df.index)
    return df


class MarketStructureContrarian(SignalModule):
    """
    Contrarian strategy: fade the crowd when positioning is extreme.
    """

    def __init__(self):
        self._structure = load_btc_structure()

    @property
    def name(self) -> str:
        return "market_structure_contrarian"

    @property
    def parameter_space(self) -> dict:
        return {
            "crowd_quantile_high": ("float", 0.75, 0.95),
            "crowd_quantile_low": ("float", 0.05, 0.25),
            "rolling_window": ("int", 48, 336),
            "max_hold_bars": ("int", 1, 12),
            "require_smart_confirm": ("categorical", [True, False]),
            "smart_div_threshold": ("float", -0.2, 0.2),
        }

    def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
        """Generate contrarian signal from crowd positioning."""
        # Align structure data to OHLCV index
        struct = self._structure.reindex(data.index, method='ffill')

        ls = struct.get('ls_ratio')
        smart_div = struct.get('smart_dumb_div')

        signal = pd.Series(0, index=data.index, dtype=int)

        if ls is None or ls.isna().all():
            return signal

        window = params['rolling_window']
        q_high = params['crowd_quantile_high']
        q_low = params['crowd_quantile_low']
        max_hold = params['max_hold_bars']
        use_smart = params['require_smart_confirm']
        smart_thresh = params['smart_div_threshold']

        # Rolling quantiles
        rolling_high = ls.rolling(window).quantile(q_high)
        rolling_low = ls.rolling(window).quantile(q_low)

        # Entry conditions
        crowd_long = ls > rolling_high   # crowd maximally long → short signal
        crowd_short = ls < rolling_low   # crowd maximally short → long signal

        # Smart money confirmation
        if use_smart and smart_div is not None:
            smart_confirms_short = smart_div < smart_thresh
            smart_confirms_long = smart_div > -smart_thresh
            short_entry = crowd_long & smart_confirms_short
            long_entry = crowd_short & smart_confirms_long
        else:
            short_entry = crowd_long
            long_entry = crowd_short

        # Generate signal with max_hold_bars duration
        # Track position state
        position = 0  # 0=flat, 1=long, -1=short
        hold_counter = 0

        for i in range(len(data)):
            if position != 0:
                hold_counter += 1
                # Exit conditions: max hold reached OR crowd normalised
                crowd_normalised = not (crowd_long.iloc[i] or crowd_short.iloc[i])
                if hold_counter >= max_hold or crowd_normalised:
                    position = 0
                    hold_counter = 0
                    signal.iloc[i] = 0
                else:
                    signal.iloc[i] = position
            else:
                if short_entry.iloc[i]:
                    position = -1
                    hold_counter = 0
                    signal.iloc[i] = -1
                elif long_entry.iloc[i]:
                    position = 1
                    hold_counter = 0
                    signal.iloc[i] = 1
                else:
                    signal.iloc[i] = 0

        return signal
