"""Strategy 1C Regime Filter Variants for SOL."""
import pandas as pd
import numpy as np
from crypto_infra.signal_module import SignalModule


class TrendRegimeVolFilter(SignalModule):
    """1C-RF-A: Trend + ADX + volatility regime filter. Only trade in low vol."""
    @property
    def name(self): return "1C_RF_A_vol_filter"
    @property
    def parameter_space(self):
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 200),
            "adx_period": ("int", 10, 30),
            "adx_threshold": ("int", 20, 40),
            "vol_lookback": ("int", 60, 180),
        }

    def generate(self, data, params):
        fast = data["close"].ewm(span=params["fast_period"], adjust=False).mean()
        slow = data["close"].ewm(span=params["slow_period"], adjust=False).mean()
        adx = self._compute_adx(data, params["adx_period"])

        # Vol filter: only trade when realised vol < 90-day median
        returns = data["close"].pct_change()
        vol_20 = returns.rolling(20).std()
        vol_median = vol_20.rolling(params["vol_lookback"]).median()

        trending = adx > params["adx_threshold"]
        low_vol = vol_20 <= vol_median

        signal = pd.Series(0, index=data.index, dtype=int)
        signal[(fast > slow) & trending & low_vol] = 1
        signal[(fast < slow) & trending & low_vol] = -1
        return signal

    @staticmethod
    def _compute_adx(data, period):
        high, low, close = data["high"], data["low"], data["close"]
        plus_dm = high.diff(); minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm < 0] = 0
        plus_dm[plus_dm <= minus_dm] = 0; minus_dm[minus_dm <= plus_dm] = 0
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        return dx.ewm(span=period, adjust=False).mean().fillna(0)


class TrendRegimeBTCFilter(SignalModule):
    """1C-RF-C: Trend + ADX on SOL, but only when BTC is in uptrend."""
    @property
    def name(self): return "1C_RF_C_btc_filter"
    @property
    def parameter_space(self):
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 200),
            "adx_period": ("int", 10, 30),
            "adx_threshold": ("int", 20, 40),
        }

    def generate(self, data, params):
        fast = data["close"].ewm(span=params["fast_period"], adjust=False).mean()
        slow = data["close"].ewm(span=params["slow_period"], adjust=False).mean()
        adx = TrendRegimeVolFilter._compute_adx(data, params["adx_period"])

        # BTC filter: use SOL data's own long-term trend as proxy
        # (Cannot fetch BTC inside generate — use 200-bar MA of SOL as regime proxy)
        ma50 = data["close"].rolling(50).mean()
        ma200 = data["close"].rolling(200).mean()
        bull_regime = ma50 > ma200

        trending = adx > params["adx_threshold"]

        signal = pd.Series(0, index=data.index, dtype=int)
        # In bull regime: allow both long and short
        # In bear regime: only allow short (follow the trend down)
        signal[(fast > slow) & trending & bull_regime] = 1
        signal[(fast < slow) & trending] = -1
        return signal


class TrendRegimeFundingFilter(SignalModule):
    """1C-RF-D: Trend + ADX + funding rate filter (avoid crowded trades)."""
    @property
    def name(self): return "1C_RF_D_funding_filter"
    @property
    def parameter_space(self):
        return {
            "fast_period": ("int", 5, 50),
            "slow_period": ("int", 20, 200),
            "adx_period": ("int", 10, 30),
            "adx_threshold": ("int", 20, 40),
            "funding_lookback": ("int", 30, 120),
        }

    def generate(self, data, params):
        fast = data["close"].ewm(span=params["fast_period"], adjust=False).mean()
        slow = data["close"].ewm(span=params["slow_period"], adjust=False).mean()
        adx = TrendRegimeVolFilter._compute_adx(data, params["adx_period"])

        # Funding proxy: use 8h momentum as funding rate approximation
        mom_8 = data["close"].pct_change(8)
        funding_median = mom_8.rolling(params["funding_lookback"]).median()

        trending = adx > params["adx_threshold"]
        # Only long when funding below median (not crowded long)
        not_crowded_long = mom_8 <= funding_median
        # Only short when funding above median (not crowded short)
        not_crowded_short = mom_8 >= funding_median

        signal = pd.Series(0, index=data.index, dtype=int)
        signal[(fast > slow) & trending & not_crowded_long] = 1
        signal[(fast < slow) & trending & not_crowded_short] = -1
        return signal
