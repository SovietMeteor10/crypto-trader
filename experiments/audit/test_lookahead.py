"""Look-ahead bias test."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import pandas as pd
import numpy as np
from crypto_infra import DataModule, BacktestEngine, CostModule, SizerModule, MetricsModule
from crypto_infra.signal_module import SignalModule

class PerfectFutureSignal(SignalModule):
    @property
    def name(self): return "perfect_future_lookahead"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params):
        future_return = data['close'].shift(-1) / data['close'] - 1
        signal = pd.Series(0, index=data.index, dtype=int)
        signal[future_return > 0] = 1
        signal[future_return < 0] = -1
        signal.iloc[-1] = 0
        return signal

class LagMomentumSignal(SignalModule):
    @property
    def name(self): return "lag1_momentum"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params):
        past_return = data['close'].pct_change()
        signal = pd.Series(0, index=data.index, dtype=int)
        signal[past_return > 0] = 1
        signal[past_return < 0] = -1
        return signal.fillna(0).astype(int)

dm = DataModule()
cm = CostModule()
sm = SizerModule(leverage=1.0, fraction=0.02)
engine = BacktestEngine(dm, cm, sm)
mm = MetricsModule()

print("=== LOOK-AHEAD BIAS AUDIT ===\n")

result_future = engine.run(
    PerfectFutureSignal(), {}, 'BTC/USDT:USDT', '4h',
    '2022-01-01', '2022-12-31', initial_equity=1000.0
)
metrics_future = mm.compute(result_future)
print(f"Perfect-future signal Sharpe: {metrics_future.sharpe_ratio:.2f}")
print(f"Perfect-future total return: {metrics_future.total_return_pct:.2f}%")
print(f"Perfect-future trades: {metrics_future.total_trades}")
if metrics_future.sharpe_ratio > 10:
    print("CRITICAL: LOOK-AHEAD BIAS CONFIRMED.")
elif metrics_future.sharpe_ratio > 5:
    print("WARNING: Sharpe suspiciously high.")
else:
    print(f"OK: Sharpe {metrics_future.sharpe_ratio:.2f} is plausible with costs.")

result_lag = engine.run(
    LagMomentumSignal(), {}, 'BTC/USDT:USDT', '4h',
    '2022-01-01', '2022-12-31', initial_equity=1000.0
)
metrics_lag = mm.compute(result_lag)
print(f"\nLag-1 momentum signal Sharpe: {metrics_lag.sharpe_ratio:.2f}")
print(f"Lag-1 momentum total return: {metrics_lag.total_return_pct:.2f}%")
if metrics_lag.sharpe_ratio > 3:
    print("WARNING: Lag-1 momentum Sharpe > 3 is suspicious.")
else:
    print(f"OK: Sharpe {metrics_lag.sharpe_ratio:.2f} is plausible for lag-1 momentum.")

# Additional test: always long during a known bull period
class AlwaysLong(SignalModule):
    @property
    def name(self): return "always_long"
    @property
    def parameter_space(self): return {}
    def generate(self, data, params):
        return pd.Series(1, index=data.index, dtype=int)

print("\n--- Buy-and-hold sanity check ---")
result_bh = engine.run(
    AlwaysLong(), {}, 'BTC/USDT:USDT', '4h',
    '2021-01-01', '2023-12-31', initial_equity=1000.0
)
metrics_bh = mm.compute(result_bh)
print(f"Always-long BTC 2021-2023, 1x leverage:")
print(f"  Total return: {metrics_bh.total_return_pct:.2f}%")
print(f"  Sharpe: {metrics_bh.sharpe_ratio:.2f}")
print(f"  Trades: {metrics_bh.total_trades}")
print(f"  BTC went ~30k to ~42k = ~+40%. Backtest should show similar minus costs.")

print("\n=== END LOOK-AHEAD AUDIT ===")
