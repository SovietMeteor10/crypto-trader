"""
Phase 2 Experiment C: BTC→SOL cross-asset.
Trade SOL based on BTC smart_dumb_div. Simple hold-24H strategy.
"""

import sys
import json
import logging
import time

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import optuna
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule
from crypto_infra.signal_module import SignalModule
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

SYMBOL = 'SOL/USDT:USDT'
TIMEFRAME = '4h'
TRAIN_START = '2022-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'

BTC_STRUCTURE_PATH = Path(
    "/home/ubuntu/projects/crypto-trader/data_cache/market_structure/"
    "BTC_USDT_USDT_unified_1h.parquet"
)


class BTCtoSOLCrossAsset(SignalModule):
    """Trade SOL based on BTC smart money divergence. Hold 24H (6 bars at 4H)."""

    def __init__(self):
        df = pd.read_parquet(BTC_STRUCTURE_PATH)
        df.index = pd.DatetimeIndex(df.index)
        self._btc_smart_div = df['smart_dumb_div'].resample('4h').last()

    @property
    def name(self):
        return "btc_sol_cross_asset"

    @property
    def parameter_space(self):
        return {
            "smart_div_threshold": ("float", 0.0, 0.5),
            "hold_bars": ("int", 2, 12),
        }

    def generate(self, data, params):
        smart_div = self._btc_smart_div.reindex(data.index, method='ffill')
        threshold = params['smart_div_threshold']
        hold_bars = params['hold_bars']

        signal = pd.Series(0, index=data.index, dtype=int)
        position = 0
        hold_count = 0

        for i in range(len(data)):
            sd = smart_div.iloc[i]
            if pd.isna(sd):
                signal.iloc[i] = 0
                continue

            if position != 0:
                hold_count += 1
                if hold_count >= hold_bars:
                    position = 0
                    hold_count = 0
                signal.iloc[i] = position
            else:
                if sd > threshold:
                    position = 1
                    hold_count = 0
                    signal.iloc[i] = 1
                elif sd < -threshold:
                    position = -1
                    hold_count = 0
                    signal.iloc[i] = -1
                else:
                    signal.iloc[i] = 0

        return signal


def compute_sharpe(bundle):
    m = bundle.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    return float(np.clip((m.mean() / m.std()) * np.sqrt(12), -10.0, 10.0))


def main():
    start_time = time.time()

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics_mod = MetricsModule()

    btc_returns = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2020-01-01', HOLDOUT_END)['close'].pct_change().dropna()

    signal = BTCtoSOLCrossAsset()
    result = {'experiment': 'Phase2-C: BTC→SOL cross-asset'}

    # Optuna: 20 trials
    log.info("OPTUNA: 20 trials")

    def objective(trial):
        params = {
            'smart_div_threshold': trial.suggest_float('smart_div_threshold', 0.0, 0.5),
            'hold_bars': trial.suggest_int('hold_bars', 2, 12),
        }
        try:
            b = engine.run(signal, params, SYMBOL, TIMEFRAME,
                           TRAIN_START, TRAIN_END, 1000.0, 'opt')
            return compute_sharpe(b)
        except:
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best = study.best_params
    log.info(f"Best: {best}, train Sharpe: {study.best_value:.4f}")
    result['best_params'] = best

    train_b = engine.run(signal, best, SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END, 1000.0, 'train')
    val_b = engine.run(signal, best, SYMBOL, TIMEFRAME, VAL_START, VAL_END, 1000.0, 'val')
    train_s, val_s = compute_sharpe(train_b), compute_sharpe(val_b)

    result['train_sharpe'] = round(train_s, 4)
    result['val_sharpe'] = round(val_s, 4)
    result['train_trades'] = len(train_b.trades)
    result['val_trades'] = len(val_b.trades)

    log.info(f"Train: {train_s:.4f} ({len(train_b.trades)} trades)")
    log.info(f"Val:   {val_s:.4f} ({len(val_b.trades)} trades)")

    if train_s > 3.0 and val_s < 1.0:
        result['conclusion'] = 'FAIL: overfit'
        _save(result, start_time)
        return

    overfit_ok = val_s >= 0.5 * train_s if train_s > 0 else val_s > 0
    if not overfit_ok:
        result['conclusion'] = 'FAIL: overfit'
        _save(result, start_time)
        return

    # WF
    log.info("Walk-forward...")
    wf = engine.run_walk_forward(signal, SYMBOL, TIMEFRAME, TRAIN_START, HOLDOUT_END,
                                  train_months=6, test_months=2, gap_weeks=2,
                                  n_optuna_trials=10, initial_equity=1000.0)
    wf_sharpes = [compute_sharpe(r) for r in wf]
    wf_pos = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100

    result['wf_pct_positive'] = round(wf_pos, 1)
    result['wf_mean_sharpe'] = round(float(np.mean(wf_sharpes)), 4)
    log.info(f"WF: {wf_pos:.1f}% positive")

    if wf_pos < 60:
        result['holdout_sharpe'] = 'not run'
        result['conclusion'] = f'FAIL: WF too low ({wf_pos:.1f}%)'
        _save(result, start_time)
        return

    # Holdout
    ho_b = engine.run(signal, best, SYMBOL, TIMEFRAME, HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')
    ho_s = compute_sharpe(ho_b)
    ho_m = metrics_mod.compute(ho_b, btc_returns, train_s)
    result['holdout_sharpe'] = round(ho_s, 4)
    result['holdout_passes_all'] = ho_m.passes_all_checks
    result['delta_vs_v3'] = round(ho_s - 0.9065, 4)

    if ho_s > 1.0 and ho_m.passes_all_checks:
        result['conclusion'] = 'PASS'
    else:
        result['conclusion'] = f'FAIL: holdout {ho_s:.2f}'

    _save(result, start_time)


def _save(result, start_time):
    elapsed = time.time() - start_time
    log.info(f"Done in {elapsed:.0f}s. {result.get('conclusion', '?')}")

    with open('/home/ubuntu/projects/crypto-trader/phase2_c_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    delta = result.get('delta_vs_v3')
    delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "N/A"

    entry = f"""

### Phase2-C: BTC→SOL cross-asset

Best params: {result.get('best_params', 'N/A')}
Train Sharpe: {result.get('train_sharpe', 'N/A')}
Val Sharpe: {result.get('val_sharpe', 'N/A')}
WF % positive: {result.get('wf_pct_positive', 'N/A')}
Holdout Sharpe: {result.get('holdout_sharpe', 'N/A')}
All flags clear: {'YES' if result.get('holdout_passes_all') else 'NO'}
Delta vs V3 (0.91): {delta_str}
Conclusion: {result.get('conclusion', 'N/A')}
"""

    log_path = '/home/ubuntu/projects/crypto-trader/STRATEGY_LOG.md'
    with open(log_path, 'r') as f:
        content = f.read()
    with open(log_path, 'w') as f:
        f.write(content + entry)


if __name__ == '__main__':
    main()
