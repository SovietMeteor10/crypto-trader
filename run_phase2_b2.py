"""
Phase 2 Experiment B2: V3 + BTC market structure as SJM regime classifier.
Joint optimisation of all parameters (40 trials).
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
from strategies.sol_1c_btc_structure import SOL1C_BTCStructure

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

SYMBOL = 'SOL/USDT:USDT'
TIMEFRAME = '4h'
TRAIN_START = '2021-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'


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

    log.info("Loading data...")
    btc_data = dm.get_ohlcv('BTC/USDT:USDT', TIMEFRAME, '2020-06-01', HOLDOUT_END)
    btc_returns = btc_data['close'].pct_change().dropna()

    signal = SOL1C_BTCStructure(btc_data=btc_data)
    result = {'experiment': 'Phase2-B2: V3 + BTC structure SJM'}

    # ── Diagnostic: V3 default params with BTC structure ───────
    log.info("\n" + "="*70)
    log.info("DIAGNOSTIC: BTC structure SJM with V3-like params")
    log.info("="*70)

    diag_params = {
        'fast_period': 42, 'slow_period': 129,
        'adx_period': 24, 'adx_threshold': 27,
        'sjm_lambda': 1.66, 'sjm_window': 378,
        'trade_in_neutral': True,
    }
    diag_b = engine.run(signal, diag_params, SYMBOL, TIMEFRAME,
                         VAL_START, VAL_END, 1000.0, 'diagnostic')
    diag_s = compute_sharpe(diag_b)
    log.info(f"Diagnostic val: Sharpe={diag_s:.4f}, trades={len(diag_b.trades)}")
    result['diagnostic'] = {'val_sharpe': round(diag_s, 4), 'trades': len(diag_b.trades)}

    # ── Optuna: 40 trials (joint) ─────────────────────────────
    log.info("\n" + "="*70)
    log.info("OPTUNA: 40 trials (joint optimisation)")
    log.info("="*70)

    def objective(trial):
        params = {
            'fast_period': trial.suggest_int('fast_period', 5, 50),
            'slow_period': trial.suggest_int('slow_period', 50, 200),
            'adx_period': trial.suggest_int('adx_period', 7, 30),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 40),
            'sjm_lambda': trial.suggest_float('sjm_lambda', 0.01, 5.0),
            'sjm_window': trial.suggest_int('sjm_window', 60, 500),
            'trade_in_neutral': trial.suggest_categorical('trade_in_neutral', [True, False]),
        }
        try:
            bundle = engine.run(signal, params, SYMBOL, TIMEFRAME,
                                TRAIN_START, TRAIN_END, 1000.0, 'opt')
            return compute_sharpe(bundle)
        except Exception as e:
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=40)

    best_params = study.best_params
    log.info(f"Best params: {best_params}")
    log.info(f"Best train Sharpe: {study.best_value:.4f}")
    result['best_params'] = best_params

    # ── Three-split ────────────────────────────────────────────
    train_b = engine.run(signal, best_params, SYMBOL, TIMEFRAME,
                          TRAIN_START, TRAIN_END, 1000.0, 'train')
    val_b = engine.run(signal, best_params, SYMBOL, TIMEFRAME,
                        VAL_START, VAL_END, 1000.0, 'validation')

    train_s = compute_sharpe(train_b)
    val_s = compute_sharpe(val_b)
    result['train_sharpe'] = round(train_s, 4)
    result['val_sharpe'] = round(val_s, 4)
    result['train_trades'] = len(train_b.trades)
    result['val_trades'] = len(val_b.trades)

    log.info(f"Train: {train_s:.4f} ({len(train_b.trades)} trades)")
    log.info(f"Val:   {val_s:.4f} ({len(val_b.trades)} trades)")

    if train_s > 3.0 and val_s < 1.0:
        log.info("OVERFIT. Stopping.")
        result['conclusion'] = 'FAIL: overfit'
        _save(result, start_time)
        return

    overfit_ok = val_s >= 0.5 * train_s if train_s > 0 else val_s > 0
    result['overfit_check'] = 'PASS' if overfit_ok else 'FAIL'

    if not overfit_ok:
        result['conclusion'] = 'FAIL: overfit'
        _save(result, start_time)
        return

    # ── Walk-forward ───────────────────────────────────────────
    log.info("\nWalk-forward validation...")
    wf = engine.run_walk_forward(
        signal, SYMBOL, TIMEFRAME, TRAIN_START, HOLDOUT_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=15, initial_equity=1000.0,
    )
    wf_sharpes = [compute_sharpe(r) for r in wf]
    wf_pos = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100

    result['wf_windows'] = len(wf)
    result['wf_pct_positive'] = round(wf_pos, 1)
    result['wf_mean_sharpe'] = round(float(np.mean(wf_sharpes)), 4)

    log.info(f"WF: {wf_pos:.1f}% positive, mean={np.mean(wf_sharpes):.4f}")

    if wf_pos < 60:
        result['holdout_sharpe'] = 'not run'
        result['conclusion'] = f'FAIL: WF too low ({wf_pos:.1f}%)'
        _save(result, start_time)
        return

    # ── Holdout ────────────────────────────────────────────────
    log.info("\nHoldout evaluation...")
    ho_b = engine.run(signal, best_params, SYMBOL, TIMEFRAME,
                       HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')
    ho_s = compute_sharpe(ho_b)
    ho_m = metrics_mod.compute(ho_b, btc_returns, train_s)

    result['holdout_sharpe'] = round(ho_s, 4)
    result['holdout_trades'] = len(ho_b.trades)
    result['holdout_max_dd_days'] = ho_m.max_drawdown_duration_days
    result['holdout_passes_all'] = ho_m.passes_all_checks
    result['delta_vs_v3'] = round(ho_s - 0.9065, 4)

    flags = {
        'overfit': ho_m.flag_overfit,
        'insufficient_trades': ho_m.flag_insufficient_trades,
        'high_btc_corr': ho_m.flag_high_btc_correlation,
        'negative_skew': ho_m.flag_negative_skew,
        'long_drawdown': ho_m.flag_long_drawdown,
        'consecutive_losses': ho_m.flag_consecutive_losses,
    }
    result['holdout_flags'] = flags

    log.info(f"Holdout: {ho_s:.4f} ({len(ho_b.trades)} trades)")
    log.info(f"Delta vs V3: {ho_s - 0.9065:+.4f}")

    if ho_s > 1.0 and ho_m.passes_all_checks:
        result['conclusion'] = 'PASS'
    elif ho_s > 1.0:
        failed = [k for k, v in flags.items() if v]
        result['conclusion'] = f'CONDITIONAL PASS (flags: {failed})'
    else:
        result['conclusion'] = f'FAIL: holdout {ho_s:.2f}'

    _save(result, start_time)


def _save(result, start_time):
    elapsed = time.time() - start_time
    log.info(f"\nDone in {elapsed:.0f}s. Conclusion: {result.get('conclusion', '?')}")

    with open('/home/ubuntu/projects/crypto-trader/phase2_b2_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    delta = result.get('delta_vs_v3')
    delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "N/A"

    entry = f"""

### Phase2-B2: V3 + BTC structure SJM

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
