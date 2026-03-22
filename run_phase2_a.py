"""
Phase 2 Experiment A: Standalone contrarian on BTC 1H.
Fade the crowd when L/S positioning is extreme.
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
from strategies.market_structure_contrarian import MarketStructureContrarian

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '1h'
TRAIN_START = '2022-01-01'
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

    btc_returns_4h = dm.get_ohlcv(SYMBOL, '4h', '2020-01-01', HOLDOUT_END)['close'].pct_change().dropna()

    signal = MarketStructureContrarian()
    result = {'experiment': 'Phase2-A: Standalone contrarian BTC 1H'}

    # ── Diagnostic ─────────────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("DIAGNOSTIC: default params (q85, max_hold=4)")
    log.info("="*70)

    diag_params = {
        'crowd_quantile_high': 0.85, 'crowd_quantile_low': 0.15,
        'rolling_window': 168, 'max_hold_bars': 4,
        'require_smart_confirm': False, 'smart_div_threshold': 0.0,
    }

    diag_b = engine.run(signal, diag_params, SYMBOL, TIMEFRAME,
                         VAL_START, VAL_END, 1000.0, 'diagnostic')
    diag_s = compute_sharpe(diag_b)
    n_trades = len(diag_b.trades)
    log.info(f"Val: Sharpe={diag_s:.4f}, trades={n_trades}")

    if n_trades > 0:
        signals_per_week = n_trades / 52
        log.info(f"Signals/week: {signals_per_week:.1f}")

    result['diagnostic'] = {
        'val_sharpe': round(diag_s, 4), 'trades': n_trades,
    }

    # ── Optuna: 30 trials ──────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("OPTUNA: 30 trials")
    log.info("="*70)

    def objective(trial):
        params = {
            'crowd_quantile_high': trial.suggest_float('crowd_quantile_high', 0.75, 0.95),
            'crowd_quantile_low': trial.suggest_float('crowd_quantile_low', 0.05, 0.25),
            'rolling_window': trial.suggest_int('rolling_window', 48, 336),
            'max_hold_bars': trial.suggest_int('max_hold_bars', 1, 12),
            'require_smart_confirm': trial.suggest_categorical('require_smart_confirm', [True, False]),
            'smart_div_threshold': trial.suggest_float('smart_div_threshold', -0.2, 0.2),
        }
        try:
            bundle = engine.run(signal, params, SYMBOL, TIMEFRAME,
                                TRAIN_START, TRAIN_END, 1000.0, 'opt')
            return compute_sharpe(bundle)
        except Exception as e:
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

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

    # ── Walk-forward (1H: 60-day train, 20-day test) ──────────
    log.info("\nWalk-forward validation (1H resolution)...")
    wf = engine.run_walk_forward(
        signal, SYMBOL, TIMEFRAME, TRAIN_START, HOLDOUT_END,
        train_months=3, test_months=1, gap_weeks=1,
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
    ho_m = metrics_mod.compute(ho_b, btc_returns_4h, train_s)

    result['holdout_sharpe'] = round(ho_s, 4)
    result['holdout_trades'] = len(ho_b.trades)
    result['holdout_passes_all'] = ho_m.passes_all_checks
    result['delta_vs_v3'] = round(ho_s - 0.9065, 4)

    flags = {k: getattr(ho_m, f'flag_{k}', False) for k in
             ['overfit', 'insufficient_trades', 'high_btc_correlation',
              'negative_skew', 'long_drawdown', 'consecutive_losses']}
    result['holdout_flags'] = flags

    log.info(f"Holdout: {ho_s:.4f} ({len(ho_b.trades)} trades)")

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

    with open('/home/ubuntu/projects/crypto-trader/phase2_a_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    delta = result.get('delta_vs_v3')
    delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "N/A"

    entry = f"""

### Phase2-A: Standalone contrarian BTC 1H

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
