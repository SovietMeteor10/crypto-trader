"""
Phase 2 Experiment B1: V3 + BTC smart money divergence gate.
All V3 params frozen. Only optimise smart_div_threshold.
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
from strategies.sol_1c_sjm_smartmoney import SOL1C_SJM_SmartMoney

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# V3 frozen params
SOL_1C_PARAMS = {
    'fast_period': 42, 'slow_period': 129,
    'adx_period': 24, 'adx_threshold': 27,
}
V3_SJM_PARAMS = {
    'sjm_lambda': 1.6573239546018446,
    'sjm_window': 378,
    'trade_in_neutral': True,
}

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

    signal = SOL1C_SJM_SmartMoney(
        btc_data=btc_data, feature_set='A', use_sol_features=True,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )

    result = {'experiment': 'Phase2-B1: V3 + smart money gate'}

    # ── Diagnostic: threshold=0 survival rate ──────────────────
    log.info("\n" + "="*70)
    log.info("DIAGNOSTIC: smart_div_threshold=0")
    log.info("="*70)

    from strategies.sol_1c_sjm import SOL1C_SJM
    baseline_signal = SOL1C_SJM(
        btc_data=btc_data, feature_set='A', use_sol_features=True,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )
    baseline_bundle = engine.run(baseline_signal, V3_SJM_PARAMS, SYMBOL, TIMEFRAME,
                                  VAL_START, VAL_END, 1000.0, 'diag_baseline')
    base_trades = len(baseline_bundle.trades)

    diag_params = {**V3_SJM_PARAMS, 'smart_div_threshold': 0.0}
    diag_bundle = engine.run(signal, diag_params, SYMBOL, TIMEFRAME,
                              VAL_START, VAL_END, 1000.0, 'diag_smart')
    smart_trades = len(diag_bundle.trades)
    survival = (smart_trades / base_trades * 100) if base_trades > 0 else 0

    log.info(f"Baseline trades (val): {base_trades}")
    log.info(f"Smart money trades (val): {smart_trades}")
    log.info(f"Survival: {survival:.1f}%")
    log.info(f"Baseline Sharpe: {compute_sharpe(baseline_bundle):.4f}")
    log.info(f"Smart money Sharpe: {compute_sharpe(diag_bundle):.4f}")

    result['diagnostic'] = {
        'baseline_trades': base_trades, 'smart_trades': smart_trades,
        'survival_pct': round(survival, 1),
        'baseline_sharpe': round(compute_sharpe(baseline_bundle), 4),
        'smart_sharpe': round(compute_sharpe(diag_bundle), 4),
    }

    # Adjust threshold range based on survival
    if survival < 20:
        thresh_range = (-0.5, 0.5)
        log.info(f"Survival <20%, widening range to {thresh_range}")
    elif survival > 80:
        thresh_range = (-0.1, 0.1)
        log.info(f"Survival >80%, tightening range to {thresh_range}")
    else:
        thresh_range = (-0.3, 0.3)
        log.info(f"Survival in target range, using {thresh_range}")

    # ── Optuna: 30 trials ──────────────────────────────────────
    log.info("\n" + "="*70)
    log.info("OPTUNA: 30 trials")
    log.info("="*70)

    def objective(trial):
        threshold = trial.suggest_float('smart_div_threshold', *thresh_range)
        params = {**V3_SJM_PARAMS, 'smart_div_threshold': threshold}
        try:
            bundle = engine.run(signal, params, SYMBOL, TIMEFRAME,
                                TRAIN_START, TRAIN_END, 1000.0, 'opt')
            return compute_sharpe(bundle)
        except Exception as e:
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    best_thresh = study.best_params['smart_div_threshold']
    log.info(f"Best threshold: {best_thresh:.4f}")
    log.info(f"Best train Sharpe: {study.best_value:.4f}")
    result['best_threshold'] = round(best_thresh, 4)

    # ── Three-split evaluation ─────────────────────────────────
    full_params = {**V3_SJM_PARAMS, 'smart_div_threshold': best_thresh}

    train_b = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                          TRAIN_START, TRAIN_END, 1000.0, 'train')
    val_b = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                        VAL_START, VAL_END, 1000.0, 'validation')

    train_s = compute_sharpe(train_b)
    val_s = compute_sharpe(val_b)
    result['train_sharpe'] = round(train_s, 4)
    result['val_sharpe'] = round(val_s, 4)
    result['train_trades'] = len(train_b.trades)
    result['val_trades'] = len(val_b.trades)

    log.info(f"Train: {train_s:.4f} ({len(train_b.trades)} trades)")
    log.info(f"Val:   {val_s:.4f} ({len(val_b.trades)} trades)")

    # Overfit check
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
    result['wf_sharpes'] = [round(s, 4) for s in wf_sharpes]

    log.info(f"WF: {wf_pos:.1f}% positive, mean={np.mean(wf_sharpes):.4f}")

    if wf_pos < 60:
        result['holdout_sharpe'] = 'not run'
        result['conclusion'] = f'FAIL: WF too low ({wf_pos:.1f}%)'
        _save(result, start_time)
        return

    # ── Holdout ────────────────────────────────────────────────
    log.info("\nHoldout evaluation...")
    ho_b = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
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
    log.info(f"Flags: {flags}")
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

    with open('/home/ubuntu/projects/crypto-trader/phase2_b1_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to STRATEGY_LOG.md
    diag = result.get('diagnostic', {})
    delta = result.get('delta_vs_v3')
    delta_str = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "N/A"

    entry = f"""

### Phase2-B1: V3 + smart money gate

smart_div_threshold: {result.get('best_threshold', 'N/A')}
Entries surviving filter: {diag.get('survival_pct', 'N/A')}%
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

    log.info("Results saved.")


if __name__ == '__main__':
    main()
