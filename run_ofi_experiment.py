"""
OFI-Filtered V3 Experiment.

Tests whether OFI MA 1H filtering in neutral SJM regime improves V3.
Uses exact V3 best params, optimises only ofi_neutral_threshold.

Stopping rules:
- If < 20% of neutral-regime entries survive OFI filter at threshold=0: STOP
- If overfit detected: STOP
- Run WF only if val passes overfit check
- Run holdout only if WF >= 60%
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
from strategies.sol_1c_sjm_ofi import SOL1C_SJM_OFI

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# V3 best params (frozen)
SOL_1C_PARAMS = {
    'fast_period': 42,
    'slow_period': 129,
    'adx_period': 24,
    'adx_threshold': 27,
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


def compute_sharpe(bundle: ResultsBundle) -> float:
    m = bundle.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    raw = (m.mean() / m.std()) * np.sqrt(12)
    return np.clip(raw, -10.0, 10.0)


def run_diagnostic(engine, signal, dm, btc_data):
    """
    Diagnostic: with ofi_neutral_threshold=0, what fraction of V3
    neutral-regime entries survive the OFI filter?

    Compare V3 baseline (no OFI filter) vs OFI-filtered on validation period.
    """
    log.info("\n" + "=" * 70)
    log.info("DIAGNOSTIC: OFI filter survival rate at threshold=0")
    log.info("=" * 70)

    # Import baseline strategy
    from strategies.sol_1c_sjm import SOL1C_SJM

    baseline_signal = SOL1C_SJM(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    # Run baseline on validation period
    baseline_params = V3_SJM_PARAMS.copy()
    baseline_bundle = engine.run(
        baseline_signal, baseline_params, SYMBOL, TIMEFRAME,
        VAL_START, VAL_END, 1000.0, 'diagnostic_baseline',
    )
    baseline_trades = len(baseline_bundle.trades)

    # Run OFI-filtered on validation period
    ofi_params = {**V3_SJM_PARAMS, 'ofi_neutral_threshold': 0.0}
    ofi_bundle = engine.run(
        signal, ofi_params, SYMBOL, TIMEFRAME,
        VAL_START, VAL_END, 1000.0, 'diagnostic_ofi',
    )
    ofi_trades = len(ofi_bundle.trades)

    survival_pct = (ofi_trades / baseline_trades * 100) if baseline_trades > 0 else 0

    log.info(f"Baseline V3 trades (val): {baseline_trades}")
    log.info(f"OFI-filtered trades (val): {ofi_trades}")
    log.info(f"Survival rate: {survival_pct:.1f}%")

    baseline_sharpe = compute_sharpe(baseline_bundle)
    ofi_sharpe = compute_sharpe(ofi_bundle)
    log.info(f"Baseline val Sharpe: {baseline_sharpe:.4f}")
    log.info(f"OFI-filtered val Sharpe: {ofi_sharpe:.4f}")

    return {
        'baseline_trades': baseline_trades,
        'ofi_trades': ofi_trades,
        'survival_pct': round(survival_pct, 1),
        'baseline_sharpe': round(baseline_sharpe, 4),
        'ofi_sharpe': round(ofi_sharpe, 4),
    }


def main():
    start_time = time.time()

    # Setup
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics_mod = MetricsModule()

    # Load data
    log.info("Loading data...")
    btc_data = dm.get_ohlcv('BTC/USDT:USDT', TIMEFRAME, '2020-06-01', HOLDOUT_END)
    btc_returns = btc_data['close'].pct_change().dropna()
    sol_data = dm.get_ohlcv(SYMBOL, TIMEFRAME, TRAIN_START, HOLDOUT_END)
    log.info(f"BTC: {len(btc_data)} bars, SOL: {len(sol_data)} bars")

    # Create OFI-filtered signal
    signal = SOL1C_SJM_OFI(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    result = {
        'experiment': 'OFI-Filtered V3',
        'v3_params': {**SOL_1C_PARAMS, **V3_SJM_PARAMS},
    }

    # ── Diagnostic ──────────────────────────────────────────────
    diag = run_diagnostic(engine, signal, dm, btc_data)
    result['diagnostic'] = diag

    if diag['survival_pct'] < 20:
        log.info(f"\nSTOPPING: Only {diag['survival_pct']}% of neutral entries survive OFI filter.")
        log.info("Filter is too aggressive for meaningful trade count.")
        result['conclusion'] = f"FAIL: OFI filter too aggressive ({diag['survival_pct']}% survival)"
        _save_and_report(result, start_time)
        return

    log.info(f"\nDiagnostic PASS: {diag['survival_pct']}% survival, proceeding with optimisation.")

    # ── Optuna: optimise only ofi_neutral_threshold ─────────────
    log.info("\n" + "=" * 70)
    log.info("OPTUNA: Optimising ofi_neutral_threshold (30 trials)")
    log.info("=" * 70)

    def objective(trial):
        threshold = trial.suggest_float('ofi_neutral_threshold', -0.1, 0.1)
        params = {**V3_SJM_PARAMS, 'ofi_neutral_threshold': threshold}
        try:
            bundle = engine.run(
                signal, params, SYMBOL, TIMEFRAME,
                TRAIN_START, TRAIN_END, 1000.0, 'optimisation',
            )
            return compute_sharpe(bundle)
        except Exception as e:
            log.warning(f"Trial failed: {e}")
            return -10.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best_threshold = study.best_params['ofi_neutral_threshold']
    log.info(f"Best ofi_neutral_threshold: {best_threshold:.4f}")
    log.info(f"Best train Sharpe: {study.best_value:.4f}")

    result['best_ofi_threshold'] = round(best_threshold, 4)

    # ── Three-split evaluation ─────────────────────────────────
    full_params = {**V3_SJM_PARAMS, 'ofi_neutral_threshold': best_threshold}

    log.info("\nRunning three-split evaluation...")
    train_bundle = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                              TRAIN_START, TRAIN_END, 1000.0, 'train')
    val_bundle = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                            VAL_START, VAL_END, 1000.0, 'validation')

    train_sharpe = compute_sharpe(train_bundle)
    val_sharpe = compute_sharpe(val_bundle)

    result['train_sharpe'] = round(train_sharpe, 4)
    result['val_sharpe'] = round(val_sharpe, 4)
    result['train_trades'] = len(train_bundle.trades)
    result['val_trades'] = len(val_bundle.trades)

    log.info(f"Train Sharpe: {train_sharpe:.4f} ({len(train_bundle.trades)} trades)")
    log.info(f"Val Sharpe:   {val_sharpe:.4f} ({len(val_bundle.trades)} trades)")

    # Overfit check
    if train_sharpe > 3.0 and val_sharpe < 1.0:
        log.info("OVERFIT detected (train > 3.0, val < 1.0). Stopping.")
        result['conclusion'] = 'FAIL: overfit'
        _save_and_report(result, start_time)
        return

    overfit_ok = val_sharpe >= 0.5 * train_sharpe if train_sharpe > 0 else val_sharpe > 0
    result['overfit_check'] = 'PASS' if overfit_ok else 'FAIL'
    log.info(f"Overfit check: {'PASS' if overfit_ok else 'FAIL'}")

    if not overfit_ok:
        result['conclusion'] = 'FAIL: overfit'
        _save_and_report(result, start_time)
        return

    # ── Walk-forward ───────────────────────────────────────────
    log.info("\nRunning walk-forward validation...")
    wf_results = engine.run_walk_forward(
        signal, SYMBOL, TIMEFRAME,
        TRAIN_START, HOLDOUT_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=15,
        initial_equity=1000.0,
    )

    wf_sharpes = [compute_sharpe(r) for r in wf_results]
    wf_positive = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100
    wf_mean = np.mean(wf_sharpes)

    result['wf_windows'] = len(wf_results)
    result['wf_pct_positive'] = round(wf_positive, 1)
    result['wf_mean_sharpe'] = round(wf_mean, 4)
    result['wf_sharpes'] = [round(s, 4) for s in wf_sharpes]

    log.info(f"WF: {wf_positive:.1f}% positive ({sum(1 for s in wf_sharpes if s > 0)}/{len(wf_sharpes)})")
    log.info(f"WF mean OOS Sharpe: {wf_mean:.4f}")

    if wf_positive < 60:
        log.info(f"WF < 60% ({wf_positive:.1f}%), skipping holdout.")
        result['holdout_sharpe'] = 'not run'
        result['conclusion'] = f'FAIL: WF too low ({wf_positive:.1f}%)'
        _save_and_report(result, start_time)
        return

    # ── Holdout ────────────────────────────────────────────────
    log.info("\nRunning holdout evaluation...")
    holdout_bundle = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                                HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')
    holdout_sharpe = compute_sharpe(holdout_bundle)
    holdout_metrics = metrics_mod.compute(holdout_bundle, btc_returns, train_sharpe)

    result['holdout_sharpe'] = round(holdout_sharpe, 4)
    result['holdout_trades'] = len(holdout_bundle.trades)
    result['holdout_max_dd_days'] = holdout_metrics.max_drawdown_duration_days
    result['holdout_passes_all'] = holdout_metrics.passes_all_checks

    flags = {
        'overfit': holdout_metrics.flag_overfit,
        'insufficient_trades': holdout_metrics.flag_insufficient_trades,
        'high_btc_corr': holdout_metrics.flag_high_btc_correlation,
        'negative_skew': holdout_metrics.flag_negative_skew,
        'long_drawdown': holdout_metrics.flag_long_drawdown,
        'consecutive_losses': holdout_metrics.flag_consecutive_losses,
    }
    result['holdout_flags'] = flags

    log.info(f"Holdout Sharpe: {holdout_sharpe:.4f} ({len(holdout_bundle.trades)} trades)")
    log.info(f"Holdout flags: {flags}")
    log.info(f"All flags clear: {holdout_metrics.passes_all_checks}")

    delta = holdout_sharpe - 0.9065  # V3 baseline
    result['delta_vs_v3'] = round(delta, 4)

    if holdout_sharpe > 1.0 and holdout_metrics.passes_all_checks:
        result['conclusion'] = 'PASS'
    elif holdout_sharpe > 1.0:
        failed = [k for k, v in flags.items() if v]
        result['conclusion'] = f'CONDITIONAL PASS (flags: {failed})'
    else:
        result['conclusion'] = f'FAIL: holdout Sharpe {holdout_sharpe:.2f}'

    log.info(f"\nDelta vs V3 baseline: {delta:+.4f}")
    log.info(f"Conclusion: {result['conclusion']}")

    _save_and_report(result, start_time)


def _save_and_report(result, start_time):
    elapsed = time.time() - start_time
    log.info(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Save JSON
    with open('/home/ubuntu/projects/crypto-trader/ofi_experiment_results.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)

    # Append to STRATEGY_LOG.md
    _append_strategy_log(result)

    log.info("Results saved.")


def _append_strategy_log(result):
    log_path = '/home/ubuntu/projects/crypto-trader/STRATEGY_LOG.md'
    with open(log_path, 'r') as f:
        content = f.read()

    diag = result.get('diagnostic', {})
    entry = f"""

### OFI-Filtered V3

OFI neutral threshold: {result.get('best_ofi_threshold', 'N/A')}
Neutral regime entries surviving filter: {diag.get('survival_pct', 'N/A')}%
Train Sharpe: {result.get('train_sharpe', 'N/A')}
Val Sharpe: {result.get('val_sharpe', 'N/A')}
WF % positive: {result.get('wf_pct_positive', 'N/A')}
Holdout Sharpe: {result.get('holdout_sharpe', 'N/A')}
All flags clear: {'YES' if result.get('holdout_passes_all') else 'NO'}
Conclusion: {result.get('conclusion', 'N/A')}
"""
    delta = result.get('delta_vs_v3')
    if isinstance(delta, (int, float)):
        entry += f"Delta vs V3 baseline (0.91): {delta:+.2f}\n"
    else:
        entry += f"Delta vs V3 baseline (0.91): N/A\n"

    with open(log_path, 'w') as f:
        f.write(content + entry)


if __name__ == '__main__':
    main()
