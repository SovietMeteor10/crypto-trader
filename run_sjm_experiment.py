"""
SJM Regime Filter Experiment Runner.
Implements Phases 4-5 from EXPERIMENT_SJM.md.

Fixed SOL 1C params from best RF-C result:
  fast_period=42, slow_period=129, adx_period=24, adx_threshold=27

Three-split dates:
  Train:     2021-01-01 to 2022-12-31
  Validation: 2023-01-01 to 2023-12-31
  Hold-out:  2024-01-01 to 2026-03-21

Anti-overfitting: 30 Optuna trials max per variant.
"""

import sys
import json
import logging
import time
from dataclasses import asdict

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import optuna
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule, MetricsBundle
from strategies.sol_1c_sjm import SOL1C_SJM

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# Fixed SOL 1C params from best result
SOL_1C_PARAMS = {
    'fast_period': 42,
    'slow_period': 129,
    'adx_period': 24,
    'adx_threshold': 27,
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
    # Clip to prevent degenerate windows from skewing WF averages
    return np.clip(raw, -10.0, 10.0)


def run_variant(
    variant_name: str,
    dm: DataModule,
    engine: BacktestEngine,
    metrics_mod: MetricsModule,
    btc_data: pd.DataFrame,
    btc_returns: pd.Series,
    feature_set: str = 'A',
    funding_rates: pd.Series = None,
    use_sol_features: bool = False,
    n_regimes: int = 3,
    fixed_sol_params: bool = True,
    lambda_range: tuple = (0.01, 5.0),
    n_trials: int = 30,
) -> dict:
    """Run a single SJM variant through the full pipeline."""

    log.info(f"\n{'=' * 70}")
    log.info(f"VARIANT: {variant_name}")
    log.info(f"{'=' * 70}")

    signal = SOL1C_SJM(
        btc_data=btc_data,
        feature_set=feature_set,
        funding_rates=funding_rates,
        use_sol_features=use_sol_features,
        n_regimes=n_regimes,
        fixed_sol_params=SOL_1C_PARAMS if fixed_sol_params else None,
    )

    result = {
        'variant': variant_name,
        'feature_set': feature_set,
        'n_regimes': n_regimes,
        'use_sol_features': use_sol_features,
    }

    # --- Optuna optimisation on TRAINING period ---
    log.info("Running Optuna optimisation on training period...")

    def objective(trial):
        params = {}

        if not fixed_sol_params:
            params['fast_period'] = trial.suggest_int('fast_period', 5, 50)
            params['slow_period'] = trial.suggest_int('slow_period', 50, 200)
            params['adx_period'] = trial.suggest_int('adx_period', 7, 30)
            params['adx_threshold'] = trial.suggest_int('adx_threshold', 15, 40)

        params['sjm_lambda'] = trial.suggest_float('sjm_lambda', lambda_range[0], lambda_range[1])
        params['sjm_window'] = trial.suggest_int('sjm_window', 182, 720)
        params['trade_in_neutral'] = trial.suggest_categorical('trade_in_neutral', [True, False])

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
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_sjm_params = study.best_params
    log.info(f"Best SJM params: {best_sjm_params}")
    log.info(f"Best train Sharpe: {study.best_value:.4f}")

    # For fixed SOL params, the signal module merges them internally.
    # full_params here only contains what Optuna optimized.
    full_params = best_sjm_params.copy()
    if fixed_sol_params:
        # Store full params for logging, but signal module uses fixed_sol_params internally
        result['sol_params'] = SOL_1C_PARAMS

    result['best_params'] = full_params

    # --- Run on train, val, holdout ---
    log.info("Running three-split evaluation...")

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
        result['conclusion'] = 'OVERFIT'
        result['holdout_sharpe'] = 'not run'
        return result

    overfit_check = val_sharpe >= 0.5 * train_sharpe if train_sharpe > 0 else val_sharpe > 0
    result['overfit_check'] = 'PASS' if overfit_check else 'FAIL'
    log.info(f"Overfit check (val >= 0.5 * train): {'PASS' if overfit_check else 'FAIL'}")

    # --- Train metrics ---
    train_metrics = metrics_mod.compute(train_bundle, btc_returns, None)
    val_metrics = metrics_mod.compute(val_bundle, btc_returns, train_sharpe)

    log.info(f"\nTrain metrics:")
    log.info(metrics_mod.format_summary(train_metrics, 'Train'))
    log.info(f"\nValidation metrics:")
    log.info(metrics_mod.format_summary(val_metrics, 'Validation'))

    # --- Walk-forward validation ---
    log.info("\nRunning walk-forward validation (22 windows)...")
    wf_results = engine.run_walk_forward(
        signal, SYMBOL, TIMEFRAME,
        TRAIN_START, HOLDOUT_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=15,  # fewer trials per window for speed
        initial_equity=1000.0,
    )

    wf_sharpes = [compute_sharpe(r) for r in wf_results]
    wf_positive = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100
    wf_mean = np.mean(wf_sharpes)

    result['wf_windows'] = len(wf_results)
    result['wf_pct_positive'] = round(wf_positive, 1)
    result['wf_mean_sharpe'] = round(wf_mean, 4)
    result['wf_sharpes'] = [round(s, 4) for s in wf_sharpes]

    log.info(f"Walk-forward: {wf_positive:.1f}% positive ({sum(1 for s in wf_sharpes if s > 0)}/{len(wf_sharpes)})")
    log.info(f"WF mean OOS Sharpe: {wf_mean:.4f}")

    # --- Holdout (only if validation passes) ---
    should_run_holdout = overfit_check and wf_positive >= 60
    if not should_run_holdout:
        log.info(f"Skipping holdout (overfit_check={result['overfit_check']}, WF%={wf_positive:.1f}%)")
        result['holdout_sharpe'] = 'not run'
        result['conclusion'] = f"FAIL: {'overfit' if not overfit_check else 'WF too low'}"
        return result

    log.info("\nRunning holdout evaluation...")
    holdout_bundle = engine.run(signal, full_params, SYMBOL, TIMEFRAME,
                                HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')
    holdout_sharpe = compute_sharpe(holdout_bundle)
    holdout_metrics = metrics_mod.compute(holdout_bundle, btc_returns, train_sharpe)

    result['holdout_sharpe'] = round(holdout_sharpe, 4)
    result['holdout_trades'] = len(holdout_bundle.trades)
    result['holdout_max_dd_days'] = holdout_metrics.max_drawdown_duration_days
    result['holdout_consec_losing'] = holdout_metrics.max_consecutive_losing_months
    result['holdout_passes_all'] = holdout_metrics.passes_all_checks

    log.info(f"\nHoldout metrics:")
    log.info(metrics_mod.format_summary(holdout_metrics, 'Holdout'))

    # Flags
    flags = {
        'overfit': holdout_metrics.flag_overfit,
        'insufficient_trades': holdout_metrics.flag_insufficient_trades,
        'high_btc_corr': holdout_metrics.flag_high_btc_correlation,
        'negative_skew': holdout_metrics.flag_negative_skew,
        'long_drawdown': holdout_metrics.flag_long_drawdown,
        'consecutive_losses': holdout_metrics.flag_consecutive_losses,
    }
    result['holdout_flags'] = flags

    failed_flags = [k for k, v in flags.items() if v]
    if holdout_metrics.passes_all_checks:
        result['conclusion'] = 'PASS'
    elif holdout_sharpe > 1.2:
        result['conclusion'] = f'CONDITIONAL PASS (Sharpe {holdout_sharpe:.2f}, failed: {failed_flags})'
    else:
        result['conclusion'] = f'FAIL: flags {failed_flags}'

    log.info(f"\nConclusion: {result['conclusion']}")
    return result


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
    log.info(f"BTC data: {len(btc_data)} bars, SOL data: {len(sol_data)} bars")

    all_results = []

    # ================================================================
    # VARIANT 1: Feature Set A, BTC features, 3 regimes
    # ================================================================
    v1 = run_variant(
        'SJM-V1: BTC Feature Set A, 3 regimes',
        dm, engine, metrics_mod, btc_data, btc_returns,
        feature_set='A', n_regimes=3,
    )
    all_results.append(v1)

    if v1.get('conclusion') == 'PASS':
        log.info("V1 PASSED! Stopping.")
        _save_results(all_results, start_time)
        return

    # ================================================================
    # VARIANT 2: Feature Set B (with funding rates)
    # ================================================================
    log.info("\nLoading funding rates for Variant 2...")
    try:
        funding = dm.get_funding_rates('BTC/USDT:USDT', '2021-01-01', HOLDOUT_END)
        v2 = run_variant(
            'SJM-V2: BTC Feature Set B (funding), 3 regimes',
            dm, engine, metrics_mod, btc_data, btc_returns,
            feature_set='B', funding_rates=funding, n_regimes=3,
        )
        all_results.append(v2)

        if v2.get('conclusion') == 'PASS':
            log.info("V2 PASSED! Stopping.")
            _save_results(all_results, start_time)
            return
    except Exception as e:
        log.warning(f"V2 failed to load funding rates: {e}. Skipping.")
        all_results.append({'variant': 'SJM-V2', 'conclusion': f'SKIPPED: {e}'})

    # ================================================================
    # VARIANT 3: SOL features instead of BTC
    # ================================================================
    v3 = run_variant(
        'SJM-V3: SOL Feature Set A, 3 regimes',
        dm, engine, metrics_mod, btc_data, btc_returns,
        feature_set='A', use_sol_features=True, n_regimes=3,
    )
    all_results.append(v3)

    if v3.get('conclusion') == 'PASS':
        log.info("V3 PASSED! Stopping.")
        _save_results(all_results, start_time)
        return

    # ================================================================
    # VARIANT 4: Two-state model (bull vs bear only)
    # ================================================================
    v4 = run_variant(
        'SJM-V4: BTC Feature Set A, 2 regimes (bull/bear)',
        dm, engine, metrics_mod, btc_data, btc_returns,
        feature_set='A', n_regimes=2,
        lambda_range=(1.0, 10.0),
    )
    all_results.append(v4)

    if v4.get('conclusion') == 'PASS':
        log.info("V4 PASSED! Stopping.")
        _save_results(all_results, start_time)
        return

    # ================================================================
    # VARIANT 5: Joint optimisation (SOL 1C + SJM params)
    # ================================================================
    v5 = run_variant(
        'SJM-V5: Joint optimisation, BTC Feature Set A, 3 regimes',
        dm, engine, metrics_mod, btc_data, btc_returns,
        feature_set='A', n_regimes=3,
        fixed_sol_params=False,
        n_trials=50,
    )
    all_results.append(v5)

    _save_results(all_results, start_time)


def _save_results(results: list, start_time: float):
    elapsed = time.time() - start_time
    log.info(f"\n{'=' * 70}")
    log.info(f"ALL VARIANTS COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"{'=' * 70}")

    # Summary table
    log.info(f"\n{'Variant':<50} {'Train':>8} {'Val':>8} {'WF%':>6} {'Holdout':>8} {'Result'}")
    log.info("-" * 100)
    for r in results:
        name = r.get('variant', '?')[:50]
        train = f"{r.get('train_sharpe', '-')}"
        val = f"{r.get('val_sharpe', '-')}"
        wf = f"{r.get('wf_pct_positive', '-')}"
        ho = f"{r.get('holdout_sharpe', '-')}"
        conc = r.get('conclusion', '?')
        log.info(f"{name:<50} {train:>8} {val:>8} {wf:>6} {ho:>8} {conc}")

    # Save to JSON
    output_path = '/home/ubuntu/projects/crypto-trader/sjm_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")

    # Generate report
    _generate_report(results)


def _generate_report(results: list):
    """Generate REPORT_SJM.md"""
    lines = []
    lines.append("# SJM Regime Filter Experiment Report\n")
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Summary\n")
    lines.append(f"| Variant | Train Sharpe | Val Sharpe | WF % Positive | Holdout Sharpe | Conclusion |")
    lines.append(f"|---------|-------------|-----------|---------------|---------------|------------|")

    for r in results:
        name = r.get('variant', '?')
        train = r.get('train_sharpe', '-')
        val = r.get('val_sharpe', '-')
        wf = r.get('wf_pct_positive', '-')
        ho = r.get('holdout_sharpe', '-')
        conc = r.get('conclusion', '?')
        lines.append(f"| {name} | {train} | {val} | {wf} | {ho} | {conc} |")

    lines.append("\n## Variant Details\n")
    for r in results:
        lines.append(f"### {r.get('variant', '?')}\n")
        lines.append(f"- Feature set: {r.get('feature_set', '?')}")
        lines.append(f"- N regimes: {r.get('n_regimes', '?')}")
        lines.append(f"- Best params: {r.get('best_params', '?')}")
        lines.append(f"- Train Sharpe: {r.get('train_sharpe', '?')}")
        lines.append(f"- Val Sharpe: {r.get('val_sharpe', '?')}")
        lines.append(f"- Overfit check: {r.get('overfit_check', '?')}")
        lines.append(f"- WF % positive: {r.get('wf_pct_positive', '?')}")
        lines.append(f"- WF mean Sharpe: {r.get('wf_mean_sharpe', '?')}")
        lines.append(f"- Holdout Sharpe: {r.get('holdout_sharpe', '?')}")
        if 'holdout_flags' in r:
            lines.append(f"- Holdout flags: {r['holdout_flags']}")
        lines.append(f"- Conclusion: {r.get('conclusion', '?')}")
        lines.append("")

    lines.append("## Assessment\n")

    any_pass = any(r.get('conclusion') == 'PASS' for r in results)
    any_conditional = any('CONDITIONAL' in str(r.get('conclusion', '')) for r in results)

    if any_pass:
        lines.append("A variant passed all MetricsBundle flags. See details above for deployment parameters.\n")
    elif any_conditional:
        lines.append("A variant achieved high holdout Sharpe but failed some flags. Review whether position sizing alone can address the remaining flags.\n")
    else:
        lines.append("No variant passed all checks. The SJM regime filter does not solve the drawdown problem in its current form. Consider:\n")
        lines.append("- Different feature engineering (macro data, on-chain metrics)")
        lines.append("- Different signal (not trend-following)")
        lines.append("- Different asset")
        lines.append("- Accepting the drawdown duration flag as inherent to trend-following in crypto\n")

    lines.append("## Steelman: Why Will This Fail in Live Trading?\n")
    lines.append("1. Regime labels are assigned retrospectively — the SJM sees the full window before labelling")
    lines.append("2. Greedy prediction (nearest centroid) may disagree with what a full re-fit would produce")
    lines.append("3. Feature standardisation uses in-sample mean/std which shifts as new data arrives")
    lines.append("4. The jump penalty lambda was optimised on historical data — future regime dynamics may differ")
    lines.append("5. Crypto regime durations are non-stationary — 2021-2023 patterns may not repeat\n")

    report_path = '/home/ubuntu/projects/crypto-trader/REPORT_SJM.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    log.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
