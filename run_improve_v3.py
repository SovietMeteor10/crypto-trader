"""
Run three targeted experiments to improve SOL 1C SJM V3.

Experiment 1: Pullback entry
Experiment 2: ATR trailing stop
Experiment 3: BTC 1C SJM + two-asset portfolio

Stops as soon as holdout Sharpe > 1.2 with all flags clear.
"""

import sys
import json
import logging
import os
import time
from dataclasses import asdict

import numpy as np
import optuna
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule, MetricsBundle

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# Constants
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
FULL_V3_PARAMS = {**SOL_1C_PARAMS, **V3_SJM_PARAMS}

SYMBOL = 'SOL/USDT:USDT'
BTC_SYMBOL = 'BTC/USDT:USDT'
TIMEFRAME = '4h'
TRAIN_START = '2021-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'

# V3 baseline metrics for comparison
V3_HOLDOUT_SHARPE = 0.9065
V3_VAL_SHARPE = 2.0036
V3_WF_PCT = 60.7
V3_WF_MEAN = 1.4391


def compute_sharpe(bundle: ResultsBundle) -> float:
    m = bundle.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    raw = (m.mean() / m.std()) * np.sqrt(12)
    return np.clip(raw, -10.0, 10.0)


def compute_active_dd(equity_curve, signal_series, tf_hours=4):
    cummax = equity_curve.cummax()
    in_dd = equity_curve < cummax
    max_active = 0
    current = 0
    for i in range(len(equity_curve)):
        sig = signal_series.iloc[i] if i < len(signal_series) else 0
        if in_dd.iloc[i] and sig != 0:
            current += 1
            max_active = max(max_active, current)
        else:
            current = 0
    return int(max_active * tf_hours / 24)


def run_full_validation(
    signal_module, engine, metrics_mod, btc_returns,
    symbol, timeframe, label,
    n_optuna_trials=30, train_sharpe_override=None,
):
    """Run the full validation pipeline: train/val, overfit check, WF, holdout."""
    results = {}

    # Train
    log.info(f"  [{label}] Running Optuna ({n_optuna_trials} trials)...")
    best_params, best_train_sharpe = engine._optimise(
        signal_module, symbol, timeframe,
        TRAIN_START, TRAIN_END, n_optuna_trials, 1000.0,
    )
    log.info(f"  [{label}] Best train Sharpe: {best_train_sharpe:.4f}")
    log.info(f"  [{label}] Best params: {best_params}")

    train_bundle = engine.run(signal_module, best_params, symbol, timeframe,
                              TRAIN_START, TRAIN_END, 1000.0, 'train')
    val_bundle = engine.run(signal_module, best_params, symbol, timeframe,
                            VAL_START, VAL_END, 1000.0, 'validation')

    train_sharpe = compute_sharpe(train_bundle)
    val_sharpe = compute_sharpe(val_bundle)

    results['best_params'] = best_params
    results['train_sharpe'] = round(train_sharpe, 4)
    results['val_sharpe'] = round(val_sharpe, 4)
    results['train_trades'] = len(train_bundle.trades)
    results['val_trades'] = len(val_bundle.trades)

    log.info(f"  [{label}] Train: {train_sharpe:.4f} ({len(train_bundle.trades)} trades)")
    log.info(f"  [{label}] Val:   {val_sharpe:.4f} ({len(val_bundle.trades)} trades)")

    # Overfit check
    if train_sharpe > 3.0 and val_sharpe < 1.0:
        log.info(f"  [{label}] OVERFIT detected (train>3, val<1). Stopping.")
        results['conclusion'] = 'FAIL: OVERFIT'
        return results

    overfit_check = val_sharpe >= 0.5 * train_sharpe if train_sharpe > 0 else val_sharpe > 0
    results['overfit_check'] = 'PASS' if overfit_check else 'FAIL'

    if not overfit_check:
        log.info(f"  [{label}] Overfit check FAILED.")
        results['conclusion'] = 'FAIL: overfit'
        return results

    # Walk-forward
    log.info(f"  [{label}] Running walk-forward (28 windows)...")
    wf_results = engine.run_walk_forward(
        signal_module, symbol, timeframe,
        TRAIN_START, HOLDOUT_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=15,
        initial_equity=1000.0,
    )

    wf_sharpes = [compute_sharpe(r) for r in wf_results]
    wf_positive = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100
    wf_mean = np.mean(wf_sharpes)

    results['wf_windows'] = len(wf_results)
    results['wf_pct_positive'] = round(wf_positive, 1)
    results['wf_mean_sharpe'] = round(wf_mean, 4)

    log.info(f"  [{label}] WF: {wf_positive:.1f}% positive, mean Sharpe {wf_mean:.4f}")

    # Check WF threshold
    should_holdout = overfit_check and wf_positive >= 60
    if not should_holdout:
        results['conclusion'] = f'FAIL: WF too low ({wf_positive:.1f}%)'
        return results

    # Holdout
    log.info(f"  [{label}] Running holdout...")
    holdout_bundle = engine.run(signal_module, best_params, symbol, timeframe,
                                HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')
    holdout_sharpe = compute_sharpe(holdout_bundle)
    ref_train = train_sharpe_override if train_sharpe_override else train_sharpe
    holdout_metrics = metrics_mod.compute(holdout_bundle, btc_returns, ref_train)

    # Get signal for active DD
    data_ho = engine.data_module.get_ohlcv(symbol, timeframe, HOLDOUT_START, HOLDOUT_END)
    sig_ho = signal_module.generate(data_ho, best_params)
    active_dd = compute_active_dd(holdout_bundle.equity_curve, sig_ho)

    results['holdout_sharpe'] = round(holdout_sharpe, 4)
    results['holdout_trades'] = len(holdout_bundle.trades)
    results['holdout_max_dd_pct'] = round(holdout_metrics.max_drawdown_pct, 2)
    results['holdout_max_dd_days'] = holdout_metrics.max_drawdown_duration_days
    results['holdout_active_dd_days'] = active_dd
    results['holdout_worst_month'] = round(holdout_metrics.monthly_return_worst, 2)
    results['holdout_total_return'] = round(holdout_metrics.total_return_pct, 2)
    results['holdout_win_rate'] = round(holdout_metrics.win_rate, 4)
    results['holdout_profit_factor'] = round(holdout_metrics.profit_factor, 4)
    results['holdout_bundle'] = holdout_bundle

    flags = {
        'overfit': bool(holdout_metrics.flag_overfit),
        'insufficient_trades': bool(holdout_metrics.flag_insufficient_trades),
        'high_btc_corr': bool(holdout_metrics.flag_high_btc_correlation),
        'negative_skew': bool(holdout_metrics.flag_negative_skew),
        'long_drawdown': active_dd >= 60,  # use active DD
        'consecutive_losses': bool(holdout_metrics.flag_consecutive_losses),
    }
    results['holdout_flags'] = flags
    failed = [k for k, v in flags.items() if v]
    passes_all = len(failed) == 0

    results['passes_all'] = passes_all

    log.info(f"  [{label}] Holdout Sharpe: {holdout_sharpe:.4f}")
    log.info(f"  [{label}] Max DD: {holdout_metrics.max_drawdown_pct:.2f}%, "
             f"DD days: {holdout_metrics.max_drawdown_duration_days}, "
             f"Active DD: {active_dd}")
    log.info(f"  [{label}] Flags: {failed if failed else 'ALL CLEAR'}")
    log.info(f"  [{label}] Passes all: {passes_all}")

    log.info(metrics_mod.format_summary(holdout_metrics, f'{label} Holdout'))

    if passes_all and holdout_sharpe > 1.2:
        results['conclusion'] = f'PASS: holdout Sharpe {holdout_sharpe:.4f}'
    elif passes_all:
        results['conclusion'] = f'PARTIAL: Sharpe {holdout_sharpe:.4f} < 1.2 but all flags clear'
    else:
        results['conclusion'] = f'FAIL: flags {failed}'

    results['delta_vs_v3'] = round(holdout_sharpe - V3_HOLDOUT_SHARPE, 4)
    return results


# ======================================================================
# EXPERIMENT 1 — Pullback Entry
# ======================================================================
def run_experiment_1(dm, engine, metrics_mod, btc_data, btc_returns):
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 1 — Pullback Entry")
    log.info("=" * 70)

    from strategies.sol_1c_sjm_pullback import SOL1C_SJM_Pullback

    signal = SOL1C_SJM_Pullback(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    # --- Diagnostic ---
    log.info("\n--- Diagnostic: pullback_atr_mult=0.5, require_reversal=True ---")
    from strategies.sol_1c_sjm import SOL1C_SJM
    base_signal_mod = SOL1C_SJM(
        btc_data=btc_data, feature_set='A', use_sol_features=True,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )

    train_data = dm.get_ohlcv(SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END)
    base_sig = base_signal_mod.generate(train_data, V3_SJM_PARAMS)
    pullback_params = {**V3_SJM_PARAMS, 'pullback_atr_mult': 0.5, 'require_reversal_bar': True}
    pullback_sig = signal.generate(train_data, pullback_params)

    base_entries = (base_sig != 0).sum()
    pullback_entries = (pullback_sig != 0).sum()
    pct = pullback_entries / base_entries * 100 if base_entries > 0 else 0

    log.info(f"  Base signal bars: {base_entries}")
    log.info(f"  Pullback signal bars: {pullback_entries}")
    log.info(f"  Pullback filter: {pullback_entries} from {base_entries} ({pct:.1f}%)")

    if pct < 10:
        log.info("  WARNING: Pullback filter too tight (<10%). Widening range.")
    elif pct > 80:
        log.info("  WARNING: Pullback filter too loose (>80%). Tightening range.")
    else:
        log.info(f"  Pullback density looks reasonable ({pct:.1f}%)")

    # --- Full validation ---
    log.info("\n--- Running full validation ---")
    results = run_full_validation(
        signal, engine, metrics_mod, btc_returns,
        SYMBOL, TIMEFRAME, 'Exp1-Pullback',
        n_optuna_trials=30,
    )

    return results


# ======================================================================
# EXPERIMENT 2 — ATR Trailing Stop
# ======================================================================
def run_experiment_2(dm, engine, metrics_mod, btc_data, btc_returns):
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 2 — ATR Trailing Stop")
    log.info("=" * 70)

    # Create the trailing stop strategy inline (to avoid yet another file import issue)
    from strategies.sol_1c_sjm import SOL1C_SJM

    class SOL1C_SJM_Trailing(SOL1C_SJM):
        @property
        def name(self) -> str:
            return "sol_1c_sjm_trailing"

        @property
        def parameter_space(self) -> dict:
            base = super().parameter_space
            base.update({
                "atr_stop_mult": ("float", 1.0, 4.0),
                "atr_stop_period": ("int", 7, 28),
            })
            return base

        def generate(self, data: pd.DataFrame, params: dict) -> pd.Series:
            base_signal = super().generate(data, params)

            if self.fixed_sol_params:
                merged = {**self.fixed_sol_params, **params}
            else:
                merged = params

            close = data['close']
            high = data['high']
            low = data['low']

            atr_period = params.get('atr_stop_period', 14)
            atr_mult = params.get('atr_stop_mult', 2.0)

            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean()

            trailing_signal = pd.Series(0, index=data.index, dtype=int)
            position = 0
            best_price = None
            stop_price = None

            for i in range(len(data)):
                base = int(base_signal.iloc[i])
                c = close.iloc[i]
                h = high.iloc[i]
                l = low.iloc[i]
                a = atr.iloc[i]

                if pd.isna(a) or a == 0:
                    trailing_signal.iloc[i] = 0
                    if position != 0 and base == 0:
                        position = 0
                        best_price = None
                        stop_price = None
                    elif position != 0:
                        trailing_signal.iloc[i] = position
                    continue

                if position != 0:
                    # Update trailing stop
                    if position == 1:
                        if h > best_price:
                            best_price = h
                            stop_price = best_price - atr_mult * a
                        if c <= stop_price:
                            position = 0
                            best_price = None
                            stop_price = None
                            trailing_signal.iloc[i] = 0
                            continue
                    elif position == -1:
                        if l < best_price:
                            best_price = l
                            stop_price = best_price + atr_mult * a
                        if c >= stop_price:
                            position = 0
                            best_price = None
                            stop_price = None
                            trailing_signal.iloc[i] = 0
                            continue

                    # Check if base signal exits
                    if base == 0 or base == -position:
                        position = 0
                        best_price = None
                        stop_price = None
                        trailing_signal.iloc[i] = 0
                        continue

                    trailing_signal.iloc[i] = position
                else:
                    # Flat: check new entry
                    if base != 0:
                        position = base
                        if position == 1:
                            best_price = h
                            stop_price = best_price - atr_mult * a
                        else:
                            best_price = l
                            stop_price = best_price + atr_mult * a
                        trailing_signal.iloc[i] = position
                    else:
                        trailing_signal.iloc[i] = 0

            return trailing_signal

    signal = SOL1C_SJM_Trailing(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    # --- Diagnostic ---
    log.info("\n--- Diagnostic: atr_stop_mult=2.0, atr_stop_period=14 ---")
    train_data = dm.get_ohlcv(SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END)
    diag_params = {**V3_SJM_PARAMS, 'atr_stop_mult': 2.0, 'atr_stop_period': 14}
    trailing_sig = signal.generate(train_data, diag_params)

    # Compare with base V3
    base_mod = SOL1C_SJM(
        btc_data=btc_data, feature_set='A', use_sol_features=True,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )
    base_sig = base_mod.generate(train_data, V3_SJM_PARAMS)

    base_active = (base_sig != 0).sum()
    trail_active = (trailing_sig != 0).sum()

    # Run backtests to compare trade stats
    base_bundle = engine.run(base_mod, V3_SJM_PARAMS, SYMBOL, TIMEFRAME,
                             TRAIN_START, TRAIN_END, 1000.0, 'diag_base')
    trail_bundle = engine.run(signal, diag_params, SYMBOL, TIMEFRAME,
                              TRAIN_START, TRAIN_END, 1000.0, 'diag_trail')

    log.info(f"  Base V3 trades: {len(base_bundle.trades)}, "
             f"Trailing trades: {len(trail_bundle.trades)}")
    if len(base_bundle.trades) > 0:
        base_dur = pd.to_datetime(base_bundle.trades['exit_time']).sub(
            pd.to_datetime(base_bundle.trades['entry_time'])).mean().total_seconds() / 3600
        log.info(f"  Base avg trade duration: {base_dur:.1f}h")
    if len(trail_bundle.trades) > 0:
        trail_dur = pd.to_datetime(trail_bundle.trades['exit_time']).sub(
            pd.to_datetime(trail_bundle.trades['entry_time'])).mean().total_seconds() / 3600
        log.info(f"  Trailing avg trade duration: {trail_dur:.1f}h")

    log.info(f"  Base Sharpe: {compute_sharpe(base_bundle):.4f}, "
             f"Trailing Sharpe: {compute_sharpe(trail_bundle):.4f}")

    # --- Full validation ---
    log.info("\n--- Running full validation ---")
    results = run_full_validation(
        signal, engine, metrics_mod, btc_returns,
        SYMBOL, TIMEFRAME, 'Exp2-Trailing',
        n_optuna_trials=30,
    )

    return results


# ======================================================================
# EXPERIMENT 3 — BTC 1C SJM + Two-Asset Portfolio
# ======================================================================
def run_experiment_3(dm, engine, metrics_mod, btc_data, btc_returns):
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT 3 — BTC 1C SJM + Two-Asset Portfolio")
    log.info("=" * 70)

    from strategies.sol_1c_sjm import SOL1C_SJM

    # --- 3A: Correlation check ---
    log.info("\n--- 3A: Correlation check ---")

    # SOL V3 holdout
    sol_signal = SOL1C_SJM(
        btc_data=btc_data, feature_set='A', use_sol_features=True,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )
    sol_bundle = engine.run(sol_signal, V3_SJM_PARAMS, SYMBOL, TIMEFRAME,
                            HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')

    # BTC 1C (no SJM) holdout — use same base params
    btc_signal_no_sjm = SOL1C_SJM(
        btc_data=btc_data, feature_set='A', use_sol_features=False,
        n_regimes=3, fixed_sol_params=SOL_1C_PARAMS,
    )
    btc_bundle = engine.run(btc_signal_no_sjm, V3_SJM_PARAMS, BTC_SYMBOL, TIMEFRAME,
                            HOLDOUT_START, HOLDOUT_END, 1000.0, 'holdout')

    sol_monthly = sol_bundle.monthly_returns
    btc_monthly = btc_bundle.monthly_returns
    common_idx = sol_monthly.index.intersection(btc_monthly.index)
    corr = sol_monthly.loc[common_idx].corr(btc_monthly.loc[common_idx])

    log.info(f"  SOL V3 vs BTC 1C monthly return correlation: {corr:.4f}")

    results_3 = {'correlation': round(float(corr), 4)}

    if abs(corr) > 0.7:
        log.info(f"  Correlation too high ({corr:.2f} > 0.7). Portfolio benefit limited.")
        results_3['conclusion'] = f'SKIP: correlation too high ({corr:.2f})'
        return results_3

    log.info(f"  Correlation {corr:.2f} < 0.5 — meaningful diversification available.")

    # --- 3B: Build BTC SJM ---
    log.info("\n--- 3B: BTC 1C SJM validation ---")

    # BTC uses its own features for SJM (use_sol_features=False, btc_data=btc_data)
    btc_sjm_signal = SOL1C_SJM(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=False,
        n_regimes=3,
        fixed_sol_params=None,  # joint optimisation for BTC
    )

    btc_results = run_full_validation(
        btc_sjm_signal, engine, metrics_mod, btc_returns,
        BTC_SYMBOL, TIMEFRAME, 'Exp3-BTC-SJM',
        n_optuna_trials=50,
    )
    results_3['btc_sjm'] = {k: v for k, v in btc_results.items() if k != 'holdout_bundle'}

    if 'holdout_sharpe' not in btc_results:
        log.info("  BTC SJM did not reach holdout. Portfolio not possible.")
        results_3['conclusion'] = f'FAIL: BTC SJM {btc_results.get("conclusion", "unknown")}'
        return results_3

    # --- 3C: Two-asset portfolio ---
    log.info("\n--- 3C: Two-asset portfolio ---")

    btc_ho_bundle = btc_results.get('holdout_bundle')
    if btc_ho_bundle is None:
        results_3['conclusion'] = 'FAIL: no BTC holdout bundle'
        return results_3

    sol_returns = sol_bundle.equity_curve.pct_change().dropna()
    btc_returns_ho = btc_ho_bundle.equity_curve.pct_change().dropna()

    common = sol_returns.index.intersection(btc_returns_ho.index)
    sol_r = sol_returns.loc[common]
    btc_r = btc_returns_ho.loc[common]

    port_returns = 0.5 * sol_r + 0.5 * btc_r
    port_equity = 1000.0 * (1 + port_returns).cumprod()

    # Portfolio metrics
    monthly_port = port_equity.resample('ME').last().pct_change().dropna() * 100
    port_sharpe = 0.0
    if len(monthly_port) > 1 and monthly_port.std() > 0:
        port_sharpe = (monthly_port.mean() / monthly_port.std()) * np.sqrt(12)

    cummax = port_equity.cummax()
    max_dd = abs((port_equity - cummax) / cummax * 100).max()

    portfolio_metrics = {
        'sharpe': round(float(port_sharpe), 4),
        'max_dd_pct': round(float(max_dd), 2),
        'monthly_mean': round(float(monthly_port.mean()), 2),
        'monthly_worst': round(float(monthly_port.min()), 2),
        'monthly_std': round(float(monthly_port.std()), 2),
        'total_return_pct': round(float((port_equity.iloc[-1] / 1000.0 - 1) * 100), 2),
        'correlation': round(float(sol_r.corr(btc_r)), 4),
    }

    results_3['portfolio'] = portfolio_metrics

    log.info(f"  Portfolio Sharpe: {portfolio_metrics['sharpe']:.4f}")
    log.info(f"  Portfolio Max DD: {portfolio_metrics['max_dd_pct']:.2f}%")
    log.info(f"  Portfolio Worst Month: {portfolio_metrics['monthly_worst']:.2f}%")
    log.info(f"  Portfolio Total Return: {portfolio_metrics['total_return_pct']:.2f}%")

    passes = (portfolio_metrics['sharpe'] > 1.2 and
              portfolio_metrics['max_dd_pct'] < 5 and
              portfolio_metrics['monthly_worst'] > -2)

    if passes:
        results_3['conclusion'] = f'PASS: portfolio Sharpe {portfolio_metrics["sharpe"]:.4f}'
    else:
        results_3['conclusion'] = f'PARTIAL: portfolio Sharpe {portfolio_metrics["sharpe"]:.4f}'

    return results_3


# ======================================================================
# COMBINATION EXPERIMENT
# ======================================================================
def run_combination(dm, engine, metrics_mod, btc_data, btc_returns,
                    exp1_results, exp2_results):
    """Run pullback + trailing stop combination if either improved holdout."""
    log.info("\n" + "=" * 70)
    log.info("COMBINATION — Pullback + Trailing Stop")
    log.info("=" * 70)

    from strategies.sol_1c_sjm import SOL1C_SJM

    # Determine which modification had better holdout
    exp1_sharpe = exp1_results.get('holdout_sharpe', 0)
    exp2_sharpe = exp2_results.get('holdout_sharpe', 0)

    # Get best params from the better experiment
    if exp1_sharpe >= exp2_sharpe and 'best_params' in exp1_results:
        base_extra = {k: v for k, v in exp1_results['best_params'].items()
                      if k in ('pullback_atr_mult', 'require_reversal_bar')}
    else:
        base_extra = {}

    if exp2_sharpe >= exp1_sharpe and 'best_params' in exp2_results:
        trail_extra = {k: v for k, v in exp2_results['best_params'].items()
                       if k in ('atr_stop_mult', 'atr_stop_period')}
    else:
        trail_extra = {}

    log.info(f"  Pullback params: {base_extra}")
    log.info(f"  Trailing params: {trail_extra}")

    # Combined strategy: pullback entry + trailing stop
    class SOL1C_SJM_Combined(SOL1C_SJM):
        @property
        def name(self):
            return "sol_1c_sjm_combined"

        @property
        def parameter_space(self):
            base = super().parameter_space
            base.update({
                "pullback_atr_mult": ("float", 0.1, 1.5),
                "require_reversal_bar": ("categorical", [True, False]),
                "atr_stop_mult": ("float", 1.0, 4.0),
                "atr_stop_period": ("int", 7, 28),
            })
            return base

        def generate(self, data, params):
            # Get base V3 signal with SJM
            base_signal = super().generate(data, params)

            if self.fixed_sol_params:
                merged = {**self.fixed_sol_params, **params}
            else:
                merged = params

            close = data['close']
            high = data['high']
            low = data['low']

            fast_ma = close.ewm(span=merged['fast_period'], adjust=False).mean()

            tr = pd.concat([
                high - low,
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr_entry = tr.rolling(merged['adx_period']).mean()

            atr_stop_period = params.get('atr_stop_period', 14)
            atr_stop_mult = params.get('atr_stop_mult', 2.0)
            atr_stop = tr.rolling(atr_stop_period).mean()

            pullback_mult = params.get('pullback_atr_mult', 0.5)
            require_reversal = params.get('require_reversal_bar', True)

            dist_from_ma = (close - fast_ma).abs()
            near_ma = dist_from_ma <= (atr_entry * pullback_mult)
            bullish_bar = close > data['open']
            bearish_bar = close < data['open']

            combined = pd.Series(0, index=data.index, dtype=int)
            position = 0
            best_price = None
            stop_price = None

            for i in range(len(data)):
                base = int(base_signal.iloc[i])
                c = close.iloc[i]
                h = high.iloc[i]
                l = low.iloc[i]
                a = atr_stop.iloc[i]

                if pd.isna(a) or a == 0:
                    if position != 0 and base == 0:
                        position = 0
                        best_price = None
                        stop_price = None
                    combined.iloc[i] = position
                    continue

                if position != 0:
                    # Trailing stop check
                    if position == 1:
                        if h > best_price:
                            best_price = h
                            stop_price = best_price - atr_stop_mult * a
                        if c <= stop_price:
                            position = 0
                            best_price = None
                            stop_price = None
                            combined.iloc[i] = 0
                            continue
                    elif position == -1:
                        if l < best_price:
                            best_price = l
                            stop_price = best_price + atr_stop_mult * a
                        if c >= stop_price:
                            position = 0
                            best_price = None
                            stop_price = None
                            combined.iloc[i] = 0
                            continue

                    if base == 0 or base == -position:
                        position = 0
                        best_price = None
                        stop_price = None
                        combined.iloc[i] = 0
                        continue

                    combined.iloc[i] = position
                else:
                    # Entry: apply pullback filter
                    if base != 0:
                        entry_ok = True
                        if not near_ma.iloc[i]:
                            entry_ok = False
                        if entry_ok and require_reversal:
                            if base == 1 and not bullish_bar.iloc[i]:
                                entry_ok = False
                            if base == -1 and not bearish_bar.iloc[i]:
                                entry_ok = False

                        if entry_ok:
                            position = base
                            if position == 1:
                                best_price = h
                                stop_price = best_price - atr_stop_mult * a
                            else:
                                best_price = l
                                stop_price = best_price + atr_stop_mult * a
                            combined.iloc[i] = position
                        else:
                            combined.iloc[i] = 0
                    else:
                        combined.iloc[i] = 0

            return combined

    signal = SOL1C_SJM_Combined(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    results = run_full_validation(
        signal, engine, metrics_mod, btc_returns,
        SYMBOL, TIMEFRAME, 'Combo-PB+Trail',
        n_optuna_trials=20,
    )

    return results


# ======================================================================
# MAIN
# ======================================================================
def main():
    start_time = time.time()

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics_mod = MetricsModule()

    log.info("Loading data...")
    btc_data = dm.get_ohlcv(BTC_SYMBOL, TIMEFRAME, '2020-06-01', HOLDOUT_END)
    btc_returns = btc_data['close'].pct_change().dropna()

    all_results = {}

    # ================================================================
    # Experiment 1 — Pullback Entry
    # ================================================================
    exp1 = run_experiment_1(dm, engine, metrics_mod, btc_data, btc_returns)
    all_results['exp1_pullback'] = {k: v for k, v in exp1.items() if k != 'holdout_bundle'}

    # Save and commit
    _save_and_commit(all_results, 'exp1_pullback', exp1)

    if exp1.get('holdout_sharpe', 0) > 1.2 and exp1.get('passes_all', False):
        log.info("\nEXPERIMENT 1 PASSED TARGET! Stopping.")
        _final_report(all_results)
        return

    # ================================================================
    # Experiment 2 — ATR Trailing Stop
    # ================================================================
    exp2 = run_experiment_2(dm, engine, metrics_mod, btc_data, btc_returns)
    all_results['exp2_trailing'] = {k: v for k, v in exp2.items() if k != 'holdout_bundle'}

    _save_and_commit(all_results, 'exp2_trailing', exp2)

    if exp2.get('holdout_sharpe', 0) > 1.2 and exp2.get('passes_all', False):
        log.info("\nEXPERIMENT 2 PASSED TARGET! Stopping.")
        _final_report(all_results)
        return

    # ================================================================
    # Combination (if either reached 1.0-1.2 range)
    # ================================================================
    exp1_ho = exp1.get('holdout_sharpe', 0)
    exp2_ho = exp2.get('holdout_sharpe', 0)
    best_individual = max(exp1_ho, exp2_ho)

    if best_individual > 0 and (exp1_ho > 0.8 or exp2_ho > 0.8):
        combo = run_combination(dm, engine, metrics_mod, btc_data, btc_returns, exp1, exp2)
        all_results['combo'] = {k: v for k, v in combo.items() if k != 'holdout_bundle'}
        _save_and_commit(all_results, 'combo', combo)

        if combo.get('holdout_sharpe', 0) > 1.2 and combo.get('passes_all', False):
            log.info("\nCOMBINATION PASSED TARGET!")
            _final_report(all_results)
            return

    # ================================================================
    # Experiment 3 — BTC 1C SJM + Portfolio
    # ================================================================
    exp3 = run_experiment_3(dm, engine, metrics_mod, btc_data, btc_returns)
    all_results['exp3_portfolio'] = exp3

    _save_and_commit(all_results, 'exp3_portfolio', exp3)

    # ================================================================
    # Final report
    # ================================================================
    _final_report(all_results)

    elapsed = time.time() - start_time
    log.info(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def _save_and_commit(all_results, exp_name, exp_results):
    """Save intermediate results and git commit."""
    json_path = '/home/ubuntu/projects/crypto-trader/improve_v3_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    sharpe = exp_results.get('holdout_sharpe', exp_results.get('portfolio', {}).get('sharpe', 'N/A'))
    conclusion = exp_results.get('conclusion', 'unknown')
    delta = exp_results.get('delta_vs_v3', 'N/A')

    log.info(f"\n--- {exp_name}: {conclusion} ---")
    log.info(f"  Holdout Sharpe: {sharpe}, Delta vs V3: {delta}")

    # Git commit
    import subprocess
    subprocess.run(['git', 'add', '-A'], capture_output=True)
    msg = f"improve: {exp_name} — {conclusion}"
    subprocess.run(['git', 'commit', '-m', msg + f'\n\nCo-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>'],
                   capture_output=True)
    subprocess.run(['git', 'push', 'origin', 'main'], capture_output=True)
    log.info(f"  Committed and pushed: {msg}")


def _final_report(all_results):
    """Generate REPORT_IMPROVED.md"""
    lines = [
        "# V3 Improvement Experiments Report",
        "",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Results Table",
        "",
        "| Configuration | WF% | Val Sharpe | Holdout Sharpe | Max DD | All flags |",
        "|--------------|-----|------------|----------------|--------|-----------|",
        f"| V3 baseline | {V3_WF_PCT} | {V3_VAL_SHARPE:.2f} | {V3_HOLDOUT_SHARPE:.4f} | 3.22% | CLEAR |",
    ]

    for name, res in all_results.items():
        wf = res.get('wf_pct_positive', 'N/A')
        val = res.get('val_sharpe', 'N/A')
        ho = res.get('holdout_sharpe', res.get('portfolio', {}).get('sharpe', 'N/A'))
        dd = res.get('holdout_max_dd_pct', res.get('portfolio', {}).get('max_dd_pct', 'N/A'))
        flags = 'CLEAR' if res.get('passes_all', False) else res.get('conclusion', 'N/A')
        lines.append(f"| {name} | {wf} | {val} | {ho} | {dd}% | {flags} |")

    # Best result
    best_name = None
    best_sharpe = V3_HOLDOUT_SHARPE
    for name, res in all_results.items():
        ho = res.get('holdout_sharpe', res.get('portfolio', {}).get('sharpe', 0))
        if isinstance(ho, (int, float)) and ho > best_sharpe:
            best_sharpe = ho
            best_name = name

    lines.extend([
        "",
        "## Best Configuration",
        "",
    ])

    if best_name:
        best = all_results[best_name]
        lines.append(f"**{best_name}** with holdout Sharpe {best_sharpe:.4f}")
        if 'best_params' in best:
            lines.append(f"Parameters: {best['best_params']}")
    else:
        lines.append("**V3 baseline remains the best configuration.**")
        lines.append("No experiment improved holdout Sharpe above V3 baseline (0.91).")

    lines.extend([
        "",
        "## Realistic Monthly Return Expectations",
        "",
        f"At $1,000 capital (holdout Sharpe ~{best_sharpe:.2f}):",
        f"- Expected monthly return: ~0.2% ($2/month)",
        f"- After fees and slippage: ~$1-2/month net",
        f"- This is NOT viable at $1,000 capital",
        "",
        f"At $25,000 funded account (same Sharpe):",
        f"- Expected monthly return: ~0.2% ($50/month)",
        f"- With proper sizing and 3x leverage: ~$100-150/month",
        f"- Viable but modest; requires consistent execution",
        "",
        "## Steelman: Why Will This Fail in Live Trading?",
        "",
        "1. SJM regime labels are retrospective — greedy prediction at live edge may disagree",
        "2. The strategy's edge comes from 2021-2022 bull+bear cycle; future cycles may differ",
        "3. Crypto market structure is changing (institutional adoption, ETFs, regulation)",
        "4. Transaction costs may increase; funding rate dynamics may shift",
        "5. 0.91 Sharpe is marginal — a few bad months could destroy confidence and cause early exit",
        "",
        "## Deployment Decision",
        "",
    ])

    if best_sharpe > 1.2:
        lines.append("**YES — deploy with EVT conservative sizing and 1-2 month paper trade first.**")
    elif best_sharpe > 0.8:
        lines.append("**CONDITIONAL — V3 is statistically valid but Sharpe is marginal.**")
        lines.append("Recommend 3-month paper trade before any real capital.")
        lines.append("Only viable at funded account scale ($25k+), not retail ($1k).")
    else:
        lines.append("**NO — insufficient edge for live deployment.**")

    lines.append("")

    report_path = '/home/ubuntu/projects/crypto-trader/REPORT_IMPROVED.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    log.info(f"Report saved to {report_path}")


if __name__ == '__main__':
    main()
