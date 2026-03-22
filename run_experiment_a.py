"""
Experiment A — Drawdown Audit on V3.

Investigates whether the 241-day drawdown in V3 holdout is:
  (a) A flat equity curve (regime filter sitting out bear market), or
  (b) A sustained losing period.

Steps:
  A1: Re-run V3 holdout, extract equity curve and signal/regime series
  A2: Analyse drawdown structure (flat vs losing)
  A3: Determine appropriate drawdown threshold for regime-gated strategies
  A4: Add active_drawdown metric and re-evaluate
"""

import sys
import json
import logging
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule, MetricsBundle
from strategies.sol_1c_sjm import SOL1C_SJM

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# V3 parameters from sjm_results.json
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
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'
TRAIN_START = '2021-01-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'

OUTPUT_DIR = '/home/ubuntu/projects/crypto-trader/outputs/experiments/drawdown_audit'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_sharpe(bundle: ResultsBundle) -> float:
    m = bundle.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    raw = (m.mean() / m.std()) * np.sqrt(12)
    return np.clip(raw, -10.0, 10.0)


def run_with_signals(engine, signal_module, params, symbol, timeframe,
                     start, end, initial_equity=1000.0, split_label='holdout'):
    """
    Run backtest but also return the signal and regime series for analysis.
    """
    data = engine.data_module.get_ohlcv(symbol, timeframe, start, end)

    # Get signal and regime separately for analysis
    full_params = {**SOL_1C_PARAMS, **params}
    signal_series = signal_module.generate(data, full_params)
    regime_series = signal_module._get_regime_series(data, full_params)

    # Run the actual backtest
    bundle = engine.run(signal_module, params, symbol, timeframe,
                        start, end, initial_equity, split_label)

    return bundle, signal_series, regime_series, data


def analyse_drawdown(equity_curve, signal_series, regime_series, data, tf_hours=4):
    """
    Detailed analysis of the longest drawdown period.
    """
    eq = equity_curve
    cummax = eq.cummax()
    in_drawdown = eq < cummax
    drawdown_pct = (eq - cummax) / cummax * 100

    # Find longest drawdown period
    dd_start = None
    dd_end = None
    current_start = None
    current_len = 0
    max_len = 0
    max_start = None
    max_end = None

    for i, val in enumerate(in_drawdown):
        if val:
            if current_start is None:
                current_start = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                max_start = current_start
                max_end = i
        else:
            current_start = None
            current_len = 0

    if max_start is None:
        return None

    dd_start_date = eq.index[max_start]
    dd_end_date = eq.index[max_end]
    dd_duration_bars = max_len
    dd_duration_days = int(dd_duration_bars * tf_hours / 24)

    # Equity during drawdown
    dd_equity = eq.iloc[max_start:max_end+1]
    peak_equity = cummax.iloc[max_start]

    # Max dollar and percentage loss during drawdown
    max_dollar_loss = peak_equity - dd_equity.min()
    max_pct_loss = (dd_equity.min() - peak_equity) / peak_equity * 100

    # Signal analysis during drawdown
    dd_signal = signal_series.reindex(dd_equity.index).fillna(0)
    dd_regime = regime_series.reindex(dd_equity.index).fillna('unknown')

    flat_bars = (dd_signal == 0).sum()
    active_bars = (dd_signal != 0).sum()
    total_bars = len(dd_signal)
    flat_fraction = flat_bars / total_bars if total_bars > 0 else 0

    # Regime breakdown
    regime_counts = dd_regime.value_counts()

    # Active drawdown: only count consecutive bars where equity is below peak
    # AND strategy had an open position (signal != 0)
    active_dd_bars = 0
    max_active_dd_bars = 0
    current_active_dd = 0

    for i in range(max_start, max_end + 1):
        if in_drawdown.iloc[i] and signal_series.iloc[i] != 0:
            current_active_dd += 1
            max_active_dd_bars = max(max_active_dd_bars, current_active_dd)
        else:
            current_active_dd = 0

    active_dd_days = int(max_active_dd_bars * tf_hours / 24)

    return {
        'dd_start_date': dd_start_date,
        'dd_end_date': dd_end_date,
        'dd_duration_bars': dd_duration_bars,
        'dd_duration_days': dd_duration_days,
        'max_dollar_loss': max_dollar_loss,
        'max_pct_loss': max_pct_loss,
        'peak_equity': peak_equity,
        'trough_equity': dd_equity.min(),
        'flat_bars': int(flat_bars),
        'active_bars': int(active_bars),
        'total_bars': int(total_bars),
        'flat_fraction': flat_fraction,
        'regime_counts': {str(k): int(v) for k, v in regime_counts.items()},
        'active_drawdown_duration_bars': max_active_dd_bars,
        'active_drawdown_duration_days': active_dd_days,
    }


def monthly_breakdown(equity_curve, signal_series, regime_series, dd_start, dd_end):
    """Monthly P&L breakdown during the drawdown period."""
    mask = (equity_curve.index >= dd_start) & (equity_curve.index <= dd_end)
    dd_eq = equity_curve[mask]

    monthly_close = dd_eq.resample('ME').last().dropna()
    monthly_open = dd_eq.resample('MS').first().dropna()

    rows = []
    for date in monthly_close.index:
        month_start = date.replace(day=1)
        month_mask = (dd_eq.index >= month_start) & (dd_eq.index <= date)
        month_eq = dd_eq[month_mask]
        if len(month_eq) < 2:
            continue

        start_eq = month_eq.iloc[0]
        end_eq = month_eq.iloc[-1]
        ret_pct = (end_eq - start_eq) / start_eq * 100

        # Signal and regime for this month
        month_signal = signal_series.reindex(month_eq.index).fillna(0)
        month_regime = regime_series.reindex(month_eq.index).fillna('unknown')

        active_pct = (month_signal != 0).sum() / len(month_signal) * 100
        dominant_regime = month_regime.value_counts().index[0] if len(month_regime) > 0 else 'unknown'

        rows.append({
            'month': date.strftime('%Y-%m'),
            'start_equity': round(start_eq, 2),
            'end_equity': round(end_eq, 2),
            'return_pct': round(ret_pct, 2),
            'dominant_regime': dominant_regime,
            'signal_active_pct': round(active_pct, 1),
        })

    return rows


def plot_drawdown_audit(equity_curve, signal_series, regime_series,
                        dd_analysis, output_path):
    """Equity curve with drawdown and regime shading."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # Panel 1: Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, color='navy', linewidth=1.2, label='Equity')
    ax1.axhline(y=equity_curve.iloc[0], color='grey', linestyle='--', alpha=0.5, label='Initial')

    # Shade longest drawdown in red
    dd_start = dd_analysis['dd_start_date']
    dd_end = dd_analysis['dd_end_date']
    ax1.axvspan(dd_start, dd_end, alpha=0.15, color='red', label=f'Longest DD ({dd_analysis["dd_duration_days"]}d)')

    # Shade bear regime periods in grey
    bear_mask = regime_series == 'bear'
    bear_changes = bear_mask.astype(int).diff().fillna(0)
    bear_starts = bear_changes[bear_changes == 1].index
    bear_ends = bear_changes[bear_changes == -1].index

    # Handle edge cases
    if bear_mask.iloc[0]:
        bear_starts = bear_starts.insert(0, regime_series.index[0])
    if bear_mask.iloc[-1]:
        bear_ends = bear_ends.append(pd.Index([regime_series.index[-1]]))

    for i in range(min(len(bear_starts), len(bear_ends))):
        label = 'Bear regime' if i == 0 else ''
        ax1.axvspan(bear_starts[i], bear_ends[i], alpha=0.1, color='grey', label=label)

    ax1.set_ylabel('Equity (USDT)')
    ax1.set_title('V3 Holdout — Drawdown Audit')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Signal (1=long, -1=short, 0=flat)
    ax2 = axes[1]
    ax2.fill_between(signal_series.index, signal_series.values,
                     step='post', alpha=0.7, color='steelblue')
    ax2.axvspan(dd_start, dd_end, alpha=0.15, color='red')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Regime
    ax3 = axes[2]
    regime_num = regime_series.map({'bull': 1, 'neutral': 0, 'bear': -1, 'unknown': 0}).fillna(0)
    colors = regime_num.map({1: 'green', 0: 'orange', -1: 'red'}).fillna('grey')
    ax3.bar(regime_num.index, regime_num.values, width=0.2, color=colors.values, alpha=0.7)
    ax3.axvspan(dd_start, dd_end, alpha=0.15, color='red')
    ax3.set_ylabel('Regime')
    ax3.set_ylim(-1.5, 1.5)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Plot saved to {output_path}")


def compute_active_drawdown_for_full_curve(equity_curve, signal_series, tf_hours=4):
    """
    Compute the active drawdown duration for the entire equity curve.
    Active drawdown = max consecutive bars where equity < peak AND signal != 0.
    """
    cummax = equity_curve.cummax()
    in_dd = equity_curve < cummax

    max_active_dd_bars = 0
    current_active_dd = 0

    for i in range(len(equity_curve)):
        sig = signal_series.iloc[i] if i < len(signal_series) else 0
        if in_dd.iloc[i] and sig != 0:
            current_active_dd += 1
            max_active_dd_bars = max(max_active_dd_bars, current_active_dd)
        else:
            current_active_dd = 0

    return int(max_active_dd_bars * tf_hours / 24)


def main():
    log.info("=" * 70)
    log.info("EXPERIMENT A — Drawdown Audit on V3")
    log.info("=" * 70)

    # Setup
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics_mod = MetricsModule()

    # Load BTC data for regime features and correlation
    log.info("Loading data...")
    btc_data = dm.get_ohlcv('BTC/USDT:USDT', TIMEFRAME, '2020-06-01', HOLDOUT_END)
    btc_returns = btc_data['close'].pct_change().dropna()

    # Create signal module
    signal = SOL1C_SJM(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    # ================================================================
    # A1 — Run V3 holdout and extract equity curve + signals
    # ================================================================
    log.info("\n--- A1: Running V3 holdout ---")
    bundle, signal_series, regime_series, sol_data = run_with_signals(
        engine, signal, V3_SJM_PARAMS, SYMBOL, TIMEFRAME,
        HOLDOUT_START, HOLDOUT_END,
    )
    holdout_sharpe = compute_sharpe(bundle)
    log.info(f"Holdout Sharpe: {holdout_sharpe:.4f}")
    log.info(f"Holdout trades: {len(bundle.trades)}")
    log.info(f"Equity curve: {len(bundle.equity_curve)} bars")

    # ================================================================
    # A2 — Analyse drawdown structure
    # ================================================================
    log.info("\n--- A2: Analysing drawdown structure ---")
    dd_analysis = analyse_drawdown(
        bundle.equity_curve, signal_series, regime_series, sol_data,
    )

    if dd_analysis is None:
        log.info("No drawdown found!")
        return

    log.info(f"Longest drawdown: {dd_analysis['dd_start_date']} to {dd_analysis['dd_end_date']}")
    log.info(f"Duration: {dd_analysis['dd_duration_days']} days ({dd_analysis['dd_duration_bars']} bars)")
    log.info(f"Max dollar loss: ${dd_analysis['max_dollar_loss']:.2f}")
    log.info(f"Max percentage loss: {dd_analysis['max_pct_loss']:.2f}%")
    log.info(f"Peak equity: ${dd_analysis['peak_equity']:.2f}")
    log.info(f"Trough equity: ${dd_analysis['trough_equity']:.2f}")
    log.info(f"")
    log.info(f"SIGNAL ANALYSIS during drawdown:")
    log.info(f"  Flat bars (signal=0): {dd_analysis['flat_bars']} ({dd_analysis['flat_fraction']:.1%})")
    log.info(f"  Active bars (signal!=0): {dd_analysis['active_bars']}")
    log.info(f"  Total bars: {dd_analysis['total_bars']}")
    log.info(f"")
    log.info(f"REGIME BREAKDOWN during drawdown:")
    for regime, count in dd_analysis['regime_counts'].items():
        pct = count / dd_analysis['total_bars'] * 100
        log.info(f"  {regime}: {count} bars ({pct:.1f}%)")
    log.info(f"")
    log.info(f"ACTIVE DRAWDOWN (only bars with open position):")
    log.info(f"  Max consecutive active DD bars: {dd_analysis['active_drawdown_duration_bars']}")
    log.info(f"  Active drawdown duration: {dd_analysis['active_drawdown_duration_days']} days")

    # Monthly P&L breakdown
    log.info(f"\nMONTHLY P&L during drawdown:")
    log.info(f"{'Month':>8} {'Start $':>10} {'End $':>10} {'Return%':>9} {'Regime':>10} {'Active%':>9}")
    log.info("-" * 60)
    monthly = monthly_breakdown(
        bundle.equity_curve, signal_series, regime_series,
        dd_analysis['dd_start_date'], dd_analysis['dd_end_date'],
    )
    for row in monthly:
        log.info(f"{row['month']:>8} {row['start_equity']:>10.2f} {row['end_equity']:>10.2f} "
                 f"{row['return_pct']:>8.2f}% {row['dominant_regime']:>10} {row['signal_active_pct']:>8.1f}%")

    # ================================================================
    # A2b — Plot
    # ================================================================
    log.info("\n--- A2b: Generating plot ---")
    plot_drawdown_audit(
        bundle.equity_curve, signal_series, regime_series,
        dd_analysis,
        os.path.join(OUTPUT_DIR, 'drawdown_audit.png'),
    )

    # ================================================================
    # A3 — Determine appropriate threshold
    # ================================================================
    log.info("\n--- A3: Threshold analysis ---")
    flat_pct = dd_analysis['flat_fraction'] * 100
    active_dd_days = dd_analysis['active_drawdown_duration_days']
    total_dd_days = dd_analysis['dd_duration_days']

    log.info(f"Flat percentage during drawdown: {flat_pct:.1f}%")
    log.info(f"Total drawdown duration: {total_dd_days} days")
    log.info(f"Active drawdown duration: {active_dd_days} days")

    predominantly_flat = dd_analysis['flat_fraction'] > 0.70
    log.info(f"Predominantly flat (>70%): {predominantly_flat}")

    if predominantly_flat:
        log.info("CONCLUSION: The drawdown is predominantly a flat equity period.")
        log.info("The strategy was correctly sitting out via regime filter.")
        log.info(f"Active drawdown of {active_dd_days} days {'< 60 — PASSES' if active_dd_days < 60 else '>= 60 — STILL FAILS'}")
    else:
        log.info("CONCLUSION: The drawdown has significant active losing periods.")
        log.info("The regime filter is not fully protecting against drawdown.")

    # ================================================================
    # A4 — Re-run metrics with active_drawdown
    # ================================================================
    log.info("\n--- A4: Computing active drawdown metric ---")

    # Compute active drawdown for full holdout (not just longest DD period)
    full_active_dd_days = compute_active_drawdown_for_full_curve(
        bundle.equity_curve, signal_series,
    )
    log.info(f"Full holdout active drawdown: {full_active_dd_days} days")

    # Standard metrics
    holdout_metrics = metrics_mod.compute(bundle, btc_returns, 1.7705)  # train_sharpe from V3

    log.info(f"\nStandard MetricsBundle:")
    log.info(metrics_mod.format_summary(holdout_metrics, 'Holdout'))

    # Active drawdown assessment
    passes_active_dd = full_active_dd_days < 60
    log.info(f"\n--- FINAL ASSESSMENT ---")
    log.info(f"Original max_drawdown_duration_days: {holdout_metrics.max_drawdown_duration_days}")
    log.info(f"Active drawdown_duration_days: {full_active_dd_days}")
    log.info(f"Active DD < 60 days: {passes_active_dd}")

    # Check all flags with active DD
    original_flags = {
        'overfit': holdout_metrics.flag_overfit,
        'insufficient_trades': holdout_metrics.flag_insufficient_trades,
        'high_btc_corr': holdout_metrics.flag_high_btc_correlation,
        'negative_skew': holdout_metrics.flag_negative_skew,
        'long_drawdown': holdout_metrics.flag_long_drawdown,
        'consecutive_losses': holdout_metrics.flag_consecutive_losses,
    }

    adjusted_flags = original_flags.copy()
    adjusted_flags['long_drawdown_active'] = not passes_active_dd
    # For the adjusted check, use active DD instead of original
    adjusted_passes = not any([
        holdout_metrics.flag_overfit,
        holdout_metrics.flag_insufficient_trades,
        holdout_metrics.flag_high_btc_correlation,
        holdout_metrics.flag_negative_skew,
        not passes_active_dd,  # use active DD
        holdout_metrics.flag_consecutive_losses,
    ])

    log.info(f"Original passes_all_checks: {holdout_metrics.passes_all_checks}")
    log.info(f"Adjusted passes_all_checks (active DD): {adjusted_passes}")

    # ================================================================
    # Save analysis
    # ================================================================
    analysis = {
        'experiment': 'A — Drawdown Audit on V3',
        'v3_params': {**SOL_1C_PARAMS, **V3_SJM_PARAMS},
        'holdout_sharpe': round(holdout_sharpe, 4),
        'holdout_trades': len(bundle.trades),
        'drawdown_analysis': {
            'dd_start': str(dd_analysis['dd_start_date']),
            'dd_end': str(dd_analysis['dd_end_date']),
            'total_duration_days': dd_analysis['dd_duration_days'],
            'max_dollar_loss': round(dd_analysis['max_dollar_loss'], 2),
            'max_pct_loss': round(dd_analysis['max_pct_loss'], 2),
            'flat_fraction': round(dd_analysis['flat_fraction'], 4),
            'flat_bars': dd_analysis['flat_bars'],
            'active_bars': dd_analysis['active_bars'],
            'regime_counts': dd_analysis['regime_counts'],
            'active_drawdown_duration_days': dd_analysis['active_drawdown_duration_days'],
        },
        'full_holdout_active_dd_days': full_active_dd_days,
        'monthly_breakdown': monthly,
        'metrics': {
            'sharpe': holdout_metrics.sharpe_ratio,
            'max_dd_pct': holdout_metrics.max_drawdown_pct,
            'max_dd_days': holdout_metrics.max_drawdown_duration_days,
            'total_return_pct': holdout_metrics.total_return_pct,
            'win_rate': holdout_metrics.win_rate,
            'profit_factor': holdout_metrics.profit_factor,
        },
        'original_flags': {k: bool(v) for k, v in original_flags.items()},
        'adjusted_assessment': {
            'active_drawdown_duration_days': full_active_dd_days,
            'passes_active_dd_check': passes_active_dd,
            'passes_all_checks_adjusted': adjusted_passes,
        },
        'conclusion': (
            f"Active DD = {full_active_dd_days} days. "
            f"{'PASSES' if passes_active_dd else 'FAILS'} the 60-day threshold. "
            f"Flat fraction during drawdown: {dd_analysis['flat_fraction']:.1%}. "
            f"Adjusted passes_all: {adjusted_passes}"
        ),
    }

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, 'analysis.json')
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    log.info(f"\nJSON saved to {json_path}")

    # Save ANALYSIS.md
    md_lines = [
        "# Experiment A — Drawdown Audit on V3",
        "",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        f"- **Holdout Sharpe**: {holdout_sharpe:.4f}",
        f"- **Holdout Trades**: {len(bundle.trades)}",
        f"- **Original max DD duration**: {holdout_metrics.max_drawdown_duration_days} days",
        f"- **Active DD duration**: {full_active_dd_days} days",
        f"- **Flat fraction during drawdown**: {dd_analysis['flat_fraction']:.1%}",
        "",
        "## Drawdown Structure",
        "",
        f"The longest drawdown period: {dd_analysis['dd_start_date']} to {dd_analysis['dd_end_date']}",
        f"- Duration: {dd_analysis['dd_duration_days']} days",
        f"- Max dollar loss: ${dd_analysis['max_dollar_loss']:.2f}",
        f"- Max percentage loss: {dd_analysis['max_pct_loss']:.2f}%",
        "",
        f"During this period:",
        f"- **{dd_analysis['flat_fraction']:.1%}** of bars had signal=0 (strategy was flat)",
        f"- **{(1-dd_analysis['flat_fraction']):.1%}** of bars had active positions",
        "",
        "### Regime breakdown during drawdown:",
        "",
    ]
    for regime, count in dd_analysis['regime_counts'].items():
        pct = count / dd_analysis['total_bars'] * 100
        md_lines.append(f"- {regime}: {count} bars ({pct:.1f}%)")

    md_lines.extend([
        "",
        "## Monthly P&L During Drawdown",
        "",
        "| Month | Start $ | End $ | Return% | Regime | Signal Active% |",
        "|-------|---------|-------|---------|--------|---------------|",
    ])
    for row in monthly:
        md_lines.append(
            f"| {row['month']} | {row['start_equity']:.2f} | {row['end_equity']:.2f} | "
            f"{row['return_pct']:.2f}% | {row['dominant_regime']} | {row['signal_active_pct']:.1f}% |"
        )

    md_lines.extend([
        "",
        "## Active Drawdown Analysis",
        "",
        f"The active drawdown metric counts only consecutive bars where:",
        f"1. Equity is below its previous peak, AND",
        f"2. The strategy has an open position (signal != 0)",
        "",
        f"- Original max drawdown duration: **{holdout_metrics.max_drawdown_duration_days} days**",
        f"- Active drawdown duration: **{full_active_dd_days} days**",
        f"- Active DD threshold (60 days): **{'PASS' if passes_active_dd else 'FAIL'}**",
        "",
        "## Flag Assessment",
        "",
        "| Flag | Original | Adjusted |",
        "|------|----------|----------|",
    ])
    for flag_name, flag_val in original_flags.items():
        if flag_name == 'long_drawdown':
            adj_val = not passes_active_dd
            md_lines.append(f"| {flag_name} | {'FAIL' if flag_val else 'PASS'} | {'FAIL' if adj_val else 'PASS'} (active DD) |")
        else:
            md_lines.append(f"| {flag_name} | {'FAIL' if flag_val else 'PASS'} | {'FAIL' if flag_val else 'PASS'} |")

    md_lines.extend([
        "",
        f"**Original passes_all_checks**: {holdout_metrics.passes_all_checks}",
        f"**Adjusted passes_all_checks**: {adjusted_passes}",
        "",
        "## Conclusion",
        "",
        analysis['conclusion'],
        "",
    ])

    md_path = os.path.join(OUTPUT_DIR, 'ANALYSIS.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    log.info(f"ANALYSIS.md saved to {md_path}")

    log.info(f"\n{'=' * 70}")
    log.info(f"EXPERIMENT A COMPLETE")
    log.info(f"Active drawdown: {full_active_dd_days} days")
    log.info(f"Adjusted passes all checks: {adjusted_passes}")
    log.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
