"""
Experiment B — EVT-based position sizing on V3.

Replaces fixed fractional sizing with GARCH(1,1)-EVT dynamic leverage.
Tests three configurations (conservative/moderate/aggressive).
Does NOT re-optimise signal or SJM parameters — only swaps the sizer.
"""

import sys
import json
import logging
import os
import time
from dataclasses import dataclass

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
from risk.garch_evt import GARCHEVTSizer, EVTResult, build_daily_returns

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# V3 parameters
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

OUTPUT_DIR = '/home/ubuntu/projects/crypto-trader/outputs/experiments/evt_sizing'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EVT configs to test
EVT_CONFIGS = {
    'conservative': {
        'loss_budget_pct': 0.03,
        'max_leverage_cap': 3.0,
        'min_leverage': 0.3,
    },
    'moderate': {
        'loss_budget_pct': 0.05,
        'max_leverage_cap': 5.0,
        'min_leverage': 0.3,
    },
    'aggressive': {
        'loss_budget_pct': 0.08,
        'max_leverage_cap': 8.0,
        'min_leverage': 0.5,
    },
}


def compute_sharpe(bundle: ResultsBundle) -> float:
    m = bundle.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    raw = (m.mean() / m.std()) * np.sqrt(12)
    return np.clip(raw, -10.0, 10.0)


def run_evt_backtest(
    data_module: DataModule,
    cost_module: CostModule,
    signal_module: SOL1C_SJM,
    params: dict,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    evt_sizer: GARCHEVTSizer,
    daily_returns_full: pd.Series,
    initial_equity: float = 1000.0,
    split_label: str = "holdout",
) -> tuple:
    """
    Run backtest with EVT-based position sizing.
    Returns (ResultsBundle, leverage_series) where leverage_series tracks
    the EVT leverage used at each bar.
    """
    data = data_module.get_ohlcv(symbol, timeframe, start, end)
    full_params = {**SOL_1C_PARAMS, **params}
    signal = signal_module.generate(data, full_params)
    signal_module.validate_output(signal, data)

    cash = initial_equity
    equity_series = []
    trades = []
    leverage_series = []
    position = None

    # Rolling vol for fallback
    returns = data["close"].pct_change()
    tf_hours = BacktestEngine._timeframe_to_hours(timeframe)

    # Precompute: at each bar, find the daily returns up to that point
    # EVT refit every 6 bars (= 1 day at 4H)
    refit_interval = 6
    cached_evt_result = None
    last_refit_bar = -refit_interval  # force refit on first bar

    equity_series.append(initial_equity)
    leverage_series.append(0.0)

    for i in range(1, len(data)):
        sig = signal.iloc[i - 1]
        price = data["close"].iloc[i]
        ts = data.index[i]

        # Refit EVT periodically
        if i - last_refit_bar >= refit_interval:
            dr = daily_returns_full.loc[:ts]
            if len(dr) >= 30:
                cached_evt_result = evt_sizer.compute(dr, cash, price)
            last_refit_bar = i

        current_leverage = cached_evt_result.max_leverage if cached_evt_result else evt_sizer.min_leverage
        leverage_series.append(current_leverage)

        # Close position if signal changed
        if position is not None:
            should_close = False
            if sig == 0:
                should_close = True
            elif sig != position["direction"]:
                should_close = True

            if should_close:
                close_result = cost_module.apply_close(
                    position["entry_price"], price, position["size"],
                    symbol, position["direction"],
                )
                pnl_raw = (close_result["fill_price"] - position["entry_price"]) * \
                          position["size"] * position["direction"]
                pnl_net = pnl_raw - close_result["fee_usdt"] - position.get("funding_cost", 0)

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "entry_price": position["entry_price"],
                    "exit_price": close_result["fill_price"],
                    "size": position["size"],
                    "direction": position["direction"],
                    "pnl_usdt": pnl_net,
                    "pnl_pct": pnl_net / cash * 100 if cash > 0 else 0,
                    "cost_usdt": close_result["fee_usdt"] + position.get("open_fee", 0),
                    "funding_cost_usdt": position.get("funding_cost", 0),
                })

                cash += pnl_net
                position = None

        # Open new position with EVT-based sizing
        if sig != 0 and position is None:
            # EVT-based size: notional = equity * evt_leverage * fraction
            # We use fraction=0.02 (same as base) but replace fixed leverage with EVT leverage
            fraction = 0.02
            notional = cash * fraction * current_leverage
            max_notional = cash * 0.95 * current_leverage
            notional = min(notional, max_notional)
            size = abs(notional / price)

            if size > 0:
                # Liquidation check
                liq_margin = size * price / current_leverage
                if current_leverage > 0:
                    if sig == 1:
                        liq_price = price * (1 - 1 / current_leverage)
                    else:
                        liq_price = price * (1 + 1 / current_leverage)
                    distance_pct = abs(price - liq_price) / price
                    safe = distance_pct >= 0.15
                else:
                    safe = True

                if safe:
                    open_result = cost_module.apply_open(price, size, symbol, sig)
                    cash -= open_result["fee_usdt"]

                    position = {
                        "direction": sig,
                        "size": size,
                        "entry_price": open_result["fill_price"],
                        "entry_time": ts,
                        "open_fee": open_result["fee_usdt"],
                        "funding_cost": 0.0,
                    }

        # Mark-to-market
        if position is not None:
            mtm_pnl = (price - position["entry_price"]) * \
                       position["size"] * position["direction"]
            equity_series.append(cash + mtm_pnl)
        else:
            equity_series.append(cash)

    # Close remaining position
    if position is not None:
        last_price = data["close"].iloc[-1]
        last_ts = data.index[-1]
        close_result = cost_module.apply_close(
            position["entry_price"], last_price, position["size"],
            symbol, position["direction"],
        )
        pnl_raw = (close_result["fill_price"] - position["entry_price"]) * \
                  position["size"] * position["direction"]
        pnl_net = pnl_raw - close_result["fee_usdt"] - position.get("funding_cost", 0)
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": last_ts,
            "entry_price": position["entry_price"],
            "exit_price": close_result["fill_price"],
            "size": position["size"],
            "direction": position["direction"],
            "pnl_usdt": pnl_net,
            "pnl_pct": pnl_net / cash * 100 if cash > 0 else 0,
            "cost_usdt": close_result["fee_usdt"] + position.get("open_fee", 0),
            "funding_cost_usdt": position.get("funding_cost", 0),
        })
        cash += pnl_net
        equity_series[-1] = cash

    eq_series = pd.Series(equity_series, index=data.index, name="equity")
    lev_series = pd.Series(leverage_series, index=data.index, name="leverage")
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
        "entry_time", "exit_time", "entry_price", "exit_price",
        "size", "direction", "pnl_usdt", "pnl_pct", "cost_usdt", "funding_cost_usdt",
    ])

    monthly = eq_series.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna() * 100

    bundle = ResultsBundle(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        strategy_name=signal_module.name,
        params=params,
        equity_curve=eq_series,
        trades=trades_df,
        monthly_returns=monthly_ret,
        split=split_label,
    )

    return bundle, lev_series


def compute_active_drawdown(equity_curve, signal_series, tf_hours=4):
    """Active drawdown: max consecutive bars where equity < peak AND signal != 0."""
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


def plot_evt_diagnostic(sol_data, leverage_series, equity_curve, config_name, output_path):
    """Three-panel diagnostic: SOL price, EVT leverage, equity curve."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: SOL price
    ax1 = axes[0]
    ax1.plot(sol_data.index, sol_data['close'].values, color='orange', linewidth=0.8)
    ax1.set_ylabel('SOL Price (USDT)')
    ax1.set_title(f'EVT Sizing Diagnostic — {config_name}')
    ax1.grid(True, alpha=0.3)

    # Panel 2: EVT leverage
    ax2 = axes[1]
    ax2.plot(leverage_series.index, leverage_series.values, color='steelblue', linewidth=0.8)
    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Fixed lev (3.0)')
    ax2.set_ylabel('EVT Max Leverage')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Equity curve
    ax3 = axes[2]
    ax3.plot(equity_curve.index, equity_curve.values, color='navy', linewidth=1.0)
    ax3.axhline(y=1000, color='grey', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Equity (USDT)')
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Plot saved to {output_path}")


def main():
    start_time = time.time()
    log.info("=" * 70)
    log.info("EXPERIMENT B — EVT-based Position Sizing on V3")
    log.info("=" * 70)

    # Setup
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    metrics_mod = MetricsModule()

    # Load data
    log.info("Loading data...")
    btc_data = dm.get_ohlcv('BTC/USDT:USDT', TIMEFRAME, '2020-06-01', HOLDOUT_END)
    btc_returns = btc_data['close'].pct_change().dropna()

    # We need SOL data from before train_start for GARCH warmup
    sol_data_full = dm.get_ohlcv(SYMBOL, TIMEFRAME, '2020-06-01', HOLDOUT_END)
    sol_daily_returns = build_daily_returns(sol_data_full)
    log.info(f"SOL daily returns: {len(sol_daily_returns)} days")

    # Signal module
    signal = SOL1C_SJM(
        btc_data=btc_data,
        feature_set='A',
        use_sol_features=True,
        n_regimes=3,
        fixed_sol_params=SOL_1C_PARAMS,
    )

    # ================================================================
    # Run fixed fractional baseline for comparison
    # ================================================================
    log.info("\n--- Running fixed fractional baseline ---")
    sizer_base = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine_base = BacktestEngine(dm, cost, sizer_base)

    splits = {
        'train': (TRAIN_START, TRAIN_END),
        'validation': (VAL_START, VAL_END),
        'holdout': (HOLDOUT_START, HOLDOUT_END),
    }

    baseline_results = {}
    for split_name, (s, e) in splits.items():
        bundle = engine_base.run(signal, V3_SJM_PARAMS, SYMBOL, TIMEFRAME,
                                 s, e, 1000.0, split_name)
        metrics = metrics_mod.compute(bundle, btc_returns,
                                      compute_sharpe(bundle) if split_name == 'train' else None)
        baseline_results[split_name] = {
            'bundle': bundle,
            'metrics': metrics,
            'sharpe': compute_sharpe(bundle),
        }

    train_sharpe_base = baseline_results['train']['sharpe']
    log.info(f"Baseline Train Sharpe: {train_sharpe_base:.4f}")
    log.info(f"Baseline Val Sharpe:   {baseline_results['validation']['sharpe']:.4f}")
    log.info(f"Baseline Holdout Sharpe: {baseline_results['holdout']['sharpe']:.4f}")

    # Get signal series for active DD calc
    holdout_data = dm.get_ohlcv(SYMBOL, TIMEFRAME, HOLDOUT_START, HOLDOUT_END)
    full_params = {**SOL_1C_PARAMS, **V3_SJM_PARAMS}
    signal_series_holdout = signal.generate(holdout_data, full_params)

    # ================================================================
    # Run EVT configs
    # ================================================================
    all_results = {}

    for config_name, config in EVT_CONFIGS.items():
        log.info(f"\n{'=' * 50}")
        log.info(f"EVT Config: {config_name}")
        log.info(f"  loss_budget={config['loss_budget_pct']}, max_lev={config['max_leverage_cap']}, min_lev={config['min_leverage']}")
        log.info(f"{'=' * 50}")

        evt_sizer = GARCHEVTSizer(
            loss_budget_pct=config['loss_budget_pct'],
            max_leverage_cap=config['max_leverage_cap'],
            min_leverage=config['min_leverage'],
        )

        config_results = {}

        for split_name, (s, e) in splits.items():
            log.info(f"  Running {split_name}...")
            bundle, lev_series = run_evt_backtest(
                dm, cost, signal, V3_SJM_PARAMS,
                SYMBOL, TIMEFRAME, s, e,
                evt_sizer, sol_daily_returns,
                initial_equity=1000.0,
                split_label=split_name,
            )

            sharpe = compute_sharpe(bundle)
            metrics = metrics_mod.compute(bundle, btc_returns,
                                          train_sharpe_base if split_name != 'train' else None)

            # Active DD for holdout
            active_dd_days = None
            if split_name == 'holdout':
                sig_for_dd = signal_series_holdout.reindex(bundle.equity_curve.index).fillna(0)
                active_dd_days = compute_active_drawdown(bundle.equity_curve, sig_for_dd)

            config_results[split_name] = {
                'bundle': bundle,
                'metrics': metrics,
                'sharpe': sharpe,
                'leverage_series': lev_series,
                'active_dd_days': active_dd_days,
            }

            lev_vals = lev_series[lev_series > 0]
            log.info(f"  {split_name} Sharpe: {sharpe:.4f}, "
                     f"DD: {metrics.max_drawdown_pct:.2f}%, "
                     f"DD days: {metrics.max_drawdown_duration_days}, "
                     f"Trades: {metrics.total_trades}, "
                     f"Lev mean: {lev_vals.mean():.2f}, min: {lev_vals.min():.2f}, max: {lev_vals.max():.2f}")

        all_results[config_name] = config_results

        # Plot diagnostic for holdout
        holdout_sol = dm.get_ohlcv(SYMBOL, TIMEFRAME, HOLDOUT_START, HOLDOUT_END)
        plot_evt_diagnostic(
            holdout_sol,
            config_results['holdout']['leverage_series'],
            config_results['holdout']['bundle'].equity_curve,
            config_name,
            os.path.join(OUTPUT_DIR, f'leverage_over_time_{config_name}.png'),
        )

    # ================================================================
    # Comparison table
    # ================================================================
    log.info(f"\n{'=' * 70}")
    log.info("COMPARISON TABLE")
    log.info(f"{'=' * 70}")

    header = f"{'Config':<20} {'Train Sh':>10} {'Val Sh':>10} {'HO Sh':>10} {'Max DD%':>10} {'DD days':>10} {'Active DD':>10} {'Worst Mo':>10} {'Trades':>8} {'Flags':>30}"
    log.info(header)
    log.info("-" * len(header))

    # Baseline
    bm = baseline_results['holdout']['metrics']
    base_sig = signal_series_holdout.reindex(baseline_results['holdout']['bundle'].equity_curve.index).fillna(0)
    base_active_dd = compute_active_drawdown(baseline_results['holdout']['bundle'].equity_curve, base_sig)
    failed_flags_base = [f for f, v in {
        'overfit': bm.flag_overfit, 'insuf_trades': bm.flag_insufficient_trades,
        'btc_corr': bm.flag_high_btc_correlation, 'neg_skew': bm.flag_negative_skew,
        'long_dd': bm.flag_long_drawdown, 'consec_loss': bm.flag_consecutive_losses,
    }.items() if v]
    log.info(f"{'V3 fixed frac':<20} "
             f"{baseline_results['train']['sharpe']:>10.4f} "
             f"{baseline_results['validation']['sharpe']:>10.4f} "
             f"{baseline_results['holdout']['sharpe']:>10.4f} "
             f"{bm.max_drawdown_pct:>10.2f} "
             f"{bm.max_drawdown_duration_days:>10} "
             f"{base_active_dd:>10} "
             f"{bm.monthly_return_worst:>10.2f} "
             f"{bm.total_trades:>8} "
             f"{','.join(failed_flags_base) if failed_flags_base else 'NONE':>30}")

    # EVT configs
    comparison_data = []
    for config_name, config_results in all_results.items():
        hm = config_results['holdout']['metrics']
        active_dd = config_results['holdout']['active_dd_days']
        failed_flags = [f for f, v in {
            'overfit': hm.flag_overfit, 'insuf_trades': hm.flag_insufficient_trades,
            'btc_corr': hm.flag_high_btc_correlation, 'neg_skew': hm.flag_negative_skew,
            'long_dd': hm.flag_long_drawdown, 'consec_loss': hm.flag_consecutive_losses,
        }.items() if v]

        log.info(f"{'V3+EVT '+config_name:<20} "
                 f"{config_results['train']['sharpe']:>10.4f} "
                 f"{config_results['validation']['sharpe']:>10.4f} "
                 f"{config_results['holdout']['sharpe']:>10.4f} "
                 f"{hm.max_drawdown_pct:>10.2f} "
                 f"{hm.max_drawdown_duration_days:>10} "
                 f"{active_dd:>10} "
                 f"{hm.monthly_return_worst:>10.2f} "
                 f"{hm.total_trades:>8} "
                 f"{','.join(failed_flags) if failed_flags else 'NONE':>30}")

        # Leverage stats for holdout
        lev = config_results['holdout']['leverage_series']
        lev_active = lev[lev > 0]

        comparison_data.append({
            'config': config_name,
            'train_sharpe': round(config_results['train']['sharpe'], 4),
            'val_sharpe': round(config_results['validation']['sharpe'], 4),
            'holdout_sharpe': round(config_results['holdout']['sharpe'], 4),
            'max_dd_pct': round(hm.max_drawdown_pct, 2),
            'max_dd_days': hm.max_drawdown_duration_days,
            'active_dd_days': active_dd,
            'worst_month': round(hm.monthly_return_worst, 2),
            'total_trades': hm.total_trades,
            'total_return_pct': round(hm.total_return_pct, 2),
            'win_rate': round(hm.win_rate, 4),
            'profit_factor': round(hm.profit_factor, 4),
            'leverage_mean': round(float(lev_active.mean()), 2) if len(lev_active) > 0 else 0,
            'leverage_min': round(float(lev_active.min()), 2) if len(lev_active) > 0 else 0,
            'leverage_max': round(float(lev_active.max()), 2) if len(lev_active) > 0 else 0,
            'failed_flags': failed_flags,
            'passes_all_original': hm.passes_all_checks,
            'passes_all_active_dd': not any([
                hm.flag_overfit, hm.flag_insufficient_trades,
                hm.flag_high_btc_correlation, hm.flag_negative_skew,
                (active_dd or 999) >= 60,
                hm.flag_consecutive_losses,
            ]),
        })

    # ================================================================
    # Determine best config
    # ================================================================
    best = max(comparison_data, key=lambda x: x['holdout_sharpe'])
    log.info(f"\nBest holdout Sharpe: {best['config']} ({best['holdout_sharpe']:.4f})")
    log.info(f"Best passes all (active DD): {best['passes_all_active_dd']}")

    # ================================================================
    # Save results
    # ================================================================
    elapsed = time.time() - start_time

    summary = {
        'experiment': 'B — EVT-based position sizing on V3',
        'elapsed_seconds': round(elapsed, 1),
        'baseline': {
            'train_sharpe': round(baseline_results['train']['sharpe'], 4),
            'val_sharpe': round(baseline_results['validation']['sharpe'], 4),
            'holdout_sharpe': round(baseline_results['holdout']['sharpe'], 4),
            'max_dd_pct': round(bm.max_drawdown_pct, 2),
            'max_dd_days': bm.max_drawdown_duration_days,
            'active_dd_days': base_active_dd,
            'worst_month': round(bm.monthly_return_worst, 2),
            'total_trades': bm.total_trades,
        },
        'evt_configs': comparison_data,
        'best_config': best['config'],
        'best_holdout_sharpe': best['holdout_sharpe'],
        'conclusion': (
            f"Best config: {best['config']} with holdout Sharpe {best['holdout_sharpe']:.4f}. "
            f"Passes all (active DD): {best['passes_all_active_dd']}. "
            f"Max DD: {best['max_dd_pct']:.2f}%, DD days: {best['max_dd_days']}, "
            f"Active DD: {best['active_dd_days']} days."
        ),
    }

    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nResults saved to {json_path}")

    # Generate SUMMARY.md
    md_lines = [
        "# Experiment B — EVT-based Position Sizing on V3",
        "",
        f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"Runtime: {elapsed:.0f}s",
        "",
        "## Comparison Table",
        "",
        "| Config | Train Sharpe | Val Sharpe | Holdout Sharpe | Max DD% | DD days | Active DD | Worst Month | Trades | Flags |",
        "|--------|-------------|-----------|---------------|---------|---------|-----------|-------------|--------|-------|",
    ]

    # Baseline row
    md_lines.append(
        f"| V3 fixed frac | {baseline_results['train']['sharpe']:.4f} | "
        f"{baseline_results['validation']['sharpe']:.4f} | "
        f"{baseline_results['holdout']['sharpe']:.4f} | "
        f"{bm.max_drawdown_pct:.2f} | {bm.max_drawdown_duration_days} | {base_active_dd} | "
        f"{bm.monthly_return_worst:.2f}% | {bm.total_trades} | "
        f"{','.join(failed_flags_base) if failed_flags_base else 'NONE'} |"
    )

    for cd in comparison_data:
        md_lines.append(
            f"| V3+EVT {cd['config']} | {cd['train_sharpe']:.4f} | "
            f"{cd['val_sharpe']:.4f} | {cd['holdout_sharpe']:.4f} | "
            f"{cd['max_dd_pct']:.2f} | {cd['max_dd_days']} | {cd['active_dd_days']} | "
            f"{cd['worst_month']:.2f}% | {cd['total_trades']} | "
            f"{','.join(cd['failed_flags']) if cd['failed_flags'] else 'NONE'} |"
        )

    md_lines.extend([
        "",
        "## EVT Leverage Statistics (Holdout)",
        "",
        "| Config | Mean Lev | Min Lev | Max Lev |",
        "|--------|---------|---------|---------|",
    ])
    for cd in comparison_data:
        md_lines.append(
            f"| {cd['config']} | {cd['leverage_mean']:.2f} | {cd['leverage_min']:.2f} | {cd['leverage_max']:.2f} |"
        )

    md_lines.extend([
        "",
        "## Best Result",
        "",
        f"- **Config**: {best['config']}",
        f"- **Holdout Sharpe**: {best['holdout_sharpe']:.4f}",
        f"- **Max DD%**: {best['max_dd_pct']:.2f}%",
        f"- **DD days**: {best['max_dd_days']}",
        f"- **Active DD days**: {best['active_dd_days']}",
        f"- **Passes all (original)**: {best['passes_all_original']}",
        f"- **Passes all (active DD)**: {best['passes_all_active_dd']}",
        "",
        "## Conclusion",
        "",
        summary['conclusion'],
        "",
    ])

    md_path = os.path.join(OUTPUT_DIR, 'SUMMARY.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    log.info(f"SUMMARY.md saved to {md_path}")

    log.info(f"\n{'=' * 70}")
    log.info(f"EXPERIMENT B COMPLETE in {elapsed:.0f}s")
    log.info(f"Best holdout: {best['config']} Sharpe={best['holdout_sharpe']:.4f}")
    log.info(f"{'=' * 70}")


if __name__ == '__main__':
    main()
