"""
Supertrend verification: trade analysis, WF failure analysis, perturbation test.
"""

import sys, json
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from strategies.supertrend_sol import SupertrendSOL

SYMBOL = 'SOL/USDT:USDT'
TF = '4h'
HO_START, HO_END = '2024-01-01', '2026-03-21'
TRAIN_START = '2021-01-01'

BEST_PARAMS = {
    'atr_period': 19,
    'multiplier': 3.5104482600764686,
    'use_daily_filter': True,
    'daily_ma_period': 26,
    'daily_buffer_pct': 0.1049375497071539,
}


def sharpe(b):
    m = b.monthly_returns
    if len(m) < 2 or m.std() == 0: return 0.0
    return float(np.clip((m.mean() / m.std()) * np.sqrt(12), -10, 10))


def main():
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    btc_4h = dm.get_ohlcv('BTC/USDT:USDT', TF, '2020-06-01', HO_END)
    sol_daily = dm.get_ohlcv(SYMBOL, '1d', '2020-01-01', HO_END)
    signal = SupertrendSOL(btc_data=btc_4h, daily_ohlcv=sol_daily)

    results = {}

    # ══════════════════════════════════════════════════════════════
    # CHECK 1: Print all 41 holdout trades
    # ══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CHECK 1: All holdout trades")
    print("=" * 70)

    ho = engine.run(signal, BEST_PARAMS, SYMBOL, TF, HO_START, HO_END, 1000, 'holdout')
    trades = ho.trades

    print(f"\nTotal trades: {len(trades)}")
    print(f"Holdout Sharpe: {sharpe(ho):.4f}")
    print(f"\n{'#':>3} {'Entry':>12} {'Exit':>12} {'Dir':>5} {'PnL ($)':>10} {'PnL (%)':>8}")
    print("-" * 55)

    trade_data = []
    for i, t in enumerate(trades):
        # Extract trade details — handle both dict and other formats
        if isinstance(t, dict):
            entry_time = t.get('entry_time', '')
            exit_time = t.get('exit_time', '')
            direction = t.get('direction', 0)
            pnl = t.get('pnl_net', t.get('pnl', 0))
            entry_price = t.get('entry_price', 0)
        else:
            entry_time = getattr(t, 'entry_time', '')
            exit_time = getattr(t, 'exit_time', '')
            direction = getattr(t, 'direction', 0)
            pnl = getattr(t, 'pnl_net', getattr(t, 'pnl', 0))
            entry_price = getattr(t, 'entry_price', 0)

        dir_str = 'LONG' if direction == 1 else 'SHORT' if direction == -1 else str(direction)
        entry_str = str(entry_time)[:10] if entry_time else '?'
        exit_str = str(exit_time)[:10] if exit_time else '?'

        trade_data.append({
            'idx': i + 1,
            'entry': entry_str,
            'exit': exit_str,
            'direction': dir_str,
            'pnl': float(pnl) if pnl else 0,
        })

        print(f"{i+1:3d} {entry_str:>12} {exit_str:>12} {dir_str:>5} {pnl:10.2f}")

    # Top 5 trades by PnL
    sorted_trades = sorted(trade_data, key=lambda x: abs(x['pnl']), reverse=True)
    total_pnl = sum(t['pnl'] for t in trade_data)
    top5_pnl = sum(sorted_trades[i]['pnl'] for i in range(min(5, len(sorted_trades))))

    print(f"\nTotal holdout P&L: ${total_pnl:.2f}")
    print(f"\nTop 5 trades by |P&L|:")
    for t in sorted_trades[:5]:
        print(f"  #{t['idx']}: {t['entry']} to {t['exit']} {t['direction']} "
              f"P&L=${t['pnl']:.2f}")

    if total_pnl != 0:
        top5_fraction = top5_pnl / total_pnl * 100
        print(f"\nTop 5 trades as % of total P&L: {top5_fraction:.1f}%")
    else:
        top5_fraction = 0
        print("\nTotal P&L is zero")

    results['check1'] = {
        'total_trades': len(trades),
        'total_pnl': round(total_pnl, 2),
        'top5_pnl': round(top5_pnl, 2),
        'top5_fraction_pct': round(top5_fraction, 1),
        'trades': trade_data,
    }

    # ══════════════════════════════════════════════════════════════
    # CHECK 2: Failed walk-forward windows
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CHECK 2: Failed walk-forward windows")
    print("=" * 70)

    # Re-run walk-forward to identify failing windows
    from ml.walk_forward_ml import compute_sharpe_from_signals
    wf = engine.run_walk_forward(
        signal, SYMBOL, TF, TRAIN_START, HO_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=15, initial_equity=1000,
    )

    failed_windows = []
    print(f"\nAll {len(wf)} WF windows:")
    print(f"{'Win':>4} {'Test Start':>12} {'Test End':>12} {'Sharpe':>8} {'Trades':>7} {'Status':>8}")
    print("-" * 55)

    for i, r in enumerate(wf):
        s = sharpe(r)
        status = "PASS" if s > 0 else "FAIL"
        # Get date range from the result
        if hasattr(r, 'equity_curve') and len(r.equity_curve) > 0:
            start_d = str(r.equity_curve.index[0])[:10]
            end_d = str(r.equity_curve.index[-1])[:10]
        else:
            start_d = '?'
            end_d = '?'

        print(f"{i+1:4d} {start_d:>12} {end_d:>12} {s:8.2f} {len(r.trades):7d} {status:>8}")

        if s <= 0:
            failed_windows.append({
                'window': i + 1,
                'start': start_d,
                'end': end_d,
                'sharpe': round(s, 4),
                'trades': len(r.trades),
            })

    print(f"\nFailed windows ({len(failed_windows)}):")
    for fw in failed_windows:
        print(f"  Window {fw['window']}: {fw['start']} to {fw['end']}, "
              f"Sharpe={fw['sharpe']:.2f}, trades={fw['trades']}")

    results['check2'] = {
        'total_windows': len(wf),
        'failed_windows': failed_windows,
        'wf_pct_positive': round(sum(1 for r in wf if sharpe(r) > 0) / len(wf) * 100, 1),
    }

    # ══════════════════════════════════════════════════════════════
    # CHECK 3: Perturbation test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("CHECK 3: Perturbation test")
    print("=" * 70)

    variations = [
        ('baseline', BEST_PARAMS),
        ('ATR 17', {**BEST_PARAMS, 'atr_period': 17}),
        ('ATR 21', {**BEST_PARAMS, 'atr_period': 21}),
        ('mult 3.0', {**BEST_PARAMS, 'multiplier': 3.0}),
        ('mult 4.0', {**BEST_PARAMS, 'multiplier': 4.0}),
        ('no daily filter', {**BEST_PARAMS, 'use_daily_filter': False}),
    ]

    perturbation = []
    print(f"\n{'Variation':>20} {'Sharpe':>8} {'Trades':>7} {'Max DD%':>8}")
    print("-" * 48)

    for name, params in variations:
        b = engine.run(signal, params, SYMBOL, TF, HO_START, HO_END, 1000, f'perturb_{name}')
        s = sharpe(b)
        n = len(b.trades)

        # Max drawdown from equity curve
        eq = b.equity_curve
        if hasattr(eq, 'values'):
            eq_vals = eq.values if isinstance(eq, pd.Series) else eq
        else:
            eq_vals = np.array(eq)
        peak = np.maximum.accumulate(eq_vals)
        dd = (eq_vals - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(dd.min() * 100)

        print(f"{name:>20} {s:8.2f} {n:7d} {max_dd:7.1f}%")

        perturbation.append({
            'name': name,
            'params': {k: v for k, v in params.items()},
            'holdout_sharpe': round(s, 4),
            'trades': n,
            'max_dd_pct': round(max_dd, 2),
        })

    # Robustness assessment
    sharpes = [p['holdout_sharpe'] for p in perturbation]
    min_sharpe = min(sharpes)
    max_sharpe = max(sharpes)
    all_above_2 = all(s > 2.0 for s in sharpes)
    any_below_1 = any(s < 1.0 for s in sharpes)

    print(f"\nSharpe range: {min_sharpe:.2f} to {max_sharpe:.2f}")
    if all_above_2:
        print("ROBUST: All variations above Sharpe 2.0")
        robustness = "ROBUST"
    elif any_below_1:
        print("FRAGILE: At least one variation below Sharpe 1.0")
        robustness = "FRAGILE"
    else:
        print("MODERATE: Sharpes between 1.0 and 2.0 for some variations")
        robustness = "MODERATE"

    results['check3'] = {
        'perturbation': perturbation,
        'robustness': robustness,
        'min_sharpe': round(min_sharpe, 4),
        'max_sharpe': round(max_sharpe, 4),
        'all_above_2': all_above_2,
        'any_below_1': any_below_1,
    }

    # Save
    with open('supertrend_perturbation.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to supertrend_perturbation.json")


if __name__ == '__main__':
    main()
