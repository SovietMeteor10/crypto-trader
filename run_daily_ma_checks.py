"""
Three verification checks on the daily MA strategy before deployment.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
import json

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from strategies.daily_ma_sol import DailyMASOL

SYMBOL = 'SOL/USDT:USDT'
TF = '4h'
HO_START, HO_END = '2024-01-01', '2026-03-21'
MA_PARAMS = {'ma_period': 26, 'buffer_pct': 0.0}


def metrics(bundle):
    eq = bundle.equity_curve
    m = bundle.monthly_returns
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min() * 100
    s = (m.mean() / m.std()) * np.sqrt(12) if len(m) > 1 and m.std() > 0 else 0
    worst_mo = m.min() * 100
    return {
        'total_ret': round(float(total_ret), 1),
        'max_dd': round(float(max_dd), 1),
        'worst_month': round(float(worst_mo), 2),
        'sharpe': round(float(s), 2),
        'trades': len(bundle.trades),
        'win_months': int((m > 0).sum()),
        'total_months': len(m),
    }


def main():
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sol_daily = dm.get_ohlcv(SYMBOL, '1d', '2020-01-01', '2026-03-21')
    signal = DailyMASOL(daily_ohlcv=sol_daily)
    results = {}

    # ══════════════════════════════════════════════════════════════
    # CHECK 1: Position sizing sensitivity
    # ══════════════════════════════════════════════════════════════
    print("=" * 80)
    print("CHECK 1: Position Sizing Sensitivity (Daily MA SOL, holdout)")
    print("=" * 80)

    fractions = [0.02, 0.05, 0.10, 0.20, 0.30]
    sizing_results = []

    print(f"\n{'Fraction':>10} {'Sharpe':>8} {'Total Ret':>10} {'Max DD':>8} {'Worst Mo':>10}")
    print("-" * 50)

    for frac in fractions:
        sizer = SizerModule(method='fixed_fractional', fraction=frac, leverage=3.0)
        engine = BacktestEngine(dm, cost, sizer)
        b = engine.run(signal, MA_PARAMS, SYMBOL, TF, HO_START, HO_END, 1000, 'sizing')
        m = metrics(b)
        m['fraction'] = frac
        sizing_results.append(m)
        print(f"{frac:10.0%} {m['sharpe']:8.2f} {m['total_ret']:9.1f}% "
              f"{m['max_dd']:7.1f}% {m['worst_month']:9.2f}%")

    # Max sizing where DD < 15%
    max_safe = max((r for r in sizing_results if r['max_dd'] > -15),
                   key=lambda r: r['fraction'], default=None)
    if max_safe:
        print(f"\nMax sizing with DD < 15%: {max_safe['fraction']:.0%} "
              f"(DD={max_safe['max_dd']:.1f}%, ret={max_safe['total_ret']:.1f}%)")
    else:
        print("\nAll sizing levels exceed 15% max drawdown")

    results['check1'] = sizing_results

    # ══════════════════════════════════════════════════════════════
    # CHECK 2: Benchmark comparison (buy-and-hold)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CHECK 2: Benchmark Comparison — Daily MA vs Buy-and-Hold SOL")
    print("=" * 80)

    # Buy-and-hold return
    sol_4h = dm.get_ohlcv(SYMBOL, TF, HO_START, HO_END)
    bh_start = sol_4h['close'].iloc[0]
    bh_end = sol_4h['close'].iloc[-1]
    bh_ret = (bh_end / bh_start - 1) * 100

    # Strategy return (use 2% sizing for fair comparison)
    sizer_2pct = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine_2pct = BacktestEngine(dm, cost, sizer_2pct)
    strat_b = engine_2pct.run(signal, MA_PARAMS, SYMBOL, TF, HO_START, HO_END, 1000, 'bench')
    strat_ret = (strat_b.equity_curve.iloc[-1] / strat_b.equity_curve.iloc[0] - 1) * 100

    # Long vs short fraction
    sig = signal.generate(sol_4h, MA_PARAMS)
    long_frac = (sig == 1).mean() * 100
    short_frac = (sig == -1).mean() * 100
    flat_frac = (sig == 0).mean() * 100

    # Alpha calculation: strategy return above what buy-and-hold would give
    # with same leverage and sizing
    # Simple alpha: strategy_monthly - beta * market_monthly
    sol_ret = sol_4h['close'].pct_change()
    strat_monthly = strat_b.monthly_returns
    bh_equity = 1000 * (1 + sol_ret.fillna(0) * 0.02 * 3).cumprod()  # approximate BH with same sizing
    bh_monthly_eq = bh_equity.resample('ME').last()
    bh_monthly_ret = bh_monthly_eq.pct_change().dropna()

    common = strat_monthly.index.intersection(bh_monthly_ret.index)
    if len(common) > 3:
        corr = strat_monthly.loc[common].corr(bh_monthly_ret.loc[common])
        alpha_monthly = strat_monthly.loc[common].mean() - bh_monthly_ret.loc[common].mean()
    else:
        corr = np.nan
        alpha_monthly = np.nan

    print(f"\nSOL price: ${bh_start:.2f} → ${bh_end:.2f}")
    print(f"Buy-and-hold return: {bh_ret:+.1f}%")
    print(f"Daily MA return (2% sizing): {strat_ret:+.1f}%")
    print(f"\nPosition distribution:")
    print(f"  Long:  {long_frac:.1f}%")
    print(f"  Short: {short_frac:.1f}%")
    print(f"  Flat:  {flat_frac:.1f}%")
    print(f"\nCorrelation with buy-and-hold monthly: {corr:.4f}")
    print(f"Monthly alpha over buy-and-hold: {alpha_monthly*100:.2f}%")

    results['check2'] = {
        'sol_start_price': round(float(bh_start), 2),
        'sol_end_price': round(float(bh_end), 2),
        'buy_hold_ret_pct': round(float(bh_ret), 1),
        'strategy_ret_pct': round(float(strat_ret), 1),
        'long_pct': round(float(long_frac), 1),
        'short_pct': round(float(short_frac), 1),
        'flat_pct': round(float(flat_frac), 1),
        'correlation_with_bh': round(float(corr), 4) if not np.isnan(corr) else None,
        'monthly_alpha_pct': round(float(alpha_monthly * 100), 2) if not np.isnan(alpha_monthly) else None,
    }

    # ══════════════════════════════════════════════════════════════
    # CHECK 3: 2022 bear market stress test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CHECK 3: 2022 Bear Market Stress Test")
    print("=" * 80)

    bear_start, bear_end = '2022-01-01', '2022-12-31'
    sizer_stress = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine_stress = BacktestEngine(dm, cost, sizer_stress)
    bear_b = engine_stress.run(signal, MA_PARAMS, SYMBOL, TF, bear_start, bear_end, 1000, 'bear')
    bear_m = metrics(bear_b)

    # SOL buy-and-hold 2022
    sol_2022 = dm.get_ohlcv(SYMBOL, TF, bear_start, bear_end)
    bh_2022 = (sol_2022['close'].iloc[-1] / sol_2022['close'].iloc[0] - 1) * 100

    # Position distribution in 2022
    sig_2022 = signal.generate(sol_2022, MA_PARAMS)
    long_2022 = (sig_2022 == 1).mean() * 100
    short_2022 = (sig_2022 == -1).mean() * 100

    # Monthly breakdown
    mo = bear_b.monthly_returns
    losing_months = (mo < 0).sum()

    print(f"\nSOL buy-and-hold 2022: {bh_2022:+.1f}%")
    print(f"Daily MA 2022: {bear_m['total_ret']:+.1f}% (Sharpe {bear_m['sharpe']:.2f})")
    print(f"Max drawdown: {bear_m['max_dd']:.1f}%")
    print(f"Trades: {bear_m['trades']}")
    print(f"Winning months: {bear_m['win_months']}/{bear_m['total_months']}")
    print(f"Losing months: {losing_months}")
    print(f"Worst month: {bear_m['worst_month']:.2f}%")
    print(f"Position: long {long_2022:.0f}%, short {short_2022:.0f}%")

    print(f"\nMonthly returns 2022:")
    for dt, ret in mo.items():
        status = "+" if ret > 0 else "-"
        print(f"  {dt.strftime('%Y-%m')}: {ret*100:+.2f}% {status}")

    results['check3'] = {
        'sol_bh_2022_pct': round(float(bh_2022), 1),
        'strategy_2022_ret_pct': bear_m['total_ret'],
        'strategy_2022_sharpe': bear_m['sharpe'],
        'strategy_2022_max_dd': bear_m['max_dd'],
        'strategy_2022_trades': bear_m['trades'],
        'strategy_2022_losing_months': int(losing_months),
        'strategy_2022_worst_month': bear_m['worst_month'],
        'long_pct_2022': round(float(long_2022), 1),
        'short_pct_2022': round(float(short_2022), 1),
    }

    # ══════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)

    print(f"\nCheck 1 — Position sizing (holdout 2024-2026):")
    print(f"{'Sizing':>8} {'Sharpe':>8} {'Return':>8} {'Max DD':>8} {'Worst Mo':>9}")
    for r in sizing_results:
        print(f"{r['fraction']:7.0%} {r['sharpe']:8.2f} {r['total_ret']:7.1f}% "
              f"{r['max_dd']:7.1f}% {r['worst_month']:8.2f}%")

    print(f"\nCheck 2 — vs Buy-and-Hold (holdout):")
    print(f"  Buy-and-hold SOL: {bh_ret:+.1f}%")
    print(f"  Daily MA (2% sizing): {strat_ret:+.1f}%")
    print(f"  Long: {long_frac:.0f}% / Short: {short_frac:.0f}%")
    print(f"  Correlation: {corr:.2f}")

    print(f"\nCheck 3 — 2022 bear market:")
    print(f"  SOL buy-hold: {bh_2022:+.1f}%")
    print(f"  Daily MA: {bear_m['total_ret']:+.1f}% (Sharpe {bear_m['sharpe']:.2f}, DD {bear_m['max_dd']:.1f}%)")
    print(f"  Losing months: {losing_months}/{bear_m['total_months']}")

    with open('daily_ma_checks.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved to daily_ma_checks.json")


if __name__ == '__main__':
    main()
