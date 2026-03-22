"""
Corrected metrics comparison: V3 vs Daily MA SOL vs Daily MA BTC.
Uses fixed monthly_returns (decimal, not percentage).
Computes max drawdown from equity curve.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from strategies.daily_ma_sol import DailyMASOL
from strategies.sol_1c_sjm import SOL1C_SJM

SYMBOL_SOL = 'SOL/USDT:USDT'
SYMBOL_BTC = 'BTC/USDT:USDT'
TF = '4h'
HO_START, HO_END = '2024-01-01', '2026-03-21'

V3_SOL_PARAMS = {'fast_period': 42, 'slow_period': 129, 'adx_period': 24, 'adx_threshold': 27}
V3_SJM_PARAMS = {'sjm_lambda': 1.6573239546018446, 'sjm_window': 378, 'trade_in_neutral': True}


def compute_metrics(bundle, label):
    eq = bundle.equity_curve
    m = bundle.monthly_returns  # now decimal

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    sharpe = (m.mean() / m.std()) * np.sqrt(12) if len(m) > 1 and m.std() > 0 else 0

    # Max drawdown from equity curve
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min() * 100

    return {
        'label': label,
        'sharpe': round(float(sharpe), 2),
        'total_return_pct': round(float(total_ret), 1),
        'max_dd_pct': round(float(max_dd), 1),
        'trades': len(bundle.trades),
        'monthly_mean_pct': round(float(m.mean() * 100), 2),
        'monthly_std_pct': round(float(m.std() * 100), 2),
        'worst_month_pct': round(float(m.min() * 100), 2),
        'best_month_pct': round(float(m.max() * 100), 2),
        'win_months': int((m > 0).sum()),
        'total_months': len(m),
        'annualised_return_pct': round(float(m.mean() * 12 * 100), 1),
    }


def main():
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    btc_4h = dm.get_ohlcv(SYMBOL_BTC, TF, '2020-06-01', HO_END)
    sol_daily = dm.get_ohlcv(SYMBOL_SOL, '1d', '2020-01-01', HO_END)
    btc_daily = dm.get_ohlcv(SYMBOL_BTC, '1d', '2020-01-01', HO_END)

    results = []

    # ── V3 SOL ──────────────────────────────────────────────────
    print("Running V3 SOL...")
    v3 = SOL1C_SJM(btc_data=btc_4h, feature_set='A', use_sol_features=True,
                     n_regimes=3, fixed_sol_params=V3_SOL_PARAMS)
    v3_b = engine.run(v3, V3_SJM_PARAMS, SYMBOL_SOL, TF, HO_START, HO_END, 1000, 'holdout')
    results.append(compute_metrics(v3_b, 'V3 (SOL)'))

    # ── Daily MA SOL (unoptimised) ──────────────────────────────
    print("Running Daily MA SOL (MA=26, buf=0)...")
    dma_sol = DailyMASOL(daily_ohlcv=sol_daily)
    dma_sol_b = engine.run(dma_sol, {'ma_period': 26, 'buffer_pct': 0.0},
                            SYMBOL_SOL, TF, HO_START, HO_END, 1000, 'holdout')
    results.append(compute_metrics(dma_sol_b, 'Daily MA SOL (26)'))

    # ── Daily MA BTC (unoptimised) ──────────────────────────────
    print("Running Daily MA BTC (MA=26, buf=0)...")
    dma_btc = DailyMASOL(daily_ohlcv=btc_daily)
    dma_btc_b = engine.run(dma_btc, {'ma_period': 26, 'buffer_pct': 0.0},
                            SYMBOL_BTC, TF, HO_START, HO_END, 1000, 'holdout')
    results.append(compute_metrics(dma_btc_b, 'Daily MA BTC (26)'))

    # ── Print comparison table ──────────────────────────────────
    print("\n" + "=" * 90)
    print("CORRECTED HOLDOUT METRICS (2024-01-01 to 2026-03-21)")
    print("=" * 90)
    print(f"{'Strategy':<20} {'Sharpe':>8} {'Total Ret':>10} {'Max DD':>8} {'Trades':>7} "
          f"{'Mo Mean':>8} {'Mo Std':>8} {'Worst Mo':>9} {'Best Mo':>8} {'Win%':>6}")
    print("-" * 90)

    for r in results:
        win_pct = f"{r['win_months']}/{r['total_months']}"
        print(f"{r['label']:<20} {r['sharpe']:8.2f} {r['total_return_pct']:9.1f}% "
              f"{r['max_dd_pct']:7.1f}% {r['trades']:7d} "
              f"{r['monthly_mean_pct']:7.2f}% {r['monthly_std_pct']:7.2f}% "
              f"{r['worst_month_pct']:8.2f}% {r['best_month_pct']:7.2f}% {win_pct:>6}")

    print(f"\nAnnualised return estimates:")
    for r in results:
        print(f"  {r['label']}: {r['annualised_return_pct']:.1f}%/year")

    # Save
    import json
    with open('corrected_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSaved to corrected_metrics.json")


if __name__ == '__main__':
    main()
