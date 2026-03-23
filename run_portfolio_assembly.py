"""
Multi-strategy portfolio assembly.

Combines:
  - Daily MA (BTC/ETH/SOL) — always included as baseline
  - Phase 1 LightGBM BTC (if WF >= 60%)
  - Phase 2 Kyle+LS filter (if WF >= 55%)
  - Phase 3 BTC/ETH arb (if 3+ of 36 positive on holdout)

Equal weight. Compute combined Sharpe, max DD, worst month, trades/month, capital for 1000/month.
Correlation matrix — drop strategies with corr > 0.7.
"""

import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.signal_module import SignalModule
from strategies.daily_ma_sol import DailyMASOL
from itertools import product

HO_START = '2024-01-01'
HO_END = '2026-03-23'
TF_1H = '1h'
TF_4H = '4h'


def sharpe_from_returns(ret):
    """Annualised Sharpe from hourly returns."""
    if ret.std() == 0 or len(ret) < 100:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(8760))


def sharpe_from_monthly(monthly):
    """Annualised Sharpe from monthly returns."""
    if len(monthly) < 2 or monthly.std() == 0:
        return 0.0
    return float((monthly.mean() / monthly.std()) * np.sqrt(12))


def get_daily_ma_returns(dm, symbol, daily_ohlcv, ma_period=26, buffer_pct=0.0):
    """Get hourly returns for daily MA strategy on a given symbol."""
    signal_mod = DailyMASOL(daily_ohlcv=daily_ohlcv)
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    params = {'ma_period': ma_period, 'buffer_pct': buffer_pct}
    bundle = engine.run(signal_mod, params, symbol, TF_4H, HO_START, HO_END, 1000, 'holdout')

    # Convert equity to returns
    eq = bundle.equity_curve
    ret = eq.pct_change().dropna()
    # Resample to 1H for alignment
    ret_1h = ret.resample('1h').sum().fillna(0)
    return ret_1h, bundle


def get_phase3_returns(dm, z_window, entry_z, exit_z):
    """Get hourly returns for BTC/ETH stat arb."""
    btc = dm.get_ohlcv('BTC/USDT:USDT', TF_1H, '2020-09-01', HO_END)
    eth = dm.get_ohlcv('ETH/USDT:USDT', TF_1H, '2020-09-01', HO_END)

    common = btc.index.intersection(eth.index)
    btc_close = btc.loc[common, 'close']
    eth_close = eth.loc[common, 'close']

    ratio = btc_close / eth_close
    ratio_mean = ratio.rolling(z_window, min_periods=z_window//2).mean()
    ratio_std = ratio.rolling(z_window, min_periods=z_window//2).std()
    z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    position = pd.Series(0.0, index=z.index)
    current_pos = 0
    for i in range(1, len(z)):
        zval = z.iloc[i]
        if np.isnan(zval):
            position.iloc[i] = current_pos
            continue
        if current_pos == 0:
            if zval > entry_z:
                current_pos = -1
            elif zval < -entry_z:
                current_pos = 1
        elif current_pos == 1:
            if abs(zval) <= exit_z:
                current_pos = 0
        elif current_pos == -1:
            if abs(zval) <= exit_z:
                current_pos = 0
        position.iloc[i] = current_pos

    btc_ret = btc_close.pct_change()
    eth_ret = eth_close.pct_change()
    spread_ret = btc_ret - eth_ret

    strat_ret = position.shift(1).fillna(0) * spread_ret
    pos_changes = position.diff().abs().fillna(0)
    strat_ret = strat_ret - pos_changes * 0.001 / 2

    # Filter to holdout
    ho_mask = common >= pd.Timestamp(HO_START, tz='UTC')
    return strat_ret[ho_mask]


def main():
    t0 = time.time()
    print("=" * 70)
    print("Multi-Strategy Portfolio Assembly")
    print("=" * 70)

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    # Load phase results
    try:
        with open('/home/ubuntu/projects/crypto-trader/phase1_lgbm_btc_results.json') as f:
            p1 = json.load(f)
    except FileNotFoundError:
        p1 = {'summary': {'best_wf_pct': 0}}

    try:
        with open('/home/ubuntu/projects/crypto-trader/phase2_kyle_ls_results.json') as f:
            p2 = json.load(f)
    except FileNotFoundError:
        p2 = {'wf_pct': 0, 'qualifies': False}

    try:
        with open('/home/ubuntu/projects/crypto-trader/phase3_btc_eth_arb_results.json') as f:
            p3 = json.load(f)
    except FileNotFoundError:
        p3 = {'qualifies': False}

    strategies = {}
    strategy_returns = {}

    # ── Strategy 1: Daily MA (BTC) — always included ──────────────
    print("\n--- Daily MA BTC ---")
    btc_daily = dm.get_ohlcv('BTC/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_ma_btc, bundle_ma_btc = get_daily_ma_returns(dm, 'BTC/USDT:USDT', btc_daily, 26, 0.0)
    s_btc = sharpe_from_returns(ret_ma_btc)
    print(f"  Sharpe: {s_btc:.2f}, trades: {len(bundle_ma_btc.trades)}")
    strategies['daily_ma_btc'] = {'sharpe': s_btc, 'trades': len(bundle_ma_btc.trades)}
    strategy_returns['daily_ma_btc'] = ret_ma_btc

    # ── Strategy 2: Daily MA (ETH) ───────────────────────────────
    print("\n--- Daily MA ETH ---")
    eth_daily = dm.get_ohlcv('ETH/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_ma_eth, bundle_ma_eth = get_daily_ma_returns(dm, 'ETH/USDT:USDT', eth_daily, 26, 0.0)
    s_eth = sharpe_from_returns(ret_ma_eth)
    print(f"  Sharpe: {s_eth:.2f}, trades: {len(bundle_ma_eth.trades)}")
    strategies['daily_ma_eth'] = {'sharpe': s_eth, 'trades': len(bundle_ma_eth.trades)}
    strategy_returns['daily_ma_eth'] = ret_ma_eth

    # ── Strategy 3: Daily MA (SOL) ───────────────────────────────
    print("\n--- Daily MA SOL ---")
    sol_daily = dm.get_ohlcv('SOL/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_ma_sol, bundle_ma_sol = get_daily_ma_returns(dm, 'SOL/USDT:USDT', sol_daily, 26, 0.0)
    s_sol = sharpe_from_returns(ret_ma_sol)
    print(f"  Sharpe: {s_sol:.2f}, trades: {len(bundle_ma_sol.trades)}")
    strategies['daily_ma_sol'] = {'sharpe': s_sol, 'trades': len(bundle_ma_sol.trades)}
    strategy_returns['daily_ma_sol'] = ret_ma_sol

    # ── Strategy 4: Phase 1 LightGBM (if WF >= 60%) ──────────────
    p1_wf = p1.get('summary', {}).get('best_wf_pct', 0)
    if isinstance(p1_wf, str):
        p1_wf = 0
    # WF pct might be 0-1 or 0-100
    p1_wf_norm = p1_wf if p1_wf <= 1 else p1_wf / 100
    p1_qualifies = p1_wf_norm >= 0.60
    print(f"\n--- Phase 1 LightGBM BTC: WF={p1_wf_norm:.1%} -> {'INCLUDED' if p1_qualifies else 'EXCLUDED'} ---")
    if p1_qualifies:
        ho_sharpe = p1.get('summary', {}).get('holdout_sharpe', 0)
        strategies['lgbm_btc'] = {'sharpe': ho_sharpe or 0}
        print(f"  Holdout Sharpe: {ho_sharpe}")

    # ── Strategy 5: Phase 2 Kyle+LS (if WF >= 55%) ───────────────
    p2_wf = p2.get('wf_pct', 0)
    p2_wf_norm = p2_wf if p2_wf <= 1 else p2_wf / 100
    p2_qualifies = p2_wf_norm >= 0.55
    print(f"\n--- Phase 2 Kyle+LS: WF={p2_wf_norm:.1%} -> {'INCLUDED' if p2_qualifies else 'EXCLUDED'} ---")
    if p2_qualifies:
        ho_sharpe = p2.get('holdout', {}).get('sharpe', 0)
        strategies['kyle_ls'] = {'sharpe': ho_sharpe or 0}
        print(f"  Holdout Sharpe: {ho_sharpe}")

    # ── Strategy 6: Phase 3 BTC/ETH arb (if 3+ positive) ─────────
    p3_qualifies = p3.get('qualifies', False)
    p3_positive = p3.get('positive_holdout_count', 0)
    print(f"\n--- Phase 3 BTC/ETH Arb: {p3_positive} positive -> {'INCLUDED' if p3_qualifies else 'EXCLUDED'} ---")
    if p3_qualifies:
        best_p3 = p3.get('best_params', {})
        p3_ret = get_phase3_returns(dm, best_p3.get('z_window', 168),
                                     best_p3.get('entry_z', 2.0),
                                     best_p3.get('exit_z', 0.5))
        s_p3 = sharpe_from_returns(p3_ret)
        strategies['btc_eth_arb'] = {'sharpe': s_p3}
        strategy_returns['btc_eth_arb'] = p3_ret
        print(f"  Holdout Sharpe: {s_p3:.2f}")

    # ── Correlation matrix ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Correlation Matrix (holdout period)")
    print("=" * 50)

    # Align all return series
    all_names = list(strategy_returns.keys())
    if len(all_names) >= 2:
        ret_df = pd.DataFrame(strategy_returns)
        ret_df = ret_df.fillna(0)
        corr = ret_df.corr()
        print(corr.round(2).to_string())

        # Check for high correlations
        dropped = set()
        for i in range(len(all_names)):
            for j in range(i+1, len(all_names)):
                c = corr.iloc[i, j]
                if abs(c) > 0.7:
                    # Drop the one with lower Sharpe
                    s_i = strategies[all_names[i]].get('sharpe', 0)
                    s_j = strategies[all_names[j]].get('sharpe', 0)
                    drop_name = all_names[j] if s_i >= s_j else all_names[i]
                    print(f"\n  HIGH CORR: {all_names[i]} vs {all_names[j]} = {c:.2f}")
                    print(f"  -> Dropping {drop_name} (lower Sharpe)")
                    dropped.add(drop_name)

        for d in dropped:
            if d in strategy_returns:
                del strategy_returns[d]
            if d in strategies:
                del strategies[d]
    else:
        corr = pd.DataFrame()

    # ── Combined portfolio ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Combined Portfolio (equal weight)")
    print("=" * 50)

    remaining = list(strategy_returns.keys())
    print(f"Strategies: {remaining}")

    if len(remaining) > 0:
        combined_df = pd.DataFrame(strategy_returns).fillna(0)
        # Equal weight
        combined_ret = combined_df.mean(axis=1)

        # Metrics
        combined_sharpe = sharpe_from_returns(combined_ret)
        equity = (1 + combined_ret).cumprod()
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = float(drawdown.min()) * 100

        monthly = combined_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        worst_month = float(monthly.min()) * 100
        mean_monthly = float(monthly.mean()) * 100

        # Total trades across strategies
        total_trades = sum(strategies[s].get('trades', 0) for s in remaining)
        ho_days = (combined_ret.index[-1] - combined_ret.index[0]).days
        total_tpm = total_trades / max(ho_days / 30.44, 1)

        # Capital for 1000/month at current monthly return
        if mean_monthly > 0:
            capital_needed = 1000 / (mean_monthly / 100)
        else:
            capital_needed = float('inf')

        print(f"\n  Combined Sharpe:     {combined_sharpe:.2f}")
        print(f"  Max Drawdown:        {max_dd:.1f}%")
        print(f"  Worst Month:         {worst_month:.1f}%")
        print(f"  Mean Monthly Return: {mean_monthly:.2f}%")
        print(f"  Trades/Month:        {total_tpm:.1f}")
        print(f"  Capital for GBP1000/mo: GBP{capital_needed:,.0f}")

        results = {
            'strategies_included': remaining,
            'strategies_excluded_corr': list(dropped) if 'dropped' in dir() else [],
            'combined_sharpe': round(combined_sharpe, 2),
            'max_dd_pct': round(max_dd, 1),
            'worst_month_pct': round(worst_month, 1),
            'mean_monthly_pct': round(mean_monthly, 2),
            'trades_per_month': round(total_tpm, 1),
            'capital_for_1000_per_month_gbp': round(capital_needed, 0),
            'correlation_matrix': corr.round(4).to_dict() if len(corr) > 0 else {},
            'individual_sharpes': {k: round(v.get('sharpe', 0), 2) for k, v in strategies.items()},
            'phase1_included': p1_qualifies,
            'phase2_included': p2_qualifies,
            'phase3_included': p3_qualifies,
        }
    else:
        combined_sharpe = 0
        total_tpm = 0
        results = {'error': 'No strategies available'}
        print("  No strategies available!")

    results['elapsed_seconds'] = time.time() - t0

    with open('/home/ubuntu/projects/crypto-trader/portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to portfolio_results.json ({time.time()-t0:.0f}s)")

    return combined_sharpe, total_tpm


if __name__ == '__main__':
    combined_sharpe, trades_mo = main()
    print(f"\nFinal: combined_sharpe={combined_sharpe:.2f} trades_mo={trades_mo:.0f}")
