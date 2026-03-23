"""
Task 2: Portfolio at increased leverage.

Combines four strategies with specified sizing:
  - Daily MA BTC:  fraction=0.20, leverage=3.0 (0.6x effective)
  - Daily MA ETH:  fraction=0.20, leverage=3.0 (0.6x effective)
  - Daily MA SOL:  fraction=0.10, leverage=3.0 (0.3x effective)
  - Stat arb:      fraction=0.36 (from Task 1, keeps max DD < 15%)

Reports: combined Sharpe, max DD, worst month, trades/month, capital for GBP1000/mo.
Also computes maximum single-day equity loss for prop firm 5% daily limit check.
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
from strategies.daily_ma_sol import DailyMASOL

HO_START = '2024-01-01'
HO_END = '2026-03-23'
TF_4H = '4h'
TF_1H = '1h'

# Strategy sizing
MA_BTC_FRACTION = 0.20
MA_ETH_FRACTION = 0.20
MA_SOL_FRACTION = 0.10
ARB_FRACTION = 0.36  # From Task 1: keeps holdout max DD < 15%

# Stat arb parameters
Z_WINDOW = 336
ENTRY_Z = 2.5
EXIT_Z = 0.5
COST_PER_TRADE = 0.001


def sharpe_from_returns(ret):
    """Annualised Sharpe from hourly returns."""
    if ret.std() == 0 or len(ret) < 100:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(8760))


def get_daily_ma_returns(dm, symbol, daily_ohlcv, fraction, ma_period=26, buffer_pct=0.0):
    """Get hourly returns for daily MA strategy."""
    signal_mod = DailyMASOL(daily_ohlcv=daily_ohlcv)
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=fraction, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    params = {'ma_period': ma_period, 'buffer_pct': buffer_pct}
    bundle = engine.run(signal_mod, params, symbol, TF_4H, HO_START, HO_END, 1000, 'holdout')

    eq = bundle.equity_curve
    ret = eq.pct_change().dropna()
    ret_1h = ret.resample('1h').sum().fillna(0)
    return ret_1h, bundle


def get_arb_returns(dm, z_window, entry_z, exit_z, fraction):
    """Get hourly returns for BTC/ETH stat arb at given fraction."""
    btc = dm.get_ohlcv('BTC/USDT:USDT', TF_1H, '2020-09-01', HO_END)
    eth = dm.get_ohlcv('ETH/USDT:USDT', TF_1H, '2020-09-01', HO_END)

    common = btc.index.intersection(eth.index)
    btc_close = btc.loc[common, 'close']
    eth_close = eth.loc[common, 'close']

    ratio = btc_close / eth_close
    ratio_mean = ratio.rolling(z_window, min_periods=z_window // 2).mean()
    ratio_std = ratio.rolling(z_window, min_periods=z_window // 2).std()
    z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    position = pd.Series(0.0, index=z.index)
    current_pos = 0
    n_trades = 0
    for i in range(1, len(z)):
        zval = z.iloc[i]
        if np.isnan(zval):
            position.iloc[i] = current_pos
            continue
        if current_pos == 0:
            if zval > entry_z:
                current_pos = -1
                n_trades += 1
            elif zval < -entry_z:
                current_pos = 1
                n_trades += 1
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

    strat_ret = position.shift(1).fillna(0) * spread_ret * fraction
    pos_changes = position.diff().abs().fillna(0)
    strat_ret = strat_ret - pos_changes * COST_PER_TRADE / 2 * fraction

    # Filter to holdout
    ho_mask = common >= pd.Timestamp(HO_START, tz='UTC')
    ho_ret = strat_ret[ho_mask]

    # Count trades in holdout
    ho_trades = int(position[ho_mask].diff().abs().fillna(0).gt(0).sum())

    return ho_ret, ho_trades


def main():
    t0 = time.time()
    print("=" * 80)
    print("Task 2: Portfolio at Increased Leverage")
    print("=" * 80)

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    strategy_returns = {}
    strategy_info = {}

    # ── Daily MA BTC (fraction=0.20, 0.6x effective) ──
    print(f"\n--- Daily MA BTC (fraction={MA_BTC_FRACTION}, eff_lev={MA_BTC_FRACTION*3:.1f}x) ---")
    btc_daily = dm.get_ohlcv('BTC/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_btc, bundle_btc = get_daily_ma_returns(dm, 'BTC/USDT:USDT', btc_daily, MA_BTC_FRACTION)
    s_btc = sharpe_from_returns(ret_btc)
    print(f"  Sharpe: {s_btc:.2f}, trades: {len(bundle_btc.trades)}")
    strategy_returns['daily_ma_btc'] = ret_btc
    strategy_info['daily_ma_btc'] = {
        'sharpe': round(s_btc, 2),
        'trades': len(bundle_btc.trades),
        'fraction': MA_BTC_FRACTION,
        'effective_leverage': MA_BTC_FRACTION * 3,
    }

    # ── Daily MA ETH (fraction=0.20, 0.6x effective) ──
    print(f"\n--- Daily MA ETH (fraction={MA_ETH_FRACTION}, eff_lev={MA_ETH_FRACTION*3:.1f}x) ---")
    eth_daily = dm.get_ohlcv('ETH/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_eth, bundle_eth = get_daily_ma_returns(dm, 'ETH/USDT:USDT', eth_daily, MA_ETH_FRACTION)
    s_eth = sharpe_from_returns(ret_eth)
    print(f"  Sharpe: {s_eth:.2f}, trades: {len(bundle_eth.trades)}")
    strategy_returns['daily_ma_eth'] = ret_eth
    strategy_info['daily_ma_eth'] = {
        'sharpe': round(s_eth, 2),
        'trades': len(bundle_eth.trades),
        'fraction': MA_ETH_FRACTION,
        'effective_leverage': MA_ETH_FRACTION * 3,
    }

    # ── Daily MA SOL (fraction=0.10, 0.3x effective) ──
    print(f"\n--- Daily MA SOL (fraction={MA_SOL_FRACTION}, eff_lev={MA_SOL_FRACTION*3:.1f}x) ---")
    sol_daily = dm.get_ohlcv('SOL/USDT:USDT', '1d', '2020-01-01', HO_END)
    ret_sol, bundle_sol = get_daily_ma_returns(dm, 'SOL/USDT:USDT', sol_daily, MA_SOL_FRACTION)
    s_sol = sharpe_from_returns(ret_sol)
    print(f"  Sharpe: {s_sol:.2f}, trades: {len(bundle_sol.trades)}")
    strategy_returns['daily_ma_sol'] = ret_sol
    strategy_info['daily_ma_sol'] = {
        'sharpe': round(s_sol, 2),
        'trades': len(bundle_sol.trades),
        'fraction': MA_SOL_FRACTION,
        'effective_leverage': MA_SOL_FRACTION * 3,
    }

    # ── Stat Arb BTC/ETH (fraction=0.36) ──
    print(f"\n--- Stat Arb BTC/ETH (fraction={ARB_FRACTION}) ---")
    ret_arb, arb_trades = get_arb_returns(dm, Z_WINDOW, ENTRY_Z, EXIT_Z, ARB_FRACTION)
    s_arb = sharpe_from_returns(ret_arb)
    print(f"  Sharpe: {s_arb:.2f}, trades: {arb_trades}")
    strategy_returns['btc_eth_arb'] = ret_arb
    strategy_info['btc_eth_arb'] = {
        'sharpe': round(s_arb, 2),
        'trades': arb_trades,
        'fraction': ARB_FRACTION,
    }

    # ── Correlation Matrix ──
    print("\n" + "=" * 60)
    print("Correlation Matrix (holdout)")
    print("=" * 60)
    ret_df = pd.DataFrame(strategy_returns).fillna(0)
    corr = ret_df.corr()
    print(corr.round(3).to_string())

    # ── Individual strategy metrics ──
    print("\n" + "=" * 60)
    print("Individual Strategy Metrics (holdout)")
    print("=" * 60)
    for name, ret in strategy_returns.items():
        eq = (1 + ret).cumprod()
        dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        monthly = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        worst_mo = monthly.min() * 100
        mean_mo = monthly.mean() * 100
        daily_ret = ret.resample('1D').sum()
        worst_day = daily_ret.min() * 100
        print(f"  {name:15s}: Sharpe={strategy_info[name]['sharpe']:5.2f}, "
              f"MaxDD={dd:6.1f}%, WrstMo={worst_mo:6.1f}%, "
              f"Monthly={mean_mo:5.2f}%, WrstDay={worst_day:5.2f}%")

    # ── Combined Portfolio (equal-weight) ──
    print("\n" + "=" * 60)
    print("Combined Portfolio (equal-weight sum of all four)")
    print("=" * 60)

    combined_ret = ret_df.mean(axis=1)

    # Metrics
    combined_sharpe = sharpe_from_returns(combined_ret)
    equity = (1 + combined_ret).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min()) * 100

    monthly = combined_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    worst_month = float(monthly.min()) * 100
    mean_monthly = float(monthly.mean()) * 100

    # Total trades
    total_trades = sum(info['trades'] for info in strategy_info.values())
    ho_days = (combined_ret.index[-1] - combined_ret.index[0]).days
    total_tpm = total_trades / max(ho_days / 30.44, 1)

    # Capital for GBP1000/month
    capital_needed = 1000 / (mean_monthly / 100) if mean_monthly > 0 else float('inf')

    # Maximum single-day equity loss
    daily_combined = combined_ret.resample('1D').sum()
    worst_day_pct = float(daily_combined.min()) * 100
    best_day_pct = float(daily_combined.max()) * 100

    # 5% daily limit check
    daily_safe = worst_day_pct >= -5.0

    print(f"\n  Combined daily Sharpe:    {combined_sharpe:.2f}")
    print(f"  Max Drawdown:            {max_dd:.1f}%")
    print(f"  Worst Month:             {worst_month:.1f}%")
    print(f"  Mean Monthly Return:     {mean_monthly:.2f}%")
    print(f"  Trades/Month:            {total_tpm:.1f}")
    cap_str = f"GBP{capital_needed:,.0f}" if capital_needed < 1e9 else "N/A"
    print(f"  Capital for GBP1000/mo:  {cap_str}")
    print(f"\n  --- Prop Firm Daily Drawdown Check ---")
    print(f"  Worst single-day loss:   {worst_day_pct:.2f}%")
    print(f"  Best single-day gain:    {best_day_pct:.2f}%")
    print(f"  Safe under 5% daily DD:  {'YES' if daily_safe else 'NO'}")

    if not daily_safe:
        # Find what scaling keeps worst day above -5%
        for scale in np.arange(0.10, 1.01, 0.01):
            scaled_daily = daily_combined * scale
            if scaled_daily.min() * 100 >= -5.0:
                print(f"  Scale factor for 5% safety: {scale:.2f} "
                      f"(worst day = {scaled_daily.min()*100:.2f}%)")
                # Recompute monthly at this scale
                scaled_combined = combined_ret * scale
                scaled_monthly = scaled_combined.resample('ME').apply(lambda x: (1 + x).prod() - 1)
                scaled_mean_mo = scaled_monthly.mean() * 100
                scaled_cap = 1000 / (scaled_mean_mo / 100) if scaled_mean_mo > 0 else float('inf')
                print(f"  At this scale: monthly={scaled_mean_mo:.2f}%, "
                      f"capital={f'GBP{scaled_cap:,.0f}' if scaled_cap < 1e9 else 'N/A'}")
                break

    # Bottom 10 worst days
    print(f"\n  Worst 10 trading days:")
    worst_days = daily_combined.nsmallest(10)
    for dt, val in worst_days.items():
        print(f"    {dt.strftime('%Y-%m-%d')}: {val*100:+.2f}%")

    # ── Results ──
    results = {
        'strategy_config': {
            'daily_ma_btc': {'fraction': MA_BTC_FRACTION, 'eff_leverage': MA_BTC_FRACTION * 3},
            'daily_ma_eth': {'fraction': MA_ETH_FRACTION, 'eff_leverage': MA_ETH_FRACTION * 3},
            'daily_ma_sol': {'fraction': MA_SOL_FRACTION, 'eff_leverage': MA_SOL_FRACTION * 3},
            'btc_eth_arb': {'fraction': ARB_FRACTION},
        },
        'individual_metrics': strategy_info,
        'correlation_matrix': corr.round(4).to_dict(),
        'combined_portfolio': {
            'sharpe': round(combined_sharpe, 2),
            'max_dd_pct': round(max_dd, 1),
            'worst_month_pct': round(worst_month, 1),
            'mean_monthly_pct': round(mean_monthly, 2),
            'trades_per_month': round(total_tpm, 1),
            'capital_for_1000_gbp': round(capital_needed, 0) if capital_needed < 1e9 else None,
        },
        'daily_drawdown_check': {
            'worst_day_pct': round(worst_day_pct, 2),
            'safe_under_5pct': daily_safe,
        },
        'elapsed_seconds': time.time() - t0,
    }

    with open('/home/ubuntu/projects/crypto-trader/task2_portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to task2_portfolio_results.json ({time.time()-t0:.0f}s)")
    return results


if __name__ == '__main__':
    results = main()
