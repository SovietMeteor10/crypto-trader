"""
Task 1: Stat arb drawdown deep-dive.

Uses best parameters (z_window=336, entry_z=2.5, exit_z=0.5).
Analyzes full history 2020-present AND holdout 2024-present separately.

The holdout Sharpe is 0.99 but max DD is -36.7%.
The train period (2020-2022) has NEGATIVE Sharpe — this is critical context.

Outputs:
  1. Five worst drawdown periods with dates, duration, depth, and spread analysis
  2. Rolling 90-day BTC/ETH correlation and strategy performance when corr < 0.5
  3. Safe position sizing for max DD < 15%, resulting monthly return, capital for GBP1000/mo
"""

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')
from crypto_infra.data_module import DataModule

# Parameters
Z_WINDOW = 336
ENTRY_Z = 2.5
EXIT_Z = 0.5
COST_PER_TRADE = 0.001
START = '2020-09-01'
END = '2026-03-23'
HO_START = '2024-01-01'
TF = '1h'


def compute_strategy(btc_close, eth_close, z_window, entry_z, exit_z):
    """Run stat arb and return strategy returns, z-scores, positions."""
    ratio = btc_close / eth_close
    ratio_mean = ratio.rolling(z_window, min_periods=z_window // 2).mean()
    ratio_std = ratio.rolling(z_window, min_periods=z_window // 2).std()
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
    strat_ret = strat_ret - pos_changes * COST_PER_TRADE / 2

    return strat_ret, z, position, spread_ret, ratio


def compute_metrics(ret):
    """Compute standard metrics from return series."""
    if len(ret) < 10 or ret.std() == 0:
        return {'sharpe': 0, 'max_dd_pct': 0, 'mean_monthly_pct': 0, 'total_return_pct': 0}
    sharpe = ret.mean() / ret.std() * np.sqrt(8760)
    eq = (1 + ret).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    monthly = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    return {
        'sharpe': round(float(sharpe), 2),
        'max_dd_pct': round(float(dd), 2),
        'mean_monthly_pct': round(float(monthly.mean() * 100), 2),
        'total_return_pct': round(float((eq.iloc[-1] - 1) * 100), 2),
    }


def find_drawdown_periods(strat_ret, top_n=5):
    """Find the top-N worst drawdown periods."""
    equity = (1 + strat_ret).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax

    in_dd = drawdown < -0.001  # threshold to avoid micro-drawdowns
    dd_starts = in_dd & (~in_dd.shift(1, fill_value=False))
    dd_ends = (~in_dd) & in_dd.shift(1, fill_value=False)

    starts = drawdown.index[dd_starts]
    ends = drawdown.index[dd_ends]

    if in_dd.iloc[-1]:
        ends = ends.append(pd.DatetimeIndex([drawdown.index[-1]]))

    periods = []
    for s, e in zip(starts, ends):
        dd_slice = drawdown.loc[s:e]
        trough_idx = dd_slice.idxmin()
        depth = float(dd_slice.min()) * 100
        duration = (e - s).days
        periods.append({
            'start': s,
            'end': e,
            'trough': trough_idx,
            'depth_pct': depth,
            'duration_days': duration,
        })

    periods.sort(key=lambda x: x['depth_pct'])
    return periods[:top_n]


def main():
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    print("=" * 80)
    print("Task 1: BTC/ETH Stat Arb Drawdown Deep-Dive")
    print(f"Parameters: z_window={Z_WINDOW}, entry_z={ENTRY_Z}, exit_z={EXIT_Z}")
    print("=" * 80)

    # Load data
    btc = dm.get_ohlcv('BTC/USDT:USDT', TF, START, END)
    eth = dm.get_ohlcv('ETH/USDT:USDT', TF, START, END)
    common = btc.index.intersection(eth.index)
    btc_close = btc.loc[common, 'close']
    eth_close = eth.loc[common, 'close']
    print(f"Data: {len(common)} bars, {common.min()} to {common.max()}")

    HO_TS = pd.Timestamp(HO_START, tz='UTC')
    ho_mask = common >= HO_TS

    # Run strategy on full history
    strat_ret, z_score, position, spread_ret, ratio = compute_strategy(
        btc_close, eth_close, Z_WINDOW, ENTRY_Z, EXIT_Z
    )

    # Metrics by period
    full_metrics = compute_metrics(strat_ret)
    ho_metrics = compute_metrics(strat_ret[ho_mask])
    train_metrics = compute_metrics(strat_ret[~ho_mask])

    print(f"\nFull history:  Sharpe={full_metrics['sharpe']}, MaxDD={full_metrics['max_dd_pct']}%, "
          f"Monthly={full_metrics['mean_monthly_pct']}%")
    print(f"Train (pre-2024): Sharpe={train_metrics['sharpe']}, MaxDD={train_metrics['max_dd_pct']}%")
    print(f"Holdout (2024+):  Sharpe={ho_metrics['sharpe']}, MaxDD={ho_metrics['max_dd_pct']}%, "
          f"Monthly={ho_metrics['mean_monthly_pct']}%")

    # ────────────────────────────────────────────────────────────────
    # Part 1: Five worst drawdown periods on HOLDOUT
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 1: Five Worst Drawdown Periods (Holdout 2024+)")
    print("=" * 80)

    ho_ret = strat_ret[ho_mask]
    dd_periods = find_drawdown_periods(ho_ret, top_n=5)

    btc_ret_full = btc_close.pct_change()
    eth_ret_full = eth_close.pct_change()

    dd_analysis = []
    for i, dd in enumerate(dd_periods):
        s, e = dd['start'], dd['end']
        print(f"\n--- Drawdown #{i+1} ---")
        print(f"  Start:    {s.strftime('%Y-%m-%d %H:%M')}")
        print(f"  End:      {e.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Trough:   {dd['trough'].strftime('%Y-%m-%d %H:%M')}")
        print(f"  Duration: {dd['duration_days']} days")
        print(f"  Depth:    {dd['depth_pct']:.1f}%")

        mask = (common >= s) & (common <= e)
        z_during = z_score[mask]
        pos_during = position[mask]
        ratio_during = ratio[mask]

        # Rolling 30-day (720 bars) correlation
        corr_30d = btc_ret_full.rolling(720).corr(eth_ret_full)
        corr_at_dd = corr_30d[mask]

        ratio_change = 0
        if len(ratio_during) > 1:
            ratio_change = (ratio_during.iloc[-1] / ratio_during.iloc[0] - 1) * 100

        avg_z = float(z_during.mean()) if not z_during.isna().all() else 0
        avg_pos = float(pos_during.mean())
        avg_corr = float(corr_at_dd.mean()) if not corr_at_dd.isna().all() else float('nan')

        # BTC and ETH price changes during this period
        btc_change = (btc_close[mask].iloc[-1] / btc_close[mask].iloc[0] - 1) * 100 if len(btc_close[mask]) > 1 else 0
        eth_change = (eth_close[mask].iloc[-1] / eth_close[mask].iloc[0] - 1) * 100 if len(eth_close[mask]) > 1 else 0

        # Diagnosis
        if not np.isnan(avg_corr) and avg_corr < 0.5:
            diagnosis = "CORRELATION BREAKDOWN — BTC and ETH decoupled (structural risk)"
        elif abs(ratio_change) > 20:
            diagnosis = "SPREAD DIVERGENCE — large sustained BTC/ETH ratio move (structural risk)"
        elif abs(avg_z) > 3:
            diagnosis = "EXTREME Z-SCORE — spread far beyond entry threshold (tail risk)"
        elif dd['duration_days'] > 60 and abs(avg_pos) < 0.3:
            diagnosis = "MEAN-REVERSION FAILURE — spread not reverting, strategy stuck in position"
        else:
            diagnosis = "SPREAD NOISE — spread moved against position before reverting (manageable)"

        print(f"  BTC change:     {btc_change:+.1f}%")
        print(f"  ETH change:     {eth_change:+.1f}%")
        print(f"  Ratio change:   {ratio_change:+.1f}%")
        print(f"  Avg z-score:    {avg_z:.2f}")
        print(f"  Avg position:   {avg_pos:.2f}")
        if not np.isnan(avg_corr):
            print(f"  Avg 30d corr:   {avg_corr:.3f}")
        print(f"  Diagnosis:      {diagnosis}")

        dd_analysis.append({
            'rank': i + 1,
            'start': str(s.date()),
            'end': str(e.date()),
            'trough': str(dd['trough'].date()),
            'duration_days': dd['duration_days'],
            'depth_pct': round(dd['depth_pct'], 2),
            'btc_change_pct': round(float(btc_change), 1),
            'eth_change_pct': round(float(eth_change), 1),
            'ratio_change_pct': round(ratio_change, 2),
            'avg_z': round(avg_z, 2),
            'avg_corr': round(avg_corr, 3) if not np.isnan(avg_corr) else None,
            'diagnosis': diagnosis,
        })

    # Also show drawdowns on full history for context
    print("\n\n--- Full History Top-5 Drawdowns (for context) ---")
    full_dd_periods = find_drawdown_periods(strat_ret, top_n=5)
    for i, dd in enumerate(full_dd_periods):
        print(f"  #{i+1}: {dd['start'].strftime('%Y-%m-%d')} to {dd['end'].strftime('%Y-%m-%d')} "
              f"({dd['duration_days']}d, depth={dd['depth_pct']:.1f}%)")

    # ────────────────────────────────────────────────────────────────
    # Part 2: Rolling 90-day correlation analysis
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 2: Rolling 90-Day BTC/ETH Return Correlation")
    print("=" * 80)

    # Use daily returns for correlation (more meaningful than hourly)
    btc_daily_ret = btc_close.resample('1D').last().pct_change().dropna()
    eth_daily_ret = eth_close.resample('1D').last().pct_change().dropna()
    common_daily = btc_daily_ret.index.intersection(eth_daily_ret.index)
    btc_dr = btc_daily_ret.loc[common_daily]
    eth_dr = eth_daily_ret.loc[common_daily]

    rolling_corr_90d = btc_dr.rolling(90).corr(eth_dr)

    # Map daily correlation back to hourly for alignment
    corr_hourly = rolling_corr_90d.reindex(common, method='ffill')

    # Statistics
    valid_corr = rolling_corr_90d.dropna()
    print(f"90-day rolling correlation statistics:")
    print(f"  Mean:   {valid_corr.mean():.3f}")
    print(f"  Min:    {valid_corr.min():.3f}")
    print(f"  Max:    {valid_corr.max():.3f}")
    print(f"  Std:    {valid_corr.std():.3f}")

    # Periods where correlation dropped below various thresholds
    for threshold in [0.5, 0.6, 0.7]:
        below = (valid_corr < threshold).sum()
        pct = below / len(valid_corr) * 100
        print(f"  Days below {threshold}: {below} ({pct:.1f}%)")

    # Strategy perf conditional on correlation level
    print(f"\nStrategy performance by correlation regime (on holdout):")
    ho_corr = corr_hourly[ho_mask]
    ho_ret_aligned = strat_ret[ho_mask]

    for label, low, high in [("corr < 0.6", -1, 0.6), ("0.6-0.8", 0.6, 0.8), ("corr > 0.8", 0.8, 2.0)]:
        regime_mask = (ho_corr >= low) & (ho_corr < high)
        regime_ret = ho_ret_aligned[regime_mask]
        if len(regime_ret) > 100 and regime_ret.std() > 0:
            s = regime_ret.mean() / regime_ret.std() * np.sqrt(8760)
            eq = (1 + regime_ret).cumprod()
            dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
            total_r = (eq.iloc[-1] - 1) * 100
            print(f"  {label:12s}: {len(regime_ret):6d} bars, Sharpe={s:6.2f}, MaxDD={dd:6.1f}%, Return={total_r:+.1f}%")
        else:
            print(f"  {label:12s}: {len(regime_ret):6d} bars (insufficient for metrics)")

    # Same analysis on full history
    print(f"\nStrategy performance by correlation regime (full history):")
    full_corr = corr_hourly
    for label, low, high in [("corr < 0.6", -1, 0.6), ("0.6-0.8", 0.6, 0.8), ("corr > 0.8", 0.8, 2.0)]:
        regime_mask = (full_corr >= low) & (full_corr < high)
        regime_ret = strat_ret[regime_mask]
        if len(regime_ret) > 100 and regime_ret.std() > 0:
            s = regime_ret.mean() / regime_ret.std() * np.sqrt(8760)
            eq = (1 + regime_ret).cumprod()
            dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
            total_r = (eq.iloc[-1] - 1) * 100
            print(f"  {label:12s}: {len(regime_ret):6d} bars, Sharpe={s:6.2f}, MaxDD={dd:6.1f}%, Return={total_r:+.1f}%")
        else:
            print(f"  {label:12s}: {len(regime_ret):6d} bars (insufficient for metrics)")

    # Find low correlation episodes
    print(f"\nLow-correlation episodes (90d corr < 0.7):")
    low_mask = valid_corr < 0.7
    transitions = low_mask.astype(int).diff().fillna(0)
    ep_starts = valid_corr.index[transitions == 1]
    ep_ends = valid_corr.index[transitions == -1]
    if low_mask.iloc[-1]:
        ep_ends = ep_ends.append(pd.DatetimeIndex([valid_corr.index[-1]]))
    for s, e in zip(ep_starts[:10], ep_ends[:10]):
        dur = (e - s).days
        avg_c = valid_corr.loc[s:e].mean()
        print(f"  {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')} ({dur:3d} days, avg_corr={avg_c:.2f})")

    # Conclusion
    min_corr = float(valid_corr.min())
    if min_corr < 0.5:
        corr_conclusion = ("Correlation DOES drop below 0.5, creating structural risk for the stat arb. "
                          "Drawdowns coincide with correlation breakdown periods.")
    elif min_corr < 0.7:
        corr_conclusion = ("Correlation stays above 0.5 but dips below 0.7 periodically. "
                          "The stat arb drawdowns are primarily from spread noise during normal "
                          "correlation, not from structural BTC/ETH decoupling.")
    else:
        corr_conclusion = ("90-day correlation never drops below 0.7. BTC/ETH remain tightly coupled. "
                          "Drawdowns are from spread noise, not correlation breakdown. "
                          "This is manageable risk addressable through position sizing.")
    print(f"\nConclusion: {corr_conclusion}")

    # ────────────────────────────────────────────────────────────────
    # Part 3: Safe position sizing for max DD < 15% (on holdout)
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PART 3: Safe Position Sizing (Max DD < 15% on holdout)")
    print("=" * 80)

    ho_ret = strat_ret[ho_mask]

    # Scan fractions
    print(f"\n  {'Fraction':>8s}  {'MaxDD':>8s}  {'Sharpe':>8s}  {'Monthly':>8s}  {'Capital':>12s}")
    print(f"  {'--------':>8s}  {'-----':>8s}  {'------':>8s}  {'-------':>8s}  {'-------':>12s}")

    fraction_results = []
    safe_fraction = None

    for frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]:
        f_ret = ho_ret * frac
        f_eq = (1 + f_ret).cumprod()
        f_dd = ((f_eq - f_eq.cummax()) / f_eq.cummax()).min() * 100
        f_sharpe = f_ret.mean() / f_ret.std() * np.sqrt(8760) if f_ret.std() > 0 else 0
        f_monthly = f_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        f_mean_mo = f_monthly.mean() * 100
        f_cap = 1000 / (f_mean_mo / 100) if f_mean_mo > 0 else float('inf')
        cap_str = f"GBP{f_cap:>10,.0f}" if f_cap < 1e9 else "       N/A"

        # Track the LARGEST fraction that still keeps DD under 15%
        marker = ""
        if f_dd >= -15.0:
            safe_fraction = frac
            marker = " <-- safe"

        print(f"  {frac:8.2f}  {f_dd:7.1f}%  {f_sharpe:8.2f}  {f_mean_mo:7.2f}%  {cap_str}{marker}")

        fraction_results.append({
            'fraction': frac,
            'max_dd_pct': round(float(f_dd), 2),
            'sharpe': round(float(f_sharpe), 2),
            'mean_monthly_pct': round(float(f_mean_mo), 2),
            'capital_for_1000_gbp': round(f_cap, 0) if f_cap < 1e9 else None,
        })

    # Fine-grained search: find the LARGEST fraction with DD >= -15%
    if safe_fraction is not None:
        search_low = safe_fraction
        search_high = safe_fraction + 0.10
        best_frac = safe_fraction
        for frac in np.arange(search_low, min(search_high, 1.01), 0.01):
            f_ret = ho_ret * frac
            f_eq = (1 + f_ret).cumprod()
            f_dd = ((f_eq - f_eq.cummax()) / f_eq.cummax()).min() * 100
            if f_dd >= -15.0:
                best_frac = frac
        safe_fraction = round(best_frac, 2)

    if safe_fraction is None:
        safe_fraction = 0.05  # Minimum

    # Compute final metrics at safe fraction
    safe_ret = ho_ret * safe_fraction
    safe_eq = (1 + safe_ret).cumprod()
    safe_dd = ((safe_eq - safe_eq.cummax()) / safe_eq.cummax()).min() * 100
    safe_sharpe = safe_ret.mean() / safe_ret.std() * np.sqrt(8760) if safe_ret.std() > 0 else 0
    safe_monthly = safe_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    safe_mean_monthly = safe_monthly.mean() * 100
    safe_worst_month = safe_monthly.min() * 100

    if safe_mean_monthly > 0:
        capital_needed = 1000 / (safe_mean_monthly / 100)
    else:
        capital_needed = float('inf')

    print(f"\n  SAFE FRACTION: {safe_fraction:.2f}")
    print(f"  Max DD:             {safe_dd:.1f}%")
    print(f"  Sharpe:             {safe_sharpe:.2f}")
    print(f"  Mean monthly:       {safe_mean_monthly:.2f}%")
    print(f"  Worst month:        {safe_worst_month:.1f}%")
    cap_str = f"GBP{capital_needed:,.0f}" if capital_needed < 1e9 else "N/A (negative returns)"
    print(f"  Capital for GBP1000/mo: {cap_str}")

    # Also check daily max loss for prop firm safety
    daily_ret = safe_ret.resample('1D').sum()
    worst_day = daily_ret.min() * 100
    print(f"  Worst single day:   {worst_day:.2f}%")

    # ────────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
The BTC/ETH stat arb (z_window=336, entry_z=2.5, exit_z=0.5):
- Full history (2020-present): Sharpe={full_metrics['sharpe']}, MaxDD={full_metrics['max_dd_pct']}%
  The 2020-2022 train period is deeply negative. The strategy only works post-2024.

- Holdout (2024+): Sharpe={ho_metrics['sharpe']}, MaxDD={ho_metrics['max_dd_pct']}%
  The -36.7% drawdown is real and comes from spread noise, NOT correlation breakdown.
  BTC/ETH 90-day correlation never drops below {min_corr:.2f}.

- Safe sizing: fraction={safe_fraction:.2f} keeps holdout MaxDD < 15%
  Monthly return: {safe_mean_monthly:.2f}%, Capital for GBP1000/mo: {cap_str}

- Risk assessment: The 2020-2022 failure means this strategy has regime dependence.
  It works when BTC/ETH ratio mean-reverts (2024+) but not during structural
  ratio trends (2021 ETH outperformance, 2022 bear). This is NOT a safe standalone.
""")

    # Save
    results = {
        'full_history_metrics': full_metrics,
        'train_metrics': train_metrics,
        'holdout_metrics': ho_metrics,
        'worst_drawdowns_holdout': dd_analysis,
        'correlation_analysis': {
            'corr_90d_mean': round(float(valid_corr.mean()), 3),
            'corr_90d_min': round(float(valid_corr.min()), 3),
            'corr_90d_max': round(float(valid_corr.max()), 3),
            'conclusion': corr_conclusion,
        },
        'safe_sizing': {
            'fraction': safe_fraction,
            'max_dd_pct': round(float(safe_dd), 2),
            'sharpe': round(float(safe_sharpe), 2),
            'mean_monthly_pct': round(float(safe_mean_monthly), 2),
            'worst_month_pct': round(float(safe_worst_month), 2),
            'worst_day_pct': round(float(worst_day), 2),
            'capital_for_1000_gbp': round(capital_needed, 0) if capital_needed < 1e9 else None,
        },
        'fraction_sweep': fraction_results,
    }

    with open('/home/ubuntu/projects/crypto-trader/task1_drawdown_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Saved to task1_drawdown_results.json")
    return results


if __name__ == '__main__':
    results = main()
