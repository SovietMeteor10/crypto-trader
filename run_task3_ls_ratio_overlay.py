"""
Task 3: ls_ratio as continuous position sizing signal.

Instead of binary entry/exit based on ls_ratio, use it to continuously scale
the daily MA strategy position size:
  - Bottom tercile (crowd lightly positioned): full position size (1.0x)
  - Middle tercile: half position size (0.5x)
  - Top tercile (crowd heavily positioned): quarter position size (0.25x)

Tested on BTC and SOL daily MA strategies during the holdout period
where ls_ratio data is available (2024-01 to 2024-12).

Compares to flat (unscaled) position sizing baseline.
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

# The ls_ratio data ends at 2024-12-30
# Use 2024-01-01 to 2024-12-30 as the test window
TEST_START = '2024-01-01'
TEST_END = '2024-12-31'
TF_4H = '4h'
TF_1H = '1h'

# Use expanding window for tercile computation to avoid look-ahead bias
# The first year of ls_ratio data (2020-09 to 2023-12) serves as the initial window


def compute_metrics(ret, label=""):
    """Compute standard metrics from return series."""
    if len(ret) < 10 or ret.std() == 0:
        return {'sharpe': 0, 'max_dd_pct': 0, 'mean_monthly_pct': 0, 'worst_month_pct': 0}
    sharpe = ret.mean() / ret.std() * np.sqrt(8760)
    eq = (1 + ret).cumprod()
    dd = ((eq - eq.cummax()) / eq.cummax()).min() * 100
    monthly = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    worst_mo = monthly.min() * 100
    mean_mo = monthly.mean() * 100
    return {
        'sharpe': round(float(sharpe), 2),
        'max_dd_pct': round(float(dd), 2),
        'mean_monthly_pct': round(float(mean_mo), 2),
        'worst_month_pct': round(float(worst_mo), 2),
        'total_return_pct': round(float((eq.iloc[-1] - 1) * 100), 2),
    }


def get_daily_ma_signal(dm, symbol, daily_ohlcv, ma_period=26, buffer_pct=0.0):
    """Get the raw daily MA signal (not returns) at 4H resolution."""
    signal_mod = DailyMASOL(daily_ohlcv=daily_ohlcv)
    data_4h = dm.get_ohlcv(symbol, TF_4H, TEST_START, TEST_END)
    params = {'ma_period': ma_period, 'buffer_pct': buffer_pct}
    signal = signal_mod.generate(data_4h, params)
    return signal, data_4h


def backtest_with_scaling(data_4h, signal, scaling_series, fraction=0.20, leverage=3.0):
    """
    Manual backtest of daily MA with position sizing scaled by scaling_series.

    scaling_series: Series indexed like data_4h with values in [0, 1]
    representing the fraction of full position to take.

    Returns hourly return series.
    """
    close = data_4h['close']
    ret = close.pct_change()

    # Align scaling to 4H bars
    scaling_4h = scaling_series.reindex(data_4h.index, method='ffill').fillna(1.0)

    # Position: signal * scaling * effective_leverage
    effective_lev = fraction * leverage
    position_size = signal.shift(1).fillna(0) * scaling_4h * effective_lev

    # Strategy returns
    strat_ret = position_size * ret

    # Transaction costs on position size changes
    pos_changes = position_size.diff().abs().fillna(0)
    # Cost: 10 bps round trip
    strat_ret = strat_ret - pos_changes * 0.0005

    # Resample to 1H
    ret_1h = strat_ret.resample('1h').sum().fillna(0)
    return ret_1h, position_size


def compute_ls_scaling(ls_ratio_1h, method='expanding_tercile'):
    """
    Compute position scaling factor from ls_ratio using expanding window terciles.

    Uses data up to each point (no look-ahead) to determine tercile boundaries.

    Returns Series with values: 1.0 (bottom tercile), 0.5 (middle), 0.25 (top)
    """
    # Expanding quantiles — uses all data up to this point
    expanding_q33 = ls_ratio_1h.expanding(min_periods=720).quantile(0.333)  # ~30 days min
    expanding_q67 = ls_ratio_1h.expanding(min_periods=720).quantile(0.667)

    scaling = pd.Series(0.5, index=ls_ratio_1h.index)  # default middle

    # Bottom tercile: crowd lightly positioned -> full size
    bottom = ls_ratio_1h <= expanding_q33
    scaling[bottom] = 1.0

    # Middle tercile: half size
    middle = (ls_ratio_1h > expanding_q33) & (ls_ratio_1h <= expanding_q67)
    scaling[middle] = 0.5

    # Top tercile: crowd heavily positioned -> quarter size
    top = ls_ratio_1h > expanding_q67
    scaling[top] = 0.25

    return scaling


def test_asset(dm, symbol, asset_name, ls_ratio_1h, fraction):
    """Test ls_ratio scaling overlay on one asset."""
    print(f"\n{'='*60}")
    print(f"  {asset_name} Daily MA — ls_ratio scaling test")
    print(f"  fraction={fraction}, leverage=3.0, eff_lev={fraction*3:.1f}x")
    print(f"{'='*60}")

    daily_ohlcv = dm.get_ohlcv(symbol, '1d', '2020-01-01', TEST_END)
    signal, data_4h = get_daily_ma_signal(dm, symbol, daily_ohlcv)

    # Compute scaling from ls_ratio
    scaling = compute_ls_scaling(ls_ratio_1h)

    # Filter to test period
    test_start_ts = pd.Timestamp(TEST_START, tz='UTC')
    test_end_ts = pd.Timestamp(TEST_END, tz='UTC')

    # Check overlap
    scaling_in_test = scaling[(scaling.index >= test_start_ts) & (scaling.index <= test_end_ts)]
    print(f"  ls_ratio scaling bars in test period: {len(scaling_in_test)}")
    print(f"  Scaling distribution: "
          f"full={float((scaling_in_test == 1.0).mean()*100):.0f}%, "
          f"half={float((scaling_in_test == 0.5).mean()*100):.0f}%, "
          f"quarter={float((scaling_in_test == 0.25).mean()*100):.0f}%")

    # Baseline: flat sizing (scaling = 1.0 everywhere)
    flat_scaling = pd.Series(1.0, index=scaling.index)
    ret_baseline, pos_baseline = backtest_with_scaling(
        data_4h, signal, flat_scaling, fraction=fraction
    )

    # Overlay: ls_ratio scaling
    ret_overlay, pos_overlay = backtest_with_scaling(
        data_4h, signal, scaling, fraction=fraction
    )

    # Also try inverted scaling (crowd heavily positioned = full size)
    # This tests whether the signal direction matters
    inverted_scaling = pd.Series(0.5, index=scaling.index)
    inverted_scaling[scaling == 1.0] = 0.25
    inverted_scaling[scaling == 0.25] = 1.0
    ret_inverted, _ = backtest_with_scaling(
        data_4h, signal, inverted_scaling, fraction=fraction
    )

    # Compute metrics
    m_baseline = compute_metrics(ret_baseline, "baseline")
    m_overlay = compute_metrics(ret_overlay, "overlay")
    m_inverted = compute_metrics(ret_inverted, "inverted")

    # Average position size for context
    avg_pos_baseline = pos_baseline.abs().mean()
    avg_pos_overlay = pos_overlay.abs().mean()

    print(f"\n  {'Metric':<20s}  {'Baseline':>10s}  {'LS Overlay':>10s}  {'Inverted':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    for key in ['sharpe', 'max_dd_pct', 'mean_monthly_pct', 'worst_month_pct', 'total_return_pct']:
        label = key.replace('_pct', '%').replace('_', ' ')
        b = m_baseline.get(key, 0)
        o = m_overlay.get(key, 0)
        inv = m_inverted.get(key, 0)
        print(f"  {label:<20s}  {b:10.2f}  {o:10.2f}  {inv:10.2f}")
    print(f"  {'avg position':20s}  {avg_pos_baseline:10.3f}  {avg_pos_overlay:10.3f}  {'':>10s}")

    # Delta analysis
    sharpe_delta = m_overlay['sharpe'] - m_baseline['sharpe']
    dd_delta = m_overlay['max_dd_pct'] - m_baseline['max_dd_pct']
    monthly_delta = m_overlay['mean_monthly_pct'] - m_baseline['mean_monthly_pct']

    print(f"\n  Delta (overlay - baseline):")
    print(f"    Sharpe:  {sharpe_delta:+.2f}")
    print(f"    MaxDD:   {dd_delta:+.2f}% (positive = less drawdown)")
    print(f"    Monthly: {monthly_delta:+.2f}%")

    # Is the overlay actually helpful?
    if sharpe_delta > 0.1 and dd_delta >= 0:
        verdict = "IMPROVEMENT — higher Sharpe AND same/less drawdown"
    elif sharpe_delta > 0.1:
        verdict = "MIXED — higher Sharpe but more drawdown"
    elif dd_delta > 1.0 and sharpe_delta > -0.1:
        verdict = "IMPROVEMENT — less drawdown with minimal Sharpe loss"
    elif abs(sharpe_delta) < 0.1 and abs(dd_delta) < 1:
        verdict = "NO EFFECT — scaling does not materially change performance"
    else:
        verdict = "DEGRADATION — overlay hurts performance"

    print(f"  Verdict: {verdict}")

    return {
        'asset': asset_name,
        'baseline': m_baseline,
        'overlay': m_overlay,
        'inverted': m_inverted,
        'sharpe_delta': round(sharpe_delta, 2),
        'dd_delta': round(dd_delta, 2),
        'monthly_delta': round(monthly_delta, 2),
        'avg_pos_baseline': round(float(avg_pos_baseline), 4),
        'avg_pos_overlay': round(float(avg_pos_overlay), 4),
        'verdict': verdict,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Task 3: ls_ratio as Continuous Position Sizing Signal")
    print("=" * 80)
    print("Test period: 2024-01 to 2024-12 (ls_ratio data availability)")
    print("Method: Expanding-window terciles (no look-ahead)")
    print("  Bottom tercile -> 1.0x (full position)")
    print("  Middle tercile -> 0.5x")
    print("  Top tercile    -> 0.25x")

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    # Load ls_ratio data
    btc_ms = pd.read_parquet('/home/ubuntu/projects/crypto-trader/data_cache/market_structure/BTC_USDT_USDT_unified_1h.parquet')
    sol_ms = pd.read_parquet('/home/ubuntu/projects/crypto-trader/data_cache/market_structure/SOL_USDT_USDT_unified_1h.parquet')

    btc_ls = btc_ms['ls_ratio'].dropna()
    sol_ls = sol_ms['ls_ratio'].dropna()

    print(f"\nBTC ls_ratio: {len(btc_ls)} bars, {btc_ls.index.min().date()} to {btc_ls.index.max().date()}")
    print(f"SOL ls_ratio: {len(sol_ls)} bars, {sol_ls.index.min().date()} to {sol_ls.index.max().date()}")

    results = {}

    # Test BTC
    btc_result = test_asset(dm, 'BTC/USDT:USDT', 'BTC', btc_ls, fraction=0.20)
    results['btc'] = btc_result

    # Test SOL
    sol_result = test_asset(dm, 'SOL/USDT:USDT', 'SOL', sol_ls, fraction=0.10)
    results['sol'] = sol_result

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for asset in ['btc', 'sol']:
        r = results[asset]
        print(f"\n  {r['asset']}:")
        print(f"    Baseline Sharpe: {r['baseline']['sharpe']}, Overlay Sharpe: {r['overlay']['sharpe']} (delta: {r['sharpe_delta']:+.2f})")
        print(f"    Baseline MaxDD:  {r['baseline']['max_dd_pct']}%, Overlay MaxDD: {r['overlay']['max_dd_pct']}% (delta: {r['dd_delta']:+.2f}%)")
        print(f"    Baseline Monthly: {r['baseline']['mean_monthly_pct']}%, Overlay Monthly: {r['overlay']['mean_monthly_pct']}%")
        print(f"    Verdict: {r['verdict']}")

    # Overall conclusion — require Sharpe loss < 0.2 AND DD improvement > 2% to call it helpful
    btc_net_positive = btc_result['sharpe_delta'] > -0.2 and btc_result['dd_delta'] > 2.0
    sol_net_positive = sol_result['sharpe_delta'] > -0.2 and sol_result['dd_delta'] > 2.0

    # But also check: does the overlay simply reduce position size (and thus both return AND DD)?
    btc_pos_reduction = btc_result['avg_pos_overlay'] / max(btc_result['avg_pos_baseline'], 0.001)
    sol_pos_reduction = sol_result['avg_pos_overlay'] / max(sol_result['avg_pos_baseline'], 0.001)

    # If the DD improvement is proportional to the position size reduction,
    # then the overlay is just reducing leverage, not adding risk-adjusted value.
    # Check: DD improvement ratio vs position size ratio
    btc_dd_ratio = btc_result['overlay']['max_dd_pct'] / min(btc_result['baseline']['max_dd_pct'], -0.01)
    sol_dd_ratio = sol_result['overlay']['max_dd_pct'] / min(sol_result['baseline']['max_dd_pct'], -0.01)
    btc_genuine_improvement = btc_dd_ratio < btc_pos_reduction  # DD improved MORE than position size decreased
    sol_genuine_improvement = sol_dd_ratio < sol_pos_reduction

    print(f"\n  Position size analysis:")
    print(f"    BTC: pos_reduction={btc_pos_reduction:.2f}x, dd_ratio={btc_dd_ratio:.2f}x, "
          f"genuine_improvement={'YES' if btc_genuine_improvement else 'NO'}")
    print(f"    SOL: pos_reduction={sol_pos_reduction:.2f}x, dd_ratio={sol_dd_ratio:.2f}x, "
          f"genuine_improvement={'YES' if sol_genuine_improvement else 'NO'}")

    if btc_genuine_improvement and sol_genuine_improvement:
        conclusion = ("ls_ratio scaling genuinely improves risk-adjusted returns for both assets. "
                      "The drawdown reduction exceeds what simple position reduction would give. "
                      "The signal correctly identifies high-risk crowd positioning periods.")
    elif btc_genuine_improvement or sol_genuine_improvement:
        asset = 'BTC' if btc_genuine_improvement else 'SOL'
        conclusion = (f"ls_ratio scaling shows genuine improvement for {asset} but not both. "
                      "Mixed evidence — the overlay primarily acts as a leverage reducer. "
                      "For the other asset, DD improvement is proportional to smaller positions.")
    elif btc_net_positive or sol_net_positive:
        conclusion = ("ls_ratio scaling reduces drawdown but ONLY because it reduces average position size. "
                      "The Sharpe ratio is not improved. The signal does not add alpha — it just reduces "
                      "leverage. You could achieve the same effect by simply using lower fraction.")
    else:
        conclusion = ("ls_ratio scaling does NOT materially improve either strategy. "
                      "The signal is too noisy to be useful as a position sizing overlay.")

    print(f"\n  Overall: {conclusion}")
    results['conclusion'] = conclusion

    with open('/home/ubuntu/projects/crypto-trader/task3_ls_overlay_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to task3_ls_overlay_results.json ({time.time()-t0:.0f}s)")
    return results


if __name__ == '__main__':
    results = main()
