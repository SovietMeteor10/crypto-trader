"""
Order flow statistical characterisation — Steps 3A through 3E.
Produces all analyses and saves plots + results for the final report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from data.order_flow import compute_bar_features_chunked, compute_vpin_chunked

CACHE_DIR = Path("/home/ubuntu/projects/crypto-trader/data_cache/aggtrades")
OUTPUT_DIR = Path("/home/ubuntu/projects/crypto-trader/outputs/research")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "basic_stats").mkdir(exist_ok=True)

SYMBOLS = ['BTCUSDT', 'SOLUSDT']
START = '2023-01-01'
END = '2024-12-31'

# Regime periods for 3C
REGIMES = {
    'P1_recovery': ('2023-01-01', '2023-06-30'),
    'P2_bull_begin': ('2023-07-01', '2023-12-31'),
    'P3_etf_bull': ('2024-01-01', '2024-06-30'),
    'P4_correction': ('2024-07-01', '2024-12-31'),
}


def predictive_regression(feature, returns, horizon, name):
    """Simple OLS predictive regression."""
    fwd_ret = returns.shift(-horizon)
    df = pd.DataFrame({'feature': feature, 'fwd_ret': fwd_ret}).dropna()
    if len(df) < 100:
        return {'name': name, 'horizon': horizon, 'beta': np.nan,
                'tstat': np.nan, 'pvalue': np.nan, 'r2': np.nan, 'n': len(df)}

    df['feature_std'] = (df['feature'] - df['feature'].mean()) / df['feature'].std()
    slope, intercept, r, p, se = stats.linregress(df['feature_std'], df['fwd_ret'])
    tstat = slope / se if se > 0 else 0

    return {'name': name, 'horizon': horizon, 'beta': slope,
            'tstat': tstat, 'pvalue': p, 'r2': r**2, 'n': len(df)}


def run_3a(bars, symbol, vpin_series=None):
    """3A: Basic order flow statistics."""
    print(f"\n{'='*60}")
    print(f"3A: Basic OFI Statistics — {symbol}")
    print('='*60)

    results = {}
    ofi = bars['ofi'].dropna()

    # 1. OFI distribution
    results['ofi_mean'] = float(ofi.mean())
    results['ofi_std'] = float(ofi.std())
    results['ofi_skew'] = float(ofi.skew())
    results['ofi_kurtosis'] = float(ofi.kurtosis())
    results['ofi_strongly_directional_pct'] = float((ofi.abs() > 0.7).mean() * 100)

    print(f"OFI: mean={results['ofi_mean']:.4f}, std={results['ofi_std']:.4f}")
    print(f"     skew={results['ofi_skew']:.4f}, kurtosis={results['ofi_kurtosis']:.4f}")
    print(f"     |OFI| > 0.7: {results['ofi_strongly_directional_pct']:.2f}%")

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(ofi, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.3)
    x = np.linspace(-1, 1, 200)
    ax.plot(x, stats.norm.pdf(x, ofi.mean(), ofi.std()), 'r--', label='Normal fit')
    ax.set_xlabel('OFI')
    ax.set_ylabel('Density')
    ax.set_title(f'{symbol} Order Flow Imbalance Distribution (15min bars)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "basic_stats" / f"ofi_distribution_{symbol}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved OFI histogram")

    # 2. Serial correlation of OFI
    acf_lags = 48
    acf_values = [ofi.autocorr(lag=i) for i in range(1, acf_lags + 1)]
    results['ofi_acf'] = {i+1: float(v) for i, v in enumerate(acf_values)}

    # Find lag where ACF drops below significance threshold
    sig_threshold = 1.96 / np.sqrt(len(ofi))
    first_insignificant = next((i+1 for i, v in enumerate(acf_values) if abs(v) < sig_threshold), acf_lags)
    results['ofi_acf_decay_lag'] = first_insignificant
    print(f"  OFI ACF: lag-1={acf_values[0]:.4f}, decay to insignificance at lag {first_insignificant}")

    # ACF plot
    fig, ax = plt.subplots(figsize=(10, 5))
    lags = range(1, acf_lags + 1)
    ax.bar(lags, acf_values, color='steelblue', alpha=0.7)
    ax.axhline(sig_threshold, color='red', linestyle='--', alpha=0.5, label=f'95% CI (±{sig_threshold:.4f})')
    ax.axhline(-sig_threshold, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Lag (15-min bars)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'{symbol} OFI Autocorrelation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "basic_stats" / f"ofi_acf_{symbol}.png", dpi=150)
    plt.close(fig)

    # 3. OFI vs next-bar return
    ret = bars['log_ret'].dropna()
    ofi_aligned = bars['ofi'].dropna()
    common_idx = ofi_aligned.index.intersection(ret.index)
    ofi_t = ofi_aligned.loc[common_idx]
    ret_t1 = ret.shift(-1).loc[common_idx].dropna()
    common = ofi_t.index.intersection(ret_t1.index)
    ofi_t = ofi_t.loc[common]
    ret_t1 = ret_t1.loc[common]

    corr_all = ofi_t.corr(ret_t1)
    results['ofi_ret_corr_all'] = float(corr_all)

    # High vs low volume split
    vol_median = bars['volume'].median()
    high_vol_mask = bars.loc[common, 'volume'] > vol_median
    low_vol_mask = ~high_vol_mask

    if high_vol_mask.sum() > 50:
        corr_high = ofi_t[high_vol_mask].corr(ret_t1[high_vol_mask])
        results['ofi_ret_corr_high_vol'] = float(corr_high)
    else:
        results['ofi_ret_corr_high_vol'] = np.nan

    if low_vol_mask.sum() > 50:
        corr_low = ofi_t[low_vol_mask].corr(ret_t1[low_vol_mask])
        results['ofi_ret_corr_low_vol'] = float(corr_low)
    else:
        results['ofi_ret_corr_low_vol'] = np.nan

    print(f"  OFI→ret(t+1) corr: all={results['ofi_ret_corr_all']:.4f}, "
          f"high_vol={results.get('ofi_ret_corr_high_vol', 'N/A'):.4f}, "
          f"low_vol={results.get('ofi_ret_corr_low_vol', 'N/A'):.4f}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sample_idx = np.random.choice(len(ofi_t), min(5000, len(ofi_t)), replace=False)
    ax.scatter(ofi_t.iloc[sample_idx], ret_t1.iloc[sample_idx], alpha=0.1, s=1, c='steelblue')
    ax.set_xlabel('OFI(t)')
    ax.set_ylabel('Return(t+1)')
    ax.set_title(f'{symbol} OFI vs Next-Bar Return (r={corr_all:.4f})')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "basic_stats" / f"ofi_vs_return_{symbol}.png", dpi=150)
    plt.close(fig)

    # 4. VPIN statistics
    if vpin_series is not None and len(vpin_series) > 0:
        vpin_clean = vpin_series.dropna()
        results['vpin_mean'] = float(vpin_clean.mean())
        results['vpin_std'] = float(vpin_clean.std())
        results['vpin_p90'] = float(vpin_clean.quantile(0.90))
        results['vpin_p95'] = float(vpin_clean.quantile(0.95))
        print(f"  VPIN: mean={results['vpin_mean']:.4f}, p90={results['vpin_p90']:.4f}")

        fig, ax = plt.subplots(figsize=(12, 5))
        vpin_clean.plot(ax=ax, alpha=0.5, linewidth=0.5, color='steelblue')
        ax.axhline(results['vpin_mean'], color='red', linestyle='--', label=f"Mean={results['vpin_mean']:.3f}")
        ax.axhline(results['vpin_p90'], color='orange', linestyle='--', label=f"P90={results['vpin_p90']:.3f}")
        ax.set_ylabel('VPIN')
        ax.set_title(f'{symbol} VPIN Time Series')
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "basic_stats" / f"vpin_timeseries_{symbol}.png", dpi=150)
        plt.close(fig)
    else:
        results['vpin_mean'] = np.nan
        print("  VPIN: not computed")

    # 5. Roll measure statistics
    roll = bars['roll_spread'].dropna()
    # Convert to basis points (roll is in log-return units)
    roll_bps = roll * 10000
    results['roll_mean_bps'] = float(roll_bps.mean())
    results['roll_std_bps'] = float(roll_bps.std())
    results['roll_median_bps'] = float(roll_bps.median())
    print(f"  Roll spread: mean={results['roll_mean_bps']:.2f} bps, "
          f"median={results['roll_median_bps']:.2f} bps")

    # Time-of-day variation
    bars_with_hour = bars.copy()
    bars_with_hour['hour'] = bars_with_hour.index.hour
    hourly_roll = bars_with_hour.groupby('hour')['roll_spread'].mean() * 10000
    results['roll_by_hour'] = {int(h): float(v) for h, v in hourly_roll.items()}

    fig, ax = plt.subplots(figsize=(10, 5))
    hourly_roll.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Roll Spread (bps)')
    ax.set_title(f'{symbol} Roll Spread by Hour of Day')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "basic_stats" / f"roll_by_hour_{symbol}.png", dpi=150)
    plt.close(fig)

    return results


def run_3b(bars, symbol):
    """3B: Predictive power analysis."""
    print(f"\n{'='*60}")
    print(f"3B: Predictive Power — {symbol}")
    print('='*60)

    features = {
        'OFI': bars['ofi'],
        'OFI_MA_1h': bars['ofi_ma_1h'],
        'OFI_MA_4h': bars['ofi_ma_4h'],
        'Kyle_Lambda': bars['kyle_lambda'],
        'Roll_Spread': bars['roll_spread'],
        'Arrival_Rate': bars['arrival_rate'],
        'Amihud': bars['amihud'],
    }

    horizons = [1, 4, 16, 64]  # 15min, 1h, 4h, 16h
    horizon_labels = {1: '15min', 4: '1h', 16: '4h', 64: '16h'}
    returns = bars['log_ret']

    all_results = []
    for feat_name, feat_series in features.items():
        for h in horizons:
            r = predictive_regression(feat_series, returns, h, feat_name)
            r['horizon_label'] = horizon_labels[h]
            r['symbol'] = symbol
            all_results.append(r)
            sig = " ***" if abs(r['tstat']) > 2.0 else ""
            print(f"  {feat_name:15s} → {horizon_labels[h]:5s}: "
                  f"t={r['tstat']:7.2f}, R²={r['r2']:.6f}, "
                  f"β={r['beta']:.8f}{sig}")

    return all_results


def run_3c(bars, symbol, best_feature_name, best_horizon):
    """3C: Regime-conditional analysis."""
    print(f"\n{'='*60}")
    print(f"3C: Regime-Conditional Analysis — {symbol}")
    print('='*60)

    feature = bars[best_feature_name]
    returns = bars['log_ret']

    results = []
    for regime_name, (start, end) in REGIMES.items():
        mask = (bars.index >= start) & (bars.index <= end)
        feat_slice = feature[mask]
        ret_slice = returns[mask]
        r = predictive_regression(feat_slice, ret_slice, best_horizon,
                                  f"{best_feature_name}_{regime_name}")
        r['regime'] = regime_name
        r['symbol'] = symbol
        results.append(r)
        sig = " ***" if abs(r['tstat']) > 2.0 else ""
        print(f"  {regime_name:20s}: t={r['tstat']:7.2f}, R²={r['r2']:.6f}, "
              f"β={r['beta']:.8f}, n={r['n']:,}{sig}")

    return results


def run_3d(bars, vpin_series, symbol):
    """3D: Informed trading events."""
    print(f"\n{'='*60}")
    print(f"3D: Informed Trading Events — {symbol}")
    print('='*60)

    if vpin_series is None or len(vpin_series) == 0:
        print("  VPIN not available, skipping")
        return {}

    # Align VPIN to bars (VPIN has irregular index, resample to bar freq)
    vpin_resampled = vpin_series.resample('15min').last().ffill()
    common_idx = bars.index.intersection(vpin_resampled.index)

    if len(common_idx) < 100:
        print("  Insufficient overlap between VPIN and bars")
        return {}

    vpin_aligned = vpin_resampled.loc[common_idx]
    bars_aligned = bars.loc[common_idx]

    # Find high-VPIN periods
    threshold = vpin_aligned.quantile(0.95)  # Top 5% as "high VPIN"
    high_vpin_mask = vpin_aligned > threshold

    # Forward 4h return (16 bars at 15min)
    fwd_4h = bars_aligned['log_ret'].rolling(16).sum().shift(-16)

    high_vpin_returns = fwd_4h[high_vpin_mask].dropna()
    baseline_returns = fwd_4h.dropna()

    if len(high_vpin_returns) < 10:
        print("  Too few high-VPIN events")
        return {}

    results = {
        'vpin_threshold': float(threshold),
        'n_high_vpin_bars': int(high_vpin_mask.sum()),
        'high_vpin_4h_mean_ret': float(high_vpin_returns.mean()),
        'high_vpin_4h_abs_mean_ret': float(high_vpin_returns.abs().mean()),
        'baseline_4h_mean_ret': float(baseline_returns.mean()),
        'baseline_4h_abs_mean_ret': float(baseline_returns.abs().mean()),
    }

    # T-test: do high-VPIN periods have larger absolute moves?
    t_stat, p_val = stats.ttest_ind(
        high_vpin_returns.abs().values,
        baseline_returns.abs().values,
        equal_var=False
    )
    results['abs_move_ttest_t'] = float(t_stat)
    results['abs_move_ttest_p'] = float(p_val)

    print(f"  VPIN threshold (P95): {results['vpin_threshold']:.4f}")
    print(f"  High-VPIN bars: {results['n_high_vpin_bars']:,}")
    print(f"  Avg |4H move| after high VPIN: {results['high_vpin_4h_abs_mean_ret']*100:.4f}%")
    print(f"  Avg |4H move| baseline:        {results['baseline_4h_abs_mean_ret']*100:.4f}%")
    print(f"  T-test: t={t_stat:.2f}, p={p_val:.4f}")

    return results


def run_3e(btc_bars, sol_bars):
    """3E: Cross-asset order flow analysis."""
    print(f"\n{'='*60}")
    print("3E: Cross-Asset Order Flow")
    print('='*60)

    # Align on common index
    common = btc_bars.index.intersection(sol_bars.index)
    if len(common) < 100:
        print("  Insufficient common bars for cross-asset analysis")
        return []

    btc = btc_bars.loc[common]
    sol = sol_bars.loc[common]

    pairs = [
        ('BTC_OFI', btc['ofi'], 'BTC_ret', btc['log_ret'], 'own-asset BTC'),
        ('SOL_OFI', sol['ofi'], 'SOL_ret', sol['log_ret'], 'own-asset SOL'),
        ('BTC_OFI', btc['ofi'], 'SOL_ret', sol['log_ret'], 'BTC→SOL cross'),
        ('SOL_OFI', sol['ofi'], 'BTC_ret', btc['log_ret'], 'SOL→BTC cross'),
    ]

    results = []
    for feat_name, feat, ret_name, ret, desc in pairs:
        for h in [1, 4, 16]:
            r = predictive_regression(feat, ret, h, desc)
            r['feature'] = feat_name
            r['target'] = ret_name
            r['horizon_bars'] = h
            results.append(r)
            sig = " ***" if abs(r['tstat']) > 2.0 else ""
            print(f"  {desc:20s} h={h:2d}: t={r['tstat']:7.2f}, R²={r['r2']:.6f}{sig}")

    return results


def main():
    all_results = {}

    # Load or compute bars for each symbol
    bars_dict = {}
    vpin_dict = {}

    for symbol in SYMBOLS:
        parquet_path = str(CACHE_DIR / f"{symbol}_{START}_{END}.parquet")
        bars_cache = CACHE_DIR / f"{symbol}_bars_15min.parquet"

        if bars_cache.exists():
            print(f"\nLoading cached bars for {symbol}")
            bars = pd.read_parquet(bars_cache)
            bars.index = pd.DatetimeIndex(bars.index)
        else:
            print(f"\nComputing 15min bars for {symbol}...")
            bars = compute_bar_features_chunked(parquet_path, freq='15min', chunk_days=14)
            bars.to_parquet(bars_cache)
            print(f"Saved bars to {bars_cache}")

        bars_dict[symbol] = bars
        print(f"{symbol}: {len(bars):,} bars, {bars.index.min()} to {bars.index.max()}")

        # VPIN — compute on 3-month sample for speed
        vpin_cache = CACHE_DIR / f"{symbol}_vpin.parquet"
        if vpin_cache.exists():
            print(f"Loading cached VPIN for {symbol}")
            vpin_df = pd.read_parquet(vpin_cache)
            vpin_series = vpin_df.iloc[:, 0]
            vpin_series.index = pd.DatetimeIndex(vpin_series.index)
        else:
            print(f"Computing VPIN for {symbol} (sampled months)...")
            try:
                vpin_series = compute_vpin_chunked(
                    parquet_path,
                    sample_months=['2023-03', '2023-07', '2023-11', '2024-01', '2024-05', '2024-09'],
                )
                vpin_series.to_frame('vpin').to_parquet(vpin_cache)
                print(f"Saved VPIN to {vpin_cache}")
            except Exception as e:
                print(f"VPIN computation failed: {e}")
                vpin_series = pd.Series(dtype=float)

        vpin_dict[symbol] = vpin_series

    # Run analyses
    for symbol in SYMBOLS:
        bars = bars_dict[symbol]
        vpin = vpin_dict.get(symbol)

        # 3A
        stats_3a = run_3a(bars, symbol, vpin)
        all_results[f'{symbol}_3a'] = stats_3a

        # 3B
        results_3b = run_3b(bars, symbol)
        all_results[f'{symbol}_3b'] = results_3b

        # Find best feature/horizon for 3C
        sig_results = [r for r in results_3b if abs(r['tstat']) > 1.5]
        if sig_results:
            best = max(sig_results, key=lambda r: abs(r['tstat']))
            best_feat_col = {
                'OFI': 'ofi', 'OFI_MA_1h': 'ofi_ma_1h', 'OFI_MA_4h': 'ofi_ma_4h',
                'Kyle_Lambda': 'kyle_lambda', 'Roll_Spread': 'roll_spread',
                'Arrival_Rate': 'arrival_rate', 'Amihud': 'amihud',
            }.get(best['name'], 'ofi')
            best_horizon = best['horizon']
            print(f"\n  Best feature for {symbol}: {best['name']} at {best['horizon']}-bar horizon "
                  f"(t={best['tstat']:.2f})")
        else:
            best_feat_col = 'ofi'
            best_horizon = 1
            print(f"\n  No significant features for {symbol}, using OFI h=1 for 3C")

        # 3C
        results_3c = run_3c(bars, symbol, best_feat_col, best_horizon)
        all_results[f'{symbol}_3c'] = results_3c

        # 3D
        results_3d = run_3d(bars, vpin, symbol)
        all_results[f'{symbol}_3d'] = results_3d

    # 3E: Cross-asset
    if 'BTCUSDT' in bars_dict and 'SOLUSDT' in bars_dict:
        results_3e = run_3e(bars_dict['BTCUSDT'], bars_dict['SOLUSDT'])
        all_results['cross_asset_3e'] = results_3e

    # Save all results
    results_path = OUTPUT_DIR / "characterisation_results.json"

    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\nAll results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    main()
