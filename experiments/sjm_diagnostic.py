"""
SJM Diagnostic: Verify regime classifications match known market history.
Run this BEFORE any backtest to confirm the SJM is working correctly.
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
from crypto_infra.data_module import DataModule
from regime.sjm import StatisticalJumpModel
from regime.features import compute_feature_set_A, standardise


def run_diagnostic():
    print("=" * 70)
    print("SJM REGIME DIAGNOSTIC")
    print("=" * 70)

    # Load BTC 4H data
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    btc = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2021-01-01', '2026-03-21')
    print(f"BTC 4H data: {btc.index[0]} to {btc.index[-1]}, {len(btc)} bars")

    # Compute features
    features = compute_feature_set_A(btc)
    print(f"Features computed: {len(features)} bars, {features.shape[1]} features")
    print(f"Feature columns: {list(features.columns)}")
    print()

    # Test multiple lambda values
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
        print(f"\n{'=' * 70}")
        print(f"Lambda = {lam}")
        print(f"{'=' * 70}")

        features_std = standardise(features)
        sjm = StatisticalJumpModel(n_regimes=3, jump_penalty=lam)
        sjm.fit(features_std.values)

        regimes = sjm.result_.regimes
        returns = features['mom_1d'].values

        # Label regimes
        label_map = sjm.label_regimes(regimes, returns)
        named_regimes = pd.Series(
            [label_map[r] for r in regimes],
            index=features.index,
        )

        # Overall stats
        print(f"\nRegime counts: {sjm.result_.regime_counts}")
        print(f"Number of transitions: {sjm.result_.n_jumps}")
        print(f"Label mapping: {label_map}")

        # Mean return per regime
        for regime_name in ['bull', 'neutral', 'bear']:
            mask = named_regimes == regime_name
            if mask.sum() > 0:
                mean_ret = returns[mask.values].mean()
                print(f"  {regime_name}: mean 1d return = {mean_ret:.4f}, count = {mask.sum()}")

        # Year-by-year breakdown
        print(f"\nYear-by-year regime fractions:")
        print(f"{'Year':<8} {'Bull':>8} {'Neutral':>8} {'Bear':>8} {'Bars':>6}")
        print("-" * 40)

        for year in range(2021, 2027):
            mask = named_regimes.index.year == year
            year_regimes = named_regimes[mask]
            if len(year_regimes) == 0:
                continue
            n = len(year_regimes)
            bull_frac = (year_regimes == 'bull').sum() / n
            neutral_frac = (year_regimes == 'neutral').sum() / n
            bear_frac = (year_regimes == 'bear').sum() / n
            print(f"{year:<8} {bull_frac:>8.1%} {neutral_frac:>8.1%} {bear_frac:>8.1%} {n:>6}")

        # Half-year breakdown for 2024
        h1_mask = (named_regimes.index.year == 2024) & (named_regimes.index.month <= 6)
        h2_mask = (named_regimes.index.year == 2024) & (named_regimes.index.month > 6)
        for label, mask in [("2024 H1", h1_mask), ("2024 H2", h2_mask)]:
            yr = named_regimes[mask]
            if len(yr) == 0:
                continue
            n = len(yr)
            print(f"{label:<8} {(yr == 'bull').sum()/n:>8.1%} {(yr == 'neutral').sum()/n:>8.1%} {(yr == 'bear').sum()/n:>8.1%} {n:>6}")

        # Longest regime stretches
        print(f"\n10 longest BULL regimes:")
        _print_longest_stretches(named_regimes, 'bull', 10)
        print(f"\n10 longest BEAR regimes:")
        _print_longest_stretches(named_regimes, 'bear', 10)


def _print_longest_stretches(regimes: pd.Series, target: str, top_n: int):
    """Find and print the longest consecutive stretches of a given regime."""
    stretches = []
    start = None
    count = 0

    for i, (ts, regime) in enumerate(regimes.items()):
        if regime == target:
            if start is None:
                start = ts
            count += 1
        else:
            if start is not None:
                stretches.append((start, regimes.index[i - 1], count))
                start = None
                count = 0
    if start is not None:
        stretches.append((start, regimes.index[-1], count))

    stretches.sort(key=lambda x: x[2], reverse=True)
    for s, e, c in stretches[:top_n]:
        days = (e - s).days
        print(f"  {s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')} — {c} bars ({days} days)")


if __name__ == '__main__':
    run_diagnostic()
