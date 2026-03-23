"""
Phase 3: BTC/ETH statistical arbitrage.

Rolling z-score of BTC/ETH price ratio (normalised to 100 at window start).
  z > 2.0: short BTC, long ETH
  z < -2.0: long BTC, short ETH
  Exit when z within 0.5 of zero.

Grid search 36 combos:
  z_window:  48, 96, 168, 336
  entry_z:   1.5, 2.0, 2.5
  exit_z:    0.0, 0.5, 1.0

Report best 5 by holdout Sharpe.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from itertools import product

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.data_module import DataModule

BTC_SYMBOL = 'BTC/USDT:USDT'
ETH_SYMBOL = 'ETH/USDT:USDT'
TF = '1h'
START = '2020-09-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HO_START = '2024-01-01'
HO_END = '2026-03-23'

Z_WINDOWS = [48, 96, 168, 336]
ENTRY_ZS = [1.5, 2.0, 2.5]
EXIT_ZS = [0.0, 0.5, 1.0]
COST_PER_TRADE = 0.001  # 10 bps round trip for pair


def compute_pair_returns(btc_close, eth_close, z_window, entry_z, exit_z):
    """
    Compute strategy returns for BTC/ETH stat arb.

    Returns Series of bar-level returns and trade count.
    """
    # Price ratio
    ratio = btc_close / eth_close

    # Rolling z-score
    ratio_mean = ratio.rolling(z_window, min_periods=z_window//2).mean()
    ratio_std = ratio.rolling(z_window, min_periods=z_window//2).std()
    z = (ratio - ratio_mean) / ratio_std.replace(0, np.nan)

    # Position tracking
    position = pd.Series(0.0, index=z.index)
    n_trades = 0

    current_pos = 0  # +1 = long BTC/short ETH, -1 = short BTC/long ETH
    for i in range(1, len(z)):
        zval = z.iloc[i]
        if np.isnan(zval):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if zval > entry_z:
                current_pos = -1  # short BTC, long ETH (ratio is high)
                n_trades += 1
            elif zval < -entry_z:
                current_pos = 1   # long BTC, short ETH (ratio is low)
                n_trades += 1
        elif current_pos == 1:
            if abs(zval) <= exit_z:
                current_pos = 0
        elif current_pos == -1:
            if abs(zval) <= exit_z:
                current_pos = 0

        position.iloc[i] = current_pos

    # Returns: long BTC + short ETH when position=1, opposite when -1
    btc_ret = btc_close.pct_change()
    eth_ret = eth_close.pct_change()
    # Spread return: long BTC/short ETH = btc_ret - eth_ret
    spread_ret = btc_ret - eth_ret

    # Strategy return with 1-bar execution delay
    strat_ret = position.shift(1).fillna(0) * spread_ret

    # Costs on position changes
    pos_changes = position.diff().abs().fillna(0)
    strat_ret = strat_ret - pos_changes * COST_PER_TRADE / 2

    return strat_ret, n_trades, z, position


def compute_metrics(strat_ret, split_name=""):
    """Compute Sharpe, max DD, worst month from bar-level returns."""
    if strat_ret.std() == 0 or len(strat_ret) < 100:
        return {'sharpe': 0.0, 'max_dd': 0.0, 'worst_month': 0.0,
                'total_return': 0.0, 'mean_monthly': 0.0}

    # Annualised Sharpe (8760 hours/year)
    sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(8760)

    # Equity curve
    equity = (1 + strat_ret).cumprod()
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min()) * 100

    # Monthly
    monthly = strat_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    worst_month = float(monthly.min()) * 100 if len(monthly) > 0 else 0
    mean_monthly = float(monthly.mean()) * 100 if len(monthly) > 0 else 0

    total_return = float((equity.iloc[-1] - 1) * 100) if len(equity) > 0 else 0

    return {
        'sharpe': round(sharpe, 4),
        'max_dd': round(max_dd, 2),
        'worst_month': round(worst_month, 2),
        'total_return': round(total_return, 2),
        'mean_monthly': round(mean_monthly, 4),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 3: BTC/ETH statistical arbitrage")
    print("=" * 70)

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    # Load data
    btc = dm.get_ohlcv(BTC_SYMBOL, TF, START, HO_END)
    eth = dm.get_ohlcv(ETH_SYMBOL, TF, START, HO_END)

    # Align indices
    common_idx = btc.index.intersection(eth.index)
    btc_close = btc.loc[common_idx, 'close']
    eth_close = eth.loc[common_idx, 'close']

    print(f"BTC: {len(btc)} bars, ETH: {len(eth)} bars, Common: {len(common_idx)}")
    print(f"Date range: {common_idx.min()} to {common_idx.max()}")

    # Split masks
    train_mask = common_idx <= pd.Timestamp(TRAIN_END, tz='UTC')
    val_mask = (common_idx >= pd.Timestamp(VAL_START, tz='UTC')) & (common_idx <= pd.Timestamp(VAL_END, tz='UTC'))
    ho_mask = common_idx >= pd.Timestamp(HO_START, tz='UTC')

    # Grid search
    print(f"\nGrid search: {len(Z_WINDOWS)}x{len(ENTRY_ZS)}x{len(EXIT_ZS)} = {len(Z_WINDOWS)*len(ENTRY_ZS)*len(EXIT_ZS)} combos")
    print("=" * 80)

    all_combos = []
    for zw, ez, xz in product(Z_WINDOWS, ENTRY_ZS, EXIT_ZS):
        strat_ret, n_trades, z_score, position = compute_pair_returns(
            btc_close, eth_close, zw, ez, xz
        )

        # Split metrics
        train_metrics = compute_metrics(strat_ret[train_mask])
        val_metrics = compute_metrics(strat_ret[val_mask])
        ho_metrics = compute_metrics(strat_ret[ho_mask])

        # Trade counts per split
        pos_full = position
        train_trades = int(pos_full[train_mask].diff().abs().fillna(0).gt(0).sum())
        val_trades = int(pos_full[val_mask].diff().abs().fillna(0).gt(0).sum())
        ho_trades = int(pos_full[ho_mask].diff().abs().fillna(0).gt(0).sum())

        # Trades per month on holdout
        ho_days = (common_idx[ho_mask][-1] - common_idx[ho_mask][0]).days if ho_mask.sum() > 1 else 1
        ho_tpm = ho_trades / max(ho_days / 30.44, 1)

        combo = {
            'z_window': zw, 'entry_z': ez, 'exit_z': xz,
            'train': train_metrics, 'val': val_metrics, 'holdout': ho_metrics,
            'n_trades_total': n_trades,
            'train_trades': train_trades, 'val_trades': val_trades, 'ho_trades': ho_trades,
            'ho_trades_per_month': round(ho_tpm, 1),
        }
        all_combos.append(combo)

        print(f"  zw={zw:3d} ez={ez:.1f} xz={xz:.1f} | "
              f"Train:{train_metrics['sharpe']:6.2f} Val:{val_metrics['sharpe']:6.2f} "
              f"HO:{ho_metrics['sharpe']:6.2f} | trades={ho_trades}")

    # Sort by holdout Sharpe
    all_combos.sort(key=lambda x: x['holdout']['sharpe'], reverse=True)

    # Best 5
    print(f"\n{'='*70}")
    print("Best 5 by holdout Sharpe:")
    print(f"{'='*70}")
    for i, c in enumerate(all_combos[:5]):
        print(f"  {i+1}. zw={c['z_window']} ez={c['entry_z']} xz={c['exit_z']} | "
              f"HO Sharpe={c['holdout']['sharpe']:.2f}, "
              f"MaxDD={c['holdout']['max_dd']:.1f}%, "
              f"Trades={c['ho_trades']}, TPM={c['ho_trades_per_month']:.1f}")

    # Count positive holdout combos
    positive_ho = sum(1 for c in all_combos if c['holdout']['sharpe'] > 0)
    print(f"\nPositive holdout: {positive_ho}/{len(all_combos)}")

    # Qualification: 3+ of 36 positive on holdout
    qualifies = positive_ho >= 3
    print(f"Qualifies for portfolio (3+ positive): {'YES' if qualifies else 'NO'}")

    # Best combo details
    best = all_combos[0]
    print(f"\nBest combo: zw={best['z_window']}, entry_z={best['entry_z']}, exit_z={best['exit_z']}")
    print(f"  Holdout: Sharpe={best['holdout']['sharpe']:.4f}, MaxDD={best['holdout']['max_dd']:.2f}%, "
          f"Return={best['holdout']['total_return']:.2f}%")
    print(f"  Val:     Sharpe={best['val']['sharpe']:.4f}")
    print(f"  Train:   Sharpe={best['train']['sharpe']:.4f}")

    # Save
    results = {
        'all_combos': all_combos,
        'best_5': all_combos[:5],
        'positive_holdout_count': positive_ho,
        'qualifies': qualifies,
        'best_params': {
            'z_window': best['z_window'],
            'entry_z': best['entry_z'],
            'exit_z': best['exit_z'],
        },
        'best_holdout_sharpe': best['holdout']['sharpe'],
        'best_holdout_max_dd': best['holdout']['max_dd'],
        'elapsed_seconds': time.time() - t0,
    }

    with open('/home/ubuntu/projects/crypto-trader/phase3_btc_eth_arb_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to phase3_btc_eth_arb_results.json ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
