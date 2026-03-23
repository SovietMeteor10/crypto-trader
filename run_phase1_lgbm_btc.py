"""
Phase 1: LightGBM on BTC 1H with correct design.

Fixes from previous attempt:
1. Uses BTC market structure data (back to Sept 2020, 4+ years) instead of SOL
2. Only 15 features (market structure 8, OHLCV 5, regime 2)
3. Binary target (up/down), remove bars with |return| < 0.1%, confidence threshold 0.53

Design: BTC/USDT:USDT 1H
  Train: 2020-09-01 to 2022-12-31
  Val:   2023-01-01 to 2023-12-31
  Holdout: 2024-01-01 to present
  Walk-forward: 12 windows, 6-month train, 2-month test, 5-bar gap
  LightGBM: max_depth=4, max 200 trees, early stopping, 5 configs
"""

import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from ml.lgbm_model import LGBMTrader
from ml.walk_forward_ml import run_walk_forward_ml, compute_sharpe_from_signals
from pathlib import Path

SYMBOL = 'BTC/USDT:USDT'
TF = '1h'
START = '2020-09-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-23'
TARGET_THRESHOLD = 0.001   # 0.1% filter for flat bars
CONF_THRESHOLD = 0.53
WF_PASS = 0.60

# The 15 selected features
SELECTED_FEATURES = [
    # Market structure (8) — computed from 1H MS data
    'ls_ratio_last', 'ls_ratio_mean',
    'smart_dumb_div_last', 'smart_dumb_div_mean',
    'taker_ratio_last', 'basis_vs_ma_last',
    'crowd_long_last', 'oi_chg_1h_mean',
    # OHLCV (5)
    'ret_1d', 'ret_1w', 'rvol_1w', 'adx_14', 'ma_cross_12_26',
    # Regime (2)
    'btc_macro_trend', 'vol_regime',
]

LGBM_CONFIGS = [
    # (confidence, max_depth, num_leaves, learning_rate)
    (0.53, 4, 15, 0.05),
    (0.55, 4, 15, 0.05),
    (0.53, 3, 8, 0.05),
    (0.53, 4, 15, 0.02),
    (0.55, 4, 31, 0.05),
]


def load_btc_market_structure():
    """Load BTC market structure 1H data."""
    path = Path("/home/ubuntu/projects/crypto-trader/data_cache/market_structure/BTC_USDT_USDT_unified_1h.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index)
    return df


def compute_ms_features_1h(ms_1h: pd.DataFrame, ohlcv_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1H market structure to rolling features aligned to 1H OHLCV."""
    result = pd.DataFrame(index=ohlcv_1h.index)

    cols_map = {
        'ls_ratio': ['last', 'mean'],
        'smart_dumb_div': ['last', 'mean'],
        'taker_ratio': ['last'],
        'oi_chg_1h': ['mean'],
        'crowd_long': ['last'],
    }

    for col, aggs in cols_map.items():
        if col not in ms_1h.columns:
            continue
        series = ms_1h[col].reindex(ohlcv_1h.index, method='ffill')
        if 'last' in aggs:
            result[f'{col}_last'] = series
        if 'mean' in aggs:
            result[f'{col}_mean'] = series.rolling(4).mean()

    # basis_vs_ma: not available in this dataset, skip gracefully
    # It will be flagged as missing

    return result


def compute_ohlcv_features_1h(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV features on 1H bars."""
    close = ohlcv['close']
    high = ohlcv['high']
    low = ohlcv['low']
    ret = close.pct_change()

    f = pd.DataFrame(index=ohlcv.index)

    # Returns at various horizons (in 1H bars)
    f['ret_1d'] = close.pct_change(24)
    f['ret_1w'] = close.pct_change(168)

    # Realised vol
    f['rvol_1w'] = ret.rolling(168).std() * np.sqrt(8760)

    # ADX
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                     (low - close.shift(1)).abs()], axis=1).max(axis=1)
    dm_plus = (high - high.shift(1)).clip(lower=0)
    dm_minus = (low.shift(1) - low).clip(lower=0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    dip = dm_plus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
    dim = dm_minus.ewm(span=14, adjust=False).mean() / atr14.replace(0, np.nan) * 100
    dx = ((dip - dim).abs() / (dip + dim).replace(0, np.nan)) * 100
    f['adx_14'] = dx.ewm(span=14, adjust=False).mean()

    # MA cross
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    f['ma_cross_12_26'] = (ema_12 - ema_26) / ema_26 * 100

    return f


def compute_regime_features_1h(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute regime features on 1H bars."""
    close = ohlcv['close']
    f = pd.DataFrame(index=ohlcv.index)

    # BTC macro trend: 50-period vs 200-period (in 1H bars ~2 days vs ~8 days)
    # Use longer periods for meaningful regime: 50*24=1200H for weekly, 200*24=4800H monthly
    # Actually stick to period counts matching 4H logic: 50 bars, 200 bars
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()
    f['btc_macro_trend'] = (ma50 > ma200).astype(int)

    # Vol regime
    rvol_short = close.pct_change().rolling(24).std()    # ~1 day
    rvol_long = close.pct_change().rolling(168*4).std()  # ~4 weeks
    f['vol_regime'] = (rvol_short > rvol_long).astype(int)

    return f


def build_feature_matrix():
    """Build feature matrix for BTC 1H with binary target."""
    from crypto_infra import DataModule
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')

    print("Loading data...")
    btc_1h = dm.get_ohlcv(SYMBOL, TF, START, HOLDOUT_END)
    btc_ms = load_btc_market_structure()

    print(f"BTC 1H OHLCV: {len(btc_1h)} bars, {btc_1h.index.min()} to {btc_1h.index.max()}")
    print(f"BTC MS 1H: {len(btc_ms)} bars, {btc_ms.index.min()} to {btc_ms.index.max()}")

    # Compute features
    ms_feats = compute_ms_features_1h(btc_ms, btc_1h)
    ohlcv_feats = compute_ohlcv_features_1h(btc_1h)
    regime_feats = compute_regime_features_1h(btc_1h)

    X = pd.concat([ms_feats, ohlcv_feats, regime_feats], axis=1)

    # Select only the specified features (skip missing ones)
    available = [f for f in SELECTED_FEATURES if f in X.columns]
    missing = [f for f in SELECTED_FEATURES if f not in X.columns]
    if missing:
        print(f"Missing features (skipped): {missing}")

    X = X[available]
    feature_cols = list(X.columns)

    # Binary target: up (1) vs down (0)
    fwd_ret = btc_1h['close'].pct_change().shift(-1)

    df = X.copy()
    df['fwd_ret'] = fwd_ret
    df['target'] = (fwd_ret > 0).astype(int)  # simple binary: up=1, down=0

    # Lag features by 1 bar to prevent lookahead
    df[feature_cols] = df[feature_cols].shift(1)

    # Drop NaN
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], 0)

    # Remove flat bars (|fwd_ret| < threshold)
    n_before = len(df)
    mask = df['fwd_ret'].abs() > TARGET_THRESHOLD
    df = df[mask]

    print(f"\n{'='*60}")
    print(f"Feature matrix shape: {df.shape}")
    print(f"  Total bars: {n_before}")
    print(f"  After removing flat (|ret|<{TARGET_THRESHOLD*100:.1f}%): {len(df)} ({len(df)/n_before*100:.0f}%)")
    print(f"  Features: {len(feature_cols)}")
    ratio = len(df) / len(feature_cols)
    print(f"  Bars-to-features ratio: {ratio:.0f}")
    print(f"  Target distribution: {df['target'].value_counts().to_dict()}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Expected WF windows
    total_bars = len(df)
    train_bars_est = 6 * 30 * 24   # 6 months of 1H bars
    test_bars_est = 2 * 30 * 24    # 2 months
    expected_windows = (total_bars - train_bars_est) // test_bars_est
    print(f"  Expected WF windows: ~{expected_windows}")

    # Check ratio
    if ratio < 100:
        print(f"\n  WARNING: bars/features ratio {ratio:.0f} < 100")
        # Drop least significant features until ratio > 100
        while ratio < 100 and len(feature_cols) > 5:
            dropped = feature_cols.pop()
            X = X.drop(columns=[dropped], errors='ignore')
            df = df.drop(columns=[dropped], errors='ignore')
            ratio = len(df) / len(feature_cols)
            print(f"  Dropped {dropped}, new ratio: {ratio:.0f}")

    return df, feature_cols


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 1: LightGBM on BTC 1H — correct design")
    print("=" * 70)

    feature_df, feature_cols = build_feature_matrix()

    # Split data
    train_val_df = feature_df.loc[:TRAIN_END]
    val_df = feature_df.loc[VAL_START:VAL_END]
    holdout_df = feature_df.loc[HOLDOUT_START:]

    print(f"\n  Train+val: {len(train_val_df)} bars")
    print(f"  Validation: {len(val_df)} bars")
    print(f"  Holdout:    {len(holdout_df)} bars")

    # ── Sanity check: feature importance ──────────────────────────
    print("\n" + "=" * 50)
    print("Sanity check: feature importance")
    print("=" * 50)

    X_all = feature_df[feature_cols]
    y_all = feature_df['target']
    n_split = int(len(X_all) * 0.7)

    sanity = LGBMTrader(confidence_threshold=CONF_THRESHOLD,
                         n_estimators=200, max_depth=4)
    sanity.fit(X_all.iloc[:n_split], y_all.iloc[:n_split],
               X_all.iloc[n_split:], y_all.iloc[n_split:])
    fi = sanity.get_feature_importance(feature_cols)
    disguised = sanity.check_disguised_momentum(fi)

    print("\nTop 10 features by importance:")
    for i, (fname, fval) in enumerate(fi.head(10).items()):
        flag = " <-- PRICE" if any(kw in fname for kw in ['ret_', 'mom_', 'ma_cross_', 'body_direction']) else ""
        print(f"  {i+1:2d}. {fname:30s} {fval:10.1f}{flag}")

    top5_price = sum(1 for f in fi.head(5).index
                     if any(kw in f for kw in ['ret_', 'mom_', 'ma_cross_']))
    if top5_price >= 4:
        print("\n  FLAG: Top 5 are mostly price return features!")
    print(f"  Disguised momentum: {'YES' if disguised else 'NO'}")

    # ── Walk-forward ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Walk-forward: {len(LGBM_CONFIGS)} configs, 12 windows")
    print("=" * 50)

    best_config = None
    best_wf_pct = 0
    best_wf_mean = -999
    all_results = {}

    for i, (ct, md, nl, lr) in enumerate(LGBM_CONFIGS):
        print(f"\n{'='*40}")
        print(f"Config {i+1}/{len(LGBM_CONFIGS)}: conf={ct}, depth={md}, leaves={nl}, lr={lr}")

        kwargs = dict(confidence_threshold=ct, max_depth=md, num_leaves=nl,
                      learning_rate=lr, n_estimators=200, min_child_samples=50)

        wf = run_walk_forward_ml(
            feature_df=train_val_df, feature_cols=feature_cols,
            model_class=LGBMTrader, model_kwargs=kwargs,
            n_windows=12, train_months=6, test_months=2, gap_bars=5,
        )

        if not wf:
            print("  No windows generated")
            continue

        sharpes = [r.test_sharpe for r in wf]
        wf_pct = sum(1 for s in sharpes if s > 0) / len(sharpes)
        wf_mean = np.mean(sharpes)
        mean_cov = np.mean([r.coverage for r in wf])
        mean_acc = np.mean([r.directional_accuracy for r in wf])

        print(f"\n  WF%={wf_pct:.1%}, Mean={wf_mean:.2f}, "
              f"Cov={mean_cov:.1%}, Acc={mean_acc:.1%}")

        all_results[f'config_{i}'] = {
            'params': {k: v for k, v in kwargs.items()},
            'wf_pct': wf_pct, 'wf_mean': wf_mean,
            'coverage': mean_cov, 'accuracy': mean_acc,
            'windows': [{
                'test_sharpe': r.test_sharpe, 'train_sharpe': r.train_sharpe,
                'dir_acc': r.directional_accuracy, 'coverage': r.coverage,
                'n_trades': r.n_trades,
            } for r in wf],
        }

        if wf_pct > best_wf_pct or (wf_pct == best_wf_pct and wf_mean > best_wf_mean):
            best_wf_pct = wf_pct
            best_wf_mean = wf_mean
            best_config = kwargs

    print(f"\nBest config: {best_config}")
    print(f"Best WF: {best_wf_pct:.1%} positive, mean={best_wf_mean:.2f}")

    # ── Comparison to baseline ────────────────────────────────────
    print(f"\nBaseline: Daily MA — Sharpe 3.14, 3.2 trades/month")
    print(f"LightGBM WF: {best_wf_pct:.1%}")

    # ── Holdout ──────────────────────────────────────────────────
    holdout_sharpe = None
    holdout_trades = 0
    holdout_coverage = 0
    holdout_dir_acc = 0
    holdout_trades_per_month = 0

    if best_wf_pct >= WF_PASS and best_config is not None:
        print(f"\n{'='*50}")
        print(f"WF PASSED ({best_wf_pct:.1%}). Running holdout...")
        print(f"{'='*50}")

        X_train = train_val_df[feature_cols]
        y_train = train_val_df['target']
        X_ho = holdout_df[feature_cols]
        y_ho = holdout_df['target']

        n_val = int(len(X_train) * 0.9)
        model = LGBMTrader(**best_config)
        model.fit(X_train.iloc[:n_val], y_train.iloc[:n_val],
                   X_train.iloc[n_val:], y_train.iloc[n_val:])

        # Generate directional signals for binary model
        probs = model.model.predict(X_ho)
        classes = model.label_encoder.classes_
        conf = best_config['confidence_threshold']

        dir_signals = pd.Series(0, index=X_ho.index)
        for j in range(len(X_ho)):
            p_down = probs[j][0]
            p_up = probs[j][1]
            if p_up >= conf:
                dir_signals.iloc[j] = 1
            elif p_down >= conf:
                dir_signals.iloc[j] = -1

        holdout_ret = holdout_df['fwd_ret']
        holdout_sharpe, holdout_trades = compute_sharpe_from_signals(dir_signals, holdout_ret)
        holdout_coverage = float((dir_signals != 0).mean())

        mask = dir_signals != 0
        correct = ((dir_signals == 1) & (y_ho == 1)) | ((dir_signals == -1) & (y_ho == 0))
        holdout_dir_acc = float(correct[mask].mean()) if mask.sum() > 0 else 0.0

        # Trades per month
        n_months = (holdout_df.index[-1] - holdout_df.index[0]).days / 30.44
        holdout_trades_per_month = holdout_trades / max(n_months, 1)

        # Monthly returns
        position = dir_signals.shift(1).fillna(0)
        costs = dir_signals.diff().abs().fillna(0) * 0.0005
        strat_ret = position * holdout_ret - costs
        monthly = strat_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        fi_ho = model.get_feature_importance(feature_cols)

        print(f"\nHoldout results:")
        print(f"  Sharpe:        {holdout_sharpe:.3f}")
        print(f"  Trades:        {holdout_trades}")
        print(f"  Trades/month:  {holdout_trades_per_month:.1f}")
        print(f"  Coverage:      {holdout_coverage:.1%}")
        print(f"  Dir accuracy:  {holdout_dir_acc:.1%}")
        print(f"  Worst month:   {monthly.min():.2%}")
        print(f"  Best month:    {monthly.max():.2%}")
        print(f"\nTop 10 features (holdout model):")
        for idx_f, (fname, fval) in enumerate(fi_ho.head(10).items()):
            print(f"  {idx_f+1:2d}. {fname:30s} {fval:10.1f}")

        all_results['holdout'] = {
            'sharpe': holdout_sharpe, 'trades': holdout_trades,
            'trades_per_month': holdout_trades_per_month,
            'coverage': holdout_coverage, 'dir_accuracy': holdout_dir_acc,
            'worst_month': float(monthly.min()), 'best_month': float(monthly.max()),
            'top_features': fi_ho.head(10).to_dict(),
        }
    else:
        print(f"\nWF FAILED ({best_wf_pct:.1%}). No holdout run.")

    # ── Save ──────────────────────────────────────────────────────
    ho_str = f"{holdout_sharpe:.2f}" if holdout_sharpe is not None else "N/A"

    print(f"\n{'='*70}")
    print(f"Baseline: Daily MA — Sharpe 3.14, 3.2 trades/month")
    print(f"Phase 1 LightGBM BTC: Sharpe {ho_str}, WF {best_wf_pct:.1%}")
    print(f"{'='*70}")

    all_results['summary'] = {
        'best_wf_pct': best_wf_pct, 'best_wf_mean': best_wf_mean,
        'holdout_sharpe': holdout_sharpe,
        'holdout_trades_per_month': holdout_trades_per_month,
        'holdout_coverage': holdout_coverage,
        'holdout_dir_accuracy': holdout_dir_acc,
        'features_used': feature_cols, 'n_features': len(feature_cols),
        'disguised_momentum': disguised,
        'best_config': {k: v for k, v in best_config.items()} if best_config else None,
        'elapsed_seconds': time.time() - t0,
    }

    with open('/home/ubuntu/projects/crypto-trader/phase1_lgbm_btc_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nSaved to phase1_lgbm_btc_results.json ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()
