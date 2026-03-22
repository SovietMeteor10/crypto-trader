"""
LightGBM V2: Fixed data + features + target.

Fixes from V1:
1. Extended data to 2020-09 (BTC) / 2021-12 (SOL) — more training data
2. Reduced to 15 features from 79 — prevents overfitting
3. Binary classification (up/down) instead of ternary — simpler target
"""

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from ml.lgbm_model import LGBMTrader
from ml.walk_forward_ml import run_walk_forward_ml, compute_sharpe_from_signals
from ml.features import (load_market_structure, aggregate_1h_to_4h,
                          compute_ohlcv_features, compute_regime_features)
from pathlib import Path

SYMBOL = 'SOL/USDT:USDT'
BTC_SYMBOL = 'BTC/USDT:USDT'
START = '2020-09-01'
END_TRAIN = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'
TARGET_THRESHOLD = 0.003
WF_PASS = 0.60

# The 15 selected features
SELECTED_FEATURES = [
    # Market structure (8)
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
    (0.52, 4, 15, 0.05),
    (0.55, 4, 15, 0.05),
    (0.52, 3, 8, 0.05),
    (0.52, 4, 15, 0.02),
    (0.55, 4, 31, 0.05),
]


def build_feature_matrix_v2():
    """Build reduced feature matrix with binary target."""
    from crypto_infra import DataModule
    dm = DataModule()

    sol_4h = dm.get_ohlcv(SYMBOL, '4h', START, HOLDOUT_END)
    btc_4h = dm.get_ohlcv(BTC_SYMBOL, '4h', START, HOLDOUT_END)

    try:
        sol_ms = load_market_structure(SYMBOL)
    except FileNotFoundError:
        print("SOL market structure not found, using empty")
        sol_ms = pd.DataFrame()

    try:
        btc_ms = load_market_structure(BTC_SYMBOL)
    except FileNotFoundError:
        print("BTC market structure not found, using empty")
        btc_ms = pd.DataFrame()

    # Build all features (we'll select the 15 later)
    ms_feats = aggregate_1h_to_4h(sol_ms, sol_4h) if len(sol_ms) > 0 else pd.DataFrame(index=sol_4h.index)
    ohlcv_feats = compute_ohlcv_features(sol_4h)
    regime_feats = compute_regime_features(btc_4h, btc_ms if len(btc_ms) > 0 else None)

    X = pd.concat([ms_feats, ohlcv_feats, regime_feats], axis=1)

    # Select only the 15 features
    available = [f for f in SELECTED_FEATURES if f in X.columns]
    missing = [f for f in SELECTED_FEATURES if f not in X.columns]
    if missing:
        print(f"Missing features: {missing}")

    X = X[available]
    feature_cols = list(X.columns)

    # Binary target: up (1) vs down (0)
    fwd_ret = sol_4h['close'].pct_change().shift(-1)

    df = X.copy()
    df['fwd_ret'] = fwd_ret
    df['target'] = (fwd_ret > TARGET_THRESHOLD).astype(int)

    # Lag features by 1 bar
    df[feature_cols] = df[feature_cols].shift(1)

    # Drop NaN
    df = df.dropna()
    df = df.replace([np.inf, -np.inf], 0)

    # Remove flat bars (|fwd_ret| < threshold)
    mask = df['fwd_ret'].abs() > TARGET_THRESHOLD
    n_before = len(df)
    df = df[mask]

    print(f"\nFeature matrix V2:")
    print(f"  Total bars: {n_before}")
    print(f"  After removing flat: {len(df)} ({len(df)/n_before*100:.0f}%)")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Bars/features ratio: {len(df)/len(feature_cols):.0f}")
    print(f"  Target: {df['target'].value_counts().to_dict()}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Check ratio
    ratio = len(df) / len(feature_cols)
    if ratio < 100:
        print(f"\n  WARNING: bars/features ratio {ratio:.0f} < 100")
        # Drop least important features until ratio > 100
        while ratio < 100 and len(feature_cols) > 5:
            dropped = feature_cols.pop()
            df = df.drop(columns=[dropped])
            ratio = len(df) / len(feature_cols)
            print(f"  Dropped {dropped}, new ratio: {ratio:.0f}")

    return df, feature_cols


def main():
    print("=" * 70)
    print("LightGBM V2: Fixed data + features + target")
    print("=" * 70)

    feature_df, feature_cols = build_feature_matrix_v2()

    # Expected WF windows
    total_bars = len(feature_df)
    train_bars = 6 * 30 * 6   # 6 months
    test_bars = 2 * 30 * 6    # 2 months
    expected_windows = (total_bars - train_bars) // test_bars
    print(f"\n  Expected WF windows: ~{expected_windows}")

    train_val_df = feature_df.loc[:END_TRAIN]
    holdout_df = feature_df.loc[HOLDOUT_START:]
    print(f"  Train+val: {len(train_val_df)} bars")
    print(f"  Holdout:   {len(holdout_df)} bars")

    # ── Sanity check ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Sanity check: feature importance")
    print("=" * 50)

    X_all = feature_df[feature_cols]
    y_all = feature_df['target']
    n_split = int(len(X_all) * 0.7)

    sanity = LGBMTrader(confidence_threshold=0.52)
    sanity.fit(X_all.iloc[:n_split], y_all.iloc[:n_split],
               X_all.iloc[n_split:], y_all.iloc[n_split:])
    fi = sanity.get_feature_importance(feature_cols)
    disguised = sanity.check_disguised_momentum(fi)

    print("\nFeature importance:")
    print(fi.to_string())
    print(f"Disguised momentum: {disguised}")

    # ── Walk-forward ─────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Walk-forward: {len(LGBM_CONFIGS)} configs")
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
            n_windows=8, train_months=6, test_months=2, gap_bars=5,
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
            'params': kwargs, 'wf_pct': wf_pct, 'wf_mean': wf_mean,
            'coverage': mean_cov, 'accuracy': mean_acc,
            'windows': [{
                'test_sharpe': r.test_sharpe, 'train_sharpe': r.train_sharpe,
                'dir_acc': r.directional_accuracy, 'coverage': r.coverage,
                'n_trades': r.n_trades, 'top_features': r.top_features[:5],
            } for r in wf],
        }

        if wf_pct > best_wf_pct or (wf_pct == best_wf_pct and wf_mean > best_wf_mean):
            best_wf_pct = wf_pct
            best_wf_mean = wf_mean
            best_config = kwargs

    print(f"\nBest: {best_config}")
    print(f"Best WF: {best_wf_pct:.1%} positive, mean={best_wf_mean:.2f}")

    # ── Holdout ──────────────────────────────────────────────────
    holdout_sharpe = None
    holdout_trades = 0
    holdout_coverage = 0
    holdout_dir_acc = 0

    if best_wf_pct >= WF_PASS:
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

        signals = model.predict_signals(X_ho)
        # Convert binary predictions to directional: 1→+1, 0→-1 (when confident)
        dir_signals = signals.copy()
        dir_signals[signals == 0] = -1  # predicted down → short
        # But respect confidence: where original was 0 (no prediction), stay flat
        # Actually the model predicts 0 or 1. We need to handle the confidence threshold differently.
        # predict_signals returns 0 when not confident, 1 when confident up, -1 when confident down
        # But with binary target, classes are [0, 1]. Let me check what predict_signals does.
        # It uses label_encoder.classes_ which for binary will be [0, 1].
        # predicted_class 0 = down signal → should be -1
        # predicted_class 1 = up signal → should be +1
        # When not confident → 0 (flat)
        # So we need to remap: 0 → -1 for shorts
        # Actually predict_signals already does: if predicted != 0 → emit predicted
        # But with binary classes [0,1], class 0 IS a prediction (down), not flat.
        # Need to fix the signal mapping for binary.

        # Re-generate signals manually for binary
        probs = model.model.predict(X_ho)
        classes = model.label_encoder.classes_  # [0, 1]
        conf = best_config['confidence_threshold']

        dir_signals = pd.Series(0, index=X_ho.index)
        for j in range(len(X_ho)):
            p_down = probs[j][0]  # P(class 0) = P(down)
            p_up = probs[j][1]    # P(class 1) = P(up)
            if p_up >= conf:
                dir_signals.iloc[j] = 1   # long
            elif p_down >= conf:
                dir_signals.iloc[j] = -1  # short
            # else: 0 (flat)

        holdout_ret = holdout_df['fwd_ret']
        holdout_sharpe, holdout_trades = compute_sharpe_from_signals(dir_signals, holdout_ret)
        holdout_coverage = float((dir_signals != 0).mean())

        mask = dir_signals != 0
        # Map target: 1=up (correct if signal=1), 0=down (correct if signal=-1)
        correct = ((dir_signals == 1) & (y_ho == 1)) | ((dir_signals == -1) & (y_ho == 0))
        holdout_dir_acc = float(correct[mask].mean()) if mask.sum() > 0 else 0.0

        position = dir_signals.shift(1).fillna(0)
        costs = dir_signals.diff().abs().fillna(0) * 0.0005
        strat_ret = position * holdout_ret - costs
        monthly = strat_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)

        fi_ho = model.get_feature_importance(feature_cols)

        print(f"\nHoldout: Sharpe={holdout_sharpe:.3f}, Trades={holdout_trades}")
        print(f"Coverage={holdout_coverage:.1%}, DirAcc={holdout_dir_acc:.1%}")
        print(f"Worst month: {monthly.min():.2%}, Best: {monthly.max():.2%}")
        print(f"\nTop features:\n{fi_ho.to_string()}")

        all_results['holdout'] = {
            'sharpe': holdout_sharpe, 'trades': holdout_trades,
            'coverage': holdout_coverage, 'dir_accuracy': holdout_dir_acc,
            'worst_month': float(monthly.min()), 'best_month': float(monthly.max()),
        }
    else:
        print(f"\nWF FAILED ({best_wf_pct:.1%}). No holdout.")

    # ── Save ─────────────────────────────────────────────────────
    delta = holdout_sharpe - 0.91 if holdout_sharpe is not None else None
    ho_str = f"{holdout_sharpe:.2f}" if holdout_sharpe is not None else "N/A"
    delta_str = f"{delta:+.2f}" if delta is not None else "N/A"

    print(f"\n{'='*70}")
    print(f"V3 baseline: Sharpe 0.91, WF 60.7%")
    print(f"LGBM-V2:     Sharpe {ho_str}, WF {best_wf_pct:.1%}")
    print(f"Delta:       {delta_str}")
    print(f"{'='*70}")

    all_results['summary'] = {
        'best_wf_pct': best_wf_pct, 'best_wf_mean': best_wf_mean,
        'holdout_sharpe': holdout_sharpe, 'delta_vs_v3': delta,
        'features_used': feature_cols, 'n_features': len(feature_cols),
        'disguised_momentum': disguised,
    }

    with open('lgbm_v2_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    entry = f"""

### LGBM-V2: LightGBM fixed data+features+target

Fixes: extended data to 2020, reduced 79→{len(feature_cols)} features, binary classification
Features: {feature_cols}
Best config: conf={best_config.get('confidence_threshold') if best_config else 'N/A'}, depth={best_config.get('max_depth') if best_config else 'N/A'}, leaves={best_config.get('num_leaves') if best_config else 'N/A'}, lr={best_config.get('learning_rate') if best_config else 'N/A'}
Disguised momentum: {'YES' if disguised else 'NO'}
WF % positive: {best_wf_pct:.1%}
WF mean OOS Sharpe: {best_wf_mean:.2f}
Holdout Sharpe: {ho_str}
Dir accuracy: {holdout_dir_acc:.1%}
Coverage: {holdout_coverage:.1%}
Delta vs V3: {delta_str}
Conclusion: {'PASS' if holdout_sharpe and holdout_sharpe > 1.0 else f'FAIL: holdout {ho_str}' if holdout_sharpe else f'FAIL: WF {best_wf_pct:.1%}'}
"""

    with open('STRATEGY_LOG.md', 'r') as f:
        content = f.read()
    with open('STRATEGY_LOG.md', 'w') as f:
        f.write(content + entry)

    print("Saved to lgbm_v2_results.json and STRATEGY_LOG.md")


if __name__ == '__main__':
    main()
