"""
LightGBM market structure experiment runner.
"""

import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from ml.features import build_feature_matrix
from ml.lgbm_model import LGBMTrader
from ml.walk_forward_ml import run_walk_forward_ml, compute_sharpe_from_signals

SYMBOL = 'SOL/USDT:USDT'
BTC_SYMBOL = 'BTC/USDT:USDT'
START = '2022-01-01'
END_TRAIN = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END = '2026-03-21'

LGBM_CONFIGS = [
    (0.52, 4, 15, 0.05),
    (0.55, 4, 15, 0.05),
    (0.52, 3, 8, 0.05),
    (0.52, 4, 15, 0.02),
    (0.55, 4, 31, 0.05),
]

TARGET_THRESHOLD = 0.003
WF_PASS_THRESHOLD = 0.60

print("=" * 70)
print("LightGBM Market Structure Experiment")
print("=" * 70)

# ── Step 1: Build feature matrix ─────────────────────────────────────────

print("\nBuilding feature matrix...")
feature_df, feature_cols = build_feature_matrix(
    symbol_futures=SYMBOL, symbol_btc=BTC_SYMBOL,
    start=START, end=HOLDOUT_END,
    target_horizon=1, target_threshold=TARGET_THRESHOLD,
)

print(f"\nFeature count: {len(feature_cols)}")
print(f"Sample count: {len(feature_df)}")
print(f"Class balance:\n{feature_df['target'].value_counts()}")

train_val_df = feature_df.loc[:END_TRAIN]
holdout_df = feature_df.loc[HOLDOUT_START:]

print(f"\nTrain+val: {len(train_val_df)} bars")
print(f"Holdout:   {len(holdout_df)} bars")

# ── Step 2: Sanity check ─────────────────────────────────────────────────

print("\n" + "=" * 50)
print("Sanity check: feature importance on full sample")
print("=" * 50)

X_all = feature_df[feature_cols]
y_all = feature_df['target']
n_split = int(len(X_all) * 0.7)

sanity_model = LGBMTrader(confidence_threshold=0.52)
sanity_model.fit(X_all.iloc[:n_split], y_all.iloc[:n_split],
                  X_all.iloc[n_split:], y_all.iloc[n_split:])
fi = sanity_model.get_feature_importance(feature_cols)
disguised = sanity_model.check_disguised_momentum(fi)

print("\nTop 15 features by importance:")
print(fi.head(15).to_string())
print(f"\nDisguised momentum: {disguised}")

# ── Step 3: Walk-forward ─────────────────────────────────────────────────

print("\n" + "=" * 50)
print(f"Walk-forward: {len(LGBM_CONFIGS)} configurations")
print("=" * 50)

best_config = None
best_wf_sharpe = -999
best_wf_consistency = 0
all_wf_results = {}

for i, (conf_thresh, max_depth, num_leaves, lr) in enumerate(LGBM_CONFIGS):
    print(f"\n{'='*40}")
    print(f"Config {i+1}/{len(LGBM_CONFIGS)}: "
          f"conf={conf_thresh}, depth={max_depth}, leaves={num_leaves}, lr={lr}")

    model_kwargs = dict(
        confidence_threshold=conf_thresh, max_depth=max_depth,
        num_leaves=num_leaves, learning_rate=lr,
        n_estimators=200, min_child_samples=50,
    )

    wf_results = run_walk_forward_ml(
        feature_df=train_val_df, feature_cols=feature_cols,
        model_class=LGBMTrader, model_kwargs=model_kwargs,
        n_windows=12, train_months=9, test_months=3, gap_bars=5,
    )

    test_sharpes = [r.test_sharpe for r in wf_results]
    wf_pct = sum(1 for s in test_sharpes if s > 0) / len(test_sharpes) if test_sharpes else 0
    wf_mean = np.mean(test_sharpes) if test_sharpes else 0
    mean_cov = np.mean([r.coverage for r in wf_results])
    mean_acc = np.mean([r.directional_accuracy for r in wf_results])

    print(f"\nResults: WF%={wf_pct:.1%}, Mean={wf_mean:.2f}, "
          f"Cov={mean_cov:.1%}, Acc={mean_acc:.1%}")

    config_result = {
        'config': {k: v for k, v in model_kwargs.items()},
        'wf_pct': wf_pct, 'wf_mean_sharpe': wf_mean,
        'mean_coverage': mean_cov, 'mean_dir_accuracy': mean_acc,
        'per_window': [{
            'window': r.window_id, 'test_sharpe': r.test_sharpe,
            'train_sharpe': r.train_sharpe, 'dir_acc': r.directional_accuracy,
            'coverage': r.coverage, 'n_trades': r.n_trades,
            'top_features': r.top_features[:5],
        } for r in wf_results]
    }

    if wf_pct > best_wf_consistency or (wf_pct == best_wf_consistency and wf_mean > best_wf_sharpe):
        best_wf_consistency = wf_pct
        best_wf_sharpe = wf_mean
        best_config = model_kwargs
        all_wf_results['best'] = config_result

    all_wf_results[f'config_{i}'] = config_result

print(f"\nBest config: {best_config}")
print(f"Best WF: {best_wf_consistency:.1%} positive, mean={best_wf_sharpe:.2f}")

# ── Step 4: Holdout ──────────────────────────────────────────────────────

holdout_sharpe = None
holdout_trades = 0
holdout_coverage = 0
holdout_dir_acc = 0

if best_wf_consistency >= WF_PASS_THRESHOLD:
    print(f"\n{'='*50}")
    print(f"WF PASSED ({best_wf_consistency:.1%}). Running holdout...")
    print(f"{'='*50}")

    X_train_full = train_val_df[feature_cols]
    y_train_full = train_val_df['target']
    X_holdout = holdout_df[feature_cols]
    y_holdout = holdout_df['target']

    n_val = int(len(X_train_full) * 0.9)
    final_model = LGBMTrader(**best_config)
    final_model.fit(X_train_full.iloc[:n_val], y_train_full.iloc[:n_val],
                     X_train_full.iloc[n_val:], y_train_full.iloc[n_val:])

    holdout_signals = final_model.predict_signals(X_holdout)
    holdout_ret = holdout_df['fwd_ret']
    holdout_sharpe, holdout_trades = compute_sharpe_from_signals(holdout_signals, holdout_ret)

    holdout_coverage = float((holdout_signals != 0).mean())
    mask = holdout_signals != 0
    holdout_dir_acc = float((holdout_signals[mask] == y_holdout[mask]).mean()) if mask.sum() > 0 else 0.0

    position = holdout_signals.shift(1).fillna(0)
    costs = holdout_signals.diff().abs().fillna(0) * 0.0005
    strat_ret = position * holdout_ret - costs
    monthly = strat_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)

    fi_ho = final_model.get_feature_importance(feature_cols)
    disguised_ho = final_model.check_disguised_momentum(fi_ho)

    print(f"\nHoldout: Sharpe={holdout_sharpe:.3f}, Trades={holdout_trades}, "
          f"Cov={holdout_coverage:.1%}, DirAcc={holdout_dir_acc:.1%}")
    print(f"Worst month: {monthly.min():.2%}, Best: {monthly.max():.2%}")
    print(f"Disguised momentum: {disguised_ho}")
    print(f"\nTop 10 features:")
    print(fi_ho.head(10).to_string())

    all_wf_results['holdout'] = {
        'sharpe': holdout_sharpe, 'trades': holdout_trades,
        'coverage': holdout_coverage, 'dir_accuracy': holdout_dir_acc,
        'worst_month': float(monthly.min()), 'best_month': float(monthly.max()),
        'mean_month': float(monthly.mean()),
        'disguised_momentum': disguised_ho,
        'top_features': fi_ho.head(15).index.tolist(),
    }
else:
    print(f"\nWF FAILED ({best_wf_consistency:.1%} < {WF_PASS_THRESHOLD:.0%}). No holdout.")

# ── Step 5: Final comparison ─────────────────────────────────────────────

print(f"\n{'='*70}")
print("FINAL COMPARISON")
print(f"{'='*70}")
print(f"V3 baseline:  Sharpe 0.91, WF 60.7%")
ho_str = f"{holdout_sharpe:.2f}" if holdout_sharpe is not None else "N/A"
print(f"LGBM:         Sharpe {ho_str}, WF {best_wf_consistency:.1%}")
delta = holdout_sharpe - 0.91 if holdout_sharpe is not None else None
print(f"Delta:        {delta:+.2f}" if delta is not None else "Delta:        N/A")

# Save
all_wf_results['summary'] = {
    'best_wf_consistency': best_wf_consistency,
    'best_wf_mean_sharpe': best_wf_sharpe,
    'holdout_sharpe': holdout_sharpe,
    'holdout_trades': holdout_trades,
    'holdout_coverage': holdout_coverage,
    'holdout_dir_acc': holdout_dir_acc,
    'disguised_momentum': disguised,
    'delta_vs_v3': delta,
}

with open('lgbm_results.json', 'w') as f:
    json.dump(all_wf_results, f, indent=2, default=str)

# Append to STRATEGY_LOG.md
delta_str = f"{delta:+.2f}" if delta is not None else "N/A"
entry = f"""

### LGBM: LightGBM market structure model

Features: 1H MS aggregated + 4H OHLCV + regime context + time ({len(feature_cols)} total)
Best config: conf={best_config.get('confidence_threshold')}, depth={best_config.get('max_depth')}, leaves={best_config.get('num_leaves')}, lr={best_config.get('learning_rate')}
Disguised momentum: {'YES' if disguised else 'NO'}
Top features: {fi.head(5).index.tolist()}
WF % positive: {best_wf_consistency:.1%}
WF mean OOS Sharpe: {best_wf_sharpe:.2f}
Holdout Sharpe: {ho_str}
Dir accuracy: {holdout_dir_acc:.1%}
Coverage: {holdout_coverage:.1%}
Delta vs V3: {delta_str}
Conclusion: {'PASS' if holdout_sharpe and holdout_sharpe > 1.0 else 'FAIL' if holdout_sharpe is None or holdout_sharpe <= 0.91 else 'MARGINAL'}
"""

with open('STRATEGY_LOG.md', 'r') as f:
    content = f.read()
with open('STRATEGY_LOG.md', 'w') as f:
    f.write(content + entry)

print("\nResults saved to lgbm_results.json and STRATEGY_LOG.md")
