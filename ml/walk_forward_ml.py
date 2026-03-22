"""
Walk-forward validation for ML models with purging.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class WFMLResult:
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_sharpe: float
    test_sharpe: float
    directional_accuracy: float
    coverage: float
    n_trades: int
    top_features: list


def compute_sharpe_from_signals(signals, returns, costs_per_trade=0.001):
    position_changes = signals.diff().fillna(0).abs()
    cost_series = position_changes * costs_per_trade / 2
    strategy_ret = signals.shift(1).fillna(0) * returns - cost_series

    n_trades = int((position_changes > 0).sum())
    if n_trades == 0 or strategy_ret.std() == 0:
        return 0.0, n_trades

    sharpe = strategy_ret.mean() / strategy_ret.std() * np.sqrt(252 * 6)
    return float(sharpe), n_trades


def run_walk_forward_ml(feature_df, feature_cols, model_class, model_kwargs,
                         n_windows=12, train_months=9, test_months=3, gap_bars=5):
    results = []
    total_bars = len(feature_df)
    train_bars = train_months * 30 * 6
    test_bars = test_months * 30 * 6
    step = test_bars

    windows = list(range(0, total_bars - train_bars - test_bars, step))[:n_windows]

    for w_id, start_pos in enumerate(windows):
        train_end_pos = start_pos + train_bars
        test_start_pos = train_end_pos + gap_bars
        test_end_pos = test_start_pos + test_bars

        if test_end_pos > total_bars:
            break

        train_df = feature_df.iloc[start_pos:train_end_pos]
        test_df = feature_df.iloc[test_start_pos:test_end_pos]

        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_val = test_df[feature_cols]
        y_test = test_df['target']
        ret_test = test_df['fwd_ret']

        print(f"\nWindow {w_id+1}/{len(windows)}: "
              f"train {train_df.index[0].date()} to {train_df.index[-1].date()}, "
              f"test {test_df.index[0].date()} to {test_df.index[-1].date()}")

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train, X_val, y_test)

        signals = model.predict_signals(X_val)
        coverage = float((signals != 0).mean())

        sharpe_test, n_trades = compute_sharpe_from_signals(signals, ret_test)
        sharpe_train, _ = compute_sharpe_from_signals(
            model.predict_signals(X_train), train_df['fwd_ret'])

        mask = signals != 0
        dir_acc = float((signals[mask] == y_test[mask]).mean()) if mask.sum() > 0 else 0.0

        fi = model.get_feature_importance(feature_cols)
        top_feats = fi.head(10).index.tolist()

        results.append(WFMLResult(
            window_id=w_id,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            test_start=str(test_df.index[0].date()),
            test_end=str(test_df.index[-1].date()),
            train_sharpe=sharpe_train,
            test_sharpe=sharpe_test,
            directional_accuracy=dir_acc,
            coverage=coverage,
            n_trades=n_trades,
            top_features=top_feats,
        ))

        print(f"  Train: {sharpe_train:.2f} | Test: {sharpe_test:.2f} | "
              f"DirAcc: {dir_acc:.1%} | Cov: {coverage:.1%} | Trades: {n_trades}")

    return results
