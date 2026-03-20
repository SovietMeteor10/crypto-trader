#!/usr/bin/env python3
"""Exhaustive strategy search — runs all strategies through the full testing protocol."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

from crypto_infra import DataModule, CostModule, SizerModule, BacktestEngine, MetricsModule

# Strategy imports
from strategies.s1a_dual_ma import DualMACrossSignal
from strategies.s1b_breakout_volume import BreakoutVolumeSignal
from strategies.s1c_trend_regime import TrendRegimeSignal
from strategies.s2a_rsi_reversion import RSIReversionSignal
from strategies.s2b_bollinger_reversion import BollingerReversionSignal
from strategies.s2c_zscore_reversion import ZScoreReversionSignal
from strategies.s3a_funding_carry import FundingCarrySignal
from strategies.s4a_vol_breakout import VolBreakoutSignal
from strategies.s5a_cross_momentum import CrossMomentumSignal

# Constants
TRAIN_START = "2021-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"
HOLDOUT_END = "2025-03-20"

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
TIMEFRAMES_BY_STRATEGY = {
    "1A": ["4h", "1d"],
    "1B": ["4h"],
    "1C": ["4h", "1d"],
    "2A": ["1h", "4h"],
    "2B": ["1h", "4h"],
    "2C": ["1h", "4h"],
    "3A": ["1h"],
    "4A": ["1h", "4h"],
    "5A": ["4h"],
}

INITIAL_EQUITY = 1000.0
N_OPTUNA_TRIALS = 50

dm = DataModule()
cm = CostModule()
sm = SizerModule(fraction=0.02, leverage=3.0)
engine = BacktestEngine(dm, cm, sm)
mm = MetricsModule()

log_entries = []


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def quick_screen(signal_module, params, symbol, timeframe):
    """Step 1: Quick viability screen with default params."""
    try:
        bundle = engine.run(
            signal_module, params, symbol, timeframe,
            TRAIN_START, TRAIN_END, INITIAL_EQUITY, "screen",
        )
        n_trades = len(bundle.trades)
        weeks = (pd.Timestamp(TRAIN_END) - pd.Timestamp(TRAIN_START)).days / 7
        trades_per_week = n_trades / max(weeks, 1)
        return {
            "trades": n_trades,
            "trades_per_week": round(trades_per_week, 2),
            "final_equity": round(bundle.equity_curve.iloc[-1], 2),
            "viable": trades_per_week >= 0.5,
        }
    except Exception as e:
        return {"trades": 0, "trades_per_week": 0, "viable": False, "error": str(e)}


def run_three_split_test(signal_module, symbol, timeframe):
    """Step 2: Optuna optimisation on train, then validate."""
    log(f"  Optimising {symbol} {timeframe}...")
    best_params, train_sharpe = engine._optimise(
        signal_module, symbol, timeframe,
        TRAIN_START, TRAIN_END, N_OPTUNA_TRIALS, INITIAL_EQUITY,
    )
    log(f"  Train Sharpe: {train_sharpe:.2f}, params: {best_params}")

    # Run validation
    val_bundle = engine.run(
        signal_module, best_params, symbol, timeframe,
        VAL_START, VAL_END, INITIAL_EQUITY, "validation",
    )
    val_metrics = mm.compute(val_bundle, train_sharpe=train_sharpe)

    log(f"  Val Sharpe: {val_metrics.sharpe_ratio:.2f}, overfit: {val_metrics.flag_overfit}")

    return {
        "best_params": best_params,
        "train_sharpe": round(train_sharpe, 4),
        "val_sharpe": round(val_metrics.sharpe_ratio, 4),
        "val_metrics": val_metrics,
        "overfit": bool(val_metrics.flag_overfit),
    }


def run_walk_forward_test(signal_module, symbol, timeframe):
    """Step 3: Walk-forward validation."""
    log(f"  Walk-forward {symbol} {timeframe}...")
    results = engine.run_walk_forward(
        signal_module, symbol, timeframe,
        TRAIN_START, HOLDOUT_END,
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=30, initial_equity=INITIAL_EQUITY,
    )

    sharpes = []
    for r in results:
        m = mm.compute(r)
        sharpes.append(m.sharpe_ratio)

    mean_oos = np.mean(sharpes) if sharpes else 0
    pct_positive = sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1) * 100

    log(f"  WF mean OOS Sharpe: {mean_oos:.2f}, {pct_positive:.0f}% positive ({len(results)} windows)")

    return {
        "n_windows": len(results),
        "mean_oos_sharpe": round(mean_oos, 4),
        "pct_positive": round(pct_positive, 1),
        "consistent": pct_positive >= 60,
        "window_sharpes": [round(s, 4) for s in sharpes],
    }


def run_holdout_test(signal_module, params, symbol, timeframe, train_sharpe):
    """Step 5: Holdout test."""
    log(f"  Holdout {symbol} {timeframe}...")
    bundle = engine.run(
        signal_module, params, symbol, timeframe,
        HOLDOUT_START, HOLDOUT_END, INITIAL_EQUITY, "holdout",
    )

    # Get BTC returns for correlation
    btc_data = dm.get_ohlcv("BTC/USDT:USDT", timeframe, HOLDOUT_START, HOLDOUT_END)
    btc_returns = btc_data["close"].pct_change().dropna()

    metrics = mm.compute(bundle, btc_returns=btc_returns, train_sharpe=train_sharpe)
    summary = mm.format_summary(metrics, f"{signal_module.name} holdout")

    return {
        "metrics": metrics,
        "summary": summary,
        "bundle": bundle,
    }


def test_strategy(signal_module, family_code, default_params):
    """Run full test pipeline for one strategy."""
    strategy_name = signal_module.name
    timeframes = TIMEFRAMES_BY_STRATEGY.get(family_code, ["4h"])

    log(f"\n{'='*60}")
    log(f"TESTING: {strategy_name} (Family {family_code})")
    log(f"{'='*60}")

    best_result = None
    best_val_sharpe = -999

    for symbol in SYMBOLS:
        for tf in timeframes:
            entry = {
                "strategy": strategy_name,
                "family": family_code,
                "symbol": symbol,
                "timeframe": tf,
                "date": datetime.now().strftime("%Y-%m-%d"),
            }

            try:
                # Step 1: Quick screen
                log(f"\n  Quick screen: {symbol} {tf}")
                screen = quick_screen(signal_module, default_params, symbol, tf)
                entry["quick_screen"] = screen

                if not screen["viable"]:
                    entry["conclusion"] = "REJECTED: INFREQUENT"
                    log(f"  REJECTED: {screen['trades_per_week']} trades/week (need >= 0.5)")
                    log_entries.append(entry)
                    continue

                log(f"  VIABLE: {screen['trades_per_week']} trades/week")

                # Step 2: Three-split optimisation
                split = run_three_split_test(signal_module, symbol, tf)
                entry["optimisation"] = {
                    "train_sharpe": split["train_sharpe"],
                    "val_sharpe": split["val_sharpe"],
                    "overfit": split["overfit"],
                    "best_params": split["best_params"],
                }

                if split["overfit"]:
                    entry["conclusion"] = "REJECTED: OVERFIT"
                    log(f"  REJECTED: overfit (train={split['train_sharpe']:.2f}, val={split['val_sharpe']:.2f})")
                    log_entries.append(entry)
                    continue

                # Track best for walk-forward / holdout
                if split["val_sharpe"] > best_val_sharpe:
                    best_val_sharpe = split["val_sharpe"]
                    best_result = {
                        "symbol": symbol,
                        "tf": tf,
                        "params": split["best_params"],
                        "train_sharpe": split["train_sharpe"],
                        "val_sharpe": split["val_sharpe"],
                    }

                # Step 3: Walk-forward
                wf = run_walk_forward_test(signal_module, symbol, tf)
                entry["walk_forward"] = wf

                if not wf["consistent"]:
                    entry["conclusion"] = "REJECTED: INCONSISTENT"
                    log(f"  REJECTED: inconsistent ({wf['pct_positive']:.0f}% positive windows)")
                    log_entries.append(entry)
                    continue

                # Step 4: Perturbation test
                log(f"  Perturbation test...")
                perturb = mm.run_perturbation_test(
                    signal_module, split["best_params"], engine,
                    symbol, tf, VAL_START, VAL_END,
                )
                entry["perturbation"] = {
                    "max_drop_pct": perturb["max_sharpe_drop_pct"],
                    "fragile": perturb["fragile"],
                }
                log(f"  Perturbation max drop: {perturb['max_sharpe_drop_pct']:.1f}%, fragile: {perturb['fragile']}")

                # Step 5: Holdout
                holdout = run_holdout_test(
                    signal_module, split["best_params"], symbol, tf,
                    split["train_sharpe"],
                )
                entry["holdout"] = {
                    "sharpe": holdout["metrics"].sharpe_ratio,
                    "total_return": holdout["metrics"].total_return_pct,
                    "max_dd": holdout["metrics"].max_drawdown_pct,
                    "win_rate": holdout["metrics"].win_rate,
                    "trades": holdout["metrics"].total_trades,
                    "btc_corr": holdout["metrics"].btc_correlation,
                    "passes_all": holdout["metrics"].passes_all_checks,
                    "monthly_mean": holdout["metrics"].monthly_return_mean,
                    "monthly_worst": holdout["metrics"].monthly_return_worst,
                }
                entry["holdout_summary"] = holdout["summary"]

                if holdout["metrics"].passes_all_checks:
                    entry["conclusion"] = "CANDIDATE"
                    log(f"  *** CANDIDATE: Sharpe={holdout['metrics'].sharpe_ratio:.2f}, "
                        f"Return={holdout['metrics'].total_return_pct:.1f}%")
                else:
                    flags = []
                    m = holdout["metrics"]
                    if m.flag_overfit: flags.append("overfit")
                    if m.flag_insufficient_trades: flags.append("insufficient_trades")
                    if m.flag_high_btc_correlation: flags.append("high_btc_corr")
                    if m.flag_negative_skew: flags.append("neg_skew")
                    if m.flag_long_drawdown: flags.append("long_dd")
                    if m.flag_consecutive_losses: flags.append("consec_losses")
                    entry["conclusion"] = f"REJECTED: FLAGS [{', '.join(flags)}]"
                    log(f"  REJECTED: flags {flags}")

            except Exception as e:
                entry["conclusion"] = f"ERROR: {str(e)}"
                log(f"  ERROR: {e}")
                traceback.print_exc()

            log_entries.append(entry)

    return best_result


# ============================================================================
# MAIN: Run all strategies
# ============================================================================

strategies = [
    (DualMACrossSignal(), "1A", {"fast_period": 20, "slow_period": 50, "atr_mult": 2.0}),
    (BreakoutVolumeSignal(), "1B", {"lookback": 30, "vol_mult": 2.0}),
    (TrendRegimeSignal(), "1C", {"fast_period": 20, "slow_period": 50, "adx_period": 14, "adx_threshold": 25}),
    (RSIReversionSignal(), "2A", {"rsi_period": 14, "oversold": 30, "overbought": 70, "max_hold_bars": 12}),
    (BollingerReversionSignal(), "2B", {"period": 20, "std_dev": 2.0}),
    (ZScoreReversionSignal(), "2C", {"lookback": 48, "entry_z": 2.0, "exit_z": 0.2}),
    (FundingCarrySignal(), "3A", {"funding_threshold": 0.0003, "hold_periods": 3}),
    (VolBreakoutSignal(), "4A", {"vol_lookback": 48, "spike_mult": 2.0}),
    (CrossMomentumSignal(), "5A", {"momentum_period": 72, "rebalance_bars": 12}),
]

candidates = []

for signal, family, defaults in strategies:
    try:
        result = test_strategy(signal, family, defaults)
        # Check if any entry was a candidate
        for entry in log_entries:
            if entry.get("strategy") == signal.name and entry.get("conclusion") == "CANDIDATE":
                candidates.append(entry)
    except Exception as e:
        log(f"FATAL ERROR testing {signal.name}: {e}")
        traceback.print_exc()

# ============================================================================
# Save results
# ============================================================================

log(f"\n\n{'='*60}")
log(f"SEARCH COMPLETE")
log(f"{'='*60}")
log(f"Total entries: {len(log_entries)}")
log(f"Candidates: {len(candidates)}")

# Save full log as JSON
with open("strategy_results.json", "w") as f:
    # Convert non-serialisable types
    def serialise(obj):
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return str(obj)
        return str(obj)
    json.dump(log_entries, f, indent=2, default=serialise)

log("Results saved to strategy_results.json")

# Print summary table
log("\n\nSUMMARY TABLE:")
log(f"{'Strategy':<25} {'Symbol':<15} {'TF':<5} {'Screen':<12} {'Train SR':<10} {'Val SR':<10} {'WF OOS SR':<10} {'Holdout SR':<10} {'Result'}")
log("-" * 120)
for entry in log_entries:
    strat = entry.get("strategy", "?")
    sym = entry.get("symbol", "?").split("/")[0]
    tf = entry.get("timeframe", "?")
    screen = entry.get("quick_screen", {}).get("trades_per_week", "?")
    opt = entry.get("optimisation", {})
    train_sr = opt.get("train_sharpe", "-")
    val_sr = opt.get("val_sharpe", "-")
    wf = entry.get("walk_forward", {})
    wf_sr = wf.get("mean_oos_sharpe", "-")
    ho = entry.get("holdout", {})
    ho_sr = ho.get("sharpe", "-")
    conclusion = entry.get("conclusion", "?")
    log(f"{strat:<25} {sym:<15} {tf:<5} {str(screen):<12} {str(train_sr):<10} {str(val_sr):<10} {str(wf_sr):<10} {str(ho_sr):<10} {conclusion}")
