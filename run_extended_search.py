#!/usr/bin/env python3
"""Extended strategy search — regime filters + new families 6-9."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import json, traceback
from datetime import datetime
import numpy as np, pandas as pd

from crypto_infra import DataModule, CostModule, SizerModule, BacktestEngine, MetricsModule

# Regime filter variants
from strategies.s1c_rf_variants import TrendRegimeVolFilter, TrendRegimeBTCFilter, TrendRegimeFundingFilter
# New families
from strategies.s6a_session_breakout import SessionBreakoutSignal
from strategies.s6b_funding_momentum import FundingMomentumSignal
from strategies.s7a_basis_carry import BasisCarrySignal
from strategies.s7b_funding_trend import FundingTrendSignal
from strategies.s8a_volume_weighted_momentum import VolumeWeightedMomentumSignal
from strategies.s8b_nr7_breakout import NR7BreakoutSignal
from strategies.s8c_volume_range_squeeze import VolumeRangeSqueezeSignal

TRAIN_START, TRAIN_END = "2021-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
HOLDOUT_START, HOLDOUT_END = "2024-01-01", "2025-03-20"
INITIAL_EQUITY = 1000.0
N_OPTUNA_TRIALS = 50

dm = DataModule()
cm = CostModule()
sm = SizerModule(fraction=0.02, leverage=3.0)
engine = BacktestEngine(dm, cm, sm)
mm = MetricsModule()
log_entries = []

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def quick_screen(sig, params, symbol, tf):
    try:
        b = engine.run(sig, params, symbol, tf, TRAIN_START, TRAIN_END, INITIAL_EQUITY, "screen")
        weeks = (pd.Timestamp(TRAIN_END) - pd.Timestamp(TRAIN_START)).days / 7
        tpw = len(b.trades) / max(weeks, 1)
        return {"trades": len(b.trades), "trades_per_week": round(tpw, 2), "viable": tpw >= 0.5}
    except Exception as e:
        return {"trades": 0, "trades_per_week": 0, "viable": False, "error": str(e)}

def test_strategy(signal, family, defaults, symbols, timeframes):
    name = signal.name
    log(f"\n{'='*60}\nTESTING: {name} (Family {family})\n{'='*60}")

    for sym in symbols:
        for tf in timeframes:
            entry = {"strategy": name, "family": family, "symbol": sym, "timeframe": tf, "date": "2026-03-20"}
            try:
                log(f"  Screen: {sym} {tf}")
                screen = quick_screen(signal, defaults, sym, tf)
                entry["quick_screen"] = screen
                if not screen["viable"]:
                    entry["conclusion"] = "REJECTED: INFREQUENT"
                    log(f"  REJECTED: {screen['trades_per_week']} tpw")
                    log_entries.append(entry); continue
                log(f"  VIABLE: {screen['trades_per_week']} tpw")

                # Optimise
                log(f"  Optimising...")
                best_params, train_sr = engine._optimise(signal, sym, tf, TRAIN_START, TRAIN_END, N_OPTUNA_TRIALS, INITIAL_EQUITY)
                val_b = engine.run(signal, best_params, sym, tf, VAL_START, VAL_END, INITIAL_EQUITY, "validation")
                val_m = mm.compute(val_b, train_sharpe=train_sr)
                log(f"  Train SR: {train_sr:.2f}, Val SR: {val_m.sharpe_ratio:.2f}, overfit: {val_m.flag_overfit}")
                entry["optimisation"] = {"train_sharpe": round(train_sr, 4), "val_sharpe": round(val_m.sharpe_ratio, 4), "overfit": bool(val_m.flag_overfit), "best_params": best_params}

                if val_m.flag_overfit:
                    entry["conclusion"] = "REJECTED: OVERFIT"
                    log_entries.append(entry); continue

                # Walk-forward
                log(f"  Walk-forward...")
                wf = engine.run_walk_forward(signal, sym, tf, TRAIN_START, HOLDOUT_END, train_months=6, test_months=2, gap_weeks=2, n_optuna_trials=30, initial_equity=INITIAL_EQUITY)
                sharpes = [mm.compute(r).sharpe_ratio for r in wf]
                mean_oos = np.mean(sharpes)
                pct_pos = sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1) * 100
                log(f"  WF: mean OOS SR {mean_oos:.2f}, {pct_pos:.0f}% positive ({len(wf)} windows)")
                entry["walk_forward"] = {"mean_oos_sharpe": round(mean_oos, 4), "pct_positive": round(pct_pos, 1), "consistent": pct_pos >= 60}

                if pct_pos < 60:
                    entry["conclusion"] = "REJECTED: INCONSISTENT"
                    log_entries.append(entry); continue

                # Perturbation
                log(f"  Perturbation test...")
                perturb = mm.run_perturbation_test(signal, best_params, engine, sym, tf, VAL_START, VAL_END)
                entry["perturbation"] = {"max_drop_pct": perturb["max_sharpe_drop_pct"], "fragile": perturb["fragile"]}
                log(f"  Perturbation: {perturb['max_sharpe_drop_pct']:.1f}% drop, fragile: {perturb['fragile']}")

                # Holdout
                log(f"  Holdout...")
                ho_b = engine.run(signal, best_params, sym, tf, HOLDOUT_START, HOLDOUT_END, INITIAL_EQUITY, "holdout")
                btc_data = dm.get_ohlcv("BTC/USDT:USDT", tf, HOLDOUT_START, HOLDOUT_END)
                btc_ret = btc_data["close"].pct_change().dropna()
                ho_m = mm.compute(ho_b, btc_returns=btc_ret, train_sharpe=train_sr)
                entry["holdout"] = {"sharpe": ho_m.sharpe_ratio, "total_return": ho_m.total_return_pct, "max_dd": ho_m.max_drawdown_pct, "win_rate": ho_m.win_rate, "trades": ho_m.total_trades, "btc_corr": ho_m.btc_correlation, "passes_all": ho_m.passes_all_checks, "monthly_mean": ho_m.monthly_return_mean, "monthly_worst": ho_m.monthly_return_worst}

                if ho_m.passes_all_checks:
                    entry["conclusion"] = "CANDIDATE"
                    log(f"  *** CANDIDATE *** SR={ho_m.sharpe_ratio:.2f} Ret={ho_m.total_return_pct:.1f}%")
                else:
                    flags = [n for n, v in [("overfit", ho_m.flag_overfit), ("insuff", ho_m.flag_insufficient_trades), ("btc_corr", ho_m.flag_high_btc_correlation), ("neg_skew", ho_m.flag_negative_skew), ("long_dd", ho_m.flag_long_drawdown), ("consec", ho_m.flag_consecutive_losses)] if v]
                    entry["conclusion"] = f"REJECTED: FLAGS [{', '.join(flags)}]"
                    log(f"  REJECTED: flags {flags}")
            except Exception as e:
                entry["conclusion"] = f"ERROR: {str(e)}"
                log(f"  ERROR: {e}")
                traceback.print_exc()
            log_entries.append(entry)

# ============ STRATEGIES TO TEST ============

strategies = [
    # Regime filter variants (SOL only, 4h)
    (TrendRegimeVolFilter(), "1C-RF-A", {"fast_period": 43, "slow_period": 113, "adx_period": 10, "adx_threshold": 23, "vol_lookback": 120}, ["SOL/USDT:USDT"], ["4h"]),
    (TrendRegimeBTCFilter(), "1C-RF-C", {"fast_period": 43, "slow_period": 113, "adx_period": 10, "adx_threshold": 23}, ["SOL/USDT:USDT"], ["4h"]),
    (TrendRegimeFundingFilter(), "1C-RF-D", {"fast_period": 43, "slow_period": 113, "adx_period": 10, "adx_threshold": 23, "funding_lookback": 60}, ["SOL/USDT:USDT"], ["4h"]),
    # New families — test on SOL (best asset) + BTC
    (SessionBreakoutSignal(), "6A", {"session": "london", "confirmation_pct": 0.3, "stop_atr_mult": 2.0}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["1h"]),
    (FundingMomentumSignal(), "6B", {"funding_threshold": 0.0005, "entry_delay_bars": 2, "hold_bars": 4}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["1h"]),
    (BasisCarrySignal(), "7A", {"entry_funding_threshold": 0.0005, "exit_funding_threshold": 0.0001, "price_stop_pct": 1.5}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["1h"]),
    (FundingTrendSignal(), "7B", {"lookback_periods": 6, "trend_threshold": 0.0001}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["1h", "4h"]),
    (VolumeWeightedMomentumSignal(), "8A", {"volume_lookback": 48, "volume_z_threshold": 2.0, "price_move_pct": 1.0, "hold_bars": 6}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["1h", "4h"]),
    (NR7BreakoutSignal(), "8B", {"lookback": 7, "breakout_pct": 0.3}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["4h"]),
    (VolumeRangeSqueezeSignal(), "8C", {"vol_surge_mult": 2.0, "range_contraction_pct": 0.5, "lookback": 12, "hold_bars": 8}, ["BTC/USDT:USDT", "SOL/USDT:USDT"], ["4h"]),
]

for sig, fam, defaults, symbols, tfs in strategies:
    try:
        test_strategy(sig, fam, defaults, symbols, tfs)
    except Exception as e:
        log(f"FATAL: {sig.name}: {e}")
        traceback.print_exc()

# ============ SUMMARY ============
log(f"\n{'='*60}\nEXTENDED SEARCH COMPLETE\n{'='*60}")
log(f"Entries: {len(log_entries)}")
candidates = [e for e in log_entries if e.get("conclusion") == "CANDIDATE"]
log(f"Candidates: {len(candidates)}")

with open("extended_results.json", "w") as f:
    def ser(o):
        if hasattr(o, '__dict__'): return {k: v for k, v in o.__dict__.items() if not k.startswith('_')}
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.bool_,)): return bool(o)
        return str(o)
    json.dump(log_entries, f, indent=2, default=ser)

log("\nSUMMARY:")
log(f"{'Strategy':<30} {'Sym':<6} {'TF':<4} {'Screen':<8} {'Train':<8} {'Val':<8} {'WF%':<8} {'HO SR':<8} {'Result'}")
log("-" * 110)
for e in log_entries:
    s = e.get("strategy","?"); sym = e.get("symbol","?").split("/")[0][:3]; tf = e.get("timeframe","?")
    scr = e.get("quick_screen",{}).get("trades_per_week","-"); o = e.get("optimisation",{})
    tr = o.get("train_sharpe","-"); v = o.get("val_sharpe","-")
    wf = e.get("walk_forward",{}).get("pct_positive","-"); ho = e.get("holdout",{}).get("sharpe","-")
    log(f"{s:<30} {sym:<6} {tf:<4} {str(scr):<8} {str(tr):<8} {str(v):<8} {str(wf):<8} {str(ho):<8} {e.get('conclusion','?')}")
