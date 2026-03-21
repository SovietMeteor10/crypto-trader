#!/usr/bin/env python3
"""V4 Targeted Research — Three Literature-Informed Experiments."""
import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import json, traceback, time
from datetime import datetime
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

from crypto_infra import DataModule, CostModule, SizerModule, BacktestEngine, MetricsModule
from strategies.exp1_large_move_reversion import LargeMoveReversionSignal
from strategies.s1c_trend_regime import TrendRegimeSignal
from strategies.s1c_rf_variants import TrendRegimeBTCFilter

TRAIN_START, TRAIN_END = "2021-01-01", "2022-12-31"
VAL_START, VAL_END = "2023-01-01", "2023-12-31"
HOLDOUT_START, HOLDOUT_END = "2024-01-01", "2025-03-20"
INITIAL_EQUITY = 1000.0

dm = DataModule()
cm = CostModule()
mm = MetricsModule()

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# EXPERIMENT 1: Large-Move Mean Reversion at 2H
# ============================================================
def run_experiment_1():
    log("\n" + "="*60)
    log("EXPERIMENT 1: Large-Move Mean Reversion at 2H")
    log("="*60)

    sm = SizerModule(fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cm, sm)
    signal = LargeMoveReversionSignal()

    results = {}
    for sym in ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']:
        base = sym.split('/')[0]
        log(f"\n--- {base} 2H ---")

        # Quick screen
        try:
            screen_b = engine.run(signal, {"atr_period": 20, "entry_threshold": 2.0, "max_hold_bars": 4},
                                  sym, '2h', TRAIN_START, TRAIN_END, INITIAL_EQUITY, "screen")
            weeks = (pd.Timestamp(TRAIN_END) - pd.Timestamp(TRAIN_START)).days / 7
            tpw = len(screen_b.trades) / max(weeks, 1)
            log(f"  Screen: {tpw:.2f} tpw, {len(screen_b.trades)} trades")

            # Optimise
            log(f"  Optimising (50 trials)...")
            best_params, train_sr = engine._optimise(signal, sym, '2h', TRAIN_START, TRAIN_END, 50, INITIAL_EQUITY)
            log(f"  Train Sharpe: {train_sr:.2f}, params: {best_params}")

            # Validate
            val_b = engine.run(signal, best_params, sym, '2h', VAL_START, VAL_END, INITIAL_EQUITY, "validation")
            val_m = mm.compute(val_b, train_sharpe=train_sr)
            log(f"  Val Sharpe: {val_m.sharpe_ratio:.2f}, overfit: {val_m.flag_overfit}")

            if val_m.flag_overfit:
                log(f"  REJECTED: OVERFIT")
                results[base] = {"result": "OVERFIT", "train_sr": train_sr, "val_sr": val_m.sharpe_ratio}
                continue

            # Check entry threshold
            if best_params.get("entry_threshold", 0) < 1.0:
                log(f"  WARNING: entry_threshold {best_params['entry_threshold']:.2f} < 1.0 — noise fade, not large-move")

            # Check avg hold duration
            if len(val_b.trades) > 0 and 'entry_time' in val_b.trades.columns:
                durations = pd.to_datetime(val_b.trades['exit_time']) - pd.to_datetime(val_b.trades['entry_time'])
                avg_hold_hours = durations.mean().total_seconds() / 3600
                log(f"  Avg hold: {avg_hold_hours:.1f}h ({avg_hold_hours/2:.1f} bars)")
                if avg_hold_hours > 8:
                    log(f"  WARNING: avg hold > 8h — not short-term MR")

            # Walk-forward
            log(f"  Walk-forward (22 windows)...")
            wf = engine.run_walk_forward(signal, sym, '2h', TRAIN_START, HOLDOUT_END,
                                         train_months=6, test_months=2, gap_weeks=2,
                                         n_optuna_trials=30, initial_equity=INITIAL_EQUITY)
            sharpes = [mm.compute(r).sharpe_ratio for r in wf]
            mean_oos = np.mean(sharpes)
            pct_pos = sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1) * 100
            log(f"  WF: mean OOS Sharpe {mean_oos:.2f}, {pct_pos:.0f}% positive ({len(wf)} windows)")

            entry = {"result": "INCONSISTENT" if pct_pos < 60 else "WF_PASS",
                     "train_sr": round(train_sr, 4), "val_sr": round(val_m.sharpe_ratio, 4),
                     "wf_pct": round(pct_pos, 1), "wf_mean_sr": round(mean_oos, 4),
                     "params": best_params}

            if pct_pos < 60:
                log(f"  REJECTED: INCONSISTENT ({pct_pos:.0f}%)")
            else:
                # Perturbation test
                perturb = mm.run_perturbation_test(signal, best_params, engine, sym, '2h', VAL_START, VAL_END)
                entry["perturbation_drop"] = perturb["max_sharpe_drop_pct"]
                entry["fragile"] = perturb["fragile"]
                log(f"  Perturbation: {perturb['max_sharpe_drop_pct']:.1f}% drop, fragile: {perturb['fragile']}")

                # Holdout
                ho_b = engine.run(signal, best_params, sym, '2h', HOLDOUT_START, HOLDOUT_END, INITIAL_EQUITY, "holdout")
                btc_data = dm.get_ohlcv("BTC/USDT:USDT", '2h', HOLDOUT_START, HOLDOUT_END)
                btc_ret = btc_data["close"].pct_change().dropna()
                ho_m = mm.compute(ho_b, btc_returns=btc_ret, train_sharpe=train_sr)
                entry["holdout_sr"] = round(ho_m.sharpe_ratio, 4)
                entry["holdout_return"] = round(ho_m.total_return_pct, 2)
                entry["holdout_maxdd"] = round(ho_m.max_drawdown_pct, 2)
                entry["passes_all"] = ho_m.passes_all_checks
                log(f"  Holdout: Sharpe {ho_m.sharpe_ratio:.2f}, Return {ho_m.total_return_pct:.1f}%, DD {ho_m.max_drawdown_pct:.1f}%")
                log(f"  All flags clear: {ho_m.passes_all_checks}")
                log(mm.format_summary(ho_m, f"{base} 2H holdout"))

            results[base] = entry
        except Exception as e:
            log(f"  ERROR: {e}")
            traceback.print_exc()
            results[base] = {"result": "ERROR", "error": str(e)}

    return results


# ============================================================
# EXPERIMENT 2: HAR Volatility Targeting on SOL 1C
# ============================================================
def compute_har_vol_forecast(returns):
    """HAR model for 4H bars."""
    rv_1d = returns.rolling(6).std() * np.sqrt(252 * 6)
    rv_1w = returns.rolling(42).std() * np.sqrt(252 * 6)
    rv_1m = returns.rolling(182).std() * np.sqrt(252 * 6)

    forecasts = pd.Series(np.nan, index=returns.index)
    min_fit = 252

    for i in range(min_fit, len(returns)):
        sl = slice(i - min_fit, i)
        X = pd.DataFrame({'rv_1d': rv_1d.iloc[sl], 'rv_1w': rv_1w.iloc[sl], 'rv_1m': rv_1m.iloc[sl]}).dropna()
        if len(X) < 100:
            continue
        y = rv_1d.iloc[sl].shift(-1).reindex(X.index).dropna()
        X = X.loc[y.index]
        if len(X) < 50:
            continue
        model = LinearRegression()
        model.fit(X, y)
        xp = np.array([[rv_1d.iloc[i], rv_1w.iloc[i], rv_1m.iloc[i]]])
        if not np.isnan(xp).any():
            forecasts.iloc[i] = max(model.predict(xp)[0], 0.01)

    return forecasts.fillna(rv_1d)


def run_experiment_2():
    log("\n" + "="*60)
    log("EXPERIMENT 2: HAR Volatility Targeting on SOL 1C")
    log("="*60)

    # Best params from v2 search for SOL 1C 4h
    sol_1c_params = {'fast_period': 43, 'slow_period': 113, 'adx_period': 10, 'adx_threshold': 23}

    results = {}

    for target_vol in [0.15, 0.20, 0.25]:
        for leverage in [3.0, 5.0]:
            label = f"tv{target_vol}_lev{leverage}"
            log(f"\n--- SOL 1C + HAR: target_vol={target_vol}, leverage={leverage} ---")

            # Get SOL 4h data for full period
            data_train = dm.get_ohlcv('SOL/USDT:USDT', '4h', TRAIN_START, TRAIN_END)
            data_val = dm.get_ohlcv('SOL/USDT:USDT', '4h', VAL_START, VAL_END)
            data_ho = dm.get_ohlcv('SOL/USDT:USDT', '4h', HOLDOUT_START, HOLDOUT_END)

            signal = TrendRegimeSignal()

            # For each split, compute HAR forecast and run with vol-targeted sizing
            split_results = {}
            for split_name, data, start, end in [
                ("train", data_train, TRAIN_START, TRAIN_END),
                ("validation", data_val, VAL_START, VAL_END),
                ("holdout", data_ho, HOLDOUT_START, HOLDOUT_END),
            ]:
                returns = data['close'].pct_change()
                har_forecast = compute_har_vol_forecast(returns)

                # Generate signal
                sig = signal.generate(data, sol_1c_params)
                signal.validate_output(sig, data)

                # Manual backtest with HAR sizing
                cash = INITIAL_EQUITY
                equity_series = [cash]
                trades = []
                position = None

                for i in range(1, len(data)):
                    s = sig.iloc[i - 1]
                    price = data['close'].iloc[i]
                    ts = data.index[i]
                    hv = har_forecast.iloc[i] if not pd.isna(har_forecast.iloc[i]) else 0.5

                    # Close logic
                    if position is not None:
                        should_close = (s == 0) or (s != position['direction'])
                        if should_close:
                            cr = cm.apply_close(position['entry_price'], price, position['size'], 'SOL/USDT:USDT', position['direction'])
                            pnl = (cr['fill_price'] - position['entry_price']) * position['size'] * position['direction'] - cr['fee_usdt']
                            trades.append({"pnl_usdt": pnl, "pnl_pct": pnl / cash * 100,
                                          "entry_time": position['entry_time'], "exit_time": ts,
                                          "entry_price": position['entry_price'], "exit_price": cr['fill_price'],
                                          "size": position['size'], "direction": position['direction'],
                                          "cost_usdt": cr['fee_usdt'] + position.get('open_fee', 0), "funding_cost_usdt": 0})
                            cash += pnl
                            position = None

                    # Open logic with HAR sizing
                    if s != 0 and position is None:
                        vol_scalar = min(max(target_vol / hv, 0.1), 2.0) if hv > 0 else 0.5
                        notional = cash * vol_scalar * leverage
                        notional = min(notional, cash * 0.95 * leverage)
                        size = notional / price
                        if size > 0:
                            opr = cm.apply_open(price, size, 'SOL/USDT:USDT', s)
                            cash -= opr['fee_usdt']
                            position = {'direction': s, 'size': size, 'entry_price': opr['fill_price'],
                                       'entry_time': ts, 'open_fee': opr['fee_usdt']}

                    # MTM equity
                    if position:
                        mtm = (price - position['entry_price']) * position['size'] * position['direction']
                        equity_series.append(cash + mtm)
                    else:
                        equity_series.append(cash)

                # Close remaining
                if position:
                    price = data['close'].iloc[-1]
                    cr = cm.apply_close(position['entry_price'], price, position['size'], 'SOL/USDT:USDT', position['direction'])
                    pnl = (cr['fill_price'] - position['entry_price']) * position['size'] * position['direction'] - cr['fee_usdt']
                    trades.append({"pnl_usdt": pnl, "pnl_pct": pnl / cash * 100,
                                  "entry_time": position['entry_time'], "exit_time": data.index[-1],
                                  "entry_price": position['entry_price'], "exit_price": cr['fill_price'],
                                  "size": position['size'], "direction": position['direction'],
                                  "cost_usdt": cr['fee_usdt'] + position.get('open_fee', 0), "funding_cost_usdt": 0})
                    cash += pnl
                    equity_series[-1] = cash

                from crypto_infra.backtest_engine import ResultsBundle
                eq = pd.Series(equity_series, index=data.index)
                trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(columns=[
                    "entry_time","exit_time","entry_price","exit_price","size","direction","pnl_usdt","pnl_pct","cost_usdt","funding_cost_usdt"])
                monthly = eq.resample("ME").last().pct_change().dropna() * 100
                bundle = ResultsBundle(symbol='SOL/USDT:USDT', timeframe='4h', start=start, end=end,
                                      strategy_name=f"1C_HAR_{label}", params=sol_1c_params,
                                      equity_curve=eq, trades=trades_df, monthly_returns=monthly, split=split_name)

                btc_ret = dm.get_ohlcv("BTC/USDT:USDT", '4h', start, end)['close'].pct_change().dropna()
                train_sr = None
                if split_name == "train":
                    train_sr_val = (monthly.mean() / monthly.std() * np.sqrt(12)) if len(monthly) > 1 and monthly.std() > 0 else 0
                    split_results["train_sr"] = round(float(train_sr_val), 4)
                m = mm.compute(bundle, btc_returns=btc_ret,
                              train_sharpe=split_results.get("train_sr") if split_name != "train" else None)
                split_results[split_name] = {
                    "sharpe": round(m.sharpe_ratio, 4),
                    "return": round(m.total_return_pct, 2),
                    "max_dd": round(m.max_drawdown_pct, 2),
                    "max_dd_days": m.max_drawdown_duration_days,
                    "consec_losing": m.max_consecutive_losing_months,
                    "trades": m.total_trades,
                    "passes_all": m.passes_all_checks,
                }
                log(f"  {split_name}: Sharpe={m.sharpe_ratio:.2f}, Return={m.total_return_pct:.1f}%, DD={m.max_drawdown_pct:.1f}%, DD_days={m.max_drawdown_duration_days}, Consec={m.max_consecutive_losing_months}, Passes={m.passes_all_checks}")

            results[label] = split_results

    return results


# ============================================================
# EXPERIMENT 3: BTC Lead-Lag at 5m
# ============================================================
def run_experiment_3():
    log("\n" + "="*60)
    log("EXPERIMENT 3: BTC Lead-Lag at 5m")
    log("="*60)

    # Check if 5m data is available
    import os
    btc_5m_path = None
    for f in os.listdir('data_cache'):
        if 'BTC' in f and '5m' in f:
            btc_5m_path = os.path.join('data_cache', f)
            break

    if not btc_5m_path:
        log("  5m data not yet available — attempting download...")
        try:
            btc_5m = dm.get_ohlcv('BTC/USDT:USDT', '5m', '2023-01-01', '2024-12-31')
            sol_5m = dm.get_ohlcv('SOL/USDT:USDT', '5m', '2023-01-01', '2024-12-31')
            log(f"  BTC 5m: {len(btc_5m)} bars, SOL 5m: {len(sol_5m)} bars")
        except Exception as e:
            log(f"  5m download failed: {e}")
            log("  EXPERIMENT 3 SKIPPED — insufficient data")
            return {"result": "SKIPPED", "reason": "5m data download failed"}
    else:
        btc_5m = dm.get_ohlcv('BTC/USDT:USDT', '5m', '2023-01-01', '2024-12-31')
        sol_5m = dm.get_ohlcv('SOL/USDT:USDT', '5m', '2023-01-01', '2024-12-31')

    from strategies.exp3_btc_lead_lag import BTCLeadLagSignal
    signal = BTCLeadLagSignal(btc_5m)

    sm = SizerModule(fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cm, sm)

    # Quick screen with defaults
    log(f"  Quick screen SOL 5m...")
    screen_b = engine.run(signal, {"btc_lookback": 3, "btc_threshold": 0.003, "hold_bars": 6, "min_btc_vol_mult": 1.5},
                          'SOL/USDT:USDT', '5m', '2023-01-01', '2023-06-30', INITIAL_EQUITY, "screen")
    weeks = (pd.Timestamp('2023-06-30') - pd.Timestamp('2023-01-01')).days / 7
    tpw = len(screen_b.trades) / max(weeks, 1)
    log(f"  Screen: {tpw:.2f} tpw, {len(screen_b.trades)} trades")

    if tpw < 0.5:
        log(f"  REJECTED: INFREQUENT ({tpw:.2f} tpw)")
        return {"result": "INFREQUENT", "tpw": tpw}

    # Optimise on 2023 H1
    log(f"  Optimising (20 trials on 2023 H1)...")
    best_params, train_sr = engine._optimise(signal, 'SOL/USDT:USDT', '5m',
                                              '2023-01-01', '2023-06-30', 20, INITIAL_EQUITY)
    log(f"  Train Sharpe: {train_sr:.2f}, params: {best_params}")

    # Validate on 2023 H2
    val_b = engine.run(signal, best_params, 'SOL/USDT:USDT', '5m', '2023-07-01', '2023-12-31', INITIAL_EQUITY, "validation")
    val_m = mm.compute(val_b, train_sharpe=train_sr)
    log(f"  Val Sharpe: {val_m.sharpe_ratio:.2f}, overfit: {val_m.flag_overfit}")

    if val_m.flag_overfit:
        log(f"  REJECTED: OVERFIT")
        return {"result": "OVERFIT", "train_sr": train_sr, "val_sr": val_m.sharpe_ratio}

    # Check hold duration
    if best_params.get("hold_bars", 0) > 6:
        log(f"  WARNING: hold_bars={best_params['hold_bars']} > 6 — delayed momentum, not lead-lag")

    # Walk-forward with shorter windows (60d train, 20d test, 5d gap)
    log(f"  Walk-forward (compressed windows on 2023-2024)...")
    wf = engine.run_walk_forward(signal, 'SOL/USDT:USDT', '5m',
                                  '2023-01-01', '2024-12-31',
                                  train_months=2, test_months=1, gap_weeks=1,
                                  n_optuna_trials=15, initial_equity=INITIAL_EQUITY)
    sharpes = [mm.compute(r).sharpe_ratio for r in wf]
    mean_oos = np.mean(sharpes)
    pct_pos = sum(1 for s in sharpes if s > 0) / max(len(sharpes), 1) * 100
    log(f"  WF: mean OOS Sharpe {mean_oos:.2f}, {pct_pos:.0f}% positive ({len(wf)} windows)")

    return {
        "result": "WF_PASS" if pct_pos >= 60 else "INCONSISTENT",
        "train_sr": round(train_sr, 4), "val_sr": round(val_m.sharpe_ratio, 4),
        "wf_pct": round(pct_pos, 1), "wf_mean_sr": round(mean_oos, 4),
        "params": best_params,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_results = {}

    log("Starting V4 Targeted Research")
    log("="*60)

    # Experiment 1
    try:
        all_results["exp1"] = run_experiment_1()
    except Exception as e:
        log(f"EXP1 FATAL: {e}")
        traceback.print_exc()
        all_results["exp1"] = {"error": str(e)}

    # Experiment 2
    try:
        all_results["exp2"] = run_experiment_2()
    except Exception as e:
        log(f"EXP2 FATAL: {e}")
        traceback.print_exc()
        all_results["exp2"] = {"error": str(e)}

    # Experiment 3
    try:
        all_results["exp3"] = run_experiment_3()
    except Exception as e:
        log(f"EXP3 FATAL: {e}")
        traceback.print_exc()
        all_results["exp3"] = {"error": str(e)}

    # Save
    with open("v4_results.json", "w") as f:
        def ser(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            return str(o)
        json.dump(all_results, f, indent=2, default=ser)

    log("\n" + "="*60)
    log("V4 RESEARCH COMPLETE")
    log("="*60)
    log(f"Results saved to v4_results.json")

    # Print summary
    for exp_name, exp_data in all_results.items():
        log(f"\n{exp_name}:")
        if isinstance(exp_data, dict):
            for k, v in exp_data.items():
                if isinstance(v, dict):
                    log(f"  {k}: {json.dumps(v, default=ser)}")
                else:
                    log(f"  {k}: {v}")
