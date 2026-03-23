"""
Phase 2: Kyle's lambda + L/S ratio as 1H entry filter within daily MA direction.

Two-layer system:
  Layer 1: Daily 26-MA on BTC (long when close > 26-day MA, short when below)
  Layer 2: Within daily direction, enter at 1H bars where BOTH:
    - Kyle's lambda > rolling 70th percentile
    - L/S ratio not at extreme (< 80th pctl for longs, > 20th pctl for shorts)

Optuna 30 trials optimising: kyle_quantile, ls_quantile_high, ls_quantile_low, rolling_window.
Same splits/WF as Phase 1.
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import optuna

sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule
from crypto_infra.signal_module import SignalModule

optuna.logging.set_verbosity(optuna.logging.WARNING)

SYMBOL = 'BTC/USDT:USDT'
TF = '1h'
TRAIN_START = '2020-09-01'
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-12-31'
HO_START = '2024-01-01'
HO_END = '2026-03-23'
MA_PERIOD = 26  # Fixed


class KyleLSFilter(SignalModule):
    """
    Layer 1: Daily 26-MA direction.
    Layer 2: Kyle's lambda + L/S ratio filter on 1H bars.
    """

    def __init__(self, daily_ohlcv, kyle_lambda_1h, ls_ratio_1h):
        self._daily = daily_ohlcv
        self._kyle = kyle_lambda_1h
        self._ls = ls_ratio_1h

    @property
    def name(self):
        return "kyle_ls_filter"

    @property
    def parameter_space(self):
        return {
            "kyle_quantile": ("float", 0.60, 0.85),
            "ls_quantile_high": ("float", 0.70, 0.90),
            "ls_quantile_low": ("float", 0.10, 0.30),
            "rolling_window": ("int", 48, 336),
        }

    def generate(self, data, params):
        kq = params['kyle_quantile']
        lq_high = params['ls_quantile_high']
        lq_low = params['ls_quantile_low']
        rw = params['rolling_window']

        # Layer 1: Daily MA direction
        daily_close = self._daily['close']
        daily_ma = daily_close.rolling(MA_PERIOD).mean()
        daily_signal = pd.Series(0, index=daily_close.index)
        daily_signal[daily_close > daily_ma] = 1
        daily_signal[daily_close < daily_ma] = -1

        # Forward fill to 1H
        direction_1h = daily_signal.reindex(data.index, method='ffill').fillna(0).astype(int)

        # Layer 2: Kyle's lambda and L/S ratio filters
        kyle = self._kyle.reindex(data.index, method='ffill')
        ls = self._ls.reindex(data.index, method='ffill')

        # Rolling quantiles
        kyle_threshold = kyle.rolling(rw, min_periods=max(rw//2, 1)).quantile(kq)
        ls_high_threshold = ls.rolling(rw, min_periods=max(rw//2, 1)).quantile(lq_high)
        ls_low_threshold = ls.rolling(rw, min_periods=max(rw//2, 1)).quantile(lq_low)

        kyle_pass = kyle > kyle_threshold

        # For longs: L/S ratio not at extreme high (< 80th pctl)
        # For shorts: L/S ratio not at extreme low (> 20th pctl)
        ls_pass_long = ls < ls_high_threshold
        ls_pass_short = ls > ls_low_threshold

        signal = pd.Series(0, index=data.index)

        long_bars = (direction_1h == 1) & kyle_pass & ls_pass_long
        short_bars = (direction_1h == -1) & kyle_pass & ls_pass_short

        signal[long_bars] = 1
        signal[short_bars] = -1

        return signal.astype(int)


def compute_kyle_lambda_from_aggtrades():
    """Compute Kyle's lambda at 1H from 15-min aggtrade bars."""
    bars_path = '/home/ubuntu/projects/crypto-trader/data_cache/aggtrades/BTCUSDT_bars_15m.parquet'
    print(f"Loading aggtrade bars from {bars_path}...")
    bars = pd.read_parquet(bars_path)
    bars.index = pd.DatetimeIndex(bars.index)

    # Resample to 1H
    bars_1h = bars.resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'buy_volume': 'sum', 'sell_volume': 'sum',
        'n_trades': 'sum',
    }).dropna(subset=['open'])

    # Kyle's lambda: rolling regression of returns on signed volume
    signed_vol = bars_1h['buy_volume'] - bars_1h['sell_volume']
    returns = bars_1h['close'].pct_change()

    # Rolling 20-bar estimate
    kyle_lambda = pd.Series(np.nan, index=bars_1h.index)
    window = 20
    for i in range(window, len(bars_1h)):
        r = returns.iloc[i-window:i].values
        sv = signed_vol.iloc[i-window:i].values
        mask = ~(np.isnan(r) | np.isnan(sv))
        r_clean, sv_clean = r[mask], sv[mask]
        if len(r_clean) < window // 2:
            continue
        cov = np.cov(r_clean, sv_clean)
        if cov[1, 1] > 0:
            kyle_lambda.iloc[i] = cov[0, 1] / cov[1, 1]

    print(f"Kyle's lambda: {kyle_lambda.notna().sum()} valid bars, "
          f"{kyle_lambda.index.min()} to {kyle_lambda.index.max()}")
    return kyle_lambda


def sharpe(b):
    m = b.monthly_returns
    if len(m) < 2 or m.std() == 0:
        return 0.0
    return float(np.clip((m.mean() / m.std()) * np.sqrt(12), -10, 10))


def trades_per_month(b):
    if len(b.trades) == 0:
        return 0.0
    days = (b.equity_curve.index[-1] - b.equity_curve.index[0]).days
    months = max(days / 30.44, 1)
    return len(b.trades) / months


def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 2: Kyle's lambda + L/S ratio entry filter")
    print("=" * 70)

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics_mod = MetricsModule()

    # Load data
    btc_daily = dm.get_ohlcv(SYMBOL, '1d', '2020-01-01', HO_END)
    btc_1h = dm.get_ohlcv(SYMBOL, TF, TRAIN_START, HO_END)

    # Load market structure for L/S ratio
    ms = pd.read_parquet('/home/ubuntu/projects/crypto-trader/data_cache/market_structure/BTC_USDT_USDT_unified_1h.parquet')
    ms.index = pd.DatetimeIndex(ms.index)
    ls_ratio_1h = ms['ls_ratio']
    print(f"L/S ratio: {ls_ratio_1h.notna().sum()} bars, {ls_ratio_1h.index.min()} to {ls_ratio_1h.index.max()}")

    # Kyle's lambda from aggtrades
    kyle_lambda_1h = compute_kyle_lambda_from_aggtrades()

    signal_mod = KyleLSFilter(btc_daily, kyle_lambda_1h, ls_ratio_1h)
    results = {}

    # ── Diagnostic: what fraction of bars pass Layer 2? ───────────
    print("\n" + "=" * 50)
    print("Diagnostic: Layer 2 pass rate at defaults")
    print("=" * 50)

    default_params = {
        'kyle_quantile': 0.70,
        'ls_quantile_high': 0.80,
        'ls_quantile_low': 0.20,
        'rolling_window': 168,
    }

    test_signal = signal_mod.generate(btc_1h, default_params)
    pass_rate = float((test_signal != 0).mean())
    print(f"  Layer 2 pass rate: {pass_rate:.1%} (target: 20-40%)")

    # Run default params to check trades/month
    try:
        default_bundle = engine.run(signal_mod, default_params, SYMBOL, TF,
                                     TRAIN_START, HO_END, 1000, 'diagnostic')
        default_tpm = trades_per_month(default_bundle)
        default_sharpe = sharpe(default_bundle)
        print(f"  Trades/month (full period): {default_tpm:.1f}")
        print(f"  Sharpe (full period): {default_sharpe:.2f}")
    except Exception as e:
        print(f"  Diagnostic run failed: {e}")
        default_tpm = 0

    results['diagnostic'] = {
        'layer2_pass_rate': pass_rate,
        'trades_per_month_default': default_tpm,
    }

    # ── Optuna: 30 trials ─────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Optuna: 30 trials")
    print("=" * 50)

    def obj(trial):
        p = {
            'kyle_quantile': trial.suggest_float('kyle_quantile', 0.60, 0.85),
            'ls_quantile_high': trial.suggest_float('ls_quantile_high', 0.70, 0.90),
            'ls_quantile_low': trial.suggest_float('ls_quantile_low', 0.10, 0.30),
            'rolling_window': trial.suggest_int('rolling_window', 48, 336),
        }
        try:
            b = engine.run(signal_mod, p, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'opt')
            return sharpe(b)
        except Exception:
            return -10

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=30)
    best = study.best_params
    print(f"Best params: {best}")
    print(f"Best train Sharpe: {study.best_value:.4f}")
    results['best_params'] = best

    # ── Three-split ──────────────────────────────────────────────
    train_b = engine.run(signal_mod, best, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'train')
    val_b = engine.run(signal_mod, best, SYMBOL, TF, VAL_START, VAL_END, 1000, 'val')

    ts = sharpe(train_b)
    vs = sharpe(val_b)
    train_tpm = trades_per_month(train_b)
    val_tpm = trades_per_month(val_b)

    print(f"\nTrain: Sharpe={ts:.4f}, trades/mo={train_tpm:.1f}")
    print(f"Val:   Sharpe={vs:.4f}, trades/mo={val_tpm:.1f}")

    results['train_sharpe'] = round(ts, 4)
    results['val_sharpe'] = round(vs, 4)
    results['train_trades_per_month'] = round(train_tpm, 1)
    results['val_trades_per_month'] = round(val_tpm, 1)

    # ── Walk-forward ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Walk-forward: up to 20 windows")
    print("=" * 50)

    wf = engine.run_walk_forward(signal_mod, SYMBOL, TF, TRAIN_START, HO_END,
                                  train_months=6, test_months=2, gap_weeks=2,
                                  n_optuna_trials=15, initial_equity=1000)

    wf_sharpes = [sharpe(r) for r in wf]
    wf_pct = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100 if wf_sharpes else 0
    wf_mean = float(np.mean(wf_sharpes)) if wf_sharpes else 0
    wf_wr_list = []
    for r in wf:
        if len(r.trades) > 0:
            wr = len(r.trades[r.trades['pnl_usdt'] > 0]) / len(r.trades)
            wf_wr_list.append(wr)

    print(f"\nWF: {wf_pct:.1f}% positive ({sum(1 for s in wf_sharpes if s > 0)}/{len(wf_sharpes)})")
    print(f"WF mean Sharpe: {wf_mean:.4f}")
    if wf_wr_list:
        print(f"WF mean win rate: {np.mean(wf_wr_list):.1%}")

    results['wf_pct'] = round(wf_pct, 1)
    results['wf_mean'] = round(wf_mean, 4)
    results['wf_sharpes'] = [round(s, 4) for s in wf_sharpes]

    # ── Holdout ──────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Holdout evaluation")
    print("=" * 50)

    ho = engine.run(signal_mod, best, SYMBOL, TF, HO_START, HO_END, 1000, 'holdout')
    ho_s = sharpe(ho)
    ho_tpm = trades_per_month(ho)
    ho_wr = 0.0
    if len(ho.trades) > 0:
        ho_wr = len(ho.trades[ho.trades['pnl_usdt'] > 0]) / len(ho.trades)

    print(f"Holdout: Sharpe={ho_s:.4f}, trades/mo={ho_tpm:.1f}, WR={ho_wr:.1%}, trades={len(ho.trades)}")

    # Targets: trades/month >= 10, win rate >= 55%, Sharpe >= 1.0
    targets_met = ho_tpm >= 10 and ho_wr >= 0.55 and ho_s >= 1.0
    print(f"\nTargets: trades/mo>={10}: {'PASS' if ho_tpm >= 10 else 'FAIL'}")
    print(f"         win_rate>=55%:  {'PASS' if ho_wr >= 0.55 else 'FAIL'}")
    print(f"         Sharpe>=1.0:    {'PASS' if ho_s >= 1.0 else 'FAIL'}")
    print(f"All targets met: {'YES' if targets_met else 'NO'}")

    results['holdout'] = {
        'sharpe': round(ho_s, 4),
        'trades': len(ho.trades),
        'trades_per_month': round(ho_tpm, 1),
        'win_rate': round(ho_wr, 4),
        'targets_met': targets_met,
    }

    # Qualification for portfolio
    results['qualifies'] = wf_pct >= 55

    all_pass = wf_pct >= 55 and ho_s > 0
    results['conclusion'] = 'PASS' if all_pass else f'FAIL: WF={wf_pct:.1f}%, Sharpe={ho_s:.2f}'

    # Save
    results['elapsed_seconds'] = time.time() - t0
    with open('/home/ubuntu/projects/crypto-trader/phase2_kyle_ls_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to phase2_kyle_ls_results.json ({time.time()-t0:.0f}s)")
    print(f"Conclusion: {results['conclusion']}")


if __name__ == '__main__':
    main()
