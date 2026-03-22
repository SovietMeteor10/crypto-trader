"""
Test the daily MA signal directly — the actual mechanism behind the Supertrend result.
"""

import sys, json, logging, time
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import optuna
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule
from strategies.daily_ma_sol import DailyMASOL

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

SYMBOL = 'SOL/USDT:USDT'
TF = '4h'
TRAIN_START, TRAIN_END = '2021-01-01', '2022-12-31'
VAL_START, VAL_END = '2023-01-01', '2023-12-31'
HO_START, HO_END = '2024-01-01', '2026-03-21'


def sharpe(b):
    m = b.monthly_returns
    if len(m) < 2 or m.std() == 0: return 0.0
    return float(np.clip((m.mean() / m.std()) * np.sqrt(12), -10, 10))


def main():
    t0 = time.time()
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)
    metrics = MetricsModule()

    sol_daily = dm.get_ohlcv(SYMBOL, '1d', '2020-01-01', HO_END)
    btc_4h = dm.get_ohlcv('BTC/USDT:USDT', TF, '2020-06-01', HO_END)
    btc_ret = btc_4h['close'].pct_change().dropna()

    signal = DailyMASOL(daily_ohlcv=sol_daily)
    results = {}

    # ══════════════════════════════════════════════════════════════
    # SANITY CHECK: Unoptimised daily 26-MA, no buffer
    # ══════════════════════════════════════════════════════════════
    log.info("=" * 70)
    log.info("SANITY CHECK: Unoptimised daily 26-MA (buffer=0)")
    log.info("=" * 70)

    raw_params = {'ma_period': 26, 'buffer_pct': 0.0}

    for split_name, start, end in [('train', TRAIN_START, TRAIN_END),
                                    ('val', VAL_START, VAL_END),
                                    ('holdout', HO_START, HO_END)]:
        b = engine.run(signal, raw_params, SYMBOL, TF, start, end, 1000, split_name)
        s = sharpe(b)
        log.info(f"  {split_name:8s}: Sharpe={s:7.4f}, trades={len(b.trades)}")
        results[f'raw_{split_name}'] = {'sharpe': round(s, 4), 'trades': len(b.trades)}

    raw_ho_sharpe = results['raw_holdout']['sharpe']
    log.info(f"\n  RAW HOLDOUT SHARPE: {raw_ho_sharpe}")
    if raw_ho_sharpe > 2.0:
        log.info("  Signal is REAL — unoptimised version achieves Sharpe > 2.0")
    elif raw_ho_sharpe < 1.0:
        log.info("  Signal is SPURIOUS — unoptimised version achieves Sharpe < 1.0")
    else:
        log.info("  Signal is MODERATE — between 1.0 and 2.0")

    # ══════════════════════════════════════════════════════════════
    # OPTUNA: 30 trials
    # ══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("OPTUNA: 30 trials (ma_period + buffer_pct)")
    log.info("=" * 70)

    def obj(trial):
        p = {'ma_period': trial.suggest_int('ma_period', 10, 60),
             'buffer_pct': trial.suggest_float('buffer_pct', 0.0, 2.0)}
        try:
            return sharpe(engine.run(signal, p, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'opt'))
        except:
            return -10

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=30)
    best = study.best_params
    log.info(f"Best params: {best}, train Sharpe: {study.best_value:.4f}")
    results['best_params'] = best

    # Three-split
    train_b = engine.run(signal, best, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'train')
    val_b = engine.run(signal, best, SYMBOL, TF, VAL_START, VAL_END, 1000, 'val')
    ts, vs = sharpe(train_b), sharpe(val_b)
    log.info(f"Train: {ts:.4f} ({len(train_b.trades)}), Val: {vs:.4f} ({len(val_b.trades)})")

    results['train_sharpe'] = round(ts, 4)
    results['val_sharpe'] = round(vs, 4)
    results['train_trades'] = len(train_b.trades)
    results['val_trades'] = len(val_b.trades)

    if ts > 3.0 and vs < 1.0:
        log.info("OVERFIT")
        results['conclusion'] = 'FAIL: overfit'
        _save(results, t0)
        return

    overfit_ok = vs >= 0.5 * ts if ts > 0 else vs > 0
    results['overfit_check'] = 'PASS' if overfit_ok else 'FAIL'
    if not overfit_ok:
        results['conclusion'] = 'FAIL: overfit'
        _save(results, t0)
        return

    # Walk-forward
    log.info("\nWalk-forward: 28 windows")
    wf = engine.run_walk_forward(signal, SYMBOL, TF, TRAIN_START, HO_END,
                                  train_months=6, test_months=2, gap_weeks=2,
                                  n_optuna_trials=15, initial_equity=1000)
    wf_sharpes = [sharpe(r) for r in wf]
    wf_pct = sum(1 for s in wf_sharpes if s > 0) / len(wf_sharpes) * 100
    wf_mean = float(np.mean(wf_sharpes))
    log.info(f"WF: {wf_pct:.1f}% positive ({sum(1 for s in wf_sharpes if s > 0)}/{len(wf_sharpes)}), mean={wf_mean:.4f}")

    results['wf_pct'] = round(wf_pct, 1)
    results['wf_mean'] = round(wf_mean, 4)
    results['wf_sharpes'] = [round(s, 4) for s in wf_sharpes]

    if wf_pct < 60:
        results['holdout_sharpe'] = 'not run'
        results['conclusion'] = f'FAIL: WF {wf_pct:.1f}%'
        _save(results, t0)
        return

    # Holdout
    log.info("\nHoldout evaluation")
    ho = engine.run(signal, best, SYMBOL, TF, HO_START, HO_END, 1000, 'holdout')
    ho_s = sharpe(ho)
    ho_m = metrics.compute(ho, btc_ret, ts)

    results['holdout_sharpe'] = round(ho_s, 4)
    results['holdout_trades'] = len(ho.trades)
    results['holdout_passes_all'] = ho_m.passes_all_checks
    results['delta_vs_v3'] = round(ho_s - 0.9065, 4)

    flags = {k: getattr(ho_m, f'flag_{k}', False) for k in
             ['overfit', 'insufficient_trades', 'high_btc_correlation',
              'negative_skew', 'long_drawdown', 'consecutive_losses']}
    results['holdout_flags'] = flags

    log.info(f"Holdout: Sharpe={ho_s:.4f}, trades={len(ho.trades)}, delta={ho_s-0.9065:+.4f}")
    log.info(f"Flags: {flags}")

    if ho_s > 1.0 and ho_m.passes_all_checks:
        results['conclusion'] = 'PASS'
    elif ho_s > 1.0:
        failed = [k for k, v in flags.items() if v]
        results['conclusion'] = f'CONDITIONAL PASS (flags: {failed})'
    else:
        results['conclusion'] = f'FAIL: holdout {ho_s:.2f}'

    _save(results, t0)


def _save(results, t0):
    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.0f}s. {results.get('conclusion', '?')}")

    with open('daily_ma_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    ho = results.get('holdout_sharpe', 'N/A')
    delta = results.get('delta_vs_v3')
    ds = f"{delta:+.2f}" if isinstance(delta, (int, float)) else "N/A"
    raw_ho = results.get('raw_holdout', {}).get('sharpe', 'N/A')

    entry = f"""

### Daily MA SOL: Pure daily MA trend signal

Best params: ma_period={results.get('best_params', {}).get('ma_period', 'N/A')}, buffer_pct={results.get('best_params', {}).get('buffer_pct', 'N/A')}
Unoptimised (MA=26, buf=0) holdout Sharpe: {raw_ho}
Train Sharpe: {results.get('train_sharpe', 'N/A')}
Val Sharpe: {results.get('val_sharpe', 'N/A')}
WF % positive: {results.get('wf_pct', 'N/A')}
Holdout Sharpe: {ho}
Delta vs V3 (0.91): {ds}
All flags clear: {'YES' if results.get('holdout_passes_all') else 'NO'}
Conclusion: {results.get('conclusion', 'N/A')}
"""

    with open('STRATEGY_LOG.md', 'r') as f:
        content = f.read()
    with open('STRATEGY_LOG.md', 'w') as f:
        f.write(content + entry)

    log.info("Saved.")


if __name__ == '__main__':
    main()
