"""
Multi-Timeframe Experiments A (MTF V3) and B (Supertrend).
Runs both sequentially, then portfolio C if both pass.
"""

import sys, json, logging, time
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import optuna
import pandas as pd

from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine, ResultsBundle
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule
from strategies.sol_1c_sjm_mtf import SOL1C_SJM_MTF
from strategies.sol_1c_sjm import SOL1C_SJM
from strategies.supertrend_sol import SupertrendSOL

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

SOL_PARAMS = {'fast_period': 42, 'slow_period': 129, 'adx_period': 24, 'adx_threshold': 27}
V3_SJM = {'sjm_lambda': 1.6573239546018446, 'sjm_window': 378, 'trade_in_neutral': True}

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

    log.info("Loading data...")
    btc_4h = dm.get_ohlcv('BTC/USDT:USDT', TF, '2020-06-01', HO_END)
    btc_ret = btc_4h['close'].pct_change().dropna()
    sol_daily = dm.get_ohlcv(SYMBOL, '1d', '2020-01-01', HO_END)
    ms_1h = pd.read_parquet('data_cache/market_structure/SOL_USDT_USDT_unified_1h.parquet')
    ms_1h.index = pd.DatetimeIndex(ms_1h.index)
    log.info(f"BTC 4H: {len(btc_4h)}, SOL daily: {len(sol_daily)}, MS 1H: {len(ms_1h)}")

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT A — MTF V3
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "="*70)
    log.info("EXPERIMENT A: Multi-Timeframe V3")
    log.info("="*70)

    mtf_signal = SOL1C_SJM_MTF(
        btc_data=btc_4h, daily_ohlcv=sol_daily, ms_1h=ms_1h,
        feature_set='A', use_sol_features=True, n_regimes=3,
        fixed_sol_params=SOL_PARAMS,
    )

    # Diagnostic
    log.info("\n--- Diagnostic ---")
    v3_baseline = SOL1C_SJM(btc_data=btc_4h, feature_set='A',
                             use_sol_features=True, n_regimes=3,
                             fixed_sol_params=SOL_PARAMS)
    base_b = engine.run(v3_baseline, V3_SJM, SYMBOL, TF, VAL_START, VAL_END, 1000, 'diag')
    base_trades = len(base_b.trades)

    diag_p = {**V3_SJM, 'daily_ma_period': 50, 'daily_buffer_pct': 0.5,
              'ls_quantile_high': 0.80, 'ls_quantile_low': 0.20,
              'ls_rolling_window': 168, 'use_ls_filter': True}
    diag_b = engine.run(mtf_signal, diag_p, SYMBOL, TF, VAL_START, VAL_END, 1000, 'diag')
    surv = len(diag_b.trades) / base_trades * 100 if base_trades > 0 else 0
    log.info(f"V3 trades: {base_trades}, MTF trades: {len(diag_b.trades)}, survival: {surv:.0f}%")
    log.info(f"V3 Sharpe: {sharpe(base_b):.4f}, MTF Sharpe: {sharpe(diag_b):.4f}")

    results['A_diagnostic'] = {'base_trades': base_trades, 'mtf_trades': len(diag_b.trades),
                                'survival': round(surv, 1)}

    # Optuna: 40 trials, freeze V3 params, only MTF params free
    log.info("\n--- Optuna: 40 trials ---")

    def obj_a(trial):
        p = {**V3_SJM,
             'daily_ma_period': trial.suggest_int('daily_ma_period', 20, 100),
             'daily_buffer_pct': trial.suggest_float('daily_buffer_pct', 0.0, 1.0),
             'ls_quantile_high': trial.suggest_float('ls_quantile_high', 0.70, 0.90),
             'ls_quantile_low': trial.suggest_float('ls_quantile_low', 0.10, 0.30),
             'ls_rolling_window': trial.suggest_int('ls_rolling_window', 48, 336),
             'use_ls_filter': trial.suggest_categorical('use_ls_filter', [True, False])}
        try:
            return sharpe(engine.run(mtf_signal, p, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'opt'))
        except: return -10

    study_a = optuna.create_study(direction='maximize')
    study_a.optimize(obj_a, n_trials=40)
    best_a = {**V3_SJM, **study_a.best_params}
    log.info(f"Best MTF params: {study_a.best_params}")
    log.info(f"Best train Sharpe: {study_a.best_value:.4f}")

    # Three-split
    train_a = engine.run(mtf_signal, best_a, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'train')
    val_a = engine.run(mtf_signal, best_a, SYMBOL, TF, VAL_START, VAL_END, 1000, 'val')
    ts_a, vs_a = sharpe(train_a), sharpe(val_a)
    log.info(f"Train: {ts_a:.4f} ({len(train_a.trades)}), Val: {vs_a:.4f} ({len(val_a.trades)})")

    results['A'] = {'best_params': study_a.best_params, 'train_sharpe': round(ts_a, 4),
                     'val_sharpe': round(vs_a, 4), 'train_trades': len(train_a.trades),
                     'val_trades': len(val_a.trades)}

    if ts_a > 3.0 and vs_a < 1.0:
        log.info("OVERFIT A")
        results['A']['conclusion'] = 'FAIL: overfit'
    else:
        overfit_ok = vs_a >= 0.5 * ts_a if ts_a > 0 else vs_a > 0
        results['A']['overfit_check'] = 'PASS' if overfit_ok else 'FAIL'

        if overfit_ok:
            # Walk-forward
            log.info("\n--- Walk-forward A ---")
            wf_a = engine.run_walk_forward(mtf_signal, SYMBOL, TF, TRAIN_START, HO_END,
                                            train_months=6, test_months=2, gap_weeks=2,
                                            n_optuna_trials=15, initial_equity=1000)
            wf_sharpes_a = [sharpe(r) for r in wf_a]
            wf_pct_a = sum(1 for s in wf_sharpes_a if s > 0) / len(wf_sharpes_a) * 100
            results['A']['wf_pct'] = round(wf_pct_a, 1)
            results['A']['wf_mean'] = round(float(np.mean(wf_sharpes_a)), 4)
            log.info(f"WF A: {wf_pct_a:.1f}% positive, mean={np.mean(wf_sharpes_a):.4f}")

            if wf_pct_a >= 60:
                log.info("\n--- Holdout A ---")
                ho_a = engine.run(mtf_signal, best_a, SYMBOL, TF, HO_START, HO_END, 1000, 'holdout')
                ho_s_a = sharpe(ho_a)
                ho_m_a = metrics.compute(ho_a, btc_ret, ts_a)
                results['A']['holdout_sharpe'] = round(ho_s_a, 4)
                results['A']['holdout_trades'] = len(ho_a.trades)
                results['A']['holdout_passes'] = ho_m_a.passes_all_checks
                results['A']['delta'] = round(ho_s_a - 0.9065, 4)
                flags = {k: getattr(ho_m_a, f'flag_{k}', False) for k in
                         ['overfit', 'insufficient_trades', 'high_btc_correlation',
                          'negative_skew', 'long_drawdown', 'consecutive_losses']}
                results['A']['flags'] = flags
                log.info(f"Holdout A: {ho_s_a:.4f} ({len(ho_a.trades)} trades), delta={ho_s_a-0.9065:+.4f}")
                results['A']['conclusion'] = 'PASS' if ho_s_a > 1.0 and ho_m_a.passes_all_checks else f'holdout {ho_s_a:.2f}'
            else:
                results['A']['holdout_sharpe'] = 'not run'
                results['A']['conclusion'] = f'FAIL: WF {wf_pct_a:.1f}%'
        else:
            results['A']['conclusion'] = 'FAIL: overfit'

    # ═══════════════════════════════════════════════════════════════
    # EXPERIMENT B — Supertrend SOL
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "="*70)
    log.info("EXPERIMENT B: Supertrend SOL")
    log.info("="*70)

    st_signal = SupertrendSOL(btc_data=btc_4h, daily_ohlcv=sol_daily)

    # Optuna: 40 trials
    log.info("\n--- Optuna B: 40 trials ---")

    def obj_b(trial):
        p = {'atr_period': trial.suggest_int('atr_period', 7, 21),
             'multiplier': trial.suggest_float('multiplier', 1.5, 4.0),
             'use_daily_filter': trial.suggest_categorical('use_daily_filter', [True, False]),
             'daily_ma_period': trial.suggest_int('daily_ma_period', 20, 100),
             'daily_buffer_pct': trial.suggest_float('daily_buffer_pct', 0.0, 1.0)}
        try:
            return sharpe(engine.run(st_signal, p, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'opt'))
        except: return -10

    study_b = optuna.create_study(direction='maximize')
    study_b.optimize(obj_b, n_trials=40)
    best_b = study_b.best_params
    log.info(f"Best ST params: {best_b}")

    train_b = engine.run(st_signal, best_b, SYMBOL, TF, TRAIN_START, TRAIN_END, 1000, 'train')
    val_b = engine.run(st_signal, best_b, SYMBOL, TF, VAL_START, VAL_END, 1000, 'val')
    ts_b, vs_b = sharpe(train_b), sharpe(val_b)
    log.info(f"Train B: {ts_b:.4f} ({len(train_b.trades)}), Val B: {vs_b:.4f} ({len(val_b.trades)})")

    results['B'] = {'best_params': best_b, 'train_sharpe': round(ts_b, 4),
                     'val_sharpe': round(vs_b, 4), 'train_trades': len(train_b.trades),
                     'val_trades': len(val_b.trades)}

    if ts_b > 3.0 and vs_b < 1.0:
        results['B']['conclusion'] = 'FAIL: overfit'
    else:
        overfit_ok_b = vs_b >= 0.5 * ts_b if ts_b > 0 else vs_b > 0
        results['B']['overfit_check'] = 'PASS' if overfit_ok_b else 'FAIL'

        if overfit_ok_b:
            log.info("\n--- Walk-forward B ---")
            wf_b = engine.run_walk_forward(st_signal, SYMBOL, TF, TRAIN_START, HO_END,
                                            train_months=6, test_months=2, gap_weeks=2,
                                            n_optuna_trials=15, initial_equity=1000)
            wf_sharpes_b = [sharpe(r) for r in wf_b]
            wf_pct_b = sum(1 for s in wf_sharpes_b if s > 0) / len(wf_sharpes_b) * 100
            results['B']['wf_pct'] = round(wf_pct_b, 1)
            results['B']['wf_mean'] = round(float(np.mean(wf_sharpes_b)), 4)
            log.info(f"WF B: {wf_pct_b:.1f}% positive")

            if wf_pct_b >= 55:
                log.info("\n--- Holdout B ---")
                ho_b = engine.run(st_signal, best_b, SYMBOL, TF, HO_START, HO_END, 1000, 'holdout')
                ho_s_b = sharpe(ho_b)
                ho_m_b = metrics.compute(ho_b, btc_ret, ts_b)
                results['B']['holdout_sharpe'] = round(ho_s_b, 4)
                results['B']['holdout_trades'] = len(ho_b.trades)
                results['B']['delta'] = round(ho_s_b - 0.9065, 4)

                # Correlation with V3
                v3_ho = engine.run(v3_baseline, V3_SJM, SYMBOL, TF, HO_START, HO_END, 1000, 'ho_v3')
                v3_monthly = v3_ho.monthly_returns
                st_monthly = ho_b.monthly_returns
                common = v3_monthly.index.intersection(st_monthly.index)
                if len(common) > 3:
                    corr = v3_monthly.loc[common].corr(st_monthly.loc[common])
                else:
                    corr = np.nan
                results['B']['corr_with_v3'] = round(float(corr), 4) if not np.isnan(corr) else None
                log.info(f"Holdout B: {ho_s_b:.4f}, corr with V3: {corr:.4f}")
                results['B']['conclusion'] = f'holdout {ho_s_b:.2f}, corr {corr:.2f}'
            else:
                results['B']['holdout_sharpe'] = 'not run'
                results['B']['conclusion'] = f'FAIL: WF {wf_pct_b:.1f}%'
        else:
            results['B']['conclusion'] = 'FAIL: overfit'

    # ═══════════════════════════════════════════════════════════════
    # SAVE
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    with open('mtf_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Append to STRATEGY_LOG
    _append_log(results)
    log.info("Saved to mtf_results.json and STRATEGY_LOG.md")


def _append_log(results):
    a = results.get('A', {})
    b = results.get('B', {})
    diag = results.get('A_diagnostic', {})

    a_delta = a.get('delta')
    a_ds = f"{a_delta:+.2f}" if isinstance(a_delta, (int, float)) else "N/A"
    a_ho = a.get('holdout_sharpe', 'not run')

    b_ho = b.get('holdout_sharpe', 'not run')
    b_corr = b.get('corr_with_v3', 'N/A')

    entry = f"""

### MTF-A: V3 + Daily Filter + LS Gate

Daily MA period: {a.get('best_params', {}).get('daily_ma_period', 'N/A')}
Daily buffer: {a.get('best_params', {}).get('daily_buffer_pct', 'N/A')}%
LS quantile high: {a.get('best_params', {}).get('ls_quantile_high', 'N/A')}
Use LS filter: {a.get('best_params', {}).get('use_ls_filter', 'N/A')}
Signal survival rate vs V3: {diag.get('survival', 'N/A')}%
Train Sharpe: {a.get('train_sharpe', 'N/A')}
Val Sharpe: {a.get('val_sharpe', 'N/A')}
WF % positive: {a.get('wf_pct', 'N/A')}
Holdout Sharpe: {a_ho}
Delta vs V3 (0.91): {a_ds}
All flags clear: {'YES' if a.get('holdout_passes') else 'NO'}
Conclusion: {a.get('conclusion', 'N/A')}

### MTF-B: Supertrend SOL 4H

ATR period: {b.get('best_params', {}).get('atr_period', 'N/A')}
Multiplier: {b.get('best_params', {}).get('multiplier', 'N/A')}
Use daily filter: {b.get('best_params', {}).get('use_daily_filter', 'N/A')}
Train Sharpe: {b.get('train_sharpe', 'N/A')}
Val Sharpe: {b.get('val_sharpe', 'N/A')}
WF % positive: {b.get('wf_pct', 'N/A')}
Holdout Sharpe: {b_ho}
Correlation with V3 holdout: {b_corr}
Conclusion: {b.get('conclusion', 'N/A')}
"""

    with open('STRATEGY_LOG.md', 'r') as f:
        content = f.read()
    with open('STRATEGY_LOG.md', 'w') as f:
        f.write(content + entry)


if __name__ == '__main__':
    main()
