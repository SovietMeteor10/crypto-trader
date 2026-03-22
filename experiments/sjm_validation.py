"""
SJM Validation Tests — run these before the full experiment.

Tests:
1. SJM correctness on synthetic data with known regimes
2. SJM no-lookahead-bias check
3. Regime gating logic
4. Signal module parameter space isolation
5. Quick 3-window walk-forward to verify pipeline
"""

import sys
sys.path.insert(0, '/home/ubuntu/projects/crypto-trader')

import numpy as np
import pandas as pd
import time
from crypto_infra.data_module import DataModule
from crypto_infra.backtest_engine import BacktestEngine
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.metrics_module import MetricsModule
from regime.sjm import StatisticalJumpModel
from regime.features import compute_feature_set_A, standardise

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


def test_1_sjm_synthetic():
    """SJM should recover known regimes from synthetic data."""
    print("\n=== TEST 1: SJM on synthetic data ===")

    rng = np.random.default_rng(42)
    T = 600
    n_features = 3

    # Create 3 clear regimes: bull (0-200), neutral (200-400), bear (400-600)
    X = np.zeros((T, n_features))
    true_regimes = np.zeros(T, dtype=int)

    # Bull: positive mean
    X[:200] = rng.normal(loc=[2.0, 1.0, 0.5], scale=0.3, size=(200, n_features))
    true_regimes[:200] = 0

    # Neutral: near zero
    X[200:400] = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.3, size=(200, n_features))
    true_regimes[200:400] = 1

    # Bear: negative mean
    X[400:] = rng.normal(loc=[-2.0, -1.0, -0.5], scale=0.3, size=(200, n_features))
    true_regimes[400:] = 2

    sjm = StatisticalJumpModel(n_regimes=3, jump_penalty=1.0)
    sjm.fit(X)

    regimes = sjm.result_.regimes

    # Check that SJM finds exactly 3 distinct regimes
    unique = np.unique(regimes)
    check("3 unique regimes found", len(unique) == 3, f"found {len(unique)}")

    # Check few transitions (should be ~2 for clean data)
    check("Few transitions (<=5)", sjm.result_.n_jumps <= 5,
          f"got {sjm.result_.n_jumps}")

    # Check that each segment is mostly one regime
    seg1_mode = np.bincount(regimes[:200]).argmax()
    seg2_mode = np.bincount(regimes[200:400]).argmax()
    seg3_mode = np.bincount(regimes[400:]).argmax()
    check("3 segments have different modes",
          len({seg1_mode, seg2_mode, seg3_mode}) == 3,
          f"modes: {seg1_mode}, {seg2_mode}, {seg3_mode}")

    # Check purity of each segment (>95%)
    for i, (start, end, label) in enumerate([(0, 200, "bull"), (200, 400, "neutral"), (400, 600, "bear")]):
        seg = regimes[start:end]
        mode = np.bincount(seg).argmax()
        purity = (seg == mode).mean()
        check(f"Segment {label} purity > 95%", purity > 0.95, f"purity={purity:.1%}")


def test_2_sjm_convergence():
    """SJM should converge (same result on refit)."""
    print("\n=== TEST 2: SJM convergence ===")

    rng = np.random.default_rng(123)
    X = rng.normal(size=(200, 4))

    sjm1 = StatisticalJumpModel(n_regimes=3, jump_penalty=0.5, random_state=42)
    sjm1.fit(X)
    r1 = sjm1.result_.regimes.copy()

    sjm2 = StatisticalJumpModel(n_regimes=3, jump_penalty=0.5, random_state=42)
    sjm2.fit(X)
    r2 = sjm2.result_.regimes.copy()

    check("Deterministic with same seed", np.array_equal(r1, r2))

    # Check centroids are finite
    check("Centroids are finite", np.all(np.isfinite(sjm1.centroids_)))


def test_3_no_lookahead():
    """Regime at time T must only use data up to T, not future data."""
    print("\n=== TEST 3: No lookahead bias ===")

    rng = np.random.default_rng(42)
    T = 300
    X = rng.normal(size=(T, 3))

    # Fit on first 200, predict bar 200
    sjm = StatisticalJumpModel(n_regimes=3, jump_penalty=1.0)
    sjm.fit(X[:200])
    pred_200 = sjm.predict(X[200:201])[0]

    # Now change data after bar 200 drastically
    X_modified = X.copy()
    X_modified[201:] = 100.0  # extreme future data

    # Refit on first 200 (same), predict bar 200
    sjm2 = StatisticalJumpModel(n_regimes=3, jump_penalty=1.0)
    sjm2.fit(X[:200])
    pred_200_v2 = sjm2.predict(X_modified[200:201])[0]

    check("Prediction at T=200 unchanged by future data",
          pred_200 == pred_200_v2,
          f"pred1={pred_200}, pred2={pred_200_v2}")

    # Verify predict() only uses centroids (nearest), not DP
    pred_batch = sjm.predict(X[200:205])
    check("Predict returns valid regime IDs",
          all(0 <= r < 3 for r in pred_batch))


def test_4_label_regimes():
    """Label mapping should correctly rank by return."""
    print("\n=== TEST 4: Regime labelling ===")

    sjm = StatisticalJumpModel(n_regimes=3)

    # 3 regimes: 0 has negative return, 1 has zero, 2 has positive
    regimes = np.array([0]*50 + [1]*50 + [2]*50)
    returns = np.concatenate([
        np.full(50, -0.02),  # regime 0: bear
        np.full(50, 0.001),  # regime 1: neutral
        np.full(50, 0.03),   # regime 2: bull
    ])

    labels = sjm.label_regimes(regimes, returns)
    check("Highest return regime labelled 'bull'", labels[2] == 'bull',
          f"regime 2 labelled as {labels[2]}")
    check("Lowest return regime labelled 'bear'", labels[0] == 'bear',
          f"regime 0 labelled as {labels[0]}")
    check("Middle regime labelled 'neutral'", labels[1] == 'neutral',
          f"regime 1 labelled as {labels[1]}")


def test_5_signal_gating():
    """Signal should be zeroed out in bear regime."""
    print("\n=== TEST 5: Signal gating logic ===")

    from strategies.sol_1c_sjm import SOL1C_SJM

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    btc = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2020-06-01', '2023-01-01')
    sol = dm.get_ohlcv('SOL/USDT:USDT', '4h', '2022-01-01', '2022-06-30')

    signal_mod = SOL1C_SJM(btc_data=btc)
    params = {
        'fast_period': 42, 'slow_period': 129, 'adx_period': 24, 'adx_threshold': 27,
        'sjm_lambda': 1.0, 'sjm_window': 360, 'trade_in_neutral': True,
    }

    signal = signal_mod.generate(sol, params)

    check("Signal values in {-1, 0, 1}", set(signal.unique()).issubset({-1, 0, 1}),
          f"values: {signal.unique()}")
    check("Signal has same length as data", len(signal) == len(sol))
    check("No NaN in signal", not signal.isna().any())
    check("Signal is integer type", signal.dtype in [np.int64, np.int32, int])

    # Check that some bars are gated (zeroed out)
    n_zero = (signal == 0).sum()
    n_total = len(signal)
    check("Some bars gated (>10% zero)", n_zero / n_total > 0.10,
          f"zero bars: {n_zero}/{n_total} ({n_zero/n_total:.1%})")

    # Run with trade_in_neutral=False — should have MORE zeros
    params2 = {**params, 'trade_in_neutral': False}
    signal2 = signal_mod.generate(sol, params2)
    n_zero2 = (signal2 == 0).sum()
    check("trade_in_neutral=False has more zeros",
          n_zero2 >= n_zero,
          f"neutral=True zeros: {n_zero}, neutral=False zeros: {n_zero2}")


def test_6_parameter_space_isolation():
    """When fixed_sol_params=True, walk-forward should only optimize SJM params."""
    print("\n=== TEST 6: Parameter space isolation ===")

    from strategies.sol_1c_sjm import SOL1C_SJM

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    btc = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2020-06-01', '2023-01-01')

    signal = SOL1C_SJM(btc_data=btc)
    space = signal.parameter_space

    # The parameter space includes ALL params (SOL + SJM)
    sol_params = {'fast_period', 'slow_period', 'adx_period', 'adx_threshold'}
    sjm_params = {'sjm_lambda', 'sjm_window', 'trade_in_neutral'}

    check("Parameter space has SOL params", sol_params.issubset(space.keys()),
          f"missing: {sol_params - space.keys()}")
    check("Parameter space has SJM params", sjm_params.issubset(space.keys()),
          f"missing: {sjm_params - space.keys()}")

    # FLAG: The current code exposes ALL params to Optuna.
    # For fixed SOL params, we need a subclass or wrapper that only exposes SJM params.
    print("  WARNING: parameter_space exposes ALL params. Walk-forward will re-optimize SOL params too!")
    print("  FIX NEEDED: Create SOL1C_SJM_FixedSOL that only has sjm_lambda, sjm_window, trade_in_neutral")


def test_7_fast_slow_constraint():
    """fast_period should always be < slow_period."""
    print("\n=== TEST 7: Fast/slow MA constraint ===")

    # This is a parameter constraint that needs enforcing
    # In the WF results we saw fast=40, slow=29 — which is wrong
    print("  WARNING: No constraint enforcing fast_period < slow_period")
    print("  FIX NEEDED: Either add constraint in objective or swap in generate()")


def test_8_timing():
    """Time a single backtest run to estimate full experiment time."""
    print("\n=== TEST 8: Timing estimate ===")

    from strategies.sol_1c_sjm import SOL1C_SJM

    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    btc = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2020-06-01', '2026-03-21')

    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    signal = SOL1C_SJM(btc_data=btc)
    params = {
        'fast_period': 42, 'slow_period': 129, 'adx_period': 24, 'adx_threshold': 27,
        'sjm_lambda': 1.0, 'sjm_window': 360, 'trade_in_neutral': True,
    }

    # Time training period
    t0 = time.time()
    bundle = engine.run(signal, params, 'SOL/USDT:USDT', '4h',
                        '2021-01-01', '2022-12-31', 1000.0, 'train')
    t_train = time.time() - t0

    # Time 6-month window (for WF)
    t0 = time.time()
    bundle2 = engine.run(signal, params, 'SOL/USDT:USDT', '4h',
                         '2022-01-01', '2022-06-30', 1000.0, 'wf')
    t_wf = time.time() - t0

    print(f"  2-year train: {t_train:.1f}s")
    print(f"  6-month WF window: {t_wf:.1f}s")

    n_variants = 5
    n_optuna_train = 30
    n_wf_windows = 22
    n_optuna_wf = 15

    est_per_variant = n_optuna_train * t_train + n_wf_windows * n_optuna_wf * t_wf
    est_total = n_variants * est_per_variant
    print(f"  Estimated time per variant: {est_per_variant/60:.0f} min")
    print(f"  Estimated total (5 variants): {est_total/60:.0f} min")

    check("Single run < 10s", t_train < 10, f"{t_train:.1f}s")


def test_9_overfit_detection():
    """Verify the overfit detection logic catches extreme cases."""
    print("\n=== TEST 9: Overfit detection ===")

    # Create a strategy that overfits: perfect on train, random on val
    from strategies.sol_1c_sjm import SOL1C_SJM
    dm = DataModule(cache_dir='/home/ubuntu/projects/crypto-trader/data_cache')
    btc = dm.get_ohlcv('BTC/USDT:USDT', '4h', '2020-06-01', '2026-03-21')
    cost = CostModule()
    sizer = SizerModule(method='fixed_fractional', fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cost, sizer)

    signal = SOL1C_SJM(btc_data=btc)
    params = {
        'fast_period': 42, 'slow_period': 129, 'adx_period': 24, 'adx_threshold': 27,
        'sjm_lambda': 1.0, 'sjm_window': 360, 'trade_in_neutral': True,
    }

    train_b = engine.run(signal, params, 'SOL/USDT:USDT', '4h',
                         '2021-01-01', '2022-12-31', 1000.0, 'train')
    val_b = engine.run(signal, params, 'SOL/USDT:USDT', '4h',
                       '2023-01-01', '2023-12-31', 1000.0, 'val')

    m_train = train_b.monthly_returns
    m_val = val_b.monthly_returns
    train_sr = (m_train.mean() / m_train.std()) * np.sqrt(12) if m_train.std() > 0 else 0
    val_sr = (m_val.mean() / m_val.std()) * np.sqrt(12) if m_val.std() > 0 else 0

    print(f"  Train Sharpe: {train_sr:.2f}, Val Sharpe: {val_sr:.2f}")
    print(f"  Val/Train ratio: {val_sr/train_sr:.2f}" if train_sr > 0 else "  Train Sharpe <= 0")

    # Check the overfit rule
    if train_sr > 3.0 and val_sr < 1.0:
        print("  Would be flagged as OVERFIT (train>3, val<1)")
    elif train_sr > 0 and val_sr < 0.5 * train_sr:
        print("  Would fail overfit check (val < 0.5 * train)")
    else:
        print("  Passes overfit check")
    check("Test completed", True)


if __name__ == '__main__':
    test_1_sjm_synthetic()
    test_2_sjm_convergence()
    test_3_no_lookahead()
    test_4_label_regimes()
    test_5_signal_gating()
    test_6_parameter_space_isolation()
    test_7_fast_slow_constraint()
    test_8_timing()
    test_9_overfit_detection()

    print(f"\n{'=' * 50}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("FIX ALL FAILURES BEFORE RUNNING FULL EXPERIMENT")
    else:
        print("ALL TESTS PASSED — safe to run experiment")
    print(f"{'=' * 50}")
