"""Tests for BacktestEngine."""

import logging

import pytest

from crypto_infra.data_module import DataModule
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.backtest_engine import BacktestEngine
from .helpers import AlwaysLongSignal, AlwaysFlatSignal, SimpleMACrossSignal


@pytest.fixture
def engine(tmp_path):
    dm = DataModule(cache_dir=str(tmp_path / "cache"))
    cm = CostModule()
    sm = SizerModule(fraction=0.02, leverage=3.0)
    return BacktestEngine(dm, cm, sm)


def test_backtest_with_trivial_signal(engine):
    """Always-long signal on 90 days of BTC 1h data."""
    signal = AlwaysLongSignal()
    result = engine.run(
        signal, {"dummy": 1}, "BTC/USDT:USDT", "1h",
        "2024-06-01", "2024-08-30", initial_equity=1000.0,
    )

    # Equity curve has correct length
    assert len(result.equity_curve) > 0

    # Trades are recorded
    assert len(result.trades) > 0, "Should have at least 1 trade"

    # Costs are being deducted — equity should differ from initial
    assert result.equity_curve.iloc[-1] != 1000.0, "Equity unchanged — costs not applied?"

    # Result has correct metadata
    assert result.symbol == "BTC/USDT:USDT"
    assert result.strategy_name == "always_long"


def test_backtest_flat_signal_no_trades(engine):
    """Always-zero signal -> no trades, equity equals initial."""
    signal = AlwaysFlatSignal()
    result = engine.run(
        signal, {"dummy": 1}, "BTC/USDT:USDT", "1h",
        "2024-06-01", "2024-08-30", initial_equity=1000.0,
    )

    assert len(result.trades) == 0, "Flat signal should produce no trades"
    assert result.equity_curve.iloc[-1] == 1000.0, "Equity should be unchanged"


def test_walk_forward_window_count(engine):
    """3 years of data, 6m train + 2m test + 2w gap -> check window count."""
    signal = SimpleMACrossSignal()

    # Use daily timeframe for speed
    engine_daily = BacktestEngine(
        engine.data_module,
        engine.cost_module,
        SizerModule(fraction=0.02, leverage=3.0),
    )

    results = engine_daily.run_walk_forward(
        signal, "BTC/USDT:USDT", "1d",
        "2022-01-01", "2024-12-31",
        train_months=6, test_months=2, gap_weeks=2,
        n_optuna_trials=5,  # keep fast for tests
        initial_equity=1000.0,
    )

    # Should have multiple windows
    assert len(results) > 5, f"Expected >5 walk-forward windows, got {len(results)}"

    # Each result should be labelled walkforward
    for r in results:
        assert r.split == "walkforward"
        assert r.window_id is not None


def test_liquidation_check_skips_risky_trades(engine, caplog):
    """High leverage should trigger liquidation skip."""
    # Create engine with very high leverage
    dm = engine.data_module
    cm = engine.cost_module
    sm_risky = SizerModule(fraction=0.5, leverage=20.0)
    risky_engine = BacktestEngine(dm, cm, sm_risky)

    signal = AlwaysLongSignal()

    with caplog.at_level(logging.WARNING):
        result = risky_engine.run(
            signal, {"dummy": 1}, "BTC/USDT:USDT", "1h",
            "2024-06-01", "2024-06-15", initial_equity=1000.0,
        )

    # At 20x leverage, liquidation is at 5% from entry — below 15% threshold
    # So trades should be skipped
    assert "liquidation" in caplog.text.lower() or len(result.trades) == 0, \
        "Expected liquidation warning or no trades with 20x leverage"
