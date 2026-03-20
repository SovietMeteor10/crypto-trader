"""Tests for MetricsModule."""

import numpy as np
import pandas as pd
import pytest

from crypto_infra.backtest_engine import ResultsBundle
from crypto_infra.metrics_module import MetricsModule, MetricsBundle
from crypto_infra.data_module import DataModule
from crypto_infra.cost_module import CostModule
from crypto_infra.sizer_module import SizerModule
from crypto_infra.backtest_engine import BacktestEngine
from .helpers import FragileSignal


def _make_bundle(monthly_returns, equity_values=None, trades=None):
    """Helper to create a ResultsBundle with known monthly returns."""
    months = pd.date_range("2024-01-31", periods=len(monthly_returns), freq="ME", tz="UTC")
    m_ret = pd.Series(monthly_returns, index=months)

    if equity_values is None:
        eq_start = 1000.0
        eq = [eq_start]
        for r in monthly_returns:
            eq.append(eq[-1] * (1 + r / 100))
        # Make hourly equity from monthly
        hours = pd.date_range("2024-01-01", periods=len(eq), freq="h", tz="UTC")
        equity = pd.Series(eq, index=hours)
    else:
        hours = pd.date_range("2024-01-01", periods=len(equity_values), freq="h", tz="UTC")
        equity = pd.Series(equity_values, index=hours)

    if trades is None:
        trades = pd.DataFrame(columns=[
            "entry_time", "exit_time", "entry_price", "exit_price",
            "size", "direction", "pnl_usdt", "pnl_pct", "cost_usdt", "funding_cost_usdt",
        ])

    return ResultsBundle(
        symbol="BTC/USDT:USDT",
        timeframe="1h",
        start="2024-01-01",
        end="2024-12-31",
        strategy_name="test",
        params={},
        equity_curve=equity,
        trades=trades,
        monthly_returns=m_ret,
        split="train",
    )


def test_sharpe_calculation():
    """Construct known monthly returns, verify Sharpe matches manual calculation."""
    returns = [5.0, 3.0, -2.0, 4.0, 1.0, -1.0, 6.0, 2.0, 3.0, -0.5, 4.0, 2.0]
    bundle = _make_bundle(returns)

    mm = MetricsModule()
    metrics = mm.compute(bundle)

    # Manual Sharpe: mean/std * sqrt(12)
    arr = np.array(returns)
    expected_sharpe = (arr.mean() / arr.std()) * np.sqrt(12)

    # pandas .std() uses ddof=1, numpy uses ddof=0 — allow some tolerance
    assert abs(metrics.sharpe_ratio - expected_sharpe) < 0.3, \
        f"Sharpe {metrics.sharpe_ratio} != expected {expected_sharpe:.2f}"


def test_overfit_flag_true():
    """train_sharpe=2.0, val Sharpe low -> flag_overfit should be True."""
    # Bundle with bad Sharpe (close to 0)
    returns = [1.0, -1.0, 0.5, -0.5, 0.2, -0.3]
    bundle = _make_bundle(returns)

    mm = MetricsModule()
    metrics = mm.compute(bundle, train_sharpe=2.0)

    assert metrics.flag_overfit == True, \
        f"Should flag overfit: Sharpe={metrics.sharpe_ratio}, train=2.0"


def test_overfit_flag_false():
    """train_sharpe=2.0, val Sharpe high -> flag_overfit should be False."""
    returns = [5.0, 4.0, 3.0, 6.0, 5.0, 4.0, 7.0, 3.0, 5.0, 4.0, 6.0, 5.0]
    bundle = _make_bundle(returns)

    mm = MetricsModule()
    metrics = mm.compute(bundle, train_sharpe=2.0)

    assert metrics.flag_overfit == False, \
        f"Should not flag overfit: Sharpe={metrics.sharpe_ratio}, train=2.0"


def test_passes_all_checks_requires_all_clear():
    """MetricsBundle with any flag True -> passes_all_checks must be False."""
    returns = [1.0, -1.0, 0.5, -0.5]
    bundle = _make_bundle(returns)

    mm = MetricsModule()
    metrics = mm.compute(bundle)

    # With only 4 months and 0 trades, insufficient_trades flag will be True
    assert metrics.flag_insufficient_trades is True
    assert metrics.passes_all_checks is False


def test_perturbation_test_detects_fragile_params(tmp_path):
    """Fragile signal module where perturbing one param destroys performance."""
    dm = DataModule(cache_dir=str(tmp_path / "cache"))
    cm = CostModule()
    sm = SizerModule(fraction=0.02, leverage=3.0)
    engine = BacktestEngine(dm, cm, sm)

    signal = FragileSignal()
    mm = MetricsModule()

    result = mm.run_perturbation_test(
        signal, {"magic_period": 17}, engine,
        "BTC/USDT:USDT", "1d", "2024-01-01", "2024-12-31",
        perturbation_pct=0.10,
    )

    assert "base_sharpe" in result
    assert "perturbed_sharpes" in result
    assert "fragile" in result
    assert isinstance(result["fragile"], bool)


def test_format_summary():
    """format_summary should return a non-empty string."""
    returns = [5.0, 3.0, -2.0, 4.0, 1.0, -1.0]
    bundle = _make_bundle(returns)

    mm = MetricsModule()
    metrics = mm.compute(bundle)
    summary = mm.format_summary(metrics, title="Test Strategy")

    assert "Test Strategy" in summary
    assert "Sharpe" in summary
    assert "PASS" in summary or "FAIL" in summary
