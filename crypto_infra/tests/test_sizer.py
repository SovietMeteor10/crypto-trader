"""Tests for SizerModule."""

from crypto_infra.sizer_module import SizerModule


def test_fixed_fractional_respects_leverage():
    """equity=1000, fraction=0.02, leverage=3 -> notional should be 60 USDT."""
    sm = SizerModule(method="fixed_fractional", fraction=0.02, leverage=3.0)
    size = sm.compute_size(signal=1, equity=1000.0, price=50000.0, volatility=0.5)

    # notional = 1000 * 0.02 * 3 = 60
    # size = 60 / 50000 = 0.0012
    expected_size = 60.0 / 50000.0
    assert abs(size - expected_size) < 1e-10, f"Size {size} != expected {expected_size}"


def test_max_position_cap():
    """Fraction so high it would exceed max_position_pct -> size is capped."""
    sm = SizerModule(
        method="fixed_fractional",
        fraction=2.0,  # 200% of equity — way over max
        leverage=3.0,
        max_position_pct=0.95,
    )
    size = sm.compute_size(signal=1, equity=1000.0, price=50000.0, volatility=0.5)

    # Max notional = 1000 * 0.95 * 3 = 2850
    # Max size = 2850 / 50000 = 0.057
    max_size = 1000.0 * 0.95 * 3.0 / 50000.0
    assert size <= max_size + 1e-10, f"Size {size} exceeds max {max_size}"


def test_liquidation_check_safe():
    """3x leverage long -> liquidation at ~33% below entry -> safe=True (>15%)."""
    sm = SizerModule(leverage=3.0)
    result = sm.check_liquidation_risk(
        entry_price=50000.0,
        position_size=0.01,
        equity=1000.0,
        leverage=3.0,
        direction=1,
    )

    assert result["safe"] is True, f"3x leverage should be safe, got distance={result['distance_pct']:.2%}"
    assert result["distance_pct"] > 0.15


def test_liquidation_check_unsafe():
    """10x leverage long -> liquidation at ~10% below entry -> safe=False."""
    sm = SizerModule(leverage=10.0)
    result = sm.check_liquidation_risk(
        entry_price=50000.0,
        position_size=0.1,
        equity=1000.0,
        leverage=10.0,
        direction=1,
    )

    assert result["safe"] is False, f"10x leverage should be unsafe, got distance={result['distance_pct']:.2%}"
    assert result["distance_pct"] < 0.15


def test_fixed_usdt_method():
    """fixed_usdt uses fraction as USDT notional, applies leverage."""
    sm = SizerModule(method="fixed_usdt", fraction=100.0, leverage=5.0)
    size = sm.compute_size(signal=1, equity=10000.0, price=50000.0, volatility=0.5)

    # notional = 100 * 5 = 500
    # size = 500 / 50000 = 0.01
    expected = 500.0 / 50000.0
    assert abs(size - expected) < 1e-10, f"Size {size} != expected {expected}"


def test_zero_signal_returns_zero():
    """Signal of 0 should return size 0."""
    sm = SizerModule()
    size = sm.compute_size(signal=0, equity=1000.0, price=50000.0, volatility=0.5)
    assert size == 0.0
