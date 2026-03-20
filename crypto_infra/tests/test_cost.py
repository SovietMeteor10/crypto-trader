"""Tests for CostModule."""

from crypto_infra.cost_module import CostModule


def test_round_trip_cost_floor():
    """Open and close a $1000 notional BTC position, confirm cost >= min_round_trip_pct."""
    cm = CostModule(min_round_trip_pct=0.15)
    rt_cost = cm.get_round_trip_cost_pct("BTC/USDT:USDT")
    assert rt_cost >= 0.15, f"Round trip cost {rt_cost} < min 0.15%"


def test_funding_long_pays_positive_rate():
    """Long position, positive funding rate -> cost should be positive (expense)."""
    cm = CostModule()
    cost = cm.apply_funding(
        position_size=1.0,
        price=50000,
        direction=1,
        funding_rate=0.0001,
        n_periods=1,
    )
    assert cost > 0, f"Long should pay positive funding, got {cost}"


def test_funding_short_receives_positive_rate():
    """Short position, positive funding rate -> cost should be negative (income)."""
    cm = CostModule()
    cost = cm.apply_funding(
        position_size=1.0,
        price=50000,
        direction=-1,
        funding_rate=0.0001,
        n_periods=1,
    )
    assert cost < 0, f"Short should receive positive funding, got {cost}"


def test_slippage_per_symbol():
    """BTC slippage should be lower than SOL slippage."""
    cm = CostModule()
    btc_open = cm.apply_open(50000, 1.0, "BTC/USDT:USDT", 1)
    sol_open = cm.apply_open(50, 1000.0, "SOL/USDT:USDT", 1)

    # Both at same notional ($50k) - BTC slippage should be less
    btc_slip = abs(btc_open["fill_price"] - 50000) / 50000
    sol_slip = abs(sol_open["fill_price"] - 50) / 50
    assert btc_slip < sol_slip, f"BTC slippage {btc_slip} >= SOL slippage {sol_slip}"


def test_open_close_fee_deducted():
    """Fees should be positive for both open and close."""
    cm = CostModule()
    open_res = cm.apply_open(50000, 0.1, "BTC/USDT:USDT", 1)
    assert open_res["fee_usdt"] > 0

    close_res = cm.apply_close(50000, 51000, 0.1, "BTC/USDT:USDT", 1)
    assert close_res["fee_usdt"] > 0


def test_funding_multiple_periods():
    """Funding cost scales linearly with n_periods."""
    cm = CostModule()
    cost_1 = cm.apply_funding(1.0, 50000, 1, 0.0001, 1)
    cost_3 = cm.apply_funding(1.0, 50000, 1, 0.0001, 3)
    assert abs(cost_3 - 3 * cost_1) < 0.01, "Funding should scale linearly with periods"
