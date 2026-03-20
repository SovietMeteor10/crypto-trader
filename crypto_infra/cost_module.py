"""CostModule — fees, slippage, funding rate application."""


DEFAULT_SLIPPAGE_BPS = {
    "BTC": 5,
    "ETH": 5,
    "SOL": 10,
    "default": 10,
}


class CostModule:
    def __init__(
        self,
        taker_fee_pct: float = 0.05,
        slippage_bps: dict = None,
        min_round_trip_pct: float = 0.15,
    ):
        self.taker_fee_pct = taker_fee_pct
        self.slippage_bps = slippage_bps or DEFAULT_SLIPPAGE_BPS.copy()
        self.min_round_trip_pct = min_round_trip_pct

    def _get_slippage_bps(self, symbol: str) -> float:
        # Extract base currency from symbol like "BTC/USDT:USDT"
        base = symbol.split("/")[0] if "/" in symbol else symbol
        return self.slippage_bps.get(base, self.slippage_bps.get("default", 10))

    def apply_open(
        self,
        price: float,
        size: float,
        symbol: str,
        direction: int,
    ) -> dict:
        slippage_bps = self._get_slippage_bps(symbol)
        slippage_frac = slippage_bps / 10000

        # Slippage works against us: longs pay more, shorts receive less
        if direction == 1:
            fill_price = price * (1 + slippage_frac)
        else:
            fill_price = price * (1 - slippage_frac)

        notional = fill_price * size
        fee_usdt = notional * (self.taker_fee_pct / 100)
        cost_pct = (abs(fill_price - price) / price * 100) + (self.taker_fee_pct)

        return {
            "fill_price": fill_price,
            "fee_usdt": fee_usdt,
            "cost_pct": cost_pct,
        }

    def apply_close(
        self,
        entry_price: float,
        exit_price: float,
        size: float,
        symbol: str,
        direction: int,
    ) -> dict:
        slippage_bps = self._get_slippage_bps(symbol)
        slippage_frac = slippage_bps / 10000

        # Closing slippage works against us: closing longs sell lower, closing shorts buy higher
        if direction == 1:
            fill_price = exit_price * (1 - slippage_frac)
        else:
            fill_price = exit_price * (1 + slippage_frac)

        notional = fill_price * size
        fee_usdt = notional * (self.taker_fee_pct / 100)
        cost_pct = (abs(fill_price - exit_price) / exit_price * 100) + (self.taker_fee_pct)

        return {
            "fill_price": fill_price,
            "fee_usdt": fee_usdt,
            "cost_pct": cost_pct,
        }

    def apply_funding(
        self,
        position_size: float,
        price: float,
        direction: int,
        funding_rate: float,
        n_periods: int,
    ) -> float:
        """
        Returns funding cost in USDT (positive = cost, negative = income).
        Long pays when rate is positive. Short receives when rate is positive.
        """
        notional = position_size * price
        # Long: cost = notional * rate * periods (positive rate = expense)
        # Short: cost = -notional * rate * periods (positive rate = income)
        cost = direction * notional * funding_rate * n_periods
        return cost

    def get_round_trip_cost_pct(self, symbol: str) -> float:
        slippage_bps = self._get_slippage_bps(symbol)
        # Open slippage + close slippage + open fee + close fee
        slippage_pct = 2 * slippage_bps / 100  # convert bps to pct
        fee_pct = 2 * self.taker_fee_pct
        total = slippage_pct + fee_pct
        return max(total, self.min_round_trip_pct)
